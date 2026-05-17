#!/usr/bin/env python3
"""Continuous backtesting loop — persona-driven, scorer-trained.

Each cycle:
  1. Runs RUNS_PER_CYCLE (5) parallel year-long backtests. Each run uses
     a distinct persona; signal differences come from per-persona ticker
     boosts and different RNG seeds.
  2. Sorts results by total_return_pct and keeps top TOP_RUNS_TO_TRAIN (3)
     positive runs (or the single best if none are positive).
  3. Appends those runs' decisions to data/winner_training.jsonl tagged
     with the cycle number. (Does NOT overwrite — accumulates forever.)
  4. Computes 5-trading-day forward returns for every BUY/SELL decision
     across ALL runs (winners and losers — losing decisions are critical
     signal for the scorer too) and appends them to
     data/decision_outcomes.jsonl, then retrains DecisionScorer.
  5. Spawns a background Opus 4.7 annotator to label the top run's
     decisions GOOD/NEUTRAL/BAD and write a trading lesson — fed back into
     ArticleNet training.
  6. Trims backtest_runs to the most recent KEEP_LAST_RUNS (500) entries.
  7. Sleeps COOLDOWN_SECONDS (60) and loops.

SIGTERM/SIGINT exits cleanly between cycles.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from paper_trader.backtest import (
    BacktestEngine,
    BacktestRun,
    ROOT,
    _get_quant_signals,
    _market_regime,
)

RUNS_PER_CYCLE = 5  # reduced from 10 — 5 runs × 3 max-concurrent claude = safe on 14 GB RAM
TOP_RUNS_TO_TRAIN = 3  # aggregate top-N runs per cycle into training data
KEEP_LAST_RUNS = 500
MAX_OUTCOMES_FOR_TRAINING = 5000  # cap decision_outcomes.jsonl tail used per retrain
COOLDOWN_SECONDS = 60
DISCORD_CHANNEL = "channel:1496099475838603324"
WINNER_JSONL = ROOT / "data" / "winner_training.jsonl"
# digital-intern's article DB that `_inject_and_train` writes winner rows into.
# Module-level (not a function local) so it can be redirected in tests.
DIGITAL_INTERN_ARTICLES_DB = "/home/zeph/digital-intern/data/articles.db"

# How often to run the validation suite (label audit + permutation test) on the
# current cycle's engine. Validation is *expensive* (one full backtest per
# permutation × ~5 perms = ~25 min total), so it runs in a background thread —
# every 10 cycles is enough to catch regressions without dominating compute.
VALIDATION_EVERY_N_CYCLES = 10
VALIDATION_PERMUTATIONS = 5     # background-thread budget — keep this low
VALIDATION_RESULTS_PATH = ROOT / "data" / "validation_results.json"
VALIDATION_RESULTS_KEEP = 50    # cap file growth

EARLIEST_WINDOW_START = date(1993, 2, 1)  # SPY inception — ~30+ years of history
WINDOW_END_BUFFER_DAYS = 180  # never end a window within 6 months of today
MIN_WINDOW_YEARS = 1
MAX_WINDOW_YEARS = 10


def _pick_window(seed: int) -> tuple[date, date]:
    """Pick a deterministic random backtest window given a seed.

    Duration is 1–10 years; start lies between 1993-02-01 (~30yr history) and
    (today - duration - 180d) so the window always ends at least 6 months before today.
    """
    rng = random.Random(seed)
    duration_years = rng.randint(MIN_WINDOW_YEARS, MAX_WINDOW_YEARS)
    duration_days = duration_years * 365

    latest_start = date.today() - timedelta(days=duration_days + WINDOW_END_BUFFER_DAYS)
    span = (latest_start - EARLIEST_WINDOW_START).days
    if span < 0:
        # Pathological: today is within ~5.5yr of EARLIEST. Clamp.
        start = EARLIEST_WINDOW_START
    else:
        start = EARLIEST_WINDOW_START + timedelta(days=rng.randint(0, span))
    end = start + timedelta(days=duration_days)
    return start, end


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _next_run_id(engine: BacktestEngine) -> int:
    # Serialise through the store lock — a background _opus_annotate thread from
    # the previous cycle may still be using the same sqlite3 connection, and
    # concurrent use of one connection across threads corrupts cursor state.
    with engine.store._lock:
        row = engine.store.conn.execute(
            "SELECT COALESCE(MAX(run_id), 0) FROM backtest_runs"
        ).fetchone()
    return int(row[0]) + 1


def _trim_history(engine: BacktestEngine, keep: int = KEEP_LAST_RUNS) -> int:
    conn = engine.store.conn
    with engine.store._lock:
        row = conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()
        total = int(row[0])
        if total <= keep:
            return 0
        cutoff = conn.execute(
            "SELECT run_id FROM backtest_runs "
            "ORDER BY run_id DESC LIMIT 1 OFFSET ?",
            (keep,),
        ).fetchone()
        if cutoff is None:
            return 0
        cutoff_id = int(cutoff[0])
        conn.execute("DELETE FROM backtest_trades WHERE run_id <= ?", (cutoff_id,))
        conn.execute("DELETE FROM backtest_decisions WHERE run_id <= ?", (cutoff_id,))
        cur = conn.execute("DELETE FROM backtest_runs WHERE run_id <= ?", (cutoff_id,))
        conn.commit()
        return cur.rowcount or 0


def _append_top_decisions(engine: BacktestEngine, top_runs: list[BacktestRun],
                          cycle: int) -> int:
    """Aggregate BUY/SELL decisions from top N runs into WINNER_JSONL.

    Records are weighted by each run's return — higher-return runs contribute
    decisions with higher ai_score so the ML trainer up-weights them.
    """
    WINNER_JSONL.parent.mkdir(parents=True, exist_ok=True)
    # Normalise returns to [0.5, 1.0] weight range so even 2nd/3rd place matter
    returns = [r.total_return_pct for r in top_runs]
    max_ret = max(returns) if returns else 1.0
    min_ret = min(returns) if returns else 0.0
    span = max_ret - min_ret or 1.0

    written = 0
    with WINNER_JSONL.open("a") as fh:
        for run in top_runs:
            weight = 0.5 + 0.5 * (run.total_return_pct - min_ret) / span
            try:
                # Hold the store lock — the background _opus_annotate thread
                # may share this sqlite3 connection across threads.
                with engine.store._lock:
                    rows = engine.store.conn.execute(
                        "SELECT action, ticker, sim_date, reasoning, qty, confidence "
                        "FROM backtest_decisions "
                        "WHERE run_id = ? AND action IS NOT NULL AND action != 'HOLD'",
                        (run.run_id,),
                    ).fetchall()
            except Exception as e:
                print(f"[continuous] run {run.run_id} read failed: {e}")
                continue
            rank = top_runs.index(run) + 1
            for row in rows:
                action = (row["action"] or "").upper()
                if action not in ("BUY", "SELL"):
                    continue
                rec = {
                    "cycle": cycle,
                    "run_id": run.run_id,
                    "rank": rank,
                    "title": f"{action} {row['ticker']} on {row['sim_date']}",
                    "source": f"backtest_cycle_{cycle}_rank{rank}",
                    "ai_score": round(weight * (5.0 if action == "BUY" else 0.5), 2),
                    "urgency": 1 if rank == 1 else 0,
                    "label": action,
                    "ticker": row["ticker"] or "",
                    "sim_date": row["sim_date"] or "",
                    "qty": row["qty"],
                    "confidence": row["confidence"],
                    "reasoning": row["reasoning"] or "",
                    "return_pct": run.total_return_pct,
                    "weight": round(weight, 3),
                }
                fh.write(json.dumps(rec) + "\n")
                written += 1
    print(f"[continuous] appended {written} records from top {len(top_runs)} runs → {WINNER_JSONL}")
    return written


def _compute_decision_outcomes(engine: "BacktestEngine",
                               top_runs: list["BacktestRun"]) -> list[dict]:
    """Compute actual 5-trading-day forward returns for BUY/SELL decisions.

    Re-uses PriceCache for returns and _get_quant_signals for features so no
    network calls are needed. Returns a list of outcome records ready to pass
    to train_scorer().

    Uses a 5-trading-day forward window (not calendar days) so weekends and
    holidays don't introduce inconsistent windows across decisions.
    """
    import bisect

    outcomes: list[dict] = []
    _quant_cache: dict[tuple, dict] = {}
    # SPY-based regime depends only on sim_date — cache per date so a cycle of
    # 5 runs × ~250 decisions doesn't recompute 1250 identical SPY 50/200 MAs.
    _regime_cache: dict[str, str] = {}

    trading_days = engine.prices.trading_days
    if not trading_days:
        return outcomes

    def _td_index(d: date) -> int:
        # bisect for exact-match trading-day lookup; -1 if not a trading day.
        i = bisect.bisect_left(trading_days, d)
        if i < len(trading_days) and trading_days[i] == d:
            return i
        return -1

    for run in top_runs:
        try:
            # Hold the store lock — the background _opus_annotate thread may
            # share this sqlite3 connection across threads.
            with engine.store._lock:
                rows = engine.store.conn.execute(
                    "SELECT action, ticker, sim_date, reasoning "
                    "FROM backtest_decisions "
                    "WHERE run_id=? AND action IN ('BUY','SELL') "
                    "AND ticker IS NOT NULL AND ticker != ''",
                    (run.run_id,),
                ).fetchall()
        except Exception as exc:
            print(f"[outcomes] run {run.run_id} read failed: {exc}")
            continue

        for r in rows:
            ticker = r["ticker"] or ""
            sim_date_str = r["sim_date"] or ""
            if not ticker or not sim_date_str:
                continue
            try:
                sim_d = date.fromisoformat(sim_date_str)
            except ValueError:
                continue

            # 5-trading-day forward window. Skip decisions whose window extends
            # past the cached price history — otherwise price_on() falls back to
            # the latest close, which equals sim_d's close and injects fake 0%
            # outcomes into training.
            idx = _td_index(sim_d)
            if idx < 0:
                continue
            target_idx = idx + 5
            if target_idx >= len(trading_days):
                continue
            end_d = trading_days[target_idx]

            # Both price lookups must hit real cached data for this ticker.
            if (engine.prices.price_on(ticker, sim_d) is None
                    or engine.prices.price_on(ticker, end_d) is None):
                continue
            fwd_ret = engine.prices.returns_pct(ticker, sim_d, end_d)

            cache_key = (sim_date_str, ticker)
            if cache_key not in _quant_cache:
                sigs = _get_quant_signals(sim_d, [ticker], engine.prices)
                _quant_cache[cache_key] = sigs.get(ticker, {})
            q = _quant_cache[cache_key]

            regime = _regime_cache.get(sim_date_str)
            if regime is None:
                regime = _market_regime(sim_d, engine.prices)
                _regime_cache[sim_date_str] = regime
            # Match _ml_decide: "unknown" is treated as neutral 1.0, not bear.
            if regime == "bull":
                regime_mult = 1.0
            elif regime == "sideways":
                regime_mult = 0.6
            elif regime == "bear":
                regime_mult = 0.3
            else:
                regime_mult = 1.0

            reasoning = r["reasoning"] or ""
            ml_score = 0.0
            m = re.search(r"score=([0-9.+-]+)", reasoning)
            if m:
                try:
                    ml_score = float(m.group(1))
                except ValueError:
                    pass

            news_urgency: float | None = None
            news_article_count: float | None = None
            m_urg = re.search(r"news_urg=([0-9.+-]+)", reasoning)
            if m_urg:
                try:
                    news_urgency = float(m_urg.group(1))
                except ValueError:
                    pass
            m_cnt = re.search(r"news_count=(\d+)", reasoning)
            if m_cnt:
                try:
                    news_article_count = float(m_cnt.group(1))
                except ValueError:
                    pass
            # Match the inference-side convention: when there is no supporting
            # news, fall back to the build_features neutral defaults (urg=50,
            # cnt=1) by passing None. Otherwise training would see (0, 0) for
            # no-news while predict sees (50, 1) — model gets two encodings
            # of the same condition.
            if news_article_count is not None and news_article_count <= 0:
                news_urgency = None
                news_article_count = None

            outcomes.append({
                "run_id": run.run_id,
                "sim_date": sim_date_str,
                "ticker": ticker,
                "action": r["action"],
                "ml_score": ml_score,
                # Use only numeric quant fields; the legacy uppercase "MACD"
                # is a string label and would corrupt scorer features if it
                # leaked through via `or`-fallback when macd_signal==0.0.
                "rsi": q.get("rsi"),
                "macd": q.get("macd_signal"),
                "mom5": q.get("mom_5d"),
                "mom20": q.get("mom_20d"),
                "regime_mult": regime_mult,
                "vol_ratio": q.get("vol_ratio"),
                "bb_position": q.get("bb_position"),
                "news_urgency": news_urgency,
                "news_article_count": news_article_count,
                "forward_return_5d": round(fwd_ret, 4),
                "return_pct": run.total_return_pct,
            })

    return outcomes


def _train_decision_scorer(outcome_records: list[dict]) -> str:
    """Train DecisionScorer on the historical 80% of outcomes; report OOS RMSE
    on the most recent 20% (true temporal holdout — never seen during training).

    `train_scorer`'s built-in val_rmse uses a *random* 80/20 split which leaks
    future information into validation when records span time. The temporal
    split here is the trustworthy generalization metric.
    """
    if not outcome_records:
        return "no outcome records"
    try:
        from paper_trader.ml.decision_scorer import train_scorer, DecisionScorer
    except Exception as exc:
        return f"scorer err: {exc}"

    # The temporal holdout is a *diagnostic* refinement (an honest OOS RMSE),
    # NOT part of the essential operation. Previously the validation import +
    # split_outcomes_temporal sat in the same try as train_scorer, so if the
    # validation module was unavailable or the split raised on pathological
    # data, training was skipped entirely and the operator saw `scorer err:` —
    # silently wedging the per-cycle retrain invariant (CLAUDE.md §6) and
    # freezing the conviction gate (#5) for as long as the condition lasted.
    # Mirror the already-separated OOS-eval guard below: degrade to "train on
    # everything, no honest holdout" rather than "don't train at all".
    oos_records: list[dict] = []
    train_records = outcome_records
    try:
        from paper_trader.validation import split_outcomes_temporal
        train_records, oos_records = split_outcomes_temporal(
            outcome_records, oos_fraction=0.2
        )
    except Exception as exc:
        print(f"[continuous] temporal split unavailable ({exc}) — training on "
              f"all {len(outcome_records)} records, OOS holdout skipped")

    try:
        result = train_scorer(train_records)
    except Exception as exc:
        return f"scorer err: {exc}"
    val_rmse = result.get("val_rmse", float("nan"))
    val_s = f"{val_rmse:.2f}" if val_rmse == val_rmse else "n/a"

    # OOS evaluation runs AFTER train_scorer has already pickled the model to
    # SCORER_PATH. A crash here (transient pickle/IO race, validation-module
    # change, …) does NOT mean training failed — the scorer is trained and the
    # next cycle's singleton reset will deploy it. Guard it separately so a
    # post-train diagnostic failure degrades to oos_rmse=n/a instead of being
    # reported to the operator as `scorer err` (a false "scorer broken" signal
    # that would make an operator think the conviction gate never engages).
    oos_rmse_s = "n/a"
    if result.get("status") == "ok" and oos_records:
        try:
            from paper_trader.validation import evaluate_scorer_oos
            # Re-load the freshly pickled model from disk so OOS predictions
            # use the exact serialized state (catches any save/load bugs).
            scorer = DecisionScorer()
            oos = evaluate_scorer_oos(scorer, oos_records)
            r = oos.get("rmse")
            if r is not None and r == r:
                oos_rmse_s = f"{r:.2f}"
        except Exception as exc:
            oos_rmse_s = f"n/a (oos-eval err: {type(exc).__name__})"

    return (f"scorer {result['status']} train_n={result['n']} "
            f"val_rmse={val_s} oos_n={len(oos_records)} oos_rmse={oos_rmse_s}")


def _parse_published_date(published) -> date | None:
    """Parse a `published` value (ISO or RFC822) into a date; None if unparseable."""
    if not published:
        return None
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(published)
        if dt is not None:
            return dt.date()
    except Exception:
        pass
    try:
        return date.fromisoformat(str(published)[:10])
    except Exception:
        return None


def _query_news_context(ticker: str, sim_date_str: str, n: int = 4) -> list[str]:
    """Fetch recent article titles from digital-intern DB near sim_date for ticker."""
    DB = ROOT.parent / "digital-intern" / "data" / "articles.db"
    if not DB.exists():
        return []
    try:
        d = date.fromisoformat(sim_date_str)
    except ValueError:
        return []
    lo = d - timedelta(days=3)
    hi = d + timedelta(days=1)
    conn = None
    try:
        conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
        # `published` in articles.db is stored in mixed formats — ISO for some
        # sources, RFC822 ("Wed, 14 May 2026 ...") for RSS. A SQL
        # `published BETWEEN` range filter silently drops every RFC822 row
        # (their leading weekday letter lex-sorts after any ISO date string),
        # so it would have excluded most live news. Fetch a generous candidate
        # set ordered by ai_score and apply the date window in Python after
        # parsing each timestamp robustly.
        rows = conn.execute(
            "SELECT title, published FROM articles "
            "WHERE (title LIKE ? OR title LIKE ?) "
            "AND (url IS NULL OR url NOT LIKE 'backtest://%') "
            "AND (source IS NULL OR (source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%')) "
            "ORDER BY ai_score DESC LIMIT ?",
            (f"%{ticker}%", f"%{ticker.lower()}%", max(n * 20, 40)),
        ).fetchall()
    except Exception:
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    out: list[str] = []
    for title, published in rows:
        if not title:
            continue
        pub_d = _parse_published_date(published)
        # Drop rows that parse to a date outside the window; keep unparseable
        # ones (can't prove they leak) so the context isn't emptied entirely.
        if pub_d is not None and not (lo <= pub_d <= hi):
            continue
        out.append(title)
        if len(out) >= n:
            break
    return out


def _opus_annotate(engine: "BacktestEngine", top_runs: list[BacktestRun],
                   cycle: int, outcome_records: list[dict] | None = None) -> int:
    """Ask Opus 4.7 to annotate ALL decisions (BUY, SELL, HOLD) in the winner run.

    Enhanced over previous version:
    - Covers every decision, not just trades, so HOLDs can also be critiqued
    - Attaches actual 5-day forward returns so Opus sees what happened after each call
    - Pulls relevant scraped news articles from articles DB near each decision date
    - Outcome records (from _compute_decision_outcomes) included as context when available

    Annotations are appended to WINNER_JSONL. Returns number of records written.
    """
    if not shutil.which("claude"):
        print("[opus_annotate] claude CLI not found — skipping annotation")
        return 0
    if not top_runs:
        return 0

    winner = top_runs[0]
    try:
        # This runs in a background thread that overlaps _trim_history and the
        # next cycle's run threads — all writing through the SAME sqlite3
        # connection. Concurrent use of one connection across threads corrupts
        # cursor state, so serialise this read through the store lock.
        with engine.store._lock:
            rows = engine.store.conn.execute(
                "SELECT action, ticker, sim_date, reasoning, qty, total_value "
                "FROM backtest_decisions WHERE run_id=? ORDER BY sim_date",
                (winner.run_id,),
            ).fetchall()
    except Exception as e:
        print(f"[opus_annotate] DB read failed: {e}")
        return 0

    # Build outcome lookup: (sim_date, ticker) -> forward_return_5d
    outcome_lookup: dict[tuple, float] = {}
    for o in (outcome_records or []):
        if o.get("run_id") == winner.run_id:
            outcome_lookup[(o["sim_date"], o["ticker"])] = o["forward_return_5d"]

    # Build enriched decision log — all actions, not just BUY/SELL
    decision_lines = []
    for r in rows:
        action = r["action"] or "HOLD"
        ticker = r["ticker"] or ""
        sim_date_str = r["sim_date"] or ""
        fwd_str = ""
        if ticker and sim_date_str:
            fwd = outcome_lookup.get((sim_date_str, ticker))
            if fwd is not None:
                fwd_str = f" →5d={fwd:+.1f}%"
            # Fetch scraped news snippets for this ticker/date
            news = _query_news_context(ticker, sim_date_str, n=2)
            news_str = " | NEWS: " + "; ".join(news[:2]) if news else ""
        else:
            news_str = ""
        qty_str = f" qty={r['qty']}" if r["qty"] else ""
        val_str = f" portfolio=${r['total_value']:.0f}" if r["total_value"] else ""
        reasoning_short = str(r["reasoning"] or "")[:100]
        decision_lines.append(
            f"  {sim_date_str} {action} {ticker}{qty_str}{val_str}{fwd_str}"
            f" | {reasoning_short}{news_str}"
        )

    if not decision_lines:
        return 0

    other_returns = " / ".join(f"run{r.run_id}={r.total_return_pct:+.1f}%" for r in top_runs[1:])
    prompt = f"""You are a quantitative trading analyst reviewing a backtest run for ML training purposes.

Backtest run #{winner.run_id} achieved {winner.total_return_pct:+.2f}% return over a 1-year simulation
using ML article sentiment + RSI/MACD/momentum signals. No live Claude calls were used — decisions
are pure quantitative signals. Other top runs this cycle: {other_returns or "none"}

FULL DECISION LOG (including HOLDs):
Format: date ACTION TICKER qty portfolio →5d_actual_return | reasoning | NEWS_CONTEXT
{chr(10).join(decision_lines[:60])}

Your task:
1. For EVERY decision (BUY, SELL, and HOLD), assign quality: GOOD / NEUTRAL / BAD
   - GOOD: the decision led to profit or correctly avoided loss (5d return confirms it)
   - BAD: the decision lost money or missed a clear profitable opportunity
   - NEUTRAL: outcome was mixed or the 5d return was near zero
   - For HOLDs: was holding the right call? Did a missed trade (5d return > +2%) mean BAD HOLD?
2. For BAD decisions: specify what signal should have triggered differently
3. For GOOD decisions: identify the specific signal that made it right
4. Provide an OVERALL LESSON as a concise trading rule derived from this run's outcomes

Respond as JSON with this schema (no markdown fences):
{{
  "trade_labels": [
    {{
      "sim_date": "YYYY-MM-DD",
      "action": "BUY/SELL/HOLD",
      "ticker": "...",
      "quality": "GOOD/NEUTRAL/BAD",
      "rationale": "...",
      "forward_return_5d": <number or null>,
      "signal_fix": "what signal should have changed this decision (if BAD or missed opportunity)"
    }}
  ],
  "overall_lesson": "...",
  "key_patterns": ["pattern1", "pattern2"],
  "improvement_suggestions": ["specific change to ML scoring or thresholds"]
}}"""

    try:
        r = subprocess.run(
            ["claude", "--model", "claude-opus-4-7", "--print",
             "--permission-mode", "bypassPermissions"],
            input=prompt, capture_output=True, text=True, timeout=240,
            env={**os.environ, "HOME": "/home/zeph"},
        )
    except subprocess.TimeoutExpired:
        print("[opus_annotate] timeout")
        return 0
    except Exception as e:
        print(f"[opus_annotate] subprocess error: {e}")
        return 0

    if r.returncode != 0 or not r.stdout.strip():
        print(f"[opus_annotate] claude rc={r.returncode} stderr={r.stderr.strip()[:200]!r}")
        return 0

    raw = r.stdout.strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        print("[opus_annotate] no JSON in response")
        return 0
    try:
        annotation = json.loads(m.group(0))
    except Exception as e:
        print(f"[opus_annotate] JSON parse error: {e}")
        return 0

    written = 0
    WINNER_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with WINNER_JSONL.open("a") as fh:
        # Opus occasionally emits JSON null for list/string fields — dict.get
        # returns None in that case, so use `or` to fall back to safe defaults.
        lesson = annotation.get("overall_lesson") or ""
        patterns = annotation.get("key_patterns") or []
        suggestions = annotation.get("improvement_suggestions") or []
        if lesson:
            fh.write(json.dumps({
                "cycle": cycle,
                "run_id": winner.run_id,
                "type": "opus_lesson",
                "title": f"Lesson run {winner.run_id} ({winner.total_return_pct:+.1f}%): {lesson[:120]}",
                "source": f"opus_annotation_cycle_{cycle}",
                "ai_score": 5.0,
                "urgency": 1,
                "label": "LESSON",
                "return_pct": winner.total_return_pct,
                "reasoning": lesson,
                "key_patterns": patterns,
                "improvement_suggestions": suggestions,
                "weight": 1.0,
            }) + "\n")
            written += 1

        quality_score = {"GOOD": 5.0, "NEUTRAL": 2.5, "BAD": 0.5}
        for tl in (annotation.get("trade_labels") or []):
            q = tl.get("quality", "NEUTRAL")
            action = tl.get("action", "HOLD")
            fh.write(json.dumps({
                "cycle": cycle,
                "run_id": winner.run_id,
                "type": "opus_trade_label",
                "title": f"{action} {tl.get('ticker','')} {tl.get('sim_date','')} [{q}]",
                "source": f"opus_annotation_cycle_{cycle}",
                "ai_score": quality_score.get(q, 2.5),
                "urgency": 1 if q == "GOOD" else 0,
                "label": action,
                "ticker": tl.get("ticker", ""),
                "sim_date": tl.get("sim_date", ""),
                "reasoning": tl.get("rationale", ""),
                "signal_fix": tl.get("signal_fix", ""),
                "forward_return_5d": tl.get("forward_return_5d"),
                "return_pct": winner.total_return_pct,
                "quality": q,
                "weight": 1.0 if q == "GOOD" else (0.5 if q == "NEUTRAL" else 0.1),
            }) + "\n")
            written += 1

    print(f"[opus_annotate] wrote {written} annotation records for run {winner.run_id} "
          f"({len(decision_lines)} decisions reviewed)")
    return written


def _inject_and_train() -> str:
    """Inject winner JSONL into article store then retrain. Returns short status string."""
    import hashlib
    import zlib

    DB_PATH = DIGITAL_INTERN_ARTICLES_DB

    def _compress(text: str) -> bytes:
        return zlib.compress(text.encode("utf-8", errors="replace"), level=6)

    def _aid(url: str, title: str) -> str:
        return hashlib.sha256(f"{url}||{title}".encode()).hexdigest()[:20]

    if not WINNER_JSONL.exists():
        return "no jsonl"

    # Cap the JSONL read to the most recent records — older ones are already
    # in articles.db (INSERT OR IGNORE de-dups by id), so re-reading them every
    # cycle wastes memory and IO as the file grows without bound.
    _MAX_INJECT_RECORDS = 10000
    # winner_training.jsonl accumulates forever; read_text() would pull the whole
    # (eventually multi-hundred-MB) file into memory every cycle. Stream it line
    # by line through a bounded deque so peak memory is capped at the tail we use.
    try:
        from collections import deque
        with WINNER_JSONL.open("r") as _fh:
            recent = list(deque((ln for ln in _fh if ln.strip()),
                                maxlen=_MAX_INJECT_RECORDS))
    except Exception as e:
        return f"jsonl read err: {e}"
    # Per-line parse so a single corrupt line doesn't drop the whole batch
    records: list[dict] = []
    for l in recent:
        try:
            records.append(json.loads(l))
        except Exception:
            pass

    now = datetime.now(timezone.utc).isoformat()
    aconn = None
    try:
        aconn = sqlite3.connect(DB_PATH, timeout=15)
        inserted = 0
        for rec in records:
            # `.get(k, default)` only substitutes the default when the key is
            # ABSENT — an explicit JSON `null` value still returns None, and
            # `float(None)` raises TypeError. A single such line in
            # winner_training.jsonl would abort the whole injection batch via
            # the outer `except` (returning "inject err: …"), so ArticleNet
            # never retrains that cycle. `or` coerces None/0/"" to the safe
            # default, matching the hardening idiom already used in
            # backtest._ml_decide and _opus_annotate.
            ai = float(rec.get("ai_score") or 0.0)
            w = float(rec.get("weight") or 1.0)
            eff = min(10.0, ai * w)
            title = rec.get("title", "")
            ticker = rec.get("ticker", "")
            reasoning = rec.get("reasoning", "")
            sim_date = rec.get("sim_date", "")
            label = rec.get("label", "")
            run_id = rec.get("run_id", 0)
            if not title:
                continue
            url = f"backtest://run_{run_id}/{sim_date}/{label}/{ticker}"
            aid = _aid(url, title)
            full_text = f"[{ticker}] {title}. {reasoning}"
            aconn.execute(
                "INSERT OR IGNORE INTO articles "
                "(id,url,title,source,published,kw_score,ai_score,urgency,first_seen,cycle,full_text) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (aid, url, title, f"backtest_run_{run_id}", sim_date or now[:10],
                 eff, eff, 0, now, rec.get("cycle", 0),
                 _compress(full_text)),
            )
            if aconn.execute("SELECT changes()").fetchone()[0]:
                inserted += 1
        aconn.commit()
    except Exception as e:
        return f"inject err: {e}"
    finally:
        if aconn is not None:
            try:
                aconn.close()
            except Exception:
                pass

    # Now trigger actual training
    try:
        r = subprocess.run(
            ["python3", "-c",
             "import sys; sys.path.insert(0,'.'); "
             "from storage.article_store import ArticleStore; "
             "from ml.trainer import train; "
             "s=ArticleStore(); res=train(s,force=True); "
             "print(f\"trainer n={res.get('n',0)} loss={res.get('final_loss',0):.4f} "
             "val={res.get('val_loss',0):.4f}\")"],
            cwd="/home/zeph/digital-intern",
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            out = (r.stdout or "").strip().splitlines()
            return f"injected {inserted} new | {out[-1] if out else 'ok'}"
        return f"trainer rc={r.returncode} injected={inserted}"
    except subprocess.TimeoutExpired:
        return f"trainer timeout (injected {inserted})"
    except Exception as e:
        return f"trainer exc: {type(e).__name__}"


def _try_train_ml() -> str:
    return _inject_and_train()


def _llm_annotate_outcomes(
    engine,
    winner: "BacktestRun",
    loser: "BacktestRun",
    outcome_records: list[dict],
    cycle: int,
) -> list[dict]:
    """Call LLM to annotate best/worst run trades with quality labels.

    Endorsed trades get llm_quality_label=+1 (3x training weight).
    Condemned trades get llm_quality_label=-1 (0.1x training weight).
    Unlabeled records get llm_quality_label=0 (1x weight).

    Returns outcome_records with llm_quality_label filled in.
    """
    try:
        import anthropic
    except ImportError:
        return outcome_records

    for r in outcome_records:
        r.setdefault("llm_quality_label", 0)

    def _summarize_run(run, label: str, max_trades: int = 5) -> str:
        trades = []
        run_records = [r for r in outcome_records if r.get("run_id") == run.run_id][:max_trades]
        for r in run_records:
            trades.append(
                f"  {r.get('ticker','?')} {r.get('action','BUY')}: "
                f"ml_score={r.get('ml_score', 0):.1f}, "
                f"rsi={r.get('rsi') or '?'}, "
                f"5d_return={r.get('forward_return_5d', 0):.1f}%"
            )
        return f"{label} (total return: {run.total_return_pct:.1f}%):\n" + "\n".join(trades or ["  (no trades)"])

    winner_summary = _summarize_run(winner, "BEST RUN")
    loser_summary = _summarize_run(loser, "WORST RUN")

    prompt = f"""You are reviewing trades from a paper-trading backtest system.

{winner_summary}

{loser_summary}

For each trade listed above, output one line:
TICKER ACTION: ENDORSE or CONDEMN [one sentence reason based on whether this trade reflects good news analysis and momentum alignment]

Be concise. Only output the labeled lines, no intro text."""

    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        annotation_text = resp.content[0].text.strip()
        print(f"[continuous] LLM annotation cycle {cycle}:\n{annotation_text}")

        # The LLM only reviewed trades from the best and worst runs (see the
        # prompt above). Restrict label application to those two run_ids —
        # matching on (ticker, action) alone would leak a verdict derived from
        # one run's trade onto identically-named trades in the three unreviewed
        # middle runs, corrupting their training sample weights.
        allowed_run_ids = {winner.run_id, loser.run_id}
        for line in annotation_text.splitlines():
            # [\w\-]* (not +) so single-letter tickers like V are not dropped.
            m = re.match(r"(\w[\w\-]*)\s+(BUY|SELL|HOLD)[:\s]+(ENDORSE|CONDEMN)", line.upper())
            if not m:
                continue
            ticker, action, verdict = m.group(1), m.group(2), m.group(3)
            label = 1 if verdict == "ENDORSE" else -1
            for r in outcome_records:
                if r.get("run_id") not in allowed_run_ids:
                    continue
                if (str(r.get("ticker", "")).upper() == ticker and
                        str(r.get("action", "")).upper() == action):
                    r["llm_quality_label"] = label

    except Exception as e:
        print(f"[continuous] LLM annotation failed: {e}")

    return outcome_records


def _post_discord(message: str) -> None:
    """Best-effort Discord post via openclaw. Silent on failure — never raise."""
    if not shutil.which("openclaw"):
        return
    try:
        subprocess.run(
            ["openclaw", "message", "send",
             "--channel", "discord",
             "--target", DISCORD_CHANNEL,
             "--message", message],
            capture_output=True, timeout=20,
        )
    except Exception as e:
        print(f"[discord] post failed: {e}")


def _run_validation_async(engine, cycle: int, win_start: date, win_end: date,
                          articles_db: str | None) -> None:
    """Run the full validation suite (label audit + permutation test) and
    persist results to ``data/validation_results.json``.

    Designed to be invoked from a background daemon thread — a permutation
    test runs ``VALIDATION_PERMUTATIONS`` full backtests serially, which can
    take 20+ minutes. Running this synchronously would block the next
    backtest cycle indefinitely.

    The function never raises — every step has a best-effort try/except so
    a validation failure cannot kill the loop.
    """
    print(f"[validation] cycle {cycle} starting (this runs in background)")
    out: dict = {
        "cycle": cycle,
        "timestamp": _now(),
        "window": f"{win_start}→{win_end}",
        "permutation_test": None,
        "label_audit": None,
    }

    # 1. Label contamination audit — fast, just SQL.
    try:
        from paper_trader.validation import audit_label_contamination
        if articles_db:
            audit = audit_label_contamination(articles_db, win_start, win_end)
            out["label_audit"] = audit
            if audit.get("contamination_rate", 0.0) > 0.5:
                _post_discord(
                    f"WARN: high label contamination "
                    f"({audit['contamination_rate']:.0%}) "
                    f"in window {win_start}→{win_end}. "
                    f"Backtest returns may be inflated."
                )
    except Exception as e:
        out["label_audit"] = {"error": str(e)}

    # 2. Permutation test — slow.
    try:
        from paper_trader.validation import run_permutation_test
        import tempfile
        with tempfile.TemporaryDirectory(prefix="perm_cycle_") as tmp:
            perm = run_permutation_test(
                engine,
                seed=cycle,
                n_permutations=VALIDATION_PERMUTATIONS,
                isolated_db_path=Path(tmp) / "perm.db",
            )
        out["permutation_test"] = perm
        v = perm.get("verdict")
        if v == "WORSE_THAN_RANDOM":
            _post_discord(
                f"ALERT: permutation test cycle {cycle} — strategy WORSE than "
                f"random signal ordering "
                f"(p={perm.get('p_value', 0):.2f}, "
                f"z={perm.get('z_score', 0):.2f}). "
                f"Signals may not carry real predictive value."
            )
        elif v == "SIGNIFICANT":
            _post_discord(
                f"OK: permutation test cycle {cycle} PASSED — "
                f"p={perm.get('p_value', 0):.3f}, "
                f"z={perm.get('z_score', 0):.1f}. "
                f"Signal time-ordering carries real value."
            )
    except Exception as e:
        out["permutation_test"] = {"error": str(e)}

    # 3. Persist (capped tail).
    try:
        VALIDATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if VALIDATION_RESULTS_PATH.exists():
            try:
                existing = json.loads(VALIDATION_RESULTS_PATH.read_text())
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        existing.append(out)
        existing = existing[-VALIDATION_RESULTS_KEEP:]
        # Atomic write — torn JSON would break the dashboard's /api/validation.
        tmp_p = VALIDATION_RESULTS_PATH.with_suffix(".json.tmp")
        tmp_p.write_text(json.dumps(existing, indent=2))
        tmp_p.replace(VALIDATION_RESULTS_PATH)
        print(f"[validation] cycle {cycle} done — wrote {len(existing)} entries")
    except Exception as e:
        print(f"[validation] persist failed: {e}")


_STOP = False


def _handle_sig(_signum, _frame) -> None:
    global _STOP
    _STOP = True
    print(f"\n[continuous] {_now()} signal received — stopping after current cycle")


def main() -> None:
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    print(f"[continuous] {_now()} starting ENSEMBLE-COMMITTEE loop "
          f"({RUNS_PER_CYCLE} runs/cycle, keep last {KEEP_LAST_RUNS}, "
          f"cooldown {COOLDOWN_SECONDS}s, variable {MIN_WINDOW_YEARS}–{MAX_WINDOW_YEARS}yr "
          f"windows in {EARLIEST_WINDOW_START}–present)")

    cycle = 0
    while not _STOP:
        cycle += 1
        # Each cycle picks its own random window. Engine is recreated because
        # PriceCache and the per-window volume cache are window-keyed; reusing
        # the previous engine would silently mix cache state from a different
        # date range.
        cycle_seed = int(time.time()) ^ (cycle * 2654435761) & 0xFFFFFFFF
        win_start, win_end = _pick_window(cycle_seed)
        win_years = (win_end - win_start).days / 365.0
        print(f"\n[continuous] {_now()} ─── cycle {cycle} window: "
              f"{win_start} → {win_end} ({win_years:.1f}yr, seed={cycle_seed}) ───")

        try:
            engine = BacktestEngine(start=win_start, end=win_end)
        except Exception as e:
            print(f"[continuous] engine init failed for {win_start}→{win_end}: {e}")
            traceback.print_exc()
            # Sleep briefly then move to next cycle; a yfinance hiccup shouldn't
            # kill the loop.
            time.sleep(30)
            continue

        # Optional pre-warmer for historical news. Background by default so
        # backtests proceed on quant signals while news fills in.
        # `tickers=None` lets the collector pick its own narrow SEC ticker set
        # (~17 names). Passing the full 117-ticker watchlist would issue an
        # SEC request for each on every cycle, wasting rate budget on names
        # that aren't tracked by the signal pipeline anyway.
        try:
            from paper_trader.historical_collector import prewarm_window
            prewarm_window(win_start, win_end, tickers=None, background=True)
        except Exception as e:
            # Pre-warmer is best-effort — failure must not stop a cycle.
            print(f"[continuous] prewarm dispatch failed: {e}")

        start_id = _next_run_id(engine)
        t0 = time.time()
        print(f"[continuous] cycle {cycle} runs {start_id}..{start_id + RUNS_PER_CYCLE - 1}")

        # Refresh local article cache so the engine sees news digital-intern
        # has written since the last engine init. (Engine is fresh, but this
        # also covers articles written during the current cycle's lifetime.)
        try:
            n_arts = engine.refresh_local_articles()
            print(f"[continuous] refreshed local_news: {n_arts} articles")
        except Exception as e:
            print(f"[continuous] refresh_local_articles failed: {e}")

        results: list[BacktestRun] = []
        try:
            results = engine.run_all(RUNS_PER_CYCLE, start_run_id=start_id) or []
        except Exception as e:
            print(f"[continuous] {_now()} cycle {cycle} crashed: {e}")
            traceback.print_exc()

        winner = None
        top_runs: list[BacktestRun] = []
        if results:
            sorted_results = sorted(results, key=lambda r: r.total_return_pct, reverse=True)
            # Only include runs that beat a flat 0% return (filter out pure losers)
            top_runs = [r for r in sorted_results[:TOP_RUNS_TO_TRAIN]
                        if r.total_return_pct > 0]
            if not top_runs:
                top_runs = sorted_results[:1]  # always train on best even if negative
            winner = top_runs[0]
            try:
                _append_top_decisions(engine, top_runs, cycle)
            except Exception as e:
                print(f"[continuous] top-runs append failed: {e}")

            # Compute 5d forward return outcomes for every BUY/SELL decision
            # across ALL runs (winners and losers) so the scorer learns from
            # losing decisions too — training only on top runs caused survivorship
            # bias and an overly optimistic model.
            outcome_records: list[dict] = []
            try:
                outcome_records = _compute_decision_outcomes(engine, sorted_results)
                print(f"[continuous] computed {len(outcome_records)} decision outcomes "
                      f"from {len(sorted_results)} runs")
            except Exception as e:
                print(f"[continuous] outcome compute failed: {e}")

            # LLM annotation: endorse/condemn individual trades to improve training signal
            if outcome_records and winner and sorted_results:
                loser = sorted_results[-1]
                try:
                    outcome_records = _llm_annotate_outcomes(
                        engine, winner, loser, outcome_records, cycle
                    )
                    endorsed = sum(1 for r in outcome_records if r.get("llm_quality_label") == 1)
                    condemned = sum(1 for r in outcome_records if r.get("llm_quality_label") == -1)
                    print(f"[continuous] LLM labels: {endorsed} endorsed, {condemned} condemned")
                except Exception as e:
                    print(f"[continuous] LLM annotation outer failed: {e}")

            # Train DecisionScorer on accumulated outcomes (accumulate across cycles)
            _all_outcomes_path = ROOT / "data" / "decision_outcomes.jsonl"
            if outcome_records:
                try:
                    _all_outcomes_path.parent.mkdir(parents=True, exist_ok=True)
                    with _all_outcomes_path.open("a") as _of:
                        for _o in outcome_records:
                            _of.write(json.dumps(_o) + "\n")
                except Exception as e:
                    print(f"[continuous] outcome append failed: {e}")

            # Load most recent outcomes and retrain scorer.
            # Capped at MAX_OUTCOMES_FOR_TRAINING — older outcomes describe a stale
            # signal regime and the file would otherwise grow unbounded.
            try:
                all_lines: list[str] = []
                if _all_outcomes_path.exists():
                    all_lines = [l for l in _all_outcomes_path.read_text().splitlines() if l.strip()]
                # Trim the file on disk when it grows past 2× the training cap so
                # it doesn't accumulate indefinitely across cycles. The model only
                # ever sees the tail anyway.
                if len(all_lines) > MAX_OUTCOMES_FOR_TRAINING * 2:
                    kept = all_lines[-MAX_OUTCOMES_FOR_TRAINING:]
                    # Atomic rewrite: a torn write (process killed mid-truncate)
                    # would corrupt or empty the accumulated outcomes file —
                    # permanently losing the scorer's training history. Write to
                    # a temp file then atomically replace.
                    _tmp = _all_outcomes_path.with_suffix(".jsonl.tmp")
                    _tmp.write_text("\n".join(kept) + "\n")
                    _tmp.replace(_all_outcomes_path)
                    print(f"[continuous] trimmed outcomes file "
                          f"{len(all_lines)} → {len(kept)} lines")
                    all_lines = kept
                all_outcomes: list[dict] = []
                for _line in all_lines:
                    try:
                        all_outcomes.append(json.loads(_line))
                    except Exception:
                        pass
                all_outcomes = all_outcomes[-MAX_OUTCOMES_FOR_TRAINING:]
                scorer_status = _train_decision_scorer(all_outcomes)
                print(f"[continuous] {scorer_status}")
                # Reset the singleton under its lock so next cycle reloads the
                # freshly-trained scorer. Bare assignment races with any backtest
                # thread mid-call to _get_decision_scorer().
                import paper_trader.backtest as _bt
                with _bt._DECISION_SCORER_LOCK:
                    _bt._DECISION_SCORER = None
            except Exception as e:
                print(f"[continuous] scorer train failed: {e}")

            # Opus 4.7 annotation in background thread — don't block next cycle
            import threading as _threading
            _threading.Thread(
                target=_opus_annotate, args=(engine, top_runs, cycle, outcome_records),
                daemon=True, name=f"opus-annotate-{cycle}"
            ).start()

        ml_status = _try_train_ml() if winner else "no winner"
        print(f"[continuous] ml: {ml_status}")

        # Backtest results are silent — check the dashboard at :8090

        # IMPORTANT: trim history BEFORE dispatching validation. The
        # validation thread mutates `engine.store` (swaps in an isolated
        # store for permutation runs), so anything that operates on the
        # real backtest.db via `engine.store` must run first. Validation
        # is the *last* thing scheduled on `engine` per cycle.
        try:
            deleted = _trim_history(engine, keep=KEEP_LAST_RUNS)
            if deleted:
                print(f"[continuous] trimmed {deleted} old runs "
                      f"(keeping last {KEEP_LAST_RUNS})")
        except Exception as e:
            print(f"[continuous] trim failed: {e}")

        # Validation suite — runs in a background thread so the next cycle
        # isn't blocked by the ~25-min permutation test. Must be the LAST
        # thing scheduled on `engine` because the validation function
        # mutates `engine.store` (swaps in an isolated store for permutation
        # runs); any subsequent code reading `engine.store` would silently
        # read the empty isolated DB.
        if cycle % VALIDATION_EVERY_N_CYCLES == 0:
            try:
                from paper_trader.backtest import LOCAL_ARTICLES_DB
                articles_db = (str(LOCAL_ARTICLES_DB)
                               if LOCAL_ARTICLES_DB.exists() else None)
                import threading as _threading
                _threading.Thread(
                    target=_run_validation_async,
                    args=(engine, cycle, win_start, win_end, articles_db),
                    daemon=True, name=f"validation-{cycle}",
                ).start()
                print(f"[continuous] validation cycle {cycle} dispatched (background)")
            except Exception as e:
                print(f"[continuous] validation dispatch failed: {e}")

        elapsed = time.time() - t0
        if winner:
            print(f"[continuous] {_now()} cycle {cycle} done in {elapsed/60:.1f}min. "
                  f"Best run {winner.run_id} {winner.total_return_pct:+.2f}%")
        else:
            print(f"[continuous] {_now()} cycle {cycle} done in {elapsed/60:.1f}min")

        if _STOP:
            break

        print(f"[continuous] sleeping {COOLDOWN_SECONDS}s before cycle {cycle + 1}")
        slept = 0
        while slept < COOLDOWN_SECONDS and not _STOP:
            chunk = min(2, COOLDOWN_SECONDS - slept)
            time.sleep(chunk)
            slept += chunk

    print(f"[continuous] {_now()} loop stopped after {cycle} cycle(s)")
    sys.exit(0)


if __name__ == "__main__":
    main()
