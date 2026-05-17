"""Walk-forward validation, permutation testing, and label-contamination
auditing for the backtest engine.

These tools provide rigorous evidence that observed backtest returns are
driven by real predictive signal — not by overfitting, hindsight-contaminated
labels, or survivorship bias.

Three distinct checks live here:

  1. ``audit_label_contamination`` — what fraction of articles in a window
     have ``first_seen`` lagging ``published`` by more than the staleness
     threshold? High contamination means historical backtests were eating
     hindsight-labeled signals.

  2. ``run_permutation_test`` — shuffle article dates within the backtest
     window and re-run. If the original return is statistically higher
     than the shuffled distribution, the time-ordering of signals carries
     real predictive value. This is the gold-standard "is my edge real?"
     check.

  3. ``run_walk_forward_validation`` — split the period into N folds, train
     on folds [0..i-1], test on fold i. If in-sample dwarfs out-of-sample
     across folds, the system is overfitting.

All three are designed to leave the live ``backtest.db`` untouched —
permutation runs use an isolated store with negative ``run_id`` keys.
"""
from __future__ import annotations

import json
import random
import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Helpers shared across the three checks
# ─────────────────────────────────────────────────────────────────────────


def _parse_published_date(raw):
    """Parse a published-timestamp string. Returns date or None.

    `published` in articles.db is a mix of ISO ("2024-06-15T...") and
    RFC822 ("Wed, 14 May 2026 ...") because different collectors normalize
    differently. A naive SQL `BETWEEN` filter silently drops RFC822 rows
    (they lex-sort before any ISO date), so all date filtering happens in
    Python here.
    """
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            return dt.date()
    except Exception:
        pass
    try:
        return date.fromisoformat(str(raw)[:10])
    except Exception:
        return None


def _make_isolated_store(db_path: Path):
    """Create a BacktestStore at `db_path` (NOT the live BACKTEST_DB).

    The permutation test invokes `BacktestEngine.run_one()` repeatedly,
    and that method writes through `engine.store`. Without this isolation,
    every permutation pollutes the live dashboard with junk runs.
    """
    from paper_trader.backtest import BacktestStore
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return BacktestStore(db_path)


# ─────────────────────────────────────────────────────────────────────────
# 1. Label contamination audit
# ─────────────────────────────────────────────────────────────────────────


def audit_label_contamination(
    articles_db_path: str,
    window_start: date,
    window_end: date,
    staleness_days: int = 60,
) -> dict:
    """Audit how many articles in a window had hindsight-contaminated labels.

    "Contaminated" means `first_seen - published > staleness_days`: the
    article was collected long after publication, so any Claude-assigned
    `ai_score` may reflect knowledge of what happened next.

    Returns a dict with overall stats and a per-source breakdown so callers
    can identify which collectors are responsible for most of the leakage.
    """
    p = Path(articles_db_path)
    if not p.exists():
        return {"error": f"DB not found: {articles_db_path}",
                "total_articles": 0, "contaminated_count": 0,
                "contamination_rate": 0.0, "verdict": "UNKNOWN", "sources": {}}

    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=10.0)
    try:
        # Fetch *all* candidates and filter dates in Python — see
        # `_parse_published_date` for why a SQL BETWEEN cannot be trusted.
        rows = conn.execute(
            "SELECT url, published, first_seen, ai_score, kw_score, source "
            "FROM articles "
            "WHERE published IS NOT NULL AND first_seen IS NOT NULL "
            "AND (url IS NULL OR url NOT LIKE 'backtest://%') "
            "AND (source IS NULL OR (source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%'))"
        ).fetchall()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    total = 0
    contaminated = 0
    source_stats: dict[str, dict] = {}

    for url, published, first_seen, ai_score, kw_score, source in rows:
        pub_d = _parse_published_date(published)
        if pub_d is None:
            continue
        if pub_d < window_start or pub_d > window_end:
            continue
        total += 1
        seen_d = _parse_published_date(first_seen)
        is_hindsight = False
        if seen_d is not None:
            days_lag = (seen_d - pub_d).days
            is_hindsight = days_lag > staleness_days
        if is_hindsight:
            contaminated += 1
        src_key = (source or "unknown").split("/")[0]
        s = source_stats.setdefault(
            src_key, {"total": 0, "contaminated": 0, "has_ai_score": 0}
        )
        s["total"] += 1
        if is_hindsight:
            s["contaminated"] += 1
        if ai_score is not None and float(ai_score) > 0:
            s["has_ai_score"] += 1

    rate = contaminated / total if total else 0.0
    verdict = (
        "HIGH_CONTAMINATION" if rate > 0.5
        else "MODERATE" if rate > 0.2
        else "LOW"
    )

    return {
        "total_articles": total,
        "contaminated_count": contaminated,
        "contamination_rate": rate,
        "clean_rate": 1.0 - rate,
        "sources": source_stats,
        "verdict": verdict,
        "staleness_days": staleness_days,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────
# 2. Permutation test
# ─────────────────────────────────────────────────────────────────────────


_PERM_RUN_ID_BASE = -1_000_000  # well below any real run_id; trivially identifiable


def _shuffle_news_dates(news: dict[str, list[dict]],
                        rng: random.Random) -> dict[str, list[dict]]:
    """Reassign each day's article list to a random other day in the same window.

    The article *content* is preserved; only the date assignment is permuted.
    Quant signals are unaffected because they derive from prices, not news.
    """
    dates = list(news.keys())
    shuffled = dates.copy()
    rng.shuffle(shuffled)
    return {shuffled[i]: news[dates[i]] for i in range(len(dates))}


def run_permutation_test(
    engine,
    seed: int = 42,
    n_permutations: int = 20,
    isolated_db_path: Path | None = None,
) -> dict:
    """Permutation test for backtest signal integrity.

    Steps:
      1. Run one real backtest, record `total_return_pct`.
      2. For each of `n_permutations`:
         a. Shuffle the dates in `engine._local_news` within the window.
         b. Re-run the same strategy, same seed.
         c. Record the permuted return.
      3. Compare original vs. permuted distribution.

    Returns a verdict dict with `p_value`, `z_score`, and SIGNIFICANT /
    INCONCLUSIVE / WORSE_THAN_RANDOM classification.

    All runs (real + permuted) write to an *isolated* BacktestStore so
    the live ``backtest.db`` is never touched. The engine's store is
    swapped back before return.
    """
    rng = random.Random(seed)

    # Snapshot original state we'll need to restore.
    original_news = engine._local_news
    original_store = engine.store

    # Use a temp DB for all permutation writes — never pollute the live dashboard.
    if isolated_db_path is None:
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp(prefix="perm_test_"))
        isolated_db_path = tmp_dir / "perm.db"

    isolated = _make_isolated_store(isolated_db_path)
    engine.store = isolated

    permuted_returns: list[float] = []
    original_return: float | None = None

    try:
        # 1. Real run with original news
        real = engine.run_one(_PERM_RUN_ID_BASE, seed=seed)
        original_return = float(real.total_return_pct)

        # 2. Permutations
        for i in range(n_permutations):
            engine._local_news = _shuffle_news_dates(original_news, rng)
            run_id = _PERM_RUN_ID_BASE - 1 - i
            try:
                run = engine.run_one(run_id, seed=seed)
                permuted_returns.append(float(run.total_return_pct))
            except Exception as e:
                print(f"[permutation] run {i} failed: {e}")
    finally:
        engine._local_news = original_news
        engine.store = original_store

    if not permuted_returns:
        return {
            "error": "all permutations failed",
            "original_return": original_return,
            "n_permutations": 0,
            "verdict": "UNKNOWN",
        }

    arr = np.array(permuted_returns, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    p_value = float(np.mean(arr >= original_return))
    z_score = float((original_return - mean) / (std + 1e-9))

    if original_return < mean:
        verdict = "WORSE_THAN_RANDOM"
    elif p_value < 0.05:
        verdict = "SIGNIFICANT"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "original_return": original_return,
        "permuted_mean": mean,
        "permuted_std": std,
        "permuted_min": float(arr.min()),
        "permuted_max": float(arr.max()),
        "p_value": p_value,
        "z_score": z_score,
        "n_permutations": len(permuted_returns),
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────
# 3. Walk-forward validation
# ─────────────────────────────────────────────────────────────────────────


def _compute_fold_windows(
    start: date, end: date, fold_years: int = 1
) -> list[dict]:
    """Compute the (train_end, test_start, test_end) windows for walk-forward.

    Splits [start, end] into fixed `fold_years`-wide chunks. Returns a list
    of fold descriptors for each OOS test window. Fold 0 is reserved as
    pure training history (no test), so the first OOS fold is fold 1.
    """
    fold_days = fold_years * 365
    total_days = (end - start).days
    n_chunks = total_days // fold_days
    if n_chunks < 2:
        return []

    out: list[dict] = []
    for i in range(1, n_chunks):
        train_end = start + timedelta(days=i * fold_days)
        test_start = train_end
        test_end = min(start + timedelta(days=(i + 1) * fold_days), end)
        out.append({
            "fold": i,
            "train_start": start.isoformat(),
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
        })
    return out


def run_walk_forward_validation(
    start: date, end: date, fold_years: int = 1,
    isolated_db_path: Path | None = None,
) -> dict:
    """Walk-forward validation over [start, end] using `fold_years`-wide folds.

    For each OOS fold, builds a fresh BacktestEngine over the test window
    and runs one backtest. Reports OOS returns vs. SPY benchmark.

    NB: this is slow — each fold creates a PriceCache and runs ~250 sim
    days. Run as a background task or from the audit script, never inline
    on the continuous loop.
    """
    folds = _compute_fold_windows(start, end, fold_years=fold_years)
    if not folds:
        return {"error": "Period too short for walk-forward validation",
                "n_folds": 0, "verdict": "UNKNOWN"}

    from paper_trader.backtest import BacktestEngine

    if isolated_db_path is None:
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp(prefix="walkfwd_"))
        isolated_db_path = tmp_dir / "walkfwd.db"

    isolated = _make_isolated_store(isolated_db_path)

    results: list[dict] = []
    oos_returns: list[float] = []
    spy_returns: list[float] = []

    for f in folds:
        test_start = date.fromisoformat(f["test_start"])
        test_end = date.fromisoformat(f["test_end"])
        try:
            engine = BacktestEngine(start=test_start, end=test_end)
            engine.store = isolated
            run_id = _PERM_RUN_ID_BASE - 100_000 - f["fold"]
            run = engine.run_one(run_id, seed=f["fold"] * 17)
            oos_ret = float(run.total_return_pct)
            spy_ret = float(run.spy_return_pct)
            f_result = {
                **f,
                "oos_return": oos_ret,
                "spy_return": spy_ret,
                "vs_spy": oos_ret - spy_ret,
                "n_trades": run.n_trades,
            }
            results.append(f_result)
            oos_returns.append(oos_ret)
            spy_returns.append(spy_ret)
            print(f"[walk-forward] fold {f['fold']} {test_start}→{test_end}: "
                  f"OOS {oos_ret:+.1f}% vs SPY {spy_ret:+.1f}%")
        except Exception as e:
            results.append({**f, "error": str(e)})
            print(f"[walk-forward] fold {f['fold']} failed: {e}")

    if not oos_returns:
        return {"folds": results, "n_folds": len(results),
                "error": "all folds failed", "verdict": "UNKNOWN"}

    arr = np.array(oos_returns, dtype=np.float64)
    spy_arr = np.array(spy_returns, dtype=np.float64)
    consistency = float(np.mean(arr > 0))
    mean_vs_spy = float((arr - spy_arr).mean())

    verdict = (
        "ROBUST" if consistency >= 0.6 and mean_vs_spy > 0
        else "OVERFIT" if consistency < 0.4
        else "MIXED"
    )

    return {
        "folds": results,
        "n_folds": len(results),
        "mean_oos_return": float(arr.mean()),
        "mean_spy_return": float(spy_arr.mean()),
        "mean_vs_spy": mean_vs_spy,
        "consistency": consistency,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────
# Temporal split for DecisionScorer training
# ─────────────────────────────────────────────────────────────────────────


def split_outcomes_temporal(
    records: list[dict],
    oos_fraction: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    """Sort outcome records by sim_date; hold out the most recent fraction as OOS.

    The DecisionScorer's built-in `train_test_split` is *random* (random_state=42),
    which leaks future information into validation when records span time. Use this
    to get a proper temporal holdout: train on history, evaluate on what came after.

    Returns (train_records, oos_records). Records without a parseable sim_date
    are placed in the training set (better than discarding training data).
    """
    if not records:
        return [], []
    if len(records) < 5:
        # Too few to split meaningfully — give all to training.
        return list(records), []

    def _key(r):
        s = r.get("sim_date") or r.get("date") or "0000-01-01"
        try:
            return date.fromisoformat(str(s)[:10])
        except Exception:
            return date(1, 1, 1)

    sorted_recs = sorted(records, key=_key)
    n_oos = max(1, int(len(sorted_recs) * oos_fraction))
    if n_oos >= len(sorted_recs):
        return list(sorted_recs), []
    train = sorted_recs[: len(sorted_recs) - n_oos]
    oos = sorted_recs[len(sorted_recs) - n_oos:]
    return train, oos


def evaluate_scorer_oos(scorer, oos_records: list[dict]) -> dict:
    """Compute RMSE of `scorer` predictions against actual `forward_return_5d`
    on a held-out set. Returns ``{"n": int, "rmse": float|None}``."""
    if not oos_records:
        return {"n": 0, "rmse": None}
    if not getattr(scorer, "is_trained", False):
        return {"n": len(oos_records), "rmse": None,
                "error": "scorer not trained"}

    from paper_trader.ml.decision_scorer import _to_float

    preds: list[float] = []
    actuals: list[float] = []
    for r in oos_records:
        try:
            p = scorer.predict(
                ml_score=_to_float(r.get("ml_score"), 0.0),
                rsi=r.get("rsi"),
                macd=r.get("macd"),
                mom5=r.get("mom5"),
                mom20=r.get("mom20"),
                regime_mult=_to_float(r.get("regime_mult"), 1.0),
                ticker=str(r.get("ticker") or ""),
                vol_ratio=r.get("vol_ratio"),
                bb_pos=r.get("bb_position"),
                news_urgency=r.get("news_urgency"),
                news_article_count=r.get("news_article_count"),
            )
            actual = _to_float(r.get("forward_return_5d"), 0.0)
            action = str(r.get("action") or "BUY").upper()
            # Mirror train_scorer's sign-flip convention so the OOS error
            # measures the same thing the training loss minimized.
            if action == "SELL":
                actual = -actual
            preds.append(float(p))
            actuals.append(float(actual))
        except Exception:
            continue

    if not preds:
        return {"n": 0, "rmse": None, "error": "no predictable records"}

    arr_p = np.array(preds, dtype=np.float64)
    arr_a = np.array(actuals, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((arr_p - arr_a) ** 2)))
    return {"n": len(preds), "rmse": rmse}
