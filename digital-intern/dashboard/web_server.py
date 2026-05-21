"""Flask web dashboard — public, bound to 0.0.0.0:8080.

The existing FastAPI dashboard (dashboard/server.py, port 8765) is the rich
ops view that runs as its own systemd unit. This module is a smaller,
read-only public-facing dashboard wired into the daemon as a worker thread so
external users can see briefings, portfolio P&L, and top signals without
running anything extra.

API key (``WEB_API_KEY``) protects ``/api/*`` only; the HTML dashboard at
``/`` is public.
"""
from __future__ import annotations

import os
import json
import re
import sqlite3
import zlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

from core.claude_cli import claude_call as _claude_cli_call

BASE_DIR = Path(__file__).resolve().parent.parent

# Resolved lazily so importing this module doesn't require an instantiated store.
_store = None
_log = None


def _logger():
    global _log
    if _log is None:
        try:
            from core.logger import get_logger
            _log = get_logger("web_server")
        except Exception:
            import logging
            _log = logging.getLogger("web_server")
    return _log


def _store_handle():
    """Return a shared ArticleStore — the daemon passes one in via init_app()."""
    return _store


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _briefings_from_log(limit: int = 10) -> list[dict]:
    """Pull recent heartbeat briefings from the structured JSONL log."""
    log_path = BASE_DIR / "logs" / "structured.jsonl"
    if not log_path.exists():
        return []
    try:
        size = log_path.stat().st_size
        chunk = min(size, 512 * 1024)
        with log_path.open("rb") as f:
            f.seek(max(0, size - chunk))
            data = f.read()
    except Exception:
        return []
    out: list[dict] = []
    for raw in reversed(data.splitlines()):
        try:
            ln = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            continue
        msg = ln.get("msg", "")
        if "[heartbeat]" in msg and ("sent" in msg.lower() or "generating" in msg.lower()):
            out.append({
                "ts": ln.get("ts", ""),
                "msg": msg,
            })
            if len(out) >= limit:
                break
    return out


def _ro_query(sql: str, params: tuple = ()) -> list[tuple]:
    """Run a read-only query on a dedicated short-lived sqlite connection.

    The dashboard runs threaded (``run_server`` → ``app.run(threaded=True)``)
    but the daemon's ``ArticleStore.conn`` is a *single* ``sqlite3.Connection``
    shared across ~30 writer threads (``check_same_thread=False``). sqlite3
    connections are not safe for concurrent use: a dashboard read racing a
    writer's implicit ``conn.execute("SELECT changes()")`` inside
    ``insert_batch`` returned a wrong-shaped 1-tuple where the 9-column row
    was expected, crashing ``/api/articles`` with ``IndexError`` (observed
    10× in production; ``IndexError`` is not a ``sqlite3.Error`` so the
    caller's ``except`` did not absorb it and the request 500'd).

    A separate ``mode=ro`` connection does lock-free WAL reads fully isolated
    from the writer connection's cursor state. One connection per call is
    inherently thread-safe and sub-millisecond to open; it also never
    competes for the daemon's write lock, so the dashboard cannot add to the
    documented USB write-contention pressure."""
    from storage.article_store import _get_db_path

    db = _get_db_path()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=15)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _articles_from_db(limit: int = 50, min_score: float = 0.0) -> list[dict]:
    try:
        rows = _ro_query(
            "SELECT id, url, title, source, published, kw_score, ai_score, urgency, first_seen "
            "FROM articles "
            "WHERE (CASE WHEN ai_score>kw_score THEN ai_score ELSE kw_score END) >= ? "
            "AND url NOT LIKE 'backtest://%' "
            "AND source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%' "
            "ORDER BY ai_score DESC, kw_score DESC, first_seen DESC LIMIT ?",
            (min_score, max(1, min(500, int(limit)))),
        )
    except sqlite3.Error:
        return []
    out = []
    for r in rows:
        ai = float(r[6] or 0)
        kw = float(r[5] or 0)
        out.append({
            "id": r[0], "url": r[1], "title": r[2], "source": r[3],
            "published": r[4], "kw_score": kw, "ai_score": ai,
            "score": ai if ai > 0 else kw,
            "urgency": int(r[7] or 0),
            "first_seen": r[8],
        })
    return out


def _tail_risk_chat_lines(an: dict) -> list[str]:
    """Render the paper-trader tail-risk block (added to /api/analytics by
    the trader's analytics endpoint) as compact chat-context lines.

    Pure / total: any missing key, a NO_DATA gate, an upstream error, or a
    non-dict payload yields ``[]`` (the block is simply omitted, never an
    exception into the chat handler). INSUFFICIENT collapses to one honest
    "still building history" line — a 5-day VaR must not read as a verdict.
    """
    if not isinstance(an, dict):
        return []
    tr = an.get("tail_risk")
    if not isinstance(tr, dict):
        return []
    state = tr.get("state")
    if state == "INSUFFICIENT":
        return [
            f"Tail risk: still building history "
            f"({tr.get('n_returns', 0)}/{tr.get('min_returns', 20)} "
            f"daily observations — verdict withheld)."
        ]
    if state != "OK":
        return []

    def _p(key: str) -> str:
        v = tr.get(key)
        return f"{v:.2f}%" if isinstance(v, (int, float)) else "n/a"

    skew = tr.get("return_skew")
    skew_s = f"{skew:+.2f}" if isinstance(skew, (int, float)) else "n/a"
    return [
        f"95% 1-day VaR {_p('var_95_pct')} / CVaR {_p('cvar_95_pct')}, "
        f"99% VaR {_p('var_99_pct')} "
        f"(historical, {tr.get('n_returns', 0)} daily obs)",
        f"ann.vol {_p('annualized_vol_pct')}, downside-dev "
        f"{_p('downside_deviation_pct')}, skew {skew_s}, worst day "
        f"{_p('worst_day_pct')}, max down-streak "
        f"{tr.get('max_consecutive_down_days', 'n/a')}d, Ulcer "
        f"{_p('ulcer_index_pct')}",
    ]


# ── alert-confidence-trend enrichment ──────────────────────────────────
# Clusters urgent / near-urgent articles in the last 24h by title-token
# Jaccard similarity (reusing `ml/dedup.py`'s normalizer) and reports the
# corroborating-source count delta between the recent half (0-6h) and the
# earlier half (6-24h). A story whose unique-source count is GROWING is
# being corroborated by additional outlets — high-trust, possibly act soon;
# a flat count from a single outlet is PR / spam; a high count fading to
# zero recent sources is the wire moving on.
#
# The existing `news_corroboration` endpoint reports CURRENT-state source
# counts; this is the temporal-DELTA companion the chat carried no view of.
_ALERT_TREND_DELTA = 1     # min new sources in recent half to call RISING
_ALERT_CLUSTER_THRESHOLD = 0.6  # Jaccard match — same as ml.dedup default


def build_alert_confidence_trend(
    articles: Any,
    *,
    now: datetime | None = None,
    min_cluster_size: int = 2,
    max_clusters: int = 6,
) -> dict:
    """Cluster urgent articles by title and report source-count trend.

    Pure / total — never raises, never reads DB. Caller supplies article
    rows (already filtered to `urgency >= 1` and through `_LIVE_ONLY_SQL`).

    Each output cluster carries:
      - `anchor_title`: the highest-ai_score title in the cluster (the
        canonical headline the analyst recognizes)
      - `n_total`, `n_recent` (0-6h), `n_earlier` (6-24h): UNIQUE source
        counts (a single outlet syndicating itself doesn't inflate trust)
      - `delta`: n_recent − n_earlier
      - `trend`: ``RISING`` (delta ≥ +ALERT_TREND_DELTA), ``FADING``
        (delta ≤ -ALERT_TREND_DELTA), ``STABLE``, or ``SINGLE_SOURCE``
        (only one unique source across the window — likely PR/spam, not a
        corroborated story)
      - `max_ai_score`: peak ai_score in the cluster (lets the analyst
        prioritise across clusters by both trend AND urgency)

    Clustering reuses `ml.dedup.title_tokens` + `jaccard_similarity` so
    chat / briefing / dashboard cluster identically (no drift).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if not isinstance(articles, (list, tuple)):
        return {"as_of": now.isoformat(timespec="seconds"),
                "window_hours": 24, "clusters": []}
    # Import inline so a missing module never breaks the chat handler at
    # import time (dedup is part of the same repo, this is belt-and-braces).
    try:
        from ml.dedup import title_tokens, jaccard_similarity
    except ImportError:
        return {"as_of": now.isoformat(timespec="seconds"),
                "window_hours": 24, "clusters": []}
    cutoff = now - timedelta(hours=24)
    boundary = now - timedelta(hours=6)
    clusters: list[dict] = []
    for art in articles:
        if not isinstance(art, dict):
            continue
        ts = _parse_first_seen(art.get("first_seen"))
        if ts is None or ts < cutoff or ts > now:
            continue
        title = art.get("title") or ""
        toks = title_tokens(title)
        if not toks:
            continue
        try:
            score = float(art.get("ai_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        source = (art.get("source") or "").strip().lower()
        is_recent = ts >= boundary
        placed = False
        for cl in clusters:
            if jaccard_similarity(toks, cl["anchor_tokens"]) >= _ALERT_CLUSTER_THRESHOLD:
                if score > cl["max_ai_score"]:
                    cl["max_ai_score"] = score
                    cl["anchor_title"] = title
                if source:
                    if is_recent:
                        cl["recent_sources"].add(source)
                    else:
                        cl["earlier_sources"].add(source)
                    cl["all_sources"].add(source)
                cl["n_articles"] += 1
                placed = True
                break
        if not placed:
            clusters.append({
                "anchor_tokens": toks,
                "anchor_title": title,
                "max_ai_score": score,
                "recent_sources": ({source} if (source and is_recent) else set()),
                "earlier_sources": ({source} if (source and not is_recent) else set()),
                "all_sources": ({source} if source else set()),
                "n_articles": 1,
            })
    out: list[dict] = []
    for cl in clusters:
        if cl["n_articles"] < min_cluster_size:
            continue
        n_recent = len(cl["recent_sources"])
        n_earlier = len(cl["earlier_sources"])
        n_total = len(cl["all_sources"])
        delta = n_recent - n_earlier
        if n_total <= 1:
            trend = "SINGLE_SOURCE"
        elif delta >= _ALERT_TREND_DELTA:
            trend = "RISING"
        elif delta <= -_ALERT_TREND_DELTA:
            trend = "FADING"
        else:
            trend = "STABLE"
        out.append({
            "anchor_title": cl["anchor_title"],
            "max_ai_score": round(cl["max_ai_score"], 2),
            "n_articles": cl["n_articles"],
            "n_total_sources": n_total,
            "n_recent_sources": n_recent,
            "n_earlier_sources": n_earlier,
            "delta": delta,
            "trend": trend,
        })
    # Rank: RISING first (most actionable), then STABLE/FADING by score.
    _trend_order = {"RISING": 0, "STABLE": 1, "FADING": 2, "SINGLE_SOURCE": 3}
    out.sort(key=lambda c: (
        _trend_order.get(c["trend"], 9),
        -c["max_ai_score"],
    ))
    return {
        "as_of": now.isoformat(timespec="seconds"),
        "window_hours": 24,
        "boundary_hours": 6,
        "clusters": out[:max_clusters],
    }


_ALERT_TREND_REAL = ("RISING", "FADING")  # STABLE/SINGLE_SOURCE → silence


def _alert_confidence_trend_chat_lines(rep) -> list[str]:
    """Render `build_alert_confidence_trend` as chat-context lines.

    Same pure / total contract as siblings — non-dict / empty clusters →
    ``[]``. STABLE & SINGLE_SOURCE are silent (the chat already shows
    `articles_block` for current top signals; this surface adds value only
    by flagging stories whose corroboration is actively moving).
    """
    if not isinstance(rep, dict):
        return []
    clusters = rep.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        return []
    lines: list[str] = []
    for cl in clusters:
        if not isinstance(cl, dict):
            continue
        trend = cl.get("trend")
        if trend not in _ALERT_TREND_REAL:
            continue
        title = (cl.get("anchor_title") or "").strip()
        if not title:
            continue
        n_total = cl.get("n_total_sources")
        n_recent = cl.get("n_recent_sources")
        n_earlier = cl.get("n_earlier_sources")
        score = cl.get("max_ai_score")
        score_s = (f", score {score:.1f}"
                   if isinstance(score, (int, float)) else "")
        # Truncate title to keep chat budget tight; the operator can ask
        # for full details if needed.
        title_short = title if len(title) <= 90 else title[:87] + "..."
        # Render arrow direction so the line reads left-to-right as time.
        if (isinstance(n_earlier, int) and isinstance(n_recent, int)
                and isinstance(n_total, int)):
            corr_s = (
                f" ({n_earlier} src in 6-24h → {n_recent} src in 0-6h, "
                f"{n_total} total)"
            )
        else:
            corr_s = ""
        lines.append(f"[{trend}{score_s}] {title_short}{corr_s}")
    return lines


# ── position-conviction-decay enrichment ───────────────────────────────
# Per-held-ticker 24h ai_score trend, bucketed into 4 × 6h slices.
# Closes the temporal gap in `_portfolio_signals` (snapshot only). A held
# position whose ai_score band is RISING means the wire is increasingly
# focused on it; FADING means the story is going quiet. Pure builder + pure
# render helper — no Flask, no DB; the chat handler injects ticker list +
# pre-fetched article rows.
_CONV_DELTA_THRESHOLD = 0.5  # avg ai_score delta to call RISING / FADING
_CONV_MIN_ARTICLES = 2       # below this, trend is INSUFFICIENT_DATA


def _parse_first_seen(s: Any) -> datetime | None:
    """Robust ISO timestamp parser — returns aware UTC datetime or None.
    Mirrors the tolerance of `analysis.claude_analyst` so the briefing and
    the chat helper bucket articles into the same windows."""
    if not isinstance(s, str) or not s:
        return None
    try:
        # Accept both '...+00:00' and '...Z'
        v = s.replace("Z", "+00:00") if s.endswith("Z") else s
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def build_position_conviction_decay(
    held_tickers: Any,
    articles: Any,
    *,
    now: datetime | None = None,
) -> dict:
    """Bucket the last 24h of articles into 4 × 6h slices per held ticker
    and report the avg ai_score per bucket + a coarse trend verdict.

    Pure / total — never raises, never reads DB, never sub-fetches. Caller
    supplies the held ticker list (from paper-trader's /api/state.positions)
    and the recent article rows (already filtered through `_LIVE_ONLY_SQL`).

    Trend rule (deliberately blunt — a finer model would over-fit the
    sparse-sample regime this runs in): compare the avg ai_score of the
    earlier half (12-24h bucket) against the recent half (0-12h bucket).
    Δ > +CONV_DELTA_THRESHOLD → RISING; Δ < -CONV_DELTA_THRESHOLD →
    FADING; else STABLE. Fewer than CONV_MIN_ARTICLES total → trend is
    INSUFFICIENT_DATA and the bucket numbers still surface honestly.

    Buckets are oldest → newest in `buckets`:
      bucket[0] = 18-24h, bucket[1] = 12-18h, bucket[2] = 6-12h, bucket[3] = 0-6h
    So the *last* element is the freshest. Each bucket carries `n` (count)
    and `avg` (mean ai_score; None when n == 0).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    # Normalize held tickers — accept list[str] or list[dict] with "ticker".
    held: set[str] = set()
    if isinstance(held_tickers, (list, tuple, set)):
        for h in held_tickers:
            if isinstance(h, str):
                t = h.strip().upper()
                if t:
                    held.add(t)
            elif isinstance(h, dict):
                t = (h.get("ticker") or "").strip().upper()
                if t:
                    held.add(t)
    out_tickers: list[dict] = []
    if not held or not isinstance(articles, (list, tuple)):
        return {
            "as_of": now.isoformat(timespec="seconds"),
            "window_hours": 24,
            "bucket_hours": 6,
            "tickers": [],
        }
    # Compile a case-insensitive word-boundary regex per held ticker; we
    # match the title (cheap and consistent with the rest of the daemon's
    # ticker extraction patterns — the brittler "tickers" column may be
    # missing on legacy rows).
    patterns = {
        t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in held
    }
    cutoff = now - timedelta(hours=24)
    # buckets[ticker] = [list_for_18-24, list_for_12-18, list_for_6-12, list_for_0-6]
    buckets: dict[str, list[list[float]]] = {t: [[], [], [], []] for t in held}
    for art in articles:
        if not isinstance(art, dict):
            continue
        ts = _parse_first_seen(art.get("first_seen"))
        if ts is None or ts < cutoff or ts > now:
            continue
        age_h = (now - ts).total_seconds() / 3600.0
        # idx = 0 (18-24h) … 3 (0-6h). Round down conservatively.
        if age_h >= 24:
            continue
        idx = 3 - min(3, int(age_h // 6))
        title = (art.get("title") or "")
        try:
            score = float(art.get("ai_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        for tk, pat in patterns.items():
            if pat.search(title):
                buckets[tk][idx].append(score)
    for tk in sorted(held):
        slots = buckets[tk]
        bucket_view = []
        for slot in slots:
            n = len(slot)
            avg = (sum(slot) / n) if n else None
            bucket_view.append({"n": n, "avg": round(avg, 2) if avg is not None else None})
        n_total = sum(b["n"] for b in bucket_view)
        # Trend: recent half (idx 2+3 = 0-12h) vs earlier half (idx 0+1 = 12-24h).
        recent_scores = slots[2] + slots[3]
        earlier_scores = slots[0] + slots[1]
        if n_total < _CONV_MIN_ARTICLES or not recent_scores or not earlier_scores:
            trend = "INSUFFICIENT_DATA"
            delta: float | None = None
        else:
            r_avg = sum(recent_scores) / len(recent_scores)
            e_avg = sum(earlier_scores) / len(earlier_scores)
            delta = r_avg - e_avg
            if delta > _CONV_DELTA_THRESHOLD:
                trend = "RISING"
            elif delta < -_CONV_DELTA_THRESHOLD:
                trend = "FADING"
            else:
                trend = "STABLE"
        row = {
            "ticker": tk,
            "n_articles": n_total,
            "buckets": bucket_view,    # oldest → newest
            "trend": trend,
        }
        if delta is not None:
            row["recent_minus_earlier"] = round(delta, 2)
        out_tickers.append(row)
    return {
        "as_of": now.isoformat(timespec="seconds"),
        "window_hours": 24,
        "bucket_hours": 6,
        "tickers": out_tickers,
    }


_CONV_REAL_TRENDS = ("RISING", "STABLE", "FADING")


def _position_conviction_decay_chat_lines(rep) -> list[str]:
    """Render `build_position_conviction_decay` output as chat-context lines.

    Pure / total — mirrors the `_baseline_compare_chat_lines` contract:
    non-dict / missing structure → ``[]`` (silence, never an exception).
    A ticker with INSUFFICIENT_DATA collapses to one honest line; a real
    trend gets a one-line summary with the bucket counts so the analyst can
    audit the verdict (a 'RISING' that's really 1→2 articles deserves the
    raw n visible, not buried).
    """
    if not isinstance(rep, dict):
        return []
    tickers = rep.get("tickers")
    if not isinstance(tickers, list) or not tickers:
        return []
    lines: list[str] = []
    # Stable trend → silence (chat budget is finite; only surface motion).
    # INSUFFICIENT_DATA → silence per ticker (no value in saying "I don't know"
    # for every held name); but if EVERY held ticker is INSUFFICIENT_DATA the
    # block stays empty (handler omits the section).
    for row in tickers:
        if not isinstance(row, dict):
            continue
        trend = row.get("trend")
        if trend not in _CONV_REAL_TRENDS:
            continue
        if trend == "STABLE":
            continue
        tk = row.get("ticker") or "?"
        n = row.get("n_articles")
        delta = row.get("recent_minus_earlier")
        buckets = row.get("buckets") or []
        # Render oldest → newest so the chat reads left-to-right as time forward.
        if isinstance(buckets, list) and len(buckets) == 4:
            parts = []
            for label, b in zip(("18-24h", "12-18h", "6-12h", "0-6h"), buckets):
                if not isinstance(b, dict):
                    continue
                bn = b.get("n") or 0
                avg = b.get("avg")
                if bn == 0 or avg is None:
                    parts.append(f"{label} —")
                else:
                    parts.append(f"{label} {avg:.1f}({bn})")
            bucket_s = " → ".join(parts)
        else:
            bucket_s = ""
        delta_s = (f", recent−earlier {delta:+.2f}"
                   if isinstance(delta, (int, float)) else "")
        n_s = (f", n={n}" if isinstance(n, int) else "")
        line = f"{tk}: 24h ai_score trend {trend}{n_s}{delta_s}"
        if bucket_s:
            line += f" [{bucket_s}]"
        lines.append(line)
    return lines


_BC_REAL_VERDICTS = (
    "MLP_ADDS_SKILL", "MLP_NO_BETTER_THAN_TRIVIAL", "MLP_WORSE_THAN_TRIVIAL",
)


def _baseline_compare_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/baseline-compare` (the read-only OOS-skill
    diagnostic — does the 17-feature DecisionScorer beat a one-line rule out
    of sample, or is its sizing nudge noise?) as compact chat-context lines.

    SSOT (paper-trader invariant #10): the module's own ``hint`` string is
    composed **verbatim** — no chat-side re-derived verdict that could drift
    from the trader endpoint.

    Pure / total — exactly the ``_tail_risk_chat_lines`` contract:

    - non-dict, missing/unknown ``verdict`` → ``[]`` (block omitted, never
      an exception into the chat handler)
    - ``INSUFFICIENT_DATA`` → ONE honest withheld line; ``hint`` is **not**
      surfaced (the never-raises trader endpoint stuffs an exception/stack
      string into ``hint`` on fault — that must never reach the analyst)
    - a real verdict → the verdict headline + the module's verbatim ``hint``
      + (when finite) the scale-invariant rank-IC race a quant checks; any
      missing numeric simply drops that one line, never raises
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    if verdict == "INSUFFICIENT_DATA":
        return [
            "ML gate skill (OOS): insufficient out-of-sample history — "
            "verdict withheld."
        ]
    if verdict not in _BC_REAL_VERDICTS:
        return []

    def _num(v):
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    n = rep.get("n")
    n_s = f", n={n}" if isinstance(n, int) else ""
    lines = [f"ML gate skill (OOS{n_s}): {verdict}"]

    hint = rep.get("hint")
    if isinstance(hint, str) and hint.strip():
        lines.append(f"  {hint}")          # verbatim SSOT — invariant #10

    mlp = rep.get("mlp") if isinstance(rep.get("mlp"), dict) else {}
    mlp_ic = _num(mlp.get("rank_ic"))
    best_ic = _num(rep.get("best_baseline_ic"))
    gap = _num(rep.get("ic_gap"))
    if mlp_ic is not None and best_ic is not None and gap is not None:
        best = rep.get("best_baseline") or "?"
        n_train = rep.get("n_train")
        nt = (f"; scorer n_train={n_train}"
              if isinstance(n_train, int) else "")
        lines.append(
            f"  MLP rank_ic {mlp_ic:+.3f} vs best one-liner "
            f"'{best}' {best_ic:+.3f} (gap {gap:+.3f}){nt}"
        )
    return lines


_CORR_REAL_VERDICTS = (
    "SINGLE_NAME_RISK", "CONCENTRATED", "MODERATE", "DIVERSIFIED",
)


def _correlation_chat_lines(corr) -> list[str]:
    """Render paper-trader's ``/api/correlation`` (the diagnostic that exposes
    *factor* concentration — do the held names actually move together?) as
    compact chat-context lines.

    ``/api/risk`` reports name-level concentration honestly (already surfaced
    upstream by ``analytics_block``), but a 2-position 59/41 book can be
    either two uncorrelated bets or — if both names are high-β semis that
    move as one — a single bet wearing two tickers. The chat carried the
    name-level view but was blind to the FACTOR view; this closes the gap
    exactly as ``_baseline_compare_chat_lines`` closed the ML-honesty one.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is
    composed **verbatim** — no chat-side re-derived verdict that could drift
    from the trader endpoint (the verdict label, mean ρ, effective-bets
    count, and the optional most-coupled-pair clause all already live inside
    the ``headline`` string).

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict, missing ``state``, or ``state`` ``NO_DATA`` (no stock
      positions — concentration is undefined) → ``[]`` (silence, not noise;
      the ``_behavioural_chat_lines`` NO_DATA-omit precedent)
    - ``INSUFFICIENT`` → ONE honest withheld line; the builder's own
      headline already names what is missing (e.g. "Only 1 correlatable
      stock name(s) — correlation verdict withheld"), so it passes through
      verbatim
    - a real verdict (``SINGLE_NAME_RISK`` / ``CONCENTRATED`` / ``MODERATE``
      / ``DIVERSIFIED``) → the builder's verbatim headline (which the unit
      tests for ``build_correlation`` already pin in shape)
    - any other ``state`` or unknown verdict on an ``OK`` row → ``[]`` (the
      never-raises builder must never have manufactured this; degrade
      silently rather than parrot a label the chat cannot validate)
    """
    if not isinstance(corr, dict):
        return []
    state = corr.get("state")
    if state == "NO_DATA":
        return []
    headline = corr.get("headline")
    if not isinstance(headline, str) or not headline.strip():
        return []
    if state == "INSUFFICIENT":
        return [f"Correlation: {headline}"]
    if state != "OK":
        return []
    if corr.get("verdict") not in _CORR_REAL_VERDICTS:
        return []
    return [f"Correlation: {headline}"]


def _behavioural_chat_lines(scorecard, paralysis, churn) -> list[str]:
    """Render the paper-trader's own self-review verdicts
    (``/api/scorecard``, ``/api/capital-paralysis``, ``/api/churn``) as
    compact chat-context lines.

    Composes builder verdicts **verbatim** (paper-trader invariant #10 —
    single source of truth: no re-derived metrics; each builder's own
    ``headline`` / ``focus`` / ``flags`` / ``recommended_unlock.reason``
    string passes through unchanged). One derived ``▶ PRIORITY`` line by
    precedence: a paralysis unlock beats a scorecard focus beats a
    CHURNING state; none-applicable → no priority line.

    Pure / total — exactly the ``_tail_risk_chat_lines`` contract: an
    input that is not a dict, carries an ``error`` key, lacks a ``state``,
    or is gated ``NO_DATA`` contributes nothing; all three contributing
    nothing → ``[]`` (the block is simply omitted, never an exception
    into the chat handler).
    """
    def _ok(d) -> bool:
        return (
            isinstance(d, dict)
            and "error" not in d
            and "state" in d
            and d.get("state") != "NO_DATA"
        )

    sc_ok, cp_ok, ch_ok = _ok(scorecard), _ok(paralysis), _ok(churn)
    lines: list[str] = []

    if sc_ok:
        hl = scorecard.get("headline")
        if hl:
            lines.append(f"Scorecard: {hl}")
        focus = scorecard.get("focus")
        if isinstance(focus, dict) and focus.get("headline"):
            lines.append(f"  focus: {focus['headline']}")

    if cp_ok:
        hl = paralysis.get("headline")
        if hl:
            lines.append(f"Capital: {hl}")
        flags = paralysis.get("flags")
        if isinstance(flags, list):
            for fl in flags[:3]:
                lines.append(f"  • {fl}")

    if ch_ok:
        hl = churn.get("headline")
        if hl:
            lines.append(f"Churn: {hl}")

    # One derived priority line, by precedence (verbatim reason/headline).
    unlock = paralysis.get("recommended_unlock") if cp_ok else None
    focus = scorecard.get("focus") if sc_ok else None
    if isinstance(unlock, dict) and unlock.get("ticker") and unlock.get("reason"):
        lines.append(f"▶ PRIORITY: sell {unlock['ticker']} — {unlock['reason']}")
    elif isinstance(focus, dict) and focus.get("theme") and focus.get("headline"):
        lines.append(f"▶ PRIORITY: {focus['theme']} — {focus['headline']}")
    elif ch_ok and churn.get("state") == "CHURNING" and churn.get("headline"):
        lines.append(f"▶ PRIORITY: overtrading — {churn['headline']}")

    return lines


def _macro_calendar_chat_lines(mc) -> list[str]:
    """Render paper-trader's `/api/macro-calendar` (the forward FOMC
    rate-decision awareness already fed into the live trader's OWN decision
    prompt) as compact chat-context lines. FOMC is the single biggest
    market-wide event for this leveraged-ETF-heavy book; the chat carried
    rich BACKWARD analytics but zero FORWARD macro-event awareness — this
    closes that gap, exactly as `_baseline_compare_chat_lines` closed the
    ML-gate-honesty one.

    SSOT (paper-trader invariant #10): the builder's own ``summary`` string
    is the verbatim headline — no chat-side re-derived verdict that could
    drift from the trader endpoint.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never an exception into the chat
      handler)
    - no events → ``[]``: the builder sets ``events: []`` for EVERY
      non-actionable branch (no FOMC within horizon, schedule-not-loaded,
      builder error), so a "no FOMC within 14d" / error string never
      becomes chat filler (the ``_behavioural_chat_lines`` NO_DATA-omit
      precedent — silence, not noise)
    - events present → the builder's verbatim ``summary`` headline (only
      when a usable string) + one restated detail line per event (when_et /
      tier / day-or-hour timing restated from the builder's OWN fields —
      the ``earnings_block`` precedent, never a recomputation); a within-24h
      ``IMMINENT_HOURS`` event surfaces the HOUR figure (a day figure rounds
      a 6h-away decision to a misleading 0.2d); a malformed row is skipped,
      never raises (the ``_paper_trader_position_lines`` precedent)
    """
    if not isinstance(mc, dict):
        return []
    events = mc.get("events")
    if not isinstance(events, list) or not events:
        return []

    def _num(v):
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    lines: list[str] = []
    summary = mc.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines.append(summary)              # verbatim SSOT — invariant #10

    for e in events:
        if not isinstance(e, dict):
            continue
        label = e.get("label") or e.get("event") or "FOMC rate decision"
        when_et = e.get("when_et") or "?"
        tier = e.get("tier") or "?"
        if tier == "IMMINENT_HOURS":
            ha = _num(e.get("hours_away"))
            timing = f"in {ha:.1f}h" if ha is not None else "imminent"
        else:
            da = _num(e.get("days_away"))
            timing = f"in {da:.1f}d" if da is not None else "upcoming"
        lines.append(f"  {label} {timing} — {when_et} [{tier}]")

    return lines


def _event_readiness_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/event-readiness` (will the live trader
    actually be able to react before the next earnings print?) as compact
    chat-context lines.

    The chat now carries forward macro + earnings timing (`_macro_calendar_
    chat_lines`) and pre-print dollarized shock (`_earnings_shock_chat_
    lines`), but ALL of that assumes the bot will *make a decision* before
    the print. The live failure mode (NO_DECISION storms / Claude empty
    streaks / wedged-supervisor PARALYSIS) breaks that assumption silently
    — and the chat-side analyst answers "is your book at risk?" without
    ever questioning whether the bot can act. This closes that gap.

    SSOT (paper-trader invariant #10): the builder's own ``summary`` string
    is the verbatim chat headline — no chat-side re-derived verdict that
    could drift from the trader endpoint. Per-event lines restate the
    builder's *own* fields (ticker / exposure_usd / hours_until_event /
    verdict / recommended_action) — never a recomputation (the
    ``_macro_calendar_chat_lines`` precedent).

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never an exception into the chat
      handler)
    - non-actionable verdicts (``READY`` / ``NO_EVENTS`` / ``NO_DECISIONS``)
      → ``[]``: only ``BLIND`` / ``DEGRADED`` / ``IMMINENT_OVERDUE`` events
      are worth surfacing to the analyst. A healthy pipeline is silence
      (the ``_behavioural_chat_lines`` NO_DATA-omit precedent — never
      chat filler)
    - actionable verdicts → builder's verbatim ``summary`` headline (only
      when a usable string) + one line per BLIND/DEGRADED/OVERDUE event
      restating the builder's OWN fields (ticker, hours-until, verdict,
      and the verbatim recommended_action — the
      ``_earnings_shock_chat_lines`` per-row precedent); a malformed row is
      skipped, never raises (the ``_paper_trader_position_lines`` precedent)
    """
    if not isinstance(rep, dict):
        return []
    worst = rep.get("worst_verdict")
    actionable = {"BLIND", "DEGRADED", "IMMINENT_OVERDUE"}
    if worst not in actionable:
        return []
    events = rep.get("events")
    if not isinstance(events, list) or not events:
        return []

    lines: list[str] = []
    summary = rep.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines.append(summary)                  # verbatim SSOT — invariant #10

    for e in events:
        if not isinstance(e, dict):
            continue
        v = e.get("verdict")
        if v not in actionable:
            continue
        tk = e.get("ticker") or "?"
        hours = e.get("hours_until_event")
        if isinstance(hours, (int, float)) and not isinstance(hours, bool):
            timing = f"in {hours:.1f}h"
        else:
            timing = "imminent"
        exposure = e.get("exposure_usd")
        exp_str = (f"${exposure:.0f}"
                   if isinstance(exposure, (int, float))
                   and not isinstance(exposure, bool) else "$?")
        action = e.get("recommended_action") or ""
        lines.append(f"  {tk} {timing} — exposure {exp_str} [{v}]")
        if isinstance(action, str) and action.strip():
            lines.append(f"    → {action}")    # verbatim — builder SSOT

    return lines


def _decision_paralysis_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/decision-paralysis` (consecutive HOLD
    streak detector — the HOLD_LOCK pathology) as compact chat-context
    lines.

    The chat already carries the NO_DECISION-storm path via
    `_event_readiness_chat_lines` (which folds the current NO_DECISION
    streak into BLIND/DEGRADED verdicts) and the broad action-mix via the
    decision-health 24h aggregate, but neither flags the *other* "the loop
    is alive but nothing is happening" failure mode: a contiguous run of
    pure-HOLD decisions. A 95% HOLD share over 24h looks identical whether
    spread across the day or stacked into a single immovable block where
    Opus is deciding every cycle and never moving the book. The chat
    answers "should I be doing something?" — and a stacked HOLD_LOCK is the
    exact pathology that question is asked about. This closes that gap.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` string
    is the verbatim chat headline — no re-derived verdict. Per-row lines
    restate the builder's *own* fields (current_hold_streak,
    hours_since_last_active) — never a recomputation (the
    ``_event_readiness_chat_lines`` precedent).

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never an exception into the chat
      handler)
    - non-actionable verdicts (``ACTIVE`` / ``NO_DATA``) → ``[]``: only
      ``HOLD_LOCK`` / ``IDLE_STORM`` / ``PASSIVE_LOOP`` are worth surfacing
      to the analyst. A healthy decision loop is silence (the
      ``_event_readiness_chat_lines`` silence precedent — never chat filler)
    - actionable verdicts → builder's verbatim ``headline`` (only when a
      usable string) + one detail line restating
      current_hold_streak / current_passive_streak / hours_since_last_active
      from the builder's OWN fields (the ``_macro_calendar_chat_lines``
      precedent); a missing field degrades to a "?" placeholder rather than
      raises (the ``_paper_trader_position_lines`` precedent)
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    actionable = {"HOLD_LOCK", "IDLE_STORM", "PASSIVE_LOOP"}
    if verdict not in actionable:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)                  # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    hold_run = _num(rep.get("current_hold_streak"))
    pass_run = _num(rep.get("current_passive_streak"))
    nd_run = _num(rep.get("current_no_decision_streak"))
    hsla = _num(rep.get("hours_since_last_active"))

    detail_parts: list[str] = []
    if hold_run is not None:
        detail_parts.append(f"HOLD streak {int(hold_run)}")
    if nd_run:
        detail_parts.append(f"NO_DECISION streak {int(nd_run)}")
    if pass_run is not None and (hold_run is None or pass_run > hold_run):
        detail_parts.append(f"passive streak {int(pass_run)}")
    if hsla is not None:
        detail_parts.append(f"last FILLED/BLOCKED {hsla:.1f}h ago")

    if detail_parts:
        lines.append("  " + " | ".join(detail_parts))

    return lines


def _thesis_drift_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/thesis-drift` (every open position re-tested
    against the verbatim reason it was opened for, graded INTACT / WEAKENING /
    BROKEN) as compact chat-context lines.

    The chat already carries the bot's per-name closed-trade memory (the
    behavioural block), and the OPEN book by *position* (the portfolio
    snapshot) and by *factor* (the correlation block) — but neither answers
    the single discipline question that drives most discretionary trims:
    *"is the thing the bot bought this for still true?"* That answer sits
    verbatim in `trades.reason` of each opening fill, and only thesis-drift
    re-scores each holding against it. Surfacing the WEAKENING/BROKEN cards
    here lets the analyst answer "should the bot have already sold X?"
    honestly instead of re-deriving from raw signals.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is the
    chat headline and each card's ``drift_reasons`` are surfaced **verbatim**
    — no re-derived verdict (the ``_decision_paralysis_chat_lines``
    precedent).

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict, missing/unknown shape → ``[]`` (block omitted, never an
      exception into the chat handler)
    - state ``NO_DATA`` (no open positions) or *every* position INTACT →
      ``[]``: a healthy book is silence (the silence precedent — never
      chat filler when the loop is fine)
    - WEAKENING / BROKEN cards present → builder's verbatim ``headline``
      + one detail line per non-INTACT card restating its OWN
      ``ticker``/``health``/``pl_pct``/``days_held``/``drift_reasons``
      (a missing field degrades to a "?" placeholder rather than raises —
      the ``_paper_trader_position_lines`` precedent).
    """
    if not isinstance(rep, dict):
        return []
    cards = rep.get("positions")
    if not isinstance(cards, list) or not cards:
        return []
    bad = [c for c in cards
           if isinstance(c, dict)
           and c.get("health") in ("BROKEN", "WEAKENING")]
    if not bad:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)              # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    for c in bad:
        tk = c.get("ticker") or "?"
        typ = c.get("type") or "stock"
        health = c.get("health") or "?"
        pl_pct = _num(c.get("pl_pct"))
        days = _num(c.get("days_held"))
        bits = [f"{tk} {typ} {health}"]
        if pl_pct is not None:
            bits.append(f"P/L {pl_pct:+.2f}%")
        if days is not None:
            bits.append(f"held {days:.2f}d")
        reasons = c.get("drift_reasons")
        if isinstance(reasons, list) and reasons:
            reason_s = "; ".join(str(r) for r in reasons if r)
            if reason_s:
                bits.append(f"drift: {reason_s}")
        lines.append("  " + " | ".join(bits))

    return lines


def _earnings_shock_chat_lines(es) -> list[str]:
    """Render paper-trader's `/api/earnings-shock` (pre-earnings dollarized
    1σ shock per held imminent print) as compact chat-context lines.

    The chat carried rich BACKWARD analytics + held-book context, and the
    macro / earnings calendars covered the FORWARD-event timing, but
    nothing translated *"NVDA earnings in 0.9d"* into *"if NVDA gaps the
    typical 1σ, your book moves $X (Y % of equity)."* — exactly the
    pre-print question the analyst gets asked. This closes that gap, in
    the same shape as `_macro_calendar_chat_lines` / `_baseline_compare_
    chat_lines` (verbatim SSOT headline; non-actionable states silenced;
    pure/total).

    SSOT (paper-trader invariant #10): the builder's own ``headline``
    string is the verbatim chat headline — no re-derived verdict that
    could drift from the trader endpoint. Per-row lines restate the
    builder's *own* fields (ticker / days_to_earnings / current_value /
    weight_pct / sigma_pct / sigma_dollar_move / sigma_book_pct) — never
    a recomputation (the ``_macro_calendar_chat_lines`` precedent).

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never an exception into the chat
      handler — the ``_macro_calendar_chat_lines`` contract; a non-actionable
      state must never become chat filler).
    - state in {NO_DATA, NO_EVENTS} or events empty → ``[]``: the empty
      book / quiet calendar paths must be silence, mirroring how
      ``_macro_calendar_chat_lines`` omits the no-FOMC and not-loaded
      branches and how ``_behavioural_chat_lines`` omits NO_DATA.
    - state OK with events → builder ``headline`` verbatim + one line per
      held event (ticker · in Nd · $value · weight · σ if known).
    - per-row ``INSUFFICIENT_HISTORY``: the event still surfaces (so
      "NVDA reports in 0.9d" is never hidden) but σ is reported as
      *withheld* — never fabricated (the builder's per-row honesty
      precedent; mirrors the ``baseline_compare`` INSUFFICIENT_DATA →
      silent-but-honest contract).
    - malformed event row → skipped, never raises (the
      ``_macro_calendar_chat_lines`` malformed-row precedent).
    """
    if not isinstance(es, dict):
        return []
    state = es.get("state")
    if state in (None, "NO_DATA", "NO_EVENTS"):
        return []
    events = es.get("events")
    if not isinstance(events, list) or not events:
        return []

    def _num(v):
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    lines: list[str] = []
    headline = es.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)            # verbatim SSOT — invariant #10

    for e in events:
        if not isinstance(e, dict):
            continue
        tk = e.get("ticker")
        if not isinstance(tk, str) or not tk:
            continue
        days = _num(e.get("days_to_earnings"))
        cv = _num(e.get("current_value_usd"))
        wt = _num(e.get("weight_pct"))
        ev_state = e.get("state")
        sigma_pct = _num(e.get("sigma_pct"))
        sigma_dollar = _num(e.get("sigma_dollar_move"))
        sigma_book = _num(e.get("sigma_book_pct"))
        # ticker + timing (always)
        when = f"in {days:.1f}d" if days is not None else "imminent"
        # exposure (always when known)
        if cv is not None and wt is not None:
            expo = f" — ${cv:.2f} ({wt:.1f}% of book)"
        elif cv is not None:
            expo = f" — ${cv:.2f}"
        else:
            expo = ""
        # σ tail: report when known; honestly say "σ withheld" otherwise
        if (sigma_pct is not None and sigma_dollar is not None
                and sigma_book is not None and ev_state == "OK"):
            sig = (f" · σ ±{sigma_pct:.1f}% → ±${sigma_dollar:.2f} "
                   f"(book ±{sigma_book:.2f}%)")
        elif ev_state == "INSUFFICIENT_HISTORY":
            sig = " · σ withheld (insufficient earnings history)"
        else:
            sig = ""
        lines.append(f"  {tk} {when}{expo}{sig}")

    return lines


def _norm_title(t: Any) -> str:
    return str(t or "").strip().casefold()


def _partition_thesis_articles(
    breaking: list[dict], thesis: list[dict], max_thesis: int
) -> list[dict]:
    """Pick up-to-``max_thesis`` wider-window 'thesis context' articles,
    excluding any whose (case-insensitive, trimmed) title already appears
    in the short-window ``breaking`` set or earlier in ``thesis`` itself.

    Pure, deterministic, order-preserving. The dedup is what makes the
    second news tier additive rather than a near-duplicate of the first.
    """
    if max_thesis <= 0:
        return []
    seen = {_norm_title(a.get("title")) for a in breaking}
    seen.discard("")
    out: list[dict] = []
    for a in thesis:
        key = _norm_title(a.get("title"))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(a)
        if len(out) >= max_thesis:
            break
    return out


def _paper_trader_position_lines(pt: Any) -> list[str]:
    """Render the live trader's OPEN book for the chat prompt.

    Reads the **marked** ``portfolio.positions`` array from :8090
    ``/api/state`` — it carries a real ``pl_pct`` and the ``stale_mark``
    flag — instead of the raw top-level ``positions`` array, which has
    neither (``store.open_positions()`` is an unenriched row read). Two
    correctness wins over the prior inline code:

    * the raw array has no ``pl_pct`` key, so ``(p.get('pl_pct') or 0)``
      printed ``(0.0%)`` for **every** stock regardless of actual P/L —
      the marked array surfaces the true percentage;
    * a position whose live price lookup failed (``stale_mark=True``,
      ``current_price == avg_cost``, P/L $0.00) is indistinguishable from
      a genuinely flat position. The chat is the user's primary surface;
      it now annotates it, mirroring the trader prompt's ``[STALE MARK …]``
      suffix (strategy.py) and the reporter's ``⚠ STALE`` line — both
      shipped for this exact live MU pathology.

    Falls back to the raw top-level array when the marked one is empty (a
    degraded ``get_portfolio()`` returns ``positions=[]`` while
    ``open_positions()`` still has rows) so a transient store blip never
    *loses* the book — it just shows it without a fabricated %.

    Pure / total — the ``_tail_risk_chat_lines`` contract: any bad shape
    degrades to the honest ``"Open positions: (none)"`` placeholder, never
    an exception into the chat handler.
    """
    if not isinstance(pt, dict):
        return ["Open positions: (none)"]
    pf = pt.get("portfolio")
    marked = pf.get("positions") if isinstance(pf, dict) else None
    rows = marked if isinstance(marked, list) and marked else None
    if rows is None:
        raw = pt.get("positions")
        rows = raw if isinstance(raw, list) and raw else None
    if not rows:
        return ["Open positions: (none)"]

    lines = ["Open positions:"]
    for p in rows[:15]:
        try:
            if not isinstance(p, dict):
                continue
            tk = p.get("ticker", "?")
            stale = " [STALE MARK: shown at cost, P/L unreliable]" if p.get(
                "stale_mark") else ""
            upl = p.get("unrealized_pl") or 0.0
            pl_pct = p.get("pl_pct")
            pct = f" ({pl_pct:+.1f}%)" if isinstance(
                pl_pct, (int, float)) else ""
            if p.get("type") in ("call", "put"):
                lines.append(
                    f"  {tk} {str(p.get('type', '')).upper()} "
                    f"{p.get('strike')} {p.get('expiry')}: "
                    f"qty={p.get('qty')} avg=${p.get('avg_cost')} "
                    f"mark=${p.get('current_price')} "
                    f"P/L=${upl:.2f}{pct}{stale}"
                )
            else:
                lines.append(
                    f"  {tk}: qty={p.get('qty')} "
                    f"avg=${(p.get('avg_cost') or 0):.2f} "
                    f"mark=${(p.get('current_price') or 0):.2f} "
                    f"P/L=${upl:.2f}{pct}{stale}"
                )
        except Exception:  # noqa: BLE001 — a chat block must never raise
            continue
    return lines


def _game_plan_chat_lines(gp: Any) -> list[str]:
    """Render the trader's own prioritised next-session action plan
    (``/api/game-plan``) as compact chat-context lines.

    Composed **verbatim** (paper-trader invariant #10 — single source of
    truth): the builder's ``headline`` and each HIGH ``portfolio_directives``
    ``text`` pass through UNCHANGED. The chat is the surface the operator
    actually asks "what should I do" on; before this it had to re-derive
    the answer from the raw analytics blocks.

    Pure / total — the ``_behavioural_chat_lines`` contract: a non-dict, an
    ``error`` key, a missing ``state``, or a ``NO_DATA`` gate → ``[]`` (the
    block is omitted, never an exception into the chat handler).
    """
    if (not isinstance(gp, dict) or "error" in gp
            or "state" not in gp or gp.get("state") == "NO_DATA"):
        return []
    hl = gp.get("headline")
    if not hl:
        return []
    lines = [f"Game plan: {hl}"]
    for c in (gp.get("position_actions") or []):
        if not isinstance(c, dict):
            continue
        act = str(c.get("action") or "").upper()
        if act in ("", "HOLD"):
            continue
        try:
            lines.append(
                f"  {act} {c.get('ticker')} "
                f"(conv {float(c.get('conviction') or 0):.2f}, "
                f"${float(c.get('unrealized_pl') or 0):+.2f}, "
                f"{float(c.get('pct_port') or 0):.1f}% port)"
            )
        except Exception:  # noqa: BLE001
            continue
    for d in (gp.get("portfolio_directives") or []):
        if isinstance(d, dict) and d.get("severity") == "HIGH" and d.get(
                "text"):
            lines.append(f"  • {d['text']}")
    opps = [o for o in (gp.get("opportunities") or [])
            if isinstance(o, dict) and o.get("ticker")]
    if opps:
        lines.append(
            "  opportunities: " + ", ".join(
                f"{o['ticker']}({str(o.get('action') or '').upper()} "
                f"{float(o.get('conviction') or 0):.2f})"
                for o in opps[:5]
            )
        )
    return lines


def _hold_discipline_chat_lines(hd: Any) -> list[str]:
    """Render the disposition-trap verdict (``/api/hold-discipline``).

    Mirrors ``reporter._hold_discipline_line`` exactly: emit the builder's
    ``headline`` **verbatim** (invariant #10) ONLY on ``DISPOSITION_DRAG``;
    ``DISCIPLINED`` / ``INSUFFICIENT`` / ``NO_DATA`` / error / bad shape →
    ``[]``. A "you're holding within discipline" verdict is not chat-worthy
    noise — the operator only needs to hear it when a losing position is
    being overstayed past the desk's own empirical losing-cut time.
    """
    if not isinstance(hd, dict) or "error" in hd:
        return []
    if hd.get("state") != "DISPOSITION_DRAG":
        return []
    hl = hd.get("headline")
    return [f"Hold discipline: {hl}"] if hl else []


def _coverage_gap_chat_lines(report: Any) -> list[str]:
    """Render the news-coverage-gap block for the chat prompt.

    A news analyst's most dangerous failure is a *silent* one: a high-value
    intel channel goes dark and the chat simply contains nothing from it, so
    the absence reads as "calm" rather than "blind here". The 5h Opus briefing
    already surfaces this; the chat — the operator's primary interactive
    surface — did not, so it would confidently answer "nothing notable on
    filings" while SEC 8-K had been dark all session.

    Composed from ``analysis.claude_analyst._coverage_gap_lines`` **verbatim**
    (the same single source of truth the briefing reproduces — invariant #10:
    no re-derived gap logic that can drift from the briefing the operator also
    reads); each SSOT line is wrapped as one bullet, ordering/cap unchanged.

    Pure / total — the ``_behavioural_chat_lines`` contract: a non-dict, an
    empty report, an import failure, or no curated channel disabled → ``[]``
    (the block is omitted, never an exception into the chat handler).
    """
    if not isinstance(report, dict) or not report:
        return []
    try:
        from analysis.claude_analyst import _coverage_gap_lines
        lines = _coverage_gap_lines(report)
    except Exception:  # noqa: BLE001 — a chat block must never raise
        return []
    return [f"• {ln}" for ln in lines] if lines else []


def _rerank_chat_news(rows: Any, limit: int, now=None) -> list:
    """Recency-decay rerank of chat news candidates, truncated to ``limit``.

    The chat news tiers were ordered by raw ``ai_score DESC``, so a stale
    high-score "SURGED TODAY" outranked a fresh slightly-lower one on the
    operator's primary surface. ArticleNet already trains a
    ``time_sensitivity`` head and the 5h briefing already decays by it;
    this applies that **same single source of truth**
    (``analysis.claude_analyst._rank_by_decayed_score`` /  ``_effective_score``:
    ``effective = ai_score * 0.5 ** (age_h * ts / 12h)``) so the chat and the
    briefing rank consistently instead of forking a second decay curve. The
    system-wide NULL policy (unscored ``time_sensitivity`` → mild default
    decay; unparseable ``first_seen`` → age 0 → no decay) lives in
    ``_effective_score`` and is locked by this module's tests.

    Pure / total: ``limit <= 0`` → ``[]``; on any failure (import error,
    malformed row) it degrades to the incoming order truncated to ``limit``
    — recency decay may only ever *help* the chat ranking, never sink the
    block or raise into the chat handler.
    """
    if limit <= 0:
        return []
    try:
        from analysis.claude_analyst import _rank_by_decayed_score
        return _rank_by_decayed_score(list(rows), now=now)[:limit]
    except Exception:  # noqa: BLE001 — decay can only help; never sink chat
        return list(rows)[:limit]


# ── Native news-density SECTOR PULSE ────────────────────────────────────────
# The only sector heatmap in the stack is cross-fetched from paper-trader
# (:8090/api/sector-heatmap — pure price momentum), so it blanks exactly when
# the trader is down/stale (its documented chronic state). digital-intern owns
# ~1000+ live scored articles/h; this derives a sector view *natively* from
# the wire — which slices of the book the news is lighting up right now,
# weighted toward fresh items — with zero dependence on paper-trader uptime.
#
# This is an explicit, test-locked taxonomy (NOT heuristic_scorer's keyword
# scoring tiers — those are not a sector map). It deliberately covers the
# user's stated universe: DRAM/memory, semis-equipment, the HBM-ramp design
# winners, and the leveraged semis ETFs the trader actually uses.
_SECTOR_MAP: dict[str, str] = {
    # DRAM / NAND / storage-media
    "MU": "DRAM/Memory", "WDC": "DRAM/Memory", "STX": "DRAM/Memory",
    # Wafer-fab equipment & materials
    "ASML": "Semis Equipment", "LRCX": "Semis Equipment",
    "AMAT": "Semis Equipment", "KLAC": "Semis Equipment",
    "TER": "Semis Equipment", "ENTG": "Semis Equipment",
    # GPU / accelerated-compute / HBM-design winners
    "NVDA": "GPU/AI Compute", "AMD": "GPU/AI Compute",
    "AVGO": "GPU/AI Compute",
    # Foundry / logic
    "TSM": "Foundry/Logic", "INTC": "Foundry/Logic", "GFS": "Foundry/Logic",
    # Mega-cap tech (hyperscaler demand)
    "AAPL": "Mega-Cap Tech", "MSFT": "Mega-Cap Tech",
    "GOOGL": "Mega-Cap Tech", "GOOG": "Mega-Cap Tech",
    "META": "Mega-Cap Tech", "AMZN": "Mega-Cap Tech",
    # Optical / networking (the trader holds LITE)
    "LITE": "Networking/Optical", "COHR": "Networking/Optical",
    "CIEN": "Networking/Optical",
    # EDA / IP
    "SNPS": "EDA/IP", "CDNS": "EDA/IP",
    # Semis index / leveraged ETFs the trader uses
    "SOXX": "Semis Index/ETF", "SMH": "Semis Index/ETF",
    "SOXL": "Semis Index/ETF", "SOXS": "Semis Index/ETF",
}
# Case-SENSITIVE, word-bounded. Tickers in real headlines are uppercase;
# a case-insensitive match would false-positive on ordinary prose ("mu"
# meson, "amd" in a URL). \b stops substring hits (EMU/SAMUEL → no MU).
# Longest-first alternation so GOOGL can't be shadowed by GOOG.
_SECTOR_TICKER_RE = re.compile(
    r"\b(?:" + "|".join(
        re.escape(t) for t in sorted(_SECTOR_MAP, key=len, reverse=True)
    ) + r")\b"
)
_SECTOR_PULSE_HALFLIFE_H = 6.0  # a sector's "velocity" halves every 6h dark
_SECTOR_PULSE_MAX_CHAT_LINES = 6


def _extract_tickers(text: Any) -> set:
    """Pure: the set of known watchlist tickers literally present (uppercase,
    word-bounded) in ``text``. Non-str / empty → ``set()`` (never raises)."""
    if not isinstance(text, str) or not text:
        return set()
    return {t for t in _SECTOR_TICKER_RE.findall(text) if t in _SECTOR_MAP}


def _aggregate_sector_pulse(
    articles: Any, window_hours=None, now=None
) -> dict:
    """Pure: roll a list of live article dicts into a per-sector news pulse.

    Each article maps (via its title) to zero or more sectors; per sector we
    track article count, mean/max ai_score, max urgency, the freshest
    timestamp, the highest-scored headline, and a **recency-weighted
    velocity** (``Σ 0.5 ** (age_h / 6h)``) so a sector lit by fresh wire
    outranks one with the same count of stale items — that recency tilt is
    the whole point of a *pulse* vs a flat count. Age uses the shared
    ``claude_analyst._seen_age_hours`` parser (same convention as the chat
    decay / alert pipeline); if that import fails every age degrades to 0
    (→ no recency tilt, velocity == count) rather than raising.

    Total: a non-list, or rows that aren't dicts / lack a usable title, are
    skipped — the result is always the well-formed skeleton, never an
    exception into the endpoint or chat handler.
    """
    out = {
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "window_hours": window_hours,
        "n_scanned": 0,
        "n_mapped": 0,
        "sectors": [],
    }
    if not isinstance(articles, list):
        return out
    try:
        from analysis.claude_analyst import _seen_age_hours
    except Exception:  # noqa: BLE001
        def _seen_age_hours(_fs, now=None):  # type: ignore
            return 0.0

    agg: dict[str, dict] = {}
    n_scanned = 0
    n_mapped = 0
    for art in articles:
        if not isinstance(art, dict):
            continue
        n_scanned += 1
        title = art.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        tks = _extract_tickers(title)
        sectors = {_SECTOR_MAP[t] for t in tks}
        if not sectors:
            continue
        n_mapped += 1
        try:
            ai = float(art.get("ai_score") or 0.0)
        except (TypeError, ValueError):
            ai = 0.0
        try:
            urg = int(art.get("urgency") or 0)
        except (TypeError, ValueError):
            urg = 0
        fs = art.get("first_seen")
        age_h = _seen_age_hours(fs, now=now)
        weight = 0.5 ** (age_h / _SECTOR_PULSE_HALFLIFE_H)
        for sec in sectors:
            s = agg.setdefault(sec, {
                "n": 0, "scores": [], "max_urg": 0, "vel": 0.0,
                "last_seen": None, "top": ("", -1.0), "tickers": set(),
            })
            s["n"] += 1
            s["scores"].append(ai)
            s["max_urg"] = max(s["max_urg"], urg)
            s["vel"] += weight
            if fs and (s["last_seen"] is None or str(fs) > str(s["last_seen"])):
                s["last_seen"] = fs
            if ai > s["top"][1]:
                s["top"] = (title, ai)
            s["tickers"] |= {t for t in tks if _SECTOR_MAP[t] == sec}

    sectors_out = []
    for sec, s in agg.items():
        scores = s["scores"] or [0.0]
        sectors_out.append({
            "sector": sec,
            "n_articles": s["n"],
            "avg_score": round(sum(scores) / len(scores), 2),
            "max_score": round(max(scores), 2),
            "max_urgency": s["max_urg"],
            "velocity": round(s["vel"], 4),
            "last_seen": s["last_seen"],
            "top_headline": s["top"][0],
            "tickers": sorted(s["tickers"]),
        })
    sectors_out.sort(
        key=lambda d: (-d["velocity"], -d["n_articles"], d["sector"]))
    out["n_scanned"] = n_scanned
    out["n_mapped"] = n_mapped
    out["sectors"] = sectors_out
    return out


def _sector_pulse_chat_lines(pulse: Any) -> list:
    """Render the native sector pulse as compact chat-context lines.

    One line per hottest sector (cap ``_SECTOR_PULSE_MAX_CHAT_LINES``), in the
    aggregator's velocity order, surfacing count, velocity, avg score, an
    URGENT flag, and the sector's lead headline so the analyst sees where the
    wire is concentrated even when paper-trader's price heatmap is dark.

    Pure / total — the ``_tail_risk_chat_lines`` contract: non-dict / no
    sectors → ``[]`` (block omitted, never an exception into the handler).
    """
    if not isinstance(pulse, dict):
        return []
    sectors = pulse.get("sectors")
    if not isinstance(sectors, list) or not sectors:
        return []
    lines = []
    for s in sectors[:_SECTOR_PULSE_MAX_CHAT_LINES]:
        if not isinstance(s, dict):
            continue
        urgent = " URGENT" if (s.get("max_urgency") or 0) >= 1 else ""
        try:
            lines.append(
                f"{s.get('sector', '?')}: {int(s.get('n_articles') or 0)} "
                f"arts (vel {float(s.get('velocity') or 0):.1f}, avg "
                f"{float(s.get('avg_score') or 0):.1f}{urgent}) · "
                f"{s.get('top_headline', '')}"
            )
        except Exception:  # noqa: BLE001 — a chat block must never raise
            continue
    return lines


_PORTFOLIO_SIGNALS_MAX_HEADLINES = 5


def build_portfolio_signals(articles: list, now: datetime | None = None) -> dict:
    """Deterministic, always-fresh per-held-ticker live-news digest.

    The 5h Opus heartbeat briefing (``/api/briefings``) is the synthesised
    "what matters" view but is up to 5h stale AND skipped entirely whenever the
    Claude org usage limit is exhausted — a chronic, repeatedly-documented
    failure mode (paper-trader CLAUDE.md §11). ``/api/articles`` is the raw
    unbucketed feed; ``/api/sector-pulse`` is sector-level density;
    ``/api/trends`` is aggregate sentiment over time. None answers the held-book
    trader's between-briefings question: *"what fresh news touches MY exact
    positions right now, ranked, with the one headline to read first?"* — and
    none answers it WITHOUT a Claude call, so they are all dark on quota
    exactly when the trader most needs a read.

    This composes the briefing SSOT helpers verbatim
    (``_filter_quote_widget_noise`` + ``_book_tickers`` +
    ``_rank_by_decayed_score`` from ``analysis.claude_analyst``) so the held
    universe and the recency decay can NEVER drift from what the 5h Opus
    digest itself uses. Pure: no LLM, no subprocess, no network — feed it the
    DB rows; ``now`` is injectable for deterministic tests.

    The held universe is ``claude_analyst._BOOK_TICKERS`` (daemon-parity,
    parity-pinned by ``tests/test_briefing_book_tag.py``), **not** a live
    ``config/portfolio.json`` read — a name added via
    ``PUT /api/portfolio/config`` will not appear here until ``_BOOK_TICKERS``
    updates. That SSOT-with-the-briefing tradeoff is deliberate: this panel and
    the Opus digest must always agree on what "the book" is.

    Every held ticker is returned (even with zero fresh articles — a pinned
    book explicitly wants to see "no news on LITE in 24h"), sorted by top
    recency-decayed score desc, canonical ``_BOOK_TICKERS`` order breaking ties.
    """
    from analysis.claude_analyst import (
        _BOOK_TICKERS,
        _book_tickers,
        _effective_score,
        _filter_quote_widget_noise,
        _rank_by_decayed_score,
    )

    now = now or datetime.now(timezone.utc)
    kept, suppressed = _filter_quote_widget_noise(list(articles or []))

    buckets: dict[str, list] = {tk: [] for tk in _BOOK_TICKERS}
    for a in kept:
        for tk in _book_tickers(a):
            if tk in buckets:
                buckets[tk].append(a)

    out_tickers: list[dict] = []
    for tk in _BOOK_TICKERS:
        arts = _rank_by_decayed_score(buckets[tk], now=now)
        if arts:
            top = arts[0]
            out_tickers.append({
                "ticker": tk,
                "n_articles": len(arts),
                "top_score": round(
                    max(_effective_score(x, now=now) for x in arts), 4),
                "max_urgency": max(int(x.get("urgency") or 0) for x in arts),
                "top_headline": top.get("title"),
                "top_source": top.get("source"),
                "top_first_seen": top.get("first_seen"),
                "headlines": [
                    {
                        "title": x.get("title"),
                        "source": x.get("source"),
                        "ai_score": float(x.get("ai_score") or 0.0),
                        "urgency": int(x.get("urgency") or 0),
                        "first_seen": x.get("first_seen"),
                    }
                    for x in arts[:_PORTFOLIO_SIGNALS_MAX_HEADLINES]
                ],
            })
        else:
            out_tickers.append({
                "ticker": tk,
                "n_articles": 0,
                "top_score": 0.0,
                "max_urgency": 0,
                "top_headline": None,
                "top_source": None,
                "top_first_seen": None,
                "headlines": [],
            })

    book_index = {tk: i for i, tk in enumerate(_BOOK_TICKERS)}
    out_tickers.sort(key=lambda r: (-r["top_score"], book_index[r["ticker"]]))

    return {
        "as_of": now.isoformat(),
        "n_articles_scanned": len(kept),
        "n_quote_widget_suppressed": len(suppressed),
        "n_tickers_with_news": sum(1 for r in out_tickers if r["n_articles"]),
        "tickers": out_tickers,
    }


def build_news_corroboration(
    articles: list,
    *,
    min_sources: int = 2,
    jaccard_threshold: float = 0.6,
    max_clusters: int = 50,
    now: datetime | None = None,
) -> dict:
    """Group syndicated articles by title-token Jaccard similarity and rank by
    how many DISTINCT sources carried the same story.

    Single-source urgency is the dominant false-positive in the live feed —
    one wire-recap headline ("Why <ticker> Trading Up Today") can spike
    ``ai_score`` past 9.0 and even trip ``urgency=2`` while no other outlet
    has touched the story. Multi-source corroboration is the cheap, model-
    independent signal a desk-side analyst would use to triage: if Reuters +
    Bloomberg + Yahoo + CNBC all carry the same beat, it's real news; if a
    single Google News aggregator wrapper carries it, it's noise to ignore
    until a second source confirms.

    SSOT: clustering reuses ``ml.dedup.title_tokens`` and
    ``ml.dedup.jaccard_similarity`` verbatim — the same near-duplicate
    primitives the briefing's domain-diversity / near-dup-collapse rely on —
    so this view of "what story is this article about" cannot drift from how
    the rest of the pipeline normalises titles.

    Pure: no DB, no LLM, no network. Caller passes a list of article dicts
    (must have ``title``, ``source``; ``ai_score`` / ``urgency`` /
    ``first_seen`` / ``url`` optional). Output cluster ranking:
      1. ``n_sources`` DESC  (most-corroborated first)
      2. ``max_ai_score`` DESC  (within tie, highest-quality)
      3. ``latest_first_seen`` DESC  (within tie, freshest)

    Articles whose title yields no tokens (empty / pure-punct) form
    standalone clusters — they will never reach ``min_sources >= 2`` unless
    the impossible (multiple identical empty-title articles from distinct
    sources) which is fine: this defaults the no-signal case to "skip".
    """
    from ml.dedup import jaccard_similarity, title_tokens

    now = now or datetime.now(timezone.utc)
    arts = list(articles or [])

    clusters: list[dict] = []
    for art in arts:
        toks = title_tokens(art.get("title"))
        placed = False
        if toks:
            for cl in clusters:
                anchor = cl["anchor_tokens"]
                if anchor and jaccard_similarity(toks, anchor) >= jaccard_threshold:
                    cl["articles"].append(art)
                    src = (art.get("source") or "").strip()
                    if src:
                        cl["sources"].add(src)
                    placed = True
                    break
        if not placed:
            src = (art.get("source") or "").strip()
            clusters.append({
                "anchor_tokens": toks,
                "anchor_title": art.get("title") or "",
                "articles": [art],
                "sources": {src} if src else set(),
            })

    def _ai(a: dict) -> float:
        try:
            return float(a.get("ai_score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _urg(a: dict) -> int:
        try:
            return int(a.get("urgency") or 0)
        except (TypeError, ValueError):
            return 0

    out_clusters: list[dict] = []
    for cl in clusters:
        n_sources = len(cl["sources"])
        if n_sources < min_sources:
            continue
        members = cl["articles"]
        top = max(members, key=_ai)
        first_seens = [a.get("first_seen") for a in members if a.get("first_seen")]
        latest = max(first_seens) if first_seens else None
        out_clusters.append({
            "headline": top.get("title") or cl["anchor_title"],
            "top_source": top.get("source"),
            "top_url": top.get("url"),
            "n_articles": len(members),
            "n_sources": n_sources,
            "sources": sorted(cl["sources"]),
            "max_ai_score": round(max((_ai(a) for a in members), default=0.0), 3),
            "max_urgency": max((_urg(a) for a in members), default=0),
            "latest_first_seen": latest,
        })

    out_clusters.sort(
        key=lambda c: (-c["n_sources"], -c["max_ai_score"],
                       -(len(c["latest_first_seen"]) if c["latest_first_seen"] else 0),
                       c["latest_first_seen"] or "")
    )

    n_kept = len(out_clusters)
    if max_clusters and n_kept > max_clusters:
        out_clusters = out_clusters[:max_clusters]

    return {
        "as_of": now.isoformat(),
        "n_articles_scanned": len(arts),
        "n_clusters_formed": len(clusters),
        "n_multi_source": n_kept,
        "min_sources": int(min_sources),
        "jaccard_threshold": float(jaccard_threshold),
        "clusters": out_clusters,
    }


# ── breaking-confluence (NOW-focused velocity view) ─────────────────────────
# /api/news-corroboration is a 6h trust filter ranked by n_sources — a stale
# but well-corroborated cluster sits at the top forever. /api/event-threads
# is a 24h recency-decayed impact ranking that keeps singletons. Neither
# answers the desk question on a fresh login at 13:57 EDT: "what is breaking
# RIGHT NOW with confirmation building?" — i.e. small window (60m default),
# urgency-weighted, with arrival VELOCITY and a verdict that distinguishes a
# CONFIRMED 5-source story from one still EMERGING in the last 30 minutes.
# Same Jaccard primitive (ml.dedup) so this can never drift from the other
# clustering surfaces; the differentiation is purely the window, the
# velocity term, and the verdict ladder. Single-source clusters are
# filtered by default (min_sources=2 — the corroboration discipline) but a
# fresh hot singleton (urgency ≥ 1 AND ai_score ≥ 9) is kept under a
# SINGLETON_HOT verdict so a solo Reuters 8-K isn't lost before the wire
# picks it up (event_threads precedent for keeping single-article threads).
_BREAKING_DEFAULT_WINDOW_MIN = 60
_BREAKING_EMERGING_MIN = 30  # latest-seen within this -> EMERGING
_BREAKING_JACCARD = 0.6  # match build_news_corroboration / event_threads
_BREAKING_CONFIRMED_SOURCES = 3
_BREAKING_HOT_SINGLETON_URGENCY = 1
_BREAKING_HOT_SINGLETON_SCORE = 9.0


def build_breaking_confluence(
    articles: list,
    *,
    now: datetime | None = None,
    window_minutes: int = _BREAKING_DEFAULT_WINDOW_MIN,
    emerging_window_minutes: int = _BREAKING_EMERGING_MIN,
    jaccard_threshold: float = _BREAKING_JACCARD,
    min_sources: int = 2,
    min_score: float = 5.0,
    max_clusters: int = 30,
) -> dict:
    """Group hot recent articles by title-token Jaccard, rank by arrival
    velocity, classify CONFIRMED / EMERGING / SINGLETON_HOT.

    Pure: no DB, no LLM, no network. Caller passes a list of article dicts
    with the same shape ``build_news_corroboration`` expects (``title``,
    ``source``, ``first_seen`` required; ``ai_score`` / ``urgency`` /
    ``url`` optional). Articles outside the ``window_minutes`` window are
    dropped before clustering; articles whose ``ai_score < min_score`` are
    also dropped (default 5.0 — drops kw_score-only rows that never made it
    past the model).

    Output cluster shape:
      headline, top_source, top_url, n_articles, n_sources, sources,
      max_ai_score, max_urgency, first_seen, latest_seen,
      first_seen_min_ago, latest_seen_min_ago,
      velocity_per_30min  (articles per 30 minutes within window),
      verdict  (CONFIRMED / EMERGING / SINGLETON_HOT)

    Ranking: (-verdict_rank, -recency_score, -n_sources, -max_ai_score).
    ``recency_score = 1 / (1 + latest_seen_min_ago / 10)`` — soft, so a
    cluster with 3 sources 12 min ago beats one with 2 sources 1 min ago.
    """
    from ml.dedup import jaccard_similarity, title_tokens

    now = now or datetime.now(timezone.utc)
    window_minutes = max(5, int(window_minutes))
    emerging_window_minutes = max(1, int(emerging_window_minutes))
    arts_all = list(articles or [])
    window_start = now - timedelta(minutes=window_minutes)

    def _ai(a: dict) -> float:
        try:
            return float(a.get("ai_score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _urg(a: dict) -> int:
        try:
            return int(a.get("urgency") or 0)
        except (TypeError, ValueError):
            return 0

    def _ts(a: dict) -> datetime | None:
        s = a.get("first_seen")
        if not s:
            return None
        try:
            d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d

    # Window + score floor pre-filter so clustering only ever sees fresh+hot.
    arts: list[dict] = []
    for a in arts_all:
        if _ai(a) < min_score:
            continue
        ts = _ts(a)
        if ts is None or ts < window_start:
            continue
        arts.append(a)

    # Greedy single-link Jaccard clustering — same recipe as
    # build_news_corroboration; the anchor token set is fixed at first
    # article so later additions cannot widen it (drift guard).
    clusters: list[dict] = []
    for art in arts:
        toks = title_tokens(art.get("title"))
        placed = False
        if toks:
            for cl in clusters:
                anchor = cl["anchor_tokens"]
                if anchor and jaccard_similarity(toks, anchor) >= jaccard_threshold:
                    cl["articles"].append(art)
                    src = (art.get("source") or "").strip()
                    if src:
                        cl["sources"].add(src)
                    placed = True
                    break
        if not placed:
            src = (art.get("source") or "").strip()
            clusters.append({
                "anchor_tokens": toks,
                "anchor_title": art.get("title") or "",
                "articles": [art],
                "sources": {src} if src else set(),
            })

    window_min_f = float(window_minutes)
    out_clusters: list[dict] = []
    for cl in clusters:
        n_sources = len(cl["sources"])
        members = cl["articles"]
        n_articles = len(members)
        max_score = max((_ai(a) for a in members), default=0.0)
        max_urg = max((_urg(a) for a in members), default=0)
        top = max(members, key=_ai)
        member_ts = [t for t in (_ts(a) for a in members) if t is not None]
        if not member_ts:
            continue
        first_ts = min(member_ts)
        latest_ts = max(member_ts)
        first_min_ago = max(0.0, (now - first_ts).total_seconds() / 60.0)
        latest_min_ago = max(0.0, (now - latest_ts).total_seconds() / 60.0)

        # Singletons: keep only if hot enough to matter on their own.
        if n_sources < min_sources:
            if not (n_sources == 1
                    and max_urg >= _BREAKING_HOT_SINGLETON_URGENCY
                    and max_score >= _BREAKING_HOT_SINGLETON_SCORE
                    and latest_min_ago <= emerging_window_minutes):
                continue
            verdict = "SINGLETON_HOT"
        elif n_sources >= _BREAKING_CONFIRMED_SOURCES:
            verdict = "CONFIRMED"
        elif latest_min_ago <= emerging_window_minutes:
            verdict = "EMERGING"
        else:
            verdict = "CONFIRMED"  # 2 sources but stale → still corroborated

        velocity_per_30min = round(
            n_articles * (30.0 / window_min_f), 3
        )

        out_clusters.append({
            "headline": top.get("title") or cl["anchor_title"],
            "top_source": top.get("source"),
            "top_url": top.get("url"),
            "n_articles": n_articles,
            "n_sources": n_sources,
            "sources": sorted(cl["sources"]),
            "max_ai_score": round(max_score, 3),
            "max_urgency": max_urg,
            "first_seen": first_ts.isoformat(),
            "latest_seen": latest_ts.isoformat(),
            "first_seen_min_ago": round(first_min_ago, 2),
            "latest_seen_min_ago": round(latest_min_ago, 2),
            "velocity_per_30min": velocity_per_30min,
            "verdict": verdict,
        })

    _VERDICT_RANK = {"CONFIRMED": 0, "EMERGING": 1, "SINGLETON_HOT": 2}

    def _sort_key(c: dict) -> tuple:
        latest = c["latest_seen_min_ago"]
        recency = 1.0 / (1.0 + latest / 10.0)
        return (
            _VERDICT_RANK.get(c["verdict"], 9),
            -recency,
            -c["n_sources"],
            -c["max_ai_score"],
        )

    out_clusters.sort(key=_sort_key)
    n_kept = len(out_clusters)
    if max_clusters and n_kept > max_clusters:
        out_clusters = out_clusters[:max_clusters]

    counts_by_verdict = {"CONFIRMED": 0, "EMERGING": 0, "SINGLETON_HOT": 0}
    for c in out_clusters:
        counts_by_verdict[c["verdict"]] = counts_by_verdict.get(
            c["verdict"], 0) + 1

    return {
        "as_of": now.isoformat(),
        "window_minutes": int(window_minutes),
        "emerging_window_minutes": int(emerging_window_minutes),
        "min_sources": int(min_sources),
        "min_score": float(min_score),
        "jaccard_threshold": float(jaccard_threshold),
        "n_articles_scanned": len(arts_all),
        "n_articles_in_window": len(arts),
        "n_clusters_formed": len(clusters),
        "n_surfaced": n_kept,
        "counts_by_verdict": counts_by_verdict,
        "clusters": out_clusters,
    }


# ── event-thread clustering (trader-facing event view of the feed) ──────────
# /api/news-corroboration answers "what stories are multi-source confirmed?"
# (trust filter, ranks by n_sources). /api/event-threads answers a different
# question the live feed currently fails: "what *distinct events* happened
# recently, ranked by impact × recency?" Same Jaccard clustering primitive,
# but:
#   - single-article threads are KEPT (a solo Reuters 8-K matters before
#     anyone else picks it up — the corroboration view filters these out)
#   - per-thread tickers + sectors are extracted (the analyst routes a thread
#     to a held-position decision via the same _SECTOR_MAP the sector-pulse
#     and chat use, no drift)
#   - ranking is recency-decayed impact (max_ai_score × 0.5^(age_h/halflife))
#     so the eye lands on what just broke, not on stale-but-corroborated
#   - member articles are surfaced (cap ``_EVENT_THREAD_MEMBER_CAP``) so the
#     trader can drill into supporting evidence without a second query
_EVENT_THREAD_HALFLIFE_H = 6.0  # match _SECTOR_PULSE_HALFLIFE_H — same eye
_EVENT_THREAD_MEMBER_CAP = 5
_EVENT_THREAD_JACCARD = 0.6  # match build_news_corroboration default


def build_event_threads(
    articles: list,
    *,
    min_score: float = 5.0,
    min_articles: int = 1,
    jaccard_threshold: float = _EVENT_THREAD_JACCARD,
    max_threads: int = 30,
    now: datetime | None = None,
) -> dict:
    """Cluster recent articles into distinct event threads, ranked by impact.

    Pure / total — no DB, no LLM, no network. Caller passes article dicts
    (must have ``title``; ``source`` / ``url`` / ``ai_score`` / ``urgency`` /
    ``first_seen`` optional). Empty / non-list / titleless inputs collapse to
    the well-formed empty skeleton, never an exception.

    Clustering is greedy Jaccard on ``ml.dedup.title_tokens`` — same primitive
    ``build_news_corroboration``, ``build_alert_confidence_trend``, and the
    briefing's near-dup-collapse use, so this view of "what story is this
    article about" cannot drift from the rest of the pipeline.

    Per-thread enrichment:
      - ``tickers``: ``_extract_tickers`` over the union of member titles
        (case-sensitive, word-bounded; longest-first so GOOGL beats GOOG —
        the same regex the sector-pulse uses, no separate copy)
      - ``sectors``: ``_SECTOR_MAP`` lookup over those tickers
      - ``impact_score``: ``max_ai_score × 0.5 ** (age_h / halflife)`` —
        same recency-decay shape as the sector-pulse velocity, so a fresh
        max=8 thread outranks a stale max=10 (which is what a trader sees
        when scrolling the feed)
      - ``members``: up to ``_EVENT_THREAD_MEMBER_CAP`` (title, source, url,
        ai_score, urgency, first_seen), highest-score first

    Filtering:
      - ``min_score`` — drop threads whose ``max_ai_score`` is below this
        (default 5.0; the same threshold ``/api/signals`` uses internally
        to skip noise). Set 0 to see everything.
      - ``min_articles`` — drop threads with fewer member articles than this
        (default 1; KEEPS single-article threads — the differentiator from
        ``build_news_corroboration``'s ``min_sources >= 2`` filter).
    """
    from ml.dedup import jaccard_similarity, title_tokens

    now = now or datetime.now(timezone.utc)
    # Strict input contract: must be a list/tuple of dicts. Anything else
    # (None, dict, str, int) collapses to the empty skeleton — never raises,
    # never scans an unintended iterable (e.g. a dict's keys).
    if not isinstance(articles, (list, tuple)):
        arts: list = []
    else:
        arts = list(articles)

    def _f(v, default=0.0):
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    def _i(v, default=0):
        try:
            return int(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    clusters: list[dict] = []
    for art in arts:
        if not isinstance(art, dict):
            continue
        title = art.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        toks = title_tokens(title)
        placed = False
        if toks:
            for cl in clusters:
                if cl["anchor_tokens"] and jaccard_similarity(
                    toks, cl["anchor_tokens"]
                ) >= jaccard_threshold:
                    cl["members"].append(art)
                    src = (art.get("source") or "").strip()
                    if src:
                        cl["sources"].add(src)
                    placed = True
                    break
        if not placed:
            src = (art.get("source") or "").strip()
            clusters.append({
                "anchor_tokens": toks,
                "members": [art],
                "sources": {src} if src else set(),
            })

    out_threads: list[dict] = []
    for cl in clusters:
        members = cl["members"]
        if len(members) < max(1, int(min_articles)):
            continue
        scores = [_f(m.get("ai_score")) for m in members]
        max_score = max(scores) if scores else 0.0
        if max_score < float(min_score):
            continue
        urgs = [_i(m.get("urgency")) for m in members]
        first_seens = [m.get("first_seen") for m in members if m.get("first_seen")]
        # ISO timestamps sort lexically — same convention sector_pulse uses.
        first_seen = min(first_seens) if first_seens else None
        latest_seen = max(first_seens) if first_seens else None
        # Tickers: union over all member titles (a Samsung HBM4 piece may
        # mention only WDC while the Reuters companion piece mentions MU —
        # union is the trader's actual exposure surface).
        all_tickers: set[str] = set()
        for m in members:
            all_tickers |= _extract_tickers(m.get("title"))
        sectors = sorted({_SECTOR_MAP[t] for t in all_tickers if t in _SECTOR_MAP})

        # Recency-decayed impact. Reuse claude_analyst._seen_age_hours so the
        # ranking time-zero is identical to the chat decay + sector-pulse
        # velocity time-zero (no drift). If it can't be imported, fall back
        # to a parse here — never raise into the route.
        try:
            from analysis.claude_analyst import _seen_age_hours
            age_h = _seen_age_hours(latest_seen, now=now) if latest_seen else 0.0
        except Exception:  # noqa: BLE001
            ts = _parse_first_seen(latest_seen) if latest_seen else None
            age_h = ((now - ts).total_seconds() / 3600.0) if ts else 0.0
        impact = max_score * (0.5 ** (max(0.0, age_h) / _EVENT_THREAD_HALFLIFE_H))

        # Anchor = highest-scoring member title (canonical headline).
        top = max(members, key=lambda m: _f(m.get("ai_score")))
        # Surface members highest-score first, cap to _EVENT_THREAD_MEMBER_CAP.
        sorted_members = sorted(
            members, key=lambda m: -_f(m.get("ai_score"))
        )[: _EVENT_THREAD_MEMBER_CAP]
        members_out = [{
            "title": m.get("title") or "",
            "source": m.get("source") or "",
            "url": m.get("url"),
            "ai_score": round(_f(m.get("ai_score")), 2),
            "urgency": _i(m.get("urgency")),
            "first_seen": m.get("first_seen"),
        } for m in sorted_members]

        out_threads.append({
            "anchor_title": top.get("title") or "",
            "anchor_url": top.get("url"),
            "n_articles": len(members),
            "n_sources": len(cl["sources"]),
            "sources": sorted(cl["sources"]),
            "tickers": sorted(all_tickers),
            "sectors": sectors,
            "max_ai_score": round(max_score, 2),
            "max_urgency": max(urgs) if urgs else 0,
            "first_seen": first_seen,
            "latest_seen": latest_seen,
            "age_hours": round(age_h, 2),
            "impact_score": round(impact, 3),
            "members": members_out,
        })

    out_threads.sort(
        key=lambda t: (-t["impact_score"], -t["n_articles"],
                       -t["n_sources"], t["anchor_title"])
    )
    if max_threads and len(out_threads) > max_threads:
        out_threads = out_threads[:max_threads]

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "n_articles_scanned": len(arts),
        "n_clusters_formed": len(clusters),
        "n_threads_kept": len(out_threads),
        "min_score": float(min_score),
        "min_articles": int(min_articles),
        "jaccard_threshold": float(jaccard_threshold),
        "halflife_hours": _EVENT_THREAD_HALFLIFE_H,
        "threads": out_threads,
    }


def create_app(store=None) -> Flask:
    """Build the Flask app. ``store`` is the shared ArticleStore from daemon.py."""
    global _store
    if store is not None:
        _store = store

    app = Flask(__name__)

    @app.after_request
    def _cors(resp):
        # Public read-only dashboard — wide-open CORS so the Paper Trader
        # dashboard (different port, same host) can fetch /api/articles for
        # the cross-linked signal feed.
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
        resp.headers.setdefault("Access-Control-Allow-Methods", "GET, OPTIONS")
        resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
        return resp

    api_key_required = os.environ.get("WEB_API_KEY", "").strip()

    def _check_api_key() -> bool:
        if not api_key_required:
            return True
        return request.args.get("key", "") == api_key_required

    @app.get("/")
    def index() -> Response:
        prefix = request.headers.get("X-Forwarded-Prefix", "").rstrip("/")
        html = _DASHBOARD_HTML.replace("__API_PREFIX__", prefix)
        return Response(html, mimetype="text/html")

    @app.get("/api/articles")
    def api_articles():
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        limit = max(1, min(500, int(request.args.get("limit", 50))))
        min_score = float(request.args.get("min_score", 0.0))
        return jsonify(_articles_from_db(limit, min_score))

    @app.get("/api/portfolio")
    def api_portfolio():
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        snap = _read_json(BASE_DIR / "data" / "portfolio_pl.json")
        if snap is None:
            return jsonify({"error": "no snapshot yet"}), 503
        return jsonify(snap)

    CONFIG_PATH = BASE_DIR / "config" / "portfolio.json"

    @app.get("/api/portfolio/config")
    def api_portfolio_config_get():
        import json as _json
        try:
            data = _json.loads(CONFIG_PATH.read_text())
        except Exception:
            data = {"positions": [], "options": [], "sector_watchlist": []}
        return jsonify(data)

    @app.put("/api/portfolio/config")
    def api_portfolio_config_put():
        import json as _json
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "invalid JSON"}), 400
        # Keep _note and _account metadata if present
        existing = {}
        try:
            existing = _json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
        body["_note"] = f"Sao's trading portfolio - updated via UI"
        body.setdefault("_account", existing.get("_account", {}))
        CONFIG_PATH.write_text(_json.dumps(body, indent=2))
        return jsonify({"ok": True})

    @app.get("/api/stats")
    def api_stats():
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        store = _store_handle()
        if store is None:
            return jsonify({"error": "store unavailable"}), 503
        # ``store.stats()`` is the core gauge (total/urgent/backlog). If it
        # exhausts the @_retry_on_lock budget under a sustained writer-
        # contention / shared-conn cursor-collision storm it genuinely means
        # the store is unreachable this instant → 500.
        try:
            s = dict(store.stats())
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        # ``last_hour``/``last_24h`` are SUPPLEMENTARY window tiles. Each is its
        # own @_retry_on_lock call; before this split, a transient retry-
        # exhaustion on EITHER of these blanked the entire /api/stats payload
        # with a 500 — the whole dashboard went dark over a slow supplementary
        # count while the core gauge was perfectly healthy. Degrade them
        # independently: keep the keys present (so a client doing ``s.last_hour``
        # gets null, never an undefined-property crash) and flag ``degraded``.
        degraded = False
        for key, hours in (("last_hour", 1), ("last_24h", 24)):
            try:
                s[key] = store.stats_since(hours)
            except Exception:
                s[key] = None
                degraded = True
        if degraded:
            s["degraded"] = True
        try:
            trends = _read_json(BASE_DIR / "data" / "sentiment_trends.json")
            if trends:
                s["trends_as_of"] = trends.get("as_of")
                s["trends_tracked"] = len(trends.get("tickers", {}))
        except Exception:
            pass
        return jsonify(s)

    @app.get("/api/trends")
    def api_trends():
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        data = _read_json(BASE_DIR / "data" / "sentiment_trends.json")
        if data is None:
            return jsonify({"error": "no trends yet"}), 503
        return jsonify(data)

    @app.get("/api/briefings")
    def api_briefings():
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        return jsonify(_briefings_from_log(10))

    @app.get("/api/earnings")
    def api_earnings():
        """Upcoming earnings within the snapshot's horizon (default 14d).

        Reads ``data/earnings_calendar.json`` written by ``write_snapshot()``.
        Also reports the snapshot age so the dashboard can render a freshness
        indicator and trigger a background refresh when stale.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        snap = _read_json(BASE_DIR / "data" / "earnings_calendar.json")
        if snap is None:
            return jsonify({
                "error": "no snapshot yet",
                "hint": "run: python3 -m collectors.earnings_calendar",
            }), 503
        # Recompute days_away on each request so a stale snapshot still shows
        # accurate counters until the daemon refreshes it.
        now = datetime.now(timezone.utc)
        try:
            for ev in snap.get("events", []) or []:
                ts = ev.get("earnings_date")
                if not ts:
                    continue
                try:
                    ed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    continue
                if ed.tzinfo is None:
                    ed = ed.replace(tzinfo=timezone.utc)
                ev["days_away"] = round((ed - now).total_seconds() / 86400.0, 2)
        except Exception:
            pass
        # Drop events that have already passed once the recompute runs,
        # then sort soonest-first so dashboards can render in order.
        snap["events"] = sorted(
            [e for e in snap.get("events", []) or []
             if (e.get("days_away") or 0) >= -0.5],
            key=lambda e: e.get("days_away") if e.get("days_away") is not None else float("inf"),
        )
        snap["n_events"] = len(snap["events"])
        snap["n_within_7d"] = sum(
            1 for e in snap["events"]
            if -0.5 <= (e.get("days_away") or 0) <= 7.0
        )
        snap["next_event"] = snap["events"][0] if snap["events"] else None
        # Snapshot age for staleness rendering.
        try:
            as_of = datetime.fromisoformat((snap.get("as_of") or "").replace("Z", "+00:00"))
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=timezone.utc)
            snap["age_hours"] = round((now - as_of).total_seconds() / 3600.0, 2)
        except Exception:
            snap["age_hours"] = None
        return jsonify(snap)

    def _ro_conn():
        """Open a fresh read-only sqlite connection to the daemon's articles.db.

        Mirrors the resolution logic used in /api/chat: prefer the path the
        ArticleStore actually opened, then fall back to USB and local repo
        paths. Returns ``None`` if no DB can be located.
        """
        db_path: Path | None = None
        store = _store_handle()
        if store is not None:
            try:
                for _id, name, file in store.conn.execute("PRAGMA database_list").fetchall():
                    if name == "main" and file:
                        db_path = Path(file)
                        break
            except Exception:
                pass
        if db_path is None:
            for cand in (
                Path("/media/zeph/projects/digital-intern/db/articles.db"),
                BASE_DIR / "data" / "articles.db",
                BASE_DIR / "db" / "articles.db",
            ):
                if cand.exists():
                    db_path = cand
                    break
        if db_path is None:
            return None
        try:
            uri = f"file:{db_path}?mode=ro"
            return sqlite3.connect(uri, uri=True, timeout=5.0)
        except sqlite3.Error:
            return None

    _LIVE_ONLY_SQL = (
        "url NOT LIKE 'backtest://%' "
        "AND source NOT LIKE 'backtest_%' "
        "AND source NOT LIKE 'opus_annotation%'"
    )

    @app.get("/api/sector-pulse")
    def api_sector_pulse():
        """Native news-density sector heatmap — which slices of the book the
        wire is lighting up right now, recency-weighted, computed purely from
        ``articles.db`` (independent of paper-trader's price heatmap, which
        blanks when the trader is down/stale). ``?hours=`` clamped 1..168.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            hours = int(request.args.get("hours", 24))
        except (TypeError, ValueError):
            hours = 24
        hours = max(1, min(168, hours))
        since = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        try:
            rows = _ro_query(
                "SELECT title, ai_score, urgency, first_seen FROM articles "
                f"WHERE first_seen >= ? AND {_LIVE_ONLY_SQL} "
                "ORDER BY first_seen DESC LIMIT 4000",
                (since,),
            )
        except sqlite3.Error:
            rows = []
        arts = [
            {"title": r[0] or "", "ai_score": float(r[1] or 0),
             "urgency": int(r[2] or 0), "first_seen": r[3]}
            for r in rows
        ]
        return jsonify(_aggregate_sector_pulse(arts, window_hours=hours))

    @app.get("/api/portfolio-signals")
    def api_portfolio_signals():
        """Deterministic, always-fresh per-held-ticker live-news digest —
        the between-briefings read that survives Claude-quota exhaustion.

        The 5h Opus briefing is stale up to 5h and is skipped entirely when
        the org usage limit is hit; ``/api/articles`` is raw/unbucketed;
        ``/api/sector-pulse`` is sector-level. This buckets fresh live
        articles onto the exact held book (``_BOOK_TICKERS`` daemon-parity),
        recency-decay-ranks them with the briefing's own SSOT curve, and
        surfaces the one headline to read first per position — with NO LLM
        call. ``?hours=`` clamped 1..168 (default 24).
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            hours = int(request.args.get("hours", 24))
        except (TypeError, ValueError):
            hours = 24
        hours = max(1, min(168, hours))
        since = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        try:
            rows = _ro_query(
                "SELECT title, ai_score, urgency, first_seen, "
                "time_sensitivity, source FROM articles "
                f"WHERE first_seen >= ? AND {_LIVE_ONLY_SQL} "
                "ORDER BY first_seen DESC LIMIT 4000",
                (since,),
            )
        except sqlite3.Error:
            rows = []
        arts = [
            {"title": r[0] or "", "ai_score": float(r[1] or 0),
             "urgency": int(r[2] or 0), "first_seen": r[3],
             "time_sensitivity": r[4], "source": r[5] or ""}
            for r in rows
        ]
        out = build_portfolio_signals(arts)
        out["window_hours"] = hours
        return jsonify(out)

    @app.get("/api/news-corroboration")
    def api_news_corroboration():
        """Multi-source story confirmation feed — rank live clusters by how
        many distinct collectors carried the same headline.

        Single-source urgency is the dominant false-positive in the feed
        (a wire-recap "Why X Trading Up Today" can spike ai_score past 9
        without any other outlet confirming). This buckets fresh articles
        by ``ml.dedup`` title-token Jaccard (same primitive the briefing's
        near-dup-collapse uses) and returns clusters with ``n_sources >=
        min_sources`` ranked by corroboration count, then quality, then
        freshness. ``?hours=`` clamped 1..168 (default 6).
        ``?min_sources=`` clamped 2..10 (default 2).
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            hours = int(request.args.get("hours", 6))
        except (TypeError, ValueError):
            hours = 6
        hours = max(1, min(168, hours))
        try:
            min_sources = int(request.args.get("min_sources", 2))
        except (TypeError, ValueError):
            min_sources = 2
        min_sources = max(2, min(10, min_sources))
        since = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        try:
            rows = _ro_query(
                "SELECT title, source, url, ai_score, urgency, first_seen "
                "FROM articles "
                f"WHERE first_seen >= ? AND {_LIVE_ONLY_SQL} "
                "ORDER BY first_seen DESC LIMIT 4000",
                (since,),
            )
        except sqlite3.Error:
            rows = []
        arts = [
            {"title": r[0] or "", "source": r[1] or "", "url": r[2],
             "ai_score": float(r[3] or 0), "urgency": int(r[4] or 0),
             "first_seen": r[5]}
            for r in rows
        ]
        out = build_news_corroboration(arts, min_sources=min_sources)
        out["window_hours"] = hours
        return jsonify(out)

    @app.get("/api/breaking-confluence")
    def api_breaking_confluence():
        """What is BREAKING right now with confirmation building?

        Same Jaccard clustering primitive as ``/api/news-corroboration``
        and ``/api/event-threads`` (``ml.dedup`` — no drift) but anchored
        to a tight NOW-focused window so a desk reopening at 14:00 EDT
        sees only the clusters that grew in the last hour, not a stale
        well-corroborated wire from this morning. Each cluster carries
        a verdict:

          * CONFIRMED — ≥3 distinct sources have the story
          * EMERGING — 2 sources AND latest article within
            ``emerging_window_minutes``
          * SINGLETON_HOT — 1 source but ``urgency >= 1`` AND
            ``ai_score >= 9`` AND latest within the emerging window;
            keeps a solo Reuters 8-K visible before the wire confirms.

        Query (all optional):
          - ``window_minutes`` — recency window (default 60, clamp 5..720)
          - ``emerging_minutes`` — EMERGING latest-seen cutoff (default
            30, clamp 1..window_minutes)
          - ``min_score`` — ai_score floor (default 5.0, clamp 0..10)
          - ``min_sources`` — corroboration floor for non-singletons
            (default 2, clamp 1..10; <2 = also include all non-hot
            singletons, noisy)
          - ``max_clusters`` — cap (default 30, clamp 1..100)
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401

        def _qi(name: str, default: int, lo: int, hi: int) -> int:
            try:
                v = int(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        def _qf(name: str, default: float, lo: float, hi: float) -> float:
            try:
                v = float(request.args.get(name, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        window_minutes = _qi("window_minutes", 60, 5, 720)
        emerging_minutes = _qi(
            "emerging_minutes", 30, 1, window_minutes)
        min_score = _qf("min_score", 5.0, 0.0, 10.0)
        min_sources = _qi("min_sources", 2, 1, 10)
        max_clusters = _qi("max_clusters", 30, 1, 100)

        since = (
            datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        ).isoformat()
        try:
            rows = _ro_query(
                "SELECT title, source, url, ai_score, urgency, first_seen "
                "FROM articles "
                f"WHERE first_seen >= ? AND {_LIVE_ONLY_SQL} "
                "ORDER BY first_seen DESC LIMIT 4000",
                (since,),
            )
        except sqlite3.Error:
            rows = []
        arts = [
            {"title": r[0] or "", "source": r[1] or "", "url": r[2],
             "ai_score": float(r[3] or 0), "urgency": int(r[4] or 0),
             "first_seen": r[5]}
            for r in rows
        ]
        out = build_breaking_confluence(
            arts,
            window_minutes=window_minutes,
            emerging_window_minutes=emerging_minutes,
            min_score=min_score,
            min_sources=min_sources,
            max_clusters=max_clusters,
        )
        return jsonify(out)

    @app.get("/api/event-threads")
    def api_event_threads():
        """Distinct event view of the live feed — clustered by title-token
        Jaccard, enriched with watchlist tickers + sectors, ranked by
        recency-decayed impact (max_ai_score × 0.5^(age_h/halflife)).

        Complements ``/api/news-corroboration`` (which filters to ≥2 sources,
        ranks by corroboration count): this surface KEEPS single-article
        threads — a solo Reuters 8-K matters before the wire picks it up —
        and routes each thread to held positions via ``_SECTOR_MAP``.

        Query:
          - ``hours`` clamped 1..168, default 24
          - ``min_score`` clamped 0..10, default 5.0 (drops noise; 0 = all)
          - ``min_articles`` clamped 1..20, default 1 (keep solo events)
          - ``max_threads`` clamped 1..100, default 30
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            hours = int(request.args.get("hours", 24))
        except (TypeError, ValueError):
            hours = 24
        hours = max(1, min(168, hours))
        try:
            min_score = float(request.args.get("min_score", 5.0))
        except (TypeError, ValueError):
            min_score = 5.0
        min_score = max(0.0, min(10.0, min_score))
        try:
            min_articles = int(request.args.get("min_articles", 1))
        except (TypeError, ValueError):
            min_articles = 1
        min_articles = max(1, min(20, min_articles))
        try:
            max_threads = int(request.args.get("max_threads", 30))
        except (TypeError, ValueError):
            max_threads = 30
        max_threads = max(1, min(100, max_threads))
        since = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        try:
            rows = _ro_query(
                "SELECT title, source, url, ai_score, urgency, first_seen "
                "FROM articles "
                f"WHERE first_seen >= ? AND {_LIVE_ONLY_SQL} "
                "ORDER BY first_seen DESC LIMIT 4000",
                (since,),
            )
        except sqlite3.Error:
            rows = []
        arts = [
            {"title": r[0] or "", "source": r[1] or "", "url": r[2],
             "ai_score": float(r[3] or 0), "urgency": int(r[4] or 0),
             "first_seen": r[5]}
            for r in rows
        ]
        out = build_event_threads(
            arts,
            min_score=min_score,
            min_articles=min_articles,
            max_threads=max_threads,
        )
        out["window_hours"] = hours
        return jsonify(out)

    @app.get("/api/collector-health")
    def api_collector_health():
        """Per-source article counts for last 1h and 24h with status thresholds.

        - active: ≥10 articles in the last hour
        - slow:   1..9  articles in the last hour
        - stale:  0 articles in the last 2 hours
        - idle:   anything else (e.g. recently active but quiet in the past hour)
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        conn = _ro_conn()
        if conn is None:
            return jsonify({"sources": [], "error": "articles.db not reachable"})
        try:
            rows_1h = conn.execute(
                f"SELECT source, COUNT(*) FROM articles "
                f"WHERE first_seen >= datetime('now','-1 hour') AND {_LIVE_ONLY_SQL} "
                f"GROUP BY source"
            ).fetchall()
            rows_24h = conn.execute(
                f"SELECT source, COUNT(*) FROM articles "
                f"WHERE first_seen >= datetime('now','-24 hours') AND {_LIVE_ONLY_SQL} "
                f"GROUP BY source"
            ).fetchall()
            rows_2h = conn.execute(
                f"SELECT source, COUNT(*) FROM articles "
                f"WHERE first_seen >= datetime('now','-2 hours') AND {_LIVE_ONLY_SQL} "
                f"GROUP BY source"
            ).fetchall()
        finally:
            conn.close()
        c1h = {r[0] or "?": int(r[1] or 0) for r in rows_1h}
        c2h = {r[0] or "?": int(r[1] or 0) for r in rows_2h}
        c24 = {r[0] or "?": int(r[1] or 0) for r in rows_24h}
        names = set(c1h) | set(c2h) | set(c24)
        out = []
        for n in names:
            h1 = c1h.get(n, 0)
            h2 = c2h.get(n, 0)
            h24 = c24.get(n, 0)
            if h2 == 0:
                status = "stale"
            elif h1 >= 10:
                status = "active"
            elif h1 >= 1:
                status = "slow"
            else:
                status = "idle"
            out.append({
                "source": n,
                "articles_1h": h1,
                "articles_24h": h24,
                "status": status,
            })
        out.sort(key=lambda r: (-r["articles_1h"], -r["articles_24h"], r["source"]))
        return jsonify({"sources": out, "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    @app.get("/api/ml-status")
    def api_ml_status():
        """ArticleNet snapshot — last trained, training-set size, predictions today.

        Pulls the checkpoint mtime as ``last_trained`` and grep-scans the
        structured log for the most recent ``[ml_trainer] Bootstrap done`` line
        to recover ``val_loss`` (the trainer logs it on every retrain, see
        ml/trainer.py).
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        ckpt = BASE_DIR / "data" / "ml" / "model_gpu.pt"
        last_trained = None
        if ckpt.exists():
            try:
                last_trained = datetime.fromtimestamp(
                    ckpt.stat().st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds")
            except Exception:
                last_trained = None
        training_set_size = None
        predictions_24h = None
        urgent_24h = None
        conn = _ro_conn()
        if conn is not None:
            try:
                # ArticleNet trains on rows with any ML/LLM-assigned score;
                # `kw_score` is the pure-heuristic fallback we exclude here.
                training_set_size = int(conn.execute(
                    "SELECT COUNT(*) FROM articles WHERE ai_score > 0"
                ).fetchone()[0] or 0)
                # Articles scored in the past 24h are a reasonable proxy for
                # inference throughput; there is no `score_source` column in
                # this schema (see articles table definition).
                predictions_24h = int(conn.execute(
                    "SELECT COUNT(*) FROM articles WHERE ai_score > 0 "
                    f"AND first_seen >= datetime('now','-24 hours') AND {_LIVE_ONLY_SQL}"
                ).fetchone()[0] or 0)
                urgent_24h = int(conn.execute(
                    "SELECT COUNT(*) FROM articles WHERE urgency >= 1 "
                    f"AND first_seen >= datetime('now','-24 hours') AND {_LIVE_ONLY_SQL}"
                ).fetchone()[0] or 0)
            except Exception:
                pass
            finally:
                conn.close()
        val_loss = None
        try:
            log_path = BASE_DIR / "logs" / "structured.jsonl"
            if log_path.exists():
                size = log_path.stat().st_size
                with log_path.open("rb") as f:
                    f.seek(max(0, size - 512 * 1024))
                    data = f.read()
                for raw in reversed(data.splitlines()):
                    try:
                        ln = json.loads(raw.decode("utf-8", errors="replace"))
                    except Exception:
                        continue
                    msg = ln.get("msg", "")
                    if "[ml_trainer] Bootstrap done" in msg or "val_loss" in msg:
                        # crude extraction: find val_loss=… or 'val_loss': …
                        import re as _re
                        m = _re.search(r"val_loss['\":=\s]+([0-9]+\.?[0-9]*)", msg)
                        if m:
                            try:
                                val_loss = float(m.group(1))
                                break
                            except Exception:
                                pass
        except Exception:
            pass
        return jsonify({
            "last_trained": last_trained,
            "training_set_size": training_set_size,
            "predictions_24h": predictions_24h,
            "urgent_24h": urgent_24h,
            "val_loss": val_loss,
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    @app.get("/api/volume-history")
    def api_volume_history():
        """Hourly article ingest counts for the last 24 hours, live rows only."""
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        conn = _ro_conn()
        if conn is None:
            return jsonify({"hours": [], "error": "articles.db not reachable"})
        try:
            rows = conn.execute(
                "SELECT strftime('%Y-%m-%dT%H:00', first_seen) AS hour, COUNT(*) "
                "FROM articles "
                f"WHERE first_seen >= datetime('now','-24 hours') AND {_LIVE_ONLY_SQL} "
                "GROUP BY hour ORDER BY hour"
            ).fetchall()
        finally:
            conn.close()
        return jsonify({
            "hours": [{"hour": r[0], "count": int(r[1] or 0)} for r in rows],
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    @app.get("/api/invariants")
    def api_invariants():
        """Backtest data-isolation status.

        Per the cross-system invariant (digital-intern CLAUDE.md §5): any
        ``backtest://`` row that has been alerted is a contamination breach —
        live alerts must only fire on live news.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        conn = _ro_conn()
        if conn is None:
            return jsonify({"backtest_isolation": "unknown", "error": "articles.db not reachable"})
        try:
            breach = int(conn.execute(
                "SELECT COUNT(*) FROM articles "
                "WHERE (url LIKE 'backtest://%' OR source LIKE 'backtest_%' "
                "      OR source LIKE 'opus_annotation%') "
                "AND urgency >= 2"
            ).fetchone()[0] or 0)
            n_backtest_total = int(conn.execute(
                "SELECT COUNT(*) FROM articles "
                "WHERE url LIKE 'backtest://%' OR source LIKE 'backtest_%' "
                "      OR source LIKE 'opus_annotation%'"
            ).fetchone()[0] or 0)
        finally:
            conn.close()
        return jsonify({
            "backtest_isolation": "breach" if breach > 0 else "active",
            "breach_count": breach,
            "backtest_rows_total": n_backtest_total,
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    @app.get("/api/urgent-label-split")
    def api_urgent_label_split():
        """Per-``score_source`` breakdown of urgent rows in the last ``hours``.

        Exposes ``ArticleStore.urgency_label_split`` so an analyst consuming
        the standalone-push channel can see the aggregate calibration story
        at a glance: of every urgency>=1 row in the window, what fraction
        carry a real LLM ground-truth label vs only a model self-prediction?

        Live evidence in ``article_store.py`` (2026-05-19): every urgent row
        the alerter saw in a 6h window had ``ai_score=0`` (model-only). The
        per-row "[unverified — model-only urgent]" alert tag exists for that
        case, but nothing exposed the AGGREGATE rate at a glance — when the
        Sonnet urgency_scorer is dark, quota-throttled or flooring everything
        to noise, the standalone-push channel becomes single-headed and the
        analyst should know.

        Query params:
          ``hours`` — window size, clamped 1..168 (default 24).

        Returns (passthrough from ``urgency_label_split`` plus a verdict):
          ``window_h``       — int
          ``total``          — total urgent rows in window (urgency>=1)
          ``by_source``      — {"llm": N, "ml": N, "briefing_boost": N, "null": N}
          ``llm_fraction``   — (llm + briefing_boost) / total (0.0 when total==0)
          ``status``         — "quiet" (total==0), "healthy" (>=50% LLM),
                                "mostly_unverified" (<50% LLM with total>=5),
                                "unverified_storm" (0% LLM with total>=3)
          ``as_of``          — ISO8601 UTC timestamp
        Read-only — no DB writes; backtest rows excluded by the underlying
        method's ``_LIVE_ONLY_CLAUSE``; all four load-bearing invariants are
        preserved by construction.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        store = _store_handle()
        if store is None:
            return jsonify({"error": "store unavailable"}), 503
        try:
            hours = int(request.args.get("hours", 24))
        except (TypeError, ValueError):
            hours = 24
        hours = max(1, min(168, hours))
        try:
            data = store.urgency_label_split(hours=hours)
        except Exception as exc:
            return jsonify({"error": f"urgency_label_split failed: {exc!s}"}), 500
        total = int(data.get("total") or 0)
        llm_fraction = float(data.get("llm_fraction") or 0.0)
        # Verdict — same silence-vs-signal discipline the chat enrichment
        # blocks use (event-readiness / macro-calendar): a quiet window
        # collapses to "quiet" instead of inventing a problem, and only an
        # actionable miscalibration emits a non-healthy status.
        if total == 0:
            status = "quiet"
        elif llm_fraction == 0.0 and total >= 3:
            status = "unverified_storm"
        elif llm_fraction < 0.5 and total >= 5:
            status = "mostly_unverified"
        else:
            status = "healthy"
        data["status"] = status
        data["as_of"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return jsonify(data)

    @app.get("/api/source-throughput")
    def api_source_throughput():
        """Per-source recent-vs-prior article rate + deceleration percentage.

        Exposes ``ArticleStore.source_throughput`` so the analyst gets a
        leading indicator BEFORE a collector goes fully dark. A source can
        decelerate sharply (40/h → 3/h) while its newest item is still only
        minutes old — ``/api/collector-health`` (which only carries 1h/24h
        counts) won't flag it; this endpoint will.

        Query params:
          ``window_min`` — window size in minutes, clamped 5..720 (default 60).
          ``limit``      — max rows returned, clamped 1..200 (default 50).

        Returns (passthrough from ``source_throughput`` plus a verdict):
          ``window_min``  — int
          ``sources``     — [{source, recent, prior, delta, decel_pct}, ...],
                             most-decelerated first
          ``n_critical``  — count of sources with decel_pct >= 75
          ``n_degraded``  — count of sources with decel_pct >= 40 and < 75
          ``status``      — "ok" if no critical/degraded; "degraded" if any
                             degraded but no critical; "critical" if any
                             critical
          ``as_of``       — ISO8601 UTC timestamp
        Read-only — backtest rows excluded by the underlying method's
        ``_LIVE_ONLY_CLAUSE``; all four load-bearing invariants preserved.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        store = _store_handle()
        if store is None:
            return jsonify({"error": "store unavailable"}), 503
        try:
            window_min = int(request.args.get("window_min", 60))
        except (TypeError, ValueError):
            window_min = 60
        window_min = max(5, min(720, window_min))
        try:
            limit = int(request.args.get("limit", 50))
        except (TypeError, ValueError):
            limit = 50
        limit = max(1, min(200, limit))
        try:
            rows = store.source_throughput(window_min=window_min)
        except Exception as exc:
            return jsonify({"error": f"source_throughput failed: {exc!s}"}), 500
        # Verdict thresholds chosen to match the analyst's "do I need to
        # look at this?" bar: 75%+ drop = source effectively dark for the
        # window; 40-75% = degrading enough to investigate before the next
        # briefing. None=brand-new sources are NOT flagged (no baseline).
        #
        # ``MIN_PRIOR_FOR_VERDICT`` excludes one-off sub-tag noise (a long-
        # tail aggregator key seen once last hour, zero this hour) from the
        # verdict count. Live evidence (2026-05-20, 60-min window): 8+
        # ``GDELT/<host>`` / ``AlphaVantage/<host>`` rows hit decel_pct=100
        # purely because prior was 1 — a normal long-tail fluctuation, not a
        # degradation worth alerting on. Without this floor the verdict
        # collapsed to ``critical`` on every cycle. The full ``sources``
        # list is still returned (the operator can still see them), but
        # only sources with a meaningful baseline drive the verdict.
        MIN_PRIOR_FOR_VERDICT = 5
        n_critical = sum(
            1 for r in rows
            if isinstance(r.get("decel_pct"), (int, float))
            and r["decel_pct"] >= 75
            and int(r.get("prior") or 0) >= MIN_PRIOR_FOR_VERDICT
        )
        n_degraded = sum(
            1 for r in rows
            if isinstance(r.get("decel_pct"), (int, float))
            and 40 <= r["decel_pct"] < 75
            and int(r.get("prior") or 0) >= MIN_PRIOR_FOR_VERDICT
        )
        if n_critical:
            status = "critical"
        elif n_degraded:
            status = "degraded"
        else:
            status = "ok"
        return jsonify({
            "window_min": window_min,
            "sources": rows[:limit],
            "n_critical": n_critical,
            "n_degraded": n_degraded,
            "status": status,
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    @app.get("/api/db-lock-health")
    def api_db_lock_health():
        """Live article_store DB-lock contention surface.

        Pairs ``article_store.lock_metrics()`` (process-lifetime retry +
        failure counters; the dashboard runs in-process with the daemon so
        the numbers are the same ones article_store has been incrementing)
        with a recent-window tail of ``logs/daemon.log`` for
        ``lock retry exhausted`` lines, which is the operator-visible
        symptom of WAL+busy_timeout=60s+5-retry exhaustion under heavy
        writer contention (see article_store._retry_on_lock).

        Status thresholds (1h window):
          - ok:        0 exhausted failures
          - degraded:  1..9 failures (transient contention)
          - critical:  ≥10 failures (sustained contention storm)
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        from storage import article_store as _as
        try:
            metrics = _as.lock_metrics()
        except Exception:
            metrics = {"lock_retries": None, "lock_failures": None}
        log_path = BASE_DIR / "logs" / "daemon.log"
        failures_1h = 0
        retries_1h = 0
        last_failure_ts: str | None = None
        if log_path.exists():
            try:
                size = log_path.stat().st_size
                # Tail the last ~256KB; lock-retry storms produce many
                # lines so a generous window survives bursts without
                # scanning the whole file.
                with open(log_path, "rb") as f:
                    if size > 256 * 1024:
                        f.seek(size - 256 * 1024)
                        f.readline()  # drop partial line
                    tail = f.read().decode("utf-8", errors="replace")
                cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
                for line in tail.splitlines():
                    if "lock retry exhausted" in line:
                        ts_match = line[:20]
                        try:
                            ts = datetime.strptime(
                                ts_match, "%Y-%m-%dT%H:%M:%SZ"
                            ).replace(tzinfo=timezone.utc)
                        except ValueError:
                            continue
                        if ts >= cutoff:
                            failures_1h += 1
                            last_failure_ts = ts.isoformat(timespec="seconds")
                    elif "transient DB error" in line and "retrying in" in line:
                        ts_match = line[:20]
                        try:
                            ts = datetime.strptime(
                                ts_match, "%Y-%m-%dT%H:%M:%SZ"
                            ).replace(tzinfo=timezone.utc)
                        except ValueError:
                            continue
                        if ts >= cutoff:
                            retries_1h += 1
            except Exception as exc:
                return jsonify({
                    "error": f"log scan failed: {exc!s}",
                    "lock_retries": metrics.get("lock_retries"),
                    "lock_failures": metrics.get("lock_failures"),
                }), 500
        if failures_1h >= 10:
            status = "critical"
        elif failures_1h >= 1:
            status = "degraded"
        else:
            status = "ok"
        return jsonify({
            "status": status,
            "lock_retries_lifetime": metrics.get("lock_retries"),
            "lock_failures_lifetime": metrics.get("lock_failures"),
            "retries_1h": retries_1h,
            "failures_1h": failures_1h,
            "last_failure_ts": last_failure_ts,
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    @app.get("/api/score-distribution")
    def api_score_distribution():
        """ai_score drift surface: 24h histogram + 7d baseline + mean delta.

        Backed by ml.score_distribution.snapshot() — read-only, SQL-aggregated,
        with bounded-sample percentiles so it stays cheap on the multi-GB
        USB-backed DB.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            from ml.score_distribution import snapshot as _score_snapshot
            snap = _score_snapshot()
        except Exception as exc:
            return jsonify({"error": f"snapshot failed: {exc}"}), 500
        snap["as_of"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return jsonify(snap)

    @app.get("/api/news-arrival-rhythm")
    def api_news_arrival_rhythm():
        """Per-source hour-of-day urgent-article distribution.

        Operator visibility into *when* urgent news lands and *from
        which source*. Complementary to ``collector_uptime`` (silence
        gaps) and ``source_throughput`` (rate deceleration) — those
        flag failure; this surfaces baseline cadence.

        Query params (all clamped):
          ``hours``       — lookback window, 1..168 (default 24)
          ``min_urgency`` — floor, 0..2 (default 1)
          ``top_sources`` — display cap on the per-source list,
                            1..50 (default 10)

        Reads articles.db via the dashboard's ``_ro_query`` short-lived
        read-only connection (the source_throughput / score_distribution
        precedent — never competes for the daemon's writer lock). The
        ``_LIVE_ONLY_CLAUSE`` is applied — backtest-injected rows
        cannot leak into the operator panel (invariant #5).

        Pure builder (``analytics.news_arrival_rhythm``) handles the
        bucketing; this route is the SQL adapter only.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            hours = int(request.args.get("hours", 24))
        except (TypeError, ValueError):
            hours = 24
        hours = max(1, min(168, hours))
        try:
            min_urgency = int(request.args.get("min_urgency", 1))
        except (TypeError, ValueError):
            min_urgency = 1
        min_urgency = max(0, min(2, min_urgency))
        try:
            top_sources = int(request.args.get("top_sources", 10))
        except (TypeError, ValueError):
            top_sources = 10
        top_sources = max(1, min(50, top_sources))

        from storage.article_store import _LIVE_ONLY_CLAUSE
        from analytics.news_arrival_rhythm import build_news_arrival_rhythm

        cutoff = (datetime.now(timezone.utc) -
                  timedelta(hours=hours)).isoformat(timespec="seconds")
        try:
            rows = _ro_query(
                f"""SELECT source, urgency, first_seen
                      FROM articles
                     WHERE {_LIVE_ONLY_CLAUSE}
                       AND urgency >= ?
                       AND first_seen >= ?
                     ORDER BY first_seen DESC
                     LIMIT 20000""",
                (min_urgency, cutoff),
            )
        except sqlite3.Error as exc:
            return jsonify({"error": f"db: {exc!s}"}), 500
        arts = [{"source": r[0], "urgency": r[1], "first_seen": r[2]}
                for r in rows]
        return jsonify(build_news_arrival_rhythm(
            arts, hours=hours, min_urgency=min_urgency,
            top_sources=top_sources,
        ))

    @app.get("/api/briefing-coverage-audit")
    def api_briefing_coverage_audit():
        """Did the latest 5h briefing actually mention the urgent tickers?

        Retrospective audit: pulls the latest ``briefings`` row, pulls every
        ``urgency >= 1`` article that fired between the prior briefing's
        timestamp and the latest briefing's timestamp (5h fallback when no
        prior briefing is on record), and classifies each book ticker with
        urgent flow as COVERED (mentioned anywhere in the briefing text) or
        MISSED (absent despite urgent stories).

        Complementary to the *prospective* enrichment helpers
        ``_coverage_gap_lines`` / ``_book_silence_lines`` (which tell Opus
        what to mention *before* he writes) — this verifies the published
        output *after* he's written.

        Query params:
          ``card_cap`` — per-side row cap on covered/missed lists, 1..50
                          (default 12). The aggregate counts always reflect
                          the full set; the cap truncates display rows only.

        Reads articles.db + briefings via ``_ro_query`` (short-lived
        ``mode=ro`` connection — same precedent as
        ``api_news_arrival_rhythm`` / ``api_score_distribution``). The
        ``_LIVE_ONLY_CLAUSE`` is applied to the urgent-article scan so
        backtest rows can never poison the audit (invariant #5).
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            card_cap = int(request.args.get("card_cap", 12))
        except (TypeError, ValueError):
            card_cap = 12
        card_cap = max(1, min(50, card_cap))

        from storage.article_store import _LIVE_ONLY_CLAUSE
        from analytics.briefing_coverage_audit import (
            build_briefing_coverage_audit,
        )

        try:
            briefing_rows = _ro_query(
                "SELECT ts, text, article_count FROM briefings "
                "ORDER BY ts DESC LIMIT 2"
            )
        except sqlite3.Error as exc:
            return jsonify({"error": f"db: {exc!s}"}), 500

        if not briefing_rows:
            return jsonify(build_briefing_coverage_audit(None, []))

        latest = {
            "ts": briefing_rows[0][0],
            "text": briefing_rows[0][1],
            "article_count": briefing_rows[0][2],
        }
        # Window = (prior_briefing_ts, latest_briefing_ts]. Falls back to a
        # 5h heartbeat-cadence lookback when only one briefing exists.
        try:
            latest_dt = datetime.fromisoformat(
                str(latest["ts"]).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return jsonify(build_briefing_coverage_audit(latest, []))
        if not latest_dt.tzinfo:
            latest_dt = latest_dt.replace(tzinfo=timezone.utc)

        if len(briefing_rows) > 1 and briefing_rows[1][0]:
            try:
                start_dt = datetime.fromisoformat(
                    str(briefing_rows[1][0]).replace("Z", "+00:00"))
                if not start_dt.tzinfo:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                start_dt = latest_dt - timedelta(hours=5)
        else:
            start_dt = latest_dt - timedelta(hours=5)

        # The articles table has no ``summary`` column — wire copy stores
        # body in ``full_text`` (zlib BLOB) and the analyst-pass-through
        # routes lazily decode it. For a coverage audit, the title alone is
        # the high-signal source: financial wires write the ticker in the
        # title; body decompression for thousands of rows would dominate the
        # request budget. The builder already tolerates missing ``summary``.
        try:
            urgent_rows = _ro_query(
                f"""SELECT title, urgency, source, first_seen
                      FROM articles
                     WHERE {_LIVE_ONLY_CLAUSE}
                       AND urgency >= 1
                       AND first_seen >= ?
                       AND first_seen <= ?
                     ORDER BY first_seen DESC
                     LIMIT 5000""",
                (start_dt.isoformat(timespec="seconds"),
                 latest_dt.isoformat(timespec="seconds")),
            )
        except sqlite3.Error as exc:
            return jsonify({"error": f"db: {exc!s}"}), 500
        arts = [{"title": r[0], "urgency": r[1],
                 "source": r[2], "first_seen": r[3]}
                for r in urgent_rows]
        return jsonify(build_briefing_coverage_audit(
            latest, arts,
            window_start=start_dt, window_end=latest_dt,
            card_cap=card_cap,
        ))

    @app.get("/healthz")
    @app.get("/api/health")
    def healthz():
        # /api/health is an alias for /healthz so dashboard.html's health
        # badge (and any external probe using the conventional /api/health
        # path) stops silently 404'ing. Same body either way.
        store = _store_handle()
        return jsonify({"ok": store is not None})

    @app.get("/chat")
    def chat_page() -> Response:
        return Response(_CHAT_HTML, mimetype="text/html",
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

    @app.post("/api/chat")
    def api_chat():
        try:
            payload = request.get_json(force=True, silent=True) or {}
        except Exception:
            payload = {}
        user_msg = (payload.get("message") or "").strip()
        history = payload.get("history") or []
        if not user_msg:
            return jsonify({"error": "empty message"}), 400

        # Pull article context — open a fresh read-only sqlite connection so we
        # don't clash with the daemon's writer thread.
        articles_ctx: list[dict] = []
        thesis_ctx: list[dict] = []
        try:
            # Resolve the actual sqlite file the daemon's ArticleStore is using
            # (could be the USB mount at /media/zeph/projects/digital-intern/db).
            db_path: Path | None = None
            store = _store_handle()
            if store is not None:
                try:
                    for _id, name, file in store.conn.execute("PRAGMA database_list").fetchall():
                        if name == "main" and file:
                            db_path = Path(file)
                            break
                except Exception:
                    pass
            if db_path is None:
                # Fallbacks: USB mount first, then local repo path.
                for cand in (
                    Path("/media/zeph/projects/digital-intern/db/articles.db"),
                    BASE_DIR / "db" / "articles.db",
                ):
                    if cand.exists():
                        db_path = cand
                        break
            since = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
            since_thesis = (
                datetime.now(timezone.utc) - timedelta(hours=48)
            ).isoformat()
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            try:
                # Pull a WIDER candidate set than we ultimately show (10 / 8)
                # and order by ai_score DESC so the recency-decay rerank below
                # can actually promote a fresh slightly-lower row over a stale
                # high one — and so the degrade path (rerank import/parse
                # failure) still hands back the old ai_score-DESC order.
                rows = conn.execute(
                    "SELECT title, source, ai_score, full_text, "
                    "time_sensitivity, first_seen FROM articles "
                    "WHERE first_seen >= ? "
                    "AND url NOT LIKE 'backtest://%' "
                    "AND source NOT LIKE 'backtest_%' "
                    "AND source NOT LIKE 'opus_annotation%' "
                    "ORDER BY ai_score DESC LIMIT 40",
                    (since,),
                ).fetchall()
                # Wider 48h "thesis context" tier — multi-day narrative the
                # single 6h breaking window cannot carry. Same live-only
                # filter (invariant: backtest/opus rows must never reach a
                # live surface); recency-reranked then deduped against
                # `breaking` below.
                thesis_rows = conn.execute(
                    "SELECT title, source, ai_score, full_text, "
                    "time_sensitivity, first_seen FROM articles "
                    "WHERE first_seen >= ? "
                    "AND url NOT LIKE 'backtest://%' "
                    "AND source NOT LIKE 'backtest_%' "
                    "AND source NOT LIKE 'opus_annotation%' "
                    "ORDER BY ai_score DESC LIMIT 60",
                    (since_thesis,),
                ).fetchall()
            finally:
                conn.close()

            def _decode_summary(blob) -> str:
                if blob is None:
                    return ""
                try:
                    return zlib.decompress(blob).decode("utf-8", errors="replace")[:300]
                except Exception:
                    try:
                        return (blob if isinstance(blob, str)
                                else blob.decode("utf-8", "replace"))[:300]
                    except Exception:
                        return ""

            def _mk(r) -> dict:
                # `time_sensitivity` / `first_seen` feed the shared recency
                # decay (analysis.claude_analyst._effective_score); they are
                # ranking inputs only and are NOT rendered into the prompt.
                return {
                    "title": r[0] or "",
                    "source": r[1] or "",
                    "ai_score": float(r[2] or 0),
                    "summary": _decode_summary(r[3]),
                    "time_sensitivity": r[4],
                    "first_seen": r[5],
                }

            # Recency-decay rerank both tiers with the SAME curve the 5h Opus
            # briefing uses, then cut to the displayed 10 / 8. A stale 9.0 no
            # longer outranks a fresh 8.6 on the operator's primary surface.
            articles_ctx = _rerank_chat_news([_mk(r) for r in rows], 10)
            thesis_ranked = _rerank_chat_news(
                [_mk(r) for r in thesis_rows], len(thesis_rows))
            thesis_ctx = _partition_thesis_articles(
                articles_ctx, thesis_ranked, 8)
        except Exception as e:
            _logger().warning("chat: article context fetch failed: %s", e)

        # News-coverage gap — which curated intel channels were dark this
        # window. Without it the chat answers "nothing notable on filings"
        # when SEC 8-K has been blind all session. Same SSOT the 5h briefing
        # uses; best-effort, a read failure simply omits the block.
        coverage_gap_block = ""
        try:
            from analysis.claude_analyst import _collect_source_health
            coverage_gap_block = "\n".join(
                _coverage_gap_chat_lines(_collect_source_health()))
        except Exception as e:  # noqa: BLE001 — never sink the chat
            _logger().warning("chat: coverage-gap fetch failed: %s", e)

        # Native news-density sector pulse — computed from articles.db, so it
        # answers "where is the wire concentrated" even when paper-trader's
        # price heatmap (fetched below) is dark. Best-effort; a read failure
        # simply omits the block.
        sector_pulse_block = ""
        try:
            sp_since = (
                datetime.now(timezone.utc) - timedelta(hours=24)
            ).isoformat()
            sp_rows = _ro_query(
                "SELECT title, ai_score, urgency, first_seen FROM articles "
                "WHERE first_seen >= ? "
                "AND url NOT LIKE 'backtest://%' "
                "AND source NOT LIKE 'backtest_%' "
                "AND source NOT LIKE 'opus_annotation%' "
                "ORDER BY first_seen DESC LIMIT 4000",
                (sp_since,),
            )
            sp = _aggregate_sector_pulse(
                [{"title": r[0] or "", "ai_score": float(r[1] or 0),
                  "urgency": int(r[2] or 0), "first_seen": r[3]}
                 for r in sp_rows],
                window_hours=24,
            )
            sector_pulse_block = "\n".join(_sector_pulse_chat_lines(sp))
        except Exception as e:  # noqa: BLE001 — never sink the chat
            _logger().warning("chat: sector-pulse fetch failed: %s", e)

        # Portfolio snapshot
        portfolio = _read_json(BASE_DIR / "data" / "portfolio_pl.json") or {}

        # Compact portfolio summary for the prompt
        portfolio_lines: list[str] = []
        try:
            s = (portfolio.get("summary") or {}) if isinstance(portfolio, dict) else {}
            if s:
                portfolio_lines.append(
                    f"Total value: ${s.get('grand_value', s.get('total_value', 'n/a'))} "
                    f"P&L: ${s.get('grand_pnl', s.get('total_pnl', 'n/a'))} "
                    f"({s.get('grand_pnl_pct', s.get('total_pnl_pct', 'n/a'))}%)"
                )
            for p in (portfolio.get("positions") or [])[:20]:
                try:
                    if float(p.get("qty") or 0) == 0:
                        continue
                except Exception:
                    pass
                portfolio_lines.append(
                    f"  {p.get('ticker','?')}: qty={p.get('qty')} avg=${p.get('avg_cost')} "
                    f"px=${p.get('price')} pnl=${p.get('pnl')} ({p.get('pnl_pct')}%)"
                )
            for o in (portfolio.get("options") or [])[:10]:
                portfolio_lines.append(
                    f"  OPT {o.get('symbol', o.get('ticker','?'))}: qty={o.get('qty')} pnl=${o.get('pnl')}"
                )
        except Exception:
            pass
        portfolio_block = "\n".join(portfolio_lines) if portfolio_lines else "(no portfolio snapshot available)"

        # Articles block
        if articles_ctx:
            art_lines = []
            for a in articles_ctx:
                art_lines.append(
                    f"{a['ai_score']:.2f} | {a['source']} | {a['title']}\n    {a['summary']}"
                )
            articles_block = "\n".join(art_lines)
        else:
            articles_block = "(no recent articles in the last 6 hours)"

        # Wider 48h thesis-context tier (deduped vs the 6h breaking set).
        thesis_block = ""
        if thesis_ctx:
            thesis_block = "\n".join(
                f"{a['ai_score']:.2f} | {a['source']} | {a['title']}\n    {a['summary']}"
                for a in thesis_ctx
            )

        # Live paper-trader state — fetch from :8090/api/state. Adds positions,
        # recent trades, recent decisions so the chat can answer "what did the
        # paper trader do today" / "why is SOXL the position".
        paper_trader_block = "(paper trader unreachable)"
        try:
            import urllib.request as _urllib
            with _urllib.urlopen("http://127.0.0.1:8090/api/state", timeout=3) as resp:
                pt = json.loads(resp.read().decode("utf-8"))
            pt_pf = pt.get("portfolio") or {}
            pt_total = pt_pf.get("total_value")
            pt_cash = pt_pf.get("cash")
            pt_pl = (pt_total - 1000.0) if isinstance(pt_total, (int, float)) else None
            pt_pl_pct = (pt_pl / 1000.0 * 100.0) if pt_pl is not None else None

            lines = []
            if pt_total is not None:
                lines.append(
                    f"Total: ${pt_total:.2f}  Cash: ${pt_cash:.2f}  "
                    f"P/L vs $1000 start: {('+' if (pt_pl or 0)>=0 else '')}${pt_pl:.2f} "
                    f"({pt_pl_pct:+.2f}%)"
                )
            sp = pt.get("sp500")
            if sp:
                lines.append(f"S&P 500: {sp:.2f}")

            # Marked snapshot (real pl_pct + stale_mark) via the pure
            # helper; falls back to the raw array if the store is degraded.
            lines += _paper_trader_position_lines(pt)

            pt_trades = pt.get("trades") or []
            if pt_trades:
                lines.append(f"Last {min(5, len(pt_trades))} trades:")
                for t in pt_trades[:5]:
                    extra = ""
                    if t.get("option_type"):
                        extra = f" {t.get('strike')}{t.get('option_type','')[0].upper()} {t.get('expiry')}"
                    lines.append(
                        f"  [{(t.get('timestamp') or '')[5:16].replace('T',' ')}] "
                        f"{t.get('action')} {t.get('qty')} {t.get('ticker')}{extra} @ ${(t.get('price') or 0):.2f}"
                    )

            pt_decisions = pt.get("decisions") or []
            if pt_decisions:
                lines.append(f"Last {min(3, len(pt_decisions))} decisions:")
                for d in pt_decisions[:3]:
                    reasoning = ""
                    try:
                        j = json.loads(d.get("reasoning") or "{}")
                        reasoning = (j.get("decision") or {}).get("reasoning") or j.get("detail") or ""
                    except Exception:
                        reasoning = d.get("reasoning") or ""
                    lines.append(
                        f"  [{(d.get('timestamp') or '')[5:16].replace('T',' ')}] "
                        f"{d.get('action_taken','')}: {reasoning[:160]}"
                    )

            # Equity curve trend (last ~6 points spaced over recent history)
            eq = pt.get("equity") or []
            if len(eq) >= 6:
                step = max(1, len(eq) // 6)
                sample = eq[::step][-6:]
                trend = " → ".join(f"${(p.get('total_value') or 0):.2f}" for p in sample)
                lines.append(f"Equity trend (recent): {trend}")

            paper_trader_block = "\n".join(lines) if lines else "(no paper-trader state)"
        except Exception as e:
            _logger().warning("chat: paper trader state fetch failed: %s", e)

        # Pull options Greeks (live trader's portfolio-level delta/gamma/theta/vega).
        # Useful when the user asks "am I overexposed?" or "what happens if NVDA drops 5%?"
        greeks_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen("http://127.0.0.1:8090/api/greeks", timeout=3) as resp:
                gk = json.loads(resp.read().decode("utf-8"))
            if not gk.get("error"):
                t = gk.get("totals") or {}
                rows = gk.get("positions") or []
                opt_rows = [r for r in rows if r.get("type") in ("call", "put")]
                if opt_rows:
                    lines = [
                        f"Net delta: {t.get('delta', 0):+.2f} (gross $ {t.get('gross_notional', 0):.0f})",
                        f"Net gamma: {t.get('gamma', 0):+.5f}",
                        f"Theta / day: ${t.get('theta', 0):+.2f}",
                        f"Vega / 1% IV: ${t.get('vega', 0):+.2f}",
                    ]
                    for o in opt_rows[:8]:
                        dte = o.get("days_to_expiry")
                        lines.append(
                            f"  {o.get('ticker')} {str(o.get('type','')).upper()} "
                            f"{o.get('strike')}/{o.get('expiry')} "
                            f"({dte}d): Δ {o.get('delta'):+.2f} Θ {o.get('theta'):+.2f} "
                            f"IV {(o.get('iv') or 0)*100:.0f}%"
                        )
                    greeks_block = "\n".join(lines)
        except Exception as e:
            _logger().warning("chat: greeks fetch failed: %s", e)

        # Pull DRAM/semis sector heatmap so the chat can answer "which semis are
        # leading today" without the user having to look at the dashboard.
        heatmap_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen("http://127.0.0.1:8090/api/sector-heatmap", timeout=4) as resp:
                hm = json.loads(resp.read().decode("utf-8"))
            if not hm.get("error"):
                ref = hm.get("reference_mom_5d")
                bits = [f"Reference: {hm.get('reference', 'SOXX')} 5d {ref:+.2f}%" if ref is not None else "Reference: —"]
                for b in (hm.get("buckets") or []):
                    avg = b.get("avg_mom_5d")
                    if avg is None:
                        continue
                    # Sort tickers within each bucket by mom_5d desc and keep top 3.
                    ticks = sorted(
                        [t for t in (b.get("tickers") or []) if t.get("mom_5d") is not None],
                        key=lambda t: -t["mom_5d"],
                    )
                    head = ", ".join(
                        f"{t['ticker']} {t['mom_5d']:+.1f}%" for t in ticks[:3]
                    )
                    urg = sum((t.get("urgent") or 0) for t in (b.get("tickers") or []))
                    urg_str = f" [{urg} urgent news]" if urg else ""
                    bits.append(f"  {b.get('name','?'):18s} avg {avg:+.2f}%{urg_str}  · top: {head}")
                heatmap_block = "\n".join(bits)
        except Exception as e:
            _logger().warning("chat: heatmap fetch failed: %s", e)

        # Pull portfolio analytics (sector exposure, drawdown, win rate, daily P/L)
        analytics_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen("http://127.0.0.1:8090/api/analytics", timeout=3) as resp:
                an = json.loads(resp.read().decode("utf-8"))
            if not an.get("error"):
                bits = []
                if an.get("daily_pl_usd") is not None:
                    bits.append(
                        f"Today P/L: ${an['daily_pl_usd']:+.2f} ({an.get('daily_pl_pct', 0):+.2f}%)"
                    )
                if an.get("max_drawdown_pct"):
                    bits.append(
                        f"Max DD: -${an['max_drawdown_usd']:.2f} ({an['max_drawdown_pct']:.2f}%)"
                    )
                if an.get("sharpe_annualized") is not None:
                    bits.append(f"Sharpe (ann.): {an['sharpe_annualized']}")
                if an.get("sortino_annualized") is not None:
                    bits.append(f"Sortino (ann.): {an['sortino_annualized']}")
                if an.get("calmar_ratio") is not None:
                    bits.append(f"Calmar: {an['calmar_ratio']}")
                if an.get("win_rate_pct") is not None:
                    bits.append(
                        f"Win rate: {an['win_rate_pct']}% over {an.get('n_round_trips', 0)} round-trips"
                    )
                if an.get("profit_factor") is not None:
                    bits.append(f"Profit factor: {an['profit_factor']}")
                if an.get("avg_holding_days") is not None:
                    bits.append(f"Avg holding period: {an['avg_holding_days']}d")
                if an.get("sp500_beta") is not None:
                    corr = an.get("sp500_correlation")
                    bits.append(
                        f"S&P 500 beta: {an['sp500_beta']}"
                        + (f" (corr {corr})" if corr is not None else "")
                    )
                if an.get("realized_pl_usd"):
                    bits.append(f"Realized P/L: ${an['realized_pl_usd']:+.2f}")
                sectors = an.get("sector_exposure_pct") or {}
                if sectors:
                    top_secs = sorted(sectors.items(), key=lambda kv: -kv[1])[:5]
                    bits.append("Sector exposure: " +
                                ", ".join(f"{s}={p:.1f}%" for s, p in top_secs) +
                                f", cash={an.get('cash_pct', 0):.1f}%")
                # Left-tail view (VaR/CVaR/skew/Ulcer) — the trader's
                # /api/analytics now carries a `tail_risk` block; surface
                # it so the analyst can answer "what's a realistic bad
                # day" not just "what was the worst drawdown".
                bits += _tail_risk_chat_lines(an)
                if bits:
                    analytics_block = "\n".join(bits)
        except Exception as e:
            _logger().warning("chat: analytics fetch failed: %s", e)

        # Behavioural diagnosis — the paper trader's OWN self-review
        # verdicts (scorecard / capital-paralysis / churn), not just the
        # raw stats /api/analytics already gave us. Lets the chat answer
        # "why is my bot losing money / what should it do?" with the
        # diagnosis the bot itself produced. Each is its own guarded
        # 3s-timeout read (one upstream fault degrades that input to
        # None, never sinks the block or the chat). Composed verbatim by
        # the pure _behavioural_chat_lines helper (unit-tested). Only
        # appears once :8090 is restarted onto these endpoints — a
        # stale/absent trader silently omits the block (sibling contract).
        behavioural_block = ""
        try:
            import urllib.request as _urllib

            def _bfetch(path: str):
                try:
                    with _urllib.urlopen(
                            f"http://127.0.0.1:8090{path}", timeout=3) as resp:
                        return json.loads(resp.read().decode("utf-8"))
                except Exception as e:
                    _logger().warning("chat: %s fetch failed: %s", path, e)
                    return None

            _sc = _bfetch("/api/scorecard")
            _cp = _bfetch("/api/capital-paralysis")
            _ch = _bfetch("/api/churn")
            behavioural_block = "\n".join(
                _behavioural_chat_lines(_sc, _cp, _ch)
            )
        except Exception as e:
            _logger().warning("chat: behavioural fetch failed: %s", e)

        # ML-gate honesty — does the 17-feature DecisionScorer that modulates
        # the bot's live position sizing (invariant #5) actually beat a
        # one-line rule OUT OF SAMPLE, or is its nudge noise? Previously this
        # truth lived only in `python3 -m paper_trader.ml.baseline_compare`
        # (a CLI no operator runs) — the analytics endpoints the chat already
        # pulls report the IN-SAMPLE story that flatters the net. Surfacing
        # it here lets the analyst answer "is the bot's ML edge real?"
        # honestly. Composed verbatim by the pure _baseline_compare_chat_lines
        # helper (unit-tested; SSOT — no re-derived verdict). Guarded 3s read
        # like every sibling; only appears once :8090 is restarted onto the
        # endpoint — a stale/absent trader silently omits the block.
        baseline_compare_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/baseline-compare",
                    timeout=3) as resp:
                _bc = json.loads(resp.read().decode("utf-8"))
            baseline_compare_block = "\n".join(
                _baseline_compare_chat_lines(_bc))
        except Exception as e:
            _logger().warning("chat: baseline-compare fetch failed: %s", e)

        # Correlation / factor concentration — does the held book move as
        # ONE bet however many tickers are on it? /api/risk (carried in the
        # analytics block above) reports NAME-level concentration; this is
        # the FACTOR-level companion (pairwise return correlation among the
        # held stocks, the weight-Herfindahl effective-position count, and
        # the correlation-adjusted effective number of *independent* bets).
        # A 2-name 59/41 book that /api/risk grades CONCENTRATED-by-name can
        # still hide single-factor risk: both names high-β semis → the
        # diversification claim is illusory. Composed verbatim by the pure
        # _correlation_chat_lines helper (unit-tested; SSOT — no re-derived
        # verdict). Guarded 3s sub-fetch like every sibling; a stale/absent
        # trader silently omits the block.
        correlation_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/correlation",
                    timeout=3) as resp:
                _corr = json.loads(resp.read().decode("utf-8"))
            correlation_block = "\n".join(
                _correlation_chat_lines(_corr))
        except Exception as e:
            _logger().warning("chat: correlation fetch failed: %s", e)

        # Thesis drift — every open position re-tested against the verbatim
        # reason it was opened for, graded INTACT/WEAKENING/BROKEN. The chat
        # already carries the open book (portfolio_block) and the closed-trade
        # behavioural mirror, but neither answers "is the thing the bot bought
        # this for still true?" — the single discipline question that drives
        # most desk trims. Surfacing the WEAKENING/BROKEN cards here lets the
        # analyst answer "should the bot have already sold X?" honestly
        # instead of re-deriving from raw signals. Composed verbatim by the
        # pure _thesis_drift_chat_lines helper (unit-tested; SSOT — no
        # re-derived verdict). Guarded 3s sub-fetch like every sibling; an
        # all-INTACT book collapses to silence (the silence precedent).
        thesis_drift_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/thesis-drift",
                    timeout=3) as resp:
                _td = json.loads(resp.read().decode("utf-8"))
            thesis_drift_block = "\n".join(
                _thesis_drift_chat_lines(_td))
        except Exception as e:
            _logger().warning("chat: thesis-drift fetch failed: %s", e)

        # Earnings radar — scheduled gap risk on the paper trader's holdings.
        # Lets the chat warn "you hold NVDA and it prints in 4 days".
        earnings_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/earnings-risk", timeout=3) as resp:
                er = json.loads(resp.read().decode("utf-8"))
            if not er.get("error") and er.get("events"):
                e_lines = []
                for ev in er["events"][:10]:
                    days = ev.get("days_away")
                    days_s = f"{days:.1f}d" if isinstance(days, (int, float)) else "?"
                    flag = " [HELD]" if ev.get("held") else " (watchlist)"
                    e_lines.append(
                        f"  {ev.get('ticker')}: reports in {days_s}{flag}"
                        + (f", ${ev.get('exposure_usd',0):.0f} exposure" if ev.get("held") else "")
                    )
                if e_lines:
                    hdr = (f"{er.get('n_held_reporting',0)} holding(s) reporting soon, "
                           f"{er.get('n_imminent',0)} within 3 days:")
                    earnings_block = hdr + "\n" + "\n".join(e_lines)
        except Exception as e:
            _logger().warning("chat: earnings-risk fetch failed: %s", e)

        # Macro calendar — the forward FOMC rate-decision awareness already
        # fed into the live trader's OWN decision prompt (macro_calendar.py).
        # The chat carried rich BACKWARD analytics + earnings, but was blind
        # to the single biggest MARKET-WIDE event — a rate decision that
        # moves the whole book (leveraged ETFs most violently, and this
        # watchlist is full of them). Surfacing it lets the analyst answer
        # "is the Fed about to move everything?" honestly. Composed verbatim
        # by the pure _macro_calendar_chat_lines helper (unit-tested; SSOT —
        # the builder's own `summary` is the headline, no re-derived
        # verdict). Guarded 3s read like every sibling; a no-FOMC / error /
        # not-loaded payload (all events:[]) silently omits the block; only
        # appears once :8090 is restarted onto the endpoint.
        macro_calendar_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/macro-calendar",
                    timeout=3) as resp:
                _mc = json.loads(resp.read().decode("utf-8"))
            macro_calendar_block = "\n".join(
                _macro_calendar_chat_lines(_mc))
        except Exception as e:
            _logger().warning("chat: macro-calendar fetch failed: %s", e)

        # Earnings shock — the pre-earnings dollarized 1σ shock per held
        # imminent print (paper-trader /api/earnings-shock, the forward
        # $-at-risk complement to /api/event-calendar). The chat already
        # carries event-calendar timing and macro-calendar FOMC risk, but
        # nothing translated "NVDA earnings in 0.9d" into "if NVDA gaps the
        # typical 1σ on its print, the book moves $X (Y% of equity)" — the
        # actual pre-print question the analyst gets asked. Composed
        # verbatim by the pure _earnings_shock_chat_lines helper (unit-
        # tested; SSOT — the builder's own `headline` is the headline, no
        # re-derived verdict). Guarded 4s read (yfinance is the slowest
        # per-name shape upstream; the SWR cache makes the *cached* response
        # fast); NO_DATA / NO_EVENTS / INSUFFICIENT_HISTORY rows silently
        # omit or report "σ withheld" honestly (the
        # _macro_calendar_chat_lines / _baseline_compare_chat_lines
        # precedents — never chat filler, never fabricated numerics).
        # Only appears once :8090 is restarted onto the endpoint.
        earnings_shock_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/earnings-shock",
                    timeout=4) as resp:
                _es = json.loads(resp.read().decode("utf-8"))
            earnings_shock_block = "\n".join(
                _earnings_shock_chat_lines(_es))
        except Exception as e:
            _logger().warning("chat: earnings-shock fetch failed: %s", e)

        # Event-readiness — will the live trader actually be able to react
        # before the next earnings print? Every block above describes the
        # event (timing, σ shock, sector exposure) and implicitly assumes
        # the bot will *make a decision* before the print. The live
        # failure mode (PARALYSIS / NO_DECISION storms / Claude empty
        # streaks) silently breaks that assumption — and a chat-side
        # analyst answering "is your book at risk?" without questioning
        # whether the bot can act is dangerously incomplete. Composed
        # verbatim by the pure _event_readiness_chat_lines helper (unit-
        # tested; SSOT — the builder's own `summary` is the headline, no
        # re-derived verdict). Guarded 3s read; READY / NO_EVENTS /
        # NO_DECISIONS payloads silently omit the block (the
        # _macro_calendar_chat_lines silence precedent — never chat filler
        # when the pipeline is healthy). Only appears once :8090 is
        # restarted onto /api/event-readiness.
        event_readiness_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/event-readiness",
                    timeout=3) as resp:
                _er = json.loads(resp.read().decode("utf-8"))
            event_readiness_block = "\n".join(
                _event_readiness_chat_lines(_er))
        except Exception as e:
            _logger().warning("chat: event-readiness fetch failed: %s", e)

        # Consecutive-HOLD paralysis (HOLD_LOCK) — the other "alive but
        # nothing happens" failure mode that event-readiness covers only
        # the NO_DECISION storm half of. A stacked HOLD-only block where
        # Opus decides every cycle but never moves the book reads as
        # HEALTHY on /api/runner-heartbeat and /api/decision-health, yet
        # the operator's "should I be doing something?" question is
        # exactly the one this pathology breaks. Composed verbatim by the
        # pure _decision_paralysis_chat_lines helper (unit-tested; SSOT —
        # the builder's own `headline` is the chat headline, no
        # re-derived verdict). Guarded 3s read; ACTIVE / NO_DATA payloads
        # silently omit the block (the _event_readiness_chat_lines
        # silence precedent — never chat filler when the loop is
        # healthy). Only appears once :8090 is restarted onto
        # /api/decision-paralysis.
        decision_paralysis_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/decision-paralysis",
                    timeout=3) as resp:
                _dp = json.loads(resp.read().decode("utf-8"))
            decision_paralysis_block = "\n".join(
                _decision_paralysis_chat_lines(_dp))
        except Exception as e:
            _logger().warning(
                "chat: decision-paralysis fetch failed: %s", e)

        # "What materially changed since you last looked" — the one temporal-
        # change view. Every sub-fetch above is a current-state snapshot; this
        # lets the chat answer "what happened while I was away / since I last
        # asked?" without the user scanning the dashboard. Restates
        # /api/session-delta's own ranked headline + top events (a restatement
        # of event fields, no re-derivation). ACTIVE-only — a QUIET/NO_DATA
        # window is silence, matching the unified :8888 chat's
        # _fetch_session_delta so the two conversational surfaces stay
        # consistent. Network guarded like every sibling (never raises into
        # chat); only appears once :8090 is restarted onto the endpoint.
        session_delta_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/session-delta?minutes=360",
                    timeout=4) as resp:
                sd = json.loads(resp.read().decode("utf-8"))
            if (not sd.get("error") and sd.get("state") == "ACTIVE"
                    and sd.get("headline")):
                sd_lines = [sd["headline"]]
                for ev in (sd.get("events") or [])[:6]:
                    s = ev.get("summary")
                    if s:
                        sd_lines.append(f"  • {s}")
                session_delta_block = "\n".join(sd_lines)
        except Exception as e:
            _logger().warning("chat: session-delta fetch failed: %s", e)

        # The system's OWN actionable synthesis — the prioritised
        # next-session plan (/api/game-plan: fuses the co-pilot verb,
        # disposition trap, concentration and earnings into one ranked
        # list) and the open-book disposition-trap verdict
        # (/api/hold-discipline). Every other block above is descriptive
        # state; this is the one "what should I actually do" surface, and
        # the chat is where the operator asks exactly that. Composed
        # verbatim by the pure _game_plan_chat_lines / _hold_discipline_
        # chat_lines helpers (invariant #10 — no re-derived verdicts).
        # Each is its own guarded read (one upstream fault degrades that
        # input to silence, never sinks the block or the chat). Only
        # appears once :8090 is restarted onto these endpoints — a
        # stale/absent trader silently omits the block (sibling contract).
        game_plan_block = ""
        hold_discipline_block = ""
        try:
            import urllib.request as _urllib

            def _afetch(path: str):
                try:
                    with _urllib.urlopen(
                            f"http://127.0.0.1:8090{path}", timeout=4) as resp:
                        return json.loads(resp.read().decode("utf-8"))
                except Exception as e:
                    _logger().warning("chat: %s fetch failed: %s", path, e)
                    return None

            game_plan_block = "\n".join(
                _game_plan_chat_lines(_afetch("/api/game-plan")))
            hold_discipline_block = "\n".join(
                _hold_discipline_chat_lines(_afetch("/api/hold-discipline")))
        except Exception as e:
            _logger().warning("chat: action-plan fetch failed: %s", e)

        # Per-held-ticker 24h ai_score TREND (RISING / FADING) — the temporal
        # complement to /api/portfolio-signals' current-state snapshot. The
        # chat already carries "what's the wire saying about MU right now";
        # this answers "is that signal RAMPING UP, going QUIET, or stable
        # since you last asked?" — exactly the question an analyst gets when
        # the portfolio block shows a held name and the operator wants to
        # know whether to push or fade. Pure helpers (`build_position_
        # conviction_decay` + `_position_conviction_decay_chat_lines`); held
        # tickers come from the same /api/state already fetched above so we
        # don't double the upstream load. STABLE & INSUFFICIENT_DATA collapse
        # to silence per-ticker; an all-silent block omits the section.
        conviction_decay_block = ""
        try:
            # `pt` was set by the paper-trader-state fetch above; extract
            # held tickers from its positions. Falls back to silence if
            # paper-trader was unreachable (pt would be undefined here, so
            # we guard via locals()).
            _pt = locals().get("pt")
            held_tickers: list[str] = []
            if isinstance(_pt, dict):
                for p in (_pt.get("positions") or []):
                    if not isinstance(p, dict):
                        continue
                    if p.get("type") and p.get("type") != "stock":
                        continue
                    qty = p.get("qty") or 0
                    if not (isinstance(qty, (int, float)) and qty > 0):
                        continue
                    tk = (p.get("ticker") or "").strip().upper()
                    if tk:
                        held_tickers.append(tk)
            if held_tickers:
                cutoff = (
                    datetime.now(timezone.utc) - timedelta(hours=24)
                ).isoformat()
                try:
                    rows = _ro_query(
                        "SELECT title, ai_score, first_seen FROM articles "
                        f"WHERE first_seen >= ? AND {_LIVE_ONLY_SQL} "
                        "ORDER BY first_seen DESC LIMIT 4000",
                        (cutoff,),
                    )
                except sqlite3.Error:
                    rows = []
                arts = [
                    {"title": r[0] or "", "ai_score": float(r[1] or 0),
                     "first_seen": r[2]}
                    for r in rows
                ]
                rep = build_position_conviction_decay(held_tickers, arts)
                conviction_decay_block = "\n".join(
                    _position_conviction_decay_chat_lines(rep))
        except Exception as e:
            _logger().warning(
                "chat: position-conviction-decay enrichment failed: %s", e)

        # Alert-confidence trend — which urgent stories are gaining new
        # corroborating sources (RISING) vs which started loud and are
        # losing the wire (FADING). The existing news_corroboration
        # endpoint reports CURRENT-state source counts; this is the
        # temporal-delta companion that tells the analyst "Nvidia earnings
        # was 3-source 18h ago → 8-source now (RISING)" vs "rate cut
        # rumour was 5-source then → 1-source now (FADING)". RISING
        # corroboration is the highest-trust BUY signal; STABLE +
        # SINGLE_SOURCE clusters silently drop (chat budget, and a
        # single-source story is usually PR/spam not corroboration).
        alert_trend_block = ""
        try:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(hours=24)
            ).isoformat()
            try:
                rows = _ro_query(
                    "SELECT title, ai_score, first_seen, source FROM articles "
                    f"WHERE first_seen >= ? AND urgency >= 1 "
                    f"AND {_LIVE_ONLY_SQL} "
                    "ORDER BY first_seen DESC LIMIT 4000",
                    (cutoff,),
                )
            except sqlite3.Error:
                rows = []
            arts = [
                {"title": r[0] or "", "ai_score": float(r[1] or 0),
                 "first_seen": r[2], "source": r[3] or ""}
                for r in rows
            ]
            rep = build_alert_confidence_trend(arts)
            alert_trend_block = "\n".join(
                _alert_confidence_trend_chat_lines(rep))
        except Exception as e:
            _logger().warning(
                "chat: alert-confidence-trend enrichment failed: %s", e)

        now_iso = datetime.now(timezone.utc).isoformat()
        system_prompt = (
            "You are a market intelligence analyst with access to a real-time news feed, "
            "the user's real portfolio, and a separate live paper trading bot (Claude Opus 4.7) "
            "running on a $1000 simulated portfolio.\n"
            f"Current date: {now_iso}\n\n"
            "TOP NEWS SIGNALS (last 6h, ranked by ML score recency-decayed by the time-sensitivity head — a fresher item can outrank a higher-scored stale one):\n"
            f"{articles_block}\n\n"
            + (f"THESIS CONTEXT (last 48h, same recency-decayed ML ranking, deduped vs the 6h set above):\n{thesis_block}\n\n" if thesis_block else "")
            + (f"NEWS COVERAGE GAP — these curated intel channels could NOT be collected this window. Absence of news here is BLINDNESS, not calm: do NOT infer 'nothing happened' on these; if the user asks about one, say it is dark and why:\n{coverage_gap_block}\n\n" if coverage_gap_block else "")
            + "USER'S REAL PORTFOLIO SNAPSHOT:\n"
            f"{portfolio_block}\n\n"
            "PAPER TRADER LIVE STATE (separate $1000 sim run by Opus 4.7 every 30 min):\n"
            f"{paper_trader_block}\n\n"
            + (f"PAPER TRADER — WHAT MATERIALLY CHANGED SINCE YOU LAST LOOKED (ranked, last 6h):\n{session_delta_block}\n\n" if session_delta_block else "")
            + (f"PAPER TRADER ANALYTICS:\n{analytics_block}\n\n" if analytics_block else "")
            + (f"PAPER TRADER — BEHAVIOURAL DIAGNOSIS (the bot's own self-review verdicts):\n{behavioural_block}\n\n" if behavioural_block else "")
            + (f"PAPER TRADER — ML GATE HONESTY (does the DecisionScorer that modulates the bot's live position sizing beat a one-line rule OUT OF SAMPLE? the analytics above report the flattering in-sample story; this is the generalization-relevant verdict):\n{baseline_compare_block}\n\n" if baseline_compare_block else "")
            + (f"PAPER TRADER — FACTOR CONCENTRATION (the held book's pairwise return correlation + effective-independent-bets count; complements /api/risk's NAME-level view — a 59/41 book that is concentrated by weight may STILL be a single FACTOR bet if both names move as one):\n{correlation_block}\n\n" if correlation_block else "")
            + (f"PAPER TRADER — OPEN-POSITION THESIS DRIFT (every holding re-tested against the verbatim reason it was opened for, graded INTACT/WEAKENING/BROKEN by P/L since entry + live quant/momentum; an INTACT book collapses to silence so this block ONLY fires when at least one held position is materially off its entry thesis — drift_reasons carry verbatim from the trader endpoint, never re-derived):\n{thesis_drift_block}\n\n" if thesis_drift_block else "")
            + (f"PAPER TRADER — PRIORITISED ACTION PLAN (the bot's own next-session game plan):\n{game_plan_block}\n\n" if game_plan_block else "")
            + (f"PAPER TRADER — HOLD-DISCIPLINE ALERT (a losing position overstayed past the desk's own median losing-cut):\n{hold_discipline_block}\n\n" if hold_discipline_block else "")
            + (f"HELD-TICKER 24h NEWS-CONVICTION TREND (per held name, ai_score bucketed into 4 × 6h slices; only RISING / FADING are surfaced — STABLE/quiet collapses to silence. Complements /api/portfolio-signals' current-state snapshot with the temporal direction: is the wire focusing MORE or LESS on this position over the last day?):\n{conviction_decay_block}\n\n" if conviction_decay_block else "")
            + (f"ALERT-CONFIDENCE TREND (urgent-story clusters whose unique-source count is GROWING or FADING vs the prior 6-24h half; corroboration that's still expanding is the highest-trust read — a single-source story is usually PR not news, a fading story has been priced):\n{alert_trend_block}\n\n" if alert_trend_block else "")
            + (f"PAPER TRADER OPTIONS GREEKS (Black-Scholes, live IV):\n{greeks_block}\n\n" if greeks_block else "")
            + (f"NEWS SECTOR PULSE (native — where the wire is concentrated right now, recency-weighted, last 24h; independent of the price heatmap below so it survives a stale/down paper-trader):\n{sector_pulse_block}\n\n" if sector_pulse_block else "")
            + (f"DRAM / SEMIS 5d MOMENTUM HEATMAP (paper-trader price momentum):\n{heatmap_block}\n\n" if heatmap_block else "")
            + (f"EARNINGS RADAR (scheduled gap risk):\n{earnings_block}\n\n" if earnings_block else "")
            + (f"PAPER TRADER — PRE-EARNINGS DOLLARIZED 1σ SHOCK (per HELD imminent print: 'if NVDA gaps the typical 1σ on its release, the book moves $X / Y% of equity' — the forward $-at-risk view that complements the EARNINGS RADAR's timing-only listing):\n{earnings_shock_block}\n\n" if earnings_shock_block else "")
            + (f"MACRO CALENDAR — FOMC RATE DECISION (the single biggest MARKET-WIDE event; it moves the whole book at once, leveraged ETFs most violently — surfaced only when one is actually within the 14d horizon):\n{macro_calendar_block}\n\n" if macro_calendar_block else "")
            + (f"PAPER TRADER — EVENT READINESS (will the live trader actually be able to react before the next earnings print? Joins earnings-risk + decision-velocity + Claude-empty rate + the *current* NO_DECISION streak into a single per-event verdict — surfaced ONLY when BLIND / DEGRADED / IMMINENT_OVERDUE, never as filler when the pipeline is healthy. Recommended_action carries verbatim from the trader endpoint — restate, never re-derive):\n{event_readiness_block}\n\n" if event_readiness_block else "")
            + (f"PAPER TRADER — DECISION PARALYSIS (consecutive HOLD-only / NO_DECISION runs on the live decision loop — a stacked HOLD_LOCK block reads HEALTHY on runner-heartbeat and decision-health 24h aggregate but means Opus decided every cycle and never moved the book. Surfaced ONLY when HOLD_LOCK / IDLE_STORM / PASSIVE_LOOP, never filler when ACTIVE. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{decision_paralysis_block}\n\n" if decision_paralysis_block else "")
            + "Answer questions about current market conditions, global events, specific "
            "stocks, the user's real portfolio, or the paper trader's positions/decisions. "
            "Be concise and data-driven. Cite specific articles when relevant. When the user "
            "asks 'how am I doing', show real-portfolio first then paper-trader as separate "
            "lines so they aren't confused. The user's thesis-focus is DRAM/memory (MU, WDC, "
            "STX) plus semis equipment (LRCX, AMAT, KLAC, ASML) and the HBM-ramp design "
            "winners (NVDA, AMD, AVGO); weight your reads of the heatmap and news through "
            "that lens unless the user asks otherwise."
        )

        # Build messages
        msgs: list[dict] = []
        for h in history[-20:]:
            role = h.get("role")
            content = (h.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": user_msg})

        # Build a single prompt: system block + conversation history + final user turn.
        # Claude CLI (core.claude_cli) handles auth via its own login — no API key needed.
        convo_parts = [system_prompt, "\n\n--- Conversation ---"]
        for m in msgs:
            convo_parts.append(f"{m['role'].upper()}: {m['content']}")
        convo_parts.append("ASSISTANT:")
        prompt = "\n\n".join(convo_parts)

        response_text = _claude_cli_call(prompt, model="claude-opus-4-7", timeout=120) or ""

        if not response_text:
            return jsonify({"error": "claude CLI returned no response"}), 502

        return jsonify({
            "response": response_text,
            "sources": [a["title"] for a in articles_ctx],
        })

    return app


def run_server(store, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Blocking entry point used by the daemon worker thread."""
    app = create_app(store)
    # werkzeug's dev server is fine for read-only public dashboard at this scale.
    app.run(host=host, port=port, threaded=True, use_reloader=False, debug=False)


# ── HTML payload (Bootstrap dark theme via CDN, no npm) ─────────────────────
_DASHBOARD_HTML = """<!doctype html>
<html lang="en" data-bs-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Digital Intern — Dashboard</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%230d1117'/%3E%3Cpolyline points='3,24 9,16 14,20 20,10 29,14' fill='none' stroke='%2300b4d8' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='20' cy='10' r='2.5' fill='%23e94560'/%3E%3Cline x1='3' y1='27' x2='29' y2='27' stroke='%2330363d' stroke-width='1'/%3E%3C/svg%3E">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Outfit:wght@400;500;600;700&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap');
    :root {
      --bg: #0c0d0f;
      --bg-panel: #111316;
      --bg-elevated: #17191d;
      --bg-hover: #1c1f24;
      --bg-input: #0e1012;
      --border: rgba(255,255,255,0.07);
      --border-strong: rgba(255,255,255,0.13);
      --text: #dde1e7;
      --text-secondary: #8b929d;
      --text-muted: #50565f;
      --amber: #f0b429;
      --amber-dim: rgba(240,180,41,0.12);
      --cyan: #0acdff;
      --cyan-dim: rgba(10,205,255,0.12);
      --green: #00c896;
      --green-dim: rgba(0,200,150,0.12);
      --red: #ff4455;
      --red-dim: rgba(255,68,85,0.12);
      --blue: #4d9eff;
      --blue-dim: rgba(77,158,255,0.12);
      --yellow: #fbbf24;
      --yellow-dim: rgba(251,191,36,0.12);
      --pink: #f472b6;
      --font-sans: 'Outfit', system-ui, sans-serif;
      --font-mono: 'DM Mono', 'JetBrains Mono', monospace;
      --font-display: 'Syne', system-ui, sans-serif;
      --radius: 8px;
      --radius-sm: 5px;
    }
    * { box-sizing: border-box; }
    html { overflow-x: hidden; max-width: 100%; }
    body { overflow-x: hidden; }
    body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: var(--font-sans); font-size: 15px; line-height: 1.5; }
    .brand, h1, h2, h3 { font-family: var(--font-display); }
    .topbar {
      background: var(--bg-panel);
      border-bottom: 1px solid var(--border);
      padding: 0 20px;
      height: 48px;
      display: flex;
      align-items: center;
      gap: 2px;
      position: sticky;
      top: 0;
      z-index: 100;
      margin: 0;
      overflow: hidden;
      max-width: 100%;
    }
    .brand {
      font-weight: 700;
      color: var(--amber);
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-right: 16px;
      flex-shrink: 0;
    }
    .topbar a {
      color: var(--text-secondary);
      text-decoration: none;
      font-size: 13px;
      font-weight: 500;
      padding: 5px 12px;
      border-radius: var(--radius-sm);
      transition: color 0.15s, background 0.15s;
      white-space: nowrap;
    }
    .topbar a:hover { color: var(--text); background: var(--bg-hover); }
    .topbar a.active { color: var(--amber); background: var(--amber-dim); }
    .page-content { padding: 16px 20px; width: 100%; }
    /* Bootstrap overrides — apply our palette to the existing Bootstrap card structure */
    .card { background: var(--bg-panel) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
    .card-header { background: var(--bg-elevated) !important; font-weight: 600; border-bottom: 1px solid var(--border) !important; color: var(--text); }
    .badge-urgent { background: var(--red-dim); color: var(--red); }
    .badge-score { background: var(--blue-dim); color: var(--blue); }
    .pl-pos { color: var(--green); font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
    .pl-neg { color: var(--red); font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
    a { color: var(--cyan); }
    a:hover { color: var(--blue); }
    .scroll-pane { max-height:520px; overflow-y:auto; }
    .ticker { font-weight:600; color: var(--text); }
    .small-muted { color: var(--text-secondary); font-size: 0.85em; }
    table { color: var(--text); }
    .table-borderless td, .table-borderless th { color: var(--text); }
    /* Markdown in floating chat widget */
    .md-body p { margin: 0 0 0.5em; }
    .md-body p:last-child { margin-bottom: 0; }
    .md-body ul, .md-body ol { margin: 0.3em 0 0.5em 1.2em; padding: 0; }
    .md-body li { margin-bottom: 0.15em; }
    .md-body code { background: var(--bg-input); border: 1px solid var(--border-strong); padding: 1px 4px; border-radius: 3px; font-size: 0.85em; font-family: var(--font-mono); }
    .md-body pre { background: var(--bg-input); border: 1px solid var(--border-strong); padding: 8px 10px; border-radius: var(--radius-sm); overflow-x: auto; margin: 0.3em 0; }
    .md-body pre code { background: none; border: none; padding: 0; }
    .md-body strong { color: var(--text); }
    .md-body h1, .md-body h2, .md-body h3 { margin: 0.5em 0 0.25em; color: var(--text); font-size: 0.95em; }
    .md-body blockquote { border-left: 3px solid var(--border-strong); margin: 0.3em 0; padding-left: 8px; color: var(--text-secondary); }
    .md-body table { border-collapse: collapse; margin: 0.3em 0; width: 100%; font-size: 0.85em; }
    .md-body th, .md-body td { border: 1px solid var(--border-strong); padding: 3px 7px; text-align: left; }
    .md-body th { background: var(--bg-panel); }
    /* === Mobile-first responsive additions ============================== */
    .nav-hamburger {
      display: none; flex-direction: column; justify-content: space-between;
      width: 32px; height: 22px; background: none; border: none; cursor: pointer;
      padding: 0; margin-left: auto;
    }
    .nav-hamburger span {
      display: block; height: 2px; background: var(--text); border-radius: 2px;
      transition: all 0.2s;
    }
    .nav-drawer {
      position: fixed; top: 0; left: -280px; width: 280px; height: 100vh;
      background: var(--bg-panel); border-right: 1px solid #1e2028;
      z-index: 1000; transition: left 0.25s ease; overflow-y: auto; padding: 20px 0;
    }
    .nav-drawer.open { left: 0; }
    .nav-drawer-header {
      font-family: var(--font-display); font-weight: 700; color: var(--amber);
      font-size: 13px; letter-spacing: 0.1em; padding: 0 20px 20px;
      border-bottom: 1px solid #1e2028; margin-bottom: 8px;
    }
    .nav-drawer a {
      display: block; padding: 12px 20px; color: var(--text-secondary);
      text-decoration: none; font-size: 14px; transition: all 0.15s;
    }
    .nav-drawer a:hover, .nav-drawer a.active {
      color: var(--text); background: var(--bg-elevated);
    }
    .nav-overlay {
      display: none; position: fixed; inset: 0;
      background: rgba(0,0,0,0.6); z-index: 999;
    }
    .nav-overlay.open { display: block; }
    .bottom-nav {
      display: none; position: fixed; bottom: 0; left: 0; right: 0; height: 64px;
      background: var(--bg-panel); border-top: 1px solid #1e2028;
      grid-template-columns: repeat(5, 1fr); z-index: 200; align-items: stretch;
    }
    .bottom-tab {
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; gap: 4px; color: var(--text-secondary);
      text-decoration: none; font-size: 10px; min-height: 44px; transition: color 0.15s;
    }
    .bottom-tab svg { width: 20px; height: 20px; }
    .bottom-tab.active, .bottom-tab:hover { color: var(--amber); }
    .table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .table-scroll table { min-width: 500px; }
    @media (max-width: 768px) {
      .topbar-nav { display: none; }
      .nav-hamburger { display: flex; }
      body { font-size: 14px; }
      button, .btn, a.btn, [role="button"] { min-height: 44px; min-width: 44px; }
    }
    @media (max-width: 480px) {
      body { padding-bottom: 72px; }
      .bottom-nav { display: grid; }
      .topbar { padding: 0 16px; }
      .card { min-height: auto !important; padding: 14px 16px; }
      .grid-2, .grid2 { grid-template-columns: 1fr !important; }
      .scroll-pane { max-height: 60vh !important; }
      table { font-size: 12px; }
      th, td { padding: 8px 10px; }
      #dichat-panel { width: calc(100vw - 24px) !important; right: 12px !important;
        height: 70vh !important; bottom: 80px !important; }
    }
  </style>
</head>
<body>
<nav class="topbar">
  <span class="brand">◈ TRADING STACK</span>
  <span class="topbar-nav" style="display:flex;align-items:center;gap:2px;">
    <a href="/">Command Center</a>
    <a href="/intern/" class="active">Digital Intern</a>
    <a href="/trader/">Paper Trader</a>
    <a href="/trader/backtests">Backtests</a>
    <a href="/backtests/compare">Compare</a>
    <a href="/journal">Journal</a>
    <a href="/ops/">Ops View</a>
    <a href="/intern/chat">Chat</a>
    <a href="/system/">System</a>
  </span>
  <button class="nav-hamburger" id="navToggle" aria-label="Menu">
    <span></span><span></span><span></span>
  </button>
</nav>
<div class="nav-drawer" id="navDrawer">
  <div class="nav-drawer-header">◈ TRADING STACK</div>
  <a href="/">Command Center</a>
  <a href="/intern/" class="active">Digital Intern</a>
  <a href="/trader/">Paper Trader</a>
  <a href="/trader/backtests">Backtests</a>
  <a href="/backtests/compare">Compare</a>
  <a href="/journal">Journal</a>
  <a href="/ops/">Ops View</a>
  <a href="/intern/chat">Chat</a>
  <a href="/system/">System</a>
</div>
<div class="nav-overlay" id="navOverlay"></div>
<div class="page-content">
<div class="container-fluid p-3">
  <div class="d-flex justify-content-between align-items-center mb-3">
    <h2 class="mb-0">Digital Intern</h2>
    <div class="small-muted" id="last-updated">loading…</div>
  </div>

  <div class="card mb-3" id="paper-trader-card">
    <div class="card-header d-flex justify-content-between align-items-center">
      <span>Live Paper Trader</span>
      <a href="/trader/" style="font-size:0.85em">View Full Trader →</a>
    </div>
    <div class="card-body p-2">
      <div id="paper-trader-summary" class="small-muted">loading…</div>
    </div>
  </div>

  <!-- News source edge — which of MY collectors actually precede the move? (new 2026-05-16, agent 4) -->
  <div class="card mb-3" id="se-card">
    <div class="card-header d-flex justify-content-between align-items-center">
      <span>News source edge <span class="small-muted">— which collectors' scored headlines actually precede the move (vs SPY)</span></span>
      <span id="se-state" class="badge bg-secondary">—</span>
    </div>
    <div class="card-body p-2">
      <div id="se-headline" class="small-muted mb-2">loading…</div>
      <div class="table-responsive">
        <table class="table table-sm table-dark mb-1" style="font-size:0.82em;">
          <thead><tr><th>collector</th><th>abn% @ref</th><th>hit</th><th>resolved</th><th>verdict</th></tr></thead>
          <tbody id="se-rows"><tr><td colspan="5" class="small-muted">—</td></tr></tbody>
        </table>
      </div>
      <div id="se-meta" class="small-muted">—</div>
    </div>
  </div>

  <div class="row g-3">
    <div class="col-12 col-lg-4">
      <div class="card mb-3">
        <div class="card-header d-flex justify-content-between align-items-center">
          Portfolio P&amp;L
          <button class="btn btn-sm btn-outline-secondary py-0 px-2" style="font-size:11px" onclick="togglePortfolioEdit()">✎ Edit</button>
        </div>
        <div class="card-body p-2">
          <div id="pnl-summary" class="mb-2 small-muted">loading…</div>
          <div class="table-responsive scroll-pane table-scroll">
            <table class="table table-sm table-borderless mb-0">
              <thead><tr>
                <th>TICKER</th><th class="text-end">QTY</th>
                <th class="text-end">PRICE</th><th class="text-end">P/L</th>
              </tr></thead>
              <tbody id="pnl-rows"></tbody>
            </table>
          </div>
          <!-- Inline portfolio editor -->
          <div id="portfolio-editor" style="display:none;margin-top:10px;">
            <div style="font-size:11px;color:var(--text-secondary);margin-bottom:6px;">Edit positions — changes save to config and refresh live P&L</div>
            <table class="table table-sm table-borderless mb-1" style="font-size:12px;">
              <thead><tr style="color:var(--text-secondary);">
                <th>TICKER</th><th>TYPE</th><th>QTY</th><th>AVG COST</th><th></th>
              </tr></thead>
              <tbody id="edit-pos-rows"></tbody>
            </table>
            <button class="btn btn-sm btn-outline-success py-0 px-2 me-1" style="font-size:11px" onclick="addEditRow()">+ Add</button>
            <button class="btn btn-sm btn-primary py-0 px-2" style="font-size:11px" onclick="savePortfolioConfig()">Save</button>
            <span id="edit-save-status" style="font-size:11px;color:var(--text-secondary);margin-left:8px;"></span>
          </div>
        </div>
      </div>

      <div class="card mb-3">
        <div class="card-header">Top Signals</div>
        <div class="card-body p-2 scroll-pane">
          <ul class="list-unstyled mb-0" id="signals-list"><li class="small-muted">loading…</li></ul>
        </div>
      </div>
    </div>

    <div class="col-12 col-lg-8">
      <div class="card mb-3">
        <div class="card-header">Recent Briefings</div>
        <div class="card-body p-2 scroll-pane">
          <ul class="list-unstyled mb-0" id="briefings-list"><li class="small-muted">loading…</li></ul>
        </div>
      </div>
      <div class="card mb-3">
        <div class="card-header">High-Score Articles</div>
        <div class="card-body p-2 scroll-pane">
          <ul class="list-unstyled mb-0" id="articles-list"><li class="small-muted">loading…</li></ul>
        </div>
      </div>

      <!-- Collector health table -->
      <div class="card mb-3" id="collectors-card">
        <div class="card-header d-flex justify-content-between align-items-center" id="collectors">
          <span>Collectors <span class="small-muted ms-1">— live source pulse</span></span>
          <button class="btn btn-sm btn-outline-secondary py-0 px-2" style="font-size:11px;" onclick="refreshCollectors()">↻ refresh</button>
        </div>
        <div class="card-body p-2">
          <div class="small-muted mb-1" id="collectors-meta">loading…</div>
          <div class="table-responsive scroll-pane table-scroll" style="max-height:280px;">
            <table class="table table-sm table-borderless mb-0">
              <thead><tr>
                <th>SOURCE</th>
                <th class="text-end">1h</th>
                <th class="text-end">24h</th>
                <th class="text-end">STATUS</th>
              </tr></thead>
              <tbody id="collectors-rows"></tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- ML model status -->
      <div class="card mb-3" id="ml-card">
        <div class="card-header">ML Model Status</div>
        <div class="card-body p-2">
          <div class="row g-2" style="font-size:13px;">
            <div class="col-6 col-md-3">
              <div class="small-muted" style="font-size:11px;text-transform:uppercase;letter-spacing:0.05em;">last trained</div>
              <div id="ml-last-trained" class="ticker">—</div>
            </div>
            <div class="col-6 col-md-3">
              <div class="small-muted" style="font-size:11px;text-transform:uppercase;letter-spacing:0.05em;">training set</div>
              <div id="ml-set" class="ticker">—</div>
            </div>
            <div class="col-6 col-md-3">
              <div class="small-muted" style="font-size:11px;text-transform:uppercase;letter-spacing:0.05em;">scored 24h</div>
              <div id="ml-preds" class="ticker">—</div>
            </div>
            <div class="col-6 col-md-3">
              <div class="small-muted" style="font-size:11px;text-transform:uppercase;letter-spacing:0.05em;">val loss</div>
              <div id="ml-val" class="ticker">—</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Article volume chart (last 24h, hourly) -->
      <div class="card mb-3" id="volume-card">
        <div class="card-header">Article volume — last 24h</div>
        <div class="card-body p-2">
          <div style="position:relative;height:180px;"><canvas id="volume-chart"></canvas></div>
        </div>
      </div>

      <!-- Backtest isolation invariant badge -->
      <div class="card mb-3" id="invariants-card">
        <div class="card-header">System invariants</div>
        <div class="card-body p-2">
          <div id="iso-badge" class="small-muted">checking…</div>
          <div class="small-muted mt-1" id="iso-detail" style="font-size:11px;"></div>
        </div>
      </div>
    </div>
  </div>
</div>
</div>

<script>
const API_PREFIX = "__API_PREFIX__";
const params = new URLSearchParams(location.search);
const KEY = params.get("key") || "";
const qs = KEY ? ("?key=" + encodeURIComponent(KEY)) : "";

async function getJSON(path) {
  try {
    const r = await fetch(API_PREFIX + path + qs);
    if (!r.ok) return null;
    return await r.json();
  } catch (e) { return null; }
}

function fmt(n) {
  if (n === null || n === undefined) return "—";
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, {maximumFractionDigits: 0});
  return Number(n).toFixed(2);
}

function plClass(v) { if (v === null || v === undefined) return ""; return v >= 0 ? "pl-pos" : "pl-neg"; }

function parseDate(ts) {
    if (!ts && ts !== 0) return null;
    // Unix epoch as string or number (10+ digit seconds, optional fractional)
    if (/^\\d{10,}(\\.\\d+)?$/.test(String(ts).trim())) return new Date(parseFloat(ts) * 1000);
    // Try direct parse (handles ISO8601 and RFC822)
    const d = new Date(ts);
    return isNaN(d.getTime()) ? null : d;
}
function relativeTime(ts) {
    const d = parseDate(ts);
    if (!d) return '';
    const diffMs = Date.now() - d.getTime();
    if (diffMs < 0) return 'just now';
    const diffM = Math.floor(diffMs / 60000);
    if (diffM < 1) return 'just now';
    if (diffM < 60) return diffM + 'm ago';
    const diffH = Math.floor(diffM / 60);
    if (diffH < 24) return diffH + 'h ago';
    const diffD = Math.floor(diffH / 24);
    if (diffD < 7) return diffD + 'd ago';
    return d.toLocaleDateString();
}

async function refresh() {
  const [pnl, articles, briefings, stats] = await Promise.all([
    getJSON("/api/portfolio"),
    getJSON("/api/articles?limit=20"),
    getJSON("/api/briefings"),
    getJSON("/api/stats"),
  ]);

  const sumDiv = document.getElementById("pnl-summary");
  const rowsBody = document.getElementById("pnl-rows");
  rowsBody.innerHTML = "";
  if (!pnl || pnl.error) {
    sumDiv.textContent = (pnl && pnl.error) ? pnl.error : "no snapshot";
  } else {
    const s = pnl.summary || {};
    const totPnl = s.grand_pnl ?? s.total_pnl ?? 0;
    const totPnlPct = s.grand_pnl_pct ?? s.total_pnl_pct ?? 0;
    sumDiv.innerHTML = `<span class="ticker">Total</span> $${fmt(s.grand_value ?? s.total_value)} ` +
      `<span class="${plClass(totPnl)}">${(totPnl>=0?'+':'')}${fmt(totPnl)} (${(totPnlPct>=0?'+':'')}${fmt(totPnlPct)}%)</span>` +
      `<br><span class="small-muted">as of ${pnl.as_of || ""}</span>`;
    const allPositions = (pnl.positions || []).concat(pnl.options || []);
    for (const p of allPositions) {
      const tr = document.createElement("tr");
      const pnlV = p.pnl;
      tr.innerHTML = `<td class="ticker">${p.ticker || p.symbol || ""}</td>` +
        `<td class="text-end">${fmt(p.qty)}</td>` +
        `<td class="text-end">${p.price === null ? '—' : fmt(p.price)}</td>` +
        `<td class="text-end ${plClass(pnlV)}">${pnlV === null ? '—' : ((pnlV>=0?'+':'')+fmt(pnlV))}</td>`;
      rowsBody.appendChild(tr);
    }
  }

  const sigs = document.getElementById("signals-list");
  sigs.innerHTML = "";
  if (articles && Array.isArray(articles)) {
    const top = articles.filter(a => (a.urgency||0) >= 1 || (a.score||0) >= 6.0).slice(0, 10);
    if (!top.length) { sigs.innerHTML = '<li class="small-muted">no urgent signals</li>'; }
    for (const a of top) {
      const li = document.createElement("li");
      const urgent = (a.urgency||0) >= 1;
      li.className = "mb-1";
      li.innerHTML = `<span class="badge ${urgent?'badge-urgent':'badge-score'} me-1">${urgent?'URG':fmt(a.score)}</span>` +
        `<a href="${a.url}" target="_blank" rel="noopener">${a.title}</a>` +
        `<div class="small-muted">${a.source || ''} · ${relativeTime(a.published)}</div>`;
      sigs.appendChild(li);
    }
  } else {
    sigs.innerHTML = '<li class="small-muted">API requires &amp;key=… (see WEB_API_KEY)</li>';
  }

  const arts = document.getElementById("articles-list");
  arts.innerHTML = "";
  if (articles && Array.isArray(articles)) {
    for (const a of articles) {
      const li = document.createElement("li");
      li.className = "mb-2";
      li.innerHTML = `<div><span class="badge badge-score me-1">${fmt(a.score)}</span>` +
        `<a href="${a.url}" target="_blank" rel="noopener">${a.title}</a></div>` +
        `<div class="small-muted">${a.source || ''} · ${relativeTime(a.published)}</div>`;
      arts.appendChild(li);
    }
  }

  const brs = document.getElementById("briefings-list");
  brs.innerHTML = "";
  if (briefings && Array.isArray(briefings) && briefings.length) {
    for (const b of briefings) {
      const li = document.createElement("li");
      li.className = "mb-1 small-muted";
      li.innerHTML = `<span>${b.ts || ''}</span> — ${b.msg || ''}`;
      brs.appendChild(li);
    }
  } else {
    brs.innerHTML = '<li class="small-muted">no briefings logged yet</li>';
  }

  document.getElementById("last-updated").textContent = "updated " + new Date().toISOString();
}

async function refreshPaperTrader() {
  const sumDiv = document.getElementById("paper-trader-summary");
  try {
    const r = await fetch("/trader/api/portfolio");
    if (!r.ok) { sumDiv.textContent = "paper trader unavailable (HTTP " + r.status + ")"; return; }
    const j = await r.json();
    const tv = j.total_value;
    const pl = tv - 1000;
    const plPct = (pl / 1000) * 100;
    const cls = pl >= 0 ? "pl-pos" : "pl-neg";
    sumDiv.innerHTML = `<span class="ticker">Portfolio</span> $${fmt(tv)} ` +
      `<span class="${cls}">${(pl>=0?'+':'')}${fmt(pl)} (${(pl>=0?'+':'')}${fmt(plPct)}%)</span> ` +
      `<span class="small-muted">vs $1000 start · cash $${fmt(j.cash)}</span>`;
  } catch (e) {
    sumDiv.textContent = "paper trader unreachable";
  }
}

// ── News source edge (cross-fetched from the trader; matures with history) ──
// Mirrors refreshPaperTrader's exact /trader/ cross-fetch + r.ok degradation
// (don't re-derive it). A stale trader process predates /api/source-edge and
// 404s — surface that honestly instead of a blank card.
async function refreshSourceEdge() {
  const stEl = document.getElementById("se-state");
  const hlEl = document.getElementById("se-headline");
  const rowsEl = document.getElementById("se-rows");
  try {
    const r = await fetch("/trader/api/source-edge");
    if (r.status === 404) {
      stEl.textContent = "stale"; stEl.className = "badge bg-warning text-dark";
      hlEl.textContent = "restart paper-trader to apply (process predates /api/source-edge)";
      return;
    }
    if (!r.ok) { stEl.textContent = "n/a"; hlEl.textContent = "trader unavailable (HTTP " + r.status + ")"; return; }
    const j = await r.json();
    const v = j.verdict || "—";
    const cls = v === "EDGE_FOUND" ? "bg-success"
              : v === "NO_EDGE" ? "bg-danger"
              : v === "INSUFFICIENT_DATA" ? "bg-warning text-dark"
              : "bg-secondary";
    stEl.textContent = v.replace(/_/g, " "); stEl.className = "badge " + cls;
    hlEl.textContent = j.verdict_reason || "";
    const ref = String(j.reference_horizon || 3);
    const vcls = { EXPLOITABLE: "pl-pos", NEGATIVE: "pl-neg" };
    const rows = (j.sources || []).slice(0, 10).map(s => {
      const h = (s.horizons || {})[ref] || {};
      const abn = h.mean_abnormal_pct, hit = h.abnormal_hit_rate;
      return "<tr><td>" + s.source + "</td>"
        + "<td class='" + plClass(abn) + "'>" + (abn != null ? (abn>=0?"+":"") + fmt(abn) + "%" : "—") + "</td>"
        + "<td>" + (hit != null ? fmt(hit) + "%" : "—") + "</td>"
        + "<td>" + (s.n_resolved != null ? s.n_resolved : "—") + "</td>"
        + "<td class='" + (vcls[s.verdict] || "") + "'>" + (s.verdict || "—") + "</td></tr>";
    });
    rowsEl.innerHTML = rows.length ? rows.join("")
      : "<tr><td colspan='5' class='small-muted'>no collector resolved a watchlist move yet</td></tr>";
    document.getElementById("se-meta").textContent =
      "ref " + (j.reference_horizon != null ? j.reference_horizon + "d" : "—")
      + " · " + (j.n_resolved != null ? j.n_resolved : "—") + " resolved / "
      + (j.n_scored != null ? j.n_scored : "—") + " scored"
      + (j.spy_adjusted ? " · SPY-adjusted" : " · raw only")
      + (j.lookback_days != null ? " · " + j.lookback_days + "d lookback" : "");
  } catch (e) {
    stEl.textContent = "n/a"; hlEl.textContent = "trader unreachable";
  }
}

refresh();
refreshPaperTrader();
refreshSourceEdge();
refreshCollectors();
refreshMlStatus();
refreshVolumeChart();
refreshInvariants();
setInterval(refresh, 15000);
setInterval(refreshPaperTrader, 15000);
setInterval(refreshSourceEdge, 300000);
setInterval(refreshCollectors, 60000);
setInterval(refreshMlStatus, 120000);
setInterval(refreshVolumeChart, 300000);
setInterval(refreshInvariants, 60000);

// ── Collector health ─────────────────────────────────────────────────────────
async function refreshCollectors() {
  const meta = document.getElementById("collectors-meta");
  const tbody = document.getElementById("collectors-rows");
  const d = await getJSON("/api/collector-health");
  if (!d || d.error) {
    if (meta) meta.textContent = (d && d.error) ? d.error : "loading…";
    if (tbody) tbody.innerHTML = '<tr><td colspan="4" class="small-muted">—</td></tr>';
    return;
  }
  const sources = d.sources || [];
  const active = sources.filter(s => s.status === "active").length;
  const slow = sources.filter(s => s.status === "slow").length;
  const stale = sources.filter(s => s.status === "stale").length;
  if (meta) {
    meta.innerHTML =
      `<span style="color:var(--green);">●</span> ${active} active · ` +
      `<span style="color:var(--yellow);">●</span> ${slow} slow · ` +
      `<span style="color:var(--red);">●</span> ${stale} stale ` +
      `<span class="small-muted">(${sources.length} sources)</span>`;
  }
  if (!sources.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="small-muted">no sources reporting</td></tr>';
    return;
  }
  tbody.innerHTML = sources.map(s => {
    const dot = s.status === "active" ? '<span style="color:var(--green);">●</span> Active'
              : s.status === "slow"   ? '<span style="color:var(--yellow);">●</span> Slow'
              : s.status === "stale"  ? '<span style="color:var(--red);">●</span> Stale'
              :                          '<span class="small-muted">●</span> Idle';
    return `<tr>
      <td class="ticker">${(s.source||'?').replace(/</g,'&lt;')}</td>
      <td class="text-end">${s.articles_1h}</td>
      <td class="text-end">${s.articles_24h}</td>
      <td class="text-end" style="font-size:12px;">${dot}</td>
    </tr>`;
  }).join("");
}

// ── ML model status ──────────────────────────────────────────────────────────
async function refreshMlStatus() {
  const d = await getJSON("/api/ml-status");
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  if (!d || d.error) { set("ml-last-trained","—"); set("ml-set","—"); set("ml-preds","—"); set("ml-val","—"); return; }
  let last = "—";
  if (d.last_trained) {
    try {
      const t = new Date(d.last_trained);
      const ago = Math.max(0, (Date.now() - t.getTime())/60000);
      last = ago < 60 ? Math.round(ago) + "m ago"
           : ago < 1440 ? (ago/60).toFixed(1) + "h ago"
           : (ago/1440).toFixed(1) + "d ago";
    } catch (e) { last = d.last_trained; }
  }
  set("ml-last-trained", last);
  set("ml-set", d.training_set_size != null ? Number(d.training_set_size).toLocaleString() : "—");
  set("ml-preds", d.predictions_24h != null ? Number(d.predictions_24h).toLocaleString() : "—");
  set("ml-val", d.val_loss != null ? Number(d.val_loss).toFixed(4) : "—");
}

// ── Article volume bar chart ─────────────────────────────────────────────────
let _volumeChart = null;
async function refreshVolumeChart() {
  const d = await getJSON("/api/volume-history");
  const canvas = document.getElementById("volume-chart");
  if (!canvas || typeof Chart === "undefined") return;
  const rows = (d && d.hours) || [];
  const labels = rows.map(r => (r.hour || "").slice(11, 16));
  const counts = rows.map(r => r.count || 0);
  if (_volumeChart) _volumeChart.destroy();
  _volumeChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "articles/h",
        data: counts,
        backgroundColor: "rgba(10,205,255,0.55)",
        borderColor: "#0acdff",
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8b929d", maxRotation: 0, autoSkip: true, maxTicksLimit: 8 }, grid: { display: false } },
        y: { ticks: { color: "#8b929d", precision: 0 }, grid: { color: "rgba(255,255,255,0.04)" } },
      },
    },
  });
}

// ── Backtest isolation invariant ─────────────────────────────────────────────
async function refreshInvariants() {
  const badge = document.getElementById("iso-badge");
  const detail = document.getElementById("iso-detail");
  if (!badge) return;
  const d = await getJSON("/api/invariants");
  if (!d || d.error) {
    badge.innerHTML = '<span class="badge badge-score">Backtest isolation: UNKNOWN</span>';
    if (detail) detail.textContent = (d && d.error) ? d.error : "";
    return;
  }
  if (d.backtest_isolation === "active") {
    badge.innerHTML = '<span class="badge" style="background:rgba(0,200,150,0.18);color:var(--green);">Backtest isolation: ACTIVE ✓</span>';
    if (detail) {
      const n = d.backtest_rows_total || 0;
      detail.textContent = `${n.toLocaleString()} synthetic rows isolated from live alerts.`;
    }
  } else {
    badge.innerHTML = '<span class="badge" style="background:rgba(255,68,85,0.20);color:var(--red);">BREACH DETECTED ✗</span>';
    if (detail) detail.textContent = `${d.breach_count} backtest row(s) alerted as urgent.`;
  }
}

// ── Portfolio config editor ────────────────────────────────────────────────
let _editConfig = null;
async function togglePortfolioEdit() {
  const el = document.getElementById("portfolio-editor");
  if (el.style.display !== "none") { el.style.display = "none"; return; }
  const cfg = await getJSON("/api/portfolio/config");
  _editConfig = cfg;
  renderEditRows(cfg.positions || []);
  el.style.display = "block";
}
function renderEditRows(positions) {
  const tbody = document.getElementById("edit-pos-rows");
  tbody.innerHTML = "";
  positions.forEach((p, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><input class="form-control form-control-sm p-1" style="font-size:11px;background:var(--bg-input);color:var(--text);border-color:var(--border-strong);width:70px" value="${p.ticker||""}" data-i="${i}" data-f="ticker"></td>
      <td><select class="form-select form-select-sm p-1" style="font-size:11px;background:var(--bg-input);color:var(--text);border-color:var(--border-strong);width:90px" data-i="${i}" data-f="type">
        ${["stock","etf_leveraged","etf","option"].map(t=>`<option${p.type===t?" selected":""}>${t}</option>`).join("")}
      </select></td>
      <td><input class="form-control form-control-sm p-1" style="font-size:11px;background:var(--bg-input);color:var(--text);border-color:var(--border-strong);width:70px" value="${p.qty??""}" data-i="${i}" data-f="qty"></td>
      <td><input class="form-control form-control-sm p-1" style="font-size:11px;background:var(--bg-input);color:var(--text);border-color:var(--border-strong);width:80px" value="${p.avg_cost??""}" data-i="${i}" data-f="avg_cost"></td>
      <td><button class="btn btn-sm btn-outline-danger py-0 px-1" style="font-size:10px" onclick="removeEditRow(${i})">✕</button></td>`;
    tbody.appendChild(tr);
  });
}
function addEditRow() {
  if (!_editConfig) return;
  (_editConfig.positions = _editConfig.positions || []).push({ticker:"",type:"stock",qty:0,avg_cost:0});
  renderEditRows(_editConfig.positions);
}
function removeEditRow(i) {
  _editConfig.positions.splice(i, 1);
  renderEditRows(_editConfig.positions);
}
async function savePortfolioConfig() {
  // Collect all field edits from DOM
  document.querySelectorAll("#edit-pos-rows input, #edit-pos-rows select").forEach(el => {
    const i = parseInt(el.dataset.i), f = el.dataset.f;
    let v = el.value.trim();
    if (f === "qty" || f === "avg_cost") v = parseFloat(v) || 0;
    _editConfig.positions[i][f] = v;
  });
  const status = document.getElementById("edit-save-status");
  status.textContent = "saving…";
  try {
    const r = await fetch("/api/portfolio/config", {method:"PUT", headers:{"Content-Type":"application/json"}, body: JSON.stringify(_editConfig)});
    const d = await r.json();
    status.textContent = d.ok ? "✓ saved" : "error: " + d.error;
    if (d.ok) setTimeout(() => { refresh(); status.textContent = ""; }, 800);
  } catch(e) { status.textContent = "error: " + e; }
}
</script>

<!-- Floating chat widget -->
<button id="dichat-btn" aria-label="Open chat"
  style="position:fixed;bottom:20px;right:20px;width:56px;height:56px;border-radius:50%;background:var(--amber-dim);color:var(--amber);border:1px solid rgba(240,180,41,0.3);font-size:24px;cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,0.5);z-index:9998">✦</button>
<div id="dichat-panel"
  style="display:none;position:fixed;bottom:88px;right:20px;width:360px;height:480px;background:var(--bg-panel);color:var(--text);border:1px solid var(--border);border-radius:var(--radius);box-shadow:0 8px 24px rgba(0,0,0,0.5);z-index:9999;flex-direction:column;font-family:var(--font-sans)">
  <div style="padding:12px 14px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center">
    <span style="font-weight:600;color:var(--text)">Market Intel</span>
    <a href="/intern/chat" style="color:var(--text-secondary);font-size:0.8em;text-decoration:none;margin-left:auto;margin-right:10px">full ↗</a>
    <button id="dichat-close" style="background:none;border:none;color:var(--text-secondary);cursor:pointer;font-size:20px;line-height:1;padding:0 4px">×</button>
  </div>
  <div id="dichat-history" style="flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px;font-size:14px;line-height:1.45"></div>
  <form id="dichat-form" style="display:flex;gap:6px;padding:10px;border-top:1px solid var(--border)">
    <input id="dichat-input" type="text" autocomplete="off" placeholder="Ask about markets…"
      style="flex:1;background:var(--bg-input);border:1px solid var(--border-strong);color:var(--text);padding:8px 10px;border-radius:var(--radius-sm);font-size:14px;font-family:inherit">
    <button type="submit" id="dichat-send"
      style="background:var(--amber-dim);color:var(--amber);border:1px solid rgba(240,180,41,0.3);padding:8px 14px;border-radius:var(--radius-sm);cursor:pointer;font-weight:600">Send</button>
  </form>
</div>
<script>
(function(){
  const btn = document.getElementById('dichat-btn');
  const panel = document.getElementById('dichat-panel');
  const closeBtn = document.getElementById('dichat-close');
  const hist = document.getElementById('dichat-history');
  const form = document.getElementById('dichat-form');
  const inp = document.getElementById('dichat-input');
  const sendBtn = document.getElementById('dichat-send');
  const convo = [];
  let greeted = false;
  function openPanel(){ panel.style.display='flex'; btn.style.display='none'; inp.focus();
    if(!greeted){ bubble('assistant','Hi — ask about markets, news, or your portfolio.'); greeted=true; } }
  function closePanel(){ panel.style.display='none'; btn.style.display='block'; }
  btn.addEventListener('click', openPanel);
  closeBtn.addEventListener('click', closePanel);
  function renderMd(text) {
    if (typeof marked !== 'undefined') {
      try { return marked.parse(text, {breaks: true, gfm: true}); } catch(e) {}
    }
    return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>');
  }
  function bubble(role, text){
    const d = document.createElement('div');
    const bg = role==='user' ? 'var(--blue-dim)' : role==='error' ? 'var(--red-dim)' : 'var(--bg-elevated)';
    const fg = role==='user' ? 'var(--blue)' : role==='error' ? 'var(--red)' : 'var(--text)';
    const align = role==='user' ? 'flex-end' : 'flex-start';
    d.style.cssText = 'background:'+bg+';color:'+fg+';padding:8px 12px;border-radius:10px;max-width:85%;align-self:'+align+';word-wrap:break-word;overflow-wrap:anywhere';
    if (role === 'user') {
      d.textContent = text;
    } else {
      d.classList.add('md-body');
      d.innerHTML = renderMd(text);
    }
    hist.appendChild(d);
    hist.scrollTop = hist.scrollHeight;
    return d;
  }
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const m = inp.value.trim();
    if (!m || sendBtn.disabled) return;
    bubble('user', m);
    inp.value = '';
    sendBtn.disabled = true;
    const tmp = bubble('assistant', 'thinking…');
    tmp.style.opacity = '0.6';
    try {
      const r = await fetch('/api/chat', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({message:m, history: convo.slice()})
      });
      const j = await r.json();
      tmp.remove();
      if (!r.ok || j.error) {
        bubble('error', 'Error: ' + (j.error || ('HTTP '+r.status)));
      } else {
        const txt = j.response || j.reply || '(empty response)';
        bubble('assistant', txt);
        convo.push({role:'user', content:m});
        convo.push({role:'assistant', content:txt});
      }
    } catch (err) {
      tmp.remove();
      bubble('error', 'Network error: ' + err.message);
    } finally {
      sendBtn.disabled = false;
      inp.focus();
    }
  });
})();
</script>

<nav class="bottom-nav" id="bottomNav">
  <a href="/" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9.5L12 3l9 6.5V20a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1V9.5z"/></svg>
    <span>Home</span>
  </a>
  <a href="/intern/" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 14l4-4 3 3 5-6"/></svg>
    <span>Intern</span>
  </a>
  <a href="/trader/" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l6-6 4 4 8-8"/><path d="M17 7h4v4"/></svg>
    <span>Trader</span>
  </a>
  <a href="/intern/chat" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-8.5 8.5 8.5 8.5 0 0 1-3.8-.9L3 21l1.9-5.7A8.38 8.38 0 0 1 4 11.5 8.5 8.5 0 0 1 12.5 3 8.38 8.38 0 0 1 21 11.5z"/></svg>
    <span>Chat</span>
  </a>
  <a href="/trader/backtests" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><path d="M12 7v5l4 2"/></svg>
    <span>Backtests</span>
  </a>
</nav>
<script>
(function(){
  const navToggle = document.getElementById('navToggle');
  const navDrawer = document.getElementById('navDrawer');
  const navOverlay = document.getElementById('navOverlay');
  if (navToggle) {
    navToggle.addEventListener('click', () => {
      navDrawer.classList.toggle('open');
      navOverlay.classList.toggle('open');
    });
    navOverlay.addEventListener('click', () => {
      navDrawer.classList.remove('open');
      navOverlay.classList.remove('open');
    });
  }
  document.querySelectorAll('.bottom-tab').forEach(tab => {
    if (tab.getAttribute('href') === window.location.pathname) {
      tab.classList.add('active');
    }
  });
})();
</script>
</body>
</html>
"""


_CHAT_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Market Intel — Digital Intern</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Outfit:wght@400;500;600;700&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap');
    :root {
      --bg: #0c0d0f;
      --bg-panel: #111316;
      --bg-elevated: #17191d;
      --bg-hover: #1c1f24;
      --bg-input: #0e1012;
      --border: rgba(255,255,255,0.07);
      --border-strong: rgba(255,255,255,0.13);
      --text: #dde1e7;
      --text-secondary: #8b929d;
      --text-muted: #50565f;
      --amber: #f0b429;
      --amber-dim: rgba(240,180,41,0.12);
      --cyan: #0acdff;
      --cyan-dim: rgba(10,205,255,0.12);
      --green: #00c896;
      --green-dim: rgba(0,200,150,0.12);
      --red: #ff4455;
      --red-dim: rgba(255,68,85,0.12);
      --blue: #4d9eff;
      --blue-dim: rgba(77,158,255,0.12);
      --yellow: #fbbf24;
      --yellow-dim: rgba(251,191,36,0.12);
      --pink: #f472b6;
      --font-sans: 'Outfit', system-ui, sans-serif;
      --font-mono: 'DM Mono', 'JetBrains Mono', monospace;
      --font-display: 'Syne', system-ui, sans-serif;
      --radius: 8px;
      --radius-sm: 5px;
    }
    * { box-sizing: border-box; }
    html { overflow-x: hidden; max-width: 100%; }
    body { overflow-x: hidden; }
    html, body { margin: 0; padding: 0; height: 100%; }
    body {
      background: var(--bg); color: var(--text);
      font-family: var(--font-sans);
      font-size: 15px;
      line-height: 1.5;
      display: flex; flex-direction: column; height: 100vh;
    }
    .brand, h1, h2, h3 { font-family: var(--font-display); }
    .topbar {
      background: var(--bg-panel);
      border-bottom: 1px solid var(--border);
      padding: 0 20px;
      height: 48px;
      display: flex;
      align-items: center;
      gap: 2px;
      position: sticky;
      top: 0;
      z-index: 100;
      margin: 0;
      flex-shrink: 0;
      overflow: hidden;
      max-width: 100%;
    }
    .brand {
      font-weight: 700;
      color: var(--amber);
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-right: 16px;
      flex-shrink: 0;
    }
    .topbar a {
      color: var(--text-secondary);
      text-decoration: none;
      font-size: 13px;
      font-weight: 500;
      padding: 5px 12px;
      border-radius: var(--radius-sm);
      transition: color 0.15s, background 0.15s;
      white-space: nowrap;
    }
    .topbar a:hover { color: var(--text); background: var(--bg-hover); }
    .topbar a.active { color: var(--amber); background: var(--amber-dim); }
    .page-content {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-height: 0;
      width: 100%;
    }
    header.page {
      padding: 18px 24px 12px; border-bottom: 1px solid var(--border);
      background: var(--bg);
    }
    header.page h1 { margin: 0; font-size: 22px; font-weight: 600; color: var(--text); }
    header.page .sub { color: var(--text-secondary); font-size: 13px; margin-top: 4px; }
    .chat-wrap {
      flex: 1; overflow-y: auto; padding: 20px 24px;
      display: flex; flex-direction: column; gap: 14px;
    }
    .msg { max-width: 760px; padding: 12px 16px; border-radius: var(--radius); word-wrap: break-word; overflow-wrap: anywhere; line-height: 1.5; font-size: 14px; }
    .msg.user { white-space: pre-wrap; }
    .msg.user { align-self: flex-end; background: var(--blue-dim); color: var(--blue); border: 1px solid rgba(77,158,255,0.3); }
    .msg.assistant { align-self: flex-start; background: var(--bg-panel); border: 1px solid var(--border); color: var(--text); }
    .msg.error { align-self: flex-start; background: var(--red-dim); border: 1px solid var(--red); color: var(--red); }
    .sources { align-self: flex-start; max-width: 760px; display: flex; flex-wrap: wrap; gap: 6px; margin-top: -6px; }
    .chip {
      background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary);
      font-size: 11px; padding: 3px 8px; border-radius: 10px;
    }
    .suggestions { display: flex; flex-wrap: wrap; gap: 8px; padding: 14px 24px 4px; }
    .suggestion {
      background: var(--bg-elevated); border: 1px solid var(--border-strong); color: var(--cyan);
      padding: 6px 14px; border-radius: 18px; cursor: pointer; font-size: 13px;
      font-family: var(--font-sans);
    }
    .suggestion:hover { background: var(--bg-hover); color: var(--text); }
    .input-bar {
      display: flex; gap: 10px; padding: 14px 24px; border-top: 1px solid var(--border); background: var(--bg-panel);
    }
    input.msg-input {
      flex: 1; background: var(--bg-input); border: 1px solid var(--border-strong); color: var(--text);
      padding: 10px 14px; border-radius: var(--radius-sm); font-size: 14px;
      font-family: var(--font-sans);
    }
    input.msg-input:focus { outline: none; border-color: var(--amber); }
    button.send {
      background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(240,180,41,0.3); padding: 10px 20px;
      border-radius: var(--radius-sm); font-size: 13px; cursor: pointer; font-weight: 600;
      font-family: var(--font-sans);
      transition: background 0.15s;
    }
    button.send:hover { background: var(--bg-hover); }
    button.send:disabled { background: var(--bg-elevated); color: var(--text-muted); border-color: var(--border); cursor: not-allowed; }
    .spinner {
      display: inline-block; width: 10px; height: 10px;
      border: 2px solid var(--border-strong); border-top-color: var(--cyan);
      border-radius: 50%; animation: spin 0.8s linear infinite; vertical-align: middle; margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.5;} }
    .typing { color: var(--text-secondary); font-style: italic; }
    /* Markdown rendered content */
    .md-body p { margin: 0 0 0.6em; }
    .md-body p:last-child { margin-bottom: 0; }
    .md-body ul, .md-body ol { margin: 0.3em 0 0.6em 1.2em; padding: 0; }
    .md-body li { margin-bottom: 0.2em; }
    .md-body code { background: var(--bg-input); border: 1px solid var(--border-strong); padding: 1px 5px; border-radius: 4px; font-size: 0.87em; font-family: var(--font-mono); }
    .md-body pre { background: var(--bg-input); border: 1px solid var(--border-strong); padding: 10px 14px; border-radius: var(--radius-sm); overflow-x: auto; margin: 0.4em 0; }
    .md-body pre code { background: none; border: none; padding: 0; }
    .md-body strong { color: var(--text); }
    .md-body h1, .md-body h2, .md-body h3 { margin: 0.6em 0 0.3em; color: var(--text); font-size: 1em; }
    .md-body blockquote { border-left: 3px solid var(--border-strong); margin: 0.4em 0; padding-left: 10px; color: var(--text-secondary); }
    .md-body table { border-collapse: collapse; margin: 0.4em 0; width: 100%; }
    .md-body th, .md-body td { border: 1px solid var(--border-strong); padding: 4px 8px; text-align: left; }
    .md-body th { background: var(--bg-panel); }
    /* === Mobile-first responsive additions ============================== */
    .nav-hamburger {
      display: none; flex-direction: column; justify-content: space-between;
      width: 32px; height: 22px; background: none; border: none; cursor: pointer;
      padding: 0; margin-left: auto;
    }
    .nav-hamburger span {
      display: block; height: 2px; background: var(--text); border-radius: 2px;
      transition: all 0.2s;
    }
    .nav-drawer {
      position: fixed; top: 0; left: -280px; width: 280px; height: 100vh;
      background: var(--bg-panel); border-right: 1px solid #1e2028;
      z-index: 1000; transition: left 0.25s ease; overflow-y: auto; padding: 20px 0;
    }
    .nav-drawer.open { left: 0; }
    .nav-drawer-header {
      font-family: var(--font-display); font-weight: 700; color: var(--amber);
      font-size: 13px; letter-spacing: 0.1em; padding: 0 20px 20px;
      border-bottom: 1px solid #1e2028; margin-bottom: 8px;
    }
    .nav-drawer a {
      display: block; padding: 12px 20px; color: var(--text-secondary);
      text-decoration: none; font-size: 14px; transition: all 0.15s;
    }
    .nav-drawer a:hover, .nav-drawer a.active {
      color: var(--text); background: var(--bg-elevated);
    }
    .nav-overlay {
      display: none; position: fixed; inset: 0;
      background: rgba(0,0,0,0.6); z-index: 999;
    }
    .nav-overlay.open { display: block; }
    .bottom-nav {
      display: none; position: fixed; bottom: 0; left: 0; right: 0; height: 64px;
      background: var(--bg-panel); border-top: 1px solid #1e2028;
      grid-template-columns: repeat(5, 1fr); z-index: 200; align-items: stretch;
    }
    .bottom-tab {
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; gap: 4px; color: var(--text-secondary);
      text-decoration: none; font-size: 10px; min-height: 44px; transition: color 0.15s;
    }
    .bottom-tab svg { width: 20px; height: 20px; }
    .bottom-tab.active, .bottom-tab:hover { color: var(--amber); }
    .table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .table-scroll table { min-width: 500px; }
    .input-bar { position: sticky; bottom: 0; }
    @media (max-width: 768px) {
      .topbar-nav { display: none; }
      .nav-hamburger { display: flex; }
      body { font-size: 14px; }
      button, .btn, a.btn, [role="button"] { min-height: 44px; min-width: 44px; }
    }
    @media (max-width: 480px) {
      .bottom-nav { display: grid; }
      .topbar { padding: 0 16px; }
      .msg { max-width: 90%; }
      .chat-wrap { padding: 14px 16px 88px; }
      .input-bar { padding: 12px 16px; margin-bottom: 64px; }
      .suggestions { padding: 12px 16px 4px; }
      button.send { min-height: 44px; min-width: 44px; }
      table { font-size: 12px; }
      th, td { padding: 8px 10px; }
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js" async></script>
</head>
<body>
<nav class="topbar">
  <span class="brand">◈ TRADING STACK</span>
  <span class="topbar-nav" style="display:flex;align-items:center;gap:2px;">
    <a href="/">Command Center</a>
    <a href="/intern/">Digital Intern</a>
    <a href="/trader/">Paper Trader</a>
    <a href="/trader/backtests">Backtests</a>
    <a href="/backtests/compare">Compare</a>
    <a href="/journal">Journal</a>
    <a href="/ops/">Ops View</a>
    <a href="/intern/chat" class="active">Chat</a>
    <a href="/system/">System</a>
  </span>
  <button class="nav-hamburger" id="navToggle" aria-label="Menu">
    <span></span><span></span><span></span>
  </button>
</nav>
<div class="nav-drawer" id="navDrawer">
  <div class="nav-drawer-header">◈ TRADING STACK</div>
  <a href="/">Command Center</a>
  <a href="/intern/">Digital Intern</a>
  <a href="/trader/">Paper Trader</a>
  <a href="/trader/backtests">Backtests</a>
  <a href="/backtests/compare">Compare</a>
  <a href="/journal">Journal</a>
  <a href="/ops/">Ops View</a>
  <a href="/intern/chat" class="active">Chat</a>
  <a href="/system/">System</a>
</div>
<div class="nav-overlay" id="navOverlay"></div>
<div class="page-content">
<header class="page">
  <h1>Market Intel</h1>
  <div class="sub">Powered by Claude Opus 4.7 + Live News Feed</div>
</header>
<div class="suggestions" id="suggestions">
  <button class="suggestion" data-q="What's moving markets today?">What's moving markets today?</button>
  <button class="suggestion" data-q="Nokia surge analysis">Nokia surge analysis</button>
  <button class="suggestion" data-q="Best opportunities right now?">Best opportunities right now?</button>
  <button class="suggestion" data-q="What should I watch in Asia overnight?">What should I watch in Asia overnight?</button>
</div>
<div class="chat-wrap" id="chat"></div>
<div class="input-bar">
  <input class="msg-input" id="input" type="text" placeholder="Ask about markets, news, or your portfolio…" autocomplete="off" autofocus>
  <button class="send" id="send" type="button" onclick="sendMsg()">Send</button>
</div>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const msgs = [];

function sendMsg() { ask(input.value.trim()); }
input.addEventListener('keydown', function(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); } });

function scrollDown() {
  requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
}

function renderMd(text) {
  if (typeof marked !== 'undefined') {
    try { return marked.parse(text, {breaks: true, gfm: true}); } catch(e) {}
  }
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>');
}
function addMsg(role, content, sources) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (role === 'user') {
    div.textContent = content;
  } else {
    div.classList.add('md-body');
    div.innerHTML = renderMd(content);
  }
  chat.appendChild(div);
  if (sources && sources.length) {
    const wrap = document.createElement('div');
    wrap.className = 'sources';
    for (const s of sources) {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = s.length > 80 ? s.slice(0, 78) + '…' : s;
      wrap.appendChild(chip);
    }
    chat.appendChild(wrap);
  }
  scrollDown();
  return div;
}

function addLoader() {
  const div = document.createElement('div');
  div.className = 'msg assistant typing';
  div.innerHTML = '<span class="spinner"></span>Thinking…';
  chat.appendChild(div);
  scrollDown();
  return div;
}

async function ask(message) {
  if (!message || sendBtn.disabled) return;
  addMsg('user', message);
  msgs.push({role: 'user', content: message});
  input.value = '';
  sendBtn.disabled = true;
  const loader = addLoader();
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: message, history: msgs.slice(0, -1)}),
    });
    const j = await r.json();
    loader.remove();
    if (!r.ok || j.error) {
      addMsg('error', 'Error: ' + (j.error || ('HTTP ' + r.status)));
    } else {
      addMsg('assistant', j.response, j.sources || []);
      msgs.push({role: 'assistant', content: j.response});
    }
  } catch (e) {
    loader.remove();
    addMsg('error', 'Network error: ' + e.message);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

document.getElementById('suggestions').addEventListener('click', function(e) {
  var b = e.target.closest('.suggestion');
  if (b) ask(b.dataset.q);
});
</script>

<nav class="bottom-nav" id="bottomNav">
  <a href="/" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9.5L12 3l9 6.5V20a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1V9.5z"/></svg>
    <span>Home</span>
  </a>
  <a href="/intern/" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 14l4-4 3 3 5-6"/></svg>
    <span>Intern</span>
  </a>
  <a href="/trader/" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l6-6 4 4 8-8"/><path d="M17 7h4v4"/></svg>
    <span>Trader</span>
  </a>
  <a href="/intern/chat" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-8.5 8.5 8.5 8.5 0 0 1-3.8-.9L3 21l1.9-5.7A8.38 8.38 0 0 1 4 11.5 8.5 8.5 0 0 1 12.5 3 8.38 8.38 0 0 1 21 11.5z"/></svg>
    <span>Chat</span>
  </a>
  <a href="/trader/backtests" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><path d="M12 7v5l4 2"/></svg>
    <span>Backtests</span>
  </a>
</nav>
<script>
(function(){
  const navToggle = document.getElementById('navToggle');
  const navDrawer = document.getElementById('navDrawer');
  const navOverlay = document.getElementById('navOverlay');
  if (navToggle) {
    navToggle.addEventListener('click', () => {
      navDrawer.classList.toggle('open');
      navOverlay.classList.toggle('open');
    });
    navOverlay.addEventListener('click', () => {
      navDrawer.classList.remove('open');
      navOverlay.classList.remove('open');
    });
  }
  document.querySelectorAll('.bottom-tab').forEach(tab => {
    if (tab.getAttribute('href') === window.location.pathname) {
      tab.classList.add('active');
    }
  });
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # Standalone runtime: open the store directly and serve.
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from storage.article_store import ArticleStore  # noqa: E402
    run_server(ArticleStore())
