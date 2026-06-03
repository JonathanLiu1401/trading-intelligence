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
import threading
import time
import zlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

from core.claude_cli import claude_call as _claude_cli_call

BASE_DIR = Path(__file__).resolve().parent.parent
_LIVE_ONLY_SQL = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)

# Resolved lazily so importing this module doesn't require an instantiated store.
_store = None
_log = None
_dashboard_cache: dict[str, tuple[float, Any]] = {}
_dashboard_cache_lock = threading.Lock()


def _ttl_cache(key: str, ttl_s: float, build):
    now = time.monotonic()
    with _dashboard_cache_lock:
        hit = _dashboard_cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    value = build()
    with _dashboard_cache_lock:
        _dashboard_cache[key] = (now + ttl_s, value)
    return value


def _ttl_get(key: str):
    now = time.monotonic()
    with _dashboard_cache_lock:
        hit = _dashboard_cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    return None


def _ttl_set(key: str, ttl_s: float, value):
    with _dashboard_cache_lock:
        _dashboard_cache[key] = (time.monotonic() + ttl_s, value)
    return value


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


def _stats_from_db() -> dict:
    from storage.article_store import _get_db_path
    db_file = _get_db_path()
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True, timeout=1.0)
    try:
        total = int(conn.execute(
            "SELECT MAX(rowid) FROM articles"
        ).fetchone()[0] or 0)
        urgent = int(conn.execute(
            "SELECT COUNT(*) FROM "
            "(SELECT 1 FROM articles WHERE urgency>=1 LIMIT 10000)"
        ).fetchone()[0] or 0)
        last_hour = int(conn.execute(
            "SELECT COUNT(*) FROM articles "
            f"WHERE first_seen >= datetime('now','-1 hour') AND {_LIVE_ONLY_SQL}"
        ).fetchone()[0] or 0)
        last_24h = int(conn.execute(
            "SELECT COUNT(*) FROM articles "
            f"WHERE first_seen >= datetime('now','-24 hours') AND {_LIVE_ONLY_SQL}"
        ).fetchone()[0] or 0)
    finally:
        conn.close()
    size_bytes = 0
    if db_file is not None:
        for suffix in ("", "-wal", "-shm"):
            p = db_file.with_name(db_file.name + suffix)
            if p.exists():
                size_bytes += p.stat().st_size
    return {
        "total": total,
        "urgent": urgent,
        "unscored": None,
        "below_threshold": None,
        "db_mb": round(size_bytes / 1024 / 1024, 1),
        "last_hour": last_hour,
        "last_24h": last_24h,
        "standalone": True,
    }


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


_SE_REAL_VERDICTS = ("EDGE_FOUND", "NO_EDGE")


def _news_source_edge_chat_lines(rep) -> list[str]:
    """Render paper-trader's ``/api/source-edge`` (the read-only per-collector
    predictive-edge diagnostic — which of digital-intern's ~17 news sources'
    scored headlines actually precede the SPY-abnormal move?) as compact
    chat-context lines.

    The chat already carries ML GATE HONESTY (does the DecisionScorer beat a
    one-liner OOS?) which grades the *gate*; this is its read-COLLECTOR
    companion: even if the gate works, are the *inputs feeding it* edge-bearing
    or wire-noise? An analyst answering "should I trust this MarketWatch
    headline?" or "which sources actually move the tape?" has no other surface
    to compose this from — the per-source verdict only lives in the trader
    endpoint and the JS-only ``se-card`` dashboard panel.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` string
    is composed **verbatim** — no chat-side re-derived verdict that could
    drift from the trader endpoint. ``best_source`` / ``worst_source`` and
    the per-source pp/horizon math already live inside the headline; we never
    reconstruct them here.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict, missing/unknown ``verdict`` → ``[]`` (block omitted, never
      an exception into the chat handler)
    - ``INSUFFICIENT_DATA`` / ``NO_DATA`` → ONE honest withheld line; the
      module's ``headline`` is **not** surfaced (the trader endpoint never
      raises and a fault/empty-DB branch may put a stack/exception string
      in adjacent fields — keep the withheld line minimal & opaque so a
      pipeline fault cannot leak into the analyst prompt, mirroring the
      ``_baseline_compare_chat_lines`` INSUFFICIENT_DATA contract)
    - ``EDGE_FOUND`` / ``NO_EDGE`` → the verdict headline + the module's
      verbatim ``headline`` line (which already encodes best/worst source,
      pp at ref horizon, sample counts, and graded/total counts); a missing
      headline simply drops the second line, never raises
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    if verdict in ("INSUFFICIENT_DATA", "NO_DATA"):
        return [
            "News-source edge: insufficient resolved history — verdict "
            "withheld."
        ]
    if verdict not in _SE_REAL_VERDICTS:
        return []

    lookback = rep.get("lookback_days")
    ref = rep.get("reference_horizon")
    n_resolved = rep.get("n_resolved")
    tag_bits: list[str] = []
    if isinstance(lookback, int):
        tag_bits.append(f"{lookback}d lookback")
    if isinstance(ref, int):
        tag_bits.append(f"{ref}d ref")
    if isinstance(n_resolved, int):
        tag_bits.append(f"n_resolved={n_resolved}")
    tag = f" ({', '.join(tag_bits)})" if tag_bits else ""

    lines = [f"News-source edge{tag}: {verdict}"]

    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(f"  {headline}")          # verbatim SSOT — invariant #10
    return lines


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


def _idle_opportunity_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/idle-opportunity` (high-score watchlist
    arrivals during the *current* NO_DECISION drought) as compact chat-
    context lines.

    The chat already carries `_decision_paralysis_chat_lines` (the loop is
    alive but nothing happens) and `_opportunity_cost_chat_lines` (the
    backward forward-return read on past sit-outs). Neither answers the
    *right-now* version: while the bot is *currently* dark, are
    high-score watchlist signals arriving on names it could have acted on?
    A decision-storm with the SPY tape running is structurally different
    from a decision-storm with the tape quiet — and only `idle-opportunity`
    measures that difference live. The endpoint exists; this wrapper closes
    the chat-prompt gap (the documented `_*_chat_lines` SSOT/test pattern).

    SSOT (paper-trader invariant #10): the builder's own ``headline`` string
    is the verbatim chat headline — no re-derived verdict. The detail line
    restates the builder's *own* fields (drought.duration_hours,
    missed_top_ticker, missed_top_score) — never a recomputation (the
    ``_event_readiness_chat_lines`` / ``_decision_paralysis_chat_lines``
    precedent). Held-name regret is called out with a `(HELD)` suffix on
    the ticker, mirroring the trader-side ``opportunities[].held`` field
    so the chat can answer "the bot was dark on MY OWN position's news"
    without re-deriving holdings.

    Pure / total — exactly the ``_decision_paralysis_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never an exception into the chat
      handler)
    - non-actionable verdicts (``NO_DATA`` / ``NO_DROUGHT`` / ``OK`` with
      zero missed signals) → ``[]``: a healthy decision loop or an honest
      silence is silence in the chat (the
      ``_event_readiness_chat_lines`` / ``_decision_paralysis_chat_lines``
      silence precedent — never chat filler when nothing is wrong)
    - actionable (``OK`` with ``n_opportunities > 0``) → builder's verbatim
      ``headline`` (only when a usable string) + one detail line restating
      drought duration, NO_DECISION count, top missed ticker + score from
      the builder's OWN fields; a missing field degrades to a "?"
      placeholder rather than raises (the
      ``_paper_trader_position_lines`` precedent)
    """
    if not isinstance(rep, dict):
        return []
    state = rep.get("state")
    # Silence on healthy / drought-clear / no-regret states.
    if state in (None, "NO_DATA", "NO_DROUGHT", "ERROR"):
        return []
    n_opps = rep.get("n_opportunities")
    # Reject bool explicitly (Python's True passes isinstance(_, int)) and
    # non-integer types — the field is a count, not a flag.
    if (not isinstance(n_opps, int) or isinstance(n_opps, bool)
            or n_opps <= 0):
        # OK with zero missed signals — the "silence is honest" branch
        # the builder's own headline already labels. Chat carries it as
        # silence to match.
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)                  # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    drought = rep.get("drought") if isinstance(rep.get("drought"), dict) else {}
    dur = _num(drought.get("duration_hours"))
    nd = _num(drought.get("n_no_decision"))
    top_tk = rep.get("missed_top_ticker")
    top_score = _num(rep.get("missed_top_score"))

    # If the top opportunity flags ``held=True`` (the bot was dark on a
    # name we OWN), surface that — the chat-side version of the operator's
    # "the bot was blind on MY position" question. Read from the row
    # itself, not a re-derived predicate.
    held_suffix = ""
    opps = rep.get("opportunities")
    if isinstance(opps, list) and opps and isinstance(opps[0], dict):
        if opps[0].get("held") is True:
            held_suffix = " (HELD)"

    detail_parts: list[str] = []
    if dur is not None:
        detail_parts.append(f"drought {dur:.1f}h")
    if nd is not None:
        detail_parts.append(f"{int(nd)} NO_DECISION")
    if isinstance(top_tk, str) and top_tk.strip():
        if top_score is not None:
            detail_parts.append(
                f"loudest {top_tk}{held_suffix} @ {top_score:.1f}")
        else:
            detail_parts.append(f"loudest {top_tk}{held_suffix}")

    if detail_parts:
        lines.append("  " + " | ".join(detail_parts))

    return lines


def _persona_book_fit_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/persona-book-fit` (does the live book look
    like a backtest persona that actually carries alpha, or one rated DRAG?)
    as compact chat-context lines.

    The chat carries forward Kelly-sizing, regime-leverage fit, exit-intent
    audit — every block analyses *position-by-position* fitness — but no
    block surfaces the **structural** question of whether the entire book's
    weight distribution mirrors a persona archetype that historically loses
    money. ALIGNED_DRAG is the only "your book IS the persona that doesn't
    work" signal in the desk; nowhere else is this surfaced.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` string
    is the verbatim chat headline — no chat-side re-derived verdict that
    could drift from the trader endpoint. The dominant persona name + the
    EDGE alternatives carry through verbatim from the builder
    (``_event_readiness_chat_lines`` recommended_action precedent).

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]``
    - non-actionable verdicts (``ALIGNED_EDGE`` / ``ALIGNED_FLAT`` /
      ``NO_BOOK`` / ``WEAK_OVERLAP`` / ``INSUFFICIENT_PERSONA``) → ``[]``.
      Only ``ALIGNED_DRAG`` is worth surfacing — every other verdict is
      either healthy (silence — the ``_event_readiness_chat_lines``
      precedent) or insufficient (never chat filler).
    - ``ALIGNED_DRAG`` → builder's verbatim ``headline`` + one detail line
      restating the builder's own fields (overlap_pct, runner_up).
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("verdict") != "ALIGNED_DRAG":
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)             # verbatim SSOT — invariant #10

    dom = rep.get("dominant")
    runner_up = rep.get("runner_up")
    detail = []
    if isinstance(dom, dict):
        ov = dom.get("overlap_pct")
        if isinstance(ov, (int, float)) and not isinstance(ov, bool):
            detail.append(f"dominant overlap {ov:.1f}%")
    if isinstance(runner_up, dict):
        ru_name = runner_up.get("persona")
        ru_ov = runner_up.get("overlap_pct")
        if isinstance(ru_name, str) and isinstance(ru_ov, (int, float)) \
                and not isinstance(ru_ov, bool):
            detail.append(f"runner-up {ru_name} ({ru_ov:.1f}%)")
    if detail:
        lines.append("  " + " | ".join(detail))

    return lines


def _inverse_pair_conflict_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/inverse-pair-conflict-skill` (leveraged
    long + leveraged inverse of the same underlying family simultaneously
    held — the carry-waste pathology) as compact chat-context lines.

    Why this block exists. The watchlist is leveraged-ETF heavy
    (TQQQ/SQQQ, SOXL/SOXS, SPXL/SPXS, FNGU/FNGD, TECL/TECS) and Opus has
    full sizing autonomy. When the bot opens both sides of the same
    underlying the directional exposure largely cancels but the book
    keeps paying leverage decay on BOTH sleeves. Every other risk block
    misses this:
    - regime-leverage-fit reads "high leveraged %" without distinguishing
      paired vs one-sided
    - etf-lookthrough reports the net single-name outcome, not the carry
      waste fact
    - sector-exposure puts both into the same ``broad_lev`` bucket
    - correlation-cluster-warning flags positively-correlated clusters
      and lets the negatively-correlated TQQQ/SQQQ pair through

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is
    the chat headline; the worst-family `family_label` and dollar
    cancellation magnitudes pass through verbatim — no chat-side
    re-derived verdict.

    Pure / total — exactly the ``_decision_paralysis_chat_lines`` contract:

    - non-dict → ``[]``
    - non-actionable verdicts (``NO_BOOK`` / ``CLEAN`` /
      ``OPPOSING_UNLEVERED``) → ``[]``. OPPOSING_UNLEVERED is silenced
      because a 1x core + leveraged inverse pays only ONE decay tab —
      operationally distinct from the both-sides-burn CARRY_WASTE case
      and not worth the chat slot. Only CARRY_WASTE is loud.
    - ``CARRY_WASTE`` → builder's verbatim ``headline`` + one detail line
      restating the builder's own per-family fields (``cancelled_delta_usd``,
      ``daily_drag_estimate_usd``, ``severity``).
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("verdict") != "CARRY_WASTE":
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)              # verbatim SSOT — invariant #10

    conflicts = rep.get("conflicts")
    if isinstance(conflicts, list):
        # Surface the worst CARRY_WASTE conflict (already sorted by
        # cancelled_delta_usd DESC inside the builder, so the first
        # CARRY_WASTE in the list is the worst). Restate fields verbatim.
        worst = next(
            (c for c in conflicts
             if isinstance(c, dict)
             and c.get("classification") == "CARRY_WASTE"),
            None,
        )
        if worst is not None:
            detail = []
            cancelled = worst.get("cancelled_delta_usd")
            if isinstance(cancelled, (int, float)) and not isinstance(cancelled, bool):
                detail.append(f"cancelled Δ ${cancelled:g}")
            drag = worst.get("daily_drag_estimate_usd")
            if isinstance(drag, (int, float)) and not isinstance(drag, bool):
                detail.append(f"~${drag:g}/day drag")
            sev = worst.get("severity")
            if isinstance(sev, str) and sev:
                detail.append(f"severity {sev}")
            if detail:
                lines.append("  " + " | ".join(detail))

    return lines


def _watchlist_news_silence_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/watchlist-news-silence-skill` (per-
    WATCHLIST-ticker live-news coverage map; how many of the ~47 names
    Opus can choose from had zero recent news flow) as compact chat-
    context lines.

    Why this block exists. The chat already carries the held-news-silence
    block (digital-intern's own /api/held-news-silence), but the trader's
    UNIVERSE is far bigger than the book — and a silent-universe blind
    spot is the structural pathology behind a stretch of stale decisions:
    Opus is being asked to choose between AMD (38 articles, max_score 8.5)
    and AMAT (zero articles) and the prompt makes them look equally
    available. This is the chat-side mirror of the universe-coverage gap.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is
    the chat headline. The silent_tickers + hot_storms sub-lists pass
    through verbatim (cap-respected).

    Pure / total — exactly the ``_decision_paralysis_chat_lines`` contract:

    - non-dict → ``[]``
    - non-actionable verdicts (``WELL_COVERED`` / ``NO_DATA``) → ``[]``.
      Healthy or insufficient — never chat filler.
    - ``BLIND_UNIVERSE`` / ``SPARSE_COVERAGE`` → builder's verbatim
      ``headline`` + one detail line listing the top silent tickers and
      the top hot-storm tickers (so the analyst can see WHICH names are
      dark vs WHICH names are over-flooded).
    """
    if not isinstance(rep, dict):
        return []
    actionable = {"BLIND_UNIVERSE", "SPARSE_COVERAGE"}
    if rep.get("verdict") not in actionable:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)              # verbatim SSOT — invariant #10

    detail = []
    silent = rep.get("silent_tickers")
    if isinstance(silent, list) and silent:
        # Cap at 8 inline so the chat line stays readable. The builder
        # already capped at 10; we slice to 8 here for line length.
        sample = [t for t in silent if isinstance(t, str)][:8]
        if sample:
            detail.append("silent: " + ", ".join(sample))
    storms = rep.get("hot_storms")
    if isinstance(storms, list) and storms:
        names = [
            s.get("ticker") for s in storms
            if isinstance(s, dict) and isinstance(s.get("ticker"), str)
        ][:3]
        if names:
            detail.append("storms: " + ", ".join(names))
    if detail:
        lines.append("  " + " | ".join(detail))

    return lines


def _concurrent_opus_attribution_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/concurrent-opus-attribution` (per-parent-
    tree breakdown of concurrent ``claude --model claude-opus`` subprocesses
    saturating the host) as compact chat-context lines.

    Why this block exists. The chat already carries the host-saturation
    *count* (via `/api/host-guard` surfaced indirectly through
    runner-heartbeat headlines) but no chat block answers the operator's
    next question: WHICH parent tree owns the rogue Opus, and which
    targeted-kill command will restore the live runner's decision call?
    The 2026-05-23 17:47 paralysis (>55h frozen, 17 concurrent Opus all
    rooted in `scripts/hourly_review.sh`) made the gap explicit — every
    existing block describes the *consequence* (NO_DECISION storm,
    decision drought, alpha drift) and none names the rogue parent.

    SSOT (paper-trader invariant #10): the builder's own `headline` is
    the chat headline verbatim, and the `recommendation` string passes
    through unchanged. Detail line restates the dominant-culprit's own
    fields (n_opus, kill_command) — no chat-side re-derived verdict.

    Pure / total — exactly the ``_decision_paralysis_chat_lines`` contract:

    - non-dict → ``[]``
    - non-actionable verdicts (``NO_OPUS`` / ``CLEAN`` / ``BENIGN``) → ``[]``.
      Healthy or within host_guard's own threshold — never chat filler.
    - ``ELEVATED`` / ``SATURATED`` → builder's verbatim ``headline`` + one
      detail line restating the builder's own `recommendation`. SATURATED
      is the operator-critical case the >55h paralysis exposed.
    """
    if not isinstance(rep, dict):
        return []
    actionable = {"ELEVATED", "SATURATED"}
    if rep.get("verdict") not in actionable:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)              # verbatim SSOT — invariant #10

    recommendation = rep.get("recommendation")
    if isinstance(recommendation, str) and recommendation.strip():
        lines.append("  " + recommendation)  # verbatim restatement

    return lines


def _intent_followthrough_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/intent-followthrough` (does the bot
    actually execute its own STANDING conditional intents?) as compact
    chat-context lines.

    Complements `_standing_intents_chat_lines` (which lists the intents
    themselves) with the observational follow-up: of the actionable
    intents the bot stated, how many were FOLLOWED by a matching trade
    inside the evaluation window, how many ABANDONED, and what's the
    aggregate followthrough rate. A bot that emits crisp "wait for X,
    then buy Y" statements but never executes Y has perfect specificity
    on decision-vapor and zero followthrough — only this block catches
    the say-do gap.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is
    the chat headline — no chat-side re-derived verdict (the
    ``_decision_paralysis_chat_lines`` precedent). Detail line restates
    the builder's *own* count fields (n_followed / n_abandoned /
    followthrough_rate / abstention.preserve_dead /
    abstention.restraint_broken) — never a recomputation (the
    ``_event_readiness_chat_lines`` precedent).

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - non-actionable verdicts (``DISCIPLINED`` / ``NO_DATA`` /
      ``NO_RESOLVED`` / ``ERROR``) → ``[]``: a disciplined desk is silence,
      matching the ``_decision_paralysis_chat_lines`` silence precedent —
      never chat filler when the bot is following through. ``NO_RESOLVED``
      collapses because the verdict isn't in yet (intents still pending);
      ``ERROR`` collapses because the endpoint's error envelope carries no
      actionable signal
    - actionable verdicts (``DRIFTING`` / ``ABANDONED``) → builder's
      verbatim ``headline`` (only when a usable string) + one detail line
      composed from the builder's count fields. A missing field degrades
      silently rather than raises (the ``_paper_trader_position_lines``
      precedent)
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    actionable = {"DRIFTING", "ABANDONED"}
    if verdict not in actionable:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)                  # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    n_followed = _num(rep.get("n_followed"))
    n_abandoned = _num(rep.get("n_abandoned"))
    n_pending = _num(rep.get("n_pending"))
    rate = _num(rep.get("followthrough_rate"))
    abst = rep.get("abstention") if isinstance(rep.get("abstention"),
                                                dict) else {}
    preserve_dead = _num(abst.get("n_preserve_dead"))
    restraint_broken = _num(abst.get("n_restraint_broken"))

    detail_parts: list[str] = []
    if (n_followed is not None and n_abandoned is not None and rate is not None
            and (n_followed + n_abandoned) > 0):
        detail_parts.append(
            f"followed {int(n_followed)} / abandoned {int(n_abandoned)} "
            f"({100.0 * rate:.0f}% followthrough)"
        )
    elif n_followed is not None and n_abandoned is not None:
        detail_parts.append(
            f"followed {int(n_followed)} / abandoned {int(n_abandoned)}"
        )
    if n_pending is not None and n_pending > 0:
        detail_parts.append(f"{int(n_pending)} pending")
    if preserve_dead is not None and preserve_dead > 0:
        detail_parts.append(f"{int(preserve_dead)} dry-powder dead-weight")
    if restraint_broken is not None and restraint_broken > 0:
        detail_parts.append(f"{int(restraint_broken)} restraint broken")

    if detail_parts:
        lines.append("  " + " | ".join(detail_parts))

    return lines


def _opportunity_cost_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/opportunity-cost` (graded HOLD-CASH /
    NO_DECISION sit-outs vs forward returns of the top-news watchlist
    ticker) as compact chat-context lines.

    The chat already carries forward-looking gap analytics (idle-opportunity
    during a current drought, cash_pct in the portfolio block, the FORWARD
    intent surface via standing-intents) but no block surfaces the
    *hindsight* read: of the sit-outs the bot already made, did they earn
    their keep, or did the watchlist run while we sat? A persistent
    MISSED_ALPHA verdict means cash discipline is COSTING alpha; a
    persistent DEFENSIVE_WIN means cash discipline is SAVING the book.
    Neither shows up as a discrete signal anywhere else.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is
    the chat headline — no chat-side re-derived verdict (the
    ``_decision_paralysis_chat_lines`` precedent). Detail line restates
    the builder's *own* `stats` fields (missed_pct / defensive_pct /
    mean_fwd_3d_pct / n_classified) — never a recomputation.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - non-actionable verdicts (``NEUTRAL`` / ``NO_DATA`` / ``ERROR``) →
      ``[]``: a neutral cash discipline is silence, matching the
      ``_decision_paralysis_chat_lines`` silence precedent — never chat
      filler when the bot's sit-outs are neither costing nor earning
      material alpha
    - actionable verdicts (``MISSED_ALPHA`` / ``DEFENSIVE_WIN``) → builder's
      verbatim ``headline`` (only when a usable string) + one detail line
      composed from `stats`. A missing field degrades silently rather than
      raises (the ``_paper_trader_position_lines`` precedent)
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    actionable = {"MISSED_ALPHA", "DEFENSIVE_WIN"}
    if verdict not in actionable:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)                  # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    stats = rep.get("stats") if isinstance(rep.get("stats"), dict) else {}
    n_classified = _num(stats.get("n_classified"))
    missed_pct = _num(stats.get("missed_pct"))
    defensive_pct = _num(stats.get("defensive_pct"))
    mean_3d = _num(stats.get("mean_fwd_3d_pct"))
    n_sitout = _num(stats.get("n_sitout_total"))

    detail_parts: list[str] = []
    if (n_classified is not None and missed_pct is not None
            and defensive_pct is not None):
        detail_parts.append(
            f"{int(n_classified)} sit-outs graded "
            f"(missed {missed_pct:.0f}% / defensive {defensive_pct:.0f}%)"
        )
    if mean_3d is not None:
        detail_parts.append(f"mean 3d {mean_3d:+.2f}%")
    if n_sitout is not None and n_classified is not None and n_sitout > n_classified:
        detail_parts.append(
            f"{int(n_sitout - n_classified)} too recent to grade"
        )

    if detail_parts:
        lines.append("  " + " | ".join(detail_parts))

    return lines


def _cash_redeployment_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/cash-redeployment-latency-skill` (post-SELL
    cash-to-next-BUY latency distribution; the sold-then-sat pathology) as
    compact chat-context lines.

    The chat already carries idle-cash snapshots (the `/api/risk` cash_pct in
    the portfolio block) and a point-in-time conviction-vs-cash fit, but no
    block surfaces the *interval-distribution* question: when the desk SELLs,
    how long does the freed capital sit before it's working again? A book that
    sells into a thesis weakening then sits for 5 days has the same headline
    cash_pct as one that redeploys in 6h — the desk in question is materially
    different. This is the chat-side surface for the existing builder.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is the
    chat headline — no chat-side re-derived verdict (the
    ``_decision_paralysis_chat_lines`` precedent). Detail line restates the
    builder's *own* `stats` fields (median, p25/p75 latency, n_stalled,
    total_freed_usd) — never a recomputation.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - non-actionable verdicts (``FAST_REDEPLOY`` / ``STEADY`` / ``NO_DATA``)
      → ``[]``: a healthy redeployment cadence is silence, matching the
      ``_decision_paralysis_chat_lines`` silence precedent — never chat
      filler when capital is moving fine
    - actionable verdicts (``SLOW`` / ``STALLED``) → builder's verbatim
      ``headline`` (only when a usable string) + one detail line composed
      from `stats` (the ``_macro_calendar_chat_lines`` precedent); a missing
      field degrades silently rather than raises (the
      ``_paper_trader_position_lines`` precedent)
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    actionable = {"SLOW", "STALLED"}
    if verdict not in actionable:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    stats = rep.get("stats") if isinstance(rep.get("stats"), dict) else {}
    median_h = _num(stats.get("median_latency_h"))
    p25_h = _num(stats.get("p25_latency_h"))
    p75_h = _num(stats.get("p75_latency_h"))
    n_stalled = _num(stats.get("n_stalled"))
    n_classifiable = _num(stats.get("n_classifiable"))
    total_freed = _num(stats.get("total_freed_usd"))
    total_redep = _num(stats.get("total_redeployed_usd"))

    detail_parts: list[str] = []
    if median_h is not None and p25_h is not None and p75_h is not None:
        detail_parts.append(
            f"latency p25/median/p75 = {p25_h:.1f}/{median_h:.1f}/{p75_h:.1f}h"
        )
    elif median_h is not None:
        detail_parts.append(f"median latency {median_h:.1f}h")
    if (n_stalled is not None and n_classifiable is not None
            and n_stalled > 0):
        detail_parts.append(
            f"{int(n_stalled)}/{int(n_classifiable)} SELLs never redeployed")
    if total_freed is not None and total_redep is not None and total_freed > 0:
        idle = total_freed - total_redep
        if idle > 0:
            detail_parts.append(f"${idle:,.0f} freed but unworked")

    if detail_parts:
        lines.append("  " + " | ".join(detail_parts))

    return lines


def _decision_vapor_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/decision-vapor-skill` (per-FILLED-decision
    grounded-reasoning detector — SPECIFIC / SEMI / VAPOR) as compact
    chat-context lines.

    The chat already carries the *what* of recent decisions (the trader
    snapshot, recent trades) but nothing answers the structural-quality
    question: are the FILLED decisions grounded in concrete numbers +
    catalysts + tickers, or has Opus been writing generic vapor? A vapor
    trade that fails has nothing for the next decision to learn from. This
    surfaces the gate-level honesty so the analyst can answer "is the bot
    thinking, or rationalising?"

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is the
    chat headline. When verdict is ``VAPOR_DECISIONS`` and a VAPOR sample is
    available, the sample excerpt is surfaced **verbatim** — no chat-side
    paraphrase. The ``_decision_paralysis_chat_lines`` precedent.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - non-actionable verdicts (``SPECIFIC`` / ``NO_DATA``) → ``[]``: a
      grounded decision pool is silence, matching the
      ``_decision_paralysis_chat_lines`` silence precedent — never chat
      filler when reasoning is fine
    - actionable verdicts (``MIXED`` / ``VAPOR_DECISIONS``) → builder's
      verbatim ``headline`` (only when a usable string) + one detail line
      from `stats`; ``VAPOR_DECISIONS`` additionally surfaces one verbatim
      VAPOR sample excerpt (if any), the `_thesis_drift_chat_lines`
      drift_reasons precedent for verbatim-passthrough rendering
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    actionable = {"MIXED", "VAPOR_DECISIONS"}
    if verdict not in actionable:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    stats = rep.get("stats") if isinstance(rep.get("stats"), dict) else {}
    n_filled = _num(stats.get("n_filled"))
    n_specific = _num(stats.get("n_specific"))
    n_semi = _num(stats.get("n_semi"))
    n_vapor = _num(stats.get("n_vapor"))

    detail_parts: list[str] = []
    if n_filled is not None and n_filled > 0:
        bits = []
        if n_specific is not None:
            bits.append(f"{int(n_specific)} SPECIFIC")
        if n_semi is not None:
            bits.append(f"{int(n_semi)} SEMI")
        if n_vapor is not None:
            bits.append(f"{int(n_vapor)} VAPOR")
        if bits:
            detail_parts.append(
                f"{int(n_filled)} FILLED: " + " / ".join(bits))
    if detail_parts:
        lines.append("  " + " | ".join(detail_parts))

    # For VAPOR_DECISIONS, surface one VAPOR exemplar so the analyst can see
    # what the bot is actually saying when reasoning collapses.
    if verdict == "VAPOR_DECISIONS":
        samples = rep.get("samples")
        if isinstance(samples, list):
            for s in samples:
                if not isinstance(s, dict):
                    continue
                if s.get("klass") != "VAPOR":
                    continue
                excerpt = s.get("excerpt")
                action = s.get("action_taken")
                if isinstance(excerpt, str) and excerpt.strip():
                    head = f"e.g. {action}: " if isinstance(action, str) else "e.g. "
                    lines.append(f"  {head}{excerpt.strip()}")
                    break

    return lines


def _regime_leverage_fit_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/regime-leverage-fit-skill` (book-leverage
    alignment vs prevailing SPY momentum regime) as compact chat-context
    lines.

    The watchlist is leveraged-ETF-heavy (TQQQ / SOXL / SQQQ / SOXS / SPXL /
    SPXS), so the analyst's single highest-stakes structural question is
    "are we positioned with or against the regime?" The chat carries no
    block that answers this — the portfolio block reports leveraged_pct as a
    scalar, but the *fit* (lev% × regime sign × flow direction) is what
    actually matters. A 0% leveraged book during a bull tape is just as
    structurally wrong as a 40% leveraged book during a bear — both are
    fightable in chat, neither shows up as a discrete signal anywhere else.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is the
    chat headline — no chat-side re-derived verdict (the
    ``_decision_paralysis_chat_lines`` precedent). Detail line restates the
    builder's *own* fields (regime, spy_mom_20d, leveraged_pct, recent flow)
    — never a recomputation.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - non-actionable verdicts (``ALIGNED`` / ``DEFENSIVE`` / ``NEUTRAL`` /
      ``NO_DATA``) → ``[]``: a regime-fit book is silence, matching the
      ``_decision_paralysis_chat_lines`` silence precedent — never chat
      filler when the structural tilt is fine
    - actionable verdicts (``BLIND_LEVERING`` / ``DANGEROUS_HEADWIND`` /
      ``MISSED_TAILWIND``) → builder's verbatim ``headline`` (only when a
      usable string) + one detail line composed from the builder's own
      `regime`, `spy_mom_20d`, `portfolio.leveraged_pct` and `recent_flow`
      fields (the ``_macro_calendar_chat_lines`` precedent); a missing field
      degrades silently rather than raises (the
      ``_paper_trader_position_lines`` precedent)
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    actionable = {"BLIND_LEVERING", "DANGEROUS_HEADWIND", "MISSED_TAILWIND"}
    if verdict not in actionable:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    regime = rep.get("regime") if isinstance(rep.get("regime"), str) else None
    spy_mom = _num(rep.get("spy_mom_20d"))
    portfolio = (rep.get("portfolio")
                 if isinstance(rep.get("portfolio"), dict) else {})
    lev_pct = _num(portfolio.get("leveraged_pct"))
    lev_usd = _num(portfolio.get("leveraged_usd"))
    n_lev_positions = _num(portfolio.get("n_leveraged_positions"))
    flow = (rep.get("recent_flow")
            if isinstance(rep.get("recent_flow"), dict) else {})
    buy_flow_pct = _num(flow.get("buy_flow_pct"))
    sell_flow_pct = _num(flow.get("sell_flow_pct"))
    flow_window = _num(flow.get("window_hours"))

    detail_parts: list[str] = []
    bits = []
    if regime is not None:
        bits.append(f"regime={regime}")
    if spy_mom is not None:
        bits.append(f"spy_mom_20d={spy_mom:.2f}%")
    if bits:
        detail_parts.append(" / ".join(bits))
    lev_bits = []
    if lev_pct is not None:
        lev_bits.append(f"leveraged={lev_pct}%")
    if lev_usd is not None and lev_usd > 0:
        lev_bits.append(f"${lev_usd:,.0f}")
    if n_lev_positions is not None and n_lev_positions > 0:
        lev_bits.append(f"{int(n_lev_positions)} pos")
    if lev_bits:
        detail_parts.append(" ".join(lev_bits))
    flow_bits = []
    if buy_flow_pct is not None and buy_flow_pct > 0:
        flow_bits.append(f"lev BUY flow {buy_flow_pct}%")
    if sell_flow_pct is not None and sell_flow_pct > 0:
        flow_bits.append(f"lev SELL flow {sell_flow_pct}%")
    if flow_bits and flow_window is not None:
        detail_parts.append(
            " ".join(flow_bits) + f" ({flow_window:g}h)")
    elif flow_bits:
        detail_parts.append(" ".join(flow_bits))

    if detail_parts:
        lines.append("  " + " | ".join(detail_parts))

    return lines


def _kelly_sizing_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/kelly-sizing` (Kelly-criterion sizing
    diagnostic — how the current top-position weight compares to a
    Kelly-optimal allocation derived from realised win-rate × payoff
    ratio) as compact chat-context lines.

    The portfolio block reports `concentration_top1_pct` as a scalar and
    `/api/concentration-cap` warns at a fixed threshold, but neither
    answers the *statistical* sizing question: given my realised edge
    (the same `payoff_ratio` and `actual_win_rate_pct` `/api/trade-
    asymmetry` already exposes), what fraction would Kelly allocate to
    the single best position, and how does the current top weight
    compare? A 65% concentration is justified by a 13× payoff and 67%
    win-rate; the same 65% on a flat edge is ruin-risk territory. This
    is the chat-side surface for that decision.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is
    the chat headline — no chat-side re-derived verdict (the
    ``_decision_paralysis_chat_lines`` precedent). Detail line restates
    the builder's own ``half_kelly_pct`` / ``full_kelly_pct`` /
    ``top_position_pct`` / ``top_position_ticker`` — never recomputed.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - non-actionable verdicts (``KELLY_ALIGNED`` / ``None``) → ``[]``: a
      Kelly-aligned book is silence, matching the
      ``_decision_paralysis_chat_lines`` silence precedent — never chat
      filler when the sizing is in the safety cushion
    - actionable verdicts (``UNDERSIZED`` / ``OVERSIZED`` /
      ``EXTREMELY_OVERSIZED`` / ``NEGATIVE_EDGE`` / ``INTENT_UNCLEAR``)
      → builder's verbatim ``headline`` (only when a usable string) + one
      detail line composed from the builder's own ``half_kelly_pct`` /
      ``full_kelly_pct`` / ``top_position_pct`` / ``top_position_ticker``
      (the ``_macro_calendar_chat_lines`` precedent); a missing field
      degrades silently rather than raises (the
      ``_paper_trader_position_lines`` precedent)
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    actionable = {"UNDERSIZED", "OVERSIZED", "EXTREMELY_OVERSIZED",
                  "NEGATIVE_EDGE"}
    if verdict not in actionable:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)                  # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    full_k = _num(rep.get("full_kelly_pct"))
    half_k = _num(rep.get("half_kelly_pct"))
    top_pct = _num(rep.get("top_position_pct"))
    top_tk = rep.get("top_position_ticker")

    detail_parts: list[str] = []
    if half_k is not None:
        detail_parts.append(f"half-Kelly target {half_k:.1f}%")
    if full_k is not None:
        detail_parts.append(f"full-Kelly {full_k:.1f}%")
    if top_pct is not None:
        top_line = f"current top {top_pct:.1f}%"
        if isinstance(top_tk, str) and top_tk.strip():
            top_line += f" ({top_tk})"
        detail_parts.append(top_line)

    if detail_parts:
        lines.append("  " + " | ".join(detail_parts))

    return lines


def _exit_intent_audit_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/exit-intent-audit` (per-closed-sell
    intent classification → outcome per bucket) as compact chat-context
    lines.

    `/api/loser-autopsy` classifies losers by an OBJECTIVE failure mode
    (hold-days × magnitude); `/api/winner-autopsy` looks at entry text
    on winners; `/api/round-trip-postmortem` judges exit *timing*
    against the next price drift. None classify the trader's STATED
    REASON for the sell. The DRAM whipsaw (2026-05-19, -17.7% in 1.1h)
    was exited with reasoning that cited "raising dry powder" and
    "post-earnings dip" — the bot was bleeding on DEFENSIVE_CASH_RAISE
    exits without anything in chat ever surfacing that pattern.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is
    the chat headline — no chat-side re-derived verdict (the
    ``_decision_paralysis_chat_lines`` precedent). Detail line restates
    the builder's own ``dominant_intent`` + that bucket's own stats
    (``n``, ``total_pnl_usd``, ``avg_pnl_pct``, ``win_rate_pct``) — never
    recomputed.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - non-actionable verdicts (``DOMINANT_INTENT_HEALTHY`` / ``None``) →
      ``[]``: a healthy dominant-intent mix is silence, matching the
      ``_decision_paralysis_chat_lines`` silence precedent — never chat
      filler when the most common exit reason is profitable on average
    - actionable verdicts (``DOMINANT_INTENT_BLEED`` / ``INTENT_UNCLEAR``)
      → builder's verbatim ``headline`` (only when a usable string) + one
      detail line composed from the builder's own ``dominant_intent`` +
      that bucket's stats (the ``_macro_calendar_chat_lines`` precedent);
      a missing field degrades silently rather than raises (the
      ``_paper_trader_position_lines`` precedent)
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    actionable = {"DOMINANT_INTENT_BLEED", "INTENT_UNCLEAR"}
    if verdict not in actionable:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)                  # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    dominant = rep.get("dominant_intent")
    buckets = rep.get("buckets") if isinstance(rep.get("buckets"), list) else []
    dom_bucket = next(
        (b for b in buckets
         if isinstance(b, dict) and b.get("intent") == dominant),
        None,
    )

    detail_parts: list[str] = []
    if isinstance(dominant, str) and dominant.strip():
        detail_parts.append(f"dominant={dominant}")
    if isinstance(dom_bucket, dict):
        n = _num(dom_bucket.get("n"))
        if n is not None:
            detail_parts.append(f"n={int(n)}")
        total_pnl = _num(dom_bucket.get("total_pnl_usd"))
        if total_pnl is not None:
            detail_parts.append(f"total ${total_pnl:+.2f}")
        avg_pct = _num(dom_bucket.get("avg_pnl_pct"))
        if avg_pct is not None:
            detail_parts.append(f"avg {avg_pct:+.2f}%/trip")
        wr = _num(dom_bucket.get("win_rate_pct"))
        if wr is not None:
            detail_parts.append(f"wr {wr:.0f}%")

    if detail_parts:
        lines.append("  " + " | ".join(detail_parts))

    return lines


def _realized_vs_unrealized_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/realized-vs-unrealized` (banked-vs-paper
    P&L split) as compact chat-context lines.

    Every other equity-shape block describes a scalar (portfolio total
    pnl%, drawdown%, β-attribution). None answers the single composition
    question that distinguishes a disciplined book from a lucky one:
    "of today's net P&L, how much is locked-in realized vs paper that
    can evaporate on the next adverse mark?" A +$50 book that is 100%
    realized is fundamentally different from the same headline that is
    100% open-paper, and the chat is where the analyst makes that
    distinction.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is
    the chat headline — no chat-side re-derived verdict (the
    ``_decision_paralysis_chat_lines`` precedent). Detail line restates
    the builder's *own* ``realized_pnl_usd`` / ``unrealized_pnl_usd``
    fields — never a recomputation.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - non-actionable verdicts (``BANKED`` / ``BALANCED`` / ``NO_DATA``)
      → ``[]``: a healthy split is silence, matching the
      ``_decision_paralysis_chat_lines`` silence precedent — never chat
      filler when the desk is in good shape
    - actionable verdicts (``DRAWING_DOWN`` / ``LEAKING_PAPER`` /
      ``PAPER_HEAVY``) → builder's verbatim ``headline`` + one detail
      line composed from the builder's own ``realized_pnl_usd`` /
      ``unrealized_pnl_usd`` (the ``_macro_calendar_chat_lines``
      precedent); a missing field degrades silently rather than raises
      (the ``_paper_trader_position_lines`` precedent)
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    actionable = {"DRAWING_DOWN", "LEAKING_PAPER", "PAPER_HEAVY"}
    if verdict not in actionable:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    r_usd = _num(rep.get("realized_pnl_usd"))
    u_usd = _num(rep.get("unrealized_pnl_usd"))
    net_pct = _num(rep.get("net_pnl_pct"))

    detail_parts: list[str] = []
    if r_usd is not None and u_usd is not None:
        detail_parts.append(
            f"realized ${r_usd:+,.2f} + unrealized ${u_usd:+,.2f}")
    if net_pct is not None:
        detail_parts.append(f"net {net_pct:+.2f}% of starting")

    if detail_parts:
        lines.append("  " + " | ".join(detail_parts))

    return lines


def _watchlist_coverage_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/watchlist-coverage` (per-watchlist-ticker
    attention scan over the recent decision stream) as compact chat-context
    lines.

    The chat carries plenty of *position*-centric and *trade*-centric
    blocks but nothing names a ticker the bot has stopped attending
    to. The live WATCHLIST has 48 tickers; if 36 of them have not
    appeared in 1000 decisions while NVDA absorbs 100+ actions, the
    analyst should see "STAGNANT — 75% of universe untouched" before
    the next prompt — that is opportunity cost that no other surface
    exposes.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is
    the chat headline — no chat-side re-derived verdict (the
    ``_decision_paralysis_chat_lines`` precedent). Detail line restates
    builder's *own* ``n_never_seen`` / ``n_active_24h`` fields and a
    sample of the stalest tickers from ``by_ticker`` (verbatim
    passthrough — the ``_thesis_drift_chat_lines`` drift_reasons
    precedent).

    Pure / total — same contract as the sibling helpers:

    - non-dict → ``[]``
    - non-actionable verdicts (``DIVERSIFIED`` / ``NO_DATA``) → ``[]``:
      healthy coverage breadth is silence
    - actionable verdicts (``STAGNANT`` / ``CONCENTRATED``) → builder's
      verbatim ``headline`` (only when usable) + one detail line +
      (for STAGNANT only) up to ``MAX_STALE_TICKERS_SHOWN`` ticker
      symbols verbatim from ``by_ticker``'s most-stale entries; missing
      fields degrade silently rather than raise
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    actionable = {"STAGNANT", "CONCENTRATED"}
    if verdict not in actionable:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    n_never = _num(rep.get("n_never_seen"))
    n_stale = _num(rep.get("n_stale_7d"))
    n_active = _num(rep.get("n_active_24h"))
    n_wl = _num(rep.get("n_watchlist"))
    top3 = _num(rep.get("top_3_share_24h"))

    detail_parts: list[str] = []
    if (n_never is not None and n_stale is not None
            and n_active is not None and n_wl is not None and n_wl > 0):
        detail_parts.append(
            f"{int(n_never)} never-seen / {int(n_stale)} stale-7d / "
            f"{int(n_active)} active-24h of {int(n_wl)} watchlist")
    elif n_wl is not None:
        detail_parts.append(f"{int(n_wl)}-ticker watchlist")
    if verdict == "CONCENTRATED" and top3 is not None:
        detail_parts.append(f"top-3 24h share {top3*100:.0f}%")

    if detail_parts:
        lines.append("  " + " | ".join(detail_parts))

    if verdict == "STAGNANT":
        # Surface a sample of the stalest tickers verbatim — the
        # analyst's "names you should be looking at" prompt. Capped so
        # the chat block stays compact.
        MAX_STALE_TICKERS_SHOWN = 8
        by_ticker = rep.get("by_ticker")
        if isinstance(by_ticker, list):
            sample: list[str] = []
            for row in by_ticker:
                if not isinstance(row, dict):
                    continue
                if not (row.get("never_seen")
                        or (isinstance(row.get("hours_since_last_seen"),
                                       (int, float))
                            and row["hours_since_last_seen"] > 168.0)):
                    continue
                tk = row.get("ticker")
                if isinstance(tk, str) and tk:
                    sample.append(tk)
                    if len(sample) >= MAX_STALE_TICKERS_SHOWN:
                        break
            if sample:
                lines.append("  stale: " + ", ".join(sample))

    return lines


def _concentration_trajectory_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/concentration-trajectory` (the daily-snapshot
    slope view of single-name concentration) as compact chat-context lines.

    Every existing chat block describing book shape is **point-in-time**: the
    portfolio snapshot reports current cash%, /api/risk reports current
    top1_pct, the correlation block reports current factor structure. None
    answers the first-derivative question: *over the past N days, has the
    book's single-name concentration been rising, falling, or steady?* A book
    sitting at 65% top-1 today reads identically in every other surface
    whether it ramped from 30% → 65% over a week (concentration creep — the
    desk drifted in) or jumped 0% → 65% in the last cycle (a single fill blew
    it up — different operator response).

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is the
    chat headline — no chat-side re-derived verdict (the
    ``_decision_paralysis_chat_lines`` precedent). Detail line restates the
    builder's *own* ``current`` + ``delta_top1_pct`` fields — never a
    recomputation.

    Pure / total — same contract as the sibling helpers:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - non-actionable verdicts (``DECONCENTRATING`` / ``DIVERSIFIED`` /
      ``BALANCED`` / ``INSUFFICIENT_DATA`` / ``NO_DATA``) → ``[]``: a healthy
      or improving trajectory is silence, matching the
      ``_decision_paralysis_chat_lines`` silence precedent — never chat
      filler when the book is diversifying or already broad
    - actionable verdicts (``CONCENTRATION_SPIKE`` / ``RAMPING_UP`` /
      ``CONCENTRATED_STEADY``) → builder's verbatim ``headline`` (only when
      usable) + one detail line composed from the builder's own ``current``
      snapshot and ``delta_top1_pct`` (the ``_macro_calendar_chat_lines``
      precedent); a missing field degrades silently rather than raises (the
      ``_paper_trader_position_lines`` precedent)
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    actionable = {"CONCENTRATION_SPIKE", "RAMPING_UP", "CONCENTRATED_STEADY"}
    if verdict not in actionable:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    current = rep.get("current") if isinstance(rep.get("current"), dict) else {}
    top1_pct = _num(current.get("top1_pct"))
    top1_ticker = current.get("top1_ticker")
    top3_pct = _num(current.get("top3_pct"))
    n_pos = _num(current.get("n_positions"))
    delta = _num(rep.get("delta_top1_pct"))
    window_days = _num(rep.get("window_days"))

    detail_parts: list[str] = []
    if (top1_pct is not None and isinstance(top1_ticker, str)
            and top1_ticker.strip() and n_pos is not None):
        detail_parts.append(
            f"top-1: {top1_ticker} {top1_pct:.1f}% of book "
            f"({int(n_pos)} name{'s' if int(n_pos) != 1 else ''})"
        )
    elif top1_pct is not None and n_pos is not None:
        detail_parts.append(
            f"top-1 {top1_pct:.1f}% "
            f"({int(n_pos)} name{'s' if int(n_pos) != 1 else ''})"
        )
    if top3_pct is not None:
        detail_parts.append(f"top-3 {top3_pct:.1f}%")
    if (delta is not None and window_days is not None
            and verdict in {"RAMPING_UP", "CONCENTRATION_SPIKE"}):
        detail_parts.append(
            f"{delta:+.1f}pp over {int(window_days)}d"
        )

    if detail_parts:
        lines.append("  " + " | ".join(detail_parts))

    return lines


def _streak_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/streak` (current run + historical extremes
    on the closed round-trip series) as compact chat-context lines.

    The chat already carries plenty of *aggregate* behavioural reads — the
    scorecard summary, churn metrics, decision paralysis, hold discipline —
    but none surface the *streak structure* of the closed round-trips
    themselves. Two questions a desk asks the analyst that have no other
    chat block:

      * Am I on a hot hand or a cold streak right now? (Recent consecutive
        same-sign closes.) Useful for surfacing potential **tilt** after a
        loss cluster or **overconfidence** after a win cluster.
      * What are the historical extremes? (Longest W / L runs.) Context for
        whether the current run is normal or unusual.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is the
    chat headline — no chat-side re-derived verdict (the
    ``_decision_paralysis_chat_lines`` precedent). Detail line restates the
    builder's *own* ``current_streak`` + ``longest_win_streak`` /
    ``longest_loss_streak`` + ``n_round_trips`` fields — never a recompute.

    Pure / total — same contract as the sibling helpers:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - non-actionable verdicts (``NEUTRAL`` / ``None`` from EMERGING /
      NO_DATA states) → ``[]``: a stable book that just hasn't strung
      together a run is silence, matching the
      ``_decision_paralysis_chat_lines`` silence precedent — never chat
      filler when no streak signal is active
    - actionable verdicts (``HOT_HAND`` / ``TILT_RISK``) → builder's
      verbatim ``headline`` (only when usable) + one detail line composed
      from the builder's own ``current_streak`` / ``longest_*_streak`` /
      ``n_round_trips`` (the ``_macro_calendar_chat_lines`` precedent); a
      missing field degrades silently rather than raises (the
      ``_paper_trader_position_lines`` precedent)
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    actionable = {"HOT_HAND", "TILT_RISK"}
    if verdict not in actionable:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    cur = rep.get("current_streak") if isinstance(rep.get("current_streak"), dict) else {}
    cur_len = _num(cur.get("length"))
    cur_kind = cur.get("kind")
    longest_w = _num(rep.get("longest_win_streak"))
    longest_l = _num(rep.get("longest_loss_streak"))
    n_rts = _num(rep.get("n_round_trips"))

    detail_parts: list[str] = []
    if (cur_len is not None and isinstance(cur_kind, str)
            and cur_kind in ("WIN", "LOSS")):
        n = int(cur_len)
        if cur_kind == "WIN":
            word = "win" if n == 1 else "wins"
        else:
            word = "loss" if n == 1 else "losses"
        detail_parts.append(f"current run: {n} {word}")
    if longest_w is not None and longest_l is not None:
        detail_parts.append(
            f"longest W={int(longest_w)} / L={int(longest_l)}"
        )
    if n_rts is not None:
        detail_parts.append(
            f"{int(n_rts)} round-trip{'s' if int(n_rts) != 1 else ''}"
        )

    if detail_parts:
        lines.append("  " + " | ".join(detail_parts))

    return lines


def _standing_intents_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/decision-conditionals` (STANDING conditional
    intents extracted from recent decisions' reasoning) as compact chat lines.

    Every other reasoning-side chat block looks BACKWARD at what was said and
    done: ``_decision_vapor_chat_lines`` grades reasoning specificity on
    FILLED trades, ``_thesis_drift_chat_lines`` re-tests the open-position
    thesis, ``_exit_intent_audit_chat_lines`` classifies *closed* sells by
    stated motive. None surface the FORWARD slice — the explicit conditional
    intents the bot itself stated ("wait for the cash session", "rotating
    into LITE/LNOK", "premature to dump") and that are still STANDING
    (within freshness window) without a follow-up action.

    Answers the operator question no other block answers: *"what did the bot
    SAY it would do next, that it has not yet done?"*

    SSOT (paper-trader invariant #10): the builder's own ``headline`` passes
    through verbatim — no chat-side re-derived verdict (the
    ``_decision_paralysis_chat_lines`` precedent). Detail lines restate up
    to 3 intent snippets verbatim from ``intents`` — also no re-derivation
    (the ``_thesis_drift_chat_lines`` drift_reasons-verbatim precedent).

    Pure / total — same contract as the sibling helpers:

    - non-dict / missing ``verdict`` → ``[]`` (silence, never raises into the
      chat handler)
    - non-actionable verdicts (``NO_DATA`` / ``NO_INTENTS``) → ``[]``: a
      desk whose bot stated nothing forward-looking is silence, matching
      the ``_decision_paralysis_chat_lines`` silence precedent — never
      chat filler
    - actionable verdicts (``STANDING_INTENTS`` / ``STALE_INTENTS``) →
      builder's verbatim ``headline`` + up to 3 newest intent snippets
      (each: kind, ticker, age, verbatim text) so the analyst can read
      the bot's own words rather than a chat-side paraphrase. STALE_INTENTS
      additionally tags the line with ``[stale]`` so the operator can see
      at a glance which standing plans aged out without follow-up.
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    actionable = {"STANDING_INTENTS", "STALE_INTENTS"}
    if verdict not in actionable:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(f"Standing intents: {headline}")  # verbatim SSOT — invariant #10

    intents = rep.get("intents")
    if not isinstance(intents, list):
        return lines  # headline-only is honest; degrade rather than raise

    shown = 0
    for it in intents:
        if not isinstance(it, dict):
            continue
        kind = it.get("kind")
        ticker = it.get("ticker")
        text = it.get("text")
        age = it.get("age_hours")
        is_stale = bool(it.get("stale"))
        if not isinstance(kind, str) or not isinstance(text, str) or not text.strip():
            continue
        # Compose: "  • [watch-for] NVDA (0.3h): wait for cash session ..."
        # The text is already builder-clipped to ≤120 chars.
        tk = (ticker if isinstance(ticker, str) and ticker else "—")
        if isinstance(age, (int, float)) and not isinstance(age, bool):
            age_s = f"{float(age):.1f}h"
        else:
            age_s = "?"
        tag = " [stale]" if is_stale else ""
        lines.append(f"  • [{kind}] {tk} ({age_s}){tag}: {text}")
        shown += 1
        if shown >= 3:
            break

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


def _no_decision_reasons_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/no-decision-reasons` (per-bucket histogram
    of recent NO_DECISION causes — host_saturated / cli_nonzero_rc / parse_
    failed / claude_timeout / claude_empty / blocked / unknown) as compact
    chat-context lines.

    The chat already carries the decision-paralysis FACT ("we are not
    deciding") and the runner-heartbeat AVAILABILITY signal. Neither answers
    the operator's first follow-up: **WHY is the bot empty? is it the
    runner's fault, or is the box saturated by review agents / backtests?**
    The trader endpoint already buckets the cause and emits a verbatim
    `recommendation` — host saturation requires reducing parallel Opus jobs,
    not a runner restart; a parse_failed cluster is a prompt-shape bug, not
    a host issue. The chat-side surface lets the analyst answer "why is the
    bot silent right now?" without parsing daemon logs.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is the
    chat headline — no chat-side re-derived verdict that could drift from
    the trader endpoint (the ``_decision_paralysis_chat_lines`` precedent).
    The ``recommendation`` text is already inlined into the builder's
    headline (`"<n>/<N> cycles NO_DECISION; dominant cause: <bucket> (P%) —
    <recommendation>"`) so the headline alone is self-contained and
    actionable.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - state in {None, NO_DATA, NORMAL, MIXED} → ``[]``: only DOMINANT is
      actionable (one bucket exceeds the builder's own
      ``DOMINANT_THRESHOLD_PCT`` of the NO_DECISION rows); MIXED is the
      "no single cause owns this" state where the recommendation would be
      hand-wavy, matching the ``_decision_paralysis_chat_lines`` silence
      precedent — never chat filler when the cause is diffuse.
    - state DOMINANT → builder's verbatim ``headline`` (only when a usable
      string) + an optional detail line restating the top 3 buckets from
      ``buckets`` (counts only, no re-derivation of percentages); a missing
      / non-dict ``buckets`` degrades silently (the
      ``_paper_trader_position_lines`` precedent).
    """
    if not isinstance(rep, dict):
        return []
    state = rep.get("state")
    if state != "DOMINANT":
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    buckets = rep.get("buckets")
    if isinstance(buckets, dict) and buckets:
        ranked: list[tuple[str, int]] = []
        for k, v in buckets.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            try:
                n = int(v)
            except Exception:
                continue
            if n > 0:
                ranked.append((k, n))
        ranked.sort(key=lambda kv: kv[1], reverse=True)
        if ranked:
            top = ", ".join(f"{k}: {n}" for k, n in ranked[:3])
            lines.append(f"  bucket counts → {top}")

    return lines


def _round_trip_postmortem_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/round-trip-postmortem` (post-exit drift
    verdict per closed round-trip — CORRECT / PREMATURE / MISSED_RUNNER /
    WHIPSAW / NEUTRAL) as compact chat-context lines.

    Every existing realized-P&L chat surface reduces a closed trip to a
    P&L number: winner_autopsy / loser_autopsy / streak / scorecard all
    answer *what closed* and *at what dollar*. None ask the falsifiable
    hindsight question: **was the exit good — did the price keep moving
    against the bot after the sell?** Selling DRAM at -0.1% looks fine on
    track-record; it reads catastrophically if DRAM rallied +5% the hour
    after the sell. This block fills that gap so the analyst can answer
    "should we have held that exit?" without re-pricing trips manually.

    SSOT (paper-trader invariant #10): the builder's own top-level
    ``headline`` is the chat headline AND the surfaced worst-trip's own
    per-row ``headline`` passes through verbatim — no chat-side paraphrase
    of the bot's own per-trip narrative (the ``_decision_vapor_chat_lines``
    /  ``_thesis_drift_chat_lines`` verbatim-passthrough precedent).

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - state in {None, NO_DATA, INSUFFICIENT} → ``[]``: insufficient post-
      exit drift history means the verdict is silently withheld, not
      fabricated (the ``baseline_compare`` INSUFFICIENT_DATA → silent-but-
      honest precedent).
    - state OK with zero PREMATURE / MISSED_RUNNER / WHIPSAW trips → ``[]``:
      an all-CORRECT/NEUTRAL exit ladder is silence, matching the
      ``_decision_paralysis_chat_lines`` silence precedent — never chat
      filler when exits are timing fine.
    - actionable (≥1 unfavourable trip) → builder's verbatim top-level
      ``headline`` + the verbatim ``headline`` of the single worst trip
      (largest absolute post-exit drift among the unfavourable verdicts),
      ticker passthrough for traceability. Missing fields degrade silently.
    """
    if not isinstance(rep, dict):
        return []
    state = rep.get("state")
    if state != "OK":                      # only OK carries a usable verdict
        return []
    trips = rep.get("trips")
    if not isinstance(trips, list) or not trips:
        return []

    unfavourable = {"PREMATURE", "MISSED_RUNNER", "WHIPSAW"}

    def _drift_mag(t):
        d = t.get("post_exit_drift_pct") if isinstance(t, dict) else None
        if isinstance(d, bool) or not isinstance(d, (int, float)):
            return -1.0
        return abs(float(d))

    bad = [
        t for t in trips
        if isinstance(t, dict) and t.get("verdict") in unfavourable
    ]
    if not bad:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    # Pick the trip with the largest absolute post-exit drift among the
    # unfavourable verdicts (the most painful sample, by definition).
    worst = max(bad, key=_drift_mag)
    worst_headline = worst.get("headline")
    if isinstance(worst_headline, str) and worst_headline.strip():
        lines.append(f"  • {worst_headline}")

    return lines


def _cash_drag_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/cash-drag` (SPY-benchmarked $ cost of
    sitting in cash, per rolling window) as compact chat-context lines.

    The chat already carries idle-cash snapshots (the ``/api/risk`` cash_pct
    in the portfolio block), cash_redeployment latency (the post-SELL sit
    pathology) and opportunity_cost (signal-specific hindsight). None
    surface the BENCHMARKED dollar cost — "while you sat at avg cash $358
    over the last 168h, SPY ran +0.96% — that's $3.44 of beta you forfeited
    by being out". That's the answer to "is sitting in cash actually costing
    me?" the operator asks at the end of a multi-day cash stretch.

    SSOT (paper-trader invariant #10): the builder's own top-level
    ``headline`` is the chat headline — no chat-side re-derived verdict
    that could drift from the trader endpoint (the
    ``_decision_paralysis_chat_lines`` precedent). The detail line restates
    the worst window's *own* fields (window_hours / sp500_return_pct /
    avg_cash_usd / cash_drag_usd) — never a recomputation
    (``_macro_calendar_chat_lines`` field-passthrough precedent).

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - state != "OK" or top-level verdict not in {"COSTLY_CASH"} → ``[]``:
      NEUTRAL / HELPFUL_CASH / INSUFFICIENT / NO_DATA collapse to silence
      — cash that SAVED you money or had no benchmark to compare against
      is not actionable (the ``_decision_paralysis_chat_lines`` silence
      precedent — never chat filler when cash is fine or unscored).
    - actionable (COSTLY_CASH) → builder's verbatim ``headline`` (only when
      a usable string) + a worst-window detail line from ``windows`` (the
      single COSTLY_CASH window with the highest cash_drag_usd; ties
      broken by longer window_hours). Missing windows / fields degrade
      silently (the ``_paper_trader_position_lines`` precedent).
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("state") != "OK":
        return []
    if rep.get("verdict") != "COSTLY_CASH":
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    windows = rep.get("windows")
    if isinstance(windows, list):
        # Find the worst COSTLY_CASH window — highest drag $, ties broken
        # by longer window_hours (longer windows are weightier evidence).
        best: dict | None = None
        best_drag = -1.0
        best_hours = -1.0
        for w in windows:
            if not isinstance(w, dict):
                continue
            if w.get("state") != "OK" or w.get("verdict") != "COSTLY_CASH":
                continue
            drag = _num(w.get("cash_drag_usd"))
            hrs = _num(w.get("window_hours"))
            if drag is None:
                continue
            hkey = hrs if hrs is not None else -1.0
            if (drag > best_drag) or (drag == best_drag and hkey > best_hours):
                best = w
                best_drag = drag
                best_hours = hkey
        if best is not None:
            hrs = _num(best.get("window_hours"))
            spy = _num(best.get("sp500_return_pct"))
            avg = _num(best.get("avg_cash_usd"))
            drag = _num(best.get("cash_drag_usd"))
            parts: list[str] = []
            if hrs is not None:
                parts.append(f"window {hrs:.0f}h")
            if spy is not None:
                parts.append(f"SPY {spy:+.2f}%")
            if avg is not None:
                parts.append(f"avg cash ${avg:,.0f}")
            if drag is not None:
                parts.append(f"drag ${drag:,.2f}")
            if parts:
                lines.append("  " + " | ".join(parts))

    return lines


def _notify_health_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/notify-health` (the Discord-channel
    delivery health of the live trader) as compact chat-context lines.

    The chat carries every flavour of book / decision / skill analytics
    but is blind to the operator-fitness question "is the trader even
    able to reach Discord right now?". A DEGRADED channel means trade
    alerts, hourly summaries, and daily-close posts are silently being
    dropped — the analyst is talking about a book whose ops surface is
    DARK, and recommendations like "consider trimming TQQQ here" never
    reach the operator. That is a top-priority context every other block
    glosses over.

    SSOT (paper-trader invariant #10): the builder's own top-level
    ``headline`` is the chat headline — the trader endpoint already
    composes the "Discord channel DARK — N consecutive send failure(s),
    last OK <ts>, last error: <msg>" form with the right tone. The
    detail line restates only the builder's own ``consecutive_failures``
    / ``last_error`` / ``restart_recommended`` fields verbatim — never
    a recomputation (the ``_macro_calendar_chat_lines`` field-passthrough
    precedent).

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - top-level verdict not in {"DEGRADED"} → ``[]``: HEALTHY / UNKNOWN
      collapse to silence (the ``_decision_paralysis_chat_lines`` silence
      precedent — never chat filler when the channel is working).
    - actionable (DEGRADED) → builder's verbatim ``headline`` (only when
      a usable string) + a one-line detail restating builder fields.
      Missing fields degrade silently to the safe subset.
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("verdict") != "DEGRADED":
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    parts: list[str] = []
    n = rep.get("consecutive_failures")
    if isinstance(n, (int, float)) and not isinstance(n, bool):
        parts.append(f"consecutive_failures={int(n)}")
    last_err = rep.get("last_error")
    if isinstance(last_err, str) and last_err.strip():
        # Cap long error strings so a multi-line traceback can't blow the
        # chat block — defensive, mirrors the AGENTS.md raw_capture_chars
        # discipline on the trader side.
        msg = last_err.strip()
        if len(msg) > 160:
            msg = msg[:157] + "..."
        parts.append(f"last_error={msg!r}")
    rr = rep.get("restart_recommended")
    if isinstance(rr, bool):
        parts.append(f"restart_recommended={'YES' if rr else 'no'}")
    if parts:
        lines.append("  " + " | ".join(parts))

    return lines


def _all_cash_streak_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/all-cash-streak` (the chronic-flat-book
    surface) as compact chat-context lines.

    The chat carries cash_pct (point-in-time), cash_redeployment latency
    (post-SELL sit), opportunity_cost (signal-specific hindsight), and
    cash_drag (SPY-benchmarked dollar cost). None of those answer the
    OPERATOR-VISIBILITY question "how long has the bot ACTUALLY been
    100% cash, and at what compounding alpha cost so far?". The book
    looks identical at 100% cash whether it's been flat for 2h or 6 days,
    yet the second case is a strong "is anything broken / is the
    decision loop too risk-off?" signal the analyst should call out
    when answering "what's been going on?".

    SSOT (paper-trader invariant #10): the builder's own top-level
    ``headline`` is the chat headline ("all-cash 22.2h on $987.39; SPY
    +0.00% → no alpha cost — EXTENDED_HOLDOUT") — no chat-side
    re-derivation. The detail line restates only the builder's own
    current_streak fields (``hours_elapsed_to_now`` / ``cash_usd`` /
    ``spy_return_pct`` / ``alpha_cost_usd``) verbatim — never a
    recomputation.

    Pure / total — exactly the ``_cash_drag_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted)
    - ``state`` != "OK" → ``[]``: builder didn't run cleanly, stay silent
    - top-level ``verdict`` not in {"EXTENDED_HOLDOUT",
      "PROLONGED_HOLDOUT"} → ``[]``: BRIEF_HOLDOUT / NOT_ALL_CASH /
      NO_DATA / INSUFFICIENT_HISTORY collapse to silence (short flats
      and not-flat books are not chat filler).
    - actionable → builder's verbatim ``headline`` (only when a usable
      string) + a detail line from ``current_streak`` (the single
      currently-running streak; missing fields degrade silently).
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("state") != "OK":
        return []
    if rep.get("verdict") not in ("EXTENDED_HOLDOUT", "PROLONGED_HOLDOUT"):
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    cs = rep.get("current_streak")
    if isinstance(cs, dict):
        def _num(v):
            if isinstance(v, bool):
                return None
            if isinstance(v, (int, float)):
                return float(v)
            return None

        parts: list[str] = []
        hrs = _num(cs.get("hours_elapsed_to_now"))
        if hrs is None:
            hrs = _num(cs.get("hours"))
        if hrs is not None:
            parts.append(f"flat {hrs:.1f}h")
        cash = _num(cs.get("cash_usd"))
        if cash is not None:
            parts.append(f"cash ${cash:,.2f}")
        spy = _num(cs.get("spy_return_pct"))
        if spy is not None:
            parts.append(f"SPY {spy:+.2f}%")
        ac = _num(cs.get("alpha_cost_usd"))
        if ac is not None:
            parts.append(f"alpha_cost ${ac:,.2f}")
        if parts:
            lines.append("  " + " | ".join(parts))

    return lines


def _feed_health_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/feed-health` (the live-news pipeline
    fitness surface) as compact chat-context lines.

    The chat carries dozens of decision/book/skill analytics that all
    assume the news feed is alive. When `/api/feed-health` flips to
    BLIND (N consecutive 0-signal decisions), STALE_FEED (newest live
    article > stale_hours old), or fires the UNSCORED clause (live
    articles arriving but ai_score=0 — digital-intern ML scoring
    pipeline silently down), every downstream block's verdicts become
    interpretively suspect — a CASH_REDEPLOYMENT verdict of STALLED
    means a different thing if the bot is BLIND vs if the wire is live.
    The analyst needs to flag this BEFORE answering "what should we
    do?" because the right answer becomes "restart the scorer" or
    "wait for the feed to recover", not "trim NVDA". This block is the
    operator-fitness layer that gates every other read.

    SSOT (paper-trader invariant #10): the builder's own top-level
    ``headline`` is the chat headline — the trader endpoint already
    composes the right wording for each verdict AND already appends
    the UNSCORED-clause sub-message under BLIND / STALE_FEED. No
    chat-side re-derivation. The detail line restates only the
    builder's own counts (``resolved_live_2h`` / ``resolved_scored_2h``
    / ``blind_streak`` / ``resolved_newest_age_h``) verbatim.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted)
    - top-level ``verdict`` not in {"BLIND", "STALE_FEED"} → ``[]``:
      HEALTHY / NO_DATA collapse to silence (the
      ``_decision_paralysis_chat_lines`` silence precedent — never
      chat filler when the feed is working; NO_DATA is a probe-side
      defect, not an actionable feed-health verdict).
    - actionable → builder's verbatim ``headline`` (only when a usable
      string) + a one-line detail restating builder fields. Missing
      fields degrade silently to the safe subset.
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("verdict") not in ("BLIND", "STALE_FEED"):
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    parts: list[str] = []
    age = _num(rep.get("resolved_newest_age_h"))
    if age is not None:
        parts.append(f"newest age {age:.1f}h")
    live = _num(rep.get("resolved_live_2h"))
    if live is not None:
        parts.append(f"live_2h={int(live)}")
    scored = _num(rep.get("resolved_scored_2h"))
    if scored is not None:
        parts.append(f"scored_2h={int(scored)}")
    bs = _num(rep.get("blind_streak"))
    if bs is not None and bs > 0:
        parts.append(f"blind_streak={int(bs)}")
    if rep.get("unscored_feed") is True:
        parts.append("unscored_feed=YES")
    rr = rep.get("restart_recommended")
    if isinstance(rr, bool) and rr:
        parts.append("restart_recommended=YES")
    if parts:
        lines.append("  " + " | ".join(parts))

    return lines


def _hourly_pnl_fingerprint_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/hourly-pnl-fingerprint` (the per-hour-of-day
    alpha-vs-SPY fingerprint) into compact chat-context lines.

    The chat carries ~50 paper-trader analytics blocks for book/decisions/
    skills/news/sectors, but NOTHING answers the structural question
    "WHEN in the trading day has this bot actually earned alpha vs SPY?".
    A live trader with a clear MORNING_EDGE / AFTERNOON_EDGE fingerprint
    should be acting differently at hour 11 than at hour 15; the chat can
    answer "is now a good time to be aggressive?" only when this empirical
    time-of-day verdict is in the prompt. A FLAT_CLOCK book has no
    discernible hour-of-day edge — the silence precedent is correct,
    don't fill the chat with non-actionable readings.

    SSOT (paper-trader invariant #10): the builder's own top-level
    ``headline`` is the chat headline — no chat-side re-derivation of the
    verdict labeling or the spread numeric. The detail line restates only
    the builder's own ``best_hour`` / ``worst_hour`` / ``alpha_spread_pp``
    / ``n_alpha_samples`` fields verbatim.

    Pure / total — exactly the ``_feed_health_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - top-level ``verdict`` not in {"MORNING_EDGE", "MIDDAY_EDGE",
      "AFTERNOON_EDGE", "OFF_HOURS_EDGE"} → ``[]``: FLAT_CLOCK /
      INSUFFICIENT_DATA / NO_SPY_DATA / ERROR collapse to silence (the
      ``_feed_health_chat_lines`` silence precedent — never chat filler
      when the fingerprint is flat or the sample is too small).
    - actionable → builder's verbatim ``headline`` (only when a usable
      string) + a detail line restating best_hour / worst_hour /
      alpha_spread_pp / n_alpha_samples. Missing fields degrade silently.
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("verdict") not in (
        "MORNING_EDGE", "MIDDAY_EDGE", "AFTERNOON_EDGE", "OFF_HOURS_EDGE",
    ):
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    def _hour_phrase(hr, suffix):
        if not isinstance(hr, dict):
            return None
        h = hr.get("hour")
        if isinstance(h, bool) or not isinstance(h, (int, float)):
            return None
        alpha = _num(hr.get("mean_alpha_pct"))
        n = _num(hr.get("n_alpha_samples"))
        chunks = [f"{suffix} hour {int(h):02d}"]
        if alpha is not None:
            chunks.append(f"alpha {alpha:+.3f}%")
        if n is not None:
            chunks.append(f"n={int(n)}")
        return " ".join(chunks)

    parts: list[str] = []
    best = _hour_phrase(rep.get("best_hour"), "best")
    if best:
        parts.append(best)
    worst = _hour_phrase(rep.get("worst_hour"), "worst")
    if worst:
        parts.append(worst)
    spread = _num(rep.get("alpha_spread_pp"))
    if spread is not None:
        parts.append(f"spread {spread:.2f}pp")
    n_alpha = _num(rep.get("n_alpha_samples"))
    if n_alpha is not None:
        parts.append(f"n_alpha={int(n_alpha)}")
    if parts:
        lines.append("  " + " | ".join(parts))

    return lines


def _weekday_pnl_fingerprint_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/weekday-pnl-fingerprint` (the per-weekday
    alpha-vs-SPY fingerprint) into compact chat-context lines.

    Companion to ``_hourly_pnl_fingerprint_chat_lines`` — the chat carries
    ~50 paper-trader analytics blocks but nothing answers "is TODAY
    historically a good day for this bot vs SPY?". A live trader with a
    WEEKDAY_EDGE on Wed and a -0.13pp drag on Fri should be sizing
    differently on those days; the chat can answer "should I be more
    cautious today?" only when this DOW verdict is in the prompt. A
    FLAT_WEEK book has no discernible weekday edge — the silence
    precedent is correct, don't fill the chat with FLAT readings.

    SSOT (paper-trader invariant #10): the builder's own top-level
    ``headline`` is the chat headline — no chat-side re-derivation. The
    detail line restates only the builder's own ``best_weekday`` /
    ``worst_weekday`` / ``alpha_spread_pp`` / ``n_alpha_samples`` fields.

    Pure / total — exactly the ``_hourly_pnl_fingerprint_chat_lines``
    contract:

    - non-dict → ``[]``
    - top-level ``verdict`` not in {"WEEKDAY_EDGE", "WEEKEND_EDGE"} →
      ``[]``: FLAT_WEEK / INSUFFICIENT_DATA / NO_SPY_DATA / ERROR
      collapse to silence (the hourly precedent — never chat filler
      when the fingerprint is flat or the sample is too small).
    - actionable → builder's verbatim ``headline`` + detail-line.
      Missing fields degrade silently.
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("verdict") not in ("WEEKDAY_EDGE", "WEEKEND_EDGE"):
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    def _wd_phrase(wd, suffix):
        if not isinstance(wd, dict):
            return None
        name = wd.get("weekday_name")
        if not isinstance(name, str) or not name.strip():
            return None
        alpha = _num(wd.get("mean_alpha_pct"))
        n = _num(wd.get("n_alpha_samples"))
        chunks = [f"{suffix} {name}"]
        if alpha is not None:
            chunks.append(f"alpha {alpha:+.3f}%")
        if n is not None:
            chunks.append(f"n={int(n)}")
        return " ".join(chunks)

    parts: list[str] = []
    best = _wd_phrase(rep.get("best_weekday"), "best")
    if best:
        parts.append(best)
    worst = _wd_phrase(rep.get("worst_weekday"), "worst")
    if worst:
        parts.append(worst)
    spread = _num(rep.get("alpha_spread_pp"))
    if spread is not None:
        parts.append(f"spread {spread:.2f}pp")
    n_alpha = _num(rep.get("n_alpha_samples"))
    if n_alpha is not None:
        parts.append(f"n_alpha={int(n_alpha)}")
    if parts:
        lines.append("  " + " | ".join(parts))

    return lines


def _cash_conviction_fit_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/cash-conviction-fit` (the cash-level-vs-
    loudest-live-signal calibration check) into compact chat-context lines.

    The chat already carries `cash_pct` (point-in-time), `all_cash_streak`
    (chronic-flat duration), `cash_redeployment` (post-SELL latency), and
    `cash_drag` (SPY-benchmarked dollar). None of those answer the
    structural calibration question: "is the CURRENT cash level wrong
    given the loudest CURRENT live signal right now?". A book at 95%
    cash while ai_score 9.2 NVDA screams is structurally wrong in a way
    none of the other surfaces flag — and a book at 0% cash with the
    loudest signal only ai 5.5 is overdeployed for that conviction.

    SSOT (paper-trader invariant #10): the builder's own top-level
    ``headline`` is the chat headline — no chat-side re-derivation of
    the verdict naming or threshold logic. The detail line restates only
    the builder's own ``cash_pct`` / ``cash_usd`` / ``top_signal.ticker``
    / ``top_signal.ai_score`` / ``last_decision.verb`` / ``age_min``
    fields verbatim.

    Pure / total — exactly the ``_feed_health_chat_lines`` contract:

    - non-dict → ``[]``
    - top-level ``verdict`` not in {"IDLE_DESPITE_SURGE", "OVERDEPLOYED",
      "IDLE_LOW_CONVICTION"} → ``[]``: BALANCED / NO_DATA collapse to
      silence — the silence precedent (never chat filler when the cash
      level fits the conviction, and NO_DATA is a probe-side defect).
    - actionable → builder's verbatim ``headline`` + a detail-line
      restating book + signal + last-decision fields. Missing fields
      degrade silently.
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("verdict") not in (
        "IDLE_DESPITE_SURGE", "OVERDEPLOYED", "IDLE_LOW_CONVICTION",
    ):
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    parts: list[str] = []
    port = rep.get("portfolio")
    if isinstance(port, dict):
        cash_pct = _num(port.get("cash_pct"))
        if cash_pct is not None:
            parts.append(f"cash {cash_pct:.0f}%")
        cash_usd = _num(port.get("cash_usd"))
        if cash_usd is not None:
            parts.append(f"${cash_usd:,.0f}")
    sig = rep.get("top_signal")
    if isinstance(sig, dict):
        tkr = sig.get("ticker")
        score = _num(sig.get("ai_score"))
        if isinstance(tkr, str) and tkr.strip() and score is not None:
            parts.append(f"top={tkr} ai={score:.1f}")
        elif isinstance(tkr, str) and tkr.strip():
            parts.append(f"top={tkr}")
        elif score is not None:
            parts.append(f"top_ai={score:.1f}")
    last = rep.get("last_decision")
    if isinstance(last, dict):
        verb = last.get("verb")
        age = _num(last.get("age_min"))
        if isinstance(verb, str) and verb.strip() and age is not None:
            parts.append(f"last={verb} {age:.0f}m ago")
        elif isinstance(verb, str) and verb.strip():
            parts.append(f"last={verb}")
    if parts:
        lines.append("  " + " | ".join(parts))

    return lines


def _actionable_opportunities_chat_lines(rep) -> list[str]:
    """Render paper-trader's ``/api/actionable-opportunities`` — composite
    ranker for unheld watchlist names that crosses three orthogonal SSOT
    surfaces — into compact chat-context lines.

    The other unheld-pick blocks in chat each carry ONE axis: scorer-
    opportunities = pure quant predicted return, persistent-watchlist =
    contiguous-hours of news heat, watchlist-news-silence = coverage
    inverse. None answer the analyst's most-common synthesis question:
    *"of the strong scorer picks, which one is the wire ALSO talking
    about right now?"*. A NEWS_CONFIRMED or HIGH_CONVICTION_FOUND verdict
    means the quant model AND the wire agree on a name — high-trust
    actionability that survives the disagreement no other panel can detect.

    The SCORER_BUT_NO_NEWS verdict is equally important: it documents the
    failure mode where the scorer screams STRONG_HOLD on many names but
    the wire is silent on every one (live snapshot 2026-05-27 02:49 ET:
    46 STRONG_HOLD picks, all news-cold). The analyst can answer "should
    I act?" with explicit awareness that one axis is hot and the other
    quiet — neither surface alone surfaces that asymmetry.

    SSOT (paper-trader invariant #10): the builder's own top-level
    ``headline`` is the chat headline — no chat-side re-derivation of the
    verdict naming. The detail block restates per-ticker rows in builder
    field order. ``reasons`` strings pass through verbatim — the bot's
    own quant + wire phrasing, never a chat-side paraphrase.

    Pure / total — the ``_cash_conviction_fit_chat_lines`` contract:

    - non-dict → ``[]``
    - top-level ``verdict`` not in {"HIGH_CONVICTION_FOUND",
      "NEWS_CONFIRMED", "PERSISTENT_FOLLOWUP", "SCORER_BUT_NO_NEWS",
      "NEWS_BUT_NO_SCORER"} → ``[]``: ALL_QUIET / INSUFFICIENT_DATA /
      ERROR collapse to silence (the silence-on-healthy precedent;
      INSUFFICIENT_DATA is a probe-side defect not a verdict).
    - actionable → builder's verbatim ``headline`` + up to 3 ticker
      rows with ``reasons`` verbatim. Missing fields degrade silently.
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("verdict") not in (
        "HIGH_CONVICTION_FOUND",
        "NEWS_CONFIRMED",
        "PERSISTENT_FOLLOWUP",
        "SCORER_BUT_NO_NEWS",
        "NEWS_BUT_NO_SCORER",
    ):
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    rows = rep.get("by_ticker")
    if isinstance(rows, list):
        actionable_levels = {
            "HIGH_CONVICTION", "NEWS_CONFIRMED", "PERSISTENT_FOLLOWUP",
            "SCORER_ONLY", "NEWS_ONLY",
        }
        surfaced = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("actionability") not in actionable_levels:
                continue
            tk = row.get("ticker")
            if not isinstance(tk, str) or not tk.strip():
                continue
            reasons = row.get("reasons")
            reasons_str = ""
            if isinstance(reasons, list):
                # reasons strings carry verbatim from the builder — they
                # already encode the % / × / hours numbers in compact form
                kept = [r for r in reasons if isinstance(r, str) and r.strip()]
                if kept:
                    reasons_str = "; ".join(kept[:3])
            verdict_tag = str(row.get("actionability") or "")
            if reasons_str:
                lines.append(f"  {tk} [{verdict_tag}] — {reasons_str}")
            else:
                lines.append(f"  {tk} [{verdict_tag}]")
            surfaced += 1
            if surfaced >= 3:
                break

    return lines


def _passive_signal_density_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/passive-signal-density` (the smoking-gun
    detector for "engine idle while news is loud") into compact chat lines.

    The chat already carries decision_paralysis (the FACT we are not
    deciding — by streak length) and idle_opportunity (forward signals
    we are missing right now). Neither answers the structural question
    behind a multi-hour passive run: **was the news QUIET (informed
    passive — correct silence) or LOUD (deafening silence — engine sat
    on its hands during a real news window)?** A 38-cycle passive run
    in the first regime is the bot doing its job; in the second it is a
    book-wide failure to participate. The trader endpoint already
    answers this and exposes the `DEAFENING_SILENCE` verdict — the chat
    has been blind to it.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is
    the chat headline — no chat-side re-derived verdict that could
    drift from the trader endpoint (the ``_decision_paralysis_chat_lines``
    precedent). The detail line restates the trader endpoint's *own*
    ``median_signal_count`` / ``n_passive`` / ``high_signal_threshold``
    fields — never a recomputation.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - verdict not in {"DEAFENING_SILENCE"} → ``[]``: INFORMED_PASSIVE /
      SIGNAL_RICH_PASSIVE / NO_PASSIVE_RUN / INSUFFICIENT / NO_DATA
      collapse to silence — only the LOUD-news-and-idle-engine case is
      actionable, the ``_decision_paralysis_chat_lines`` silence
      precedent (never chat filler when the engine is quiet for the
      right reason, or when there is no passive run to grade).
      SIGNAL_RICH_PASSIVE is borderline (some signal, no trade) but is
      kept silent here to mirror the trader-side Discord block contract
      (`reporter._passive_signal_density_line` — only DEAFENING_SILENCE
      ships) so the two surfaces never disagree on what is "the alert".
    - actionable (DEAFENING_SILENCE) → builder's verbatim ``headline``
      (only when a usable string) + a numeric detail line restating
      median signals / passive run length / high-signal threshold.
      Missing fields degrade silently.
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("verdict") != "DEAFENING_SILENCE":
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    median = _num(rep.get("median_signal_count"))
    n_passive = _num(rep.get("n_passive"))
    threshold = _num(rep.get("high_signal_threshold"))
    parts: list[str] = []
    if median is not None:
        parts.append(f"median {median:g} signals/cycle")
    if n_passive is not None:
        parts.append(f"{int(n_passive)} passive cycles")
    if threshold is not None:
        parts.append(f"high-signal floor >{int(threshold)}")
    if parts:
        lines.append("  " + " | ".join(parts))

    return lines


def _news_to_trade_lag_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/news-to-trade-lag` (distribution of the
    freshest plausibly-causal article's minutes-before each FILLED trade)
    into compact chat lines.

    The chat already carries decision_vapor (per-decision reasoning
    quality), exit_intent_audit (per-sell stated-reason classification),
    and trade_attribution-derived blocks. None answer the *reactivity*
    question: when a real catalyst hits the wire, does the bot act in 30
    minutes — or two hours later, by which time the leverage has bled
    and the price has moved? A book that consistently trades 2h+ behind
    the news is structurally different from one that reacts in 30min,
    and only this block surfaces that gap.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is
    the chat headline — no chat-side re-derived verdict that could
    drift from the trader endpoint. The detail line restates the
    trader's *own* ``median_lag_minutes`` / ``p75_lag_minutes`` /
    ``n_attributed`` fields — never a recomputation.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - verdict not in {"DELAYED"} → ``[]``: REACTIVE_FAST / REACTIVE
      collapse to silence (the bot is reacting on time — no
      intervention needed); NO_ATTRIBUTION / NO_DATA / ERROR also
      collapse to silence — "unmeasurable" is not an alert (the
      ``_baseline_compare_chat_lines`` INSUFFICIENT_DATA → silent-but-
      honest precedent), and a sample of 1 trade with no attributed
      news must not become chat filler.
    - actionable (DELAYED) → builder's verbatim ``headline`` (only when
      a usable string) + a percentile detail line restating
      median / p75 lag in minutes + the attributed-trade count.
      Missing fields degrade silently.
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("verdict") != "DELAYED":
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    median = _num(rep.get("median_lag_minutes"))
    p75 = _num(rep.get("p75_lag_minutes"))
    n_att = _num(rep.get("n_attributed"))
    parts: list[str] = []
    if median is not None:
        parts.append(f"median lag {median:.0f}min")
    if p75 is not None:
        parts.append(f"p75 {p75:.0f}min")
    if n_att is not None:
        parts.append(f"n={int(n_att)} attributed trades")
    if parts:
        lines.append("  " + " | ".join(parts))

    return lines


def _catalyst_expiry_chat_lines(rep) -> list[str]:
    """Render paper-trader's `/api/catalyst-expiry-skill` (per-open-position
    catalyst-class + age vs catalyst-type expiry window) into compact
    chat lines.

    The chat already carries thesis_drift (P/L-driven verdict on each
    open position) and hold_discipline (a losing position overstayed
    the desk's median losing-cut). Neither tracks the *catalyst clock*:
    a position opened on an earnings beat that has now sat for 5 days is
    on a STALE thesis even if it's still green — earnings beats price-in
    within ~2 days, so the original alpha source has decayed and the
    holding is now riding sympathy / momentum, not the named catalyst.
    Selling at a small green on a zombie thesis is rational; selling at
    -1% on an INTACT structural thesis is not. The chat had no surface
    for the catalyst-expiry distinction until now.

    SSOT (paper-trader invariant #10): the builder's own ``headline`` is
    the chat headline — no chat-side re-derived verdict that could
    drift from the trader endpoint. The detail line surfaces ONE worst
    zombie position by restating its *own* ``ticker`` / ``days_held`` /
    ``catalyst_class`` fields — never a recomputation. Worst is defined
    as the ZOMBIE position with the largest ``days_held`` (the most
    aged thesis is the most exposed to catalyst-decay).

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - verdict not in {"ZOMBIE_HOLDINGS"} → ``[]``: ALL_FRESH /
      STRUCTURAL_BOOK / MIXED_BOOK / NO_DATA collapse to silence — a
      book whose catalysts have not yet expired (or a structural book
      with no time markers, or no book at all) is not actionable, the
      ``_decision_paralysis_chat_lines`` silence precedent (never chat
      filler when the book has nothing aged out).
    - actionable (ZOMBIE_HOLDINGS) → builder's verbatim ``headline``
      (only when a usable string) + a detail line surfacing the single
      worst zombie position (largest days_held among ZOMBIE rows; ties
      broken alphabetically by ticker for stability). Missing positions
      / non-dict positions / unparseable days_held degrade silently.
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("verdict") != "ZOMBIE_HOLDINGS":
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    positions = rep.get("positions")
    if isinstance(positions, list):
        # Worst zombie = largest days_held; ties broken alphabetically by
        # ticker for stability (so the same DB yields the same chat line).
        worst: dict | None = None
        worst_days = -1.0
        worst_ticker = ""
        for p in positions:
            if not isinstance(p, dict):
                continue
            if p.get("verdict") != "ZOMBIE":
                continue
            days = _num(p.get("days_held"))
            if days is None:
                continue
            tk = str(p.get("ticker") or "")
            if (days > worst_days) or (
                days == worst_days and (worst_ticker == "" or tk < worst_ticker)
            ):
                worst = p
                worst_days = days
                worst_ticker = tk
        if worst is not None:
            tk = worst.get("ticker")
            cls = worst.get("catalyst_class")
            parts: list[str] = []
            if isinstance(tk, str) and tk:
                parts.append(tk)
            if worst_days >= 0:
                parts.append(f"{worst_days:.1f}d held")
            if isinstance(cls, str) and cls:
                parts.append(f"catalyst {cls}")
            if parts:
                lines.append("  worst zombie → " + " | ".join(parts))

    return lines


_TA_REAL_VERDICTS = ("PAYOFF_TRAP", "DISPOSITION_BLEED")


def _trade_asymmetry_chat_lines(rep) -> list[str]:
    """Render paper-trader's ``/api/trade-asymmetry`` (the payoff /
    win-rate / disposition diagnostic — are we cutting winners short
    while letting losers run?) as compact chat-context lines.

    The chat already carries ``_baseline_compare_chat_lines`` (does the
    scorer have OOS skill?), ``_realized_vs_unrealized_chat_lines``
    (banked vs paper), ``_exit_intent_audit_chat_lines`` (which intent
    bucket bleeds), and ``_kelly_sizing_chat_lines`` (sizing fit to the
    realised edge). None expose the classic disposition-effect /
    payoff-trap pathology — a high win-rate made of small wins and large
    losses (negative expectancy hiding under a positive record), or a
    winner / loser hold-time skew that says the desk is patient with
    losers and impatient with winners. ``_kelly_sizing_chat_lines``
    consumes payoff_ratio downstream but reads it as a fixed input, not
    a verdict in its own right.

    SSOT (paper-trader invariant #10): the builder's own ``headline``
    passes through UNCHANGED — no chat-side re-derived verdict that
    could drift from the trader endpoint (the
    ``_decision_paralysis_chat_lines`` precedent). The detail line
    restates the endpoint's OWN ``payoff_ratio`` /
    ``actual_win_rate_pct`` / ``breakeven_win_rate_pct`` /
    ``avg_winner_hold_days`` / ``avg_loser_hold_days`` fields — never a
    recomputation.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - ``verdict`` not in {"PAYOFF_TRAP", "DISPOSITION_BLEED"} → ``[]``:
      EDGE_POSITIVE / FLAT / null (EMERGING / NO_DATA) all collapse to
      silence — only the actionable behavioural-edge pathologies are
      surfaced (the ``_decision_paralysis_chat_lines`` silence
      precedent — a healthy or sample-thin record is not chat filler;
      the trader endpoint's own ``stable_min_round_trips=20`` gate
      already withholds the verdict label when n is too small).
    - actionable verdict → builder's verbatim ``headline`` (only when a
      usable string) + one detail line restating the numeric fields
      above; each missing numeric simply drops its own fragment, never
      raises (the ``_decision_paralysis_chat_lines`` precedent).
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("verdict") not in _TA_REAL_VERDICTS:
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)              # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    pr = _num(rep.get("payoff_ratio"))
    awr = _num(rep.get("actual_win_rate_pct"))
    bwr = _num(rep.get("breakeven_win_rate_pct"))
    wh = _num(rep.get("avg_winner_hold_days"))
    lh = _num(rep.get("avg_loser_hold_days"))

    parts: list[str] = []
    if pr is not None:
        parts.append(f"payoff {pr:.2f}")
    if awr is not None and bwr is not None:
        parts.append(f"win-rate {awr:.1f}% (need {bwr:.1f}% to break even)")
    elif awr is not None:
        parts.append(f"win-rate {awr:.1f}%")
    if wh is not None and lh is not None:
        parts.append(f"winner hold {wh:.2f}d / loser hold {lh:.2f}d")
    if parts:
        lines.append("  " + " | ".join(parts))

    return lines


def _rebuy_regret_chat_lines(rep) -> list[str]:
    """Render paper-trader's ``/api/rebuy-regret`` (sell-then-rebuy DOLLAR
    regret quantifier — did the desk save or lose money on the
    close→re-entry hop?) as compact chat-context lines.

    ``/api/reentry-velocity`` (already wired upstream) covers the cadence
    question (how fast does the bot re-enter after a sell — CHURN_RISK
    vs STABLE). This is the orthogonal DOLLAR question: a bot that
    re-enters fast at materially worse price is bleeding edge that no
    other block exposes. ``round_trip_postmortem`` grades the next
    drift post-exit (was the sell premature?); this grades the actual
    BUY that followed (sold low and bought back higher?). Sign
    convention follows the trader endpoint: ``net_regret_usd > 0``
    means LOST money on the round-trip-to-re-entry hop.

    SSOT (paper-trader invariant #10): the builder's own ``headline``
    passes through UNCHANGED — no chat-side re-derived verdict that
    could drift from the trader endpoint. The worst-ticker detail
    restates the ticker's own ``net_regret_usd`` from the endpoint's
    ``per_ticker`` array — never a recomputation.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - ``verdict`` not in {"REGRETTING"} → ``[]``: SAVINGS / NET_NEUTRAL /
      NO_DATA / NO_REBUYS / ERROR collapse to silence — only the
      money-losing case is actionable (the
      ``_decision_paralysis_chat_lines`` silence precedent — a saving
      or flat re-entry record is not chat filler).
    - REGRETTING → builder's verbatim ``headline`` (only when a usable
      string) + one detail line restating the worst per-ticker entry's
      ``ticker`` / ``n_events`` / ``net_regret_usd``. Missing fields
      degrade silently.
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("verdict") != "REGRETTING":
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)              # verbatim SSOT — invariant #10

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    per_ticker = (rep.get("per_ticker")
                  if isinstance(rep.get("per_ticker"), list) else [])
    worst: dict | None = None
    worst_regret: float = float("-inf")
    for r in per_ticker:
        if not isinstance(r, dict):
            continue
        nr = _num(r.get("net_regret_usd"))
        if nr is None:
            continue
        if nr > worst_regret:
            worst_regret = nr
            worst = r
    if worst is not None and worst_regret > 0:
        tk = worst.get("ticker") or "?"
        n = worst.get("n_events")
        n_s = f" over {int(n)} event(s)" if isinstance(n, int) and n > 0 else ""
        lines.append(
            f"  worst → {tk}: ${worst_regret:,.2f} net regret{n_s}"
        )

    return lines


def _scorer_book_disagreement_chat_lines(rep) -> list[str]:
    """Render paper-trader's ``/api/disagreement`` (the scorer-vs-Opus
    per-position disagreement panel — where does the bot's own ML say
    EXIT/TRIM while Opus is still long?) as compact chat-context lines.

    The chat already carries ``_baseline_compare_chat_lines`` (does the
    bot's 17-feature DecisionScorer have OOS skill at all?) and the
    behavioural / thesis-drift / hold-discipline blocks (each open
    position graded against its own rationale or P/L). None answer the
    meta question: **does the bot's own ML — the one already trained on
    its outcomes — currently agree with the actual book?** A
    HIGH-severity row is the red flag: the live decision loop is sitting
    on a position the scorer would EXIT/TRIM. Both panels point at the
    same desk; this is the one that calls out when they disagree.

    Pre-filter: rows where ``off_distribution=True`` are dropped — per
    the trader endpoint's own docstring those are "clamped extrapolation"
    (a feature vector outside the trained distribution), not a real
    scorer/Opus fight. They surface on the dashboard for completeness
    but a chat alert from a clamped row would misrepresent the conflict.

    SSOT (paper-trader invariant #10): the surfaced row's ``ticker`` /
    ``scorer_verdict`` / ``last_action`` / ``scorer_pred_5d_pct`` all
    pass through verbatim from the trader endpoint — no chat-side
    re-derived classification. The composed headline only restates
    ``counts.HIGH`` and the worst row's own fields — never a metric the
    endpoint did not already emit. Unlike most chat-block siblings the
    trader endpoint ships no top-level ``headline`` string (per
    `disagreement_api`'s implementation), so the headline is composed
    here from endpoint-emitted counts + the worst row's verbatim
    fields rather than passed through; the composition is still SSOT
    in the sense that no NEW numeric is derived chat-side.

    Pure / total — exactly the ``_baseline_compare_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises into the chat handler)
    - ``scorer_trained=False`` (the builder's own qualification gate —
      mirrors the trader endpoint's empty-``rows`` behaviour when the
      scorer is unqualified) → ``[]``
    - zero HIGH-severity rows post-filter → ``[]``: ALIGNED-only or
      MEDIUM-only books collapse to silence (the
      ``_decision_paralysis_chat_lines`` silence precedent — never chat
      filler when the bot's ML is on board with what it's holding).
      MEDIUM-only is deliberately kept silent here: it mirrors the
      trader's own three-tier severity ladder, where MEDIUM is "mild
      discomfort" and only HIGH is "the ML is screaming"; surfacing
      MEDIUM in chat would dilute the signal.
    - ≥1 HIGH row → a composed headline restating the trader endpoint's
      own ``counts.HIGH`` + the worst row's ticker / scorer_verdict /
      last_action; one detail line restating the same row's
      ``scorer_pred_5d_pct``. Missing fields drop their own fragment,
      never raise.
    """
    if not isinstance(rep, dict):
        return []
    if not rep.get("scorer_trained"):
        return []
    rows = rep.get("rows")
    if not isinstance(rows, list):
        return []

    high = [
        r for r in rows
        if isinstance(r, dict)
        and r.get("severity") == "HIGH"
        and not r.get("off_distribution")
    ]
    if not high:
        return []

    def _num(v):
        return (v if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None)

    def _row_key(r: dict) -> tuple[float, str]:
        # Worst = most negative scorer prediction (scorer wants OUT hardest).
        # Tie-break alphabetically by ticker for deterministic chat output.
        p = _num(r.get("scorer_pred_5d_pct"))
        return (p if p is not None else 0.0, str(r.get("ticker") or ""))

    high.sort(key=_row_key)
    worst = high[0]

    counts = (rep.get("counts")
              if isinstance(rep.get("counts"), dict) else {})
    n_high_raw = counts.get("HIGH")
    n_high = (n_high_raw if isinstance(n_high_raw, int) and n_high_raw >= 0
              else len(high))

    tk = worst.get("ticker") or "?"
    verdict_label = worst.get("scorer_verdict") or "?"
    last_act = worst.get("last_action") or "?"

    plural = "" if n_high == 1 else "s"
    headline = (
        f"Scorer vs book: {n_high} HIGH-severity scorer/Opus conflict"
        f"{plural} — {tk} (scorer: {verdict_label}, last Opus action: "
        f"{last_act})."
    )
    lines = [headline]

    pred = _num(worst.get("scorer_pred_5d_pct"))
    if pred is not None:
        lines.append(
            f"  worst → {tk}: scorer 5d {pred:+.2f}% vs Opus {last_act}"
        )

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


_SECTOR_COHERENCE_MAX_CHAT_LINES = 5
_HELD_WIRE_BALANCE_MAX_CHAT_LINES = 5


def _held_wire_balance_chat_lines(rep: Any) -> list:
    """Render the per-held-ticker wire-balance report as compact chat lines.

    Silence-on-healthy in the ``_sector_coherence_chat_lines`` mould: only
    surface held names whose verdict is ``BEAR_LEAN`` — the case where the
    wire opposes the desk's long bias on a held position. ``BULL_LEAN``
    names are aligned-with-book good news (no operator action needed);
    ``MIXED`` and ``INSUFFICIENT`` collapse to silence — chat filler is
    never information.

    Pure / total — the ``_sector_coherence_chat_lines`` contract: non-dict
    / empty / no actionable names → ``[]`` (block omitted, never an
    exception into the chat handler).
    """
    if not isinstance(rep, dict):
        return []
    per = rep.get("per_ticker")
    if not isinstance(per, list) or not per:
        return []
    bear = [
        t for t in per
        if isinstance(t, dict) and t.get("verdict") == "BEAR_LEAN"
    ]
    if not bear:
        return []
    lines = []
    # The BOOK_BEAR headline is the most-actionable single line; prepend it
    # when the book itself has tipped bearish — operators want the summary
    # before the per-name detail (mirrors `_sector_coherence_chat_lines`
    # which surfaces the actionable sector verdicts directly).
    book_verdict = rep.get("book_verdict")
    if book_verdict == "BOOK_BEAR":
        headline = rep.get("headline")
        if isinstance(headline, str) and headline:
            lines.append(headline)
    for t in bear[:_HELD_WIRE_BALANCE_MAX_CHAT_LINES]:
        try:
            lead = t.get("lead_headline") or ""
            lead_cut = (lead[:120] + "…") if len(lead) > 120 else lead
            lines.append(
                f"{t.get('ticker', '?')} BEAR_LEAN "
                f"({t.get('n_bear')}↓ of {t.get('n_classified')} "
                f"classified, {t.get('coherence_pct')}% coh) · {lead_cut}"
            )
        except Exception:  # noqa: BLE001 — a chat block must never raise
            continue
    return lines


def _sector_coherence_chat_lines(rep: Any) -> list:
    """Render the per-sector coherence report as compact chat lines.

    Silence-on-healthy in the ``_sector_pulse_chat_lines`` /
    ``_macro_calendar_chat_lines`` mould: only surface sectors whose
    verdict is actionable (MACRO_BULL / MACRO_BEAR / TILT_BULL / TILT_BEAR).
    SPLIT and INSUFFICIENT collapse to silence — a chat block whose only
    contribution is "we don't know" is filler, never information.

    Pure / total — the ``_tail_risk_chat_lines`` contract: non-dict /
    empty / no actionable sectors → ``[]`` (block omitted, never an
    exception into the chat handler).
    """
    if not isinstance(rep, dict):
        return []
    sectors = rep.get("sectors")
    if not isinstance(sectors, list) or not sectors:
        return []
    actionable = [
        s for s in sectors
        if isinstance(s, dict)
        and s.get("verdict") in
        ("MACRO_BULL", "MACRO_BEAR", "TILT_BULL", "TILT_BEAR")
    ]
    if not actionable:
        return []
    lines = []
    for s in actionable[:_SECTOR_COHERENCE_MAX_CHAT_LINES]:
        try:
            lead = s.get("lead_headline") or ""
            lead_cut = (lead[:120] + "…") if len(lead) > 120 else lead
            lines.append(
                f"{s.get('sector', '?')} {s.get('verdict')} "
                f"({s.get('n_bull')}↑/{s.get('n_bear')}↓ of "
                f"{s.get('n_classified')} classified, "
                f"{s.get('coherence_pct')}% coh) · {lead_cut}"
            )
        except Exception:  # noqa: BLE001 — a chat block must never raise
            continue
    return lines


def _publish_lag_chat_lines(rep: Any) -> list[str]:
    """Render the ``/api/publish-lag`` envelope as compact chat-context lines.

    The companion to ``stale_source_alerter`` (which answers "is this collector
    still ingesting at all?") on the *latency* axis: when a collector DOES
    ingest, how far behind the publisher clock is it? A 30-min RSS poll reading
    publisher-dated items from 6h ago is still ingesting but is feeding
    ArticleNet stale news — items that score the same `time_sensitivity`
    weight as fresh wire copy, and end up in briefings as if they were live.
    No other chat block surfaces this latency dimension.

    SSOT (paper-trader invariant #10): the endpoint's own top-level
    ``headline`` is the chat headline. The detail lines restate the endpoint's
    own per-collector counts verbatim — never a chat-side re-derived verdict.

    Pure / total — the ``_cash_redeployment_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted, never raises)
    - verdict not in {``STALE_FEEDS``, ``MIXED``} → ``[]``: FRESH / NO_DATA /
      ERROR collapse to silence — the ``_decision_paralysis_chat_lines``
      silence precedent. A healthy ingest pipeline is filler.
    - actionable → builder's verbatim ``headline`` + one detail line composed
      from the stalest collector's own n / median / p90 fields.
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    if verdict not in ("STALE_FEEDS", "MIXED"):
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)

    stalest = rep.get("ranked_stalest")
    if isinstance(stalest, list) and stalest:
        top = stalest[0] if isinstance(stalest[0], dict) else {}
        try:
            name = str(top.get("collector", "?"))[:32]
            n = top.get("n")
            med = top.get("median_lag_min")
            p90 = top.get("p90_lag_min")
            stale60 = top.get("stale_60m_pct")
            parts: list[str] = []
            if isinstance(n, (int, float)) and not isinstance(n, bool):
                parts.append(f"n={int(n)}")
            if isinstance(med, (int, float)) and not isinstance(med, bool):
                parts.append(f"median={float(med):.1f}m")
            if isinstance(p90, (int, float)) and not isinstance(p90, bool):
                parts.append(f"p90={float(p90):.1f}m")
            if isinstance(stale60, (int, float)) and not isinstance(stale60, bool):
                parts.append(f">60m={float(stale60):.0f}%")
            if parts:
                lines.append(f"  STALE {name}: " + " | ".join(parts))
        except Exception:
            pass
    return lines


def _rotation_skill_chat_lines(rep: Any) -> list[str]:
    """Render paper-trader's ``/api/rotation-skill`` envelope as compact
    chat-context lines.

    The sibling to ``_cash_redeployment_chat_lines`` on the return-spread
    axis: cash_redeployment says HOW FAST the desk recycles freed capital;
    this one says whether the recycling was SKILLED (paired-rotation alpha
    measured at the forward horizon). A FAST_REDEPLOY desk that rotates
    DRAM → MSTR while DRAM rips and MSTR sags is fast AND lazy — both chat
    blocks together diagnose the right pathology.

    SSOT (paper-trader invariant #10): the endpoint's own ``headline`` is
    the chat headline — no chat-side re-derived verdict (the
    ``_cash_redeployment_chat_lines`` precedent).

    Pure / total — the ``_cash_redeployment_chat_lines`` contract:

    - non-dict → ``[]`` (block omitted)
    - verdict not in {``LAZY_ROTATION``, ``NET_NEGATIVE``} → ``[]``:
      SKILLED_ROTATION / NET_POSITIVE / NEUTRAL / INSUFFICIENT_DATA / ERROR
      collapse to silence — a profitable / neutral rotation cadence is the
      silence precedent. Surfacing only the actionable-bad verdicts matches
      every other ``_*_chat_lines`` helper.
    - actionable → builder's verbatim ``headline`` + one detail line composed
      from the endpoint's own ``stats`` (median, p25/p75 alpha, n_negative).
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    if verdict not in ("LAZY_ROTATION", "NET_NEGATIVE"):
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)

    def _num(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    stats = rep.get("stats") if isinstance(rep.get("stats"), dict) else {}
    median_pp = _num(stats.get("median_alpha_pp"))
    p25_pp = _num(stats.get("p25_alpha_pp"))
    p75_pp = _num(stats.get("p75_alpha_pp"))
    n_neg = _num(stats.get("n_negative"))
    n_scored = _num(stats.get("n_pairs_scored"))
    neg_pct = _num(stats.get("negative_alpha_pct"))

    parts: list[str] = []
    if median_pp is not None and p25_pp is not None and p75_pp is not None:
        parts.append(
            f"alpha p25/median/p75 = {p25_pp:+.2f}/{median_pp:+.2f}/{p75_pp:+.2f}pp"
        )
    elif median_pp is not None:
        parts.append(f"median alpha {median_pp:+.2f}pp")
    if n_neg is not None and n_scored is not None and n_scored > 0:
        if neg_pct is not None:
            parts.append(
                f"{int(n_neg)}/{int(n_scored)} rotations destroyed value "
                f"({neg_pct:.0f}%)"
            )
        else:
            parts.append(
                f"{int(n_neg)}/{int(n_scored)} rotations destroyed value"
            )
    if parts:
        lines.append("  " + " | ".join(parts))

    return lines


def _sentiment_reversal_chat_lines(rep: Any) -> list[str]:
    """Render ``/api/sentiment-reversal`` as compact chat-context lines.

    The native DI companion to NEWS SECTOR PULSE / NEWS SECTOR COHERENCE —
    those answer "where is the wire concentrated" and "do the lit sectors
    agree on direction"; this one answers the per-TICKER directional-flip
    question neither of them can: which tickers' avg ml_score has just
    *changed sign* (pos→neg or neg→pos) between the prior 2h window and the
    current 2h window, with both windows carrying enough articles to make
    the flip more than noise?

    A sentiment flip is materially different from a high score or a high
    urgency — those are LEVELS; a flip is a DIRECTIONAL change in the wire's
    own consensus, and the most actionable read on a position that has been
    sitting on a thesis the news has now turned against (or toward).

    Pure / total — the ``_sector_coherence_chat_lines`` contract:

    - non-dict / missing ``reversals`` → ``[]`` (silence)
    - zero reversals → ``[]`` (silence — never chat filler when the windows
      agree)
    - any reversals → headline ("N ticker(s) flipped sentiment in the last
      2h") plus up to ``_SENTIMENT_REVERSAL_TOP_SHOWN`` per-ticker lines
      verbatim from the builder's own ``reversals`` list (direction, prev
      avg → curr avg, article counts)
    """
    if not isinstance(rep, dict):
        return []
    reversals = rep.get("reversals")
    if not isinstance(reversals, list) or not reversals:
        return []
    n = len(reversals)
    window_h = rep.get("window_hours")
    win_s = f" {int(window_h)}h" if isinstance(window_h, (int, float)) else ""
    lines: list[str] = [
        f"{n} ticker(s) flipped sentiment in the last{win_s} window "
        f"(prev → curr avg ml_score sign change, both windows ≥ "
        f"{rep.get('min_articles_per_window', 2)} articles)."
    ]
    for r in reversals[:_SENTIMENT_REVERSAL_TOP_SHOWN]:
        if not isinstance(r, dict):
            continue
        ticker = r.get("ticker") or "?"
        direction = r.get("direction") or "?"
        try:
            ap = float(r.get("avg_prev"))
            ac = float(r.get("avg_curr"))
            d = float(r.get("delta"))
        except (TypeError, ValueError):
            continue
        n_prev = r.get("articles_prev") or 0
        n_curr = r.get("articles_curr") or 0
        lines.append(
            f"  {ticker} {direction}: prev {ap:+.2f}({n_prev}art) → "
            f"curr {ac:+.2f}({n_curr}art) Δ{d:+.2f}"
        )
    return lines


_SENTIMENT_REVERSAL_TOP_SHOWN = 6


def _ticker_score_dispersion_chat_lines(rep: Any) -> list[str]:
    """Render ``/api/ticker-score-dispersion`` as compact chat-context lines.

    Complements ``_sentiment_reversal_chat_lines``: reversal asks "did the
    direction flip across 2h windows?" (cross-window), this one asks "are
    the articles WITHIN the current window agreeing or disagreeing on this
    ticker?" (intra-window). A ticker with five articles all scoring 7.5–8.0
    is consensus; a ticker with the same mean spread 1.0–9.5 is contested —
    structurally different signals, identical to every other surface.

    Pure / total — the ``_sentiment_reversal_chat_lines`` contract:

    - non-dict → ``[]`` (silence)
    - verdict in {NO_DATA, NO_DISPERSION, CONSENSUS} → ``[]`` (silence — a
      consensus wire is the silence precedent; nothing actionable)
    - verdict MIXED_BOOK / CONFLICTED_NEWS → headline + per-ticker lines
      restricted to MIXED/CONFLICTED entries (TIGHT entries collapse from the
      detail block — same rule as the helper-wide silence: only surface the
      actionable rows)
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    if verdict not in ("MIXED_BOOK", "CONFLICTED_NEWS"):
        return []
    tickers = rep.get("tickers")
    if not isinstance(tickers, list) or not tickers:
        return []
    n_conf = rep.get("n_conflicted") or 0
    n_mix = rep.get("n_mixed") or 0
    window_h = rep.get("window_hours") or "?"
    lines: list[str] = [
        f"Ticker-score dispersion {verdict} over {window_h}h: "
        f"{int(n_conf)} CONFLICTED / {int(n_mix)} MIXED "
        f"(tight std ≤ {rep.get('tight_std_threshold')}, "
        f"conflicted std > {rep.get('conflicted_std_threshold')})."
    ]
    shown = 0
    for r in tickers:
        if not isinstance(r, dict):
            continue
        v = r.get("verdict")
        if v not in ("CONFLICTED", "MIXED"):
            continue
        try:
            mean = float(r.get("mean"))
            std = float(r.get("std"))
            lo = float(r.get("min"))
            hi = float(r.get("max"))
        except (TypeError, ValueError):
            continue
        ticker = r.get("ticker") or "?"
        n = r.get("n") or 0
        lines.append(
            f"  {ticker} {v}: n={int(n)} mean {mean:+.2f} std {std:.2f} "
            f"range [{lo:+.2f}, {hi:+.2f}]"
        )
        shown += 1
        if shown >= 8:
            break
    return lines


_PORTFOLIO_SIGNALS_MAX_HEADLINES = 5

_TICKER_VELOCITY_TOP_SHOWN = 6
_TICKER_COMENTIONS_TOP_SHOWN = 6


def _ticker_velocity_chat_lines(rep: Any) -> list[str]:
    """Render ``/api/ticker-velocity`` as compact chat-context lines.

    The arrival-count axis sibling of the score-based velocity surfaces:
    sentiment-reversal asks "did the direction flip?", dispersion asks
    "do the articles agree within the window?", and *this* answers
    "which tickers' raw mention volume is accelerating right now?".

    Pure / total — the ``_sentiment_reversal_chat_lines`` contract:

    - non-dict / missing keys → ``[]`` (silence)
    - verdict in {NO_DATA, QUIET} → ``[]`` (silence — never chat filler
      when the wire is structurally flat)
    - verdict BREAKING / WARMING → headline (verbatim from the builder)
      plus up to ``_TICKER_VELOCITY_TOP_SHOWN`` per-ticker rows restricted
      to BREAKING/WARMING entries (QUIET rows drop out of the actionable
      block — same precedent as ``_ticker_score_dispersion_chat_lines``
      excluding TIGHT rows).
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    if verdict not in ("BREAKING", "WARMING"):
        return []
    tickers = rep.get("tickers")
    if not isinstance(tickers, list) or not tickers:
        return []
    headline = rep.get("headline") or f"Ticker velocity {verdict}"
    lines: list[str] = [str(headline)]
    shown = 0
    for r in tickers:
        if not isinstance(r, dict):
            continue
        v = r.get("verdict")
        if v not in ("BREAKING", "WARMING"):
            continue
        try:
            ratio = float(r.get("ratio"))
        except (TypeError, ValueError):
            continue
        ticker = r.get("ticker") or "?"
        recent = r.get("recent") or 0
        prior = r.get("prior") or 0
        age = r.get("newest_age_s")
        age_str = (
            f"newest {age:.0f}s ago"
            if isinstance(age, (int, float)) else "no hits"
        )
        lines.append(
            f"  {ticker} {v}: prior {int(prior)} → recent {int(recent)} "
            f"(ratio {ratio:.2f}, {age_str})"
        )
        shown += 1
        if shown >= _TICKER_VELOCITY_TOP_SHOWN:
            break
    return lines


def _ticker_comentions_chat_lines(rep: Any) -> list[str]:
    """Render ``/api/ticker-comentions`` as compact chat-context lines.

    The pair-axis sibling of single-ticker velocity / dispersion /
    reversal: when two tickers light up TOGETHER repeatedly, the move is
    usually a sector ETF rip / peer-readthrough / M&A pairing rather than
    a single-name story. None of the per-ticker surfaces can separate
    "NVDA velocity from idiosyncratic catalysts" from "NVDA velocity as
    part of a semis basket move" — this one does.

    Pure / total — silence-on-healthy precedent:

    - non-dict → ``[]``
    - verdict in {NO_DATA, DISCONNECTED} → ``[]``
    - verdict COUPLED_NAMES / SECTOR_BURST → headline + top pairs
    """
    if not isinstance(rep, dict):
        return []
    verdict = rep.get("verdict")
    if verdict not in ("COUPLED_NAMES", "SECTOR_BURST"):
        return []
    top = rep.get("top")
    if not isinstance(top, list) or not top:
        return []
    headline = rep.get("headline") or f"Ticker comentions {verdict}"
    lines: list[str] = [str(headline)]
    shown = 0
    for r in top:
        if not isinstance(r, dict):
            continue
        pair = r.get("pair")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            lift = float(r.get("lift"))
            co = int(r.get("co_count"))
            a_total = int(r.get("a_total"))
            b_total = int(r.get("b_total"))
        except (TypeError, ValueError):
            continue
        a, b = pair
        lines.append(
            f"  {a}+{b}: co={co} lift={lift:.2f} "
            f"(a_total={a_total} b_total={b_total})"
        )
        shown += 1
        if shown >= _TICKER_COMENTIONS_TOP_SHOWN:
            break
    return lines


def _repeat_loser_chat_lines(rep: Any) -> list[str]:
    """Render paper-trader's ``/api/repeat-loser`` (the chronic-pattern
    behavioural read — tickers where the bot has lost the last
    ``threshold`` closed round-trips in a row) into compact chat lines.

    The chat already carries loser_autopsy, winner_autopsy, trade_asymmetry,
    streak. Each grades the bot in aggregate: which class of trade loses,
    payoff-trap, current W/L run. None answer the per-NAME chronic-pattern
    question: *"have I lost the last N trips on ticker X in a row?"*. A
    book that closes the same ticker for the third loss in a row is
    grinding the same setup against the same outcome — a behavioural blind
    spot every other surface aggregates away.

    SSOT (paper-trader invariant #10): the builder's own top-level
    ``headline`` is the chat headline — no chat-side re-derivation of
    the verdict naming. The detail line restates the worst offender's
    ``ticker`` / ``current_loss_streak`` / ``current_loss_usd`` /
    ``last_loss_exit_ts`` verbatim. Threshold is restated from the
    builder's own ``threshold`` field — never hardcoded chat-side.

    Pure / total — exactly the ``_cash_conviction_fit_chat_lines``
    contract:

    - non-dict → ``[]``
    - top-level ``state`` not equal to ``"REPEAT_LOSER"`` → ``[]``: OK
      / NO_DATA collapse to silence (the silence-on-healthy precedent;
      NO_DATA is a probe-side defect, OK means no offender).
    - actionable → builder's verbatim ``headline`` + a detail line
      restating the worst offender. Missing fields degrade silently.
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("state") != "REPEAT_LOSER":
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    offenders = rep.get("offenders")
    worst = None
    if isinstance(offenders, list) and offenders:
        # Builder sorts worst-first; take the first dict offender defensively.
        for o in offenders:
            if isinstance(o, dict):
                worst = o
                break

    if worst is not None:
        parts: list[str] = []
        tkr = worst.get("ticker")
        if isinstance(tkr, str) and tkr.strip():
            parts.append(tkr.strip())
        streak = worst.get("current_loss_streak")
        if isinstance(streak, int) and not isinstance(streak, bool):
            parts.append(f"{streak}L in a row")
        loss_usd = _num(worst.get("current_loss_usd"))
        if loss_usd is not None:
            parts.append(f"${loss_usd:,.2f} bled")
        last_exit = worst.get("last_loss_exit_ts")
        if isinstance(last_exit, str) and last_exit.strip():
            parts.append(f"last exit {last_exit}")
        thr = rep.get("threshold")
        if isinstance(thr, int) and not isinstance(thr, bool):
            parts.append(f"threshold {thr}")
        if parts:
            lines.append("  " + " | ".join(parts))

    return lines


def _exit_only_streak_chat_lines(rep: Any) -> list[str]:
    """Render paper-trader's ``/api/exit-only-streak`` (consecutive SELLs
    since the last entry at the book level) into compact chat lines.

    The chat carries ``/api/streak`` (W/L on closed round-trips),
    ``/api/churn`` (re-entry cadence), and ``/api/cash-drag`` (idle-cash
    dollar cost). None surface the *trade-direction* sequence: "the last
    6 fills were all SELLs — the engine is liquidating, not running the
    strategy". A defensive-trim run preceding a market drop reads as
    DISCIPLINED on every backward-looking block; the same run preceding
    a rip reads as PANIC.  Only this block surfaces the structural fact.

    SSOT (paper-trader invariant #10): the builder's own top-level
    ``headline`` is the chat headline. The detail line restates only the
    builder's own ``exit_run_length`` / ``exit_run_tickers`` /
    ``hours_since_last_entry`` / ``most_recent_action`` fields verbatim.

    Pure / total — silence precedent:

    - non-dict → ``[]``
    - ``state`` != ``"STABLE"`` → ``[]`` (NO_DATA is a probe-side defect)
    - ``verdict`` not in {``DEFENSIVE_TRIM``, ``DEFENSIVE_LIQUIDATION``}
      → ``[]``: MOST_RECENT_IS_ENTRY collapses to silence — never chat
      filler when the newest fill is an entry.
    - actionable → builder's verbatim ``headline`` + a detail line
      restating book-level run fields. Missing fields degrade silently.
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("state") != "STABLE":
        return []
    if rep.get("verdict") not in ("DEFENSIVE_TRIM", "DEFENSIVE_LIQUIDATION"):
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    parts: list[str] = []
    run_len = rep.get("exit_run_length")
    if isinstance(run_len, int) and not isinstance(run_len, bool):
        parts.append(f"run={run_len} consec exits")
    tickers = rep.get("exit_run_tickers")
    if isinstance(tickers, list) and tickers:
        sym_list: list[str] = []
        for t in tickers:
            if isinstance(t, str) and t.strip():
                sym_list.append(t.strip())
            if len(sym_list) >= 5:
                break
        if sym_list:
            parts.append("→".join(sym_list))
    hours = _num(rep.get("hours_since_last_entry"))
    if hours is not None:
        parts.append(f"{hours:.1f}h since last entry")
    most_recent = rep.get("most_recent_action")
    if isinstance(most_recent, str) and most_recent.strip():
        parts.append(f"most recent={most_recent}")
    if parts:
        lines.append("  " + " | ".join(parts))

    return lines


def _catalyst_class_autopsy_chat_lines(rep: Any) -> list[str]:
    """Render paper-trader's ``/api/catalyst-class-autopsy`` (per-catalyst-
    class win-rate / PnL leaderboard over closed round-trips) into compact
    chat lines.

    The chat carries trade_asymmetry (payoff trap), winner_autopsy /
    loser_autopsy (entry-class breakdown), per_ticker_skill (per-name
    edge). None answer the per-CATALYST-CLASS question:
    *"which class of trade — ML_ADVISOR vs ANALYST_PT vs TECHNICALS vs
    EARNINGS_PLAY vs MACRO vs BREAKING_NEWS vs PUNDIT vs SECTOR_SYMPATHY
    vs CONCENTRATION — has biased my realised P&L up or down?"*. A book
    that wins on ML_ADVISOR trips and bleeds on EARNINGS_PLAY trips is
    structurally different from one with the reverse profile, and the
    weight-allocation recommendation is opposite.

    SSOT (paper-trader invariant #10): the builder's own top-level
    ``headline`` is the chat headline. The detail line restates only
    the builder's own ``top_biased_winner`` / ``top_biased_loser`` /
    ``biased_wr_delta_pct`` / ``pool_win_rate_pct`` fields verbatim.

    Pure / total — silence precedent:

    - non-dict → ``[]``
    - ``state`` != ``"STABLE"`` → ``[]`` (NO_DATA / EMERGING collapse:
      no class has crossed the sample-size gate so no verdict yet).
    - ``state == "STABLE"`` AND both ``top_biased_winner`` and
      ``top_biased_loser`` are falsy → ``[]``: a STABLE-but-NEUTRAL
      panel where no class has reached BIASED is correctly silent —
      the leaderboard is interesting but not actionable.
    - actionable → builder's verbatim ``headline`` + a detail line
      restating the bias fields. Missing fields degrade silently.
    """
    if not isinstance(rep, dict):
        return []
    if rep.get("state") != "STABLE":
        return []
    winner = rep.get("top_biased_winner")
    loser = rep.get("top_biased_loser")
    has_winner = isinstance(winner, str) and winner.strip()
    has_loser = isinstance(loser, str) and loser.strip()
    if not (has_winner or has_loser):
        return []

    lines: list[str] = []
    headline = rep.get("headline")
    if isinstance(headline, str) and headline.strip():
        lines.append(headline)               # verbatim SSOT — invariant #10

    def _num(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    parts: list[str] = []
    if has_winner:
        parts.append(f"winner={winner}")
    if has_loser:
        parts.append(f"loser={loser}")
    delta = _num(rep.get("biased_wr_delta_pct"))
    if delta is not None:
        parts.append(f"Δwr≥{delta:.0f}pp")
    pool = _num(rep.get("pool_win_rate_pct"))
    if pool is not None:
        parts.append(f"pool wr={pool:.1f}%")
    n_trips = rep.get("n_round_trips")
    if isinstance(n_trips, int) and not isinstance(n_trips, bool):
        parts.append(f"n={n_trips} trips")
    if parts:
        lines.append("  " + " | ".join(parts))

    return lines


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


# ── chat-suggestions builder ─────────────────────────────────────────
# Drives the 4 click-friendly question chips above the chat input on
# /intern/chat. Re-runs at most every 10 min (`_CHAT_SUGGESTIONS_TTL_S`)
# so the buttons don't flicker on every page load. Pure heuristic, no
# LLM, no full_text BLOB read — must stay <300ms even when the USB
# articles.db is under writer contention.
_CHAT_SUGGESTIONS_TTL_S = 600.0
_CHAT_SUGGESTIONS_FALLBACK = [
    "What's moving markets today?",
    "Nokia surge analysis",
    "Best opportunities right now?",
    "What should I watch in Asia overnight?",
]
_CHAT_SUGGESTIONS_FILLERS = [
    "What's hot in tech?",
    "What's happening in macro?",
    "Asia overnight setup?",
]
_CHAT_LLM_FALLBACK_MODELS = ("gpt-5.5", "claude-sonnet-4-6")
_CHAT_LLM_TIMEOUT_S = max(
    2,
    int(os.environ.get("DIGITAL_INTERN_CHAT_LLM_TIMEOUT", "4")),
)
_CHAT_DEEP_KEYWORDS = (
    "paper trader",
    "paper-trader",
    "trading bot",
    "bot",
    "decision",
    "decisions",
    "position",
    "positions",
    "cash",
    "p/l",
    "pnl",
    "why did",
    "what did",
    "diagnose",
    "diagnosis",
    "deep",
)


def _chat_needs_deep_context(message: str) -> bool:
    low = (message or "").lower()
    return any(k in low for k in _CHAT_DEEP_KEYWORDS)


def _chat_model_candidates(env: dict[str, str] | None = None) -> list[str]:
    """Ordered chat LLM backends.

    ``DIGITAL_INTERN_CHAT_MODELS`` may name a comma-separated override. If it is
    absent, keep the daemon-wide default first, then append both supported
    runtimes so a Codex outage can fall through to Claude and a Claude outage
    can fall through to Codex.
    """
    env = env or os.environ
    configured = (
        env.get("DIGITAL_INTERN_CHAT_MODELS")
        or env.get("DIGITAL_INTERN_LLM_MODEL")
        or ""
    )
    raw = [m.strip() for m in configured.split(",") if m.strip()]
    raw.extend(_CHAT_LLM_FALLBACK_MODELS)
    out: list[str] = []
    seen: set[str] = set()
    for model in raw:
        if model not in seen:
            seen.add(model)
            out.append(model)
    return out


def _call_chat_llm(prompt: str, timeout: int = 120) -> tuple[str, str, list[str]]:
    failures: list[str] = []
    for model in _chat_model_candidates():
        try:
            text = (_claude_cli_call(prompt, model=model, timeout=timeout) or "").strip()
        except Exception as exc:  # noqa: BLE001 - chat must degrade, not 500
            _logger().warning("chat llm backend %s raised: %s", model, exc)
            failures.append(model)
            continue
        if text:
            return text, model, failures
        failures.append(model)
    return "", "", failures


def _chat_backend_unavailable_response(
    user_msg: str,
    articles_ctx: list[dict],
    paper_trader_block: str,
    failed_models: list[str],
) -> str:
    lines = [
        "LLM backends are unavailable right now, so I cannot run the full chat reasoning pass.",
        "",
        "I can still read the live feed context:",
    ]
    if articles_ctx:
        for art in articles_ctx[:5]:
            title = str(art.get("title") or "").strip()
            source = str(art.get("source") or "").strip()
            score = art.get("ai_score")
            if title:
                suffix = (
                    f" ({source}, score {score:.1f})"
                    if source and isinstance(score, (int, float))
                    else (f" ({source})" if source else "")
                )
                lines.append(f"- {title[:160]}{suffix}")
    else:
        lines.append("- No fresh article context was available from the local feed.")

    pt_lines = [
        ln.strip()
        for ln in (paper_trader_block or "").splitlines()
        if ln.strip()
    ][:8]
    if pt_lines:
        lines.extend(["", "Paper trader snapshot:"])
        lines.extend(f"- {ln[:180]}" for ln in pt_lines)

    failed = ", ".join(failed_models) if failed_models else "configured models"
    lines.extend([
        "",
        f"Backend status: tried {failed}; all returned empty/unavailable.",
        "Retry in a minute. This route now tries both Codex and Claude automatically.",
    ])
    if user_msg:
        lines.append(f"Question queued context: {user_msg[:180]}")
    return "\n".join(lines)


def _build_chat_suggestions() -> list[str]:
    """Top-3 watchlist tickers in the last 24h → 3 ticker-anchored
    questions + the always-on 4th. <3 tickers → fill with static
    fillers. Any exception → caller substitutes ``_CHAT_SUGGESTIONS_FALLBACK``.
    """
    from storage.article_store import _get_db_path
    db = _get_db_path()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
    try:
        rows = conn.execute(
            "SELECT title FROM articles "
            f"WHERE first_seen >= datetime('now','-1 day') AND {_LIVE_ONLY_SQL}"
            " ORDER BY first_seen DESC LIMIT 1000"
        ).fetchall()
    finally:
        conn.close()
    counts: dict[str, int] = {}
    for (title,) in rows:
        for tk in _extract_tickers(title):
            counts[tk] = counts.get(tk, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top = [tk for tk, _ in ranked[:3]]
    templates = [
        "Why is ${tk} moving today?",
        "What's the latest on ${tk}?",
        "${tk} — buy, hold, or trim?",
    ]
    out: list[str] = []
    for tk, tpl in zip(top, templates):
        out.append(tpl.replace("${tk}", tk))
    if len(out) < 3:
        for f in _CHAT_SUGGESTIONS_FILLERS:
            if len(out) >= 3:
                break
            out.append(f)
    out.append("What's moving markets today?")
    # Hard ≤60-char cap defensively (templates + watchlist tickers are
    # already short, but never trust the wire).
    return [s if len(s) <= 60 else s[:60] for s in out[:4]]


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
        return jsonify(_ttl_cache(
            f"articles:{limit}:{min_score:.3f}",
            5.0,
            lambda: _articles_from_db(limit, min_score),
        ))

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

    def _proxy_paper_trader(path: str, timeout: float = 2.5):
        import urllib.error as _urlerror
        import urllib.request as _urllib
        try:
            with _urllib.urlopen(
                f"http://127.0.0.1:8090{path}",
                timeout=timeout,
            ) as resp:
                body = resp.read()
                content_type = resp.headers.get(
                    "Content-Type", "application/json")
                return Response(
                    body,
                    status=resp.status,
                    mimetype=content_type.split(";", 1)[0],
                )
        except _urlerror.HTTPError as e:
            body = e.read() or b'{"error":"paper trader error"}'
            return Response(
                body,
                status=e.code,
                mimetype="application/json",
            )
        except Exception as e:
            return jsonify({"error": f"paper trader unreachable: {e}"}), 503

    @app.get("/trader/api/portfolio")
    def proxy_trader_portfolio():
        return _proxy_paper_trader("/api/portfolio")

    @app.get("/trader/api/equity-tail")
    def proxy_trader_equity_tail():
        qs = request.query_string.decode("utf-8", errors="ignore")
        path = "/api/equity-tail" + (f"?{qs}" if qs else "")
        return _proxy_paper_trader(path, timeout=5.0)

    @app.get("/trader/api/source-edge")
    def proxy_trader_source_edge():
        return _proxy_paper_trader("/api/source-edge", timeout=4.0)

    @app.get("/api/stats")
    def api_stats():
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        store = _store_handle()

        def _build_stats():
            if store is None:
                return _stats_from_db()
            # ``store.stats()`` is the core gauge (total/urgent/backlog). If it
            # exhausts the @_retry_on_lock budget under a sustained writer-
            # contention / shared-conn cursor-collision storm it genuinely means
            # the store is unreachable this instant → 500.
            s = dict(store.stats())
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
            return s

        # ``store.stats()`` is the core gauge (total/urgent/backlog). If it
        # exhausts the @_retry_on_lock budget under a sustained writer-
        # contention / shared-conn cursor-collision storm it genuinely means
        # the store is unreachable this instant → 500.
        try:
            s = _ttl_cache("stats", 5.0, _build_stats)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
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
        return jsonify(_ttl_cache(
            "briefings:10",
            30.0,
            lambda: _briefings_from_log(10),
        ))

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

    def _ro_conn(timeout: float = 5.0, use_store: bool = True):
        """Open a fresh read-only sqlite connection to the daemon's articles.db.

        Mirrors the resolution logic used in /api/chat: prefer the path the
        ArticleStore actually opened, then fall back to USB and local repo
        paths. Returns ``None`` if no DB can be located.
        """
        db_path: Path | None = None
        store = _store_handle() if use_store else None
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
            return sqlite3.connect(uri, uri=True, timeout=timeout)
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

    @app.get("/api/sector-coherence")
    def api_sector_coherence():
        """Per-sector bullish/bearish coherence — the structural companion
        to ``/api/sector-pulse``. PULSE answers "where is the wire
        concentrated?"; COHERENCE answers "is the concentration agreeing
        on a direction (macro story, sector-wide positioning is the trade)
        or split (idiosyncratic catalysts, name-level only)?". Pure SQL
        over the same live-only article rows; honours ``_LIVE_ONLY_SQL``
        so backtest-injected rows are excluded. ``?hours=`` clamped 1..168.
        Observational only — never gates Opus.
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
        try:
            from analysis.sector_coherence import build_sector_coherence
            return jsonify(build_sector_coherence(arts, window_hours=hours))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/api/held-wire-balance")
    def api_held_wire_balance():
        """Per-held-ticker bull/bear coherence — the per-name companion to
        ``/api/sector-coherence``. SECTOR-COHERENCE answers "is the wire
        agreeing on a direction at the sector level?"; HELD-WIRE-BALANCE
        answers the per-name follow-up: "is the wire on *my specific held
        names* aligned with my long bias?".

        Pure SQL over the same live-only article rows; honours
        ``_LIVE_ONLY_SQL`` so backtest-injected rows are excluded.
        Held universe is ``ml.features.LIVE_PORTFOLIO_TICKERS`` (matches
        ``/api/held-news-silence``). ``?hours=`` clamped 1..168 (default 24).
        Observational only — never gates Opus.
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
                "SELECT title, ai_score, first_seen FROM articles "
                f"WHERE first_seen >= ? AND {_LIVE_ONLY_SQL} "
                "ORDER BY first_seen DESC LIMIT 4000",
                (since,),
            )
        except sqlite3.Error:
            rows = []
        arts = [
            {"title": r[0] or "", "ai_score": float(r[1] or 0),
             "first_seen": r[2]}
            for r in rows
        ]
        try:
            from analysis.held_wire_balance import build_held_wire_balance
            return jsonify(build_held_wire_balance(
                arts, window_hours=hours))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/api/wire-stance")
    def api_wire_stance():
        """Per-ticker bull/bear wire stance on an ARBITRARY ticker
        list — the cross-validation companion to ``/api/held-wire-balance``.

        ``/api/held-wire-balance`` is locked to the held book (the
        ``LIVE_PORTFOLIO_TICKERS`` universe) — it answers "is the wire
        bearish on a name I'm long?". The trader's *next* decision
        looks at scorer-driven candidates the desk has NOT yet bought
        (e.g. the deployment_plan ranks MUU + KLAC for the next BUY).
        Those names are not in the held book, so they don't appear in
        ``/api/held-wire-balance``.

        This endpoint accepts ``?tickers=MUU,KLAC,MRVL`` and returns
        the same per-name bull/bear coherence verdict over that
        arbitrary universe. SSOT classifier — taxonomy is shared with
        ``/api/held-wire-balance`` (and through it,
        ``/api/sector-coherence``), so a name's bull/bear verdict is
        identical regardless of which lens called it.

        Use cases:

          * Cross-validate a scorer-driven candidate set against the
            wire's directional read before fanning out cash.
          * Read the wire on a single watchlist ticker without
            standing it up as a held position first.

        Query params:

          * ``tickers`` — comma-separated, REQUIRED (1..40 names).
            Empty / missing returns the empty-skeleton with
            ``BOOK_INSUFFICIENT``.
          * ``hours`` — lookback window, clamped 1..168 (default 24).

        Pure SQL over the live-only article rows; honours
        ``_LIVE_ONLY_SQL`` so backtest-injected rows are excluded.
        Observational only — never gates Opus.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            hours = int(request.args.get("hours", 24))
        except (TypeError, ValueError):
            hours = 24
        hours = max(1, min(168, hours))

        raw = request.args.get("tickers") or ""
        # Split + cap at 40 names to bound the SQL regex and avoid an
        # operator pasting their entire watchlist into one URL — the
        # held-book report has ~13 names total, 40 is comfortable
        # headroom for plan-side + watchlist-side joint queries.
        tickers_in = [
            s.strip() for s in raw.split(",") if s.strip()
        ][:40]

        since = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()
        try:
            rows = _ro_query(
                "SELECT title, ai_score, first_seen FROM articles "
                f"WHERE first_seen >= ? AND {_LIVE_ONLY_SQL} "
                "ORDER BY first_seen DESC LIMIT 4000",
                (since,),
            )
        except sqlite3.Error:
            rows = []
        arts = [
            {"title": r[0] or "", "ai_score": float(r[1] or 0),
             "first_seen": r[2]}
            for r in rows
        ]
        try:
            from analysis.wire_stance import build_wire_stance
            return jsonify(build_wire_stance(
                arts, tickers=tickers_in, window_hours=hours))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/api/sentiment-reversal")
    def api_sentiment_reversal():
        """Per-ticker sentiment-flip detector across two consecutive 2h windows.

        Identifies tickers whose average ml_score has crossed sign
        (positive → negative or vice versa) between PREV [4h, 2h ago) and
        CURR [2h ago, now), with both windows carrying at least
        ``MIN_ARTICLES`` mentions so the flip is more than single-article
        noise. Read counterpart: NEWS SECTOR PULSE / COHERENCE answer the
        sector-level question; this is the per-ticker directional-change
        view neither of them can compose.

        Pure builder lives in ``analytics.sentiment_reversal`` and is
        unit-tested independently of the DB. Honours ``_LIVE_ONLY_SQL`` —
        backtest-injected rows are excluded. Observational only.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        from analytics.sentiment_reversal import (
            build_sentiment_reversal,
            FETCH_LIMIT,
            WINDOW_HOURS,
        )
        now = datetime.now(timezone.utc)
        # Same cutoff/limit as the CLI snapshot writer — bounded by the
        # 4h prev-window so a single fetch covers both windows.
        cutoff = (now - timedelta(hours=WINDOW_HOURS * 2)).isoformat()
        try:
            rows = _ro_query(
                "SELECT first_seen, title, ml_score FROM articles "
                f"WHERE {_LIVE_ONLY_SQL} "
                "AND ml_score IS NOT NULL "
                "AND first_seen >= ? "
                "ORDER BY first_seen DESC LIMIT ?",
                (cutoff, FETCH_LIMIT),
            )
        except sqlite3.Error:
            rows = []
        articles = [
            {"first_seen": r[0], "title": r[1], "ml_score": r[2]}
            for r in rows
        ]
        return jsonify(build_sentiment_reversal(articles, now=now))

    @app.get("/api/ticker-score-dispersion")
    def api_ticker_score_dispersion():
        """Per-ticker intra-window score dispersion — are the articles on
        each ticker agreeing or disagreeing on the score within the current
        window?

        Complements ``/api/sentiment-reversal`` (cross-window directional
        flip) by surfacing the WITHIN-window consensus axis: a ticker with
        five articles all scoring 7.5–8.0 is consensus-bullish; a ticker
        with the same mean spread 1.0–9.5 is contested. Pure builder lives
        in ``analytics.ticker_score_dispersion`` and is unit-tested
        independently of the DB. ``?hours=`` clamped 1..168 (default 24).
        Honours ``_LIVE_ONLY_SQL``. Observational only.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            hours = int(request.args.get("hours", 24))
        except (TypeError, ValueError):
            hours = 24
        hours = max(1, min(168, hours))
        from analytics.ticker_score_dispersion import (
            build_ticker_score_dispersion,
            FETCH_LIMIT,
        )
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=hours)).isoformat()
        try:
            rows = _ro_query(
                "SELECT first_seen, title, ml_score FROM articles "
                f"WHERE {_LIVE_ONLY_SQL} "
                "AND ml_score IS NOT NULL "
                "AND first_seen >= ? "
                "ORDER BY first_seen DESC LIMIT ?",
                (cutoff, FETCH_LIMIT),
            )
        except sqlite3.Error:
            rows = []
        articles = [
            {"first_seen": r[0], "title": r[1], "ml_score": r[2]}
            for r in rows
        ]
        return jsonify(build_ticker_score_dispersion(
            articles, window_hours=hours, now=now))

    @app.get("/api/ticker-velocity")
    def api_ticker_velocity():
        """Top tickers by arrival-count velocity (recent vs prior window).

        Pure builder ``analytics.ticker_velocity_runner.build_ticker_velocity``;
        discovers top-N tickers by raw mention count over the full
        ``2 * window_min`` window and returns recent/prior counts, ratio,
        newest_age_s, and per-ticker BREAKING/WARMING/QUIET verdict plus a
        top-level verdict (BREAKING/WARMING/QUIET/NO_DATA).

        Arrival-axis sibling of ``/api/sentiment-reversal`` (directional
        cross-window flip) and ``/api/ticker-score-dispersion`` (intra-
        window consensus). ``?window_min=`` clamped 30..720 (default 120).
        Honours ``_LIVE_ONLY_SQL`` — backtest-injected rows excluded.
        Observational only.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            window_min = int(request.args.get("window_min", 120))
        except (TypeError, ValueError):
            window_min = 120
        window_min = max(30, min(720, window_min))
        from analytics.ticker_velocity_runner import (
            build_ticker_velocity,
            FETCH_LIMIT,
        )
        now = datetime.now(timezone.utc)
        cutoff = (
            now - timedelta(minutes=2 * window_min)
        ).isoformat()
        try:
            rows = _ro_query(
                "SELECT first_seen, title FROM articles "
                f"WHERE {_LIVE_ONLY_SQL} "
                "AND first_seen >= ? "
                "ORDER BY first_seen DESC LIMIT ?",
                (cutoff, FETCH_LIMIT),
            )
        except sqlite3.Error:
            rows = []
        articles = [
            {"first_seen": r[0], "title": r[1]} for r in rows
        ]
        return jsonify(build_ticker_velocity(
            articles, window_min=window_min, now=now))

    @app.get("/api/ticker-comentions")
    def api_ticker_comentions():
        """Top ticker pairs co-occurring in recent articles — the sector-axis
        sibling of single-ticker velocity / dispersion / reversal.

        Pure builder ``analytics.ticker_comentions.build_ticker_comentions``;
        emits pair list with co_count, solo totals, and ``lift`` (co_count
        divided by the rarer ticker's total mentions), plus a top-level
        verdict (SECTOR_BURST / COUPLED_NAMES / DISCONNECTED / NO_DATA).
        Answers "is NVDA velocity idiosyncratic or part of a semis basket
        move?" which the per-ticker surfaces cannot.

        ``?hours=`` clamped 1..24 (default 2). Honours ``_LIVE_ONLY_SQL``.
        Observational only.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            hours = int(request.args.get("hours", 2))
        except (TypeError, ValueError):
            hours = 2
        hours = max(1, min(24, hours))
        from analytics.ticker_comentions import (
            build_ticker_comentions,
            FETCH_LIMIT,
        )
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=hours)).isoformat()
        try:
            rows = _ro_query(
                "SELECT first_seen, title FROM articles "
                f"WHERE {_LIVE_ONLY_SQL} "
                "AND first_seen >= ? "
                "ORDER BY first_seen DESC LIMIT ?",
                (cutoff, FETCH_LIMIT),
            )
        except sqlite3.Error:
            rows = []
        articles = [
            {"first_seen": r[0], "title": r[1]} for r in rows
        ]
        return jsonify(build_ticker_comentions(
            articles, window_hours=hours, now=now))

    @app.get("/api/ticker-news-burst")
    def api_ticker_news_burst():
        """Per-ticker news-volume burst vs per-hour baseline for the held
        book — answers "is the wire heating up on a held / watched name
        RIGHT NOW?".

        Live evidence (2026-05-26, 1h vs 23h prior): SOXX 18×, MU 12.67×,
        QBTS 12.55×, DRAM 10.31× — none surfaced anywhere else in the
        system because every other live volume surface (ticker-velocity,
        ticker-comentions, sector-pulse) discovers tickers from the wire
        rather than taking a known universe.

        Pure builder ``analytics.ticker_news_burst_runner.build_ticker_news_burst``
        — same verdict ladder, baseline-per-h floor, and sort order as the
        in-process ``ArticleStore.ticker_news_burst`` storage method so the
        endpoint and the daemon's analytics surface cannot diverge.

        ``?window_h=`` clamped 0.25..12.0 (default 1.0).
        ``?baseline_h=`` clamped to ``[1.5 * window_h, 168]`` (default 24).
        ``?tickers=`` optional CSV; defaults to ``LIVE_PORTFOLIO_TICKERS``
        (the held + watched universe the daemon's alert ``book:`` tag and
        ``ticker_news_burst`` already use).

        Honours ``_LIVE_ONLY_SQL`` — backtest-injected rows excluded.
        Read-only — no ai_score / ml_score / score_source / urgency mutation.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            window_h = float(request.args.get("window_h", 1.0))
        except (TypeError, ValueError):
            window_h = 1.0
        window_h = max(0.25, min(12.0, window_h))
        try:
            baseline_h = float(request.args.get("baseline_h", 24.0))
        except (TypeError, ValueError):
            baseline_h = 24.0
        baseline_h = max(window_h * 1.5, min(168.0, baseline_h))

        tickers_arg = (request.args.get("tickers") or "").strip()
        if tickers_arg:
            tickers = [t.strip() for t in tickers_arg.split(",") if t.strip()]
        else:
            try:
                from ml.features import LIVE_PORTFOLIO_TICKERS
                tickers = sorted(LIVE_PORTFOLIO_TICKERS)
            except Exception:
                tickers = []

        from analytics.ticker_news_burst_runner import (
            build_ticker_news_burst,
            FETCH_LIMIT,
        )
        now = datetime.now(timezone.utc)
        cutoff = (
            now - timedelta(hours=baseline_h + window_h)
        ).isoformat()
        try:
            rows = _ro_query(
                "SELECT first_seen, title FROM articles "
                f"WHERE {_LIVE_ONLY_SQL} "
                "AND first_seen >= ? "
                "ORDER BY first_seen DESC LIMIT ?",
                (cutoff, FETCH_LIMIT),
            )
        except sqlite3.Error:
            rows = []
        articles = [{"first_seen": r[0], "title": r[1]} for r in rows]
        return jsonify(build_ticker_news_burst(
            articles,
            tickers=tickers,
            window_h=window_h,
            baseline_h=baseline_h,
            now=now,
        ))

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

    @app.get("/api/overnight-gaps")
    def api_overnight_gaps():
        """Pre-open gap-risk scan — tickers carried by urgent / high-ml_score
        news that broke during market-closed ET hours in the last 24h.

        The wire never sleeps but the tape does: a catalyst that prints at
        2 AM ET sits unpriced until 9:30. ``/api/articles`` is the raw feed
        and ``/api/breaking-confluence`` clusters intraday bursts, but
        neither isolates the *overnight* slice that actually gaps the open.
        This reuses ``analytics.overnight_gap_scanner.build_overnight_gaps``
        verbatim — the same ranking the CLI digest writes — so the live
        panel and the log file can never disagree. Pure DB read, no LLM.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        from analytics.overnight_gap_scanner import build_overnight_gaps
        try:
            rows = _ro_query(
                "SELECT first_seen, title, urgency, ml_score, source "
                "FROM articles "
                f"WHERE {_LIVE_ONLY_SQL} "
                "ORDER BY first_seen DESC LIMIT 5000",
            )
        except sqlite3.Error:
            rows = []
        return jsonify(build_overnight_gaps(rows))

    @app.get("/api/held-news-silence")
    def api_held_news_silence():
        """Per-held-ticker news-coverage audit — which names is the analyst
        flying blind on?

        The 5h Opus briefing's book-silence line answers this for the digest
        window only and goes dark on Claude-quota exhaustion. This surfaces
        the standing 1h / 6h / 24h coverage view for every book ticker as a
        live, no-LLM panel: a ``DARK`` name has zero live mentions in 24h
        (operating blind), an ``ECHO`` name has coverage from a single
        publisher only (one outlet repeating itself). Reuses
        ``analytics.held_ticker_news_silence`` verbatim — the held set is the
        canonical ``LIVE_PORTFOLIO_TICKERS`` so it never drifts from the
        briefing's ``[BOOK:]`` tag.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        from analytics.held_ticker_news_silence import (
            SCAN_WINDOW_H, build_report, compute_silence,
        )
        from ml.features import LIVE_PORTFOLIO_TICKERS
        since = (
            datetime.now(timezone.utc) - timedelta(hours=SCAN_WINDOW_H)
        ).isoformat()
        try:
            rows = _ro_query(
                "SELECT title, source, first_seen FROM articles "
                f"WHERE first_seen >= ? AND {_LIVE_ONLY_SQL}",
                (since,),
            )
        except sqlite3.Error:
            rows = []
        per_ticker = compute_silence(rows, LIVE_PORTFOLIO_TICKERS)
        return jsonify(build_report(per_ticker))

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

    @app.get("/api/publish-lag")
    def api_publish_lag():
        """Per-collector publish→first_seen latency snapshot.

        Wraps ``analytics.publish_lag_audit.compute()`` (existing builder) with
        a verdict + headline so chat + ops surfaces can read the same SSOT.
        ``collector-health`` says *whether* a collector is ingesting;
        publish-lag says *how stale the items it ingests are* — a 30-min RSS
        poll seeing publisher-dated 6h-old items reads HEALTHY on
        collector-health but is feeding ArticleNet stale news.

        Verdict ladder (most-severe-first):

          * ``STALE_FEEDS`` — stalest collector median lag > ``STALE_MEDIAN_MIN``
            (60min) AND its sample count ≥ ``MIN_OBS`` (10). The collector is
            consistently lagged enough to bleed into briefing freshness.
          * ``MIXED`` — stalest median lag > ``MIXED_MEDIAN_MIN`` (15min) but
            below STALE_MEDIAN_MIN. Some lag but not a structural failure.
          * ``FRESH`` — every reported collector's median lag ≤ MIXED_MEDIAN_MIN.
          * ``NO_DATA`` — zero parseable-lag samples OR no collector cleared
            ``MIN_PER_COLLECTOR``.

        Pure read — never raises. Observational only. The underlying
        ``publish_lag_audit.compute()`` snapshot is bounded by
        ``SCAN_LIMIT=5000`` (recent-id slice), sub-second on the USB DB.

        Query params (clamped):
          ``scan_limit`` 500..20000 default 5000.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            scan_limit = int(request.args.get("scan_limit", 5000))
        except (TypeError, ValueError):
            scan_limit = 5000
        scan_limit = max(500, min(20000, scan_limit))

        # Verdict thresholds — pinned here so chat surfaces / tests / callers
        # read the same constants. Mirror the publish_lag_audit module's
        # bucket boundaries (5m fresh / 60m stale) so the verdict + the
        # stats panel agree on what "stale" means.
        STALE_MEDIAN_MIN = 60.0
        MIXED_MEDIAN_MIN = 15.0
        MIN_OBS = 10

        try:
            from analytics.publish_lag_audit import compute as _pl_compute
            rep = _pl_compute(scan_limit=scan_limit)
        except Exception as e:
            return jsonify({
                "verdict": "ERROR",
                "headline": f"error: {e}",
                "error": str(e),
                "collectors": {},
                "ranked_freshest": [],
                "ranked_stalest": [],
            }), 500

        collectors = rep.get("collectors") if isinstance(rep.get("collectors"), dict) else {}
        if not collectors or not rep.get("rows_with_parseable_lag"):
            verdict = "NO_DATA"
            headline = (
                f"no parseable-lag samples in last {scan_limit} rows "
                f"(scanned {rep.get('scanned', 0)})"
            )
        else:
            ranked_stalest = rep.get("ranked_stalest") or []
            ranked_freshest = rep.get("ranked_freshest") or []
            stalest = ranked_stalest[0] if ranked_stalest else None
            freshest = ranked_freshest[0] if ranked_freshest else None

            stalest_med = (stalest.get("median_lag_min")
                           if isinstance(stalest, dict) else None)
            stalest_n = (stalest.get("n")
                         if isinstance(stalest, dict) else 0)

            if (isinstance(stalest_med, (int, float))
                    and stalest_med > STALE_MEDIAN_MIN
                    and isinstance(stalest_n, int) and stalest_n >= MIN_OBS):
                verdict = "STALE_FEEDS"
                headline = (
                    f"stalest: {stalest['collector']} p50="
                    f"{stalest_med:.1f}m (n={stalest_n})"
                )
                if isinstance(freshest, dict):
                    fmed = freshest.get("median_lag_min")
                    if isinstance(fmed, (int, float)):
                        headline += (
                            f"; freshest: {freshest['collector']} p50="
                            f"{fmed:.1f}m"
                        )
            elif (isinstance(stalest_med, (int, float))
                    and stalest_med > MIXED_MEDIAN_MIN):
                verdict = "MIXED"
                headline = (
                    f"mixed: stalest {stalest['collector']} p50="
                    f"{stalest_med:.1f}m; "
                    f"{len(collectors)} collectors reported"
                )
            else:
                verdict = "FRESH"
                headline = (
                    f"fresh: all {len(collectors)} reported collectors "
                    f"median ≤ {MIXED_MEDIAN_MIN:g}m"
                )

        return jsonify({
            "verdict": verdict,
            "headline": headline,
            "as_of": rep.get("generated_at"),
            "scanned": rep.get("scanned"),
            "rows_with_parseable_lag": rep.get("rows_with_parseable_lag"),
            "scan_limit": rep.get("scan_limit"),
            "collectors": rep.get("collectors"),
            "ranked_freshest": rep.get("ranked_freshest"),
            "ranked_stalest": rep.get("ranked_stalest"),
            "thresholds": {
                "stale_median_min": STALE_MEDIAN_MIN,
                "mixed_median_min": MIXED_MEDIAN_MIN,
                "min_obs": MIN_OBS,
            },
        })

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
        conn = _ro_conn(timeout=0.25, use_store=False)
        if conn is None:
            return jsonify({"sources": [], "error": "articles.db not reachable"})
        try:
            rows = conn.execute(
                "SELECT source, "
                "SUM(CASE WHEN first_seen >= datetime('now','-1 hour') THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN first_seen >= datetime('now','-2 hours') THEN 1 ELSE 0 END), "
                "COUNT(*) "
                "FROM articles "
                f"WHERE first_seen >= datetime('now','-24 hours') AND {_LIVE_ONLY_SQL} "
                "GROUP BY source"
            ).fetchall()
        finally:
            conn.close()
        out = []
        for n, h1_raw, h2_raw, h24_raw in rows:
            h1 = int(h1_raw or 0)
            h2 = int(h2_raw or 0)
            h24 = int(h24_raw or 0)
            name = n or "?"
            if h2 == 0:
                status = "stale"
            elif h1 >= 10:
                status = "active"
            elif h1 >= 1:
                status = "slow"
            else:
                status = "idle"
            out.append({
                "source": name,
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
        cached = _ttl_get("ml-status")
        if cached is not None:
            return jsonify(cached)
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
        standalone = _store_handle() is None
        conn = _ro_conn()
        if conn is not None:
            try:
                # ArticleNet trains on rows with any ML/LLM-assigned score;
                # `kw_score` is the pure-heuristic fallback we exclude here.
                # Articles scored in the past 24h are a reasonable proxy for
                # inference throughput; there is no `score_source` column in
                # this schema (see articles table definition).
                if standalone:
                    row = conn.execute(
                        "SELECT "
                        "SUM(CASE WHEN ai_score > 0 THEN 1 ELSE 0 END), "
                        "SUM(CASE WHEN urgency >= 1 THEN 1 ELSE 0 END) "
                        "FROM articles "
                        f"WHERE first_seen >= datetime('now','-24 hours') AND {_LIVE_ONLY_SQL}"
                    ).fetchone()
                    predictions_24h = int(row[0] or 0)
                    urgent_24h = int(row[1] or 0)
                else:
                    row = conn.execute(
                        "SELECT "
                        "SUM(CASE WHEN ai_score > 0 THEN 1 ELSE 0 END), "
                        "SUM(CASE WHEN ai_score > 0 "
                        f"AND first_seen >= datetime('now','-24 hours') AND {_LIVE_ONLY_SQL} "
                        "THEN 1 ELSE 0 END), "
                        "SUM(CASE WHEN urgency >= 1 "
                        f"AND first_seen >= datetime('now','-24 hours') AND {_LIVE_ONLY_SQL} "
                        "THEN 1 ELSE 0 END) "
                        "FROM articles"
                    ).fetchone()
                    training_set_size = int(row[0] or 0)
                    predictions_24h = int(row[1] or 0)
                    urgent_24h = int(row[2] or 0)
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
        payload = {
            "last_trained": last_trained,
            "training_set_size": training_set_size,
            "predictions_24h": predictions_24h,
            "urgent_24h": urgent_24h,
            "val_loss": val_loss,
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return jsonify(_ttl_set("ml-status", 60.0, payload))

    @app.get("/api/label-quality")
    def api_label_quality():
        """Composite ML training-input health view.

        Wires three previously-dark analyzer modules into one endpoint so the
        operator can answer "are the model's labels still trustworthy?" in
        one call:

          * ``ml.label_audit.audit`` — strong-pool integrity (Claude-LLM vs
            heuristic-inferred vs synthetic backtest provenance, column
            hygiene violations, reconciliation check). The single most load-
            bearing invariant of the system (CLAUDE.md §5).
          * ``ml.score_agreement.compute_agreement`` — ml_score vs ai_score
            agreement on the LLM-graded overlap. The cheap-model drift signal:
            if ArticleNet stops tracking Sonnet's judgement on items the LLM
            actually graded, the cheap model is no longer a trustworthy filter.

        Adds a single roll-up ``verdict``:
          OK         — hygiene clean AND |bias| < 1.0 AND strong-divergence < 15%
          DIVERGING  — hygiene clean BUT ml/ai disagreement is structurally large
          DIRTY      — hygiene violations present OR strong-pool buckets fail to
                       reconcile (this is the analyst's "stop trusting the model"
                       signal — surfaces immediately, never hidden behind a 2nd-
                       order metric).

        Read-only against articles.db (one ``mode=ro`` connection per call,
        WAL-isolated from the daemon's writer — adds zero lock contention).
        Returns a 200 with degraded ``status`` rather than raising on any
        partial failure (mirrors the existing `/api/ml-status` discipline).
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401

        out = {
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label_audit": None,
            "score_agreement": None,
            "verdict": "UNKNOWN",
            "errors": [],
        }

        # 1) Strong-pool integrity (label_audit) — needs a `.conn`-bearing
        # shim. The audit's _RoStore exists exactly for this case but is
        # private; build our own thin shim around _ro_conn() to keep the
        # public surface stable.
        conn = _ro_conn()
        if conn is None:
            out["errors"].append("articles_db_unreachable")
        else:
            try:
                from ml.label_audit import audit as _label_audit
                class _Shim:
                    pass
                shim = _Shim()
                shim.conn = conn
                out["label_audit"] = _label_audit(shim)
            except Exception as e:
                out["errors"].append(f"label_audit:{type(e).__name__}")
            finally:
                conn.close()

        # 2) ml_score vs ai_score agreement — operates on rows, not a
        # connection. Pull the most recent overlap window directly so the
        # endpoint never imports score_agreement's filesystem-writing run().
        conn2 = _ro_conn()
        if conn2 is not None:
            try:
                from ml.score_agreement import compute_agreement, _MIN_AI
                cur = conn2.execute(
                    "SELECT ml_score, ai_score, title, source, first_seen "
                    "FROM articles "
                    "WHERE ml_score IS NOT NULL AND ai_score >= ? "
                    f"AND {_LIVE_ONLY_SQL} "
                    "ORDER BY first_seen DESC LIMIT 20000",
                    (_MIN_AI,),
                )
                rows = [
                    {"ml_score": r[0], "ai_score": r[1],
                     "title": r[2], "source": r[3], "first_seen": r[4]}
                    for r in cur.fetchall()
                ]
                out["score_agreement"] = compute_agreement(rows)
            except Exception as e:
                out["errors"].append(f"score_agreement:{type(e).__name__}")
            finally:
                conn2.close()

        # 3) Single roll-up verdict. Hygiene takes precedence over drift —
        # a column hygiene violation or a non-reconciling pool is a code-
        # invariant break and must surface as DIRTY even if the cheap model
        # is currently agreeing with the LLM (today's clean agreement says
        # nothing about tomorrow's training run).
        la = out["label_audit"]
        sa = out["score_agreement"]
        if isinstance(la, dict) and la.get("ok") is False:
            out["verdict"] = "DIRTY"
        elif isinstance(la, dict) and isinstance(sa, dict) and sa.get("n", 0) >= 100:
            strong_pct = sa.get("strong_disagreement_pct", 0.0) or 0.0
            bias = abs(sa.get("bias_ml_minus_ai", 0.0) or 0.0)
            if strong_pct >= 15.0 or bias >= 1.0:
                out["verdict"] = "DIVERGING"
            else:
                out["verdict"] = "OK"
        elif isinstance(la, dict) and la.get("ok") is True:
            # Hygiene clean but we don't have enough overlap to judge drift.
            out["verdict"] = "OK_LOW_OVERLAP"

        return jsonify(out)

    @app.get("/api/active-learning-queue")
    def api_active_learning_queue():
        """Surface the model's most-uncertain articles for analyst review.

        The recursive labeler writes ``data/active_learning_queue.jsonl`` —
        one row per article that the MC-Dropout inference flagged as
        high-variance ("the model could not make up its mind"). Capped at
        5000 lines by the labeler. Until now, the queue was consumed only
        by the labeler itself; an operator had no way to see *what* the
        model is uncertain about.

        This endpoint returns the most-recent ``limit`` rows (default 25,
        max 100). Read-only file streaming — never raises on a missing or
        partially-written JSONL (best-effort, mirrors
        ``ml.conviction_calibration.load_outcomes`` discipline).
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            limit = max(1, min(100, int(request.args.get("limit", 25))))
        except (TypeError, ValueError):
            limit = 25

        queue_path = BASE_DIR / "data" / "active_learning_queue.jsonl"
        items: list[dict] = []
        total = 0
        if queue_path.exists():
            try:
                # Tail-read: seek the last ~64 KB × max-limit-headroom bytes
                # so we never load a 50 MB JSONL into memory for a 25-row
                # query. 8 KB/row average is a generous bound; this scans
                # comfortably under 4 MB even for limit=100.
                approx_row_bytes = 8 * 1024
                window = approx_row_bytes * (limit + 8)
                size = queue_path.stat().st_size
                with queue_path.open("rb") as fh:
                    fh.seek(max(0, size - window))
                    tail = fh.read()
                lines = tail.splitlines()
                # Drop the first line if we seeked mid-line (small risk: a
                # partial JSON header would fail json.loads — best-effort).
                if size > window and lines:
                    lines = lines[1:]
                for raw in lines:
                    try:
                        items.append(json.loads(raw.decode("utf-8", errors="replace")))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                items = items[-limit:]
                # Total row count is informative but expensive on a huge
                # file; the labeler's 5000-cap means a full scan is cheap.
                with queue_path.open("rb") as fh:
                    for total, _ in enumerate(fh, start=1):
                        pass
            except OSError:
                items = []

        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "queue_path": str(queue_path),
            "total_queued": total,
            "returned": len(items),
            "limit": limit,
            "items": list(reversed(items)),  # newest-first for analyst UX
        })

    @app.get("/api/kw-ai-divergence")
    def api_kw_ai_divergence():
        """Surface the kw vs ai divergence analyzer to the dashboard.

        The ``analytics/kw_ai_divergence.py`` module finds two regimes worth
        knowing:

          * ``false_positives`` — kw_score fired strongly (>=KW_HIGH) but
            Sonnet found no signal (ai_score <= AI_LOW). High counts from a
            source mean its vocabulary trips keywords without substance —
            ideal "which feeders to prune" input.
          * ``hidden_gems`` — Sonnet rated relevant (>=AI_HIGH) but the
            keyword filter almost missed it (kw_score < KW_LOW). Tells the
            analyst which topics the keyword dictionary under-weights.

        Until this endpoint, the analyzer's snapshot only landed in
        ``/home/zeph/logs/kw_ai_divergence.json`` — readable only via SSH.
        Same shape as the sibling ``/api/label-quality`` /
        ``/api/active-learning-queue`` "expose dark analyzer" endpoints
        added in the 2026-05-23 feature-dev pass.

        Computes on demand (bounded SCAN_LIMIT=6000 read, idx_first_seen
        served, ~100ms). Returns the analyzer's full payload verbatim plus
        the ``as_of`` timestamp this endpoint stamped — so a UI caller can
        tell whether the displayed figures are seconds old or stale. Errors
        absorbed into a 200 with an ``error`` key (mirrors the existing
        ``/api/ml-status`` / ``/api/label-quality`` discipline).
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            from analytics.kw_ai_divergence import compute as _compute
            payload = _compute()
        except Exception as e:
            return jsonify({
                "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "error": f"{type(e).__name__}:{e}",
                "false_positives": None,
                "hidden_gems": None,
            })
        # Stamp our own as_of so the UI can show "this view was computed at"
        # alongside whatever ``generated_at`` the analyzer wrote. Preserve
        # the analyzer's keys verbatim — never re-derive a verdict.
        payload["as_of"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return jsonify(payload)

    @app.get("/api/urgency-drought")
    def api_urgency_drought():
        """Surface the urgency drought monitor to the dashboard.

        The ``analytics/urgency_drought.py`` module tracks elapsed time since
        the LAST urgency>=2 (pushed) and LAST urgency>=1 (queued) live row.
        A long drought means the LLM triage / alert pipeline went silent
        (quota exhaustion, Sonnet throttle, scoring backlog) — the analyst's
        "is the standalone-push channel still alive?" view, complementary to
        the urgent_queue_health backlog view (which counts what's queued,
        not how long ago anything reached the head).

        Until this endpoint, the drought monitor's snapshot only landed in
        ``/home/zeph/logs/urgency_drought.json`` (cron-written). Same
        compute-on-demand discipline as the sibling kw_ai_divergence
        endpoint above — two indexed ``ORDER BY first_seen DESC LIMIT 1``
        reads, negligible cost. Read-only WAL connection, never contends
        with the daemon's writer.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            from analytics.urgency_drought import compute as _compute
            payload = _compute()
        except Exception as e:
            return jsonify({
                "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "error": f"{type(e).__name__}:{e}",
                "status": "unknown",
            })
        payload["as_of"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return jsonify(payload)

    @app.get("/api/volume-history")
    def api_volume_history():
        """Hourly article ingest counts for the last 24 hours, live rows only."""
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        cached = _ttl_get("volume-history")
        if cached is not None:
            return jsonify(cached)
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
        payload = {
            "hours": [{"hour": r[0], "count": int(r[1] or 0)} for r in rows],
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return jsonify(_ttl_set("volume-history", 60.0, payload))

    @app.get("/api/invariants")
    def api_invariants():
        """Backtest data-isolation status.

        Per the cross-system invariant (digital-intern CLAUDE.md §5): any
        ``backtest://`` row that has been alerted is a contamination breach —
        live alerts must only fire on live news.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        cached = _ttl_get("invariants")
        if cached is not None:
            return jsonify(cached)
        conn = _ro_conn()
        if conn is None:
            return jsonify({"backtest_isolation": "unknown", "error": "articles.db not reachable"})
        try:
            row = conn.execute(
                "SELECT "
                "SUM(CASE WHEN urgency >= 2 THEN 1 ELSE 0 END), "
                "COUNT(*) "
                "FROM articles "
                "WHERE url LIKE 'backtest://%' OR source LIKE 'backtest_%' "
                "      OR source LIKE 'opus_annotation%'"
            ).fetchone()
            breach = int(row[0] or 0)
            n_backtest_total = int(row[1] or 0)
        finally:
            conn.close()
        payload = {
            "backtest_isolation": "breach" if breach > 0 else "active",
            "breach_count": breach,
            "backtest_rows_total": n_backtest_total,
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return jsonify(_ttl_set("invariants", 60.0, payload))

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

    @app.get("/api/urgent-queue-health")
    def api_urgent_queue_health():
        """Health of the *unalerted* urgent backlog — "am I about to silently
        miss an urgent item?".

        ``urgency_label_split`` reports the calibration of urgent rows the
        alerter already SAW; this is the complement — what is still WAITING in
        the queue. A ``urgency=1`` row is "scored urgent, not yet pushed";
        once its ``first_seen`` ages past 24h the alert worker can never see
        it again and ``reap_stale_urgent`` demotes it — the push is silently
        lost, with no trace. This endpoint surfaces that backlog before the
        loss: how many urgent items are queued, how old the oldest is, how
        many are within ``near_reap_hours`` of the reap deadline, and how many
        are already ``overdue`` (push lost).

        The per-held-ticker breakdown answers the analyst's sharper question
        — "is my BOOK the thing going un-alerted?" — using the canonical
        ``LIVE_PORTFOLIO_TICKERS`` so it never drifts from the briefing's
        ``[BOOK:]`` tag or ``urgency_label_split_by_ticker``.

        Query params:
          ``reap_hours``  — reap deadline, clamped 1..168 (default 24).
          ``near_hours``  — near-reap warning band, clamped 0..reap (default 3).

        Returns (passthrough from ``ArticleStore.urgent_queue_health`` plus a
        verdict):
          ``queued`` / ``oldest_age_h`` / ``near_reap`` / ``overdue``
          ``reap_age_hours`` / ``near_reap_hours`` / ``by_ticker``
          ``status`` — "quiet" (queued==0), "ok" (no near/overdue rows),
                        "near_reap" (>=1 near the deadline, none overdue),
                        "items_lost" (>=1 overdue — urgent pushes silently
                        dropped; the analyst's worst case)
          ``as_of``  — ISO8601 UTC timestamp
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
            reap_hours = int(request.args.get("reap_hours", 24))
        except (TypeError, ValueError):
            reap_hours = 24
        reap_hours = max(1, min(168, reap_hours))
        try:
            near_hours = float(request.args.get("near_hours", 3.0))
        except (TypeError, ValueError):
            near_hours = 3.0
        near_hours = max(0.0, min(float(reap_hours), near_hours))
        try:
            from ml.features import LIVE_PORTFOLIO_TICKERS
        except Exception:
            LIVE_PORTFOLIO_TICKERS = set()
        try:
            data = store.urgent_queue_health(
                tickers=sorted(LIVE_PORTFOLIO_TICKERS),
                reap_age_hours=reap_hours,
                near_reap_hours=near_hours,
            )
        except Exception as exc:
            return jsonify(
                {"error": f"urgent_queue_health failed: {exc!s}"}
            ), 500
        # Verdict — same silence-vs-signal discipline as api_urgent_label_split:
        # a quiet/healthy queue collapses to a benign status, only a genuine
        # near-miss or a confirmed lost push escalates.
        if int(data.get("queued") or 0) == 0:
            status = "quiet"
        elif int(data.get("overdue") or 0) > 0:
            status = "items_lost"
        elif int(data.get("near_reap") or 0) > 0:
            status = "near_reap"
        else:
            status = "ok"
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

    @app.get("/api/source-urgency-yield")
    def api_source_urgency_yield():
        """Per-source urgent-yield audit — which collectors are signal vs noise.

        For each source in the last ``hours`` window, returns total /
        urgent / alerted counts plus an operator verdict
        (``NOISY`` / ``CLEAN`` / ``MIXED`` / ``QUIET`` / ``UNKNOWN``).
        Operator question: "of my 60+ collectors, which are producing
        urgent flags that all get suppressed by the alert-side gates
        before reaching Discord, and which are clean signal worth keeping?"

        Complementary to existing analytics:

        * ``ArticleStore.source_freshness`` — answers "how stale is each
          source's NEWEST article".
        * ``ArticleStore.source_throughput`` — answers "which sources are
          slowing down right now".
        * ``analytics/publish_lag_audit.py`` — per-source publication
          latency.

        None of those measures whether a source's *urgent* flags survive
        the alert-side gates (recap-template, quote-widget, low-authority,
        cross-cycle dedup, paraphrase). This route does — the
        ``suppression_rate`` field is precisely "how many urgent rows from
        this source got gate-dropped before Discord push".

        Query params (clamped):
          ``hours``       — lookback window, 1..168 (default 24)
          ``min_samples`` — verdict floor; below this a source returns
                             ``"UNKNOWN"``, 1..1000 (default 20)
          ``top_sources`` — display cap on the per-source list, 1..100
                             (default 15). The aggregate ``totals`` always
                             reflects every kept article.

        Reads articles.db via the dashboard's ``_ro_query`` short-lived
        read-only connection (same precedent as
        ``api_news_arrival_rhythm`` / ``api_score_distribution``). The
        ``_LIVE_ONLY_CLAUSE`` is applied — backtest-injected rows can
        never poison the operator panel (invariant #5 preserved).

        Pure builder (``analytics.source_urgency_yield``) handles the
        verdict policy; this route is the SQL adapter only.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            hours = int(request.args.get("hours", 24))
        except (TypeError, ValueError):
            hours = 24
        hours = max(1, min(168, hours))
        try:
            min_samples = int(request.args.get("min_samples", 20))
        except (TypeError, ValueError):
            min_samples = 20
        min_samples = max(1, min(1000, min_samples))
        try:
            top_sources = int(request.args.get("top_sources", 15))
        except (TypeError, ValueError):
            top_sources = 15
        top_sources = max(1, min(100, top_sources))

        from storage.article_store import _LIVE_ONLY_CLAUSE
        from analytics.source_urgency_yield import build_source_urgency_yield

        cutoff = (datetime.now(timezone.utc) -
                  timedelta(hours=hours)).isoformat(timespec="seconds")
        try:
            rows = _ro_query(
                f"""SELECT source, urgency, first_seen
                      FROM articles
                     WHERE {_LIVE_ONLY_CLAUSE}
                       AND first_seen >= ?
                     ORDER BY first_seen DESC
                     LIMIT 30000""",
                (cutoff,),
            )
        except sqlite3.Error as exc:
            return jsonify({"error": f"db: {exc!s}"}), 500
        arts = [{"source": r[0], "urgency": r[1], "first_seen": r[2]}
                for r in rows]
        return jsonify(build_source_urgency_yield(
            arts, hours=hours, min_samples=min_samples,
            top_sources=top_sources,
        ))

    @app.get("/api/alert-delivery-audit")
    def api_alert_delivery_audit():
        """Did the urgent rows the dashboard counts actually push to Discord?

        The ``urgent`` tile counts every ``urgency=2`` row, but the alert
        worker marks a row alerted whenever *any* defense-in-depth gate
        (synthetic / quote-widget / recap-template / low-authority /
        stale-published) absorbs it — so the tile silently conflates "the
        analyst was pushed" with "a gate quietly suppressed it". This route
        joins ``articles.db`` (urgency=2 rows) against
        ``alert_recency.db`` (signatures that actually fired to Discord) and
        partitions into ``delivered`` vs ``suppressed``, attributing each
        suppressed row to the gate that caught it.

        Operator questions answered:

        * ``delivery_rate`` — what fraction of urgency-head fires actually
          reached Discord. A persistent drop means the model is producing
          more false positives the gates are absorbing.
        * ``suppressed_by`` — which gate is doing the most work (a
          ``recap_template`` spike = a new SEO variant slipping the urgency
          head; a ``low_authority`` spike = a social-tier feed over-firing).
        * ``suppressed_llm_fraction`` — if high, gates are absorbing
          *LLM-vetted* (ground-truth) urgent rows: a calibration red flag.

        This is the dashboard surface for the chronic, repeatedly
        hand-diagnosed "alerts not firing" pain (daemon.log ``No response
        from Claude — skipping`` storms feeding an urgency=1 backlog).

        Query param:
          ``hours`` — window, floored at 0.5, ceiling = ``alert_recency``
                      TTL (``run_audit`` applies the ceiling itself because
                      a wider window would compare urgency=2 rows against an
                      already-pruned signature set and over-count
                      ``suppressed``).

        Reuses ``analytics.alert_delivery_audit.run_audit`` verbatim (the
        same dual-DB read-only shell the CLI digest uses) so the panel and
        the CLI can never disagree. Both DBs are opened ``mode=ro``; a
        missing recency DB degrades to "all suppressed", never a 500.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401

        from analytics.alert_delivery_audit import (
            run_audit, DEFAULT_WINDOW_HOURS,
        )
        try:
            hours = float(request.args.get("hours", DEFAULT_WINDOW_HOURS))
        except (TypeError, ValueError):
            hours = DEFAULT_WINDOW_HOURS
        # Floor only — run_audit applies the recency-TTL ceiling itself.
        hours = max(0.5, hours)
        try:
            return jsonify(run_audit(hours=hours))
        except Exception as exc:  # noqa: BLE001 — any DB/IO fault → 500, not crash
            return jsonify({"error": f"audit: {exc!s}"}), 500

    @app.get("/api/alert-freshness")
    def api_alert_freshness():
        """How stale were the urgent rows at the moment they were detected?

        A 🚨 BREAKING alert fired on a 3-hour-old article is nearly
        worthless — the move is already priced. Every other monitor
        (collector uptime, source throughput, urgency calibration) can read
        HEALTHY while the alerts the analyst actually gets are too old to
        act on. This route is the bottom-line freshness view: of the
        ``urgency>=1`` rows in the window, the ``published`` → ``first_seen``
        staleness distribution, aggregate and split by ``score_source``.

        Dual of ``ingestion_latency`` (all live rows, per-source —
        "is a collector slow?"): this scopes to urgent rows only and answers
        "were the alerts stale *content*?". A high ``p90_min`` /
        ``pct_over_1h`` is the quality failure the volume monitors miss.

        Query param:
          ``hours`` — lookback window, 1 .. 168 (default 24).

        Reads ``articles.db`` via the dashboard's ``_ro_query`` short-lived
        ``mode=ro`` connection (the ``news-arrival-rhythm`` /
        ``source-urgency-yield`` precedent — never competes for the daemon's
        writer lock). ``_LIVE_ONLY_CLAUSE`` is applied so backtest-injected
        rows cannot colour the latency view (invariant #5). The pure builder
        ``analytics.alert_freshness.compute_alert_freshness`` owns the clock
        parsing, percentile maths and the LLM-vs-ML split.
        """
        if not _check_api_key():
            return jsonify({"error": "unauthorized"}), 401
        try:
            hours = int(request.args.get("hours", 24))
        except (TypeError, ValueError):
            hours = 24
        hours = max(1, min(168, hours))

        from storage.article_store import _LIVE_ONLY_CLAUSE
        from analytics.alert_freshness import compute_alert_freshness

        cutoff = (datetime.now(timezone.utc) -
                  timedelta(hours=hours)).isoformat(timespec="seconds")
        try:
            rows = _ro_query(
                f"""SELECT published, first_seen, score_source, urgency
                      FROM articles
                     WHERE {_LIVE_ONLY_CLAUSE}
                       AND urgency >= 1
                       AND first_seen >= ?
                     ORDER BY first_seen DESC
                     LIMIT 100000""",
                (cutoff,),
            )
        except sqlite3.Error as exc:
            return jsonify({"error": f"db: {exc!s}"}), 500
        report = compute_alert_freshness(rows)
        report["window_hours"] = hours
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        return jsonify(report)

    @app.get("/healthz")
    @app.get("/api/health")
    def healthz():
        # /api/health is an alias for /healthz so dashboard.html's health
        # badge (and any external probe using the conventional /api/health
        # path) stops silently 404'ing. Same body either way.
        store = _store_handle()
        return jsonify({"ok": True, "store_attached": store is not None})

    @app.get("/api/chat-suggestions")
    def api_chat_suggestions():
        # Public — the chat page JS has no API key. Matches /api/chat's
        # public surface (it's intentionally ungated; the page is public).
        cached = _ttl_get("chat_suggestions")
        if cached is not None:
            return jsonify({"suggestions": cached})
        try:
            suggestions = _build_chat_suggestions()
        except Exception as exc:  # noqa: BLE001
            _logger().warning("chat_suggestions build failed: %s", exc)
            return jsonify({"suggestions": list(_CHAT_SUGGESTIONS_FALLBACK)})
        _ttl_set("chat_suggestions", _CHAT_SUGGESTIONS_TTL_S, suggestions)
        return jsonify({"suggestions": suggestions})

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
        # Companion block: sector COHERENCE — is the wire actually agreed on
        # a direction in the lit sectors, or is the concentration mostly
        # idiosyncratic (in which case sector-wide positioning is wrong)?
        # Reuses the same SQL fetch as sector-pulse so we don't double-query.
        sector_coherence_block = ""
        # Per-held-ticker companion: BEAR_LEAN on a held name = wire opposes
        # the desk's long bias. Reuses the same 24h sp_arts fetch; initialised
        # at the outer scope so the f-string reference is always bound even if
        # the sector-pulse fetch itself raises.
        held_wire_balance_block = ""
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
            sp_arts = [
                {"title": r[0] or "", "ai_score": float(r[1] or 0),
                 "urgency": int(r[2] or 0), "first_seen": r[3]}
                for r in sp_rows
            ]
            sp = _aggregate_sector_pulse(sp_arts, window_hours=24)
            sector_pulse_block = "\n".join(_sector_pulse_chat_lines(sp))
            try:
                from analysis.sector_coherence import build_sector_coherence
                sc = build_sector_coherence(sp_arts, window_hours=24)
                sector_coherence_block = "\n".join(
                    _sector_coherence_chat_lines(sc))
            except Exception as e2:  # noqa: BLE001 — never sink the chat
                _logger().warning(
                    "chat: sector-coherence build failed: %s", e2)
            # Per-held-ticker wire-balance: companion to sector-coherence,
            # reuses the same 24h article fetch. Silence-on-healthy: only
            # emits when at least one held name is BEAR_LEAN (wire opposes
            # the desk's long bias). Healthy/BULL_LEAN names produce no
            # chat line — the chat-pattern memory `silence-on-healthy`.
            try:
                from analysis.held_wire_balance import build_held_wire_balance
                hwb = build_held_wire_balance(sp_arts, window_hours=24)
                held_wire_balance_block = "\n".join(
                    _held_wire_balance_chat_lines(hwb))
            except Exception as e3:  # noqa: BLE001 — never sink the chat
                _logger().warning(
                    "chat: held-wire-balance build failed: %s", e3)
                held_wire_balance_block = ""
        except Exception as e:  # noqa: BLE001 — never sink the chat
            _logger().warning("chat: sector-pulse fetch failed: %s", e)

        # Native sentiment-reversal block — per-ticker avg ml_score
        # sign-flip across two consecutive 2h windows. The directional
        # counterpart to PULSE/COHERENCE (those answer sector-level
        # questions) and the cross-window counterpart to dispersion below
        # (which answers the intra-window consensus question). Pure DI
        # surface (no :8090 cross-fetch); honours _LIVE_ONLY_SQL. Block
        # collapses to silence when zero reversals are detected — never
        # chat filler when both windows agree.
        sentiment_reversal_block = ""
        try:
            from analytics.sentiment_reversal import (
                build_sentiment_reversal,
                FETCH_LIMIT as _SR_LIMIT,
                WINDOW_HOURS as _SR_WIN,
            )
            sr_now = datetime.now(timezone.utc)
            sr_since = (
                sr_now - timedelta(hours=_SR_WIN * 2)
            ).isoformat()
            sr_rows = _ro_query(
                "SELECT first_seen, title, ml_score FROM articles "
                "WHERE first_seen >= ? "
                "AND url NOT LIKE 'backtest://%' "
                "AND source NOT LIKE 'backtest_%' "
                "AND source NOT LIKE 'opus_annotation%' "
                "AND ml_score IS NOT NULL "
                "ORDER BY first_seen DESC LIMIT ?",
                (sr_since, _SR_LIMIT),
            )
            sr_arts = [
                {"first_seen": r[0], "title": r[1], "ml_score": r[2]}
                for r in sr_rows
            ]
            sr = build_sentiment_reversal(sr_arts, now=sr_now)
            sentiment_reversal_block = "\n".join(
                _sentiment_reversal_chat_lines(sr))
        except Exception as e:  # noqa: BLE001 — never sink the chat
            _logger().warning(
                "chat: sentiment-reversal build failed: %s", e)

        # Native ticker-score-dispersion block — per-ticker intra-window
        # std-dev of ml_score over the last 24h. Companion to
        # sentiment-reversal: reversal asks "did the direction flip
        # across windows?"; dispersion asks "are the articles within the
        # current window AGREEING or DISAGREEING on the score?". A
        # CONFLICTED ticker has news pulling in two directions at once —
        # opposite read of a TIGHT consensus, identical to every other
        # surface that only carries the mean. Block fires ONLY on
        # MIXED_BOOK / CONFLICTED_NEWS — CONSENSUS / NO_DATA collapse to
        # silence (never filler when the wire is consistent).
        ticker_score_dispersion_block = ""
        try:
            from analytics.ticker_score_dispersion import (
                build_ticker_score_dispersion,
                FETCH_LIMIT as _DISP_LIMIT,
            )
            disp_now = datetime.now(timezone.utc)
            disp_since = (disp_now - timedelta(hours=24)).isoformat()
            disp_rows = _ro_query(
                "SELECT first_seen, title, ml_score FROM articles "
                "WHERE first_seen >= ? "
                "AND url NOT LIKE 'backtest://%' "
                "AND source NOT LIKE 'backtest_%' "
                "AND source NOT LIKE 'opus_annotation%' "
                "AND ml_score IS NOT NULL "
                "ORDER BY first_seen DESC LIMIT ?",
                (disp_since, _DISP_LIMIT),
            )
            disp_arts = [
                {"first_seen": r[0], "title": r[1], "ml_score": r[2]}
                for r in disp_rows
            ]
            disp = build_ticker_score_dispersion(
                disp_arts, window_hours=24, now=disp_now)
            ticker_score_dispersion_block = "\n".join(
                _ticker_score_dispersion_chat_lines(disp))
        except Exception as e:  # noqa: BLE001 — never sink the chat
            _logger().warning(
                "chat: ticker-score-dispersion build failed: %s", e)

        # Native ticker-velocity block — top tickers' raw arrival-count
        # ratio recent vs prior. The arrival-volume axis sibling of
        # reversal (cross-window direction) and dispersion (intra-window
        # consensus). Block fires ONLY on BREAKING / WARMING — QUIET /
        # NO_DATA collapse to silence (never filler when the wire is
        # structurally flat).
        ticker_velocity_block = ""
        try:
            from analytics.ticker_velocity_runner import (
                build_ticker_velocity,
                FETCH_LIMIT as _TV_LIMIT,
                WINDOW_MIN as _TV_WIN,
            )
            tv_now = datetime.now(timezone.utc)
            tv_since = (
                tv_now - timedelta(minutes=2 * _TV_WIN)
            ).isoformat()
            tv_rows = _ro_query(
                "SELECT first_seen, title FROM articles "
                "WHERE first_seen >= ? "
                "AND url NOT LIKE 'backtest://%' "
                "AND source NOT LIKE 'backtest_%' "
                "AND source NOT LIKE 'opus_annotation%' "
                "ORDER BY first_seen DESC LIMIT ?",
                (tv_since, _TV_LIMIT),
            )
            tv_arts = [
                {"first_seen": r[0], "title": r[1]} for r in tv_rows
            ]
            tv = build_ticker_velocity(
                tv_arts, window_min=_TV_WIN, now=tv_now)
            ticker_velocity_block = "\n".join(
                _ticker_velocity_chat_lines(tv))
        except Exception as e:  # noqa: BLE001 — never sink the chat
            _logger().warning(
                "chat: ticker-velocity build failed: %s", e)

        # Native ticker-comentions block — top ticker pairs co-occurring
        # in the last 2h. The sector-axis sibling of single-ticker
        # velocity/dispersion/reversal: separates idiosyncratic-name
        # velocity from sector-basket co-movement. Block fires ONLY on
        # COUPLED_NAMES / SECTOR_BURST — DISCONNECTED / NO_DATA collapse
        # to silence (never filler when no pair is recurring).
        ticker_comentions_block = ""
        try:
            from analytics.ticker_comentions import (
                build_ticker_comentions,
                FETCH_LIMIT as _CM_LIMIT,
                WINDOW_HOURS as _CM_WIN,
            )
            cm_now = datetime.now(timezone.utc)
            cm_since = (
                cm_now - timedelta(hours=_CM_WIN)
            ).isoformat()
            cm_rows = _ro_query(
                "SELECT first_seen, title FROM articles "
                "WHERE first_seen >= ? "
                "AND url NOT LIKE 'backtest://%' "
                "AND source NOT LIKE 'backtest_%' "
                "AND source NOT LIKE 'opus_annotation%' "
                "ORDER BY first_seen DESC LIMIT ?",
                (cm_since, _CM_LIMIT),
            )
            cm_arts = [
                {"first_seen": r[0], "title": r[1]} for r in cm_rows
            ]
            cm = build_ticker_comentions(
                cm_arts, window_hours=_CM_WIN, now=cm_now)
            ticker_comentions_block = "\n".join(
                _ticker_comentions_chat_lines(cm))
        except Exception as e:  # noqa: BLE001 — never sink the chat
            _logger().warning(
                "chat: ticker-comentions build failed: %s", e)

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

        deep_chat = bool(payload.get("deep")) or _chat_needs_deep_context(user_msg)
        if not deep_chat:
            now_iso = datetime.now(timezone.utc).isoformat()
            fast_prompt = (
                "You are a market intelligence analyst with access to a live "
                "news feed and the user's portfolio snapshot.\n"
                f"Current date: {now_iso}\n\n"
                "TOP NEWS SIGNALS (last 6h, recency-decayed ML ranking):\n"
                f"{articles_block}\n\n"
                + (
                    "THESIS CONTEXT (last 48h, deduped vs 6h set):\n"
                    f"{thesis_block}\n\n"
                    if thesis_block else ""
                )
                + (
                    "NEWS COVERAGE GAP:\n"
                    f"{coverage_gap_block}\n\n"
                    if coverage_gap_block else ""
                )
                + "USER'S REAL PORTFOLIO SNAPSHOT:\n"
                f"{portfolio_block}\n\n"
                + (
                    "NEWS SECTOR PULSE:\n"
                    f"{sector_pulse_block}\n\n"
                    if sector_pulse_block else ""
                )
                + (
                    "NEWS SECTOR COHERENCE:\n"
                    f"{sector_coherence_block}\n\n"
                    if sector_coherence_block else ""
                )
                + (
                    "NEWS SENTIMENT REVERSAL:\n"
                    f"{sentiment_reversal_block}\n\n"
                    if sentiment_reversal_block else ""
                )
                + (
                    "NEWS TICKER VELOCITY:\n"
                    f"{ticker_velocity_block}\n\n"
                    if ticker_velocity_block else ""
                )
                + (
                    "NEWS TICKER COMENTIONS:\n"
                    f"{ticker_comentions_block}\n\n"
                    if ticker_comentions_block else ""
                )
                + "Answer concisely and data-driven. If the user asks for "
                "paper-trader internals, tell them to ask for a deep bot "
                "diagnosis so the heavy trader diagnostics are fetched."
            )
            msgs: list[dict] = []
            for h in history[-20:]:
                role = h.get("role")
                content = (h.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    msgs.append({"role": role, "content": content})
            msgs.append({"role": "user", "content": user_msg})

            convo_parts = [fast_prompt, "\n\n--- Conversation ---"]
            for m in msgs:
                convo_parts.append(f"{m['role'].upper()}: {m['content']}")
            convo_parts.append("ASSISTANT:")
            prompt = "\n\n".join(convo_parts)

            response_text, llm_model, failed_models = _call_chat_llm(
                prompt, timeout=_CHAT_LLM_TIMEOUT_S
            )
            if not response_text:
                _logger().warning(
                    "chat fast-path: all LLM backends unavailable; tried=%s",
                    ",".join(failed_models) or "<none>",
                )
                return jsonify({
                    "response": _chat_backend_unavailable_response(
                        user_msg, articles_ctx, "", failed_models
                    ),
                    "sources": [a["title"] for a in articles_ctx],
                    "degraded": True,
                    "failed_models": failed_models,
                    "mode": "fast",
                })

            return jsonify({
                "response": response_text,
                "sources": [a["title"] for a in articles_ctx],
                "model": llm_model,
                "mode": "fast",
            })

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

        # News-source edge — which of digital-intern's ~17 collectors actually
        # precede SPY-abnormal moves vs which are wire-noise? ML-gate-honesty
        # above grades the GATE; this is its read-COLLECTOR companion: an
        # analyst answering "should I trust this MarketWatch headline?" or
        # "which sources actually move the tape?" has no other surface to
        # compose this from — the per-source verdict only lives in the trader
        # endpoint and the JS-only se-card dashboard panel. Composed verbatim
        # by the pure _news_source_edge_chat_lines helper (unit-tested; SSOT
        # — no re-derived verdict). Guarded 3s sub-fetch like every sibling;
        # a stale/absent trader silently omits the block.
        news_source_edge_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/source-edge",
                    timeout=3) as resp:
                _se = json.loads(resp.read().decode("utf-8"))
            news_source_edge_block = "\n".join(
                _news_source_edge_chat_lines(_se))
        except Exception as e:
            _logger().warning("chat: source-edge fetch failed: %s", e)

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

        # Intent-followthrough — the say-do gap. /api/decision-conditionals
        # lists the STANDING intents; this block grades whether they were
        # actually executed. A bot that emits crisp "wait for X, then buy
        # Y" statements but never executes Y has perfect decision-vapor
        # specificity and zero followthrough; only this block catches it.
        # Fires ONLY on DRIFTING / ABANDONED (DISCIPLINED / NO_DATA /
        # NO_RESOLVED / ERROR collapse to silence — the silence precedent,
        # never chat filler when the bot is following through). Composed
        # verbatim by the pure _intent_followthrough_chat_lines helper
        # (unit-tested; SSOT — the builder's own `headline` is the chat
        # headline, no re-derived verdict). Guarded 3s read; appears once
        # :8090 is restarted onto /api/intent-followthrough.
        intent_followthrough_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/intent-followthrough",
                    timeout=3) as resp:
                _ift = json.loads(resp.read().decode("utf-8"))
            intent_followthrough_block = "\n".join(
                _intent_followthrough_chat_lines(_ift))
        except Exception as e:
            _logger().warning(
                "chat: intent-followthrough fetch failed: %s", e)

        # Opportunity-cost — the hindsight read on past sit-outs. The chat
        # already carries idle-opportunity (current drought) and cash_pct
        # (snapshot) but neither answers "did past HOLD-CASH calls actually
        # save money?". A persistent MISSED_ALPHA verdict says cash
        # discipline is COSTING alpha; a persistent DEFENSIVE_WIN says it's
        # SAVING the book. Fires ONLY on MISSED_ALPHA / DEFENSIVE_WIN
        # (NEUTRAL / NO_DATA / ERROR collapse to silence — the silence
        # precedent, never chat filler when sit-outs are neutral). Composed
        # verbatim by the pure _opportunity_cost_chat_lines helper
        # (unit-tested; SSOT — the builder's own `headline` is the chat
        # headline, no re-derived verdict). Guarded 3s read; appears once
        # :8090 is restarted onto /api/opportunity-cost.
        opportunity_cost_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/opportunity-cost",
                    timeout=3) as resp:
                _oc = json.loads(resp.read().decode("utf-8"))
            opportunity_cost_block = "\n".join(
                _opportunity_cost_chat_lines(_oc))
        except Exception as e:
            _logger().warning(
                "chat: opportunity-cost fetch failed: %s", e)

        # Idle-opportunity — the RIGHT-NOW companion to opportunity-cost.
        # opportunity-cost grades PAST sit-outs by forward return (hindsight);
        # idle-opportunity grades the CURRENT drought by what's arriving on
        # the watchlist while the bot is dark. A decision-storm with the
        # tape running is structurally different from a decision-storm with
        # the tape quiet, and that difference is only visible live —
        # opportunity-cost can't see it for hours yet. Composed verbatim by
        # the pure _idle_opportunity_chat_lines helper (unit-tested; SSOT —
        # the builder's own `headline` is the chat headline, no re-derived
        # verdict). Guarded 3s read; NO_DATA / NO_DROUGHT / OK-with-zero
        # silently omit the block (the _decision_paralysis_chat_lines
        # silence precedent — never chat filler when the loop is filling
        # or nothing was actually missed).
        idle_opportunity_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/idle-opportunity",
                    timeout=3) as resp:
                _io = json.loads(resp.read().decode("utf-8"))
            idle_opportunity_block = "\n".join(
                _idle_opportunity_chat_lines(_io))
        except Exception as e:
            _logger().warning(
                "chat: idle-opportunity fetch failed: %s", e)

        # Persona-book-fit — the structural "does the live book mirror a
        # backtest persona rated DRAG by the leaderboard?" question. Every
        # other block analyses position-by-position fitness; none surface
        # the whole-book archetype-overlap angle. Composed verbatim by the
        # pure _persona_book_fit_chat_lines helper (unit-tested; SSOT —
        # the builder's own headline is the chat headline, no re-derived
        # verdict). Guarded 3s read; ALIGNED_EDGE / ALIGNED_FLAT / NO_BOOK
        # / WEAK_OVERLAP / INSUFFICIENT_PERSONA payloads silently omit the
        # block (the _decision_paralysis_chat_lines silence precedent —
        # never chat filler when the book is well-aligned). Only appears
        # once :8090 is restarted onto /api/persona-book-fit.
        persona_book_fit_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/persona-book-fit",
                    timeout=3) as resp:
                _pbf = json.loads(resp.read().decode("utf-8"))
            persona_book_fit_block = "\n".join(
                _persona_book_fit_chat_lines(_pbf))
        except Exception as e:
            _logger().warning(
                "chat: persona-book-fit fetch failed: %s", e)

        # Inverse-pair-conflict — the leveraged carry-waste structural
        # block. When the bot holds TQQQ AND SQQQ simultaneously the
        # directional exposure cancels but both sleeves keep paying
        # leverage decay (etf-lookthrough reports the net outcome but
        # not the carry-waste fact; correlation-cluster catches positive
        # correlation only). Fires ONLY on CARRY_WASTE; CLEAN / NO_BOOK
        # / OPPOSING_UNLEVERED collapse to silence (the silence
        # precedent — never chat filler when the book is one-sided).
        # Headline + worst-family details carry verbatim. Only appears
        # once :8090 restarts onto /api/inverse-pair-conflict-skill.
        inverse_pair_conflict_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/inverse-pair-conflict-skill",
                    timeout=3) as resp:
                _ipc = json.loads(resp.read().decode("utf-8"))
            inverse_pair_conflict_block = "\n".join(
                _inverse_pair_conflict_chat_lines(_ipc))
        except Exception as e:
            _logger().warning(
                "chat: inverse-pair-conflict fetch failed: %s", e)

        # Watchlist news silence — the universe-blind-spot block. Of the
        # ~47 tickers Opus may choose from, how many had ZERO live
        # articles in the last 24h, and which are mention-storming?
        # Complements digital-intern's own /api/held-news-silence
        # (held-only) by surfacing the UNIVERSE-wide coverage map. Fires
        # ONLY on BLIND_UNIVERSE / SPARSE_COVERAGE; WELL_COVERED /
        # NO_DATA collapse to silence. Headline + silent / hot lists
        # carry verbatim. Only appears once :8090 restarts onto
        # /api/watchlist-news-silence-skill.
        watchlist_news_silence_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/watchlist-news-silence-skill",
                    timeout=3) as resp:
                _wns = json.loads(resp.read().decode("utf-8"))
            watchlist_news_silence_block = "\n".join(
                _watchlist_news_silence_chat_lines(_wns))
        except Exception as e:
            _logger().warning(
                "chat: watchlist-news-silence fetch failed: %s", e)

        # Concurrent-Opus attribution — the host-saturation parent-tree
        # breakdown. /api/host-guard reports the COUNT of concurrent Opus
        # but not WHICH parent trees own them. Without that the operator
        # either kills indiscriminately (also nukes the legitimate runner
        # Opus) or waits hours for the storm to clear. Fires ONLY on
        # ELEVATED / SATURATED (NO_OPUS / CLEAN / BENIGN collapse to
        # silence — the silence precedent, never chat filler when the
        # host is within host_guard's own threshold). Headline +
        # recommendation carry verbatim from the trader endpoint —
        # restate, never re-derive. Only appears once :8090 restarts onto
        # /api/concurrent-opus-attribution.
        concurrent_opus_attribution_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/concurrent-opus-attribution",
                    timeout=3) as resp:
                _coa = json.loads(resp.read().decode("utf-8"))
            concurrent_opus_attribution_block = "\n".join(
                _concurrent_opus_attribution_chat_lines(_coa))
        except Exception as e:
            _logger().warning(
                "chat: concurrent-opus-attribution fetch failed: %s", e)

        # Post-SELL cash-redeployment latency — the sold-then-sat pathology.
        # /api/risk reports cash_pct as a *snapshot*, but a book that sells
        # into a thesis weakening then sits for 5 days has the same headline
        # cash% as one that redeploys in 6h. The chat block fires ONLY on
        # SLOW / STALLED (FAST_REDEPLOY, STEADY, NO_DATA collapse to silence
        # — the _decision_paralysis_chat_lines silence precedent). Composed
        # verbatim by the pure _cash_redeployment_chat_lines helper
        # (unit-tested; SSOT — the builder's own `headline` is the chat
        # headline, no re-derived verdict). Guarded 3s read; only appears
        # once :8090 is restarted onto /api/cash-redeployment-latency-skill.
        cash_redeployment_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/cash-redeployment-latency-skill",
                    timeout=3) as resp:
                _crl = json.loads(resp.read().decode("utf-8"))
            cash_redeployment_block = "\n".join(
                _cash_redeployment_chat_lines(_crl))
        except Exception as e:
            _logger().warning(
                "chat: cash-redeployment-latency fetch failed: %s", e)

        # Decision-vapor (reasoning grounded-ness) — does Opus cite specific
        # numbers + catalysts + tickers, or is FILLED reasoning generic
        # "strong setup, building position" vapor? A vapor trade that fails
        # has nothing for the next decision to learn from. Block fires ONLY
        # on MIXED / VAPOR_DECISIONS (SPECIFIC / NO_DATA collapse to silence
        # — the _decision_paralysis_chat_lines silence precedent). VAPOR_
        # DECISIONS additionally surfaces ONE verbatim VAPOR sample excerpt
        # so the analyst sees what the bot is actually saying when reasoning
        # collapses. Composed verbatim by the pure _decision_vapor_chat_lines
        # helper (unit-tested; SSOT — the builder's own `headline` is the
        # chat headline, no re-derived verdict). Guarded 3s read; only
        # appears once :8090 is restarted onto /api/decision-vapor-skill.
        decision_vapor_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/decision-vapor-skill",
                    timeout=3) as resp:
                _dv = json.loads(resp.read().decode("utf-8"))
            decision_vapor_block = "\n".join(
                _decision_vapor_chat_lines(_dv))
        except Exception as e:
            _logger().warning(
                "chat: decision-vapor fetch failed: %s", e)

        # Regime-leverage fit — the watchlist is leveraged-ETF-heavy (TQQQ /
        # SOXL / SQQQ / SOXS / SPXL / SPXS), so the structural question
        # "are we positioned with or against the regime?" is high-stakes
        # and not answered anywhere else in chat. Portfolio block reports
        # leveraged_pct as a scalar, but the *fit* (lev% × regime × flow)
        # is what actually matters. Block fires ONLY on BLIND_LEVERING /
        # DANGEROUS_HEADWIND / MISSED_TAILWIND (ALIGNED / DEFENSIVE /
        # NEUTRAL / NO_DATA collapse to silence — the _decision_paralysis_
        # chat_lines silence precedent). Composed verbatim by the pure
        # _regime_leverage_fit_chat_lines helper (unit-tested; SSOT — the
        # builder's own `headline` is the chat headline, no re-derived
        # verdict). Guarded 3s read; only appears once :8090 is restarted
        # onto /api/regime-leverage-fit-skill.
        regime_leverage_fit_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/regime-leverage-fit-skill",
                    timeout=3) as resp:
                _rlf = json.loads(resp.read().decode("utf-8"))
            regime_leverage_fit_block = "\n".join(
                _regime_leverage_fit_chat_lines(_rlf))
        except Exception as e:
            _logger().warning(
                "chat: regime-leverage-fit fetch failed: %s", e)

        # Kelly-criterion sizing — given my realised payoff × win-rate, what
        # fraction would Kelly allocate to the single best position, and
        # how does the current top weight compare? The portfolio block
        # reports concentration_top1_pct as a scalar and concentration-cap
        # warns at a fixed threshold, but neither answers the *statistical*
        # sizing question. A 65% concentration is justified by a 13× payoff;
        # the same 65% on a flat edge is ruin-risk. Block fires ONLY on
        # UNDERSIZED / OVERSIZED / EXTREMELY_OVERSIZED / NEGATIVE_EDGE
        # (KELLY_ALIGNED collapses to silence — the
        # _decision_paralysis_chat_lines silence precedent). Composed
        # verbatim by the pure _kelly_sizing_chat_lines helper (unit-tested;
        # SSOT — the builder's own `headline` is the chat headline, no
        # re-derived verdict). Guarded 3s read; only appears once :8090 is
        # restarted onto /api/kelly-sizing.
        kelly_sizing_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/kelly-sizing",
                    timeout=3) as resp:
                _ks = json.loads(resp.read().decode("utf-8"))
            kelly_sizing_block = "\n".join(
                _kelly_sizing_chat_lines(_ks))
        except Exception as e:
            _logger().warning(
                "chat: kelly-sizing fetch failed: %s", e)

        # Exit-intent audit — per-closed-sell intent classification into
        # EARNINGS_CLEAR / STOP_LOSS / TARGET_HIT / THESIS_FLIP /
        # DEFENSIVE_CASH_RAISE / UNCLASSIFIED, rolled up to outcome per
        # bucket. /api/loser-autopsy classifies losers by OBJECTIVE failure
        # mode (hold × magnitude), /api/winner-autopsy looks at entry
        # rationale; neither classifies the trader's STATED REASON for
        # selling. The DRAM whipsaw (2026-05-19, -17.7% in 1.1h) was
        # exited citing "raising dry powder" — DEFENSIVE_CASH_RAISE bleed
        # surfaces here when n≥10 round-trips and the dominant intent has
        # negative avg P&L. Block fires ONLY on DOMINANT_INTENT_BLEED /
        # INTENT_UNCLEAR (DOMINANT_INTENT_HEALTHY collapses to silence —
        # the _decision_paralysis_chat_lines silence precedent). Composed
        # verbatim by the pure _exit_intent_audit_chat_lines helper
        # (unit-tested; SSOT — the builder's own `headline` is the chat
        # headline, no re-derived verdict). Guarded 3s read; only appears
        # once :8090 is restarted onto /api/exit-intent-audit.
        exit_intent_audit_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/exit-intent-audit",
                    timeout=3) as resp:
                _eia = json.loads(resp.read().decode("utf-8"))
            exit_intent_audit_block = "\n".join(
                _exit_intent_audit_chat_lines(_eia))
        except Exception as e:
            _logger().warning(
                "chat: exit-intent-audit fetch failed: %s", e)

        # Realized-vs-unrealized P&L split — the "banked vs paper" composition
        # question. Every other P&L block is a scalar (portfolio total pnl%,
        # drawdown%, β-attribution); none answers "of today's gain, how much
        # is locked-in vs paper that can evaporate?". A +$50 book that is
        # 100% realized is a fundamentally different desk than 100% open
        # paper. Block fires ONLY on DRAWING_DOWN / LEAKING_PAPER /
        # PAPER_HEAVY (BANKED / BALANCED / NO_DATA collapse to silence — the
        # _decision_paralysis_chat_lines silence precedent). Composed
        # verbatim by the pure _realized_vs_unrealized_chat_lines helper
        # (unit-tested; SSOT — the builder's own `headline` is the chat
        # headline, no re-derived verdict). Guarded 3s read; only appears
        # once :8090 is restarted onto /api/realized-vs-unrealized.
        realized_vs_unrealized_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/realized-vs-unrealized",
                    timeout=3) as resp:
                _rvu = json.loads(resp.read().decode("utf-8"))
            realized_vs_unrealized_block = "\n".join(
                _realized_vs_unrealized_chat_lines(_rvu))
        except Exception as e:
            _logger().warning(
                "chat: realized-vs-unrealized fetch failed: %s", e)

        # Watchlist coverage — which WATCHLIST tickers has the bot stopped
        # attending to? Every other panel is position-centric (what was
        # traded); none names a ticker that was IGNORED. The live WATCHLIST
        # has 48 tickers; if 36 are silent across 1000 decisions while NVDA
        # absorbs 100+ actions, that is opportunity cost the operator never
        # sees. Block fires ONLY on STAGNANT / CONCENTRATED (DIVERSIFIED /
        # NO_DATA collapse to silence — the _decision_paralysis_chat_lines
        # silence precedent). Composed verbatim by the pure
        # _watchlist_coverage_chat_lines helper (unit-tested; SSOT — the
        # builder's own `headline` is the chat headline; STAGNANT surfaces
        # up to 8 stalest ticker symbols verbatim from `by_ticker`). Guarded
        # 3s read; only appears once :8090 is restarted onto
        # /api/watchlist-coverage.
        watchlist_coverage_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/watchlist-coverage",
                    timeout=3) as resp:
                _wlc = json.loads(resp.read().decode("utf-8"))
            watchlist_coverage_block = "\n".join(
                _watchlist_coverage_chat_lines(_wlc))
        except Exception as e:
            _logger().warning(
                "chat: watchlist-coverage fetch failed: %s", e)

        # Concentration trajectory — the slope view of single-name exposure.
        # Every other concentration block on chat is point-in-time (portfolio
        # snapshot cash%, /api/risk top1_pct, correlation factor structure).
        # None answers the first-derivative question: has the book's top-1
        # weight been rising, falling, or steady over the last N days? A
        # book at 65% top-1 today reads identically in every other surface
        # whether it ramped from 30% → 65% over a week (concentration creep)
        # or jumped 0% → 65% in one cycle (single-fill blow-up — different
        # operator response). Block fires ONLY on CONCENTRATION_SPIKE /
        # RAMPING_UP / CONCENTRATED_STEADY (DECONCENTRATING / DIVERSIFIED /
        # BALANCED / INSUFFICIENT_DATA / NO_DATA collapse to silence — the
        # _decision_paralysis_chat_lines silence precedent). Composed
        # verbatim by the pure _concentration_trajectory_chat_lines helper
        # (unit-tested; SSOT — the builder's own `headline` is the chat
        # headline, no re-derived verdict). Guarded 3s read; only appears
        # once :8090 is restarted onto /api/concentration-trajectory.
        concentration_trajectory_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/concentration-trajectory",
                    timeout=3) as resp:
                _ct = json.loads(resp.read().decode("utf-8"))
            concentration_trajectory_block = "\n".join(
                _concentration_trajectory_chat_lines(_ct))
        except Exception as e:
            _logger().warning(
                "chat: concentration-trajectory fetch failed: %s", e)

        # Streak (current run + historical extremes on closed round-trips) —
        # the behavioural read no other chat block carries. The scorecard
        # gives win-rate aggregate; churn counts re-entries; hold-discipline
        # times the median losing-cut. None surface "am I on a HOT_HAND right
        # now, am I on a TILT_RISK loss-cluster?" — the textbook
        # behavioural-edge questions a desk asks after a streak forms. Block
        # fires ONLY on HOT_HAND / TILT_RISK (NEUTRAL / EMERGING / NO_DATA
        # collapse to silence — the _decision_paralysis_chat_lines silence
        # precedent — the verdict is gated to STABLE n>=8 round-trips, so a
        # three-trip "streak" stays silent by construction). Composed
        # verbatim by the pure _streak_chat_lines helper (unit-tested; SSOT
        # — the builder's own `headline` is the chat headline, no re-derived
        # verdict). Guarded 3s read; only appears once :8090 is restarted
        # onto /api/streak.
        streak_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/streak",
                    timeout=3) as resp:
                _sk = json.loads(resp.read().decode("utf-8"))
            streak_block = "\n".join(_streak_chat_lines(_sk))
        except Exception as e:
            _logger().warning(
                "chat: streak fetch failed: %s", e)

        # Standing intents — the FORWARD slice of the reasoning surface.
        # Every other reasoning chat block looks backward: decision-vapor
        # grades specificity on FILLED trades, thesis-drift re-tests the
        # open-position thesis, exit-intent-audit classifies CLOSED sells.
        # None surface the FORWARD slice — the explicit conditional intents
        # the bot itself stated ("wait for the cash session", "rotating
        # into LITE/LNOK", "premature to dump") that are still STANDING
        # without follow-up action. Answers the operator question no other
        # block answers: "what did the bot SAY it would do next, that it has
        # not yet done?" Fires ONLY on STANDING_INTENTS / STALE_INTENTS
        # (NO_INTENTS / NO_DATA collapse to silence — the _decision_
        # paralysis_chat_lines silence precedent, never chat filler when
        # the bot is reasoning without forward commitments). Composed
        # verbatim by the pure _standing_intents_chat_lines helper
        # (unit-tested; SSOT — the builder's own `headline` is the chat
        # headline AND each surfaced intent's text passes through verbatim
        # — no chat-side paraphrase of the bot's own words, the
        # _thesis_drift_chat_lines drift_reasons verbatim-passthrough
        # precedent). STALE intents tag-line ``[stale]`` so the operator
        # can see plans that aged out without action at a glance. Guarded
        # 3s read; only appears once :8090 is restarted onto
        # /api/decision-conditionals.
        standing_intents_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/decision-conditionals",
                    timeout=3) as resp:
                _si = json.loads(resp.read().decode("utf-8"))
            standing_intents_block = "\n".join(
                _standing_intents_chat_lines(_si))
        except Exception as e:
            _logger().warning(
                "chat: decision-conditionals fetch failed: %s", e)

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

        # NO_DECISION cause attribution — when the bot is silent for a
        # stretch, the operator's first question is "WHY?". The chat
        # already carries decision-paralysis (the FACT we are not deciding)
        # and runner-heartbeat (the availability signal) but neither
        # answers "is it the runner's fault, or is the box saturated by
        # review agents / backtest committee?". The trader endpoint
        # buckets the cause and emits a verbatim recommendation
        # (host_saturated requires reducing parallel Opus jobs, NOT a
        # runner restart; a parse_failed cluster is a prompt-shape bug,
        # not a host issue). Block fires ONLY on DOMINANT (NO_DATA /
        # NORMAL / MIXED collapse to silence — the
        # _decision_paralysis_chat_lines silence precedent, never chat
        # filler when the bot is deciding or when the cause is diffuse).
        # Composed verbatim by the pure _no_decision_reasons_chat_lines
        # helper (unit-tested; SSOT — the builder's own `headline` is
        # the chat headline AND already contains the recommendation
        # verbatim, no chat-side re-derivation). Guarded 3s read; only
        # appears once :8090 is restarted onto /api/no-decision-reasons.
        no_decision_reasons_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/no-decision-reasons",
                    timeout=3) as resp:
                _ndr = json.loads(resp.read().decode("utf-8"))
            no_decision_reasons_block = "\n".join(
                _no_decision_reasons_chat_lines(_ndr))
        except Exception as e:
            _logger().warning(
                "chat: no-decision-reasons fetch failed: %s", e)

        # Round-trip post-mortem — was each closed exit timed correctly
        # relative to the *next* price drift? Every existing realized-P&L
        # chat surface (winner_autopsy / loser_autopsy / streak / scorecard)
        # reduces a closed trip to a P&L number; none answer the falsifiable
        # hindsight question "did the price keep moving against the bot
        # after the sell?". A trade closed at -0.1% looks fine on
        # track-record; it reads catastrophic if the name rallied +5% the
        # hour after. Block fires ONLY when ≥1 PREMATURE / MISSED_RUNNER /
        # WHIPSAW trip exists (all-CORRECT / NEUTRAL ladders collapse to
        # silence — the _decision_paralysis_chat_lines silence precedent,
        # never chat filler when exits are timing fine). Composed verbatim
        # by the pure _round_trip_postmortem_chat_lines helper (unit-tested;
        # SSOT — the builder's top-level `headline` AND the worst trip's
        # own per-row `headline` pass through verbatim, no chat-side
        # paraphrase). Guarded 3s read; only appears once :8090 is
        # restarted onto /api/round-trip-postmortem.
        round_trip_postmortem_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/round-trip-postmortem",
                    timeout=3) as resp:
                _rtp = json.loads(resp.read().decode("utf-8"))
            round_trip_postmortem_block = "\n".join(
                _round_trip_postmortem_chat_lines(_rtp))
        except Exception as e:
            _logger().warning(
                "chat: round-trip-postmortem fetch failed: %s", e)

        # Cash drag — SPY-benchmarked $ cost of sitting in cash, per
        # rolling window. The chat already carries idle-cash snapshots
        # (/api/risk cash_pct), cash_redeployment latency (the post-SELL
        # sit pathology), and opportunity_cost (signal-specific hindsight).
        # None surface the BENCHMARKED dollar cost — "while you sat at avg
        # cash $358 over the last 168h, SPY ran +0.96% — that's $3.44 of
        # beta you forfeited by being out". That's the answer to "is
        # sitting in cash actually costing me?" the operator asks at the
        # end of a multi-day cash stretch. Block fires ONLY on COSTLY_CASH
        # (NEUTRAL / HELPFUL_CASH / INSUFFICIENT / NO_DATA collapse to
        # silence — the _decision_paralysis_chat_lines silence precedent,
        # never chat filler when cash saved you money or there's no
        # benchmark). Composed verbatim by the pure _cash_drag_chat_lines
        # helper (unit-tested; SSOT — the builder's own `headline` is the
        # chat headline, no re-derived verdict). Guarded 3s read; only
        # appears once :8090 is restarted onto /api/cash-drag.
        cash_drag_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/cash-drag",
                    timeout=3) as resp:
                _cd = json.loads(resp.read().decode("utf-8"))
            cash_drag_block = "\n".join(_cash_drag_chat_lines(_cd))
        except Exception as e:
            _logger().warning(
                "chat: cash-drag fetch failed: %s", e)

        # Passive-signal-density — the smoking-gun read for "engine idle
        # during loud news". decision-paralysis / streak surface the FACT
        # of a multi-cycle HOLD-only run; neither answers whether the
        # news during that run was QUIET (informed passive — correct
        # silence) or LOUD (deafening silence — engine sat on its hands
        # while a real news window was open). The trader endpoint
        # already discriminates these and emits DEAFENING_SILENCE; the
        # chat had been blind to that verdict. Block fires ONLY on
        # DEAFENING_SILENCE — INFORMED_PASSIVE / SIGNAL_RICH_PASSIVE /
        # NO_PASSIVE_RUN / INSUFFICIENT / NO_DATA all collapse to
        # silence, mirroring the trader-side Discord block
        # (reporter._passive_signal_density_line — same single-verdict
        # contract, so the two surfaces never disagree on what is "the
        # alert"). Composed verbatim by the pure
        # _passive_signal_density_chat_lines helper (unit-tested; SSOT
        # — the builder's own headline is the chat headline). Guarded
        # 3s read; only appears once :8090 is restarted onto
        # /api/passive-signal-density.
        passive_signal_density_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/passive-signal-density",
                    timeout=3) as resp:
                _psd = json.loads(resp.read().decode("utf-8"))
            passive_signal_density_block = "\n".join(
                _passive_signal_density_chat_lines(_psd))
        except Exception as e:
            _logger().warning(
                "chat: passive-signal-density fetch failed: %s", e)

        # News-to-trade lag — is the bot actually reacting to fresh
        # news, or is it consistently 2h behind? trade-attribution
        # enumerates per-trade article precedence; this block compresses
        # that to one reactivity verdict. A book trading 2h+ behind the
        # wire on leveraged ETFs has bled significant edge before the
        # entry — and only this block surfaces it. Block fires ONLY on
        # DELAYED — REACTIVE_FAST / REACTIVE collapse to silence (the
        # bot is reacting on time, no intervention), NO_ATTRIBUTION /
        # NO_DATA / ERROR also collapse (unmeasurable is not an alert).
        # Composed verbatim by the pure _news_to_trade_lag_chat_lines
        # helper (unit-tested; SSOT — the builder's own headline is
        # the chat headline, no re-derived verdict). Guarded 3s read;
        # only appears once :8090 is restarted onto
        # /api/news-to-trade-lag.
        news_to_trade_lag_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/news-to-trade-lag",
                    timeout=3) as resp:
                _ntl = json.loads(resp.read().decode("utf-8"))
            news_to_trade_lag_block = "\n".join(
                _news_to_trade_lag_chat_lines(_ntl))
        except Exception as e:
            _logger().warning(
                "chat: news-to-trade-lag fetch failed: %s", e)

        # Catalyst-expiry — per-open-position catalyst-class + age vs
        # catalyst-type expiry window. thesis_drift verdicts each open
        # position on P/L-since-entry; hold_discipline flags losers
        # overstayed past the desk's median losing-cut. Neither tracks
        # the *catalyst clock* — a position opened on an earnings beat
        # that has now sat for 5 days is on a STALE thesis even if it's
        # still green, because earnings beats price-in within ~2 days.
        # The chat had no surface for the catalyst-decay distinction
        # until now. Block fires ONLY on ZOMBIE_HOLDINGS — ALL_FRESH /
        # STRUCTURAL_BOOK / MIXED_BOOK / NO_DATA collapse to silence
        # (the _decision_paralysis_chat_lines silence precedent, never
        # chat filler when the book has nothing aged out). Composed
        # verbatim by the pure _catalyst_expiry_chat_lines helper
        # (unit-tested; SSOT — the builder's own headline is the chat
        # headline AND the worst zombie's own ticker / days_held /
        # catalyst_class fields are restated verbatim, no chat-side
        # re-derivation). Guarded 3s read; only appears once :8090 is
        # restarted onto /api/catalyst-expiry-skill.
        catalyst_expiry_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/catalyst-expiry-skill",
                    timeout=3) as resp:
                _ce = json.loads(resp.read().decode("utf-8"))
            catalyst_expiry_block = "\n".join(
                _catalyst_expiry_chat_lines(_ce))
        except Exception as e:
            _logger().warning(
                "chat: catalyst-expiry-skill fetch failed: %s", e)

        # Trade-asymmetry — the payoff / win-rate / disposition diagnostic.
        # Every other realised-P&L block (analytics_block, behavioural,
        # streak, exit-intent-audit) reduces closed trips to either an
        # aggregate or a per-bucket count. None expose the classic
        # payoff-trap / disposition-bleed pathology — a high win-rate made
        # of small wins and large losses, or a winner/loser hold-time skew
        # that says the desk is impatient with winners. Composed verbatim
        # by the pure _trade_asymmetry_chat_lines helper (unit-tested;
        # SSOT — the builder's own ``headline`` is the chat headline,
        # detail-line numerics restate the endpoint's own payoff_ratio /
        # actual_win_rate_pct / breakeven_win_rate_pct / hold-day fields,
        # no chat-side re-derivation). Guarded 3s read; EDGE_POSITIVE /
        # FLAT / EMERGING / NO_DATA collapse to silence (the
        # _decision_paralysis_chat_lines silence precedent — never chat
        # filler when the record is healthy or sample-thin).
        trade_asymmetry_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/trade-asymmetry",
                    timeout=3) as resp:
                _ta = json.loads(resp.read().decode("utf-8"))
            trade_asymmetry_block = "\n".join(
                _trade_asymmetry_chat_lines(_ta))
        except Exception as e:
            _logger().warning("chat: trade-asymmetry fetch failed: %s", e)

        # Rebuy-regret — the DOLLAR question on sell-then-rebuy hops.
        # reentry-velocity (upstream) grades the CADENCE of re-entries
        # (CHURN_RISK / STABLE); round-trip-postmortem grades whether the
        # SELL was well-timed against the next drift; neither answers
        # "did the actual BUY that followed the sell come at a worse
        # price?" — the canonical sold-low-bought-high failure mode.
        # Composed verbatim by the pure _rebuy_regret_chat_lines helper
        # (unit-tested; SSOT — the builder's own ``headline`` is the chat
        # headline, no chat-side re-derived verdict). Guarded 3s read;
        # SAVINGS / NET_NEUTRAL / NO_DATA / NO_REBUYS / ERROR collapse to
        # silence (the _decision_paralysis_chat_lines silence precedent
        # — never chat filler when re-entries are saving money or flat).
        rebuy_regret_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/rebuy-regret",
                    timeout=3) as resp:
                _rr = json.loads(resp.read().decode("utf-8"))
            rebuy_regret_block = "\n".join(
                _rebuy_regret_chat_lines(_rr))
        except Exception as e:
            _logger().warning("chat: rebuy-regret fetch failed: %s", e)

        # Scorer-vs-book disagreement — the meta question the chat had
        # been blind to: does the bot's OWN ML (the 17-feature
        # DecisionScorer that ``_baseline_compare_chat_lines`` grades for
        # OOS skill) currently agree with the actual book? A HIGH-severity
        # row is the live decision loop sitting on a position the scorer
        # would EXIT/TRIM — the trader endpoint already exposes this
        # ("scorer-vs-Opus disagreement panel" per its docstring) and the
        # dashboard surfaces it, but the chat had been blind to it; the
        # analyst could never ask "is the bot's ML on board with what
        # it's holding?". Composed verbatim by the pure
        # _scorer_book_disagreement_chat_lines helper (unit-tested; SSOT
        # — the worst row's ticker / scorer_verdict / last_action /
        # scorer_pred_5d_pct pass through unchanged, headline is composed
        # only from counts.HIGH + the worst row's own fields). Guarded
        # 3s read; off_distribution rows (clamped extrapolation, per the
        # trader endpoint's own docstring) are pre-filtered out so a
        # chat alert never misrepresents a clamped row as a real
        # scorer/Opus fight; scorer_trained=False / zero HIGH rows
        # collapse to silence.
        scorer_book_disagreement_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/disagreement",
                    timeout=3) as resp:
                _sbd = json.loads(resp.read().decode("utf-8"))
            scorer_book_disagreement_block = "\n".join(
                _scorer_book_disagreement_chat_lines(_sbd))
        except Exception as e:
            _logger().warning("chat: disagreement fetch failed: %s", e)

        # Notify-health — is Discord delivery from the trader healthy?
        # Every other block describes the BOOK / DECISIONS / SKILLS the
        # analyst can reason about; none answer the operator-fitness
        # question "can the trader even reach its own ops channel right
        # now?". A DEGRADED notify means trade alerts, hourly summaries
        # and daily-close posts are being silently dropped — the
        # analyst is talking about a book whose ops surface is DARK and
        # recommendations like "consider trimming TQQQ here" never
        # reach the operator. Block fires ONLY on DEGRADED (HEALTHY /
        # UNKNOWN collapse to silence — the _decision_paralysis_chat_lines
        # silence precedent, never chat filler when the channel works).
        # Composed verbatim by the pure _notify_health_chat_lines helper
        # (unit-tested; SSOT — the builder's own `headline` is the chat
        # headline, no re-derived verdict). Guarded 3s read.
        notify_health_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/notify-health",
                    timeout=3) as resp:
                _nh = json.loads(resp.read().decode("utf-8"))
            notify_health_block = "\n".join(_notify_health_chat_lines(_nh))
        except Exception as e:
            _logger().warning("chat: notify-health fetch failed: %s", e)

        # All-cash streak — how long has the live trader actually been
        # 100% cash, and at what compounding alpha cost so far? The
        # chat carries cash_pct (point-in-time), cash_redeployment
        # latency (post-SELL sit), opportunity_cost (signal-specific
        # hindsight) and cash_drag (SPY-benchmarked dollar). None
        # answer the OPERATOR-VISIBILITY question "is the book
        # CHRONICALLY flat right now?". A book that has been flat for
        # 2h looks identical on cash_pct to one that has been flat for
        # 6 days yet the second case is a strong "decision loop too
        # risk-off / something is wrong" tell. Block fires ONLY on
        # EXTENDED_HOLDOUT / PROLONGED_HOLDOUT (BRIEF_HOLDOUT /
        # NOT_ALL_CASH / NO_DATA / INSUFFICIENT_HISTORY collapse to
        # silence). Composed verbatim by the pure
        # _all_cash_streak_chat_lines helper (unit-tested; SSOT — the
        # builder's own `headline` is the chat headline). Guarded 3s
        # read.
        all_cash_streak_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/all-cash-streak",
                    timeout=3) as resp:
                _acs = json.loads(resp.read().decode("utf-8"))
            all_cash_streak_block = "\n".join(_all_cash_streak_chat_lines(_acs))
        except Exception as e:
            _logger().warning("chat: all-cash-streak fetch failed: %s", e)

        # Feed-health — the live-news pipeline fitness layer. Every
        # downstream block (decision/book/skill/news analytics) assumes
        # the feed is alive. When `/api/feed-health` flips to BLIND
        # (N consecutive 0-signal decisions), STALE_FEED (newest live
        # article > stale_hours old), or fires the UNSCORED clause
        # (articles arriving but ai_score=0 — digital-intern ML scoring
        # pipeline silently down), every other block's verdicts become
        # interpretively suspect — CASH_REDEPLOYMENT=STALLED means
        # something different when the bot is BLIND vs when the wire
        # is live. The analyst must flag this BEFORE answering "what
        # should we do?" because the right answer becomes "restart the
        # scorer" or "wait for the feed", not "trim NVDA". The
        # UNSCORED-clause sub-message is already part of the builder's
        # `headline` under BLIND/STALE_FEED. Block fires ONLY on BLIND
        # / STALE_FEED (HEALTHY / NO_DATA collapse to silence — never
        # chat filler when the feed works; NO_DATA is a probe-side
        # defect, not an actionable verdict). Composed verbatim by
        # the pure _feed_health_chat_lines helper (unit-tested; SSOT).
        # Guarded 3s read.
        feed_health_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/feed-health",
                    timeout=3) as resp:
                _fh = json.loads(resp.read().decode("utf-8"))
            feed_health_block = "\n".join(_feed_health_chat_lines(_fh))
        except Exception as e:
            _logger().warning("chat: feed-health fetch failed: %s", e)

        # Publish-lag — per-collector publish→first_seen latency. Sibling to
        # FEED HEALTH and COLLECTOR HEALTH on the latency dimension: feed-health
        # asks "is the wire alive?", collector-health asks "is each source
        # ingesting?", this one asks "when each source DOES ingest, how stale
        # are the items it sees?". A 30-min RSS poll feeding 6h-old items reads
        # HEALTHY on both other surfaces but is silently bleeding stale rows
        # into briefings. Composed verbatim by the pure _publish_lag_chat_lines
        # helper (unit-tested; SSOT — the endpoint's own `headline` is the
        # chat headline, no re-derived verdict). Read intra-process (the
        # endpoint lives on THIS dashboard) — no urlopen needed, no :8090
        # restart dependency. STALE_FEEDS / MIXED actionable; FRESH / NO_DATA /
        # ERROR collapse to silence (the _feed_health_chat_lines precedent).
        publish_lag_block = ""
        try:
            from analytics.publish_lag_audit import compute as _pl_compute
            _pl = _pl_compute()
            # Mirror the endpoint's verdict-ladder inline so the chat block
            # reads the same SSOT as the /api/publish-lag response.
            _collectors = _pl.get("collectors") or {}
            if _collectors and _pl.get("rows_with_parseable_lag"):
                _stalest = (_pl.get("ranked_stalest") or [None])[0]
                _freshest = (_pl.get("ranked_freshest") or [None])[0]
                _sm = _stalest.get("median_lag_min") if isinstance(_stalest, dict) else None
                _sn = _stalest.get("n") if isinstance(_stalest, dict) else 0
                if (isinstance(_sm, (int, float)) and _sm > 60.0
                        and isinstance(_sn, int) and _sn >= 10):
                    _v = "STALE_FEEDS"
                    _h = (f"stalest: {_stalest['collector']} p50={_sm:.1f}m "
                          f"(n={_sn})")
                    if isinstance(_freshest, dict):
                        _fm = _freshest.get("median_lag_min")
                        if isinstance(_fm, (int, float)):
                            _h += (f"; freshest: {_freshest['collector']} "
                                   f"p50={_fm:.1f}m")
                elif isinstance(_sm, (int, float)) and _sm > 15.0:
                    _v = "MIXED"
                    _h = (f"mixed: stalest {_stalest['collector']} "
                          f"p50={_sm:.1f}m; {len(_collectors)} collectors "
                          f"reported")
                else:
                    _v = "FRESH"
                    _h = ""
                _pl_envelope = {
                    "verdict": _v,
                    "headline": _h,
                    "ranked_stalest": _pl.get("ranked_stalest"),
                }
                publish_lag_block = "\n".join(
                    _publish_lag_chat_lines(_pl_envelope))
        except Exception as e:
            _logger().warning("chat: publish-lag fetch failed: %s", e)

        # Rotation skill — when the desk SELLs X and within hours BUYs a
        # *different* ticker Y, does Y outperform X over the next 5d? The
        # return-spread sibling to cash_redeployment_latency: that endpoint
        # says HOW FAST cash redeploys; this one says whether the
        # redeployment was SKILLED (paired-rotation alpha at the forward
        # horizon). A FAST_REDEPLOY desk that rotates DRAM→MSTR while DRAM
        # rips and MSTR sags is fast AND lazy — both verdicts together
        # diagnose the right pathology. Composed verbatim by the pure
        # _rotation_skill_chat_lines helper (unit-tested; SSOT — the
        # endpoint's own `headline` is the chat headline, no re-derived
        # verdict). Guarded 3s read; only appears once :8090 is restarted
        # onto /api/rotation-skill. LAZY_ROTATION / NET_NEGATIVE actionable;
        # SKILLED_ROTATION / NET_POSITIVE / NEUTRAL / INSUFFICIENT_DATA
        # collapse to silence.
        rotation_skill_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/rotation-skill",
                    timeout=3) as resp:
                _rs = json.loads(resp.read().decode("utf-8"))
            rotation_skill_block = "\n".join(
                _rotation_skill_chat_lines(_rs))
        except Exception as e:
            _logger().warning("chat: rotation-skill fetch failed: %s", e)

        # Hourly-PnL fingerprint — WHEN in the trading day has this bot
        # actually earned alpha vs SPY? The chat carries dozens of
        # book/decisions/skills blocks but is BLIND to the structural
        # time-of-day verdict — a MORNING_EDGE bot should be more
        # aggressive at hour 11 than at hour 15, and the chat can answer
        # "is now a good time to lean into this signal?" only when this
        # empirical hourly verdict is in the prompt. Fires ONLY on
        # MORNING_EDGE / MIDDAY_EDGE / AFTERNOON_EDGE / OFF_HOURS_EDGE
        # (FLAT_CLOCK / INSUFFICIENT_DATA / NO_SPY_DATA collapse to
        # silence). Composed verbatim by the pure
        # _hourly_pnl_fingerprint_chat_lines helper (unit-tested; SSOT —
        # the builder's own `headline` is the chat headline, no
        # re-derived verdict). Guarded 3s read.
        hourly_pnl_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/hourly-pnl-fingerprint",
                    timeout=3) as resp:
                _hp = json.loads(resp.read().decode("utf-8"))
            hourly_pnl_block = "\n".join(_hourly_pnl_fingerprint_chat_lines(_hp))
        except Exception as e:
            _logger().warning("chat: hourly-pnl-fingerprint fetch failed: %s", e)

        # Weekday-PnL fingerprint — is TODAY historically a good day
        # for this bot vs SPY? Companion to the hourly fingerprint — the
        # chat carries no DOW-edge verdict, so the analyst can't answer
        # "should I be more cautious today?" with empirical-edge data.
        # Fires ONLY on WEEKDAY_EDGE / WEEKEND_EDGE (FLAT_WEEK /
        # INSUFFICIENT_DATA / NO_SPY_DATA collapse to silence). Composed
        # verbatim by the pure _weekday_pnl_fingerprint_chat_lines
        # helper (unit-tested; SSOT). Guarded 3s read.
        weekday_pnl_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/weekday-pnl-fingerprint",
                    timeout=3) as resp:
                _wp = json.loads(resp.read().decode("utf-8"))
            weekday_pnl_block = "\n".join(_weekday_pnl_fingerprint_chat_lines(_wp))
        except Exception as e:
            _logger().warning("chat: weekday-pnl-fingerprint fetch failed: %s", e)

        # Cash-conviction fit — is the CURRENT cash level wrong given the
        # loudest CURRENT live signal right now? The chat carries
        # cash_pct (point-in-time), all_cash_streak (chronic-flat
        # duration), cash_redeployment (post-SELL latency), cash_drag
        # (SPY-benchmarked dollar). None of those answer the structural
        # calibration question — a book 95% cash while ai_score 9.2
        # screams is structurally wrong in a way none of the other
        # surfaces flag. Fires ONLY on IDLE_DESPITE_SURGE / OVERDEPLOYED
        # / IDLE_LOW_CONVICTION (BALANCED / NO_DATA collapse to silence).
        # Composed verbatim by the pure _cash_conviction_fit_chat_lines
        # helper (unit-tested; SSOT). Guarded 3s read.
        cash_conviction_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/cash-conviction-fit",
                    timeout=3) as resp:
                _cc = json.loads(resp.read().decode("utf-8"))
            cash_conviction_block = "\n".join(_cash_conviction_fit_chat_lines(_cc))
        except Exception as e:
            _logger().warning("chat: cash-conviction-fit fetch failed: %s", e)

        # Actionable opportunities — the composite ranker for UNHELD
        # watchlist names that crosses three orthogonal axes (scorer
        # predicted 5d return × this-dashboard's ticker-news-burst ×
        # persistent-watchlist hot-run). The chat already carries each
        # axis SEPARATELY (scorer-opportunities, persistent-watchlist
        # under the trader analytics block, this dashboard's ticker-
        # velocity / ticker-comentions) but none answer the analyst's
        # synthesis question: "of the strong scorer picks, which one is
        # the wire ALSO talking about RIGHT NOW?". A HIGH_CONVICTION /
        # NEWS_CONFIRMED verdict means quant + wire agree on a name — the
        # specific cross-confirmation no single panel surfaces. The
        # SCORER_BUT_NO_NEWS verdict explicitly documents the live-state
        # disagreement (scorer screaming STRONG_HOLD on dozens of names,
        # wire silent) so the analyst can answer "should I act?" with
        # honest source-availability awareness. Fires ONLY on
        # HIGH_CONVICTION_FOUND / NEWS_CONFIRMED / PERSISTENT_FOLLOWUP /
        # SCORER_BUT_NO_NEWS / NEWS_BUT_NO_SCORER (ALL_QUIET /
        # INSUFFICIENT_DATA / ERROR collapse to silence). Composed
        # verbatim by the pure _actionable_opportunities_chat_lines
        # helper (unit-tested; SSOT). Guarded 3s read.
        actionable_opportunities_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/actionable-opportunities",
                    timeout=3) as resp:
                _ao = json.loads(resp.read().decode("utf-8"))
            actionable_opportunities_block = "\n".join(
                _actionable_opportunities_chat_lines(_ao)
            )
        except Exception as e:
            _logger().warning(
                "chat: actionable-opportunities fetch failed: %s", e
            )

        # Repeat-loser — chronic-pattern behavioural read: tickers where
        # the bot has lost the last N closed round-trips in a row. Every
        # other realised-P&L block aggregates by class or in total; this
        # is the only block that names a specific ticker the bot keeps
        # losing on. Fires ONLY on state=="REPEAT_LOSER" (OK / NO_DATA
        # collapse to silence — never chat filler when there is no
        # offender). Composed verbatim by the pure
        # _repeat_loser_chat_lines helper (unit-tested; SSOT — the
        # builder's own `headline` is the chat headline, no re-derived
        # verdict). Guarded 3s read.
        repeat_loser_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/repeat-loser",
                    timeout=3) as resp:
                _rl = json.loads(resp.read().decode("utf-8"))
            repeat_loser_block = "\n".join(_repeat_loser_chat_lines(_rl))
        except Exception as e:
            _logger().warning("chat: repeat-loser fetch failed: %s", e)

        # Exit-only-streak — consecutive SELLs since the last entry at
        # the BOOK level. /api/streak grades W/L on closed round-trips;
        # /api/churn measures re-entry cadence; cash-drag measures idle-
        # cash dollar cost. None surface the trade-DIRECTION sequence
        # that says "the last 6 fills were all SELLs — the engine is
        # liquidating, not running the strategy". Fires ONLY on verdict
        # DEFENSIVE_TRIM (≥3 consec exits) / DEFENSIVE_LIQUIDATION (≥6)
        # (MOST_RECENT_IS_ENTRY collapses to silence). Composed verbatim
        # by the pure _exit_only_streak_chat_lines helper (unit-tested;
        # SSOT — the builder's own `headline` is the chat headline, no
        # re-derived verdict). Guarded 3s read.
        exit_only_streak_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/exit-only-streak",
                    timeout=3) as resp:
                _es = json.loads(resp.read().decode("utf-8"))
            exit_only_streak_block = "\n".join(
                _exit_only_streak_chat_lines(_es)
            )
        except Exception as e:
            _logger().warning("chat: exit-only-streak fetch failed: %s", e)

        # Catalyst-class autopsy — per-CATALYST-CLASS win-rate / PnL
        # leaderboard. trade_asymmetry / winner_autopsy / loser_autopsy
        # aggregate; per_ticker_skill is per-NAME edge. None answer the
        # per-CLASS question: of the 9 catalyst classes (ML_ADVISOR,
        # ANALYST_PT, TECHNICALS, EARNINGS_PLAY, MACRO, BREAKING_NEWS,
        # PUNDIT, SECTOR_SYMPATHY, CONCENTRATION) which has biased my
        # realised P&L up or down? Surfaces a structural weight-
        # allocation recommendation invisible to every other surface.
        # Fires ONLY when state==STABLE AND (top_biased_winner OR
        # top_biased_loser) — NO_DATA / EMERGING / STABLE-but-no-bias
        # all collapse to silence. Composed verbatim by the pure
        # _catalyst_class_autopsy_chat_lines helper (unit-tested; SSOT —
        # the builder's own `headline` is the chat headline, no re-
        # derived verdict). Guarded 3s read.
        catalyst_class_autopsy_block = ""
        try:
            import urllib.request as _urllib
            with _urllib.urlopen(
                    "http://127.0.0.1:8090/api/catalyst-class-autopsy",
                    timeout=3) as resp:
                _cc = json.loads(resp.read().decode("utf-8"))
            catalyst_class_autopsy_block = "\n".join(
                _catalyst_class_autopsy_chat_lines(_cc)
            )
        except Exception as e:
            _logger().warning(
                "chat: catalyst-class-autopsy fetch failed: %s", e
            )

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
            + (f"PAPER TRADER — DISCORD-DELIVERY HEALTH (operator-fitness — when DEGRADED, trade alerts / hourly summaries / daily-close posts are silently being dropped: the analyst is talking about a book whose ops surface is DARK and any 'consider trimming X' recommendation never reaches the operator. Surfaced ONLY when DEGRADED, never filler when HEALTHY / UNKNOWN. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{notify_health_block}\n\n" if notify_health_block else "")
            + (f"PAPER TRADER — ALL-CASH STREAK (the chronic-flat-book surface — cash_pct snapshots are point-in-time; cash_redeployment latency is the post-SELL sit; opportunity_cost is signal-specific hindsight; cash_drag is SPY-benchmarked dollar; none answer 'how long has the book ACTUALLY been 100% cash right now, and at what compounding alpha cost so far?'. A book flat for 2h vs 6 days reads identically on cash_pct yet the second case is a strong 'decision loop too risk-off / something is wrong' tell. Surfaced ONLY when EXTENDED_HOLDOUT / PROLONGED_HOLDOUT, never filler when BRIEF_HOLDOUT / NOT_ALL_CASH / NO_DATA / INSUFFICIENT_HISTORY. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{all_cash_streak_block}\n\n" if all_cash_streak_block else "")
            + (f"PAPER TRADER — FEED HEALTH (the live-news pipeline fitness — every downstream block assumes the feed is alive; when feed-health is BLIND (consecutive 0-signal decisions), STALE_FEED (newest article > stale_hours old), or UNSCORED (articles arriving but ai_score=0 — digital-intern ML scoring pipeline silently down), every other verdict becomes interpretively suspect: CASH_REDEPLOYMENT=STALLED means something different when the bot is BLIND vs when the wire is live. The analyst must flag this BEFORE answering 'what should we do?' because the right answer becomes 'restart the scorer' or 'wait for the feed', not 'trim NVDA'. The UNSCORED-clause is already part of the headline under BLIND/STALE_FEED. Surfaced ONLY when BLIND / STALE_FEED, never filler when HEALTHY / NO_DATA. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{feed_health_block}\n\n" if feed_health_block else "")
            + (f"DIGITAL INTERN — PUBLISH LAG (per-collector publish→first_seen latency. The companion to FEED HEALTH and COLLECTOR HEALTH on the latency dimension: feed-health asks 'is the wire alive?', collector-health asks 'is each source ingesting?', this asks 'when each source DOES ingest, how stale are the items?'. A 30-min RSS poll feeding 6h-old items reads HEALTHY on both other surfaces but is silently bleeding stale rows into briefings — items that score the same time_sensitivity weight as fresh wire copy. Surfaced ONLY when STALE_FEEDS / MIXED (stalest collector median > 60m AND n≥10, or median > 15m), never filler when FRESH / NO_DATA. Headline carries verbatim from THIS dashboard's intra-process compute() — restate, never re-derive):\n{publish_lag_block}\n\n" if publish_lag_block else "")
            + (f"PAPER TRADER — ROTATION SKILL (when the desk SELLs X and within hours BUYs a different ticker Y, did Y outperform X over the next 5d? The return-spread sibling to CASH REDEPLOYMENT LATENCY: that block says HOW FAST cash redeploys; this one says whether the redeployment was SKILLED. A FAST_REDEPLOY desk that rotates DRAM→MSTR while DRAM rips and MSTR sags is fast AND lazy — both verdicts together diagnose the right pathology. Distinct from round_trip_postmortem (per-position post-exit drift, no pair) and rebuy_regret / reentry_velocity (same-ticker, not cross-ticker). Surfaced ONLY when LAZY_ROTATION / NET_NEGATIVE, never filler when SKILLED_ROTATION / NET_POSITIVE / NEUTRAL / INSUFFICIENT_DATA. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{rotation_skill_block}\n\n" if rotation_skill_block else "")
            + (f"PAPER TRADER — TIME-OF-DAY EDGE (the per-hour-of-day alpha-vs-SPY fingerprint over the bot's equity-curve history — 'WHEN in the trading day has this bot actually earned alpha?'. The chat carries dozens of book/decisions/skills blocks but no temporal-edge verdict; a MORNING_EDGE bot should be more aggressive at the best alpha hour and lighter at the worst-alpha hour, and the analyst can answer 'is now a good time to lean into this signal?' only with this empirical hour-of-day view. Surfaced ONLY when MORNING_EDGE / MIDDAY_EDGE / AFTERNOON_EDGE / OFF_HOURS_EDGE (i.e. a non-flat fingerprint with adequate samples), never filler when FLAT_CLOCK / INSUFFICIENT_DATA / NO_SPY_DATA. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{hourly_pnl_block}\n\n" if hourly_pnl_block else "")
            + (f"PAPER TRADER — DAY-OF-WEEK EDGE (the per-weekday alpha-vs-SPY fingerprint over the bot's equity-curve history — 'is TODAY historically a good day for this bot vs SPY?'. Companion to TIME-OF-DAY EDGE — answers 'should I be more cautious today?' on a different time axis. Surfaced ONLY when WEEKDAY_EDGE / WEEKEND_EDGE (non-flat fingerprint with adequate samples), never filler when FLAT_WEEK / INSUFFICIENT_DATA / NO_SPY_DATA. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{weekday_pnl_block}\n\n" if weekday_pnl_block else "")
            + (f"PAPER TRADER — CASH-CONVICTION FIT (is the CURRENT cash level wrong given the loudest CURRENT live signal? The chat already carries cash_pct (point-in-time), all_cash_streak (chronic-flat duration), cash_redeployment (post-SELL latency), cash_drag (SPY-benchmarked dollar) — none of those answer the structural calibration question. A book 95% cash while ai_score 9.2 screams is structurally wrong in a way none of the other cash surfaces flag, and a fully-deployed book against a 5.5 loudest-live signal is overdeployed for that conviction. Surfaced ONLY when IDLE_DESPITE_SURGE / OVERDEPLOYED / IDLE_LOW_CONVICTION, never filler when BALANCED / NO_DATA. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{cash_conviction_block}\n\n" if cash_conviction_block else "")
            + (f"PAPER TRADER — ACTIONABLE OPPORTUNITIES (the composite ranker for UNHELD watchlist names that crosses three orthogonal axes: scorer predicted 5d return × ticker-news-burst (this dashboard's per-held-ticker volume-spike detector) × persistent-watchlist-opportunity hot-run hours. The chat already carries each axis SEPARATELY but none answer the synthesis question 'of the strong scorer picks, which one is the wire ALSO talking about RIGHT NOW?'. HIGH_CONVICTION_FOUND / NEWS_CONFIRMED means quant + wire AGREE on a name — high-trust cross-confirmation. SCORER_BUT_NO_NEWS is the documented live failure mode (46 STRONG_HOLD picks while the wire is silent on every one); surfacing it explicitly lets the analyst answer 'should I act?' with honest source-availability awareness rather than assuming silence means absence-of-signal. Surfaced ONLY when HIGH_CONVICTION_FOUND / NEWS_CONFIRMED / PERSISTENT_FOLLOWUP / SCORER_BUT_NO_NEWS / NEWS_BUT_NO_SCORER, never filler when ALL_QUIET / INSUFFICIENT_DATA / ERROR. Headline + per-ticker `reasons` strings pass verbatim from the trader endpoint — restate, never re-derive):\n{actionable_opportunities_block}\n\n" if actionable_opportunities_block else "")
            + (f"PAPER TRADER — WHAT MATERIALLY CHANGED SINCE YOU LAST LOOKED (ranked, last 6h):\n{session_delta_block}\n\n" if session_delta_block else "")
            + (f"PAPER TRADER ANALYTICS:\n{analytics_block}\n\n" if analytics_block else "")
            + (f"PAPER TRADER — BEHAVIOURAL DIAGNOSIS (the bot's own self-review verdicts):\n{behavioural_block}\n\n" if behavioural_block else "")
            + (f"PAPER TRADER — ML GATE HONESTY (does the DecisionScorer that modulates the bot's live position sizing beat a one-line rule OUT OF SAMPLE? the analytics above report the flattering in-sample story; this is the generalization-relevant verdict):\n{baseline_compare_block}\n\n" if baseline_compare_block else "")
            + (f"PAPER TRADER — NEWS-SOURCE EDGE (the read-COLLECTOR companion to ML GATE HONESTY — even if the gate works, are the inputs feeding it edge-bearing or wire-noise? Per-collector predictive-edge leaderboard across digital-intern's ~17 sources: which collectors' scored headlines actually precede the SPY-abnormal move at the reference horizon, and which are weakest? The analyst can answer 'should I trust this source's headline?' or 'which collectors actually move the tape?' from no other surface in chat — this verdict lives only in the trader endpoint and the JS-only se-card dashboard panel. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{news_source_edge_block}\n\n" if news_source_edge_block else "")
            + (f"PAPER TRADER — FACTOR CONCENTRATION (the held book's pairwise return correlation + effective-independent-bets count; complements /api/risk's NAME-level view — a 59/41 book that is concentrated by weight may STILL be a single FACTOR bet if both names move as one):\n{correlation_block}\n\n" if correlation_block else "")
            + (f"PAPER TRADER — OPEN-POSITION THESIS DRIFT (every holding re-tested against the verbatim reason it was opened for, graded INTACT/WEAKENING/BROKEN by P/L since entry + live quant/momentum; an INTACT book collapses to silence so this block ONLY fires when at least one held position is materially off its entry thesis — drift_reasons carry verbatim from the trader endpoint, never re-derived):\n{thesis_drift_block}\n\n" if thesis_drift_block else "")
            + (f"PAPER TRADER — PRIORITISED ACTION PLAN (the bot's own next-session game plan):\n{game_plan_block}\n\n" if game_plan_block else "")
            + (f"PAPER TRADER — HOLD-DISCIPLINE ALERT (a losing position overstayed past the desk's own median losing-cut):\n{hold_discipline_block}\n\n" if hold_discipline_block else "")
            + (f"HELD-TICKER 24h NEWS-CONVICTION TREND (per held name, ai_score bucketed into 4 × 6h slices; only RISING / FADING are surfaced — STABLE/quiet collapses to silence. Complements /api/portfolio-signals' current-state snapshot with the temporal direction: is the wire focusing MORE or LESS on this position over the last day?):\n{conviction_decay_block}\n\n" if conviction_decay_block else "")
            + (f"ALERT-CONFIDENCE TREND (urgent-story clusters whose unique-source count is GROWING or FADING vs the prior 6-24h half; corroboration that's still expanding is the highest-trust read — a single-source story is usually PR not news, a fading story has been priced):\n{alert_trend_block}\n\n" if alert_trend_block else "")
            + (f"PAPER TRADER OPTIONS GREEKS (Black-Scholes, live IV):\n{greeks_block}\n\n" if greeks_block else "")
            + (f"NEWS SECTOR PULSE (native — where the wire is concentrated right now, recency-weighted, last 24h; independent of the price heatmap below so it survives a stale/down paper-trader):\n{sector_pulse_block}\n\n" if sector_pulse_block else "")
            + (f"NEWS SECTOR COHERENCE (per-sector bullish/bearish stance dispersion across the same 24h wire — the companion to PULSE that answers the structural question PULSE cannot: is the concentration a MACRO STORY all agreeing on direction (sector-wide positioning is the trade) or IDIOSYNCRATIC catalysts pulling in different directions (sector-wide positioning is the wrong move, name-level only)? Surfaced ONLY when at least one sector reaches MACRO_BULL / MACRO_BEAR / TILT_BULL / TILT_BEAR; SPLIT / INSUFFICIENT / no-news collapses to silence. Headline + per-sector verdict carry verbatim from the builder — restate, never re-derive):\n{sector_coherence_block}\n\n" if sector_coherence_block else "")
            + (f"HELD-WIRE BALANCE (per-held-ticker bullish/bearish stance from the same 24h wire — the per-NAME companion to SECTOR COHERENCE. SECTOR COHERENCE answers 'is the wire agreed on a direction at the sector level?'; HELD-WIRE BALANCE answers the per-name follow-up 'is the wire on MY SPECIFIC HELD names aligned with my long bias, or quietly opposing it?'. A sector can be MACRO_BULL while the wire on the specific held single-name (LITE, AXTI, QBTS) is bearish — that is the structural gap between sector positioning and name-level positioning, invisible to every other block. Silence-on-healthy: surfaced ONLY when at least one held name reaches BEAR_LEAN (wire opposes the desk's long bias). BULL_LEAN names are aligned-with-book good news — collapse to silence. MIXED / INSUFFICIENT also silence: chat filler is not information. BOOK_BEAR additionally prepends the headline so the operator sees the book-level verdict before the per-name detail. Per-name n_bear / n_classified / coherence_pct / lead_headline carry verbatim from the builder — restate, never re-derive):\n{held_wire_balance_block}\n\n" if held_wire_balance_block else "")
            + (f"NEWS SENTIMENT REVERSAL (native — per-ticker avg ml_score sign-flip between consecutive 2h windows (PREV [4h, 2h ago) → CURR [2h ago, now)). The directional-change view PULSE/COHERENCE cannot compose: those answer the SECTOR-level question, this answers the per-TICKER question 'which positions have just had the news turn against (or toward) the thesis?'. Both windows are gated to ≥ MIN_ARTICLES articles so single-row noise can't fire a flip. Surfaced ONLY when at least one reversal qualifies; zero-reversal windows collapse to silence — never chat filler when the wire is directionally stable. Per-ticker direction + prev → curr means + article counts carry verbatim from the builder — restate, never re-derive):\n{sentiment_reversal_block}\n\n" if sentiment_reversal_block else "")
            + (f"NEWS TICKER-SCORE DISPERSION (native — per-ticker intra-window std-dev of ml_score over the last 24h. The within-window consensus companion to SENTIMENT REVERSAL: reversal asks 'did the direction flip across windows?'; dispersion asks 'are the articles WITHIN the current window AGREEING or DISAGREEING on this ticker?'. A ticker with five articles all scoring 7.5–8.0 is consensus; the same mean spread 1.0–9.5 is contested news — structurally different signals invisible to every other panel that only carries the mean. Surfaced ONLY when MIXED_BOOK / CONFLICTED_NEWS (CONSENSUS / NO_DATA / NO_DISPERSION collapse to silence — never filler when the wire is consistent). Per-ticker n / mean / std / [min, max] carry verbatim from the builder — restate, never re-derive):\n{ticker_score_dispersion_block}\n\n" if ticker_score_dispersion_block else "")
            + (f"NEWS TICKER VELOCITY (native — top tickers by raw arrival-count ratio recent vs prior, 2h vs 2h prior. The arrival-VOLUME axis sibling of REVERSAL (cross-window direction) and DISPERSION (intra-window consensus): those describe HOW the wire feels about a ticker; this describes HOW MUCH the wire is talking about it. A BREAKING name has both substantial recent count AND a sharp acceleration vs the prior window — the early indicator before the score-based surfaces catch up. Surfaced ONLY when at least one ticker reaches BREAKING / WARMING; QUIET / NO_DATA collapse to silence — never filler when the wire is structurally flat. Per-ticker recent / prior / ratio / newest_age_s carry verbatim from the builder — restate, never re-derive):\n{ticker_velocity_block}\n\n" if ticker_velocity_block else "")
            + (f"NEWS TICKER COMENTIONS (native — top ticker pairs co-occurring in the last 2h. The SECTOR-axis sibling of single-ticker VELOCITY / DISPERSION / REVERSAL: when two tickers light up TOGETHER repeatedly, it's usually a sector ETF rip, peer-readthrough, or M&A pairing — not a single-name story. Separates 'NVDA velocity from idiosyncratic catalysts' from 'NVDA velocity as part of a semis basket move'. Lift = co_count / min(solo_a, solo_b). Surfaced ONLY when at least one pair reaches COUPLED_NAMES / SECTOR_BURST; DISCONNECTED / NO_DATA collapse to silence — never filler when no pair is recurring. Top pairs' co_count / solo totals / lift carry verbatim from the builder — restate, never re-derive):\n{ticker_comentions_block}\n\n" if ticker_comentions_block else "")
            + (f"DRAM / SEMIS 5d MOMENTUM HEATMAP (paper-trader price momentum):\n{heatmap_block}\n\n" if heatmap_block else "")
            + (f"EARNINGS RADAR (scheduled gap risk):\n{earnings_block}\n\n" if earnings_block else "")
            + (f"PAPER TRADER — PRE-EARNINGS DOLLARIZED 1σ SHOCK (per HELD imminent print: 'if NVDA gaps the typical 1σ on its release, the book moves $X / Y% of equity' — the forward $-at-risk view that complements the EARNINGS RADAR's timing-only listing):\n{earnings_shock_block}\n\n" if earnings_shock_block else "")
            + (f"MACRO CALENDAR — FOMC RATE DECISION (the single biggest MARKET-WIDE event; it moves the whole book at once, leveraged ETFs most violently — surfaced only when one is actually within the 14d horizon):\n{macro_calendar_block}\n\n" if macro_calendar_block else "")
            + (f"PAPER TRADER — EVENT READINESS (will the live trader actually be able to react before the next earnings print? Joins earnings-risk + decision-velocity + Claude-empty rate + the *current* NO_DECISION streak into a single per-event verdict — surfaced ONLY when BLIND / DEGRADED / IMMINENT_OVERDUE, never as filler when the pipeline is healthy. Recommended_action carries verbatim from the trader endpoint — restate, never re-derive):\n{event_readiness_block}\n\n" if event_readiness_block else "")
            + (f"PAPER TRADER — DECISION PARALYSIS (consecutive HOLD-only / NO_DECISION runs on the live decision loop — a stacked HOLD_LOCK block reads HEALTHY on runner-heartbeat and decision-health 24h aggregate but means Opus decided every cycle and never moved the book. Surfaced ONLY when HOLD_LOCK / IDLE_STORM / PASSIVE_LOOP, never filler when ACTIVE. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{decision_paralysis_block}\n\n" if decision_paralysis_block else "")
            + (f"PAPER TRADER — PERSONA-BOOK FIT (does the live book's weight distribution mirror a backtest-persona rated DRAG by the persona-leaderboard? Every other block analyses position-by-position fitness; none surface the whole-book archetype-overlap angle. A book that mirrors a known underperforming persona is structurally adding variance, not alpha, regardless of how reasonable each individual trade looked at entry. Surfaced ONLY when ALIGNED_DRAG, never filler when ALIGNED_EDGE / ALIGNED_FLAT / NO_BOOK / WEAK_OVERLAP / INSUFFICIENT_PERSONA. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{persona_book_fit_block}\n\n" if persona_book_fit_block else "")
            + (f"PAPER TRADER — INVERSE-PAIR CONFLICT (leveraged-long + leveraged-inverse ETFs of the same underlying family simultaneously held — TQQQ+SQQQ, SOXL+SOXS, SPXL+SPXS, FNGU+FNGD, TECL+TECS. Directional exposure largely cancels but both sleeves keep paying leverage decay every single day. etf-lookthrough reports the net single-name outcome but not the carry-waste fact; correlation-cluster-warning flags positively-correlated clusters and lets the negatively-correlated pair through; regime-leverage-fit reads 'high leveraged %' without distinguishing a paired book from a clean one-sided bet. Surfaced ONLY when CARRY_WASTE, never filler when CLEAN / NO_BOOK / OPPOSING_UNLEVERED. Headline + worst-family fields carry verbatim from the trader endpoint — restate, never re-derive):\n{inverse_pair_conflict_block}\n\n" if inverse_pair_conflict_block else "")
            + (f"PAPER TRADER — WATCHLIST NEWS SILENCE (per-WATCHLIST-ticker live-news coverage map. Of the ~47 tickers Opus is allowed to consider this cycle, how many had ZERO live articles in the last 24h, and which are mention-storming? Complements held-news-silence by surfacing the UNIVERSE blind spot every other panel ignores — a silent name in the prompt looks operationally equivalent to a well-covered one and Opus has no way to know whether silence means 'nothing happened' or 'the collector failed'. Surfaced ONLY when BLIND_UNIVERSE / SPARSE_COVERAGE, never filler when WELL_COVERED / NO_DATA. Headline + silent / hot lists carry verbatim from the trader endpoint — restate, never re-derive):\n{watchlist_news_silence_block}\n\n" if watchlist_news_silence_block else "")
            + (f"PAPER TRADER — CONCURRENT-OPUS ATTRIBUTION (per-parent-tree breakdown of concurrent ``claude --model claude-opus`` subprocesses on the box — /api/host-guard reports the count but not WHICH parent trees own them, so killing safely requires either inspecting every PID or blanket ``pkill -9 claude`` which also nukes the legitimate live runner Opus. This block names the rogue parent (hourly_review.sh / continuous backtest / runner / digital-intern daemon / unknown) and prescribes the exact targeted-kill command. Surfaced ONLY when ELEVATED / SATURATED, never filler when NO_OPUS / CLEAN / BENIGN. Headline and recommendation carry verbatim from the trader endpoint — restate, never re-derive):\n{concurrent_opus_attribution_block}\n\n" if concurrent_opus_attribution_block else "")
            + (f"PAPER TRADER — CASH REDEPLOYMENT LATENCY (post-SELL cash-to-next-BUY interval distribution — the sold-then-sat pathology. A book that sells then sits for 5 days has the same headline cash% as one that redeploys in 6h, but the desk in question is materially different. Surfaced ONLY when SLOW / STALLED, never filler when FAST_REDEPLOY / STEADY. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{cash_redeployment_block}\n\n" if cash_redeployment_block else "")
            + (f"PAPER TRADER — DECISION VAPOR (per-FILLED-decision structural-quality detector — does the reasoning cite specific numbers + catalysts + tickers, or is Opus writing generic 'strong setup, building position' vapor? A vapor trade that fails has nothing for the next decision to learn from — this is the only block that answers 'is the bot thinking, or rationalising?'. Surfaced ONLY when MIXED / VAPOR_DECISIONS, never filler when SPECIFIC. Headline + any VAPOR sample excerpt carry verbatim from the trader endpoint — restate, never re-derive):\n{decision_vapor_block}\n\n" if decision_vapor_block else "")
            + (f"PAPER TRADER — REGIME-LEVERAGE FIT (book-leverage alignment vs prevailing SPY momentum regime. The watchlist is leveraged-ETF-heavy — the structural question 'are we positioned with or against the regime?' is high-stakes and answered nowhere else in chat. A 0% leveraged book during a bull tape is just as structurally wrong as a 40% leveraged book during a bear. Surfaced ONLY when BLIND_LEVERING / DANGEROUS_HEADWIND / MISSED_TAILWIND, never filler when ALIGNED / DEFENSIVE / NEUTRAL. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{regime_leverage_fit_block}\n\n" if regime_leverage_fit_block else "")
            + (f"PAPER TRADER — KELLY SIZING (Kelly-criterion sizing diagnostic — given my realised win-rate and payoff ratio, what fraction would Kelly allocate to the single best position, and how does the current top weight compare? The portfolio block reports concentration_top1_pct as a scalar and concentration-cap warns at a fixed threshold; neither answers whether ANY fixed threshold is statistically justified by the realised edge. A 65% concentration is justified by a 13× payoff and 67% win-rate; the same 65% on a flat edge is ruin-risk territory. Surfaced ONLY when UNDERSIZED / OVERSIZED / EXTREMELY_OVERSIZED / NEGATIVE_EDGE, never filler when KELLY_ALIGNED. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{kelly_sizing_block}\n\n" if kelly_sizing_block else "")
            + (f"PAPER TRADER — EXIT-INTENT AUDIT (per-closed-sell intent classification: EARNINGS_CLEAR / STOP_LOSS / TARGET_HIT / THESIS_FLIP / DEFENSIVE_CASH_RAISE / UNCLASSIFIED, rolled up to outcome per bucket. Loser-autopsy classifies losers by OBJECTIVE failure mode (hold × magnitude); winner-autopsy looks at entry rationale; neither classifies the trader's STATED REASON for selling. When the most common stated reason to sell is also a money loser, the desk has a behavioural blind spot the other blocks cannot see. Surfaced ONLY when DOMINANT_INTENT_BLEED / INTENT_UNCLEAR, never filler when DOMINANT_INTENT_HEALTHY. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{exit_intent_audit_block}\n\n" if exit_intent_audit_block else "")
            + (f"PAPER TRADER — REALIZED vs UNREALIZED P&L SPLIT (the banked-vs-paper composition question — of today's net P&L, how much is locked-in realized vs paper that can evaporate on the next adverse mark? A +$50 book that is 100% realized is fundamentally different from the same headline that is 100% open-paper. Surfaced ONLY when DRAWING_DOWN / LEAKING_PAPER / PAPER_HEAVY, never filler when BANKED / BALANCED. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{realized_vs_unrealized_block}\n\n" if realized_vs_unrealized_block else "")
            + (f"PAPER TRADER — WATCHLIST COVERAGE (per-watchlist-ticker attention scan over the recent decision stream — which tickers has the bot stopped looking at? Every other panel is position-centric and never names an IGNORED ticker; this is the only block that surfaces opportunity cost from neglected names. Surfaced ONLY when STAGNANT / CONCENTRATED, never filler when DIVERSIFIED. Headline and the stale-ticker sample carry verbatim from the trader endpoint — restate, never re-derive):\n{watchlist_coverage_block}\n\n" if watchlist_coverage_block else "")
            + (f"PAPER TRADER — CONCENTRATION TRAJECTORY (the slope view of single-name exposure over the last N days — has top-1 weight been RISING, FALLING, or STEADY? Every other concentration surface is point-in-time and reads identically whether the book ramped 30%→65% over a week or jumped 0%→65% in one cycle. Surfaced ONLY when CONCENTRATION_SPIKE / RAMPING_UP / CONCENTRATED_STEADY, never filler when DECONCENTRATING / DIVERSIFIED / BALANCED. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{concentration_trajectory_block}\n\n" if concentration_trajectory_block else "")
            + (f"PAPER TRADER — STREAK (current win/loss run + historical extremes on closed round-trips — the behavioural-edge read no other surface carries. Surfaced ONLY when HOT_HAND / TILT_RISK, never filler when NEUTRAL or EMERGING; verdict is gated to ≥8 closed round-trips so a 3-trip 'streak' stays silent. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{streak_block}\n\n" if streak_block else "")
            + (f"PAPER TRADER — STANDING INTENTS (the FORWARD slice of the reasoning surface — conditional intents the bot itself stated in recent decisions' reasoning ('wait for cash session', 'rotating into LITE/LNOK', 'premature to dump') that are still STANDING within the freshness window without follow-up action. Every other reasoning block looks BACKWARD (vapor on FILLED, thesis-drift on opens, exit-intent on closed sells); none answers 'what did the bot SAY it would do next, that it has not yet done?'. Surfaced ONLY when STANDING_INTENTS / STALE_INTENTS, never filler when NO_INTENTS / NO_DATA. Headline AND each surfaced intent text pass verbatim from the trader endpoint — restate, never re-derive; the bot's own words, never a chat-side paraphrase. ``[stale]`` tag flags plans that aged past the freshness window without action):\n{standing_intents_block}\n\n" if standing_intents_block else "")
            + (f"PAPER TRADER — INTENT FOLLOWTHROUGH (the observational companion to STANDING INTENTS — of the actionable intents the bot stated, did it actually execute them? A bot that emits crisp 'wait for X, then buy Y' statements every cycle but never executes Y has perfect specificity on decision-vapor and zero followthrough; only this block catches the say-do gap. Surfaced ONLY when DRIFTING / ABANDONED, never filler when DISCIPLINED / NO_DATA / NO_RESOLVED. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{intent_followthrough_block}\n\n" if intent_followthrough_block else "")
            + (f"PAPER TRADER — OPPORTUNITY COST (the hindsight read on past HOLD-CASH / NO_DECISION sit-outs — when the bot sat in cash, did the top-news watchlist ticker run without it, or did sitting in cash dodge a drawdown? The chat already carries cash_pct snapshots and idle-opportunity (current drought) but neither answers 'did past cash discipline COST or SAVE alpha?'. Surfaced ONLY when MISSED_ALPHA / DEFENSIVE_WIN, never filler when NEUTRAL / NO_DATA. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{opportunity_cost_block}\n\n" if opportunity_cost_block else "")
            + (f"PAPER TRADER — IDLE OPPORTUNITY (the RIGHT-NOW companion to OPPORTUNITY COST — opportunity-cost grades PAST sit-outs by forward return (hindsight, takes hours to mature); idle-opportunity grades the CURRENT NO_DECISION drought by what HIGH-SCORE watchlist signals are arriving while the bot is dark. A decision-storm with the tape running is structurally different from a decision-storm with the tape quiet, and that difference is invisible to every other block until forward returns mature. A held-name (HELD) tag flags 'the bot was blind on news for a position WE OWN'. Surfaced ONLY when a missed signal exists (NO_DATA / NO_DROUGHT / OK-with-zero collapse to silence — never filler when the loop is filling or nothing is actually being missed). Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{idle_opportunity_block}\n\n" if idle_opportunity_block else "")
            + (f"PAPER TRADER — NO_DECISION CAUSE ATTRIBUTION (when the live trader is silent for a stretch, the operator's first follow-up to decision-paralysis is 'WHY is it empty?'. The trader endpoint buckets the recent NO_DECISION rows into host_saturated / cli_nonzero_rc / parse_failed / claude_timeout / claude_empty / blocked / unknown and emits a verbatim recommendation — host saturation requires reducing parallel Opus jobs, NOT a runner restart; a parse_failed cluster is a prompt-shape bug, not a host issue. Surfaced ONLY when DOMINANT (one bucket exceeds the threshold), never filler when NORMAL / MIXED (diffuse causes do not yield a specific recommendation). Headline carries verbatim from the trader endpoint AND already contains the recommendation — restate, never re-derive):\n{no_decision_reasons_block}\n\n" if no_decision_reasons_block else "")
            + (f"PAPER TRADER — ROUND-TRIP POSTMORTEM (per closed exit, the post-exit price drift verdict CORRECT / PREMATURE / MISSED_RUNNER / WHIPSAW / NEUTRAL — was the sell well-timed relative to the NEXT drift? Every existing realized-P&L surface — winner_autopsy, loser_autopsy, streak, scorecard — reduces a closed trip to a P&L number; only this block asks 'did the price keep running against the bot after the sell?'. A trade closed at -0.1% looks fine on track-record yet reads catastrophic if the name rallied +5% the hour after. Surfaced ONLY when ≥1 PREMATURE / MISSED_RUNNER / WHIPSAW trip exists, never filler when all CORRECT / NEUTRAL. Top-level headline AND the surfaced worst trip's own per-row headline both carry verbatim from the trader endpoint — restate, never re-derive):\n{round_trip_postmortem_block}\n\n" if round_trip_postmortem_block else "")
            + (f"PAPER TRADER — CASH DRAG (SPY-benchmarked $ cost of sitting in cash per rolling window — 'while you sat at avg cash $X over the last Yh, SPY ran +Z% — that's $W of beta you forfeited by being out'. Complements cash_pct snapshots, cash_redeployment latency, and opportunity_cost (signal-specific): this is the BENCHMARKED dollar-cost answer to 'is sitting in cash actually costing me?'. Surfaced ONLY when COSTLY_CASH, never filler when NEUTRAL / HELPFUL_CASH / INSUFFICIENT / NO_DATA. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{cash_drag_block}\n\n" if cash_drag_block else "")
            + (f"PAPER TRADER — PASSIVE-SIGNAL DENSITY (the smoking-gun read for 'engine idle during loud news' — decision-paralysis surfaces the FACT of a HOLD-only run; passive-signal-density discriminates whether the news during that run was QUIET (informed passive — correct silence) or LOUD (deafening silence — engine sat on its hands while a real news window was open). Surfaced ONLY when DEAFENING_SILENCE, never filler when INFORMED_PASSIVE / SIGNAL_RICH_PASSIVE / NO_PASSIVE_RUN / INSUFFICIENT / NO_DATA. Mirrors the trader-side Discord block (reporter._passive_signal_density_line) so the two surfaces never disagree on what is the alert. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{passive_signal_density_block}\n\n" if passive_signal_density_block else "")
            + (f"PAPER TRADER — NEWS-TO-TRADE LAG (is the bot actually reacting to fresh news, or is it consistently 2h+ behind? trade-attribution enumerates per-trade article precedence; this block compresses that to one reactivity verdict over recent FILLED trades. A book trading 2h+ behind the wire on leveraged ETFs has bled significant edge before the entry. Surfaced ONLY when DELAYED, never filler when REACTIVE_FAST / REACTIVE / NO_ATTRIBUTION / NO_DATA — 'unmeasurable' is not an alert. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{news_to_trade_lag_block}\n\n" if news_to_trade_lag_block else "")
            + (f"PAPER TRADER — CATALYST EXPIRY (per-open-position catalyst-class + age vs catalyst-type expiry window — a position opened on an earnings beat that has sat for 5 days is on a STALE thesis even if it's still green, because earnings beats price-in within ~2 days. thesis_drift verdicts each position on P/L-since-entry; hold_discipline flags losers overstayed; only this block tracks the *catalyst clock*. Selling at small green on a zombie thesis is rational; selling at -1% on an INTACT structural thesis is not. Surfaced ONLY when ZOMBIE_HOLDINGS, never filler when ALL_FRESH / STRUCTURAL_BOOK / MIXED_BOOK / NO_DATA. Headline + worst-zombie ticker/days/class carry verbatim from the trader endpoint — restate, never re-derive):\n{catalyst_expiry_block}\n\n" if catalyst_expiry_block else "")
            + (f"PAPER TRADER — TRADE ASYMMETRY (the payoff-ratio / disposition-effect diagnostic — are we cutting winners short while letting losers run? Every other realised-P&L block reduces closed trips to an aggregate or a per-bucket count; none expose the classic payoff-trap pathology where a high win-rate hides a negative expectancy (small wins, big losses). Surfaced ONLY when PAYOFF_TRAP / DISPOSITION_BLEED, never filler when EDGE_POSITIVE / FLAT / EMERGING / NO_DATA — the builder's own n≥20 gate keeps thin samples silent. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{trade_asymmetry_block}\n\n" if trade_asymmetry_block else "")
            + (f"PAPER TRADER — REBUY REGRET (the DOLLAR question on sell-then-rebuy hops: did the bot sell low and buy back higher? reentry-velocity grades CADENCE (CHURN_RISK / STABLE); round-trip-postmortem grades whether the SELL was well-timed against the next drift; only this block grades whether the actual BUY that followed came at a materially worse price. Surfaced ONLY when REGRETTING, never filler when SAVINGS / NET_NEUTRAL / NO_DATA / NO_REBUYS — re-entries that save money or net flat are not chat filler. Headline carries verbatim from the trader endpoint — restate, never re-derive):\n{rebuy_regret_block}\n\n" if rebuy_regret_block else "")
            + (f"PAPER TRADER — SCORER vs BOOK DISAGREEMENT (the meta question — does the bot's OWN ML (the 17-feature DecisionScorer that ML-GATE-HONESTY grades for OOS skill) currently agree with what it's holding? A HIGH-severity row means the live decision loop is sitting on a position the scorer would EXIT/TRIM — the dashboard already surfaces this panel but the chat had been blind to it. Surfaced ONLY when ≥1 HIGH-severity row exists (clamped off-distribution rows pre-filtered to avoid misrepresenting extrapolation as a real fight); ALIGNED-only / MEDIUM-only / scorer_trained=False collapse to silence. Worst row's ticker / scorer_verdict / last_action / scorer_pred_5d_pct pass through verbatim from the trader endpoint — restate, never re-derive):\n{scorer_book_disagreement_block}\n\n" if scorer_book_disagreement_block else "")
            + (f"PAPER TRADER — REPEAT-LOSER WATCH (chronic-pattern behavioural read — tickers where the bot has lost the last N closed round-trips IN A ROW on the same ticker. loser_autopsy / winner_autopsy / trade_asymmetry / streak each grade the bot in aggregate (which class of trade loses, payoff-trap, current W/L run); none answer the per-NAME chronic-pattern question 'have I lost the last N trips on ticker X in a row?'. A book that closes the same ticker for the third loss in a row is grinding the same setup against the same outcome — a behavioural blind spot every other surface aggregates away. Surfaced ONLY when state==REPEAT_LOSER, never filler when OK / NO_DATA. Headline + worst-offender fields carry verbatim from the trader endpoint — restate, never re-derive):\n{repeat_loser_block}\n\n" if repeat_loser_block else "")
            + (f"PAPER TRADER — EXIT-ONLY STREAK (consecutive SELLs since the last entry at the BOOK level — 'the last 6 fills were all SELLs; the engine is liquidating, not running the strategy'. /api/streak grades W/L on closed round-trips; /api/churn measures re-entry cadence; cash-drag measures idle-cash dollar cost. None surface the trade-DIRECTION sequence. A defensive-trim run preceding a market drop reads as DISCIPLINED on every backward block; the same run preceding a rip reads as PANIC — only this block names the structural fact in real time. Surfaced ONLY when DEFENSIVE_TRIM (≥3 consec exits) / DEFENSIVE_LIQUIDATION (≥6), never filler when MOST_RECENT_IS_ENTRY. Headline + run fields carry verbatim from the trader endpoint — restate, never re-derive):\n{exit_only_streak_block}\n\n" if exit_only_streak_block else "")
            + (f"PAPER TRADER — CATALYST-CLASS AUTOPSY (per-CATALYST-CLASS win-rate / PnL leaderboard over closed round-trips. trade_asymmetry surfaces the payoff trap; winner_autopsy / loser_autopsy bucket by ENTRY-class; per_ticker_skill is per-NAME edge. None answer the per-CLASS question: of the 9 catalyst classes (ML_ADVISOR, ANALYST_PT, TECHNICALS, EARNINGS_PLAY, MACRO, BREAKING_NEWS, PUNDIT, SECTOR_SYMPATHY, CONCENTRATION) which has biased my realised P&L UP (a class to weight INTO) or DOWN (a class to weight OUT OF)? Surfaces a structural class-level weight-allocation recommendation invisible to every other surface — answers 'should I lean more on the ML-advisor track and less on the pundit-takes track?' with realised-PnL data. Surfaced ONLY when state==STABLE AND (top_biased_winner OR top_biased_loser); NO_DATA / EMERGING / STABLE-but-no-bias collapse to silence (a class-leaderboard where no class has crossed both the sample-size gate AND the win-rate margin is interesting but not actionable). Headline + bias fields carry verbatim from the trader endpoint — restate, never re-derive):\n{catalyst_class_autopsy_block}\n\n" if catalyst_class_autopsy_block else "")
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
        # core.claude_cli handles auth/routing for the configured LLM CLI.
        convo_parts = [system_prompt, "\n\n--- Conversation ---"]
        for m in msgs:
            convo_parts.append(f"{m['role'].upper()}: {m['content']}")
        convo_parts.append("ASSISTANT:")
        prompt = "\n\n".join(convo_parts)

        response_text, llm_model, failed_models = _call_chat_llm(
            prompt, timeout=_CHAT_LLM_TIMEOUT_S
        )

        if not response_text:
            _logger().warning(
                "chat: all LLM backends unavailable; tried=%s",
                ",".join(failed_models) or "<none>",
            )
            return jsonify({
                "response": _chat_backend_unavailable_response(
                    user_msg, articles_ctx, paper_trader_block, failed_models
                ),
                "sources": [a["title"] for a in articles_ctx],
                "degraded": True,
                "failed_models": failed_models,
            })

        return jsonify({
            "response": response_text,
            "sources": [a["title"] for a in articles_ctx],
            "model": llm_model,
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
      --font-sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --font-mono: "SFMono-Regular", "Cascadia Mono", "JetBrains Mono", monospace;
      --font-display: var(--font-sans);
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
    <h1 class="h2 mb-0">Digital Intern</h1>
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
    const basis = j.capital_basis ?? j.starting_value ?? 1000;
    const pl = j.deposit_adjusted_pnl ?? (tv - basis);
    const plPct = j.deposit_adjusted_return_pct ?? ((pl / basis) * 100);
    const cls = pl >= 0 ? "pl-pos" : "pl-neg";
    sumDiv.innerHTML = `<span class="ticker">Portfolio</span> $${fmt(tv)} ` +
      `<span class="${cls}">${(pl>=0?'+':'')}${fmt(pl)} (${(pl>=0?'+':'')}${fmt(plPct)}%)</span> ` +
      `<span class="small-muted">vs $${fmt(basis)} capital basis · cash $${fmt(j.cash)}</span>`;
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
      <td><input aria-label="Ticker symbol for position ${i+1}" class="form-control form-control-sm p-1" style="font-size:11px;background:var(--bg-input);color:var(--text);border-color:var(--border-strong);width:70px" value="${p.ticker||""}" data-i="${i}" data-f="ticker"></td>
      <td><select aria-label="Asset type for position ${i+1}" class="form-select form-select-sm p-1" style="font-size:11px;background:var(--bg-input);color:var(--text);border-color:var(--border-strong);width:90px" data-i="${i}" data-f="type">
        ${["stock","etf_leveraged","etf","option"].map(t=>`<option${p.type===t?" selected":""}>${t}</option>`).join("")}
      </select></td>
      <td><input aria-label="Quantity for position ${i+1}" class="form-control form-control-sm p-1" style="font-size:11px;background:var(--bg-input);color:var(--text);border-color:var(--border-strong);width:70px" value="${p.qty??""}" data-i="${i}" data-f="qty"></td>
      <td><input aria-label="Average cost for position ${i+1}" class="form-control form-control-sm p-1" style="font-size:11px;background:var(--bg-input);color:var(--text);border-color:var(--border-strong);width:80px" value="${p.avg_cost??""}" data-i="${i}" data-f="avg_cost"></td>
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
      const raw = await r.text();
      let j = {};
      try {
        j = raw ? JSON.parse(raw) : {};
      } catch (parseErr) {
        const preview = (raw || '').slice(0, 160) || '<empty body>';
        throw new Error('upstream returned non-JSON: ' + preview);
      }
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
      --font-sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --font-mono: "SFMono-Regular", "Cascadia Mono", "JetBrains Mono", monospace;
      --font-display: var(--font-sans);
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
    .return-panel {
      margin: 12px 24px 0;
      padding: 12px 14px;
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      flex-shrink: 0;
    }
    .return-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 8px;
      font-size: 12px;
      color: var(--text-secondary);
    }
    .return-title {
      color: var(--text);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .return-stats {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      font-family: var(--font-mono);
      font-size: 12px;
    }
    .return-pos { color: var(--green); }
    .return-neg { color: var(--red); }
    .return-chart { position: relative; height: 180px; }
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
      .return-panel { margin: 10px 16px 0; padding: 10px 12px; }
      .return-chart { height: 150px; }
      .return-head { align-items: flex-start; flex-direction: column; gap: 4px; }
      .chat-wrap { padding: 14px 16px 88px; }
      .input-bar { padding: 12px 16px; margin-bottom: 64px; }
      .suggestions { padding: 12px 16px 4px; }
      button.send { min-height: 44px; min-width: 44px; }
      table { font-size: 12px; }
      th, td { padding: 8px 10px; }
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js" async></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
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
<section class="return-panel">
  <div class="return-head">
    <span class="return-title">Paper Trader Return Graph</span>
    <span class="return-stats">
      <span id="return-total">value —</span>
      <span id="return-basis">basis —</span>
      <span id="return-pct">return —</span>
    </span>
  </div>
  <div class="return-chart"><canvas id="return-chart"></canvas></div>
</section>
<div class="suggestions" id="suggestions"></div>
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
let returnChart = null;

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
    const raw = await r.text();
    let j = {};
    try {
      j = raw ? JSON.parse(raw) : {};
    } catch (parseErr) {
      const preview = (raw || '').slice(0, 160) || '<empty body>';
      throw new Error('upstream returned non-JSON: ' + preview);
    }
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

const SUGGESTIONS_FALLBACK = [
  "What's moving markets today?",
  "Nokia surge analysis",
  "Best opportunities right now?",
  "What should I watch in Asia overnight?"
];
function renderSuggestions(list) {
  const box = document.getElementById('suggestions');
  if (!box) return;
  const items = (Array.isArray(list) && list.length) ? list.slice(0, 4) : SUGGESTIONS_FALLBACK;
  box.innerHTML = '';
  for (const q of items) {
    const b = document.createElement('button');
    b.className = 'suggestion';
    b.dataset.q = q;
    b.textContent = q;
    box.appendChild(b);
  }
}

function money(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return '—';
  const x = Number(n);
  return (x < 0 ? '-$' : '$') + Math.abs(x).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
}
function signedPct(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return '—';
  const x = Number(n);
  return (x >= 0 ? '+' : '') + x.toFixed(2) + '%';
}
async function refreshReturnGraph() {
  const canvas = document.getElementById('return-chart');
  if (!canvas || typeof Chart === 'undefined') return;
  let j;
  try {
    const r = await fetch('/trader/api/equity-tail?limit=500', {cache: 'no-store'});
    if (!r.ok) return;
    j = await r.json();
  } catch (e) { return; }
  const rows = Array.isArray(j.equity) ? j.equity : [];
  if (rows.length < 2) return;
  const labels = rows.map(p => (p.timestamp || '').replace('T', ' ').slice(0, 16));
  const values = rows.map(p => {
    const pct = Number(p.deposit_adjusted_return_pct);
    if (Number.isFinite(pct)) return pct;
    const tv = Number(p.total_value), basis = Number(p.capital_basis || 1000);
    return basis > 0 ? (tv / basis - 1) * 100 : null;
  });
  const last = j.portfolio || rows[rows.length - 1] || {};
  const pct = Number(last.deposit_adjusted_return_pct);
  const pnl = Number(last.deposit_adjusted_pnl);
  const pctEl = document.getElementById('return-pct');
  document.getElementById('return-total').textContent = 'value ' + money(last.total_value);
  document.getElementById('return-basis').textContent = 'basis ' + money(last.capital_basis);
  pctEl.textContent = 'return ' + signedPct(pct);
  pctEl.className = (pct >= 0 ? 'return-pos' : 'return-neg');
  if (Number.isFinite(pnl)) pctEl.title = 'Deposit-adjusted P/L ' + money(pnl);
  if (returnChart) returnChart.destroy();
  returnChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Deposit-adjusted return %',
        data: values,
        borderColor: '#0acdff',
        backgroundColor: 'rgba(10,205,255,0.12)',
        pointRadius: 0,
        pointHoverRadius: 3,
        borderWidth: 2,
        tension: 0.25,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => 'return ' + signedPct(ctx.parsed.y),
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#8b929d', maxRotation: 0, autoSkip: true, maxTicksLimit: 7 },
          grid: { display: false },
        },
        y: {
          ticks: { color: '#8b929d', callback: v => signedPct(Number(v)) },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
      },
    },
  });
}
async function loadSuggestions() {
  try {
    const r = await fetch('/api/chat-suggestions', {cache: 'no-store'});
    const j = await r.json();
    renderSuggestions(j && j.suggestions);
  } catch (e) {
    renderSuggestions(null);
  }
}
renderSuggestions(SUGGESTIONS_FALLBACK);
document.addEventListener('DOMContentLoaded', loadSuggestions);
if (document.readyState !== 'loading') loadSuggestions();
document.addEventListener('DOMContentLoaded', refreshReturnGraph);
if (document.readyState !== 'loading') refreshReturnGraph();
setInterval(loadSuggestions, 10 * 60 * 1000);
setInterval(refreshReturnGraph, 15000);
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
    # Standalone runtime: serve via short-lived read-only DB connections.
    import sys
    sys.path.insert(0, str(BASE_DIR))
    run_server(None)
