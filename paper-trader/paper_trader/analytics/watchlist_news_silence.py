"""Watchlist news silence — for each WATCHLIST ticker, how recently did
live news touch it, and at what signal grade?

The trader carries multiple endpoints that watch the *held* book's news
flow (digital-intern's ``/api/held-news-silence`` is the canonical
held-only surface; ``position_news_cooldown`` is its trader-side mirror;
``today_action_tape`` summarises *trades*-from-news). None of them
answer the *complementary* question:

    "Of the 47 tickers Opus is *allowed to consider*, how many had no
    live news in the last 24h? Which are mention-storming right now?
    Where are the blind spots in the universe I'm asking Opus to pick
    from?"

A blind universe is the structural risk that makes ``no_decision``
forensics so hard: Opus sees the same WATCHLIST prices every cycle but
the news context behind those prices is uneven. A clean coverage map
lets the operator see "Opus is being asked to choose between AMD (38
articles, max_score 8.5) and AMAT (zero articles, never seen) and the
prompt makes them look equally available." The bot has no way to know
whether silence means "nothing happened" or "the collector failed for
this ticker".

Pure & offline builder. The endpoint owns the article-DB scan (a
single bulk query, same shape as ``signals.ticker_sentiments``) and
passes a per-ticker summary dict in. The builder classifies and
verdicts. Identical ``_safe`` discipline to ``persona_book_fit`` /
``inverse_pair_conflict``: malformed input degrades to a deterministic
``NO_DATA`` verdict, never raises.

Advisory only. It mints no directive, imposes no cap, and has no path
to ``_execute()`` (AGENTS.md invariants #2 / #12).

Per-ticker classification (recency / volume buckets):

| Bucket    | Trigger                                                                   |
|-----------|---------------------------------------------------------------------------|
| ``SILENT``| Zero live articles in the lookback window — the universe blind spot       |
| ``STALE`` | At least one article but newest is older than ``STALE_HOURS_FLOOR``       |
| ``LIVE``  | Newest article within ``STALE_HOURS_FLOOR`` and ``n_in_window < HOT_N``   |
| ``HOT``   | Newest within window AND ``n_in_window ≥ HOT_N`` — mention storm           |

Universe-level verdict (book-wide rollup):

| Verdict           | Trigger                                                          |
|-------------------|------------------------------------------------------------------|
| ``NO_DATA``       | Empty watchlist or every entry malformed                         |
| ``BLIND_UNIVERSE``| Silent fraction ≥ ``BLIND_UNIVERSE_PCT_FLOOR`` (default 50%)     |
| ``SPARSE_COVERAGE``| Silent fraction in ``[SPARSE_COVERAGE_PCT_FLOOR, BLIND_…)``     |
| ``WELL_COVERED``  | Silent fraction below ``SPARSE_COVERAGE_PCT_FLOOR``               |
"""
from __future__ import annotations

from typing import Any

# Default lookback for the per-ticker classification. The endpoint should
# read the same window (the SQL scan and the builder agree).
DEFAULT_HOURS = 24

# A ticker's newest article older than this is classified ``STALE``
# even if some article exists in the window.
STALE_HOURS_FLOOR = 12.0

# Articles-in-window ≥ this flips ``LIVE`` → ``HOT``.
HOT_N = 8

# Signal-grade article gate (``ai_score`` threshold). The article-DB scan
# can apply this server-side; the builder also accepts the unfiltered
# count for context. Mirror of the ``signals.get_top_signals`` default
# ``min_score`` to keep cross-surface parity.
SIGNAL_GRADE_SCORE = 4.0

# Universe-level verdict thresholds (silent fraction). Tuned so a typical
# 47-name watchlist with 10 silent reads ``WELL_COVERED`` (21% silent),
# 17 silent reads ``SPARSE_COVERAGE`` (36%), 25 silent ``BLIND_UNIVERSE``
# (53%). Calibrated against the actual live watchlist composition.
BLIND_UNIVERSE_PCT_FLOOR = 50.0
SPARSE_COVERAGE_PCT_FLOOR = 20.0

# Below this many evaluable watchlist tickers the rollup verdict is
# withheld (small-watchlist edge case where 1-of-3 silent reads "33%
# silent" but is one collector hiccup). Mirrors the trade_asymmetry /
# add_discipline EMERGING / STABLE precedent.
MIN_EVALUABLE_TICKERS = 5


def _safe_int(x, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_float_or_none(x):
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _classify_ticker(summary: dict) -> str:
    """Returns one of ``SILENT`` / ``STALE`` / ``LIVE`` / ``HOT``.

    Pure on the summary shape: ``n_in_window`` (int), ``hours_since_last``
    (float-or-None). Treats malformed/missing entries as ``SILENT`` — the
    operator-actionable default (no evidence of coverage).
    """
    n = _safe_int(summary.get("n_in_window"))
    hsl = _safe_float_or_none(summary.get("hours_since_last"))
    if n <= 0 or hsl is None:
        return "SILENT"
    if hsl > STALE_HOURS_FLOOR:
        return "STALE"
    if n >= HOT_N:
        return "HOT"
    return "LIVE"


def build_watchlist_news_silence(
    watchlist: Any,
    per_ticker: Any = None,
    *,
    hours: int = DEFAULT_HOURS,
    hot_n: int = HOT_N,
    stale_hours_floor: float = STALE_HOURS_FLOOR,
    blind_universe_pct_floor: float = BLIND_UNIVERSE_PCT_FLOOR,
    sparse_coverage_pct_floor: float = SPARSE_COVERAGE_PCT_FLOOR,
    min_evaluable: int = MIN_EVALUABLE_TICKERS,
) -> dict:
    """Pure builder. ``watchlist`` is an iterable of ticker strings;
    ``per_ticker`` is a dict keyed by uppercase ticker with values of
    shape ``{n_in_window, n_signal_grade, max_score, hours_since_last,
    last_seen_iso}`` (any field may be missing — defaults applied).

    The endpoint must produce ``per_ticker`` via a single bulk scan of
    ``articles.db`` (mirroring ``signals.ticker_sentiments``); this
    builder never touches I/O.
    """
    tickers_clean: list[str] = []
    seen: set[str] = set()
    for raw in watchlist or []:
        if not isinstance(raw, str):
            continue
        t = raw.strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        tickers_clean.append(t)

    per_ticker = per_ticker if isinstance(per_ticker, dict) else {}

    rows: list[dict] = []
    bucket_counts = {"SILENT": 0, "STALE": 0, "LIVE": 0, "HOT": 0}
    for t in tickers_clean:
        summary = per_ticker.get(t) or {}
        if not isinstance(summary, dict):
            summary = {}
        n = max(0, _safe_int(summary.get("n_in_window")))
        n_sig = max(0, _safe_int(summary.get("n_signal_grade")))
        max_sc = _safe_float(summary.get("max_score"))
        hsl = _safe_float_or_none(summary.get("hours_since_last"))
        last_seen = summary.get("last_seen_iso") if isinstance(summary.get("last_seen_iso"), str) else None
        bucket = _classify_ticker(
            {"n_in_window": n, "hours_since_last": hsl}
        )
        bucket_counts[bucket] += 1
        rows.append({
            "ticker": t,
            "bucket": bucket,
            "n_in_window": n,
            "n_signal_grade": n_sig,
            "max_score": round(max_sc, 2) if max_sc else 0.0,
            "hours_since_last": round(hsl, 2) if hsl is not None else None,
            "last_seen_iso": last_seen,
        })

    # Sort: SILENT first (so the blind spots lead the report), then
    # by newest-last-seen DESC (HOT/LIVE storms surface after silence
    # block, freshest first). Deterministic — ties broken by ticker.
    bucket_order = {"SILENT": 0, "STALE": 1, "HOT": 2, "LIVE": 3}
    rows.sort(key=lambda r: (
        bucket_order.get(r["bucket"], 9),
        r["hours_since_last"] if r["hours_since_last"] is not None else 1e9,
        r["ticker"],
    ))

    n_total = len(rows)
    n_silent = bucket_counts["SILENT"]
    n_stale = bucket_counts["STALE"]
    n_live = bucket_counts["LIVE"]
    n_hot = bucket_counts["HOT"]
    silent_pct = (n_silent / n_total * 100.0) if n_total else 0.0

    if n_total < min_evaluable:
        verdict = "NO_DATA"
        headline = (
            f"NO_DATA — watchlist has {n_total} evaluable tickers "
            f"(need ≥{min_evaluable}); coverage verdict withheld"
        )
    elif silent_pct >= blind_universe_pct_floor:
        verdict = "BLIND_UNIVERSE"
        headline = (
            f"BLIND_UNIVERSE — {n_silent}/{n_total} watchlist tickers "
            f"({silent_pct:.0f}%) had zero live articles in the last {hours}h"
        )
    elif silent_pct >= sparse_coverage_pct_floor:
        verdict = "SPARSE_COVERAGE"
        headline = (
            f"SPARSE_COVERAGE — {n_silent}/{n_total} watchlist tickers "
            f"({silent_pct:.0f}%) silent in the last {hours}h; "
            f"{n_hot} mention-storms"
        )
    else:
        verdict = "WELL_COVERED"
        headline = (
            f"WELL_COVERED — {n_silent}/{n_total} silent ({silent_pct:.0f}%); "
            f"{n_hot} hot, {n_live} live"
        )

    # Top mention-storms (HOT bucket) — concise so the chat enrichment
    # has a stable, deterministic shortlist to compose into a one-line
    # detail block. Capped at 5; ordered by ``n_in_window`` DESC then
    # ``max_score`` DESC, ties broken by ticker.
    hot_storms = sorted(
        (r for r in rows if r["bucket"] == "HOT"),
        key=lambda r: (-r["n_in_window"], -r["max_score"], r["ticker"]),
    )[:5]

    # Silent tickers — capped at 10. Ordered alphabetically so the list
    # is stable cycle-to-cycle (rather than re-ordering on every refresh).
    silent_tickers = sorted(
        r["ticker"] for r in rows if r["bucket"] == "SILENT"
    )[:10]

    return {
        "verdict": verdict,
        "headline": headline,
        "hours": hours,
        "n_total": n_total,
        "n_silent": n_silent,
        "n_stale": n_stale,
        "n_live": n_live,
        "n_hot": n_hot,
        "silent_pct": round(silent_pct, 2),
        "bucket_counts": bucket_counts,
        "thresholds": {
            "stale_hours_floor": stale_hours_floor,
            "hot_n": hot_n,
            "blind_universe_pct_floor": blind_universe_pct_floor,
            "sparse_coverage_pct_floor": sparse_coverage_pct_floor,
            "min_evaluable_tickers": min_evaluable,
            "signal_grade_score": SIGNAL_GRADE_SCORE,
        },
        "silent_tickers": silent_tickers,
        "hot_storms": [
            {"ticker": r["ticker"], "n_in_window": r["n_in_window"],
             "max_score": r["max_score"]}
            for r in hot_storms
        ],
        "by_ticker": rows,
    }
