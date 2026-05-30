"""Realized P&L bucketed by entry position size — is the desk Kelly-coherent?

``conviction_deployment`` asks the *forward* question: when news ai_score on
a ticker is high, does the bot deploy more capital? That endpoint scores the
sizing *intent*. It does NOT score the sizing *outcome* — whether those
larger entries actually produced better realized P&L.

This module is the matched-pair *backward* read. Every closed round-trip
gets bucketed by its entry-side dollar cost expressed as a percentage of
the book at entry (SMALL / MEDIUM / LARGE / MAX). Per bucket it reports
n_trips, win_rate_pct, total_pl_usd, avg_pl_pct, avg_hold_days, plus the
average actual entry_book_pct that landed in the bucket. The verdict
contrasts the big-bet buckets (LARGE + MAX) against the small-bet buckets
(SMALL + MEDIUM):

* ``KELLY_COHERENT``   — big-bet net P/L > 0 AND > small-bet net P/L
                         (the desk's biggest sizes are also its biggest
                         dollar winners)
* ``ANTI_KELLY``       — big-bet net P/L < 0 AND small-bet net P/L > 0
                         (the desk's small bets pay; its big bets lose —
                         sizing is structurally broken)
* ``BIG_BETS_NEUTRAL`` — big-bet net P/L > 0 but ≤ small-bet net P/L
                         (oversizing isn't paying off relative to the
                         distributed-bet path)
* ``ALL_BLEED``        — every populated bucket has net negative P/L
* ``EMERGING``         — total closed trips < ``min_for_verdict`` (4)
* ``NO_DATA``          — no closed round-trips

Distinct from its neighbours (AGENTS.md invariant #10, do not consolidate):

* ``conviction_deployment`` is the *sizing-intent* curve (catalyst score →
  size at entry). This module is the *sizing-outcome* curve (size at
  entry → realized $ on close).
* ``holding_period_distribution`` buckets by HOLD DURATION; this buckets
  by SIZE. A trip can be in the same hold bucket but a different size
  bucket — orthogonal axes.
* ``trade_asymmetry`` measures the disposition gap (winner-hold vs
  loser-hold). No size axis.
* ``track_record`` is the aggregate ledger with no per-bucket stratification.

Book-at-entry comes from the equity_curve sample at-or-immediately-before
the round-trip's ``opened_at`` timestamp. If equity_curve is unavailable
(empty list, malformed, no sample at-or-before the entry), the builder
falls back to the canonical $1000 INITIAL_CASH so a fresh-DB call still
emits a meaningful bucketing rather than degrading every row to SMALL.

Pure builder, never raises, never trains, never writes, never has a path
to ``_execute``. Errors on a single row degrade that row, never the verdict.

Run as a CLI::

    python3 -m paper_trader.analytics.position_size_pl_fit
"""
from __future__ import annotations

from datetime import datetime
from statistics import median


# Bucket edges as fractions of book-at-entry. Closed at the upper edge on
# the lower bucket so 25.0% lands in MEDIUM, not SMALL (the standard
# half-open-on-the-right convention for quantile bins).
_BUCKET_EDGES = (
    ("SMALL", 0.0, 0.25),
    ("MEDIUM", 0.25, 0.50),
    ("LARGE", 0.50, 0.80),
    ("MAX", 0.80, float("inf")),
)
_BUCKETS = tuple(b[0] for b in _BUCKET_EDGES)
_BIG_BUCKETS = ("LARGE", "MAX")
_SMALL_BUCKETS = ("SMALL", "MEDIUM")

# Fallback book value when equity_curve has no sample at or before a trip's
# entry time — matches store.INITIAL_CASH so a fresh-DB caller doesn't get
# every trip classified as MAX by accident.
_FALLBACK_BOOK_USD = 1000.0


def _parse_ts(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _book_at(equity_curve, opened_at_dt):
    """Return the ``total_value`` from the latest equity_curve sample whose
    timestamp is ≤ ``opened_at_dt``. ``None`` if no covering sample exists.

    ``equity_curve`` is assumed ascending by timestamp (``store.equity_curve``
    contract). The walk is linear; the typical limit=500 sample window is
    cheap. Robust to malformed rows (missing total_value, unparseable ts) —
    they're skipped, never raise.
    """
    if not equity_curve or opened_at_dt is None:
        return None
    best = None
    for sample in equity_curve:
        if not isinstance(sample, dict):
            continue
        sample_ts = _parse_ts(sample.get("timestamp"))
        if sample_ts is None:
            continue
        if sample_ts > opened_at_dt:
            # Past the entry — equity_curve is ascending, so we're done.
            break
        try:
            tv = float(sample.get("total_value"))
        except (TypeError, ValueError):
            continue
        if tv > 0:
            best = tv
    return best


def _classify_size(entry_book_pct):
    """Return the bucket label for ``entry_book_pct`` (fraction, e.g. 0.42).

    None / non-finite degrade to ``SMALL`` (mirrors the safe-default
    convention of round_trip_postmortem — unknown rows do not poison the
    verdict by landing in MAX)."""
    if entry_book_pct is None:
        return "SMALL"
    try:
        v = float(entry_book_pct)
    except (TypeError, ValueError):
        return "SMALL"
    if v != v or v < 0:  # NaN or impossible negative
        return "SMALL"
    for label, lo, hi in _BUCKET_EDGES:
        if lo <= v < hi:
            return label
    return "MAX"


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def annotate_round_trip(rt, equity_curve):
    """Enrich one round-trip dict with ``entry_book_pct`` and ``size_bucket``.

    Pure; on any field-level error the returned dict still ships a fallback
    classification (SMALL) so the row participates in the rollup."""
    if not isinstance(rt, dict):
        return None
    cost = _safe_float(rt.get("cost"))
    opened_at = _parse_ts(rt.get("opened_at"))
    book = _book_at(equity_curve, opened_at) if opened_at else None
    if book is None or book <= 0:
        book = _FALLBACK_BOOK_USD
    pct = cost / book if cost > 0 and book > 0 else 0.0
    bucket = _classify_size(pct)
    return {
        **rt,
        "entry_book_pct": round(pct, 4),
        "size_bucket": bucket,
    }


def _bucket_stats(rows):
    """Per-bucket rollup. Same shape as exit_trigger_pl_mix._bucket_stats so
    a downstream consumer can render the two endpoints side-by-side."""
    rows = [r for r in rows if isinstance(r, dict)]
    n = len(rows)
    wins = sum(1 for r in rows if _safe_float(r.get("realized_pl")) > 0)
    losses = sum(1 for r in rows if _safe_float(r.get("realized_pl")) < 0)
    decided = wins + losses
    total_pl = round(sum(_safe_float(r.get("realized_pl")) for r in rows), 2)
    total_cost = round(sum(_safe_float(r.get("cost")) for r in rows), 2)
    win_rate = round(100.0 * wins / decided, 2) if decided else None
    avg_pl_pct = (round(total_pl / total_cost * 100.0, 2)
                  if total_cost > 1e-9 else None)
    holds = [r["hold_days"] for r in rows
             if r.get("hold_days") is not None]
    avg_hold = round(sum(holds) / len(holds), 4) if holds else None
    med_hold = round(median(holds), 4) if holds else None
    pcts = [r["entry_book_pct"] for r in rows
            if r.get("entry_book_pct") is not None]
    avg_pct = round(sum(pcts) / len(pcts) * 100.0, 2) if pcts else None
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "flat": n - wins - losses,
        "win_rate_pct": win_rate,
        "total_pl_usd": total_pl,
        "total_cost_usd": total_cost,
        "avg_pl_pct": avg_pl_pct,
        "avg_hold_days": avg_hold,
        "median_hold_days": med_hold,
        "avg_entry_book_pct": avg_pct,
    }


def _verdict(buckets, total_n, min_for_verdict=4):
    """Verdict ladder over big-bucket vs small-bucket net P/L.

    See module docstring for the ladder definition.
    """
    if total_n == 0:
        return "NO_DATA"
    if total_n < min_for_verdict:
        return "EMERGING"
    big = sum(buckets[b]["total_pl_usd"] for b in _BIG_BUCKETS)
    small = sum(buckets[b]["total_pl_usd"] for b in _SMALL_BUCKETS)
    populated = [b for b in _BUCKETS if buckets[b]["n"] > 0]
    if (populated and
            all(buckets[b]["total_pl_usd"] < 0 for b in populated)):
        return "ALL_BLEED"
    if big > 0 and big > small:
        return "KELLY_COHERENT"
    if big < 0 and small > 0:
        return "ANTI_KELLY"
    if big > 0:
        return "BIG_BETS_NEUTRAL"
    # Big-bet bucket non-positive, small-bet bucket non-positive — rare
    # mixed shape (e.g. all big-bets flat, all small-bets flat).
    return "BIG_BETS_NEUTRAL"


def _headline(buckets, verdict, total_n):
    """One-line summary contrasting big-bet $ with small-bet $.

    Always identifies which side carried the desk (or both, when neither did).
    """
    if total_n == 0:
        return "No closed round-trips on record yet."
    big = sum(buckets[b]["total_pl_usd"] for b in _BIG_BUCKETS)
    small = sum(buckets[b]["total_pl_usd"] for b in _SMALL_BUCKETS)
    n_big = sum(buckets[b]["n"] for b in _BIG_BUCKETS)
    n_small = sum(buckets[b]["n"] for b in _SMALL_BUCKETS)
    if verdict == "EMERGING":
        return (f"Emerging — {total_n} closed round-trip"
                f"{'s' if total_n != 1 else ''}; verdict withheld until "
                f"≥4. Big-bet (LARGE+MAX, n={n_big}) ${big:+.2f}; "
                f"small-bet (SMALL+MEDIUM, n={n_small}) ${small:+.2f}.")
    label = verdict.replace("_", " ").title()
    return (
        f"{label} — big-bet (LARGE+MAX, n={n_big}) ${big:+.2f}; "
        f"small-bet (SMALL+MEDIUM, n={n_small}) ${small:+.2f} "
        f"across {total_n} closed round-trip"
        f"{'s' if total_n != 1 else ''}."
    )


def build(trades, equity_curve=None, min_for_verdict=4):
    """Top-level: take raw trade rows + equity_curve, return the size-bucket
    rollup envelope.

    Parameters
    ----------
    trades : iterable of dict
        Output of ``store.recent_trades(limit=N)`` (append-only).
    equity_curve : iterable of dict or None
        Output of ``store.equity_curve(limit=M)`` — used to look up the
        ``total_value`` at each round-trip's entry timestamp. ``None`` or
        empty falls back to ``_FALLBACK_BOOK_USD = $1000`` per row.
    min_for_verdict : int
        Floor below which the verdict reports ``EMERGING`` rather than a
        directional read. Mirrors exit_trigger_pl_mix's 4-trip floor.

    Returns
    -------
    dict
        ``{verdict, headline, buckets, n_round_trips, min_for_verdict,
           fallback_book_used_count}``. ``buckets`` is a dict of
        ``{SMALL, MEDIUM, LARGE, MAX}`` → bucket-stats dict.
        Never raises.
    """
    # Late import — avoids a top-level circular if anything in the analytics
    # package ever imports from this module via __init__.
    from .round_trips_derived import derive_round_trips

    round_trips = derive_round_trips(trades)
    ec = list(equity_curve) if equity_curve else []
    annotated = []
    fallback_count = 0
    for rt in round_trips:
        ar = annotate_round_trip(rt, ec)
        if ar is None:
            continue
        annotated.append(ar)
        # Track how many rows had to use the fallback book — useful for
        # diagnostics. Detection: the entry sample lookup returned None
        # AND the trip has a usable opened_at (otherwise the fallback is
        # unavoidable, not a defect).
        opened = _parse_ts(rt.get("opened_at"))
        if opened is not None and _book_at(ec, opened) is None:
            fallback_count += 1

    by_bucket = {b: [] for b in _BUCKETS}
    for r in annotated:
        by_bucket[r["size_bucket"]].append(r)
    buckets = {b: _bucket_stats(by_bucket[b]) for b in _BUCKETS}
    total_n = len(annotated)
    verdict = _verdict(buckets, total_n, min_for_verdict=min_for_verdict)
    return {
        "verdict": verdict,
        "headline": _headline(buckets, verdict, total_n),
        "buckets": buckets,
        "n_round_trips": total_n,
        "min_for_verdict": min_for_verdict,
        "fallback_book_used_count": fallback_count,
    }


if __name__ == "__main__":  # pragma: no cover
    import json
    from ..store import get_store
    store = get_store()
    trades = store.recent_trades(limit=5000)
    eq = store.equity_curve(limit=500)
    out = build(trades, equity_curve=eq)
    print(json.dumps(out, indent=2, default=str))
