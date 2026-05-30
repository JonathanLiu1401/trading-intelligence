"""Realized P&L bucketed by SPY tape direction during the holding period.

Every closed round-trip lived through some segment of the tape — sometimes
the broader market rose during the hold (TAILWIND), sometimes it fell
(HEADWIND), sometimes it was flat (FLAT). Whether the desk made money
during HEADWIND segments is the textbook beta-vs-alpha test: a book that
profits only on TAILWIND days is riding market beta, not earning alpha.

Adjacent reads don't answer this question:

* ``portfolio_beta`` / ``portfolio_beta_drift`` measure CURRENT exposure
  to SPY — point-in-time, not realized-outcome.
* ``risk_adjusted_returns`` / ``benchmark`` compare aggregate book return
  to SPY — no per-trip stratification.
* ``sector_pl_history`` slices realized $ by SECTOR (which names), not
  by REGIME (what the tape did during the trade).
* ``exit_trigger_pl_mix`` slices by EXIT MECHANISM (HARD_SL/HARD_TP/
  discretionary), not by market backdrop.

This module is the matched-pair tape-fit slice. For each closed round-trip
it looks up the SPY price at ``opened_at`` and ``closed_at`` from the
equity_curve (which already records ``sp500_price`` at every cycle), then
buckets by the during-hold % change:

* ``TAILWIND``  — SPY rose by more than ``_TAPE_BAND_PCT`` (0.3%) during the hold
* ``HEADWIND``  — SPY fell by more than ``_TAPE_BAND_PCT`` during the hold
* ``FLAT``      — SPY moved within ±``_TAPE_BAND_PCT`` during the hold
* ``UNKNOWN``   — SPY price was missing at one or both endpoints (rare —
                  the first cycles after fresh-DB startup before benchmark_sp500
                  populates the column, or yfinance was down)

Per bucket it emits the standard rollup (n_trips, n_wins, n_losses,
win_rate_pct, total_pl_usd, avg_pl_pct, avg_hold_days) plus the average
SPY move during those trips (so the operator can see the "how much tape
did this bucket actually digest" detail).

Verdict ladder (with min_for_verdict=4 floor):

* ``ALPHA_VS_TAPE``      — HEADWIND bucket net P/L > 0 AND book is
                            net positive (proves edge is not pure beta)
* ``RIDING_BETA``        — TAILWIND bucket net P/L > 0 AND HEADWIND bucket
                            net P/L < 0 (typical retail pattern)
* ``FIGHTING_TAPE``      — HEADWIND bucket net P/L > 0 AND TAILWIND bucket
                            net P/L ≤ 0 (rare; counter-trend edge)
* ``TAPE_TRAPPED``       — both directional buckets net negative
* ``TAPE_NEUTRAL``       — both directional buckets within ±$1 of zero
                            (effectively no tape sensitivity)
* ``EMERGING``           — n < min_for_verdict
* ``NO_DATA``            — n = 0

Pure builder, never raises, never trains, never writes, never has a path
to ``_execute``. A trip with missing SPY context degrades to ``UNKNOWN``
bucket — never poisons the verdict.

Run as a CLI::

    python3 -m paper_trader.analytics.tape_fit_pl
"""
from __future__ import annotations

from datetime import datetime
from statistics import median


# SPY % move band that separates TAILWIND/HEADWIND from FLAT. 0.30% is
# tight enough to capture meaningful drift on a single-day hold but wide
# enough that intraday noise on multi-day holds doesn't constantly flip
# the classification.
_TAPE_BAND_PCT = 0.30

# Tape-direction buckets. UNKNOWN sits outside the directional triplet so
# the verdict ladder never reads a missing SPY context as evidence either
# way; populated UNKNOWN rows still ship in the envelope for diagnostics.
_DIRECTIONAL = ("TAILWIND", "HEADWIND", "FLAT")
_BUCKETS = _DIRECTIONAL + ("UNKNOWN",)


def _parse_ts(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _spy_at(equity_curve, target_dt):
    """Return ``sp500_price`` from the equity_curve sample with the smallest
    absolute timestamp delta to ``target_dt`` AND covering ts. Two-sided
    nearest-neighbour (within an open-cycle window of 60s on each side); a
    daily-close trip looks up the post-close mark, an intraday open looks
    up the next-cycle mark, both with no per-side bias.

    ``None`` when no sample on either side carries a non-null sp500_price.
    """
    if not equity_curve or target_dt is None:
        return None
    best = None
    best_dist = None
    for sample in equity_curve:
        if not isinstance(sample, dict):
            continue
        sp = sample.get("sp500_price")
        if sp is None:
            continue
        try:
            sp_f = float(sp)
        except (TypeError, ValueError):
            continue
        if sp_f <= 0:
            continue
        ts = _parse_ts(sample.get("timestamp"))
        if ts is None:
            continue
        dist = abs((ts - target_dt).total_seconds())
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = sp_f
    return best


def _classify_tape(spy_change_pct):
    """Map a during-hold SPY % change to one of TAILWIND / HEADWIND / FLAT.

    ``None`` ⇒ UNKNOWN. NaN-safe (NaN > 0 is False on both sides, so a
    NaN would land in HEADWIND otherwise — explicit guard here)."""
    if spy_change_pct is None:
        return "UNKNOWN"
    try:
        v = float(spy_change_pct)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if v != v:  # NaN
        return "UNKNOWN"
    if v > _TAPE_BAND_PCT:
        return "TAILWIND"
    if v < -_TAPE_BAND_PCT:
        return "HEADWIND"
    return "FLAT"


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def annotate_round_trip(rt, equity_curve):
    """Enrich one round-trip dict with ``spy_open``, ``spy_close``,
    ``spy_change_pct``, and ``tape_bucket``. Pure; on any field-level
    error the row still ships with bucket=``UNKNOWN``."""
    if not isinstance(rt, dict):
        return None
    opened_at = _parse_ts(rt.get("opened_at"))
    closed_at = _parse_ts(rt.get("closed_at"))
    spy_o = _spy_at(equity_curve, opened_at) if opened_at else None
    spy_c = _spy_at(equity_curve, closed_at) if closed_at else None
    change_pct = None
    if spy_o is not None and spy_c is not None and spy_o > 0:
        change_pct = round((spy_c - spy_o) / spy_o * 100.0, 4)
    bucket = _classify_tape(change_pct)
    return {
        **rt,
        "spy_open": spy_o,
        "spy_close": spy_c,
        "spy_change_pct": change_pct,
        "tape_bucket": bucket,
    }


def _bucket_stats(rows):
    """Per-bucket rollup including the average SPY % change observed."""
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
    spy_moves = [r["spy_change_pct"] for r in rows
                 if r.get("spy_change_pct") is not None]
    avg_spy = (round(sum(spy_moves) / len(spy_moves), 4)
               if spy_moves else None)
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
        "avg_spy_change_pct": avg_spy,
    }


def _verdict(buckets, total_directional_n, total_pl, min_for_verdict=4):
    """Verdict ladder over directional buckets (UNKNOWN never gates a
    directional verdict).

    See module docstring for ladder definition.
    """
    if total_directional_n == 0:
        # No SPY context on any closed trip yet — likely a fresh equity_curve.
        return "NO_DATA"
    if total_directional_n < min_for_verdict:
        return "EMERGING"
    head = buckets["HEADWIND"]["total_pl_usd"]
    tail = buckets["TAILWIND"]["total_pl_usd"]
    if head > 0 and total_pl > 0:
        return "ALPHA_VS_TAPE"
    if tail > 0 and head < 0:
        return "RIDING_BETA"
    if head > 0 and tail <= 0:
        return "FIGHTING_TAPE"
    if head < 0 and tail < 0:
        return "TAPE_TRAPPED"
    if abs(head) <= 1.0 and abs(tail) <= 1.0:
        return "TAPE_NEUTRAL"
    # Mixed shape that doesn't cleanly land in any of the above (e.g. head=0,
    # tail>0 with overall book negative). Surface as RIDING_BETA-flavoured —
    # the desk's positive bucket is tailwind, and the negative side isn't
    # headwind itself, so the safest read is "leaning on tape direction".
    return "RIDING_BETA"


def _headline(buckets, verdict, total_n, total_directional_n):
    """One-line summary naming the headwind vs tailwind net $ and the
    UNKNOWN diagnostic when it is non-trivial."""
    if total_n == 0:
        return "No closed round-trips on record yet."
    head = buckets["HEADWIND"]
    tail = buckets["TAILWIND"]
    flat = buckets["FLAT"]
    unknown = buckets["UNKNOWN"]
    parts = [
        f"TAILWIND {tail['n']} trips ${tail['total_pl_usd']:+.2f}",
        f"HEADWIND {head['n']} trips ${head['total_pl_usd']:+.2f}",
        f"FLAT {flat['n']} trips ${flat['total_pl_usd']:+.2f}",
    ]
    if unknown["n"]:
        parts.append(f"UNKNOWN {unknown['n']} trips (SPY context missing)")
    if verdict == "EMERGING":
        return (
            f"Emerging — {total_directional_n} directional closed trip"
            f"{'s' if total_directional_n != 1 else ''}; verdict withheld "
            f"until ≥4. " + " · ".join(parts) + "."
        )
    label = verdict.replace("_", " ").title()
    return f"{label} — " + " · ".join(parts) + "."


def build(trades, equity_curve=None, min_for_verdict=4):
    """Top-level: take raw trade rows + equity_curve, return the tape-bucket
    rollup envelope.

    Parameters
    ----------
    trades : iterable of dict
        Output of ``store.recent_trades(limit=N)``.
    equity_curve : iterable of dict or None
        Output of ``store.equity_curve(limit=M)``. The ``sp500_price``
        column drives the per-trip lookup. ``None`` or empty causes every
        directional bucket to be empty and verdict to be ``NO_DATA``.
    min_for_verdict : int
        Floor below which the verdict reports ``EMERGING``.

    Returns
    -------
    dict
        ``{verdict, headline, buckets, n_round_trips, n_directional,
           min_for_verdict, tape_band_pct}``.
        ``buckets`` is a dict of ``{TAILWIND, HEADWIND, FLAT, UNKNOWN}`` →
        bucket-stats dict. Never raises.
    """
    from .round_trips_derived import derive_round_trips

    round_trips = derive_round_trips(trades)
    ec = list(equity_curve) if equity_curve else []
    annotated = []
    for rt in round_trips:
        ar = annotate_round_trip(rt, ec)
        if ar is not None:
            annotated.append(ar)

    by_bucket = {b: [] for b in _BUCKETS}
    for r in annotated:
        by_bucket[r["tape_bucket"]].append(r)
    buckets = {b: _bucket_stats(by_bucket[b]) for b in _BUCKETS}

    n_total = len(annotated)
    n_directional = sum(buckets[b]["n"] for b in _DIRECTIONAL)
    total_pl = sum(buckets[b]["total_pl_usd"] for b in _DIRECTIONAL)
    verdict = _verdict(buckets, n_directional, total_pl,
                       min_for_verdict=min_for_verdict)
    return {
        "verdict": verdict,
        "headline": _headline(buckets, verdict, n_total, n_directional),
        "buckets": buckets,
        "n_round_trips": n_total,
        "n_directional": n_directional,
        "min_for_verdict": min_for_verdict,
        "tape_band_pct": _TAPE_BAND_PCT,
    }


if __name__ == "__main__":  # pragma: no cover
    import json
    from ..store import get_store
    store = get_store()
    trades = store.recent_trades(limit=5000)
    eq = store.equity_curve(limit=500)
    out = build(trades, equity_curve=eq)
    print(json.dumps(out, indent=2, default=str))
