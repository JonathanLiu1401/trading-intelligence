"""Post-exit drift analysis for closed round-trips.

`build_round_trips` (analytics/round_trips.py) is the canonical aggregator of
realised P&L — it says WHAT closed, when, for how much. Every realised-P&L
surface (track-record, churn, streak, winner/loser autopsy, trade-asymmetry)
reduces its output to summary stats.

What none of them ask is the operator-facing follow-up: *was the exit good?*
Selling DRAM at -0.18% looks fine on paper-trader's loss-tally; it looks
catastrophic if DRAM rallied +5% in the hour after the sell. The current
analytics surface has no post-exit price signal — once a round-trip closes
it's reduced to a P&L number and disappears from the learning loop.

This module fills that gap with a per-RT verdict that hindsight makes
falsifiable:

  * **CORRECT** — post-exit price fell at least ``-CORRECT_MAX_DRIFT_PCT``.
    The exit captured the local high (or avoided further loss).
  * **PREMATURE** — post-exit price rose between ``PREMATURE_MIN_DRIFT_PCT``
    and ``MISSED_RUNNER_MIN_DRIFT_PCT``. Bot sold and the move continued
    against the exit direction; could have sat tighter.
  * **MISSED_RUNNER** — post-exit price rose ≥ ``MISSED_RUNNER_MIN_DRIFT_PCT``.
    Bot exited a big winner before it ran.
  * **WHIPSAW** — short hold (≤ ``WHIPSAW_MAX_HOLD_HOURS``) + small loss
    (≥ ``-WHIPSAW_MAX_LOSS_PCT``) + post-exit recovery > half of
    ``PREMATURE_MIN_DRIFT_PCT``. The specific pathology of "bot panicked out
    of a flat-to-up name at a paper-cut loss". Distinct from PREMATURE: the
    pnl signal + short hold are the discriminator (a winning-trip rise-after
    is PREMATURE, not WHIPSAW).
  * **NEUTRAL** — drift inside the band; verdict withheld.
  * **INSUFFICIENT** — exit too recent (≤ ``MIN_HOURS_SINCE_EXIT``) or no
    current price. Sample-size-honest: numerics emitted whenever defined, but
    verdict withheld until the post-exit window has matured. Same discipline
    as ``build_tail_risk`` / ``build_correlation`` / ``build_news_velocity``.

Aggregate ``exit_quality_score`` (sum of +1/-1/-2 weights / n_scored)
condenses the per-RT pattern to a single number an operator can watch over
time: persistently negative ⇒ the bot is exiting too early, persistently
positive ⇒ exits are well-timed. A single trip is not load-bearing — the
score only matures with N≥3 scored trips.

Pure builder: takes already-computed round_trips + a current-price dict +
``now``. No DB, no network, never raises on garbage rows. Observational only
— never gates Opus, no caps (AGENTS.md #2/#12). The endpoint
(``/api/round-trip-postmortem``) owns the IO (yfinance via market.get_prices,
store.recent_trades).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

# Verdict thresholds. Conservative defaults so a noisy 0.3% post-exit drift
# is not over-read as PREMATURE on a $50 stock (paper-trader's typical name).
MIN_HOURS_SINCE_EXIT = 2.0
PREMATURE_MIN_DRIFT_PCT = 1.0
MISSED_RUNNER_MIN_DRIFT_PCT = 5.0
CORRECT_MAX_DRIFT_PCT = -1.0  # post-exit drop ≤ this ⇒ CORRECT

# WHIPSAW: short hold + small loss + post-exit recovery. Tuned for the live
# pathology (DRAM 1h hold, -0.18% loss, "raise dry powder" reasoning) —
# different ladder from PREMATURE because the pnl + hold signal makes it
# falsifiably distinct, not just a sub-band of drift.
WHIPSAW_MAX_HOLD_HOURS = 4.0
WHIPSAW_MAX_LOSS_PCT = 1.5

# Verdict → score (+ = good, − = bad). Used to compute exit_quality_score.
_VERDICT_SCORE = {
    "CORRECT": 1,
    "PREMATURE": -1,
    "MISSED_RUNNER": -2,
    "WHIPSAW": -2,
    "NEUTRAL": 0,
}


def _parse_ts(ts):
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _per_share_avg(value, qty):
    try:
        v = float(value)
        q = float(qty)
        if q <= 1e-9:
            return None
        return v / q
    except Exception:
        return None


def _verdict_for(*, hold_hours, pnl_pct, drift_pct):
    """Order matters: WHIPSAW is the most specific positive-drift band."""
    if drift_pct is None:
        return "INSUFFICIENT"
    is_short_hold = hold_hours is not None and hold_hours <= WHIPSAW_MAX_HOLD_HOURS
    is_small_loss = (
        pnl_pct is not None and pnl_pct < 0 and pnl_pct >= -WHIPSAW_MAX_LOSS_PCT
    )
    drifted_up = drift_pct >= (PREMATURE_MIN_DRIFT_PCT / 2.0)
    if is_short_hold and is_small_loss and drifted_up:
        return "WHIPSAW"
    if drift_pct >= MISSED_RUNNER_MIN_DRIFT_PCT:
        return "MISSED_RUNNER"
    if drift_pct >= PREMATURE_MIN_DRIFT_PCT:
        return "PREMATURE"
    if drift_pct <= CORRECT_MAX_DRIFT_PCT:
        return "CORRECT"
    return "NEUTRAL"


def _trip_headline(trip):
    tk = trip["ticker"]
    exit_p = trip["exit_price_avg"]
    drift = trip["post_exit_drift_pct"]
    cur = trip["current_price"]
    h_since = trip["hours_since_exit"]
    v = trip["verdict"]
    if v == "INSUFFICIENT":
        if cur is None:
            return f"{tk}: no current price — exit verdict withheld."
        return (
            f"{tk}: exit {h_since:.1f}h ago — under the "
            f"{MIN_HOURS_SINCE_EXIT:.1f}h post-exit window, verdict withheld."
        )
    pnl = trip.get("pnl_pct")
    pnl_str = f"({pnl:+.2f}%)" if pnl is not None else ""
    drift_str = f"{drift:+.2f}%"
    if v == "CORRECT":
        return (
            f"{tk}: sold ${exit_p:.2f} {pnl_str}; ${cur:.2f} now "
            f"({drift_str} post-exit, {h_since:.1f}h). Exit captured the move."
        )
    if v == "PREMATURE":
        return (
            f"{tk}: sold ${exit_p:.2f} {pnl_str}; ${cur:.2f} now "
            f"({drift_str} post-exit, {h_since:.1f}h). May have exited too early."
        )
    if v == "MISSED_RUNNER":
        return (
            f"{tk}: sold ${exit_p:.2f} {pnl_str}; ${cur:.2f} now "
            f"({drift_str} post-exit, {h_since:.1f}h). Big runner missed."
        )
    if v == "WHIPSAW":
        return (
            f"{tk}: short {trip['hold_hours']:.1f}h round-trip closed at "
            f"{pnl_str}, then bounced {drift_str} ({h_since:.1f}h). "
            f"Whipsaw — bot may be over-eager to raise cash."
        )
    return (
        f"{tk}: sold ${exit_p:.2f} {pnl_str}; ${cur:.2f} now "
        f"({drift_str} post-exit, {h_since:.1f}h). No edge in the post-exit move."
    )


def _aggregate_headline(counts, n_scored):
    if n_scored == 0:
        return "Exit-quality verdicts pending — no round-trip has matured past the post-exit window."
    n_prem = counts.get("PREMATURE", 0)
    n_whip = counts.get("WHIPSAW", 0)
    n_miss = counts.get("MISSED_RUNNER", 0)
    n_corr = counts.get("CORRECT", 0)
    bad = n_prem + n_whip + n_miss
    if bad == 0 and n_corr > 0:
        return f"{n_corr}/{n_scored} exits were CORRECT — exits look well-timed."
    if n_miss + n_whip + n_prem > n_corr and (n_prem + n_miss + n_whip) >= 2:
        return (
            f"{n_prem + n_miss + n_whip}/{n_scored} exits ran against the bot "
            f"(premature/whipsaw/missed) — bot may be selling too early."
        )
    if n_corr > 0:
        return (
            f"{n_corr}/{n_scored} CORRECT, {bad}/{n_scored} ran against the exit. "
            "Mixed."
        )
    return f"{bad}/{n_scored} exits ran against the bot post-close."


def build_round_trip_postmortem(
    round_trips,
    current_prices,
    now=None,
    max_n=10,
):
    """Pure builder. See module docstring for the verdict ladder.

    Inputs:
        round_trips: list of dicts in build_round_trips() shape — must carry
            ticker, type, entry_ts, exit_ts, cost, proceeds, qty, pnl_pct.
        current_prices: dict[ticker → float|None]. Missing or <=0 ⇒ verdict
            INSUFFICIENT for that ticker. Negatives treated as missing.
        now: datetime (default UTC now). Tests inject a fixed clock.
        max_n: clip surfaced trips to the newest-by-exit N (default 10).

    Output:
        Dict with: as_of, n_input, n_scored, verdict_counts,
        exit_quality_score (None when no trip scored), state
        (NO_DATA/INSUFFICIENT/OK), headline, trips (newest-first).
    """
    now = now or datetime.now(timezone.utc)
    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_input": 0,
        "n_scored": 0,
        "verdict_counts": {
            "CORRECT": 0, "PREMATURE": 0, "MISSED_RUNNER": 0,
            "WHIPSAW": 0, "NEUTRAL": 0, "INSUFFICIENT": 0,
        },
        "exit_quality_score": None,
        "state": "NO_DATA",
        "headline": _aggregate_headline({}, 0),
        "trips": [],
        "min_hours_since_exit": MIN_HOURS_SINCE_EXIT,
        "max_n": int(max_n),
    }

    if not round_trips:
        return base

    prices = current_prices or {}

    candidates = []
    for rt in round_trips:
        if not isinstance(rt, dict):
            continue
        tk = rt.get("ticker")
        if not tk:
            continue
        exit_ts = _parse_ts(rt.get("exit_ts"))
        entry_ts = _parse_ts(rt.get("entry_ts"))
        if exit_ts is None or entry_ts is None:
            continue
        cost = rt.get("cost")
        proceeds = rt.get("proceeds")
        qty = rt.get("qty")
        entry_p = _per_share_avg(cost, qty)
        exit_p = _per_share_avg(proceeds, qty)
        if entry_p is None or exit_p is None or exit_p <= 0:
            continue

        cur = prices.get(tk)
        try:
            cur_f = float(cur) if cur is not None else None
        except Exception:
            cur_f = None
        if cur_f is not None and cur_f <= 0:
            cur_f = None

        hours_since_exit = (now - exit_ts).total_seconds() / 3600.0
        hold_hours = (exit_ts - entry_ts).total_seconds() / 3600.0
        if hold_hours < 0:
            hold_hours = 0.0

        if cur_f is None or hours_since_exit < MIN_HOURS_SINCE_EXIT:
            verdict = "INSUFFICIENT"
            drift_pct = None
        else:
            drift_pct = (cur_f - exit_p) / exit_p * 100.0
            verdict = _verdict_for(
                hold_hours=hold_hours,
                pnl_pct=rt.get("pnl_pct"),
                drift_pct=drift_pct,
            )

        trip = {
            "ticker": tk,
            "type": rt.get("type") or "stock",
            "strike": rt.get("strike"),
            "expiry": rt.get("expiry"),
            "entry_ts": rt.get("entry_ts"),
            "exit_ts": rt.get("exit_ts"),
            "qty": rt.get("qty"),
            "cost": rt.get("cost"),
            "proceeds": rt.get("proceeds"),
            "pnl_usd": rt.get("pnl_usd"),
            "pnl_pct": rt.get("pnl_pct"),
            "hold_hours": round(hold_hours, 2),
            "hours_since_exit": round(hours_since_exit, 2),
            "entry_price_avg": round(entry_p, 4),
            "exit_price_avg": round(exit_p, 4),
            "current_price": (round(cur_f, 4) if cur_f is not None else None),
            "post_exit_drift_pct": (
                round(drift_pct, 2) if drift_pct is not None else None
            ),
            "verdict": verdict,
            "entry_trade_ids": rt.get("entry_trade_ids") or [],
            "exit_trade_ids": rt.get("exit_trade_ids") or [],
        }
        trip["headline"] = _trip_headline(trip)
        candidates.append(trip)

    if not candidates:
        return base

    candidates.sort(key=lambda r: r["exit_ts"] or "", reverse=True)
    clipped = candidates[: max(1, int(max_n))]

    counts = {k: 0 for k in base["verdict_counts"]}
    score_sum = 0
    n_scored = 0
    for tr in clipped:
        v = tr["verdict"]
        counts[v] = counts.get(v, 0) + 1
        if v != "INSUFFICIENT":
            n_scored += 1
            score_sum += _VERDICT_SCORE.get(v, 0)

    base["n_input"] = len(round_trips)
    base["verdict_counts"] = counts
    base["n_scored"] = n_scored
    base["exit_quality_score"] = (
        round(score_sum / n_scored, 4) if n_scored > 0 else None
    )
    base["trips"] = clipped
    base["headline"] = _aggregate_headline(counts, n_scored)
    base["state"] = "OK" if n_scored > 0 else "INSUFFICIENT"
    return base
