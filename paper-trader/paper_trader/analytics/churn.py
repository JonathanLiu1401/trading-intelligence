"""Overtrading & re-entry-churn diagnostic.

Observed live (2026-05-16): ``avg_holding_days 0.26``, ``profit_factor
0.04``, and a trade log that SELLs NVDA to fund LITE then plans to re-add
NVDA next cycle — the textbook *exit-then-re-enter-the-same-name* churn
that turns a thesis into round-trip friction. ``/api/analytics`` already
shows the raw aggregates and ``/api/trade-asymmetry`` already grades the
exit/sizing *payoff* pathology (DISPOSITION_BLEED, breakeven-vs-actual
win-rate). Neither answers the turnover question a desk risk manager asks:

* **How often does the book re-buy a name it just fully closed, and how
  fast?** (the headline contribution here — nothing else measures it.)
* What is the round-trip *cadence* (round-trips per active trading day)?
* What share of realised *loss* was booked in sub-one-day round-trips?

This module is deliberately distinct from its neighbours:

* ``/api/analytics`` — raw aggregates (win_rate, profit_factor, hold avg).
* ``/api/trade-asymmetry`` — payoff ratio & winner/loser hold-time skew.
* ``build_churn`` — **turnover & same-name re-entry frequency**: the
  count and speed of "closed a key, re-opened the same key within
  ``REENTRY_WINDOW_DAYS``" events, the per-active-day round-trip cadence,
  and how concentrated realised losses are in <1-day trips.

Single source of truth: it consumes ``round_trips.build_round_trips`` and
never recomputes P&L or hold time (AGENTS.md invariant #10). It is a
*diagnostic / advisory* panel only — it never gates Opus and adds no caps
(AGENTS.md #2/#12).

Sample-size honesty mirrors ``trade_asymmetry.py`` / ``news_edge.py``:
numeric metrics are emitted from the first closed round-trip, but the
**verdict label** is withheld until ``STABLE`` (``n ≥ STABLE_MIN_RTS``) —
a five-trip "you're churning" verdict is noise.
"""
from __future__ import annotations

from datetime import datetime, timezone
from statistics import median

from .round_trips import build_round_trips

# Verdict is only labelled once the realised sample is large enough that it
# will not flip on the next trade (identical idiom & threshold to
# trade_asymmetry so the two panels never disagree on STABLE-ness).
STABLE_MIN_RTS = 20

# A same-(ticker,type,strike,expiry) re-BUY this many calendar days (or
# fewer) after the key's prior full close counts as a *re-entry churn*
# event. Rationale: the live trader's open-market cadence is
# ``OPEN_INTERVAL_S = 1800`` (~48 decision cycles per trading day), so a
# genuine thesis *reversal* on the very name just exited rarely matures
# within three calendar days — a re-buy that fast is turnover, not new
# conviction. Calendar days (not trading days) are used so this stays
# consistent with ``round_trips.hold_days``, which is also calendar.
REENTRY_WINDOW_DAYS = 3.0

# STABLE verdict thresholds (all exact-value test-locked):
#   ≥25% of round-trips being fast same-name re-entries, OR a >1 round-trip
#   /active-day cadence with a sub-day median hold ⇒ CHURNING.
REENTRY_CHURN_PCT = 25.0
CHURN_RT_PER_DAY = 1.0
#   A ≥10-day median hold with negligible cadence and no fast re-entries
#   ⇒ BUY_AND_HOLD. Everything in between is ACTIVE_TURNOVER.
HOLD_LONG_DAYS = 10.0
QUIET_RT_PER_DAY = 0.2


def _parse_ts(ts: str | None) -> datetime | None:
    # Identical to round_trips._parse_ts; kept local rather than importing a
    # private name, matching how trade_asymmetry keeps its own _mean.
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def build_churn(trades: list[dict], now: datetime | None = None) -> dict:
    """Turnover / re-entry-churn decomposition over closed round-trips. Pure.

    ``trades`` must be a ``Store.recent_trades()``-shaped ledger ordered
    **oldest→newest** — exactly what ``/api/analytics`` and
    ``/api/trade-asymmetry`` pass:
    ``list(reversed(store.recent_trades(2000)))``. ``build_round_trips``
    reads rows in sequence and does not sort.
    """
    now = now or datetime.now(timezone.utc)
    rts = build_round_trips(trades)
    n = len(rts)

    # ---- same-name re-entry detection ---------------------------------
    # build_round_trips appends a round-trip the moment its key's qty
    # returns to zero, iterating trades oldest→newest, so per key the
    # round-trips appear in chronological *close* order. Group by key
    # preserving that order, then compare each close to the next open of
    # the same key.
    by_key: dict[tuple, list[dict]] = {}
    for rt in rts:
        key = (rt["ticker"], rt["type"], rt["strike"], rt["expiry"])
        by_key.setdefault(key, []).append(rt)

    reentry_events: list[dict] = []
    for key, seq in by_key.items():
        for prev, nxt in zip(seq, seq[1:]):
            exit_dt = _parse_ts(prev.get("exit_ts"))
            entry_dt = _parse_ts(nxt.get("entry_ts"))
            if exit_dt is None or entry_dt is None:
                continue
            gap = (entry_dt - exit_dt).total_seconds() / 86400.0
            if gap < 0:
                continue  # chronological by construction; guard anyway
            if gap <= REENTRY_WINDOW_DAYS:
                reentry_events.append({
                    "ticker": key[0],
                    "type": key[1],
                    "strike": key[2],
                    "expiry": key[3],
                    "gap_days": round(gap, 4),
                    "prior_pnl_usd": prev.get("pnl_usd"),
                    "prior_exit_ts": prev.get("exit_ts"),
                    "next_entry_ts": nxt.get("entry_ts"),
                })
    reentry_events.sort(key=lambda e: e["gap_days"])  # fastest churn first
    n_reentries = len(reentry_events)
    reentry_rate_pct = round(n_reentries / n * 100.0, 2) if n else None

    # ---- cadence (round-trips per active calendar day) ----------------
    entry_dts = [d for d in (_parse_ts(rt.get("entry_ts")) for rt in rts)
                 if d is not None]
    exit_dts = [d for d in (_parse_ts(rt.get("exit_ts")) for rt in rts)
                if d is not None]
    span_days = None
    round_trips_per_day = None
    if entry_dts and exit_dts:
        span_days = (max(exit_dts) - min(entry_dts)).total_seconds() / 86400.0
        span_days = round(span_days, 4)
        if span_days > 1e-9:
            round_trips_per_day = round(n / span_days, 4)
        # span ≤ 0 (all trips inside one instant) ⇒ leave None rather than
        # divide by zero — mirrors decision_reliability's dead-cycle guard.

    # ---- hold-time / sub-day loss concentration -----------------------
    holds = [rt["hold_days"] for rt in rts if rt.get("hold_days") is not None]
    median_hold_days = round(median(holds), 4) if holds else None
    n_sub_day = sum(1 for h in holds if h < 1.0)
    sub_day_trip_pct = (round(n_sub_day / len(holds) * 100.0, 2)
                        if holds else None)

    total_loss = sum(rt["pnl_usd"] for rt in rts
                     if (rt.get("pnl_usd") or 0.0) < 0)
    sub_day_loss = sum(
        rt["pnl_usd"] for rt in rts
        if (rt.get("pnl_usd") or 0.0) < 0
        and rt.get("hold_days") is not None and rt["hold_days"] < 1.0)
    # Honest framing: this is the *share of realised losses booked in
    # sub-1-day round-trips*, NOT a slippage/friction model (paper book has
    # no spread). None when there were no losses to attribute.
    churn_loss_concentration_pct = (
        round(sub_day_loss / total_loss * 100.0, 2) if total_loss < 0 else None)

    # ---- verdict (gated to STABLE) ------------------------------------
    state = ("NO_DATA" if n == 0
             else "STABLE" if n >= STABLE_MIN_RTS
             else "EMERGING")

    verdict = None
    verdict_reason = None
    if state == "STABLE":
        rr = reentry_rate_pct or 0.0
        rtpd = round_trips_per_day
        mh = median_hold_days
        fast_cadence = (rtpd is not None and rtpd >= CHURN_RT_PER_DAY
                        and mh is not None and mh < 1.0)
        if rr >= REENTRY_CHURN_PCT or fast_cadence:
            verdict = "CHURNING"
            if rr >= REENTRY_CHURN_PCT:
                verdict_reason = (
                    f"{rr:.1f}% of round-trips re-buy a name within "
                    f"{REENTRY_WINDOW_DAYS:g}d of fully closing it — "
                    f"turnover, not conviction")
            else:
                verdict_reason = (
                    f"{rtpd:.2f} round-trips per active day with a "
                    f"{mh:.2f}d median hold — trading the same capital in "
                    f"circles")
        elif (mh is not None and mh >= HOLD_LONG_DAYS
              and (rtpd is None or rtpd < QUIET_RT_PER_DAY)
              and rr < REENTRY_CHURN_PCT):
            verdict = "BUY_AND_HOLD"
            verdict_reason = (
                f"{mh:.1f}d median hold, "
                f"{(f'{rtpd:.2f}' if rtpd is not None else 'n/a')} "
                f"round-trips/day — positions are given time to work")
        else:
            verdict = "ACTIVE_TURNOVER"
            verdict_reason = (
                f"{(f'{rtpd:.2f}' if rtpd is not None else 'n/a')} "
                f"round-trips/day, {rr:.1f}% fast re-entries — active but "
                f"below the churn line")

    # ---- headline ------------------------------------------------------
    rr_disp = "n/a" if reentry_rate_pct is None else f"{reentry_rate_pct:.1f}%"
    rtpd_disp = ("n/a" if round_trips_per_day is None
                 else f"{round_trips_per_day:.2f}")
    mh_disp = "n/a" if median_hold_days is None else f"{median_hold_days:.2f}d"

    if state == "NO_DATA":
        headline = "No closed round-trips yet — turnover undefined."
    elif state == "EMERGING":
        headline = (
            f"Emerging — {n} of {STABLE_MIN_RTS} round-trips for a stable "
            f"read. So far: {n_reentries} fast same-name re-entries "
            f"({rr_disp}), {rtpd_disp} round-trips/day, {mh_disp} median "
            f"hold (verdict withheld until n≥{STABLE_MIN_RTS}).")
    elif verdict == "CHURNING":
        head = reentry_events[0] if reentry_events else None
        ex = (f" Fastest: {head['ticker']} re-bought "
              f"{head['gap_days']:.2f}d after closing it"
              f"{'' if head.get('prior_pnl_usd') is None else f' for ${head['prior_pnl_usd']:+.2f}'}."
              if head else "")
        headline = (
            f"CHURNING — {rr_disp} of {n} round-trips are <"
            f"{REENTRY_WINDOW_DAYS:g}d same-name re-entries; {rtpd_disp} "
            f"round-trips/day; {mh_disp} median hold." + ex)
    elif verdict == "BUY_AND_HOLD":
        headline = (
            f"BUY_AND_HOLD — {mh_disp} median hold, {rtpd_disp} "
            f"round-trips/day, {rr_disp} fast re-entries over {n} "
            f"round-trips; capital is given time to compound.")
    elif verdict == "ACTIVE_TURNOVER":
        headline = (
            f"ACTIVE_TURNOVER — {rtpd_disp} round-trips/day, {rr_disp} fast "
            f"re-entries, {mh_disp} median hold over {n} round-trips; "
            f"active but below the churn line.")
    else:
        headline = f"{n} round-trips analysed."

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "headline": headline,
        "n_round_trips": n,
        "n_reentries": n_reentries,
        "reentry_rate_pct": reentry_rate_pct,
        "reentry_window_days": REENTRY_WINDOW_DAYS,
        "reentry_events": reentry_events[:10],
        "span_days": span_days,
        "round_trips_per_day": round_trips_per_day,
        "median_hold_days": median_hold_days,
        "n_sub_day_trips": n_sub_day,
        "sub_day_trip_pct": sub_day_trip_pct,
        "realized_loss_usd": round(total_loss, 4),
        "sub_day_loss_usd": round(sub_day_loss, 4),
        "churn_loss_concentration_pct": churn_loss_concentration_pct,
        "stable_min_round_trips": STABLE_MIN_RTS,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json

    from paper_trader.store import get_store
    s = get_store()
    rep = build_churn(list(reversed(s.recent_trades(2000))))
    print(json.dumps(rep, indent=2, default=str))
