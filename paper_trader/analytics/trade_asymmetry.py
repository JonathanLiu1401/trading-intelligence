"""Trade-asymmetry / behavioural-edge diagnostic — exit & sizing pathology.

Observed live (2026-05-16): ``win_rate 20%``, ``profit_factor 0.04``,
``avg_winner_usd +0.57`` vs ``avg_loser_usd −3.75`` (losers 6.6× winners),
``avg_holding_days 0.26``. The raw aggregates already exist in
``/api/analytics``; what no panel answers is the desk question — *given my
payoff ratio, what win-rate do I need to break even, am I above or below it,
and am I cutting winners faster than losers (the disposition effect that
produces exactly this P&L shape)?*

This module is deliberately distinct from its neighbours:

* ``/api/analytics`` — **raw aggregates** (win_rate, profit_factor, $ avgs).
* ``/api/calibration`` — **is the confidence axis accurate** (stated
  confidence vs realised win-rate by bucket).
* ``build_trade_asymmetry`` — **exit/sizing behaviour pathology**: payoff
  ratio, per-trade expectancy, the breakeven win-rate *implied by the
  payoff ratio* vs the *actual* win-rate (the gap is the verdict), and the
  **disposition gap** = mean winner hold-days − mean loser hold-days
  (negative ⇒ cutting winners faster than losers).

Single source of truth: it consumes ``round_trips.build_round_trips`` and
never recomputes P&L (AGENTS.md invariant #10). It is a *diagnostic /
advisory* panel only — it never gates Opus and adds no caps (#2/#12).

Sample-size honesty mirrors ``news_edge.py``'s INSUFFICIENT_DATA idiom:
numeric metrics are emitted as soon as there is any closed round-trip, but
the **verdict label** is withheld until ``STABLE`` (``n ≥ STABLE_MIN_RTS``)
— a five-trade verdict is noise.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .round_trips import build_round_trips

# Verdict is only labelled once the realised sample is large enough that it
# will not flip on the next trade. Below this it's EMERGING (metrics only).
STABLE_MIN_RTS = 20
# A disposition gap inside ±0.01d (~15 min; meaningful at the observed
# 0.26d average hold) is "no skew".
DISPOSITION_EPS_DAYS = 0.01
# Per-trade expectancy within ±$0.01 is "flat".
FLAT_EPS_USD = 0.01


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def build_trade_asymmetry(trades: list[dict],
                          now: datetime | None = None) -> dict:
    """Behavioural-edge decomposition over closed round-trips. Pure.

    ``trades`` must be a ``Store.recent_trades()``-shaped ledger ordered
    **oldest→newest** — ``build_round_trips`` reads rows in sequence and
    does not sort. Pass exactly what ``/api/analytics`` passes:
    ``list(reversed(store.recent_trades(2000)))``.
    """
    now = now or datetime.now(timezone.utc)
    rts = build_round_trips(trades)
    n = len(rts)

    winners = [rt for rt in rts if (rt.get("pnl_usd") or 0.0) > 0]
    losers = [rt for rt in rts if (rt.get("pnl_usd") or 0.0) < 0]
    n_wins = len(winners)
    n_losses = len(losers)
    n_decided = n_wins + n_losses  # washes (pnl == 0) excluded from both

    realized_pl = round(sum((rt.get("pnl_usd") or 0.0) for rt in rts), 4)
    expectancy = round(realized_pl / n, 4) if n else None

    avg_win = _mean([rt["pnl_usd"] for rt in winners])
    avg_loss = _mean([rt["pnl_usd"] for rt in losers])  # negative or None
    avg_win_r = round(avg_win, 4) if avg_win is not None else None
    avg_loss_r = round(avg_loss, 4) if avg_loss is not None else None

    # payoff_ratio needs BOTH a winner mean and a (non-zero) loser mean.
    # No losers → None, never ∞ / a sentinel huge number.
    if avg_win is not None and avg_loss is not None and abs(avg_loss) > 1e-12:
        payoff_ratio = round(avg_win / abs(avg_loss), 4)
        breakeven_wr = round(1.0 / (1.0 + payoff_ratio) * 100.0, 2)
    else:
        payoff_ratio = None
        breakeven_wr = None

    actual_wr = round(n_wins / n_decided * 100.0, 2) if n_decided else None
    win_rate_gap = (round(actual_wr - breakeven_wr, 2)
                    if actual_wr is not None and breakeven_wr is not None
                    else None)

    # Disposition gap: only round-trips that carry a parseable hold time feed
    # the respective mean (a wash with hold_days set still does NOT count as a
    # win or loss, so it never reaches these lists).
    win_holds = [rt["hold_days"] for rt in winners
                 if rt.get("hold_days") is not None]
    loss_holds = [rt["hold_days"] for rt in losers
                  if rt.get("hold_days") is not None]
    mean_win_hold = _mean(win_holds)
    mean_loss_hold = _mean(loss_holds)
    if mean_win_hold is not None and mean_loss_hold is not None:
        disposition_gap = round(mean_win_hold - mean_loss_hold, 4)
    else:
        disposition_gap = None

    # ---- verdict (gated to STABLE) -------------------------------------
    state = ("NO_DATA" if n == 0
             else "STABLE" if n >= STABLE_MIN_RTS
             else "EMERGING")

    verdict = None
    verdict_reason = None
    if state == "STABLE":
        skew_negative = (disposition_gap is not None
                         and disposition_gap < -DISPOSITION_EPS_DAYS)
        # The trap = losing money with losers on the book. When a payoff
        # ratio exists this is exactly `actual_wr < breakeven_wr`
        # (sign(expectancy) ≡ actual-vs-breakeven, washes contribute 0); the
        # `payoff_ratio is None` arm additionally catches an all-losers book
        # (no winner mean ⇒ no ratio) so it reads as PAYOFF_TRAP, not FLAT.
        if (n_losses > 0 and expectancy is not None
                and expectancy < -FLAT_EPS_USD):
            verdict = "PAYOFF_TRAP"
            if payoff_ratio is not None and breakeven_wr is not None:
                verdict_reason = (
                    f"win-rate {actual_wr:.1f}% is below the "
                    f"{breakeven_wr:.1f}% this {payoff_ratio:.2f} payoff "
                    f"ratio needs to break even — the math cannot carry it")
            else:
                verdict_reason = (
                    f"no winning round-trips — every closed trade lost "
                    f"(${expectancy:+.2f}/trade over {n} round-trips)")
        elif expectancy is not None and expectancy > FLAT_EPS_USD:
            if skew_negative:
                verdict = "DISPOSITION_BLEED"
                verdict_reason = (
                    f"net positive (${expectancy:+.2f}/trade) but winners are "
                    f"held {abs(disposition_gap):.2f}d less than losers — the "
                    f"disposition effect is capping the edge")
            else:
                verdict = "EDGE_POSITIVE"
                verdict_reason = (
                    f"positive expectancy (${expectancy:+.2f}/trade) with no "
                    f"adverse hold-time skew — a genuine, well-managed edge")
        else:
            verdict = "FLAT"
            verdict_reason = (
                f"expectancy ${expectancy:+.2f}/trade — no statistical edge "
                f"either way")

    # ---- headline ------------------------------------------------------
    disp_clause = ""
    if (disposition_gap is not None and disposition_gap < -DISPOSITION_EPS_DAYS
            and avg_win_r is not None and avg_loss_r is not None
            and mean_win_hold is not None and mean_loss_hold is not None):
        disp_clause = (
            f" Disposition effect: cutting winners at ${avg_win_r:+.2f} after "
            f"{mean_win_hold:.2f}d, riding losers to ${avg_loss_r:+.2f} over "
            f"{mean_loss_hold:.2f}d.")

    if state == "NO_DATA":
        headline = "No closed round-trips yet — behavioural edge undefined."
    elif state == "EMERGING":
        headline = (
            f"Emerging — {n} of {STABLE_MIN_RTS} round-trips for a stable "
            f"read. So far: ${expectancy:+.2f}/trade expectancy, "
            f"{(str(payoff_ratio) if payoff_ratio is not None else 'n/a')} "
            f"payoff ratio (verdict withheld until n≥{STABLE_MIN_RTS})."
            + disp_clause)
    elif verdict == "PAYOFF_TRAP":
        if payoff_ratio is not None and breakeven_wr is not None:
            trap_lead = (
                f"PAYOFF_TRAP — {actual_wr:.1f}% win-rate vs "
                f"{breakeven_wr:.1f}% needed for this {payoff_ratio:.2f} "
                f"payoff ratio; ")
        else:
            trap_lead = ("PAYOFF_TRAP — no winning round-trips; ")
        headline = (
            trap_lead
            + f"${expectancy:+.2f}/trade over {n} round-trips." + disp_clause)
    elif verdict == "DISPOSITION_BLEED":
        headline = (
            f"DISPOSITION_BLEED — profitable (${expectancy:+.2f}/trade) but "
            f"cutting winners {abs(disposition_gap):.2f}d faster than losers "
            f"over {n} round-trips." + disp_clause)
    elif verdict == "EDGE_POSITIVE":
        wr_clause = (
            f"{actual_wr:.1f}% win-rate vs {breakeven_wr:.1f}% breakeven, "
            if actual_wr is not None and breakeven_wr is not None
            else "no losing round-trips yet, ")
        headline = (
            f"EDGE_POSITIVE — ${expectancy:+.2f}/trade, {wr_clause}"
            f"healthy hold-time discipline over {n} round-trips.")
    else:  # FLAT
        headline = (
            f"FLAT — ${expectancy:+.2f}/trade over {n} round-trips; no "
            f"statistical edge." + disp_clause)

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "headline": headline,
        "n_round_trips": n,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_decided": n_decided,
        "n_washes": n - n_decided,
        "realized_pl_usd": realized_pl,
        "expectancy_usd": expectancy,
        "avg_winner_usd": avg_win_r,
        "avg_loser_usd": avg_loss_r,
        "payoff_ratio": payoff_ratio,
        "breakeven_win_rate_pct": breakeven_wr,
        "actual_win_rate_pct": actual_wr,
        "win_rate_gap_pct": win_rate_gap,
        "disposition_gap_days": disposition_gap,
        "avg_winner_hold_days": (round(mean_win_hold, 4)
                                 if mean_win_hold is not None else None),
        "avg_loser_hold_days": (round(mean_loss_hold, 4)
                                if mean_loss_hold is not None else None),
        "stable_min_round_trips": STABLE_MIN_RTS,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json
    from paper_trader.store import get_store
    s = get_store()
    rep = build_trade_asymmetry(list(reversed(s.recent_trades(2000))))
    print(json.dumps(rep, indent=2, default=str))
