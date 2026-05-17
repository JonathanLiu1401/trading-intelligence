"""Loser autopsy — a per-closed-trade post-mortem on the round-trips that lost.

The desk question nothing else answers. Its neighbours each see a *different*
slice:

* ``/api/thesis-drift`` — re-tests an **open** position against its own
  opening rationale. Says nothing about trades already closed.
* ``/api/trade-asymmetry`` — **aggregate** payoff math (expectancy, payoff
  ratio, the disposition gap). A single number for the whole book; no
  per-trade story.
* ``/api/churn`` — turnover / same-name re-entry **cadence**. Counts how
  often, not *why each loss happened*.

``build_loser_autopsy`` is the missing one: for every **closed losing
round-trip** it surfaces the *verbatim* entry reason (the thesis the trader
wrote when it bought) and the *verbatim* exit reason (why it sold), the hold
time, the loss, and an objective, deterministic **failure-mode label** — then
rolls them up so the operator can see *which name is the bleed*, *which
failure mode dominates*, and *which losing names keep recurring*.

Single source of truth: it consumes ``round_trips.build_round_trips`` and
never recomputes P&L or hold-time (AGENTS.md invariant #10). The verbatim
reasons are joined back from the contributing trade rows by their DB ``id``
(the same "surface the reason verbatim, never NLP-parse it for trading
logic" discipline ``thesis_drift`` uses). It is a *diagnostic / advisory*
panel only — it never gates Opus and adds no caps (AGENTS.md #2/#12).

Sample-size honesty mirrors ``trade_asymmetry.py``: numeric metrics and the
per-loser cards are emitted from the first closed round-trip, but the
**pattern verdict** (dominant failure mode + dominant losing name) is
withheld until ``STABLE`` (``n_losers >= STABLE_MIN_LOSERS``) — a two-loss
"pattern" is noise.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .round_trips import build_round_trips

# The dominant-pattern verdict only labels once enough losers have closed
# that the modal failure mode will not flip on the next bad trade. Below
# this it is EMERGING (cards + numerics only, no verdict).
STABLE_MIN_LOSERS = 8
# Hold-time / loss thresholds for the objective failure-mode taxonomy.
# Documented module constants (tests assert exact behaviour at each
# boundary) — not tunables Opus ever sees.
FAST_HOLD_DAYS = 1.0     # closed inside a day ⇒ a fast cut / whipsaw
SLOW_HOLD_DAYS = 5.0     # held ≥ this through the loss ⇒ rode it down
BIG_LOSS_PCT = -15.0     # ≤ this ⇒ the thesis was badly wrong
SMALL_LOSS_PCT = -3.0    # > this (closer to 0) ⇒ a shallow loss


def _classify(hold_days: float | None, pnl_pct: float | None) -> str:
    """Objective, deterministic failure-mode label for one losing round-trip.

    Precedence is intentional and documented:

    * ``KNIFE_CATCH`` — loss ≤ ``BIG_LOSS_PCT`` regardless of hold time: the
      entry thesis was badly wrong (caught a falling knife).
    * ``WHIPSAW`` — closed inside ``FAST_HOLD_DAYS`` at a shallow loss
      (``> SMALL_LOSS_PCT``): cut fast for almost nothing, likely noise.
    * ``SLOW_BLEED`` — held ``≥ SLOW_HOLD_DAYS`` and still closed red: rode a
      loser down (the disposition behaviour ``trade_asymmetry`` measures in
      aggregate, surfaced here per-trade).
    * ``STOPPED_OUT`` — everything else: a moderate loss on a moderate hold.

    ``hold_days``/``pnl_pct`` may be ``None`` (parse failure / zero-cost
    round-trip). A ``None`` pnl_pct can never trip the magnitude arms, and a
    ``None`` hold_days can never trip the duration arms, so the function
    always returns a label and never raises.
    """
    big = pnl_pct is not None and pnl_pct <= BIG_LOSS_PCT
    if big:
        return "KNIFE_CATCH"
    fast = hold_days is not None and hold_days < FAST_HOLD_DAYS
    shallow = pnl_pct is not None and pnl_pct > SMALL_LOSS_PCT
    if fast and shallow:
        return "WHIPSAW"
    slow = hold_days is not None and hold_days >= SLOW_HOLD_DAYS
    if slow:
        return "SLOW_BLEED"
    return "STOPPED_OUT"


def _reason_for(trade_ids: list, by_id: dict, pick_last: bool) -> str | None:
    """Verbatim ``reason`` of the opening (first) or closing (last) trade.

    Never NLP-parsed — surfaced exactly as the trader wrote it. Missing id /
    absent row / empty string all degrade to ``None`` (no error).
    """
    if not trade_ids:
        return None
    tid = trade_ids[-1] if pick_last else trade_ids[0]
    row = by_id.get(tid)
    if not row:
        return None
    r = row.get("reason")
    return r if (r is not None and str(r).strip() != "") else None


def build_loser_autopsy(trades: list[dict],
                        worst_n: int = 10,
                        now: datetime | None = None) -> dict:
    """Per-closed-losing-round-trip post-mortem. Pure, never raises.

    ``trades`` must be a ``Store.recent_trades()``-shaped ledger ordered
    **oldest→newest** — ``build_round_trips`` reads rows in sequence and does
    not sort. Pass exactly what ``/api/analytics`` & ``/api/trade-asymmetry``
    pass: ``list(reversed(store.recent_trades(2000)))``.

    ``worst_n`` caps the per-loser card list (sorted most-negative first);
    the aggregates are always over *all* losers.
    """
    now = now or datetime.now(timezone.utc)
    rts = build_round_trips(trades)
    n_rts = len(rts)

    by_id: dict = {}
    for t in trades:
        tid = t.get("id")
        if tid is not None:
            by_id[tid] = t

    # Strict < 0 loser convention — identical to round_trips/#10 &
    # trade_asymmetry (a sub-cent wash reads as a non-loss, not a loss).
    losers = [rt for rt in rts if (rt.get("pnl_usd") or 0.0) < 0]
    n_losers = len(losers)

    cards: list[dict] = []
    for rt in losers:
        pnl = round(float(rt.get("pnl_usd") or 0.0), 4)
        pnl_pct = rt.get("pnl_pct")
        hold = rt.get("hold_days")
        mode = _classify(hold, pnl_pct)
        cards.append({
            "ticker": rt.get("ticker"),
            "type": rt.get("type"),
            "entry_ts": rt.get("entry_ts"),
            "exit_ts": rt.get("exit_ts"),
            "hold_days": hold,
            "qty": rt.get("qty"),
            "cost": rt.get("cost"),
            "proceeds": rt.get("proceeds"),
            "pnl_usd": pnl,
            "pnl_pct": pnl_pct,
            "failure_mode": mode,
            "entry_reason": _reason_for(rt.get("entry_trade_ids") or [],
                                        by_id, pick_last=False),
            "exit_reason": _reason_for(rt.get("exit_trade_ids") or [],
                                       by_id, pick_last=True),
        })

    # Most painful first; ticker is a deterministic tie-break so the card
    # order is stable for two identical-loss round-trips.
    cards.sort(key=lambda c: (c["pnl_usd"], c["ticker"] or ""))

    total_loss = round(sum(c["pnl_usd"] for c in cards), 4) if cards else 0.0
    worst = cards[0] if cards else None
    avg_loss = round(total_loss / n_losers, 4) if n_losers else None
    hold_vals = [c["hold_days"] for c in cards if c["hold_days"] is not None]
    median_hold = None
    if hold_vals:
        sv = sorted(hold_vals)
        m = len(sv) // 2
        median_hold = (sv[m] if len(sv) % 2
                       else round((sv[m - 1] + sv[m]) / 2.0, 4))

    # Which name is the bleed — $ lost per ticker (most negative first).
    by_ticker: dict[str, dict] = {}
    for c in cards:
        b = by_ticker.setdefault(c["ticker"], {"ticker": c["ticker"],
                                               "n": 0, "loss_usd": 0.0})
        b["n"] += 1
        b["loss_usd"] = round(b["loss_usd"] + c["pnl_usd"], 4)
    ticker_breakdown = sorted(by_ticker.values(),
                              key=lambda b: (b["loss_usd"], b["ticker"] or ""))

    # Which failure mode dominates.
    by_mode: dict[str, int] = {}
    for c in cards:
        by_mode[c["failure_mode"]] = by_mode.get(c["failure_mode"], 0) + 1
    # Deterministic dominant: most frequent, ties broken by a fixed
    # severity order so the verdict never flips on dict insertion order.
    _SEVERITY = ["KNIFE_CATCH", "SLOW_BLEED", "STOPPED_OUT", "WHIPSAW"]
    dominant_mode = None
    if by_mode:
        dominant_mode = max(
            by_mode, key=lambda m: (by_mode[m], -_SEVERITY.index(m)))

    # Names that lost more than once — distinct from churn (which counts
    # *re-entry cadence*); this is "which losing names keep recurring".
    repeat_offenders = [b["ticker"] for b in ticker_breakdown if b["n"] >= 2]

    # ---- state / verdict (verdict gated to STABLE) ---------------------
    if n_rts == 0:
        state = "NO_DATA"
    elif n_losers == 0:
        state = "NO_LOSSES"
    elif n_losers >= STABLE_MIN_LOSERS:
        state = "STABLE"
    else:
        state = "EMERGING"

    verdict = None
    if state == "STABLE":
        verdict = dominant_mode

    # ---- headline ------------------------------------------------------
    if state == "NO_DATA":
        headline = "No closed round-trips yet — nothing to autopsy."
    elif state == "NO_LOSSES":
        headline = (f"No losing round-trips across {n_rts} closed — "
                    f"nothing to autopsy.")
    else:
        worst_clause = ""
        if worst is not None:
            wp = (f" ({worst['pnl_pct']:+.1f}%)"
                  if worst.get("pnl_pct") is not None else "")
            worst_clause = (f" Worst: {worst['ticker']} "
                            f"${worst['pnl_usd']:+.2f}{wp}.")
        bleed_clause = ""
        if ticker_breakdown:
            tb = ticker_breakdown[0]
            bleed_clause = (f" {tb['ticker']} is the bleed "
                            f"(${tb['loss_usd']:+.2f} over {tb['n']} "
                            f"loss{'es' if tb['n'] != 1 else ''}).")
        if state == "EMERGING":
            headline = (
                f"Emerging — {n_losers} of {STABLE_MIN_LOSERS} losing "
                f"round-trips for a stable pattern read (verdict withheld). "
                f"${total_loss:+.2f} lost so far.{worst_clause}{bleed_clause}")
        else:
            mc = by_mode.get(dominant_mode, 0)
            headline = (
                f"{dominant_mode} dominates — {mc} of {n_losers} losing "
                f"round-trips; ${total_loss:+.2f} total realised "
                f"loss.{worst_clause}{bleed_clause}")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "n_round_trips": n_rts,
        "n_losers": n_losers,
        "total_loss_usd": total_loss,
        "avg_loss_usd": avg_loss,
        "median_loser_hold_days": median_hold,
        "dominant_failure_mode": dominant_mode,
        "failure_mode_counts": by_mode,
        "ticker_breakdown": ticker_breakdown,
        "repeat_offenders": repeat_offenders,
        "worst_losers": cards[:max(0, worst_n)],
        "stable_min_losers": STABLE_MIN_LOSERS,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json

    from paper_trader.store import get_store
    s = get_store()
    rep = build_loser_autopsy(list(reversed(s.recent_trades(2000))))
    print(json.dumps(rep, indent=2, default=str))
