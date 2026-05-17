"""Winner autopsy — a per-closed-trade post-mortem on the round-trips that won.

The symmetric counterpart to ``loser_autopsy``. Every behavioural builder on
this desk reflects a *pathology* — ``loser_autopsy`` narrates each loss,
``trade_asymmetry`` flags ``DISPOSITION_BLEED``, ``churn`` counts overtrading,
``self_review`` mirrors only the failures back into the decision prompt.
Nothing tells the operator (or, via the dashboard, the human reading it)
**which winning behaviour to repeat** — which entry theses actually produced
gains, and whether the desk *let its winners run* or *scalped them flat*.

``build_winner_autopsy`` is the missing positive mirror: for every **closed
winning round-trip** it surfaces the *verbatim* entry reason (the thesis the
trader wrote when it bought) and the *verbatim* exit reason (why it sold), the
hold time, the gain, and an objective, deterministic **success-mode label** —
then rolls them up so the operator can see *which name is the engine*, *which
win mode dominates*, and *which winning names keep recurring*.

The success-mode taxonomy is the exact reflection of ``loser_autopsy``'s
failure modes, and two of its labels are behaviourally load-bearing because
they connect to ``trade_asymmetry``'s disposition gap from the *winning* side:

* ``SLOW_GRIND`` — held ``≥ SLOW_HOLD_DAYS`` and still closed green: the desk
  *let a winner run*. This is the **good** disposition behaviour (the exact
  opposite of ``loser_autopsy``'s ``SLOW_BLEED``) — the one to repeat.
* ``SCALP`` — closed inside ``FAST_HOLD_DAYS`` for a shallow gain: cut a
  winner fast for almost nothing. This is the disposition effect
  ``trade_asymmetry`` measures in aggregate, surfaced here per-trade on the
  *winning* side — money likely left on the table.

Single source of truth: it consumes ``round_trips.build_round_trips`` and
never recomputes P&L or hold-time (AGENTS.md invariant #10). The verbatim
reasons are joined back from the contributing trade rows by their DB ``id``
(the same "surface the reason verbatim, never NLP-parse it for trading logic"
discipline ``thesis_drift`` / ``loser_autopsy`` use). It is a *diagnostic /
advisory* panel only — it never gates Opus, is never injected into the
decision prompt, and adds no caps (AGENTS.md #2/#12).

Sample-size honesty mirrors ``loser_autopsy`` / ``trade_asymmetry``: numeric
metrics and the per-winner cards are emitted from the first closed round-trip,
but the **pattern verdict** (dominant success mode) is withheld until
``STABLE`` (``n_winners >= STABLE_MIN_WINNERS``) — a two-win "pattern" is
noise.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .round_trips import build_round_trips

# The dominant-pattern verdict only labels once enough winners have closed
# that the modal success mode will not flip on the next good trade. Below
# this it is EMERGING (cards + numerics only, no verdict). Identical to
# loser_autopsy.STABLE_MIN_LOSERS so the two panels never disagree on what
# "a stable pattern" means.
STABLE_MIN_WINNERS = 8
# Hold-time / gain thresholds for the objective success-mode taxonomy.
# Mirror images of loser_autopsy's loss thresholds (sign-flipped). Documented
# module constants (tests assert exact behaviour at each boundary) — not
# tunables Opus ever sees.
FAST_HOLD_DAYS = 1.0    # closed inside a day ⇒ a fast cut / scalp
SLOW_HOLD_DAYS = 5.0    # held ≥ this through the gain ⇒ let it run
BIG_WIN_PCT = 15.0      # ≥ this ⇒ the thesis was strongly right
SMALL_WIN_PCT = 3.0     # < this ⇒ a shallow gain


def _classify(hold_days: float | None, pnl_pct: float | None) -> str:
    """Objective, deterministic success-mode label for one winning round-trip.

    Precedence is intentional and documented — the exact mirror of
    ``loser_autopsy._classify``:

    * ``HOME_RUN`` — gain ≥ ``BIG_WIN_PCT`` regardless of hold time: the
      entry thesis was strongly right.
    * ``SCALP`` — closed inside ``FAST_HOLD_DAYS`` at a shallow gain
      (``< SMALL_WIN_PCT``): cut a winner fast for almost nothing — the
      disposition effect, surfaced per-trade.
    * ``SLOW_GRIND`` — held ``≥ SLOW_HOLD_DAYS`` and still closed green: let
      a winner run (the good disposition behaviour to repeat).
    * ``TARGET_HIT`` — everything else: a moderate gain on a moderate hold.

    ``hold_days``/``pnl_pct`` may be ``None`` (parse failure / zero-cost
    round-trip). A ``None`` pnl_pct can never trip the magnitude arms, and a
    ``None`` hold_days can never trip the duration arms, so the function
    always returns a label and never raises.
    """
    big = pnl_pct is not None and pnl_pct >= BIG_WIN_PCT
    if big:
        return "HOME_RUN"
    fast = hold_days is not None and hold_days < FAST_HOLD_DAYS
    shallow = pnl_pct is not None and pnl_pct < SMALL_WIN_PCT
    if fast and shallow:
        return "SCALP"
    slow = hold_days is not None and hold_days >= SLOW_HOLD_DAYS
    if slow:
        return "SLOW_GRIND"
    return "TARGET_HIT"


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


def build_winner_autopsy(trades: list[dict],
                         best_n: int = 10,
                         now: datetime | None = None) -> dict:
    """Per-closed-winning-round-trip post-mortem. Pure, never raises.

    ``trades`` must be a ``Store.recent_trades()``-shaped ledger ordered
    **oldest→newest** — ``build_round_trips`` reads rows in sequence and does
    not sort. Pass exactly what ``/api/analytics`` & ``/api/loser-autopsy``
    pass: ``list(reversed(store.recent_trades(2000)))``.

    ``best_n`` caps the per-winner card list (sorted most-positive first);
    the aggregates are always over *all* winners.
    """
    now = now or datetime.now(timezone.utc)
    rts = build_round_trips(trades)
    n_rts = len(rts)

    by_id: dict = {}
    for t in trades:
        tid = t.get("id")
        if tid is not None:
            by_id[tid] = t

    # Strict > 0 winner convention — identical to round_trips/#10 &
    # trade_asymmetry/loser_autopsy (a sub-cent wash reads as a non-win,
    # exactly as it reads as a non-loss).
    winners = [rt for rt in rts if (rt.get("pnl_usd") or 0.0) > 0]
    n_winners = len(winners)

    cards: list[dict] = []
    for rt in winners:
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
            "success_mode": mode,
            "entry_reason": _reason_for(rt.get("entry_trade_ids") or [],
                                        by_id, pick_last=False),
            "exit_reason": _reason_for(rt.get("exit_trade_ids") or [],
                                       by_id, pick_last=True),
        })

    # Most profitable first; ticker is a deterministic tie-break so the card
    # order is stable for two identical-gain round-trips.
    cards.sort(key=lambda c: (-c["pnl_usd"], c["ticker"] or ""))

    total_gain = round(sum(c["pnl_usd"] for c in cards), 4) if cards else 0.0
    best = cards[0] if cards else None
    avg_gain = round(total_gain / n_winners, 4) if n_winners else None
    hold_vals = [c["hold_days"] for c in cards if c["hold_days"] is not None]
    median_hold = None
    if hold_vals:
        sv = sorted(hold_vals)
        m = len(sv) // 2
        median_hold = (sv[m] if len(sv) % 2
                       else round((sv[m - 1] + sv[m]) / 2.0, 4))

    # Which name is the engine — $ gained per ticker (most positive first).
    by_ticker: dict[str, dict] = {}
    for c in cards:
        b = by_ticker.setdefault(c["ticker"], {"ticker": c["ticker"],
                                               "n": 0, "gain_usd": 0.0})
        b["n"] += 1
        b["gain_usd"] = round(b["gain_usd"] + c["pnl_usd"], 4)
    ticker_breakdown = sorted(by_ticker.values(),
                              key=lambda b: (-b["gain_usd"], b["ticker"] or ""))

    # Which success mode dominates.
    by_mode: dict[str, int] = {}
    for c in cards:
        by_mode[c["success_mode"]] = by_mode.get(c["success_mode"], 0) + 1
    # Deterministic dominant: most frequent, ties broken by a fixed
    # significance order so the verdict never flips on dict insertion order
    # (the exact mirror of loser_autopsy's _SEVERITY tie-break).
    _SIGNIFICANCE = ["HOME_RUN", "SLOW_GRIND", "TARGET_HIT", "SCALP"]
    dominant_mode = None
    if by_mode:
        dominant_mode = max(
            by_mode, key=lambda m: (by_mode[m], -_SIGNIFICANCE.index(m)))

    # Names that won more than once — distinct from churn (which counts
    # *re-entry cadence*); this is "which winning names keep recurring".
    repeat_winners = [b["ticker"] for b in ticker_breakdown if b["n"] >= 2]

    # ---- state / verdict (verdict gated to STABLE) ---------------------
    if n_rts == 0:
        state = "NO_DATA"
    elif n_winners == 0:
        state = "NO_WINS"
    elif n_winners >= STABLE_MIN_WINNERS:
        state = "STABLE"
    else:
        state = "EMERGING"

    verdict = None
    if state == "STABLE":
        verdict = dominant_mode

    # ---- headline ------------------------------------------------------
    if state == "NO_DATA":
        headline = "No closed round-trips yet — nothing to autopsy."
    elif state == "NO_WINS":
        headline = (f"No winning round-trips across {n_rts} closed — "
                    f"nothing to autopsy.")
    else:
        best_clause = ""
        if best is not None:
            bp = (f" ({best['pnl_pct']:+.1f}%)"
                  if best.get("pnl_pct") is not None else "")
            best_clause = (f" Best: {best['ticker']} "
                           f"${best['pnl_usd']:+.2f}{bp}.")
        engine_clause = ""
        if ticker_breakdown:
            tb = ticker_breakdown[0]
            engine_clause = (f" {tb['ticker']} is the engine "
                             f"(${tb['gain_usd']:+.2f} over {tb['n']} "
                             f"win{'s' if tb['n'] != 1 else ''}).")
        if state == "EMERGING":
            headline = (
                f"Emerging — {n_winners} of {STABLE_MIN_WINNERS} winning "
                f"round-trips for a stable pattern read (verdict withheld). "
                f"${total_gain:+.2f} gained so far."
                f"{best_clause}{engine_clause}")
        else:
            mc = by_mode.get(dominant_mode, 0)
            headline = (
                f"{dominant_mode} dominates — {mc} of {n_winners} winning "
                f"round-trips; ${total_gain:+.2f} total realised "
                f"gain.{best_clause}{engine_clause}")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "n_round_trips": n_rts,
        "n_winners": n_winners,
        "total_gain_usd": total_gain,
        "avg_gain_usd": avg_gain,
        "median_winner_hold_days": median_hold,
        "dominant_success_mode": dominant_mode,
        "success_mode_counts": by_mode,
        "ticker_breakdown": ticker_breakdown,
        "repeat_winners": repeat_winners,
        "best_winners": cards[:max(0, best_n)],
        "stable_min_winners": STABLE_MIN_WINNERS,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json

    from paper_trader.store import get_store
    s = get_store()
    rep = build_winner_autopsy(list(reversed(s.recent_trades(2000))))
    print(json.dumps(rep, indent=2, default=str))
