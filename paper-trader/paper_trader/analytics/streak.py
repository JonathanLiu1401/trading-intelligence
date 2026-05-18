"""Win/loss streak analysis on closed round-trips.

Behavioural-edge diagnostic that no existing panel covers. ``trade_asymmetry``
gives payoff math, ``winner_autopsy`` / ``loser_autopsy`` narrate per-trade
outcomes, ``churn`` counts re-entry cadence — none surface the *streak
structure* of the closed-trip series itself.

Two things a desk wants from a streak panel:

* The **current run** — am I on a hot hand or a cold streak right now? (The
  most recent consecutive same-sign closes counting backward from the latest
  exit.) Useful for surfacing potential *tilt* after a loss cluster or
  *overconfidence* after a win cluster.
* The **historical extremes** — what are the longest winning and losing
  streaks this book has produced? Context for whether the current run is
  normal or unusual.

Single source of truth: consumes ``round_trips.build_round_trips`` (AGENTS.md
invariant #10). No P&L recompute, no hold-time recompute. Flat closes
(``pnl_usd == 0``) are skipped from the streak run (they break neither a
winning nor a losing streak) but still counted in ``n_round_trips`` for the
sample-size view.

Sample-size honesty mirrors ``winner_autopsy`` / ``loser_autopsy`` /
``trade_asymmetry``: counts and the recent sequence are emitted from the
first closed round-trip, but the **behavioural verdict** (``HOT_HAND`` /
``TILT_RISK`` / ``NEUTRAL``) is withheld until ``STABLE``
(``n_round_trips >= STABLE_MIN_ROUND_TRIPS``) — a three-trip "streak" is
noise.

Advisory only — never gates Opus, never injected into the decision prompt,
adds no caps (AGENTS.md #2/#12).
"""
from __future__ import annotations

from datetime import datetime, timezone

from .round_trips import build_round_trips

STABLE_MIN_ROUND_TRIPS = 8
HOT_HAND_MIN = 4
TILT_RISK_MIN = 4
RECENT_SEQUENCE_LEN = 20


def _outcome(rt: dict) -> str:
    """Classify a round-trip as ``W`` / ``L`` / ``F`` (flat).

    Flats use the exact zero-pnl convention; floating-point noise on a true
    breakeven is rare for paper trading (qty × price differences are integer
    cents) but if it ever bites, the symmetric ``< 0`` / ``> 0`` test keeps
    the read honest rather than collapsing tiny gains/losses into "flat".
    """
    pnl = rt.get("pnl_usd")
    if pnl is None:
        return "F"
    if pnl > 0:
        return "W"
    if pnl < 0:
        return "L"
    return "F"


def _current_streak(outcomes: list[str], exits: list[str | None]) -> dict:
    """Count consecutive same-sign closes from the most recent backward.

    Flats are skipped (don't break a streak, don't extend it either) — they
    represent a true zero-P&L close, which is neither a win nor a loss for
    the purposes of "am I on a run?". The first non-flat from the right
    seeds the streak kind; the walk continues only while subsequent non-flat
    entries match.
    """
    if not outcomes:
        return {"kind": "NONE", "length": 0, "since_ts": None}
    kind: str | None = None
    length = 0
    since_idx: int | None = None
    for i in range(len(outcomes) - 1, -1, -1):
        o = outcomes[i]
        if o == "F":
            continue
        if kind is None:
            kind = o
            length = 1
            since_idx = i
            continue
        if o == kind:
            length += 1
            since_idx = i
        else:
            break
    if kind is None:
        return {"kind": "NONE", "length": 0, "since_ts": None}
    return {
        "kind": "WIN" if kind == "W" else "LOSS",
        "length": length,
        "since_ts": exits[since_idx] if since_idx is not None else None,
    }


def _longest_run(outcomes: list[str], target: str) -> int:
    """Longest run of ``target`` (``W`` or ``L``) in the series.

    Flats reset the running counter — they don't extend the streak but they
    do break it. (Symmetric with ``_current_streak``'s skip behaviour from
    the *right edge*; a flat in the middle of historical W's still ends that
    historical run.)
    """
    best = 0
    run = 0
    for o in outcomes:
        if o == target:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def build_streak(trades: list[dict]) -> dict:
    """Compute the current streak, longest historical streaks, and recent
    W/L sequence for the closed round-trip series.

    ``trades`` must be oldest → newest (same convention as
    ``winner_autopsy`` / ``loser_autopsy``).
    """
    now = datetime.now(timezone.utc)

    rts = build_round_trips(trades)
    n_rts = len(rts)

    outcomes = [_outcome(rt) for rt in rts]
    exits = [rt.get("exit_ts") for rt in rts]

    n_wins = sum(1 for o in outcomes if o == "W")
    n_losses = sum(1 for o in outcomes if o == "L")
    n_flats = sum(1 for o in outcomes if o == "F")

    cur = _current_streak(outcomes, exits)
    longest_win = _longest_run(outcomes, "W")
    longest_loss = _longest_run(outcomes, "L")

    recent_sequence = outcomes[-RECENT_SEQUENCE_LEN:]

    # ---- state / verdict (verdict gated to STABLE) ---------------------
    if n_rts == 0:
        state = "NO_DATA"
    elif n_rts >= STABLE_MIN_ROUND_TRIPS:
        state = "STABLE"
    else:
        state = "EMERGING"

    verdict: str | None = None
    if state == "STABLE":
        if cur["kind"] == "WIN" and cur["length"] >= HOT_HAND_MIN:
            verdict = "HOT_HAND"
        elif cur["kind"] == "LOSS" and cur["length"] >= TILT_RISK_MIN:
            verdict = "TILT_RISK"
        else:
            verdict = "NEUTRAL"

    # ---- headline ------------------------------------------------------
    if state == "NO_DATA":
        headline = "No closed round-trips yet — no streak to read."
    else:
        if cur["kind"] == "WIN":
            run_clause = f"{cur['length']}-win run"
        elif cur["kind"] == "LOSS":
            run_clause = f"{cur['length']}-loss run"
        else:
            run_clause = "no active run"
        extremes = (f"longest W={longest_win}, longest L={longest_loss} "
                    f"across {n_rts} round-trip{'s' if n_rts != 1 else ''}")
        if state == "EMERGING":
            headline = (
                f"Emerging — {n_rts} of {STABLE_MIN_ROUND_TRIPS} round-trips "
                f"for a stable streak read (verdict withheld). Current: "
                f"{run_clause}. {extremes}.")
        elif verdict == "HOT_HAND":
            headline = (f"HOT_HAND — on a {cur['length']}-win run "
                        f"(threshold {HOT_HAND_MIN}). {extremes}.")
        elif verdict == "TILT_RISK":
            headline = (f"TILT_RISK — on a {cur['length']}-loss run "
                        f"(threshold {TILT_RISK_MIN}). {extremes}.")
        else:
            headline = f"NEUTRAL — {run_clause}. {extremes}."

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "n_round_trips": n_rts,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_flats": n_flats,
        "current_streak": cur,
        "longest_win_streak": longest_win,
        "longest_loss_streak": longest_loss,
        "recent_sequence": recent_sequence,
        "stable_min_round_trips": STABLE_MIN_ROUND_TRIPS,
        "hot_hand_min": HOT_HAND_MIN,
        "tilt_risk_min": TILT_RISK_MIN,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json

    from paper_trader.store import get_store
    s = get_store()
    rep = build_streak(list(reversed(s.recent_trades(2000))))
    print(json.dumps(rep, indent=2, default=str))
