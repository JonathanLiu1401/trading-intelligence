"""Kelly-criterion position sizing diagnostic.

The desk question nothing else answers. Realised win-rate + payoff ratio
already live in ``/api/trade-asymmetry``; ``/api/concentration-cap`` warns
when the top name exceeds a hard configured cap. What no panel does is the
*statistical* sizing read — given the trader's own realised win-rate and
payoff, what fraction of the book would a Kelly-optimal sizer allocate to
the single best position, and how does the **current** top-position weight
compare?

Kelly is the gambling/portfolio-theory result for the bet fraction that
maximises long-run log-wealth growth. For a single bet with win probability
``p``, loss probability ``q = 1 − p``, and payoff ratio ``b`` (avg win /
|avg loss|), the optimal fraction is::

    f* = p − q/b

The full-Kelly fraction sits at the *peak* of the log-growth curve — it
maximises long-run growth but at the cost of large interim drawdowns. For
trading desks the standard advice is **half-Kelly** (≈ same growth at ¼ the
variance) or **quarter-Kelly** (when sampling error is real). This module
emits all three so the operator can pick a personal point on the curve, and
benchmarks the **current top-position weight** against half-Kelly.

Distinct from its neighbours:

* ``/api/trade-asymmetry`` — emits ``payoff_ratio`` and
  ``actual_win_rate_pct`` already. We **consume** that math; we do not
  recompute it.
* ``/api/concentration-cap`` — fixed-threshold concentration warning. Says
  "the cap is 30%, you are at 65%". Says nothing about whether 30% (or any
  other number) is statistically justified by the realised edge.
* ``/api/risk-adjusted-returns`` — Sharpe/Sortino on daily portfolio
  returns. A different ratio entirely; ignores per-trade win-rate.

Sample-size honesty mirrors ``trade_asymmetry.py``: numerics are emitted as
soon as a payoff ratio is defined (at least one winner and one loser), but
the **verdict label** is withheld until ``STABLE``
(``n_round_trips >= STABLE_MIN_RTS``). A 3-trade Kelly read is noise. The
verdict is also gated on a defined payoff ratio (all-winners or all-losers
books cannot be sized by Kelly).

Observational only — never gates Opus and adds no caps (#2/#12).
"""
from __future__ import annotations

from datetime import datetime, timezone

from .round_trips import build_round_trips
from .trade_asymmetry import build_trade_asymmetry

# Verdict is only labelled once the realised sample is large enough that
# Kelly inputs (p, b) will not flip on the next trade. Mirrors
# ``trade_asymmetry.STABLE_MIN_RTS``. Below this it's EMERGING.
STABLE_MIN_RTS = 20

# Half-Kelly is the standard practitioner recommendation. The verdict
# bands are anchored on the *half-Kelly* number (less aggressive than full
# Kelly, more useful than quarter-Kelly as the central comparison).
#
# Why these band edges:
#   < 0.5×HK     ⇒ UNDERSIZED   — could deploy materially more given edge
#   [0.5, 1.25]  ⇒ KELLY_ALIGNED — within ±25% of the half-Kelly target
#   (1.25, 2.0]  ⇒ OVERSIZED    — beyond the safety-margin cushion
#   > 2.0×HK     ⇒ EXTREMELY_OVERSIZED — above *full Kelly* itself (ruin tail)
UNDERSIZED_MAX_RATIO = 0.5
ALIGNED_MAX_RATIO = 1.25
OVERSIZED_MAX_RATIO = 2.0


def _kelly_fraction(win_rate_pct: float | None,
                    payoff_ratio: float | None) -> float | None:
    """Full-Kelly fraction in percent, or ``None`` if undefined.

    Returns ``f* = p − q/b`` expressed in %. May be negative
    (negative-edge book) — callers decide how to surface that. Clamped to
    ``[-100, 100]`` for downstream UI sanity.

    ``win_rate_pct`` is expected in [0, 100] (matches ``trade_asymmetry``
    output). A ``payoff_ratio <= 0`` is treated as undefined: Kelly
    requires a positive ratio of avg-win to |avg-loss| and our caller
    ensures this, but a defensive guard avoids div-by-zero.
    """
    if win_rate_pct is None or payoff_ratio is None:
        return None
    if payoff_ratio <= 0:
        return None
    p = win_rate_pct / 100.0
    q = 1.0 - p
    f = p - q / payoff_ratio
    f_pct = f * 100.0
    if f_pct < -100.0:
        return -100.0
    if f_pct > 100.0:
        return 100.0
    return round(f_pct, 4)


def build_kelly_sizing(trades: list[dict],
                       top_position_pct: float | None = None,
                       top_position_ticker: str | None = None,
                       now: datetime | None = None) -> dict:
    """Kelly-sizing diagnostic over closed round-trips. Pure, never raises.

    ``trades`` must be ``Store.recent_trades()``-shaped, oldest→newest.
    ``top_position_pct`` is the current largest position's weight as
    percentage of total book value (matches ``/api/risk``'s
    ``concentration_top1_pct``). ``top_position_ticker`` is the name.
    Both default to ``None`` for the "book is all-cash" case — the Kelly
    numerics still emit, the comparison fields degrade to ``None``.
    """
    now = now or datetime.now(timezone.utc)

    # SSOT: pull payoff_ratio / actual_win_rate_pct from trade_asymmetry,
    # never recompute (invariant #10 — keep one canonical aggregator).
    asym = build_trade_asymmetry(trades, now=now)
    n_rts = asym.get("n_round_trips") or 0
    win_rate = asym.get("actual_win_rate_pct")          # in % or None
    payoff = asym.get("payoff_ratio")                   # ratio or None
    n_wins = asym.get("n_wins") or 0
    n_losses = asym.get("n_losses") or 0

    full_kelly = _kelly_fraction(win_rate, payoff)
    half_kelly = (round(full_kelly / 2.0, 4)
                  if full_kelly is not None else None)
    quarter_kelly = (round(full_kelly / 4.0, 4)
                     if full_kelly is not None else None)

    top_pct = (round(float(top_position_pct), 4)
               if top_position_pct is not None else None)

    delta_vs_half = (round(top_pct - half_kelly, 4)
                     if (top_pct is not None and half_kelly is not None)
                     else None)

    # ---- state -----------------------------------------------------------
    if n_rts == 0:
        state = "NO_DATA"
    elif payoff is None:
        # All-winners or all-losers — Kelly is undefined.
        state = "UNDEFINED_PAYOFF"
    elif n_rts >= STABLE_MIN_RTS:
        state = "STABLE"
    else:
        state = "EMERGING"

    # ---- verdict (gated to STABLE + defined payoff + known top) ----------
    verdict = None
    verdict_reason = None
    if state == "STABLE" and half_kelly is not None and top_pct is not None:
        # Negative edge ⇒ Kelly says DO NOT BET — any positive size is too big.
        if full_kelly is not None and full_kelly <= 0.0:
            verdict = "NEGATIVE_EDGE"
            verdict_reason = (
                f"realised edge is negative (Kelly fraction "
                f"{full_kelly:+.1f}%): the math says hold cash, any sizing "
                f"is over-betting")
        elif half_kelly <= 0.0:
            # Defensive: positive full_kelly with half_kelly==0 shouldn't
            # happen, but never emit a ratio against zero.
            verdict = None
        else:
            ratio = top_pct / half_kelly
            if ratio < UNDERSIZED_MAX_RATIO:
                verdict = "UNDERSIZED"
                verdict_reason = (
                    f"top position at {top_pct:.1f}% is "
                    f"{ratio:.2f}× half-Kelly ({half_kelly:.1f}%) — "
                    f"could deploy materially more given the realised edge")
            elif ratio <= ALIGNED_MAX_RATIO:
                verdict = "KELLY_ALIGNED"
                verdict_reason = (
                    f"top position at {top_pct:.1f}% is "
                    f"{ratio:.2f}× half-Kelly ({half_kelly:.1f}%) — sized "
                    f"in the safety cushion around the optimal growth point")
            elif ratio <= OVERSIZED_MAX_RATIO:
                verdict = "OVERSIZED"
                verdict_reason = (
                    f"top position at {top_pct:.1f}% is "
                    f"{ratio:.2f}× half-Kelly ({half_kelly:.1f}%) — beyond "
                    f"the cushion, drawdown variance climbs sharply")
            else:
                verdict = "EXTREMELY_OVERSIZED"
                verdict_reason = (
                    f"top position at {top_pct:.1f}% is "
                    f"{ratio:.2f}× half-Kelly ({half_kelly:.1f}%) and above "
                    f"FULL Kelly ({full_kelly:.1f}%) — ruin tail of the "
                    f"long-run growth curve")

    # ---- headline --------------------------------------------------------
    name_clause = (f" ({top_position_ticker})"
                   if top_position_ticker and top_pct is not None else "")
    if state == "NO_DATA":
        headline = ("No closed round-trips yet — Kelly framework needs "
                    "both wins and losses to size.")
    elif state == "UNDEFINED_PAYOFF":
        if n_wins == 0:
            why = "no winning round-trips yet"
        elif n_losses == 0:
            why = "no losing round-trips yet"
        else:
            why = "payoff ratio undefined"
        headline = (f"Kelly undefined — {why} across {n_rts} closed "
                    f"round-trip(s); ratio of avg-win to |avg-loss| needs "
                    f"both sides.")
    elif state == "EMERGING":
        hk_s = f"{half_kelly:.1f}%" if half_kelly is not None else "n/a"
        top_s = (f"{top_pct:.1f}%{name_clause}"
                 if top_pct is not None else "n/a (all cash)")
        headline = (
            f"Emerging — {n_rts} of {STABLE_MIN_RTS} round-trips for a "
            f"stable Kelly read. So far: half-Kelly target {hk_s}, top "
            f"position {top_s} (verdict withheld until n≥{STABLE_MIN_RTS}).")
    elif verdict is not None:
        # STABLE + verdict assigned — headline is the verdict reason
        # prefixed with the verdict label (mirror trade_asymmetry style).
        headline = f"{verdict} — {verdict_reason}."
    else:
        # STABLE but verdict could not be assigned (top_pct is None — book
        # all cash, or some other degenerate case).
        hk_s = f"{half_kelly:.1f}%" if half_kelly is not None else "n/a"
        headline = (
            f"Stable Kelly read across {n_rts} round-trips: half-Kelly "
            f"target {hk_s}; no current top-position weight to benchmark "
            f"against.")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "headline": headline,
        "n_round_trips": n_rts,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "actual_win_rate_pct": win_rate,
        "payoff_ratio": payoff,
        "full_kelly_pct": full_kelly,
        "half_kelly_pct": half_kelly,
        "quarter_kelly_pct": quarter_kelly,
        "top_position_pct": top_pct,
        "top_position_ticker": top_position_ticker,
        "delta_vs_half_kelly_pct": delta_vs_half,
        "stable_min_round_trips": STABLE_MIN_RTS,
        "thresholds": {
            "undersized_max_ratio": UNDERSIZED_MAX_RATIO,
            "aligned_max_ratio": ALIGNED_MAX_RATIO,
            "oversized_max_ratio": OVERSIZED_MAX_RATIO,
        },
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json
    from paper_trader.store import get_store

    s = get_store()
    rep = build_kelly_sizing(list(reversed(s.recent_trades(2000))),
                             top_position_pct=65.4,
                             top_position_ticker="NVDA")
    print(json.dumps(rep, indent=2, default=str))
