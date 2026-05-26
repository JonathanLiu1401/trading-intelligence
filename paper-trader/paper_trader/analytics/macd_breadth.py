"""Watchlist MACD breadth — counts bullish / bearish / flat / unknown
names across the quant-signal snapshot.

A live trader pages on aggregate momentum breadth ("is the tape
trending or stalled?") but the prompt's quant block dumps a per-name
list — answering it requires Opus to mentally tally rows on every
decision. The MACD label now carries a "flat" state (``strategy.
_macd_live`` + ``backtest._macd`` fix, AGENTS.md pass #38 sibling),
so a one-line breadth headline is the natural roll-up: how many
names actually carry directional momentum vs. sitting in steady
state.

Returns a JSON-safe dict — verdict + counts + headline string the
caller renders verbatim (the standard analytics-builder shape used
by ``risk_mirror`` / ``sector_exposure`` / ``stress_scenarios``).

Verdicts:
  * ``NO_DATA``       — len(quant_sigs) == 0
  * ``MIXED``         — neither bullish nor bearish dominates (both
                         within ``DOMINANCE_PP=0.10`` of each other)
  * ``BULL_BREADTH``  — bullish share > bearish share by > 0.10
  * ``BEAR_BREADTH``  — bearish share > bullish share by > 0.10
  * ``FLAT_TAPE``     — flat share is the majority (≥ 0.50)

Observational only — never gates Opus, no caps, no path to
``_execute()`` (AGENTS.md invariants #2 / #12 — the
``sector_exposure`` precedent). Pure: no I/O, never raises.

CLI for an ops one-shot::

    python3 -m paper_trader.analytics.macd_breadth
"""
from __future__ import annotations

import json


# Dominance gap above which BULL/BEAR breadth wins over MIXED. Tight
# enough that a 55/45 split is MIXED (the tape is contested) but a
# 60/40 split tips the verdict (one side dominates by 0.20).
DOMINANCE_PP = 0.10

# Flat-share majority threshold. ≥ 50% of named labels reading "flat"
# means the prompt's per-name MACD lines mostly carry no momentum
# signal — the FLAT_TAPE verdict.
FLAT_MAJORITY = 0.50

# Verdict→headline templates. Each carries the bullish/bearish/flat
# counts so the operator can read the split without computing it.
# Headline kept under ~120 chars so it fits the prompt MARKET STRUCTURE
# line without wrapping the rendered token.
_HEADLINES = {
    "NO_DATA":      "MACD BREADTH: no quant data this cycle",
    "MIXED":        "MACD BREADTH: MIXED (bullish={b}, bearish={r}, flat={f}, n/a={u} of {n})",
    "BULL_BREADTH": "MACD BREADTH: BULL ({b}/{n} bullish vs {r} bearish, flat={f})",
    "BEAR_BREADTH": "MACD BREADTH: BEAR ({r}/{n} bearish vs {b} bullish, flat={f})",
    "FLAT_TAPE":    "MACD BREADTH: FLAT TAPE ({f}/{n} flat, bullish={b}, bearish={r})",
}


def _classify(quant_sigs: dict[str, dict]) -> dict[str, int]:
    """Count names per MACD label.

    Reads the ``MACD`` key per row. None / missing / unknown labels
    are counted as ``unknown`` so the total always equals
    ``len(quant_sigs)`` — the operator can sanity-check the breakdown.
    """
    counts = {"bullish": 0, "bearish": 0, "flat": 0, "unknown": 0}
    for _tk, q in (quant_sigs or {}).items():
        if not isinstance(q, dict):
            counts["unknown"] += 1
            continue
        label = q.get("MACD")
        if label == "bullish":
            counts["bullish"] += 1
        elif label == "bearish":
            counts["bearish"] += 1
        elif label == "flat":
            counts["flat"] += 1
        else:
            counts["unknown"] += 1
    return counts


def _verdict(counts: dict[str, int], n: int) -> str:
    """Pure verdict ladder over the per-label counts. ``n`` is the total
    name count (== sum of counts.values())."""
    if n == 0:
        return "NO_DATA"
    bull_share = counts["bullish"] / n
    bear_share = counts["bearish"] / n
    flat_share = counts["flat"] / n
    if flat_share >= FLAT_MAJORITY:
        return "FLAT_TAPE"
    if bull_share - bear_share > DOMINANCE_PP:
        return "BULL_BREADTH"
    if bear_share - bull_share > DOMINANCE_PP:
        return "BEAR_BREADTH"
    return "MIXED"


def build_macd_breadth(quant_sigs: dict[str, dict] | None) -> dict:
    """Single source of truth for watchlist MACD breadth (invariant
    #10). Returns the standard envelope: verdict + headline + counts.

    Never raises — any failure degrades to a ``NO_DATA`` envelope so
    the caller (prompt builder, dashboard endpoint, hourly summary)
    can ship the row verbatim and the live decision loop stays alive.
    """
    if not quant_sigs:
        return {
            "verdict": "NO_DATA",
            "headline": _HEADLINES["NO_DATA"],
            "n": 0,
            "counts": {"bullish": 0, "bearish": 0, "flat": 0, "unknown": 0},
            "shares": {"bullish": 0.0, "bearish": 0.0,
                       "flat": 0.0, "unknown": 0.0},
        }
    try:
        counts = _classify(quant_sigs)
        n = sum(counts.values())
        verdict = _verdict(counts, n)
        # Round shares to 3dp — keeps JSON lean and avoids float-noise
        # diffs in tests (the ``_z`` precedent in sibling analytics).
        shares = {
            k: round(v / n, 3) if n > 0 else 0.0
            for k, v in counts.items()
        }
        headline = _HEADLINES[verdict].format(
            b=counts["bullish"], r=counts["bearish"],
            f=counts["flat"], u=counts["unknown"], n=n,
        )
        return {
            "verdict": verdict,
            "headline": headline,
            "n": n,
            "counts": counts,
            "shares": shares,
        }
    except Exception as e:  # pragma: no cover - defensive
        return {
            "verdict": "NO_DATA",
            "headline": f"MACD BREADTH: builder error ({e})",
            "n": 0,
            "counts": {"bullish": 0, "bearish": 0, "flat": 0, "unknown": 0},
            "shares": {"bullish": 0.0, "bearish": 0.0,
                       "flat": 0.0, "unknown": 0.0},
        }


def render_prompt_line(snap: dict | None) -> str:
    """Render the single line surfaced to Opus in the prompt.

    ``snap`` is the dict returned by ``build_macd_breadth``. Returns
    ``""`` when there is no data so the caller emits no token (the
    ``silence-when-nothing-actionable`` precedent — the live prompt
    stays clean on the first-cycle / no-quant case). Otherwise
    returns the headline string the prompt block carries.
    """
    if not snap or not isinstance(snap, dict):
        return ""
    if snap.get("verdict") == "NO_DATA":
        return ""
    headline = snap.get("headline") or ""
    return str(headline)


if __name__ == "__main__":  # pragma: no cover - CLI helper only
    import sys

    # Best-effort CLI: pull the current live quant snapshot, build the
    # breadth, print the JSON envelope. Designed for a one-shot ops
    # check, not the live loop.
    try:
        from paper_trader.strategy import (  # noqa: E402
            QUANT_TICKERS_LIVE, get_quant_signals_live,
        )
        sigs = get_quant_signals_live(QUANT_TICKERS_LIVE)
        out = build_macd_breadth(sigs)
        print(json.dumps(out, indent=2))
    except Exception as e:
        print(f"macd_breadth CLI failed: {e}", file=sys.stderr)
        sys.exit(1)
