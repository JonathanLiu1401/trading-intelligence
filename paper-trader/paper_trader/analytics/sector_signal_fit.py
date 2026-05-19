"""Sector-level position vs. signal-density divergence — the per-sector
"are you allocated where the wire is pointing?" view.

The desk already has two adjacent surfaces:
  * ``/api/sector-exposure`` — % of the book in each sector RIGHT NOW (the
    risk-side concentration view fed into the prompt).
  * ``/api/news-deduped`` / ``/api/signals`` — what the wire is talking
    about right now (the opportunity-side stream).

But it had no view answering the discretionary question a portfolio manager
asks every morning: *am I allocated where the signals are pointing, or am
I holding sectors the wire has moved on from?* A 60% SEMIS book with the
last 6h of wire activity 80% MEMORY-coverage is two different stories — one
where you're overweight a quiet sector (de-risk candidate), one where the
wire is finally catching up (lean in).

This module composes the existing single-source-of-truth
``build_sector_exposure`` output with a sector-weighted ai_score rollup of
the live signal stream, and reports the per-sector GAP and a top-level
ALIGNED / MISALIGNED verdict.

**Single source of truth.** Sector classification uses the existing
``analytics.sector_exposure.classify`` function (which is itself a verbatim
mirror of ``dashboard._classify`` — drift-locked). The position weight
column comes from the already-built ``sector_exposure`` dict, never
re-derived. The signal-share denominator is the total weighted score sum
so it ALWAYS sums to 100.0% across surfaced sectors — no fabricated
percentage that doesn't add up.

**Observational, never prescriptive.** Same contract as
``sector_exposure`` / ``risk_mirror`` (AGENTS.md invariants #2/#12):
states facts, issues no directive, never gates a trade.

Pure and deterministic (no clock, no IO). Never raises — caller may invoke
without a ``_safe`` wrapper.
"""
from __future__ import annotations

from typing import Any

from .sector_exposure import classify


# A gap of this magnitude (in absolute % points) is the threshold above
# which a sector flips from ALIGNED to one of the directional verdicts.
# Tuned for a 5-10 sector book where most weights are sub-30%: a 15-point
# gap is the smallest meaningful "I'm overweight / underweight" claim.
GAP_THRESHOLD_PCT = 15.0


def _f(x: Any, default: float = 0.0) -> float:
    """Best-effort float coercion — garbage degrades to default, never raises."""
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _signal_tickers(sig: Any) -> list[str]:
    """Extract uppercased ticker list from a signal row. Tolerates missing /
    non-list / non-string elements without raising."""
    if not isinstance(sig, dict):
        return []
    raw = sig.get("tickers") or []
    if not isinstance(raw, (list, tuple, set)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        if not isinstance(t, str):
            continue
        u = t.strip().upper()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def build_sector_signal_fit(
    sector_exposure: Any,
    signals: Any,
    *,
    gap_threshold_pct: float = GAP_THRESHOLD_PCT,
) -> dict:
    """Compose sector position weights with sector-weighted signal density.

    ``sector_exposure`` — output of ``analytics.sector_exposure.build_sector_exposure``
    (must carry ``sector_pct`` — the per-sector % of total book value).
    ``signals`` — list of signal dicts (output of ``signals.get_top_signals``,
    each with a ``tickers`` list and an ``ai_score`` numeric).

    Each signal contributes its ``ai_score`` to its sector(s). A multi-sector
    signal splits its weight evenly across mentioned sectors (so a 10-ticker
    article doesn't 10× distort the share — the sector_share is per-sector,
    not per-ticker-mention). Unknown tickers map to the ``"other"`` sector
    (same as ``sector_exposure.classify``); a signal with NO ticker mentions
    contributes nothing (silence, not "other" — the wire's coverage is not
    "other-sector noise" if no ticker was extracted).

    Returns:
        {
          "state": "ALIGNED" | "MISALIGNED" | "NO_DATA",
          "summary": str,
          "sectors": [
            {"sector": str, "position_pct": float, "signal_share_pct": float,
             "n_signals": int, "max_signal_score": float, "gap_pct": float,
             "verdict": "OVERWEIGHT" | "UNDERWEIGHT" | "ALIGNED"}
          ],
          "max_gap_pct": float,         # largest absolute gap
          "max_gap_sector": str | None, # the sector that owns max_gap_pct
          "n_sectors": int,
          "n_signals_used": int,
          "n_signals_with_no_tickers": int,
        }

    Pure / total — never raises; bad inputs degrade to ``NO_DATA``.
    """
    sx = sector_exposure if isinstance(sector_exposure, dict) else {}
    raw_pos_pct = sx.get("sector_pct")
    pos_pct: dict[str, float] = {}
    if isinstance(raw_pos_pct, dict):
        for k, v in raw_pos_pct.items():
            if isinstance(k, str):
                pos_pct[k] = _f(v)

    sigs = signals if isinstance(signals, (list, tuple)) else []

    signal_weight: dict[str, float] = {}
    signal_count: dict[str, int] = {}
    max_signal_score: dict[str, float] = {}
    n_used = 0
    n_no_tickers = 0
    for sig in sigs:
        tickers = _signal_tickers(sig)
        if not tickers:
            n_no_tickers += 1
            continue
        score = _f(sig.get("ai_score") if isinstance(sig, dict) else 0.0)
        if score <= 0:
            continue
        # Distinct sectors this signal touches (so a 4-ticker article all in
        # SEMIS contributes once to SEMIS, not 4x; a 2-ticker article spanning
        # SEMIS+TECH splits its weight 50/50).
        sectors: set[str] = set()
        for tk in tickers:
            sectors.add(classify(tk))
        if not sectors:
            continue
        share = score / len(sectors)
        for sec in sectors:
            signal_weight[sec] = signal_weight.get(sec, 0.0) + share
            signal_count[sec] = signal_count.get(sec, 0) + 1
            if score > max_signal_score.get(sec, 0.0):
                max_signal_score[sec] = score
        n_used += 1

    total_signal_weight = sum(signal_weight.values())
    has_position = bool(pos_pct)
    has_signal = total_signal_weight > 0

    if not has_position and not has_signal:
        return {
            "state": "NO_DATA",
            "summary": "no positions and no scored signals — fit is undefined",
            "sectors": [],
            "max_gap_pct": 0.0,
            "max_gap_sector": None,
            "n_sectors": 0,
            "n_signals_used": n_used,
            "n_signals_with_no_tickers": n_no_tickers,
        }

    all_sectors = sorted(set(pos_pct) | set(signal_weight))
    rows: list[dict] = []
    max_gap_abs = 0.0
    max_gap_sector: str | None = None
    for sec in all_sectors:
        p_pct = pos_pct.get(sec, 0.0)
        sig_share = (
            (signal_weight.get(sec, 0.0) / total_signal_weight * 100.0)
            if total_signal_weight > 0 else 0.0
        )
        gap = p_pct - sig_share
        abs_gap = abs(gap)
        if gap > gap_threshold_pct:
            verdict = "OVERWEIGHT"
        elif gap < -gap_threshold_pct:
            verdict = "UNDERWEIGHT"
        else:
            verdict = "ALIGNED"
        rows.append({
            "sector": sec,
            "position_pct": round(p_pct, 2),
            "signal_share_pct": round(sig_share, 2),
            "n_signals": int(signal_count.get(sec, 0)),
            "max_signal_score": round(max_signal_score.get(sec, 0.0), 2),
            "gap_pct": round(gap, 2),
            "verdict": verdict,
        })
        if abs_gap > max_gap_abs:
            max_gap_abs = abs_gap
            max_gap_sector = sec

    # Top-level verdict: ALIGNED iff every sector is within threshold.
    state = "ALIGNED" if max_gap_abs <= gap_threshold_pct else "MISALIGNED"

    # Sort rows by descending |gap| so the analyst sees the biggest divergences
    # first; ties broken by sector name for determinism.
    rows.sort(key=lambda r: (-abs(r["gap_pct"]), r["sector"]))

    if state == "ALIGNED":
        summary = (
            f"sector position weights within ±{gap_threshold_pct:.0f}% of "
            f"signal share across {len(rows)} sector(s)"
        )
    else:
        # Compose a one-line summary that names the most-divergent sector and
        # its direction — so the verdict isn't just a flag, it's an answer.
        top = next((r for r in rows if r["sector"] == max_gap_sector), None)
        if top:
            direction = top["verdict"].lower()
            summary = (
                f"{top['sector']} is {direction}: position {top['position_pct']:.1f}% "
                f"vs signal share {top['signal_share_pct']:.1f}% "
                f"(gap {top['gap_pct']:+.1f} pts)"
            )
        else:
            summary = (
                f"largest sector gap {max_gap_abs:.1f} pts exceeds "
                f"the ±{gap_threshold_pct:.0f}% alignment threshold"
            )

    return {
        "state": state,
        "summary": summary,
        "sectors": rows,
        "max_gap_pct": round(max_gap_abs, 2),
        "max_gap_sector": max_gap_sector,
        "n_sectors": len(rows),
        "n_signals_used": n_used,
        "n_signals_with_no_tickers": n_no_tickers,
    }
