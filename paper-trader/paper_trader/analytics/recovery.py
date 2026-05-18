"""Path back to even — the one question a losing book asks first.

The stack measures *risk* and *drawdown* exhaustively (``/api/analytics``
Sharpe/Sortino/Calmar, ``/api/drawdown`` peak-to-trough + claw-back %,
``/api/tail-risk`` VaR, ``/api/stress-scenarios`` forward shock). But a
``grep`` across the 50+ builders confirms **nothing** answers the single
question a discretionary trader staring at a −10 % book asks constantly:
*what move does it take to get back to even, which name has to do the
heavy lifting, and — given how this book actually behaves — roughly how
far is that in this book's own daily dispersion?*

``/api/drawdown`` owns the *backward* "% of trough already recovered".
``build_recovery`` is the **forward** complement and owns one distinct
thing: the rally **from here** required to return to the $1000 start
(the universal P/L baseline every Discord report is measured against)
and to the running high-water peak — per position and for the book.

SSOT composition (AGENTS.md #10 — never re-derive a number a sibling
builder already owns):

* ``current_value`` / ``peak_value`` / per-position
  ``avg_cost``/``current_price``/``unrealized_pl`` are read **verbatim**
  from ``drawdown.compute_drawdown``'s result (its ``contributors`` list
  is already the per-open-lot P&L, sorted most-negative-first, option
  ×100 baked into ``unrealized_pl``). A drift between this builder's peak
  and ``/api/drawdown``'s peak fails the no-drift test loudly.
* The daily-dispersion scale reuses ``tail_risk.build_tail_risk``'s
  realized ``annualized_vol_pct`` **verbatim**, de-annualized by
  ``/√252``. It is gated on ``tail_risk.state == "OK"``: a young book
  reads ``INSUFFICIENT`` (the live case — emit the %/$ targets, **withhold
  the dispersion figure** with an honest sentence, the
  ``tail_risk``/``correlation`` sample-size-honesty precedent).

Honesty: the σ-day figure is explicitly a **dispersion scale, not a
forecast** — a random walk's expected first-passage time to a positive
level is infinite; the headline says so (the ``stress_scenarios``
"beta-approx, not VaR" honesty tone).

State ladder: ``NO_DATA`` (no priced book) → ``ABOVE_WATER`` (book ≥ the
start — nothing to recover toward even, the Discord line self-suppresses,
the ``_drawdown_line`` at-high-water precedent) → ``UNDERWATER``.

Diagnostic / advisory only: never gates Opus, adds no caps (AGENTS.md
#2 / #12 — the ``tail_risk`` / ``stress_scenarios`` precedent). Pure, no
I/O, never raises. Applies on next paper-trader restart (the documented
pattern for every recent feature).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

_TRADING_DAYS = 252
_SQRT_TD = math.sqrt(_TRADING_DAYS)


def _z(v: float | None, ndigits: int = 2) -> float | None:
    """Round, folding -0.0 → 0.0 so the JSON never carries a signed zero."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _f(v, default: float = 0.0) -> float:
    """Coerce to float, never raise — a garbage row contributes the
    default, never sinks the builder (the ``_position_betas`` precedent)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def build_recovery(
    drawdown_result: dict,
    tail_risk_result: dict | None,
    initial_equity: float,
    now: datetime | None = None,
) -> dict:
    """Pure, no I/O, never raises.

    ``drawdown_result`` is ``analytics.drawdown.compute_drawdown(...)``'s
    dict (SSOT for current/peak value + per-position P&L). ``tail_risk_result``
    is ``analytics.tail_risk.build_tail_risk(eq)``'s dict (SSOT for realized
    vol) or ``None``. ``initial_equity`` is ``store.INITIAL_CASH`` threaded
    by the caller — never a literal 1000 (the ``benchmark``/``drawdown``
    invariant #12 precedent).
    """
    now = now or datetime.now(timezone.utc)
    init = _f(initial_equity, 1000.0)

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "initial_equity": _z(init),
        "current_value": None,
        "peak_value": None,
        "to_initial_pct": None,
        "to_initial_usd": None,
        "to_peak_pct": None,
        "to_peak_usd": None,
        "daily_vol_pct": None,
        "to_initial_sigma_days": None,
        "to_peak_sigma_days": None,
        "dispersion_state": None,
        "positions": [],
        "prompt_block": None,
    }

    if not isinstance(drawdown_result, dict):
        base["state"] = "NO_DATA"
        base["headline"] = "Recovery: no priced book to measure yet."
        return base

    current = drawdown_result.get("current_value")
    peak = drawdown_result.get("peak_value")
    try:
        current = float(current)
    except (TypeError, ValueError):
        current = None
    if current is None or current <= 0:
        base["state"] = "NO_DATA"
        base["headline"] = "Recovery: no priced book to measure yet."
        return base
    try:
        peak = float(peak)
    except (TypeError, ValueError):
        peak = current

    base["current_value"] = _z(current)
    base["peak_value"] = _z(peak)

    # ── Per-position breakeven (composed verbatim from drawdown contributors) ──
    # A loser must rally (avg_cost/current_price − 1) to return its mark to
    # the entry that zeroes its unrealized P&L. The % is a *price ratio* and
    # is multiplier-invariant — an option lot needs NO ×100 here (the ×100 is
    # already baked into unrealized_pl, which is what dollars_to_recover reads
    # directly — never re-derived from avg_cost×qty, the hold_discipline
    # invariant-#10 precedent). A winner is already above its own breakeven →
    # 0.0 needed, never negative noise.
    pos_rows = []
    for c in drawdown_result.get("contributors") or []:
        if not isinstance(c, dict):
            continue
        upl = _f(c.get("unrealized_pl"), 0.0)
        avg_cost = _f(c.get("avg_cost"), 0.0)
        cur_px = _f(c.get("current_price"), 0.0)
        if upl >= 0 or avg_cost <= 0 or cur_px <= 0:
            be_pct = 0.0
        else:
            be_pct = (avg_cost / cur_px - 1.0) * 100.0
        pos_rows.append({
            "ticker": c.get("ticker"),
            "type": c.get("type"),
            "breakeven_pct": _z(be_pct),
            "dollars_to_recover": _z(max(0.0, -upl)),
            "unrealized_pl": _z(upl),
        })
    # Heaviest $-to-recover first — the name that must do the work leads.
    pos_rows.sort(key=lambda r: -(r["dollars_to_recover"] or 0.0))
    base["positions"] = pos_rows

    # ── Book targets ──
    if current >= init:
        base["state"] = "ABOVE_WATER"
        base["to_initial_pct"] = 0.0
        base["to_initial_usd"] = 0.0
        # Still report the peak gap (a book above the start can be below an
        # intra-history high) but the *recovery-to-even* identity is met.
        if peak > current:
            base["to_peak_pct"] = _z((peak / current - 1.0) * 100.0)
            base["to_peak_usd"] = _z(peak - current)
        else:
            base["to_peak_pct"] = 0.0
            base["to_peak_usd"] = 0.0
        base["headline"] = (
            f"Recovery: book ${current:,.2f} is at/above the "
            f"${init:,.0f} start — nothing to recover to even."
        )
        return base

    base["state"] = "UNDERWATER"
    to_init_usd = init - current
    to_init_pct = (init / current - 1.0) * 100.0
    base["to_initial_usd"] = _z(to_init_usd)
    base["to_initial_pct"] = _z(to_init_pct)
    if peak > current:
        base["to_peak_usd"] = _z(peak - current)
        base["to_peak_pct"] = _z((peak / current - 1.0) * 100.0)
    else:
        base["to_peak_usd"] = 0.0
        base["to_peak_pct"] = 0.0

    # ── Daily-dispersion scale (SSOT: tail_risk realized vol, gated on OK) ──
    tr_state = (tail_risk_result or {}).get("state") if isinstance(
        tail_risk_result, dict) else None
    ann_vol = (tail_risk_result or {}).get("annualized_vol_pct") if isinstance(
        tail_risk_result, dict) else None
    dispersion_note: str
    if tr_state == "OK" and ann_vol not in (None, 0):
        daily_sigma = _f(ann_vol, 0.0) / _SQRT_TD
        if daily_sigma > 0:
            base["daily_vol_pct"] = _z(daily_sigma)
            base["dispersion_state"] = "OK"
            base["to_initial_sigma_days"] = _z(to_init_pct / daily_sigma, 1)
            if base["to_peak_pct"]:
                base["to_peak_sigma_days"] = _z(
                    base["to_peak_pct"] / daily_sigma, 1)
            dispersion_note = (
                f"≈{base['to_initial_sigma_days']:.1f}× this book's own daily "
                f"σ ({base['daily_vol_pct']:.2f}%) — a dispersion scale, "
                f"NOT a time forecast (a random walk's expected time to even "
                f"is undefined)."
            )
        else:
            base["dispersion_state"] = "WITHHELD"
            dispersion_note = (
                "Dispersion scale withheld — realized daily vol is zero."
            )
    else:
        base["dispersion_state"] = "WITHHELD"
        dispersion_note = (
            "Dispersion scale withheld — realized-vol verdict is "
            f"{tr_state or 'unavailable'} (a young book's vol is noise; "
            "the tail_risk/correlation sample-size-honesty precedent)."
        )

    lead = pos_rows[0] if pos_rows else None
    lead_seg = ""
    if lead and (lead["dollars_to_recover"] or 0.0) > 0:
        lead_seg = (
            f" {lead['ticker']} carries the most "
            f"(${lead['dollars_to_recover']:+.2f}, +"
            f"{lead['breakeven_pct']:.1f}% to its own breakeven)."
        )

    base["headline"] = (
        f"Recovery: underwater ${-to_init_usd:+.2f} "
        f"({-to_init_pct:+.2f}%) vs the ${init:,.0f} start — needs "
        f"+{to_init_pct:.2f}% (${to_init_usd:,.2f}) to even, "
        f"+{base['to_peak_pct']:.2f}% to the ${peak:,.2f} peak. "
        f"{dispersion_note}{lead_seg}"
    )
    base["prompt_block"] = (
        "RECOVERY MATH (advisory, observational — never a directive; you "
        "retain full autonomy):\n"
        f"  {base['headline']}\n"
        "  The σ figure is a dispersion scale on THIS book's realized "
        "volatility, not an estimated time-to-recover."
    )
    return base
