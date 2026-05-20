"""Per-position single-name blow-up scenarios.

``/api/stress-scenarios`` already shocks the **largest** position alone at
-10% (``single_name_gap``). It says nothing about the second-largest, third,
etc. — on the live concentrated book (NVDA 66.6 %, TQQQ 29.4 %, 95.9 %
top-2) a discretionary trader looking at the dashboard wants the FULL per-
position blow-up ladder, not just one row:

  *If NVDA gaps -10 % on earnings, I lose $X. If it gaps -25 %, $Y. If
  it goes to zero, $Z. Same for TQQQ. Sorted by worst.*

``build_position_blowup`` answers exactly that and nothing else. Pure
weight×shock arithmetic over the currently-marked book — **no beta, no
market correlation**, because the question is *idiosyncratic*: what does
THIS name alone losing X % cost me? The position's whole market value
(``current_price × qty × mult``) is the lever; the only multiplier is the
shock magnitude.

Complementary, NOT redundant, with the existing builders:

* ``stress_scenarios.single_name_gap`` — the **top** name only at -10 %.
* ``stress_scenarios.scenarios`` — **market-wide** SPY shocks with
  beta amplification.
* ``stress_scenarios.sector_shock`` — the heaviest sector cluster.
* ``earnings_shock`` — only **upcoming earnings** events (calendar-gated).

This builder is the **per-position ladder** missing from all of those: every
held name shocked individually at four magnitudes (-10/-25/-50/-100 %),
sorted by max damage so the operator sees the biggest single-name risks
first. Crucial when the book is concentrated and a single-name surprise
(downgrade, lawsuit, accounting issue) is the dominant tail risk that the
SPY-shock and earnings-σ models don't capture.

Verdict ladder (concentration-aware, mirrors the desk's other builders):

* ``NO_DATA``     — no open positions / zero total value.
* ``DIFFUSE``     — no single position going to zero would lose >20 % of
  book (the book is genuinely diversified — a single-name surprise is
  capped at "moderate dent").
* ``MODERATE``    — at least one position going to zero would lose 20-40 %
  of book (single-name exposure is meaningful — a bad print on the wrong
  name takes a real bite).
* ``CONCENTRATED``— at least one position going to zero would lose >40 %
  of book (single-name surprise is the dominant tail risk; the desk is
  effectively short one company's optionality).

Diagnostic / advisory only: never gates Opus, adds no caps (AGENTS.md
invariants #2 / #12 — the ``stress_scenarios`` / ``risk_mirror`` precedent).
Pure, no I/O, never raises. Options multiplied ×100 (the
``stress_scenarios._position_betas`` precedent) so option positions
contribute their true notional, not the per-share premium.
"""
from __future__ import annotations

from datetime import datetime, timezone

#: Single-name shock magnitudes (% drop on the position alone, no beta).
#: -10/-25/-50 cover normal-to-severe single-name surprises; -100 is the
#: company-fails-entirely floor (Enron, FTX) — vanishingly rare for the
#: large-caps the desk trades but the absolute worst-case bound, and
#: useful as the size-of-position-relative-to-book lens.
SHOCK_MAGNITUDES_PCT = (-10.0, -25.0, -50.0, -100.0)

#: Verdict thresholds — single-position-to-zero loss as % of book.
CONCENTRATED_THRESHOLD_PCT = 40.0
MODERATE_THRESHOLD_PCT = 20.0


def _z(v: float | None, ndigits: int = 2) -> float | None:
    """Round, folding -0.0 → 0.0 so the JSON never carries a signed zero
    (the ``stress_scenarios._z`` precedent)."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _position_value(p: dict) -> float:
    """Best-effort market value of a position row. Prefers the enriched
    ``market_value`` written by ``strategy._mark_to_market``; falls back to
    ``current_price × qty × mult`` (options ×100, the
    ``stress_scenarios._position_betas`` precedent); never raises.

    A garbage row contributes 0.0 (the ``_safe`` contract used across
    behavioural builders), never sinks the whole result."""
    mv = p.get("market_value")
    if mv is not None:
        try:
            return float(mv)
        except (TypeError, ValueError):
            pass
    try:
        ptype = p.get("type") or "stock"
        mult = 100 if ptype in ("call", "put") else 1
        price = p.get("current_price") or p.get("avg_cost") or 0.0
        qty = float(p.get("qty") or 0)
        return float(price) * qty * mult
    except (TypeError, ValueError):
        return 0.0


def build_position_blowup(
    positions: list[dict] | None,
    total_value: float | None,
    now: datetime | None = None,
) -> dict:
    """Per-position single-name shock ladder. Pure, no I/O, never raises.

    Each row carries the position's market_value plus a ``shocks`` list
    with (label, pnl_usd, pnl_pct) entries — one per ``SHOCK_MAGNITUDES_PCT``
    magnitude — and a ``max_loss_usd`` summary equal to the -100 % entry.
    Rows are sorted most-damaging-first by ``max_loss_usd`` so the
    operator's first read is the biggest single-name risk."""
    now = now or datetime.now(timezone.utc)
    try:
        tv = float(total_value or 0.0)
    except (TypeError, ValueError):
        tv = 0.0

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_positions": 0,
        "total_value_usd": _z(tv),
        "shock_magnitudes_pct": list(SHOCK_MAGNITUDES_PCT),
        "positions": [],
    }

    rows_in = list(positions or [])
    if not rows_in or tv <= 0:
        base["state"] = "NO_DATA"
        base["headline"] = (
            "Position blow-up: no priced book to shock yet."
        )
        return base

    out_rows: list[dict] = []
    for p in rows_in:
        val = _position_value(p)
        if val <= 0:
            # Skip a zero-value row (closed mid-cycle, garbage qty, etc.) —
            # silently, never raise. The headline counts only real lots.
            continue
        ticker = p.get("ticker")
        ptype = (p.get("type") or "stock").lower()
        shocks: list[dict] = []
        for mag in SHOCK_MAGNITUDES_PCT:
            pnl = mag / 100.0 * val
            shocks.append({
                "shock_pct": _z(mag),
                "pnl_usd": _z(pnl),
                "pnl_pct_of_book": _z(pnl / tv * 100.0),
            })
        # max_loss is the -100 % entry — guaranteed to exist since the
        # magnitudes tuple includes it; pulling it explicitly so callers
        # never have to re-find it by string match.
        max_loss = -val
        out_rows.append({
            "ticker": ticker,
            "type": ptype,
            "market_value_usd": _z(val),
            "weight_pct": _z(val / tv * 100.0),
            "max_loss_usd": _z(max_loss),
            "max_loss_pct_of_book": _z(max_loss / tv * 100.0),
            "shocks": shocks,
        })

    if not out_rows:
        # All rows skipped — same NO_DATA exit, never a ZeroDivisionError
        # / IndexError downstream.
        base["state"] = "NO_DATA"
        base["headline"] = (
            "Position blow-up: no priceable open positions."
        )
        return base

    # Most damaging first — by max_loss_usd ASC (most negative first).
    out_rows.sort(key=lambda r: (r.get("max_loss_usd") or 0.0))
    base["positions"] = out_rows
    base["n_positions"] = len(out_rows)

    worst = out_rows[0]
    worst_pct = abs(worst.get("max_loss_pct_of_book") or 0.0)
    if worst_pct >= CONCENTRATED_THRESHOLD_PCT:
        verdict = "CONCENTRATED"
    elif worst_pct >= MODERATE_THRESHOLD_PCT:
        verdict = "MODERATE"
    else:
        verdict = "DIFFUSE"
    base["state"] = verdict

    # Headline showcases the worst name's three most-decision-relevant
    # rungs (-25 / -50 / -100). The -10 % rung is left for the JSON; -25
    # is the realistic single-name surprise floor, -50 is severe, -100 is
    # the size-of-position lens.
    sh_by_mag = {s["shock_pct"]: s for s in worst["shocks"]}
    s25 = sh_by_mag.get(_z(-25.0)) or sh_by_mag.get(-25.0) or {}
    s50 = sh_by_mag.get(_z(-50.0)) or sh_by_mag.get(-50.0) or {}
    s100 = sh_by_mag.get(_z(-100.0)) or sh_by_mag.get(-100.0) or {}
    base["headline"] = (
        f"Position blow-up ({verdict}): {worst['ticker']} "
        f"({worst['weight_pct']:.1f}% of book) "
        f"−25% costs ${s25.get('pnl_usd', 0.0):+.2f} "
        f"({s25.get('pnl_pct_of_book', 0.0):+.2f}% of book), "
        f"−50% costs ${s50.get('pnl_usd', 0.0):+.2f} "
        f"({s50.get('pnl_pct_of_book', 0.0):+.2f}%), "
        f"to zero costs ${s100.get('pnl_usd', 0.0):+.2f} "
        f"({s100.get('pnl_pct_of_book', 0.0):+.2f}%). "
        f"{base['n_positions']} position"
        f"{'' if base['n_positions'] == 1 else 's'} shocked individually."
    )
    return base


def _cli_main() -> int:
    """Render the live book's position-blowup table. Read-only — opens
    the live store via the same path the other analytics CLIs use
    (``-m paper_trader.analytics.<name>`` from the repo root)."""
    import json
    from ..store import get_store
    from ..strategy import portfolio_snapshot_readonly

    store = get_store()
    snap = portfolio_snapshot_readonly(store)
    res = build_position_blowup(
        snap.get("positions"), snap.get("total_value"),
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
