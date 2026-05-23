"""Per-position upside profit ladder + trim-yield schedule.

The symmetric complement to ``position_blowup``: that builder shocks
each held name DOWN at -10/-25/-50/-100 % and tells the operator
"single-name surprise costs $X". This builder shocks the same names UP
at +5/+10/+25/+50/+100 % and tells the operator "single-name rally is
worth $Y at the mark — and trimming Z % at that rung locks $L in
realized P&L while leaving $U in unrealized upside".

Existing surfaces and what they leave open:

* ``position_blowup``  — downside, idiosyncratic; never quantifies upside.
* ``recovery`` / ``drawdown`` — book-level path back to even; flattens
  per-position upside into one $-to-peak number.
* ``cost_basis_ladder`` — per-LOT FIFO P&L vs the *current* mark; says
  nothing about future shocks or trim-yield arithmetic.
* ``trim_simulator`` — trims at the *current* mark; never projects "if
  it runs another +25 %, here's what trimming HALF at THAT level locks
  in".
* ``earnings_war_room`` — option-implied +σ/-σ envelope around a dated
  catalyst; calendar-gated, never the per-position fixed-magnitude ladder.

The headline trader question this answers and nothing else does:

  *If this position rallies another +25 %, what does it pay me? At +50
  %? If I trim half at +25 %, how much do I lock in and how much do I
  leave on the table for the +50 % rung?*

Each row carries the position's market_value plus a ``shocks`` list — one
per ``UPSIDE_SHOCK_PCT`` magnitude — and a ``trim_schedule`` block with
the realized-vs-unrealized split if the operator trims 25 / 50 / 100 %
of the position AT that rung. Rows are sorted most-upside-first by
``max_gain_usd`` (the +100 % entry) so the operator's first read is the
biggest dollar-upside name.

Verdict ladder (mirrors ``position_blowup`` / ``cost_basis_ladder``
shape, valued for upside):

* ``NO_DATA``        — no priced book to shock.
* ``RECOVERY_BOOK``  — every position underwater at the current mark.
  The +N % rungs above are the *path back to even and then above* — the
  upside math is the same shape but the operator's first move is
  recovery, not profit-taking.
* ``MIXED_BOOK``     — at least one green, at least one red.
* ``IN_PROFIT``      — every position green at the mark; no rung shows
  a single position printing > 10 % of book in additional gain.
* ``BIG_WINNERS``    — at least one position whose +25 % rung alone
  would add > 10 % of book in additional gain. Trim-schedule attention
  warranted.

Diagnostic / advisory only: never gates Opus, adds no caps (AGENTS.md
invariants #2 / #12 — the ``position_blowup`` / ``stress_scenarios``
precedent). Pure, no I/O, never raises. Options multiplied ×100 (the
``stress_scenarios._position_betas`` / ``position_blowup`` precedent)
so option positions contribute their true notional, not the per-share
premium.
"""
from __future__ import annotations

from datetime import datetime, timezone

#: Single-name UPSIDE shock magnitudes (% rally on the position alone,
#: no beta). +5/+10 cover normal continuations; +25 is a meaningful
#: rip; +50 a parabolic move; +100 a true multi-bagger (rare for
#: large-caps over short windows but actionable for leveraged ETFs and
#: catalyst names — and useful as a "if I'm right, what's the prize"
#: lens).
UPSIDE_SHOCK_PCT = (5.0, 10.0, 25.0, 50.0, 100.0)

#: Per-rung trim fractions whose realized/unrealized split we surface.
#: 25 / 50 / 100 are the operationally meaningful rungs: trim a
#: quarter (leave most upside), half (classic disposition lock), or
#: full close (cash out).
TRIM_FRACTIONS = (0.25, 0.50, 1.00)

#: Aggregate-verdict thresholds — single-position +25 % rung gain as
#: % of book.
BIG_WINNER_AT_25PCT_THRESHOLD = 10.0


def _z(v: float | None, ndigits: int = 2) -> float | None:
    """Round, folding -0.0 → 0.0 so the JSON never carries a signed zero
    (the ``position_blowup._z`` / ``stress_scenarios._z`` precedent)."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _num(v) -> float | None:
    """Permissive numeric coerce — None/blank/non-numeric → None.
    bool is rejected (the ``cost_basis_ladder._num`` precedent)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except (TypeError, ValueError):
            return None
    return None


def _position_value_and_qty(p: dict) -> tuple[float, float, float]:
    """Return (market_value_usd, qty, raw_price_per_unit) for a position.

    ``raw_price_per_unit`` is the per-share price for stocks and the
    per-contract premium for options — NOT pre-multiplied by the ×100
    options multiplier. The multiplier is applied once downstream
    (``_trim_schedule`` does ``price * qty * mult``); pre-multiplying
    here too would double-count it (the bug ``test_options_multiplier_100x``
    locks against).

    Garbage row → (0.0, 0.0, 0.0); never raises."""
    ptype = (p.get("type") or "stock").lower()
    mult = 100.0 if ptype in ("call", "put") else 1.0
    qty = _num(p.get("qty")) or 0.0
    price = _num(p.get("current_price")) or _num(p.get("avg_cost")) or 0.0
    mv = _num(p.get("market_value"))
    if mv is None:
        mv = price * qty * mult
    return float(mv), float(qty), float(price)


def _position_cost_basis(p: dict, qty: float, raw_price: float) -> float:
    """Total cost-basis dollars at entry (avg_cost × qty × multiplier).
    Falls back to current market value if avg_cost is missing — that
    collapses unrealized_pl_at_shock to gain_above_current; safer than
    raising on a corrupt row.

    ``raw_price`` is the per-share / per-contract-premium (NOT
    pre-multiplied by ×100 for options); the multiplier is applied
    inside this function on both branches."""
    ptype = (p.get("type") or "stock").lower()
    mult = 100.0 if ptype in ("call", "put") else 1.0
    avg = _num(p.get("avg_cost"))
    if avg is None or avg <= 0:
        # No reconstructable cost basis — degrade to "treat current mark
        # as basis" so trim math becomes "gain above CURRENT price",
        # never NaN / negative.
        return raw_price * qty * mult
    return float(avg) * float(qty) * mult


def _trim_schedule(
    pre_qty: float,
    cost_per_unit: float,
    shocked_price_per_unit: float,
    mult: float,
) -> list[dict]:
    """For each TRIM_FRACTIONS rung, compute the cash realized and the
    realized P&L locked in at the shocked price. ``cost_per_unit`` is
    avg_cost (per-share for stock, per-contract premium for options);
    ``shocked_price_per_unit`` likewise. Multiplier handles options.

    Empty for zero-qty positions; never raises."""
    if pre_qty <= 0:
        return []
    out: list[dict] = []
    for frac in TRIM_FRACTIONS:
        trim_qty = pre_qty * frac
        cash_freed = shocked_price_per_unit * trim_qty * mult
        cost_freed = cost_per_unit * trim_qty * mult
        realized = cash_freed - cost_freed
        remaining_qty = pre_qty - trim_qty
        # Remaining unrealized at the shocked mark, accounted on the
        # share-count basis still held.
        remaining_value = shocked_price_per_unit * remaining_qty * mult
        remaining_unrealized = (
            (shocked_price_per_unit - cost_per_unit)
            * remaining_qty
            * mult
        )
        out.append({
            "trim_pct": _z(frac * 100.0, 1),
            "trim_qty": _z(trim_qty, 4),
            "cash_freed_usd": _z(cash_freed),
            "realized_pl_usd": _z(realized),
            "remaining_qty": _z(remaining_qty, 4),
            "remaining_value_usd": _z(remaining_value),
            "remaining_unrealized_pl_usd": _z(remaining_unrealized),
        })
    return out


def build_profit_ladder(
    positions: list[dict] | None,
    total_value: float | None,
    now: datetime | None = None,
) -> dict:
    """Per-position upside shock ladder with trim-yield schedule.

    Pure, no I/O, never raises. Mirrors ``build_position_blowup``'s
    shape so callers can render both side-by-side with identical
    glue code."""
    now = now or datetime.now(timezone.utc)
    try:
        tv = float(total_value or 0.0)
    except (TypeError, ValueError):
        tv = 0.0

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_positions": 0,
        "total_value_usd": _z(tv),
        "upside_shock_pct": list(UPSIDE_SHOCK_PCT),
        "trim_fractions_pct": [_z(f * 100.0, 1) for f in TRIM_FRACTIONS],
        "positions": [],
    }

    rows_in = list(positions or [])
    if not rows_in or tv <= 0:
        base["state"] = "NO_DATA"
        base["headline"] = (
            "Profit ladder: no priced book to project upside on yet."
        )
        return base

    out_rows: list[dict] = []
    n_green = 0
    n_red = 0
    for p in rows_in:
        mv, qty, price_per_unit = _position_value_and_qty(p)
        if mv <= 0 or qty <= 0 or price_per_unit <= 0:
            # Skip a zero-value / unpriced row (closed mid-cycle,
            # garbage qty) — silently, never raise. The headline
            # counts only real lots.
            continue
        ticker = p.get("ticker")
        ptype = (p.get("type") or "stock").lower()
        mult = 100.0 if ptype in ("call", "put") else 1.0
        cost_basis = _position_cost_basis(p, qty, price_per_unit)
        # Per-unit avg cost — back out from total basis ÷ qty ÷ mult so
        # the trim_schedule arithmetic matches store conventions.
        cost_per_unit = (
            (cost_basis / qty / mult) if (qty > 0 and mult > 0) else 0.0
        )
        unrealized_pl_now = mv - cost_basis
        if unrealized_pl_now > 0:
            n_green += 1
        elif unrealized_pl_now < 0:
            n_red += 1

        shocks: list[dict] = []
        for mag in UPSIDE_SHOCK_PCT:
            shocked_price = price_per_unit * (1.0 + mag / 100.0)
            shocked_mv = mv * (1.0 + mag / 100.0)
            gain_usd = shocked_mv - mv
            shocked_unrealized = shocked_mv - cost_basis
            shocks.append({
                "shock_pct": _z(mag),
                "shocked_price_per_unit": _z(shocked_price, 4),
                "shocked_market_value_usd": _z(shocked_mv),
                "gain_above_current_usd": _z(gain_usd),
                "gain_above_current_pct_of_book": _z(gain_usd / tv * 100.0),
                "unrealized_pl_at_shock_usd": _z(shocked_unrealized),
                "trim_schedule": _trim_schedule(
                    qty,
                    cost_per_unit,
                    shocked_price,
                    mult,
                ),
            })
        # max_gain is the +100 % entry — guaranteed to exist.
        max_gain = mv  # +100% means doubling — gain == current mv
        out_rows.append({
            "ticker": ticker,
            "type": ptype,
            "qty": _z(qty, 4),
            "current_price_per_unit": _z(price_per_unit, 4),
            "cost_per_unit": _z(cost_per_unit, 4),
            "market_value_usd": _z(mv),
            "cost_basis_usd": _z(cost_basis),
            "unrealized_pl_usd_now": _z(unrealized_pl_now),
            "weight_pct": _z(mv / tv * 100.0),
            "max_gain_usd": _z(max_gain),
            "max_gain_pct_of_book": _z(max_gain / tv * 100.0),
            "shocks": shocks,
        })

    if not out_rows:
        base["state"] = "NO_DATA"
        base["headline"] = (
            "Profit ladder: no priceable open positions."
        )
        return base

    # Most upside first — by max_gain_usd DESC.
    out_rows.sort(key=lambda r: -(r.get("max_gain_usd") or 0.0))
    base["positions"] = out_rows
    base["n_positions"] = len(out_rows)

    # Aggregate verdict.
    big_winner = False
    for r in out_rows:
        sh_by_mag = {s["shock_pct"]: s for s in r["shocks"]}
        s25 = sh_by_mag.get(_z(25.0)) or sh_by_mag.get(25.0) or {}
        g25_pct = s25.get("gain_above_current_pct_of_book") or 0.0
        if g25_pct >= BIG_WINNER_AT_25PCT_THRESHOLD:
            big_winner = True
            break

    if n_green == 0 and n_red > 0:
        verdict = "RECOVERY_BOOK"
    elif n_green > 0 and n_red > 0:
        verdict = "MIXED_BOOK"
    elif big_winner:
        verdict = "BIG_WINNERS"
    elif n_green > 0:
        verdict = "IN_PROFIT"
    else:
        # n_green == 0 and n_red == 0 — every position is exactly flat
        # (rare; possible right after entry at the mark). Treat as
        # IN_PROFIT-eq (no recovery path needed) — flat is not red.
        verdict = "IN_PROFIT"
    base["state"] = verdict

    # Headline showcases the top-upside name's three most decision-
    # relevant rungs (+10 / +25 / +50). The +5 rung is left for the
    # JSON; +100 is the size-of-position lens.
    top = out_rows[0]
    sh = {s["shock_pct"]: s for s in top["shocks"]}
    s10 = sh.get(_z(10.0)) or sh.get(10.0) or {}
    s25 = sh.get(_z(25.0)) or sh.get(25.0) or {}
    s50 = sh.get(_z(50.0)) or sh.get(50.0) or {}
    base["headline"] = (
        f"Profit ladder ({verdict}): {top['ticker']} "
        f"({top['weight_pct']:.1f}% of book) "
        f"+10% pays ${s10.get('gain_above_current_usd', 0.0):+.2f} "
        f"({s10.get('gain_above_current_pct_of_book', 0.0):+.2f}% of book), "
        f"+25% pays ${s25.get('gain_above_current_usd', 0.0):+.2f} "
        f"({s25.get('gain_above_current_pct_of_book', 0.0):+.2f}%), "
        f"+50% pays ${s50.get('gain_above_current_usd', 0.0):+.2f} "
        f"({s50.get('gain_above_current_pct_of_book', 0.0):+.2f}%). "
        f"{base['n_positions']} position"
        f"{'' if base['n_positions'] == 1 else 's'} laddered."
    )
    return base


def _cli_main() -> int:
    """Render the live book's profit-ladder table. Read-only — opens
    the live store via the same path the other analytics CLIs use
    (``-m paper_trader.analytics.<name>`` from the repo root)."""
    import json
    from ..store import get_store
    from ..strategy import portfolio_snapshot_readonly

    store = get_store()
    snap = portfolio_snapshot_readonly(store)
    res = build_profit_ladder(
        snap.get("positions"), snap.get("total_value"),
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
