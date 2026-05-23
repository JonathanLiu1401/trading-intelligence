"""Per-name concentration-cap rebalance recommender.

``/api/risk`` flags concentration_top1 by *weight* and severity; the new
``/api/position-blowup`` quantifies the *single-name shock damage*. Neither
of them answers the **mechanical** question: *if I set a per-name cap of K%,
exactly how many shares of each over-cap name do I need to sell?*

``build_concentration_cap`` does exactly that. Given:

* the current portfolio (priced positions + total_value),
* a configurable per-name cap (default 25.0%; route accepts ``?cap_pct=N``),

it computes, for each over-cap name, the exact ``shares_to_trim`` and
``cash_freed_usd`` required to land at the cap. The result also carries the
``baseline`` and ``projected`` top1/top3 so the operator sees whether one
trim cycle is enough or another name is queued to climb past the cap (passive
math: freed cash isn't redistributed; the trim only reweights the trimmed
names, the rest hold weight by remaining at the same dollar value while
``total_value`` is preserved).

Pure / never-raises / no I/O. Advisory only — never gates Opus, adds no caps
(AGENTS.md #2/#12). Complementary to ``trim_simulator`` (the per-position
scorer-EV ladder): one is "what trim sizes would the *scorer* drive" and the
other is "what trim sizes does a *mechanical* per-name cap drive".
"""
from __future__ import annotations

from datetime import datetime, timezone

#: Default per-name cap in percent of book. Mirrors the
#: ``/api/risk`` ``concentration_severity`` LOW/MEDIUM band thresholds the
#: rest of the desk uses to flag concentration. The route accepts an override.
DEFAULT_CAP_PCT = 25.0
#: Clamp band. Below 1% is operational nonsense (every name is over-cap by
#: definition); above 100% is a guaranteed AT_CAP no-op.
MIN_CAP_PCT = 1.0
MAX_CAP_PCT = 100.0


def _z(v, ndigits: int = 2):
    """Round, folding ``-0.0 → 0.0`` (the ``position_blowup._z`` precedent)."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _position_value(p: dict) -> float:
    """Best-effort market value — prefers ``market_value`` (written by
    ``strategy._mark_to_market``); falls back to ``current_price × qty ×
    mult`` (options ×100). Never raises — the
    ``position_blowup._position_value`` precedent / contract."""
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


def build_concentration_cap(
    positions: list[dict] | None,
    total_value: float | None,
    cap_pct: float = DEFAULT_CAP_PCT,
    now: datetime | None = None,
) -> dict:
    """Pure: per-name trim qty to bring every position under ``cap_pct``.

    Returns a JSON-ready dict with:

    * ``state``         — NO_DATA / AT_CAP / OVER_CAP
    * ``cap_pct``       — the cap that was applied (post-clamp)
    * ``n_over_cap``    — count of positions above the cap
    * ``total_cash_freed_usd``  — sum of cash_freed across over-cap rows
    * ``over_cap_positions``    — worst-first list (current + target +
      shares_to_trim + cash_freed + weight_pct_reduction)
    * ``baseline`` / ``projected`` — {top1_pct, top1_ticker, top3_pct} pre
      and post the proposed trim (``total_value`` is preserved — equity → cash)

    Inputs handled defensively: garbage rows skipped, zero/negative
    total_value → NO_DATA, ``cap_pct`` clamped to ``[MIN_CAP_PCT,
    MAX_CAP_PCT]``.
    """
    now = now or datetime.now(timezone.utc)
    try:
        tv = float(total_value or 0.0)
    except (TypeError, ValueError):
        tv = 0.0
    try:
        cap = float(cap_pct)
    except (TypeError, ValueError):
        cap = DEFAULT_CAP_PCT
    cap = max(MIN_CAP_PCT, min(MAX_CAP_PCT, cap))

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "cap_pct": _z(cap),
        "total_value_usd": _z(tv),
        "n_positions": 0,
        "n_over_cap": 0,
        "total_cash_freed_usd": 0.0,
        "over_cap_positions": [],
        "baseline": {"top1_pct": 0.0, "top1_ticker": None, "top3_pct": 0.0},
        "projected": {"top1_pct": 0.0, "top1_ticker": None, "top3_pct": 0.0},
    }

    rows_in = list(positions or [])
    if not rows_in or tv <= 0:
        base["state"] = "NO_DATA"
        base["headline"] = "Concentration cap: no priced book to rebalance yet."
        return base

    # Collect priced rows and compute initial weights.
    priced: list[dict] = []
    for p in rows_in:
        val = _position_value(p)
        if val <= 0:
            continue
        try:
            qty = float(p.get("qty") or 0)
        except (TypeError, ValueError):
            qty = 0.0
        if qty <= 0:
            continue
        priced.append({
            "ticker": (p.get("ticker") or "").upper() or None,
            "type": (p.get("type") or "stock").lower(),
            "qty": qty,
            "value": val,
            "weight_pct": val / tv * 100.0,
        })

    if not priced:
        base["state"] = "NO_DATA"
        base["headline"] = "Concentration cap: no priceable open positions."
        return base

    base["n_positions"] = len(priced)

    # Baseline top1/top3 — heaviest-first by value.
    priced_sorted = sorted(priced, key=lambda r: -r["value"])
    base["baseline"]["top1_ticker"] = priced_sorted[0]["ticker"]
    base["baseline"]["top1_pct"] = _z(priced_sorted[0]["weight_pct"])
    base["baseline"]["top3_pct"] = _z(
        sum(r["weight_pct"] for r in priced_sorted[:3])
    )

    cap_value = cap / 100.0 * tv
    over_cap: list[dict] = []
    total_freed = 0.0
    # Start projection from current values; trims reduce the over-cap entries
    # to cap_value. Remaining names hold their dollar weight (cash replaces
    # the trimmed equity, preserving total_value).
    projected_values: dict[str | None, float] = {
        r["ticker"]: r["value"] for r in priced
    }
    for r in priced:
        if r["weight_pct"] <= cap:
            continue
        target_val = cap_value
        cash_freed = r["value"] - target_val
        shares_to_trim = r["qty"] * (cash_freed / r["value"])
        over_cap.append({
            "ticker": r["ticker"],
            "type": r["type"],
            "current_weight_pct": _z(r["weight_pct"]),
            "current_market_value_usd": _z(r["value"]),
            "current_qty": _z(r["qty"], 4),
            "target_weight_pct": _z(cap),
            "target_market_value_usd": _z(target_val),
            "shares_to_trim": _z(shares_to_trim, 4),
            "cash_freed_usd": _z(cash_freed),
            "weight_pct_reduction": _z(r["weight_pct"] - cap),
        })
        total_freed += cash_freed
        projected_values[r["ticker"]] = target_val

    # Projected top1/top3 — total_value preserved (trimmed equity → cash; cash
    # isn't itself a position with a weight here, so the proportions remain
    # against the same denominator the operator sees in /api/risk).
    proj_sorted = sorted(projected_values.items(), key=lambda kv: -kv[1])
    if proj_sorted:
        base["projected"]["top1_ticker"] = proj_sorted[0][0]
        base["projected"]["top1_pct"] = _z(proj_sorted[0][1] / tv * 100.0)
        base["projected"]["top3_pct"] = _z(
            sum(v for _, v in proj_sorted[:3]) / tv * 100.0
        )

    base["n_over_cap"] = len(over_cap)
    base["total_cash_freed_usd"] = _z(total_freed)

    # Worst-first by weight_pct_reduction (deepest cut first).
    over_cap.sort(key=lambda r: -(r["weight_pct_reduction"] or 0.0))
    base["over_cap_positions"] = over_cap

    if not over_cap:
        base["state"] = "AT_CAP"
        base["headline"] = (
            f"Concentration cap ({cap:.1f}%): {base['n_positions']} "
            f"position{'' if base['n_positions'] == 1 else 's'} all within "
            f"cap; largest {base['baseline']['top1_ticker']} "
            f"{base['baseline']['top1_pct']:.1f}%."
        )
    else:
        worst = over_cap[0]
        base["state"] = "OVER_CAP"
        base["headline"] = (
            f"Concentration cap ({cap:.1f}%): {len(over_cap)} over-cap; "
            f"largest {worst['ticker']} {worst['current_weight_pct']:.1f}% — "
            f"trim {worst['shares_to_trim']:g} shares to free "
            f"${worst['cash_freed_usd']:.2f}; book top1 "
            f"{base['baseline']['top1_pct']:.1f}%→"
            f"{base['projected']['top1_pct']:.1f}%."
        )
    return base


def _cli_main() -> int:
    """Render the live book's concentration-cap recommendation.

    Accepts ``--cap=N`` to override the default 25 % cap (any value outside
    [1, 100] is silently clamped, same as the route)."""
    import json
    import sys
    from ..store import get_store
    from ..strategy import portfolio_snapshot_readonly
    cap = DEFAULT_CAP_PCT
    for arg in sys.argv[1:]:
        if arg.startswith("--cap="):
            try:
                cap = float(arg.split("=", 1)[1])
            except ValueError:
                pass
    store = get_store()
    snap = portfolio_snapshot_readonly(store)
    res = build_concentration_cap(
        snap.get("positions"), snap.get("total_value"), cap,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
