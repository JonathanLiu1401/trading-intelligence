"""Liquidation preview — the natural complement to ``buying_power``.

``buying_power`` answers "what can my FREE CASH fund right now?".
``liquidation_preview`` answers the orthogonal "what would my book look
like RIGHT NOW if I closed every open position?" — the question a PM
asks before a pre-earnings de-risk, a drawdown trim, or a strategy reset.

Concretely, for each open position this computes:
  * the cash that SELLING it at the current mark would generate
    (``qty × current_price × (100 if option else 1)``);
  * the realized P/L that close would lock in
    (``(current_price − avg_cost) × qty × multiplier``);
  * the per-position % return on cost.

Then aggregates: total liquidation cash (current cash + sum of marks),
total realized P/L locked in (== sum of current unrealized_pl on
mark-to-market — single source of truth with ``_mark_to_market`` so
the panel and Opus never disagree on what closing the book means).

Same advisory contract as ``buying_power`` (AGENTS.md #2/#12 — the
``self_review`` / ``risk_mirror`` precedent): **observational only**,
never gates Opus, no caps, no directive. States facts about the
*hypothetical* close and lets the operator/PM act on them.

Pure arithmetic over the already-marked snapshot — NO network, NO
store reads (the risk_mirror hot-path discipline). Never raises — a
garbage cell degrades to a 0.0 contribution (the _safe contract).

Why this is a DISTINCT surface from ``/api/portfolio`` (which already
shows ``total_value`` and ``unrealized_pl``):

  * ``total_value`` aggregates ``cash + open_value`` and is the
    headline equity number — a PM looking at it cannot answer "if I
    closed everything, would my locked-in P/L be positive?" without
    re-deriving from per-position cost basis. This surface lifts that
    derivation onto its own panel with the per-position
    cash-generated / realized-PL breakdown.
  * Stale-mark positions are flagged here so the PM knows when the
    "would lock in" number is unreliable; ``total_value`` swallows the
    stale rows silently at cost.
  * The panel sorts contributors so the operator sees the biggest
    locks first — useful for partial liquidation ("close the top 2,
    keep the rest").
"""
from __future__ import annotations


def _f(x, default: float = 0.0) -> float:
    """Best-effort float coercion. A garbage cell degrades to ``default``,
    never raises (the _safe contract mirrors ``buying_power._f``)."""
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _is_option(p: dict) -> bool:
    return p.get("type") in ("call", "put")


def _multiplier(p: dict) -> int:
    return 100 if _is_option(p) else 1


def _position_close_cash(p: dict) -> float:
    """Cash a SELL at the current mark would generate.

    Prefers the enriched ``market_value`` ``_mark_to_market`` computed
    (which already bakes in the ×100 option multiplier). Falls back to
    ``current_price × qty × multiplier`` only when ``market_value`` is
    absent — so a plain ``open_positions()`` row still values sanely.
    The same precedent as ``buying_power._position_mark_value``."""
    mv = p.get("market_value")
    if isinstance(mv, (int, float)):
        return float(mv)
    mult = _multiplier(p)
    px = _f(p.get("current_price")) or _f(p.get("avg_cost"))
    return px * _f(p.get("qty")) * mult


def _position_realized_if_closed(p: dict) -> float:
    """Realized P/L locked in by a SELL at the current mark.

    Prefers the enriched ``unrealized_pl`` ``_mark_to_market`` already
    computed (single source of truth — the panel can never disagree
    with what the prompt sees). Falls back to
    ``(current_price − avg_cost) × qty × multiplier`` only when
    ``unrealized_pl`` is absent."""
    upl = p.get("unrealized_pl")
    if isinstance(upl, (int, float)):
        return float(upl)
    mult = _multiplier(p)
    qty = _f(p.get("qty"))
    avg = _f(p.get("avg_cost"))
    cur = _f(p.get("current_price"))
    if avg <= 0 or cur <= 0 or qty <= 0:
        return 0.0
    return (cur - avg) * qty * mult


def _position_pl_pct(p: dict) -> float | None:
    """Per-position % return on cost. ``None`` when avg_cost is not a
    usable positive number (so the panel does not emit a misleading
    ``+0.0%`` next to a stale or zero-cost row)."""
    avg = _f(p.get("avg_cost"))
    cur = _f(p.get("current_price"))
    if avg <= 0 or cur <= 0:
        return None
    return (cur - avg) / avg * 100.0


def _format_label(p: dict) -> str:
    """``NVDA`` for a stock, ``NVDA 600C 2026-05-30`` for an option.

    Mirrors the live Opus prompt's position line format (strategy._build_payload
    pos_lines) so an operator scanning the liquidation panel can map each
    row 1-to-1 against what the decision engine sees."""
    ticker = (p.get("ticker") or "").upper() or "?"
    if not _is_option(p):
        return ticker
    strike = p.get("strike")
    expiry = p.get("expiry") or ""
    otype = (p.get("type") or "")[:1].upper()
    try:
        sf = float(strike) if strike is not None else None
    except (TypeError, ValueError):
        sf = None
    if sf is None:
        return ticker
    strike_label = int(sf) if sf == int(sf) else sf
    return f"{ticker} {strike_label}{otype} {expiry}".strip()


def build_liquidation_preview(snapshot: dict) -> dict:
    """Compose the liquidation-preview view from a portfolio snapshot.

    ``snapshot`` — the ``strategy._portfolio_snapshot`` /
    ``portfolio_snapshot_readonly`` shape (``cash``, ``total_value``,
    ``open_value``, ``positions`` — already mark-to-market).

    Returns a dict the dashboard / Discord can render directly:
      * ``state`` — ``OK`` when there is at least one open position;
        ``NO_POSITIONS`` for an empty book (the suppression precedent
        the buying_power / capital_paralysis blocks already use).
      * ``current_cash`` — cash in book today.
      * ``liquidation_cash`` — cash after closing everything at marks.
      * ``cash_freed`` — ``liquidation_cash − current_cash`` (always >=
        0 absent a margin position — the live trader has none).
      * ``realized_pl_if_closed`` — sum of per-position unrealized_pl;
        the P/L a full liquidation locks in.
      * ``realized_pl_pct`` — that figure as % of the book's
        ``total_value`` at the time of the snapshot (the same baseline
        ``capital_paralysis`` deployed_pct uses). ``None`` when
        ``total_value`` is non-positive.
      * ``n_positions`` — open positions counted.
      * ``n_stale_marks`` — positions whose live price was unavailable
        and which mark at avg_cost (``stale_mark`` True). The lock-in
        figure is unreliable for these rows.
      * ``positions`` — per-position rows, sorted by absolute
        contribution to the lock-in DESC (biggest contributor first —
        the natural read order for partial-liquidation decisions):
        ``{ticker, label, type, qty, avg_cost, current_price,
           market_value, realized_pl, pl_pct, stale_mark}``.
      * ``headline`` — one short human-readable sentence the Discord
        path can include verbatim.

    Pure, never raises (every per-position calculation is wrapped via
    ``_f`` and the boolean/dict guards mirror ``buying_power``). The
    function is offline — zero I/O, deterministic given identical
    inputs (single source of truth with ``_mark_to_market``).
    """
    if not isinstance(snapshot, dict):
        snapshot = {}

    current_cash = _f(snapshot.get("cash"))
    total_value = _f(snapshot.get("total_value"))
    positions_raw = snapshot.get("positions") or []
    if not isinstance(positions_raw, list):
        positions_raw = []

    rows: list[dict] = []
    realized_total = 0.0
    liquid_total = current_cash
    stale_count = 0
    for p in positions_raw:
        if not isinstance(p, dict):
            continue
        cash = _position_close_cash(p)
        realized = _position_realized_if_closed(p)
        pl_pct = _position_pl_pct(p)
        stale = bool(p.get("stale_mark"))
        if stale:
            stale_count += 1
        rows.append({
            "ticker": (p.get("ticker") or "").upper() or "?",
            "label": _format_label(p),
            "type": p.get("type") or "stock",
            "qty": _f(p.get("qty")),
            "avg_cost": _f(p.get("avg_cost")),
            "current_price": _f(p.get("current_price")),
            "market_value": round(cash, 2),
            "realized_pl": round(realized, 2),
            "pl_pct": (round(pl_pct, 2) if pl_pct is not None else None),
            "stale_mark": stale,
        })
        realized_total += realized
        liquid_total += cash

    rows.sort(key=lambda r: abs(r["realized_pl"]), reverse=True)

    if not rows:
        return {
            "state": "NO_POSITIONS",
            "headline": "No open positions — nothing to liquidate.",
            "current_cash": round(current_cash, 2),
            "liquidation_cash": round(current_cash, 2),
            "cash_freed": 0.0,
            "realized_pl_if_closed": 0.0,
            "realized_pl_pct": None,
            "n_positions": 0,
            "n_stale_marks": 0,
            "positions": [],
        }

    pct = None
    if total_value > 0:
        pct = round(realized_total / total_value * 100.0, 2)

    # Render realized as "+$X.XX" / "-$X.XX" — the leading sign comes BEFORE
    # the dollar mark so a loss reads "-$200.00", not "$-200.00".
    if realized_total >= 0:
        realized_token = f"+${realized_total:.2f}"
    else:
        realized_token = f"-${abs(realized_total):.2f}"
    headline = (
        f"Liquidation at marks would free ${liquid_total:.2f} in cash "
        f"(currently ${current_cash:.2f}) and lock in "
        f"{realized_token} realized "
        f"across {len(rows)} position{'' if len(rows) == 1 else 's'}."
    )
    if stale_count:
        headline += (
            f" Warning: {stale_count} position{'' if stale_count == 1 else 's'} "
            f"mark{'s' if stale_count == 1 else ''} stale (live price unavailable)"
            "; lock-in unreliable for those rows."
        )

    return {
        "state": "OK",
        "headline": headline,
        "current_cash": round(current_cash, 2),
        "liquidation_cash": round(liquid_total, 2),
        "cash_freed": round(liquid_total - current_cash, 2),
        "realized_pl_if_closed": round(realized_total, 2),
        "realized_pl_pct": pct,
        "n_positions": len(rows),
        "n_stale_marks": stale_count,
        "positions": rows,
    }
