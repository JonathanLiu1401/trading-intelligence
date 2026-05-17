"""Capital deployment & liquidity — can the trader still act, or is it pinned?

Observed live: $6.23 cash of a $972 book (99.4% deployed) across two losing
positions, zero new entries, zero BLOCKED rows. ``/api/risk`` reports
*concentration*; nothing reports *liquidity* — the simple "you have no dry
powder, you are fully invested in N red names, and you have not opened a new
position in X days" picture a desk checks first.

``build_liquidity`` is pure: feed it the portfolio dict, the open-positions
list and the recent-trades list (as returned by ``store``); it returns a
JSON-ready dict. ``now`` is injectable for deterministic tests.
"""
from __future__ import annotations

from datetime import datetime, timezone

_OPEN_ACTIONS = {"BUY", "BUY_CALL", "BUY_PUT"}


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _market_value(pos: dict) -> float:
    """Mark-to-market value of a position. Options are ×100; an unmarked
    current_price (0/None) falls back to avg_cost so a fresh fill isn't
    counted as $0 deployed."""
    qty = float(pos.get("qty") or 0.0)
    px = pos.get("current_price")
    if not px:  # 0 or None — not yet marked
        px = pos.get("avg_cost") or 0.0
    mult = 100 if pos.get("type") in ("call", "put") else 1
    return float(px) * qty * mult


def _cost_basis(pos: dict) -> float:
    qty = float(pos.get("qty") or 0.0)
    mult = 100 if pos.get("type") in ("call", "put") else 1
    return float(pos.get("avg_cost") or 0.0) * qty * mult


def build_liquidity(portfolio: dict,
                    positions: list[dict],
                    trades: list[dict],
                    now: datetime | None = None) -> dict:
    """Deployment / liquidity snapshot. Pure — never touches the DB."""
    now = now or datetime.now(timezone.utc)
    cash = float((portfolio or {}).get("cash") or 0.0)
    total_value = float((portfolio or {}).get("total_value") or 0.0)
    positions = positions or []
    trades = trades or []

    invested = max(0.0, total_value - cash)
    cash_pct = round(cash / total_value * 100, 2) if total_value > 0 else 0.0
    deployed_pct = round(max(0.0, min(100.0, 100.0 - cash_pct)), 2) if total_value > 0 else 0.0

    detail = []
    cost_total = 0.0
    for p in positions:
        mv = _market_value(p)
        cb = _cost_basis(p)
        cost_total += cb
        pl = mv - cb
        detail.append({
            "ticker": p.get("ticker"),
            "type": p.get("type"),
            "market_value": round(mv, 2),
            "weight_pct": round(mv / total_value * 100, 2) if total_value > 0 else 0.0,
            "unrealized_pl": round(pl, 2),
            "pl_pct": round(pl / cb * 100, 2) if cb else 0.0,
        })
    detail.sort(key=lambda r: -r["market_value"])

    n_positions = len(detail)
    largest = detail[0] if detail else None
    top_weight_pct = largest["weight_pct"] if largest else 0.0
    upl_total = round(sum(d["unrealized_pl"] for d in detail), 2)
    upl_pct = round(upl_total / cost_total * 100, 2) if cost_total else 0.0
    n_losers = sum(1 for d in detail if d["unrealized_pl"] < 0)
    n_winners = sum(1 for d in detail if d["unrealized_pl"] > 0)

    # Time since the last *opening* trade vs. any trade.
    last_entry_ts = None
    last_trade_ts = None
    for t in trades:  # newest-first
        ts = _parse_ts(t.get("timestamp"))
        if ts is None:
            continue
        if last_trade_ts is None:
            last_trade_ts = ts
        if last_entry_ts is None and (t.get("action") or "").upper() in _OPEN_ACTIONS:
            last_entry_ts = ts
        if last_trade_ts is not None and last_entry_ts is not None:
            break

    def _days(ts):
        return round((now - ts).total_seconds() / 86400, 2) if ts else None

    days_since_entry = _days(last_entry_ts)
    days_since_trade = _days(last_trade_ts)

    # A trade costs ≥ a fraction of a share; treat sub-$1 / sub-1% cash as
    # effectively unable to act on a fresh signal.
    can_act = cash >= 1.0 and (total_value <= 0 or cash_pct >= 1.0)

    if total_value <= 0 and n_positions == 0:
        status = "NO_DATA"
    elif cash_pct < 2.0 and n_positions > 0:
        status = "NO_DRY_POWDER"
    elif cash_pct < 5.0:
        status = "DRY_POWDER_LOW"
    elif cash_pct > 60.0:
        status = "CASH_HEAVY"
    else:
        status = "BALANCED"

    flags: list[str] = []
    if total_value > 0:
        flags.append(f"{deployed_pct:.1f}% of book deployed")
    if not can_act:
        flags.append("no dry powder — cannot act on a new BUY signal")
    if n_positions > 0 and n_losers == n_positions:
        flags.append(f"all {n_positions} open positions underwater")
    if n_positions and top_weight_pct >= 50.0:
        flags.append(f"{largest['ticker']} is {top_weight_pct:.0f}% of the book")
    if days_since_entry is not None and days_since_entry >= 2.0:
        flags.append(f"no new position opened in {days_since_entry:.1f}d")
    elif days_since_entry is None and n_positions > 0:
        flags.append("no opening trade on record")

    if status == "NO_DRY_POWDER":
        headline = (f"Pinned: {deployed_pct:.1f}% deployed across {n_positions} "
                    f"position(s), {cash_pct:.2f}% cash — no room to act")
    elif status == "DRY_POWDER_LOW":
        headline = f"Low dry powder: {cash_pct:.1f}% cash"
    elif status == "CASH_HEAVY":
        headline = f"Cash-heavy: {cash_pct:.1f}% uninvested"
    elif status == "BALANCED":
        headline = f"Balanced: {cash_pct:.1f}% cash / {deployed_pct:.1f}% deployed"
    else:
        headline = "No portfolio data"

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "status": status,
        "headline": headline,
        "cash": round(cash, 2),
        "total_value": round(total_value, 2),
        "invested_value": round(invested, 2),
        "cash_pct": cash_pct,
        "deployed_pct": deployed_pct,
        "can_act_on_signal": can_act,
        "n_positions": n_positions,
        "top_weight_pct": top_weight_pct,
        "largest_position": largest["ticker"] if largest else None,
        "unrealized_pl": upl_total,
        "unrealized_pl_pct": upl_pct,
        "n_winners": n_winners,
        "n_losers": n_losers,
        "days_since_last_entry": days_since_entry,
        "days_since_last_trade": days_since_trade,
        "flags": flags,
        "positions": detail,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json
    from paper_trader.store import get_store
    s = get_store()
    rep = build_liquidity(s.get_portfolio(), s.open_positions(), s.recent_trades(200))
    print(json.dumps(rep, indent=2, default=str))
