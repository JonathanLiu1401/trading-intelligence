"""Group trade ledger rows into closed round-trips.

Single source of truth for the round-trip aggregation consumed by
``dashboard.analytics_api`` (``win_rate``, ``profit_factor``,
``avg_holding_days``). The logic previously lived inline in that endpoint;
extracting it here keeps a future trade-attribution caller from drifting away
from a second hand-maintained copy. No other caller exists yet — this is the
shared helper they should use, not a claim that one is already wired up.

A round-trip is the slice of trades on the same (ticker, type, strike, expiry)
key that starts when qty rises from zero and ends when it returns to zero. A
re-BUY after a full close starts a new round-trip.
"""
from __future__ import annotations

from datetime import datetime


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _hold_days(buy_ts: str | None, sell_ts: str | None) -> float | None:
    b = _parse_ts(buy_ts)
    s = _parse_ts(sell_ts)
    if b is None or s is None:
        return None
    dd = (s - b).total_seconds() / 86400.0
    return round(dd, 4) if dd >= 0 else None


def build_round_trips(trades: list[dict]) -> list[dict]:
    """Group ``trades`` (oldest → newest) into closed round-trips.

    Each output dict has:

    * ``ticker``, ``type``, ``strike``, ``expiry`` — the position key
    * ``entry_ts``, ``exit_ts`` — first BUY and closing SELL timestamps
    * ``qty`` — total quantity opened across all BUYs in the round-trip
    * ``cost`` — gross BUY value (already factors option ×100 via trades.value)
    * ``proceeds`` — gross SELL value
    * ``pnl_usd`` — proceeds − cost
    * ``pnl_pct`` — pnl_usd / cost × 100 (None if cost is 0)
    * ``hold_days`` — calendar days entry→exit (None on parse error)
    * ``n_buys``, ``n_sells`` — trade-row counts
    * ``entry_trade_ids``, ``exit_trade_ids`` — DB row ids of contributing trades

    The ``trades`` input is expected to be a list of dicts shaped like
    ``Store.recent_trades()``: ``timestamp``, ``ticker``, ``action`` (BUY/SELL
    plus option variants), ``qty``, ``price``, ``value``, ``strike``,
    ``expiry``, ``option_type``, ``id``. Action prefix match handles
    ``BUY_CALL`` etc.
    """
    per_position: dict[tuple, dict] = {}
    out: list[dict] = []
    for t in trades:
        typ = t.get("option_type") or "stock"
        key = (t["ticker"], typ, t.get("strike"), t.get("expiry"))
        rec = per_position.setdefault(
            key,
            {
                "cost": 0.0,
                "proceeds": 0.0,
                "held": 0.0,
                "qty": 0.0,
                "first_buy_ts": None,
                "n_buys": 0,
                "n_sells": 0,
                "entry_trade_ids": [],
                "exit_trade_ids": [],
            },
        )
        action = (t.get("action") or "").upper()
        if action.startswith("BUY"):
            if abs(rec["held"]) < 1e-4:
                rec["first_buy_ts"] = t.get("timestamp")
                rec["entry_trade_ids"] = []
                rec["qty"] = 0.0
            rec["cost"] += float(t.get("value") or 0.0)
            rec["held"] += float(t.get("qty") or 0.0)
            rec["qty"] += float(t.get("qty") or 0.0)
            rec["n_buys"] += 1
            if t.get("id") is not None:
                rec["entry_trade_ids"].append(t["id"])
        elif action.startswith("SELL"):
            rec["proceeds"] += float(t.get("value") or 0.0)
            rec["held"] -= float(t.get("qty") or 0.0)
            rec["n_sells"] += 1
            if t.get("id") is not None:
                rec["exit_trade_ids"].append(t["id"])
            if abs(rec["held"]) < 1e-4:
                pnl = rec["proceeds"] - rec["cost"]
                pnl_pct = round(pnl / rec["cost"] * 100, 4) if rec["cost"] > 1e-9 else None
                out.append(
                    {
                        "ticker": key[0],
                        "type": key[1],
                        "strike": key[2],
                        "expiry": key[3],
                        "entry_ts": rec["first_buy_ts"],
                        "exit_ts": t.get("timestamp"),
                        "qty": round(rec["qty"], 6),
                        "cost": round(rec["cost"], 4),
                        "proceeds": round(rec["proceeds"], 4),
                        "pnl_usd": round(pnl, 4),
                        "pnl_pct": pnl_pct,
                        "hold_days": _hold_days(rec["first_buy_ts"], t.get("timestamp")),
                        "n_buys": rec["n_buys"],
                        "n_sells": rec["n_sells"],
                        "entry_trade_ids": list(rec["entry_trade_ids"]),
                        "exit_trade_ids": list(rec["exit_trade_ids"]),
                    }
                )
                rec["cost"] = rec["proceeds"] = rec["held"] = rec["qty"] = 0.0
                rec["first_buy_ts"] = None
                rec["n_buys"] = rec["n_sells"] = 0
                rec["entry_trade_ids"] = []
                rec["exit_trade_ids"] = []
    return out
