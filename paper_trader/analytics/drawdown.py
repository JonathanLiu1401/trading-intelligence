"""Drawdown anatomy — decompose the current drawdown from peak.

Walks the equity_curve table, finds the all-time-peak total_value, computes the
current drawdown %, time-in-DD, and decomposes which open positions are
contributing most to the current shortfall (by peak-day P/L vs now-P/L delta).

When the portfolio is at a fresh high, the response still returns a structured
zero — the dashboard shows a green "at high-water" badge.
"""
from __future__ import annotations

from datetime import datetime, timezone


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def compute_drawdown(
    equity_curve: list[dict],
    open_positions: list[dict],
    starting_equity: float = 1000.0,
) -> dict:
    """Compute drawdown stats + per-position contribution.

    equity_curve: chronological list of {timestamp, total_value, cash, sp500_price}.
    open_positions: current open positions w/ avg_cost, qty, current_price, unrealized_pl.
    """
    if not equity_curve:
        return {
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "current_value": starting_equity,
            "peak_value": starting_equity,
            "peak_ts": None,
            "drawdown_pct": 0.0,
            "drawdown_abs": 0.0,
            "hours_in_dd": 0.0,
            "at_high_water": True,
            "contributors": [],
            "trough_value": starting_equity,
            "trough_ts": None,
            "recovery_pct": 0.0,
            "history": [],
        }

    # Walk forward finding running peak and trough since peak.
    peak_value = -1e18
    peak_ts = None
    trough_value = None
    trough_ts = None
    history = []
    last_total = starting_equity
    for row in equity_curve:
        tv = float(row.get("total_value") or 0.0)
        ts = row.get("timestamp")
        last_total = tv
        if tv > peak_value:
            peak_value = tv
            peak_ts = ts
            trough_value = tv
            trough_ts = ts
        else:
            if trough_value is None or tv < trough_value:
                trough_value = tv
                trough_ts = ts
        history.append({"ts": ts, "v": round(tv, 2)})

    current = last_total
    dd_abs = current - peak_value
    dd_pct = (dd_abs / peak_value * 100.0) if peak_value else 0.0
    at_hwm = dd_pct >= -0.01  # within 1bp of high-water
    hours = 0.0
    if peak_ts:
        peak_dt = _parse_ts(peak_ts)
        if peak_dt:
            hours = (datetime.now(timezone.utc) - peak_dt).total_seconds() / 3600.0
    trough_abs = (trough_value or current) - peak_value
    trough_pct = (trough_abs / peak_value * 100.0) if peak_value else 0.0
    # How much of the trough has been recovered already (0% = still at trough, 100% = back to peak).
    recovery_pct = 0.0
    if trough_value is not None and trough_value < peak_value:
        denom = peak_value - trough_value
        if denom > 0:
            recovery_pct = max(0.0, min(100.0, (current - trough_value) / denom * 100.0))

    # Per-position contribution. We don't have full per-position P/L history,
    # so we approximate "contribution to current DD" as each open position's
    # current unrealized P/L (negative positions = drag, positive = offset).
    contributors = []
    for p in open_positions:
        if (p.get("qty") or 0) <= 0:
            continue
        upl = float(p.get("unrealized_pl") or 0.0)
        cost = float(p.get("avg_cost") or 0.0) * float(p.get("qty") or 0.0)
        pl_pct = (upl / cost * 100.0) if cost > 0 else 0.0
        contributors.append({
            "ticker": p.get("ticker"),
            "type": p.get("type"),
            "qty": p.get("qty"),
            "avg_cost": round(float(p.get("avg_cost") or 0.0), 2),
            "current_price": round(float(p.get("current_price") or 0.0), 2),
            "unrealized_pl": round(upl, 2),
            "pl_pct": round(pl_pct, 2),
            "cost_basis": round(cost, 2),
            "drag": upl < 0,
        })
    contributors.sort(key=lambda c: c["unrealized_pl"])  # most-negative first

    # Compact peak-window history (down-sample to ≤ 200 points)
    if len(history) > 200:
        step = max(1, len(history) // 200)
        history = history[::step] + ([history[-1]] if (len(history) - 1) % step else [])

    return {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "current_value": round(current, 2),
        "peak_value": round(peak_value, 2),
        "peak_ts": peak_ts,
        "drawdown_pct": round(dd_pct, 2),
        "drawdown_abs": round(dd_abs, 2),
        "hours_in_dd": round(hours, 2),
        "at_high_water": at_hwm,
        "trough_value": round(trough_value, 2) if trough_value is not None else None,
        "trough_ts": trough_ts,
        "trough_pct": round(trough_pct, 2),
        "recovery_pct": round(recovery_pct, 1),
        "contributors": contributors,
        "history": history,
        "starting_equity": round(starting_equity, 2),
    }
