"""Portfolio P&L tracker — reads holdings, fetches live prices, computes unrealized P&L."""
import json
from pathlib import Path
from typing import Optional

import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"


def _load_positions() -> list[dict]:
    with open(PORTFOLIO_PATH, "r") as f:
        data = json.load(f)
    return data.get("positions", [])


def _fetch_price(ticker: str) -> Optional[float]:
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="2d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        info = t.info or {}
        for key in ("currentPrice", "regularMarketPrice", "previousClose"):
            v = info.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    except Exception:
        return None
    return None


def collect_portfolio_pnl() -> dict:
    """Fetch live prices and compute P&L for every position. Returns dict with positions + summary."""
    positions = _load_positions()
    results = []
    total_market = 0.0
    total_cost = 0.0

    for pos in positions:
        ticker = pos["ticker"]
        qty = float(pos["qty"])
        avg_cost = float(pos["avg_cost"])
        price = _fetch_price(ticker)

        cost_basis = qty * avg_cost
        if price is None:
            results.append({
                "ticker": ticker,
                "qty": qty,
                "avg_cost": avg_cost,
                "current_price": None,
                "market_value": None,
                "cost_basis": round(cost_basis, 2),
                "unrealized_pnl": None,
                "pnl_pct": None,
            })
            continue

        market_value = qty * price
        unrealized = market_value - cost_basis
        pnl_pct = (unrealized / cost_basis * 100.0) if cost_basis else 0.0

        total_market += market_value
        total_cost += cost_basis

        results.append({
            "ticker": ticker,
            "qty": qty,
            "avg_cost": avg_cost,
            "current_price": round(price, 2),
            "market_value": round(market_value, 2),
            "cost_basis": round(cost_basis, 2),
            "unrealized_pnl": round(unrealized, 2),
            "pnl_pct": round(pnl_pct, 2),
        })

    total_pnl = total_market - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100.0) if total_cost else 0.0

    return {
        "positions": results,
        "summary": {
            "total_market_value": round(total_market, 2),
            "total_cost": round(total_cost, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
        },
    }


def format_pnl_block(pnl_data: dict) -> str:
    """Render P&L data as a clean ASCII table suitable for Discord code blocks."""
    if not pnl_data or not pnl_data.get("positions"):
        return "N/A"

    lines = ["╔══ PORTFOLIO P&L ══════════════════════════════════════╗"]
    for p in pnl_data["positions"]:
        tkr = f"{p['ticker']:<5}"
        qty = f"{p['qty']:>7.2f}"
        if p["current_price"] is None:
            lines.append(f"  {tkr} {qty}  price N/A  cost ${p['cost_basis']:>7.2f}")
            continue
        price = f"${p['current_price']:>8.2f}"
        cost = f"${p['cost_basis']:>8.2f}"
        sign = "+" if p["unrealized_pnl"] >= 0 else ""
        pnl = f"{sign}${p['unrealized_pnl']:>8.2f}"
        pct_sign = "+" if p["pnl_pct"] >= 0 else ""
        pct = f"({pct_sign}{p['pnl_pct']:.1f}%)"
        lines.append(f"  {tkr} {qty}  {price}  cost {cost}  PNL {pnl} {pct}")

    s = pnl_data["summary"]
    tsign = "+" if s["total_pnl"] >= 0 else ""
    tpct_sign = "+" if s["total_pnl_pct"] >= 0 else ""
    lines.append("  " + "─" * 53)
    lines.append(
        f"  TOTAL  market ${s['total_market_value']:>9.2f}  cost ${s['total_cost']:>9.2f}  "
        f"PNL {tsign}${s['total_pnl']:.2f} ({tpct_sign}{s['total_pnl_pct']:.1f}%)"
    )
    lines.append("╚═══════════════════════════════════════════════════════╝")
    return "\n".join(lines)


if __name__ == "__main__":
    data = collect_portfolio_pnl()
    print(format_pnl_block(data))
