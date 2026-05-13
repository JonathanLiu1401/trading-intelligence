"""Portfolio P&L tracker — fetches live prices via yfinance, computes unrealized P&L."""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"

log = logging.getLogger("portfolio_pnl")

# ANSI colors (only used if stdout is a tty)
_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_ANSI_BOLD = "\033[1m"
_ANSI_RESET = "\033[0m"


def _load_positions() -> list[dict]:
    try:
        with open(PORTFOLIO_PATH, "r") as f:
            data = json.load(f)
        return data.get("positions", [])
    except Exception as e:
        log.warning(f"portfolio.json load failed: {e}")
        return []


def _fetch_price(ticker: str) -> Optional[float]:
    """Return latest available price for ticker, or None on failure."""
    if yf is None:
        return None
    # Try fast_info first (cheap, single call)
    try:
        t = yf.Ticker(ticker)
        try:
            fi = t.fast_info
            for key in ("last_price", "lastPrice", "regular_market_price",
                        "regularMarketPrice", "previous_close", "previousClose"):
                v = None
                try:
                    v = fi[key] if hasattr(fi, "__getitem__") else getattr(fi, key, None)
                except Exception:
                    v = getattr(fi, key, None)
                if isinstance(v, (int, float)) and v > 0:
                    return float(v)
        except Exception:
            pass
        # Fallback: short history
        try:
            hist = t.history(period="2d", auto_adjust=False)
            if hist is not None and not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
    except Exception as e:
        log.warning(f"yfinance fetch failed for {ticker}: {e}")
    return None


def get_portfolio_pnl() -> Optional[dict]:
    """Read portfolio.json, fetch live prices, compute per-position and total P&L.

    Returns dict {"positions": [...], "summary": {...}, "as_of": "<ISO ts>"} on success,
    or None if yfinance is unavailable or no positions could be priced.
    """
    if yf is None:
        log.warning("yfinance not installed; skipping P&L snapshot")
        return None

    positions = _load_positions()
    if not positions:
        return None

    results: list[dict] = []
    total_market = 0.0
    total_cost = 0.0
    priced_count = 0

    for pos in positions:
        ticker = pos.get("ticker")
        if not ticker:
            continue
        try:
            qty = float(pos.get("qty", 0))
            avg_cost = float(pos.get("avg_cost", 0))
        except (TypeError, ValueError):
            continue

        cost_basis = qty * avg_cost
        price = _fetch_price(ticker)

        row = {
            "ticker": ticker,
            "qty": qty,
            "avg_cost": avg_cost,
            "price": None,
            "value": None,
            "cost": round(cost_basis, 2),
            "pnl": None,
            "pnl_pct": None,
        }

        if price is not None:
            value = qty * price
            pnl = value - cost_basis
            pnl_pct = (pnl / cost_basis * 100.0) if cost_basis else 0.0
            row.update({
                "price": round(price, 2),
                "value": round(value, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
            })
            total_market += value
            total_cost += cost_basis
            priced_count += 1

        results.append(row)

    if priced_count == 0:
        log.warning("portfolio P&L: no positions could be priced")
        return None

    total_pnl = total_market - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100.0) if total_cost else 0.0

    return {
        "positions": results,
        "summary": {
            "total_value": round(total_market, 2),
            "total_cost": round(total_cost, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "priced": priced_count,
            "total_positions": len(results),
        },
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# Backward-compat alias for any older callers
collect_portfolio_pnl = get_portfolio_pnl


def _color(text: str, val: Optional[float], use_color: bool) -> str:
    if not use_color or val is None:
        return text
    if val > 0:
        return f"{_ANSI_GREEN}{text}{_ANSI_RESET}"
    if val < 0:
        return f"{_ANSI_RED}{text}{_ANSI_RESET}"
    return text


def format_pnl_block(data: Optional[dict]) -> str:
    """Render P&L data as a fixed-width ASCII table (<= 58 cols).

    Columns: TICKER  QTY  PRICE  VALUE  COST  PNL$  PNL%
    """
    if not data or not data.get("positions"):
        return "N/A"

    use_color = sys.stdout.isatty()

    # Width target: 58 chars including borders
    # Layout (56 inner chars):
    #   TKR  QTY    PRICE     VALUE     COST     PNL$    PNL%
    #   5    7      8         9         9        9       7   = 54 + 6 spaces ~ 60 — tighten
    # Use compact 56-inner layout:
    header = f"{'TKR':<5}{'QTY':>7} {'PRICE':>8} {'VALUE':>9} {'COST':>9} {'PNL$':>9} {'PNL%':>7}"
    sep = "-" * len(header)
    inner_w = len(header)
    border_top = "+" + "=" * (inner_w + 2) + "+"
    # spec says ╔ / ╚ borders
    top = "╔" + "═" * (inner_w + 2) + "╗"
    bot = "╚" + "═" * (inner_w + 2) + "╝"
    mid = "║ " + header + " ║"
    sep_row = "║ " + sep + " ║"

    lines = [top, mid, sep_row]

    for p in data["positions"]:
        tkr = f"{p['ticker'][:5]:<5}"
        qty = f"{p['qty']:>7.2f}"
        if p["price"] is None:
            row_text = f"{tkr}{qty} {'N/A':>8} {'N/A':>9} {p['cost']:>9.2f} {'N/A':>9} {'N/A':>7}"
            lines.append("║ " + row_text + " ║")
            continue
        pnl = p["pnl"]
        pct = p["pnl_pct"]
        price_s = f"{p['price']:>8.2f}"
        value_s = f"{p['value']:>9.2f}"
        cost_s = f"{p['cost']:>9.2f}"
        pnl_s = f"{pnl:>+9.2f}"
        pct_s = f"{pct:>+6.1f}%"
        plain = f"{tkr}{qty} {price_s} {value_s} {cost_s} {pnl_s} {pct_s}"
        # Apply color only to PNL fields (preserves column widths)
        if use_color:
            colored_pnl = _color(pnl_s, pnl, True)
            colored_pct = _color(pct_s, pnl, True)
            row_text = f"{tkr}{qty} {price_s} {value_s} {cost_s} {colored_pnl} {colored_pct}"
        else:
            row_text = plain
        lines.append("║ " + row_text + " ║")

    s = data["summary"]
    lines.append(sep_row)
    tot_tkr = f"{'TOTAL':<5}"
    tot_qty = f"{'':>7}"
    tot_price = f"{'':>8}"
    tot_val = f"{s['total_value']:>9.2f}"
    tot_cost = f"{s['total_cost']:>9.2f}"
    tot_pnl = f"{s['total_pnl']:>+9.2f}"
    tot_pct = f"{s['total_pnl_pct']:>+6.1f}%"
    plain = f"{tot_tkr}{tot_qty} {tot_price} {tot_val} {tot_cost} {tot_pnl} {tot_pct}"
    if use_color:
        c_pnl = _color(tot_pnl, s["total_pnl"], True)
        c_pct = _color(tot_pct, s["total_pnl"], True)
        row_text = f"{_ANSI_BOLD}{tot_tkr}{_ANSI_RESET}{tot_qty} {tot_price} {tot_val} {tot_cost} {c_pnl} {c_pct}"
    else:
        row_text = plain
    lines.append("║ " + row_text + " ║")
    lines.append(bot)

    as_of = data.get("as_of", "")
    if as_of:
        lines.append(f"as of {as_of}")

    return "\n".join(lines)


if __name__ == "__main__":
    d = get_portfolio_pnl()
    if d is None:
        print("P&L fetch failed")
    else:
        print(format_pnl_block(d))
