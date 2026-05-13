"""Stock data collector using yfinance."""
import json
from pathlib import Path

import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"

MIN_MARKET_CAP_GLOBAL = 10_000_000_000  # $10B threshold for global stocks


def _load_watchlist():
    with open(WATCHLIST_PATH, "r") as f:
        return json.load(f)


def _is_global_ticker(ticker: str) -> bool:
    return any(suffix in ticker for suffix in (".KS", ".T", ".HK", ".L", ".SS", ".SZ"))


def _fetch_one(ticker: str):
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        hist = t.history(period="5d")
        if hist.empty:
            return None

        last_close = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last_close
        pct_change = ((last_close - prev_close) / prev_close * 100.0) if prev_close else 0.0

        market_cap = info.get("marketCap")
        week52_high = info.get("fiftyTwoWeekHigh")
        week52_low = info.get("fiftyTwoWeekLow")

        return {
            "ticker": ticker,
            "price": round(last_close, 2),
            "pct_change": round(pct_change, 2),
            "52w_high": week52_high,
            "52w_low": week52_low,
            "market_cap": market_cap,
            "name": info.get("shortName") or info.get("longName") or ticker,
        }
    except Exception as e:
        print(f"[stock_data] Error fetching {ticker}: {e}")
        return None


def get_stock_data():
    """Fetch price/range/% change for watchlist tickers.

    Returns a dict with categories: macro (indices/crypto/commodities/forex/bonds)
    and equities (stocks/etfs). Global tickers filtered to market cap > $10B.
    """
    watchlist = _load_watchlist()

    macro_keys = ("indices", "crypto", "commodities", "bonds", "forex")
    equity_keys = ("memory_core", "semis_equipment", "broader_semis", "korean", "japanese", "etfs", "portfolio")

    macro_tickers, equity_tickers = [], []
    for key in macro_keys:
        macro_tickers.extend(watchlist.get(key, []))
    for key in equity_keys:
        equity_tickers.extend(watchlist.get(key, []))

    def _dedup(lst):
        seen = set()
        return [t for t in lst if not (t in seen or seen.add(t))]

    macro_tickers = _dedup(macro_tickers)
    equity_tickers = _dedup(equity_tickers)

    macro_data, equity_data = [], []

    for ticker in macro_tickers:
        data = _fetch_one(ticker)
        if data:
            macro_data.append(data)

    for ticker in equity_tickers:
        data = _fetch_one(ticker)
        if not data:
            continue
        if _is_global_ticker(ticker):
            mc = data.get("market_cap") or 0
            if mc < MIN_MARKET_CAP_GLOBAL:
                continue
        equity_data.append(data)

    return {"macro": macro_data, "equities": equity_data}


if __name__ == "__main__":
    rows = get_stock_data()
    for r in rows:
        print(f"{r['ticker']:>12}  {r['price']:>10}  {r['pct_change']:+.2f}%")
