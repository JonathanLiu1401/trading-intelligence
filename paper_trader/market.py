"""Market data + paper order execution. yfinance is the data source."""
from __future__ import annotations

import time
from datetime import datetime, date, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import yfinance as yf

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# 2026 NYSE holidays (full closes). Half-days not enforced — we'll trade through them.
NYSE_HOLIDAYS_2026 = {
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
}

_PRICE_CACHE: dict[str, tuple[float, float]] = {}  # ticker -> (price, ts)
_PRICE_TTL = 30.0  # seconds


def is_market_open(now: datetime | None = None) -> bool:
    now = (now or datetime.now(UTC)).astimezone(NY)
    if now.weekday() >= 5:
        return False
    if now.date() in NYSE_HOLIDAYS_2026:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= minutes < 16 * 60


def _cached_price(ticker: str) -> float | None:
    rec = _PRICE_CACHE.get(ticker)
    if rec and time.time() - rec[1] < _PRICE_TTL:
        return rec[0]
    return None


def _store_price(ticker: str, price: float):
    _PRICE_CACHE[ticker] = (price, time.time())


def get_price(ticker: str) -> float | None:
    """Latest trade price for the symbol, or None."""
    cached = _cached_price(ticker)
    if cached is not None:
        return cached
    try:
        t = yf.Ticker(ticker)
        fast = t.fast_info
        price = fast.get("last_price") or fast.get("regular_market_price")
        if price is None or price <= 0:
            hist = t.history(period="1d", interval="1m")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        if price and price > 0:
            _store_price(ticker, float(price))
            return float(price)
    except Exception as e:
        print(f"[market] price fetch failed {ticker}: {e}")
    return None


def get_prices(tickers: list[str]) -> dict[str, float | None]:
    """Bulk price fetch. Falls back to single requests on bulk failure."""
    if not tickers:
        return {}
    out: dict[str, float | None] = {}
    missing: list[str] = []
    for t in tickers:
        c = _cached_price(t)
        if c is not None:
            out[t] = c
        else:
            missing.append(t)
    if missing:
        try:
            data = yf.download(missing, period="1d", interval="1m",
                               group_by="ticker", progress=False, threads=True, auto_adjust=False)
            for t in missing:
                price = None
                try:
                    if len(missing) == 1:
                        closes = data["Close"].dropna()
                    else:
                        closes = data[t]["Close"].dropna()
                    if len(closes):
                        price = float(closes.iloc[-1])
                except Exception:
                    price = None
                if price is None:
                    price = get_price(t)
                if price is not None:
                    _store_price(t, price)
                out[t] = price
        except Exception:
            for t in missing:
                out[t] = get_price(t)
    return out


def get_options_chain(ticker: str, target_dte: int = 14) -> dict | None:
    """Return options chain for the expiry closest to target_dte days away."""
    try:
        t = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return None
        today = date.today()
        target = today + timedelta(days=target_dte)
        chosen = min(expiries, key=lambda d: abs((date.fromisoformat(d) - target).days))
        chain = t.option_chain(chosen)
        return {
            "ticker": ticker,
            "expiry": chosen,
            "calls": chain.calls[["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]].head(30).to_dict("records"),
            "puts": chain.puts[["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]].head(30).to_dict("records"),
        }
    except Exception as e:
        print(f"[market] options fetch failed {ticker}: {e}")
        return None


def get_option_price(ticker: str, expiry: str, strike: float, option_type: str) -> float | None:
    """Mid-of-bid-ask for a specific option contract. Hard-fails if strike not in chain
    (silently substituting the nearest strike would create position/price mismatches)."""
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiry)
        df = chain.calls if option_type == "call" else chain.puts
        row = df[df["strike"] == strike]
        if row.empty:
            print(f"[market] strike {strike} not in {ticker} {expiry} {option_type} chain")
            return None
        last = float(row["lastPrice"].iloc[0])
        bid = float(row["bid"].iloc[0])
        ask = float(row["ask"].iloc[0])
        if bid > 0 and ask > 0:
            return round((bid + ask) / 2, 4)
        return last if last > 0 else None
    except Exception as e:
        print(f"[market] option price failed {ticker} {expiry} {strike} {option_type}: {e}")
        return None


@lru_cache(maxsize=64)
def get_futures_price_cached(symbol: str, bucket: int) -> float | None:
    return get_price(symbol)


def get_futures_price(symbol: str) -> float | None:
    """yfinance futures (e.g. ES=F, NQ=F, CL=F, GC=F). 30s bucket cache."""
    return get_futures_price_cached(symbol, int(time.time() // 30))


def benchmark_sp500() -> float | None:
    return get_price("^GSPC")
