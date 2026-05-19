"""Market data + paper order execution. yfinance is the data source."""
from __future__ import annotations

import time
from datetime import datetime, date, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import yfinance as yf

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# 2026 NYSE holidays (full closes).
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

# 2026 NYSE early-close half-days — the regular session ends at 1:00 p.m. ET,
# not 4:00 p.m. Previously "not enforced — we'll trade through them", which
# meant the engine believed the market was open 13:00–16:00 ET on these days:
# it ran the fast 30-min OPEN cadence and *executed trades against frozen
# post-close yfinance marks* for three hours of a CLOSED market, twice a year.
# Enforcing the early close makes is_market_open() — and therefore the runner
# sleep cadence, the prompt's MARKET_OPEN flag, and every market-hours gate —
# correct on these days. An unknown half-day still falls through to the 16:00
# close (same conservative "trade through what we don't know" default the
# holiday calendar uses).
NYSE_HALF_DAYS_2026 = {
    date(2026, 11, 27),  # Day after Thanksgiving — 1:00 p.m. ET close
    date(2026, 12, 24),  # Christmas Eve — 1:00 p.m. ET close
}

_REGULAR_CLOSE_MIN = 16 * 60       # 16:00 ET, minutes since ET midnight
_EARLY_CLOSE_MIN = 13 * 60         # 13:00 ET half-day close
_OPEN_MIN = 9 * 60 + 30            # 09:30 ET open


def is_half_day(d: date) -> bool:
    """True if ``d`` is a known NYSE early-close (1:00 p.m. ET) session."""
    return d in NYSE_HALF_DAYS_2026


def close_minute(d: date) -> int:
    """NYSE regular-session close for ``d`` as minutes since ET midnight:
    13:00 on a known half-day, otherwise the regular 16:00 close. A weekend
    or full holiday has no session — callers gate that separately via
    ``is_market_open``; this only answers 'when does the bell ring'."""
    return _EARLY_CLOSE_MIN if d in NYSE_HALF_DAYS_2026 else _REGULAR_CLOSE_MIN

_PRICE_CACHE: dict[str, tuple[float, float]] = {}  # ticker -> (price, ts)
_PRICE_TTL = 30.0  # seconds

# Negative cache: symbols that returned no data recently (delisted/invalid like
# METAU/GOOGU, or futures closed on weekends like ES=F). Without this, every
# decision cycle re-requests these from yfinance, failing every time and
# spamming the log. TTL is short (5 min) so legitimately-transient outages
# (e.g. futures over a weekend) recover quickly once data returns.
_DEAD_CACHE: dict[str, float] = {}  # ticker -> ts when marked dead
_DEAD_TTL = 300.0  # seconds


def _is_dead(ticker: str) -> bool:
    ts = _DEAD_CACHE.get(ticker)
    return ts is not None and time.time() - ts < _DEAD_TTL


def _mark_dead(ticker: str):
    # Only log the first time a symbol goes dead within a TTL window, not every cycle.
    if not _is_dead(ticker):
        print(f"[market] no data for {ticker}; suppressing re-fetch for {int(_DEAD_TTL)}s")
    _DEAD_CACHE[ticker] = time.time()


def _clear_dead(ticker: str):
    _DEAD_CACHE.pop(ticker, None)


def is_market_open(now: datetime | None = None) -> bool:
    now = (now or datetime.now(UTC)).astimezone(NY)
    if now.weekday() >= 5:
        return False
    if now.date() in NYSE_HOLIDAYS_2026:
        return False
    minutes = now.hour * 60 + now.minute
    return _OPEN_MIN <= minutes < close_minute(now.date())


def next_session_open(now: datetime | None = None) -> datetime | None:
    """The next regular-session 09:30 ET open after ``now``.

    Returns a UTC-aware datetime, or ``None`` when no NYSE open day can be
    found within 14 forward days (defensive; the actual NYSE calendar has
    no >4-day gap). When the market is currently OPEN, returns the *next*
    open after today's close (so callers always get a future timestamp;
    "when can I act next?" semantics).

    Pure — walks the NYSE_HOLIDAYS_2026 / weekday calendar from ``now``,
    no I/O. The week-end NY date is the only state read besides
    ``NYSE_HOLIDAYS_2026``. Half-day sessions (NYSE_HALF_DAYS_2026) still
    *open* at 09:30 ET, so they are a normal open day from this helper's
    perspective; only the close is early.
    """
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    now_ny = now_utc.astimezone(NY)
    cur_min = now_ny.hour * 60 + now_ny.minute
    # If we're inside today's session OR past today's close, today is no
    # longer the next open — advance to tomorrow before scanning.
    advance_past_today = (
        now_ny.weekday() < 5
        and now_ny.date() not in NYSE_HOLIDAYS_2026
        and cur_min >= _OPEN_MIN
    )
    candidate = now_ny
    if advance_past_today:
        candidate = (candidate + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    for _ in range(14):
        if candidate.weekday() < 5 and candidate.date() not in NYSE_HOLIDAYS_2026:
            is_today = candidate.date() == now_ny.date()
            if not is_today or cur_min < _OPEN_MIN:
                open_dt = candidate.replace(
                    hour=9, minute=30, second=0, microsecond=0
                )
                return open_dt.astimezone(UTC)
        candidate = (candidate + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    return None


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
    if _is_dead(ticker):
        return None
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
            _clear_dead(ticker)
            return float(price)
    except Exception as e:
        print(f"[market] price fetch failed {ticker}: {e}")
    _mark_dead(ticker)
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
        elif _is_dead(t):
            out[t] = None  # known-dead: skip the yfinance round-trip this cycle
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
                    price = get_price(t)  # also handles dead-marking
                if price is not None:
                    _store_price(t, price)
                    _clear_dead(t)
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
