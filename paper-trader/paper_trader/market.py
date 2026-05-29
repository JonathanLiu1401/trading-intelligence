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

# Granular phase windows around the NYSE session. The trader's prompt header
# historically carried only the binary ``MARKET_OPEN: True/False`` — but a
# decision at 09:31 ET (whipsaw-prone opening minute), 15:45 ET (closing
# rebalance flow + after-close earnings about to print) and 11:00 ET (deep
# mid-session, normal book) live in three very different liquidity / spread
# regimes and a portfolio manager calibrates conviction differently in each.
# Mid-session lasts from 10:00 ET to ``close - CLOSING_HALF_HOUR_MIN`` (so a
# half-day's CLOSING_HALF_HOUR is 12:30-13:00 ET, not 15:30-16:00). Pre-/
# post-market windows mirror the typical extended-hours bands U.S. brokers
# expose (04:00-09:30 ET and 16:00-20:00 ET); outside those is OVERNIGHT.
_OPENING_BELL_MIN = 30             # first 30 min of the session
_CLOSING_HALF_HOUR_MIN = 30        # last 30 min of the session
_PRE_MARKET_OPEN_MIN = 4 * 60      # 04:00 ET — common ECN pre-market open
_AFTER_HOURS_CLOSE_MIN = 20 * 60   # 20:00 ET — common ECN after-hours close


def is_half_day(d: date) -> bool:
    """True if ``d`` is a known NYSE early-close (1:00 p.m. ET) session."""
    return d in NYSE_HALF_DAYS_2026


def close_minute(d: date) -> int:
    """NYSE regular-session close for ``d`` as minutes since ET midnight:
    13:00 on a known half-day, otherwise the regular 16:00 close. A weekend
    or full holiday has no session — callers gate that separately via
    ``is_market_open``; this only answers 'when does the bell ring'."""
    return _EARLY_CLOSE_MIN if d in NYSE_HALF_DAYS_2026 else _REGULAR_CLOSE_MIN


def market_phase(now: datetime | None = None) -> str:
    """Granular trading-day phase for ``now`` (UTC tz-aware; default real wall
    clock). Returns ONE of:

      * ``"WEEKEND"``          — Saturday/Sunday (no session of any kind)
      * ``"HOLIDAY"``          — known full NYSE holiday
      * ``"PRE_MARKET"``       — trading day, 04:00 ≤ NY-time < 09:30
      * ``"OPENING_BELL"``     — trading day, 09:30 ≤ NY-time < 10:00
      * ``"MID_SESSION"``      — trading day, 10:00 ≤ NY-time < close-30min
      * ``"CLOSING_HALF_HOUR"`` — trading day, close-30min ≤ NY-time < close
      * ``"AFTER_CLOSE"``      — trading day, close ≤ NY-time < 20:00
      * ``"OVERNIGHT"``        — trading day evening/early morning outside
                                  the pre/after windows above (20:00 ≤
                                  NY-time < 04:00 next day)

    Half-day handling is automatic — ``CLOSING_HALF_HOUR`` is 12:30-13:00 ET
    on a known half-day (not 15:30-16:00) because ``close_minute`` resolves
    to 13:00 there, so ``MID_SESSION`` ends at 12:30 ET and ``AFTER_CLOSE``
    begins at 13:00 ET on those days. The same ``is_market_open`` calendar
    drives the trading-day vs holiday/weekend split, so the phase view and
    ``MARKET_OPEN`` can never disagree about whether today is a session day.

    Pure: derived from the NY-tz wall clock and the holiday/half-day sets.
    No I/O. Never raises on a valid tz-aware ``now`` — the timezone math is
    deterministic. Designed for the live decision prompt's header (so Opus
    sees the granular phase, not just a binary open/closed)."""
    now = (now or datetime.now(UTC)).astimezone(NY)
    if now.weekday() >= 5:
        return "WEEKEND"
    if now.date() in NYSE_HOLIDAYS_2026:
        return "HOLIDAY"
    minutes = now.hour * 60 + now.minute
    close_min = close_minute(now.date())
    if minutes < _PRE_MARKET_OPEN_MIN:
        return "OVERNIGHT"
    if minutes < _OPEN_MIN:
        return "PRE_MARKET"
    if minutes < _OPEN_MIN + _OPENING_BELL_MIN:
        return "OPENING_BELL"
    if minutes < close_min - _CLOSING_HALF_HOUR_MIN:
        return "MID_SESSION"
    if minutes < close_min:
        return "CLOSING_HALF_HOUR"
    if minutes < _AFTER_HOURS_CLOSE_MIN:
        return "AFTER_CLOSE"
    return "OVERNIGHT"

_PRICE_CACHE: dict[str, tuple[float, float]] = {}  # ticker -> (price, ts)
_PRICE_TTL = 30.0  # seconds

# Negative cache: symbols that returned no data recently (delisted/invalid like
# METAU/GOOGU, or futures closed on weekends like ES=F). Without this, every
# decision cycle re-requests these from yfinance, failing every time and
# spamming the log. TTL is short (5 min) so legitimately-transient outages
# (e.g. futures over a weekend) recover quickly once data returns.
_DEAD_CACHE: dict[str, float] = {}  # ticker -> ts when marked dead
_DEAD_TTL = 300.0  # seconds

# Option-contract price cache. `get_option_price` fetches the FULL option
# chain over the network on every call — and it is called once per open
# option position on every mark-to-market (`strategy._mark_to_market`, every
# 30-min cycle) plus again from `strategy._execute` when a SELL_CALL/SELL_PUT
# fires against the same contract in the same cycle. Stocks have `_PRICE_CACHE`
# and futures have a 30s bucket cache; the option path had neither, so a book
# holding options re-pulled an unchanged chain every cycle and a bad strike
# (not in the chain — `get_option_price` returns None) re-failed forever, the
# exact "re-request a failing symbol every cycle" spam `_DEAD_CACHE` was added
# to stop for stocks. A short TTL keyed on the contract closes that gap:
# `None` results are cached too (acting as the dead-cache for an off-chain
# strike), so a 30-min cycle now does at most one chain fetch per contract.
# TTL matches `_PRICE_TTL` so an option mark is no staler than a stock mark.
_OPT_PRICE_CACHE: dict[tuple, tuple[float | None, float]] = {}
_OPT_PRICE_TTL = 30.0  # seconds


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


def next_session_close(now: datetime | None = None) -> datetime | None:
    """The next NYSE session close after ``now`` (16:00 ET regular /
    13:00 ET half-day).

    Returns a UTC-aware datetime, or ``None`` when no NYSE close day can
    be found within 14 forward days (defensive; the calendar has no
    >4-day gap). Semantics mirror ``next_session_open``:

      * Mid-session (e.g. 10:00 ET on a weekday) → TODAY's 16:00 close.
      * Pre-open today (e.g. 09:00 ET) → still TODAY's 16:00 close (the
        next session bell of any kind belongs to today's session).
      * At or past today's close (e.g. 16:00 ET exactly) → NEXT trading
        day's close.
      * Weekend / holiday → the next trading day's close.
      * Half-day → 13:00 ET, not 16:00, on the half-day itself.

    Pure: walks the NYSE_HOLIDAYS_2026 / NYSE_HALF_DAYS_2026 / weekday
    calendar from ``now``, no I/O. Companion to ``seconds_until_close``
    — both share the strict ``close_dt > now`` advance rule so a tick
    at exactly the bell always advances to the next session (never
    returns a stale "0s left" against a frozen book)."""
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    now_ny = now_utc.astimezone(NY)
    candidate = now_ny.date()
    for _ in range(14):
        is_trading_day = (
            candidate.weekday() < 5
            and candidate not in NYSE_HOLIDAYS_2026
        )
        if is_trading_day:
            close_min = close_minute(candidate)
            close_dt_ny = datetime(
                candidate.year, candidate.month, candidate.day,
                close_min // 60, close_min % 60, tzinfo=NY,
            )
            if close_dt_ny > now_ny:
                return close_dt_ny.astimezone(UTC)
        candidate = candidate + timedelta(days=1)
    return None


def seconds_until_close(now: datetime | None = None) -> int | None:
    """Integer seconds from ``now`` until the next NYSE session close.

    Returns ``None`` when no close is reachable within 14 forward days
    (mirrors ``next_session_close``'s defensive cap). Always >= 0 —
    a wall-clock step-back that would otherwise yield a negative
    countdown clamps to 0. Round-trips with ``next_session_close``:
    callers that only need the countdown for prompt/Discord rendering
    should use this; callers that need the absolute timestamp should
    use ``next_session_close``."""
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    close_dt = next_session_close(now_utc)
    if close_dt is None:
        return None
    secs = (close_dt - now_utc).total_seconds()
    return int(max(0.0, secs))


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


def previous_session_close(now: datetime | None = None) -> datetime | None:
    """The most recent NYSE session close at or before ``now`` (16:00 ET
    regular / 13:00 ET half-day).

    Returns a UTC-aware datetime, or ``None`` when no NYSE close day can
    be found within 14 backward days (defensive cap; the calendar has no
    >4-day gap). Mirror of ``next_session_close`` walking the opposite
    direction:

      * Mid-session weekday → YESTERDAY's close (today's close is in the
        future and so is not "previous").
      * Past today's close (e.g. 17:00 ET) → TODAY's close (the bell that
        just rang).
      * Weekend / holiday → the LAST trading day's close before now.
      * Half-day → 13:00 ET, not 16:00, on the half-day itself.

    Pure: walks NYSE_HOLIDAYS_2026 / NYSE_HALF_DAYS_2026 / weekday from
    ``now`` backward, no I/O. Used by the closure-window analytics to
    measure "how long has the bell been silent" without the caller
    re-deriving the holiday calendar.
    """
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    now_ny = now_utc.astimezone(NY)
    candidate = now_ny.date()
    for _ in range(14):
        is_trading_day = (
            candidate.weekday() < 5
            and candidate not in NYSE_HOLIDAYS_2026
        )
        if is_trading_day:
            close_min = close_minute(candidate)
            close_dt_ny = datetime(
                candidate.year, candidate.month, candidate.day,
                close_min // 60, close_min % 60, tzinfo=NY,
            )
            if close_dt_ny <= now_ny:
                return close_dt_ny.astimezone(UTC)
        candidate = candidate - timedelta(days=1)
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
            # Switch on the ACTUAL frame shape, not the request size. With
            # group_by="ticker" current yfinance returns a per-ticker
            # MultiIndex *even for a single symbol* — the old `len(missing)==1`
            # branch read a flat `data["Close"]`, which then raised KeyError on
            # every single-ticker bulk fetch and silently degraded to a slow
            # per-ticker get_price() fallback. nlevels keys off the real columns
            # so both a MultiIndex frame (any symbol count) and a hypothetical
            # flat single-ticker frame (older yfinance) resolve correctly.
            multiindex = getattr(getattr(data, "columns", None), "nlevels", 1) > 1
            for t in missing:
                price = None
                try:
                    if multiindex:
                        closes = data[t]["Close"].dropna()
                    else:
                        closes = data["Close"].dropna()
                    if len(closes):
                        price = float(closes.iloc[-1])
                        # Zero / negative is not a real mark (halted symbol,
                        # corrupt yfinance row). Treat it as missing so the
                        # per-ticker ``get_price(t)`` fallback runs and the
                        # mark-to-market path sees ``None`` instead of writing
                        # a $0 mark that immediately reads as a 100% loss in
                        # ``_mark_to_market`` (the single-ticker ``get_price``
                        # path already filters this; the bulk path silently
                        # didn't, so a yfinance hiccup that returned 0 for a
                        # live ticker briefly wiped the position's mark).
                        if price <= 0:
                            price = None
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


def _cached_option_price(key: tuple) -> tuple[bool, float | None]:
    """``(hit, price)`` for an option-contract cache key. ``hit`` is False
    when there is no fresh entry; ``price`` is the cached value (which may
    legitimately be ``None`` — a cached off-chain/unfetchable miss)."""
    rec = _OPT_PRICE_CACHE.get(key)
    if rec is not None and time.time() - rec[1] < _OPT_PRICE_TTL:
        return True, rec[0]
    return False, None


def _store_option_price(key: tuple, price: float | None) -> None:
    _OPT_PRICE_CACHE[key] = (price, time.time())


def get_option_price(ticker: str, expiry: str, strike: float, option_type: str) -> float | None:
    """Mid-of-bid-ask for a specific option contract. Hard-fails if strike not in chain
    (silently substituting the nearest strike would create position/price mismatches).

    Result is cached per ``(ticker, expiry, strike, option_type)`` for
    ``_OPT_PRICE_TTL`` seconds — including a ``None`` miss — so a held option
    is priced with at most one chain fetch per cycle (see ``_OPT_PRICE_CACHE``)."""
    key = (ticker, expiry, strike, option_type)
    hit, cached = _cached_option_price(key)
    if hit:
        return cached
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiry)
        df = chain.calls if option_type == "call" else chain.puts
        row = df[df["strike"] == strike]
        if row.empty:
            print(f"[market] strike {strike} not in {ticker} {expiry} {option_type} chain")
            _store_option_price(key, None)
            return None
        last = float(row["lastPrice"].iloc[0])
        bid = float(row["bid"].iloc[0])
        ask = float(row["ask"].iloc[0])
        if bid > 0 and ask > 0:
            price = round((bid + ask) / 2, 4)
        else:
            price = last if last > 0 else None
        _store_option_price(key, price)
        return price
    except Exception as e:
        # A network/parse fault is NOT cached: unlike an off-chain strike
        # (a stable "no such contract"), a transient yfinance error should be
        # retried on the next call, not pinned None for the whole TTL.
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
