"""Tests for paper_trader.market — NYSE session calendar and price helpers.

Session-calendar tests use injected fake "now" timestamps so they are fast
and independent of the actual wall clock. Price helpers are tested with
yfinance mocked so the suite never hits the network.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import market

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _ny(year, month, day, hour, minute):
    """Build a UTC datetime corresponding to a given NY wall-clock time."""
    return datetime(year, month, day, hour, minute, tzinfo=NY).astimezone(UTC)


class TestIsMarketOpen:
    def test_weekend_saturday_returns_false(self):
        # 2026-05-16 is a Saturday.
        assert market.is_market_open(_ny(2026, 5, 16, 10, 0)) is False

    def test_weekend_sunday_returns_false(self):
        assert market.is_market_open(_ny(2026, 5, 17, 10, 0)) is False

    def test_pre_open_929_returns_false(self):
        # 2026-05-14 Thursday, 9:29 AM ET is still pre-open.
        assert market.is_market_open(_ny(2026, 5, 14, 9, 29)) is False

    def test_after_close_4pm_returns_false(self):
        # The window is half-open [9:30, 16:00), so 16:00 is the close.
        assert market.is_market_open(_ny(2026, 5, 14, 16, 0)) is False

    def test_after_close_401pm_returns_false(self):
        assert market.is_market_open(_ny(2026, 5, 14, 16, 1)) is False

    def test_weekday_10am_returns_true(self):
        assert market.is_market_open(_ny(2026, 5, 14, 10, 0)) is True

    def test_weekday_exactly_930_returns_true(self):
        # Lower bound is inclusive.
        assert market.is_market_open(_ny(2026, 5, 14, 9, 30)) is True

    def test_weekday_1559_returns_true(self):
        # Upper bound is exclusive — one minute before close is still open.
        assert market.is_market_open(_ny(2026, 5, 14, 15, 59)) is True

    def test_thanksgiving_returns_false(self):
        # 2026-11-26 is Thanksgiving; even mid-day the market is closed.
        assert market.is_market_open(_ny(2026, 11, 26, 10, 0)) is False

    def test_new_years_day_returns_false(self):
        assert market.is_market_open(_ny(2026, 1, 1, 10, 0)) is False

    def test_good_friday_returns_false(self):
        # 2026-04-03 is Good Friday.
        assert market.is_market_open(_ny(2026, 4, 3, 10, 0)) is False


class TestPriceCache:
    def setup_method(self):
        # The module-level cache leaks between tests; clear before each.
        market._PRICE_CACHE.clear()

    def test_cached_price_returns_cached_value(self):
        market._store_price("NVDA", 500.0)
        assert market._cached_price("NVDA") == 500.0

    def test_cached_price_missing_returns_none(self):
        assert market._cached_price("ABSENT") is None

    def test_cache_expires_after_ttl(self, monkeypatch):
        market._store_price("NVDA", 500.0)
        # Move the module's view of time forward beyond TTL.
        import time as _t
        real = _t.time()
        monkeypatch.setattr(market.time, "time", lambda: real + market._PRICE_TTL + 1)
        assert market._cached_price("NVDA") is None


class TestGetPriceMocked:
    def setup_method(self):
        market._PRICE_CACHE.clear()

    def test_fast_info_path_returns_price_and_caches(self, monkeypatch):
        fake_ticker = MagicMock()
        fake_ticker.fast_info = {"last_price": 123.45, "regular_market_price": 0}
        monkeypatch.setattr(market.yf, "Ticker", lambda t: fake_ticker)
        assert market.get_price("FAKE") == 123.45
        assert market._cached_price("FAKE") == 123.45

    def test_zero_fast_info_falls_back_to_history(self, monkeypatch):
        import pandas as pd
        fake_ticker = MagicMock()
        fake_ticker.fast_info = {"last_price": 0, "regular_market_price": 0}
        fake_ticker.history.return_value = pd.DataFrame({"Close": [99.5]})
        monkeypatch.setattr(market.yf, "Ticker", lambda t: fake_ticker)
        assert market.get_price("FAKE") == pytest.approx(99.5)

    def test_yfinance_exception_returns_none(self, monkeypatch):
        def raise_(_t):
            raise RuntimeError("network down")
        monkeypatch.setattr(market.yf, "Ticker", raise_)
        assert market.get_price("FAKE") is None

    def test_get_prices_empty_returns_empty(self):
        assert market.get_prices([]) == {}

    def test_get_prices_uses_cache(self):
        market._store_price("AAA", 10.0)
        market._store_price("BBB", 20.0)
        out = market.get_prices(["AAA", "BBB"])
        assert out == {"AAA": 10.0, "BBB": 20.0}


class TestGetOptionPrice:
    def test_strike_not_in_chain_returns_none(self, monkeypatch):
        import pandas as pd
        # Build a fake chain that does NOT contain strike=999.
        chain = MagicMock()
        chain.calls = pd.DataFrame([{"strike": 100.0, "lastPrice": 5.0, "bid": 4.5, "ask": 5.5}])
        chain.puts = pd.DataFrame([{"strike": 100.0, "lastPrice": 1.0, "bid": 0.5, "ask": 1.5}])
        fake_ticker = MagicMock()
        fake_ticker.option_chain.return_value = chain
        monkeypatch.setattr(market.yf, "Ticker", lambda t: fake_ticker)
        assert market.get_option_price("FAKE", "2026-12-19", 999.0, "call") is None

    def test_mid_of_bid_ask_when_both_positive(self, monkeypatch):
        import pandas as pd
        chain = MagicMock()
        chain.calls = pd.DataFrame([{"strike": 100.0, "lastPrice": 5.0, "bid": 4.0, "ask": 6.0}])
        chain.puts = pd.DataFrame([{"strike": 100.0, "lastPrice": 1.0, "bid": 0.0, "ask": 0.0}])
        fake_ticker = MagicMock()
        fake_ticker.option_chain.return_value = chain
        monkeypatch.setattr(market.yf, "Ticker", lambda t: fake_ticker)
        # Mid = (4+6)/2 = 5.0
        assert market.get_option_price("FAKE", "2026-12-19", 100.0, "call") == 5.0

    def test_falls_back_to_last_when_bid_ask_zero(self, monkeypatch):
        import pandas as pd
        chain = MagicMock()
        chain.calls = pd.DataFrame([{"strike": 100.0, "lastPrice": 7.5, "bid": 0.0, "ask": 0.0}])
        chain.puts = pd.DataFrame([{"strike": 100.0, "lastPrice": 1.0, "bid": 0.0, "ask": 0.0}])
        fake_ticker = MagicMock()
        fake_ticker.option_chain.return_value = chain
        monkeypatch.setattr(market.yf, "Ticker", lambda t: fake_ticker)
        assert market.get_option_price("FAKE", "2026-12-19", 100.0, "call") == 7.5


class TestGetFuturesPriceBucketCache:
    """Locks the 30s time-bucket lru_cache in get_futures_price.

    get_futures_price(sym) -> get_futures_price_cached(sym, int(time()//30)).
    Within one 30s bucket repeated calls must NOT re-hit get_price; once the
    bucket advances they must. The cache must also key on the symbol, not the
    bucket alone (otherwise two futures would alias to one price). lru_cache is
    module-global, so each test clears it first for isolation.
    """

    def test_same_30s_bucket_serves_cached_value(self, monkeypatch):
        from paper_trader import market

        market.get_futures_price_cached.cache_clear()
        calls = []

        def fake_get_price(sym):
            calls.append(sym)
            return 100.0 + len(calls)  # changes every real fetch

        monkeypatch.setattr(market, "get_price", fake_get_price)
        # Freeze wall clock inside the same 30s bucket (t=10 -> 10//30 == 0).
        monkeypatch.setattr(market.time, "time", lambda: 10.0)

        first = market.get_futures_price("ES=F")
        second = market.get_futures_price("ES=F")
        third = market.get_futures_price("ES=F")

        assert first == 101.0
        assert second == first and third == first  # cached, not re-fetched
        assert calls == ["ES=F"]  # get_price hit exactly once

    def test_bucket_advance_triggers_refetch(self, monkeypatch):
        from paper_trader import market

        market.get_futures_price_cached.cache_clear()
        calls = []

        def fake_get_price(sym):
            calls.append(sym)
            return float(len(calls))

        monkeypatch.setattr(market, "get_price", fake_get_price)

        now = {"t": 0.0}
        monkeypatch.setattr(market.time, "time", lambda: now["t"])

        now["t"] = 5.0          # bucket 0
        a = market.get_futures_price("NQ=F")
        now["t"] = 29.999       # still bucket 0
        b = market.get_futures_price("NQ=F")
        now["t"] = 30.0         # bucket 1 -> new lru key -> refetch
        c = market.get_futures_price("NQ=F")

        assert a == 1.0
        assert b == 1.0          # same bucket: cached
        assert c == 2.0          # bucket advanced: fresh fetch
        assert calls == ["NQ=F", "NQ=F"]

    def test_distinct_symbols_keyed_independently(self, monkeypatch):
        from paper_trader import market

        market.get_futures_price_cached.cache_clear()
        prices = {"ES=F": 5000.0, "CL=F": 70.0}
        monkeypatch.setattr(market, "get_price", lambda s: prices[s])
        monkeypatch.setattr(market.time, "time", lambda: 0.0)

        # Same bucket, different symbols must not alias to one cached value.
        assert market.get_futures_price("ES=F") == 5000.0
        assert market.get_futures_price("CL=F") == 70.0
        assert market.get_futures_price("ES=F") == 5000.0


class TestGetPricesBulk:
    """Locks the `yf.download` bulk branch of get_prices — previously the only
    coverage was the empty-list and full-cache short-circuits, so the actual
    DataFrame-shape handling had ZERO direct tests.

    yfinance returns two STRUCTURALLY DIFFERENT frames depending on how many
    symbols are requested with group_by='ticker':
      * exactly one symbol  -> flat columns, code reads ``data["Close"]``
      * two or more symbols -> a per-ticker MultiIndex, code reads
        ``data[t]["Close"]``
    The `len(missing) == 1` branch is the load-bearing switch between them. A
    refactor that drops the switch (or swaps the branches) would silently
    return None for every multi-symbol fetch — these tests fail loudly on that.
    """

    def setup_method(self):
        market._PRICE_CACHE.clear()

    def _no_download(self, *a, **k):
        raise AssertionError("yf.download must not be called")

    def test_single_missing_uses_flat_close_column(self, monkeypatch):
        import pandas as pd
        # One uncached symbol -> single-ticker flat-columns frame.
        df = pd.DataFrame({"Open": [98.0, 99.0], "Close": [99.0, 100.5]})
        monkeypatch.setattr(market.yf, "download", lambda *a, **k: df)
        out = market.get_prices(["AAA"])
        # Latest non-NaN Close, and it must be cached for the next call.
        assert out == {"AAA": pytest.approx(100.5)}
        assert market._cached_price("AAA") == pytest.approx(100.5)

    def test_multi_missing_uses_per_ticker_multiindex(self, monkeypatch):
        import pandas as pd
        cols = pd.MultiIndex.from_tuples(
            [("AAA", "Close"), ("BBB", "Close")]
        )
        df = pd.DataFrame([[10.0, 20.0], [11.0, 21.0]], columns=cols)
        monkeypatch.setattr(market.yf, "download", lambda *a, **k: df)
        out = market.get_prices(["AAA", "BBB"])
        # Each ticker resolves through its OWN sub-frame, not a shared column.
        assert out == {"AAA": pytest.approx(11.0), "BBB": pytest.approx(21.0)}

    def test_all_nan_close_falls_back_to_get_price(self, monkeypatch):
        import pandas as pd
        # dropna() empties the series -> len 0 -> per-ticker get_price fallback.
        df = pd.DataFrame({"Close": [float("nan"), float("nan")]})
        monkeypatch.setattr(market.yf, "download", lambda *a, **k: df)
        monkeypatch.setattr(market, "get_price", lambda t: 42.0)
        assert market.get_prices(["AAA"]) == {"AAA": 42.0}

    def test_missing_ticker_column_falls_back_per_ticker(self, monkeypatch):
        import pandas as pd
        # Multi request but BBB's column is absent -> data["BBB"] raises
        # KeyError -> inner except -> get_price fallback for BBB only.
        cols = pd.MultiIndex.from_tuples([("AAA", "Close")])
        df = pd.DataFrame([[10.0], [12.5]], columns=cols)
        monkeypatch.setattr(market.yf, "download", lambda *a, **k: df)
        monkeypatch.setattr(market, "get_price", lambda t: 7.0 if t == "BBB" else None)
        out = market.get_prices(["AAA", "BBB"])
        assert out["AAA"] == pytest.approx(12.5)
        assert out["BBB"] == 7.0  # fell back to single fetch

    def test_unresolvable_ticker_yields_none(self, monkeypatch):
        import pandas as pd
        df = pd.DataFrame({"Close": [float("nan")]})
        monkeypatch.setattr(market.yf, "download", lambda *a, **k: df)
        monkeypatch.setattr(market, "get_price", lambda t: None)
        # Neither bulk nor single fetch produced a price: key present, value None.
        assert market.get_prices(["ZZZ"]) == {"ZZZ": None}

    def test_whole_download_exception_falls_back_to_get_price(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("yfinance bulk endpoint down")
        monkeypatch.setattr(market.yf, "download", boom)
        prices = {"AAA": 1.0, "BBB": 2.0}
        monkeypatch.setattr(market, "get_price", lambda t: prices[t])
        assert market.get_prices(["AAA", "BBB"]) == {"AAA": 1.0, "BBB": 2.0}

    def test_partial_cache_only_fetches_the_missing_symbol(self, monkeypatch):
        import pandas as pd
        market._store_price("AAA", 50.0)  # already cached
        # Only BBB is missing -> len(missing)==1 -> flat-column branch.
        df = pd.DataFrame({"Close": [60.0]})
        called = {"n": 0}

        def fake_download(syms, *a, **k):
            called["n"] += 1
            assert list(syms) == ["BBB"], f"only the uncached symbol is fetched, got {syms}"
            return df

        monkeypatch.setattr(market.yf, "download", fake_download)
        out = market.get_prices(["AAA", "BBB"])
        assert out == {"AAA": 50.0, "BBB": pytest.approx(60.0)}
        assert called["n"] == 1

    def test_full_cache_never_calls_download(self, monkeypatch):
        market._store_price("AAA", 5.0)
        market._store_price("BBB", 6.0)
        monkeypatch.setattr(market.yf, "download", self._no_download)
        assert market.get_prices(["AAA", "BBB"]) == {"AAA": 5.0, "BBB": 6.0}


class TestGetOptionsChain:
    """Locks the nearest-DTE expiry selection in get_options_chain — zero prior
    direct coverage. The contract is: of all listed expiries, pick the one
    whose distance to ``today + target_dte`` days is smallest (NOT the first
    listed), cap each side at 30 rows, and degrade to None (never raise) when
    there are no expiries or yfinance errors.
    """

    def _chain_df(self, n_rows: int):
        import pandas as pd
        cols = ["strike", "lastPrice", "bid", "ask", "volume",
                "openInterest", "impliedVolatility"]
        return pd.DataFrame(
            [[100.0 + i, 1.0, 0.9, 1.1, 10, 5, 0.5] for i in range(n_rows)],
            columns=cols,
        )

    def test_picks_expiry_nearest_target_dte(self, monkeypatch):
        from datetime import date, timedelta
        today = date.today()
        near = (today + timedelta(days=5)).isoformat()    # |5 - 14|  = 9
        far = (today + timedelta(days=40)).isoformat()     # |40 - 14| = 26
        fake_ticker = MagicMock()
        fake_ticker.options = (far, near)  # near is NOT first — must still win
        chain = MagicMock()
        chain.calls = self._chain_df(3)
        chain.puts = self._chain_df(3)
        fake_ticker.option_chain.return_value = chain
        monkeypatch.setattr(market.yf, "Ticker", lambda t: fake_ticker)

        out = market.get_options_chain("FAKE", target_dte=14)
        assert out is not None
        assert out["expiry"] == near
        fake_ticker.option_chain.assert_called_once_with(near)
        assert out["ticker"] == "FAKE"
        assert out["calls"][0]["strike"] == 100.0

    def test_head_caps_each_side_at_30(self, monkeypatch):
        from datetime import date, timedelta
        exp = (date.today() + timedelta(days=14)).isoformat()
        fake_ticker = MagicMock()
        fake_ticker.options = (exp,)
        chain = MagicMock()
        chain.calls = self._chain_df(35)
        chain.puts = self._chain_df(35)
        fake_ticker.option_chain.return_value = chain
        monkeypatch.setattr(market.yf, "Ticker", lambda t: fake_ticker)

        out = market.get_options_chain("FAKE")
        assert len(out["calls"]) == 30
        assert len(out["puts"]) == 30

    def test_no_expiries_returns_none(self, monkeypatch):
        fake_ticker = MagicMock()
        fake_ticker.options = ()
        monkeypatch.setattr(market.yf, "Ticker", lambda t: fake_ticker)
        assert market.get_options_chain("FAKE") is None

    def test_yfinance_exception_returns_none(self, monkeypatch):
        def raise_(_t):
            raise RuntimeError("network down")
        monkeypatch.setattr(market.yf, "Ticker", raise_)
        assert market.get_options_chain("FAKE") is None
