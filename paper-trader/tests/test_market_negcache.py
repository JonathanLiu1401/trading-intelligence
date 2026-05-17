"""Negative-cache behaviour for market.get_price / get_prices.

Regression guard: delisted/invalid symbols (METAU, GOOGU) and weekend-closed
futures (ES=F) must NOT be re-requested from yfinance on every decision cycle.
Once a symbol returns no data it is suppressed for _DEAD_TTL seconds; a later
success clears the suppression. All offline — yfinance is monkeypatched.
"""
import time

import pytest

from paper_trader import market


@pytest.fixture(autouse=True)
def _clean_caches():
    market._PRICE_CACHE.clear()
    market._DEAD_CACHE.clear()
    yield
    market._PRICE_CACHE.clear()
    market._DEAD_CACHE.clear()


class _EmptyHist:
    empty = True


class _DeadTicker:
    """yf.Ticker stand-in that has no price data (delisted/closed)."""

    def __init__(self, symbol):
        pass

    @property
    def fast_info(self):
        return {}

    def history(self, *a, **k):
        return _EmptyHist()


def test_dead_symbol_is_not_refetched_within_ttl(monkeypatch):
    calls = {"n": 0}

    def _ticker(sym):
        calls["n"] += 1
        return _DeadTicker(sym)

    monkeypatch.setattr(market.yf, "Ticker", _ticker)

    assert market.get_price("METAU") is None
    assert calls["n"] == 1
    assert market._is_dead("METAU")

    # Subsequent cycles must be served from the negative cache, no new fetch.
    for _ in range(5):
        assert market.get_price("METAU") is None
    assert calls["n"] == 1, "dead symbol was re-fetched despite negative cache"


def test_ttl_expiry_allows_recovery(monkeypatch):
    monkeypatch.setattr(market.yf, "Ticker", lambda s: _DeadTicker(s))
    assert market.get_price("ES=F") is None
    assert market._is_dead("ES=F")

    # Simulate TTL elapsing (futures data returns after the weekend).
    market._DEAD_CACHE["ES=F"] = time.time() - market._DEAD_TTL - 1
    assert not market._is_dead("ES=F")

    class _LiveTicker:
        def __init__(self, s):
            pass

        @property
        def fast_info(self):
            return {"last_price": 5123.5}

    monkeypatch.setattr(market.yf, "Ticker", lambda s: _LiveTicker(s))
    assert market.get_price("ES=F") == 5123.5
    assert not market._is_dead("ES=F"), "successful fetch must clear dead state"


def test_get_prices_skips_dead_in_bulk(monkeypatch):
    market._mark_dead("GOOGU")
    downloaded = {"syms": None}

    def _download(syms, **k):
        downloaded["syms"] = list(syms)
        raise AssertionError("should not bulk-download when only dead symbols remain")

    monkeypatch.setattr(market.yf, "download", _download)

    out = market.get_prices(["GOOGU"])
    assert out == {"GOOGU": None}
    assert downloaded["syms"] is None, "dead symbol leaked into bulk yfinance request"
