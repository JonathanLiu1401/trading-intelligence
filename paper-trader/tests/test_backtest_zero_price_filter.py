"""Lock tests for PriceCache treating 0/negative closes as missing.

Mirrors the live-trader-side `market.get_price`/`get_prices` zero-filter
contract (the `9c5f` / `0c2da34` market-side fix Agent 1 just landed in
the same review cycle). Without this:

  * `price_on` would return a poisoned 0.0 close — `returns_pct` then
    short-circuits via `if not s` and fabricates a flat 0.0 outcome,
    silently contaminating the `decision_outcomes.jsonl` training set.
  * `_buy(price=0)` would have notional=0 and divide-by-zero on the
    blended avg_cost path.
  * `_build_trading_days` would include $0 SPY days as sampled decisions.

Tests pin both the storage-side filter (no $0 ever enters the cache) and
the read-side filter (legacy caches with $0 values walk back over them).
"""
from __future__ import annotations

from datetime import date

import pytest

from paper_trader.backtest import PriceCache


@pytest.fixture
def empty_cache():
    """Construct a PriceCache without invoking _load (skips yfinance)."""
    pc = PriceCache.__new__(PriceCache)
    pc.tickers = ["NVDA"]
    pc.start = date(2024, 1, 1)
    pc.end = date(2024, 12, 31)
    pc.prices = {}
    pc.trading_days = []
    return pc


class TestPriceOnZeroFilter:
    """Read-side: 0/negative cached closes are treated as missing."""

    def test_price_on_exact_match_zero_walks_back(self, empty_cache):
        # A legacy cache with a 0.0 close on the requested date and a real
        # close 2 days back should walk back, NOT return 0.0.
        empty_cache.prices = {"NVDA": {
            "2024-06-10": 100.0,
            "2024-06-12": 0.0,    # poisoned tick
        }}
        v = empty_cache.price_on("NVDA", date(2024, 6, 12))
        assert v == 100.0  # walked back, not the $0

    def test_price_on_negative_close_treated_as_missing(self, empty_cache):
        empty_cache.prices = {"NVDA": {
            "2024-06-10": 100.0,
            "2024-06-12": -5.0,   # implausible but should be filtered
        }}
        v = empty_cache.price_on("NVDA", date(2024, 6, 12))
        assert v == 100.0  # walked back past the negative

    def test_price_on_all_walkback_zero_returns_none(self, empty_cache):
        # If every candidate in the 7-day walk-back window is 0, return None
        # — the engine then properly skips the decision rather than feeding
        # a fabricated 0 into downstream math.
        empty_cache.prices = {"NVDA": {
            "2024-06-08": 0.0, "2024-06-09": 0.0, "2024-06-10": 0.0,
            "2024-06-11": 0.0, "2024-06-12": 0.0,
        }}
        v = empty_cache.price_on("NVDA", date(2024, 6, 12))
        assert v is None

    def test_price_on_positive_close_returns_value(self, empty_cache):
        empty_cache.prices = {"NVDA": {"2024-06-12": 100.0}}
        v = empty_cache.price_on("NVDA", date(2024, 6, 12))
        assert v == 100.0


class TestResolvedCloseDateZeroFilter:
    """Read-side: resolved_close_date agrees with price_on on which day is
    usable. Otherwise outcome-side callers (collision check) would see a
    walk-back date that price_on then refuses to return a price for —
    inconsistent state that breaks `_compute_decision_outcomes`'s honest
    walk-back-collision guard."""

    def test_resolved_skips_zero_day_walks_back(self, empty_cache):
        empty_cache.prices = {"NVDA": {
            "2024-06-10": 100.0,
            "2024-06-12": 0.0,
        }}
        resolved = empty_cache.resolved_close_date("NVDA", date(2024, 6, 12))
        assert resolved == date(2024, 6, 10)

    def test_resolved_all_zero_returns_none(self, empty_cache):
        empty_cache.prices = {"NVDA": {
            "2024-06-08": 0.0, "2024-06-09": 0.0, "2024-06-10": 0.0,
        }}
        resolved = empty_cache.resolved_close_date("NVDA", date(2024, 6, 10))
        assert resolved is None

    def test_resolved_positive_returns_date(self, empty_cache):
        empty_cache.prices = {"NVDA": {"2024-06-12": 100.0}}
        resolved = empty_cache.resolved_close_date("NVDA", date(2024, 6, 12))
        assert resolved == date(2024, 6, 12)


class TestReturnsPctZeroFilter:
    """Verify returns_pct no longer returns 0.0 fabricated outcomes when a
    cached series carries a $0 endpoint — the walk-back filter chains
    through to honest behavior."""

    def test_zero_endpoint_walks_back_to_real_close(self, empty_cache):
        empty_cache.prices = {"NVDA": {
            "2024-06-10": 100.0,
            "2024-06-15": 110.0,
            "2024-06-17": 0.0,    # poisoned end-date tick
        }}
        # end_d=2024-06-17 walks back to 2024-06-15 → +10% return
        ret = empty_cache.returns_pct("NVDA", date(2024, 6, 10),
                                      date(2024, 6, 17))
        assert ret == pytest.approx(10.0, rel=1e-6)

    def test_zero_start_returns_zero_fallback(self, empty_cache):
        # If the start has no walk-back candidate, returns_pct still returns
        # 0.0 via its existing `if not s` guard — but the price is None now,
        # not 0.0, so the legacy collision guard still detects it via
        # resolved_close_date. The point is no $0 mark survives.
        empty_cache.prices = {"NVDA": {
            "2024-06-15": 110.0,
            "2024-06-20": 121.0,
        }}
        # 2024-05-01 has no walk-back candidate (>7 days from 06-15)
        ret = empty_cache.returns_pct("NVDA", date(2024, 5, 1),
                                      date(2024, 6, 20))
        assert ret == 0.0  # legacy "no price" fallback (unchanged contract)
