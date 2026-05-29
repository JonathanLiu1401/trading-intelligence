"""``market.get_prices`` must NEVER return 0/negative as a valid mark.

The single-ticker ``get_price`` path already filters (``if price > 0``
check, fallback to last close), but the BULK path used to surface the
raw ``closes.iloc[-1]`` value — and yfinance occasionally hands back
``0.0`` for a halted/illiquid symbol or an empty intraday bar. A ``0``
mark fed into ``strategy._mark_to_market`` registers as a 100 % loss
against ``avg_cost`` (``pl = (0 - avg_cost) * qty * 1``), silently
mis-pricing every held position whose bulk fetch hit that row.

Fix: ``get_prices`` now treats ``<= 0`` as missing and falls back to
the per-ticker path (which itself filters and falls back to today's
last close history; ultimately returns ``None`` so ``_mark_to_market``
marks the position ``stale_mark=True`` at ``avg_cost``).
"""
from __future__ import annotations

import pytest

from paper_trader import market


@pytest.fixture(autouse=True)
def _clean_caches():
    market._PRICE_CACHE.clear()
    market._DEAD_CACHE.clear()
    yield
    market._PRICE_CACHE.clear()
    market._DEAD_CACHE.clear()


class TestGetPricesZeroPriceFilter:
    """The bulk yfinance path treats ``<=0`` as missing — falls through to
    the per-ticker ``get_price(t)`` fallback, which is the canonical
    "price not available" signal feeding ``_mark_to_market``."""

    def test_zero_close_single_falls_back_to_per_ticker(self, monkeypatch):
        import pandas as pd
        # yfinance returns a perfectly valid frame whose ONLY close is 0.0.
        # The fix must reject this as a mark and trigger the per-ticker path.
        df = pd.DataFrame({"Close": [0.0]})
        monkeypatch.setattr(market.yf, "download", lambda *a, **k: df)
        called = {"n": 0}

        def _fallback(t):
            called["n"] += 1
            return 42.5  # the "real" intraday-close fallback

        monkeypatch.setattr(market, "get_price", _fallback)
        out = market.get_prices(["AAA"])
        assert out == {"AAA": 42.5}, (
            "0.0 in the bulk frame must NOT be returned as the mark; "
            "the per-ticker fallback owns 'no quote' semantics."
        )
        assert called["n"] == 1
        # And the rescued price IS cached so the next call hits the cache.
        assert market._cached_price("AAA") == 42.5

    def test_negative_close_falls_back(self, monkeypatch):
        import pandas as pd
        df = pd.DataFrame({"Close": [-3.14]})
        monkeypatch.setattr(market.yf, "download", lambda *a, **k: df)
        monkeypatch.setattr(market, "get_price", lambda t: None)
        # Negative is non-physical; treat exactly like zero.
        assert market.get_prices(["AAA"]) == {"AAA": None}

    def test_zero_in_multiindex_falls_back_only_for_that_ticker(self, monkeypatch):
        import pandas as pd
        cols = pd.MultiIndex.from_tuples([("AAA", "Close"), ("BBB", "Close")])
        # AAA has a legit price; BBB has 0 (halted intraday row).
        df = pd.DataFrame([[10.0, 0.0], [11.0, 0.0]], columns=cols)
        monkeypatch.setattr(market.yf, "download", lambda *a, **k: df)
        fallback_calls: list[str] = []

        def _fallback(t):
            fallback_calls.append(t)
            return 99.0 if t == "BBB" else None  # BBB rescued via per-ticker

        monkeypatch.setattr(market, "get_price", _fallback)
        out = market.get_prices(["AAA", "BBB"])
        assert out["AAA"] == pytest.approx(11.0)
        assert out["BBB"] == pytest.approx(99.0)
        # Only BBB hit the fallback — AAA's legit value never re-fetched.
        assert fallback_calls == ["BBB"]

    def test_zero_does_not_pollute_price_cache(self, monkeypatch):
        """A zero from yfinance must NOT poison the price cache: the next
        call (after the fallback returns None too) should leave the cache
        clean, not record 0.0 as the latest mark."""
        import pandas as pd
        df_zero = pd.DataFrame({"Close": [0.0]})
        monkeypatch.setattr(market.yf, "download", lambda *a, **k: df_zero)
        monkeypatch.setattr(market, "get_price", lambda t: None)
        assert market.get_prices(["AAA"]) == {"AAA": None}
        assert market._cached_price("AAA") is None
