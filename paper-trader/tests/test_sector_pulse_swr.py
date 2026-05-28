"""`/api/sector-pulse` stale-while-revalidate cache.

Regression lock for the 2026-05-28 core-hybrid SWR fix. Live probing
(runner cycling under load) found `/api/sector-pulse` was the only
remaining sector card NOT behind `swr_cached`: each request fired ~17
synchronous yfinance round-trips via ``get_quant_signals_live`` +
``market.get_prices`` inline with no bounded cold path and hung 8s+
under yfinance throttling, frequently exceeding the browser's 10s
panel timeout (observed live: curl --max-time 8 → 000). It is now
wrapped in `@swr_cached("sector_pulse", 60.0)` so the panel always
loads instantly while a single background build refreshes the snapshot.

Mirrors the structure of ``test_capital_paralysis_swr.py``: drive the
real Flask view through ``app.test_client()`` against monkeypatched
quant/market/news fetchers (no network) and lock the cold→warm
contract + the pytest-inert-by-default isolation that keeps the
existing exact-value sector-pulse tests from leaking through a
module-global cache.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paper_trader.dashboard as d
from paper_trader import market as market_mod
from paper_trader import strategy as strategy_mod


@pytest.fixture
def swr_client(monkeypatch):
    # Stub the three expensive read sources so the test never touches the
    # network — quant cache lookups, market price fetches, and articles DB.
    fake_quant = {
        "NVDA": {
            "RSI": 55.0, "macd_signal": 1.2, "mom_5d": 0.5, "mom_20d": 3.2,
            "vol_ratio": 1.1, "pct_from_52h": -4.0,
        },
        "MU": {
            "RSI": 70.0, "macd_signal": -0.4, "mom_5d": -1.2, "mom_20d": 0.8,
            "vol_ratio": 0.9, "pct_from_52h": -8.0,
        },
    }
    monkeypatch.setattr(strategy_mod, "_QUANT_CACHE",
                        {k: (v, 1.0e12) for k, v in fake_quant.items()})
    monkeypatch.setattr(strategy_mod, "get_quant_signals_live",
                        lambda tickers: None)
    monkeypatch.setattr(market_mod, "get_prices",
                        lambda tickers: {t: 100.0 for t in tickers})
    monkeypatch.setattr(d, "_ticker_news_pulse",
                        lambda tickers, hours=24: {
                            t.upper(): {"n": 1, "urgent": 0,
                                        "top_title": "Headline",
                                        "top_url": "u",
                                        "top_score": 5.0}
                            for t in tickers
                        })

    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="swr-sp-test")
    monkeypatch.setattr(d, "_SWR_TEST_FORCE", True)
    monkeypatch.setattr(d, "_SWR_COLD_BUDGET_S", 1.0)
    monkeypatch.setattr(d, "_SWR_STATE", {})
    monkeypatch.setattr(d, "_SWR_EXEC", pool)
    d.app.config["TESTING"] = True
    try:
        with d.app.test_client() as client:
            yield client
    finally:
        pool.shutdown(wait=True)


class TestSectorPulseSwr:
    def test_cold_call_returns_full_shape_with_honesty_keys(self, swr_client):
        r = swr_client.get("/api/sector-pulse")
        assert r.status_code == 200
        j = r.get_json()
        assert "warming" not in j
        # Real builder payload — tickers list shape preserved.
        assert "tickers" in j and isinstance(j["tickers"], list)
        assert len(j["tickers"]) > 0
        first = j["tickers"][0]
        assert "ticker" in first
        assert "rsi" in first
        # SWR command-center honesty keys prove the cache is active.
        assert j["cached"] is False
        assert j["cache_age_s"] is not None

    def test_warm_hit_served_from_cache_not_rebuilt(self, swr_client, monkeypatch):
        first = swr_client.get("/api/sector-pulse").get_json()
        assert first["cached"] is False
        as_of_first = first["as_of"]

        # Replace the underlying fetcher between calls so the *next* build
        # would produce a different payload. Within the 60s TTL the cached
        # response must be returned anyway — that's the latency win.
        monkeypatch.setattr(d, "_ticker_news_pulse",
                            lambda tickers, hours=24: {
                                t.upper(): {"n": 999, "urgent": 99,
                                            "top_title": "DIFFERENT",
                                            "top_url": "u2",
                                            "top_score": 9.9}
                                for t in tickers
                            })

        second = swr_client.get("/api/sector-pulse").get_json()
        assert second["cached"] is True
        # The cached payload still reflects the FIRST builder's result.
        assert second["as_of"] == as_of_first
        # And the per-ticker news_count_24h is still the original n=1, NOT
        # the DIFFERENT-builder's n=999 — proving we did not rebuild.
        assert second["tickers"][0]["news_count_24h"] == 1

    def test_inert_under_pytest_by_default(self, monkeypatch):
        """Without the explicit opt-in the handler runs every call with NO
        honesty keys — what keeps the existing exact-value sector-pulse
        callers (and any test that monkeypatches yfinance shape) isolated
        from a cross-test module-global cache leak."""
        fake_quant = {
            "NVDA": {"RSI": 55.0, "macd_signal": 1.0, "mom_5d": 0.0,
                     "mom_20d": 0.0, "vol_ratio": 1.0, "pct_from_52h": 0.0},
        }
        monkeypatch.setattr(strategy_mod, "_QUANT_CACHE",
                            {k: (v, 1.0e12) for k, v in fake_quant.items()})
        monkeypatch.setattr(strategy_mod, "get_quant_signals_live",
                            lambda tickers: None)
        monkeypatch.setattr(market_mod, "get_prices",
                            lambda tickers: {t: 100.0 for t in tickers})
        monkeypatch.setattr(d, "_ticker_news_pulse",
                            lambda tickers, hours=24: {})
        monkeypatch.setattr(d, "_SWR_TEST_FORCE", False)
        d.app.config["TESTING"] = True
        with d.app.test_client() as client:
            j = client.get("/api/sector-pulse").get_json()
            assert "cached" not in j
            assert "cache_age_s" not in j
            assert "tickers" in j
