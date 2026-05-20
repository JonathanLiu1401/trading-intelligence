"""`/api/buying-power` endpoint — exposes ``build_buying_power`` (the lean
prompt-facing complement to ``capital_paralysis``, already in the Opus
prompt) on the dashboard.

The endpoint is the operator-side mirror of the same advisory block Opus
already sees in the decision prompt. These tests drive the real Flask view
through ``app.test_client()`` against a fresh temp Store and a mocked
``market.get_prices`` so no yfinance traffic ever happens. The
discriminating asserts:

* the cold call returns the builder's *exact* shape — cash, deployed_pct,
  affordable list with whole-share counts, cheapest name + price, unlock —
  not the SWR warming placeholder;
* affordability is a strict ``int(cash // px)`` floor against the mocked
  prices (so a 999 cash / 500 price reads 1 share, never 2);
* the CASH_CONSTRAINED branch fires when cash is below every priced name's
  price (the live $18.49-of-$972 pathology);
* the unlock fact names the most-underwater open position (the builder's
  biggest-loser-first cut-priority);
* the SWR honesty keys ``cached`` / ``cache_age_s`` are present in
  forced-cache mode and absent in default pytest mode (the
  capital_paralysis SWR isolation precedent — the unit-level
  test_buying_power.py asserts stay stable);
* WATCHLIST is correctly scoped to the FULL universe (an unheld ticker
  with a live price is in the affordable list).
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paper_trader.dashboard as d
from paper_trader import market as _market
from paper_trader import store as store_mod
from paper_trader import strategy as strategy_mod
from paper_trader.store import Store


def _seed_store_pinned(db_path):
    """The CASH_CONSTRAINED live shape: ~$18 free of a ~$972 book."""
    s = Store()
    s.record_trade("MU", "BUY", 0.5, 724.12, reason="seed")
    s.upsert_position("MU", "stock", 0.5, 724.12)
    s.record_trade("LITE", "BUY", 0.61, 980.90, reason="seed")
    s.upsert_position("LITE", "stock", 0.61, 980.90)
    s.update_portfolio(cash=18.49, total_value=972.69, positions=[])
    s.record_equity_point(972.69, 18.49, 5000.0)
    return s


def _seed_store_deployable(db_path):
    """A book with cash that CAN buy whole shares at the watchlist's
    cheapest in-play name."""
    s = Store()
    s.record_trade("NVDA", "BUY", 1, 100.0, reason="seed")
    s.upsert_position("NVDA", "stock", 1, 100.0)
    s.update_portfolio(cash=500.0, total_value=600.0, positions=[])
    s.record_equity_point(600.0, 500.0, 5000.0)
    return s


@pytest.fixture
def swr_client(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)

    # No yfinance — deterministic mocked watch prices. Cover both held
    # names + a couple of unheld watchlist names so the affordable list
    # has both shapes.
    fake_prices = {"MU": 92.0, "LITE": 55.0, "SOXL": 30.0,
                   "SPY": 500.0, "QQQ": 480.0, "NVDA": 220.0,
                   "TQQQ": 73.0, "SMH": 250.0}
    # Restrict WATCHLIST to a small known set so the test is deterministic.
    monkeypatch.setattr(strategy_mod, "WATCHLIST",
                        ["MU", "LITE", "SOXL", "SPY", "QQQ", "NVDA",
                         "TQQQ", "SMH"])

    def _fake_get_prices(tickers):
        return {t: fake_prices.get(t) for t in tickers}
    monkeypatch.setattr(_market, "get_prices", _fake_get_prices)

    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="swr-bp-test")
    monkeypatch.setattr(d, "_SWR_TEST_FORCE", True)
    monkeypatch.setattr(d, "_SWR_COLD_BUDGET_S", 5.0)
    monkeypatch.setattr(d, "_SWR_STATE", {})
    monkeypatch.setattr(d, "_SWR_EXEC", pool)
    d.app.config["TESTING"] = True
    yield monkeypatch, db, pool
    pool.shutdown(wait=True)
    # Singleton store gets recreated in subsequent fixtures via monkeypatch.


class TestBuyingPowerEndpoint:
    def test_cash_constrained_live_pathology_shape(self, swr_client):
        _mp, db, _ = swr_client
        s = _seed_store_pinned(db)
        try:
            with d.app.test_client() as client:
                r = client.get("/api/buying-power")
                assert r.status_code == 200
                j = r.get_json()
                # Real builder payload, not the warming placeholder.
                assert "warming" not in j
                assert j["state"] == "CASH_CONSTRAINED"
                # Cheapest in-play priced name is SOXL @ $30.
                assert j["cheapest_name"] == "SOXL"
                assert j["cheapest_price"] == 30.0
                assert j["cash"] == 18.49
                # Strict int floor: $18.49 // $30 == 0 ⇒ no name affordable.
                assert all(a["whole_shares"] == 0 for a in j["affordable"])
                # Unlock = most-underwater open lot.
                # LITE: 0.61 × (55.0 − 980.90) = −564.80
                # MU:   0.5 × (92.0 − 724.12) = −316.06
                # LITE is more underwater, so it's the unlock candidate.
                assert j["unlock"]["ticker"] == "LITE"
                assert j["unlock"]["unrealized_pl"] < 0
                # Honesty keys present in forced-cache mode.
                assert j["cached"] is False
                assert j["cache_age_s"] is not None
        finally:
            s.close()

    def test_deployable_shape_has_whole_share_counts(self, swr_client):
        _mp, db, _ = swr_client
        s = _seed_store_deployable(db)
        try:
            with d.app.test_client() as client:
                r = client.get("/api/buying-power")
                assert r.status_code == 200
                j = r.get_json()
                assert j["state"] == "DEPLOYABLE"
                assert j["cash"] == 500.0
                # $500 // $30 SOXL = 16; $500 // $500 SPY = 1; $500 // $480
                # QQQ = 1; $500 // $73 TQQQ = 6; $500 // $250 SMH = 2.
                by_t = {a["ticker"]: a["whole_shares"] for a in j["affordable"]}
                assert by_t["SOXL"] == 16
                assert by_t["SPY"] == 1
                assert by_t["QQQ"] == 1
                assert by_t["TQQQ"] == 6
                assert by_t["SMH"] == 2
                # NVDA is held but also in WATCHLIST — affordability is per
                # in-play set so it's still listed: $500 // $220 = 2.
                assert by_t["NVDA"] == 2
                # portfolio_snapshot_readonly remarks open positions to live
                # marks: 1 × $220 NVDA = $220 open_value, total = cash +
                # open_value = 500 + 220 = 720. Deployed % = 220/720 ≈
                # 30.56%. Pinning the exact remarked total locks the
                # readonly-snapshot integration into this endpoint.
                assert round(j["deployed_pct"], 2) == 30.56
        finally:
            s.close()

    def test_int_floor_is_strict_not_rounded(self, swr_client):
        """A 999.99 cash / 500 price ⇒ exactly 1 share, never 2."""
        _mp, db, _ = swr_client
        s = Store()
        s.update_portfolio(cash=999.99, total_value=999.99, positions=[])
        s.record_equity_point(999.99, 999.99, 5000.0)
        try:
            with d.app.test_client() as client:
                r = client.get("/api/buying-power")
                j = r.get_json()
                by_t = {a["ticker"]: a["whole_shares"] for a in j["affordable"]}
                # 999.99 // 500 == 1, not 2.
                assert by_t["SPY"] == 1
                # 999.99 // 480 == 2.
                assert by_t["QQQ"] == 2
        finally:
            s.close()

    def test_warm_hit_served_from_cache(self, swr_client):
        """The second poll within the 60s TTL returns the cached body
        instead of re-running the yfinance bulk fetch."""
        _mp, db, _ = swr_client
        s = _seed_store_deployable(db)
        try:
            with d.app.test_client() as client:
                first = client.get("/api/buying-power").get_json()
                assert first["cached"] is False
                # Mutate the store after the cold build: a cache miss
                # would now reflect this; a hit returns stale data.
                s.update_portfolio(cash=1.0, total_value=1.0, positions=[])
                second = client.get("/api/buying-power").get_json()
                assert second["cached"] is True
                # Stale: serves the original $500 cash, not the new $1.
                assert second["cash"] == 500.0
        finally:
            s.close()

    def test_endpoint_registered_in_swr_prewarm_targets(self):
        """The prewarm == @swr_cached invariant
        (test_swr_prewarm_coverage.py) must include the new endpoint;
        re-assert here so a future change to either the route name or the
        prewarm list trips a focused test before the broader invariant."""
        import inspect
        src = inspect.getsource(d._swr_prewarm)
        assert '("buying-power", buying_power_api)' in src
