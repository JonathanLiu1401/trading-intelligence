"""End-to-end Flask-client tests for /api/mark-integrity &
/api/decision-context (the test_feed_health_endpoint.py convention — route
wiring, the read-only contract, the never-call-Opus contract, and the SWR
honesty keys, exercised through the real Flask app, not __main__ smoke).
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paper_trader.dashboard as d
import paper_trader.market as market_mod
import paper_trader.signals as signals_mod
import paper_trader.store as store_mod
from paper_trader import strategy
from paper_trader.store import Store


@pytest.fixture
def client_store(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    s.record_trade("NVDA", "BUY", 2, 100.0, reason="momentum")
    s.upsert_position("NVDA", "stock", 2, 100.0)
    d.app.config["TESTING"] = True
    try:
        with d.app.test_client() as client:
            yield client, s
    finally:
        s.close()


def _offline(monkeypatch):
    """Neutralise every network fetch assemble_inputs makes so the route runs
    fully offline and deterministically; make _claude_call EXPLODE so the
    'never invokes the model' contract is proven by a green 200.

    get_prices returns {ticker: None} per *requested* ticker — the realistic
    yfinance-starvation shape (an empty dict would read as 'nothing queried',
    not 'all missing')."""
    monkeypatch.setattr(market_mod, "is_market_open", lambda *a, **k: True)
    monkeypatch.setattr(market_mod, "get_prices",
                        lambda tks: {t: None for t in tks})
    monkeypatch.setattr(market_mod, "get_futures_price", lambda f: None)
    monkeypatch.setattr(market_mod, "benchmark_sp500", lambda *a, **k: 5800.0)
    monkeypatch.setattr(signals_mod, "get_top_signals", lambda *a, **k: [])
    monkeypatch.setattr(signals_mod, "get_urgent_articles", lambda *a, **k: [])
    monkeypatch.setattr(signals_mod, "ticker_sentiments", lambda *a, **k: [])
    monkeypatch.setattr(strategy, "get_quant_signals_live", lambda tks: {})
    monkeypatch.setattr(strategy, "_ml_is_qualified", lambda: (False, "no"))

    def _boom(*a, **k):
        raise AssertionError("_claude_call must NEVER run from the inspector")
    monkeypatch.setattr(strategy, "_claude_call", _boom)


class TestMarkIntegrityEndpoint:
    def test_clean_when_price_resolves_and_is_readonly(self, client_store,
                                                       monkeypatch):
        client, s = client_store
        monkeypatch.setattr(market_mod, "get_prices", lambda tks: {"NVDA": 120.0})
        before = s.get_portfolio()
        r = client.get("/api/mark-integrity")
        assert r.status_code == 200
        j = r.get_json()
        assert "error" not in j, j
        assert j["verdict"] == "CLEAN"
        assert j["n_positions"] == 1 and j["n_stale"] == 0
        # the endpoint must not have written marks/equity (read-only snapshot)
        assert s.get_portfolio() == before

    def test_untrustworthy_when_price_missing(self, client_store, monkeypatch):
        client, _s = client_store
        monkeypatch.setattr(market_mod, "get_prices", lambda tks: {})
        j = client.get("/api/mark-integrity").get_json()
        assert j["n_stale"] == 1
        assert j["stale_value_pct"] == 100.0
        assert j["verdict"] == "UNTRUSTWORTHY"
        assert j["stale_tickers"] == ["NVDA"]


class TestDecisionContextEndpoint:
    def test_reconstructs_prompt_without_calling_opus(self, client_store,
                                                      monkeypatch):
        client, _s = client_store
        _offline(monkeypatch)  # NVDA price missing → stale book
        r = client.get("/api/decision-context")
        assert r.status_code == 200          # green ⇒ _claude_call never ran
        j = r.get_json()
        assert "error" not in j, j
        assert j["claude_invoked"] is False
        assert "paper trading portfolio" in j["prompt"]
        assert "TOP SCORED SIGNALS" in j["prompt"]
        # no signals were fed (offline) → the trader is provably blind
        assert j["feed_state"] == "BLIND"
        assert j["input_summary"]["signal_count"] == 0
        # embedded mark-integrity saw the stale NVDA mark
        assert j["mark_integrity"]["verdict"] == "UNTRUSTWORTHY"
        # pytest-inert SWR by default → no honesty keys (cross-test isolation)
        assert "cached" not in j

    def test_buying_power_block_reaches_reconstructed_prompt(self, client_store,
                                                             monkeypatch):
        """assemble_inputs must build the buying_power / event_calendar
        blocks decide() builds — else the inspector under-reports what Opus
        sees. buying_power.build_buying_power ALWAYS returns a non-empty
        prompt_block (even NO_PRICED_NAMES under full yfinance starvation),
        so a faithful reconstruction MUST surface it. Before the fix,
        assemble_inputs never built it and the endpoint silently dropped it."""
        client, _s = client_store
        _offline(monkeypatch)  # all watch prices None → NO_PRICED_NAMES
        j = client.get("/api/decision-context").get_json()
        assert "error" not in j, j
        assert j["advisory_blocks"].get("buying_power") is True
        assert "BUYING POWER" in j["prompt"]
        # the event_calendar flag key must exist regardless of disk state —
        # a trader auditing the block set must never hit a missing key.
        assert "event_calendar" in j["advisory_blocks"]

    def test_sector_exposure_block_reaches_reconstructed_prompt(
            self, client_store, monkeypatch):
        """assemble_inputs must also build the sector_exposure block
        decide() builds (commit b471188) — else the inspector under-reports
        what Opus sees, exactly like the buying_power gap above.
        sector_exposure.build_sector_exposure ALWAYS returns a non-empty
        prompt_block (even NO_DATA), so a faithful reconstruction MUST
        surface it. Before this pass, assemble_inputs never built it and
        the endpoint silently dropped a 7th advisory block."""
        client, _s = client_store
        _offline(monkeypatch)
        j = client.get("/api/decision-context").get_json()
        assert "error" not in j, j
        assert j["advisory_blocks"].get("sector_exposure") is True
        assert "SECTOR EXPOSURE" in j["prompt"]

    def test_degraded_feed_when_half_prices_missing(self, client_store,
                                                    monkeypatch):
        client, _s = client_store
        _offline(monkeypatch)
        # one signal present → not BLIND; watchlist all-missing → DEGRADED
        monkeypatch.setattr(signals_mod, "get_top_signals",
                            lambda *a, **k: [{"id": "x", "ai_score": 9.0,
                                              "urgency": 2,
                                              "title": "NVDA pops",
                                              "tickers": ["NVDA"]}])
        j = client.get("/api/decision-context").get_json()
        assert j["input_summary"]["signal_count"] == 1
        assert j["feed_state"] == "DEGRADED"


class TestDecisionContextSwr:
    @pytest.fixture
    def swr_client(self, tmp_path, monkeypatch):
        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        s.record_trade("NVDA", "BUY", 2, 100.0)
        s.upsert_position("NVDA", "stock", 2, 100.0)
        pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="swr-dc-test")
        monkeypatch.setattr(d, "_SWR_TEST_FORCE", True)
        monkeypatch.setattr(d, "_SWR_COLD_BUDGET_S", 2.0)
        monkeypatch.setattr(d, "_SWR_STATE", {})
        monkeypatch.setattr(d, "_SWR_EXEC", pool)
        d.app.config["TESTING"] = True
        try:
            with d.app.test_client() as client:
                yield client, s
        finally:
            pool.shutdown(wait=True)
            s.close()

    def test_swr_honesty_keys_when_cache_active(self, swr_client, monkeypatch):
        client, _s = swr_client
        _offline(monkeypatch)
        r = client.get("/api/decision-context")
        assert r.status_code == 200
        j = r.get_json()
        if j.get("warming"):           # cold build still running — poll again
            j = client.get("/api/decision-context").get_json()
        assert "warming" not in j, j
        assert j["cached"] is False
        assert j["cache_age_s"] is not None
        # a warm hit within TTL is served from cache, model still never called
        assert client.get("/api/decision-context").get_json()["cached"] is True
