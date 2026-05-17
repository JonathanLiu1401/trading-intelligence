"""`/api/state` stale-while-revalidate cache + the main-page `refresh()` guard.

`/api/state` is the trader page's lifeline (polled every 15s, cross-fetched,
~145KB of eq-5000 + 500 trades over six lock-held Store reads). Live-user
testing on 2026-05-17 measured it at 8.7s under concurrent load because it
was the only high-traffic core endpoint NOT behind `swr_cached` while every
slow network endpoint already was. This locks:

  * the SWR behaviour on the real `/api/state` view (cold build → warm hit
    served from cache without re-reading the store; honesty keys injected);
  * the documented "inert under pytest unless forced" contract, so the
    existing exact-value `/api/state`-shaped tests stay isolated;
  * the `refresh()` JS guard — the Phase 1 fix that stops the main page
    freezing on the SWR cold `{"warming":true}` placeholder (or any
    transient `/api/state` error body, which has 500'd in prod).

These drive the real Flask view through `app.test_client()` against a fresh
temp `Store` — no network, no real :8090 bind.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import store as store_mod
from paper_trader.store import Store
import paper_trader.dashboard as d


def _seed_store(db_path):
    s = Store()
    # One closed round-trip + one open position so the payload has real
    # positions/trades/equity to render.
    s.record_trade("AAPL", "BUY", 10, 10.0, reason="entry thesis")
    s.record_trade("AAPL", "SELL", 10, 12.0, reason="took profit")
    s.record_trade("NVDA", "BUY", 2, 50.0, reason="momentum")
    s.upsert_position("NVDA", "stock", 2, 50.0)
    s.update_portfolio(cash=920.0, total_value=1020.0, positions=[])
    s.record_equity_point(1020.0, 920.0, 5000.0)
    return s


@pytest.fixture
def swr_client(tmp_path, monkeypatch):
    """Fresh temp Store + Flask client with the SWR cache *opted in*
    (it is deliberately pytest-inert otherwise) and an isolated state map /
    executor / short cold budget."""
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = _seed_store(db)

    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="swr-state-test")
    monkeypatch.setattr(d, "_SWR_TEST_FORCE", True)
    monkeypatch.setattr(d, "_SWR_COLD_BUDGET_S", 1.0)
    monkeypatch.setattr(d, "_SWR_STATE", {})
    monkeypatch.setattr(d, "_SWR_EXEC", pool)
    d.app.config["TESTING"] = True
    try:
        with d.app.test_client() as client:
            yield client, s
    finally:
        pool.shutdown(wait=True)
        s.close()


class TestStateSwr:
    def test_cold_call_returns_full_shape_with_honesty_keys(self, swr_client):
        client, _s = swr_client
        r = client.get("/api/state")
        assert r.status_code == 200
        j = r.get_json()
        # Real payload, not the warming placeholder.
        assert "warming" not in j
        assert j["portfolio"]["cash"] == 920.0
        assert j["portfolio"]["total_value"] == 1020.0
        assert [p["ticker"] for p in j["positions"]] == ["NVDA"]
        assert {t["ticker"] for t in j["trades"]} == {"AAPL", "NVDA"}
        assert len(j["equity"]) == 1 and j["equity"][0]["total_value"] == 1020.0
        assert j["sp500"] == 5000.0
        # SWR command-center honesty keys (proves the cache is active).
        assert j["cached"] is False
        assert j["cache_age_s"] is not None

    def test_warm_hit_served_from_cache_not_rebuilt(self, swr_client):
        client, s = swr_client
        first = client.get("/api/state").get_json()
        assert first["cached"] is False
        assert first["portfolio"]["total_value"] == 1020.0

        # Underlying store changes *after* the cold build populated the cache.
        # Within the 15s TTL the next poll must serve the OLD payload from
        # cache (the whole point: the heavy six-read build is not repeated on
        # every 15s poll). This is the latency win, asserted as behaviour.
        s.update_portfolio(cash=1.0, total_value=99999.0, positions=[])
        s.record_trade("ZZZZ", "BUY", 1, 1.0)

        second = client.get("/api/state").get_json()
        assert second["cached"] is True
        assert second["portfolio"]["total_value"] == 1020.0   # stale, cached
        assert {t["ticker"] for t in second["trades"]} == {"AAPL", "NVDA"}
        assert "ZZZZ" not in {t["ticker"] for t in second["trades"]}

    def test_inert_under_pytest_by_default(self, tmp_path, monkeypatch):
        """Without the explicit opt-in, `/api/state` calls the handler every
        time with NO honesty keys — this is what keeps the other
        `/api/state`-shaped exact-value tests isolated from a cross-test
        module-global cache leak."""
        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        monkeypatch.setattr(d, "_SWR_TEST_FORCE", False)  # default pytest state
        s = _seed_store(db)
        d.app.config["TESTING"] = True
        try:
            with d.app.test_client() as client:
                a = client.get("/api/state").get_json()
                assert "cached" not in a and "cache_age_s" not in a
                assert a["portfolio"]["total_value"] == 1020.0
                # A live change is reflected immediately (no caching at all).
                s.update_portfolio(cash=1.0, total_value=42.0, positions=[])
                b = client.get("/api/state").get_json()
                assert b["portfolio"]["total_value"] == 42.0
        finally:
            s.close()


class TestRefreshGuard:
    """Static regression lock on the main-page `refresh()` JS (the Phase 1
    fix). Mirrors `test_core_dashboard_helpers.TestTemplateIdsUnique`'s
    static-template discipline — no browser needed to prove the guard exists.
    """

    def _refresh_body(self) -> str:
        tpl = d.TEMPLATE
        i = tpl.index("async function refresh()")
        # Bound the slice to the start of the next top-level async function so
        # we only assert against refresh()'s own body.
        j = tpl.index("async function ", i + 10)
        raw = tpl[i:j]
        # Drop full-line `//` comments so the ordering assertions reason about
        # *executable* code, not the explanatory comment (which itself quotes
        # `r.portfolio.total_value` to describe the old bug).
        return "\n".join(
            ln for ln in raw.splitlines() if not ln.lstrip().startswith("//")
        )

    def test_state_fetch_is_wrapped_in_try_catch(self):
        body = self._refresh_body()
        assert "try {" in body and "catch" in body, (
            "refresh() must wrap the /api/state fetch so a non-JSON 500 "
            "(it has 500'd in prod) can't throw an unhandled rejection")

    def test_warming_and_error_body_short_circuit_before_deref(self):
        body = self._refresh_body()
        guard = body.index("!r.portfolio")
        deref = body.index("r.portfolio.total_value")
        # The guard must come BEFORE the first unguarded dereference, or a
        # warming/error body still throws and freezes the page.
        assert guard < deref
        assert "r.warming" in body and "r.error" in body
        # The guard returns early instead of throwing.
        assert "updating…" in body
