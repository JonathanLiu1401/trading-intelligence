"""`/api/capital-paralysis` stale-while-revalidate cache.

Regression lock for the 2026-05-18 core-hybrid SWR-4 fix. Live probing
(fresh runner, host load avg ~16-23) found `/api/risk`, `/api/benchmark`,
`/api/capital-paralysis` and `/api/decision-health` were the only remaining
high-traffic *pure-store-read* core panels NOT behind `swr_cached`: each ran
the heavy multi-read handler inline with no bounded cold path and hung
>15s (curl → 000) under CPU starvation, while every already-cached endpoint
(`/api/state`, `/api/feed-health`, `/api/source-edge`) served a fast
`{"warming":true}` placeholder. They were wrapped in `@swr_cached(.., 30.0)`
— well under the ≥1800s decision cadence so a ≤30s stale window can never
flip a verdict (the pass-#18 runner-heartbeat precedent).

This mirrors `test_core_state_swr.py`: drive the real Flask view through
`app.test_client()` against a fresh temp Store (no network, no :8090 bind)
and lock the cold→warm contract + the pytest-inert-by-default isolation
that keeps the existing exact-value `test_capital_paralysis.py` tests from
leaking through a module-global cache. capital-paralysis is the cleanest of
the four (pure store reads → one builder, no market calls).
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
    # One open position + a fill so build_capital_paralysis has a real book
    # (deployed, non-NO_DATA) to reason over.
    s.record_trade("NVDA", "BUY", 2, 50.0, reason="momentum")
    s.upsert_position("NVDA", "stock", 2, 50.0)
    s.update_portfolio(cash=920.0, total_value=1020.0, positions=[])
    s.record_equity_point(1020.0, 920.0, 5000.0)
    return s


@pytest.fixture
def swr_client(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = _seed_store(db)

    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="swr-cp-test")
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


class TestCapitalParalysisSwr:
    def test_cold_call_returns_full_shape_with_honesty_keys(self, swr_client):
        client, _s = swr_client
        r = client.get("/api/capital-paralysis")
        assert r.status_code == 200
        j = r.get_json()
        # Real builder payload, not the warming placeholder.
        assert "warming" not in j
        assert j["cash"] == 920.0
        assert j["total_value"] == 1020.0
        assert j["n_positions"] == 1
        assert "state" in j and "headline" in j
        # SWR command-center honesty keys prove the cache is active.
        assert j["cached"] is False
        assert j["cache_age_s"] is not None

    def test_warm_hit_served_from_cache_not_rebuilt(self, swr_client):
        client, s = swr_client
        first = client.get("/api/capital-paralysis").get_json()
        assert first["cached"] is False
        assert first["total_value"] == 1020.0

        # Underlying store changes AFTER the cold build cached. Within the
        # 30s TTL the next poll must serve the OLD payload from cache — the
        # latency win that stops the heavy multi-read build repeating every
        # poll, asserted as behaviour.
        s.update_portfolio(cash=1.0, total_value=99999.0, positions=[])

        second = client.get("/api/capital-paralysis").get_json()
        assert second["cached"] is True
        assert second["total_value"] == 1020.0     # stale, served from cache
        assert second["cash"] == 920.0

    def test_inert_under_pytest_by_default(self, tmp_path, monkeypatch):
        """Without the explicit opt-in the handler runs every call with NO
        honesty keys — what keeps the exact-value test_capital_paralysis.py
        tests isolated from a cross-test module-global cache leak."""
        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        monkeypatch.setattr(d, "_SWR_TEST_FORCE", False)  # default pytest state
        s = _seed_store(db)
        d.app.config["TESTING"] = True
        try:
            with d.app.test_client() as client:
                a = client.get("/api/capital-paralysis").get_json()
                assert "cached" not in a and "cache_age_s" not in a
                assert a["total_value"] == 1020.0
                # A live change is reflected immediately (no caching at all).
                s.update_portfolio(cash=1.0, total_value=42.0, positions=[])
                b = client.get("/api/capital-paralysis").get_json()
                assert b["total_value"] == 42.0
        finally:
            s.close()
