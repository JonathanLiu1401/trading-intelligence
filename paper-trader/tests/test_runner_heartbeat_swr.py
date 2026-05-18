"""`/api/runner-heartbeat` stale-while-revalidate cache (2026-05-18).

The heartbeat is the surface a trader checks *first* and the dashboard JS
polls it every 60s, yet it was the last high-traffic core endpoint NOT
behind `swr_cached` (the invariant #7 gap `/api/state` closed). It was
measured at 9.45s under load avg 23 (the documented host-load storm) vs
~1ms warm — a pure DB + module-global read with no network, so the latency
is pure CPU starvation, exactly what SWR absorbs. The runner cadence is
≥1800s/3600s with ≥1.25x/2x verdict multipliers and the IDLE_STORM efficacy
verdict needs ≥5 cycles × ≥1800s, so a ≤20s stale window can never flip the
verdict — the staleness is invisible while the trader gets an instant
answer instead of a 9s block.

This mirrors `tests/test_core_state_swr.py`: it drives the real Flask view
through `app.test_client()` against a fresh temp `Store`, opting the
pytest-inert SWR cache in via `_SWR_TEST_FORCE`. No network, no real :8090
bind.
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


def _seed_idle_storm(db_path):
    """A real Store with an 18-deep NO_DECISION storm + a fresh newest
    timestamp → liveness HEALTHY, decision_efficacy IDLE_STORM,
    restart_recommended True (a deterministic, discriminating verdict)."""
    s = Store()
    for _ in range(18):
        s.record_decision(False, 0, "NO_DECISION", "claude timeout",
                           1000.0, 1000.0)
    return s


@pytest.fixture
def swr_client(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = _seed_idle_storm(db)

    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="swr-hb-test")
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


class TestRunnerHeartbeatSwr:
    def test_cold_call_returns_full_verdict_with_honesty_keys(self,
                                                              swr_client):
        client, _s = swr_client
        r = client.get("/api/runner-heartbeat")
        assert r.status_code == 200
        j = r.get_json()
        assert "warming" not in j  # real payload, not the cold placeholder
        # The real heartbeat verdict shape is present.
        assert j["verdict"] == "HEALTHY"          # ts ≈ now → cadence OK
        assert j["restart_recommended"] is True   # but the efficacy storm
        assert j["decision_efficacy"]["verdict"] == "IDLE_STORM"
        assert j["decision_efficacy"]["consecutive_no_decision"] == 18
        # SWR command-center honesty keys prove the cache is active.
        assert j["cached"] is False
        assert j["cache_age_s"] is not None

    def test_warm_hit_served_from_cache_not_rebuilt(self, swr_client):
        client, s = swr_client
        first = client.get("/api/runner-heartbeat").get_json()
        assert first["cached"] is False
        assert first["restart_recommended"] is True
        assert first["decision_efficacy"]["consecutive_no_decision"] == 18

        # The storm CLEARS after the cold build: a fresh FILLED decision is
        # now the newest row, so a *rebuild* would yield consecutive 0 /
        # restart_recommended False. Within the 20s TTL the next poll must
        # serve the OLD (still-alarming) payload from cache — the latency
        # win asserted as behaviour, and proof the verdict is not silently
        # recomputed on every 60s poll.
        s.record_decision(False, 1, "BUY NVDA → FILLED", "{}", 1000.0, 999.0)

        second = client.get("/api/runner-heartbeat").get_json()
        assert second["cached"] is True
        assert second["restart_recommended"] is True            # stale
        assert second["decision_efficacy"]["consecutive_no_decision"] == 18

    def test_inert_under_pytest_by_default(self, tmp_path, monkeypatch):
        """Without the explicit opt-in the handler runs every time with NO
        honesty keys and reflects the store live — this keeps the existing
        `tests/test_runner_heartbeat.py` endpoint tests isolated from a
        cross-test module-global cache leak (the `/api/state` contract)."""
        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        monkeypatch.setattr(d, "_SWR_TEST_FORCE", False)  # default pytest
        s = _seed_idle_storm(db)
        d.app.config["TESTING"] = True
        try:
            with d.app.test_client() as client:
                a = client.get("/api/runner-heartbeat").get_json()
                assert "cached" not in a and "cache_age_s" not in a
                assert a["restart_recommended"] is True
                assert a["decision_efficacy"]["consecutive_no_decision"] == 18
                # A live change is reflected immediately (no caching at all):
                # one FILLED on top clears the storm.
                s.record_decision(False, 1, "BUY NVDA → FILLED", "{}",
                                  1000.0, 999.0)
                b = client.get("/api/runner-heartbeat").get_json()
                assert b["decision_efficacy"]["consecutive_no_decision"] == 0
        finally:
            s.close()
