"""End-to-end Flask-client test for /api/sector-exposure (the
test_feed_health_endpoint.py / test_decision_context_endpoint.py convention —
route wiring + the parity contract exercised through the real Flask app, not a
module __main__ smoke; per the paper-trader-analytics-verification note).

The discriminating lock: /api/sector-exposure must be numerically identical
to /api/analytics `sector_exposure_pct` for the same store snapshot — that is
the single-source-of-truth guarantee the whole feature rests on. Both routes
read `store.get_portfolio()` total_value + `store.open_positions()` and the
same SECTOR_MAP, so a drift here means the dashboard contradicts the prompt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paper_trader.dashboard as d
import paper_trader.store as store_mod
from paper_trader.store import Store


@pytest.fixture
def client_store(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    # A deliberately concentrated, semis-heavy book (the documented
    # pathology). current_price defaults to 0 → both routes fall back to
    # avg_cost (the analytics_api formula), so the math is deterministic.
    s.record_trade("MU", "BUY", 6, 100.0, reason="x")
    s.upsert_position("MU", "stock", 6, 100.0)        # 600 semis
    s.record_trade("SOXL", "BUY", 10, 30.0, reason="x")
    s.upsert_position("SOXL", "stock", 10, 30.0)      # 300 semis_lev
    s.record_trade("LITE", "BUY", 2, 50.0, reason="x")
    s.upsert_position("LITE", "stock", 2, 50.0)       # 100 optical
    s.update_portfolio(0.0, 1000.0, [])               # total_value = 1000
    d.app.config["TESTING"] = True
    try:
        with d.app.test_client() as client:
            yield client, s
    finally:
        s.close()


def test_endpoint_shape_and_values(client_store):
    client, _ = client_store
    r = client.get("/api/sector-exposure")
    assert r.status_code == 200
    j = r.get_json()
    assert j["state"] == "CONCENTRATED"          # semis 60% == heavy mark
    assert j["sector_pct"] == {"semis": 60.0, "semis_lev": 30.0,
                               "optical": 10.0}
    assert j["top_sector"] == "semis"
    assert j["top_sector_pct"] == 60.0
    assert "prompt_block" in j and "SECTOR EXPOSURE" in j["prompt_block"]


def test_endpoint_parity_with_analytics(client_store):
    """The whole single-source-of-truth promise: the prompt-facing endpoint
    and the dashboard analytics endpoint must report the SAME book sector
    breakdown for the same store, or they silently contradict each other."""
    client, _ = client_store
    se = client.get("/api/sector-exposure").get_json()
    an = client.get("/api/analytics").get_json()
    assert se["sector_pct"] == an["sector_exposure_pct"]
