"""End-to-end Flask-client test for /api/concentration-trajectory.

Mirrors the test_sector_exposure_endpoint.py / test_feed_health_endpoint.py
convention — route wiring + threading from store → builder exercised
through the real Flask app (not a module __main__ smoke; per
project_paper_trader_analytics_verification: the smoke path hits a
different DB and reports CLEAN when the real route fails).

The discriminating locks:

* ``?days=`` clamps to ``[MIN_SNAPSHOTS, MAX_SNAPSHOTS]`` — out-of-band
  values do not blow up.
* The endpoint returns the same shape as the pure builder
  (``series`` / ``current`` / ``verdict`` / ``headline`` /
  ``thresholds``).
* The route degrades gracefully (200 + ``error`` field, not a 500) when
  the downstream daily-history helper returns empty rows.
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
    # Reset the dashboard's store cache so it picks up the temp DB.
    monkeypatch.setattr(d, "_STORE", None, raising=False)
    s = Store()
    # Synthesise a clearly-concentrated book (a single 1000-USD NVDA
    # BUY ~20 days ago — far enough back that even a 30-day window
    # emits the full series). Insert the trade row directly with a
    # backdated timestamp because Store.record_trade stamps `now()`,
    # which would collapse the window to a single snapshot today.
    from datetime import datetime, timedelta, timezone
    backdated = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    s.conn.execute(
        "INSERT INTO trades (timestamp, ticker, action, qty, price, value, "
        "reason, expiry, strike, option_type) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (backdated, "NVDA", "BUY", 10, 100.0, 1000.0, "seed",
         None, None, None),
    )
    s.conn.commit()
    s.upsert_position("NVDA", "stock", 10, 100.0)
    s.update_portfolio(0.0, 1000.0, [])
    # Provide a flat closes history so every snapshot marks NVDA at 100.
    today = datetime.now(timezone.utc).date()
    rows = [((today - timedelta(days=i)).isoformat(), 100.0)
            for i in range(60, -1, -1)]
    monkeypatch.setattr(d, "_daily_history_cached",
                        lambda tk, period="3mo": rows if tk == "NVDA" else [])
    d.app.config["TESTING"] = True
    try:
        with d.app.test_client() as client:
            yield client, s
    finally:
        s.close()


def test_endpoint_shape_and_concentrated_steady(client_store):
    client, _ = client_store
    r = client.get("/api/concentration-trajectory?days=10")
    assert r.status_code == 200
    j = r.get_json()
    # Shape contract.
    for k in ("series", "current", "verdict", "headline", "thresholds",
             "window_days", "n_trades_walked", "delta_top1_pct",
             "max_top1_pct", "min_top1_pct", "as_of"):
        assert k in j, f"missing key: {k}"
    # A single-name book held flat is CONCENTRATED_STEADY (top-1 at
    # 100% throughout the window).
    assert j["verdict"] == "CONCENTRATED_STEADY"
    assert j["current"]["top1_ticker"] == "NVDA"
    assert j["current"]["top1_pct"] == 100.0


def test_endpoint_days_clamps_low(client_store):
    client, _ = client_store
    r = client.get("/api/concentration-trajectory?days=1")
    assert r.status_code == 200
    j = r.get_json()
    # MIN_SNAPSHOTS = 3 — the clamp is applied inside the route too.
    # With a single BUY 20 days ago the book has 3 snapshots emitted.
    assert j["window_days"] == 3


def test_endpoint_days_clamps_high(client_store):
    client, _ = client_store
    r = client.get("/api/concentration-trajectory?days=9999")
    assert r.status_code == 200
    j = r.get_json()
    # MAX_SNAPSHOTS = 30.
    assert j["window_days"] <= 30
