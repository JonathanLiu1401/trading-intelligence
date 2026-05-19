"""End-to-end Flask-client tests for /api/earnings-distribution.

Convention mirrors tests/test_baseline_compare_endpoint.py — real Flask app,
real builder, deterministic offline data; no :8090 bind, no live DB, no
yfinance call (history_provider is stubbed)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paper_trader.dashboard as d


def _stock_pos(ticker, qty, current_price, avg_cost=None):
    return {
        "ticker": ticker,
        "qty": qty,
        "current_price": current_price,
        "avg_cost": avg_cost if avg_cost is not None else current_price,
        "type": "stock",
    }


def _patch_store(monkeypatch, positions, total_value):
    fake_store = MagicMock()
    fake_store.open_positions = MagicMock(return_value=positions)
    fake_store.get_portfolio = MagicMock(return_value={"total_value": total_value})
    monkeypatch.setattr(d, "get_store", lambda: fake_store)


def _patch_event_calendar(monkeypatch, events):
    def _fake_ec(_positions, _names):
        return {"events": events, "state": "OK"}
    monkeypatch.setattr(
        "paper_trader.analytics.event_calendar.build_event_calendar",
        _fake_ec,
    )


def _patch_history(monkeypatch, history_map):
    def _fake(ticker, depth=8):
        return history_map.get(ticker, [])[:depth]
    monkeypatch.setattr(d, "_earnings_history_for", _fake)


@pytest.fixture
def client(monkeypatch):
    d.app.config["TESTING"] = True
    with d.app.test_client() as c:
        yield c, monkeypatch


def test_route_returns_payload_shape(client):
    c, mp = client
    _patch_store(mp, [_stock_pos("NVDA", 2, 222.35)], total_value=1000.0)
    _patch_event_calendar(mp, [{
        "ticker": "NVDA", "days_away": 0.9,
        "earnings_date": "2026-05-20T00:00:00+00:00",
        "tier": "HELD_IMMINENT",
    }])
    _patch_history(mp, {"NVDA": [-2.1, 3.7, 0.5, -1.2, 2.0, 1.0, -0.5, 4.0]})

    r = c.get("/api/earnings-distribution")
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    # Top-level contract.
    for k in ("as_of", "horizon_days", "history_depth", "min_history",
              "n_events", "events", "state", "headline"):
        assert k in j, f"missing key {k!r} in {sorted(j)}"
    assert j["state"] == "OK"
    assert j["n_events"] == 1
    row = j["events"][0]
    # Row-level contract.
    for k in ("ticker", "days_to_earnings", "earnings_date", "tier",
              "current_value_usd", "weight_pct", "n_history",
              "observed_quartiles", "dollar_quartiles", "book_pct_quartiles",
              "downside_worst_dollar", "downside_worst_book_pct",
              "row_verdict", "state", "headline"):
        assert k in row, f"missing row key {k!r}"
    assert row["ticker"] == "NVDA"
    assert row["state"] == "OK"


def test_quartile_keys_use_observational_naming(client):
    """Per advisor framing: keys must be q1/median/q3 (observed quartiles),
    NOT p25/p50/p75 (which would imply distributional inference n=8 can't
    support). Lock the naming to prevent drift back to misleading labels."""
    c, mp = client
    _patch_store(mp, [_stock_pos("NVDA", 2, 222.35)], total_value=1000.0)
    _patch_event_calendar(mp, [{
        "ticker": "NVDA", "days_away": 0.9,
        "earnings_date": "2026-05-20T00:00:00+00:00",
        "tier": "HELD_IMMINENT",
    }])
    _patch_history(mp, {"NVDA": [-2.1, 3.7, 0.5, -1.2, 2.0, 1.0, -0.5, 4.0]})
    row = c.get("/api/earnings-distribution").get_json()["events"][0]
    assert set(row["observed_quartiles"]) == {"worst", "q1", "median", "q3", "best"}
    assert set(row["dollar_quartiles"]) == {"worst", "q1", "median", "q3", "best"}
    assert set(row["book_pct_quartiles"]) == {"worst", "q1", "median", "q3", "best"}
    # Lock-out the misleading distributional naming.
    for bad in ("p5", "p25", "p50", "p75", "p95"):
        assert bad not in row["observed_quartiles"]


def test_dollar_quartiles_equal_position_value_times_pct(client):
    c, mp = client
    _patch_store(mp, [_stock_pos("NVDA", 2, 222.35)], total_value=1000.0)
    _patch_event_calendar(mp, [{
        "ticker": "NVDA", "days_away": 0.9,
        "earnings_date": "2026-05-20T00:00:00+00:00",
        "tier": "HELD_IMMINENT",
    }])
    _patch_history(mp, {"NVDA": [-10.0, -2.0, 1.0, 3.0]})  # worst = -10%
    row = c.get("/api/earnings-distribution").get_json()["events"][0]
    pos_value = 2 * 222.35
    assert row["dollar_quartiles"]["worst"] == pytest.approx(pos_value * -10 / 100, abs=0.01)


def test_no_data_when_no_positions(client):
    c, mp = client
    _patch_store(mp, [], total_value=1000.0)
    _patch_event_calendar(mp, [])
    _patch_history(mp, {})
    j = c.get("/api/earnings-distribution").get_json()
    assert j["state"] == "NO_DATA"


def test_no_events_when_book_held_but_calendar_clear(client):
    c, mp = client
    _patch_store(mp, [_stock_pos("NVDA", 2, 222.35)], total_value=1000.0)
    _patch_event_calendar(mp, [])  # no held imminent prints
    _patch_history(mp, {})
    j = c.get("/api/earnings-distribution").get_json()
    assert j["state"] == "NO_EVENTS"


def test_cors_header_present_for_cross_fetch(client):
    c, mp = client
    _patch_store(mp, [], total_value=0.0)
    _patch_event_calendar(mp, [])
    _patch_history(mp, {})
    r = c.get("/api/earnings-distribution")
    assert r.headers.get("Access-Control-Allow-Origin") == "*"


def test_endpoint_degrades_to_error_body_on_store_raise(monkeypatch):
    d.app.config["TESTING"] = True

    def _boom():
        raise RuntimeError("db unreachable")
    monkeypatch.setattr(d, "get_store", _boom)
    with d.app.test_client() as c:
        r = c.get("/api/earnings-distribution")
        assert r.status_code == 500
        j = r.get_json()
        assert "error" in j
