"""Flask wiring tests for /api/peer-earnings-shock.

Drives the dashboard endpoint with a fake store + monkeypatched
yfinance history seam (``_earnings_history_for``). Asserts the SSOT
composition chain (etf_lookthrough × event_calendar × earnings_shock
σ) wires correctly, and the endpoint never raises.

Pure arithmetic + verdict ladder is pinned in
test_peer_earnings_shock.py — this file only covers the IO seam.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeStore:
    def __init__(self, positions=None, portfolio=None):
        self._pos = positions or []
        self._pf = portfolio or {"cash": 100.0, "total_value": 500.0}
        self._lock = _NullLock()
        self.conn = None

    def open_positions(self):
        return self._pos

    def get_portfolio(self):
        return self._pf


@pytest.fixture
def client():
    return dashboard.app.test_client()


def _pos(ticker, qty, price, type_="stock"):
    return {"ticker": ticker, "qty": qty, "avg_cost": price,
            "current_price": price, "type": type_,
            "option_type": None, "strike": None, "expiry": None}


class TestEndpointWiring:
    def test_no_etf_held_branch(self, client, monkeypatch):
        # Book holds only NVDA directly — no leveraged ETF.
        store = _FakeStore(
            positions=[_pos("NVDA", 1, 200.0)],
            portfolio={"cash": 100.0, "total_value": 300.0},
        )
        monkeypatch.setattr(dashboard, "get_store", lambda: store)
        # Stub event_calendar's calendar reader to return no events.
        monkeypatch.setattr(
            dashboard, "_earnings_history_for", lambda _t, depth=8: [],
        )
        resp = client.get("/api/peer-earnings-shock")
        assert resp.status_code == 200
        body = resp.get_json()
        # Whether the calendar JSON exists locally is environment-
        # dependent; the endpoint must degrade to one of NO_ETF_HELD /
        # NO_PEER_EVENTS / NO_DATA — never raise — and crucially never
        # report a peer-event impact when no ETF is held.
        assert body["state"] in ("NO_ETF_HELD", "NO_PEER_EVENTS", "NO_DATA")
        assert body["n_etfs_at_risk"] == 0

    def test_no_peer_events_when_history_empty(self, client, monkeypatch):
        # Hold TQQQ but no historical reactions → every σ = None →
        # the row reads INSUFFICIENT_SIGMA per underlying, aggregate
        # excludes them. Whether peer events parse depends on the
        # local earnings_calendar.json — assert the endpoint returns a
        # well-shaped response in either branch.
        store = _FakeStore(
            positions=[_pos("TQQQ", 1, 100.0)],
            portfolio={"cash": 100.0, "total_value": 500.0},
        )
        monkeypatch.setattr(dashboard, "get_store", lambda: store)
        # Empty history → σ unknown.
        monkeypatch.setattr(
            dashboard, "_earnings_history_for", lambda _t, depth=8: [],
        )
        resp = client.get("/api/peer-earnings-shock")
        assert resp.status_code == 200
        body = resp.get_json()
        # Stable shape guaranteed; downstream callers can rely on
        # these keys whatever the branch.
        for k in ("state", "headline", "n_etfs_at_risk", "n_peer_events",
                  "total_indirect_sigma_dollar", "verdict", "etf_rows"):
            assert k in body

    def test_endpoint_never_raises_on_store_failure(self, client, monkeypatch):
        class _BrokenStore:
            def open_positions(self):
                raise RuntimeError("store explosion")

            def get_portfolio(self):
                return {"total_value": 0.0}

        monkeypatch.setattr(dashboard, "get_store", lambda: _BrokenStore())
        resp = client.get("/api/peer-earnings-shock")
        # The endpoint catches and returns 500 + error JSON; doesn't
        # propagate out to the WSGI layer.
        assert resp.status_code == 500
        assert "error" in resp.get_json()
