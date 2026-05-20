"""Flask wiring tests for /api/catalyst-class-autopsy.

Drives the dashboard endpoint with a fake store. Asserts the SSOT
builder is wired (trades come from recent_trades(2000) and are
list-reversed to oldest→newest, mirroring /api/loser-autopsy /
/api/winner-autopsy), and that the endpoint never raises.

Pure arithmetic of the taxonomy and the verdict ladder is pinned in
test_catalyst_class_autopsy.py — this file only covers the IO seam.
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
    def __init__(self, trades=None):
        self._t = trades or []
        self._lock = _NullLock()
        self.conn = None

    def recent_trades(self, limit=50):
        # store.recent_trades is newest-first; the endpoint reverses.
        return sorted(self._t, key=lambda t: t["timestamp"], reverse=True)[:limit]


@pytest.fixture
def client():
    return dashboard.app.test_client()


def _now():
    return datetime.now(timezone.utc)


def _trade(tid, ts, ticker, action, qty, price, reason=""):
    return {
        "id": tid, "timestamp": ts, "ticker": ticker, "action": action,
        "qty": qty, "price": price, "value": qty * price,
        "strike": None, "expiry": None, "option_type": None,
        "reason": reason,
    }


class TestEndpointWiring:
    def test_empty_store_returns_no_data(self, client, monkeypatch):
        store = _FakeStore(trades=[])
        monkeypatch.setattr(dashboard, "get_store", lambda: store)
        resp = client.get("/api/catalyst-class-autopsy")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["state"] == "NO_DATA"
        assert body["n_round_trips"] == 0

    def test_live_dram_whipsaw_classifies(self, client, monkeypatch):
        # Recreate a single round-trip with the DRAM-shaped reason text.
        now = _now()
        store = _FakeStore(trades=[
            _trade(
                1, (now - timedelta(hours=5)).isoformat(),
                "DRAM", "BUY", 5.0, 50.70,
                reason=("Triple-stacked catalyst: Citi bullish on DRAM, "
                        "HSBC/Melius PT, Cramer buy signal — and ML advisor "
                        "(median +143% alpha) flags DRAM. NVDA earnings "
                        "tomorrow may drag DRAM up sympathetically."),
            ),
            _trade(
                2, (now - timedelta(hours=3)).isoformat(),
                "DRAM", "SELL", 5.0, 50.61, reason="raising dry powder",
            ),
        ])
        monkeypatch.setattr(dashboard, "get_store", lambda: store)
        resp = client.get("/api/catalyst-class-autopsy")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["state"] == "EMERGING"  # n=1 < gate
        assert body["n_round_trips"] == 1
        assert body["n_scored"] == 1
        # All four classes present in the rationale showed up.
        class_names = {r["class"] for r in body["classes"]}
        assert "ML_ADVISOR" in class_names
        assert "ANALYST_PT" in class_names
        assert "PUNDIT" in class_names
        assert "EARNINGS_PLAY" in class_names
        assert "SECTOR_SYMPATHY" in class_names
        # Each class has the one losing trip → 0% win rate.
        ml = next(r for r in body["classes"] if r["class"] == "ML_ADVISOR")
        assert ml["n_trips"] == 1
        assert ml["win_rate_pct"] == 0.0
        # Verdict withheld below gate.
        assert ml["verdict"] == "UNSTABLE"

    def test_endpoint_never_raises_on_store_failure(self, client, monkeypatch):
        class _BrokenStore:
            def recent_trades(self, limit=50):
                raise RuntimeError("store explosion")
        monkeypatch.setattr(dashboard, "get_store", lambda: _BrokenStore())
        # The route's own try/except surfaces 500+error, never raises out
        # to the WSGI layer — the documented dashboard contract.
        resp = client.get("/api/catalyst-class-autopsy")
        assert resp.status_code == 500
        assert "error" in resp.get_json()
