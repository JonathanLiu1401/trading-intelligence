"""Flask wiring tests for /api/round-trip-postmortem.

Drives the dashboard endpoint with a fake store + monkeypatched
market.get_prices so the test is offline and deterministic. Asserts
that the SSOT builder is wired correctly, the lookback filter works,
and the price-fetch failure path degrades gracefully (no raise).

Pure arithmetic of the verdict ladder is pinned in
test_round_trip_postmortem.py — this file only covers the IO seam.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard, market


class _FakeStore:
    def __init__(self, trades=None):
        self._t = trades or []
        self._lock = _NullLock()
        self.conn = None  # endpoint shouldn't touch trades.conn for this path

    def recent_trades(self, limit=50):
        # store.recent_trades is newest-first; the endpoint reverses.
        return sorted(self._t, key=lambda t: t["timestamp"], reverse=True)[:limit]


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _now():
    return datetime.now(timezone.utc)


@pytest.fixture
def client():
    return dashboard.app.test_client()


def _trade(tid, ts, ticker, action, qty, price):
    return {
        "id": tid,
        "timestamp": ts,
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "price": price,
        "value": qty * price,
        "strike": None,
        "expiry": None,
        "option_type": None,
    }


class TestEndpointWiring:
    def test_dram_whipsaw_surfaces(self, client, monkeypatch):
        # Recreate the live 2026-05-19 DRAM round-trip shape: BUY 5 @50.70
        # ~5h ago, SELL 5 @50.61 ~3h ago, and price has recovered to 51.50
        # now (+1.76% post-exit). Short hold (2h) + small loss + recovery
        # ⇒ WHIPSAW.
        now = _now()
        store = _FakeStore(trades=[
            _trade(1, (now - timedelta(hours=5)).isoformat(),
                   "DRAM", "BUY", 5.0, 50.70),
            _trade(2, (now - timedelta(hours=3)).isoformat(),
                   "DRAM", "SELL", 5.0, 50.61),
        ])
        monkeypatch.setattr(dashboard, "get_store", lambda: store)
        monkeypatch.setattr(market, "get_prices",
                            lambda tk: {t: 51.50 for t in tk})

        res = client.get("/api/round-trip-postmortem")
        j = res.get_json()
        assert res.status_code == 200
        assert j["n_input"] == 1
        assert j["n_scored"] == 1
        assert len(j["trips"]) == 1
        trip = j["trips"][0]
        assert trip["ticker"] == "DRAM"
        assert trip["verdict"] == "WHIPSAW"
        # Post-exit drift = (51.50 - 50.61) / 50.61 * 100 ≈ 1.76%
        assert abs(trip["post_exit_drift_pct"] - 1.76) < 0.05
        # Exit-quality score for one WHIPSAW = -2.
        assert j["exit_quality_score"] == -2.0
        assert j["state"] == "OK"

    def test_correct_exit_when_price_falls(self, client, monkeypatch):
        now = _now()
        store = _FakeStore(trades=[
            _trade(1, (now - timedelta(hours=48)).isoformat(),
                   "NVDA", "BUY", 2.0, 200.0),
            _trade(2, (now - timedelta(hours=12)).isoformat(),
                   "NVDA", "SELL", 2.0, 220.0),
        ])
        monkeypatch.setattr(dashboard, "get_store", lambda: store)
        # Price fell post-exit.
        monkeypatch.setattr(market, "get_prices",
                            lambda tk: {t: 210.0 for t in tk})

        j = client.get("/api/round-trip-postmortem").get_json()
        trip = j["trips"][0]
        assert trip["verdict"] == "CORRECT"
        # (210 - 220) / 220 * 100 ≈ -4.55
        assert trip["post_exit_drift_pct"] < CORRECT_THRESHOLD
        assert j["exit_quality_score"] == 1.0

    def test_hours_back_filter_clips_old_trips(self, client, monkeypatch):
        now = _now()
        # Old round-trip (closed 30 days ago) + recent round-trip.
        store = _FakeStore(trades=[
            _trade(1, (now - timedelta(hours=24 * 35)).isoformat(),
                   "OLD", "BUY", 1.0, 10.0),
            _trade(2, (now - timedelta(hours=24 * 30)).isoformat(),
                   "OLD", "SELL", 1.0, 10.0),
            _trade(3, (now - timedelta(hours=48)).isoformat(),
                   "NEW", "BUY", 1.0, 100.0),
            _trade(4, (now - timedelta(hours=12)).isoformat(),
                   "NEW", "SELL", 1.0, 110.0),
        ])
        monkeypatch.setattr(dashboard, "get_store", lambda: store)
        monkeypatch.setattr(market, "get_prices",
                            lambda tk: {t: 110.0 for t in tk})

        # ?hours_back=24 should exclude both — the new round-trip closed
        # 12h ago which IS in 24h, the old one is not.
        j = client.get("/api/round-trip-postmortem?hours_back=24").get_json()
        assert all(tr["ticker"] != "OLD" for tr in j["trips"])
        # Newer trip surfaces.
        assert any(tr["ticker"] == "NEW" for tr in j["trips"])

    def test_max_n_clamped(self, client, monkeypatch):
        # 5 trips, max_n=2 → only 2 surface.
        now = _now()
        trades = []
        for i in range(5):
            entry = (now - timedelta(hours=48 + i)).isoformat()
            exit_ = (now - timedelta(hours=24 + i)).isoformat()
            trades.append(_trade(2 * i + 1, entry, f"T{i}", "BUY", 1.0, 100.0))
            trades.append(_trade(2 * i + 2, exit_, f"T{i}", "SELL", 1.0, 100.0))
        store = _FakeStore(trades=trades)
        monkeypatch.setattr(dashboard, "get_store", lambda: store)
        monkeypatch.setattr(market, "get_prices",
                            lambda tk: {t: 100.0 for t in tk})

        j = client.get("/api/round-trip-postmortem?max_n=2").get_json()
        assert len(j["trips"]) <= 2

    def test_price_fetch_failure_degrades_not_raises(self, client, monkeypatch):
        now = _now()
        store = _FakeStore(trades=[
            _trade(1, (now - timedelta(hours=48)).isoformat(),
                   "XYZ", "BUY", 1.0, 50.0),
            _trade(2, (now - timedelta(hours=12)).isoformat(),
                   "XYZ", "SELL", 1.0, 50.0),
        ])
        monkeypatch.setattr(dashboard, "get_store", lambda: store)

        def _boom(tickers):
            raise RuntimeError("yfinance unavailable")
        monkeypatch.setattr(market, "get_prices", _boom)

        res = client.get("/api/round-trip-postmortem")
        # 200, INSUFFICIENT — no price ⇒ verdict withheld but endpoint OK.
        assert res.status_code == 200
        j = res.get_json()
        assert j["n_input"] >= 1
        # No scored verdicts when prices fail to load.
        assert j["n_scored"] == 0

    def test_no_round_trips_no_data(self, client, monkeypatch):
        store = _FakeStore(trades=[])
        monkeypatch.setattr(dashboard, "get_store", lambda: store)
        monkeypatch.setattr(market, "get_prices", lambda tk: {})

        j = client.get("/api/round-trip-postmortem").get_json()
        assert j["state"] == "NO_DATA"


# Constant pulled from the module so the test moves with the threshold,
# not a stale literal — the test_round_trip_postmortem.py file already
# pins the *value*.
from paper_trader.analytics.round_trip_postmortem import (  # noqa: E402
    CORRECT_MAX_DRIFT_PCT as CORRECT_THRESHOLD,
)
