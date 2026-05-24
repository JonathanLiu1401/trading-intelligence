"""Tests for paper_trader/analytics/cash_drag.py + the /api/cash-drag endpoint.

These pins assert exact arithmetic and the sample-size-honest state ladder.
A regression — wrong drag formula, leaky window filter, missing time-weighted
average, neutral-band that collapses a real cost to NEUTRAL, or the endpoint
re-deriving the verdict instead of single-sourcing the builder — fails here.
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
from paper_trader import store as store_mod
from paper_trader.analytics.cash_drag import build_cash_drag

_NOW = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


def _row(hours_ago: float, cash: float, sp: float, tv: float | None = None) -> dict:
    return {
        "timestamp": (_NOW - timedelta(hours=hours_ago)).isoformat(),
        "total_value": tv if tv is not None else cash,
        "cash": cash,
        "sp500_price": sp,
    }


def _at(ts: datetime, cash: float, sp: float, tv: float | None = None) -> dict:
    return {
        "timestamp": ts.isoformat(),
        "total_value": tv if tv is not None else cash,
        "cash": cash,
        "sp500_price": sp,
    }


# ─────────────────────── pure builder: state ladder ────────────────────────

class TestStateLadder:
    def test_no_data_when_empty(self):
        r = build_cash_drag([], now=_NOW)
        assert r["state"] == "NO_DATA"
        assert r["verdict"] is None
        assert "No benchmarkable" in r["headline"]
        assert r["windows"] == []
        assert r["n_total_points"] == 0

    def test_no_data_when_rows_unparseable(self):
        bad = [
            {"timestamp": None, "cash": 100, "sp500_price": 7400},
            {"timestamp": "2026-05-24T00:00:00+00:00", "cash": None, "sp500_price": 7400},
            {"timestamp": "2026-05-24T00:00:00+00:00", "cash": 100, "sp500_price": None},
            {"timestamp": "2026-05-24T00:00:00+00:00", "cash": -1, "sp500_price": 7400},
            {"timestamp": "garbage", "cash": 100, "sp500_price": 7400},
            {"timestamp": "2026-05-24T00:00:00+00:00", "cash": 100, "sp500_price": 0},
        ]
        r = build_cash_drag(bad, now=_NOW)
        assert r["state"] == "NO_DATA"
        assert r["n_total_points"] == 0

    def test_state_ok_emits_per_window_blocks(self):
        # Two points 20h apart inside the 24h window — span 20h ≥ 24*0.6=14.4h.
        rows = [_row(20.0, 1000.0, 7400.0), _row(0.5, 1000.0, 7474.0)]
        r = build_cash_drag(rows, windows_h=(24.0,), now=_NOW)
        assert r["state"] == "OK"
        assert len(r["windows"]) == 1
        w = r["windows"][0]
        assert w["state"] == "OK"
        assert w["window_hours"] == 24.0
        assert w["n_points"] == 2

    def test_window_insufficient_when_span_below_coverage_floor(self):
        # Two points 1h apart — span 1h is < 24*0.6=14.4h, so 24h arm is INSUFFICIENT.
        rows = [_row(1.0, 1000.0, 7400.0), _row(0.0, 1000.0, 7400.0)]
        r = build_cash_drag(rows, windows_h=(24.0,), now=_NOW)
        # One window, INSUFFICIENT — top-level falls back to INSUFFICIENT.
        assert r["windows"][0]["state"] == "INSUFFICIENT"
        assert r["verdict"] == "INSUFFICIENT"
        assert "insufficient" in r["windows"][0]["headline"].lower()

    def test_window_insufficient_with_single_point(self):
        rows = [_row(0.5, 1000.0, 7400.0)]
        r = build_cash_drag(rows, windows_h=(24.0,), now=_NOW)
        assert r["windows"][0]["state"] == "INSUFFICIENT"
        assert r["windows"][0]["n_points"] == 1


# ─────────────────────────── arithmetic pins ────────────────────────────────

class TestCashDragArithmetic:
    def test_costly_cash_when_spy_rises(self):
        # 20h apart, spans 24h*0.6=14.4h ✓; SPY +1% over the window;
        # constant cash 1000 → drag = 1000 * 1.0 / 100 = $10.00 (COSTLY).
        rows = [_row(20.0, 1000.0, 7400.0), _row(0.5, 1000.0, 7474.0)]
        r = build_cash_drag(rows, windows_h=(24.0,), now=_NOW)
        w = r["windows"][0]
        assert w["state"] == "OK"
        assert w["sp500_return_pct"] == 1.0
        assert w["avg_cash_usd"] == 1000.0
        assert w["cash_drag_usd"] == 10.0
        assert w["verdict"] == "COSTLY_CASH"
        assert r["verdict"] == "COSTLY_CASH"
        assert "cost you $10.00" in w["headline"]
        # SPY return sign also surfaces in the headline.
        assert "+1.00%" in w["headline"]

    def test_helpful_cash_when_spy_falls(self):
        # SPY drops 1.0% — cash saved you $10.
        rows = [_row(20.0, 1000.0, 7400.0), _row(0.5, 1000.0, 7326.0)]
        r = build_cash_drag(rows, windows_h=(24.0,), now=_NOW)
        w = r["windows"][0]
        assert w["sp500_return_pct"] == -1.0
        assert w["cash_drag_usd"] == -10.0
        assert w["verdict"] == "HELPFUL_CASH"
        assert "saved you $10.00" in w["headline"]
        assert r["verdict"] == "HELPFUL_CASH"

    def test_neutral_band_absorbs_tiny_drag(self):
        # SPY +0.005% with $1000 cash → drag $0.05 < $0.50 → NEUTRAL.
        rows = [_row(20.0, 1000.0, 7400.0), _row(0.5, 1000.0, 7400.37)]
        r = build_cash_drag(rows, windows_h=(24.0,), now=_NOW)
        w = r["windows"][0]
        assert w["verdict"] == "NEUTRAL"
        assert abs(w["cash_drag_usd"]) <= 0.50
        assert r["verdict"] == "NEUTRAL"

    def test_time_weighted_average_weights_long_intervals(self):
        # 0..15h: cash 0 (15h of zero). 15..20h: cash 1000 (5h of 1000).
        # Total span 20h → time-weighted mean = (0*15 + 1000*5) / 20 = 250.
        # A simple mean of [0, 0, 1000] would be 333.3 — verifies weighting.
        rows = [
            _row(20.0, 0.0, 7400.0),
            _row(5.0, 0.0, 7400.0),
            _row(0.5, 1000.0, 7474.0),  # SPY +1% over the 19.5h span
        ]
        r = build_cash_drag(rows, windows_h=(24.0,), now=_NOW)
        w = r["windows"][0]
        # Trapezoidal: ((0+0)/2)*15h + ((0+1000)/2)*4.5h = 0 + 2250 = 2250
        # divided by 19.5h span = ~115.38 (not 333.3).
        assert w["avg_cash_usd"] == pytest.approx(115.38, abs=0.05)
        # drag = 115.38 * 1.0/100 = ~$1.15
        assert w["cash_drag_usd"] == pytest.approx(1.15, abs=0.05)

    def test_only_in_window_rows_count(self):
        # Old row OUTSIDE the 24h window must be ignored. If it counted,
        # the SPY return would be measured against the old anchor and
        # come out wildly larger.
        rows = [
            _row(48.0, 1000.0, 6000.0),  # 48h ago — outside 24h window
            _row(20.0, 1000.0, 7400.0),  # in-window anchor
            _row(0.5, 1000.0, 7474.0),   # in-window latest
        ]
        r = build_cash_drag(rows, windows_h=(24.0,), now=_NOW)
        w = r["windows"][0]
        assert w["n_points"] == 2
        assert w["sp500_return_pct"] == 1.0

    def test_top_verdict_picks_worst_costly_window(self):
        # 24h: SPY +1% → $10 COSTLY.   168h: SPY +2% → $20 COSTLY.
        # Top-level verdict picks the WORST (most costly) — $20 / 168h.
        rows = [
            _at(_NOW - timedelta(hours=160), 1000.0, 7000.0),
            _at(_NOW - timedelta(hours=23), 1000.0, 7070.0),  # 24h anchor
            _at(_NOW - timedelta(hours=0.5), 1000.0, 7140.0),  # latest
        ]
        r = build_cash_drag(rows, windows_h=(24.0, 168.0), now=_NOW)
        w24 = next(w for w in r["windows"] if w["window_hours"] == 24.0)
        w168 = next(w for w in r["windows"] if w["window_hours"] == 168.0)
        assert w24["verdict"] == "COSTLY_CASH"
        assert w168["verdict"] == "COSTLY_CASH"
        assert w168["cash_drag_usd"] > w24["cash_drag_usd"]
        assert r["verdict"] == "COSTLY_CASH"
        assert "worst window" in r["headline"]
        # The headline names the 168h drag, not the 24h one.
        assert "168h" in r["headline"]


# ───────────────────── /api/cash-drag endpoint contract ─────────────────────


class _FakeStore:
    def __init__(self, rows):
        self._rows = rows

    def equity_curve(self, limit: int = 500) -> list[dict]:
        return list(self._rows)


@pytest.fixture
def client():
    dashboard.app.config["TESTING"] = True
    return dashboard.app.test_client()


@pytest.fixture
def stub_store(monkeypatch):
    """Replace ``store.get_store`` with a list-fed fake so tests are
    deterministic and never touch a real DB. Also clears the SWR cache
    so each call hits the fresh fake."""
    # Clear the swr_cached layer the endpoint uses, so the previous
    # test's payload doesn't leak forward.
    cache = getattr(dashboard, "_SWR_STATE", None)
    if isinstance(cache, dict):
        cache.pop("cash_drag", None)

    def _install(rows):
        fake = _FakeStore(rows)
        monkeypatch.setattr(dashboard, "get_store", lambda: fake)
        monkeypatch.setattr(store_mod, "_singleton", fake)
        return fake

    return _install


class TestCashDragEndpointContract:
    def test_endpoint_returns_no_data_envelope_on_empty(self, client, stub_store):
        stub_store([])
        r = client.get("/api/cash-drag")
        assert r.status_code == 200
        body = r.get_json()
        assert body["state"] == "NO_DATA"
        assert body["verdict"] is None
        assert "as_of" in body
        assert body["windows"] == []

    def test_endpoint_passes_real_rows_to_builder(self, client, stub_store):
        # Real rows that the builder will turn into COSTLY_CASH.
        rows = [
            {"timestamp": (datetime.now(timezone.utc) - timedelta(hours=20)).isoformat(),
             "total_value": 1000.0, "cash": 1000.0, "sp500_price": 7400.0},
            {"timestamp": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
             "total_value": 1000.0, "cash": 1000.0, "sp500_price": 7474.0},
        ]
        stub_store(rows)
        r = client.get("/api/cash-drag")
        assert r.status_code == 200
        body = r.get_json()
        # The endpoint MUST single-source the builder — verdict / arithmetic
        # came from the analytics module, not re-derived by the route.
        assert body["state"] == "OK"
        assert body["verdict"] == "COSTLY_CASH"
        # One window block per default window length.
        assert {w["window_hours"] for w in body["windows"]} == {24.0, 168.0, 720.0}

    def test_endpoint_degrades_to_500_on_builder_failure(self, client, monkeypatch):
        # If get_store raises, the endpoint returns a 500 with the error.
        # Mirrors the /api/benchmark error contract.
        def _boom():
            raise RuntimeError("simulated store failure")
        cache = getattr(dashboard, "_SWR_STATE", None)
        if isinstance(cache, dict):
            cache.pop("cash_drag", None)
        monkeypatch.setattr(dashboard, "get_store", _boom)
        r = client.get("/api/cash-drag")
        assert r.status_code == 500
        body = r.get_json()
        assert "error" in body
        assert "simulated store failure" in body["error"]
