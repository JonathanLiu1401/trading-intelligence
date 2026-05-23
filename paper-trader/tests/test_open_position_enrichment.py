"""Regression lock: ``/api/state`` positions carry derived fields (``pl_pct``,
``market_value``, ``hold_seconds``, ``hold_days``) so consumers can read the
trader-essential percentage return and lot value without re-deriving them.

Without these the dashboard JS used to recompute ``market_value`` client-side
(line 2566) AND show ``—`` for per-position %P/L; any cross-port consumer (the
digital-intern dashboard, scripts) had to invent its own formula and would
silently disagree with ``_mark_to_market``.

These tests assert specific computed values against the helper directly AND
end-to-end through the Flask test client so a regression in either layer fails
loudly.
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
from paper_trader.dashboard import _enrich_open_position
from paper_trader.store import Store


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    monkeypatch.setattr(dashboard, "get_store", lambda: s)
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def client():
    dashboard.app.config["TESTING"] = True
    return dashboard.app.test_client()


class TestEnrichHelperPurity:
    """Pure helper: same inputs → same outputs, never raises, never mutates
    the input dict in-place."""

    def test_stock_pl_pct_and_market_value_correct(self):
        p = {"ticker": "NVDA", "type": "stock", "qty": 3.0,
             "avg_cost": 223.435, "current_price": 215.25,
             "unrealized_pl": -24.555, "opened_at": None}
        out = _enrich_open_position(p)
        # (215.25 - 223.435) / 223.435 * 100 = -3.6633%
        assert out["pl_pct"] == pytest.approx(-3.6633, rel=1e-3)
        # market_value = 215.25 * 3 = 645.75 (stock multiplier 1)
        assert out["market_value"] == pytest.approx(645.75, rel=1e-3)

    def test_option_market_value_uses_100x_multiplier(self):
        p = {"ticker": "NVDA", "type": "call", "qty": 2.0,
             "avg_cost": 5.00, "current_price": 7.50,
             "strike": 220.0, "expiry": "2026-06-19",
             "unrealized_pl": 500.0, "opened_at": None}
        out = _enrich_open_position(p)
        # market_value = 7.50 * 2 * 100 = 1500.0
        assert out["market_value"] == pytest.approx(1500.0, rel=1e-6)
        # pl_pct same as stock — per-contract premium % (not dollar P/L)
        assert out["pl_pct"] == pytest.approx(50.0, rel=1e-6)

    def test_put_uses_100x_multiplier_too(self):
        p = {"ticker": "SPY", "type": "put", "qty": 1.0,
             "avg_cost": 4.00, "current_price": 2.00, "opened_at": None}
        out = _enrich_open_position(p)
        assert out["market_value"] == pytest.approx(200.0, rel=1e-6)
        assert out["pl_pct"] == pytest.approx(-50.0, rel=1e-6)

    def test_zero_avg_cost_yields_none_pl_pct(self):
        # avg_cost = 0 has no meaningful base — surface None, never +inf
        # or a misleading 0.00 that masks a free position.
        p = {"ticker": "FREE", "type": "stock", "qty": 1.0,
             "avg_cost": 0.0, "current_price": 10.0, "opened_at": None}
        out = _enrich_open_position(p)
        assert out["pl_pct"] is None
        # market_value still computes correctly even with avg_cost=0
        assert out["market_value"] == pytest.approx(10.0, rel=1e-6)

    def test_missing_current_price_yields_zero_market_value(self):
        # A newly-opened lot before its first mark has current_price=0
        # (schema default). market_value reads 0, pl_pct reads -100% (loss
        # of the full premium, which is the honest read for "I just bought
        # this and the market hasn't priced it yet OR the lookup failed").
        p = {"ticker": "NEW", "type": "stock", "qty": 1.0,
             "avg_cost": 100.0, "current_price": 0.0, "opened_at": None}
        out = _enrich_open_position(p)
        assert out["market_value"] == 0.0
        assert out["pl_pct"] == pytest.approx(-100.0, rel=1e-6)

    def test_non_numeric_inputs_never_raise(self):
        # A row with corrupted string fields must degrade gracefully —
        # /api/state must never 500 on a single bad row.
        p = {"ticker": "BAD", "type": "stock", "qty": "garbage",
             "avg_cost": "nope", "current_price": "abc",
             "opened_at": None}
        out = _enrich_open_position(p)
        assert out["market_value"] == 0.0
        assert out["pl_pct"] is None
        assert out["hold_seconds"] is None
        assert out["hold_days"] is None

    def test_does_not_mutate_input(self):
        p = {"ticker": "NVDA", "type": "stock", "qty": 1.0,
             "avg_cost": 100.0, "current_price": 110.0, "opened_at": None}
        before = dict(p)
        _enrich_open_position(p)
        assert p == before, "helper must not mutate the input row in-place"

    def test_preserves_existing_fields(self):
        # Stable contract: any field already on the input row passes through
        # untouched — a consumer relying on e.g. `id`, `opened_at`, `strike`
        # must still see them on the enriched row.
        p = {"id": 42, "ticker": "NVDA", "type": "call", "qty": 1.0,
             "avg_cost": 5.0, "current_price": 6.0, "strike": 200.0,
             "expiry": "2026-06-19", "opened_at": "2026-05-20T12:00:00+00:00",
             "closed_at": None, "unrealized_pl": 100.0}
        out = _enrich_open_position(p)
        assert out["id"] == 42
        assert out["strike"] == 200.0
        assert out["expiry"] == "2026-06-19"
        assert out["closed_at"] is None
        assert out["unrealized_pl"] == 100.0


class TestHoldDuration:
    """``hold_seconds`` / ``hold_days`` from ``opened_at`` — surfaces the
    disposition-effect ammo (riding losers vs cutting winners) the dashboard
    used to bury in a separate panel."""

    def test_hold_seconds_basic(self):
        now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
        opened = now - timedelta(hours=2, minutes=30)
        p = {"ticker": "NVDA", "type": "stock", "qty": 1.0,
             "avg_cost": 100.0, "current_price": 100.0,
             "opened_at": opened.isoformat()}
        out = _enrich_open_position(p, _now=now)
        # 2h30m = 9000 seconds
        assert out["hold_seconds"] == 9000
        # 9000 / 86400 = 0.1042 days
        assert out["hold_days"] == pytest.approx(0.1042, abs=1e-4)

    def test_hold_seconds_with_z_suffix(self):
        # ISO with Z suffix (UTC abbreviated) — same format the store uses.
        now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
        p = {"ticker": "NVDA", "type": "stock", "qty": 1.0,
             "avg_cost": 100.0, "current_price": 100.0,
             "opened_at": "2026-05-22T12:00:00Z"}
        out = _enrich_open_position(p, _now=now)
        assert out["hold_seconds"] == 86400
        assert out["hold_days"] == 1.0

    def test_naive_opened_at_treated_as_utc(self):
        # A timezone-naive ISO string must be parsed as UTC (matches the
        # store's behavior — _now() writes a timezone-aware UTC string but
        # legacy data may be naive).
        now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
        p = {"ticker": "NVDA", "type": "stock", "qty": 1.0,
             "avg_cost": 100.0, "current_price": 100.0,
             "opened_at": "2026-05-23T11:00:00"}
        out = _enrich_open_position(p, _now=now)
        assert out["hold_seconds"] == 3600

    def test_future_opened_at_clamps_to_zero(self):
        # Clock stepped back (documented NTP correction hazard) — a future
        # opened_at must not render as a negative hold time.
        now = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
        future = (now + timedelta(hours=1)).isoformat()
        p = {"ticker": "NVDA", "type": "stock", "qty": 1.0,
             "avg_cost": 100.0, "current_price": 100.0,
             "opened_at": future}
        out = _enrich_open_position(p, _now=now)
        assert out["hold_seconds"] == 0
        assert out["hold_days"] == 0.0

    def test_unparseable_opened_at_yields_none(self):
        p = {"ticker": "NVDA", "type": "stock", "qty": 1.0,
             "avg_cost": 100.0, "current_price": 100.0,
             "opened_at": "not-a-date"}
        out = _enrich_open_position(p)
        assert out["hold_seconds"] is None
        assert out["hold_days"] is None

    def test_missing_opened_at_yields_none(self):
        p = {"ticker": "NVDA", "type": "stock", "qty": 1.0,
             "avg_cost": 100.0, "current_price": 100.0,
             "opened_at": None}
        out = _enrich_open_position(p)
        assert out["hold_seconds"] is None
        assert out["hold_days"] is None


class TestApiStateExposesEnrichedFields:
    """End-to-end through Flask test client — the enriched fields must
    appear on ``/api/state`` positions so the digital-intern dashboard and
    any script consumer can read them."""

    def test_api_state_position_carries_pl_pct_and_market_value(
            self, fresh_store, client):
        # Open a real lot, mark it, then hit /api/state via test client.
        fresh_store.upsert_position("NVDA", "stock", 3.0, 200.0)
        # Apply a mark — current_price=210 → pl_pct = +5.0%
        pos = fresh_store.open_positions()[0]
        fresh_store.update_position_marks({pos["id"]: (210.0, 30.0)})

        r = client.get("/api/state?fresh=1")  # bypass SWR cache for the test
        assert r.status_code == 200, r.data
        body = r.get_json()
        # If SWR is warming, retry without the fresh hint
        positions = body.get("positions") if isinstance(body, dict) else None
        if not positions:
            r = client.get("/api/state")
            body = r.get_json()
            positions = body.get("positions") or []
        assert len(positions) == 1
        p = positions[0]
        assert p["ticker"] == "NVDA"
        assert p["pl_pct"] == pytest.approx(5.0, rel=1e-3)
        assert p["market_value"] == pytest.approx(630.0, rel=1e-3)
        # hold_seconds/hold_days are derived from opened_at which the store
        # writes at insert time; they should be non-negative integers/floats.
        assert isinstance(p["hold_seconds"], int) and p["hold_seconds"] >= 0
        assert isinstance(p["hold_days"], float) and p["hold_days"] >= 0.0
