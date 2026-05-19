"""Regression lock: ``/api/portfolio`` is the lean public surface consumed by
Digital Intern's cross-port fetch. The endpoint:

  1. **MUST** preserve the legacy three-key contract (``total_value``,
     ``cash``, ``starting_value``) so existing consumers never break.
  2. Now additionally exposes at-a-glance trader-actionable fields
     (``n_positions``, ``unrealized_pl``, ``stale_marks``, …) composed
     from the already-cached ``portfolio.positions_json`` row so the
     endpoint remains the lowest-latency public surface.

These tests assert specific computed values against a real ``Store``
populated with known marks — not just "no crash" — so a regression in
the aggregation arithmetic (a wrong sign, a denominator swap, an
isinstance-blind bool falling through ``unrealized_pl``) fails loudly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard
from paper_trader import store as store_mod
from paper_trader.store import INITIAL_CASH, Store


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    """A real Store backed by a temp DB — mirrors test_core_strategy.py /
    test_core_reporter.py. The dashboard's ``get_store`` is repointed to it."""
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


def _get_json(client, path):
    r = client.get(path)
    assert r.status_code == 200, r.data
    return r.get_json()


class TestPortfolioApiLegacyContract:
    """The original three-key payload is consumed by Digital Intern's
    dashboard — it MUST keep those exact keys and shapes so a cross-port
    deploy mismatch can't silently dark-out that panel."""

    def test_fresh_store_returns_legacy_keys_with_starting_value(
            self, fresh_store, client):
        body = _get_json(client, "/api/portfolio")
        # Legacy contract — these three keys must always be present.
        assert "total_value" in body
        assert "cash" in body
        assert "starting_value" in body
        # Fresh book: cash == total_value == INITIAL_CASH, no positions.
        assert body["cash"] == pytest.approx(INITIAL_CASH)
        assert body["total_value"] == pytest.approx(INITIAL_CASH)
        assert body["starting_value"] == INITIAL_CASH

    def test_cors_header_present_for_cross_port_fetch(
            self, fresh_store, client):
        """Digital Intern (port 8080) fetches this from the browser; without
        the wildcard CORS header the cross-port read fails silently."""
        r = client.get("/api/portfolio")
        assert r.headers.get("Access-Control-Allow-Origin") == "*"


class TestPortfolioApiEnrichedFields:
    """The new at-a-glance fields are composed *from the cached
    positions_json row only* — no extra store reads."""

    @staticmethod
    def _seed_book(store: Store, cash: float, positions: list[dict]) -> None:
        """Write a snapshot directly to the portfolio row, mirroring what
        ``_portfolio_snapshot`` does post-trade."""
        open_value = sum((p.get("market_value") or 0.0) for p in positions)
        total = cash + open_value
        store.update_portfolio(cash, total, positions)

    def test_n_positions_and_open_value_summed_from_cached_marks(
            self, fresh_store, client):
        positions = [
            {"ticker": "NVDA", "type": "stock", "qty": 2,
             "avg_cost": 100.0, "current_price": 110.0,
             "market_value": 220.0, "unrealized_pl": 20.0,
             "pl_pct": 10.0, "stale_mark": False},
            {"ticker": "TQQQ", "type": "stock", "qty": 3,
             "avg_cost": 50.0, "current_price": 60.0,
             "market_value": 180.0, "unrealized_pl": 30.0,
             "pl_pct": 20.0, "stale_mark": False},
        ]
        self._seed_book(fresh_store, cash=600.0, positions=positions)

        body = _get_json(client, "/api/portfolio")
        assert body["n_positions"] == 2
        assert body["open_value"] == pytest.approx(400.0)
        assert body["cash"] == pytest.approx(600.0)
        # total_value = cash + open_value, persisted by update_portfolio
        assert body["total_value"] == pytest.approx(1000.0)

    def test_unrealized_pl_signed_sum(self, fresh_store, client):
        """The book-wide P/L is the SIGNED sum across positions: a +20 mover
        and a −5 mover net to +15. An unsigned implementation would mask
        a losing position behind a profitable one."""
        positions = [
            {"ticker": "NVDA", "type": "stock", "qty": 2,
             "avg_cost": 100.0, "current_price": 110.0,
             "market_value": 220.0, "unrealized_pl": 20.0,
             "stale_mark": False},
            {"ticker": "LITE", "type": "stock", "qty": 1,
             "avg_cost": 80.0, "current_price": 75.0,
             "market_value": 75.0, "unrealized_pl": -5.0,
             "stale_mark": False},
        ]
        self._seed_book(fresh_store, cash=705.0, positions=positions)

        body = _get_json(client, "/api/portfolio")
        assert body["unrealized_pl"] == pytest.approx(15.0)
        # 15 / 1000 = 1.5%
        assert body["unrealized_pl_pct"] == pytest.approx(1.5)

    def test_unrealized_pl_pct_uses_total_value_denominator(
            self, fresh_store, client):
        """% is over the equity baseline (= cash + open_value), aligning
        with /api/benchmark / drawdown framing. A divide-by-open-value
        bug would inflate the % whenever cash is high."""
        positions = [
            {"ticker": "NVDA", "type": "stock", "qty": 1,
             "avg_cost": 100.0, "current_price": 110.0,
             "market_value": 110.0, "unrealized_pl": 10.0,
             "stale_mark": False},
        ]
        self._seed_book(fresh_store, cash=890.0, positions=positions)

        body = _get_json(client, "/api/portfolio")
        # total_value = 890 + 110 = 1000. pl 10 / 1000 = 1.00%.
        # If denominator were open_value (110), pct would be 9.09% — wrong.
        assert body["total_value"] == pytest.approx(1000.0)
        assert body["unrealized_pl_pct"] == pytest.approx(1.0)

    def test_stale_marks_counts_only_flagged_positions(
            self, fresh_store, client):
        positions = [
            {"ticker": "NVDA", "type": "stock", "qty": 1,
             "avg_cost": 100.0, "current_price": 100.0,
             "market_value": 100.0, "unrealized_pl": 0.0,
             "stale_mark": True},
            {"ticker": "TQQQ", "type": "stock", "qty": 1,
             "avg_cost": 50.0, "current_price": 60.0,
             "market_value": 60.0, "unrealized_pl": 10.0,
             "stale_mark": False},
            {"ticker": "LITE", "type": "stock", "qty": 1,
             "avg_cost": 80.0, "current_price": 80.0,
             "market_value": 80.0, "unrealized_pl": 0.0,
             "stale_mark": True},
        ]
        self._seed_book(fresh_store, cash=760.0, positions=positions)
        body = _get_json(client, "/api/portfolio")
        assert body["stale_marks"] == 2  # NVDA + LITE, NOT TQQQ

    def test_pnl_vs_start_absolute_and_pct(self, fresh_store, client):
        """The book-wide drift vs the $1000 baseline (INITIAL_CASH) — the
        single number a trader compares against the index. Up scenario."""
        positions = [
            {"ticker": "NVDA", "type": "stock", "qty": 1,
             "avg_cost": 100.0, "current_price": 200.0,
             "market_value": 200.0, "unrealized_pl": 100.0,
             "stale_mark": False},
        ]
        self._seed_book(fresh_store, cash=900.0, positions=positions)

        body = _get_json(client, "/api/portfolio")
        # total_value = 900 + 200 = 1100.  vs 1000 baseline = +100 / +10.0%.
        assert body["pnl_vs_start"] == pytest.approx(100.0)
        assert body["pnl_vs_start_pct"] == pytest.approx(10.0)

    def test_pnl_vs_start_negative_when_underwater(
            self, fresh_store, client):
        positions = [
            {"ticker": "X", "type": "stock", "qty": 1,
             "avg_cost": 200.0, "current_price": 100.0,
             "market_value": 100.0, "unrealized_pl": -100.0,
             "stale_mark": False},
        ]
        self._seed_book(fresh_store, cash=750.0, positions=positions)
        body = _get_json(client, "/api/portfolio")
        # total_value = 850. vs 1000 baseline = -150 / -15.0%.
        assert body["pnl_vs_start"] == pytest.approx(-150.0)
        assert body["pnl_vs_start_pct"] == pytest.approx(-15.0)

    def test_last_updated_is_iso_timestamp_from_portfolio_row(
            self, fresh_store, client):
        positions = [
            {"ticker": "NVDA", "type": "stock", "qty": 1,
             "avg_cost": 100.0, "current_price": 100.0,
             "market_value": 100.0, "unrealized_pl": 0.0,
             "stale_mark": False},
        ]
        self._seed_book(fresh_store, cash=900.0, positions=positions)
        body = _get_json(client, "/api/portfolio")
        # `last_updated` must be present and parseable as an ISO timestamp
        # — Digital Intern's UI uses it as a freshness indicator.
        assert isinstance(body["last_updated"], str)
        from datetime import datetime
        # Should not raise.
        datetime.fromisoformat(body["last_updated"].replace("Z", "+00:00"))


class TestPortfolioApiEmptyAndDegradeSafe:
    """The endpoint must NEVER raise — a malformed positions row degrades
    to safe zeros while preserving the legacy three keys, mirroring the
    "never blink the public surface" reporter discipline."""

    def test_empty_book_has_zero_positions_and_zero_pl(
            self, fresh_store, client):
        body = _get_json(client, "/api/portfolio")
        assert body["n_positions"] == 0
        assert body["open_value"] == 0.0
        assert body["unrealized_pl"] == 0.0
        assert body["stale_marks"] == 0
        # No positions → pl_pct of total_value=INITIAL_CASH is exactly 0.
        assert body["unrealized_pl_pct"] == pytest.approx(0.0)
        # vs baseline: 0 (fresh book is at the start by construction).
        assert body["pnl_vs_start"] == pytest.approx(0.0)
        assert body["pnl_vs_start_pct"] == pytest.approx(0.0)

    def test_corrupt_positions_json_falls_back_to_empty_list(
            self, fresh_store, client):
        """``store.get_portfolio`` already returns ``positions=[]`` when
        ``positions_json`` is empty/NULL; we additionally guard against a
        non-list shape here so a future schema drift doesn't 500 this
        endpoint."""
        # Directly inject a non-list JSON into the row to exercise the
        # endpoint's defensive `if not isinstance(positions, list)` arm.
        fresh_store.conn.execute(
            "UPDATE portfolio SET positions_json=? WHERE id=1",
            ('{"corrupt": "not-a-list"}',),
        )
        fresh_store.conn.commit()
        # `store.get_portfolio` json.loads this; whatever shape it returns
        # must NOT make the endpoint crash. The endpoint MUST still return
        # 200 with safe defaults.
        body = _get_json(client, "/api/portfolio")
        assert body["n_positions"] == 0
        assert body["open_value"] == 0.0
        assert body["unrealized_pl"] == 0.0

    def test_non_numeric_unrealized_pl_does_not_crash(
            self, fresh_store, client):
        """A single bad row must NOT take down the aggregate — coerce
        defensively and continue."""
        positions = [
            {"ticker": "GOOD", "type": "stock", "qty": 1,
             "avg_cost": 100.0, "current_price": 110.0,
             "market_value": 110.0, "unrealized_pl": 10.0,
             "stale_mark": False},
            {"ticker": "BAD", "type": "stock", "qty": 1,
             "avg_cost": 100.0, "current_price": None,
             "market_value": None, "unrealized_pl": "n/a",
             "stale_mark": False},
        ]
        fresh_store.conn.execute(
            "UPDATE portfolio SET positions_json=?, cash=?, total_value=? "
            "WHERE id=1",
            (json.dumps(positions), 890.0, 1000.0),
        )
        fresh_store.conn.commit()
        body = _get_json(client, "/api/portfolio")
        assert body["n_positions"] == 2  # both still counted
        # GOOD contributes 10.0; BAD coerces to 0.0.
        assert body["unrealized_pl"] == pytest.approx(10.0)
        # GOOD contributes 110; BAD coerces to 0.
        assert body["open_value"] == pytest.approx(110.0)

    def test_zero_total_value_yields_null_pl_pct_not_div_zero(
            self, fresh_store, client):
        """A total_value of 0 is non-physical for the live book (cash never
        actually goes to literal zero), but the public surface must NOT
        ZeroDivisionError on the synthetic case — degrade to None."""
        fresh_store.conn.execute(
            "UPDATE portfolio SET cash=0, total_value=0 WHERE id=1"
        )
        fresh_store.conn.commit()
        body = _get_json(client, "/api/portfolio")
        # pct is None (not 0.0 — None signals "no meaningful denominator")
        assert body["unrealized_pl_pct"] is None
