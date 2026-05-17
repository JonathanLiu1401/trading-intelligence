"""Regression locks for two previously-zero-coverage ML/backtest seams.

2026-05-16 ML/backtest review pass (9th consecutive — no new core bugs;
this pass's honest contribution is closing two real coverage gaps found by
grepping every backtest symbol against tests/):

1. ``BacktestStore.run_detail`` — the read view behind
   ``GET /api/backtests/<run_id>``. Its siblings ``all_runs`` /
   ``run_curves`` were locked in ``test_ml_backtest_store_views.py`` last
   pass, but ``run_detail`` had **zero** direct coverage despite real
   logic: missing-run → ``None`` (not raise / not ``{}``), the
   ``(sim_date ASC, id ASC)`` ordering on both child tables (a
   ``sim_date DESC`` or ``id DESC`` regression silently scrambles the
   dashboard's trade/decision tables), and the corrupt-``equity_curve_json``
   → ``[]`` degradation (a raise here 500s the endpoint).

2. ``backtest._sell`` (the ``SimPortfolio`` mutator, **not**
   ``strategy._sell``) — every backtest SELL, stop-loss and take-profit
   exit routes through it, yet it had **zero** direct unit coverage (only
   transitive exercise via ``_enforce_risk_exits`` / ``_execute_decision``
   tests, which clamp qty themselves before calling it, so its own
   over-sell clamp and the ``pos["qty"] <= 1e-6`` deletion boundary were
   never asserted in isolation).

Exact values, not ranges — a formula/ordering/boundary change must update
the literals deliberately. All offline: the conftest ``_isolate_data_dir``
autouse fixture monkeypatches ``backtest.BACKTEST_DB`` to a tmp file, so
``BacktestStore()`` never touches the real persistent backtest.db.
"""
from __future__ import annotations

from datetime import date

import pytest

from paper_trader.backtest import BacktestStore, SimPortfolio, _sell


@pytest.fixture
def store():
    s = BacktestStore()  # honors conftest's monkeypatched BACKTEST_DB
    try:
        yield s
    finally:
        s.conn.close()


# ─────────────────────────── BacktestStore.run_detail ───────────────────────────


class TestRunDetail:
    def test_missing_run_returns_none(self, store):
        # Must be None — the endpoint relies on this to 404 a bad run_id.
        # A regression that returned {} or raised would 500 instead.
        assert store.run_detail(999) is None

    def test_trades_ordered_sim_date_then_id_asc(self, store):
        store.upsert_run(1, seed=1, status="running",
                         start=date(2025, 1, 1), end=date(2025, 12, 31))
        # Insert deliberately out of sim_date order, with a same-day pair so
        # the id-ASC tiebreak is exercised independently of sim_date sort.
        store.record_trade(1, "2025-01-02", "LATE", "BUY", 1, 10.0, "r")
        store.record_trade(1, "2025-01-01", "EARLY_A", "BUY", 1, 10.0, "r")
        store.record_trade(1, "2025-01-01", "EARLY_B", "BUY", 1, 10.0, "r")
        store.finalize_run(1, 1100.0, 5.0, 3, 0, [{"date": "2025-01-01",
                                                   "value": 1100.0}])
        detail = store.run_detail(1)
        # sim_date ASC → both 01-01 rows first; id ASC → insertion order
        # within the day (EARLY_A before EARLY_B); 01-02 row last.
        # A sim_date DESC regression → ["LATE", ...]; an id DESC tiebreak
        # regression → [..., "EARLY_B", "EARLY_A", ...]. Either fails here.
        assert [t["ticker"] for t in detail["trades"]] == [
            "EARLY_A", "EARLY_B", "LATE"]

    def test_decisions_ordered_sim_date_then_id_asc(self, store):
        store.upsert_run(2, seed=1, status="running",
                         start=date(2025, 1, 1), end=date(2025, 12, 31))
        store.record_decision(2, "2025-03-01",
                              {"action": "HOLD", "ticker": "Z"},
                              "HOLD", "d", 1000.0, 1000.0, 0)
        store.record_decision(2, "2025-02-01",
                              {"action": "BUY", "ticker": "A1"},
                              "FILLED", "d", 900.0, 1000.0, 1)
        store.record_decision(2, "2025-02-01",
                              {"action": "SELL", "ticker": "A2"},
                              "FILLED", "d", 950.0, 1000.0, 1)
        store.finalize_run(2, 1000.0, 0.0, 0, 3, [])
        detail = store.run_detail(2)
        assert [d["ticker"] for d in detail["decisions"]] == ["A1", "A2", "Z"]

    def test_corrupt_equity_curve_json_degrades_to_empty_list(self, store):
        store.upsert_run(3, seed=1, status="running",
                         start=date(2025, 1, 1), end=date(2025, 12, 31))
        store.finalize_run(3, 1000.0, 0.0, 0, 0, [])
        # Corrupt the column directly (finalize_run always json.dumps valid
        # JSON, so the only way to reach the degradation branch is to plant
        # a torn value). Hold the store lock — the connection is shared.
        with store._lock:
            store.conn.execute(
                "UPDATE backtest_runs SET equity_curve_json=? WHERE run_id=?",
                ("{not valid json", 3),
            )
            store.conn.commit()
        detail = store.run_detail(3)
        # Degrades, never raises — a raise here 500s /api/backtests/<id>.
        assert detail["equity_curve"] == []

    def test_valid_equity_curve_round_trips(self, store):
        curve = [{"date": "2025-01-01", "value": 1000.0, "cash": 1000.0},
                 {"date": "2025-06-01", "value": 1250.5, "cash": 12.0}]
        store.upsert_run(4, seed=1, status="running",
                         start=date(2025, 1, 1), end=date(2025, 12, 31))
        store.finalize_run(4, 1250.5, 8.0, 2, 4, curve)
        detail = store.run_detail(4)
        assert detail["equity_curve"] == curve
        assert detail["run_id"] == 4
        assert detail["status"] == "complete"


# ─────────────────────────── backtest._sell ───────────────────────────


class TestSell:
    def test_no_position_returns_zero_and_no_mutation(self):
        p = SimPortfolio(cash=500.0)
        proceeds = _sell(p, "NVDA", 3.0, 100.0)
        assert proceeds == 0.0
        assert p.cash == 500.0
        assert p.positions == {}

    def test_partial_sell_keeps_position_avg_cost_unchanged(self):
        p = SimPortfolio(cash=0.0)
        p.positions["NVDA"] = {"qty": 10.0, "avg_cost": 50.0,
                               "stop_loss": None, "take_profit": None}
        proceeds = _sell(p, "NVDA", 4.0, 60.0)
        assert proceeds == 240.0          # 4 * 60, exact (no rounding in _sell)
        assert p.cash == 240.0            # cash credited == proceeds exactly
        assert p.positions["NVDA"]["qty"] == 6.0
        assert p.positions["NVDA"]["avg_cost"] == 50.0   # entry basis untouched

    def test_oversell_clamps_to_held_and_closes_position(self):
        p = SimPortfolio(cash=0.0)
        p.positions["MU"] = {"qty": 5.0, "avg_cost": 30.0,
                             "stop_loss": None, "take_profit": None}
        proceeds = _sell(p, "MU", 8.0, 20.0)   # ask 8, only 5 held
        assert proceeds == 100.0               # 5 * 20, not 8 * 20
        assert p.cash == 100.0
        assert "MU" not in p.positions         # zeroed → deleted

    def test_residual_at_or_below_1e6_epsilon_is_deleted(self):
        # 1.0000001 - 1.0 ≈ 1e-7 ≤ 1e-6 → position removed (no dust lots).
        p = SimPortfolio(cash=0.0)
        p.positions["AMD"] = {"qty": 1.0000001, "avg_cost": 10.0,
                              "stop_loss": None, "take_profit": None}
        _sell(p, "AMD", 1.0, 100.0)
        assert "AMD" not in p.positions

    def test_residual_above_1e6_epsilon_is_kept(self):
        # 1.00001 - 1.0 ≈ 1e-5 > 1e-6 → position survives with the residual.
        p = SimPortfolio(cash=0.0)
        p.positions["AMD"] = {"qty": 1.00001, "avg_cost": 10.0,
                              "stop_loss": None, "take_profit": None}
        _sell(p, "AMD", 1.0, 100.0)
        assert "AMD" in p.positions
        assert p.positions["AMD"]["qty"] == pytest.approx(1e-5, abs=1e-9)
