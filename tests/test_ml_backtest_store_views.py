"""Regression locks for BacktestStore's dashboard-facing read views.

2026-05-16 ML/backtest review pass. `BacktestStore.run_curves` and the
`annualized_return_pct` / `duration_days` arithmetic inside
`BacktestStore.all_runs` had **zero** direct test coverage (verified by
grepping every symbol in tests/), yet both feed user-visible numbers:

- `all_runs`  → `/api/backtests` (the run grid + the annualized-return column)
- `run_curves`→ `/api/backtests/compare` (the normalized equity overlay)

A silent regression here — a flipped sign, a dropped `1.0/years` exponent,
a `365` vs `365.25` day-count, the `value_pct` losing its `-1.0` offset, or
the malformed-JSON / bad-date degradation paths throwing instead of
degrading — would corrupt the dashboard without failing any existing test
and without crashing the continuous loop (these are read-only views). These
tests assert **exact** values, not ranges, by design: if you change the
normalization formula, update the expected literals deliberately.

All offline: the conftest `_isolate_data_dir` autouse fixture monkeypatches
`backtest.BACKTEST_DB` to a tmp file, so `BacktestStore()` never touches the
real persistent backtest.db.
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from paper_trader.backtest import BacktestStore, INITIAL_CASH


@pytest.fixture
def store():
    s = BacktestStore()  # honors conftest's monkeypatched BACKTEST_DB
    try:
        yield s
    finally:
        s.conn.close()


# ─────────────────────────── all_runs() ───────────────────────────


class TestAllRunsDerivedMetrics:
    def test_duration_days_is_exact_calendar_delta(self, store):
        # 2024 is a leap year → 2024-01-01 .. 2025-01-01 is exactly 366 days.
        store.upsert_run(1, seed=7, status="running",
                         start=date(2024, 1, 1), end=date(2025, 1, 1))
        store.finalize_run(1, final_value=2000.0, spy_return_pct=10.0,
                           n_trades=3, n_decisions=9, equity_curve=[])
        run = store.all_runs()[0]
        assert run["duration_days"] == 366

    def test_annualized_return_zero_growth_is_exactly_zero(self, store):
        # final == start → growth 1.0 → (1.0 ** anything) - 1 == 0.0 exactly,
        # independent of the window length. Locks the `-1.0` offset: a
        # regression that drops it would report +100.0 here.
        store.upsert_run(2, seed=1, status="running",
                         start=date(2024, 1, 1), end=date(2024, 7, 1))
        store.finalize_run(2, final_value=INITIAL_CASH, spy_return_pct=0.0,
                           n_trades=0, n_decisions=1, equity_curve=[])
        run = store.all_runs()[0]
        assert run["annualized_return_pct"] == 0.0

    def test_annualized_return_compounding_formula(self, store):
        # 2x over 366 days. annualized = growth ** (365.25/duration) - 1.
        # Independent surface form of the production `growth ** (1.0/years)`
        # where years = duration_days / 365.25 — a 365-vs-365.25 divisor
        # regression shifts this past the round-to-3 equality and fails.
        store.upsert_run(3, seed=2, status="running",
                         start=date(2024, 1, 1), end=date(2025, 1, 1))
        store.finalize_run(3, final_value=2000.0, spy_return_pct=5.0,
                           n_trades=1, n_decisions=2, equity_curve=[])
        run = store.all_runs()[0]
        growth = 2000.0 / INITIAL_CASH
        expected = round((math.pow(growth, 365.25 / 366) - 1.0) * 100.0, 3)
        assert run["annualized_return_pct"] == expected
        # And the hand-computed literal, so an algebra change is also caught.
        assert run["annualized_return_pct"] == pytest.approx(99.716, abs=1e-3)

    def test_annualized_is_none_before_finalize(self, store):
        # upsert_run only → final_value defaults to 0.0 (falsy) → the
        # `d.get("final_value")` guard yields annualized None, but
        # duration_days is still computed from the stored dates.
        store.upsert_run(9, seed=3, status="running",
                         start=date(2025, 1, 1), end=date(2025, 1, 31))
        run = store.all_runs()[0]
        assert run["annualized_return_pct"] is None
        assert run["duration_days"] == 30

    def test_runs_ordered_by_run_id_ascending(self, store):
        for rid in (5, 2, 8):
            store.upsert_run(rid, seed=rid, status="running",
                             start=date(2025, 1, 1), end=date(2025, 6, 1))
        assert [r["run_id"] for r in store.all_runs()] == [2, 5, 8]

    def test_include_curves_parses_json_and_bad_json_degrades(self, store):
        store.upsert_run(4, seed=4, status="running",
                         start=date(2025, 1, 1), end=date(2025, 2, 1))
        store.finalize_run(4, final_value=1100.0, spy_return_pct=0.0,
                           n_trades=1, n_decisions=1,
                           equity_curve=[{"date": "2025-01-02",
                                          "value": 1050.0, "cash": 0.0}])
        run = store.all_runs(include_curves=True)[0]
        assert run["equity_curve"] == [
            {"date": "2025-01-02", "value": 1050.0, "cash": 0.0}
        ]
        # Corrupt the stored JSON → the read path must degrade to [] not raise.
        with store._lock:
            store.conn.execute(
                "UPDATE backtest_runs SET equity_curve_json='{not-json' "
                "WHERE run_id=?", (4,))
            store.conn.commit()
        run2 = store.all_runs(include_curves=True)[0]
        assert run2["equity_curve"] == []


# ─────────────────────────── run_curves() ───────────────────────────


class TestRunCurvesNormalization:
    def test_value_pct_and_day_index_exact(self, store):
        store.upsert_run(1, seed=1, status="running",
                         start=date(2025, 1, 1), end=date(2025, 12, 31))
        curve = [
            {"date": "2025-01-01", "value": 1000.0, "cash": 1000.0},
            {"date": "2025-01-11", "value": 1250.0, "cash": 0.0},
            {"date": "2025-02-01", "value": 900.0, "cash": 50.0},
        ]
        store.finalize_run(1, final_value=900.0, spy_return_pct=0.0,
                           n_trades=2, n_decisions=4, equity_curve=curve)
        out = store.run_curves([1])
        pts = out[1]
        # start_value is INITIAL_CASH (1000) from upsert_run.
        assert pts[0] == {"date": "2025-01-01", "day_index": 0,
                          "value": 1000.0, "value_pct": 0.0}
        # 2025-01-11 is 10 calendar days after 2025-01-01; +25%.
        assert pts[1] == {"date": "2025-01-11", "day_index": 10,
                          "value": 1250.0, "value_pct": 25.0}
        # 2025-02-01 is 31 days after start; -10%.
        assert pts[2] == {"date": "2025-02-01", "day_index": 31,
                          "value": 900.0, "value_pct": -10.0}

    def test_unparseable_point_date_keeps_value_but_nulls_day_index(self, store):
        store.upsert_run(2, seed=1, status="running",
                         start=date(2025, 1, 1), end=date(2025, 12, 31))
        store.finalize_run(2, final_value=1100.0, spy_return_pct=0.0,
                           n_trades=1, n_decisions=1,
                           equity_curve=[{"date": "not-a-date",
                                          "value": 1100.0, "cash": 0.0}])
        pt = store.run_curves([2])[2][0]
        assert pt["day_index"] is None
        assert pt["value"] == 1100.0
        assert pt["value_pct"] == 10.0  # still normalized vs start_value

    def test_corrupt_curve_json_degrades_to_empty_list(self, store):
        store.upsert_run(3, seed=1, status="running",
                         start=date(2025, 1, 1), end=date(2025, 12, 31))
        store.finalize_run(3, final_value=1000.0, spy_return_pct=0.0,
                           n_trades=0, n_decisions=1, equity_curve=[])
        with store._lock:
            store.conn.execute(
                "UPDATE backtest_runs SET equity_curve_json='[[[' "
                "WHERE run_id=?", (3,))
            store.conn.commit()
        assert store.run_curves([3]) == {3: []}

    def test_zero_start_value_falls_back_to_1000(self, store):
        store.upsert_run(4, seed=1, status="running",
                         start=date(2025, 1, 1), end=date(2025, 12, 31))
        store.finalize_run(4, final_value=1500.0, spy_return_pct=0.0,
                           n_trades=1, n_decisions=1,
                           equity_curve=[{"date": "2025-01-01",
                                          "value": 1500.0, "cash": 0.0}])
        # Force a falsy start_value to exercise the `float(start_val or 1000.0)`
        # guard — without it a 0 start_value would ZeroDivisionError.
        with store._lock:
            store.conn.execute(
                "UPDATE backtest_runs SET start_value=0 WHERE run_id=?", (4,))
            store.conn.commit()
        pt = store.run_curves([4])[4][0]
        assert pt["value_pct"] == 50.0  # 1500 / 1000 - 1 == +50%

    def test_empty_run_ids_returns_empty_dict(self, store):
        assert store.run_curves([]) == {}
