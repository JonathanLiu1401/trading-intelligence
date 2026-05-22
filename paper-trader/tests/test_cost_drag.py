"""Tests for paper_trader/ml/cost_drag.py — transaction-cost drag audit.

Every assertion pins exact arithmetic against a hand-built synthetic
``backtest.db`` (the AGENTS.md "tests must catch logic bugs, not just run"
discipline). A wrong cost formula, a turnover denominator that uses start
value instead of mean equity, a verdict that fires off the wrong bps level,
or a failed/empty-curve run leaking into the corpus medians all fail here.
Fully offline — no yfinance, no network.
"""
from __future__ import annotations

from datetime import date

import pytest

from paper_trader.backtest import BacktestStore
from paper_trader.ml import cost_drag


# ─────────────────────────── helpers ───────────────────────────

def _make_run(store: BacktestStore, run_id: int, *, final_value: float,
              spy_return_pct: float, trade_values: list[float],
              start: date = date(2020, 1, 1), end: date = date(2021, 1, 1),
              status: str = "complete",
              equity_values: list[float] | None = None) -> None:
    """Insert one backtest run + its trades into the synthetic store.

    `trade_values` is the list of per-trade notionals (qty*price) — the test
    controls Σ notional exactly. `record_trade` stores value = qty*price, so
    we pass qty=value, price=1.0 to make value land on the requested number.
    """
    store.upsert_run(run_id, seed=run_id, status="running", start=start, end=end)
    for i, v in enumerate(trade_values):
        store.record_trade(run_id, start.isoformat(), "NVDA", "BUY",
                            qty=v, price=1.0, reason=f"t{i}")
    curve = [{"date": start.isoformat(), "value": ev}
             for ev in (equity_values if equity_values is not None
                        else [1000.0, final_value])]
    store.finalize_run(run_id, final_value=final_value,
                       spy_return_pct=spy_return_pct,
                       n_trades=len(trade_values), n_decisions=len(trade_values),
                       equity_curve=curve, status=status)


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "backtest.db"


# ─────────────────────── pure-helper arithmetic ───────────────────────

class TestPureHelpers:
    def test_median_odd(self):
        assert cost_drag._median([3.0, 1.0, 2.0]) == 2.0

    def test_median_even(self):
        assert cost_drag._median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_median_empty_is_none(self):
        assert cost_drag._median([]) is None

    def test_median_drops_none(self):
        assert cost_drag._median([None, 4.0, 2.0]) == 3.0

    def test_years_between_one_year(self):
        # 2020 is a leap year → 366 days / 365.25.
        y = cost_drag._years_between("2020-01-01", "2021-01-01")
        assert y == pytest.approx(366 / 365.25)

    def test_years_between_non_positive_span_is_none(self):
        assert cost_drag._years_between("2021-01-01", "2020-01-01") is None
        assert cost_drag._years_between("2020-01-01", "2020-01-01") is None

    def test_years_between_unparseable_is_none(self):
        assert cost_drag._years_between("not-a-date", "2021-01-01") is None
        assert cost_drag._years_between(None, None) is None


# ─────────────────────── exact per-run arithmetic ───────────────────────

class TestPerRunArithmetic:
    def test_cost_and_turnover_exact(self, db):
        """One run, 3 trades × $1000 notional = $3000 traded. final=$2000,
        spy=30% ⇒ vs_spy=70%. equity curve [1000, 2000] ⇒ mean equity 1500.
        Pin every derived number."""
        store = BacktestStore(db)
        _make_run(store, 1, final_value=2000.0, spy_return_pct=30.0,
                  trade_values=[1000.0, 1000.0, 1000.0])
        # 11 filler robust runs so the corpus passes the MIN_RUNS gate and the
        # `runs` list is fully populated (verdict path exercised separately).
        for rid in range(2, 13):
            _make_run(store, rid, final_value=2000.0, spy_return_pct=30.0,
                      trade_values=[1.0, 1.0])
        store.conn.close()

        rep = cost_drag.analyze(db_path=db)
        assert rep["status"] == "ok"
        run = next(r for r in rep["runs"] if r["run_id"] == 1)

        assert run["traded_notional_usd"] == 3000.0
        assert run["mean_equity_usd"] == 1500.0
        assert run["years"] == pytest.approx(366 / 365.25, abs=1e-3)
        assert run["vs_spy_pct"] == 70.0
        # turnover = 3000 / 1500 / (366/365.25) = 1.9959…
        assert run["turnover_annualized"] == pytest.approx(1.996, abs=1e-3)

        # cost = notional * bps/10000 ; cost_pct = cost / start_value(1000) * 100
        c2 = run["cost"]["2"]
        assert c2["total_cost_usd"] == 0.6           # 3000 * 2/10000
        assert c2["cost_pct"] == 0.06                # 0.6 / 1000 * 100
        assert c2["cost_adjusted_vs_spy_pct"] == 69.94  # 70 - 0.06

        c10 = run["cost"]["10"]
        assert c10["total_cost_usd"] == 3.0          # 3000 * 10/10000
        assert c10["cost_pct"] == 0.3
        assert c10["cost_adjusted_vs_spy_pct"] == 69.7   # 70 - 0.3

    def test_turnover_uses_mean_equity_not_start_value(self, db):
        """A run whose equity curve averages well above start value must show
        a SMALLER turnover than the same notional measured against $1000 —
        catches a denominator regressed to start_value."""
        store = BacktestStore(db)
        _make_run(store, 1, final_value=5000.0, spy_return_pct=0.0,
                  trade_values=[2000.0],
                  equity_values=[1000.0, 9000.0])  # mean equity = 5000
        for rid in range(2, 13):
            _make_run(store, rid, final_value=2000.0, spy_return_pct=10.0,
                      trade_values=[1.0])
        store.conn.close()

        rep = cost_drag.analyze(db_path=db)
        run = next(r for r in rep["runs"] if r["run_id"] == 1)
        assert run["mean_equity_usd"] == 5000.0
        # turnover = 2000 / 5000 / (366/365.25) ≈ 0.399 — NOT 2000/1000=2.0
        assert run["turnover_annualized"] == pytest.approx(0.399, abs=1e-3)


# ─────────────────────── corpus verdict logic ───────────────────────

class TestVerdicts:
    def test_cost_robust_when_median_survives_max_bps(self, db):
        """12 runs, vs_spy=70%, tiny notional ⇒ negligible cost ⇒ median
        cost-adjusted alpha stays positive at 10bps ⇒ COST_ROBUST."""
        store = BacktestStore(db)
        for rid in range(1, 13):
            _make_run(store, rid, final_value=2000.0, spy_return_pct=30.0,
                      trade_values=[10.0, 10.0])
        store.conn.close()

        rep = cost_drag.analyze(db_path=db)
        assert rep["verdict"] == "COST_ROBUST"
        assert rep["n_runs"] == 12
        assert rep["corpus"]["median_vs_spy_pct"] == 70.0
        # 12 trades × $10 = $20 notional; 10bps ⇒ $0.02 ⇒ 0.002pp — negligible.
        assert rep["corpus"]["per_bps"]["10"][
            "median_cost_adjusted_vs_spy_pct"] == pytest.approx(70.0, abs=0.01)
        assert rep["corpus"]["per_bps"]["10"]["frac_runs_below_spy"] == 0.0

    def test_cost_fragile_when_max_bps_flips_median_negative(self, db):
        """vs_spy=+1% but $11k traded notional ⇒ at 10bps cost_pct=1.1pp ⇒
        cost-adjusted alpha = -0.1% ⇒ median negative ⇒ COST_FRAGILE."""
        store = BacktestStore(db)
        for rid in range(1, 13):
            # final=1010 ⇒ total_return 1%; spy=0 ⇒ vs_spy=+1%.
            _make_run(store, rid, final_value=1010.0, spy_return_pct=0.0,
                      trade_values=[1000.0] * 11)  # $11,000 notional
        store.conn.close()

        rep = cost_drag.analyze(db_path=db)
        assert rep["verdict"] == "COST_FRAGILE"
        assert rep["corpus"]["median_vs_spy_pct"] == 1.0
        cell10 = rep["corpus"]["per_bps"]["10"]
        # 11000 * 10/10000 = $11 ⇒ 11/1000*100 = 1.1pp ⇒ 1.0 - 1.1 = -0.1
        assert cell10["median_cost_adjusted_vs_spy_pct"] == pytest.approx(
            -0.1, abs=1e-6)
        assert cell10["frac_runs_below_spy"] == 1.0

    def test_cost_negative_when_raw_median_already_below_spy(self, db):
        """Runs that already trail SPY before any cost ⇒ COST_NEGATIVE."""
        store = BacktestStore(db)
        for rid in range(1, 13):
            # final=1010 ⇒ +1% return; spy=20% ⇒ vs_spy = -19%.
            _make_run(store, rid, final_value=1010.0, spy_return_pct=20.0,
                      trade_values=[10.0])
        store.conn.close()

        rep = cost_drag.analyze(db_path=db)
        assert rep["verdict"] == "COST_NEGATIVE"
        assert rep["corpus"]["median_vs_spy_pct"] == -19.0

    def test_insufficient_data_below_min_runs(self, db):
        store = BacktestStore(db)
        for rid in range(1, 5):  # only 4 < MIN_RUNS
            _make_run(store, rid, final_value=2000.0, spy_return_pct=10.0,
                      trade_values=[10.0])
        store.conn.close()

        rep = cost_drag.analyze(db_path=db)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_runs"] == 4


# ─────────────────────── run-exclusion correctness ───────────────────────

class TestRunExclusion:
    def test_non_complete_runs_excluded(self, db):
        """A 'running'/'failed' run must never enter the corpus."""
        store = BacktestStore(db)
        for rid in range(1, 13):
            _make_run(store, rid, final_value=2000.0, spy_return_pct=10.0,
                      trade_values=[10.0])
        _make_run(store, 99, final_value=9999.0, spy_return_pct=0.0,
                  trade_values=[100.0], status="failed")
        store.conn.close()

        rep = cost_drag.analyze(db_path=db)
        assert rep["n_runs"] == 12
        assert all(r["run_id"] != 99 for r in rep["runs"])

    def test_empty_equity_curve_run_excluded(self, db):
        """A completed run with an empty equity curve (the orphaned/manual
        run_id 99001/90001 case) is not costable — must be skipped, never
        divide-by-zero on mean equity."""
        store = BacktestStore(db)
        for rid in range(1, 13):
            _make_run(store, rid, final_value=2000.0, spy_return_pct=10.0,
                      trade_values=[10.0])
        # n_trades>0 but equity curve empty.
        store.upsert_run(900, seed=900, status="running",
                         start=date(2020, 1, 1), end=date(2021, 1, 1))
        store.record_trade(900, "2020-01-01", "NVDA", "BUY", 5.0, 1.0, "x")
        store.finalize_run(900, final_value=1500.0, spy_return_pct=0.0,
                           n_trades=1, n_decisions=1, equity_curve=[],
                           status="complete")
        store.conn.close()

        rep = cost_drag.analyze(db_path=db)
        assert rep["status"] == "ok"
        assert all(r["run_id"] != 900 for r in rep["runs"])

    def test_zero_trade_run_excluded(self, db):
        """status=complete with n_trades=0 is filtered by SQL — confirm it
        never reaches the corpus even though it has an equity curve."""
        store = BacktestStore(db)
        for rid in range(1, 13):
            _make_run(store, rid, final_value=2000.0, spy_return_pct=10.0,
                      trade_values=[10.0])
        store.upsert_run(800, seed=800, status="running",
                         start=date(2020, 1, 1), end=date(2021, 1, 1))
        store.finalize_run(800, final_value=1000.0, spy_return_pct=0.0,
                           n_trades=0, n_decisions=0,
                           equity_curve=[{"date": "2020-01-01", "value": 1000.0}],
                           status="complete")
        store.conn.close()

        rep = cost_drag.analyze(db_path=db)
        assert all(r["run_id"] != 800 for r in rep["runs"])


# ─────────────────────── robustness ───────────────────────

class TestRobustness:
    def test_missing_db_degrades_gracefully(self, tmp_path):
        rep = cost_drag.analyze(db_path=tmp_path / "nope.db")
        assert rep["status"] == "error"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "no backtest db" in rep["hint"]

    def test_custom_bps_levels_respected(self, db):
        store = BacktestStore(db)
        for rid in range(1, 13):
            _make_run(store, rid, final_value=2000.0, spy_return_pct=30.0,
                      trade_values=[1000.0])
        store.conn.close()

        rep = cost_drag.analyze(db_path=db, bps_levels=(25.0,))
        assert rep["bps_levels"] == [25.0]
        # $1000 notional × 25bps = $2.50 ⇒ 0.25pp ⇒ 70 - 0.25 = 69.75
        run = rep["runs"][0]
        assert run["cost"]["25"]["cost_adjusted_vs_spy_pct"] == 69.75

    def test_runs_sorted_by_turnover_desc(self, db):
        store = BacktestStore(db)
        # rid 1 churns hardest (largest notional), rest minimal.
        _make_run(store, 1, final_value=2000.0, spy_return_pct=10.0,
                  trade_values=[5000.0])
        for rid in range(2, 13):
            _make_run(store, rid, final_value=2000.0, spy_return_pct=10.0,
                      trade_values=[1.0])
        store.conn.close()

        rep = cost_drag.analyze(db_path=db)
        turnovers = [r["turnover_annualized"] for r in rep["runs"]]
        assert turnovers == sorted(turnovers, reverse=True)
        assert rep["runs"][0]["run_id"] == 1
