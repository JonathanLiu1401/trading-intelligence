"""Tests for paper_trader.ml.persona_leaderboard (2026-05-17 ML/backtest
pass #3).

Exact-value verdict + arithmetic locks on hand-constructed deterministic
data with known-correct answers, mirroring tests/test_calibration.py and
tests/test_label_audit_20260517.py. A threshold or classification change
must update these literals deliberately rather than silently shift a
quant-facing strategy-quality diagnostic.

The leaderboard is read-only (no train / no pickle / no jsonl / no DB
write); these tests assert that contract by construction — every dataset
is an in-memory list, and the `_load_runs` DB test opens a throwaway temp
DB strictly `mode=ro`.

`persona_for` is imported by the module (single source of truth); these
tests pin run_id→persona by *using that same function*, so a PERSONAS
reorder updates both sides together and can never silently desync the
historical aggregates.
"""
from __future__ import annotations

import sqlite3

import pytest

from paper_trader.backtest import persona_for
from paper_trader.ml.persona_leaderboard import (
    DRAG_MAX_MEDIAN_VS_SPY,
    EDGE_MIN_MEDIAN_VS_SPY,
    EDGE_MIN_WIN_RATE,
    MIN_RECORDS,
    MIN_RUNS_PER_PERSONA,
    _equity_risk,
    _load_runs,
    persona_leaderboard,
)


def _runs_for(persona_idx: int, vs_list, total_return=10.0,
              equity_curve=None, status="complete"):
    """Build run dicts whose run_ids all map to `persona_idx` (1..10).

    run_id = persona_idx + 10*k ⇒ ((run_id-1) % 10) + 1 == persona_idx,
    i.e. exactly what backtest.persona_for computes.
    """
    out = []
    for k, vs in enumerate(vs_list):
        out.append({
            "run_id": persona_idx + 10 * k,
            "total_return_pct": total_return,
            "vs_spy_pct": vs,
            "status": status,
            "equity_curve": equity_curve,
        })
    return out


# ───────────────────────── single source of truth ─────────────────────────

class TestSingleSourceOfTruth:
    def test_run_id_maps_via_persona_for(self):
        # The helper's run_id arithmetic must agree with the real mapping
        # the module imports — otherwise every aggregate below is mislabeled.
        for p in range(1, 11):
            name = persona_for(p)["name"]
            for k in range(5):
                assert persona_for(p + 10 * k)["name"] == name

    def test_aggregates_attributed_to_correct_persona(self):
        runs = _runs_for(7, [5.0] * MIN_RECORDS)  # persona 7
        rep = persona_leaderboard(runs)
        assert rep["n_personas"] == 1
        assert rep["leaderboard"][0]["persona"] == persona_for(7)["name"]


# ───────────────────────── verdict boundaries ─────────────────────────────

class TestPersonaVerdictBoundaries:
    def test_drag_at_exact_zero_median(self):
        # median vs_spy == DRAG_MAX_MEDIAN_VS_SPY (0.0) → DRAG (inclusive).
        # 30 runs all 0.0 → median 0.0, win_rate 0.0.
        rep = persona_leaderboard(_runs_for(1, [0.0] * 30))
        e = rep["leaderboard"][0]
        assert e["median_vs_spy"] == 0.0
        assert e["win_rate"] == 0.0
        assert e["verdict"] == "DRAG"
        assert rep["verdict"] == "HAS_DRAG_PERSONA"
        assert persona_for(1)["name"] in rep["drag_personas"]

    def test_flat_just_above_zero_below_edge(self):
        # median 10.0 (>0, <20 EDGE bar) → FLAT, overall HEALTHY.
        rep = persona_leaderboard(_runs_for(2, [10.0] * 30))
        e = rep["leaderboard"][0]
        assert e["median_vs_spy"] == 10.0
        assert e["verdict"] == "FLAT"
        assert rep["verdict"] == "HEALTHY"
        assert rep["drag_personas"] == []

    def test_edge_at_exact_threshold_with_winrate(self):
        # median exactly EDGE_MIN_MEDIAN_VS_SPY (20.0), all positive →
        # win_rate 1.0 ≥ EDGE_MIN_WIN_RATE → EDGE (inclusive boundary).
        rep = persona_leaderboard(_runs_for(3, [20.0] * 30))
        e = rep["leaderboard"][0]
        assert e["median_vs_spy"] == 20.0
        assert e["win_rate"] == 1.0
        assert e["verdict"] == "EDGE"

    def test_strong_median_but_low_winrate_is_not_edge(self):
        # 30 runs: 16 big winners, 14 big losers. median is a winner
        # (≥20) but win_rate = 16/30 = 0.5333 ≥ 0.5 → still EDGE.
        # Flip to 15/15 so median is the midpoint and win_rate 0.5 — make
        # losers dominate so the median sits ≤0 → DRAG, proving win_rate
        # AND median both gate (a high single value can't fake EDGE).
        vs = [50.0] * 14 + [-50.0] * 16
        rep = persona_leaderboard(_runs_for(4, vs))
        e = rep["leaderboard"][0]
        # sorted ascending median of 14×50 + 16×-50 → middle two are -50 → -50
        assert e["median_vs_spy"] == -50.0
        assert e["verdict"] == "DRAG"

    def test_insufficient_per_persona_sample(self):
        # A persona with < MIN_RUNS_PER_PERSONA runs is INSUFFICIENT and
        # never DRAG even with a negative median (small-n unstable).
        good = _runs_for(1, [30.0] * MIN_RECORDS)         # EDGE, stable
        thin = _runs_for(2, [-99.0] * (MIN_RUNS_PER_PERSONA - 1))
        rep = persona_leaderboard(good + thin)
        by = {e["persona"]: e for e in rep["leaderboard"]}
        assert by[persona_for(2)["name"]]["verdict"] == "INSUFFICIENT"
        assert by[persona_for(2)["name"]]["n"] == MIN_RUNS_PER_PERSONA - 1
        # A thin negative persona must NOT trip the overall DRAG alarm.
        assert rep["verdict"] == "HEALTHY"
        assert rep["drag_personas"] == []


# ───────────────────────── median vs mean robustness ──────────────────────

class TestMedianIsRobust:
    def test_median_ignores_leverage_outlier(self):
        # The documented failure mode: one 3×-ETF bull-window rip dominates
        # the MEAN. 29 runs at +5%, one at +2000%. median stays +5 (FLAT),
        # mean is dragged to ~+71.5 — proving the leaderboard's primary
        # statistic is the honest one.
        vs = [5.0] * 29 + [2000.0]
        rep = persona_leaderboard(_runs_for(5, vs))
        e = rep["leaderboard"][0]
        assert e["median_vs_spy"] == 5.0
        assert e["mean_vs_spy"] == round((5.0 * 29 + 2000.0) / 30, 4)
        assert e["mean_vs_spy"] > 60.0          # mean is inflated…
        assert e["verdict"] == "FLAT"           # …median verdict is honest


# ───────────────────────── equity-curve risk metrics ──────────────────────

class TestEquityRisk:
    def test_known_drawdown_and_underwater(self):
        # values 1000→1100→900→1200. Peak path: 1000,1100,1100,1200.
        # max dd = (1100-900)/1100 = 18.1818%. Underwater points: only the
        # 900 (1 of 4) = 25%.
        r = _equity_risk([{"value": v} for v in (1000, 1100, 900, 1200)])
        assert r["max_drawdown_pct"] == round(200.0 / 1100.0 * 100.0, 4)
        assert r["max_drawdown_pct"] == 18.1818
        assert r["pct_time_underwater"] == 25.0
        assert r["sharpe"] is not None and r["sharpe"] > 0  # net-up, varying

    def test_monotone_increasing_constant_return_has_zero_std(self):
        # Constant +10% steps → sample std of returns is exactly 0 →
        # Sharpe undefined (None), no drawdown, never underwater.
        r = _equity_risk([{"value": v} for v in (100, 110, 121, 133.1)])
        assert r["max_drawdown_pct"] == 0.0
        assert r["pct_time_underwater"] == 0.0
        assert r["sharpe"] is None

    def test_monotone_decreasing_negative_path(self):
        r = _equity_risk([{"value": v} for v in (1000, 900, 810)])
        assert r["max_drawdown_pct"] == 19.0          # (1000-810)/1000
        assert r["pct_time_underwater"] == round(2 / 3 * 100, 4)
        assert r["sharpe"] is None                    # both rets == -0.1

    def test_negative_sharpe_when_choppy_down(self):
        r = _equity_risk([{"value": v} for v in (1000, 900, 850, 800)])
        assert r["sharpe"] is not None and r["sharpe"] < 0

    def test_degrades_to_none_on_garbage(self):
        for bad in (None, [], [{"value": "x"}], [{"nope": 1}], "junk",
                    [{"value": None}]):
            r = _equity_risk(bad)
            assert r == {"max_drawdown_pct": None, "sharpe": None,
                         "pct_time_underwater": None}

    def test_risk_aggregates_survive_when_some_curves_missing(self):
        # Half the runs have a usable curve, half have none. Return
        # aggregates use all runs; risk medians use only the curved ones.
        curve = [{"value": v} for v in (1000, 800, 1000)]  # 20% dd
        good = _runs_for(6, [10.0] * 20, equity_curve=curve)
        nocurve = _runs_for(6, [10.0] * 20, equity_curve=None)
        rep = persona_leaderboard(good + nocurve)
        e = rep["leaderboard"][0]
        assert e["n"] == 40                       # all runs counted
        assert e["median_vs_spy"] == 10.0
        assert e["median_max_drawdown_pct"] == 20.0   # only the 20 curved


# ───────────────────────── overall guards & ordering ──────────────────────

class TestOverallGuards:
    def test_insufficient_data_below_min_records(self):
        rep = persona_leaderboard(_runs_for(1, [50.0] * (MIN_RECORDS - 1)))
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_runs"] == MIN_RECORDS - 1
        assert rep["leaderboard"] == []

    def test_non_complete_rows_ignored(self):
        runs = (_runs_for(1, [50.0] * MIN_RECORDS)
                + _runs_for(2, [99.0] * 50, status="running"))
        rep = persona_leaderboard(runs)
        assert rep["n_runs"] == MIN_RECORDS          # running rows dropped
        assert rep["n_personas"] == 1

    def test_non_finite_vs_spy_dropped(self):
        runs = _runs_for(1, [50.0] * MIN_RECORDS)
        runs.append({"run_id": 1, "vs_spy_pct": float("nan"),
                     "status": "complete"})
        runs.append({"run_id": 1, "vs_spy_pct": None, "status": "complete"})
        rep = persona_leaderboard(runs)
        assert rep["n_runs"] == MIN_RECORDS          # nan/None excluded

    def test_leaderboard_sorted_desc_insufficient_last(self):
        runs = (
            _runs_for(1, [80.0] * 30)               # EDGE, high
            + _runs_for(2, [25.0] * 30)             # EDGE, lower
            + _runs_for(3, [-5.0] * 30)             # DRAG, lowest
            + _runs_for(4, [999.0] * 2)             # INSUFFICIENT (n<5)
        )
        rep = persona_leaderboard(runs)
        lb = rep["leaderboard"]
        # INSUFFICIENT persona must be last despite its huge median.
        assert lb[-1]["verdict"] == "INSUFFICIENT"
        stable = [e for e in lb if e["verdict"] != "INSUFFICIENT"]
        meds = [e["median_vs_spy"] for e in stable]
        assert meds == sorted(meds, reverse=True)
        assert meds == [80.0, 25.0, -5.0]
        assert rep["verdict"] == "HAS_DRAG_PERSONA"


# ───────────────────────── read-only DB loader ────────────────────────────

class TestLoadRunsReadOnly:
    def test_corrupt_equity_json_degrades_not_raises(self, tmp_path):
        db = tmp_path / "bt.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE backtest_runs (run_id INTEGER, total_return_pct REAL,"
            " vs_spy_pct REAL, status TEXT, equity_curve_json TEXT)"
        )
        conn.executemany(
            "INSERT INTO backtest_runs VALUES (?,?,?,?,?)",
            [
                (1, 10.0, 5.0, "complete", '[{"value": 1000}]'),
                (2, 20.0, 8.0, "complete", "NOT JSON {{"),     # corrupt
                (3, 30.0, 9.0, "running", "[]"),               # filtered out
            ],
        )
        conn.commit()
        conn.close()

        runs = _load_runs(db)
        assert len(runs) == 2                       # only 'complete' rows
        by_id = {r["run_id"]: r for r in runs}
        assert by_id[1]["equity_curve"] == [{"value": 1000}]
        assert by_id[2]["equity_curve"] == []       # corrupt → [] not raise
        # And the full report still computes off the degraded row.
        rep = persona_leaderboard(runs * 15)        # pad past MIN_RECORDS
        assert rep["status"] == "ok"
