"""Exact-value locks for the val/oos generalization-gap trend diagnostic
(`paper_trader/ml/overfit_gap.py`, 2026-05-18 quant feature).

Mirrors test_skill_trend.py / test_baseline_trend.py: deterministic synthetic
ledgers, exact metrics and exact verdicts (not ranges) so a logic change must
update the literals deliberately. All offline; the module never touches the
model, pickle, or trade path.
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml import overfit_gap as og
from paper_trader.ml import skill_trend as st


def _row(val_rmse, oos_rmse, status="ok", gate_active=True):
    return {"status": status, "val_rmse": val_rmse, "oos_rmse": oos_rmse,
            "gate_active": gate_active}


# ───────────────── single source of truth (DRY import lock) ─────────────────

class TestSingleSourceOfTruth:
    """The trender must reuse skill_trend's ledger loader / median / window
    sizing verbatim — a ledger-schema change cannot make this verdict and
    skill_trend's disagree about which rows count. Mirrors baseline_trend's
    'single-source margin identity' lock."""

    def test_shared_symbols_are_the_same_object(self):
        assert og.MIN_CYCLES is st.MIN_CYCLES
        assert og.RECENT_CYCLES is st.RECENT_CYCLES
        assert og._median is st._median
        assert og.load_skill_ledger is st.load_skill_ledger

    def test_window_constants_have_expected_values(self):
        # If skill_trend retunes these, this test fails RED so the literals
        # in the verdict-boundary tests below are revisited deliberately.
        assert og.MIN_CYCLES == 5
        assert og.RECENT_CYCLES == 10
        assert og.SEVERE_RATIO == pytest.approx(1.40)
        assert og.MILD_RATIO == pytest.approx(1.15)
        assert og.RATIO_TOL == pytest.approx(0.10)


# ───────────────────────── usable-row filter ─────────────────────────

class TestUsableRows:
    def test_excludes_non_ok_null_nan_and_nonpositive_val(self):
        rows = [
            _row(10.0, 12.0),                              # usable
            _row(10.0, 12.0, status="insufficient_data"),  # not ok
            _row(None, 12.0),                              # null val
            _row(10.0, None),                              # null oos
            _row(float("nan"), 12.0),                      # NaN val
            _row(10.0, float("nan")),                      # NaN oos
            _row(0.0, 12.0),                               # val == 0 (÷0)
            _row(-3.0, 12.0),                              # val < 0
            _row("abc", 12.0),                             # unparseable
        ]
        assert len(og._usable_rows(rows)) == 1

    def test_insufficient_data_below_min_cycles(self):
        rep = og.overfit_gap_report([_row(10.0, 12.0) for _ in range(4)])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_cycles_usable"] == 4
        assert rep["recent_median_ratio"] is None

    def test_empty_ledger(self):
        rep = og.overfit_gap_report([])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_cycles_total"] == 0
        assert rep["n_cycles_usable"] == 0
        assert rep["gate_active_fraction"] is None


# ───────────────────────── exact verdicts ─────────────────────────

class TestVerdicts:
    def test_severe_overfit_exact(self):
        # 5 rows, ratio 15/10 = 1.5 ≥ SEVERE_RATIO(1.40).
        rep = og.overfit_gap_report([_row(10.0, 15.0) for _ in range(5)])
        assert rep["verdict"] == "SEVERE_OVERFIT"
        assert rep["n_cycles_usable"] == 5
        assert rep["recent_median_ratio"] == pytest.approx(1.5)
        assert rep["overall_median_ratio"] == pytest.approx(1.5)
        assert rep["recent_median_abs_gap"] == pytest.approx(5.0)
        assert rep["recent_median_val_rmse"] == pytest.approx(10.0)
        assert rep["recent_median_oos_rmse"] == pytest.approx(15.0)
        assert rep["trend"] == "UNKNOWN"  # exactly 5 usable, no older tail

    def test_mild_overfit_exact(self):
        # ratio 12/10 = 1.2 ∈ [1.15, 1.40).
        rep = og.overfit_gap_report([_row(10.0, 12.0) for _ in range(6)])
        assert rep["verdict"] == "MILD_OVERFIT"
        assert rep["recent_median_ratio"] == pytest.approx(1.2)
        assert rep["recent_median_abs_gap"] == pytest.approx(2.0)

    def test_well_generalized_exact(self):
        # ratio 10.5/10 = 1.05 < 1.15.
        rep = og.overfit_gap_report([_row(10.0, 10.5) for _ in range(5)])
        assert rep["verdict"] == "WELL_GENERALIZED"
        assert rep["recent_median_ratio"] == pytest.approx(1.05)
        assert rep["recent_median_abs_gap"] == pytest.approx(0.5)

    def test_severe_boundary_is_inclusive(self):
        # ratio exactly 14/10 = 1.40 → SEVERE (>= boundary).
        rep = og.overfit_gap_report([_row(10.0, 14.0) for _ in range(5)])
        assert rep["verdict"] == "SEVERE_OVERFIT"
        assert rep["recent_median_ratio"] == pytest.approx(1.40)

    def test_mild_boundary_is_inclusive(self):
        # ratio exactly 23/20 = 1.15 → MILD (>= boundary), not WELL.
        rep = og.overfit_gap_report([_row(20.0, 23.0) for _ in range(5)])
        assert rep["verdict"] == "MILD_OVERFIT"
        assert rep["recent_median_ratio"] == pytest.approx(1.15)

    def test_just_below_mild_is_well_generalized(self):
        # ratio 1140/1000 = 1.14 < 1.15 → WELL_GENERALIZED.
        rep = og.overfit_gap_report([_row(1000.0, 1140.0) for _ in range(5)])
        assert rep["verdict"] == "WELL_GENERALIZED"
        assert rep["recent_median_ratio"] == pytest.approx(1.14)

    def test_even_length_median_averages_two_middle(self):
        # 6 recent rows, val=10, oos=11..16 → ratios 1.1..1.6 sorted;
        # median = (1.3 + 1.4) / 2 = 1.35 → MILD (1.35 < 1.40).
        rows = [_row(10.0, o) for o in (11.0, 12.0, 13.0, 14.0, 15.0, 16.0)]
        rep = og.overfit_gap_report(rows)
        assert rep["recent_median_ratio"] == pytest.approx(1.35)
        assert rep["recent_median_abs_gap"] == pytest.approx(3.5)  # (3+4)/2
        assert rep["verdict"] == "MILD_OVERFIT"


# ───────────────────────── trend axis ─────────────────────────

class TestTrend:
    def test_improving_when_recent_gap_shrinks(self):
        # 2 older (ratio 1.6) + 10 recent (ratio 1.2). 1.2 ≤ 1.6·0.9=1.44.
        older = [_row(10.0, 16.0) for _ in range(2)]
        recent = [_row(10.0, 12.0) for _ in range(10)]
        rep = og.overfit_gap_report(older + recent)
        assert rep["n_cycles_usable"] == 12
        assert rep["recent_median_ratio"] == pytest.approx(1.2)
        assert rep["older_median_ratio"] == pytest.approx(1.6)
        assert rep["trend"] == "IMPROVING"
        assert rep["verdict"] == "MILD_OVERFIT"

    def test_degrading_when_recent_gap_widens(self):
        older = [_row(10.0, 11.0) for _ in range(2)]   # ratio 1.1
        recent = [_row(10.0, 15.0) for _ in range(10)]  # ratio 1.5
        rep = og.overfit_gap_report(older + recent)
        assert rep["trend"] == "DEGRADING"
        assert rep["verdict"] == "SEVERE_OVERFIT"

    def test_stable_within_band(self):
        older = [_row(10.0, 13.0) for _ in range(2)]    # ratio 1.3
        recent = [_row(10.0, 13.0) for _ in range(10)]  # ratio 1.3
        rep = og.overfit_gap_report(older + recent)
        assert rep["trend"] == "STABLE"
        assert rep["verdict"] == "MILD_OVERFIT"

    def test_unknown_when_no_older_tail(self):
        rep = og.overfit_gap_report([_row(10.0, 13.0) for _ in range(10)])
        assert rep["older_median_ratio"] is None
        assert rep["trend"] == "UNKNOWN"


# ─────────────────── gate_active_fraction (all rows) ───────────────────

class TestGateActiveFraction:
    def test_fraction_counts_every_row_not_just_usable(self):
        # 5 usable gate-active ok rows + 1 excluded gate-INACTIVE non-ok row.
        rows = [_row(10.0, 12.0, gate_active=True) for _ in range(5)]
        rows.append(_row(10.0, 12.0, status="no_outcome_records",
                         gate_active=False))
        rep = og.overfit_gap_report(rows)
        # Fraction is over ALL 6 rows: 5 active / 6 = 0.8333.
        assert rep["gate_active_fraction"] == pytest.approx(0.8333, abs=1e-4)
        assert rep["n_cycles_usable"] == 5
        assert rep["verdict"] == "MILD_OVERFIT"


# ───────────────────────── robustness ─────────────────────────

class TestNeverRaises:
    def test_non_dict_rows_degrade_to_insufficient(self):
        rep = og.overfit_gap_report([1, 2, 3])  # type: ignore[list-item]
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_analyze_missing_file(self, tmp_path):
        rep = og.analyze(tmp_path / "nope.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_cycles_total"] == 0

    def test_analyze_reads_jsonl(self, tmp_path):
        p = tmp_path / "led.jsonl"
        with p.open("w") as fh:
            for _ in range(5):
                fh.write(json.dumps(_row(10.0, 15.0)) + "\n")
            fh.write("not json\n")  # skipped by load_skill_ledger
        rep = og.analyze(p)
        assert rep["verdict"] == "SEVERE_OVERFIT"
        assert rep["n_cycles_usable"] == 5
        assert rep["recent_median_ratio"] == pytest.approx(1.5)


# ───────────────────────── CLI exit codes ─────────────────────────

class TestCliExitCodes:
    """Mirrors skill_trend / baseline_trend: a cron must be able to branch on
    'the net is persistently memorizing'. 2 on MILD/SEVERE_OVERFIT, 0
    otherwise."""

    @pytest.mark.parametrize("verdict,expected", [
        ("SEVERE_OVERFIT", 2),
        ("MILD_OVERFIT", 2),
        ("WELL_GENERALIZED", 0),
        ("INSUFFICIENT_DATA", 0),
    ])
    def test_exit_code_maps_to_verdict(self, monkeypatch, capsys,
                                       verdict, expected):
        canned = {
            "verdict": verdict, "hint": "x", "n_cycles_usable": 1,
            "n_cycles_total": 1, "gate_active_fraction": 1.0,
            "recent_median_ratio": 1.5, "older_median_ratio": None,
            "overall_median_ratio": 1.5, "trend": "UNKNOWN",
            "recent_median_val_rmse": 10.0, "recent_median_oos_rmse": 15.0,
            "recent_median_abs_gap": 5.0,
        }
        monkeypatch.setattr(og, "analyze", lambda *_a, **_k: canned)
        rc = og._cli()
        assert rc == expected
        assert verdict in capsys.readouterr().out
