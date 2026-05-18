"""Exact-value locks for the baseline-trend diagnostic
(`paper_trader/ml/baseline_trend.py`, 2026-05-18 quant feature).

Mirrors test_skill_trend.py / test_calibration.py: deterministic synthetic
ledger rows, exact verdicts (not ranges) at and just past the margin
boundaries so a logic change must update the literals deliberately. All
offline — never touches the real `data/baseline_skill_log.jsonl`.
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml import baseline_trend as bt
from paper_trader.ml import baseline_compare as bc


def _row(ic_gap, mlp_rank_ic=0.0, best_baseline="ml_score",
         best_baseline_ic=0.0, n_train=3000, gate_active=True, status="ok"):
    return {"status": status, "verdict": "x", "slice": "oos", "n": 1000,
            "n_train": n_train, "mlp_rank_ic": mlp_rank_ic,
            "mlp_dir_acc": 0.5, "best_baseline": best_baseline,
            "best_baseline_ic": best_baseline_ic, "ic_gap": ic_gap,
            "gate_active": gate_active}


class TestMarginsAreSingleSourceOfTruth:
    def test_margins_imported_from_baseline_compare(self):
        # The trender trends baseline_compare's per-cycle verdict, so the
        # margins MUST be the same object/value by construction.
        assert bt.IC_MARGIN is bc.IC_MARGIN
        assert bt.MLP_IC_MIN is bc.MLP_IC_MIN
        assert bt.IC_MARGIN == 0.05 and bt.MLP_IC_MIN == 0.10


class TestLoadBaselineLedger:
    def test_missing_file_yields_empty(self, tmp_path):
        assert bt.load_baseline_ledger(tmp_path / "nope.jsonl") == []

    def test_corrupt_and_nondict_lines_skipped(self, tmp_path):
        p = tmp_path / "led.jsonl"
        p.write_text('{"status": "ok", "ic_gap": -0.1}\n'
                      "not json\n"
                      "[1,2,3]\n"
                      "\n"
                      '{"status": "ok", "ic_gap": -0.2}\n')
        rows = bt.load_baseline_ledger(p)
        assert len(rows) == 2
        assert rows[0]["ic_gap"] == -0.1 and rows[1]["ic_gap"] == -0.2


class TestUsableRowFilter:
    def test_insufficient_when_too_few_usable(self):
        rep = bt.baseline_trend_report([_row(-0.1)] * 4)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_cycles_usable"] == 4

    def test_insufficient_data_cycles_have_null_ic_gap_excluded(self):
        # status=="ok" but ic_gap=None (scorer-untrained / <MIN_PAIRS cycle)
        # must NOT count as usable — exactly skill_trend's null-oos_rmse skip.
        rows = [_row(-0.1)] * 4 + [_row(ic_gap=None)] * 5
        rep = bt.baseline_trend_report(rows)
        assert rep["n_cycles_usable"] == 4
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_status_not_ok_rows_excluded(self):
        rows = [_row(-0.1)] * 4 + [_row(99.0, status="error")] * 5
        rep = bt.baseline_trend_report(rows)
        assert rep["n_cycles_usable"] == 4
        assert rep["verdict"] == "INSUFFICIENT_DATA"


class TestVerdicts:
    def test_mlp_worse_than_trivial(self):
        rep = bt.baseline_trend_report([_row(-0.10)] * 6)
        assert rep["verdict"] == "MLP_WORSE_THAN_TRIVIAL"
        assert rep["recent_median_ic_gap"] == -0.10

    def test_mlp_worse_inclusive_boundary(self):
        # ic_gap exactly -IC_MARGIN (-0.05): -0.05 <= -0.05 → WORSE.
        rep = bt.baseline_trend_report([_row(-0.05)] * 6)
        assert rep["verdict"] == "MLP_WORSE_THAN_TRIVIAL"

    def test_no_better_just_inside_negative_band(self):
        # -0.04 is inside ±0.05 → NO_BETTER, not WORSE.
        rep = bt.baseline_trend_report([_row(-0.04)] * 6)
        assert rep["verdict"] == "MLP_NO_BETTER_THAN_TRIVIAL"

    def test_no_better_at_zero_gap(self):
        rep = bt.baseline_trend_report([_row(0.0)] * 6)
        assert rep["verdict"] == "MLP_NO_BETTER_THAN_TRIVIAL"

    def test_mlp_adds_skill_needs_both_gap_and_floor(self):
        # ic_gap +0.10 ≥ 0.05 AND mlp_rank_ic +0.15 > 0.10 floor.
        rep = bt.baseline_trend_report(
            [_row(0.10, mlp_rank_ic=0.15)] * 6)
        assert rep["verdict"] == "MLP_ADDS_SKILL"

    def test_positive_gap_but_below_mlp_floor_is_no_better(self):
        # Gap clears +0.05 but the MLP's own rank skill (0.08) ≤ 0.10 floor —
        # it only "wins" because everything is noise → NO_BETTER.
        rep = bt.baseline_trend_report(
            [_row(0.10, mlp_rank_ic=0.08)] * 6)
        assert rep["verdict"] == "MLP_NO_BETTER_THAN_TRIVIAL"

    def test_adds_skill_floor_is_strict(self):
        # mlp_rank_ic exactly 0.10 → 0.10 > 0.10 is False → NO_BETTER.
        rep = bt.baseline_trend_report(
            [_row(0.10, mlp_rank_ic=0.10)] * 6)
        assert rep["verdict"] == "MLP_NO_BETTER_THAN_TRIVIAL"


class TestAggregates:
    def test_gate_active_fraction_exact(self):
        rows = [_row(-0.1, gate_active=True), _row(-0.1, gate_active=True),
                _row(-0.1, gate_active=True), _row(-0.1, gate_active=False),
                _row(-0.1, gate_active=False)]
        rep = bt.baseline_trend_report(rows)
        assert rep["gate_active_fraction"] == 0.6

    def test_most_common_best_baseline_and_medians(self):
        rows = ([_row(-0.12, mlp_rank_ic=0.04, best_baseline="ml_score",
                       best_baseline_ic=0.20, n_train=3000)] * 4
                + [_row(-0.08, mlp_rank_ic=0.06, best_baseline="mom20",
                        best_baseline_ic=0.14, n_train=3000)] * 2)
        rep = bt.baseline_trend_report(rows)
        assert rep["most_common_best_baseline"] == "ml_score"
        # 6 rows, even count → np.median = mean of the two middle elements.
        # ic_gap sorted [-.12,-.12,-.12,-.12,-.08,-.08] → (-.12 + -.12)/2.
        assert rep["recent_median_ic_gap"] == -0.12
        # mlp_rank_ic sorted [.04,.04,.04,.04,.06,.06] → (.04 + .04)/2.
        assert rep["recent_median_mlp_rank_ic"] == 0.04
        # best_baseline_ic sorted [.14,.14,.20,.20,.20,.20] → (.20 + .20)/2.
        assert rep["recent_median_best_baseline_ic"] == 0.2
        assert rep["recent_median_n_train"] == 3000
        assert rep["verdict"] == "MLP_WORSE_THAN_TRIVIAL"


class TestTrendDirection:
    def test_improving_when_recent_gap_higher(self):
        # 15 rows: older=first 5 (gap -0.20), recent=last 10 (gap -0.05).
        rows = [_row(-0.20)] * 5 + [_row(-0.05)] * 10
        rep = bt.baseline_trend_report(rows)
        assert rep["older_median_ic_gap"] == -0.20
        assert rep["recent_median_ic_gap"] == -0.05
        assert rep["trend"] == "IMPROVING"
        # Trend axis is independent of the verdict axis.
        assert rep["verdict"] == "MLP_WORSE_THAN_TRIVIAL"

    def test_degrading_when_recent_gap_lower(self):
        rows = [_row(0.10, mlp_rank_ic=0.15)] * 5 + [_row(-0.10)] * 10
        rep = bt.baseline_trend_report(rows)
        assert rep["trend"] == "DEGRADING"

    def test_stable_within_band(self):
        rows = [_row(0.0)] * 5 + [_row(0.02)] * 10
        rep = bt.baseline_trend_report(rows)
        assert rep["trend"] == "STABLE"

    def test_unknown_when_no_older_rows(self):
        rep = bt.baseline_trend_report([_row(-0.1)] * 6)
        assert rep["trend"] == "UNKNOWN"
        assert rep["older_median_ic_gap"] is None


class TestAnalyzeEndToEnd:
    def test_analyze_reads_file_and_verdicts(self, tmp_path):
        ledger = tmp_path / "baseline_skill_log.jsonl"
        ledger.write_text("\n".join(
            json.dumps(_row(-0.15)) for _ in range(6)) + "\n")
        rep = bt.analyze(ledger)
        assert rep["verdict"] == "MLP_WORSE_THAN_TRIVIAL"
        assert rep["recent_median_ic_gap"] == -0.15

    def test_analyze_missing_file_insufficient(self, tmp_path):
        rep = bt.analyze(tmp_path / "no_led.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_cycles_total"] == 0


class TestCliExitCodes:
    def test_exit_2_on_worse(self, tmp_path, monkeypatch, capsys):
        # CLI resolves the ledger from ROOT; redirect via a fake module ROOT
        # by writing the real path is overkill — exercise _cli through analyze
        # on a synthetic ledger by monkeypatching analyze.
        monkeypatch.setattr(
            bt, "analyze",
            lambda _p: bt.baseline_trend_report([_row(-0.2)] * 6))
        assert bt._cli([]) == 2
        out = capsys.readouterr().out
        assert "MLP_WORSE_THAN_TRIVIAL" in out

    def test_exit_0_on_insufficient(self, monkeypatch):
        monkeypatch.setattr(
            bt, "analyze",
            lambda _p: bt.baseline_trend_report([_row(-0.2)] * 2))
        assert bt._cli([]) == 0

    def test_exit_0_on_adds_skill(self, monkeypatch):
        monkeypatch.setattr(
            bt, "analyze",
            lambda _p: bt.baseline_trend_report(
                [_row(0.12, mlp_rank_ic=0.18)] * 6))
        assert bt._cli([]) == 0
