"""Exact-value locks for the scorer-skill trend diagnostic
(`paper_trader/ml/skill_trend.py`, 2026-05-17 quant feature).

Mirrors test_calibration.py: deterministic synthetic data, exact metrics
and exact verdicts (not ranges) so a logic change must update the literals
deliberately. All offline.
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml import skill_trend as st


def _row(oos_rmse, oos_ic=0.0, oos_dir_acc=0.5, gate_active=True,
         status="ok", val_rmse=12.0):
    return {"status": status, "oos_rmse": oos_rmse, "oos_ic": oos_ic,
            "oos_dir_acc": oos_dir_acc, "gate_active": gate_active,
            "val_rmse": val_rmse}


class TestMeanPredictorBaseline:
    def test_population_std_with_sell_sign_flip(self):
        # 10 records by sim_date; oos_fraction 0.2 → last 2 are the OOS slice.
        # OOS = 2025-01-09 BUY fr=+10.0  and  2025-01-10 SELL fr=+4.0
        # SELL sign-flips → target -4.0. targets=[10.0,-4.0] mean=3.0
        # population std = sqrt(((10-3)^2+(-4-3)^2)/2) = sqrt(49) = 7.0 exactly.
        recs = []
        for i in range(1, 9):
            recs.append({"sim_date": f"2025-01-0{i}", "action": "BUY",
                         "forward_return_5d": 0.0})
        recs.append({"sim_date": "2025-01-09", "action": "BUY",
                     "forward_return_5d": 10.0})
        recs.append({"sim_date": "2025-01-10", "action": "SELL",
                     "forward_return_5d": 4.0})
        assert st.mean_predictor_baseline_rmse(recs) == pytest.approx(7.0)

    def test_empty_and_tiny_inputs_return_none(self):
        assert st.mean_predictor_baseline_rmse([]) is None
        # < 5 records → split gives everything to train, OOS empty → None.
        assert st.mean_predictor_baseline_rmse(
            [{"sim_date": "2025-01-01", "forward_return_5d": 1.0}]) is None


class TestLoadSkillLedger:
    def test_missing_file_yields_empty(self, tmp_path):
        assert st.load_skill_ledger(tmp_path / "nope.jsonl") == []

    def test_corrupt_and_nondict_lines_skipped(self, tmp_path):
        p = tmp_path / "led.jsonl"
        p.write_text('{"status": "ok", "oos_rmse": 9.0}\n'
                      "not json\n"
                      "[1,2,3]\n"
                      "\n"
                      '{"status": "ok", "oos_rmse": 8.0}\n')
        rows = st.load_skill_ledger(p)
        assert len(rows) == 2
        assert rows[0]["oos_rmse"] == 9.0 and rows[1]["oos_rmse"] == 8.0


class TestSkillTrendVerdicts:
    def test_insufficient_when_too_few_cycles(self):
        rep = st.skill_trend_report([_row(9.0)] * 4, baseline_rmse=10.0)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_cycles_usable"] == 4

    def test_insufficient_when_no_baseline(self):
        rep = st.skill_trend_report([_row(9.0)] * 8, baseline_rmse=None)
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_beats_mean_predictor(self):
        # baseline 10 → lo=9.0; recent median 8.0 ≤ 9.0.
        rep = st.skill_trend_report([_row(8.0)] * 6, baseline_rmse=10.0)
        assert rep["verdict"] == "BEATS_MEAN_PREDICTOR"
        assert rep["recent_median_oos_rmse"] == 8.0

    def test_negative_oos_skill(self):
        # baseline 10 → hi=11.0; recent median 13.0 ≥ 11.0 AND oos_ic ≤ IC_MIN.
        rep = st.skill_trend_report([_row(13.0, oos_ic=0.0)] * 6,
                                    baseline_rmse=10.0)
        assert rep["verdict"] == "NEGATIVE_OOS_SKILL"

    def test_directional_but_high_error(self):
        # RMSE worse than mean (13 ≥ 11) but median oos_ic 0.20 > IC_MIN 0.05.
        rep = st.skill_trend_report([_row(13.0, oos_ic=0.20)] * 6,
                                    baseline_rmse=10.0)
        assert rep["verdict"] == "DIRECTIONAL_BUT_HIGH_ERROR"

    def test_borderline_within_band(self):
        # recent median exactly at baseline → within ±10% band.
        rep = st.skill_trend_report([_row(10.0)] * 6, baseline_rmse=10.0)
        assert rep["verdict"] == "BORDERLINE"

    def test_gate_active_fraction_exact(self):
        rows = [_row(9.0, gate_active=True), _row(9.0, gate_active=True),
                _row(9.0, gate_active=True), _row(9.0, gate_active=False),
                _row(9.0, gate_active=False)]
        rep = st.skill_trend_report(rows, baseline_rmse=10.0)
        assert rep["gate_active_fraction"] == 0.6

    def test_status_not_ok_rows_excluded_from_usable(self):
        rows = [_row(9.0)] * 4 + [_row(99.0, status="insufficient_after_dedup")]
        rep = st.skill_trend_report(rows, baseline_rmse=10.0)
        # only 4 usable → INSUFFICIENT, and the bad row didn't pollute medians.
        assert rep["n_cycles_usable"] == 4
        assert rep["verdict"] == "INSUFFICIENT_DATA"


class TestTrendDirection:
    def test_improving_when_recent_lower(self):
        # 15 ok rows: older = first 5 (rmse 20), recent = last 10 (rmse 10).
        rows = [_row(20.0)] * 5 + [_row(10.0)] * 10
        rep = st.skill_trend_report(rows, baseline_rmse=10.0)
        assert rep["older_median_oos_rmse"] == 20.0
        assert rep["recent_median_oos_rmse"] == 10.0
        assert rep["trend"] == "IMPROVING"

    def test_degrading_when_recent_higher(self):
        rows = [_row(20.0)] * 5 + [_row(25.0)] * 10
        rep = st.skill_trend_report(rows, baseline_rmse=22.0)
        assert rep["trend"] == "DEGRADING"

    def test_stable_within_band(self):
        rows = [_row(20.0)] * 5 + [_row(21.0)] * 10
        rep = st.skill_trend_report(rows, baseline_rmse=20.5)
        assert rep["trend"] == "STABLE"


class TestAnalyzeEndToEnd:
    def test_analyze_reads_files_and_verdicts(self, tmp_path):
        ledger = tmp_path / "scorer_skill_log.jsonl"
        ledger.write_text("\n".join(
            json.dumps(_row(8.0)) for _ in range(6)) + "\n")
        # Outcomes whose OOS slice has population std 7.0 (see baseline test).
        recs = [{"sim_date": f"2025-01-0{i}", "action": "BUY",
                 "forward_return_5d": 0.0} for i in range(1, 9)]
        recs.append({"sim_date": "2025-01-09", "action": "BUY",
                     "forward_return_5d": 10.0})
        recs.append({"sim_date": "2025-01-10", "action": "SELL",
                     "forward_return_5d": 4.0})
        outcomes = tmp_path / "decision_outcomes.jsonl"
        outcomes.write_text("\n".join(json.dumps(r) for r in recs) + "\n")

        rep = st.analyze(ledger, outcomes)
        assert rep["baseline_rmse"] == pytest.approx(7.0)
        # recent median oos_rmse 8.0 vs baseline 7.0: hi=7.7, 8.0 ≥ 7.7,
        # oos_ic 0.0 ≤ IC_MIN → NEGATIVE_OOS_SKILL.
        assert rep["verdict"] == "NEGATIVE_OOS_SKILL"

    def test_analyze_missing_files_insufficient(self, tmp_path):
        rep = st.analyze(tmp_path / "no_led.jsonl", tmp_path / "no_out.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
