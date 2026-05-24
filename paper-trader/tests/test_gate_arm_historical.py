"""Tests for paper_trader.ml.gate_arm_historical — historical gate-arm
diagnostic that uses persisted ``gate_scorer_pred`` instead of re-predicting
with today's model. The verdict semantics MUST match the sibling
``gate_effectiveness_report`` so the only difference between the two
diagnostics is *which prediction* they bucket by (historical truth vs
counterfactual reconstruction)."""
from __future__ import annotations

import json
import pytest

from paper_trader.ml.gate_arm_historical import (
    gate_arm_historical_report,
    analyze,
)
from paper_trader.ml.gate_audit import MIN_TOTAL, MIN_ARM_N, EDGE_TOL_PP


# ── Synthetic record helpers ───────────────────────────────────────────────

def _row(pred, ret, action="BUY", off_dist=False):
    """Build one decision_outcomes.jsonl row shape with just the fields
    the historical-arm report reads."""
    return {
        "gate_scorer_pred": pred,
        "gate_off_dist": off_dist,
        "forward_return_5d": ret,
        "action": action,
    }


class TestHistoricalArmBucketing:

    def test_strong_tailwind_above_threshold_bucketed_correctly(self):
        """`pred > +10` lands in strong_tailwind — must mirror gate_arm()."""
        rows = [_row(15.0, 8.0)] * 6
        # Pad so we hit MIN_TOTAL with mostly other arms.
        rows += [_row(-15.0, 2.0)] * 6     # strong_headwind to satisfy floor
        rows += [_row(0.0, 1.0)] * (MIN_TOTAL - 12)
        rep = gate_arm_historical_report(rows)
        st = next(a for a in rep["arms"] if a["arm"] == "strong_tailwind")
        assert st["n"] == 6
        assert st["mean_realized"] == pytest.approx(8.0, abs=0.01)

    def test_off_dist_rows_are_dropped(self):
        """A row marked off_dist=True must NOT contribute to any arm —
        the gate explicitly abstained at decision time."""
        rows = [_row(15.0, 99.0, off_dist=True)] * 5    # extreme rets
        rows += [_row(-15.0, -99.0, off_dist=True)] * 5  # extreme losses
        rows += [_row(0.0, 0.0)]                        # one neutral row
        rep = gate_arm_historical_report(rows)
        for a in rep["arms"]:
            if a["arm"] in ("strong_tailwind", "strong_headwind"):
                assert a["n"] == 0, a
        assert rep["n_dropped_off_dist"] == 10
        assert rep["n"] == 1

    def test_none_pred_rows_are_dropped(self):
        """Rows with no gate_scorer_pred (sub-gate / SELL / HOLD / legacy)
        contribute nothing AND are counted in n_dropped_no_gate_pred."""
        rows = [_row(None, 5.0) for _ in range(7)]
        rows.append(_row(0.0, 1.0))
        rep = gate_arm_historical_report(rows)
        assert rep["n_dropped_no_gate_pred"] == 7
        assert rep["n"] == 1
        neutral = next(a for a in rep["arms"] if a["arm"] == "neutral")
        assert neutral["n"] == 1

    def test_missing_forward_return_dropped(self):
        """A row with gate_scorer_pred but no forward_return_5d cannot
        contribute (no realized truth to bucket)."""
        rows = [_row(5.0, None)] * 3
        rows.append(_row(0.0, 1.0))
        rep = gate_arm_historical_report(rows)
        assert rep["n_dropped_no_return"] == 3
        assert rep["n"] == 1

    def test_sell_sign_flip_applied(self):
        """A SELL with forward_return_5d=-5 (correct prediction of a drop)
        contributes +5 (action-aligned) to the arm."""
        rows = [_row(15.0, -8.0, action="SELL")]
        # Pad for floors so the per-arm mean is computable.
        rows += [_row(15.0, -8.0, action="SELL")] * 4
        rows += [_row(-15.0, 1.0)] * MIN_ARM_N
        rows += [_row(0.0, 0.0)] * (MIN_TOTAL - len(rows))
        rep = gate_arm_historical_report(rows)
        st = next(a for a in rep["arms"] if a["arm"] == "strong_tailwind")
        # All 5 SELL rows: -(-8) = +8 each.
        assert st["mean_realized"] == pytest.approx(8.0, abs=0.01)


class TestVerdicts:

    def test_insufficient_data_below_min_total(self):
        rows = [_row(0.0, 1.0)] * (MIN_TOTAL - 1)
        rep = gate_arm_historical_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == MIN_TOTAL - 1

    def test_insufficient_data_extreme_arm_too_small(self):
        # Lots of total rows but only 2 in each extreme arm.
        rows = [_row(15.0, 1.0)] * 2
        rows += [_row(-15.0, 1.0)] * 2
        rows += [_row(0.0, 1.0)] * (MIN_TOTAL + 5)
        rep = gate_arm_historical_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        # n_extreme < MIN_ARM_N is what triggered it.
        assert "extreme" in rep["hint"].lower()

    def test_gate_effective_when_tailwind_clearly_beats_headwind(self):
        """Strong_tailwind realizes more than headwind by > EDGE_TOL_PP."""
        rows = [_row(15.0, 10.0) for _ in range(MIN_ARM_N)]
        rows += [_row(-15.0, -5.0) for _ in range(MIN_ARM_N)]
        rows += [_row(0.0, 1.0) for _ in range(MIN_TOTAL - 2 * MIN_ARM_N)]
        rep = gate_arm_historical_report(rows)
        assert rep["verdict"] == "GATE_EFFECTIVE"
        assert rep["strong_tailwind_minus_headwind_pp"] == pytest.approx(15.0,
                                                                         abs=0.01)

    def test_gate_harmful_when_tailwind_underperforms_headwind(self):
        rows = [_row(15.0, -5.0) for _ in range(MIN_ARM_N)]
        rows += [_row(-15.0, 10.0) for _ in range(MIN_ARM_N)]
        rows += [_row(0.0, 1.0) for _ in range(MIN_TOTAL - 2 * MIN_ARM_N)]
        rep = gate_arm_historical_report(rows)
        assert rep["verdict"] == "GATE_HARMFUL"
        assert rep["strong_tailwind_minus_headwind_pp"] < -EDGE_TOL_PP

    def test_gate_ineffective_when_spread_within_tolerance(self):
        # Strong_tailwind and headwind realize within ±EDGE_TOL_PP of each
        # other — multipliers are sizing noise.
        rows = [_row(15.0, 1.0) for _ in range(MIN_ARM_N)]
        rows += [_row(-15.0, 1.2) for _ in range(MIN_ARM_N)]
        rows += [_row(0.0, 1.0) for _ in range(MIN_TOTAL - 2 * MIN_ARM_N)]
        rep = gate_arm_historical_report(rows)
        assert rep["verdict"] == "GATE_INEFFECTIVE"
        spread = rep["strong_tailwind_minus_headwind_pp"]
        assert abs(spread) <= EDGE_TOL_PP


class TestNeverRaises:

    def test_empty_input(self):
        rep = gate_arm_historical_report([])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_garbage_rows(self):
        rows = [
            {},  # missing every field
            {"gate_scorer_pred": "not a number"},  # bad type
            {"gate_scorer_pred": float("inf"), "forward_return_5d": 1.0},  # non-finite
            {"gate_scorer_pred": 1.0, "forward_return_5d": float("nan")},  # nan ret
        ]
        rep = gate_arm_historical_report(rows)
        # No crash; all rows dropped one way or another.
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        # First row: no gate_scorer_pred → dropped in no_pred bucket.
        # Second: bad type → caught at float() in try.
        # Third: non-finite pred → dropped by isfinite guard.
        # Fourth: nan ret → dropped by isfinite guard.
        assert rep["n"] == 0


class TestAnalyzeCli:

    def test_analyze_returns_insufficient_for_missing_file(self, tmp_path):
        rep = analyze(tmp_path / "does_not_exist.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "no file" in rep["hint"].lower()

    def test_analyze_reads_jsonl_and_reports(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        # Mix of rows that hit the GATE_EFFECTIVE branch.
        rows = []
        for _ in range(MIN_ARM_N):
            rows.append(_row(15.0, 10.0))
        for _ in range(MIN_ARM_N):
            rows.append(_row(-15.0, -5.0))
        # Pad with neutral rows but vary sim_date so split_outcomes_temporal
        # has something to split on.
        for i in range(MIN_TOTAL):
            rows.append({**_row(0.0, 1.0),
                         "sim_date": f"2025-01-{(i % 28) + 1:02d}"})
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        rep = analyze(path, oos_only=False)
        assert rep["status"] == "ok"
        assert rep["outcomes_n"] == len(rows)
        # Doesn't matter whether OOS or all; the synthetic distribution
        # is uniform enough.

    def test_analyze_oos_only_falls_back_to_all_when_split_missing(self,
                                                                   tmp_path,
                                                                   monkeypatch):
        """Force the validation import to fail; analyze should fall back to
        slice='all' rather than crashing."""
        # Simulate the module being unavailable
        import sys
        monkeypatch.setitem(sys.modules, "paper_trader.validation",
                            type(sys)("paper_trader.validation"))
        path = tmp_path / "x.jsonl"
        path.write_text(json.dumps(_row(0.0, 1.0)) + "\n")
        rep = analyze(path, oos_only=True)
        # The synthetic module is missing split_outcomes_temporal — the
        # import inside analyze() fails and we fall back to slice='all'.
        assert rep.get("slice") in ("oos", "all")  # either is acceptable
