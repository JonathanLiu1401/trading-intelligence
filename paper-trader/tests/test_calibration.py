"""Tests for paper_trader.ml.calibration.

Asserts exact metric values and exact verdicts on deterministic synthetic
data, so a logic change in the bucketing / rank-skill / verdict thresholds
fails loudly rather than silently shifting a quant-facing diagnostic. Every
dataset here is hand-constructed with a known-correct answer.
"""
from __future__ import annotations

import pytest

from paper_trader.ml.calibration import (
    BIAS_TOL_PCT,
    MIN_PAIRS,
    SPEARMAN_GOOD,
    SPEARMAN_MIN,
    calibration_report,
    scorer_calibration,
)


class TestCalibrationReport:
    def test_perfectly_calibrated(self):
        # realized == predicted across a wide range → ideal scorer.
        rep = calibration_report([(float(i), float(i)) for i in range(100)])
        assert rep["status"] == "ok"
        assert rep["verdict"] == "WELL_CALIBRATED"
        assert rep["n"] == 100
        assert rep["spearman"] == 1.0
        assert rep["pearson"] == 1.0
        assert rep["monotone_fraction"] == 1.0
        assert rep["mean_abs_decile_error"] == 0.0
        assert len(rep["buckets"]) == 10
        # Buckets are predicted-ascending and disjoint.
        assert rep["buckets"][0]["mean_pred"] < rep["buckets"][-1]["mean_pred"]

    def test_directional_but_biased(self):
        # realized = 0.2·predicted: rank-perfect (spearman 1.0) but the
        # magnitude is 5× too big — the real DecisionScorer's tail failure
        # mode. Must NOT be called WELL_CALIBRATED.
        pred = [float(i - 50) for i in range(100)]
        rep = calibration_report([(p, 0.2 * p) for p in pred])
        assert rep["verdict"] == "DIRECTIONAL_BUT_BIASED"
        assert rep["spearman"] == 1.0
        assert rep["monotone_fraction"] == 1.0
        assert rep["mean_abs_decile_error"] == pytest.approx(20.0)
        assert rep["mean_abs_decile_error"] > BIAS_TOL_PCT

    def test_miscalibrated_anticorrelated(self):
        # Prediction is exactly backwards → no usable rank skill.
        pred = [float(i - 50) for i in range(100)]
        rep = calibration_report([(p, -p) for p in pred])
        assert rep["verdict"] == "MISCALIBRATED"
        assert rep["spearman"] == -1.0
        assert rep["pearson"] == -1.0
        assert rep["monotone_fraction"] == 0.0
        assert rep["spearman"] < SPEARMAN_MIN

    def test_weak_signal(self):
        # Coarse decile means stay perfectly monotone (period-10 zero-sum
        # noise cancels exactly per decile → bias 0, monotone 1.0) but
        # per-pair rank skill is eroded into the WEAK band.
        pred = [(i - 50) * 0.2 for i in range(100)]
        base = [(j - 4.5) for j in range(10)]   # sums to 0 over a period
        realized = [pred[i] + 10 * base[i % 10] for i in range(100)]
        rep = calibration_report(list(zip(pred, realized)))
        assert rep["verdict"] == "WEAK_SIGNAL"
        assert rep["spearman"] == pytest.approx(0.2725, abs=1e-4)
        assert rep["monotone_fraction"] == 1.0
        assert rep["mean_abs_decile_error"] == 0.0
        # The defining band for WEAK_SIGNAL.
        assert SPEARMAN_MIN <= rep["spearman"] < SPEARMAN_GOOD

    def test_insufficient_data_boundary(self):
        # MIN_PAIRS - 1 finite pairs → INSUFFICIENT_DATA, no metrics.
        rep = calibration_report([(float(i), float(i))
                                  for i in range(MIN_PAIRS - 1)])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["status"] == "insufficient_data"
        assert rep["n"] == MIN_PAIRS - 1
        assert rep["spearman"] is None
        assert rep["buckets"] == []
        # Exactly MIN_PAIRS → crosses into a real report.
        rep2 = calibration_report([(float(i), float(i))
                                   for i in range(MIN_PAIRS)])
        assert rep2["status"] == "ok"
        assert rep2["n"] == MIN_PAIRS

    def test_non_finite_pairs_are_dropped(self):
        good = [(float(i), float(i)) for i in range(40)]
        poisoned = good + [(float("nan"), 1.0), (1.0, float("inf")),
                           (float("-inf"), float("nan")), (None, 2.0)]
        rep = calibration_report(poisoned)
        # The 4 non-finite/None rows are excluded; the 40 good remain.
        assert rep["n"] == 40
        assert rep["verdict"] == "WELL_CALIBRATED"
        assert rep["spearman"] == 1.0

    def test_constant_prediction_has_no_rank_skill(self):
        # A degenerate scorer that always says the same thing has zero rank
        # variance — spearman must be 0.0 (not NaN) and the verdict must
        # not falsely claim calibration.
        rep = calibration_report([(5.0, float(i)) for i in range(60)])
        assert rep["spearman"] == 0.0
        assert rep["verdict"] == "MISCALIBRATED"


class _FakeScorer:
    """predict() echoes a chosen feature so calibration plumbing (SELL sign
    flip, kwarg names, defaults) can be tested without a trained MLP."""

    def __init__(self, field: str = "ml_score") -> None:
        self.field = field

    def predict(self, **kw) -> float:
        return float(kw[self.field])


class _RaisingScorer:
    def predict(self, **kw):
        raise ValueError("model/feature shape mismatch")


class TestScorerCalibration:
    def test_buy_records_perfect(self):
        # ml_score == forward_return_5d, scorer echoes ml_score → perfect.
        recs = [{"ml_score": float(i - 30), "forward_return_5d": float(i - 30),
                 "action": "BUY", "ticker": "NVDA"} for i in range(80)]
        rep = scorer_calibration(_FakeScorer(), recs)
        assert rep["verdict"] == "WELL_CALIBRATED"
        assert rep["spearman"] == 1.0
        assert rep["n"] == 80

    def test_sell_sign_is_flipped(self):
        # SELL-only: a drop after a SELL was the RIGHT call, so the aligned
        # target is -forward_return_5d. With ml_score == predicted and
        # forward_return_5d == -ml_score, the flip makes it perfectly
        # calibrated. WITHOUT the flip the pairs would be (v, -v) →
        # spearman -1 → MISCALIBRATED. Asserting WELL_CALIBRATED here is the
        # regression lock on the sign-alignment (mirrors train_scorer).
        recs = [{"ml_score": float(i - 30),
                 "forward_return_5d": -float(i - 30),
                 "action": "SELL", "ticker": "XLF"} for i in range(80)]
        rep = scorer_calibration(_FakeScorer(), recs)
        assert rep["verdict"] == "WELL_CALIBRATED"
        assert rep["spearman"] == 1.0

    def test_predict_exceptions_are_skipped_not_fatal(self):
        recs = [{"ml_score": float(i), "forward_return_5d": float(i),
                 "action": "BUY"} for i in range(50)]
        rep = scorer_calibration(_RaisingScorer(), recs)
        # Every predict raised → zero usable pairs → graceful, not a crash.
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_missing_forward_return_skipped(self):
        recs = ([{"ml_score": float(i), "forward_return_5d": float(i),
                  "action": "BUY"} for i in range(40)]
                + [{"ml_score": 9.0, "forward_return_5d": None,
                    "action": "BUY"}])
        rep = scorer_calibration(_FakeScorer(), recs)
        assert rep["n"] == 40   # the None-forward_return row is excluded
        assert rep["verdict"] == "WELL_CALIBRATED"
