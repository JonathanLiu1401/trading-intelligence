"""Pins ml/score_agreement — the ML-vs-LLM drift monitor.

Asserts the hand-rolled stats against values computed by hand / known
closed-form results (no numpy to cross-check against in CI), and pins the
divergence-bucketing contract that the dashboard reads.
"""
from __future__ import annotations

import math

from ml.score_agreement import (
    compute_agreement,
    pearson,
    spearman,
)


class TestPearson:
    def test_perfect_positive(self):
        assert math.isclose(pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0, abs_tol=1e-9)

    def test_perfect_negative(self):
        assert math.isclose(pearson([1, 2, 3, 4], [4, 3, 2, 1]), -1.0, abs_tol=1e-9)

    def test_known_value(self):
        # sxy=8, sxx=syy=10 → r = 8/10 = 0.8 (worked by hand).
        assert math.isclose(
            pearson([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]), 0.8, abs_tol=1e-9
        )

    def test_degenerate_flat_series_is_zero(self):
        assert pearson([5, 5, 5], [1, 2, 3]) == 0.0

    def test_too_few_points_is_zero(self):
        assert pearson([1], [2]) == 0.0


class TestSpearman:
    def test_monotonic_nonlinear_is_one(self):
        # Strictly increasing but nonlinear → Pearson < 1, Spearman == 1.
        xs = [1, 2, 3, 4, 5]
        ys = [1, 4, 9, 16, 25]
        assert math.isclose(spearman(xs, ys), 1.0, abs_tol=1e-9)
        assert pearson(xs, ys) < 1.0

    def test_handles_ties(self):
        # Two tied x's share average rank; result stays in [-1, 1].
        r = spearman([1, 1, 2, 3], [1, 2, 3, 4])
        assert -1.0 <= r <= 1.0


class TestComputeAgreement:
    def test_empty_is_safe(self):
        out = compute_agreement([])
        assert out["n"] == 0
        assert out["pearson"] == 0.0
        assert out["model_overconfident"] == []

    def test_ai_score_zero_is_excluded_from_overlap(self):
        rows = [
            {"ml_score": 8.0, "ai_score": 0.0},  # LLM never graded → dropped
            {"ml_score": 5.0, "ai_score": 5.0},
            {"ml_score": 6.0, "ai_score": 6.0},
        ]
        assert compute_agreement(rows)["n"] == 2

    def test_bias_sign_when_model_runs_hot(self):
        rows = [
            {"ml_score": 9.0, "ai_score": 5.0},
            {"ml_score": 8.0, "ai_score": 4.0},
        ]
        out = compute_agreement(rows)
        assert out["bias_ml_minus_ai"] == 4.0
        assert math.isclose(out["mean_abs_divergence"], 4.0)
        assert math.isclose(out["rmse"], 4.0)

    def test_overconfident_bucket_captures_model_false_positives(self):
        rows = [
            {"ml_score": 9.5, "ai_score": 1.0, "title": "noise flagged hot"},
            {"ml_score": 5.0, "ai_score": 5.0, "title": "agree"},
        ]
        out = compute_agreement(rows)
        assert out["strong_disagreement_count"] == 1
        assert len(out["model_overconfident"]) == 1
        assert out["model_overconfident"][0]["title"] == "noise flagged hot"
        assert out["model_overconfident"][0]["gap"] == 8.5
        assert out["model_underconfident"] == []

    def test_underconfident_bucket_captures_expensive_misses(self):
        rows = [
            {"ml_score": 1.0, "ai_score": 9.0, "title": "model would have hidden this"},
        ]
        out = compute_agreement(rows)
        assert len(out["model_underconfident"]) == 1
        assert out["model_underconfident"][0]["gap"] == 8.0
        assert out["model_overconfident"] == []

    def test_overconfident_sorted_worst_first(self):
        rows = [
            {"ml_score": 9.0, "ai_score": 4.0, "title": "gap5"},
            {"ml_score": 10.0, "ai_score": 1.0, "title": "gap9"},
            {"ml_score": 8.0, "ai_score": 3.0, "title": "gap5b"},
        ]
        out = compute_agreement(rows)
        assert out["model_overconfident"][0]["title"] == "gap9"
