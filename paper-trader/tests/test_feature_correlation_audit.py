"""Tests for paper_trader.ml.feature_correlation_audit.

Assert specific numeric verdicts (not just "no crash") so a future
refactor of the Spearman / VIF math breaks loudly and obviously.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_trader.ml.feature_correlation_audit import (
    MIN_OBS,
    MODERATE_THR,
    NUMERIC_FEATURES,
    SEVERE_THR,
    VIF_MODERATE,
    VIF_SEVERE,
    _spearman,
    _vif_via_ols,
    analyze,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _row(**overrides) -> dict:
    """Build a joint-complete numeric row (all 10 features present)."""
    base = {
        "ml_score": 5.0, "rsi": 50.0, "macd": 0.5,
        "mom5": 1.0, "mom20": 2.0, "regime_mult": 1.0,
        "vol_ratio": 1.2, "bb_position": 0.0,
        "news_urgency": 30.0, "news_article_count": 2.0,
        # Non-feature fields the loader doesn't need but production rows carry.
        "action": "BUY", "ticker": "NVDA", "forward_return_5d": 1.0,
        "sim_date": "2025-01-01", "run_id": 1,
    }
    base.update(overrides)
    return base


class TestSpearmanCore:
    """Pin the tie-aware Spearman implementation against textbook cases."""

    def test_perfect_positive(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        assert _spearman(a, b) == pytest.approx(1.0)

    def test_perfect_negative(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        assert _spearman(a, b) == pytest.approx(-1.0)

    def test_constant_returns_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        # A constant predictor has no rank skill — must read 0.0, never NaN.
        assert _spearman(a, b) == 0.0

    def test_short_series_returns_zero(self):
        a = np.array([1.0])
        b = np.array([2.0])
        assert _spearman(a, b) == 0.0


class TestVIF:
    """Closed-form VIF should match the textbook formula on a known matrix."""

    def test_independent_columns_vif_near_one(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(500, 3))
        vifs = _vif_via_ols(X)
        assert all(v is not None for v in vifs)
        # Truly independent columns produce VIF very close to 1.0.
        for v in vifs:
            assert 0.8 <= v <= 1.3, f"VIF should be ~1.0 for independent, got {v}"

    def test_perfectly_collinear_columns_vif_explodes(self):
        # X[:, 1] = 2 * X[:, 0] — perfect collinearity, VIF should hit the cap.
        rng = np.random.default_rng(42)
        col0 = rng.normal(size=500)
        col1 = 2.0 * col0
        col2 = rng.normal(size=500)
        X = np.column_stack([col0, col1, col2])
        vifs = _vif_via_ols(X)
        # The numerical guard caps R² at 0.9999, so VIF ≤ 1/(1-0.9999) = 10000.
        # Either column 0 or 1 (or both) must hit that ceiling.
        assert vifs[0] is not None and vifs[1] is not None
        assert max(vifs[0], vifs[1]) > 1000.0, \
            f"Perfect collinearity should produce massive VIF, got {vifs}"
        # Column 2 is independent — should stay near 1.0.
        assert vifs[2] is not None
        assert 0.8 <= vifs[2] <= 1.3

    def test_constant_column_returns_none(self):
        """A zero-variance column has undefined VIF — must return None,
        never crash or fabricate a value."""
        X = np.column_stack([
            np.array([5.0] * 100),  # constant
            np.random.default_rng(0).normal(size=100),
        ])
        vifs = _vif_via_ols(X)
        assert vifs[0] is None


class TestAnalyzeInsufficientData:
    def test_missing_file_is_honest(self, tmp_path):
        rep = analyze(outcomes_path=tmp_path / "does_not_exist.jsonl")
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_too_few_rows_does_not_crash(self, tmp_path):
        p = tmp_path / "outcomes.jsonl"
        _write(p, [_row() for _ in range(MIN_OBS - 1)])
        rep = analyze(outcomes_path=p)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] < MIN_OBS

    def test_incomplete_rows_dropped_from_joint_count(self, tmp_path):
        """A row with one missing feature must be dropped — joint-complete
        is the only honest sample size for cross-feature comparison."""
        p = tmp_path / "outcomes.jsonl"
        rows = [_row() for _ in range(60)]
        for r in rows[:50]:
            r["bb_position"] = None  # drop these 50 joint
        _write(p, rows)
        rep = analyze(outcomes_path=p)
        # Only 10 rows remain joint-complete → insufficient.
        assert rep["n"] == 10
        assert rep["verdict"] == "INSUFFICIENT_DATA"


class TestAnalyzeLowCollinearity:
    def test_random_features_yields_low_verdict(self, tmp_path):
        rng = np.random.default_rng(123)
        p = tmp_path / "outcomes.jsonl"
        rows = []
        for i in range(MIN_OBS * 3):
            r = _row(
                ml_score=float(rng.normal(0, 5)),
                rsi=float(rng.uniform(20, 80)),
                macd=float(rng.normal(0, 2)),
                mom5=float(rng.normal(0, 5)),
                mom20=float(rng.normal(0, 10)),
                regime_mult=float(rng.choice([0.3, 0.6, 1.0])),
                vol_ratio=float(rng.uniform(0.5, 3.0)),
                bb_position=float(rng.uniform(-2, 2)),
                news_urgency=float(rng.uniform(0, 100)),
                news_article_count=float(rng.uniform(1, 10)),
            )
            rows.append(r)
        _write(p, rows)
        rep = analyze(outcomes_path=p)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "LOW_COLLINEARITY"
        assert rep["max_abs_corr"] < MODERATE_THR
        assert rep["max_vif"] is None or rep["max_vif"] < VIF_MODERATE


class TestAnalyzeSevereCollinearity:
    def test_duplicate_feature_triggers_severe(self, tmp_path):
        """If two features carry IDENTICAL noise, Spearman = 1.0 and the
        verdict MUST be SEVERE."""
        rng = np.random.default_rng(7)
        p = tmp_path / "outcomes.jsonl"
        rows = []
        for i in range(MIN_OBS * 2):
            # ALL features random EXCEPT rsi and bb_position are the same value.
            shared = float(rng.uniform(20, 80))
            r = _row(
                ml_score=float(rng.normal(0, 5)),
                rsi=shared,           # rsi == bb_position
                macd=float(rng.normal(0, 2)),
                mom5=float(rng.normal(0, 5)),
                mom20=float(rng.normal(0, 10)),
                regime_mult=float(rng.choice([0.3, 0.6, 1.0])),
                vol_ratio=float(rng.uniform(0.5, 3.0)),
                bb_position=shared,   # bb_position == rsi → Spearman = 1.0
                news_urgency=float(rng.uniform(0, 100)),
                news_article_count=float(rng.uniform(1, 10)),
            )
            rows.append(r)
        _write(p, rows)
        rep = analyze(outcomes_path=p)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "SEVERE_COLLINEARITY"
        # The most-correlated pair must be the duplicated one with rho==1.
        top = rep["pairs"][0]
        assert {top["a"], top["b"]} == {"rsi", "bb_position"}
        assert top["spearman"] == pytest.approx(1.0, abs=1e-6)
        assert rep["max_abs_corr"] >= SEVERE_THR

    def test_severe_collinearity_exit_code(self, tmp_path):
        """The CLI returns exit code 1 on SEVERE so shell pipelines can
        gate on $?."""
        from paper_trader.ml.feature_correlation_audit import main

        rng = np.random.default_rng(9)
        p = tmp_path / "outcomes.jsonl"
        rows = []
        for i in range(MIN_OBS * 2):
            v = float(rng.normal(0, 1))
            rows.append(_row(rsi=v, bb_position=v))
        _write(p, rows)
        # exit code 1 == SEVERE.
        assert main(["--path", str(p), "--json"]) == 1


class TestAnalyzePairOrdering:
    def test_pairs_sorted_by_absolute_correlation(self, tmp_path):
        rng = np.random.default_rng(42)
        p = tmp_path / "outcomes.jsonl"
        rows = []
        for i in range(MIN_OBS * 3):
            # macd ≈ -ml_score (strong negative correlation, |rho| ≈ 1)
            x = float(rng.normal(0, 5))
            r = _row(
                ml_score=x, macd=-x,
                rsi=float(rng.uniform(20, 80)),
                mom5=float(rng.normal(0, 5)),
                mom20=float(rng.normal(0, 10)),
                regime_mult=float(rng.choice([0.3, 0.6, 1.0])),
                vol_ratio=float(rng.uniform(0.5, 3.0)),
                bb_position=float(rng.uniform(-2, 2)),
                news_urgency=float(rng.uniform(0, 100)),
                news_article_count=float(rng.uniform(1, 10)),
            )
            rows.append(r)
        _write(p, rows)
        rep = analyze(outcomes_path=p)
        # Top pair: ml_score ↔ macd, with rho == -1 (negative perfect).
        top = rep["pairs"][0]
        assert {top["a"], top["b"]} == {"ml_score", "macd"}
        assert top["spearman"] == pytest.approx(-1.0, abs=1e-6)
        # All remaining pairs must have |spearman| <= |top|.
        for p_row in rep["pairs"][1:]:
            assert abs(p_row["spearman"]) <= abs(top["spearman"])


class TestNumericFeaturesCoverage:
    def test_numeric_features_match_outcomes_schema(self):
        """The NUMERIC_FEATURES list must mirror the on-disk JSONL keys
        for the 10 numeric inputs — NOT decision_scorer.FEATURE_NAMES
        directly, because the JSONL persists ``bb_position`` (legacy
        spelling) while ``build_features`` accepts ``bb_pos`` (kwarg
        spelling). Drift between this list and the outcome JSONL would
        silently produce all-empty columns and INSUFFICIENT_DATA on a
        valid corpus."""
        from paper_trader.ml.decision_scorer import FEATURE_NAMES
        # Same length and same order as the model's numeric block.
        assert len(NUMERIC_FEATURES) == 10
        assert NUMERIC_FEATURES[:7] == FEATURE_NAMES[:7]
        # bb_position is the JSONL spelling; bb_pos is the kwarg.
        # Decision-scorer FEATURE_NAMES uses bb_pos; outcomes use bb_position.
        assert NUMERIC_FEATURES[7] == "bb_position"
        assert FEATURE_NAMES[7] == "bb_pos"
        # The trailing two news features are the same in both schemas.
        assert NUMERIC_FEATURES[8:10] == FEATURE_NAMES[8:10]
