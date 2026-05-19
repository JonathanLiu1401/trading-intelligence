"""Tests for ``DecisionScorer.feature_contributions``.

This method is the per-feature signed attribution that powers
``/api/scorer-attribution`` and the conviction-board "why does the scorer
predict this?" panel. It was previously uncovered, so a refactor that
silently inverted the contribution sign, returned the wrong feature names,
or broke the residual identity could ship to a quant operator without any
test catching it. These tests lock the structural and numeric invariants
the API consumers depend on.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from paper_trader.ml.decision_scorer import (
    DecisionScorer,
    FEATURE_NAMES,
    N_FEATURES,
    PRED_CLAMP_PCT,
    train_scorer,
)


def _outcome(sim_date: str, ml_score: float, fwd: float,
             ticker: str = "NVDA") -> dict:
    return {
        "ticker": ticker, "action": "BUY",
        "ml_score": ml_score, "rsi": 50.0, "macd": 0.1,
        "mom5": 0.0, "mom20": 0.0, "regime_mult": 1.0,
        "vol_ratio": 1.0, "bb_position": 0.0,
        "news_urgency": 50.0, "news_article_count": 1.0,
        "forward_return_5d": fwd, "return_pct": 10.0,
        "sim_date": sim_date,
    }


def _train_predictable(tmp_path, monkeypatch) -> DecisionScorer:
    """Build a scorer trained on a clean monotone ml_score → fwd relationship."""
    import paper_trader.ml.decision_scorer as ds
    monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "scorer_attr.pkl")
    recs = []
    for i in range(40):
        sc = (i - 20) * 0.5  # -10.0 .. +9.5
        recs.append(_outcome(f"2025-04-{i+1:02d}", ml_score=sc, fwd=sc * 1.5))
    assert train_scorer(recs)["status"] == "ok"
    s = DecisionScorer()
    assert s.is_trained
    return s


class TestUntrainedAttribution:
    def test_untrained_returns_empty_safe_dict(self):
        s = DecisionScorer()
        assert not s.is_trained
        out = s.feature_contributions(ml_score=2.0, rsi=None, macd=None,
                                      mom5=None, mom20=None, regime_mult=1.0,
                                      ticker="NVDA")
        # Schema lock: every key /api/scorer-attribution renders must exist.
        assert out["trained"] is False
        assert out["contributions"] == []
        assert out["pred"] == 0.0
        assert out["pred_baseline"] == 0.0
        assert out["interaction_residual"] == 0.0
        assert out["off_distribution"] is False


class TestAttributionStructure:
    def test_returns_one_row_per_feature(self, tmp_path, monkeypatch):
        s = _train_predictable(tmp_path, monkeypatch)
        out = s.feature_contributions(ml_score=2.0, rsi=50, macd=0.1,
                                      mom5=0.0, mom20=0.0, regime_mult=1.0,
                                      ticker="NVDA")
        assert out["trained"] is True
        assert len(out["contributions"]) == N_FEATURES
        names = [c["feature"] for c in out["contributions"]]
        assert set(names) == set(FEATURE_NAMES)

    def test_feature_names_match_build_features_order(
            self, tmp_path, monkeypatch):
        """A refactor that re-orders ``build_features`` outputs without
        updating ``FEATURE_NAMES`` would mislabel the attribution panel.
        Lock that the raw_value reported for each named feature matches the
        value at that slot in ``build_features``."""
        from paper_trader.ml.decision_scorer import build_features
        s = _train_predictable(tmp_path, monkeypatch)
        feats = build_features(ml_score=2.0, rsi=55.0, macd=0.3, mom5=1.0,
                               mom20=2.0, regime_mult=1.0, ticker="NVDA",
                               vol_ratio=1.2, bb_pos=0.3,
                               news_urgency=60.0, news_article_count=2.0)
        out = s.feature_contributions(ml_score=2.0, rsi=55.0, macd=0.3,
                                      mom5=1.0, mom20=2.0, regime_mult=1.0,
                                      ticker="NVDA", vol_ratio=1.2, bb_pos=0.3,
                                      news_urgency=60.0,
                                      news_article_count=2.0)
        # Re-key contributions by feature name; map to FEATURE_NAMES order to
        # compare directly against build_features positions.
        rows = {c["feature"]: c["raw_value"] for c in out["contributions"]}
        for i, name in enumerate(FEATURE_NAMES):
            assert rows[name] == pytest.approx(round(float(feats[i]), 4)), (
                f"raw_value at slot {i} ({name}) drifted from build_features")

    def test_contributions_sorted_by_magnitude(self, tmp_path, monkeypatch):
        s = _train_predictable(tmp_path, monkeypatch)
        out = s.feature_contributions(ml_score=8.0, rsi=70, macd=0.5,
                                      mom5=5.0, mom20=3.0, regime_mult=1.0,
                                      ticker="NVDA")
        mags = [abs(c["contribution"]) for c in out["contributions"]]
        assert mags == sorted(mags, reverse=True), (
            f"contributions must be sorted by |impact| descending: {mags}")


class TestAttributionIdentity:
    def test_pred_decomposes_into_baseline_plus_sum_plus_residual(
            self, tmp_path, monkeypatch):
        """The Shapley-style ablation identity must hold to within float noise:

            pred_raw == pred_baseline + sum(contributions) + interaction_residual

        If this identity breaks, the attribution stops summing to the
        prediction it explains — operators reading the panel would mistrust
        a quantitatively wrong decomposition.
        """
        s = _train_predictable(tmp_path, monkeypatch)
        out = s.feature_contributions(ml_score=6.0, rsi=70, macd=0.5,
                                      mom5=3.0, mom20=2.0, regime_mult=1.0,
                                      ticker="NVDA")
        recomposed = (out["pred_baseline"]
                      + sum(c["contribution"] for c in out["contributions"])
                      + out["interaction_residual"])
        # pred_raw is reported unclamped; pred is the clamped value an
        # operator acts on. The identity is on the raw side.
        assert out["pred_raw"] == pytest.approx(recomposed, abs=1e-3), (
            f"attribution identity broken: pred_raw={out['pred_raw']} "
            f"recomposed={recomposed}")

    def test_dominant_feature_is_the_one_we_swept(self, tmp_path, monkeypatch):
        """On a scorer trained with ml_score as the dominant driver, the
        attribution must surface ml_score as the top-magnitude contributor —
        otherwise the panel would explain predictions with the WRONG feature.
        """
        s = _train_predictable(tmp_path, monkeypatch)
        out = s.feature_contributions(ml_score=8.0, rsi=50, macd=0.1,
                                      mom5=0.0, mom20=0.0, regime_mult=1.0,
                                      ticker="NVDA")
        top = out["contributions"][0]
        assert top["feature"] == "ml_score"


class TestAttributionFailureModes:
    def test_off_distribution_flag_propagates(self, tmp_path, monkeypatch):
        """When the model extrapolates past PRED_CLAMP_PCT, the attribution
        must surface ``off_distribution=True`` so panels don't render the
        clamped value as a confident in-distribution call."""
        s = _train_predictable(tmp_path, monkeypatch)

        # Replace the model with one whose output exceeds the clamp band.
        class _Extrap:
            def predict(self, X):
                # First row = full prediction; subsequent rows are the
                # baseline + ablation batch. Make ONLY the full prediction
                # blow past PRED_CLAMP_PCT so off_distribution fires.
                out = np.zeros(len(X), dtype=np.float64)
                out[0] = 999.0
                return out
        s._model = _Extrap()
        out = s.feature_contributions(ml_score=0.0, rsi=50, macd=0.0,
                                      mom5=0.0, mom20=0.0, regime_mult=1.0,
                                      ticker="NVDA")
        assert out["off_distribution"] is True
        # pred is clamped, pred_raw exposes the unbounded value (honesty).
        assert abs(out["pred"]) <= PRED_CLAMP_PCT
        assert out["pred_raw"] == pytest.approx(999.0)

    def test_model_raise_returns_safe_error_dict(self, tmp_path, monkeypatch):
        """A predict() that raises (e.g. a feature added to build_features
        without retraining → shape mismatch) must degrade to a safe dict,
        not propagate the exception. Mirrors the predict_with_meta failure
        contract — the dashboard's /api/scorer-attribution must never 500."""
        s = _train_predictable(tmp_path, monkeypatch)

        class _Raiser:
            def predict(self, X):
                raise ValueError("shapes (3,17) and (10,) not aligned")
        s._model = _Raiser()
        out = s.feature_contributions(ml_score=0.0, rsi=50, macd=0.0,
                                      mom5=0.0, mom20=0.0, regime_mult=1.0,
                                      ticker="NVDA")
        # Trained=True but contributions empty + low-trust flag set.
        assert out["trained"] is True
        assert out["contributions"] == []
        assert out["pred"] == 0.0
        assert out["off_distribution"] is True
        assert "error" in out

    def test_non_finite_model_output_flagged(self, tmp_path, monkeypatch):
        s = _train_predictable(tmp_path, monkeypatch)

        class _NanModel:
            def predict(self, X):
                return np.full(len(X), float("nan"), dtype=np.float64)
        s._model = _NanModel()
        out = s.feature_contributions(ml_score=0.0, rsi=50, macd=0.0,
                                      mom5=0.0, mom20=0.0, regime_mult=1.0,
                                      ticker="NVDA")
        # A NaN-poisoned batch must NOT propagate non-finite values through
        # the attribution rows. The safe-fallback empty-contributions dict
        # has pred=0.0 and off_distribution=True.
        assert math.isfinite(out["pred"])
        assert out["pred"] == 0.0
        assert out["off_distribution"] is True
        assert out["contributions"] == []
