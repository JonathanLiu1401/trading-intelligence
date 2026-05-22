"""Tests for DecisionScorer quantile-mapping calibration.

`predict_calibrated()` / `_raw_to_calibrated()` map a raw MLP prediction
through two persisted 101-point quantile tables — predictions and realized
labels — onto the empirical forward-return distribution. This is the
documented fix for the `DIRECTIONAL_BUT_BIASED` OOS verdict: it preserves
the model's trustworthy rank ordering while correcting the biased %
magnitude.

These tests check the *business logic*: the quantile-mapping math is
correct, monotonic, bounded by the empirical label support, backward
compatible with legacy pickles, and never raises.
"""
from __future__ import annotations

import numpy as np
import pytest

from paper_trader.ml.decision_scorer import DecisionScorer, train_scorer


class _FixedModel:
    """Model whose predict() returns a controlled value (one row per call)."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.array([self.value] * n, dtype=np.float64)


def _scorer_with_quantiles(pred_q, label_q, model_value=50.0) -> DecisionScorer:
    """A trained scorer with controlled quantile tables and a fixed model."""
    s = DecisionScorer()
    s._model = _FixedModel(model_value)
    s._scaler = None
    s._trained = True
    s._n_train = 1000
    s._pred_quantiles = np.asarray(pred_q, dtype=np.float64)
    s._label_quantiles = np.asarray(label_q, dtype=np.float64)
    return s


# Prediction distribution: a clean 0..100 ramp so percentile(raw) == raw.
_PRED_Q = list(range(0, 101))
# Label distribution: realized returns uniformly -10%..+10% (step 0.2).
_LABEL_Q = [round(-10.0 + i * 0.2, 4) for i in range(101)]


class TestRawToCalibratedMath:
    """The quantile map must read the label value at the prediction's rank."""

    def test_median_prediction_maps_to_median_label(self):
        s = _scorer_with_quantiles(_PRED_Q, _LABEL_Q)
        # raw=50 → 50th pred percentile → 50th label = -10 + 50*0.2 = 0.0
        assert s._raw_to_calibrated(50.0) == pytest.approx(0.0, abs=1e-6)

    def test_top_prediction_maps_to_top_label(self):
        s = _scorer_with_quantiles(_PRED_Q, _LABEL_Q)
        assert s._raw_to_calibrated(100.0) == pytest.approx(10.0, abs=1e-6)

    def test_bottom_prediction_maps_to_bottom_label(self):
        s = _scorer_with_quantiles(_PRED_Q, _LABEL_Q)
        assert s._raw_to_calibrated(0.0) == pytest.approx(-10.0, abs=1e-6)

    def test_off_distribution_raw_clamps_to_label_support(self):
        """A raw far past the training prediction range must NOT extrapolate
        past the empirical label support — it clamps to the label max/min."""
        s = _scorer_with_quantiles(_PRED_Q, _LABEL_Q)
        # raw=500 is wild extrapolation; percentile clamps to 100 → label max.
        assert s._raw_to_calibrated(500.0) == pytest.approx(10.0, abs=1e-6)
        assert s._raw_to_calibrated(-500.0) == pytest.approx(-10.0, abs=1e-6)

    def test_calibration_is_monotonic_in_raw(self):
        s = _scorer_with_quantiles(_PRED_Q, _LABEL_Q)
        raws = [-50, -5, 0, 12.5, 37.5, 62.5, 88, 150]
        cals = [s._raw_to_calibrated(r) for r in raws]
        assert all(c is not None for c in cals)
        # Non-decreasing — preserves the model's rank ordering exactly.
        assert cals == sorted(cals)

    def test_calibrated_always_within_label_support(self):
        """Whatever the raw, the calibrated value stays inside [min,max] label."""
        s = _scorer_with_quantiles(_PRED_Q, _LABEL_Q)
        for raw in (-9999.0, -3.0, 0.0, 47.0, 99.9, 9999.0):
            c = s._raw_to_calibrated(raw)
            assert c is not None
            assert min(_LABEL_Q) - 1e-6 <= c <= max(_LABEL_Q) + 1e-6

    def test_biased_magnitude_is_corrected(self):
        """A model that predicts +50% (top of its own range) should calibrate
        to the realized top-rank return (+10%), not echo the biased +50%."""
        s = _scorer_with_quantiles(_PRED_Q, _LABEL_Q, model_value=50.0)
        # raw 50 sits at the 50th pred percentile here -> 0% realized.
        # Shift label dist so the realized top is +10 but predictions inflate.
        cal = s._raw_to_calibrated(100.0)
        assert cal == pytest.approx(10.0, abs=1e-6)
        # The honest magnitude (10%) is far below a naive read of raw 100.
        assert abs(cal) < 100.0


class TestLegacyAndUntrained:
    """Backward compatibility and graceful degradation."""

    def test_legacy_pickle_without_label_quantiles_returns_none(self):
        s = _scorer_with_quantiles(_PRED_Q, _LABEL_Q)
        s._label_quantiles = None  # legacy pickle: field absent
        assert s._raw_to_calibrated(50.0) is None

    def test_missing_pred_quantiles_returns_none(self):
        s = _scorer_with_quantiles(_PRED_Q, _LABEL_Q)
        s._pred_quantiles = None  # cannot locate the rank → no calibration
        assert s._raw_to_calibrated(50.0) is None

    def test_untrained_scorer_calibrated_is_none(self):
        s = DecisionScorer()
        s._trained = False
        out = s.predict_calibrated(
            ml_score=3.0, rsi=40, macd=0.2, mom5=1.0, mom20=2.0,
            regime_mult=1.0, ticker="NVDA",
        )
        assert out is None

    def test_non_finite_raw_returns_none(self):
        s = _scorer_with_quantiles(_PRED_Q, _LABEL_Q)
        assert s._raw_to_calibrated(float("nan")) is None
        assert s._raw_to_calibrated(float("inf")) is None


class TestPredictWithMetaContract:
    """predict_with_meta must always carry the `calibrated` key."""

    def test_calibrated_key_present_when_trained(self):
        s = _scorer_with_quantiles(_PRED_Q, _LABEL_Q, model_value=70.0)
        meta = s.predict_with_meta(
            ml_score=3.0, rsi=40, macd=0.2, mom5=1.0, mom20=2.0,
            regime_mult=1.0, ticker="NVDA",
        )
        assert "calibrated" in meta
        # model returns 70 → 70th pred pct → label = -10 + 70*0.2 = +4.0
        assert meta["calibrated"] == pytest.approx(4.0, abs=1e-6)

    def test_calibrated_key_present_when_untrained(self):
        s = DecisionScorer()
        s._trained = False
        meta = s.predict_with_meta(
            ml_score=0.0, rsi=None, macd=None, mom5=None, mom20=None,
            regime_mult=1.0, ticker="",
        )
        assert "calibrated" in meta and meta["calibrated"] is None

    def test_predict_calibrated_matches_meta(self):
        s = _scorer_with_quantiles(_PRED_Q, _LABEL_Q, model_value=30.0)
        common = dict(ml_score=2.0, rsi=55, macd=-0.1, mom5=-1.0,
                      mom20=0.5, regime_mult=0.6, ticker="SOXL")
        assert s.predict_calibrated(**common) == s.predict_with_meta(**common)["calibrated"]


def _synthetic_records(n: int = 80) -> list[dict]:
    """n distinct outcome records with a controlled forward-return spread."""
    recs = []
    for i in range(n):
        # Spread realized returns across roughly -15%..+15%.
        fr = -15.0 + (30.0 * i / (n - 1))
        recs.append({
            "ticker": "NVDA" if i % 2 else "SOXL",
            "sim_date": f"2024-{1 + i % 12:02d}-{1 + i % 27:02d}-{i}",
            "action": "BUY",
            "ml_score": 1.0 + (i % 5),
            "rsi": 30.0 + (i % 40),
            "macd": -0.5 + (i % 10) * 0.1,
            "mom5": -3.0 + (i % 7),
            "mom20": -5.0 + (i % 11),
            "regime_mult": 1.0,
            "forward_return_5d": round(fr, 4),
            "return_pct": 10.0,
        })
    return recs


class TestTrainScorerPersistsLabelQuantiles:
    """End-to-end: train_scorer must write a usable label-quantile table."""

    def test_pickle_carries_label_quantiles(self):
        result = train_scorer(_synthetic_records())
        assert result["status"] == "ok"
        # 101 breakpoints (percentiles 0..100), same shape as pred_quantiles.
        assert result["n_label_quantiles"] == 101

    def test_reloaded_scorer_calibrates_within_label_support(self):
        recs = _synthetic_records()
        train_scorer(recs)
        # Fresh load from the (conftest-redirected) pickle.
        s = DecisionScorer()
        assert s.is_trained
        assert s._label_quantiles is not None
        assert len(s._label_quantiles) == 101

        labels = [r["forward_return_5d"] for r in recs]
        lo, hi = min(labels), max(labels)
        # Probe a spread of feature vectors — every calibrated value must
        # land inside the empirical label support, never extrapolate past it.
        for ml in (-2.0, 0.0, 3.0, 8.0):
            cal = s.predict_calibrated(
                ml_score=ml, rsi=45, macd=0.1, mom5=1.0, mom20=2.0,
                regime_mult=1.0, ticker="NVDA",
            )
            assert cal is not None
            assert lo - 1e-3 <= cal <= hi + 1e-3, (
                f"calibrated {cal} escaped label support [{lo}, {hi}]")

    def test_calibrated_differs_from_raw_when_model_extrapolates(self):
        """The whole point: when the MLP raw output sits outside the label
        support, the calibrated reading must be pulled back inside it."""
        train_scorer(_synthetic_records())
        s = DecisionScorer()
        labels_hi = 15.0  # synthetic max realized return
        meta = s.predict_with_meta(
            ml_score=50.0, rsi=20, macd=2.0, mom5=12.0, mom20=20.0,
            regime_mult=1.0, ticker="NVDA",
        )
        # If the raw blew past the label support, calibrated must not.
        if abs(meta["raw"]) > labels_hi + 5.0:
            assert meta["calibrated"] is not None
            assert abs(meta["calibrated"]) <= labels_hi + 1e-3
