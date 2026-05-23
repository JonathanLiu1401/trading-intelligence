"""Tests for DecisionScorer.predict_percentile — the rank-calibrated reading
of a prediction (feature added 2026-05-21).

The deployed scorer's OOS calibration verdict is DIRECTIONAL_BUT_BIASED: the
predicted % magnitude is unreliable but the ordering carries real skill. These
tests pin that predict_percentile() exposes a trustworthy rank position and
that the feature is strictly backward-compatible with legacy pickles that
carry no quantile table.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

import paper_trader.ml.decision_scorer as ds
from paper_trader.ml.decision_scorer import DecisionScorer, train_scorer


def _training_records(n: int = 240) -> list[dict]:
    """Synthetic outcomes where forward_return_5d is a clean monotone
    function of ml_score — so a trained model MUST rank high ml_score above
    low ml_score, making percentile assertions deterministic."""
    recs = []
    for i in range(n):
        score = (i % 24) - 12          # -12..+11
        # Distinct (ticker, sim_date) per record so dedup keeps them all.
        day = 1 + (i % 27)
        month = 1 + (i // 27) % 12
        recs.append({
            "ticker": "NVDA",
            "sim_date": f"2024-{month:02d}-{day:02d}",
            "action": "BUY",
            "ml_score": float(score),
            "rsi": 50.0,
            "macd": 0.0,
            "mom5": 0.0,
            "mom20": 0.0,
            "regime_mult": 1.0,
            "forward_return_5d": float(score) * 1.5,
            "return_pct": 10.0,
        })
    return recs


@pytest.fixture
def trained_scorer(monkeypatch):
    """Train a scorer to SCORER_PATH (conftest redirects it to tmp) and
    return a freshly-loaded DecisionScorer instance."""
    ds._LOAD_CACHE.clear()
    result = train_scorer(_training_records())
    assert result["status"] == "ok", result
    ds._LOAD_CACHE.clear()
    return DecisionScorer(), result


class TestQuantileTablePersisted:
    def test_train_scorer_persists_101_quantiles(self, trained_scorer):
        _, result = trained_scorer
        assert result["n_pred_quantiles"] == 101

    def test_pickle_contains_pred_quantiles_key(self, trained_scorer):
        with ds.SCORER_PATH.open("rb") as f:
            state = pickle.load(f)
        assert "pred_quantiles" in state
        q = state["pred_quantiles"]
        assert q is not None and len(q) == 101

    def test_quantiles_are_non_decreasing(self, trained_scorer):
        scorer, _ = trained_scorer
        q = scorer._pred_quantiles
        assert q is not None
        # A percentile table is sorted ascending by construction.
        assert np.all(np.diff(q) >= -1e-9)


class TestPredictPercentile:
    def test_high_ml_score_ranks_above_low(self, trained_scorer):
        scorer, _ = trained_scorer
        common = dict(rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
                      regime_mult=1.0, ticker="NVDA")
        low = scorer.predict_percentile(ml_score=-10.0, **common)
        high = scorer.predict_percentile(ml_score=10.0, **common)
        assert low is not None and high is not None
        # The model learned fr5d ∝ ml_score, so the bullish input must land
        # at a strictly higher rank than the bearish one.
        assert high > low
        # And meaningfully so — not coin-flip noise.
        assert high - low > 20.0

    def test_percentile_in_unit_range(self, trained_scorer):
        scorer, _ = trained_scorer
        for s in (-50.0, -10.0, 0.0, 10.0, 50.0, 999.0):
            pct = scorer.predict_percentile(
                ml_score=s, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
                regime_mult=1.0, ticker="NVDA")
            assert pct is not None
            assert 0.0 <= pct <= 100.0

    def test_extreme_bullish_input_is_top_ranked(self, trained_scorer):
        scorer, _ = trained_scorer
        # An ml_score far above anything in training extrapolates upward;
        # its raw prediction sits at/after the training max → percentile 100.
        pct = scorer.predict_percentile(
            ml_score=500.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        assert pct == 100.0

    def test_predict_with_meta_includes_percentile(self, trained_scorer):
        scorer, _ = trained_scorer
        meta = scorer.predict_with_meta(
            ml_score=5.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        assert "percentile" in meta
        assert meta["percentile"] is not None


class TestBackwardCompatibility:
    def test_legacy_pickle_without_quantiles_loads(self, trained_scorer):
        """A pickle written before this feature (no pred_quantiles key) must
        load cleanly and predict() must still work — predict_percentile then
        degrades to None rather than crashing."""
        # Rewrite the pickle in the legacy 3-key format.
        with ds.SCORER_PATH.open("rb") as f:
            state = pickle.load(f)
        legacy = {"model": state["model"], "scaler": state["scaler"],
                  "n_train": state["n_train"]}
        with ds.SCORER_PATH.open("wb") as f:
            pickle.dump(legacy, f)
        ds._LOAD_CACHE.clear()

        scorer = DecisionScorer()
        assert scorer.is_trained
        assert scorer._pred_quantiles is None
        # predict() unaffected — still a finite float.
        pred = scorer.predict(ml_score=5.0, rsi=50.0, macd=0.0, mom5=0.0,
                              mom20=0.0, regime_mult=1.0, ticker="NVDA")
        assert isinstance(pred, float)
        # predict_percentile degrades to None, never raises.
        pct = scorer.predict_percentile(
            ml_score=5.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        assert pct is None
        # And the meta dict still carries the key (value None).
        meta = scorer.predict_with_meta(
            ml_score=5.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        assert meta["percentile"] is None

    def test_untrained_scorer_percentile_is_none(self, monkeypatch):
        """No pickle on disk → untrained → predict_percentile None, no crash."""
        ds._LOAD_CACHE.clear()
        if ds.SCORER_PATH.exists():
            ds.SCORER_PATH.unlink()
        scorer = DecisionScorer()
        assert not scorer.is_trained
        pct = scorer.predict_percentile(
            ml_score=5.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        assert pct is None

    def test_raw_to_percentile_rejects_non_finite(self, trained_scorer):
        scorer, _ = trained_scorer
        assert scorer._raw_to_percentile(float("nan")) is None
        assert scorer._raw_to_percentile(float("inf")) is None


class TestCollapsedQuantileGuard:
    """The 2026-05-23 finding #1 footprint: a synthetic n=39 retrain clobbered
    `data/ml/decision_scorer.pkl` with 101 IDENTICAL `pred_quantiles` (every
    entry was 18.934). Without an internal guard, `np.interp` on a constant
    `xp` clamps to fp[0] (0.0) or fp[-1] (100.0) for every real prediction —
    so `predict_percentile()` and `predict_calibrated()` silently surface
    fabricated extreme ranks while looking healthy to every existing test.
    These tests pin the honest degrade-to-None behaviour."""

    def test_raw_to_percentile_none_when_pred_quantiles_collapsed(self):
        """A pred_quantiles table with every entry equal must return None —
        no real rank information is available, mirror legacy-pickle behaviour."""
        s = DecisionScorer()
        s._trained = True
        s._n_train = 100
        # 101-entry collapsed table — exactly the synthetic-clobber footprint.
        s._pred_quantiles = np.asarray([18.934] * 101, dtype=np.float64)
        s._label_quantiles = np.asarray(np.linspace(-5.0, 5.0, 101),
                                        dtype=np.float64)
        # Any raw value (below, at, above the collapsed point) — all None.
        for raw in (-50.0, 0.0, 18.934, 50.0):
            assert s._raw_to_percentile(raw) is None, raw

    def test_raw_to_calibrated_none_when_pred_quantiles_collapsed(self):
        """Calibrated reading derives from `_raw_to_percentile`; the same
        collapsed table must cascade to a None calibrated magnitude
        (not the lq[0]/lq[-1] fabricated extreme that np.interp would
        otherwise produce when its xp clamps to 0/100)."""
        s = DecisionScorer()
        s._trained = True
        s._n_train = 100
        s._pred_quantiles = np.asarray([5.0] * 101, dtype=np.float64)
        s._label_quantiles = np.asarray(np.linspace(-10.0, 10.0, 101),
                                        dtype=np.float64)
        for raw in (-50.0, 0.0, 5.0, 50.0):
            assert s._raw_to_calibrated(raw) is None, raw

    def test_predict_with_meta_calibrated_none_on_collapsed_pred_quantiles(
            self, trained_scorer):
        """End-to-end: a real trained scorer whose pred_quantiles got
        replaced by the collapsed footprint must surface None in the
        consumer-visible `percentile` / `calibrated` fields."""
        scorer, _ = trained_scorer
        scorer._pred_quantiles = np.asarray([7.5] * 101, dtype=np.float64)
        meta = scorer.predict_with_meta(
            ml_score=5.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        assert meta["percentile"] is None
        assert meta["calibrated"] is None
        # predict()'s scalar contract is UNCHANGED — a collapsed quantile
        # table is a diagnostic artifact, not a model failure, so the gate
        # (which reads `pred` not `percentile`) is unaffected.
        assert meta["failed"] is False
        assert isinstance(meta["pred"], float)

    def test_healthy_quantiles_still_work(self, trained_scorer):
        """Sanity counterfactual: an honest non-collapsed table from a real
        train_scorer run continues to produce finite percentiles (proves the
        guard fires ONLY on the degenerate case, not on every healthy run)."""
        scorer, _ = trained_scorer
        assert scorer._pred_quantiles is not None
        q = np.asarray(scorer._pred_quantiles, dtype=np.float64)
        assert float(q.max()) > float(q.min())  # not collapsed
        pct = scorer._raw_to_percentile(float(q[50]))  # median raw
        assert pct is not None and 0.0 <= pct <= 100.0
