"""Tests for the ``gate_arm`` / ``gate_arm_multiplier`` fields added to
``DecisionScorer.predict_with_meta`` (2026-05-28 feature).

The conviction-gate arm (×0.6 / ×0.85 / ×1.0 / ×1.15 / ×1.3) was previously
decodable only via two paths: (a) reading the live ``_ml_decide``
if/elif chain by hand, or (b) importing ``gate_audit.gate_arm`` and calling
it with the prediction. Surfacing the arm directly on ``predict_with_meta``
gives a researcher inspecting one hypothetical prediction the answer in
the same atomic snapshot as the rest of the trust flags (clamp /
off-distribution / percentile / calibrated).

These tests pin:
  * the new fields' presence in the dict (drift guard)
  * the threshold boundaries match the live gate to the bit
  * untrained / failed paths report ``None`` honestly
  * the field decoding still works on legacy pickles without the
    `pred_quantiles` / `label_quantiles` tables
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

import paper_trader.ml.decision_scorer as ds
from paper_trader.ml.decision_scorer import DecisionScorer, train_scorer
from paper_trader.ml.gate_audit import gate_arm as _canonical_gate_arm


def _training_records(n: int = 240) -> list[dict]:
    """Synthetic outcomes where ``forward_return_5d ∝ ml_score`` so the
    trained model produces predictably ordered predictions."""
    recs = []
    for i in range(n):
        score = (i % 24) - 12          # -12..+11
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
            "forward_return_5d": float(score) * 2.0,
            "return_pct": 10.0,
        })
    return recs


@pytest.fixture
def trained_scorer(monkeypatch):
    """Train and return a fresh scorer at the test-isolated SCORER_PATH."""
    ds._LOAD_CACHE.clear()
    result = train_scorer(_training_records())
    assert result["status"] == "ok", result
    ds._LOAD_CACHE.clear()
    return DecisionScorer()


# ─────────────────────────── presence / shape ────────────────────────────

class TestGateArmFieldPresence:
    def test_meta_dict_contains_both_fields(self, trained_scorer):
        meta = trained_scorer.predict_with_meta(
            ml_score=5.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        # Drift guard — the dict-key set is part of predict_with_meta's
        # public contract (the docstring enumerates it).
        assert "gate_arm" in meta
        assert "gate_arm_multiplier" in meta

    def test_arm_name_is_canonical(self, trained_scorer):
        """Arm names must come from the canonical 5-arm set, never
        an ad-hoc string. The gate_audit module's _ARM_ORDER is the
        single source of truth."""
        from paper_trader.ml.gate_audit import _ARM_ORDER
        meta = trained_scorer.predict_with_meta(
            ml_score=5.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        assert meta["gate_arm"] in _ARM_ORDER

    def test_multiplier_in_canonical_set(self, trained_scorer):
        """Multiplier must be one of the 5 canonical values, never an
        interpolation or default."""
        meta = trained_scorer.predict_with_meta(
            ml_score=5.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        assert meta["gate_arm_multiplier"] in {0.60, 0.85, 1.00, 1.15, 1.30}


# ─────────────────────────── canonical-decode parity ─────────────────────

class TestGateArmMatchesCanonical:
    """The arm decode MUST match gate_audit.gate_arm(clamped_pred) exactly —
    same single source of truth. A drift between them would mean a
    researcher reading ``meta["gate_arm"]`` sees a different bucket than
    the live ``_ml_decide`` gate uses, defeating the whole point of
    surfacing the arm here."""

    def test_arm_matches_canonical_for_real_prediction(self, trained_scorer):
        meta = trained_scorer.predict_with_meta(
            ml_score=5.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        canonical_arm, canonical_mult = _canonical_gate_arm(meta["pred"])
        assert meta["gate_arm"] == canonical_arm
        assert meta["gate_arm_multiplier"] == canonical_mult

    def test_arm_decoded_from_clamped_not_raw(self, monkeypatch,
                                              trained_scorer):
        """The arm must come from the CLAMPED ``pred`` (matches the live
        gate which sees ``predict()``'s clamped output). If decoded from
        ``raw``, an off-distribution extrapolation like +89% would decode
        the same as +50 (both 'strong_tailwind') — but a future tighter
        clamp could break that, so pinning the source explicitly is
        defense-in-depth."""
        # Force the model to predict a value beyond the clamp range so
        # raw and pred differ.
        class _Stub:
            def predict(self, X):
                return np.array([200.0], dtype=np.float64)
        trained_scorer._model = _Stub()
        meta = trained_scorer.predict_with_meta(
            ml_score=0.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        assert meta["raw"] == 200.0
        assert meta["pred"] == 50.0  # PRED_CLAMP_PCT
        # Arm must decode from the clamped value (50.0 > 10 → strong_tailwind).
        assert meta["gate_arm"] == "strong_tailwind"
        assert meta["gate_arm_multiplier"] == 1.30


# ─────────────────────────── threshold boundaries ────────────────────────

class TestGateArmThresholdBoundaries:
    """Pin the exact boundary semantics — these match the live
    ``_ml_decide`` if/elif chain (CLAUDE.md §6 / AGENTS.md §Phase 1).
    A boundary change here without a matching live-gate change would
    silently mislead operators using ``--explain``."""

    @pytest.fixture
    def stub_scorer(self, trained_scorer):
        """Yield a function that returns the gate_arm meta for a stubbed
        prediction value, so we can exercise every threshold without
        crafting feature vectors that happen to land exactly on each."""
        def _arm_for(pred_value: float) -> tuple[str, float]:
            class _Stub:
                def predict(self, X):
                    return np.array([pred_value], dtype=np.float64)
            trained_scorer._model = _Stub()
            meta = trained_scorer.predict_with_meta(
                ml_score=0.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
                regime_mult=1.0, ticker="NVDA")
            return meta["gate_arm"], meta["gate_arm_multiplier"]
        return _arm_for

    @pytest.mark.parametrize("pred,expected_arm,expected_mult", [
        (-50.0, "strong_headwind", 0.60),
        (-15.0, "strong_headwind", 0.60),
        (-10.1, "strong_headwind", 0.60),
        # boundary: pred < -10 is strong; pred == -10 is mild
        (-10.0, "mild_headwind", 0.85),
        (-5.0, "mild_headwind", 0.85),
        (-0.001, "mild_headwind", 0.85),
        # boundary: pred < 0 is mild_headwind; pred == 0 is neutral
        (0.0, "neutral", 1.00),
        (3.0, "neutral", 1.00),
        (5.0, "neutral", 1.00),
        # boundary: pred > 5 is mild_tailwind; pred == 5 stays neutral
        (5.001, "mild_tailwind", 1.15),
        (10.0, "mild_tailwind", 1.15),
        # boundary: pred > 10 is strong_tailwind
        (10.001, "strong_tailwind", 1.30),
        (25.0, "strong_tailwind", 1.30),
        (50.0, "strong_tailwind", 1.30),
    ])
    def test_threshold_arm_assignment(self, stub_scorer, pred, expected_arm,
                                       expected_mult):
        arm, mult = stub_scorer(pred)
        assert arm == expected_arm, f"pred={pred}: got {arm!r}"
        assert mult == expected_mult, f"pred={pred}: got mult={mult}"


# ─────────────────────────── failed / untrained paths ────────────────────

class TestGateArmFailedPaths:
    """``failed=True`` paths must report ``gate_arm=None`` — the prediction
    couldn't be produced, so the arm is undefined. The OOS rank-IC consumer
    discipline (drop ``failed=True`` rows) extends naturally: a researcher
    bucketing predictions by arm should likewise drop rows where the arm
    is None rather than falsely attributing a 'neutral' arm to a broken
    predict path."""

    def test_untrained_scorer_returns_none_arm(self, monkeypatch, tmp_path):
        """Brand-new scorer with no on-disk pickle: predict_with_meta
        returns failed=True, gate_arm=None."""
        # Redirect SCORER_PATH to a non-existent file
        ghost = tmp_path / "ghost.pkl"
        monkeypatch.setattr(ds, "SCORER_PATH", ghost)
        ds._LOAD_CACHE.clear()
        s = DecisionScorer()
        assert not s.is_trained
        meta = s.predict_with_meta(
            ml_score=5.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        assert meta["failed"] is True
        assert meta["gate_arm"] is None
        assert meta["gate_arm_multiplier"] is None

    def test_predict_exception_returns_none_arm(self, trained_scorer):
        """Model.predict raises → failed=True path → arm None."""
        class _Broken:
            def predict(self, X):
                raise RuntimeError("simulated shape mismatch")
        trained_scorer._model = _Broken()
        meta = trained_scorer.predict_with_meta(
            ml_score=5.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        assert meta["failed"] is True
        assert meta["gate_arm"] is None
        assert meta["gate_arm_multiplier"] is None

    def test_non_finite_raw_returns_none_arm(self, trained_scorer):
        """Model returns NaN/Inf → non-finite raw → failed=True → arm None.
        A 'neutral' arm here would silently make the broken predict path
        look like a legitimate gate decision."""
        class _NaN:
            def predict(self, X):
                return np.array([float("nan")], dtype=np.float64)
        trained_scorer._model = _NaN()
        meta = trained_scorer.predict_with_meta(
            ml_score=5.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        assert meta["failed"] is True
        assert meta["gate_arm"] is None
        assert meta["gate_arm_multiplier"] is None


# ─────────────────────────── off-distribution semantics ──────────────────

class TestGateArmOffDistribution:
    """An off-distribution prediction (raw > PRED_CLAMP_PCT) is still a real
    prediction — failed=False — so the arm IS reported. The arm reflects
    what the gate WOULD fire absent the off-dist abstention guard in
    ``_ml_decide`` (the live gate sees ``off_distribution=True`` and
    abstains, but the arm decode itself is well-defined).

    This is the design choice the docstring locks: ``gate_arm`` is the
    'what arm fits this clamped pred', NOT 'what arm will the gate
    actually fire'. ``predict_with_meta`` already exposes
    ``off_distribution`` so the caller can combine the two to answer the
    latter question."""

    def test_clamped_extrapolation_still_has_arm(self, trained_scorer):
        class _Extrapolator:
            def predict(self, X):
                return np.array([200.0], dtype=np.float64)
        trained_scorer._model = _Extrapolator()
        meta = trained_scorer.predict_with_meta(
            ml_score=0.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        assert meta["off_distribution"] is True
        assert meta["failed"] is False
        # Arm IS reported — the gate's off-dist abstention is separate
        # from arm decoding.
        assert meta["gate_arm"] == "strong_tailwind"
        assert meta["gate_arm_multiplier"] == 1.30


# ─────────────────────────── legacy pickle compatibility ─────────────────

class TestGateArmBackwardCompatibility:
    """Existing legacy pickles (no pred_quantiles / label_quantiles) must
    still produce a gate_arm — the arm decode depends only on the
    prediction value, not on the quantile tables. A regression here would
    silently break gate arm reporting for older deployments mid-rollout."""

    def test_legacy_pickle_still_decodes_arm(self, trained_scorer):
        # Rewrite to legacy 3-key format (no pred_quantiles, no
        # label_quantiles).
        with ds.SCORER_PATH.open("rb") as f:
            state = pickle.load(f)
        legacy = {"model": state["model"], "scaler": state["scaler"],
                  "n_train": state["n_train"]}
        with ds.SCORER_PATH.open("wb") as f:
            pickle.dump(legacy, f)
        ds._LOAD_CACHE.clear()
        s = DecisionScorer()
        assert s.is_trained
        # Quantile tables are None, but arm decode still works.
        assert s._pred_quantiles is None
        assert s._label_quantiles is None
        meta = s.predict_with_meta(
            ml_score=5.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA")
        # Sibling fields legitimately None on legacy pickles
        assert meta["percentile"] is None
        assert meta["calibrated"] is None
        # But gate_arm is decoded from pred alone, no quantile dependency.
        assert meta["gate_arm"] is not None
        assert meta["gate_arm_multiplier"] is not None
        assert meta["gate_arm_multiplier"] in {0.60, 0.85, 1.00, 1.15, 1.30}
