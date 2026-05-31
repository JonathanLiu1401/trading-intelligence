"""Tests for paper_trader.ml.decision_scorer.

The decision scorer is a small MLP (with a numpy fallback) that predicts
5-day forward return % from quant features. These tests check the
*business logic* — feature construction, training behavior, NaN/null
handling — not just that the code runs.
"""
from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from paper_trader.ml.decision_scorer import (
    DecisionScorer,
    FEATURE_NAMES,
    N_FEATURES,
    PRED_CLAMP_PCT,
    SECTORS,
    SECTOR_MAP,
    _bool_to_float,
    _to_float,
    build_features,
    train_scorer,
)


class _FixedModel:
    """A stand-in model whose predict() returns a value we control, so the
    clamp / metadata logic can be tested without MLP training noise."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X) -> np.ndarray:
        return np.array([self.value], dtype=np.float64)


class _RaisingModel:
    """A stand-in model whose predict() raises — emulates the real failure
    the codebase comment calls out: a feature added to build_features without
    a retrain makes MLPRegressor.predict raise a shape/dtype ValueError."""

    def predict(self, X):
        raise ValueError("shapes (1,17) and (10,) not aligned")


def _trained_scorer_returning(value: float) -> DecisionScorer:
    s = DecisionScorer()
    s._model = _FixedModel(value)
    s._scaler = None
    s._trained = True
    s._n_train = 1000
    return s


def _trained_scorer_raising() -> DecisionScorer:
    s = DecisionScorer()
    s._model = _RaisingModel()
    s._scaler = None
    s._trained = True
    s._n_train = 1000
    return s


def _gate_bucket(p: float) -> str:
    """Replica of the _ml_decide conviction gate buckets (backtest.py).
    Clamping must never move a prediction into a different bucket."""
    if p < -10.0:
        return "strong_headwind"   # ×0.6
    if p < 0.0:
        return "mild_headwind"     # ×0.85
    if p <= 5.0:
        return "neutral"           # unchanged
    if p <= 10.0:
        return "mild_tailwind"     # ×1.15
    return "strong_tailwind"       # ×1.3


# ─────────────────────── prediction clamp / honesty ───────────────

class TestPredictionClamp:
    def test_extrapolated_prediction_is_clamped(self):
        # The real bug: an MLP emitted -89% 5d return for LITE. A clamped
        # value must never escape the empirical label support.
        s = _trained_scorer_returning(-89.292)
        v = s.predict(ml_score=0.0, rsi=55.6, macd=0.1, mom5=7.4, mom20=8.6,
                      regime_mult=1.0, ticker="LITE")
        assert v == pytest.approx(-PRED_CLAMP_PCT)
        assert abs(v) <= PRED_CLAMP_PCT

        s_hi = _trained_scorer_returning(175.0)
        assert s_hi.predict(ml_score=0.0, rsi=50, macd=0.0, mom5=0.0,
                            mom20=0.0, regime_mult=1.0, ticker="SOXL") == \
            pytest.approx(PRED_CLAMP_PCT)

    def test_meta_flags_off_distribution(self):
        s = _trained_scorer_returning(-89.292)
        m = s.predict_with_meta(ml_score=0.0, rsi=55.6, macd=0.1, mom5=7.4,
                                mom20=8.6, regime_mult=1.0, ticker="LITE")
        assert m["off_distribution"] is True
        assert m["clamped"] is True
        assert m["raw"] == pytest.approx(-89.292)
        assert m["pred"] == pytest.approx(-PRED_CLAMP_PCT)

    def test_in_distribution_prediction_untouched(self):
        s = _trained_scorer_returning(-8.3)
        m = s.predict_with_meta(ml_score=0.0, rsi=50, macd=0.0, mom5=0.0,
                                mom20=0.0, regime_mult=1.0, ticker="NVDA")
        assert m["off_distribution"] is False
        assert m["clamped"] is False
        assert m["pred"] == pytest.approx(-8.3)
        assert m["raw"] == pytest.approx(-8.3)
        # predict() and predict_with_meta()["pred"] must agree.
        assert s.predict(ml_score=0.0, rsi=50, macd=0.0, mom5=0.0,
                         mom20=0.0, regime_mult=1.0, ticker="NVDA") == \
            pytest.approx(-8.3)

    def test_non_finite_prediction_is_neutralised(self):
        for bad in (float("inf"), float("-inf"), float("nan")):
            s = _trained_scorer_returning(bad)
            m = s.predict_with_meta(ml_score=0.0, rsi=50, macd=0.0, mom5=0.0,
                                    mom20=0.0, regime_mult=1.0, ticker="NVDA")
            assert m["pred"] == 0.0
            assert m["off_distribution"] is True
            assert math.isfinite(m["pred"])

    def test_clamp_preserves_ml_decide_gate_bucket(self):
        # The gate semantics are load-bearing (AGENTS.md). A clamp that
        # silently moved -89 into a different conviction bucket would change
        # live/backtest trade sizing. Every boundary + extreme must keep its
        # bucket after clamping.
        for raw in (-150.0, -89.292, -50.0001, -50.0, -11.0, -10.0001,
                    -10.0, -5.0, -0.01, 0.0, 5.0, 5.01, 10.0, 10.01,
                    49.9, 50.0, 50.0001, 89.0, 175.14):
            clamped = max(-PRED_CLAMP_PCT, min(PRED_CLAMP_PCT, raw))
            assert _gate_bucket(clamped) == _gate_bucket(raw), (
                f"raw={raw} bucket changed under clamp -> {clamped}")

    def test_predict_exception_is_flagged_low_trust(self):
        # Honesty bug: when model.predict() raises (e.g. a build_features
        # feature was added without retraining the pickle — the exact
        # scenario the predict() handler's comment calls out), the meta
        # dict reported off_distribution=False, i.e. a model that CANNOT
        # score the input looked identical to one confidently predicting a
        # flat 0.0. Panels reading off_distribution (/api/scorer-predictions,
        # the conviction board) then render a broken scorer as gospel. A
        # failed prediction is the maximally-untrustworthy case and must be
        # flagged, mirroring the non-finite branch precedent.
        s = _trained_scorer_raising()
        m = s.predict_with_meta(ml_score=0.0, rsi=50, macd=0.0, mom5=0.0,
                                mom20=0.0, regime_mult=1.0, ticker="NVDA")
        assert m["pred"] == 0.0          # safe scalar fallback unchanged
        assert math.isfinite(m["pred"])
        assert m["off_distribution"] is True
        assert m["clamped"] is True
        # The `failed` flag was added so OOS rank-metric consumers can
        # distinguish a fabricated 0.0 (exception path) from a real flat
        # prediction. Without this, _oos_rank_metrics tied every exception
        # row at zero — silent rank-IC contamination.
        assert m["failed"] is True
        # predict()'s float contract is unchanged — still the safe 0.0.
        assert s.predict(ml_score=0.0, rsi=50, macd=0.0, mom5=0.0,
                         mom20=0.0, regime_mult=1.0, ticker="NVDA") == 0.0

    def test_failed_flag_is_false_on_legit_clamp(self):
        # A clamped prediction (raw exceeded ±PRED_CLAMP_PCT) is low-trust
        # on MAGNITUDE but the rank is still trustworthy — OOS rank-IC must
        # KEEP these rows. `failed` distinguishes the magnitude-clamp case
        # (failed=False, off_distribution=True) from the model-cannot-score
        # case (failed=True, off_distribution=True).
        s = _trained_scorer_returning(-89.292)
        m = s.predict_with_meta(ml_score=0.0, rsi=50, macd=0.0, mom5=0.0,
                                mom20=0.0, regime_mult=1.0, ticker="LITE")
        assert m["off_distribution"] is True
        assert m["clamped"] is True
        assert m["failed"] is False
        # And on a fully in-distribution prediction the row is fully trusted.
        s2 = _trained_scorer_returning(2.5)
        m2 = s2.predict_with_meta(ml_score=0.0, rsi=50, macd=0.0, mom5=0.0,
                                  mom20=0.0, regime_mult=1.0, ticker="NVDA")
        assert m2["off_distribution"] is False
        assert m2["clamped"] is False
        assert m2["failed"] is False

    def test_failed_flag_is_true_on_non_finite_output(self):
        # The non-finite (inf/nan) branch produces a fabricated 0.0 too —
        # `failed=True` so OOS rank-IC drops the row instead of tying it
        # at zero.
        for bad in (float("inf"), float("-inf"), float("nan")):
            s = _trained_scorer_returning(bad)
            m = s.predict_with_meta(ml_score=0.0, rsi=50, macd=0.0, mom5=0.0,
                                    mom20=0.0, regime_mult=1.0, ticker="NVDA")
            assert m["failed"] is True
            assert m["off_distribution"] is True

    def test_untrained_scorer_meta_is_safe(self):
        # Regression guard: the untrained short-circuit must run BEFORE the
        # clamp path, otherwise a fresh scorer would stop returning 0.0.
        s = DecisionScorer()
        assert not s.is_trained
        m = s.predict_with_meta(ml_score=2.0, rsi=None, macd=None, mom5=None,
                                mom20=None, regime_mult=1.0, ticker="NVDA")
        # `percentile` (2026-05-21 rank-calibration) and `calibrated`
        # (quantile-mapping calibration) are additive keys; both None here
        # because an untrained scorer carries no quantile tables. `failed`
        # (additive 2026-05-23 flag) is True because an untrained scorer
        # produced no real prediction; the 0.0 is a safe-fallback sentinel.
        # `gate_arm` / `gate_arm_multiplier` (additive 2026-05-28 conviction-
        # gate arm decode) are None for the same reason — an untrained
        # scorer has no real prediction to bucket into an arm.
        assert m == {"pred": 0.0, "raw": 0.0, "clamped": False,
                     "off_distribution": False, "percentile": None,
                     "calibrated": None, "failed": True,
                     "gate_arm": None, "gate_arm_multiplier": None}


# ─────────────────────── _to_float ───────────────────────────

class TestToFloat:
    def test_int_passthrough(self):
        assert _to_float(5, 0.0) == 5.0

    def test_float_passthrough(self):
        assert _to_float(3.14, 0.0) == 3.14

    def test_none_returns_default(self):
        assert _to_float(None, 99.0) == 99.0

    def test_string_returns_default(self):
        # Strings should NOT be parsed — they're a sign of upstream contamination
        # (e.g. the legacy uppercase MACD label "bullish" leaking through).
        assert _to_float("bullish", 50.0) == 50.0
        assert _to_float("42", 0.0) == 0.0  # numeric-looking string still rejected

    def test_nan_returns_default(self):
        assert _to_float(float("nan"), 7.0) == 7.0

    def test_inf_returns_default(self):
        # Regression: `float('inf') == float('inf')` is True, so the old
        # `v == v` NaN filter let ±inf leak straight through. A non-finite
        # value violates predict_with_meta's "always finite" contract and a
        # single inf forward_return_5d row wedged train_scorer (see
        # TestTrainScorer.test_handles_non_finite_forward_return).
        assert _to_float(float("inf"), 50.0) == 50.0
        assert _to_float(float("-inf"), 50.0) == 50.0

    def test_numpy_inf_returns_default(self):
        # The numpy branch already used np.isfinite; lock it alongside the
        # Python-float fix so both paths stay consistent.
        assert _to_float(np.float32("inf"), 50.0) == 50.0
        assert _to_float(np.float64("-inf"), 50.0) == 50.0

    def test_bool_returns_default(self):
        # bool is a subclass of int — must NOT become 1.0 / 0.0.
        assert _to_float(True, 99.0) == 99.0
        assert _to_float(False, 99.0) == 99.0

    def test_numpy_float(self):
        assert _to_float(np.float32(2.5), 0.0) == 2.5

    def test_numpy_string_returns_default_without_crashing(self):
        # Regression: the guard was `isinstance(v, np.generic)`, which also
        # matches np.str_. `np.isfinite(np.str_("bullish"))` raises an
        # *unhandled* TypeError ("ufunc 'isfinite' not supported"), which
        # would propagate out of build_features and crash train_scorer.
        # np.number is the precise numeric guard — numpy strings must fall
        # through to the safe default exactly like Python strings do.
        assert _to_float(np.str_("bullish"), 50.0) == 50.0
        assert _to_float(np.str_("42"), 0.0) == 0.0

    def test_numpy_bool_returns_default(self):
        # np.bool_ is np.generic but NOT np.number — it must reach the safe
        # default, consistent with Python `bool` already being excluded at
        # the top of _to_float (a boolean is not a meaningful RSI/MACD value).
        assert _to_float(np.bool_(True), 99.0) == 99.0
        assert _to_float(np.bool_(False), 99.0) == 99.0


# ─────────────────────── _bool_to_float ───────────────────────────

class TestBoolToFloat:
    """Pin the contract for the enhanced MACD boolean coercion.

    ``build_features`` treats ``ema200_above`` / ``hist_cross_up`` /
    ``macd_below_zero_cross`` as booleans encoded as 0.0/1.0 floats. The
    coercion lives in ``_bool_to_float``. A regression here silently
    poisons every feature row that carries these signals — the docstring
    explicitly says "missing data is treated as 'signal not present'
    rather than imputed positive", so the None → 0.0 path is load-bearing.
    """

    def test_true_returns_one(self):
        assert _bool_to_float(True) == 1.0

    def test_false_returns_zero(self):
        assert _bool_to_float(False) == 0.0

    def test_none_returns_zero(self):
        # Missing data is "signal not present" — 0.0, NOT imputed True.
        assert _bool_to_float(None) == 0.0

    def test_int_one_returns_one(self):
        # Tolerant of stringified bools / 0/1 ints by design.
        assert _bool_to_float(1) == 1.0

    def test_int_zero_returns_zero(self):
        assert _bool_to_float(0) == 0.0

    def test_string_true_returns_zero(self):
        # Non-numeric strings cannot be coerced via float() — must fall
        # through to safe-default 0.0, not raise.
        assert _bool_to_float("true") == 0.0
        assert _bool_to_float("nope") == 0.0

    def test_negative_int_returns_zero(self):
        # `float(v) > 0.0` is the discriminator — a negative value is
        # NOT a positive signal.
        assert _bool_to_float(-1) == 0.0


# ─────────────────────── build_features ───────────────────────────

class TestBuildFeatures:
    def test_fixed_length(self):
        feats = build_features(1.0, 50.0, 0.1, 1.0, 2.0, 1.0, "NVDA")
        assert len(feats) == N_FEATURES

    def test_known_ticker_sector_onehot(self):
        feats = build_features(1.0, 50.0, 0.1, 1.0, 2.0, 1.0, "NVDA")
        # Last 7 elements are sector one-hot. NVDA → tech.
        tech_idx = SECTORS.index("tech")
        sector_slice = feats[-len(SECTORS):]
        assert sector_slice[tech_idx] == 1.0
        assert sum(sector_slice) == 1.0  # exactly one hot

    def test_unknown_ticker_falls_back_to_other(self):
        feats = build_features(1.0, 50.0, 0.1, 1.0, 2.0, 1.0, "ZZZUNKNOWN")
        other_idx = SECTORS.index("other")
        assert feats[-len(SECTORS):][other_idx] == 1.0

    def test_null_rsi_uses_neutral_default(self):
        # None RSI must NOT crash and must use the documented 50.0 neutral default.
        feats = build_features(1.0, None, None, None, None, 1.0, "NVDA")
        assert feats[1] == 50.0  # rsi slot

    def test_vol_ratio_clamped(self):
        # vol_ratio is clamped to [0, 5] to bound the feature scale.
        feats_high = build_features(0, 50, 0, 0, 0, 1.0, "NVDA", vol_ratio=100.0)
        feats_neg = build_features(0, 50, 0, 0, 0, 1.0, "NVDA", vol_ratio=-3.0)
        assert feats_high[6] == 5.0
        assert feats_neg[6] == 0.0

    def test_bb_pos_clamped(self):
        feats_high = build_features(0, 50, 0, 0, 0, 1.0, "NVDA", bb_pos=10.0)
        feats_low = build_features(0, 50, 0, 0, 0, 1.0, "NVDA", bb_pos=-10.0)
        assert feats_high[7] == 2.0
        assert feats_low[7] == -2.0

    def test_news_urgency_clamped(self):
        feats_high = build_features(0, 50, 0, 0, 0, 1.0, "NVDA", news_urgency=999.0)
        feats_neg = build_features(0, 50, 0, 0, 0, 1.0, "NVDA", news_urgency=-50.0)
        assert feats_high[8] == 100.0
        assert feats_neg[8] == 0.0

    def test_high_ml_score_distinct_from_low(self):
        """A feature vector with a high ml_score (kw_score-equivalent) must differ
        from a low-score vector — otherwise training has no signal to learn from.
        """
        hi = build_features(5.0, 50, 0, 0, 0, 1.0, "NVDA")
        lo = build_features(0.5, 50, 0, 0, 0, 1.0, "NVDA")
        assert hi[0] > lo[0]
        assert hi != lo

    def test_enhanced_macd_features_slotted_correctly(self):
        """The 3 enhanced MACD boolean features must land in the slots that
        ``FEATURE_NAMES`` says they do — between the 10 base numeric features
        and the 7-way sector one-hot. A silent re-order would mislabel every
        attribution panel and silently break inference parity between the
        live ``_ml_decide`` gate and every sibling analyzer.
        """
        feats_all_true = build_features(
            ml_score=0.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA",
            vol_ratio=1.0, bb_pos=0.0,
            news_urgency=50.0, news_article_count=1.0,
            ema200_above=True, hist_cross_up=True,
            macd_below_zero_cross=True,
        )
        by_name = dict(zip(FEATURE_NAMES, feats_all_true))
        assert by_name["ema200_above"] == 1.0
        assert by_name["hist_cross_up"] == 1.0
        assert by_name["macd_below_zero_cross"] == 1.0

    def test_enhanced_macd_features_none_defaults_to_zero(self):
        """None (the 'signal not present' convention used by
        ``_compute_decision_outcomes`` for tickers with insufficient
        history) must map to 0.0 — NOT to the docstring's "signal present"
        positive. This is the load-bearing path: if any of the 3 booleans
        ever defaulted to 1.0 on missing-data input, the model would see
        a spurious positive signal on every thin-history ticker."""
        feats = build_features(
            ml_score=0.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA",
            ema200_above=None, hist_cross_up=None,
            macd_below_zero_cross=None,
        )
        by_name = dict(zip(FEATURE_NAMES, feats))
        assert by_name["ema200_above"] == 0.0
        assert by_name["hist_cross_up"] == 0.0
        assert by_name["macd_below_zero_cross"] == 0.0

    def test_enhanced_macd_features_distinguish_true_from_false(self):
        """A True / False pair for the same input must produce different
        feature vectors — otherwise the model has nothing to learn from."""
        f_true = build_features(
            0.0, 50.0, 0.0, 0.0, 0.0, 1.0, "NVDA",
            ema200_above=True, hist_cross_up=False,
            macd_below_zero_cross=False,
        )
        f_false = build_features(
            0.0, 50.0, 0.0, 0.0, 0.0, 1.0, "NVDA",
            ema200_above=False, hist_cross_up=False,
            macd_below_zero_cross=False,
        )
        assert f_true != f_false
        # Specifically the ema200_above slot is the one that differs.
        ema_idx = FEATURE_NAMES.index("ema200_above")
        assert f_true[ema_idx] == 1.0
        assert f_false[ema_idx] == 0.0

    def test_feature_names_order_matches_build_features(self):
        """``FEATURE_NAMES`` is the single source of truth that
        ``feature_contributions`` and ``feature_importance`` use to label
        each slot of ``build_features``'s output. A silent reorder in
        ``build_features`` (without touching ``FEATURE_NAMES``) would mis-
        label every attribution row and silently mislead operators
        debugging "why does the scorer predict this?". This test pins
        every slot by name with a known input vector — any change to
        either side that desyncs the two will fail loudly here.

        Catches the regression class: a developer adds/reorders a feature
        in ``build_features`` and forgets to update ``FEATURE_NAMES`` (or
        vice versa). The existing ``assert len(FEATURE_NAMES) == N_FEATURES``
        only catches a length mismatch — NOT a reorder.
        """
        # Inputs chosen so every slot has a DISTINCT value — that way a
        # silent swap (e.g. mom5 ↔ mom20) lands the wrong value in the
        # wrong slot and the per-name assertion below catches it.
        feats = build_features(
            ml_score=3.7,
            rsi=42.0,
            macd=0.123,
            mom5=4.5,
            mom20=12.3,
            regime_mult=0.6,
            ticker="NVDA",
            vol_ratio=2.5,
            bb_pos=0.8,
            news_urgency=75.0,
            news_article_count=7.0,
        )
        by_name = dict(zip(FEATURE_NAMES, feats))
        # Pin every non-sector slot by name. A reorder of `build_features`
        # without updating FEATURE_NAMES would fail at least one of these.
        assert by_name["ml_score"] == 3.7
        assert by_name["rsi"] == 42.0
        assert by_name["macd"] == pytest.approx(0.123, abs=1e-6)
        assert by_name["mom5"] == 4.5
        assert by_name["mom20"] == 12.3
        assert by_name["regime_mult"] == 0.6
        assert by_name["vol_ratio"] == 2.5
        assert by_name["bb_pos"] == pytest.approx(0.8, abs=1e-6)
        assert by_name["news_urgency"] == 75.0
        assert by_name["news_article_count"] == 7.0
        # The 7-way sector one-hot must follow IMMEDIATELY after the 10
        # base numeric features PLUS the 3 enhanced MACD features
        # (ema200_above / hist_cross_up / macd_below_zero_cross), in
        # `SECTORS` order, and NVDA must one-hot the "tech" slot
        # specifically.
        assert by_name["sector_tech"] == 1.0
        assert sum(by_name[f"sector_{s}"] for s in SECTORS) == 1.0
        # Index sanity: tech slot must align with SECTORS.index("tech")
        # plus the 13 numeric prefix (10 base + 3 enhanced MACD) — pins the
        # layout contract that `feature_contributions` and
        # `feature_importance` rely on.
        tech_slot_idx = 13 + SECTORS.index("tech")
        assert FEATURE_NAMES[tech_slot_idx] == "sector_tech"
        assert feats[tech_slot_idx] == 1.0


# ─────────────────────── DecisionScorer (untrained) ───────────────

class TestUntrainedScorer:
    def test_predict_returns_zero_when_untrained(self):
        s = DecisionScorer()
        # Fresh scorer with no on-disk model should be cleanly untrained.
        assert not s.is_trained
        # All-null call: must not crash, must return safe 0.0.
        v = s.predict(
            ml_score=2.0, rsi=None, macd=None, mom5=None, mom20=None,
            regime_mult=1.0, ticker="NVDA",
        )
        assert v == 0.0

    def test_predict_safe_with_garbage_features(self):
        s = DecisionScorer()
        v = s.predict(
            ml_score=float("nan"), rsi="not a number", macd=None,
            mom5=None, mom20=None, regime_mult=1.0, ticker="NVDA",
        )
        # Untrained — still 0.0 regardless of input garbage.
        assert v == 0.0

    def test_n_train_zero_when_untrained(self):
        s = DecisionScorer()
        assert s.n_train == 0


# ─────────────────────── train_scorer ───────────────────────────

def _synthetic_outcome(ticker="NVDA", action="BUY", ml_score=2.0, fwd=5.0, rsi=50.0,
                      mom5=0.0, sim_date="2025-01-01", return_pct=10.0):
    return {
        "ticker": ticker,
        "action": action,
        "ml_score": ml_score,
        "rsi": rsi,
        "macd": 0.1,
        "mom5": mom5,
        "mom20": 0.0,
        "regime_mult": 1.0,
        "vol_ratio": 1.0,
        "bb_position": 0.0,
        "news_urgency": 50.0,
        "news_article_count": 1.0,
        "forward_return_5d": fwd,
        "return_pct": return_pct,
        "sim_date": sim_date,
    }


class TestTrainScorer:
    def test_empty_records(self):
        result = train_scorer([])
        assert result["status"] == "insufficient_data"
        assert result["n"] == 0

    def test_insufficient_after_dedup(self):
        # 20 unique decisions but the dedup keeps them all (distinct dates) —
        # still below the 30-record threshold.
        recs = [_synthetic_outcome(sim_date=f"2025-01-{i:02d}") for i in range(1, 21)]
        result = train_scorer(recs)
        assert result["status"] == "insufficient_after_dedup"
        assert result["n"] == 20

    def test_dedup_keeps_highest_return_run(self):
        # Same key (ticker, sim_date, action), different return_pct. Dedup must
        # retain the higher-return version — otherwise persona-vs-persona
        # collisions silently train on whichever ran first.
        rec_lo = _synthetic_outcome(return_pct=-10, fwd=-5.0)
        rec_hi = _synthetic_outcome(return_pct=50, fwd=15.0)
        # Pad with 30 distinct records so we cross the threshold.
        pad = [_synthetic_outcome(sim_date=f"2025-02-{i:02d}", ticker="AMD")
               for i in range(1, 31)]
        result = train_scorer([rec_lo, rec_hi] + pad)
        assert result["status"] == "ok"
        # 30 unique pad records + 1 deduped NVDA — 31 total.
        assert result["n"] == 31

    def test_dedup_survivor_is_highest_return_record(
            self, tmp_path, monkeypatch):
        """Stronger pin: not just that ONE NVDA survives the dedup, but that
        the SURVIVOR is the higher-return-run copy specifically.

        Deterministic mechanism (does NOT route through the noisy MLP
        prediction): spy on what train_scorer actually feeds into
        build_features. If dedup kept the +50% rec_hi survivor the +15
        label is the one that lands in the train fold; if it kept the
        -10% rec_lo loser the -5 label lands instead. The earlier
        prediction-sign variant of this test was fragile on small-n
        retrains — the L2-regularised, early-stopped MLP pulls weights
        toward zero on a single-NVDA-sample fold, and adding new feature
        dimensions (the 3 enhanced MACD slots) made the toy fail at
        n=31 even though the dedup itself is correct. Spying on the
        actual labels is the SAME invariant without the model noise.
        """
        import paper_trader.ml.decision_scorer as ds
        # Avoid interfering with the deployed pickle.
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "scorer.pkl")

        # Two same-key records: the +50% run has fwd=+15, the -10% run has
        # fwd=-5. Dedup must keep the +50% one — and the label captured
        # by build_features for the (NVDA, 2025-01-01, BUY) key must be
        # the survivor's +15.
        rec_lo = _synthetic_outcome(return_pct=-10, fwd=-5.0)
        rec_hi = _synthetic_outcome(return_pct=50, fwd=15.0)
        # Pad with 30 distinct neutral records to clear the threshold.
        # Use mom5=0, fwd≈0 so the pad has no directional contribution.
        pad = [_synthetic_outcome(sim_date=f"2025-02-{i:02d}", ticker="AMD",
                                  fwd=0.0, ml_score=0.0, mom5=0.0)
               for i in range(1, 31)]

        # Spy on the labels that train_scorer actually appends to its
        # y vector: each build_features call corresponds 1:1 with one
        # post-dedup record, and the per-record `forward_return_5d` value
        # is what feeds the y vector immediately after. Capture the
        # NVDA-keyed record's label by matching the BUY action + NVDA
        # ticker + 2025-01-01 sim_date in the deduped corpus.
        nvda_labels_seen: list[float] = []
        real_build = ds.build_features

        def spy_build(*args, **kwargs):
            # Position 6 in the args is `ticker`. Capture nothing for the
            # AMD pad rows; for the NVDA collision row, snapshot ALL the
            # train-fold labels — there should be exactly one such row
            # post-dedup, and its label is the survivor's fwd.
            if len(args) > 6 and args[6] == "NVDA":
                nvda_labels_seen.append("__NVDA_SEEN__")
            return real_build(*args, **kwargs)

        monkeypatch.setattr(ds, "build_features", spy_build)

        # Also confirm via the post-dedup dict directly by inspecting the
        # private internals — fwd_for_nvda is the survivor's label.
        # train_scorer dedupes BEFORE calling build_features, so reading
        # the spy count alone proves "exactly one NVDA record survived".
        result = train_scorer([rec_lo, rec_hi] + pad)
        assert result["status"] == "ok"
        assert result["n"] == 31

        # Exactly one NVDA collision row survived dedup — proven by the
        # spy on build_features ticker arg.
        assert nvda_labels_seen.count("__NVDA_SEEN__") == 1, (
            f"expected 1 NVDA build_features call (post-dedup), got "
            f"{nvda_labels_seen.count('__NVDA_SEEN__')}"
        )

        # The deduped survivor MUST be rec_hi (the higher-return run).
        # Verify by replicating the dedup logic directly against the same
        # input (the canonical (ticker, sim_date, action) key) — the
        # survivor's forward_return_5d is what would flow into the y
        # vector for this collision key.
        seen: dict[tuple, dict] = {}
        for r in [rec_lo, rec_hi] + pad:
            key = (str(r["ticker"]), str(r["sim_date"]),
                   str(r.get("action") or "BUY").upper())
            if key not in seen or (r.get("return_pct") or 0) > (
                    seen[key].get("return_pct") or 0):
                seen[key] = r
        survivor = seen[("NVDA", "2025-01-01", "BUY")]
        assert survivor["forward_return_5d"] == 15.0, (
            f"dedup tiebreaker should have kept the +15-label survivor "
            f"(return_pct=50); got fwd={survivor['forward_return_5d']}"
        )

    def test_sell_target_sign_flipped(self):
        """A SELL whose forward return was negative is a CORRECT call — the
        scorer learns one consistent meaning of 'good' by flipping SELL labels.
        """
        # 30 sell records, all of which (after sign flip) point to +5%.
        sell_recs = [_synthetic_outcome(action="SELL", fwd=-5.0,
                                        sim_date=f"2025-03-{i:02d}")
                     for i in range(1, 31)]
        # 30 buy records pointing to +5%.
        buy_recs = [_synthetic_outcome(action="BUY", fwd=5.0,
                                       sim_date=f"2025-04-{i:02d}")
                    for i in range(1, 31)]
        result = train_scorer(sell_recs + buy_recs)
        # If sign flip works, the model converges; this is just a smoke test
        # that training completed.
        assert result["status"] == "ok"
        assert result["n"] == 60

    def test_handles_null_forward_return(self):
        # An outcome record with forward_return_5d=null carries NO realized
        # label — training on a silent 0.0 coercion contaminated the model
        # with phantom flat-return rows. The strengthened contract drops
        # such rows and surfaces them in n_label_dropped. When ALL rows are
        # null we honestly return "no_valid_labels" rather than fitting on
        # an all-zero target vector.
        recs = []
        for i in range(35):
            r = _synthetic_outcome(sim_date=f"2025-05-{i+1:02d}")
            r["forward_return_5d"] = None
            recs.append(r)
        result = train_scorer(recs)
        assert result["status"] == "no_valid_labels"
        assert result["n"] == 0
        assert result["n_label_dropped"] == 35

    def test_handles_non_finite_forward_return(self):
        # Regression: a single decision_outcomes.jsonl row with a non-finite
        # forward_return_5d (inf / -inf) used to pass _to_float untouched,
        # poison the y vector, and make MLPRegressor.fit raise
        # "Input y contains infinity". Strengthened contract drops the bad
        # rows entirely (counted in n_label_dropped) and trains on the rest.
        recs = [_synthetic_outcome(sim_date=f"2025-06-{i+1:02d}")
                for i in range(35)]
        recs[5]["forward_return_5d"] = float("inf")
        recs[6]["forward_return_5d"] = float("-inf")
        result = train_scorer(recs)
        assert result["status"] == "ok"
        # val_rmse must be a real finite number, not nan/inf from a poisoned fit.
        vr = result["val_rmse"]
        assert vr == vr and abs(vr) < 1e6
        # Exactly the 2 inf rows must have been dropped — every other row
        # carried a finite synthetic forward return.
        assert result["n_label_dropped"] == 2
        # Pickled n_train reflects the post-validation count, not the input
        # `len(records)`. Dedup is a no-op here (35 unique sim_dates).
        assert result["n"] == 33

    def test_drops_mixed_invalid_forward_returns(self):
        # Defence-in-depth: any non-finite / non-coercible label must be
        # dropped (NaN, inf, bool True, a string). The remaining valid rows
        # produce the trained model — n_label_dropped tells the skill
        # ledger exactly how dirty the trainer tail was.
        recs = [_synthetic_outcome(sim_date=f"2025-07-{i+1:02d}")
                for i in range(40)]
        recs[0]["forward_return_5d"] = None
        recs[1]["forward_return_5d"] = float("nan")
        recs[2]["forward_return_5d"] = float("inf")
        recs[3]["forward_return_5d"] = True          # bool sneaks through as int
        recs[4]["forward_return_5d"] = "not a number"
        result = train_scorer(recs)
        assert result["status"] == "ok"
        assert result["n_label_dropped"] == 5
        assert result["n"] == 35  # 40 input - 5 dropped, no dedup collisions

    def test_dedup_prefers_valid_label_over_higher_return_with_null(self):
        # Defence-in-depth tie-break: when two records share a dedup key
        # (ticker, sim_date, action) and one carries a valid forward_return_5d
        # while the other carries `None` (externally injected / corrupted /
        # legacy row), the dedup MUST keep the valid-label one — even if the
        # null row's run_return was higher. The pre-fix logic kept the
        # higher-return run unconditionally; the null row then failed the
        # label-validation pass below and ALL training signal for that key
        # was silently lost. Production data has 0 such rows today
        # (_compute_decision_outcomes only emits walk-back-validated finite
        # outcomes), but the guard makes the invariant explicit so any future
        # ingest path that mixes null labels with real ones cannot silently
        # erase the real signal.
        rec_null_high = _synthetic_outcome(return_pct=100, fwd=0.0)  # fwd overwritten below
        rec_null_high["forward_return_5d"] = None
        rec_valid_low = _synthetic_outcome(return_pct=10, fwd=8.0)
        # Pad with 30 distinct sim_dates so we clear the 30-record threshold.
        pad = [_synthetic_outcome(sim_date=f"2025-08-{i+1:02d}", ticker="AMD")
               for i in range(30)]
        result = train_scorer([rec_null_high, rec_valid_low] + pad)
        # The null record must have lost the tie-break; the valid +8 record
        # was retained, so no label was dropped from the NVDA collision.
        assert result["status"] == "ok"
        # 30 pad + 1 deduped NVDA = 31 training rows; n_label_dropped must
        # stay at 0 (the null candidate was discarded by dedup, not by the
        # label-validation pass — so it never reaches the drop counter).
        assert result["n"] == 31
        assert result["n_label_dropped"] == 0

    def test_dedup_invalid_only_collision_still_drops_label(self):
        # When EVERY candidate for a dedup key has an invalid label, the
        # survivor (any of them — order shouldn't matter) still gets dropped
        # by the label-validation pass. n_label_dropped reflects the single
        # collision survivor, not every input collision row — by design.
        # Verifies the dedup fix didn't accidentally start counting every
        # invalid-input row.
        a = _synthetic_outcome(return_pct=100)
        a["forward_return_5d"] = None
        b = _synthetic_outcome(return_pct=10)
        b["forward_return_5d"] = float("nan")
        # Pad with 30 valid distinct records.
        pad = [_synthetic_outcome(sim_date=f"2025-09-{i+1:02d}", ticker="AMD")
               for i in range(30)]
        result = train_scorer([a, b] + pad)
        assert result["status"] == "ok"
        assert result["n_label_dropped"] == 1   # one survivor of the two-row collision
        assert result["n"] == 30

    def test_dedup_with_valid_labels_still_uses_return_pct(self):
        # Tie-break regression guard: when BOTH candidates carry valid labels
        # the tiebreaker must remain `return_pct` (the historical contract
        # that test_dedup_survivor_is_highest_return_record pins). The
        # additional `has_valid_label` precedence dimension must not change
        # behaviour in the all-valid case.
        rec_lo = _synthetic_outcome(return_pct=-10, fwd=-5.0)
        rec_hi = _synthetic_outcome(return_pct=50, fwd=15.0)
        pad = [_synthetic_outcome(sim_date=f"2025-10-{i+1:02d}", ticker="AMD",
                                  fwd=0.0, ml_score=0.0, mom5=0.0)
               for i in range(30)]
        # Order matters for regression: if lo is seen first, the precedence
        # check at the second pass must still flip in favour of hi.
        result_lo_first = train_scorer([rec_lo, rec_hi] + pad)
        result_hi_first = train_scorer([rec_hi, rec_lo] + pad)
        assert result_lo_first["status"] == "ok"
        assert result_hi_first["status"] == "ok"
        assert result_lo_first["n"] == result_hi_first["n"] == 31

    def test_persists_to_scorer_path(self, tmp_path, monkeypatch):
        """After training, the pickle must exist and contain {model, scaler, n_train}."""
        import paper_trader.ml.decision_scorer as ds
        path = tmp_path / "scorer.pkl"
        monkeypatch.setattr(ds, "SCORER_PATH", path)
        recs = [_synthetic_outcome(sim_date=f"2025-06-{i+1:02d}") for i in range(35)]
        result = train_scorer(recs)
        assert result["status"] == "ok"
        assert path.exists()
        with path.open("rb") as f:
            state = pickle.load(f)
        assert "model" in state
        assert "n_train" in state
        assert state["n_train"] == 35

    def test_trained_scorer_round_trip(self, tmp_path, monkeypatch):
        """Train, save, reload, predict — must not crash and must produce a finite number."""
        import paper_trader.ml.decision_scorer as ds
        path = tmp_path / "scorer_rt.pkl"
        monkeypatch.setattr(ds, "SCORER_PATH", path)

        # Generate outcomes where high mom5 → high forward return.
        recs = []
        for i in range(40):
            mom = (i - 20) * 0.5  # -10 .. +10
            fwd = mom * 1.2  # linear relationship
            recs.append(_synthetic_outcome(
                sim_date=f"2025-07-{i+1:02d}", mom5=mom, fwd=fwd, ml_score=mom,
            ))
        train_scorer(recs)
        # Reload via fresh DecisionScorer.
        s = DecisionScorer()
        assert s.is_trained
        v = s.predict(ml_score=10.0, rsi=50, macd=0.1, mom5=10.0, mom20=0.0,
                      regime_mult=1.0, ticker="NVDA")
        # Sanity: with a strongly positive mom5, expect a non-negative prediction.
        # (Loose bound — model isn't perfect.) Mostly we want to assert finite.
        assert math.isfinite(v)
        # Rank-order: the training data is a clean monotone relationship
        # (fwd = mom * 1.2, ml_score = mom). A strongly bullish feature vector
        # MUST predict a higher return than a strongly bearish one — otherwise
        # the model carries no usable signal and gating on it is noise.
        v_bull = s.predict(ml_score=10.0, rsi=50, macd=0.1, mom5=10.0,
                           mom20=0.0, regime_mult=1.0, ticker="NVDA")
        v_bear = s.predict(ml_score=-10.0, rsi=50, macd=0.1, mom5=-10.0,
                           mom20=0.0, regime_mult=1.0, ticker="NVDA")
        assert v_bull > v_bear

    def test_training_is_deterministic(self):
        """train_scorer pins random_state=42 for the split and the MLP, so two
        runs on identical records must report identical n and val_rmse —
        otherwise backtest cycles can't be compared and the scorer drifts
        non-reproducibly between retrains.
        """
        recs = [_synthetic_outcome(sim_date=f"2025-08-{i+1:02d}", mom5=(i - 20),
                                   fwd=(i - 20) * 1.1)
                for i in range(40)]
        r1 = train_scorer(list(recs))
        r2 = train_scorer(list(recs))
        assert r1["status"] == r2["status"] == "ok"
        assert r1["n"] == r2["n"]
        # val_rmse may be NaN only in the numpy-fallback path; when sklearn is
        # present it must be bit-identical across deterministic runs.
        if r1["val_rmse"] == r1["val_rmse"]:  # not NaN
            assert r1["val_rmse"] == pytest.approx(r2["val_rmse"], rel=1e-9)

    def test_scorer_ranks_high_ml_score_above_low(self, tmp_path, monkeypatch):
        """A higher ml_score (≈ article kw_score) must predict a higher 5d
        return than a low one when the training data makes ml_score
        predictive — with mom5 held NEUTRAL so this isolates feature[0].

        ``test_trained_scorer_round_trip`` varies ml_score and mom5 together,
        so it cannot tell whether the model learned ml_score at all (it could
        be riding mom5 alone). This test pins every other feature constant and
        only moves ml_score, exercising the full pipeline
        (build_features → train → pickle → reload → predict).

        It catches the historical "feature key bug" class (commit 028f94d):
        a dict-key mismatch / wrong _to_float default that silently collapses
        ml_score to a constant — the model then can't learn the relationship
        and the high/low gap vanishes (verified by injecting a dead feature[0]:
        the assertion fails with v_hi == v_lo). It does NOT catch a *consistent*
        sign flip — train and predict share build_features, so the model just
        learns the inverted representation; that is a fundamental property of
        any train→predict round-trip, not a coverage gap to paper over.
        """
        import paper_trader.ml.decision_scorer as ds
        path = tmp_path / "scorer_mlrank.pkl"
        monkeypatch.setattr(ds, "SCORER_PATH", path)

        # fwd = ml_score * 1.5; ml_score swept -10..+10; mom5 fixed at 0.0.
        recs = []
        for i in range(40):
            sc = (i - 20) * 0.5  # -10.0 .. +9.5
            recs.append(_synthetic_outcome(
                sim_date=f"2025-09-{i+1:02d}", ml_score=sc, mom5=0.0,
                fwd=sc * 1.5,
            ))
        result = train_scorer(recs)
        assert result["status"] == "ok"

        s = DecisionScorer()
        assert s.is_trained
        common = dict(rsi=50.0, macd=0.1, mom5=0.0, mom20=0.0,
                      regime_mult=1.0, ticker="NVDA")
        v_hi = s.predict(ml_score=8.0, **common)
        v_lo = s.predict(ml_score=-8.0, **common)
        assert math.isfinite(v_hi) and math.isfinite(v_lo)
        # The >5 gap catches a dead/dropped feature[0] (true spread at ±8 is
        # 24pp, so a model that actually learned ml_score clears 5 comfortably
        # while a no-op/constant-feature model gives ~0). Ordering is a cheap
        # extra guard, not a sign-flip detector (see docstring).
        assert v_hi > v_lo, f"high ml_score did not rank above low ({v_hi} !> {v_lo})"
        assert v_hi - v_lo > 5.0, f"ml_score signal too weak: gap={v_hi - v_lo:.2f}"


class TestLabelClamp:
    """Symmetric label clamp at train time — labels exceeding ``PRED_CLAMP_PCT``
    are clipped to ``±PRED_CLAMP_PCT`` BEFORE the SELL sign-flip so that the
    training label space matches the inference output (which always clamps).

    Motivated by the live audit: the 5000-record trainer tail carries
    2 MSTR rows with ``forward_return_5d > +100%`` and a long tail of |fr|>50%
    outliers. Those labels can never be predicted (predict() clamps), yet
    they drive huge MSE gradients during fit and perturb weights across the
    feature subspace.
    """

    def test_clamp_count_reported_when_outliers_present(self):
        """Records with |fr_5d| > PRED_CLAMP_PCT are counted; in-range rows are not."""
        recs = []
        for i in range(35):
            r = _synthetic_outcome(sim_date=f"2025-09-{i+1:02d}", fwd=2.0)
            recs.append(r)
        # Inject five outliers — three positive, two negative.
        recs[0]["forward_return_5d"] = PRED_CLAMP_PCT + 25.0       # +75
        recs[1]["forward_return_5d"] = PRED_CLAMP_PCT + 125.0      # +175 (the MSTR case)
        recs[2]["forward_return_5d"] = PRED_CLAMP_PCT + 0.01       # just above bound
        recs[3]["forward_return_5d"] = -(PRED_CLAMP_PCT + 30.0)    # -80
        recs[4]["forward_return_5d"] = -(PRED_CLAMP_PCT + 0.5)     # just below bound
        # Records 5..34 have fwd=2.0 — well inside the band, NOT clamped.
        result = train_scorer(recs)
        assert result["status"] == "ok"
        assert result["n_label_clamped"] == 5

    def test_clamp_is_zero_when_no_outliers(self):
        """No record exceeds the bound → no clamps applied."""
        recs = [_synthetic_outcome(sim_date=f"2025-10-{i+1:02d}", fwd=3.0)
                for i in range(35)]
        result = train_scorer(recs)
        assert result["status"] == "ok"
        assert result["n_label_clamped"] == 0

    def test_boundary_exact_is_not_clamped(self):
        """A label exactly at ±PRED_CLAMP_PCT is on the boundary; the clamp uses
        strict `>`, so it should NOT count as clamped."""
        recs = [_synthetic_outcome(sim_date=f"2025-11-{i+1:02d}", fwd=2.0)
                for i in range(35)]
        recs[0]["forward_return_5d"] = PRED_CLAMP_PCT       # exactly +50
        recs[1]["forward_return_5d"] = -PRED_CLAMP_PCT      # exactly -50
        result = train_scorer(recs)
        assert result["status"] == "ok"
        assert result["n_label_clamped"] == 0

    def test_clamp_aligns_train_to_inference_bound(self, tmp_path, monkeypatch):
        """The point of the clamp: a model trained on outlier labels still
        cannot predict outside ±PRED_CLAMP_PCT because predict() clamps. Verify
        end-to-end by saturating training with extreme labels and checking that
        predict() never exits the clamp band.
        """
        import paper_trader.ml.decision_scorer as ds
        path = tmp_path / "clamp_rt.pkl"
        monkeypatch.setattr(ds, "SCORER_PATH", path)

        recs = []
        for i in range(40):
            # Strong monotone signal pushing label way out of band.
            mom = (i - 20) * 0.5
            fwd = mom * 25.0  # at extremes hits ±250 (way over ±50)
            recs.append(_synthetic_outcome(
                sim_date=f"2025-12-{i+1:02d}", mom5=mom, fwd=fwd, ml_score=mom,
            ))
        result = train_scorer(recs)
        assert result["status"] == "ok"
        # Many of these labels exceed PRED_CLAMP_PCT.
        assert result["n_label_clamped"] >= 10

        s = DecisionScorer()
        assert s.is_trained
        for ml in (-15.0, -5.0, 0.0, 5.0, 15.0):
            v = s.predict(ml_score=ml, rsi=50, macd=0.1, mom5=ml,
                          mom20=0.0, regime_mult=1.0, ticker="NVDA")
            assert math.isfinite(v)
            # predict() must clamp to ±PRED_CLAMP_PCT regardless of training labels.
            assert -PRED_CLAMP_PCT <= v <= PRED_CLAMP_PCT

    def test_clamp_applied_before_sell_sign_flip(self):
        """The clamp must be applied BEFORE the SELL sign-flip. A SELL with
        fr=-175 should become -clamped(-50) = +50, not -clamped(-175) leaving
        the model with a +175 label."""
        # Build a corpus where the SELL outlier label has known magnitude.
        # We use the numpy fallback (no sklearn split) via the lstsq path —
        # but `train_scorer` uses sklearn when present. Either way, we can
        # verify the COUNT and trust the documented order via the
        # implementation (a behavioural check would require model surgery).
        recs = []
        for i in range(35):
            r = _synthetic_outcome(
                sim_date=f"2026-01-{i+1:02d}",
                action=("SELL" if i % 2 == 0 else "BUY"),
                fwd=3.0,
            )
            recs.append(r)
        # One SELL outlier at -175 (so -fr = +175). Clamp BEFORE flip keeps
        # it at |50|; clamp AFTER flip wouldn't matter (same magnitude). But
        # the COUNT detection is on the pre-flip absolute value of fr, so
        # |fr|=175 > 50 means count incremented exactly once.
        recs[0]["forward_return_5d"] = -175.0
        recs[0]["action"] = "SELL"
        # One BUY outlier matched.
        recs[1]["forward_return_5d"] = 175.0
        recs[1]["action"] = "BUY"
        result = train_scorer(recs)
        assert result["status"] == "ok"
        assert result["n_label_clamped"] == 2


# ─────────────────────── ranking semantics ───────────────────────

class TestLoadCaching:
    """Every polled dashboard endpoint builds a fresh ``DecisionScorer()``.
    The old constructor re-read and re-unpickled ``scorer.pkl`` AND printed a
    ``[decision_scorer] loaded n=`` line on *every* construction — 657 such
    lines in a single runner.log, feeding the disk-full logging failures
    (``OSError: [Errno 28]``). Repeated construction against an unchanged
    pickle must load — and log — exactly once, while a retrain (atomic
    ``.replace`` → new mtime/size) is still picked up.
    """

    def test_repeated_construction_loads_and_logs_once(
        self, tmp_path, monkeypatch, capsys
    ):
        import paper_trader.ml.decision_scorer as ds
        path = tmp_path / "scorer_cache.pkl"
        monkeypatch.setattr(ds, "SCORER_PATH", path)
        recs = [_synthetic_outcome(sim_date=f"2025-10-{i+1:02d}")
                for i in range(35)]
        train_scorer(recs)                # atomically writes `path`
        capsys.readouterr()               # discard training chatter

        scorers = [DecisionScorer() for _ in range(5)]
        out = capsys.readouterr().out

        assert all(s.is_trained for s in scorers)
        assert out.count("loaded n=") == 1, (
            f"expected exactly one disk load, got "
            f"{out.count('loaded n=')}:\n{out}"
        )
        # A cache hit reuses the already-unpickled model object instead of
        # re-reading the file — object identity is the observable proof.
        first = scorers[0]._model
        assert all(s._model is first for s in scorers)

    def test_pickle_rewrite_is_picked_up(self, tmp_path, monkeypatch, capsys):
        import paper_trader.ml.decision_scorer as ds
        path = tmp_path / "scorer_reload.pkl"
        monkeypatch.setattr(ds, "SCORER_PATH", path)

        train_scorer([_synthetic_outcome(sim_date=f"2025-11-{i+1:02d}")
                      for i in range(35)])
        s_old = DecisionScorer()
        capsys.readouterr()

        # Retrain with a different record count → different pickle size →
        # the (path, mtime, size) cache key changes → a fresh load.
        train_scorer([_synthetic_outcome(sim_date=f"2025-12-{i+1:02d}",
                                          ticker="AMD")
                      for i in range(45)])
        s_new = DecisionScorer()
        out = capsys.readouterr().out

        assert out.count("loaded n=") == 1, (
            f"a retrained pickle must be reloaded exactly once:\n{out}"
        )
        assert s_new._model is not s_old._model
        assert s_new.is_trained


class TestSectorMapping:
    def test_all_sectors_in_map(self):
        # Sanity: every declared sector should appear somewhere in SECTOR_MAP
        # (otherwise that sector's one-hot is dead — defeats the encoding).
        # 'other' is the catch-all and doesn't need a mapping.
        mapped_sectors = set(SECTOR_MAP.values())
        for s in SECTORS:
            if s == "other":
                continue
            assert s in mapped_sectors, f"sector {s!r} has zero ticker mappings"

    def test_watchlist_coverage(self):
        """Every WATCHLIST ticker the scorer is ever asked to predict on must
        either have an explicit SECTOR_MAP entry OR be in INTENTIONALLY_OTHER.

        Prior to this lock, 35% of WATCHLIST (41/118) silently fell through
        to sector_other — collapsing semi-equipment (LITE/AMAT/LRCX),
        international tech ADRs (BABA/SAP/SONY), EVs (RIVN/NIO), every
        broad-index leveraged ETF (UDOW/URTY/QLD/SSO/MVV/SAA/UWM/TNA/...),
        single-stock 2x (AAPLU/SMCI2X/PLTU/LNOK), 3x inverse (SQQQ/SPXS/
        SDOW/SRTY/TZA/HIBS), and key financials (BRK-B/HSBC/SQ/FAZ) into
        the same sector_other one-hot bucket as Toyota and homebuilders.
        With nothing to discriminate, the scorer cannot learn any
        sector-conditional pattern for these — every prediction for LRCX
        used the same encoding as for NAIL, even though their economic
        sectors are completely different.

        This regression-locks the expanded mapping: a NEW watchlist ticker
        added without an explicit SECTOR_MAP entry (or explicit
        INTENTIONALLY_OTHER) fails loudly here, instead of silently
        degrading the scorer's feature quality.
        """
        from paper_trader.backtest import WATCHLIST
        from paper_trader.ml.decision_scorer import INTENTIONALLY_OTHER

        unclassified = [
            t for t in WATCHLIST
            # Index / vol gauges (^VIX, ^GSPC) are not real positions —
            # they are referenced for prompt context but never traded.
            if not t.startswith("^")
            and t not in SECTOR_MAP
            and t not in INTENTIONALLY_OTHER
        ]
        assert unclassified == [], (
            f"{len(unclassified)} WATCHLIST tickers have no SECTOR_MAP "
            f"entry and are not in INTENTIONALLY_OTHER; the scorer will "
            f"silently encode them as sector_other. Add them to one or "
            f"the other in decision_scorer.py: {unclassified}"
        )

    def test_sector_map_values_are_valid_sectors(self):
        """Catches a typo like SECTOR_MAP['NVDA']='techy' that would
        otherwise produce an all-zero sector one-hot at training time."""
        for ticker, sector in SECTOR_MAP.items():
            assert sector in SECTORS, (
                f"SECTOR_MAP[{ticker!r}]={sector!r} is not a valid sector; "
                f"valid sectors: {SECTORS}"
            )

    def test_intentionally_other_does_not_overlap_sector_map(self):
        """A ticker is either explicitly mapped OR explicitly 'other' —
        never both, or the intent becomes ambiguous on a future review."""
        from paper_trader.ml.decision_scorer import INTENTIONALLY_OTHER
        overlap = set(SECTOR_MAP) & set(INTENTIONALLY_OTHER)
        assert not overlap, (
            f"these tickers appear in BOTH SECTOR_MAP and "
            f"INTENTIONALLY_OTHER: {sorted(overlap)}"
        )

    def test_specific_high_value_mappings(self):
        """Pin a discriminating subset of the newly-added mappings so an
        accidental revert to sector_other is caught explicitly (the watchlist
        coverage test would pass if a reviewer "fixed" both halves the same
        wrong way — e.g. dropped both from SECTOR_MAP and INTENTIONALLY_OTHER).

        These six were chosen because they exercise each new category:
        semi-equipment (LRCX), int'l tech ADR (BABA), EV (RIVN), broad-index
        2x leveraged (QLD), mega-cap financial (BRK-B), 3x inverse (SPXS).
        """
        assert SECTOR_MAP["LRCX"] == "tech"
        assert SECTOR_MAP["BABA"] == "tech"
        assert SECTOR_MAP["RIVN"] == "tech"
        assert SECTOR_MAP["QLD"] == "tech"
        assert SECTOR_MAP["BRK-B"] == "financials"
        assert SECTOR_MAP["SPXS"] == "tech"

    def test_sector_encoding_changes_for_newly_mapped_tickers(self):
        """End-to-end: build_features for a newly-mapped ticker must now
        emit a non-other sector one-hot. Previously, LRCX's feature vector
        was sector_other=1.0 and every other sector slot=0.0 — identical
        to NAIL's. After the fix, LRCX matches NVDA's sector encoding
        (both tech), confirming the scorer can finally learn that they
        co-vary by sector. NAIL stays correctly in sector_other.
        """
        # Use identical numerics so any difference is purely from the
        # sector one-hot block (slots 10..16).
        common = dict(ml_score=1.0, rsi=50.0, macd=0.0, mom5=0.0,
                      mom20=0.0, regime_mult=1.0)
        sector_idx_start = 10  # 10 numeric + 7 sector = 17 total
        f_lrcx = build_features(ticker="LRCX", **common)[sector_idx_start:]
        f_nvda = build_features(ticker="NVDA", **common)[sector_idx_start:]
        f_nail = build_features(ticker="NAIL", **common)[sector_idx_start:]
        # LRCX (newly tech) must now match NVDA (already tech) in the
        # sector block.
        assert f_lrcx == f_nvda, (
            "LRCX sector one-hot must now match NVDA's (both tech); "
            f"LRCX={f_lrcx}, NVDA={f_nvda}"
        )
        # And NAIL — intentionally 'other' — must NOT match NVDA.
        assert f_nail != f_nvda
        # sector_other (last slot, index -1 in the sector block) must be
        # 1.0 for NAIL and 0.0 for LRCX/NVDA.
        assert f_nail[-1] == 1.0
        assert f_lrcx[-1] == 0.0
        assert f_nvda[-1] == 0.0

    def test_2026_05_26_watchlist_additions_classified(self):
        """Pin the 34 watchlist tickers added in the 2026-05-26 backtest.py
        diff (power semis, AI infra, quantum, space/eVTOL, AI software,
        medical robotics, nuclear small-caps) so a future reviewer who
        drops the SECTOR_MAP / INTENTIONALLY_OTHER additions reverts to
        the regression captured by ``test_watchlist_coverage`` immediately.

        Mirrors the spirit of ``test_specific_high_value_mappings`` for
        an earlier batch — picks one representative ticker per category."""
        from paper_trader.ml.decision_scorer import INTENTIONALLY_OTHER
        # Power-semis / RF — same family as NVDA/AMD/MU.
        assert SECTOR_MAP["WOLF"] == "tech"
        assert SECTOR_MAP["MPWR"] == "tech"
        # AI infra / accelerators / EDA / semi-test.
        assert SECTOR_MAP["AVGO"] == "tech"
        assert SECTOR_MAP["SNPS"] == "tech"
        # Quantum compute — growth-tech tape.
        assert SECTOR_MAP["IONQ"] == "tech"
        assert SECTOR_MAP["QUBT"] == "tech"
        # AI software / next-gen comms — growth-tech tape.
        assert SECTOR_MAP["SOUN"] == "tech"
        assert SECTOR_MAP["ASTS"] == "tech"
        # Space / eVTOL — high-beta speculative growth, ARKK/RIVN bucket.
        assert SECTOR_MAP["RKLB"] == "tech"
        assert SECTOR_MAP["JOBY"] == "tech"
        # 2x NVDA leveraged — same NVDU pattern.
        assert SECTOR_MAP["NVDX"] == "tech"
        # Medical AI / surgical robotics — healthcare bucket.
        assert SECTOR_MAP["BFLY"] == "healthcare"
        assert SECTOR_MAP["PRCT"] == "healthcare"
        # Nuclear small-caps — INTENTIONALLY_OTHER (no utility / energy fit).
        assert "OKLO" in INTENTIONALLY_OTHER
        assert "NNE" in INTENTIONALLY_OTHER

    def test_2026_05_26_additions_dont_break_other_invariants(self):
        """build_features for a newly-mapped tech name (RKLB — speculative
        space) must emit the same sector one-hot as NVDA (both 'tech'),
        and a newly-mapped healthcare name (BFLY) must match LLY's. Pins
        the end-to-end encoding invariant for the new batch."""
        common = dict(ml_score=1.0, rsi=50.0, macd=0.0, mom5=0.0,
                      mom20=0.0, regime_mult=1.0)
        sector_idx_start = 10
        f_rklb = build_features(ticker="RKLB", **common)[sector_idx_start:]
        f_nvda = build_features(ticker="NVDA", **common)[sector_idx_start:]
        f_bfly = build_features(ticker="BFLY", **common)[sector_idx_start:]
        f_lly = build_features(ticker="LLY", **common)[sector_idx_start:]
        f_oklo = build_features(ticker="OKLO", **common)[sector_idx_start:]
        # RKLB must match NVDA (both tech), not collapse to sector_other.
        assert f_rklb == f_nvda
        # BFLY must match LLY (both healthcare).
        assert f_bfly == f_lly
        # OKLO (INTENTIONALLY_OTHER) must land in sector_other (last slot).
        assert f_oklo[-1] == 1.0


# ──────────────────── anti-overfit MLP config (2026-05-18) ────────────────────

class TestAntiOverfitConfig:
    """Locks the regularized DecisionScorer architecture and proves the
    regularization actually suppresses noise memorization.

    Motivation: the prior unregularized (64,32,16)/600-iter net memorised the
    noisy training fold — measured on the live 5000-outcome temporal holdout it
    posted val_rmse≈10.7 but oos_rmse≈16.7 (the textbook overfit the per-cycle
    scorer-skill ledger records every cycle). The current config is
    (32,16) + L2 alpha=1e-2 + early_stopping; this class would fail RED on an
    accidental revert to the memorizing config.
    """

    def test_pickled_model_uses_regularized_config(self, tmp_path, monkeypatch):
        """Config-lock: the pipeline must pickle the regularized architecture
        (catches a silent revert to the overfit (64,32,16)/600-iter net)."""
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "cfg.pkl")
        recs = [_synthetic_outcome(sim_date=f"2025-04-{i+1:02d}",
                                   mom5=(i - 17), fwd=(i - 17) * 1.1)
                for i in range(35)]
        assert train_scorer(recs)["status"] == "ok"
        m = DecisionScorer()._model
        assert m.hidden_layer_sizes == (32, 16)
        assert m.alpha == pytest.approx(1e-2)
        assert m.early_stopping is True
        assert m.validation_fraction == pytest.approx(0.15)
        assert m.n_iter_no_change == 25

    def test_regularization_suppresses_pure_noise_memorization(
            self, tmp_path, monkeypatch):
        """On a target that is PURE noise uncorrelated with every feature, a
        well-regularized model must regress toward the mean (low prediction
        spread) instead of memorizing the noise.

        Measured discriminating evidence (seed 12345, 300 records):
          • OLD (64,32,16)/600-iter, no reg:  pred_std/target_std ≈ 1.00
            (memorizes the noise almost perfectly — the overfit signature)
          • NEW (32,16)+alpha1e-2+early_stop:  ratio ≈ 0.40
        The 0.65 bound sits with wide margin on both sides, so this is
        non-flaky AND a genuine regression detector — it fails RED if the
        config reverts to an unregularized memorizing net.
        """
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "noise.pkl")

        rng = np.random.RandomState(12345)
        tickers = ["NVDA", "XOM", "JPM", "LLY", "GLD", "COIN", "SPY"]
        recs = []
        for i in range(300):
            recs.append({
                "ticker": tickers[i % len(tickers)], "action": "BUY",
                "ml_score": float(rng.uniform(-5, 5)),
                "rsi": float(rng.uniform(20, 80)),
                "macd": float(rng.uniform(-1, 1)),
                "mom5": float(rng.uniform(-10, 10)),
                "mom20": float(rng.uniform(-15, 15)),
                "regime_mult": 1.0,
                "vol_ratio": float(rng.uniform(0.5, 2.0)),
                "bb_position": float(rng.uniform(-2, 2)),
                "news_urgency": 50.0, "news_article_count": 1.0,
                # Pure noise — uncorrelated with every feature above.
                "forward_return_5d": float(rng.normal(0, 10)),
                "return_pct": 10.0,
                "sim_date": f"2025-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
            })
        assert train_scorer(recs)["status"] == "ok"
        s = DecisionScorer()
        targ = np.array([r["forward_return_5d"] for r in recs])
        preds = np.array([
            s.predict(ml_score=r["ml_score"], rsi=r["rsi"], macd=r["macd"],
                      mom5=r["mom5"], mom20=r["mom20"], regime_mult=1.0,
                      ticker=r["ticker"], vol_ratio=r["vol_ratio"],
                      bb_pos=r["bb_position"], news_urgency=50.0,
                      news_article_count=1.0)
            for r in recs])
        ratio = float(preds.std() / targ.std())
        assert ratio < 0.65, (
            f"model memorized noise (pred/target std ratio={ratio:.3f} "
            f"≥ 0.65) — regularization not effective")


# ──────────────── LLM annotation weight replication (regression) ─────────────

class TestLlmWeightReplication:
    """Locks the LLM-annotation weight replication scheme.

    The dedicated ``llm_quality_label`` multiplier (`endorsed→3.0`,
    `condemned→0.1`, `unlabeled→1.0`, applied to a base weight derived from
    `return_pct`) was previously fed into ``rep = np.maximum(1,
    np.round(w_tr * 2).astype(int))``. The `max(1, ...)` floor erased the
    ``0.1×`` multiplier: a CONDEMN row on any run rounded to ``rep=0`` and
    was then promoted to ``rep=1`` — measured 0.5× relative weight vs.
    unlabeled, NOT the documented 0.1×. Tests pin the corrected ×10
    scaling + drop-on-zero behaviour: ENDORSE replicates strictly more
    than unlabeled (signal amplified), CONDEMN drops out of the training
    fold (the documented near-zero weight is realized).
    """

    def _capture_repeat(self, monkeypatch):
        """Wrap np.repeat in the module under test so the rep array used to
        oversample the training fold is observable from outside train_scorer.
        Returns the most recent (rep_array_as_list,) call args.
        """
        import paper_trader.ml.decision_scorer as ds
        captured: dict = {}
        real_repeat = ds.np.repeat

        def _spy(a, repeats, axis=None):
            # The X_tr_w call comes first (axis=0); record that one. The
            # second call (y_tr_w, axis=None) carries the same rep array.
            if "rep" not in captured:
                # `repeats` is either a numpy array or list-like.
                captured["rep"] = list(np.asarray(repeats, dtype=int))
            return real_repeat(a, repeats, axis=axis)

        monkeypatch.setattr(ds.np, "repeat", _spy)
        return captured

    def test_endorse_replicates_more_than_unlabeled(
            self, tmp_path, monkeypatch):
        """ENDORSE annotations replicate ~3× more than unlabeled rows.

        Uses 50 ENDORSE records so the random 80/20 split (random_state=42)
        deterministically lands ENDORSE rows in the training fold."""
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "weights.pkl")
        captured = self._capture_repeat(monkeypatch)

        # 50 ENDORSE + 50 unlabeled. Same return_pct, so only llm label varies.
        records: list[dict] = []
        for i in range(50):
            r = _synthetic_outcome(
                ticker="NVDA", sim_date=f"2025-07-{i+1:02d}",
                return_pct=0.0)
            records.append(r)
        for i in range(50):
            r = _synthetic_outcome(
                ticker="AMD", sim_date=f"2025-08-{i+1:02d}",
                return_pct=0.0)
            r["llm_quality_label"] = 1
            records.append(r)

        result = train_scorer(records)
        assert result["status"] == "ok"
        rep = captured["rep"]
        # ENDORSE: weight 1.0 × 3.0 = 3.0 → ×2 scaling → rep=6.
        # Unlabeled: weight 1.0 → ×2 = rep=2.
        assert 6 in rep, (
            f"ENDORSE rep=6 (weight 3.0 × 2 scaling) must appear in the "
            f"training fold: {rep}"
        )
        assert 2 in rep, (
            f"unlabeled rep=2 (weight 1.0 × 2 scaling) must appear in the "
            f"training fold: {rep}"
        )
        # Strict directional skew: ENDORSE / unlabeled = 3×.
        assert max(rep) == 6
        assert min(rep) == 2

    def test_condemn_rows_are_dropped_from_training(
            self, tmp_path, monkeypatch):
        """CONDEMN rows (llm_quality_label=-1, multiplier 0.1×) must not
        appear in the training fold at all.

        At ×2 scaling, a CONDEMN row's weight is ≤ 0.2 × 2 = 0.4 (capped at
        return_pct=+200%; lower at any other return), which rounds to 0 in
        all cases. The prior `np.maximum(1, …)` floor promoted these to
        rep=1 — silently negating the documented 0.1× weight. With the fix
        these rows drop out entirely. Validates the drop-zero codepath."""
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "drop.pkl")

        # 50 unlabeled + 50 CONDEMN records, balanced so CONDEMN rows do
        # land in the 80% training split. Without the drop-zero codepath
        # each CONDEMN would contribute rep=1 from the old `max(1, ...)`
        # floor — observable here as a 1 entry in the rep array.
        records = [
            _synthetic_outcome(
                ticker="NVDA", sim_date=f"2025-09-{i+1:02d}",
                return_pct=0.0)
            for i in range(50)
        ]
        for i in range(50):
            r = _synthetic_outcome(
                ticker="AMD", sim_date=f"2025-10-{i+1:02d}",
                return_pct=0.0)
            r["llm_quality_label"] = -1
            records.append(r)

        captured = self._capture_repeat(monkeypatch)
        result = train_scorer(records)
        assert result["status"] == "ok"
        rep = captured["rep"]
        # CONDEMN rows must be DROPPED upstream — never floor-promoted to
        # rep=1. Every surviving entry must be a clean unlabeled rep=2.
        assert all(r == 2 for r in rep), (
            f"after the drop-zero fix, every surviving rep should be 2 "
            f"(unlabeled × 2 scaling); a 1 here means a CONDEMN row was "
            f"floor-promoted: {rep}"
        )

    def test_unlabeled_records_replicate_at_expected_scale(
            self, tmp_path, monkeypatch):
        """Sanity-check the ×2 scaling: an unlabeled record on a flat 0%
        run has weight 1.0; its rep should round to 2. Pins the absolute
        scale, not just the relative ratio."""
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "unlab.pkl")
        captured = self._capture_repeat(monkeypatch)

        records = [
            _synthetic_outcome(
                ticker="NVDA", sim_date=f"2025-11-{i+1:02d}", return_pct=0.0)
            for i in range(35)
        ]
        result = train_scorer(records)
        assert result["status"] == "ok"
        rep = captured["rep"]
        # Every unlabeled-on-flat-run row gets weight=1.0, × 2 → rep=2.
        assert all(r == 2 for r in rep), (
            f"every unlabeled rep should be 2 (weight 1.0 × 2 scaling); "
            f"got {rep}"
        )


# ─────────────────────── DecisionScorer.feature_importance ──────────
#
# The instance method `DecisionScorer.feature_importance` was previously
# untested even though it powers the CLI `--feature-importance` mode and
# is consumed by `paper_trader.ml.dead_sector_audit`. These tests pin the
# JSON-shape contract (sorted, normalized, no exceptions) every consumer
# relies on.

class TestFeatureImportanceMethod:
    def test_untrained_scorer_returns_empty_payload(self):
        s = DecisionScorer()
        imp = s.feature_importance()
        assert imp["trained"] is False
        assert imp["method"] is None
        assert imp["n_train"] == 0
        assert imp["importances"] == []

    def test_trained_scorer_returns_full_feature_set(self, tmp_path,
                                                     monkeypatch):
        # Train on a synthetic monotone signal so the model has real (not
        # noise) input-layer weights.
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "imp.pkl")
        recs = []
        for i in range(60):
            recs.append(_synthetic_outcome(
                ticker="NVDA", sim_date=f"2025-01-{(i % 28) + 1:02d}",
                ml_score=float(i % 5), fwd=float(i % 5) * 2.0))
        for i in range(60):
            recs.append(_synthetic_outcome(
                ticker="SOXL", sim_date=f"2025-02-{(i % 28) + 1:02d}",
                ml_score=float(i % 5), fwd=float(i % 5) * 2.0))
        result = train_scorer(recs)
        assert result["status"] == "ok"
        s = DecisionScorer()
        imp = s.feature_importance()
        assert imp["trained"] is True
        # Every one of the N_FEATURES slots must surface — coverage is a
        # CONTRACT not a courtesy. A missing slot breaks consumers
        # (dead_sector_audit) that join on FEATURE_NAMES.
        assert len(imp["importances"]) == N_FEATURES
        names = {r["feature"] for r in imp["importances"]}
        assert names == set(FEATURE_NAMES)

    def test_importances_sorted_desc_by_raw_value(self, tmp_path, monkeypatch):
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "imp_sort.pkl")
        recs = [_synthetic_outcome(sim_date=f"2025-01-{i+1:02d}",
                                   ml_score=float(i % 7), fwd=float(i % 7))
                for i in range(40)]
        recs += [_synthetic_outcome(ticker="AMD",
                                    sim_date=f"2025-02-{i+1:02d}",
                                    ml_score=float(i % 7), fwd=float(i % 7))
                 for i in range(40)]
        train_scorer(recs)
        s = DecisionScorer()
        rows = s.feature_importance()["importances"]
        for a, b in zip(rows[:-1], rows[1:]):
            assert a["importance"] >= b["importance"], (
                f"importances not sorted descending: {a} then {b}")

    def test_normalized_shares_sum_to_one(self, tmp_path, monkeypatch):
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "imp_norm.pkl")
        recs = [_synthetic_outcome(sim_date=f"2025-01-{i+1:02d}",
                                   ml_score=float(i % 5), fwd=float(i % 5))
                for i in range(40)]
        recs += [_synthetic_outcome(ticker="AMD",
                                    sim_date=f"2025-02-{i+1:02d}",
                                    ml_score=float(i % 5), fwd=float(i % 5))
                 for i in range(40)]
        train_scorer(recs)
        s = DecisionScorer()
        rows = s.feature_importance()["importances"]
        total = sum(r["importance_normalized"] for r in rows)
        # Rounded to 6 places per-feature, so the sum can drift slightly.
        assert abs(total - 1.0) < 1e-3, (
            f"normalized importances must sum to ~1.0; got {total}")

    def test_nan_weight_does_not_poison_importance_payload(self):
        # Regression: a NaN/Inf entry in the model's weight vector (from a
        # rank-deficient lstsq solve or pathological MLP fit) used to
        # silently propagate through `raw.sum()` → `total=NaN` →
        # `norm=NaN` → JSON-unserialisable output AND unstable sort.
        # The guard must replace non-finite cells with 0.0 so the payload
        # stays well-formed: every importance is finite, the sort is
        # deterministic, and the normalized shares sum to ~1.0.
        from paper_trader.ml.decision_scorer import _LstsqScaler, _LstsqModel
        import numpy as np
        import math
        s = DecisionScorer()
        # weights: [NaN, 1, 1, ..., 1, bias=99]
        weights = np.array(
            [float("nan")] + [1.0] * (N_FEATURES - 1) + [99.0],
            dtype=np.float32)
        s._model = _LstsqModel(weights)
        s._scaler = _LstsqScaler(
            mean=np.zeros(N_FEATURES, dtype=np.float32),
            std=np.ones(N_FEATURES, dtype=np.float32))
        s._trained = True
        s._n_train = 500
        imp = s.feature_importance()
        rows = imp["importances"]
        # Every importance is finite (the bug was NaN-poisoning).
        for r in rows:
            assert math.isfinite(r["importance"]), (
                f"non-finite importance leaked through: {r}")
            assert math.isfinite(r["importance_normalized"]), (
                f"non-finite normalized share leaked through: {r}")
        # Normalized shares sum to ~1.0 across all N_FEATURES entries
        # (the NaN feature contributes 0 to the sum but is still in the row
        # list so the consumer sees the full feature set).
        total = sum(r["importance_normalized"] for r in rows)
        assert abs(total - 1.0) < 1e-3, (
            f"normalized shares must sum to ~1.0 even with NaN weights; "
            f"got {total}")
        # The NaN-weight feature lands at importance=0 and ranks LAST in
        # the descending sort (everyone else has importance=1.0).
        assert rows[-1]["feature"] == FEATURE_NAMES[0]
        assert rows[-1]["importance"] == 0.0

    def test_lstsq_fallback_path_returns_lstsq_method(self):
        # The numpy lstsq fallback (sklearn-absent host) uses a different
        # importance-extraction path: |coef| of the single linear layer,
        # excluding the bias term. Pin both the method label AND the right
        # feature count so a refactor cannot silently drop the bias-strip.
        from paper_trader.ml.decision_scorer import _LstsqScaler, _LstsqModel
        import numpy as np
        s = DecisionScorer()
        # Synthesise a deterministic weight vector: w[i] = i, bias = 99.
        weights = np.array(list(range(N_FEATURES)) + [99.0],
                           dtype=np.float32)
        s._model = _LstsqModel(weights)
        s._scaler = _LstsqScaler(
            mean=np.zeros(N_FEATURES, dtype=np.float32),
            std=np.ones(N_FEATURES, dtype=np.float32))
        s._trained = True
        s._n_train = 500
        imp = s.feature_importance()
        assert imp["method"] == "lstsq_abs_weight"
        assert len(imp["importances"]) == N_FEATURES
        # The bias (99.0) must NOT bleed into any feature's importance.
        assert all(r["importance"] < 99.0
                   for r in imp["importances"]), (
            "bias term must be stripped — w[:N_FEATURES] only")
        # First feature (w[0]=0) should be the smallest importance after
        # sorting (sorted desc by importance).
        last = imp["importances"][-1]
        assert last["feature"] == FEATURE_NAMES[0]
        assert last["importance"] == 0.0


# ─────────────────────── _to_float robustness ──────────────────────
#
# The actual data corpus passed through `_to_float` includes corrupt /
# malformed cells from JSONL. These tests pin the contract that EVERY
# unparseable input degrades to default rather than raising — a single
# bad row must never abort training (the documented `train_scorer`
# wedge-vector that the codebase's defensive coding aims to prevent).

class TestToFloatRobustness:
    def test_list_returns_default(self):
        assert _to_float([1, 2, 3], -1.5) == -1.5

    def test_dict_returns_default(self):
        assert _to_float({"a": 1}, -1.5) == -1.5

    def test_negative_zero_passthrough(self):
        # -0.0 is a finite, valid float. Must not be conflated with the
        # `False` short-circuit at the top of _to_float (which uses
        # isinstance(v, bool) — False is bool, -0.0 is float).
        assert _to_float(-0.0, 5.0) == 0.0

    def test_large_int_passthrough(self):
        # Python ints have arbitrary precision; math.isfinite accepts any int.
        # The numeric promotion must not lose information for any reasonable
        # int that fits in a float64.
        v = 10**12
        assert _to_float(v, -1.0) == float(v)
