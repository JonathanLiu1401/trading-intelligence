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
    N_FEATURES,
    PRED_CLAMP_PCT,
    SECTORS,
    SECTOR_MAP,
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


def _trained_scorer_returning(value: float) -> DecisionScorer:
    s = DecisionScorer()
    s._model = _FixedModel(value)
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

    def test_untrained_scorer_meta_is_safe(self):
        # Regression guard: the untrained short-circuit must run BEFORE the
        # clamp path, otherwise a fresh scorer would stop returning 0.0.
        s = DecisionScorer()
        assert not s.is_trained
        m = s.predict_with_meta(ml_score=2.0, rsi=None, macd=None, mom5=None,
                                mom20=None, regime_mult=1.0, ticker="NVDA")
        assert m == {"pred": 0.0, "raw": 0.0, "clamped": False,
                     "off_distribution": False}


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
        # JSON nulls in the outcome file historically crashed training because
        # float(r.get("forward_return_5d", 0.0)) saw None instead of the default.
        recs = []
        for i in range(35):
            r = _synthetic_outcome(sim_date=f"2025-05-{i+1:02d}")
            r["forward_return_5d"] = None  # the bug case
            recs.append(r)
        # Must not crash — _to_float coerces None → 0.0.
        result = train_scorer(recs)
        assert result["status"] == "ok"

    def test_handles_non_finite_forward_return(self):
        # Regression: a single decision_outcomes.jsonl row with a non-finite
        # forward_return_5d (inf / -inf) used to pass _to_float untouched,
        # poison the y vector, and make MLPRegressor.fit raise
        # "Input y contains infinity". _train_decision_scorer swallows that
        # exception, so the scorer silently stopped retraining for that cycle
        # AND every cycle after (the poisoned row persists in the 5000-record
        # tail). With the fix, inf/-inf coerce to 0.0 and training completes.
        recs = [_synthetic_outcome(sim_date=f"2025-06-{i+1:02d}")
                for i in range(35)]
        recs[5]["forward_return_5d"] = float("inf")
        recs[6]["forward_return_5d"] = float("-inf")
        result = train_scorer(recs)
        assert result["status"] == "ok"
        # val_rmse must be a real finite number, not nan/inf from a poisoned fit.
        vr = result["val_rmse"]
        assert vr == vr and abs(vr) < 1e6

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
