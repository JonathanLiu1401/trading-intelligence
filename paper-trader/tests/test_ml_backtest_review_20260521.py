"""Targeted ML/backtest regression tests added 2026-05-21.

These tests pin specific contracts and edge cases discovered during the
HYBRID quant-researcher review pass:

1. The `\\bscore=` / `\\bnews_urg=` / `\\bnews_count=` word-boundary anchoring
   in `_compute_decision_outcomes` — a naive `re.search("score=…")` matches
   the substring inside `underscore=` / `kw_score=` / any future longer
   token, silently poisoning the DecisionScorer's `ml_score` feature.
2. `predict_with_meta` non-finite handling — `pred` must always be finite
   and equal to 0.0 (the documented safe-fallback contract every honesty
   panel reads), regardless of how the raw model output diverged.
3. `build_features` clamp ranges for the news features — out-of-band inputs
   must not extend the feature vector past the training manifold.
4. `_compute_decision_outcomes` SELL conviction = None — the gate is
   BUY-only so SELL/HOLD rows must never carry a conviction_pct token.
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import run_continuous_backtests as rcb
from paper_trader.ml.decision_scorer import (
    DecisionScorer,
    build_features,
    PRED_CLAMP_PCT,
    SECTORS,
    N_FEATURES,
)


# ──────────────── score=/news_urg=/news_count= word boundary ───────────────


def _make_engine_with_synthetic_prices(tmp_path, synthetic_prices):
    """Helper: minimal engine with synthetic price cache and an empty
    backtest store ready for INSERTs."""
    from paper_trader.backtest import BacktestStore

    db_path = tmp_path / "bt.db"
    store = BacktestStore(path=db_path)
    start = synthetic_prices.trading_days[0]
    end = synthetic_prices.trading_days[-1]
    store.upsert_run(1, seed=1, status="complete", start=start, end=end)
    engine = MagicMock()
    engine.store = store
    engine.prices = synthetic_prices
    return engine


class TestScoreWordBoundary:
    """`_compute_decision_outcomes` must not pick `score=` from inside a
    longer identifier — `\\bscore=` mirrors the existing `\\bscorer=` /
    `\\bconviction=` discipline in `_parse_gate_decision` /
    `_parse_conviction_pct`. Without `\\b`, a `re.search` on
    ``"underscore=999 score=1.5"`` returns 999 (first match lives INSIDE
    `underscore=`), feeding a wrong ml_score into the training feature.
    """

    def test_score_not_matched_inside_underscore(
        self, tmp_path, synthetic_prices
    ):
        engine = _make_engine_with_synthetic_prices(tmp_path, synthetic_prices)
        day0 = synthetic_prices.trading_days[0].isoformat()
        # Adversarial reasoning: leading `underscore=999` BEFORE the real
        # `score=2.5`. The `\b` anchor must reject the embedded match.
        engine.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "status, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
            (1, day0, "BUY", "NVDA", "FILLED",
             "underscore=999 score=2.5 regime=bull "
             "news_count=3 news_urg=80.0"),
        )
        engine.store.conn.commit()
        from paper_trader.backtest import BacktestRun
        runs = [BacktestRun(run_id=1, seed=1,
                            start_date=synthetic_prices.trading_days[0].isoformat(),
                            end_date=synthetic_prices.trading_days[-1].isoformat())]
        outs = rcb._compute_decision_outcomes(engine, runs)
        assert len(outs) == 1
        # The fix returns 2.5, not 999. Without the \b anchor it would be 999.
        assert outs[0]["ml_score"] == pytest.approx(2.5)

    def test_score_extracted_from_typical_reasoning(
        self, tmp_path, synthetic_prices
    ):
        engine = _make_engine_with_synthetic_prices(tmp_path, synthetic_prices)
        day0 = synthetic_prices.trading_days[0].isoformat()
        engine.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "status, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
            (1, day0, "BUY", "NVDA", "FILLED",
             "ML+quant: NVDA score=3.50 regime=bull "
             "news_count=5 news_urg=70.0 conviction=25% scorer=+1.5%"),
        )
        engine.store.conn.commit()
        from paper_trader.backtest import BacktestRun
        runs = [BacktestRun(run_id=1, seed=1,
                            start_date=synthetic_prices.trading_days[0].isoformat(),
                            end_date=synthetic_prices.trading_days[-1].isoformat())]
        outs = rcb._compute_decision_outcomes(engine, runs)
        assert len(outs) == 1
        # Real-world unchanged: ml_score=3.5, news_urg=70, news_count=5.
        assert outs[0]["ml_score"] == pytest.approx(3.50)
        assert outs[0]["news_urgency"] == pytest.approx(70.0)
        assert outs[0]["news_article_count"] == pytest.approx(5.0)

    def test_negative_score_value_extracted(
        self, tmp_path, synthetic_prices
    ):
        # `score=-1.50` — the regex captures the negative sign correctly.
        engine = _make_engine_with_synthetic_prices(tmp_path, synthetic_prices)
        day0 = synthetic_prices.trading_days[0].isoformat()
        engine.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "status, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
            (1, day0, "SELL", "NVDA", "FILLED",
             "ML+quant: NVDA score=-1.50 regime=bear "
             "news_count=0 news_urg=0.0"),
        )
        engine.store.conn.commit()
        from paper_trader.backtest import BacktestRun
        runs = [BacktestRun(run_id=1, seed=1,
                            start_date=synthetic_prices.trading_days[0].isoformat(),
                            end_date=synthetic_prices.trading_days[-1].isoformat())]
        outs = rcb._compute_decision_outcomes(engine, runs)
        assert len(outs) == 1
        assert outs[0]["ml_score"] == pytest.approx(-1.50)


# ──────────────────── predict_with_meta contract ───────────────────────────


class _ScorerReturning:
    """Test double: a trained DecisionScorer whose model.predict returns
    a fixed value (or array)."""

    def __init__(self, value):
        self._value = value

    def is_trained_set(self):
        return True

    def __call__(self, X):
        return np.asarray([self._value] * len(X))


def _trained_scorer_returning(value: float) -> DecisionScorer:
    """Build a DecisionScorer wired to return a fixed scalar from its
    model. The scaler is identity (no transformation)."""
    s = DecisionScorer()
    s._trained = True

    class _M:
        def predict(self, X):
            return np.asarray([value] * len(X), dtype=np.float64)

    class _S:
        def transform(self, X):
            return np.asarray(X, dtype=np.float32)

    s._model = _M()
    s._scaler = _S()
    s._n_train = 1000
    return s


class TestPredictWithMetaContract:
    """The `pred` field must always be finite — every honesty panel reads
    it as a confident in-distribution call when it's a real float, so a
    leaked NaN would render as a fake confident 0.0 alongside an honest
    in-distribution prediction.
    """

    def test_pred_always_finite(self):
        for bad in (float("inf"), float("-inf"), float("nan")):
            s = _trained_scorer_returning(bad)
            m = s.predict_with_meta(
                ml_score=0.0, rsi=50, macd=0.0, mom5=0.0,
                mom20=0.0, regime_mult=1.0, ticker="NVDA",
            )
            assert math.isfinite(m["pred"]), \
                f"pred non-finite for raw={bad}: {m}"
            assert m["pred"] == 0.0
            assert m["off_distribution"] is True

    def test_clamp_at_boundary_50(self):
        # +50 exactly → not clamped, +50.0001 → clamped.
        s = _trained_scorer_returning(50.0)
        m = s.predict_with_meta(
            ml_score=0.0, rsi=50, macd=0.0, mom5=0.0,
            mom20=0.0, regime_mult=1.0, ticker="NVDA",
        )
        assert m["pred"] == pytest.approx(50.0)
        assert m["clamped"] is False
        assert m["off_distribution"] is False

        s2 = _trained_scorer_returning(50.0001)
        m2 = s2.predict_with_meta(
            ml_score=0.0, rsi=50, macd=0.0, mom5=0.0,
            mom20=0.0, regime_mult=1.0, ticker="NVDA",
        )
        assert m2["pred"] == pytest.approx(50.0)
        assert m2["clamped"] is True
        assert m2["off_distribution"] is True

    def test_negative_clamp_at_minus_50(self):
        s = _trained_scorer_returning(-89.0)
        m = s.predict_with_meta(
            ml_score=0.0, rsi=50, macd=0.0, mom5=0.0,
            mom20=0.0, regime_mult=1.0, ticker="NVDA",
        )
        assert m["pred"] == pytest.approx(-50.0)
        assert m["raw"] == pytest.approx(-89.0)
        assert m["clamped"] is True


# ──────────────────── build_features clamp ranges ─────────────────────────


class TestBuildFeaturesClamps:
    """Out-of-band feature values must NOT extend the training manifold —
    each numeric input has documented clamp bounds the trained model has
    never seen exceeded. Inference-time clamps mirror the bounds the model
    was trained against."""

    def test_vol_ratio_clamped_to_5(self):
        feats = build_features(
            ml_score=1.0, rsi=50, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA", vol_ratio=999.0,
        )
        # index 6 = vol_ratio (per FEATURE_NAMES order)
        assert feats[6] == 5.0

    def test_vol_ratio_clamped_to_zero(self):
        feats = build_features(
            ml_score=1.0, rsi=50, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA", vol_ratio=-3.0,
        )
        assert feats[6] == 0.0

    def test_bb_pos_clamped_to_band(self):
        feats_hi = build_features(
            ml_score=1.0, rsi=50, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA", bb_pos=10.0,
        )
        feats_lo = build_features(
            ml_score=1.0, rsi=50, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA", bb_pos=-10.0,
        )
        # index 7 = bb_pos
        assert feats_hi[7] == 2.0
        assert feats_lo[7] == -2.0

    def test_news_urgency_clamped_to_band(self):
        feats_hi = build_features(
            ml_score=1.0, rsi=50, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA", news_urgency=500.0,
        )
        feats_lo = build_features(
            ml_score=1.0, rsi=50, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA", news_urgency=-50.0,
        )
        # index 8 = news_urgency
        assert feats_hi[8] == 100.0
        assert feats_lo[8] == 0.0

    def test_news_article_count_clamped(self):
        feats_hi = build_features(
            ml_score=1.0, rsi=50, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA", news_article_count=200.0,
        )
        feats_lo = build_features(
            ml_score=1.0, rsi=50, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA", news_article_count=-5.0,
        )
        # index 9 = news_article_count
        assert feats_hi[9] == 20.0
        assert feats_lo[9] == 0.0

    def test_feature_vector_length_invariant(self):
        # The full vector must always be N_FEATURES long, regardless of
        # which optional inputs are None.
        feats_none = build_features(
            ml_score=0.0, rsi=None, macd=None, mom5=None, mom20=None,
            regime_mult=1.0, ticker="UNKNOWN_TICKER",
        )
        feats_full = build_features(
            ml_score=5.0, rsi=55, macd=0.5, mom5=2.0, mom20=4.0,
            regime_mult=1.0, ticker="NVDA",
            vol_ratio=1.5, bb_pos=0.5, news_urgency=80.0,
            news_article_count=4.0,
        )
        assert len(feats_none) == N_FEATURES
        assert len(feats_full) == N_FEATURES

    def test_unknown_ticker_goes_to_sector_other(self):
        feats = build_features(
            ml_score=0.0, rsi=50, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="UNKNOWN_PSEUDO_TICKER_XYZ",
        )
        # The last 7 slots are the sector one-hot in SECTORS order.
        sector_block = feats[-len(SECTORS):]
        other_idx = SECTORS.index("other")
        assert sector_block[other_idx] == 1.0
        # Exactly one slot active.
        assert sum(sector_block) == 1.0


# ──────────────── SELL/HOLD has no conviction (BUY-only token) ─────────────


class TestSellConvictionAlwaysNone:
    """`_parse_conviction_pct` must return None for SELL and HOLD reasoning.
    The `conviction=` token is only ever emitted by `_ml_decide`'s BUY path —
    a SELL or HOLD row with a parsed conviction would be a leak from a
    bug-introduced cross-action emission. Mirrors the `gate_scorer_pred`
    SELL convention."""

    def test_sell_reasoning_no_conviction(self):
        sell_reasoning = (
            "ML+quant: NVDA score=-1.50 regime=bear "
            "RSI=85 news_count=3 news_urg=80.0 — reducing"
        )
        assert rcb._parse_conviction_pct(sell_reasoning) is None

    def test_hold_reasoning_no_conviction(self):
        hold_reasoning = (
            "ML+quant: no high-conviction signal 2025-05-15 regime=sideways"
        )
        assert rcb._parse_conviction_pct(hold_reasoning) is None


# ──────────────── _parse_gate_decision robustness ─────────────────────────


class TestParseGateDecisionRobustness:
    """Defensive parsing — a malformed `scorer=…%` token must degrade to
    (None, None) rather than raise. `_parse_gate_decision` is consumed by
    the per-cycle ledger that must NEVER break the loop."""

    def test_empty_reasoning(self):
        assert rcb._parse_gate_decision("") == (None, None)
        assert rcb._parse_gate_decision(None) == (None, None)

    def test_scorer_without_percent_sign_no_match(self):
        # Regex requires a trailing `%` — a `scorer=5.0` (no percent) does
        # not match (defensive: future emission without percent would have
        # different semantics and we don't want a silent partial parse).
        assert rcb._parse_gate_decision("scorer=5.0") == (None, None)

    def test_scorer_with_off_dist_token(self):
        pred, off = rcb._parse_gate_decision(
            "ML+quant: NVDA score=2.5 conviction=25% "
            "scorer=-50.0%(off-dist,gate-skipped)"
        )
        assert pred == pytest.approx(-50.0)
        assert off is True

    def test_scorer_normal_inline(self):
        pred, off = rcb._parse_gate_decision(
            "ML+quant: NVDA score=2.5 conviction=30% scorer=+12.3%"
        )
        assert pred == pytest.approx(12.3)
        assert off is False


# ──────────────── DecisionScorer load cache freshness ────────────────────


class TestScorerLoadCacheFreshness:
    """The process-wide `_LOAD_CACHE` keys on (path, mtime_ns, size). A
    retrain that atomically writes a new pickle must invalidate the cached
    entry — otherwise the singleton-reset in `run_continuous_backtests`
    would still hand out the stale model.

    Uses the production `train_scorer` to produce real pickles so the
    test path matches the real retrain pipeline (lstsq fallback model
    can be pickled — both branches of `train_scorer` produce a
    `{model, scaler, n_train}` dict, which is what `_load` consumes).
    """

    def test_new_pickle_invalidates_cache(self, tmp_path):
        import paper_trader.ml.decision_scorer as ds

        # Build 30 synthetic records (`train_scorer` dedup gate floor) with
        # a different LABEL on each retrain so the model output differs.
        def _records(label_val: float):
            out = []
            # Unique (ticker, sim_date, action) per row — dedup keeps every row.
            from datetime import date as _d, timedelta as _td
            d0 = _d(2025, 1, 1)
            for i in range(40):
                out.append({
                    "ticker": "NVDA",
                    "sim_date": (d0 + _td(days=i)).isoformat(),
                    "action": "BUY",
                    "ml_score": float(i % 5),
                    "rsi": 50.0,
                    "macd": 0.0,
                    "mom5": 0.0,
                    "mom20": 0.0,
                    "regime_mult": 1.0,
                    "forward_return_5d": label_val + (i * 0.01),
                    "return_pct": 10.0,
                })
            return out

        # `train_scorer` writes to SCORER_PATH (the conftest already
        # monkeypatched it to tmp_path/data/ml/decision_scorer.pkl).
        ds._LOAD_CACHE.clear()

        # Retrain v1 — labels around +5.
        r1 = ds.train_scorer(_records(5.0))
        assert r1["status"] == "ok"
        s1 = ds.DecisionScorer()
        assert s1.is_trained
        pred1 = s1.predict(
            ml_score=2.0, rsi=50, macd=0.0, mom5=0.0,
            mom20=0.0, regime_mult=1.0, ticker="NVDA",
        )

        # Retrain v2 — labels around -10 (substantially different).
        # Force a different mtime by sleeping a hair past mtime_ns granularity.
        import time as _time
        _time.sleep(0.01)
        r2 = ds.train_scorer(_records(-10.0))
        assert r2["status"] == "ok"

        # New instance MUST observe the new pickle (cache invalidated).
        s2 = ds.DecisionScorer()
        pred2 = s2.predict(
            ml_score=2.0, rsi=50, macd=0.0, mom5=0.0,
            mom20=0.0, regime_mult=1.0, ticker="NVDA",
        )
        assert pred1 != pred2, (
            f"stale model leaked: v1 pred={pred1}, v2 pred={pred2} — "
            f"cache did not invalidate on pickle replace"
        )


# ──────────────── _to_float numpy/exotic input handling ────────────────


class TestToFloatEdgeCases:
    """`_to_float` is the safety guard around every external input to
    `build_features` — `predict_with_meta`, `train_scorer`, and the
    OOS metrics all rely on its (value, default) contract holding for
    every numeric type the JSONL outcomes file or numpy may produce."""

    def test_numpy_float32_finite_passes(self):
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(np.float32(3.14), 0.0) == pytest.approx(3.14, rel=1e-5)

    def test_numpy_float32_nan_falls_to_default(self):
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(np.float32("nan"), 99.0) == 99.0

    def test_numpy_int_finite_passes(self):
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(np.int64(7), 0.0) == 7.0

    def test_python_inf_falls_to_default(self):
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(float("inf"), 42.0) == 42.0
        assert _to_float(float("-inf"), 42.0) == 42.0

    def test_python_nan_falls_to_default(self):
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(float("nan"), 42.0) == 42.0

    def test_bool_falls_to_default(self):
        # bool is an int subclass — True/False must NOT be coerced to 1.0/0.0
        # (an outcome row with `forward_return_5d=true` is bad data).
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(True, 99.0) == 99.0
        assert _to_float(False, 99.0) == 99.0

    def test_string_falls_to_default(self):
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float("3.14", 0.0) == 0.0
        assert _to_float("abc", 0.0) == 0.0

    def test_none_falls_to_default(self):
        from paper_trader.ml.decision_scorer import _to_float
        assert _to_float(None, 0.0) == 0.0


# ──────────────── PriceCache returns_pct edge cases ────────────────────


class TestPriceCacheReturnsPctEdgeCases:
    """`returns_pct` is read by the SPY benchmark guard and by
    `_compute_decision_outcomes`. Both rely on documented edge behavior:
    a missing endpoint returns 0.0 (caller-side guards interpret that as
    `vs_spy_pct=fabricated`), NEVER a NaN or exception."""

    def test_returns_zero_when_ticker_absent(self, synthetic_prices):
        d0 = synthetic_prices.trading_days[0]
        d1 = synthetic_prices.trading_days[10]
        # MISSING_TICKER not in synthetic_prices.prices
        assert synthetic_prices.returns_pct("MISSING_TICKER", d0, d1) == 0.0

    def test_returns_correct_pct_for_known_curve(self, synthetic_prices):
        # SPY: 100 → 100 + i. Day 0 → Day 10 should be (110-100)/100*100 = 10%.
        d0 = synthetic_prices.trading_days[0]
        d10 = synthetic_prices.trading_days[10]
        ret = synthetic_prices.returns_pct("SPY", d0, d10)
        assert ret == pytest.approx(10.0, rel=1e-3)

    def test_resolved_close_date_returns_same_when_present(
        self, synthetic_prices
    ):
        d0 = synthetic_prices.trading_days[0]
        assert synthetic_prices.resolved_close_date("SPY", d0) == d0
