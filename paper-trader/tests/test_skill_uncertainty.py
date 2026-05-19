"""Tests for `paper_trader.ml.skill_uncertainty` — bootstrap CIs on
the DecisionScorer's OOS rank-IC / RMSE / dir-acc.

Tests assert specific expected values (CIs straddle 0 vs exclude 0,
synthetic-signal vs pure-noise outcomes) — not just that the function
"runs without crashing". A skeptical quant cares that
``SKILL_DETECTED`` fires only when the rank-IC CI truly excludes 0.
"""
from __future__ import annotations

import numpy as np
import pytest

from paper_trader.ml import skill_uncertainty as su


# A minimal scorer-shaped stub that consumes the 11-kwarg `predict` path
# `_predict_oos_pairs` calls. The point of these tests is to exercise the
# bootstrap logic on KNOWN (pred, realized) distributions, not to also
# test the real MLP. Instead of building a feature vector and pushing it
# through `build_features` / `MLPRegressor`, the stub returns a stored
# prediction keyed off the test-supplied `ticker` field, so each test
# can engineer the exact (pred, realized) shape it needs.
class _StubScorer:
    is_trained = True
    _n_train = 999

    def __init__(self, predictions_by_ticker: dict[str, float]):
        self._preds = predictions_by_ticker

    def predict(self, *, ml_score, rsi, macd, mom5, mom20, regime_mult,
                ticker, vol_ratio=None, bb_pos=None, news_urgency=None,
                news_article_count=None) -> float:
        return float(self._preds.get(ticker, 0.0))


def _mk_records(preds: list[float], actuals: list[float],
                action: str = "BUY") -> list[dict]:
    """Build the minimal decision_outcomes.jsonl-shape rows that
    `_predict_oos_pairs` will read. The stub scorer reads `ticker`, so
    encode the per-row pred into the ticker name.
    """
    return [
        {
            "ticker": f"T{i}",
            "action": action,
            "ml_score": 1.0,
            "rsi": 50.0,
            "macd": 0.0,
            "mom5": 0.0,
            "mom20": 0.0,
            "regime_mult": 1.0,
            "forward_return_5d": a,
        }
        for i, a in enumerate(actuals)
    ]


def _build_stub(preds: list[float]) -> _StubScorer:
    return _StubScorer({f"T{i}": p for i, p in enumerate(preds)})


class TestVerdictThresholds:
    """The verdict is the load-bearing output — these tests pin every arm."""

    def test_not_trained_returns_not_trained_verdict(self):
        class _Untrained:
            is_trained = False
            _n_train = 0
        rep = su.bootstrap_skill_ci(_Untrained(), _mk_records([], [1.0]))
        assert rep["status"] == "not_trained"
        assert rep["verdict"] == "NOT_TRAINED"
        assert rep["rank_ic"]["point"] is None

    def test_insufficient_oos_returns_insufficient(self):
        # 5 pairs < MIN_OOS (30) → INSUFFICIENT_DATA
        preds = [0.1, 0.2, -0.1, 0.3, -0.2]
        actuals = [1.0, 2.0, -1.0, 3.0, -2.0]
        scorer = _build_stub(preds)
        rep = su.bootstrap_skill_ci(
            scorer, _mk_records(preds, actuals),
            n_bootstraps=50,
        )
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_oos"] == 5
        assert rep["rank_ic"]["point"] is None

    def test_strong_signal_yields_skill_detected(self):
        # Perfect rank correlation — CI MUST exclude 0.
        n = 60
        rng = np.random.default_rng(42)
        actuals = rng.normal(0.0, 5.0, size=n).tolist()
        # preds align exactly with actuals' rank → rank-IC ≈ 1.0
        preds = [float(x) for x in actuals]
        scorer = _build_stub(preds)
        rep = su.bootstrap_skill_ci(
            scorer, _mk_records(preds, actuals),
            n_bootstraps=500, seed=42,
        )
        assert rep["status"] == "ok"
        assert rep["verdict"] == "SKILL_DETECTED"
        # Sanity: point estimate ≈ 1.0, CI tight near 1.0
        assert rep["rank_ic"]["point"] > 0.95
        assert rep["rank_ic"]["ci_low"] > 0.5  # well above 0

    def test_pure_noise_does_not_detect_skill(self):
        # Independent random preds and actuals — CI must straddle 0.
        n = 60
        rng = np.random.default_rng(7)
        actuals = rng.normal(0.0, 5.0, size=n).tolist()
        preds = rng.normal(0.0, 5.0, size=n).tolist()
        scorer = _build_stub(preds)
        rep = su.bootstrap_skill_ci(
            scorer, _mk_records(preds, actuals),
            n_bootstraps=500, seed=42,
        )
        assert rep["status"] == "ok"
        assert rep["verdict"] == "NO_SKILL_DETECTED"
        ci_lo = rep["rank_ic"]["ci_low"]
        ci_hi = rep["rank_ic"]["ci_high"]
        # The CI must straddle 0 — that's the entire definition of
        # NO_SKILL_DETECTED in this branch.
        assert ci_lo <= 0.0 <= ci_hi

    def test_negatively_correlated_does_not_detect_positive_skill(self):
        # Anti-correlated preds and actuals — the verdict is NO_SKILL_DETECTED
        # because the rank-IC's CI doesn't exclude 0 from ABOVE (the gate
        # logic only earns positive rank skill credit; an anti-correlated
        # model is no more useful than a random one without an explicit flip).
        n = 60
        rng = np.random.default_rng(99)
        actuals = rng.normal(0.0, 5.0, size=n).tolist()
        preds = [-float(x) for x in actuals]
        scorer = _build_stub(preds)
        rep = su.bootstrap_skill_ci(
            scorer, _mk_records(preds, actuals),
            n_bootstraps=500, seed=42,
        )
        assert rep["status"] == "ok"
        assert rep["verdict"] == "NO_SKILL_DETECTED"
        # Point estimate should be strongly negative; CI strictly below 0.
        assert rep["rank_ic"]["point"] < -0.95
        assert rep["rank_ic"]["ci_high"] < 0.0  # CI fully below 0


class TestMetricsShape:
    def test_all_three_metrics_have_point_and_ci(self):
        # Mild positive signal → all three metrics should populate.
        n = 50
        rng = np.random.default_rng(3)
        actuals = rng.normal(0.0, 3.0, size=n).tolist()
        preds = [a * 0.5 + rng.normal(0.0, 1.0) for a in actuals]
        scorer = _build_stub(preds)
        rep = su.bootstrap_skill_ci(
            scorer, _mk_records(preds, actuals),
            n_bootstraps=200, seed=42,
        )
        for metric in ("rank_ic", "rmse", "dir_acc"):
            cell = rep[metric]
            assert cell["point"] is not None
            assert cell["ci_low"] is not None
            assert cell["ci_high"] is not None
            # CI is ordered low <= high
            assert cell["ci_low"] <= cell["ci_high"]

    def test_n_oos_counts_only_finite_rows(self):
        # Inject NaN forward_return_5d on two rows — they must be DROPPED
        # by `_predict_oos_pairs`, not coerced to 0.0 (which would
        # silently bias the stats).
        preds = [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0]
        actuals = [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0]
        recs = _mk_records(preds, actuals)
        # Pollute two rows.
        recs[1]["forward_return_5d"] = float("nan")
        recs[7]["forward_return_5d"] = None
        scorer = _build_stub(preds)
        # n=8 after dropping NaN+None → INSUFFICIENT_DATA (< 30).
        rep = su.bootstrap_skill_ci(scorer, recs, n_bootstraps=50)
        assert rep["n_oos"] == 8

    def test_dir_acc_ignores_zero_truth_pairs(self):
        # A pair where realized==0 carries no directional truth. Construct a
        # set where 30 pairs are unanimously correct and another 30 have
        # realized=0 (and the prediction is random). dir_acc must reflect
        # the 30 informative pairs, not 0.5 across all 60.
        n = 60
        preds = [1.0 if i < 30 else 1.0 for i in range(n)]
        actuals = [1.0 if i < 30 else 0.0 for i in range(n)]
        scorer = _build_stub(preds)
        rep = su.bootstrap_skill_ci(
            scorer, _mk_records(preds, actuals),
            n_bootstraps=200, seed=42,
        )
        # 30 informative pairs, all correct → dir_acc point = 1.0
        assert rep["dir_acc"]["point"] == pytest.approx(1.0, abs=1e-6)


class TestSellSignFlip:
    def test_sell_realized_target_sign_is_flipped(self):
        # A SELL whose realized return was NEGATIVE was a GOOD call. The
        # universal SELL sign-flip means the effective realized target is
        # +|return|. A scorer that predicts NEGATIVE for SELL signals
        # would then be ANTI-correlated with the flipped target → no
        # skill. Conversely, predicting POSITIVE for SELLs that fall
        # should DETECT skill.
        n = 50
        rng = np.random.default_rng(11)
        # 50 SELL rows: realized = small negatives (good SELL calls)
        actuals = rng.uniform(-5.0, -1.0, size=n).tolist()
        # Predict POSITIVE — the model says "good SELL". After flip,
        # realized becomes +1..+5 — correlated with predictions.
        preds = [-a + rng.normal(0.0, 0.5) for a in actuals]
        scorer = _build_stub(preds)
        rep = su.bootstrap_skill_ci(
            scorer, _mk_records(preds, actuals, action="SELL"),
            n_bootstraps=300, seed=42,
        )
        # Should detect skill because of the sign flip.
        assert rep["status"] == "ok"
        assert rep["verdict"] == "SKILL_DETECTED"


class TestReproducibility:
    def test_same_seed_yields_identical_ci(self):
        # The seed is part of the contract — same inputs + same seed must
        # yield byte-identical metrics. This is what lets a quant trend
        # the diagnostic across cycles without re-running flapping.
        n = 50
        rng = np.random.default_rng(0)
        actuals = rng.normal(0.0, 3.0, size=n).tolist()
        preds = [a * 0.3 + rng.normal(0.0, 1.0) for a in actuals]
        scorer = _build_stub(preds)
        rep1 = su.bootstrap_skill_ci(
            scorer, _mk_records(preds, actuals),
            n_bootstraps=100, seed=42,
        )
        rep2 = su.bootstrap_skill_ci(
            scorer, _mk_records(preds, actuals),
            n_bootstraps=100, seed=42,
        )
        # Every numeric field must match exactly under deterministic RNG.
        assert rep1["rank_ic"] == rep2["rank_ic"]
        assert rep1["rmse"] == rep2["rmse"]
        assert rep1["dir_acc"] == rep2["dir_acc"]

    def test_different_seeds_change_ci_band(self):
        # Sanity: changing the seed must perturb the CI band (otherwise
        # we're not actually resampling). Point estimates are sample-
        # deterministic; CI bounds are seed-dependent.
        n = 50
        rng = np.random.default_rng(0)
        actuals = rng.normal(0.0, 3.0, size=n).tolist()
        preds = [a * 0.3 + rng.normal(0.0, 1.0) for a in actuals]
        scorer = _build_stub(preds)
        rep1 = su.bootstrap_skill_ci(
            scorer, _mk_records(preds, actuals),
            n_bootstraps=100, seed=17,
        )
        rep2 = su.bootstrap_skill_ci(
            scorer, _mk_records(preds, actuals),
            n_bootstraps=100, seed=23,
        )
        # Point estimates IDENTICAL — not resampled.
        assert rep1["rank_ic"]["point"] == rep2["rank_ic"]["point"]
        # But the bootstrap CI must differ (different resamples).
        assert (rep1["rank_ic"]["ci_low"] != rep2["rank_ic"]["ci_low"]
                or rep1["rank_ic"]["ci_high"] != rep2["rank_ic"]["ci_high"])


class TestNeverRaises:
    def test_predict_exception_doesnt_kill_diagnostic(self):
        # A scorer that raises on every predict — _predict_oos_pairs
        # swallows it per-row, returning empty arrays → INSUFFICIENT_DATA.
        class _Broken:
            is_trained = True
            _n_train = 999
            def predict(self, **kw):
                raise RuntimeError("boom")
        scorer = _Broken()
        rep = su.bootstrap_skill_ci(
            scorer, _mk_records([1.0] * 50, [1.0] * 50),
            n_bootstraps=10,
        )
        # Crashed predicts produce zero pairs → insufficient.
        assert rep["status"] == "insufficient_data"
        assert rep["n_oos"] == 0

    def test_empty_records_returns_insufficient(self):
        scorer = _build_stub([])
        rep = su.bootstrap_skill_ci(scorer, [], n_bootstraps=10)
        assert rep["status"] == "insufficient_data"
        assert rep["n_oos"] == 0
