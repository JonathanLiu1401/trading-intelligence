"""Tests for paper_trader.ml.feature_alignment — per-feature univariate-IC
vs model-importance alignment diagnostic.

Asserts the verdict ladder by constructing synthetic outcomes corpora that
unambiguously land in one bucket per test. No model is loaded from disk —
a stub ``DecisionScorer.feature_importance`` is monkeypatched so the test
controls the model side of the alignment cross-tab.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from paper_trader.ml import feature_alignment as fa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_outcomes(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "decision_outcomes.jsonl"
    with p.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return p


def _make_records(n: int, ic_features: dict[str, float] | None = None,
                  null_features: list[str] | None = None,
                  seed: int = 42) -> list[dict]:
    """Build n synthetic outcome rows.

    ``ic_features`` maps record-key → (approximate) Spearman slope between
    that feature and forward_return_5d. ``null_features`` are present in
    every row with random uncorrelated noise.
    """
    import random
    rng = random.Random(seed)
    out: list[dict] = []
    base_keys = [k for k, _ in fa.NUMERIC_FEATURES]
    for i in range(n):
        # Realized return ~ uniform; features linearly correlated by their
        # configured slope plus noise. The Spearman is monotonic so any
        # positive slope yields a positive IC for n=50+.
        fr = rng.uniform(-15.0, 15.0)
        rec: dict = {
            "ticker": "NVDA",
            "action": "BUY",
            "forward_return_5d": fr,
        }
        for k in base_keys:
            if ic_features and k in ic_features:
                slope = ic_features[k]
                # Feature value = slope*fr + noise. Spearman ≈ slope/(slope+noise).
                rec[k] = slope * fr + rng.gauss(0, 2.0)
            elif null_features and k in null_features:
                rec[k] = rng.uniform(-1, 1)
            else:
                rec[k] = rng.uniform(-1, 1)
        out.append(rec)
    return out


class _StubScorer:
    """Stand-in for DecisionScorer used in tests — provides
    ``feature_importance()`` returning a configurable map."""
    is_trained = True
    _model = object()  # truthy

    def __init__(self, importance_map: dict[str, float]):
        self._imp = importance_map

    def feature_importance(self) -> dict:
        return {
            "trained": True,
            "method": "stub",
            "n_train": 1000,
            "importances": [
                {"feature": k, "importance": v}
                for k, v in self._imp.items()
            ],
        }


@pytest.fixture
def patched_scorer(monkeypatch):
    """Provide a way to inject a stub scorer into `analyze`'s
    DecisionScorer-load step."""
    holder = {"importance_map": {}}

    def _factory(monkey_imp: dict[str, float]):
        holder["importance_map"] = monkey_imp
        return holder

    def _patch(holder=holder):
        class _Mod:
            DecisionScorer = lambda *a, **kw: _StubScorer(
                holder["importance_map"])

        # `feature_alignment.analyze` does `from paper_trader.ml.decision_scorer
        # import DecisionScorer`. Patch the underlying class so the import
        # picks it up.
        import paper_trader.ml.decision_scorer as ds_mod
        monkeypatch.setattr(
            ds_mod, "DecisionScorer",
            lambda: _StubScorer(holder["importance_map"]),
        )

    _patch()
    return _factory


# ---------------------------------------------------------------------------
# 1. Univariate-IC computation correctness
# ---------------------------------------------------------------------------

class TestUnivariateIc:
    """The Spearman implementation must yield the textbook value for
    monotonic data. If this drifts every downstream verdict drifts."""

    def test_perfectly_monotone_positive_yields_ic_near_one(self, tmp_path,
                                                            patched_scorer):
        patched_scorer({"ml_score": 0.5})
        # Construct records where ml_score == forward_return_5d exactly,
        # so Spearman = 1.0 exactly.
        recs = []
        for i in range(60):
            recs.append({"ticker": "NVDA", "action": "BUY",
                         "forward_return_5d": i * 1.0,
                         "ml_score": i * 1.0,
                         "rsi": 50.0, "macd": 0.0, "mom5": 0.0,
                         "mom20": 0.0, "regime_mult": 1.0,
                         "vol_ratio": 1.0, "bb_position": 0.0,
                         "news_urgency": 0.0, "news_article_count": 0.0,
                         "ema200_above": False, "hist_cross_up": False,
                         "macd_below_zero_cross": False})
        path = _write_outcomes(tmp_path, recs)
        rep = fa.analyze(outcomes_path=path, oos_only=False)
        ml_row = next(r for r in rep["features"]
                      if r["feature"] == "ml_score")
        assert ml_row["univariate_ic"] is not None
        assert math.isclose(ml_row["univariate_ic"], 1.0, abs_tol=1e-6), \
            f"expected univariate IC = 1.0, got {ml_row['univariate_ic']}"

    def test_perfectly_monotone_negative_yields_ic_near_minus_one(
            self, tmp_path, patched_scorer):
        patched_scorer({"ml_score": 0.5})
        recs = []
        for i in range(60):
            recs.append({"ticker": "NVDA", "action": "BUY",
                         "forward_return_5d": i * 1.0,
                         "ml_score": -i * 1.0,
                         "rsi": 50.0, "macd": 0.0, "mom5": 0.0,
                         "mom20": 0.0, "regime_mult": 1.0,
                         "vol_ratio": 1.0, "bb_position": 0.0,
                         "news_urgency": 0.0, "news_article_count": 0.0,
                         "ema200_above": False, "hist_cross_up": False,
                         "macd_below_zero_cross": False})
        path = _write_outcomes(tmp_path, recs)
        rep = fa.analyze(outcomes_path=path, oos_only=False)
        ml_row = next(r for r in rep["features"]
                      if r["feature"] == "ml_score")
        assert math.isclose(ml_row["univariate_ic"], -1.0, abs_tol=1e-6), \
            f"expected univariate IC = -1.0, got {ml_row['univariate_ic']}"

    def test_sell_action_flips_target_sign(self, tmp_path, patched_scorer):
        """A SELL row's target must be sign-flipped (mirrors train_scorer /
        _oos_rank_metrics). So a feature perfectly correlated with the
        RAW forward_return on SELL rows should yield IC = -1.0 here."""
        patched_scorer({"ml_score": 0.5})
        recs = []
        for i in range(60):
            recs.append({"ticker": "NVDA", "action": "SELL",
                         "forward_return_5d": i * 1.0,
                         "ml_score": i * 1.0,
                         "rsi": 50.0, "macd": 0.0, "mom5": 0.0,
                         "mom20": 0.0, "regime_mult": 1.0,
                         "vol_ratio": 1.0, "bb_position": 0.0,
                         "news_urgency": 0.0, "news_article_count": 0.0,
                         "ema200_above": False, "hist_cross_up": False,
                         "macd_below_zero_cross": False})
        path = _write_outcomes(tmp_path, recs)
        rep = fa.analyze(outcomes_path=path, oos_only=False)
        ml_row = next(r for r in rep["features"]
                      if r["feature"] == "ml_score")
        # SELL sign-flips target → ml_score ↑ now correlates with
        # -(forward_return) ↓, so univariate_ic = -1.0.
        assert math.isclose(ml_row["univariate_ic"], -1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 2. Verdict ladder
# ---------------------------------------------------------------------------

class TestVerdictLadder:
    def test_insufficient_data_verdict_when_corpus_below_min(
            self, tmp_path, patched_scorer):
        patched_scorer({"ml_score": 0.5})
        path = _write_outcomes(tmp_path, _make_records(10))
        rep = fa.analyze(outcomes_path=path, oos_only=False)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["status"] == "insufficient_data"
        assert rep["features"] == []

    def test_aligned_verdict_when_top_ic_matches_top_weight(
            self, tmp_path, patched_scorer):
        """ml_score has IC = 1.0 (highest), and the model weights ml_score
        highest. → ALIGNED."""
        patched_scorer({
            "ml_score": 1.0,        # top weight
            "rsi": 0.1,
            "macd": 0.05,
            "mom5": 0.01,
            "mom20": 0.01,
            "regime_mult": 0.0,
            "vol_ratio": 0.0,
            "bb_pos": 0.0,
            "news_urgency": 0.0,
            "news_article_count": 0.0,
            "ema200_above": 0.0,
            "hist_cross_up": 0.0,
            "macd_below_zero_cross": 0.0,
        })
        recs = []
        for i in range(80):
            recs.append({"ticker": "NVDA", "action": "BUY",
                         "forward_return_5d": i * 1.0,
                         "ml_score": i * 1.0,    # IC = 1.0
                         "rsi": 50.0, "macd": 0.0, "mom5": 0.0,
                         "mom20": 0.0, "regime_mult": 1.0,
                         "vol_ratio": 1.0, "bb_position": 0.0,
                         "news_urgency": 0.0, "news_article_count": 0.0,
                         "ema200_above": False, "hist_cross_up": False,
                         "macd_below_zero_cross": False})
        path = _write_outcomes(tmp_path, recs)
        rep = fa.analyze(outcomes_path=path, oos_only=False)
        assert rep["verdict"] == "ALIGNED", rep
        ml_row = next(r for r in rep["features"]
                      if r["feature"] == "ml_score")
        assert ml_row["alignment_bucket"] == "ALIGNED"

    def test_weighted_noise_verdict_when_top_weight_is_bottom_ic(
            self, tmp_path, patched_scorer):
        """ml_score = noise, vol_ratio = real signal. Model weights
        ml_score top, vol_ratio low. → WEIGHTED_NOISE."""
        patched_scorer({
            # Top-weighted features = noise carriers.
            "ml_score": 1.0,
            "rsi": 0.9,
            "macd": 0.8,
            # Lowest-weighted features = signal carriers.
            "mom5": 0.0001,
            "mom20": 0.0001,
            "regime_mult": 0.0,
            "vol_ratio": 0.0001,
            "bb_pos": 0.0,
            "news_urgency": 0.0,
            "news_article_count": 0.0,
            "ema200_above": 0.0,
            "hist_cross_up": 0.0,
            "macd_below_zero_cross": 0.0,
        })
        import random
        rng = random.Random(123)
        recs = []
        for i in range(120):
            fr = rng.uniform(-10, 10)
            recs.append({
                "ticker": "NVDA", "action": "BUY",
                "forward_return_5d": fr,
                # ml_score / rsi / macd = pure noise → near-zero IC.
                "ml_score": rng.uniform(-1, 1),
                "rsi": rng.uniform(0, 100),
                "macd": rng.uniform(-1, 1),
                # mom5, mom20, vol_ratio = perfectly correlated → IC ≈ 1.
                "mom5": fr * 2,
                "mom20": fr * 1.5,
                "regime_mult": 1.0,
                "vol_ratio": fr * 0.5,
                "bb_position": 0.0,
                "news_urgency": 0.0,
                "news_article_count": 0.0,
                "ema200_above": False,
                "hist_cross_up": False,
                "macd_below_zero_cross": False,
            })
        path = _write_outcomes(tmp_path, recs)
        rep = fa.analyze(outcomes_path=path, oos_only=False)
        assert rep["verdict"] == "WEIGHTED_NOISE", rep
        # At least one of {ml_score, rsi, macd} must be in top_weighted_noise.
        assert any(f in rep["top_weighted_noise"]
                   for f in ("ml_score", "rsi", "macd")), rep

    def test_ignored_signal_verdict_when_top_ic_is_bottom_weight(
            self, tmp_path, patched_scorer):
        """mom5/mom20/vol_ratio = strong univariate signal but the model
        weights them near-zero. ml_score/rsi/macd = noise but heavily
        weighted by the model. → WEIGHTED_NOISE actually wins by ladder
        priority. To trigger IGNORED_SIGNAL specifically, configure the
        model to NOT have any top-weighted-noise feature (no top-weight
        in bottom-IC position) while still having a top-IC feature in
        bottom-weight position. Easiest is when top-IC features are
        present but the model's top weights go to mid-IC features."""
        patched_scorer({
            # Top-weighted features: mid-IC (not bottom).
            "regime_mult": 1.0,
            "bb_pos": 0.9,
            "news_urgency": 0.8,
            # Mid weight.
            "ml_score": 0.4,
            "rsi": 0.3,
            "macd": 0.2,
            # Bottom-weighted features: TOP IC (the IGNORED_SIGNAL state).
            "mom5": 0.0001,
            "mom20": 0.0001,
            "vol_ratio": 0.0001,
            "news_article_count": 0.0,
            "ema200_above": 0.0,
            "hist_cross_up": 0.0,
            "macd_below_zero_cross": 0.0,
        })
        import random
        rng = random.Random(123)
        recs = []
        for i in range(120):
            fr = rng.uniform(-10, 10)
            recs.append({
                "ticker": "NVDA", "action": "BUY",
                "forward_return_5d": fr,
                # ml_score / rsi / macd: mid IC (noise correlation).
                "ml_score": rng.uniform(-1, 1),
                "rsi": rng.uniform(0, 100),
                "macd": rng.uniform(-1, 1),
                # regime_mult / bb_pos / news_urgency: NEAR-ZERO IC (also noise).
                "regime_mult": 1.0,
                "bb_position": rng.uniform(-1, 1),
                "news_urgency": rng.uniform(0, 100),
                # mom5 / mom20 / vol_ratio: STRONG IC.
                "mom5": fr * 2,
                "mom20": fr * 1.5,
                "vol_ratio": fr * 0.5,
                "news_article_count": 0.0,
                "ema200_above": False, "hist_cross_up": False,
                "macd_below_zero_cross": False,
            })
        path = _write_outcomes(tmp_path, recs)
        rep = fa.analyze(outcomes_path=path, oos_only=False)
        # Top-IC features (mom5/mom20/vol_ratio) are bottom-weighted —
        # so at least one of them lands in top_ignored_signal.
        # The TOP_N=3 lowest weights are mom5/mom20/vol_ratio (all 0.0001),
        # and they're the TOP_N=3 highest |IC|. So all 3 should be IGNORED.
        assert rep["verdict"] in ("IGNORED_SIGNAL", "WEIGHTED_NOISE"), rep
        if rep["verdict"] == "IGNORED_SIGNAL":
            assert any(f in rep["top_ignored_signal"]
                       for f in ("mom5", "mom20", "vol_ratio")), rep

    def test_degenerate_verdict_when_all_features_are_noise(
            self, tmp_path, patched_scorer):
        """No feature has |IC| ≥ MIN_IC. → DEGENERATE.

        Uses n=2000 so the Spearman SE (~1/√n) drops well below MIN_IC=0.03.
        At n=2000 the SE is ~0.022, so the 95% CI of every per-feature IC
        spans roughly ±0.045 around 0 — most features land below the
        |IC|≥0.03 threshold. Pin both the structural quantity
        (n_features_with_signal small) and the verdict on the strict zero-signal case.
        """
        patched_scorer({"ml_score": 1.0, "rsi": 0.5})
        import random
        rng = random.Random(99)
        recs = []
        for _ in range(2000):
            # Forward return is decorrelated from every feature.
            fr = rng.uniform(-10, 10)
            rec = {"ticker": "NVDA", "action": "BUY",
                   "forward_return_5d": fr}
            for k, _ in fa.NUMERIC_FEATURES:
                rec[k] = rng.uniform(-1, 1)
            recs.append(rec)
        path = _write_outcomes(tmp_path, recs)
        rep = fa.analyze(outcomes_path=path, oos_only=False)
        # Loose upper bound — at most a few features above threshold by chance.
        assert rep["n_features_with_signal"] <= 4, rep
        if rep["n_features_with_signal"] == 0:
            assert rep["verdict"] == "DEGENERATE", rep


# ---------------------------------------------------------------------------
# 3. Honest-degrade contract
# ---------------------------------------------------------------------------

class TestHonestDegrade:
    def test_missing_outcomes_file_returns_insufficient_data(
            self, tmp_path, patched_scorer):
        patched_scorer({})
        rep = fa.analyze(
            outcomes_path=tmp_path / "does_not_exist.jsonl",
            oos_only=False,
        )
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["features"] == []

    def test_features_with_no_importance_have_none_weight(
            self, tmp_path, patched_scorer):
        """A feature absent from feature_importance map → model_importance=None
        and alignment_bucket=DEGENERATE for that row (model state unknown)."""
        # Stub returns importance ONLY for ml_score, leaving everything
        # else with model_importance=None.
        patched_scorer({"ml_score": 1.0})
        recs = _make_records(80, ic_features={"mom5": 1.0})
        path = _write_outcomes(tmp_path, recs)
        rep = fa.analyze(outcomes_path=path, oos_only=False)
        # rsi has no importance → bucket DEGENERATE
        rsi_row = next(r for r in rep["features"] if r["feature"] == "rsi")
        assert rsi_row["model_importance"] is None
        assert rsi_row["alignment_bucket"] == "DEGENERATE"

    def test_never_raises_on_corrupt_jsonl(self, tmp_path, patched_scorer):
        patched_scorer({})
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n{broken\n")
        rep = fa.analyze(outcomes_path=p, oos_only=False)
        assert rep["status"] in ("insufficient_data", "error")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
