"""Exact-value locks for the permutation feature-importance diagnostic
(`paper_trader/ml/feature_importance.py`, 2026-05-18 quant feature).

Mirrors test_calibration.py / test_skill_trend.py / test_gate_audit.py:
deterministic synthetic data, exact metrics and exact verdicts (not ranges)
so a logic change must update the literals deliberately. All offline, no
network, no trained MLP.

The load-bearing assertions:
  * A feature ``predict`` provably ignores has EXACTLY 0.0 importance on all
    three metrics — proves the permutation isolates one feature and the real
    ``scorer.predict`` path is exercised per row.
  * The most-used feature drives ``top_feature`` and the verdict.
  * The sector one-hot is permuted JOINTLY via ``ticker`` (a sector-only
    model reads SECTOR_DOMINATED, not FLAT).
  * The universal SELL ``-forward_return_5d`` sign-flip is applied (without
    it the rank-IC flips sign — the regression lock).
  * ``oos_only`` restricts to the temporal-OOS slice.
  * Never raises: a raising / NaN / untrained scorer degrades to a verdict.
"""
from __future__ import annotations

import pytest

from paper_trader.ml import feature_importance as fi
from paper_trader.ml.decision_scorer import SECTOR_MAP


# ─────────────────────────── fake scorers ───────────────────────────

class _EchoScorer:
    """predict() returns ml_score — so only the ml_score feature matters and
    every other feature's permutation importance must be EXACTLY 0.0."""
    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        return float(kw["ml_score"])


class _SectorScorer:
    """predict() depends ONLY on the sector of ticker (via the real
    SECTOR_MAP). Permuting ticker scrambles the sector one-hot jointly;
    every numeric feature must read 0.0 importance."""
    is_trained = True
    n_train = 999
    _SV = {"tech": 10.0, "energy": -5.0, "financials": 3.0,
           "healthcare": 7.0, "commodities": -2.0, "crypto": 12.0,
           "other": 0.0}

    def predict(self, **kw) -> float:
        return self._SV[SECTOR_MAP.get(kw["ticker"], "other")]


class _ConstantScorer:
    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        return 3.0


class _RaisingScorer:
    is_trained = True
    n_train = 999

    def predict(self, **kw):
        raise ValueError("model/feature shape mismatch")


class _NaNScorer:
    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        return float("nan")


class _UntrainedScorer:
    is_trained = False
    n_train = 0

    def predict(self, **kw) -> float:
        return 0.0


# ─────────────────────────── record builders ───────────────────────────

# Distinct tech tickers (all map to "tech" via SECTOR_MAP) so the sector
# one-hot is constant in the echo test → sector importance is exactly 0.
_TECH = ["NVDA", "AMD", "MSFT", "AAPL"]
# One ticker per distinct sector for the sector-model test.
_PER_SECTOR = ["NVDA", "XOM", "JPM", "LLY", "GLD", "COIN"]


def _buy_records(n: int = 80, sector_cycle=_TECH):
    """ml_score == forward_return_5d == i, distinct per row; numeric features
    non-constant but irrelevant to the echo scorer."""
    recs = []
    for i in range(1, n + 1):
        recs.append({
            "ml_score": float(i),
            "rsi": float(30 + (i % 40)),
            "macd": float((-1) ** i) * 0.5,
            "mom5": float(i % 7) - 3.0,
            "mom20": float(i % 11) - 5.0,
            "regime_mult": [1.0, 0.6, 0.3][i % 3],
            "vol_ratio": 1.0 + (i % 5) * 0.1,
            "bb_position": float(i % 3) - 1.0,
            "news_urgency": float(i % 100),
            "news_article_count": float(i % 4),
            "ticker": sector_cycle[i % len(sector_cycle)],
            "action": "BUY",
            "forward_return_5d": float(i),
            "sim_date": f"2024-{1 + (i % 9):02d}-{1 + (i % 27):02d}",
        })
    return recs


# ─────────────────────────── tests ───────────────────────────

class TestIgnoredFeatureIsExactlyZero:
    """The strongest logic lock: a feature predict() ignores must move NONE
    of the three metrics — exactly 0.0, not 'small'."""

    def test_echo_scorer_only_ml_score_matters(self):
        rep = fi.feature_importance(_EchoScorer(), _buy_records(80),
                                    oos_only=False, n_repeats=3)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "SIGNAL_GROUNDED"
        assert rep["top_feature"] == "ml_score"
        # baseline: preds == actuals == i  → perfect.
        assert rep["baseline_rmse"] == 0.0
        assert rep["baseline_rank_ic"] == 1.0
        by = {f["feature"]: f for f in rep["features"]}
        # ml_score is used → strictly positive importance & material.
        assert by["ml_score"]["rmse_increase"] > 0.0
        assert by["ml_score"]["material"] is True
        # Every other feature is provably ignored → EXACTLY 0.0 everywhere.
        for name in ("rsi", "macd", "mom5", "mom20", "regime_mult",
                     "vol_ratio", "bb_position", "news_urgency",
                     "news_article_count", "sector"):
            assert by[name]["rmse_increase"] == 0.0, name
            assert by[name]["rank_ic_drop"] == 0.0, name
            assert by[name]["dir_acc_drop"] == 0.0, name
            assert by[name]["material"] is False, name


class TestSectorDominated:
    def test_sector_only_model_reads_sector_dominated(self):
        # forward_return == the sector's value so the sector model is perfect.
        recs = _buy_records(90, sector_cycle=_PER_SECTOR)
        sv = _SectorScorer._SV
        for r in recs:
            r["forward_return_5d"] = sv[SECTOR_MAP.get(r["ticker"], "other")]
        rep = fi.feature_importance(_SectorScorer(), recs,
                                    oos_only=False, n_repeats=3)
        assert rep["verdict"] == "SECTOR_DOMINATED"
        assert rep["top_feature"] == "sector"
        by = {f["feature"]: f for f in rep["features"]}
        assert by["sector"]["material"] is True
        assert by["sector"]["rmse_increase"] > 0.0
        # No quant feature is material — the model is sector memorization.
        for name in fi.QUANT_FEATURES:
            assert by[name]["material"] is False, name
            assert by[name]["rmse_increase"] == 0.0, name


class TestFlat:
    def test_constant_predictor_is_flat(self):
        rep = fi.feature_importance(_ConstantScorer(), _buy_records(80),
                                    oos_only=False, n_repeats=3)
        assert rep["verdict"] == "FLAT"
        # A constant predictor has zero rank skill (std==0 → spearman 0.0,
        # never a tie-ordering 1.0 artifact — the calibration._spearman lock).
        assert rep["baseline_rank_ic"] == 0.0
        for f in rep["features"]:
            assert f["rmse_increase"] == 0.0
            assert f["material"] is False


class TestSellSignFlip:
    """SELL realized goodness is -forward_return_5d. Construct a SELL-only
    set where higher ml_score precedes a MORE NEGATIVE raw return (a correct
    SELL). With the flip the echo model is perfectly rank-correlated (+1);
    without it it would read -1 → the regression lock."""

    def test_sell_flip_makes_rank_ic_positive(self):
        recs = _buy_records(80)
        for r in recs:
            r["action"] = "SELL"
            r["forward_return_5d"] = -r["ml_score"]   # correct SELL
        rep = fi.feature_importance(_EchoScorer(), recs,
                                    oos_only=False, n_repeats=3)
        # _realized flips SELL → a = -(-ml_score) = ml_score == pred.
        assert rep["baseline_rank_ic"] == 1.0
        assert rep["baseline_rmse"] == 0.0
        assert rep["verdict"] == "SIGNAL_GROUNDED"


class TestOosSliceRestriction:
    def test_oos_only_restricts_to_temporal_holdout(self):
        # 200 rows with strictly increasing sim_date → OOS = last 20% = 40.
        recs = []
        for i in range(1, 201):
            recs.append({
                "ml_score": float(i), "rsi": 50.0, "macd": 0.0,
                "mom5": 0.0, "mom20": 0.0, "regime_mult": 1.0,
                "vol_ratio": 1.0, "bb_position": 0.0,
                "news_urgency": None, "news_article_count": None,
                "ticker": "NVDA", "action": "BUY",
                "forward_return_5d": float(i),
                "sim_date": f"2024-01-01",
            })
        # give them an orderable sim_date
        for idx, r in enumerate(recs):
            r["sim_date"] = (
                f"2024-{1 + idx // 28:02d}-{1 + idx % 28:02d}")
        oos = fi.feature_importance(_EchoScorer(), recs,
                                    oos_only=True, n_repeats=2)
        assert oos["slice"] == "oos"
        assert oos["n_records_considered"] == 40
        full = fi.feature_importance(_EchoScorer(), recs,
                                     oos_only=False, n_repeats=2)
        assert full["slice"] == "all"
        assert full["n_records_considered"] == 200


class TestNeverRaises:
    def test_raising_scorer_degrades(self):
        rep = fi.feature_importance(_RaisingScorer(), _buy_records(80),
                                    oos_only=False, n_repeats=2)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_nan_scorer_degrades(self):
        rep = fi.feature_importance(_NaNScorer(), _buy_records(80),
                                    oos_only=False, n_repeats=2)
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_untrained_scorer(self):
        rep = fi.feature_importance(_UntrainedScorer(), _buy_records(80),
                                    oos_only=False, n_repeats=2)
        assert rep["verdict"] == "UNTRAINED"

    def test_too_few_records(self):
        rep = fi.feature_importance(_EchoScorer(), _buy_records(10),
                                    oos_only=False, n_repeats=2)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 10

    def test_none_fields_do_not_crash(self):
        recs = _buy_records(60)
        for r in recs:
            r["rsi"] = None
            r["macd"] = None
            r["news_urgency"] = None
            r["vol_ratio"] = None
        rep = fi.feature_importance(_EchoScorer(), recs,
                                    oos_only=False, n_repeats=2)
        assert rep["status"] == "ok"
        assert rep["top_feature"] == "ml_score"


class TestDegenerateFlag:
    """A null/constant column has nothing to permute → its 0.0 importance
    must be flagged `degenerate`, NOT silently read as 'model ignores it'.
    This is the honesty guard for the live news_urgency/news_article_count
    case (null for 100% of the OOS slice)."""

    def test_all_null_column_is_degenerate_not_material(self):
        recs = _buy_records(80)
        for r in recs:
            r["news_urgency"] = None          # constant-null column
            r["news_article_count"] = None
            r["macd"] = 0.5                   # constant non-null column
        rep = fi.feature_importance(_EchoScorer(), recs,
                                    oos_only=False, n_repeats=2)
        by = {f["feature"]: f for f in rep["features"]}
        for deg in ("news_urgency", "news_article_count", "macd"):
            assert by[deg]["degenerate"] is True, deg
            assert by[deg]["material"] is False, deg
            assert by[deg]["rmse_increase"] == 0.0, deg
        assert by["news_urgency"]["n_distinct"] == 0
        assert by["macd"]["n_distinct"] == 1
        # A varied feature the echo model ignores is NOT degenerate (it has
        # variance — the 0.0 is a true 'model ignores it', distinct case).
        assert by["rsi"]["degenerate"] is False
        assert by["rsi"]["n_distinct"] >= 2
        assert by["rsi"]["material"] is False
        assert set(rep["degenerate_features"]) >= {
            "news_urgency", "news_article_count", "macd"}
        assert rep["n_degenerate_features"] >= 3

    def test_fully_degenerate_inputs_read_flat(self):
        # Every feature constant except ml_score, which the echo model uses —
        # so this is SIGNAL_GROUNDED, not FLAT. Then make ml_score constant
        # too → nothing varies → FLAT with a degenerate note.
        recs = _buy_records(60)
        for r in recs:
            for k in ("ml_score", "rsi", "macd", "mom5", "mom20",
                      "regime_mult", "vol_ratio", "bb_position",
                      "news_urgency", "news_article_count"):
                r[k] = 1.0
            r["ticker"] = "NVDA"
        rep = fi.feature_importance(_EchoScorer(), recs,
                                    oos_only=False, n_repeats=2)
        assert rep["verdict"] == "FLAT"
        assert "degenerate" in rep["hint"]
        assert rep["n_degenerate_features"] == 11


class TestFeatureCoverage:
    def test_all_eleven_logical_features_reported(self):
        rep = fi.feature_importance(_EchoScorer(), _buy_records(80),
                                    oos_only=False, n_repeats=2)
        names = {f["feature"] for f in rep["features"]}
        assert names == {
            "ml_score", "rsi", "macd", "mom5", "mom20", "regime_mult",
            "vol_ratio", "bb_position", "news_urgency",
            "news_article_count", "sector",
        }
        assert len(rep["features"]) == 11
        # sorted descending by rmse_increase
        incs = [f["rmse_increase"] for f in rep["features"]]
        assert incs == sorted(incs, reverse=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
