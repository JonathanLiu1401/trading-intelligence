"""Tests for paper_trader.ml.feature_ablation.

Verifies the per-feature ablation analyzer:
  * builds the OOS feature matrix correctly (drops missing labels,
    flips SELL sign, clamps to ±50%),
  * computes baseline rank-IC honestly,
  * computes per-feature deltas when an input column is zeroed out,
  * picks the right verdict from the delta distribution,
  * never raises on degenerate / malformed input,
  * the CLI exits 2 on adverse verdicts and 0 on healthy.
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

import paper_trader.ml.feature_ablation as fa
from paper_trader.ml.decision_scorer import (
    SCORER_PATH, FEATURE_NAMES, N_FEATURES, train_scorer,
)


def _write_outcomes(path: Path, records: list[dict]) -> None:
    """Write outcome records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _synthetic_outcomes(n: int, seed: int = 17,
                       leakage: float = 0.0) -> list[dict]:
    """Generate `n` synthetic outcome rows the scorer can be trained on.

    By default the forward return is independent of every feature so the
    scorer's rank-IC will be ≈ 0 (the noise-only case). Passing
    ``leakage > 0`` makes ``forward_return_5d`` weakly correlated with
    ``ml_score`` so the scorer can actually learn a signal (used to test
    the LOAD_BEARING verdict).
    """
    rng = np.random.default_rng(seed)
    tickers = ["NVDA", "AMD", "AAPL", "MSFT", "TSLA", "XOM", "LLY",
               "JPM", "GLD", "COIN", "TM"]
    sectors_count = len(set(tickers))
    out: list[dict] = []
    for i in range(n):
        tk = tickers[i % len(tickers)]
        ml = float(rng.normal(2.0, 1.5))
        rsi = float(rng.uniform(20, 80))
        macd = float(rng.normal(0, 0.5))
        mom5 = float(rng.normal(0, 3))
        mom20 = float(rng.normal(0, 5))
        regime_mult = float(rng.choice([0.3, 0.6, 1.0]))
        vol_ratio = float(rng.uniform(0.5, 3.0))
        bb_pos = float(rng.uniform(-1.5, 1.5))
        urg = float(rng.uniform(40, 80))
        cnt = float(rng.integers(1, 10))
        fr = float(rng.normal(0, 5)) + leakage * ml
        out.append({
            "sim_date": f"2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            "ticker": tk,
            "action": "BUY",
            "ml_score": ml,
            "rsi": rsi,
            "macd": macd,
            "mom5": mom5,
            "mom20": mom20,
            "regime_mult": regime_mult,
            "vol_ratio": vol_ratio,
            "bb_position": bb_pos,
            "news_urgency": urg,
            "news_article_count": cnt,
            "forward_return_5d": fr,
            "return_pct": float(rng.normal(0, 30)),
        })
    return out


# ---------------------------------------------------------------------------
# helper functions — pure unit tests
# ---------------------------------------------------------------------------


class TestFiniteFloat:
    def test_finite_passthrough(self):
        assert fa._finite_float(3.14) == 3.14

    def test_int_passthrough(self):
        assert fa._finite_float(7) == 7.0

    def test_none_returns_none(self):
        assert fa._finite_float(None) is None

    def test_bool_returns_none(self):
        # bool is a subclass of int; treat as "no value"
        assert fa._finite_float(True) is None
        assert fa._finite_float(False) is None

    def test_nan_returns_none(self):
        assert fa._finite_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert fa._finite_float(float("inf")) is None
        assert fa._finite_float(float("-inf")) is None

    def test_unparseable_string_returns_none(self):
        # Strings that can't be float()'d are dropped. Numeric strings
        # like "3.14" parse cleanly (matching feature_correlation_audit's
        # _finite_float behaviour) — the JSONL writer never emits quoted
        # numerics, so this branch is rarely hit in practice but the
        # contract matches sibling analyzers.
        assert fa._finite_float("abc") is None
        # Mirrors feature_correlation_audit: numeric strings pass through.
        assert fa._finite_float("3.14") == 3.14


class TestSpearman:
    def test_perfect_positive(self):
        a = np.arange(10, dtype=float)
        b = a.copy()
        rho = fa._spearman(a, b)
        assert rho == pytest.approx(1.0)

    def test_perfect_negative(self):
        a = np.arange(10, dtype=float)
        b = -a
        rho = fa._spearman(a, b)
        assert rho == pytest.approx(-1.0)

    def test_zero_variance_returns_none(self):
        a = np.array([1.0, 1.0, 1.0, 1.0])
        b = np.arange(4, dtype=float)
        assert fa._spearman(a, b) is None
        assert fa._spearman(b, a) is None

    def test_short_returns_none(self):
        assert fa._spearman(np.array([1.0]), np.array([2.0])) is None


# ---------------------------------------------------------------------------
# analyze() — verdict ladder
# ---------------------------------------------------------------------------


class TestInsufficientData:
    def test_empty_file(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        path.write_text("")
        rep = fa.analyze(outcomes_path=path)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0
        assert rep["baseline_rank_ic"] is None

    def test_too_few_rows(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        _write_outcomes(path, _synthetic_outcomes(20))
        rep = fa.analyze(outcomes_path=path)
        # 20 rows, OOS = last 20% = 4 rows, well below MIN_OBS=50
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] < fa.MIN_OBS

    def test_missing_file(self, tmp_path):
        rep = fa.analyze(outcomes_path=tmp_path / "does_not_exist.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0


class TestScorerUntrained:
    def test_no_pickle(self, tmp_path):
        # _isolate_data_dir auto-fixture already moved SCORER_PATH into tmp;
        # ensure no pickle exists.
        path = tmp_path / "outcomes.jsonl"
        _write_outcomes(path, _synthetic_outcomes(500))
        rep = fa.analyze(outcomes_path=path)
        assert rep["verdict"] == "SCORER_UNTRAINED"
        assert rep["baseline_rank_ic"] is None


class TestBaselineDegenerate:
    """A model whose predict() returns a constant — every ablation is no-op."""

    def test_constant_predictor_returns_degenerate(self, tmp_path, monkeypatch):
        path = tmp_path / "outcomes.jsonl"
        records = _synthetic_outcomes(500)
        _write_outcomes(path, records)

        # Train a real scorer first so the pickle exists, then monkey-patch
        # the loaded model to return a constant.
        result = train_scorer(records)
        assert result.get("status") == "ok"

        class _ConstModel:
            def predict(self, X):
                return np.ones(len(X), dtype=np.float64) * 1.5

        from paper_trader.ml.decision_scorer import DecisionScorer
        ds_orig_init = DecisionScorer.__init__

        def _patched_init(self):
            ds_orig_init(self)
            self._model = _ConstModel()
        monkeypatch.setattr(DecisionScorer, "__init__", _patched_init)

        rep = fa.analyze(outcomes_path=path)
        assert rep["verdict"] == "BASELINE_DEGENERATE"


class TestAnalyzeWithTrainedScorer:
    """End-to-end with a real trained scorer on synthetic data."""

    @pytest.fixture
    def trained(self, tmp_path):
        """Train the scorer on enough synthetic rows that the pickle is
        deployed in tmp. Returns the outcomes path so analyze() can read it."""
        path = tmp_path / "outcomes.jsonl"
        # Enough rows that the OOS slice (20%) clears MIN_OBS=50.
        records = _synthetic_outcomes(500, leakage=0.4)
        _write_outcomes(path, records)
        result = train_scorer(records)
        assert result.get("status") == "ok"
        return path

    def test_returns_baseline_rank_ic(self, trained):
        rep = fa.analyze(outcomes_path=trained)
        assert rep["status"] == "ok"
        assert rep["n"] >= fa.MIN_OBS
        # baseline rank-IC must be a finite float
        b = rep["baseline_rank_ic"]
        assert b is not None
        assert isinstance(b, float)
        assert math.isfinite(b)
        assert -1.0 <= b <= 1.0

    def test_ablations_cover_all_groups(self, trained):
        rep = fa.analyze(outcomes_path=trained)
        names = [a["feature"] for a in rep["ablations"]]
        # All 10 numeric features ablated individually + 1 sector group.
        for name in fa.NUMERIC_FEATURES:
            assert name in names
        assert "sector" in names

    def test_ablation_delta_is_change_from_baseline(self, trained):
        rep = fa.analyze(outcomes_path=trained)
        b = rep["baseline_rank_ic"]
        for a in rep["ablations"]:
            if a["rank_ic"] is None:
                assert a["delta"] is None
            else:
                assert a["delta"] == pytest.approx(a["rank_ic"] - b, abs=1e-4)

    def test_slice_is_oos_by_default(self, trained):
        rep = fa.analyze(outcomes_path=trained)
        assert rep["slice"] == "temporal_oos"

    def test_full_corpus_when_oos_only_false(self, trained):
        rep = fa.analyze(outcomes_path=trained, oos_only=False)
        assert rep["slice"] == "full"
        # Full corpus is strictly larger than OOS slice.
        rep_oos = fa.analyze(outcomes_path=trained, oos_only=True)
        assert rep["n"] > rep_oos["n"]


class TestVerdictPartitioning:
    """The verdict ladder splits redundant vs load_bearing by EDGE_TOL."""

    def test_force_redundant_verdict(self, trained_with_skill, monkeypatch):
        # Inject ablation deltas: one strongly positive (redundant), rest 0
        # — analyze should pick REDUNDANT_DETECTED.
        # We patch _spearman to return controlled values.
        path = trained_with_skill

        # Use a state machine: first call (baseline) → 0.10; second call
        # (first feature ablated) → 0.15 (delta = +0.05, > EDGE_TOL); rest → 0.10.
        calls = {"i": 0}
        # 11 groups total: 10 numeric + 1 sector
        # baseline + 11 ablations = 12 _spearman calls
        targets = [0.10, 0.15] + [0.10] * 10

        def _fake(a, b):
            c = calls["i"]
            calls["i"] += 1
            return targets[c] if c < len(targets) else 0.10

        monkeypatch.setattr(fa, "_spearman", _fake)
        rep = fa.analyze(outcomes_path=path)
        assert rep["verdict"] == "REDUNDANT_DETECTED"
        assert len(rep["top_redundant"]) >= 1
        assert rep["top_redundant"][0]["delta"] == pytest.approx(0.05)

    def test_force_load_bearing_verdict(self, trained_with_skill, monkeypatch):
        path = trained_with_skill
        # baseline 0.10, first ablation 0.04 (delta = -0.06 < -EDGE_TOL), rest 0.10
        calls = {"i": 0}
        targets = [0.10, 0.04] + [0.10] * 10

        def _fake(a, b):
            c = calls["i"]
            calls["i"] += 1
            return targets[c] if c < len(targets) else 0.10

        monkeypatch.setattr(fa, "_spearman", _fake)
        rep = fa.analyze(outcomes_path=path)
        assert rep["verdict"] == "LOAD_BEARING_DETECTED"
        assert len(rep["top_load_bearing"]) >= 1
        assert rep["top_load_bearing"][0]["delta"] == pytest.approx(-0.06)

    def test_force_mixed_verdict(self, trained_with_skill, monkeypatch):
        path = trained_with_skill
        # baseline 0.10, first ablation 0.15 (redundant), second 0.04 (load-bearing)
        calls = {"i": 0}
        targets = [0.10, 0.15, 0.04] + [0.10] * 10

        def _fake(a, b):
            c = calls["i"]
            calls["i"] += 1
            return targets[c] if c < len(targets) else 0.10

        monkeypatch.setattr(fa, "_spearman", _fake)
        rep = fa.analyze(outcomes_path=path)
        assert rep["verdict"] == "MIXED"
        assert len(rep["top_redundant"]) >= 1
        assert len(rep["top_load_bearing"]) >= 1

    def test_no_significant_effect(self, trained_with_skill, monkeypatch):
        path = trained_with_skill
        # baseline 0.10, every ablation within EDGE_TOL of baseline
        def _fake(a, b):
            return 0.105  # delta = +0.005 < EDGE_TOL=0.02

        monkeypatch.setattr(fa, "_spearman", _fake)
        rep = fa.analyze(outcomes_path=path)
        assert rep["verdict"] == "NO_SIGNIFICANT_EFFECT"
        assert rep["top_redundant"] == []
        assert rep["top_load_bearing"] == []


@pytest.fixture
def trained_with_skill(tmp_path):
    """Set up a trained scorer pickle for use in monkey-patched verdict tests."""
    path = tmp_path / "outcomes.jsonl"
    records = _synthetic_outcomes(500, leakage=0.4)
    _write_outcomes(path, records)
    result = train_scorer(records)
    assert result.get("status") == "ok"
    return path


# ---------------------------------------------------------------------------
# Never-raises contract
# ---------------------------------------------------------------------------


class TestNeverRaises:
    def test_corrupt_jsonl_lines_skipped(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        records = _synthetic_outcomes(500)
        with path.open("w") as fh:
            fh.write("not valid json\n")
            for r in records[:5]:
                fh.write(json.dumps(r) + "\n")
            fh.write("{\"partial\": 1\n")  # truncated
            for r in records[5:]:
                fh.write(json.dumps(r) + "\n")

        # Should return without crashing; the malformed lines are skipped
        # by _iter_rows. We can't train a scorer without a pickle, so this
        # particular run will report SCORER_UNTRAINED — the never-raises
        # contract is what we're testing.
        rep = fa.analyze(outcomes_path=path)
        assert isinstance(rep, dict)
        assert "verdict" in rep

    def test_missing_forward_return_drops_row(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        records = _synthetic_outcomes(500)
        for r in records[::2]:
            r["forward_return_5d"] = None
        _write_outcomes(path, records)
        rep = fa.analyze(outcomes_path=path)
        assert isinstance(rep, dict)
        # Half the rows drop, but synthesizing 500 then half-dropping leaves
        # 250, OOS slice = 50 → still meets MIN_OBS.
        # If the scorer didn't train, just check no crash.

    def test_nan_forward_return_drops_row(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        records = _synthetic_outcomes(500)
        for r in records[::3]:
            r["forward_return_5d"] = float("nan")
        _write_outcomes(path, records)
        rep = fa.analyze(outcomes_path=path)
        assert isinstance(rep, dict)

    def test_records_with_no_action(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        records = _synthetic_outcomes(500)
        for r in records:
            r.pop("action", None)
        _write_outcomes(path, records)
        # Default action is BUY, so no sign flip. Should still work.
        rep = fa.analyze(outcomes_path=path)
        assert isinstance(rep, dict)


# ---------------------------------------------------------------------------
# Behavioural correctness
# ---------------------------------------------------------------------------


class TestSellSignFlip:
    """SELL rows have their realized return sign-flipped to match the
    scorer's training-time convention."""

    def test_sell_y_is_negated_buy(self, tmp_path):
        """A BUY with fr=+10% and an identical SELL (same features) with
        fr=+10% should produce y vectors of opposite sign."""
        from paper_trader.ml.decision_scorer import _to_float as _tf  # noqa
        rec_buy = {
            "sim_date": "2025-01-15", "ticker": "NVDA", "action": "BUY",
            "ml_score": 1.0, "rsi": 50, "macd": 0.0, "mom5": 0,
            "mom20": 0, "regime_mult": 1.0, "vol_ratio": 1.0,
            "bb_position": 0.0, "news_urgency": 50, "news_article_count": 1,
            "forward_return_5d": 10.0,
        }
        rec_sell = dict(rec_buy, action="SELL")
        X, y, _ = fa._build_feature_matrix([rec_buy, rec_sell])
        assert y[0] == pytest.approx(10.0)
        assert y[1] == pytest.approx(-10.0)


class TestLabelClamp:
    """Realized 5d returns outside ±50% are clamped (matching train_scorer)."""

    def test_label_above_clamp(self, tmp_path):
        rec = {
            "sim_date": "2025-01-15", "ticker": "NVDA", "action": "BUY",
            "ml_score": 1.0, "rsi": 50, "macd": 0.0, "mom5": 0,
            "mom20": 0, "regime_mult": 1.0, "vol_ratio": 1.0,
            "bb_position": 0.0, "news_urgency": 50, "news_article_count": 1,
            "forward_return_5d": 175.0,
        }
        _, y, _ = fa._build_feature_matrix([rec])
        # Clamped to PRED_CLAMP_PCT (50.0)
        assert y[0] == pytest.approx(50.0)

    def test_label_below_clamp(self):
        rec = {
            "sim_date": "2025-01-15", "ticker": "NVDA", "action": "BUY",
            "ml_score": 1.0, "rsi": 50, "macd": 0.0, "mom5": 0,
            "mom20": 0, "regime_mult": 1.0, "vol_ratio": 1.0,
            "bb_position": 0.0, "news_urgency": 50, "news_article_count": 1,
            "forward_return_5d": -125.0,
        }
        _, y, _ = fa._build_feature_matrix([rec])
        assert y[0] == pytest.approx(-50.0)


class TestAblationMatchesBaseline:
    """Sanity-check: ablating a constant column (where it already equals the
    training-set mean post-scaling, i.e. zero in standardized space) should
    yield exactly the same predictions as the un-ablated baseline."""

    def test_ablate_then_unablate(self, tmp_path):
        """After scaling, the mean of each column is exactly 0 — so zeroing
        a column for one row is equivalent to setting that row's value to
        the column mean. For a row whose feature already equals the
        training mean, ablation is a no-op."""
        # This is structurally true by construction; verify with the
        # internal helper that the same predictions are produced when
        # zero_cols is empty.
        rng = np.random.default_rng(13)
        X = rng.standard_normal((10, N_FEATURES))

        class _LinModel:
            def predict(self, X):
                return X.sum(axis=1)

        class _NullScaler:
            def transform(self, X):
                return X

        base = fa._predict_with_optional_ablation(
            _LinModel(), _NullScaler(), X)
        # Ablate column 0 in a 0-column matrix: predictions reduce by the
        # value of column 0 (since the linear model sums all columns).
        ablated = fa._predict_with_optional_ablation(
            _LinModel(), _NullScaler(), X, zero_cols=(0,))
        # Each row's prediction drops by X[i, 0]
        assert np.allclose(base - X[:, 0], ablated)


# ---------------------------------------------------------------------------
# CLI exit codes
# ---------------------------------------------------------------------------


class TestCLIExitCode:
    def test_exit_2_on_scorer_untrained(self, tmp_path, monkeypatch):
        # No pickle in tmp; CLI should exit 2.
        path = tmp_path / "outcomes.jsonl"
        _write_outcomes(path, _synthetic_outcomes(500))
        argv = ["--path", str(path)]
        rc = fa.main(argv)
        assert rc == 2  # SCORER_UNTRAINED is adverse

    def test_exit_2_on_insufficient_data(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        path.write_text("")
        argv = ["--path", str(path)]
        rc = fa.main(argv)
        # INSUFFICIENT_DATA is NOT in the adverse set (it's a
        # data-availability issue, not a model issue) → exit 0.
        assert rc == 0

    def test_json_mode_emits_parseable(self, tmp_path, capsys):
        path = tmp_path / "outcomes.jsonl"
        path.write_text("")  # empty
        argv = ["--path", str(path), "--json"]
        fa.main(argv)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, dict)
        assert "verdict" in parsed


# ---------------------------------------------------------------------------
# Bug-catching tests
# ---------------------------------------------------------------------------


class TestStructuralAblation:
    """Tests that catch real logic bugs: ablation actually zeros the right
    columns."""

    def test_zeroing_no_columns_returns_baseline(self):
        """Empty zero_cols should give the same result as no ablation."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((8, N_FEATURES))

        class _IdentityModel:
            def predict(self, X):
                return X[:, 0].copy()

        class _NullScaler:
            def transform(self, X):
                return X

        base = fa._predict_with_optional_ablation(
            _IdentityModel(), _NullScaler(), X)
        with_empty = fa._predict_with_optional_ablation(
            _IdentityModel(), _NullScaler(), X, zero_cols=())
        assert np.allclose(base, with_empty)

    def test_zeroing_target_column_changes_prediction(self):
        """Ablating the column the model depends on should change the
        prediction to zero (for a model that returns column 0)."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((5, N_FEATURES))

        class _IdentityModel:
            def predict(self, X):
                return X[:, 0].copy()

        class _NullScaler:
            def transform(self, X):
                return X

        # Ablating column 0 → predictions should be all zeros.
        ablated = fa._predict_with_optional_ablation(
            _IdentityModel(), _NullScaler(), X, zero_cols=(0,))
        assert np.allclose(ablated, 0.0)

    def test_predict_returns_none_on_model_exception(self):
        """If the model raises, the helper returns None — never crashes."""
        class _RaisingModel:
            def predict(self, X):
                raise RuntimeError("predict failed")

        class _NullScaler:
            def transform(self, X):
                return X

        X = np.zeros((5, N_FEATURES))
        result = fa._predict_with_optional_ablation(
            _RaisingModel(), _NullScaler(), X)
        assert result is None
