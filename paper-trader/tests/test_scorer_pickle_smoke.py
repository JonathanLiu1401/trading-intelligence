"""Tests for paper_trader.ml.scorer_pickle_smoke — read-only sanity probe
of the deployed DecisionScorer pickle (2026-05-23 Agent-2 HYBRID pass #20).

Pins the verdict ladder for the four adverse states the analyzer was
built to catch (INSUFFICIENT_N_TRAIN, COLLAPSED_PRED_QUANTILES,
COLLAPSED_LABEL_QUANTILES, DEGENERATE_PREDICTIONS) plus the
healthy / can't-tell cases. Asserts specific expected values — not just
"no crash". Every test is offline: the analyzer reads only the on-disk
pickle (redirected to tmp by conftest) and runs predict_with_meta with
deterministic synthetic features.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

import paper_trader.ml.decision_scorer as ds
from paper_trader.ml.decision_scorer import train_scorer
from paper_trader.ml import scorer_pickle_smoke as sps


class _ConstModel:
    """Module-level so pickle can serialise it (a local class would fail
    `Can't pickle local object`). Stand-in for a degenerate trained model
    that emits a constant prediction regardless of input — the gate-relevant
    failure mode the predict-variance probe was built to catch."""

    def predict(self, X):
        return np.full(len(X), 7.5, dtype=np.float64)


def _training_records(n: int = 240) -> list[dict]:
    """Synthetic outcomes shaped so a fresh train_scorer run produces a
    real (non-collapsed) model. Mirrors the test_scorer_percentile.py
    helper exactly — same fr5d ∝ ml_score signal so the trained net
    must emit varying predictions across the bullish→bearish range.
    """
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
            "forward_return_5d": float(score) * 1.5,
            "return_pct": 10.0,
        })
    return recs


def _write_state(state: dict):
    """Atomically write a synthetic pickle to the conftest-redirected
    SCORER_PATH, mirroring train_scorer's tmp+replace idiom."""
    ds.SCORER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ds.SCORER_PATH.open("wb") as f:
        pickle.dump(state, f)
    ds._LOAD_CACHE.clear()


@pytest.fixture
def healthy_pickle():
    """A real trained pickle from synthetic data — known-good baseline."""
    ds._LOAD_CACHE.clear()
    result = train_scorer(_training_records())
    assert result["status"] == "ok", result
    ds._LOAD_CACHE.clear()
    return result


# ─────────────────────────── can't-tell verdicts ──────────────────────

class TestCantTellVerdicts:
    def test_missing_pickle_returns_insufficient_data(self):
        if ds.SCORER_PATH.exists():
            ds.SCORER_PATH.unlink()
        ds._LOAD_CACHE.clear()
        rep = sps.analyze()
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_train"] is None
        # Exit-code shim agrees with the verdict ladder.
        assert sps.is_pickle_smoke_failed() is None

    def test_unreadable_pickle_handled(self):
        """A torn-write / non-pickle file must NOT crash the analyzer.
        Verdict UNREADABLE_PICKLE, exit code 0 (cannot prove regression)."""
        ds.SCORER_PATH.parent.mkdir(parents=True, exist_ok=True)
        ds.SCORER_PATH.write_bytes(b"\x00\xff not a pickle \x00")
        ds._LOAD_CACHE.clear()
        rep = sps.analyze()
        assert rep["verdict"] == "UNREADABLE_PICKLE"
        assert sps.is_pickle_smoke_failed() is None

    def test_non_dict_pickle_handled(self):
        """A pickle whose top-level is not a dict (wrong-shape artifact)
        must surface as UNREADABLE_PICKLE, not crash."""
        with ds.SCORER_PATH.open("wb") as f:
            pickle.dump(["not", "a", "scorer"], f)
        ds._LOAD_CACHE.clear()
        rep = sps.analyze()
        assert rep["verdict"] == "UNREADABLE_PICKLE"

    def test_missing_model_key_handled(self):
        """A dict pickle missing the `model` key surfaces UNREADABLE_PICKLE."""
        _write_state({"scaler": None, "n_train": 1000})
        rep = sps.analyze()
        assert rep["verdict"] == "UNREADABLE_PICKLE"
        assert "model" in rep["hint"]


# ─────────────────────────── adverse verdicts ─────────────────────────

class TestInsufficientNTrain:
    def test_n_train_below_floor_detected(self, healthy_pickle):
        """The 2026-05-23 finding #1 footprint: n_train=39 (synthetic).
        The analyzer must catch this BEFORE any other check fires."""
        with ds.SCORER_PATH.open("rb") as f:
            state = pickle.load(f)
        state["n_train"] = 39
        _write_state(state)
        rep = sps.analyze()
        assert rep["verdict"] == "INSUFFICIENT_N_TRAIN"
        assert rep["n_train"] == 39
        assert "39" in rep["hint"]
        assert sps.is_pickle_smoke_failed() is True

    def test_n_train_exactly_at_floor_is_healthy(self, healthy_pickle):
        """The check uses `< MIN_N_TRAIN` — equal must be acceptable
        (one-off retraining at exactly the floor is not a regression)."""
        with ds.SCORER_PATH.open("rb") as f:
            state = pickle.load(f)
        state["n_train"] = sps.MIN_N_TRAIN
        _write_state(state)
        rep = sps.analyze()
        # Healthy because n_train >= floor AND the healthy_pickle fixture
        # produced real (non-collapsed) quantiles + varying predictions.
        assert rep["verdict"] == "HEALTHY"

    def test_n_train_takes_precedence_over_collapsed_quantiles(self,
                                                               healthy_pickle):
        """Adverse-precedence invariant: n_train < floor is the MOST
        actionable case (delete + wait for the loop), so it must fire
        even when other adverse conditions exist."""
        with ds.SCORER_PATH.open("rb") as f:
            state = pickle.load(f)
        state["n_train"] = 39
        # Also collapse the pred_quantiles — n_train wins.
        state["pred_quantiles"] = [18.934] * 101
        _write_state(state)
        rep = sps.analyze()
        assert rep["verdict"] == "INSUFFICIENT_N_TRAIN"
        # Evidence is still surfaced honestly though.
        assert rep["pred_quantiles_collapsed"] is True


class TestCollapsedQuantiles:
    def test_collapsed_pred_quantiles_detected(self, healthy_pickle):
        """Exact 2026-05-23 finding #1 footprint: all 101 entries equal."""
        with ds.SCORER_PATH.open("rb") as f:
            state = pickle.load(f)
        state["pred_quantiles"] = [18.934] * 101
        _write_state(state)
        rep = sps.analyze()
        assert rep["verdict"] == "COLLAPSED_PRED_QUANTILES"
        assert rep["pred_quantiles_collapsed"] is True
        # The diagnostic surfaces the cascade: percentile/calibrated fields
        # in the consumer-visible predict_with_meta return None (per the
        # same-pass _raw_to_percentile guard).
        assert "consumer-visible" in rep["hint"]
        assert sps.is_pickle_smoke_failed() is True

    def test_collapsed_label_quantiles_detected(self, healthy_pickle):
        """Sibling check: every realized 5d label was the same value."""
        with ds.SCORER_PATH.open("rb") as f:
            state = pickle.load(f)
        state["label_quantiles"] = [0.0] * 101
        # Keep pred_quantiles healthy so the pred-collapse check doesn't
        # win the precedence race.
        _write_state(state)
        rep = sps.analyze()
        assert rep["verdict"] == "COLLAPSED_LABEL_QUANTILES"
        assert rep["label_quantiles_collapsed"] is True
        assert sps.is_pickle_smoke_failed() is True

    def test_pred_collapse_takes_precedence_over_label_collapse(self,
                                                                healthy_pickle):
        """If both quantile tables collapse, pred_quantiles wins (it's the
        upstream artifact — collapsed predictions explain the gate's
        consumer-visible regression more directly)."""
        with ds.SCORER_PATH.open("rb") as f:
            state = pickle.load(f)
        state["pred_quantiles"] = [5.0] * 101
        state["label_quantiles"] = [0.0] * 101
        _write_state(state)
        rep = sps.analyze()
        assert rep["verdict"] == "COLLAPSED_PRED_QUANTILES"

    def test_legacy_pickle_without_quantiles_unchecked(self, healthy_pickle):
        """A legacy pickle predating the quantile fields has the keys
        absent; the analyzer must NOT flag it as collapsed (cannot tell
        ⇒ degrade to predict-variance + n_train only)."""
        with ds.SCORER_PATH.open("rb") as f:
            state = pickle.load(f)
        del state["pred_quantiles"]
        del state["label_quantiles"]
        _write_state(state)
        rep = sps.analyze()
        assert rep["pred_quantiles_collapsed"] is None
        assert rep["label_quantiles_collapsed"] is None
        # With absent quantiles + valid n_train + varying predictions, the
        # pickle is HEALTHY for smoke purposes (rank/calibrated consumers
        # already degrade to None per the legacy-compatibility tests).
        assert rep["verdict"] == "HEALTHY"


class TestDegeneratePredictions:
    def test_constant_predictor_detected(self, healthy_pickle):
        """A model that emits the same value for every input must trigger
        DEGENERATE_PREDICTIONS — the gate would modulate every BUY by the
        same factor, pure variance with no edge."""

        with ds.SCORER_PATH.open("rb") as f:
            state = pickle.load(f)
        state["model"] = _ConstModel()
        # Even with healthy n_train and non-collapsed quantiles the
        # predict-variance probe catches the bogus model.
        # Disambiguate: keep quantiles non-collapsed so they don't win.
        _write_state(state)
        rep = sps.analyze()
        assert rep["verdict"] == "DEGENERATE_PREDICTIONS"
        assert rep["prediction_spread"] is not None
        assert rep["prediction_spread"] < sps.DEGENERATE_PRED_VAR_THRESHOLD
        # All probes succeeded, none failed — the model "predicts", it just
        # predicts the same thing every time.
        assert rep["n_predicted"] == len(sps._PROBE_ML_SCORES)
        assert rep["n_failed"] == 0
        assert sps.is_pickle_smoke_failed() is True

    def test_predict_spread_threshold_calibrated(self, healthy_pickle):
        """Sanity counterfactual: a real trained model on the synthetic
        fr5d ∝ ml_score corpus MUST produce a meaningful spread (well
        above the degeneracy floor)."""
        rep = sps.analyze()
        assert rep["verdict"] == "HEALTHY"
        assert rep["prediction_spread"] is not None
        # The synthetic training signal is monotone in ml_score so the
        # bullish→bearish probe grid must produce a spread comfortably
        # above the floor — proves the threshold catches degeneracy
        # without false-flagging real models.
        assert rep["prediction_spread"] > sps.DEGENERATE_PRED_VAR_THRESHOLD * 2


# ─────────────────────────── healthy verdict ──────────────────────────

class TestHealthyPickle:
    def test_real_trained_pickle_passes_smoke(self, healthy_pickle):
        rep = sps.analyze()
        assert rep["verdict"] == "HEALTHY"
        assert rep["n_train"] >= sps.MIN_N_TRAIN
        assert rep["pred_quantiles_collapsed"] is False
        assert rep["label_quantiles_collapsed"] is False
        assert rep["n_predicted"] == len(sps._PROBE_ML_SCORES)
        assert rep["n_failed"] == 0
        assert sps.is_pickle_smoke_failed() is False


# ─────────────────────────── CLI / exit codes ─────────────────────────

class TestCli:
    def test_cli_exit_2_on_healthy_pickle_clobbered_to_synthetic(
            self, healthy_pickle, capsys):
        """End-to-end: a healthy pickle clobbered into the n=39 footprint
        must exit 2 (operator-actionable)."""
        with ds.SCORER_PATH.open("rb") as f:
            state = pickle.load(f)
        state["n_train"] = 39
        _write_state(state)
        rc = sps._cli([])
        assert rc == 2
        out = capsys.readouterr().out
        assert "INSUFFICIENT_N_TRAIN" in out

    def test_cli_exit_0_on_healthy(self, healthy_pickle, capsys):
        rc = sps._cli([])
        assert rc == 0
        assert "HEALTHY" in capsys.readouterr().out

    def test_cli_exit_0_on_missing_pickle(self, capsys):
        if ds.SCORER_PATH.exists():
            ds.SCORER_PATH.unlink()
        ds._LOAD_CACHE.clear()
        rc = sps._cli([])
        assert rc == 0
        assert "INSUFFICIENT_DATA" in capsys.readouterr().out

    def test_cli_json_emits_valid_json(self, healthy_pickle, capsys):
        """The CLI emits structured JSON when called with --json. The
        DecisionScorer's `[decision_scorer] loaded n=…` log line (printed
        once per cold cache miss) interleaves on stdout, so the JSON parser
        must skip past it — slice from the first `{` to the matching `}`."""
        import json
        rc = sps._cli(["--json"])
        assert rc == 0
        raw = capsys.readouterr().out
        # The JSON object spans the last `{` … `}` block — robust to any
        # pre-stdout chatter from the scorer's load path.
        start = raw.index("{")
        end = raw.rindex("}")
        payload = json.loads(raw[start:end + 1])
        assert payload["verdict"] == "HEALTHY"
        assert isinstance(payload["n_train"], int)


# ─────────────────────────── never-raises discipline ──────────────────

class TestNeverRaises:
    def test_analyze_with_invalid_path_returns_insufficient_data(self):
        rep = sps.analyze("/nonexistent/path/scorer.pkl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_is_pickle_smoke_failed_handles_garbage_path(self):
        # A garbage path must return None (cannot tell), never raise.
        assert sps.is_pickle_smoke_failed(
            "/dev/null/not-a-real-path/scorer.pkl") is None
