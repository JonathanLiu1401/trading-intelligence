"""Tests for ``paper_trader.ml.dead_feature_audit`` and the per-cycle
``_append_dead_feature_audit_log`` wiring.

The audit catches the specific class of bug pass #35 fixed: a feature
added to ``DecisionScorer.build_features`` whose values are never plumbed
into ``_compute_decision_outcomes`` or ``_ml_decide``. The model sees a
constant-zero input → StandardScaler's std≈0 → L2 alpha drives weights to
exactly 0.0. These tests verify that:

  * A trained model with one feature pinned to constant zero is flagged
    HAS_DEAD with that exact feature name in the dead list.
  * A trained model with NO constant-zero features is flagged OK.
  * The audit's UNTRAINED / SHAPE_MISMATCH / ERROR paths emit honest
    sentinel envelopes (never raises).
  * The wired ``_append_dead_feature_audit_log`` ledger writes a JSON row
    with the verdict and trims the file at the documented cap.
"""
from __future__ import annotations

import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from paper_trader.ml import dead_feature_audit as dfa
from paper_trader.ml.decision_scorer import (
    DecisionScorer,
    FEATURE_NAMES,
    N_FEATURES,
    SCORER_PATH,
    train_scorer,
)


def _make_record(idx: int, *, ml_score=2.0, rsi=55.0, macd=0.01,
                 mom5=1.0, mom20=2.0, regime_mult=1.0, ticker="NVDA",
                 vol_ratio=1.0, bb_position=0.0,
                 news_urgency=50.0, news_article_count=1.0,
                 ema200_above=None, hist_cross_up=None,
                 macd_below_zero_cross=None, forward_return_5d=None,
                 action="BUY", return_pct=10.0):
    """Build one outcome record with optionally-overridable inputs.

    Default ``forward_return_5d`` is ``idx % 5 - 2`` (a small varying
    signal that gives the MLP something to fit). Tests can override any
    field via kwargs."""
    if forward_return_5d is None:
        forward_return_5d = float(idx % 5 - 2)
    return {
        "ml_score": ml_score, "rsi": rsi, "macd": macd, "mom5": mom5,
        "mom20": mom20, "regime_mult": regime_mult, "ticker": ticker,
        "vol_ratio": vol_ratio, "bb_position": bb_position,
        "news_urgency": news_urgency,
        "news_article_count": news_article_count,
        "ema200_above": ema200_above,
        "hist_cross_up": hist_cross_up,
        "macd_below_zero_cross": macd_below_zero_cross,
        "forward_return_5d": forward_return_5d,
        "action": action,
        "return_pct": return_pct,
        "sim_date": f"2024-01-{(idx % 28) + 1:02d}",
    }


class TestAuditDeadFeaturesBasicVerdicts:
    """The verdict ladder must distinguish NOT_TRAINED / OK / HAS_DEAD /
    SHAPE_MISMATCH / UNKNOWN_MODEL / ERROR. These tests target the first
    three (untrained + the two real-world common verdicts)."""

    def test_untrained_scorer_yields_not_trained_verdict(self):
        scorer = DecisionScorer()
        # The _isolate_data_dir autouse fixture redirects SCORER_PATH to a
        # tmp dir that has no pickle in it, so the scorer loads untrained.
        assert not scorer.is_trained
        rep = dfa.audit_dead_features(scorer)
        assert rep["verdict"] == "NOT_TRAINED"
        assert rep["n_features_dead"] == 0
        assert rep["dead_features"] == []
        # n_features_total still echoes — operators rely on the constant
        # being present even on a degenerate verdict.
        assert rep["n_features_total"] == N_FEATURES

    def test_trained_scorer_with_constant_feature_reports_dead(self, tmp_path):
        """Train a real (numpy lstsq fallback) model on records that pin
        ``ema200_above`` to None always (the live pass-#35 footprint).
        The audit must flag it as dead."""
        # Build a corpus where every other field varies but ema200_above
        # stays at None (build_features defaults it to 0.0). All 3 enhanced
        # MACD features go through the same path so we expect all three to
        # land in the dead list.
        # train_scorer dedups on (ticker, sim_date, action) and requires
        # ≥30 distinct rows post-dedup, so rotate ticker/sim_date together
        # to keep every record unique. Vary EVERY non-target feature
        # (including macd / regime_mult / vol_ratio / news_*) so only the
        # 3 enhanced MACD features are pinned to a constant — otherwise
        # other constant inputs share the dead-feature signature and the
        # test becomes ambiguous.
        tickers = ["NVDA", "AMD", "AAPL", "MSFT", "META", "GOOGL", "TSLA",
                   "JPM", "GS", "XOM"]
        records = []
        for i in range(120):
            tk = tickers[i % len(tickers)]
            day = (i % 28) + 1
            month = (i // 28) % 12 + 1
            records.append(_make_record(
                i,
                ml_score=1.0 + (i % 10),
                rsi=30.0 + (i * 0.7),
                macd=(i % 7 - 3) * 0.01,
                mom5=float(i % 11 - 5),
                mom20=float(i % 7 - 3),
                regime_mult=1.0 + (i % 5 - 2) * 0.1,
                vol_ratio=1.0 + (i % 7 - 3) * 0.1,
                bb_position=(i % 21 - 10) / 10.0,
                news_urgency=20.0 + (i % 60),
                news_article_count=1.0 + (i % 5),
                forward_return_5d=float(i % 9 - 4),
                ticker=tk,
                # All three enhanced MACD features omitted → build_features
                # defaults them to 0.0 always.
            ))
            records[-1]["sim_date"] = f"2024-{month:02d}-{day:02d}"
        out_path = tmp_path / "scorer.pkl"
        result = train_scorer(records, path=out_path)
        # The training must succeed for the audit to mean anything.
        assert result["status"] == "ok", (
            f"train_scorer returned {result} — corpus too small / dedup "
            f"killed it; the audit cannot validate without a model")
        # Point the scorer at the just-trained pickle.
        scorer = DecisionScorer.__new__(DecisionScorer)
        scorer._model = None
        scorer._scaler = None
        scorer._trained = False
        scorer._n_train = 0
        scorer._pred_quantiles = None
        scorer._label_quantiles = None
        with out_path.open("rb") as f:
            state = pickle.load(f)
        scorer._model = state["model"]
        scorer._scaler = state["scaler"]
        scorer._n_train = state["n_train"]
        scorer._trained = True
        # Use a loose eps for this unit test (1e-3 vs the prod default
        # 1e-6). On a 120-record corpus the L2 regularizer doesn't crush
        # constant-input weights all the way to 1e-40 like it does on a
        # 3580-record corpus (the deployed pickle); they land around 3e-5
        # with this corpus size. We're testing the audit's classification
        # logic — that constant inputs land BELOW the eps band while
        # varying inputs land ORDERS OF MAGNITUDE above it — not the
        # production eps value (which a separate test pins).
        rep = dfa.audit_dead_features(scorer, eps=1e-3)
        # All 3 enhanced-MACD features were trained on constant 0.0; their
        # mean|w| is many orders of magnitude smaller than the live (varying)
        # features'. The audit must flag them.
        assert rep["verdict"] == "HAS_DEAD"
        dead_names = {r["feature"] for r in rep["dead_features"]}
        for feat in ("ema200_above", "hist_cross_up", "macd_below_zero_cross"):
            assert feat in dead_names, (
                f"audit failed to flag constant-zero feature {feat!r} as "
                f"dead — dead features reported: {sorted(dead_names)}"
            )
        # Conversely, every VARYING feature must NOT be flagged. Sector
        # one-hot for tickers not in our corpus (healthcare/commodities/
        # crypto/other) are legitimately dead because nothing landed there
        # — those WILL show up in the dead list. The live varying features
        # (ml_score, rsi, mom*, bb_pos, vol_ratio, news_*) must NOT.
        live_features = {"ml_score", "rsi", "mom5", "mom20", "bb_pos",
                         "vol_ratio", "news_urgency", "news_article_count"}
        wrongly_flagged = live_features & dead_names
        assert not wrongly_flagged, (
            f"audit falsely flagged varying features as dead: "
            f"{wrongly_flagged}")

    def test_shape_mismatch_when_pickle_predates_feature_change(self):
        """If the pickle was trained on fewer features than the current
        ``N_FEATURES``, the audit reports SHAPE_MISMATCH rather than
        falsely flagging the missing-from-pickle slots as dead."""
        # Build a fake MLPRegressor stub with the wrong input dim.
        class _FakeMlp:
            def __init__(self, n_in: int):
                # Just enough surface area for _feature_weight_magnitudes.
                self.coefs_ = [np.ones((n_in, 8), dtype=np.float64)]

        scorer = DecisionScorer.__new__(DecisionScorer)
        scorer._model = _FakeMlp(N_FEATURES - 2)
        scorer._scaler = None
        scorer._trained = True
        scorer._n_train = 100
        scorer._pred_quantiles = None
        scorer._label_quantiles = None
        rep = dfa.audit_dead_features(scorer)
        assert rep["verdict"] == "SHAPE_MISMATCH"
        assert rep["n_features_in_pickle"] == N_FEATURES - 2
        assert rep["n_features_dead"] == 0

    def test_unknown_model_when_no_weight_matrix(self):
        """A model object with neither ``coefs_`` nor ``w_`` is flagged
        UNKNOWN_MODEL — defends against a future model swap that doesn't
        expose either attribute."""
        class _Opaque:
            pass

        scorer = DecisionScorer.__new__(DecisionScorer)
        scorer._model = _Opaque()
        scorer._scaler = None
        scorer._trained = True
        scorer._n_train = 100
        scorer._pred_quantiles = None
        scorer._label_quantiles = None
        rep = dfa.audit_dead_features(scorer)
        assert rep["verdict"] == "UNKNOWN_MODEL"
        assert rep["n_features_dead"] == 0

    def test_non_finite_weights_treated_as_dead(self):
        """NaN/Inf in coefs_[0] is itself a bug — the audit must flag any
        non-finite-weighted feature as dead rather than letting NaN
        comparisons silently slip past the eps gate."""
        class _NanMlp:
            def __init__(self):
                W = np.ones((N_FEATURES, 8), dtype=np.float64)
                W[5, :] = np.nan
                W[7, :] = np.inf
                self.coefs_ = [W]

        scorer = DecisionScorer.__new__(DecisionScorer)
        scorer._model = _NanMlp()
        scorer._scaler = None
        scorer._trained = True
        scorer._n_train = 100
        scorer._pred_quantiles = None
        scorer._label_quantiles = None
        rep = dfa.audit_dead_features(scorer)
        # Features 5 and 7 are non-finite → mean is NaN → replaced with 0
        # → ≤ eps → flagged dead. Every other feature has mean |w|=1.0,
        # which is well above eps.
        assert rep["verdict"] == "HAS_DEAD"
        dead_names = {r["feature"] for r in rep["dead_features"]}
        assert FEATURE_NAMES[5] in dead_names
        assert FEATURE_NAMES[7] in dead_names


class TestAuditDeadFeaturesNeverRaises:
    """The audit's outer try/except must catch every conceivable failure
    so a ledger consumer wired into the continuous loop can never crash
    the cycle on a degenerate model."""

    def test_audit_catches_predict_failure(self):
        """A scorer whose model raises on weight access still yields an
        ERROR envelope rather than an exception."""
        class _ExplodingModel:
            @property
            def coefs_(self):
                raise RuntimeError("boom")

        scorer = DecisionScorer.__new__(DecisionScorer)
        scorer._model = _ExplodingModel()
        scorer._scaler = None
        scorer._trained = True
        scorer._n_train = 100
        scorer._pred_quantiles = None
        scorer._label_quantiles = None
        # The _feature_weight_magnitudes() helper swallows the raise and
        # returns None, so the audit reports UNKNOWN_MODEL — exactly the
        # safer-than-crashing degradation the contract advertises.
        rep = dfa.audit_dead_features(scorer)
        assert rep["verdict"] in ("UNKNOWN_MODEL", "ERROR")
        assert rep["n_features_dead"] == 0


class TestAppendDeadFeatureAuditLog:
    """The wired ``_append_dead_feature_audit_log`` ledger must write a
    structured JSON row containing the audit's verdict and dead-feature
    summary, and survive a degenerate scorer state without crashing the
    cycle."""

    def test_ledger_writes_a_row_per_call(self, tmp_path, monkeypatch):
        """Redirect ``DEAD_FEATURE_AUDIT_LOG`` to a tmp file and confirm
        each call adds one JSONL row with the expected schema."""
        import run_continuous_backtests as rcb
        log_path = tmp_path / "dfa_log.jsonl"
        monkeypatch.setattr(rcb, "DEAD_FEATURE_AUDIT_LOG", log_path)

        ok = rcb._append_dead_feature_audit_log(
            cycle=1,
            win_start=date(2024, 1, 1),
            win_end=date(2024, 12, 31),
        )
        assert ok is True
        assert log_path.exists()
        lines = log_path.read_text().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        # Mandatory fields the ledger contract advertises.
        for key in ("cycle", "timestamp", "window_start", "window_end",
                    "verdict", "n_features_total", "n_features_dead",
                    "has_dead", "dead_features"):
            assert key in row, f"row missing required key {key!r}: {row}"
        assert row["cycle"] == 1
        assert row["window_start"] == "2024-01-01"
        assert row["window_end"] == "2024-12-31"
        # The autouse fixture's empty SCORER_PATH means the audit reads
        # NOT_TRAINED — has_dead must be False (no false alarm on cold
        # start).
        assert row["verdict"] == "NOT_TRAINED"
        assert row["has_dead"] is False

    def test_ledger_records_has_dead_when_audit_flags_features(self, tmp_path,
                                                                monkeypatch):
        """Substitute an ``audit_dead_features`` that returns HAS_DEAD and
        verify the ledger faithfully persists it. Decouples this test from
        the train_scorer end-to-end (already covered) so a regression in
        ``_append_dead_feature_audit_log`` itself can't hide behind the
        training path."""
        import run_continuous_backtests as rcb
        import paper_trader.ml.dead_feature_audit as dfa_mod

        log_path = tmp_path / "dfa_has_dead.jsonl"
        monkeypatch.setattr(rcb, "DEAD_FEATURE_AUDIT_LOG", log_path)

        def _fake_audit(*a, **k):
            return {
                "verdict": "HAS_DEAD",
                "method": "mlp_first_layer_mean_abs_weight",
                "n_train": 500,
                "n_features_total": N_FEATURES,
                "n_features_dead": 1,
                "dead_features": [
                    {"feature": "ema200_above", "mean_abs_weight": 0.0}
                ],
                "eps": 1e-6,
            }

        monkeypatch.setattr(dfa_mod, "audit_dead_features", _fake_audit)

        rcb._append_dead_feature_audit_log(
            cycle=42, win_start=date(2020, 1, 1),
            win_end=date(2021, 1, 1),
        )
        row = json.loads(log_path.read_text().splitlines()[0])
        assert row["verdict"] == "HAS_DEAD"
        assert row["has_dead"] is True
        assert row["n_features_dead"] == 1
        assert row["dead_features"][0]["feature"] == "ema200_above"

    def test_ledger_survives_audit_exception(self, tmp_path, monkeypatch):
        """A raising ``audit_dead_features`` must NOT propagate out of
        the ledger writer — the row is still appended with an ERROR
        verdict so the gap in the trend stays visible."""
        import run_continuous_backtests as rcb
        import paper_trader.ml.dead_feature_audit as dfa_mod

        log_path = tmp_path / "dfa_err.jsonl"
        monkeypatch.setattr(rcb, "DEAD_FEATURE_AUDIT_LOG", log_path)

        def _exploding_audit(*a, **k):
            raise RuntimeError("scorer corrupted")

        monkeypatch.setattr(dfa_mod, "audit_dead_features", _exploding_audit)

        ok = rcb._append_dead_feature_audit_log(
            cycle=7, win_start=date(2020, 1, 1), win_end=date(2021, 1, 1),
        )
        assert ok is True  # ledger wrote a row, even if audit failed
        row = json.loads(log_path.read_text().splitlines()[0])
        assert row["verdict"] == "ERROR"
        assert row["has_dead"] is False
        assert "error" in row

    def test_ledger_trims_when_over_cap(self, tmp_path, monkeypatch):
        """A file beyond 2× the keep cap is rewritten down to the cap.
        Mirrors the sibling-ledger trim discipline."""
        import run_continuous_backtests as rcb

        log_path = tmp_path / "dfa_trim.jsonl"
        monkeypatch.setattr(rcb, "DEAD_FEATURE_AUDIT_LOG", log_path)
        monkeypatch.setattr(rcb, "DEAD_FEATURE_AUDIT_LOG_KEEP", 5)
        # Pre-populate with 12 lines — well past 2× KEEP=10.
        log_path.write_text("\n".join(
            json.dumps({"cycle": i, "filler": True}) for i in range(12)
        ) + "\n")
        # One more append triggers the trim path.
        rcb._append_dead_feature_audit_log(
            cycle=99, win_start=date(2020, 1, 1), win_end=date(2021, 1, 1),
        )
        kept = log_path.read_text().splitlines()
        # KEEP=5 so the trimmer rewrites to the last 5 rows. The append
        # itself wrote the 13th row, after which the trim ran.
        assert len(kept) == 5
        # The freshly-appended cycle=99 row is in the tail (it was the
        # final write before trim).
        assert any(json.loads(ln).get("cycle") == 99 for ln in kept)
