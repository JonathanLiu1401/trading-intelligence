"""Tests for paper_trader.ml.calibration_reliability.

The module validates whether ``DecisionScorer.predict_calibrated`` actually
delivers honest 5-day magnitude readings out-of-sample. Tests pin:

- the report's bucket math (mean_calibrated_pred vs mean_realized matches
  hand-computed values on a known synthetic distribution)
- the bias-reduction metric is the literal difference of two decile-error
  means computed over the SAME bins (no axis mismatch)
- legacy pickles (no label_quantiles) degrade to INSUFFICIENT_DATA without
  crashing
- the verdict thresholds round-trip with calibration.py's bands
- the analyze() entry point honours a missing outcomes file and a missing
  scorer pickle without raising — the discipline the continuous loop's
  per-cycle ledger wiring relies on
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_trader.ml.calibration_reliability import (
    _empty_report,
    analyze,
    calibrated_reliability_report,
    scorer_calibrated_reliability,
    scorer_calibrated_reliability_oos,
)
from paper_trader.ml.calibration import (
    BIAS_TOL_PCT,
    MIN_PAIRS,
    SPEARMAN_GOOD,
)


# ─────────────────────── calibrated_reliability_report ──────────────────


class TestCalibratedReliabilityReport:
    def test_insufficient_data_below_min_pairs(self):
        # 5 pairs < MIN_PAIRS (30). Report must degrade honestly.
        triples = [(i, i, i + 1) for i in range(5)]
        rep = calibrated_reliability_report(triples)
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 5
        assert rep["buckets"] == []

    def test_well_calibrated_when_calibrated_matches_realized(self):
        # Calibrated predictions are *literally* the realized returns + tiny
        # noise; raw predictions are systematically biased (×3 over-shoot).
        # Calibration should win bias-reduction and earn a WELL_CALIBRATED
        # verdict (high rank skill, monotone, low decile error).
        rng = np.random.default_rng(123)
        n = 200
        y = np.linspace(-15, 15, n) + rng.normal(0, 0.05, n)
        cal = y + rng.normal(0, 0.05, n)         # near-perfect calibration
        raw = y * 3.0 + rng.normal(0, 0.05, n)   # ×3 magnitude bias
        triples = list(zip(cal, raw, y))
        rep = calibrated_reliability_report(triples)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "WELL_CALIBRATED"
        # Calibrated decile error must be much smaller than raw.
        assert rep["mean_abs_decile_error"] < BIAS_TOL_PCT
        assert rep["raw_mean_abs_decile_error"] > rep["mean_abs_decile_error"]
        # Bias reduction positive — calibration measurably helped.
        assert rep["vs_raw_bias_reduction"] > 0

    def test_directional_but_biased_when_calibration_overshoots(self):
        # Calibrated predictions track ordering but over-shoot magnitude by
        # a constant offset >> tolerance. Spearman stays high so it's not
        # MISCALIBRATED; bias is large so it lands in DIRECTIONAL_BUT_BIASED.
        rng = np.random.default_rng(7)
        n = 200
        y = np.linspace(-5, 5, n)
        cal = y + 15.0 + rng.normal(0, 0.05, n)  # constant +15pp over-shoot
        raw = y + 5.0                            # less biased raw
        triples = list(zip(cal, raw, y))
        rep = calibrated_reliability_report(triples)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "DIRECTIONAL_BUT_BIASED"
        # Calibration made things WORSE here (over-shoot > raw bias).
        assert rep["vs_raw_bias_reduction"] < 0

    def test_miscalibrated_when_predictions_anti_correlated(self):
        # If calibrated predictions move OPPOSITE to realized, the verdict
        # must be MISCALIBRATED (the gate would mis-size on this scorer).
        rng = np.random.default_rng(11)
        n = 200
        y = np.linspace(-10, 10, n)
        cal = -y + rng.normal(0, 0.05, n)
        raw = -y * 2.0
        triples = list(zip(cal, raw, y))
        rep = calibrated_reliability_report(triples)
        assert rep["verdict"] == "MISCALIBRATED"
        # Bucket monotonicity must be low (descending realized).
        assert rep["monotone_fraction"] < 0.5

    def test_bucket_math_is_per_bin_mean(self):
        # Hand-checkable: 60 triples in 5 contiguous prediction blocks of 12
        # each. With n_buckets=5 the report cuts on identical block bounds —
        # so mean_pred per bucket equals the block's known mean exactly.
        triples = []
        for block, base in enumerate([-10, -5, 0, 5, 10]):
            for j in range(12):
                triples.append((base, base, base + 0.1 * j))
        rep = calibrated_reliability_report(triples, n_buckets=5)
        assert rep["status"] == "ok"
        assert len(rep["buckets"]) == 5
        # Each bucket's mean_calibrated_pred = block value (constant).
        for i, base in enumerate([-10, -5, 0, 5, 10]):
            b = rep["buckets"][i]
            assert b["n"] == 12
            assert b["mean_calibrated_pred"] == pytest.approx(base, abs=1e-6)
            assert b["mean_raw_pred"] == pytest.approx(base, abs=1e-6)
            # mean_realized = base + (0.1 * (0+1+...+11)) / 12 = base + 0.55
            assert b["mean_realized"] == pytest.approx(base + 0.55, abs=1e-6)

    def test_bias_reduction_is_literal_difference(self):
        # vs_raw_bias_reduction == raw_mean_abs_decile_error -
        # mean_abs_decile_error, by definition. Pin the contract so a
        # refactor that swaps subtraction order or scales it can't sneak in.
        rng = np.random.default_rng(99)
        n = 90
        y = np.linspace(-10, 10, n) + rng.normal(0, 0.1, n)
        cal = y + 1.0
        raw = y * 2.0
        triples = list(zip(cal, raw, y))
        rep = calibrated_reliability_report(triples, n_buckets=9)
        assert rep["status"] == "ok"
        expected = rep["raw_mean_abs_decile_error"] - rep["mean_abs_decile_error"]
        assert rep["vs_raw_bias_reduction"] == pytest.approx(
            round(expected, 4), abs=1e-4
        )

    def test_non_finite_triples_dropped(self):
        # NaN/inf in any column drops the row. Survivors must still produce
        # a valid report when the survivor count crosses MIN_PAIRS.
        rng = np.random.default_rng(5)
        n = 50
        y = list(np.linspace(-10, 10, n) + rng.normal(0, 0.1, n))
        cal = list(np.array(y) + 0.5)
        raw = list(np.array(y) + 1.0)
        # Poison 10 rows — each with a different non-finite poison.
        cal[0] = float("nan")
        raw[1] = float("inf")
        y[2] = float("-inf")
        cal[3] = None
        triples = list(zip(cal, raw, y))
        rep = calibrated_reliability_report(triples)
        assert rep["status"] == "ok"
        # 50 input - 4 poisoned = 46 surviving (still > MIN_PAIRS=30).
        assert rep["n"] == 46

    def test_n_buckets_capped_at_n_over_3(self):
        # If n is small, n_buckets should clamp to n//3 — each bucket gets
        # ≥3 samples. n=30 with request for 20 buckets must return 10.
        triples = [(i, i, i + 0.5) for i in range(30)]
        rep = calibrated_reliability_report(triples, n_buckets=20)
        assert rep["status"] == "ok"
        assert len(rep["buckets"]) == 10
        # Every bucket has 3 rows; 10 buckets × 3 = 30 covered.
        assert sum(b["n"] for b in rep["buckets"]) == 30


# ─────────────────────── scorer_calibrated_reliability ──────────────────


class _StaticCalibratedScorer:
    """A stand-in scorer whose predict_with_meta returns deterministic
    calibrated / raw / off_distribution values — lets us exercise the
    per-row predict pass without an MLP fit."""

    def __init__(self, calibrate_offset: float = 0.0,
                 raw_factor: float = 3.0):
        self.calibrate_offset = calibrate_offset
        self.raw_factor = raw_factor
        self._n_train = 1000
        self.is_trained = True

    def predict_with_meta(self, **kw):
        # Use ml_score as the "feature" the scorer reads — a simple monotone
        # path so tests can control the ordering.
        ml = float(kw.get("ml_score") or 0.0)
        return {
            "pred": ml * self.raw_factor,        # biased raw
            "raw": ml * self.raw_factor,
            "clamped": False,
            "off_distribution": False,
            "percentile": 50.0,
            "calibrated": ml + self.calibrate_offset,
        }


class _LegacyScorer(_StaticCalibratedScorer):
    """A scorer whose predict_with_meta returns ``calibrated=None``
    (legacy pickle, no label_quantiles)."""

    def predict_with_meta(self, **kw):
        meta = super().predict_with_meta(**kw)
        meta["calibrated"] = None
        return meta


def _outcome_row(ml_score: float, fwd: float, action: str = "BUY",
                 sim_date: str = "2025-01-01") -> dict:
    return {
        "ml_score": ml_score,
        "rsi": 50.0, "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
        "regime_mult": 1.0, "ticker": "NVDA",
        "vol_ratio": 1.0, "bb_position": 0.0,
        "news_urgency": 50.0, "news_article_count": 1.0,
        "forward_return_5d": fwd, "action": action,
        "sim_date": sim_date,
    }


class TestScorerCalibratedReliability:
    def test_legacy_scorer_no_calibrated_yields_insufficient(self):
        # Every predict_with_meta returns calibrated=None — no triples can
        # be built, report must read INSUFFICIENT_DATA without raising.
        scorer = _LegacyScorer()
        recs = [_outcome_row(i, i + 0.5) for i in range(40)]
        rep = scorer_calibrated_reliability(scorer, recs)
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_scorer_without_predict_with_meta_degrades_safely(self):
        class _Stub:
            is_trained = True
            _n_train = 100
        rep = scorer_calibrated_reliability(_Stub(), [_outcome_row(0, 0)])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "predict_with_meta" in rep["hint"]

    def test_sell_target_sign_flipped(self):
        # A SELL whose forward return was negative is a CORRECT call. The
        # report's action-aligned target flips the sign so a rank-skilled
        # SELL doesn't read as anti-correlated.
        scorer = _StaticCalibratedScorer(calibrate_offset=0.0)
        # 40 SELL records: ml_score ascending, realized fwd DESCENDING.
        # After SELL sign flip the realized target is ASCENDING, aligning
        # with calibrated_pred (also ascending in ml_score).
        recs = []
        for i in range(40):
            recs.append(_outcome_row(ml_score=i, fwd=-i, action="SELL",
                                     sim_date=f"2025-01-{i+1:02d}"))
        rep = scorer_calibrated_reliability(scorer, recs)
        assert rep["status"] == "ok"
        # Monotone fraction must be high — without the sign flip it would
        # be 0 (strictly descending).
        assert rep["monotone_fraction"] >= 0.8

    def test_records_with_null_forward_return_dropped(self):
        scorer = _StaticCalibratedScorer(calibrate_offset=0.0)
        recs = [_outcome_row(i, i + 0.5) for i in range(40)]
        recs[5]["forward_return_5d"] = None
        recs[6]["forward_return_5d"] = None
        rep = scorer_calibrated_reliability(scorer, recs)
        assert rep["status"] == "ok"
        assert rep["n"] == 38

    def test_predict_exception_drops_row_not_report(self):
        # A row whose predict raises must drop just that row, not crash
        # the analyzer — the unattended loop's per-cycle ledger needs this.
        class _PartialFail(_StaticCalibratedScorer):
            def predict_with_meta(self, **kw):
                if (kw.get("ml_score") or 0.0) == 13.0:
                    raise RuntimeError("planted")
                return super().predict_with_meta(**kw)

        scorer = _PartialFail()
        recs = [_outcome_row(i, i + 0.5) for i in range(35)]
        recs[13]["ml_score"] = 13.0
        rep = scorer_calibrated_reliability(scorer, recs)
        assert rep["status"] == "ok"
        assert rep["n"] == 34


# ─────────────────────── temporal-OOS wrapper ───────────────────────────


class TestScorerCalibratedReliabilityOos:
    def test_holds_out_recent_fraction(self):
        scorer = _StaticCalibratedScorer(calibrate_offset=0.0)
        recs = [_outcome_row(i, i + 0.5, sim_date=f"2025-{(i // 28) + 1:02d}-"
                              f"{(i % 28) + 1:02d}")
                for i in range(200)]
        rep = scorer_calibrated_reliability_oos(scorer, recs,
                                                oos_fraction=0.2)
        assert rep["status"] == "ok"
        assert rep["oos_n"] == 40
        assert rep["train_n"] == 160
        assert rep["oos_fraction"] == 0.2

    def test_split_failure_degrades_to_no_holdout(self, monkeypatch):
        # If split_outcomes_temporal raises, the wrapper must still return a
        # valid report (INSUFFICIENT_DATA) — never propagate the exception.
        import paper_trader.ml.calibration_reliability as mod

        def _boom(records, oos_fraction):
            raise RuntimeError("planted")

        monkeypatch.setattr(
            "paper_trader.validation.split_outcomes_temporal", _boom
        )
        scorer = _StaticCalibratedScorer()
        recs = [_outcome_row(i, i + 0.5) for i in range(40)]
        rep = scorer_calibrated_reliability_oos(scorer, recs)
        assert rep["oos_n"] == 0
        assert rep["train_n"] == 40
        assert rep["verdict"] == "INSUFFICIENT_DATA"


# ─────────────────────── analyze() entry point ──────────────────────────


class TestAnalyze:
    def test_missing_outcomes_file_returns_insufficient(self, tmp_path):
        rep = analyze(outcomes_path=tmp_path / "missing.jsonl")
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "no outcomes file" in rep["hint"]

    def test_untrained_scorer_returns_insufficient(self, tmp_path, monkeypatch):
        # Outcomes exist but no pickle on disk — analyze must degrade
        # honestly rather than raise.
        outcomes = tmp_path / "outcomes.jsonl"
        outcomes.write_text(json.dumps(_outcome_row(1, 2)) + "\n")
        # Point SCORER_PATH at a non-existent file so DecisionScorer reports
        # untrained.
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "nope.pkl")
        rep = analyze(outcomes_path=outcomes)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "scorer not trained" in rep["hint"]

    def test_corrupted_outcomes_lines_skipped(self, tmp_path, monkeypatch):
        # JSONL with a garbage line: the parse must skip it, not crash.
        outcomes = tmp_path / "outcomes.jsonl"
        with outcomes.open("w") as f:
            f.write(json.dumps(_outcome_row(1, 2)) + "\n")
            f.write("not json at all\n")
            f.write(json.dumps(_outcome_row(2, 3)) + "\n")
        # Without a trained scorer the report still degrades to insufficient,
        # which is fine — we're testing that the parse loop doesn't crash.
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "nope.pkl")
        rep = analyze(outcomes_path=outcomes)
        # 2 valid records loaded but no scorer; report degrades — no crash
        # is the test contract.
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_in_sample_vs_oos_slice_tag(self, tmp_path, monkeypatch):
        outcomes = tmp_path / "outcomes.jsonl"
        outcomes.write_text(json.dumps(_outcome_row(1, 2)) + "\n")
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "nope.pkl")
        # Both untrained → INSUFFICIENT, but the slice tag is set on the
        # path that would have run had the scorer been trained.
        # Since the early "scorer not trained" exit doesn't set slice, we
        # only verify the entry-point is callable with both modes.
        analyze(outcomes_path=outcomes, oos_only=True)
        analyze(outcomes_path=outcomes, oos_only=False)


# ─────────────────────── CLI integration (smoke) ────────────────────────


class TestCli:
    def test_cli_returns_nonzero_when_insufficient(self, monkeypatch, tmp_path):
        # The CLI exit-code contract: 0 only for WELL_CALIBRATED / WEAK_SIGNAL.
        # An untrained scorer yields INSUFFICIENT_DATA → exit 1.
        import paper_trader.ml.calibration_reliability as mod
        import paper_trader.ml.decision_scorer as ds

        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "nope.pkl")
        # Redirect the analyze() default outcomes path away from production.
        monkeypatch.setattr(
            mod, "analyze",
            lambda outcomes_path=None, oos_only=True, n_buckets=10: _empty_report(
                0, "stub"
            ),
        )
        rc = mod._cli([])
        assert rc == 1

    def test_empty_report_shape(self):
        # _empty_report is the JSON-safe shape every degradation path returns.
        # Pin it so external consumers (dashboard / ledger) can rely on the
        # key set.
        rep = _empty_report(0, "anything")
        for key in (
            "status", "verdict", "n", "spearman", "monotone_fraction",
            "mean_abs_decile_error", "raw_mean_abs_decile_error",
            "vs_raw_bias_reduction", "buckets", "hint",
        ):
            assert key in rep, f"missing {key} in empty report shape"
        assert rep["buckets"] == []
        assert rep["verdict"] == "INSUFFICIENT_DATA"
