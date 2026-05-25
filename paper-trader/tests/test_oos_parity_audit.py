"""Tests for paper_trader.ml.oos_parity_audit.

The audit quantifies the bias the pass #36 OOS-feature-parity fix
removes. These tests pin: every verdict-ladder path, the bias direction
when enhanced MACD signal is informative, and the never-raise contract
on degenerate inputs (untrained scorer, empty corpus, all-None
enhanced-MACD rows, predict failures).
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from paper_trader.ml import oos_parity_audit as ops
from paper_trader.ml.decision_scorer import (
    DecisionScorer,
    train_scorer,
)


# ───────────────────────── helpers ──────────────────────────

def _row(forward_return_5d=5.0, ema200_above=True, hist_cross_up=False,
         macd_below_zero_cross=False, action="BUY", ticker="NVDA",
         ml_score=1.0, **extra):
    """Build a minimal outcome row with the fields ``audit_oos_parity``
    inspects. Defaults match the real outcome shape so omitting a field
    in a test means "use the default the audit would see in production".
    """
    base = {
        "ticker": ticker, "action": action,
        "ml_score": ml_score,
        "rsi": 50.0, "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
        "regime_mult": 1.0,
        "vol_ratio": 1.0, "bb_position": 0.0,
        "news_urgency": 50.0, "news_article_count": 1.0,
        "ema200_above": ema200_above,
        "hist_cross_up": hist_cross_up,
        "macd_below_zero_cross": macd_below_zero_cross,
        "forward_return_5d": forward_return_5d,
    }
    base.update(extra)
    return base


def _train_scorer_where_ema200_drives_label(tmpdir) -> Path:
    """Train a tiny scorer on a synthetic corpus where ``ema200_above``
    is the ONLY informative signal: rows with True realized +5%, False
    realized -5%, everything else held constant. The trained scorer's
    weights for ema200_above land non-zero by construction, so the
    forwarded-vs-degraded prediction diverges on every row — exactly the
    case the audit's BIAS_LARGE verdict should fire for. 30 rows = the
    train_scorer `len(records) < 30` floor."""
    records = []
    for i in range(15):
        records.append(_row(
            forward_return_5d=5.0, ema200_above=True,
            ticker="NVDA", action="BUY",
            sim_date=f"2025-01-{i+1:02d}"))
    for i in range(15):
        records.append(_row(
            forward_return_5d=-5.0, ema200_above=False,
            ticker="NVDA", action="BUY",
            sim_date=f"2025-02-{i+1:02d}"))
    tmp_pkl = tmpdir / "scorer.pkl"
    result = train_scorer(records, path=tmp_pkl)
    assert result["status"] == "ok", result
    return tmp_pkl


def _load_throwaway_scorer(tmp_pkl: Path) -> DecisionScorer:
    """Redirect ``SCORER_PATH`` so DecisionScorer loads the throwaway
    pickle without touching the live deployed file."""
    from paper_trader.ml import decision_scorer as _ds
    _ds.SCORER_PATH = tmp_pkl
    return _ds.DecisionScorer()


# ───────────────────────── verdict ladder ─────────────────────────

class TestVerdictLadder:
    """One test per ladder verdict — they must be distinguishable."""

    def test_not_trained_returns_not_trained_verdict(self):
        class _Untrained:
            is_trained = False
            n_train = 0
        out = ops.audit_oos_parity(
            [_row()], scorer=_Untrained())
        assert out["verdict"] == "NOT_TRAINED"

    def test_no_records_returns_no_data(self):
        class _Trained:
            is_trained = True
            n_train = 5000

            def predict_with_meta(self, **_kw):
                return {"pred": 0.0, "raw": 0.0, "clamped": False,
                        "off_distribution": False, "failed": False}

        out = ops.audit_oos_parity([], scorer=_Trained())
        assert out["verdict"] == "NO_DATA"
        assert out["n_records"] == 0

    def test_records_with_no_enhanced_signal_returns_no_parity_features(
            self):
        """When every row has ALL enhanced MACD fields = None, both paths
        feed identical inputs — the audit must report that distinct
        state, not BIAS_SMALL (which would imply genuine zero bias)."""
        class _Trained:
            is_trained = True
            n_train = 5000

            def predict_with_meta(self, **_kw):
                return {"pred": 1.0, "raw": 1.0, "clamped": False,
                        "off_distribution": False, "failed": False}

        records = [_row(ema200_above=None, hist_cross_up=None,
                        macd_below_zero_cross=None) for _ in range(5)]
        out = ops.audit_oos_parity(records, scorer=_Trained())
        assert out["verdict"] == "NO_PARITY_FEATURES"
        # Diagnostic must still be honest about why
        assert out["n_records"] == 5
        assert out["n_with_any_enhanced"] == 0

    def test_bias_small_when_paths_match_within_tolerance(self):
        """A scorer that returns the same prediction regardless of which
        kwargs reach it produces zero divergence — BIAS_SMALL by
        definition. Locks the band threshold contract."""
        class _Constant:
            is_trained = True
            n_train = 5000

            def predict_with_meta(self, **_kw):
                return {"pred": 0.5, "raw": 0.5, "clamped": False,
                        "off_distribution": False, "failed": False}

        records = [_row(forward_return_5d=fr, ema200_above=True)
                   for fr in (1.0, -1.0, 2.0, -2.0, 3.0)]
        out = ops.audit_oos_parity(records, scorer=_Constant())
        assert out["verdict"] == "BIAS_SMALL"
        # The constant scorer makes both RMSEs identical and rank-IC
        # undefined (constant prediction) — but the audit still
        # produces a complete envelope.
        assert out["n_with_any_enhanced"] == 5
        assert out["mean_abs_pred_diff_pp"] == 0.0
        assert out["max_abs_pred_diff_pp"] == 0.0

    def test_bias_large_when_ema200_drives_label_real_train(self):
        """End-to-end pin: train a real scorer where ema200_above carries
        the entire signal, then audit. The corrected path must report a
        meaningfully better metric than the degraded path — BIAS_LARGE.
        This proves the audit catches the exact class of bug pass #36
        fixed."""
        with tempfile.TemporaryDirectory() as td:
            tmp_pkl = _train_scorer_where_ema200_drives_label(Path(td))
            scorer = _load_throwaway_scorer(tmp_pkl)
            # OOS-style records: half True / half False on ema200_above
            # AND vary another feature (ml_score) so the degraded path
            # still produces variation (otherwise rank-IC of the degraded
            # path is undefined — all features identical, all preds
            # identical). What we're pinning is that BOTH paths produce
            # valid predictions and the corrected path is materially
            # better on the magnitude axis.
            records = []
            cases = [
                (5.0, True, 1.0), (5.0, True, 2.0), (5.0, True, 3.0),
                (-5.0, False, 1.0), (-5.0, False, 2.0), (-5.0, False, 3.0),
                (4.0, True, 1.5), (-4.0, False, 1.5),
            ]
            for fr, ema, msc in cases:
                records.append(_row(forward_return_5d=fr,
                                    ema200_above=ema,
                                    hist_cross_up=False,
                                    macd_below_zero_cross=False,
                                    ml_score=msc))
            out = ops.audit_oos_parity(records, scorer=scorer)
            assert out["verdict"] in ("BIAS_LARGE", "BIAS_MODERATE"), out
            # Direction check: the corrected path uses the true signal,
            # so RMSE should be LOWER (more negative delta).
            assert out["delta_rmse_pp"] is not None
            assert out["delta_rmse_pp"] < 0, out
            # The per-row divergence is the prediction shift between
            # ema200_above=True (~+5%) and =None→0.0 (~-5%) — must
            # be at least a couple pp.
            assert out["max_abs_pred_diff_pp"] > 1.0, out


# ───────────────────────── envelope honesty ─────────────────────────

class TestEnvelopeHonesty:
    """The audit must never raise and must always return a complete
    JSON-safe envelope with the documented keys."""

    REQUIRED_KEYS = {
        "verdict", "method", "n_records", "n_with_any_enhanced",
        "n_predict_failures", "rmse_parity", "rmse_degraded",
        "delta_rmse_pp", "rank_ic_parity", "rank_ic_degraded",
        "delta_rank_ic", "mean_abs_pred_diff_pp", "max_abs_pred_diff_pp",
        "eps_rmse", "eps_rank_ic",
    }

    def test_envelope_keys_present_for_no_data(self):
        class _Trained:
            is_trained = True
            n_train = 100

            def predict_with_meta(self, **_kw):
                return {"pred": 0.0, "raw": 0.0, "clamped": False,
                        "off_distribution": False, "failed": False}

        out = ops.audit_oos_parity([], scorer=_Trained())
        assert self.REQUIRED_KEYS.issubset(out.keys())

    def test_envelope_keys_present_for_not_trained(self):
        class _Untrained:
            is_trained = False
            n_train = 0
        out = ops.audit_oos_parity([_row()], scorer=_Untrained())
        assert self.REQUIRED_KEYS.issubset(out.keys())

    def test_audit_never_raises_on_scorer_exception(self):
        """A scorer that raises on every predict must NOT bubble — the
        audit drops failing rows and reports an honest envelope."""
        class _Raising:
            is_trained = True
            n_train = 100

            def predict_with_meta(self, **_kw):
                raise RuntimeError("simulated transient predict fault")

        out = ops.audit_oos_parity(
            [_row() for _ in range(3)], scorer=_Raising())
        # All 3 rows fail — verdict degrades to NO_DATA (no usable predictions)
        assert out["verdict"] == "NO_DATA"
        assert out["n_predict_failures"] == 3

    def test_audit_handles_non_finite_actual_by_dropping_row(self):
        """A row with NaN forward_return_5d cannot contribute to RMSE/IC
        — it must be silently dropped, not poison the metric or crash."""
        class _Trained:
            is_trained = True
            n_train = 100

            def predict_with_meta(self, **_kw):
                return {"pred": 1.0, "raw": 1.0, "clamped": False,
                        "off_distribution": False, "failed": False}

        records = [
            _row(forward_return_5d=float("nan"), ema200_above=True),
            _row(forward_return_5d=1.0, ema200_above=False),
            _row(forward_return_5d=-1.0, ema200_above=True),
        ]
        out = ops.audit_oos_parity(records, scorer=_Trained())
        # The audit ran on the 2 usable rows + 1 dropped row. Verdict
        # is well-formed (any of the bias-* tiers; constant predictor
        # produces 0 divergence so it's BIAS_SMALL).
        assert out["verdict"] in ("BIAS_SMALL", "BIAS_MODERATE",
                                  "BIAS_LARGE")
        assert out["n_records"] == 3
        # n_predict_failures here is 0 — failure is "predict raised"
        # not "actual was NaN" (that's a silent drop, not a failure).
        assert out["n_predict_failures"] == 0


# ───────────────────────── parity-feature detection ─────────────────────────

class TestRowEnhancedMacdDetection:
    """``_row_has_enhanced_macd`` decides whether a row even has the
    *potential* for path divergence. Pin its contract."""

    def test_all_none_returns_false(self):
        assert not ops._row_has_enhanced_macd({
            "ema200_above": None, "hist_cross_up": None,
            "macd_below_zero_cross": None})

    def test_missing_keys_treated_as_none(self):
        assert not ops._row_has_enhanced_macd({})

    def test_at_least_one_non_none_true(self):
        for key in ("ema200_above", "hist_cross_up", "macd_below_zero_cross"):
            row = {"ema200_above": None, "hist_cross_up": None,
                   "macd_below_zero_cross": None}
            row[key] = True
            assert ops._row_has_enhanced_macd(row)

    def test_false_value_counts_as_present(self):
        """False is a meaningful signal (the model can distinguish it
        from None → 0.0 in training, even though both inference-defaults
        produce 0.0). The audit must still flag the row as parity-
        relevant — otherwise the "no_parity_features" verdict would
        misfire for a corpus where every row has False."""
        assert ops._row_has_enhanced_macd({
            "ema200_above": False,
            "hist_cross_up": None,
            "macd_below_zero_cross": None})


# ───────────────────────── corpus loading ─────────────────────────

class TestLoadRecords:
    def test_returns_empty_for_missing_path(self, tmp_path):
        out = ops._load_records(tmp_path / "absent.jsonl")
        assert out == []

    def test_loads_jsonl_file(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        rows = [
            {"ticker": "NVDA", "forward_return_5d": 1.0},
            {"ticker": "AMD", "forward_return_5d": -1.0},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        loaded = ops._load_records(path)
        assert loaded == rows

    def test_tail_keeps_only_last_n(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        rows = [{"i": i} for i in range(10)]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        loaded = ops._load_records(path, tail=3)
        assert loaded == [{"i": 7}, {"i": 8}, {"i": 9}]

    def test_malformed_line_dropped_silently(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        path.write_text(
            '{"ticker": "NVDA"}\n'
            'this is not json\n'
            '{"ticker": "AMD"}\n'
        )
        loaded = ops._load_records(path)
        assert len(loaded) == 2


# ───────────────────────── CLI ─────────────────────────

class TestCli:
    def test_json_output_includes_verdict(self, tmp_path, capsys):
        path = tmp_path / "outcomes.jsonl"
        path.write_text(json.dumps(_row()) + "\n")

        class _Untrained:
            is_trained = False
            n_train = 0

        with patch.object(ops, "DecisionScorer", return_value=_Untrained()):
            rc = ops.main(["--outcomes", str(path), "--json"])
        # NOT_TRAINED exits 0
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["verdict"] == "NOT_TRAINED"

    def test_cli_exit_code_1_when_bias_large(self, tmp_path):
        """End-to-end CLI: a real trained scorer where ema200_above
        drives the label, exercised through main(), must exit non-zero
        — that's how a shell script gates on the audit."""
        with tempfile.TemporaryDirectory() as td:
            tmp_pkl = _train_scorer_where_ema200_drives_label(Path(td))
            from paper_trader.ml import decision_scorer as _ds
            _ds.SCORER_PATH = tmp_pkl

            outcomes_path = tmp_path / "outcomes.jsonl"
            rows = []
            for fr, ema in (
                (5.0, True), (5.0, True), (-5.0, False),
                (-5.0, False), (4.0, True), (-4.0, False),
            ):
                rows.append(_row(forward_return_5d=fr, ema200_above=ema))
            outcomes_path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n")

            rc = ops.main(["--outcomes", str(outcomes_path), "--json"])
            # BIAS_LARGE / BIAS_MODERATE both exit 1.
            assert rc == 1
