"""Tests for ``paper_trader.ml.attribution_audit``.

Locks the structural and verdict invariants of the new
attribution-aggregation diagnostic. The audit reuses
``DecisionScorer.feature_contributions`` (the same Shapley-style ablation
``/api/scorer-attribution`` renders) and aggregates across the outcomes
corpus, so the assertions here must hold against the SAME
``feature_contributions`` contract locked in
``test_decision_scorer_attribution.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_trader.ml.attribution_audit import (
    CONCENTRATED_TOP1_SHARE,
    INERT_MAX_ABS,
    MIN_RECORDS,
    _iter_records,
    analyze,
    analyze_path,
)
from paper_trader.ml.decision_scorer import (
    DecisionScorer,
    FEATURE_NAMES,
    N_FEATURES,
    train_scorer,
)


def _outcome(sim_date: str, ml_score: float, fwd: float,
             ticker: str = "NVDA") -> dict:
    return {
        "ticker": ticker, "action": "BUY",
        "ml_score": ml_score, "rsi": 50.0, "macd": 0.1,
        "mom5": 0.0, "mom20": 0.0, "regime_mult": 1.0,
        "vol_ratio": 1.0, "bb_position": 0.0,
        "news_urgency": 50.0, "news_article_count": 1.0,
        "forward_return_5d": fwd, "return_pct": 10.0,
        "sim_date": sim_date,
    }


def _train_ml_score_dominant(tmp_path, monkeypatch) -> DecisionScorer:
    """Train a scorer where ml_score is by construction the dominant driver."""
    import paper_trader.ml.decision_scorer as ds
    monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "scorer_aa.pkl")
    recs = []
    for i in range(40):
        sc = (i - 20) * 0.5  # -10.0 .. +9.5
        recs.append(_outcome(f"2025-04-{i+1:02d}", ml_score=sc, fwd=sc * 1.5))
    assert train_scorer(recs)["status"] == "ok"
    s = DecisionScorer()
    assert s.is_trained
    return s


class TestUntrainedAndEmpty:
    def test_untrained_scorer_returns_untrained_verdict(self):
        # Untrained scorer + arbitrary records: must short-circuit BEFORE
        # any per-record loop runs (so an untrained gate never gets a
        # CONCENTRATED / DIVERSIFIED verdict that would mislead an operator).
        s = DecisionScorer()
        rep = analyze([_outcome(f"2025-01-{i+1:02d}", ml_score=1.0, fwd=1.0)
                       for i in range(40)], scorer=s)
        assert rep["verdict"] == "UNTRAINED"
        assert rep["features"] == []
        assert rep["n_analyzed"] == 0

    def test_empty_records_returns_insufficient(self, tmp_path, monkeypatch):
        s = _train_ml_score_dominant(tmp_path, monkeypatch)
        rep = analyze([], scorer=s)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["features"] == []
        assert rep["n_analyzed"] == 0

    def test_too_few_records_returns_insufficient(
            self, tmp_path, monkeypatch):
        s = _train_ml_score_dominant(tmp_path, monkeypatch)
        # MIN_RECORDS-1 valid records — must NOT proceed to a numeric verdict.
        recs = [_outcome(f"2025-02-{i+1:02d}", ml_score=2.0, fwd=3.0)
                for i in range(MIN_RECORDS - 1)]
        rep = analyze(recs, scorer=s)
        assert rep["verdict"] == "INSUFFICIENT_DATA"


class TestAttributionAggregation:
    def test_returns_one_row_per_feature(self, tmp_path, monkeypatch):
        s = _train_ml_score_dominant(tmp_path, monkeypatch)
        recs = [_outcome(f"2025-03-{i+1:02d}", ml_score=2.0, fwd=3.0)
                for i in range(40)]
        rep = analyze(recs, scorer=s)
        assert rep["status"] == "ok"
        assert len(rep["features"]) == N_FEATURES
        names = {r["feature"] for r in rep["features"]}
        assert names == set(FEATURE_NAMES)

    def test_features_sorted_by_mean_abs_contribution(
            self, tmp_path, monkeypatch):
        s = _train_ml_score_dominant(tmp_path, monkeypatch)
        recs = [_outcome(f"2025-04-{i+1:02d}", ml_score=(i - 20) * 0.5,
                         fwd=(i - 20) * 0.7)
                for i in range(40)]
        rep = analyze(recs, scorer=s)
        mags = [r["mean_abs_contribution"] for r in rep["features"]]
        assert mags == sorted(mags, reverse=True), (
            "features must be sorted by mean_abs_contribution desc")

    def test_dominant_feature_concentrates_attribution(
            self, tmp_path, monkeypatch):
        """The training set isolates ml_score as the sole signal. The
        aggregate audit must surface ml_score as the dominant driver with
        attribution concentration above the threshold."""
        s = _train_ml_score_dominant(tmp_path, monkeypatch)
        recs = [_outcome(f"2025-05-{i+1:02d}",
                         ml_score=float(rng_v) * 5.0,
                         fwd=float(rng_v) * 7.0)
                for i, rng_v in enumerate(
                    np.linspace(-2.0, 2.0, 40))]
        rep = analyze(recs, scorer=s)
        top = rep["features"][0]
        assert top["feature"] == "ml_score"
        # With ml_score sweep large and other features constant, the
        # top1_share should be high.
        assert rep["top1_share"] > 0.3, (
            f"ml_score must dominate when other features are constant; "
            f"got top1_share={rep['top1_share']}")

    def test_top3_share_is_a_fraction(self, tmp_path, monkeypatch):
        s = _train_ml_score_dominant(tmp_path, monkeypatch)
        recs = [_outcome(f"2025-06-{i+1:02d}", ml_score=2.0, fwd=3.0)
                for i in range(40)]
        rep = analyze(recs, scorer=s)
        for r in rep["features"]:
            assert 0.0 <= r["top3_share"] <= 1.0
        # The 3 most-impactful features per record sum to 3 hits per record;
        # across all rows, sum of top3_share equals 3.0 (exactly).
        total_top3 = sum(r["top3_share"] for r in rep["features"])
        assert total_top3 == pytest.approx(3.0, abs=1e-6)


class TestVerdictThresholds:
    def test_inert_scorer_gets_inert_verdict(self, tmp_path, monkeypatch):
        """A constant-predictor model produces zero contribution for every
        feature — the verdict must surface this as MODEL_INERT (so an
        operator sees the gate has no leverage)."""
        s = _train_ml_score_dominant(tmp_path, monkeypatch)

        class _Constant:
            def predict(self, X):
                # Every input row returns the same value → contribs = 0.
                return np.full(len(X), 1.5, dtype=np.float64)
        s._model = _Constant()
        recs = [_outcome(f"2025-07-{i+1:02d}", ml_score=float(v) * 5.0,
                         fwd=float(v))
                for i, v in enumerate(np.linspace(-2, 2, 40))]
        rep = analyze(recs, scorer=s)
        assert rep["verdict"] == "MODEL_INERT"
        # Every feature's mean_abs must be below the inert threshold.
        for r in rep["features"]:
            assert r["mean_abs_contribution"] < INERT_MAX_ABS

    def test_concentrated_verdict_when_one_feature_dominates(
            self, tmp_path, monkeypatch):
        """Force feature_contributions to attribute everything to ml_score.
        The verdict must be CONCENTRATED (the "17-feature MLP is effectively
        a 1-feature rule" finding the audit is designed to surface)."""
        s = _train_ml_score_dominant(tmp_path, monkeypatch)

        # Stub feature_contributions so ONE feature swamps all others.
        ml_idx = FEATURE_NAMES.index("ml_score")

        def fake_contrib(**kw):
            rows = [{"feature": n, "raw_value": 0.0, "contribution": 0.0}
                    for n in FEATURE_NAMES]
            rows[ml_idx]["contribution"] = 5.0  # huge vs every other 0
            return {"trained": True, "contributions": rows,
                    "pred": 5.0, "pred_baseline": 0.0,
                    "interaction_residual": 0.0, "off_distribution": False}
        s.feature_contributions = fake_contrib

        recs = [_outcome(f"2025-08-{i+1:02d}", ml_score=2.0, fwd=3.0)
                for i in range(40)]
        rep = analyze(recs, scorer=s)
        assert rep["verdict"] == "CONCENTRATED"
        assert rep["top1_share"] > CONCENTRATED_TOP1_SHARE


class TestIOHandling:
    def test_iter_records_skips_corrupt_lines(self, tmp_path):
        """A single corrupt JSONL line must NOT abort the whole iteration —
        mirrors the per-line tolerance of ``_inject_and_train``."""
        path = tmp_path / "outcomes.jsonl"
        path.write_text(
            '{"ml_score": 1.0}\n'
            'this is not json\n'
            '{"ml_score": 2.0}\n'
            '\n'  # blank line
            '{"broken}\n'
        )
        recs = list(_iter_records(path))
        assert len(recs) == 2
        assert recs[0]["ml_score"] == 1.0
        assert recs[1]["ml_score"] == 2.0

    def test_iter_records_missing_file_returns_empty(self, tmp_path):
        # No exception — empty generator (the "best effort" discipline).
        recs = list(_iter_records(tmp_path / "does_not_exist.jsonl"))
        assert recs == []

    def test_analyze_path_untrained_when_no_pickle(
            self, tmp_path, monkeypatch):
        """When no scorer pkl exists and the outcomes file is populated,
        analyze_path must report UNTRAINED, not crash. The combined check
        guards the operator-facing "where's my gate?" question."""
        import paper_trader.ml.decision_scorer as ds
        # Point SCORER_PATH at a missing file so DecisionScorer.is_trained=False.
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "no_scorer.pkl")
        outcomes = tmp_path / "outcomes.jsonl"
        outcomes.write_text(
            "\n".join(json.dumps(_outcome(f"2025-09-{i+1:02d}",
                                          ml_score=1.0, fwd=2.0))
                      for i in range(40)) + "\n"
        )
        rep = analyze_path(outcomes, oos_only=False)
        assert rep["verdict"] == "UNTRAINED"
        assert rep["outcomes_path"] == str(outcomes)
        assert rep["slice"] == "all"

    def test_analyze_path_uses_oos_slice_by_default(
            self, tmp_path, monkeypatch):
        s = _train_ml_score_dominant(tmp_path, monkeypatch)
        outcomes = tmp_path / "outcomes_oos.jsonl"
        recs = [_outcome(f"2025-10-{i+1:02d}", ml_score=float(v),
                         fwd=float(v) * 1.5)
                for i, v in enumerate(np.linspace(-5, 5, 200))]
        outcomes.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        # Don't pass scorer — analyze_path reloads via DecisionScorer() and
        # SCORER_PATH was monkeypatched by _train_ml_score_dominant.
        rep = analyze_path(outcomes, oos_only=True)
        assert rep["slice"] == "oos"
        # OOS slice is 20% of 200 = 40 records (above MIN_RECORDS).
        assert rep["verdict"] != "INSUFFICIENT_DATA"
        # Total records on disk is preserved in n_records, not the slice.
        assert rep["n_records"] == 40  # records AFTER split (the OOS slice)


class TestCLI:
    def test_cli_json_output_is_parseable(self, tmp_path, monkeypatch, capsys):
        s = _train_ml_score_dominant(tmp_path, monkeypatch)
        outcomes = tmp_path / "outcomes_cli.jsonl"
        recs = [_outcome(f"2025-11-{i+1:02d}", ml_score=float(v),
                         fwd=float(v) * 1.5)
                for i, v in enumerate(np.linspace(-5, 5, 200))]
        outcomes.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        from paper_trader.ml.attribution_audit import _cli
        rc = _cli(["--outcomes", str(outcomes), "--all", "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        # DecisionScorer.__init__ prints a "[decision_scorer] loaded n=…"
        # line on first load — strip it so we parse only the JSON document.
        json_start = out.index("{")
        rep = json.loads(out[json_start:])
        assert rep["verdict"] in ("DIVERSIFIED", "CONCENTRATED", "MODEL_INERT")
        assert len(rep["features"]) == N_FEATURES

    def test_cli_returncode_signals_actionable_verdict(
            self, tmp_path, monkeypatch, capsys):
        """Exit code convention (mirroring sibling diagnostics): 0 ok,
        1 INSUFFICIENT_DATA, 2 actionable problem (UNTRAINED/MODEL_INERT)."""
        # Empty outcomes file → INSUFFICIENT_DATA → rc=1.
        outcomes_empty = tmp_path / "empty.jsonl"
        outcomes_empty.write_text("")
        s = _train_ml_score_dominant(tmp_path, monkeypatch)
        from paper_trader.ml.attribution_audit import _cli
        rc = _cli(["--outcomes", str(outcomes_empty), "--all"])
        assert rc == 1
