"""Exact-value locks for the regime-conditional scorer-skill audit
(`paper_trader/ml/regime_audit.py`).

Mirrors test_gate_audit.py / test_calibration.py: deterministic synthetic
data, exact metrics and exact verdicts (not ranges) so a logic change must
update the literals deliberately. All offline, no network, no trained MLP.

Load-bearing assertions:
  * `_regime_of` maps the three exact `regime_mult` multipliers and drops
    everything else (unmapped → None, counted in dropped_unmapped).
  * the verdict compares per-regime rank_ic against IC_MIN, with exact
    rank_ic literals.
  * the SELL `-forward_return_5d` sign-flip matches train_scorer (without it
    a rank-skilled SELL regime would read null — the regression lock).
  * `oos_only` restricts to the temporal-OOS slice.
  * never-raises on a raising/untrained scorer / NaN / empty.
"""
from __future__ import annotations

import math

import pytest

from paper_trader.ml import regime_audit as ra


class _FakeScorer:
    """predict() echoes ml_score so the plumbing (SELL flip, kwarg names,
    OOS split, regime decode) is testable without a trained MLP."""

    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        return float(kw["ml_score"])


class _RaisingScorer:
    is_trained = True

    def predict(self, **kw):
        raise ValueError("model/feature shape mismatch")


# ─────────────────────────── _regime_of ───────────────────────────


class TestRegimeOf:
    def test_three_known_multipliers(self):
        assert ra._regime_of(0.3) == "bear"
        assert ra._regime_of(0.6) == "sideways"
        assert ra._regime_of(1.0) == "bull_or_unknown"

    def test_rounds_before_lookup(self):
        assert ra._regime_of(0.6000000000000001) == "sideways"
        assert ra._regime_of(0.2999999) == "bear"
        assert ra._regime_of(1) == "bull_or_unknown"  # int → 1.0

    def test_unmapped_returns_none(self):
        # A future fourth multiplier must NOT masquerade as one of these.
        assert ra._regime_of(0.45) is None
        assert ra._regime_of(0.0) is None
        assert ra._regime_of(2.0) is None

    def test_nonfinite_and_garbage_return_none(self):
        assert ra._regime_of(None) is None
        assert ra._regime_of("sideways") is None
        assert ra._regime_of(float("nan")) is None
        assert ra._regime_of(float("inf")) is None


# ─────────────────────── regime_skill_report ───────────────────────


def _perfect(regime_mult, n, start=0):
    """n triples whose pred perfectly rank-correlates with realized."""
    return [(regime_mult, float(i), float(i) * 0.5)
            for i in range(start, start + n)]


def _null(regime_mult, n):
    """n triples whose realized is constant → rank_ic exactly 0.0."""
    return [(regime_mult, float(i), 5.0) for i in range(n)]


class TestRegimeSkillReport:
    def test_insufficient_data_below_min_total(self):
        rep = ra.regime_skill_report(_perfect(1.0, 10))
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 10

    def test_single_regime_only(self):
        rep = ra.regime_skill_report(_perfect(1.0, 35))
        assert rep["verdict"] == "SINGLE_REGIME_ONLY"
        assert rep["n_measurable_regimes"] == 1
        # the lone measurable regime still reports its (perfect) skill
        bull = next(r for r in rep["regimes"]
                    if r["regime"] == "bull_or_unknown")
        assert bull["rank_ic"] == 1.0
        assert bull["measurable"] is True

    def test_regime_uniform_null(self):
        rep = ra.regime_skill_report(_null(0.6, 25) + _null(1.0, 25))
        assert rep["verdict"] == "REGIME_UNIFORM_NULL"
        assert rep["n_measurable_regimes"] == 2
        assert rep["n_skilled_regimes"] == 0
        for r in rep["regimes"]:
            if r["measurable"]:
                assert r["rank_ic"] == 0.0
                assert r["skilled"] is False

    def test_regime_uniform_edge(self):
        rep = ra.regime_skill_report(_perfect(0.6, 25) + _perfect(1.0, 25))
        assert rep["verdict"] == "REGIME_UNIFORM_EDGE"
        assert rep["n_measurable_regimes"] == 2
        assert rep["n_skilled_regimes"] == 2
        for r in rep["regimes"]:
            if r["measurable"]:
                assert r["rank_ic"] == 1.0
                assert r["skilled"] is True

    def test_regime_dependent_edge(self):
        # bull perfectly skilled, sideways pure noise (constant realized).
        rep = ra.regime_skill_report(_perfect(1.0, 25) + _null(0.6, 25))
        assert rep["verdict"] == "REGIME_DEPENDENT_EDGE"
        assert rep["n_measurable_regimes"] == 2
        assert rep["n_skilled_regimes"] == 1
        bull = next(r for r in rep["regimes"]
                    if r["regime"] == "bull_or_unknown")
        side = next(r for r in rep["regimes"] if r["regime"] == "sideways")
        assert bull["rank_ic"] == 1.0 and bull["skilled"] is True
        assert side["rank_ic"] == 0.0 and side["skilled"] is False

    def test_thin_regime_excluded_from_verdict(self):
        # bull 25 (measurable, skilled), bear 5 (< MIN_REGIME_N → thin).
        rep = ra.regime_skill_report(_perfect(1.0, 25) + _perfect(0.3, 5))
        # only ONE measurable regime → SINGLE_REGIME_ONLY, the thin bear
        # bucket is reported but not counted toward the verdict.
        assert rep["verdict"] == "SINGLE_REGIME_ONLY"
        bear = next(r for r in rep["regimes"] if r["regime"] == "bear")
        assert bear["n"] == 5
        assert bear["measurable"] is False
        assert bear["skilled"] is None
        assert rep["n_measurable_regimes"] == 1

    def test_unmapped_regime_mult_dropped_and_counted(self):
        good = _perfect(1.0, 25) + _perfect(0.6, 25)
        bad = [(0.45, 1.0, 2.0), (None, 1.0, 2.0), (0.0, 1.0, 2.0)]
        rep = ra.regime_skill_report(good + bad)
        assert rep["dropped_unmapped"] == 3
        assert rep["n"] == 50  # only the mapped ones are clean

    def test_nonfinite_pred_realized_dropped(self):
        recs = _perfect(1.0, 25) + _perfect(0.6, 25)
        recs += [(1.0, float("nan"), 1.0), (0.6, 1.0, float("inf"))]
        rep = ra.regime_skill_report(recs)
        assert rep["n"] == 50  # the 2 non-finite rows dropped

    def test_dir_acc_excludes_zero_pred_and_realized(self):
        # pred=0 / realized=0 carry no directional truth.
        triples = [(1.0, 0.0, 0.0)] + [(1.0, float(i), float(i))
                                       for i in range(1, 30)]
        rep = ra.regime_skill_report(triples)
        bull = next(r for r in rep["regimes"]
                    if r["regime"] == "bull_or_unknown")
        # 29 nonzero pairs, all sign-agreeing → dir_acc 1.0
        assert bull["dir_acc"] == 1.0

    def test_gate_spread_present_when_both_extreme_arms_filled(self):
        # ≥5 strong_headwind (pred<-10) and ≥5 strong_tailwind (pred>10)
        # in one regime; tailwind realized >> headwind realized.
        triples = []
        for i in range(8):
            triples.append((1.0, -20.0 - i, -3.0))   # strong_headwind
        for i in range(8):
            triples.append((1.0, 20.0 + i, 4.0))      # strong_tailwind
        # pad to clear MIN_TOTAL with neutral rows in another regime
        triples += _null(0.6, 25)
        rep = ra.regime_skill_report(triples)
        bull = next(r for r in rep["regimes"]
                    if r["regime"] == "bull_or_unknown")
        # tail mean (4.0) − head mean (-3.0) = +7.0pp
        assert bull["gate_tail_minus_head_pp"] == 7.0


# ─────────────────────── scorer_regime_audit ───────────────────────


class TestScorerRegimeAudit:
    def test_sell_sign_flip_regression(self):
        """A SELL whose pred tracks the *negated* forward return is skilled
        only because the universal sign-flip is applied. Without the flip
        this regime would read rank_ic = -1.0 (null)."""
        recs = []
        for i in range(30):
            recs.append({
                "ml_score": float(i),          # _FakeScorer → pred = i
                "regime_mult": 1.0,
                "action": "SELL",
                "forward_return_5d": float(-i),  # SELL good → price dropped
                "sim_date": f"2025-01-{(i % 28) + 1:02d}",
            })
        rep = ra.scorer_regime_audit(_FakeScorer(), recs, oos_only=False)
        bull = next(r for r in rep["regimes"]
                    if r["regime"] == "bull_or_unknown")
        # flip: realized = -(-i) = +i, perfectly tracks pred=i → rank_ic 1.0
        assert bull["rank_ic"] == 1.0
        assert bull["skilled"] is True
        assert rep["slice"] == "all"

    def test_oos_only_restricts_to_temporal_holdout(self):
        # 100 rows; oldest 80 are bull, newest 20 are sideways. The OOS
        # slice (last 20% by sim_date) must contain only the sideways rows.
        recs = []
        for i in range(80):
            recs.append({"ml_score": float(i), "regime_mult": 1.0,
                         "action": "BUY", "forward_return_5d": float(i),
                         "sim_date": f"2025-01-{(i % 28) + 1:02d}"})
        for i in range(20):
            recs.append({"ml_score": float(i), "regime_mult": 0.6,
                         "action": "BUY", "forward_return_5d": 5.0,
                         "sim_date": f"2026-12-{i + 1:02d}"})
        rep = ra.scorer_regime_audit(_FakeScorer(), recs, oos_only=True)
        assert rep["slice"] == "oos"
        assert rep["n_records_considered"] == 20
        # only the sideways regime survives the OOS slice
        side = next(r for r in rep["regimes"] if r["regime"] == "sideways")
        assert side["n"] == 20
        bull = next(r for r in rep["regimes"]
                    if r["regime"] == "bull_or_unknown")
        assert bull["n"] == 0

    def test_raising_scorer_never_raises(self):
        recs = [{"ml_score": 1.0, "regime_mult": 1.0, "action": "BUY",
                 "forward_return_5d": 2.0, "sim_date": "2025-01-01"}]
        rep = ra.scorer_regime_audit(_RaisingScorer(), recs, oos_only=False)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_empty_records(self):
        rep = ra.scorer_regime_audit(_FakeScorer(), [], oos_only=True)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_missing_forward_return_skipped(self):
        recs = [{"ml_score": 1.0, "regime_mult": 1.0, "action": "BUY",
                 "sim_date": "2025-01-01"}] * 40  # no forward_return_5d
        rep = ra.scorer_regime_audit(_FakeScorer(), recs, oos_only=False)
        assert rep["n"] == 0


# ───────────────────────────── analyze / CLI ─────────────────────────────


class TestAnalyzeAndCli:
    def test_analyze_missing_file(self, tmp_path):
        rep = ra.analyze(tmp_path / "nope.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "no outcomes file" in rep["hint"]

    def test_cli_exit_2_only_on_regime_dependent_edge(self, monkeypatch):
        monkeypatch.setattr(ra, "analyze",
                             lambda *a, **k: {"verdict": "REGIME_DEPENDENT_EDGE",
                                              "regimes": [], "hint": ""})
        assert ra._cli() == 2
        monkeypatch.setattr(ra, "analyze",
                             lambda *a, **k: {"verdict": "REGIME_UNIFORM_NULL",
                                              "regimes": [], "hint": ""})
        assert ra._cli() == 0

    def test_analyze_untrained_scorer(self, tmp_path, monkeypatch):
        p = tmp_path / "outcomes.jsonl"
        p.write_text('{"ml_score":1.0,"regime_mult":1.0,'
                      '"action":"BUY","forward_return_5d":2.0}\n')

        class _Untrained:
            is_trained = False

        monkeypatch.setattr(ra, "DecisionScorer", _Untrained, raising=False)
        # analyze imports DecisionScorer lazily from .decision_scorer; patch there
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "DecisionScorer", _Untrained)
        rep = ra.analyze(p)
        assert "not trained" in rep["hint"]
