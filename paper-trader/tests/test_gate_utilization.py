"""Exact-value locks for the gate-utilization audit
(``paper_trader/ml/gate_utilization.py``).

Discipline mirrors ``tests/test_gate_audit.py``: deterministic synthetic
predictions, exact verdict literals, all offline, no MLP. The load-bearing
assertions:

  * arm membership is computed via ``gate_audit.gate_arm`` (single source
    of truth — never re-derived), so a boundary-operator regression in
    ``_ml_decide`` would propagate identically here.
  * verdict precedence: ``EMPTY_EXTREME_ARM`` beats ``NEUTRAL_DOMINATED``
    beats ``LOPSIDED`` beats ``BALANCED`` / ``WEAK_BALANCE``.
  * ``INSUFFICIENT_DATA`` short-circuits below ``MIN_TOTAL`` (boundary
    pinned at exactly the gate value).
  * ``LOPSIDED_PCT`` boundary tested at exactly the cutoff in both
    directions so a 0.5999 → 0.6001 tuning change is a visible diff.
  * the analyzer ``analyze()`` honestly degrades on a missing file and
    untrained scorer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_trader.ml import gate_utilization as gu


class TestArmMembershipIsSSOTPinned:
    """Arm assignment MUST use `gate_audit.gate_arm` — never a local
    re-derivation. Locking this prevents two diagnostics from disagreeing
    on which arm a `p == -10` or `p == 10` prediction belongs to (the
    documented boundary-operator hazard)."""

    def test_uses_gate_audit_gate_arm(self):
        from paper_trader.ml import gate_audit
        assert gu.gate_arm is gate_audit.gate_arm

    def test_arm_order_matches_gate_audit(self):
        from paper_trader.ml.gate_audit import _ARM_ORDER as GA_ORDER
        # The gate_utilization report iterates arms in gate_audit's order
        # so the two CLIs print arms in a stable matching order.
        rep = gu.gate_utilization_report([1.0] * 30)
        assert [o["arm"] for o in rep["arms"]] == list(GA_ORDER)


class TestInsufficientData:
    """Short-circuits below MIN_TOTAL — the only verdict allowed there."""

    def test_empty_returns_insufficient(self):
        rep = gu.gate_utilization_report([])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_exactly_below_min_total(self):
        # MIN_TOTAL - 1 predictions → still INSUFFICIENT_DATA
        rep = gu.gate_utilization_report([1.0] * (gu.MIN_TOTAL - 1))
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_exactly_at_min_total_passes_short_circuit(self):
        # MIN_TOTAL predictions — verdict is no longer INSUFFICIENT_DATA.
        # A single-arm concentration at MIN_TOTAL crosses LOPSIDED_PCT.
        rep = gu.gate_utilization_report([1.0] * gu.MIN_TOTAL)
        assert rep["verdict"] != "INSUFFICIENT_DATA"


class TestEmptyExtremeArmTrumps:
    """An empty extreme arm is the strongest single signal and must
    surface BEFORE LOPSIDED / NEUTRAL_DOMINATED verdicts."""

    def test_empty_strong_headwind_surfaces_first(self):
        # All neutral predictions: would otherwise be NEUTRAL_DOMINATED.
        # But strong_headwind AND strong_tailwind are empty, and the
        # empty-extreme verdict trumps.
        rep = gu.gate_utilization_report([2.0] * 30)
        assert rep["verdict"] == "EMPTY_EXTREME_ARM"
        # Both extremes empty
        assert "strong_headwind" in rep["empty_arms"]
        assert "strong_tailwind" in rep["empty_arms"]

    def test_one_empty_extreme_still_triggers(self):
        # populate every arm EXCEPT strong_tailwind
        preds = ([-15.0] * 10   # strong_headwind
                 + [-5.0] * 10   # mild_headwind
                 + [2.0] * 10    # neutral
                 + [7.0] * 10)   # mild_tailwind — no strong_tailwind!
        rep = gu.gate_utilization_report(preds)
        assert rep["verdict"] == "EMPTY_EXTREME_ARM"
        assert "strong_tailwind" in rep["empty_arms"]
        assert "strong_headwind" not in rep["empty_arms"]

    def test_both_extremes_populated_does_not_trigger(self):
        # 6 in each arm, total 30 → no extreme empty.
        preds = ([-15.0] * 6 + [-5.0] * 6 + [2.0] * 6 + [7.0] * 6 + [15.0] * 6)
        rep = gu.gate_utilization_report(preds)
        assert rep["verdict"] != "EMPTY_EXTREME_ARM"


class TestNeutralDominatedVerdict:
    """Neutral arm carrying ≥ LOPSIDED_PCT is its own verdict so the
    operational consequence ('gate is no-op for most decisions') is
    distinct from a single non-neutral arm dominating."""

    def test_neutral_dominated_at_lopsided_threshold(self):
        # 18 neutral + 3 each of the other 4 arms (incl. both extremes).
        # 18/30 = 60% = LOPSIDED_PCT — exact boundary, INCLUSIVE.
        preds = ([2.0] * 18 +
                 [-15.0] * 3 + [-5.0] * 3 + [7.0] * 3 + [15.0] * 3)
        rep = gu.gate_utilization_report(preds)
        assert rep["verdict"] == "NEUTRAL_DOMINATED"
        assert rep["neutral_dominated"] is True

    def test_below_lopsided_threshold_is_not_dominated(self):
        # 17 neutral (17/30 ≈ 56.7% < 60%) + extremes populated.
        # Should NOT read NEUTRAL_DOMINATED.
        preds = ([2.0] * 17 +
                 [-15.0] * 4 + [-5.0] * 3 + [7.0] * 3 + [15.0] * 3)
        rep = gu.gate_utilization_report(preds)
        assert rep["verdict"] != "NEUTRAL_DOMINATED"
        assert rep["neutral_dominated"] is False


class TestLopsidedVerdict:
    """A single non-neutral arm carrying ≥ LOPSIDED_PCT is LOPSIDED (NOT
    NEUTRAL_DOMINATED) — the difference is whether the dominant arm IS
    the gate's no-op arm."""

    def test_strong_tailwind_dominates(self):
        # 18 strong_tailwind + 3 each of the other arms — neutral not
        # dominant; LOPSIDED instead.
        preds = ([15.0] * 18 +
                 [-15.0] * 3 + [-5.0] * 3 + [2.0] * 3 + [7.0] * 3)
        rep = gu.gate_utilization_report(preds)
        assert rep["verdict"] == "LOPSIDED"
        assert rep["lopsided_arm"] == "strong_tailwind"
        assert rep["neutral_dominated"] is False

    def test_strong_headwind_dominates(self):
        preds = ([-15.0] * 18 +
                 [-5.0] * 3 + [2.0] * 3 + [7.0] * 3 + [15.0] * 3)
        rep = gu.gate_utilization_report(preds)
        assert rep["verdict"] == "LOPSIDED"
        assert rep["lopsided_arm"] == "strong_headwind"


class TestBalancedVerdict:
    """Every arm ≥ MIN_ARM_PCT (5%) and no arm ≥ LOPSIDED_PCT (60%)."""

    def test_uniform_distribution_is_balanced(self):
        # 12 per arm × 5 arms = 60 predictions, 20% per arm — fully balanced.
        preds = ([-15.0] * 12 + [-5.0] * 12 + [2.0] * 12 + [7.0] * 12 + [15.0] * 12)
        rep = gu.gate_utilization_report(preds)
        assert rep["verdict"] == "BALANCED"
        for o in rep["arms"]:
            assert o["pct_of_total"] >= gu.MIN_ARM_PCT

    def test_at_min_arm_pct_boundary_passes(self):
        # 5% each in two extreme arms (3 of 60 = 5.0% exact), 30% in each
        # of the other 3 arms — still BALANCED (not WEAK_BALANCE).
        preds = ([-15.0] * 3 + [-5.0] * 18 + [2.0] * 18 + [7.0] * 18 + [15.0] * 3)
        rep = gu.gate_utilization_report(preds)
        assert rep["verdict"] == "BALANCED"


class TestWeakBalanceVerdict:
    """Every arm populated but ≥1 below MIN_ARM_PCT — reachable but thin."""

    def test_thin_extreme_arm_demotes_to_weak_balance(self):
        # 2 in strong_headwind (~3.3% of 60), 14-15 in the others.
        preds = ([-15.0] * 2 + [-5.0] * 14 + [2.0] * 15 + [7.0] * 14 + [15.0] * 15)
        rep = gu.gate_utilization_report(preds)
        assert rep["verdict"] == "WEAK_BALANCE"

    def test_weak_arms_listed_in_hint(self):
        preds = ([-15.0] * 2 + [-5.0] * 14 + [2.0] * 15 + [7.0] * 14 + [15.0] * 15)
        rep = gu.gate_utilization_report(preds)
        assert "strong_headwind" in (rep.get("hint") or "")


class TestArmReportingFields:
    """The per-arm output must include name, multiplier (from gate_audit
    SSOT), count, and percentage. These are JSON-safe and stable."""

    def test_arms_field_includes_all_five(self):
        rep = gu.gate_utilization_report([1.0] * 30)
        names = {o["arm"] for o in rep["arms"]}
        assert names == {"strong_headwind", "mild_headwind", "neutral",
                         "mild_tailwind", "strong_tailwind"}

    def test_multipliers_match_gate_audit_ssot(self):
        from paper_trader.ml.gate_audit import _ARM_MULT
        rep = gu.gate_utilization_report([1.0] * 30)
        for o in rep["arms"]:
            assert o["multiplier"] == _ARM_MULT[o["arm"]]

    def test_pct_sums_to_one(self):
        preds = [1.0] * 25 + [-15.0] * 5
        rep = gu.gate_utilization_report(preds)
        total_pct = sum(o["pct_of_total"] for o in rep["arms"])
        assert total_pct == pytest.approx(1.0, abs=1e-6)

    def test_arm_counts_sum_to_n(self):
        preds = [1.0] * 25 + [-15.0] * 5
        rep = gu.gate_utilization_report(preds)
        total_n = sum(o["n"] for o in rep["arms"])
        assert total_n == rep["n"] == 30


class TestPredictionDistribution:
    """Quantile reporting on the FINITE subset of predictions."""

    def test_distribution_is_emitted_when_n_positive(self):
        rep = gu.gate_utilization_report([-2.0, 0.0, 2.0])
        dist = rep["pred_distribution"]
        assert dist is not None
        assert dist["min"] == -2.0 and dist["max"] == 2.0
        assert dist["median"] == 0.0

    def test_distribution_is_none_on_empty(self):
        rep = gu.gate_utilization_report([])
        assert rep["pred_distribution"] is None


class TestNonFiniteHandling:
    """A non-finite prediction routes to the neutral arm via gate_arm,
    NOT silently dropped — mirrors the live gate's off-distribution
    behaviour. Distribution stats use only finite values."""

    def test_nan_routes_to_neutral_arm(self):
        # 35 NaN + 5 in each extreme (so the EMPTY_EXTREME_ARM verdict
        # doesn't trump). All NaNs route to neutral via gate_arm's
        # non-finite contract; neutral then dominates → NEUTRAL_DOMINATED.
        preds = [float("nan")] * 35 + [-15.0] * 5 + [15.0] * 5
        rep = gu.gate_utilization_report(preds)
        assert rep["verdict"] == "NEUTRAL_DOMINATED"
        neutral = next(o for o in rep["arms"] if o["arm"] == "neutral")
        assert neutral["n"] == 35

    def test_all_nan_empties_extremes_trumps_neutral_domination(self):
        # All-NaN input: every value routes to neutral, but strong_headwind
        # and strong_tailwind end up empty. The empty-extreme verdict
        # precedence beats NEUTRAL_DOMINATED — the operator should see the
        # stronger structural finding, not "neutral is full" (which is also
        # true but less actionable).
        rep = gu.gate_utilization_report([float("nan")] * 35)
        assert rep["verdict"] == "EMPTY_EXTREME_ARM"

    def test_pred_distribution_only_uses_finite_values(self):
        # 30 finite + 5 NaN. NaN should not appear in the percentile stats.
        preds = list(range(30)) + [float("nan")] * 5
        rep = gu.gate_utilization_report(preds)
        dist = rep["pred_distribution"]
        # max is 29 (finite), not NaN
        assert dist["max"] == 29.0


class TestDegradesNeverRaises:
    """Pure-function contract: no input crash, even on garbage."""

    def test_non_iterable_returns_insufficient(self):
        rep = gu.gate_utilization_report(42)  # int is not iterable
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_generator_input_works(self):
        rep = gu.gate_utilization_report(p for p in [1.0] * 30)
        assert rep["verdict"] != "INSUFFICIENT_DATA"

    def test_garbage_values_route_to_neutral(self):
        rep = gu.gate_utilization_report(["nope", None, "0"] * 12)  # 36 items
        # All garbage maps to neutral arm via gate_arm (which TypeError-handles)
        neutral = next(o for o in rep["arms"] if o["arm"] == "neutral")
        assert neutral["n"] == 36


class TestSideBalance:
    """`side_balance` is the fraction of predictions on the strict
    tailwind side (p > 0). p == 0 counts as neutral for this metric."""

    def test_all_tailwind(self):
        rep = gu.gate_utilization_report([1.0] * 30)
        assert rep["side_balance"] == 1.0

    def test_all_headwind(self):
        rep = gu.gate_utilization_report([-1.0] * 30)
        assert rep["side_balance"] == 0.0

    def test_split_evenly(self):
        rep = gu.gate_utilization_report([-1.0] * 15 + [1.0] * 15)
        assert rep["side_balance"] == 0.5

    def test_zero_counts_as_headwind_side(self):
        # p == 0 is in neutral arm, but for side_balance we ask "is p > 0".
        # 15 zeros + 15 ones → 15 strictly tailwind → 0.5 of total.
        rep = gu.gate_utilization_report([0.0] * 15 + [1.0] * 15)
        assert rep["side_balance"] == 0.5


class TestAnalyzeAnalyzer:
    """`analyze()` is the on-disk + on-deployed-model path. Honest degrade
    on missing inputs."""

    def test_missing_file_degrades(self, tmp_path):
        rep = gu.analyze(tmp_path / "no-such.jsonl")
        assert rep["status"] == "error"
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_empty_file_degrades(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        path.write_text("")
        rep = gu.analyze(path)
        # No predictions → INSUFFICIENT_DATA, never raises.
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_untrained_scorer_degrades(self, tmp_path, monkeypatch):
        # Redirect SCORER_PATH to a nonexistent file → DecisionScorer()
        # reports untrained → analyzer must NOT raise.
        from paper_trader.ml import decision_scorer
        monkeypatch.setattr(decision_scorer, "SCORER_PATH",
                            tmp_path / "no-pickle.pkl")
        # Write one valid outcome so the file isn't empty.
        path = tmp_path / "outcomes.jsonl"
        path.write_text(json.dumps({
            "ticker": "NVDA", "ml_score": 3.0, "rsi": 50,
            "macd": 0.01, "mom5": 1.0, "mom20": 2.0,
            "regime_mult": 1.0, "forward_return_5d": 1.5,
        }) + "\n")
        rep = gu.analyze(path, oos_only=False)
        assert rep["status"] == "error"
        assert "not trained" in (rep.get("hint") or "")
