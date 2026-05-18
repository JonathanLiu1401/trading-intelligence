"""Exact-value locks for the conviction-gate effectiveness audit
(`paper_trader/ml/gate_audit.py`, 2026-05-18 quant feature).

Mirrors test_calibration.py / test_skill_trend.py: deterministic synthetic
data, exact metrics and exact verdicts (not ranges) so a logic change must
update the literals deliberately. All offline, no network, no trained MLP.

The load-bearing assertions:
  * `gate_arm` boundary operators are byte-identical to
    `backtest.py::_ml_decide`'s `if/elif` chain (a `<`/`<=` regression there
    must fail here).
  * the verdict is driven solely by the strong_tailwind − strong_headwind
    realized spread, with exact pp values.
  * the SELL `-forward_return_5d` sign-flip matches train_scorer (without it
    GATE_EFFECTIVE would read GATE_HARMFUL — the regression lock).
  * `oos_only` restricts to the temporal-OOS slice.
"""
from __future__ import annotations

import math

import pytest

from paper_trader.ml import gate_audit as ga


class _FakeScorer:
    """predict() echoes ml_score so the gate plumbing (SELL flip, kwarg
    names, OOS split) is testable without a trained MLP."""

    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        return float(kw["ml_score"])


class _RaisingScorer:
    is_trained = True

    def predict(self, **kw):
        raise ValueError("model/feature shape mismatch")


class TestGateArm:
    """Boundary semantics must match _ml_decide's exact if/elif chain."""

    def test_strong_headwind_strict_below_minus_ten(self):
        assert ga.gate_arm(-10.0001) == ("strong_headwind", 0.60)

    def test_minus_ten_exact_is_mild_headwind(self):
        # _ml_decide: `p < -10.0` is False at exactly -10 → falls to `p < 0`.
        assert ga.gate_arm(-10.0) == ("mild_headwind", 0.85)

    def test_just_below_zero_is_mild_headwind(self):
        assert ga.gate_arm(-0.0001) == ("mild_headwind", 0.85)

    def test_zero_exact_is_neutral(self):
        assert ga.gate_arm(0.0) == ("neutral", 1.00)

    def test_five_exact_is_neutral(self):
        # `p > 5.0` is False at exactly 5 → neutral (0 ≤ p ≤ 5 unchanged).
        assert ga.gate_arm(5.0) == ("neutral", 1.00)

    def test_just_above_five_is_mild_tailwind(self):
        assert ga.gate_arm(5.0001) == ("mild_tailwind", 1.15)

    def test_ten_exact_is_mild_tailwind(self):
        # `p > 10.0` is False at exactly 10 → mild_tailwind branch.
        assert ga.gate_arm(10.0) == ("mild_tailwind", 1.15)

    def test_just_above_ten_is_strong_tailwind(self):
        assert ga.gate_arm(10.0001) == ("strong_tailwind", 1.30)

    def test_nonfinite_and_bad_type_are_neutral_noop(self):
        assert ga.gate_arm(float("nan")) == ("neutral", 1.00)
        assert ga.gate_arm(float("inf")) == ("neutral", 1.00)
        assert ga.gate_arm(float("-inf")) == ("neutral", 1.00)
        assert ga.gate_arm("not-a-number") == ("neutral", 1.00)
        assert ga.gate_arm(None) == ("neutral", 1.00)

    def test_module_constants_mirror_ml_decide(self):
        assert ga.GATE_ARMS == [
            ("strong_headwind", 0.60),
            ("mild_headwind", 0.85),
            ("neutral", 1.00),
            ("mild_tailwind", 1.15),
            ("strong_tailwind", 1.30),
        ]


def _pairs(tail_realized: float, head_realized: float,
           n_each: int = 15) -> list[tuple[float, float]]:
    """n_each strong_tailwind (pred=+20) + n_each strong_headwind (pred=-20)
    pairs with the given realized means (constant per arm)."""
    out = [(20.0, tail_realized) for _ in range(n_each)]
    out += [(-20.0, head_realized) for _ in range(n_each)]
    return out


class TestGateEffectivenessReport:
    def test_gate_effective_exact_spread(self):
        rep = ga.gate_effectiveness_report(_pairs(8.0, 2.0))
        assert rep["verdict"] == "GATE_EFFECTIVE"
        assert rep["n"] == 30
        assert rep["strong_tailwind_minus_headwind_pp"] == 6.0
        arms = {a["arm"]: a for a in rep["arms"]}
        assert arms["strong_tailwind"]["mean_realized"] == 8.0
        assert arms["strong_tailwind"]["multiplier"] == 1.30
        assert arms["strong_headwind"]["mean_realized"] == 2.0
        assert arms["strong_headwind"]["n"] == 15
        # only the two extreme arms have samples → 1 adjacent step, 8 ≥ 2.
        assert rep["arm_monotone_fraction"] == 1.0

    def test_gate_harmful_inverted_sizing(self):
        rep = ga.gate_effectiveness_report(_pairs(2.0, 8.0))
        assert rep["verdict"] == "GATE_HARMFUL"
        assert rep["strong_tailwind_minus_headwind_pp"] == -6.0
        # present arms in multiplier order: headwind(8) then tailwind(2) →
        # 2 ≥ 8 is False → 0/1.
        assert rep["arm_monotone_fraction"] == 0.0

    def test_gate_ineffective_within_band(self):
        rep = ga.gate_effectiveness_report(_pairs(5.5, 5.0))
        assert rep["verdict"] == "GATE_INEFFECTIVE"
        assert rep["strong_tailwind_minus_headwind_pp"] == 0.5

    def test_tolerance_boundary_exactly_one_pp_is_ineffective(self):
        # |spread| == EDGE_TOL_PP (1.0) → still INEFFECTIVE (inclusive band).
        rep = ga.gate_effectiveness_report(_pairs(6.0, 5.0))
        assert rep["strong_tailwind_minus_headwind_pp"] == 1.0
        assert rep["verdict"] == "GATE_INEFFECTIVE"

    def test_just_outside_tolerance_is_effective(self):
        rep = ga.gate_effectiveness_report(_pairs(6.0001, 5.0))
        assert rep["verdict"] == "GATE_EFFECTIVE"

    def test_insufficient_total(self):
        rep = ga.gate_effectiveness_report(_pairs(8.0, 2.0, n_each=5))  # n=10
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 10
        assert rep["strong_tailwind_minus_headwind_pp"] is None

    def test_insufficient_extreme_arm_even_with_enough_total(self):
        # 40 neutral-arm pairs (pred=2.0): n ≥ MIN_TOTAL but BOTH extreme
        # arms are empty → still INSUFFICIENT_DATA (the gate verdict needs
        # the extremes, not just volume).
        rep = ga.gate_effectiveness_report([(2.0, 3.0) for _ in range(40)])
        assert rep["n"] == 40
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        arms = {a["arm"]: a for a in rep["arms"]}
        assert arms["neutral"]["n"] == 40
        assert arms["strong_tailwind"]["n"] == 0

    def test_nonfinite_pairs_dropped(self):
        pairs = _pairs(8.0, 2.0)
        pairs += [(float("nan"), 1.0), (10.0, float("inf")),
                  ("x", 2.0), (5.0, None)]
        rep = ga.gate_effectiveness_report(pairs)
        assert rep["n"] == 30  # the 4 junk pairs excluded
        assert rep["verdict"] == "GATE_EFFECTIVE"


class TestScorerGateAudit:
    def test_buy_records_effective(self):
        recs = ([{"ml_score": 20.0, "forward_return_5d": 8.0,
                  "action": "BUY", "ticker": "SOXL"} for _ in range(15)]
                + [{"ml_score": -20.0, "forward_return_5d": 2.0,
                    "action": "BUY", "ticker": "SOXL"} for _ in range(15)])
        rep = ga.scorer_gate_audit(_FakeScorer(), recs, oos_only=False)
        assert rep["verdict"] == "GATE_EFFECTIVE"
        assert rep["slice"] == "all"
        assert rep["strong_tailwind_minus_headwind_pp"] == 6.0

    def test_sell_sign_flip_is_applied(self):
        # SELL: aligned target is -forward_return_5d. tailwind fr=-8 → +8,
        # headwind fr=-2 → +2 → spread +6 → EFFECTIVE. WITHOUT the flip the
        # spread would be (-8) - (-2) = -6 → GATE_HARMFUL. Asserting
        # GATE_EFFECTIVE is the regression lock on the sign alignment.
        recs = ([{"ml_score": 20.0, "forward_return_5d": -8.0,
                  "action": "SELL", "ticker": "XLF"} for _ in range(15)]
                + [{"ml_score": -20.0, "forward_return_5d": -2.0,
                    "action": "SELL", "ticker": "XLF"} for _ in range(15)])
        rep = ga.scorer_gate_audit(_FakeScorer(), recs, oos_only=False)
        assert rep["verdict"] == "GATE_EFFECTIVE"
        assert rep["strong_tailwind_minus_headwind_pp"] == 6.0

    def test_predict_exceptions_skipped_not_fatal(self):
        recs = [{"ml_score": 20.0, "forward_return_5d": 8.0, "action": "BUY"}
                for _ in range(30)]
        rep = ga.scorer_gate_audit(_RaisingScorer(), recs, oos_only=False)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_missing_forward_return_skipped(self):
        recs = ([{"ml_score": 20.0, "forward_return_5d": 8.0,
                  "action": "BUY"} for _ in range(15)]
                + [{"ml_score": -20.0, "forward_return_5d": 2.0,
                    "action": "BUY"} for _ in range(15)]
                + [{"ml_score": 20.0, "forward_return_5d": None,
                    "action": "BUY"}])
        rep = ga.scorer_gate_audit(_FakeScorer(), recs, oos_only=False)
        assert rep["n"] == 30  # the None-forward_return row excluded

    def test_oos_only_restricts_to_temporal_holdout(self):
        # 10 dated records; split_outcomes_temporal oos_fraction 0.2 →
        # the 2 most-recent-by-sim_date rows are the OOS slice.
        recs = []
        for i in range(1, 9):  # 2025-01-01 .. 2025-01-08 (train)
            recs.append({"sim_date": f"2025-01-0{i}", "ml_score": 1.0,
                         "forward_return_5d": 0.0, "action": "BUY"})
        recs.append({"sim_date": "2025-01-09", "ml_score": 20.0,
                     "forward_return_5d": 8.0, "action": "BUY"})
        recs.append({"sim_date": "2025-01-10", "ml_score": -20.0,
                     "forward_return_5d": 2.0, "action": "BUY"})
        rep = ga.scorer_gate_audit(_FakeScorer(), recs, oos_only=True)
        assert rep["slice"] == "oos"
        assert rep["n_records_considered"] == 2
        assert rep["n"] == 2
        # oos_only=False sees all 10.
        rep_all = ga.scorer_gate_audit(_FakeScorer(), recs, oos_only=False)
        assert rep_all["slice"] == "all"
        assert rep_all["n_records_considered"] == 10


class TestAnalyzeNeverRaises:
    def test_missing_outcomes_file(self, tmp_path):
        rep = ga.analyze(tmp_path / "nope.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "no outcomes file" in rep["hint"]

    def test_corrupt_lines_do_not_crash(self, tmp_path, monkeypatch):
        p = tmp_path / "out.jsonl"
        p.write_text('{"ml_score": 1}\nnot json\n[1,2,3]\n\n')

        # Force the "scorer not trained" branch deterministically (no pickle
        # in the isolated tmp data dir anyway) — analyze must degrade, not
        # raise, on every input.
        rep = ga.analyze(p)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert math.isfinite(rep["n"]) if isinstance(rep["n"], (int, float)) else True
