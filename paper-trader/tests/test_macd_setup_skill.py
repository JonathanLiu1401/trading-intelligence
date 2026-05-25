"""Tests for paper_trader.ml.macd_setup_skill — the MACD-enhanced-feature
univariate skill diagnostic. Locks specific verdicts on engineered
inputs so a future tuning change is reviewable from a passing test.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from paper_trader.ml import macd_setup_skill as mss


# ---------------------------------------------------------------------------
# _to_finite_float — mirror of wk52_skill / decision_scorer sentinel discipline
# ---------------------------------------------------------------------------


class TestToFiniteFloat:
    def test_finite_passthrough(self):
        assert mss._to_finite_float(1.5) == 1.5
        assert mss._to_finite_float(0) == 0.0
        assert mss._to_finite_float(-42.7) == -42.7

    def test_none_returns_none(self):
        assert mss._to_finite_float(None) is None

    def test_bool_returns_none(self):
        """bool subclasses int — must be rejected so flag-typed columns
        never leak through as 1.0/0.0 numerics."""
        assert mss._to_finite_float(True) is None
        assert mss._to_finite_float(False) is None

    def test_nan_returns_none(self):
        assert mss._to_finite_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert mss._to_finite_float(float("inf")) is None
        assert mss._to_finite_float(float("-inf")) is None

    def test_unparseable_returns_none(self):
        assert mss._to_finite_float("not a number") is None
        assert mss._to_finite_float([1, 2]) is None


# ---------------------------------------------------------------------------
# _bool_to_int — accepts the three documented storage shapes
# ---------------------------------------------------------------------------


class TestBoolToInt:
    def test_python_true_false(self):
        assert mss._bool_to_int(True) == 1
        assert mss._bool_to_int(False) == 0

    def test_int_zero_one(self):
        assert mss._bool_to_int(1) == 1
        assert mss._bool_to_int(0) == 0

    def test_float_truthiness(self):
        # The function follows the documented "> 0" rule.
        assert mss._bool_to_int(0.5) == 1
        assert mss._bool_to_int(0.0) == 0
        assert mss._bool_to_int(-1.0) == 0

    def test_none_drops(self):
        assert mss._bool_to_int(None) is None

    def test_nan_drops(self):
        assert mss._bool_to_int(float("nan")) is None

    def test_inf_drops(self):
        assert mss._bool_to_int(float("inf")) is None

    def test_unparseable_drops(self):
        assert mss._bool_to_int("yes") is None


# ---------------------------------------------------------------------------
# _flag_value — handles combined_setup conjunction semantics
# ---------------------------------------------------------------------------


class TestFlagValue:
    def test_primitive_passthrough(self):
        rec = {"ema200_above": True, "hist_cross_up": False}
        assert mss._flag_value(rec, "ema200_above") == 1
        assert mss._flag_value(rec, "hist_cross_up") == 0

    def test_combined_setup_all_true(self):
        rec = {
            "ema200_above": True,
            "hist_cross_up": True,
            "macd_below_zero_cross": True,
        }
        assert mss._flag_value(rec, "combined_setup") == 1

    def test_combined_setup_one_false_yields_zero(self):
        rec = {
            "ema200_above": True,
            "hist_cross_up": False,
            "macd_below_zero_cross": True,
        }
        assert mss._flag_value(rec, "combined_setup") == 0

    def test_combined_setup_one_none_yields_none(self):
        """Any None primitive means we cannot reduce to True/False;
        the row drops from the combined-setup bucket honestly."""
        rec = {
            "ema200_above": True,
            "hist_cross_up": None,
            "macd_below_zero_cross": True,
        }
        assert mss._flag_value(rec, "combined_setup") is None

    def test_combined_setup_missing_field_yields_none(self):
        rec = {"ema200_above": True, "hist_cross_up": True}
        # macd_below_zero_cross missing → None
        assert mss._flag_value(rec, "combined_setup") is None


# ---------------------------------------------------------------------------
# build_macd_setup_skill — main verdict logic
# ---------------------------------------------------------------------------


def _make_records(n_true: int, n_false: int, true_mean: float = 5.0,
                  false_mean: float = -2.0, key: str = "hist_cross_up",
                  noise: float = 0.5) -> list[dict]:
    """Generate engineered outcome records with a known gap. Other flags
    are all False so each row only contributes to the target flag."""
    rng = np.random.default_rng(42)
    recs: list[dict] = []
    for i in range(n_true):
        recs.append({
            "ema200_above": (True if key == "ema200_above" else False),
            "hist_cross_up": (True if key == "hist_cross_up" else False),
            "macd_below_zero_cross": (
                True if key == "macd_below_zero_cross" else False),
            "forward_return_5d": float(true_mean + rng.normal(0, noise)),
        })
    for i in range(n_false):
        recs.append({
            "ema200_above": False,
            "hist_cross_up": False,
            "macd_below_zero_cross": False,
            "forward_return_5d": float(false_mean + rng.normal(0, noise)),
        })
    return recs


class TestBuildMacdSetupSkill:
    def test_empty_input_returns_insufficient(self):
        rep = mss.build_macd_setup_skill([])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["status"] == "insufficient_data"
        # Every flag exists in the empty payload — JSON consumers need
        # the shape regardless.
        for key in mss.FLAG_KEYS:
            assert key in rep["flags"]
            assert rep["flags"][key]["verdict"] == "INSUFFICIENT_DATA"

    def test_below_min_pairs_yields_insufficient(self):
        recs = _make_records(n_true=5, n_false=5, key="hist_cross_up")
        rep = mss.build_macd_setup_skill(recs)
        # Total 10 < MIN_PAIRS=30 → INSUFFICIENT_DATA for that flag.
        assert rep["flags"]["hist_cross_up"]["verdict"] == "INSUFFICIENT_DATA"

    def test_strong_positive_edge_predicts_up(self):
        """A flag with mean_gap ≈ +7pp and clear separation must hit
        SETUP_PREDICTS_UP."""
        recs = _make_records(n_true=30, n_false=30,
                             true_mean=5.0, false_mean=-2.0,
                             key="hist_cross_up", noise=0.3)
        rep = mss.build_macd_setup_skill(recs)
        flag = rep["flags"]["hist_cross_up"]
        assert flag["verdict"] == "SETUP_PREDICTS_UP", (
            f"expected SETUP_PREDICTS_UP, got {flag['verdict']} "
            f"(mean_gap={flag['mean_gap_pct']:+.2f}, "
            f"spearman={flag['spearman']:+.3f})"
        )
        assert flag["mean_gap_pct"] > mss.MEAN_GAP_GOOD_PCT
        assert flag["spearman"] >= mss.SPEARMAN_GOOD
        # Top-level rollup picks PREDICTS_UP since no DOWN/DIR_DOWN exists.
        assert rep["verdict"] == "SETUP_PREDICTS_UP"

    def test_strong_negative_edge_predicts_down(self):
        """A flag that's actively HARMFUL (True → worse return) must
        hit SETUP_PREDICTS_DOWN. This is the most economically decisive
        finding a quant cares about — it justifies removing or
        inverting the feature."""
        recs = _make_records(n_true=30, n_false=30,
                             true_mean=-5.0, false_mean=2.0,
                             key="ema200_above", noise=0.3)
        rep = mss.build_macd_setup_skill(recs)
        flag = rep["flags"]["ema200_above"]
        assert flag["verdict"] == "SETUP_PREDICTS_DOWN", (
            f"expected SETUP_PREDICTS_DOWN, got {flag['verdict']} "
            f"(mean_gap={flag['mean_gap_pct']:+.2f}, "
            f"spearman={flag['spearman']:+.3f})"
        )
        assert flag["mean_gap_pct"] < -mss.MEAN_GAP_GOOD_PCT
        # Top-level rollup picks DOWN over UP per the documented priority.
        assert rep["verdict"] == "SETUP_PREDICTS_DOWN"

    def test_no_edge_yields_no_skill(self):
        """When True and False groups have the same mean (within
        tolerance), verdict must be SETUP_NO_SKILL."""
        recs = _make_records(n_true=50, n_false=50,
                             true_mean=0.0, false_mean=0.0,
                             key="macd_below_zero_cross", noise=2.0)
        rep = mss.build_macd_setup_skill(recs)
        flag = rep["flags"]["macd_below_zero_cross"]
        assert flag["verdict"] in (
            "SETUP_NO_SKILL", "DIRECTIONAL_UP", "DIRECTIONAL_DOWN",
        ), (
            f"identical-mean groups should not produce a strong verdict, "
            f"got {flag['verdict']}"
        )

    def test_imbalanced_min_bucket_blocks_verdict(self):
        """When the True bucket has < MIN_BUCKET rows, the verdict must
        downgrade to INSUFFICIENT_DATA regardless of the False bucket
        size."""
        recs = _make_records(n_true=3, n_false=100,
                             key="hist_cross_up")
        rep = mss.build_macd_setup_skill(recs)
        assert rep["flags"]["hist_cross_up"]["verdict"] == "INSUFFICIENT_DATA"

    def test_nan_forward_return_dropped(self):
        """A NaN/None forward_return_5d must drop the row, not poison
        the mean computation."""
        recs = [{
            "ema200_above": True, "hist_cross_up": True,
            "macd_below_zero_cross": True,
            "forward_return_5d": float("nan"),
        } for _ in range(5)]
        recs += _make_records(n_true=30, n_false=30, true_mean=5.0,
                              false_mean=-2.0, key="hist_cross_up",
                              noise=0.3)
        rep = mss.build_macd_setup_skill(recs)
        # The dropped count is per-flag — verify at least the NaN ones
        # were rejected.
        assert rep["flags"]["hist_cross_up"]["n_dropped_return"] >= 5

    def test_combined_setup_works_when_all_three_present(self):
        """combined_setup should produce a verdict when there are
        ≥MIN_BUCKET rows on each side."""
        rng = np.random.default_rng(7)
        recs: list[dict] = []
        # True-setup: all 3 flags True, average return +6%
        for _ in range(30):
            recs.append({
                "ema200_above": True, "hist_cross_up": True,
                "macd_below_zero_cross": True,
                "forward_return_5d": float(6.0 + rng.normal(0, 0.4)),
            })
        # False-setup: NONE of the 3 are True (combined_setup=0), avg -2%
        for _ in range(30):
            recs.append({
                "ema200_above": False, "hist_cross_up": False,
                "macd_below_zero_cross": False,
                "forward_return_5d": float(-2.0 + rng.normal(0, 0.4)),
            })
        rep = mss.build_macd_setup_skill(recs)
        flag = rep["flags"]["combined_setup"]
        assert flag["n_true"] == 30
        assert flag["n_false"] == 30
        assert flag["verdict"] == "SETUP_PREDICTS_UP"

    def test_top_level_priority_down_wins_over_up(self):
        """If one flag is PREDICTS_DOWN and another is PREDICTS_UP, the
        rollup must pick DOWN (the anti-predictive signal — feature-
        removal cue is more economically actionable)."""
        # ema200_above: strongly DOWN
        bad_recs = _make_records(n_true=30, n_false=30,
                                  true_mean=-6.0, false_mean=3.0,
                                  key="ema200_above", noise=0.3)
        # hist_cross_up: strongly UP — but ema200_above mismatch in the
        # original generator means we need disjoint sample sets. Rebuild.
        good_recs = _make_records(n_true=30, n_false=30,
                                   true_mean=5.0, false_mean=-2.0,
                                   key="hist_cross_up", noise=0.3)
        # Build a single combined corpus — but the helper sets *other*
        # flags to False on each side, so the two flag verdicts will
        # use ALL the rows. That's the right test.
        all_recs = bad_recs + good_recs
        rep = mss.build_macd_setup_skill(all_recs)
        # At least one DOWN must be present (ema200_above).
        verdicts = {k: rep["flags"][k]["verdict"] for k in mss.FLAG_KEYS}
        assert "SETUP_PREDICTS_DOWN" in verdicts.values(), verdicts
        # Top-level priority picks DOWN over UP.
        assert rep["verdict"] == "SETUP_PREDICTS_DOWN"


# ---------------------------------------------------------------------------
# JSON contract — every emitted dict must serialise cleanly
# ---------------------------------------------------------------------------


class TestJSONSafe:
    def test_empty_payload_is_json_safe(self):
        rep = mss._empty("test reason")
        json.dumps(rep)  # raises if not json-safe

    def test_ok_payload_is_json_safe(self):
        recs = _make_records(n_true=30, n_false=30, key="hist_cross_up")
        rep = mss.build_macd_setup_skill(recs)
        json.dumps(rep)


# ---------------------------------------------------------------------------
# load_outcomes / analyze — file-IO contract
# ---------------------------------------------------------------------------


class TestLoadOutcomes:
    def test_missing_file_returns_empty(self):
        assert mss.load_outcomes("/no/such/path.jsonl") == []

    def test_unparseable_line_dropped(self, tmp_path):
        p = tmp_path / "outcomes.jsonl"
        p.write_text("not valid json\n{\"a\": 1}\nalso bad\n")
        out = mss.load_outcomes(p)
        assert out == [{"a": 1}]

    def test_round_trip(self, tmp_path):
        recs = _make_records(n_true=30, n_false=30, key="hist_cross_up")
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        loaded = mss.load_outcomes(p)
        assert len(loaded) == 60


class TestAnalyze:
    def test_missing_file_yields_insufficient(self):
        rep = mss.analyze("/no/such/path.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["status"] == "insufficient_data"

    def test_full_round_trip(self, tmp_path):
        recs = _make_records(n_true=30, n_false=30,
                             true_mean=5.0, false_mean=-2.0,
                             key="hist_cross_up", noise=0.3)
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        rep = mss.analyze(p)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "SETUP_PREDICTS_UP"

    def test_never_raises_on_corrupt_input(self, tmp_path):
        """A blow-up inside `analyze` must degrade to status='error',
        not raise (the documented diagnostic discipline)."""
        # Force a fault by monkey-patching build_macd_setup_skill.
        original = mss.build_macd_setup_skill
        try:
            mss.build_macd_setup_skill = lambda *_: (_ for _ in ()).throw(
                RuntimeError("boom"))
            p = tmp_path / "outcomes.jsonl"
            p.write_text("{}\n")
            rep = mss.analyze(p)
            assert rep["status"] == "error"
            assert "boom" in rep["hint"] or "boom" in str(rep)
        finally:
            mss.build_macd_setup_skill = original


# ---------------------------------------------------------------------------
# CLI — exit codes encode the verdict for shell integration
# ---------------------------------------------------------------------------


class TestCli:
    def test_predicts_up_exits_zero(self, tmp_path, capsys):
        recs = _make_records(n_true=30, n_false=30,
                             true_mean=5.0, false_mean=-2.0,
                             key="hist_cross_up", noise=0.3)
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        rc = mss._cli(["--path", str(p), "--json"])
        assert rc == 0

    def test_predicts_down_exits_two(self, tmp_path, capsys):
        recs = _make_records(n_true=30, n_false=30,
                             true_mean=-5.0, false_mean=2.0,
                             key="ema200_above", noise=0.3)
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        rc = mss._cli(["--path", str(p), "--json"])
        assert rc == 2

    def test_missing_file_exits_one(self, tmp_path):
        rc = mss._cli(["--path", str(tmp_path / "missing.jsonl"), "--json"])
        assert rc == 1
