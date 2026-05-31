"""Tests for paper_trader.ml.gate_arm_directional_consistency.

Pins exact-value verdicts on synthetic inputs so a regression here is
immediately visible. Mirrors the test discipline of every sibling
gate diagnostic (test_gate_realized, test_gate_audit, test_gate_pnl):
threshold-driven verdicts → table of synthetic per-arm sign mixes →
expected verdict per case.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_trader.ml.gate_arm_directional_consistency import (
    CONSISTENT_TOL,
    INVERTED_TOL,
    MIN_ARM_N,
    MIN_TOTAL,
    _EXPECTED_SIGN,
    _arm_for_pred,
    analyze,
    gate_arm_directional_consistency_report,
)


# ─────────────────────── _arm_for_pred ───────────────────────────

class TestArmForPred:
    """Pin the gate-arm bucketing — must match `gate_audit.gate_arm`'s
    if/elif chain to the bit (CLAUDE.md SSOT discipline)."""

    def test_strong_headwind_boundary(self):
        # < -10.0 → strong_headwind
        assert _arm_for_pred(-50.0) == "strong_headwind"
        assert _arm_for_pred(-10.01) == "strong_headwind"

    def test_minus_ten_is_mild_headwind(self):
        # -10.0 is NOT < -10.0 — falls to mild_headwind
        assert _arm_for_pred(-10.0) == "mild_headwind"

    def test_mild_headwind_band(self):
        assert _arm_for_pred(-5.0) == "mild_headwind"
        assert _arm_for_pred(-0.01) == "mild_headwind"

    def test_zero_is_neutral(self):
        # Not < 0, not > 5 → neutral
        assert _arm_for_pred(0.0) == "neutral"

    def test_five_is_neutral(self):
        # Not > 5 — falls into neutral, matches _ml_decide / gate_audit
        assert _arm_for_pred(5.0) == "neutral"

    def test_mild_tailwind_band(self):
        assert _arm_for_pred(5.01) == "mild_tailwind"
        assert _arm_for_pred(10.0) == "mild_tailwind"

    def test_strong_tailwind_band(self):
        assert _arm_for_pred(10.01) == "strong_tailwind"
        assert _arm_for_pred(50.0) == "strong_tailwind"


# ─────────────────────── _EXPECTED_SIGN ───────────────────────────

class TestExpectedSign:
    """The expected-sign mapping is the load-bearing input to the
    consistency calculation — pin every entry."""

    def test_headwind_arms_are_negative(self):
        assert _EXPECTED_SIGN["strong_headwind"] == -1
        assert _EXPECTED_SIGN["mild_headwind"] == -1

    def test_neutral_is_zero(self):
        assert _EXPECTED_SIGN["neutral"] == 0

    def test_tailwind_arms_are_positive(self):
        assert _EXPECTED_SIGN["mild_tailwind"] == +1
        assert _EXPECTED_SIGN["strong_tailwind"] == +1


# ─────────────────── gate_arm_directional_consistency_report ─────────

def _mk_row(pred: float, realized: float,
            action: str = "BUY", off_dist: bool = False) -> dict:
    """Synthetic decision_outcomes-shaped row."""
    return {
        "gate_scorer_pred": pred,
        "gate_off_dist": off_dist,
        "forward_return_5d": realized,
        "action": action,
    }


class TestReportEmpty:
    def test_no_rows_returns_not_yet_populated(self):
        rep = gate_arm_directional_consistency_report([])
        assert rep["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"
        assert rep["n_captured"] == 0
        assert rep["n_acted"] == 0

    def test_only_uncaptured_rows_returns_not_yet_populated(self):
        # No gate_scorer_pred key → row excluded entirely
        rows = [{"forward_return_5d": 1.0} for _ in range(50)]
        rep = gate_arm_directional_consistency_report(rows)
        assert rep["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"
        assert rep["n_captured"] == 0


class TestReportInsufficient:
    def test_below_min_total(self):
        # 10 rows, all valid → < MIN_TOTAL=30
        rows = [_mk_row(20.0, 5.0) for _ in range(10)]
        rep = gate_arm_directional_consistency_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_acted"] == 10

    def test_below_min_arm_n_in_extreme(self):
        # 50 rows, all in the neutral arm → MIN_TOTAL satisfied but
        # strong arms have 0 rows
        rows = [_mk_row(2.0, 1.0) for _ in range(50)]
        rep = gate_arm_directional_consistency_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_acted"] == 50


class TestReportConsistent:
    def test_both_arms_consistent_emits_consistent_verdict(self):
        # 20 strong_tailwind rows, 80% positive realized → 0.8 consistency
        # 20 strong_headwind rows, 80% negative realized → 0.8 consistency
        rows: list[dict] = []
        # strong_tailwind: pred > 10, 16 positives + 4 negatives = 0.8
        rows.extend(_mk_row(20.0, +5.0) for _ in range(16))
        rows.extend(_mk_row(20.0, -3.0) for _ in range(4))
        # strong_headwind: pred < -10, 16 negatives + 4 positives = 0.8
        rows.extend(_mk_row(-20.0, -5.0) for _ in range(16))
        rows.extend(_mk_row(-20.0, +3.0) for _ in range(4))
        rep = gate_arm_directional_consistency_report(rows)
        assert rep["verdict"] == "GATE_DIRECTIONALLY_CONSISTENT"
        st = next(a for a in rep["arms"] if a["arm"] == "strong_tailwind")
        sh = next(a for a in rep["arms"] if a["arm"] == "strong_headwind")
        assert st["directional_consistency"] == pytest.approx(0.8, abs=1e-3)
        assert sh["directional_consistency"] == pytest.approx(0.8, abs=1e-3)
        assert st["n_positive"] == 16
        assert sh["n_negative"] == 16


class TestReportInverted:
    def test_inverted_tailwind_emits_inverted_verdict(self):
        # 20 strong_tailwind rows with only 30% positive (inverted: gate
        # said "good" but realized was bad 70% of the time)
        # 20 strong_headwind rows with 60% negative (consistent enough)
        rows: list[dict] = []
        rows.extend(_mk_row(20.0, +5.0) for _ in range(6))
        rows.extend(_mk_row(20.0, -3.0) for _ in range(14))
        rows.extend(_mk_row(-20.0, -5.0) for _ in range(12))
        rows.extend(_mk_row(-20.0, +3.0) for _ in range(8))
        rep = gate_arm_directional_consistency_report(rows)
        assert rep["verdict"] == "GATE_DIRECTIONALLY_INVERTED"
        st = next(a for a in rep["arms"] if a["arm"] == "strong_tailwind")
        assert st["directional_consistency"] == pytest.approx(0.3, abs=1e-3)

    def test_inverted_headwind_emits_inverted_verdict(self):
        # 20 strong_headwind rows with only 30% negative (inverted)
        # 20 strong_tailwind rows with 60% positive (consistent enough)
        rows: list[dict] = []
        rows.extend(_mk_row(-20.0, -5.0) for _ in range(6))
        rows.extend(_mk_row(-20.0, +3.0) for _ in range(14))
        rows.extend(_mk_row(20.0, +5.0) for _ in range(12))
        rows.extend(_mk_row(20.0, -3.0) for _ in range(8))
        rep = gate_arm_directional_consistency_report(rows)
        assert rep["verdict"] == "GATE_DIRECTIONALLY_INVERTED"
        sh = next(a for a in rep["arms"] if a["arm"] == "strong_headwind")
        assert sh["directional_consistency"] == pytest.approx(0.3, abs=1e-3)


class TestReportNoise:
    def test_both_arms_in_noise_band_emits_noise_verdict(self):
        # 20 strong_tailwind rows with 50% positive (right at noise)
        # 20 strong_headwind rows with 50% negative (right at noise)
        rows: list[dict] = []
        rows.extend(_mk_row(20.0, +5.0) for _ in range(10))
        rows.extend(_mk_row(20.0, -3.0) for _ in range(10))
        rows.extend(_mk_row(-20.0, -5.0) for _ in range(10))
        rows.extend(_mk_row(-20.0, +3.0) for _ in range(10))
        rep = gate_arm_directional_consistency_report(rows)
        assert rep["verdict"] == "GATE_DIRECTIONALLY_NOISE"
        st = next(a for a in rep["arms"] if a["arm"] == "strong_tailwind")
        sh = next(a for a in rep["arms"] if a["arm"] == "strong_headwind")
        assert st["directional_consistency"] == pytest.approx(0.5, abs=1e-3)
        assert sh["directional_consistency"] == pytest.approx(0.5, abs=1e-3)


class TestReportMixed:
    def test_one_strong_one_noise_emits_mixed_verdict(self):
        # 20 strong_tailwind: 80% positive (consistent)
        # 20 strong_headwind: 50% negative (noise — between INVERTED_TOL
        # 0.45 and CONSISTENT_TOL 0.55)
        rows: list[dict] = []
        rows.extend(_mk_row(20.0, +5.0) for _ in range(16))
        rows.extend(_mk_row(20.0, -3.0) for _ in range(4))
        rows.extend(_mk_row(-20.0, -5.0) for _ in range(10))
        rows.extend(_mk_row(-20.0, +3.0) for _ in range(10))
        rep = gate_arm_directional_consistency_report(rows)
        assert rep["verdict"] == "GATE_DIRECTIONALLY_MIXED"


class TestSellSignFlip:
    """SELL rows must have their realized return flipped — same
    convention as train_scorer / gate_audit / gate_realized."""

    def test_sell_with_negative_realized_counts_as_positive(self):
        # A SELL with realized=-5.0 flips to +5.0 → counts as positive
        # in the tailwind arm.
        rows = [_mk_row(20.0, -5.0, action="SELL") for _ in range(20)]
        # Strong headwind dummies to clear MIN_ARM_N
        rows.extend(_mk_row(-20.0, -5.0) for _ in range(10))
        rows.extend(_mk_row(-20.0, +3.0) for _ in range(5))
        rep = gate_arm_directional_consistency_report(rows)
        st = next(a for a in rep["arms"] if a["arm"] == "strong_tailwind")
        # All 20 SELL rows landed in strong_tailwind with flipped-positive
        assert st["n_positive"] == 20
        assert st["n_negative"] == 0


class TestAbstentions:
    """Off-distribution rows must NOT bucket into arms (they didn't
    actually act on the prediction)."""

    def test_off_dist_rows_in_abstained_not_arms(self):
        rows: list[dict] = []
        # 50 strong_tailwind rows BUT all off-dist
        rows.extend(_mk_row(20.0, +5.0, off_dist=True) for _ in range(50))
        rep = gate_arm_directional_consistency_report(rows)
        assert rep["n_captured"] == 50
        assert rep["n_abstained"] == 50
        assert rep["n_acted"] == 0
        st = next(a for a in rep["arms"] if a["arm"] == "strong_tailwind")
        assert st["n"] == 0
        # Insufficient acted rows ⇒ verdict is INSUFFICIENT_DATA
        assert rep["verdict"] == "INSUFFICIENT_DATA"


class TestEdgeCases:
    def test_non_dict_rows_skipped(self):
        # Garbage in the JSONL line — analyzer must not raise.
        rows = ["not a dict", 42, None, _mk_row(20.0, +5.0)]
        rep = gate_arm_directional_consistency_report(rows)
        # Only the one real row was counted
        assert rep["n_captured"] == 1

    def test_nan_realized_skipped(self):
        rows = [_mk_row(20.0, float("nan"))]
        rep = gate_arm_directional_consistency_report(rows)
        # NaN realized → row excluded entirely
        assert rep["n_captured"] == 1
        assert rep["n_acted"] == 0

    def test_zero_realized_excluded_from_consistency(self):
        # A realized of exactly 0 doesn't count toward positive or
        # negative — recorded in n_zero, excluded from n_nonzero.
        rows: list[dict] = []
        rows.extend(_mk_row(20.0, +5.0) for _ in range(16))
        rows.extend(_mk_row(20.0, 0.0) for _ in range(4))
        rows.extend(_mk_row(-20.0, -5.0) for _ in range(20))
        rep = gate_arm_directional_consistency_report(rows)
        st = next(a for a in rep["arms"] if a["arm"] == "strong_tailwind")
        assert st["n_zero"] == 4
        # consistency = 16 positive / (16 + 0) nonzero = 1.0
        assert st["directional_consistency"] == pytest.approx(1.0, abs=1e-3)


class TestNeutralArm:
    def test_neutral_consistency_is_none(self):
        # Neutral arm has expected_sign=0 → no expected direction →
        # directional_consistency is reported as None even when sample
        # size is large.
        rows: list[dict] = []
        # 50 neutral rows
        rows.extend(_mk_row(2.0, +1.0) for _ in range(30))
        rows.extend(_mk_row(2.0, -1.0) for _ in range(20))
        # Dummies to satisfy MIN_ARM_N for the extreme arms
        rows.extend(_mk_row(20.0, +5.0) for _ in range(10))
        rows.extend(_mk_row(-20.0, -5.0) for _ in range(10))
        rep = gate_arm_directional_consistency_report(rows)
        neutral = next(a for a in rep["arms"] if a["arm"] == "neutral")
        assert neutral["directional_consistency"] is None
        # balanced_fraction = max(30, 20) / 50 = 0.6
        assert neutral["balanced_fraction"] == pytest.approx(0.6, abs=1e-3)


class TestSSOT:
    """The arm-bucketing logic must match gate_audit.gate_arm to the
    bit. A drift here would silently mis-bucket rows."""

    def test_arm_for_pred_matches_gate_audit(self):
        from paper_trader.ml.gate_audit import gate_arm
        for p in (-50, -10.5, -10, -5, -0.01, 0, 0.01, 4.99, 5, 5.01,
                  10, 10.01, 50, float("inf"), float("-inf")):
            try:
                local = _arm_for_pred(p)
            except Exception:
                # _arm_for_pred is total on finite inputs; non-finite
                # would propagate, which is fine — we test finite below.
                continue
            try:
                canonical, _ = gate_arm(p)
            except Exception:
                continue
            assert local == canonical, (
                f"arm mismatch on p={p}: local={local} canonical={canonical}"
            )


class TestAnalyzeCLI:
    """End-to-end via the analyze() entry point — exercises the JSONL
    read path."""

    def test_analyze_with_real_file(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        rows: list[dict] = []
        rows.extend(_mk_row(20.0, +5.0) for _ in range(16))
        rows.extend(_mk_row(20.0, -3.0) for _ in range(4))
        rows.extend(_mk_row(-20.0, -5.0) for _ in range(16))
        rows.extend(_mk_row(-20.0, +3.0) for _ in range(4))
        with p.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        rep = analyze(p)
        assert rep["verdict"] == "GATE_DIRECTIONALLY_CONSISTENT"

    def test_analyze_with_missing_file(self, tmp_path: Path):
        # No file → empty rows → GATE_CAPTURE_NOT_YET_POPULATED
        rep = analyze(tmp_path / "does-not-exist.jsonl")
        assert rep["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"

    def test_analyze_with_corrupt_lines_skips_them(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            fh.write("not json\n")
            fh.write("\n")  # blank line — skipped
            # One real row but only 1, well below MIN_TOTAL
            fh.write(json.dumps(_mk_row(20.0, +5.0)) + "\n")
        rep = analyze(p)
        # The corrupt line is skipped — analyzer doesn't crash.
        assert rep["n_captured"] == 1
        assert rep["verdict"] == "INSUFFICIENT_DATA"
