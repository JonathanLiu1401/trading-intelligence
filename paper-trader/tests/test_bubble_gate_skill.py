"""Tests for paper_trader.ml.bubble_gate_skill.

Tests assert SPECIFIC verdicts on hand-crafted synthetic outcomes —
each test exercises one verdict branch with the exact-value contract
the verdict logic claims. The bubble-gate skill diagnostic must:

* report ``BUBBLE_GATE_HARMFUL`` when at-52-week-high BUYs realize
  MORE than mid-range BUYs (the documented "near-high IS a breakout
  signal, not a bubble trap" outcome the gate's hypothesis falsifies),
* report ``BUBBLE_GATE_JUSTIFIED`` when at-high BUYs realize LESS,
* report ``BUBBLE_GATE_NEUTRAL`` within the EDGE_TOL_PP band,
* report ``INSUFFICIENT_DATA`` below MIN_TOTAL / MIN_BUCKET_N,
* report ``WK52_NOT_YET_POPULATED`` when no BUY carries wk52_pos,
* exclude SELL rows entirely (gate is BUY-only by construction).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_trader.ml.bubble_gate_skill import (
    _bucket_for,
    bubble_gate_skill_report,
    analyze,
    _BUCKETS,
    MIN_TOTAL,
    MIN_BUCKET_N,
    EDGE_TOL_PP,
)


# ─────────────────────────── _bucket_for ────────────────────────────


class TestBucketFor:
    def test_deep_discount(self):
        assert _bucket_for(0.0) == "deep_discount"
        assert _bucket_for(0.25) == "deep_discount"
        assert _bucket_for(0.4999) == "deep_discount"

    def test_mid(self):
        assert _bucket_for(0.5) == "mid"
        assert _bucket_for(0.65) == "mid"
        assert _bucket_for(0.6999) == "mid"

    def test_near_high(self):
        assert _bucket_for(0.7) == "near_high"
        assert _bucket_for(0.75) == "near_high"
        assert _bucket_for(0.7999) == "near_high"

    def test_at_high(self):
        assert _bucket_for(0.8) == "at_high"
        assert _bucket_for(0.95) == "at_high"
        # 1.0 (pure 52-week high) must be captured by the at_high bucket.
        # The upper boundary is inclusive of 1.0 (right-CLOSED on 1.0 only).
        assert _bucket_for(1.0) == "at_high"

    def test_out_of_range_returns_none(self):
        assert _bucket_for(-0.1) is None
        assert _bucket_for(1.5) is None

    def test_buckets_cover_unit_interval(self):
        """The 4 buckets must tile [0.0, 1.0] with no gaps or overlaps.
        A gap would silently drop a wk52_pos value from every bucket;
        an overlap would double-count it."""
        # 100 points from 0 to 1
        n = 100
        for i in range(n + 1):
            x = i / n
            b = _bucket_for(x)
            assert b is not None, f"wk52={x} fell into no bucket"


# ──────────────────────── verdict branches ───────────────────────────


def _row(wk52, fwd, action="BUY"):
    return {
        "action": action,
        "ticker": "NVDA",
        "wk52_pos": wk52,
        "forward_return_5d": fwd,
    }


class TestBubbleGateSkillVerdicts:
    def test_empty_input(self):
        rep = bubble_gate_skill_report([])
        assert rep["status"] == "ok"
        assert rep["verdict"] == "WK52_NOT_YET_POPULATED"
        assert rep["n_buys"] == 0
        assert rep["n_with_wk52"] == 0

    def test_all_sells_yields_no_buys(self):
        # SELL rows must be excluded — the bubble gate is BUY-only.
        rows = [_row(0.85, 5.0, action="SELL") for _ in range(50)]
        rep = bubble_gate_skill_report(rows)
        assert rep["n_buys"] == 0
        assert rep["verdict"] == "WK52_NOT_YET_POPULATED"

    def test_buys_without_wk52_yields_capture_pending(self):
        rows = [{"action": "BUY", "forward_return_5d": 5.0} for _ in range(40)]
        rep = bubble_gate_skill_report(rows)
        assert rep["n_buys"] == 40
        assert rep["n_with_wk52"] == 0
        assert rep["verdict"] == "WK52_NOT_YET_POPULATED"

    def test_insufficient_data_below_min_total(self):
        # 10 BUY rows — below MIN_TOTAL even if buckets balanced.
        rows = ([_row(0.85, 5.0) for _ in range(5)]
                + [_row(0.60, 2.0) for _ in range(5)])
        rep = bubble_gate_skill_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_insufficient_data_when_at_high_bucket_too_small(self):
        # Plenty of mid-bucket BUYs but only 3 at_high (below MIN_BUCKET_N=5).
        rows = ([_row(0.85, 5.0) for _ in range(3)]
                + [_row(0.60, 2.0) for _ in range(40)])
        rep = bubble_gate_skill_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_bubble_gate_harmful_when_athigh_outperforms(self):
        # Synthetic data: at_high BUYs realize +8%, mid BUYs realize +1%.
        # Spread = +7pp > +EDGE_TOL_PP → BUBBLE_GATE_HARMFUL.
        rows = ([_row(0.85, 8.0) for _ in range(20)]
                + [_row(0.60, 1.0) for _ in range(20)])
        rep = bubble_gate_skill_report(rows)
        assert rep["verdict"] == "BUBBLE_GATE_HARMFUL"
        assert rep["at_high_minus_mid_pp"] == pytest.approx(7.0)
        assert "OUTPERFORMED" in rep["hint"]

    def test_bubble_gate_justified_when_athigh_underperforms(self):
        # Inverse: at_high realizes -5%, mid realizes +2%.
        rows = ([_row(0.85, -5.0) for _ in range(20)]
                + [_row(0.60, 2.0) for _ in range(20)])
        rep = bubble_gate_skill_report(rows)
        assert rep["verdict"] == "BUBBLE_GATE_JUSTIFIED"
        assert rep["at_high_minus_mid_pp"] == pytest.approx(-7.0)

    def test_bubble_gate_neutral_within_edge_band(self):
        # Spread under EDGE_TOL_PP (1.0pp): at_high +2.5%, mid +2.0% → +0.5pp.
        rows = ([_row(0.85, 2.5) for _ in range(20)]
                + [_row(0.60, 2.0) for _ in range(20)])
        rep = bubble_gate_skill_report(rows)
        assert rep["verdict"] == "BUBBLE_GATE_NEUTRAL"
        assert abs(rep["at_high_minus_mid_pp"]) <= EDGE_TOL_PP

    def test_bucket_means_computed_correctly(self):
        # at_high: avg of [5, 7, 9] = 7. mid: avg of [1, 3] = 2.
        rows = [_row(0.85, 5.0), _row(0.85, 7.0), _row(0.85, 9.0),
                _row(0.85, 5.0), _row(0.85, 7.0),  # 5 total at_high (MIN_BUCKET_N)
                _row(0.60, 1.0), _row(0.60, 3.0),
                _row(0.60, 2.0), _row(0.60, 4.0),
                _row(0.60, 0.0)]  # 5 mid
        # Top up to MIN_TOTAL with a third bucket — uses near_high (not in
        # the spread, but counts toward n_with_fwd total).
        rows += [_row(0.75, 0.0) for _ in range(20)]
        rep = bubble_gate_skill_report(rows)
        # Buckets in order; the at_high bucket has mean (5+7+9+5+7)/5 = 6.6
        at_high_b = next(b for b in rep["buckets"] if b["bucket"] == "at_high")
        assert at_high_b["mean_realized_5d"] == pytest.approx(6.6)
        assert at_high_b["n"] == 5
        # mid bucket has mean (1+3+2+4+0)/5 = 2.0
        mid_b = next(b for b in rep["buckets"] if b["bucket"] == "mid")
        assert mid_b["mean_realized_5d"] == pytest.approx(2.0)


class TestBucketMonotone:
    def test_monotone_when_means_strictly_increasing(self):
        # Synthesized so each bucket has a clean mean.
        rows = ([_row(0.30, -2.0) for _ in range(5)]    # deep_discount
                + [_row(0.60, 0.0) for _ in range(5)]    # mid
                + [_row(0.75, 2.0) for _ in range(5)]    # near_high
                + [_row(0.85, 4.0) for _ in range(20)])  # at_high (needs ≥5)
        # Note: enough total BUYs for the verdict.
        rows += [_row(0.60, 0.0) for _ in range(15)]
        rep = bubble_gate_skill_report(rows)
        # 4 buckets present → 3 steps, all non-decreasing → 3/3 = 1.0
        assert rep["bucket_monotone_fraction"] == pytest.approx(1.0)


class TestRobustness:
    """Defensive: a malformed row must NOT raise, must NOT poison the
    audit — same discipline as gate_realized."""

    def test_handles_none_input(self):
        rep = bubble_gate_skill_report(None)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "WK52_NOT_YET_POPULATED"

    def test_handles_garbage_rows(self):
        rows = [None, "not a dict", 12345, {"action": "BUY"}]
        rep = bubble_gate_skill_report(rows)
        # None entries / non-dicts dropped. The single bare-dict BUY has
        # no wk52_pos so n_with_wk52 stays 0.
        assert rep["n_buys"] == 1
        assert rep["n_with_wk52"] == 0

    def test_non_finite_wk52_dropped(self):
        rows = [_row(float("inf"), 5.0), _row(float("nan"), 5.0),
                _row(2.0, 5.0),  # out of [0, 1] — dropped
                _row(-0.5, 5.0)]  # out of [0, 1] — dropped
        rep = bubble_gate_skill_report(rows)
        assert rep["n_with_wk52"] == 0

    def test_non_finite_forward_return_dropped(self):
        rows = ([_row(0.85, float("nan")) for _ in range(10)]
                + [_row(0.85, 5.0) for _ in range(5)])
        rep = bubble_gate_skill_report(rows)
        # at_high count drops to 5 (only the finite returns counted).
        at_high_b = next(b for b in rep["buckets"] if b["bucket"] == "at_high")
        assert at_high_b["n"] == 5

    def test_bool_label_treated_as_missing(self):
        # bool is an int subclass — must NOT be coerced to 1.0/0.0 (matches
        # `_to_float` in decision_scorer).
        rows = [_row(True, 5.0) for _ in range(10)]
        rep = bubble_gate_skill_report(rows)
        assert rep["n_with_wk52"] == 0  # all True wk52 values dropped


# ─────────────────────────── analyze (file IO) ───────────────────────


class TestAnalyzeFromFile:
    def test_missing_file(self, tmp_path):
        p = tmp_path / "missing.jsonl"
        rep = analyze(p)
        assert rep["status"] == "error"
        assert rep["verdict"] == "WK52_NOT_YET_POPULATED"
        assert "no outcomes file" in rep["hint"]

    def test_reads_full_file_when_oos_only_false(self, tmp_path):
        rows = ([_row(0.85, 8.0) for _ in range(20)]
                + [_row(0.60, 1.0) for _ in range(20)])
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        rep = analyze(p, oos_only=False)
        assert rep["verdict"] == "BUBBLE_GATE_HARMFUL"
        assert rep["slice"] == "all"

    def test_handles_corrupt_jsonl_lines(self, tmp_path):
        # Should skip malformed lines without crashing.
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as f:
            f.write("{not valid json\n")
            f.write(json.dumps(_row(0.85, 5.0)) + "\n")
            f.write("\n")  # empty line
            f.write("garbage\n")
        rep = analyze(p, oos_only=False)
        assert rep["status"] == "ok"
        assert rep["n_buys"] == 1
