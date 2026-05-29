"""Tests for paper_trader.ml.gate_arm_kelly.

Verifies the per-arm Kelly fraction computation, the verdict ladder,
and the row filtering (off-distribution → abstained bucket, captured
gate_scorer_pred required).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from paper_trader.ml.gate_arm_kelly import (
    KELLY_CAP,
    MIN_ARM_N,
    MIN_TOTAL,
    SHARPE_TOL,
    _ARM_MULTIPLIER,
    _ARM_ORDER,
    _kelly,
    _per_arm_stats,
    analyze,
    gate_arm_kelly_report,
)


# ──────────────────────────────────────────────────────────────────────────
# _kelly: pure function
# ──────────────────────────────────────────────────────────────────────────


def test_kelly_clamps_negative_mean_to_zero():
    """Don't bet on a losing arm — Kelly = 0 when mean ≤ 0."""
    assert _kelly(-5.0, 10.0) == 0.0
    assert _kelly(0.0, 10.0) == 0.0


def test_kelly_returns_none_on_degenerate_stdev():
    """σ ≤ 0 ⇒ Kelly undefined."""
    assert _kelly(5.0, 0.0) is None
    assert _kelly(5.0, -1.0) is None


def test_kelly_returns_none_on_nan_inputs():
    """Non-finite inputs → None (never raises)."""
    assert _kelly(float("nan"), 10.0) is None
    assert _kelly(5.0, float("inf")) is None


def test_kelly_formula_correct():
    """f* = μ / σ² where both are decimals.

    Concrete: μ=2%, σ=10% → μ_decimal=0.02, σ²=0.01 → f*=2.0, clamped to 1.0.
    """
    # μ=2%, σ=10% → 0.02 / 0.01 = 2.0 → clamped to KELLY_CAP
    assert _kelly(2.0, 10.0) == KELLY_CAP
    # μ=1%, σ=10% → 0.01 / 0.01 = 1.0 → exact
    assert abs(_kelly(1.0, 10.0) - 1.0) < 1e-9
    # μ=0.5%, σ=10% → 0.005 / 0.01 = 0.5
    assert abs(_kelly(0.5, 10.0) - 0.5) < 1e-9
    # μ=1%, σ=20% → 0.01 / 0.04 = 0.25
    assert abs(_kelly(1.0, 20.0) - 0.25) < 1e-9


def test_kelly_capped_at_one():
    """No leverage above 100% — even tiny σ doesn't escape the cap."""
    # μ=10%, σ=1% → 0.1/0.0001 = 1000 → clamped to KELLY_CAP
    assert _kelly(10.0, 1.0) == KELLY_CAP


# ──────────────────────────────────────────────────────────────────────────
# _per_arm_stats: per-arm summary
# ──────────────────────────────────────────────────────────────────────────


def test_per_arm_stats_empty_arm():
    """Empty arm → all None/0 fields."""
    s = _per_arm_stats([])
    assert s["n"] == 0
    assert s["mean_pct"] is None
    assert s["stdev_pct"] is None
    assert s["sharpe_per_trade"] is None
    assert s["kelly_fraction"] is None


def test_per_arm_stats_single_sample():
    """n=1 → mean defined, stdev undefined, sharpe None."""
    s = _per_arm_stats([5.0])
    assert s["n"] == 1
    assert s["mean_pct"] == 5.0
    assert s["stdev_pct"] is None
    assert s["sharpe_per_trade"] is None  # undefined for n=1
    assert s["kelly_fraction"] is None
    assert s["win_rate"] == 1.0


def test_per_arm_stats_known_distribution():
    """Concrete sample → exact mean/stdev/sharpe."""
    realized = [1.0, 2.0, 3.0, 4.0, 5.0]  # mean=3, sample stdev=√2.5≈1.5811
    s = _per_arm_stats(realized)
    assert s["n"] == 5
    assert s["mean_pct"] == 3.0
    assert s["median_pct"] == 3.0
    assert abs(s["stdev_pct"] - 1.5811) < 1e-3
    # Sharpe = 3 / 1.5811 ≈ 1.8974
    assert abs(s["sharpe_per_trade"] - 1.8974) < 1e-3
    # All five positive → win rate 1.0
    assert s["win_rate"] == 1.0
    # Kelly = 0.03 / (0.015811)^2 ≈ 120 → capped at 1.0
    assert s["kelly_fraction"] == KELLY_CAP


def test_per_arm_stats_mixed_wins_and_losses():
    """Win rate counts strict positives only."""
    s = _per_arm_stats([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert s["n"] == 5
    assert s["mean_pct"] == 0.0
    assert s["win_rate"] == 0.4  # 2 of 5 strictly positive
    # μ=0 → Kelly=0 (don't bet on zero EV)
    assert s["kelly_fraction"] == 0.0


# ──────────────────────────────────────────────────────────────────────────
# gate_arm_kelly_report: bucketing + verdict
# ──────────────────────────────────────────────────────────────────────────


def _row(pred, ret, action="BUY", off_dist=False):
    """Build a synthetic outcome row."""
    return {
        "gate_scorer_pred": pred,
        "forward_return_5d": ret,
        "action": action,
        "gate_off_dist": off_dist,
    }


def test_report_excludes_rows_without_gate_scorer_pred():
    """Rows without a captured gate prediction (None) are dropped entirely."""
    rows = [
        {"gate_scorer_pred": None, "forward_return_5d": 5.0, "action": "BUY"},
        _row(8.0, 3.0),
    ]
    r = gate_arm_kelly_report(rows)
    # Only 1 captured, 1 acted; no abstentions.
    assert r["n_captured"] == 1
    assert r["n_acted"] == 1
    assert r["n_abstained"] == 0


def test_report_routes_off_dist_to_abstained_bucket():
    """off_dist=True rows go to abstained, not an arm."""
    rows = [_row(12.0, 5.0, off_dist=True), _row(-15.0, -3.0, off_dist=True),
            _row(3.0, 1.0, off_dist=False)]
    r = gate_arm_kelly_report(rows)
    assert r["n_captured"] == 3
    assert r["n_acted"] == 1
    assert r["n_abstained"] == 2
    assert r["abstained"]["n"] == 2


def test_report_sell_sign_flip():
    """SELL with positive realized return is recorded as negative (the model
    learns "good" with one consistent meaning)."""
    # Place this in the neutral arm (pred=2 → 0 < 2 ≤ 5 → neutral).
    rows = [_row(2.0, 5.0, action="SELL")]  # realized -5 after flip
    r = gate_arm_kelly_report(rows)
    neutral = next(a for a in r["per_arm"] if a["arm"] == "neutral")
    assert neutral["n"] == 1
    assert neutral["mean_pct"] == -5.0


def test_report_arms_match_gate_arm_decoder():
    """Bucketing follows gate_audit.gate_arm boundaries exactly."""
    rows = [
        _row(-15.0, 1.0),  # strong_headwind  (p < -10)
        _row(-5.0, 2.0),   # mild_headwind    (-10 ≤ p < 0)
        _row(0.0, 3.0),    # neutral          (0 ≤ p ≤ 5)
        _row(7.0, 4.0),    # mild_tailwind    (5 < p ≤ 10)
        _row(20.0, 5.0),   # strong_tailwind  (p > 10)
    ]
    r = gate_arm_kelly_report(rows)
    arms_by_name = {a["arm"]: a for a in r["per_arm"]}
    assert arms_by_name["strong_headwind"]["n"] == 1
    assert arms_by_name["mild_headwind"]["n"] == 1
    assert arms_by_name["neutral"]["n"] == 1
    assert arms_by_name["mild_tailwind"]["n"] == 1
    assert arms_by_name["strong_tailwind"]["n"] == 1


def test_report_insufficient_data_verdict():
    """< MIN_TOTAL rows → INSUFFICIENT_DATA."""
    rows = [_row(8.0, 3.0)]
    r = gate_arm_kelly_report(rows)
    assert r["verdict"] == "INSUFFICIENT_DATA"


def test_report_capture_not_populated():
    """Zero captured rows → GATE_CAPTURE_NOT_YET_POPULATED."""
    rows = [{"gate_scorer_pred": None, "forward_return_5d": 5.0}]
    r = gate_arm_kelly_report(rows)
    assert r["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"


def test_report_kelly_aligned_when_tailwind_outperforms():
    """When the strong_tailwind arm has BIG positive Sharpe and
    strong_headwind has zero/negative, verdict is KELLY_ALIGNED."""
    rows = []
    # Strong_tailwind: high mean, low variance — high Sharpe.
    for _ in range(MIN_ARM_N + 10):
        rows.append(_row(15.0, 5.0))   # always +5
        rows.append(_row(15.0, 6.0))   # alternation for nonzero stdev
    # Strong_headwind: low/negative mean.
    for _ in range(MIN_ARM_N + 10):
        rows.append(_row(-15.0, -3.0))
        rows.append(_row(-15.0, -4.0))
    # Pad with mid-arm rows to hit MIN_TOTAL acted-row threshold easily.
    for _ in range(MIN_TOTAL):
        rows.append(_row(2.0, 0.5))    # neutral arm
    r = gate_arm_kelly_report(rows)
    assert r["verdict"] == "KELLY_ALIGNED", (
        f"expected KELLY_ALIGNED, got {r['verdict']} "
        f"(spread={r.get('sharpe_spread')})"
    )


def test_report_kelly_inverted_when_headwind_outperforms():
    """When strong_headwind has BETTER Sharpe than strong_tailwind, the
    gate's directionality is wrong — verdict KELLY_INVERTED."""
    rows = []
    # Strong_tailwind: NEGATIVE mean (bad outcomes for the BIG bet).
    for _ in range(MIN_ARM_N + 10):
        rows.append(_row(15.0, -3.0))
        rows.append(_row(15.0, -4.0))
    # Strong_headwind: positive mean (good outcomes for the SMALL bet).
    for _ in range(MIN_ARM_N + 10):
        rows.append(_row(-15.0, 5.0))
        rows.append(_row(-15.0, 6.0))
    # Pad to MIN_TOTAL.
    for _ in range(MIN_TOTAL):
        rows.append(_row(2.0, 0.5))
    r = gate_arm_kelly_report(rows)
    assert r["verdict"] == "KELLY_INVERTED", (
        f"expected KELLY_INVERTED, got {r['verdict']} "
        f"(spread={r.get('sharpe_spread')})"
    )


def test_report_per_arm_multiplier_match_gate_audit():
    """Each per_arm row carries the actual gate multiplier (×0.60..×1.30)."""
    rows = [_row(-15.0, 1.0)] * MIN_ARM_N + [_row(15.0, 2.0)] * MIN_ARM_N
    r = gate_arm_kelly_report(rows)
    arms = {a["arm"]: a for a in r["per_arm"]}
    assert arms["strong_headwind"]["multiplier"] == 0.60
    assert arms["strong_tailwind"]["multiplier"] == 1.30


# ──────────────────────────────────────────────────────────────────────────
# analyze: file-based entry + JSON safety
# ──────────────────────────────────────────────────────────────────────────


def test_analyze_handles_missing_file(tmp_path):
    """Missing outcomes file → status='error' with INSUFFICIENT_DATA verdict.
    Never raises."""
    bogus = tmp_path / "does_not_exist.jsonl"
    r = analyze(bogus, oos_only=False)
    assert r.get("status") == "error"
    assert r.get("verdict") == "INSUFFICIENT_DATA"


def test_analyze_full_corpus(tmp_path):
    """End-to-end: write a small JSONL and verify the analyzer reads it."""
    path = tmp_path / "outcomes.jsonl"
    rows = [
        _row(8.0, 3.0),
        _row(-12.0, 1.0),
        _row(2.0, 0.5),
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    r = analyze(path, oos_only=False)
    assert r.get("status") == "ok"
    assert r.get("n_captured") == 3
    assert r.get("oos_only") is False
    # Verify JSON-serializable
    json.dumps(r)


def test_analyze_skips_malformed_lines(tmp_path):
    """A garbled line in the middle of the JSONL doesn't crash the read."""
    path = tmp_path / "outcomes.jsonl"
    good = json.dumps(_row(8.0, 3.0))
    path.write_text(f"{good}\n{{not json}}\n{good}\n")
    r = analyze(path, oos_only=False)
    assert r.get("status") == "ok"
    assert r.get("n_captured") == 2  # one bad line skipped


# ──────────────────────────────────────────────────────────────────────────
# Sanity: never raises on degenerate inputs
# ──────────────────────────────────────────────────────────────────────────


def test_report_handles_garbage_input():
    """Non-iterable / random garbage → empty report, no crash."""
    r = gate_arm_kelly_report(None)
    assert r["n_captured"] == 0
    r2 = gate_arm_kelly_report([{"not": "a row"}, 42, "string", None])
    assert r2["n_captured"] == 0
