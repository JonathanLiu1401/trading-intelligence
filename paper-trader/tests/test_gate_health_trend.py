"""Tests for paper_trader.ml.gate_health_trend.

The analyzer reads scorer_skill_log.jsonl and reports kill-switch
health + trend. Tests exercise the verdict ladder boundaries with
hand-crafted IC time-series so each verdict is pinned at threshold
± one step.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_trader.ml.gate_health_trend import (
    GATE_THRESHOLD,
    MIN_CYCLES_FOR_TREND,
    SLOPE_SIGNIFICANCE,
    TREND_WINDOW,
    _cli,
    analyze,
    gate_health_trend_report,
)


# ─────────────────────────── helpers ───────────────────────────

def _ic_rows(ics, gate_actives=None):
    """Build skill-log-style rows from a sequence of (ic, gate_active) tuples
    or a flat ics list. Missing flag → row omits the key (legacy shape)."""
    if gate_actives is None:
        gate_actives = [None] * len(ics)
    return [
        ({"oos_buy_ic": ic, "gate_effectively_active": g}
         if g is not None
         else {"oos_buy_ic": ic})
        for ic, g in zip(ics, gate_actives)
    ]


# ─────────────────────────── INSUFFICIENT_DATA ───────────────────────────

class TestInsufficientData:
    def test_empty_rows(self):
        r = gate_health_trend_report([])
        assert r["verdict"] == "INSUFFICIENT_DATA"
        assert r["n"] == 0

    def test_below_min_cycles(self):
        r = gate_health_trend_report(_ic_rows([0.0] * (MIN_CYCLES_FOR_TREND - 1)))
        assert r["verdict"] == "INSUFFICIENT_DATA"
        assert r["n"] == MIN_CYCLES_FOR_TREND - 1

    def test_min_cycles_boundary(self):
        """Exactly MIN_CYCLES_FOR_TREND rows → NOT INSUFFICIENT."""
        # All zeros → GATE_DARK_FLAT (median=0 < threshold, slope ≈ 0).
        r = gate_health_trend_report(_ic_rows([0.0] * MIN_CYCLES_FOR_TREND))
        assert r["verdict"] != "INSUFFICIENT_DATA"
        assert r["n"] == MIN_CYCLES_FOR_TREND

    def test_invalid_ic_values_filtered(self):
        """None / NaN / strings in oos_buy_ic don't poison the count."""
        rows = (_ic_rows([0.0] * 5)
                + [{"oos_buy_ic": None}, {"oos_buy_ic": "bad"},
                   {"oos_buy_ic": float("nan")}, {"oos_buy_ic": float("inf")}]
                + _ic_rows([0.0] * 10))
        r = gate_health_trend_report(rows)
        # 15 valid IC rows, < MIN_CYCLES_FOR_TREND → INSUFFICIENT
        assert r["n"] == 15
        assert r["verdict"] == "INSUFFICIENT_DATA"


# ─────────────────────────── verdict ladder ───────────────────────────

class TestGateDarkFlat:
    def test_flat_at_zero(self):
        r = gate_health_trend_report(_ic_rows([0.0] * 30))
        assert r["verdict"] == "GATE_DARK_FLAT"
        assert r["trailing_median_20"] == 0.0
        assert abs(r["slope_20"]) <= SLOPE_SIGNIFICANCE

    def test_flat_below_threshold(self):
        """Stuck at +0.01 (below +0.03 threshold) with no slope."""
        r = gate_health_trend_report(_ic_rows([0.01] * 30))
        assert r["verdict"] == "GATE_DARK_FLAT"
        assert r["trailing_median_20"] == 0.01

    def test_cycles_to_threshold_is_none_when_flat(self):
        r = gate_health_trend_report(_ic_rows([0.0] * 30))
        assert r["cycles_to_threshold"] is None


class TestGateDarkRecovering:
    def test_steady_rise_below_threshold(self):
        """Slope clearly above significance, median still below threshold."""
        # Rising from -0.05 to +0.02 over 30 cycles. Trailing-20 median
        # ≈ middle of last 20 ≈ around 0.0; slope is +0.0024/cycle.
        ics = [-0.05 + i * 0.0024 for i in range(30)]
        r = gate_health_trend_report(_ic_rows(ics))
        assert r["verdict"] == "GATE_DARK_RECOVERING"
        assert r["slope_20"] > SLOPE_SIGNIFICANCE
        assert r["trailing_median_20"] < GATE_THRESHOLD
        assert r["cycles_to_threshold"] is not None
        assert r["cycles_to_threshold"] > 0


class TestGateDarkDeteriorating:
    def test_falling_below_zero(self):
        """Median below zero AND slope significantly negative."""
        # Falling from 0.0 to -0.06 over 30 cycles. Slope = -0.002/cycle.
        ics = [0.0 - i * 0.002 for i in range(30)]
        r = gate_health_trend_report(_ic_rows(ics))
        assert r["verdict"] == "GATE_DARK_DETERIORATING"
        assert r["trailing_median_20"] < 0.0
        assert r["slope_20"] < -SLOPE_SIGNIFICANCE


class TestGateActiveStable:
    def test_consistently_above_threshold(self):
        """Median > threshold, flat slope."""
        r = gate_health_trend_report(_ic_rows([0.10] * 30))
        assert r["verdict"] == "GATE_ACTIVE_STABLE"
        assert r["trailing_median_20"] >= GATE_THRESHOLD

    def test_above_threshold_with_positive_slope(self):
        """Active and improving → still GATE_ACTIVE_STABLE."""
        ics = [0.05 + i * 0.001 for i in range(30)]
        r = gate_health_trend_report(_ic_rows(ics))
        assert r["verdict"] == "GATE_ACTIVE_STABLE"

    def test_threshold_boundary_exactly(self):
        """Exactly at +0.03 (the threshold) → still ACTIVE per >= comparison."""
        r = gate_health_trend_report(_ic_rows([0.03] * 30))
        assert r["verdict"] == "GATE_ACTIVE_STABLE"


class TestGateActiveDeteriorating:
    def test_active_but_falling(self):
        """Active median but trending toward threshold."""
        # Falling from +0.15 to +0.06 over 30 cycles. Slope ≈ -0.003.
        ics = [0.15 - i * 0.003 for i in range(30)]
        r = gate_health_trend_report(_ic_rows(ics))
        assert r["verdict"] == "GATE_ACTIVE_DETERIORATING"
        assert r["trailing_median_20"] >= GATE_THRESHOLD
        assert r["slope_20"] < -SLOPE_SIGNIFICANCE


# ─────────────────────────── medians and slope ───────────────────────────

class TestTrailingMedians:
    def test_median_20_matches_numpy(self):
        import numpy as np
        ics = [0.01, 0.02, -0.01, 0.05] * 7  # 28 rows
        r = gate_health_trend_report(_ic_rows(ics))
        expected = round(float(np.median(ics[-TREND_WINDOW:])), 4)
        assert r["trailing_median_20"] == expected

    def test_median_5_uses_last_five(self):
        # Last 5 are [0.10] → median 0.10. Bulk is 0.0 to make this distinct.
        ics = [0.0] * 25 + [0.10] * 5
        r = gate_health_trend_report(_ic_rows(ics))
        assert r["trailing_median_5"] == 0.10

    def test_median_50_none_when_log_shorter(self):
        r = gate_health_trend_report(_ic_rows([0.0] * 25))
        # 25 < 50, so trailing_median_50 is None
        assert r["trailing_median_50"] is None

    def test_median_50_set_when_log_long_enough(self):
        r = gate_health_trend_report(_ic_rows([0.0] * 60))
        assert r["trailing_median_50"] == 0.0


class TestSlope:
    def test_slope_zero_for_flat_series(self):
        r = gate_health_trend_report(_ic_rows([0.05] * 30))
        assert abs(r["slope_20"]) < SLOPE_SIGNIFICANCE

    def test_slope_positive_for_rising_series(self):
        ics = list(range(30))  # 0, 1, 2, ...
        ics = [i * 0.001 for i in ics]
        r = gate_health_trend_report(_ic_rows(ics))
        # Slope over last 20 should be ≈ 0.001/cycle.
        assert r["slope_20"] == pytest.approx(0.001, abs=1e-4)

    def test_slope_negative_for_falling_series(self):
        ics = [(-i) * 0.002 for i in range(30)]
        r = gate_health_trend_report(_ic_rows(ics))
        assert r["slope_20"] == pytest.approx(-0.002, abs=1e-4)


class TestCyclesToThreshold:
    def test_set_only_when_dark_and_rising(self):
        # Gate active → no projection needed.
        r = gate_health_trend_report(_ic_rows([0.1] * 30))
        assert r["cycles_to_threshold"] is None

        # Gate dark but flat → no projection (slope <= 0).
        r = gate_health_trend_report(_ic_rows([0.0] * 30))
        assert r["cycles_to_threshold"] is None

    def test_projection_reasonable(self):
        """A trailing-20 median below threshold with positive slope should
        produce a finite cycles_to_threshold projection."""
        # Build a rising series whose trailing-20 stays below +0.03 but the
        # slope is significantly positive. Linear ramp from -0.04 to +0.015
        # over 30 cycles. The trailing-20 (indices 10..29) is -0.005 to
        # +0.015 → median ≈ +0.005, below threshold.
        ics = [-0.04 + i * 0.0019 for i in range(30)]
        r = gate_health_trend_report(_ic_rows(ics))
        assert r["verdict"] == "GATE_DARK_RECOVERING"
        # The exact projection depends on the trailing-20 median; just
        # check it's a small positive int (i.e. recovery is in sight).
        assert r["cycles_to_threshold"] is not None
        assert 1 <= r["cycles_to_threshold"] < 100


# ─────────────────────────── gate-active tracking ───────────────────────────

class TestActiveTracking:
    def test_no_active_flags_at_all(self):
        rows = _ic_rows([0.0] * 30, gate_actives=[False] * 30)
        r = gate_health_trend_report(rows)
        assert r["n_active"] == 0
        assert r["n_active_in_last_20"] == 0
        assert r["cycles_since_active"] is None
        assert r["has_never_been_active"] is True

    def test_has_never_been_active_when_flags_missing(self):
        """Legacy rows (no gate_effectively_active key) count as 'unknown' —
        n_active is 0 → has_never_been_active=True."""
        r = gate_health_trend_report(_ic_rows([0.0] * 30))
        assert r["n_active"] == 0
        assert r["has_never_been_active"] is True

    def test_active_in_history(self):
        # Active at index 5 from end; the rest False.
        flags = [False] * 30
        flags[-6] = True
        rows = _ic_rows([0.0] * 30, gate_actives=flags)
        r = gate_health_trend_report(rows)
        assert r["n_active"] == 1
        assert r["cycles_since_active"] == 5
        assert r["has_never_been_active"] is False

    def test_n_active_in_last_20(self):
        # 21 cycles: index 0 active, then everything False for last 20.
        flags = [True] + [False] * 20
        ics = [0.0] * 21
        # We need >= MIN_CYCLES_FOR_TREND = 20 ICs; add filler rows.
        # All 21 ICs are present (=0.0).
        r = gate_health_trend_report(_ic_rows(ics, gate_actives=flags))
        assert r["n_active"] == 1
        assert r["n_active_in_last_20"] == 0  # the True is at index 0, last_20 starts at index 1

    def test_active_in_last_20(self):
        flags = [False] * 20 + [True]
        ics = [0.0] * 21
        r = gate_health_trend_report(_ic_rows(ics, gate_actives=flags))
        assert r["n_active_in_last_20"] == 1
        assert r["cycles_since_active"] == 0


# ─────────────────────────── IC distribution ───────────────────────────

class TestIcDistribution:
    def test_ic_stats_computed(self):
        ics = [0.0, 0.05, -0.05, 0.10, -0.10] * 6  # 30 values
        r = gate_health_trend_report(_ic_rows(ics))
        assert r["ic_min"] == -0.10
        assert r["ic_max"] == 0.10
        assert r["ic_mean"] == 0.0


# ─────────────────────────── analyze() and CLI ───────────────────────────

class TestAnalyze:
    def test_missing_log_returns_error(self, tmp_path):
        log = tmp_path / "absent.jsonl"
        out = analyze(log)
        assert out["status"] == "error"
        assert "missing" in out["error"]
        assert out["verdict"] == "INSUFFICIENT_DATA"

    def test_reads_full_log(self, tmp_path):
        log = tmp_path / "skill.jsonl"
        rows = _ic_rows([0.0] * 30)
        with log.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        out = analyze(log)
        assert out["status"] == "ok"
        assert out["n"] == 30
        assert out["verdict"] == "GATE_DARK_FLAT"

    def test_corrupted_lines_skipped(self, tmp_path):
        log = tmp_path / "corrupted.jsonl"
        with log.open("w") as fh:
            fh.write("this is not json\n")
            fh.write("\n")  # blank line
            for r in _ic_rows([0.05] * 30):
                fh.write(json.dumps(r) + "\n")
            fh.write("{not_valid_json\n")
        out = analyze(log)
        # 30 valid rows survive the corruption; verdict is active-stable
        # (median 0.05 ≥ 0.03 threshold, flat slope).
        assert out["status"] == "ok"
        assert out["n"] == 30
        assert out["verdict"] == "GATE_ACTIVE_STABLE"


class TestCli:
    def test_cli_returns_0_on_active(self, tmp_path, capsys):
        log = tmp_path / "active.jsonl"
        with log.open("w") as fh:
            for r in _ic_rows([0.05] * 30):
                fh.write(json.dumps(r) + "\n")
        rc = _cli(["--log", str(log)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "GATE_ACTIVE_STABLE" in out

    def test_cli_returns_1_on_dark(self, tmp_path, capsys):
        log = tmp_path / "dark.jsonl"
        with log.open("w") as fh:
            for r in _ic_rows([0.0] * 30):
                fh.write(json.dumps(r) + "\n")
        rc = _cli(["--log", str(log)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "GATE_DARK_FLAT" in out

    def test_cli_returns_1_on_missing(self, tmp_path, capsys):
        log = tmp_path / "nope.jsonl"
        rc = _cli(["--log", str(log), "--json"])
        assert rc == 1
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["status"] == "error"

    def test_cli_json_emits_full_envelope(self, tmp_path, capsys):
        log = tmp_path / "log.jsonl"
        with log.open("w") as fh:
            for r in _ic_rows([0.05] * 30):
                fh.write(json.dumps(r) + "\n")
        rc = _cli(["--log", str(log), "--json"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        # Core fields present
        for k in ("verdict", "n", "trailing_median_20", "slope_20",
                  "gate_threshold", "slope_significance",
                  "has_never_been_active", "hint", "status"):
            assert k in data


# ─────────────────────────── never raises ───────────────────────────

class TestNeverRaises:
    def test_passing_non_list_does_not_crash(self):
        # The signature says list[dict] but real-world callers may pass garbage.
        # Should produce INSUFFICIENT_DATA, not exception.
        for bad in (None, "", 42, {"not": "a list"}, object()):
            r = gate_health_trend_report(bad)
            assert isinstance(r, dict)
            assert "verdict" in r

    def test_non_dict_rows_skipped(self):
        rows = ["bad", 42, None, ["list"]] + _ic_rows([0.0] * 30)
        r = gate_health_trend_report(rows)
        assert r["n"] == 30


# ─────────────────────────── threshold constant alignment ─────────

class TestThresholdAlignment:
    def test_gate_threshold_matches_backtest(self):
        """GATE_THRESHOLD must match backtest._GATE_SKILL_IC_TOLERANCE.

        The analyzer reports the kill-switch's behaviour; if the constants
        drift, the verdict would lie about whether the gate would fire."""
        from paper_trader.backtest import _GATE_SKILL_IC_TOLERANCE
        assert GATE_THRESHOLD == _GATE_SKILL_IC_TOLERANCE

    def test_trend_window_matches_backtest(self):
        """TREND_WINDOW must match backtest._GATE_SKILL_MIN_CYCLES.

        The analyzer reads "the gate's trailing-20 median" which is what
        backtest._should_gate_modulate_conviction actually evaluates."""
        from paper_trader.backtest import _GATE_SKILL_MIN_CYCLES
        assert TREND_WINDOW == _GATE_SKILL_MIN_CYCLES
