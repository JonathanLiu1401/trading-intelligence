"""Tests for paper_trader.ml.scorer_buy_sell_skill.

Pins the verdict ladder against synthetic IC time-series so each
verdict is exercised at threshold ± one step, and the analyzer's
PAIRED row-filtering (BUY/SELL must both be parseable per row) is
locked against silent regression.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_trader.ml.scorer_buy_sell_skill import (
    MIN_CYCLES_FOR_TREND,
    MISALIGN_TOL,
    SKILL_TOL,
    TREND_WINDOW,
    LONG_WINDOW,
    _cli,
    analyze,
    buy_sell_skill_report,
)


# ─────────────────────────── helpers ───────────────────────────

def _rows(buy_ics, sell_ics):
    """Build skill-log-style rows with paired BUY/SELL ICs."""
    assert len(buy_ics) == len(sell_ics), (
        "test helper requires equal-length BUY/SELL sequences")
    return [
        {"oos_buy_ic": b, "oos_sell_ic": s}
        for b, s in zip(buy_ics, sell_ics)
    ]


# ─────────────────────────── INSUFFICIENT_DATA ───────────────────────────

class TestInsufficientData:
    def test_empty_rows(self):
        r = buy_sell_skill_report([])
        assert r["verdict"] == "INSUFFICIENT_DATA"
        assert r["n"] == 0
        assert r["asymmetry_20"] is None
        assert r["gate_misaligned"] is False

    def test_below_min_cycles(self):
        r = buy_sell_skill_report(
            _rows([0.0] * (MIN_CYCLES_FOR_TREND - 1),
                  [0.0] * (MIN_CYCLES_FOR_TREND - 1)))
        assert r["verdict"] == "INSUFFICIENT_DATA"
        assert r["n"] == MIN_CYCLES_FOR_TREND - 1

    def test_min_cycles_boundary(self):
        """Exactly MIN_CYCLES_FOR_TREND rows → NOT INSUFFICIENT."""
        r = buy_sell_skill_report(
            _rows([0.0] * MIN_CYCLES_FOR_TREND,
                  [0.0] * MIN_CYCLES_FOR_TREND))
        assert r["verdict"] != "INSUFFICIENT_DATA"
        assert r["n"] == MIN_CYCLES_FOR_TREND

    def test_paired_filter_drops_buy_only_rows(self):
        """A row with oos_buy_ic but no oos_sell_ic is DROPPED — the
        paired discipline is essential, otherwise BUY/SELL medians would
        be computed on misaligned cycle samples and the asymmetry
        comparison silently lies."""
        rows = [{"oos_buy_ic": 0.01, "oos_sell_ic": 0.01}] * 5
        # Add 30 BUY-only rows (no SELL field) — must NOT count.
        rows += [{"oos_buy_ic": 0.05}] * 30
        r = buy_sell_skill_report(rows)
        # Only 5 valid paired rows; below MIN_CYCLES_FOR_TREND.
        assert r["n"] == 5
        assert r["verdict"] == "INSUFFICIENT_DATA"

    def test_paired_filter_drops_sell_only_rows(self):
        rows = [{"oos_buy_ic": 0.01, "oos_sell_ic": 0.01}] * 5
        rows += [{"oos_sell_ic": 0.05}] * 30
        r = buy_sell_skill_report(rows)
        assert r["n"] == 5
        assert r["verdict"] == "INSUFFICIENT_DATA"

    def test_invalid_ic_values_filtered(self):
        """None / NaN / strings / inf in either IC drop the WHOLE row."""
        rows = (_rows([0.0] * 5, [0.0] * 5)
                + [{"oos_buy_ic": None, "oos_sell_ic": 0.05}]
                + [{"oos_buy_ic": "bad", "oos_sell_ic": 0.05}]
                + [{"oos_buy_ic": float("nan"), "oos_sell_ic": 0.05}]
                + [{"oos_buy_ic": float("inf"), "oos_sell_ic": 0.05}]
                + [{"oos_buy_ic": 0.05, "oos_sell_ic": None}]
                + [{"oos_buy_ic": 0.05, "oos_sell_ic": float("nan")}]
                + _rows([0.0] * 10, [0.0] * 10))
        r = buy_sell_skill_report(rows)
        # 5 + 10 = 15 paired rows survive, below the minimum.
        assert r["n"] == 15
        assert r["verdict"] == "INSUFFICIENT_DATA"

    def test_bool_ic_values_filtered(self):
        """A literal True/False stored in oos_buy_ic is NOT 1.0/0.0 —
        the `_maybe_float` helper rejects bools so they don't silently
        contaminate the IC distribution."""
        rows = (_rows([0.0] * 5, [0.0] * 5)
                + [{"oos_buy_ic": True, "oos_sell_ic": 0.01}]
                + [{"oos_buy_ic": 0.01, "oos_sell_ic": False}])
        r = buy_sell_skill_report(rows)
        assert r["n"] == 5

    def test_non_dict_rows_filtered(self):
        rows = [
            {"oos_buy_ic": 0.0, "oos_sell_ic": 0.0},
            "not a dict",
            None,
            42,
            ["not", "a", "dict"],
        ]
        r = buy_sell_skill_report(rows)
        assert r["n"] == 1


# ─────────────────────────── NEITHER_SKILLED ───────────────────────────

class TestNeitherSkilled:
    def test_both_at_zero(self):
        r = buy_sell_skill_report(_rows([0.0] * 30, [0.0] * 30))
        assert r["verdict"] == "NEITHER_SKILLED"
        assert r["buy_trailing_median_20"] == 0.0
        assert r["sell_trailing_median_20"] == 0.0
        assert r["asymmetry_20"] == 0.0
        assert r["gate_misaligned"] is False

    def test_both_just_below_skill_tol(self):
        """SKILL_TOL minus 0.001 — neither passes the absolute threshold."""
        below = SKILL_TOL - 0.001
        r = buy_sell_skill_report(_rows([below] * 30, [below] * 30))
        assert r["verdict"] == "NEITHER_SKILLED"

    def test_neither_skilled_with_misalignment(self):
        """SELL beats BUY by > MISALIGN_TOL but both still below
        SKILL_TOL — the gate_misaligned flag must fire INDEPENDENTLY
        of the absolute skill verdict (the analyzer's two-axis design).
        This catches the early-warning case the AGENTS.md pass #5
        finding flagged: trailing-20 buy=-0.010, sell=+0.010 (asymmetry
        +0.020) — gate misaligned even with both at near-zero."""
        r = buy_sell_skill_report(
            _rows([-0.01] * 30,
                  [-0.01 + MISALIGN_TOL + 0.005] * 30))
        assert r["verdict"] == "NEITHER_SKILLED"
        assert r["gate_misaligned"] is True


# ─────────────────────────── BUY_SKILLED_ONLY ───────────────────────────

class TestBuySkilledOnly:
    def test_buy_above_tol_sell_below(self):
        above = SKILL_TOL + 0.01
        below = SKILL_TOL - 0.01
        r = buy_sell_skill_report(_rows([above] * 30, [below] * 30))
        assert r["verdict"] == "BUY_SKILLED_ONLY"
        # gate_misaligned is False because asymmetry = sell - buy = -0.02 < 0
        # (BUY beats SELL — gate is aligned).
        assert r["gate_misaligned"] is False

    def test_buy_at_tol_boundary(self):
        """Exactly SKILL_TOL → BUY_SKILLED_ONLY (>= comparison)."""
        r = buy_sell_skill_report(
            _rows([SKILL_TOL] * 30,
                  [SKILL_TOL - 0.01] * 30))
        assert r["verdict"] == "BUY_SKILLED_ONLY"


# ─────────────────────────── SELL_SKILLED_ONLY ───────────────────────────

class TestSellSkilledOnly:
    def test_sell_above_tol_buy_below(self):
        # Pick a gap > MISALIGN_TOL so misalignment is also flagged.
        # buy=0.0, sell=0.05 → asymmetry=0.05 > MISALIGN_TOL=0.02.
        r = buy_sell_skill_report(_rows([0.0] * 30, [0.05] * 30))
        assert r["verdict"] == "SELL_SKILLED_ONLY"
        # gate_misaligned fires INDEPENDENTLY of the verdict — orthogonal
        # axes by design. Here the gap is well above MISALIGN_TOL.
        assert r["gate_misaligned"] is True

    def test_sell_skilled_only_can_have_small_gap_no_misalign(self):
        """SELL_SKILLED_ONLY does NOT automatically imply
        gate_misaligned: a small gap (e.g. buy=0.02, sell=0.04, gap=0.02
        exactly at MISALIGN_TOL boundary, strict > comparison) clears
        the skill verdict but not the misalignment flag. Pins the
        orthogonal-axes design: SELL_SKILLED_ONLY says "buy at noise",
        gate_misaligned says "sell significantly beats buy"."""
        # buy=SKILL_TOL-0.01=0.02, sell=SKILL_TOL+0.01=0.04, gap=0.02
        # (exactly MISALIGN_TOL — strict > so not flagged).
        r = buy_sell_skill_report(
            _rows([SKILL_TOL - 0.01] * 30,
                  [SKILL_TOL + 0.01] * 30))
        assert r["verdict"] == "SELL_SKILLED_ONLY"
        assert r["gate_misaligned"] is False  # gap exactly at tol, not >

    def test_sell_at_tol_boundary(self):
        r = buy_sell_skill_report(
            _rows([SKILL_TOL - 0.01] * 30,
                  [SKILL_TOL] * 30))
        assert r["verdict"] == "SELL_SKILLED_ONLY"


# ─────────────────────────── BOTH_SKILLED ───────────────────────────

class TestBothSkilled:
    def test_both_above_tol(self):
        above = SKILL_TOL + 0.02
        r = buy_sell_skill_report(_rows([above] * 30, [above] * 30))
        assert r["verdict"] == "BOTH_SKILLED"

    def test_both_at_boundary(self):
        r = buy_sell_skill_report(
            _rows([SKILL_TOL] * 30, [SKILL_TOL] * 30))
        assert r["verdict"] == "BOTH_SKILLED"


# ─────────────────────────── asymmetry ───────────────────────────

class TestAsymmetry:
    def test_zero_asymmetry_when_identical(self):
        r = buy_sell_skill_report(_rows([0.05] * 30, [0.05] * 30))
        assert r["asymmetry_20"] == 0.0

    def test_positive_asymmetry_when_sell_dominates(self):
        r = buy_sell_skill_report(_rows([0.01] * 30, [0.05] * 30))
        # asymmetry = 0.05 - 0.01 = 0.04
        assert r["asymmetry_20"] == pytest.approx(0.04, abs=1e-9)

    def test_negative_asymmetry_when_buy_dominates(self):
        r = buy_sell_skill_report(_rows([0.05] * 30, [0.01] * 30))
        assert r["asymmetry_20"] == pytest.approx(-0.04, abs=1e-9)

    def test_misalign_at_exact_tol_not_flagged(self):
        """Strict > comparison — exactly MISALIGN_TOL does NOT trigger.
        Defensive: a boundary tweak should require a deliberate test
        update, not a silent toggle."""
        r = buy_sell_skill_report(
            _rows([0.0] * 30, [MISALIGN_TOL] * 30))
        assert r["asymmetry_20"] == pytest.approx(MISALIGN_TOL, abs=1e-9)
        assert r["gate_misaligned"] is False

    def test_misalign_just_above_tol_flagged(self):
        r = buy_sell_skill_report(
            _rows([0.0] * 30, [MISALIGN_TOL + 0.001] * 30))
        assert r["gate_misaligned"] is True


# ─────────────────────────── trailing window ───────────────────────────

class TestTrailingWindow:
    def test_only_last_20_used_for_t20(self):
        """The first 20 rows have BUY at +0.10, last 20 at +0.01.
        trailing_median_20 must reflect the LAST 20 only, not the
        average of the whole series — a regression that would surface
        as a silently-elevated median right after a regime shift."""
        rows = _rows([0.10] * 20 + [0.01] * 20,
                     [0.10] * 20 + [0.01] * 20)
        r = buy_sell_skill_report(rows)
        assert r["buy_trailing_median_20"] == 0.01
        assert r["sell_trailing_median_20"] == 0.01

    def test_long_window_returns_none_when_fewer_than_50(self):
        r = buy_sell_skill_report(_rows([0.0] * 30, [0.0] * 30))
        assert r["buy_trailing_median_50"] is None
        assert r["sell_trailing_median_50"] is None

    def test_long_window_populated_at_50_rows(self):
        r = buy_sell_skill_report(_rows([0.02] * 50, [0.02] * 50))
        assert r["buy_trailing_median_50"] == 0.02
        assert r["sell_trailing_median_50"] == 0.02


# ─────────────────────────── analyze() end-to-end ───────────────────────────

class TestAnalyzeFile:
    def test_missing_log(self, tmp_path):
        r = analyze(tmp_path / "does_not_exist.jsonl")
        assert r["status"] == "error"
        assert "missing" in r["error"]
        # Still has the full report skeleton (INSUFFICIENT_DATA verdict).
        assert r["verdict"] == "INSUFFICIENT_DATA"

    def test_real_jsonl_round_trip(self, tmp_path):
        log = tmp_path / "skill.jsonl"
        rows = _rows([0.05] * 30, [0.01] * 30)
        with log.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        out = analyze(log)
        assert out["status"] == "ok"
        assert out["verdict"] == "BUY_SKILLED_ONLY"
        assert out["n"] == 30

    def test_corrupt_lines_skipped(self, tmp_path):
        log = tmp_path / "skill.jsonl"
        with log.open("w") as fh:
            for r in _rows([0.05] * 25, [0.01] * 25):
                fh.write(json.dumps(r) + "\n")
            fh.write("not json\n")
            fh.write("\n")  # empty line
            fh.write("{partial\n")
            for r in _rows([0.05] * 5, [0.01] * 5):
                fh.write(json.dumps(r) + "\n")
        out = analyze(log)
        assert out["status"] == "ok"
        assert out["n"] == 30
        assert out["verdict"] == "BUY_SKILLED_ONLY"


# ─────────────────────────── _cli() exit codes ───────────────────────────

class TestCliExitCodes:
    def _write_log(self, tmp_path, buy_ics, sell_ics):
        log = tmp_path / "skill.jsonl"
        with log.open("w") as fh:
            for b, s in zip(buy_ics, sell_ics):
                fh.write(json.dumps({"oos_buy_ic": b, "oos_sell_ic": s})
                         + "\n")
        return log

    def test_exit_zero_on_buy_skilled(self, tmp_path, capsys):
        log = self._write_log(tmp_path, [0.05] * 30, [0.01] * 30)
        rc = _cli(["--log", str(log)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "BUY_SKILLED_ONLY" in out

    def test_exit_zero_on_both_skilled(self, tmp_path, capsys):
        log = self._write_log(tmp_path, [0.05] * 30, [0.05] * 30)
        rc = _cli(["--log", str(log)])
        assert rc == 0

    def test_exit_zero_on_insufficient(self, tmp_path):
        log = self._write_log(tmp_path, [0.0] * 5, [0.0] * 5)
        rc = _cli(["--log", str(log)])
        assert rc == 0  # INSUFFICIENT_DATA is benign

    def test_exit_two_on_sell_skilled_only(self, tmp_path, capsys):
        log = self._write_log(tmp_path, [0.01] * 30, [0.05] * 30)
        rc = _cli(["--log", str(log)])
        assert rc == 2
        out = capsys.readouterr().out
        assert "SELL_SKILLED_ONLY" in out
        assert "gate_misaligned" in out

    def test_exit_two_on_neither_skilled_with_misalign(self, tmp_path):
        # Both below SKILL_TOL but SELL > BUY by > MISALIGN_TOL.
        log = self._write_log(tmp_path, [-0.01] * 30,
                              [-0.01 + MISALIGN_TOL + 0.005] * 30)
        rc = _cli(["--log", str(log)])
        assert rc == 2

    def test_exit_zero_on_neither_skilled_without_misalign(self, tmp_path):
        # Both at 0 — no misalignment.
        log = self._write_log(tmp_path, [0.0] * 30, [0.0] * 30)
        rc = _cli(["--log", str(log)])
        assert rc == 0

    def test_json_output_is_machine_readable(self, tmp_path, capsys):
        log = self._write_log(tmp_path, [0.05] * 30, [0.01] * 30)
        rc = _cli(["--log", str(log), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["verdict"] == "BUY_SKILLED_ONLY"
        assert payload["n"] == 30
        assert "asymmetry_20" in payload

    def test_missing_log_exits_zero_with_insufficient(self, tmp_path, capsys):
        rc = _cli(["--log", str(tmp_path / "missing.jsonl")])
        assert rc == 0  # INSUFFICIENT_DATA → benign exit
        out = capsys.readouterr().out
        assert "INSUFFICIENT_DATA" in out
