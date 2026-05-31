"""Tests for paper_trader.ml.kill_switch_arm_breakdown.

Covers:
  * arm decode + bucket assignment (uses gate_audit.gate_arm — boundary
    operators MUST match _ml_decide to the bit, so the per-arm verdict
    decomposition is grounded in the same arithmetic the live gate would
    have used at decision time)
  * filtering: only BUYs with gate_abstention_kind == "killswitch" AND
    finite gate_scorer_pred AND finite forward_return_5d contribute
  * INSUFFICIENT_DATA threshold (n_killswitched < MIN_TOTAL_N)
  * MIN_ARM_N guard: a tiny extreme arm cannot flip the verdict
  * per_trade_tilt_pp arithmetic = (mult - 1.0) * mean_realized exactly
  * aggregate_per_trade_tilt_pp = Σ n_i (mult_i - 1) mean_i / Σ n_i
  * all 5 verdict ladder steps at threshold boundaries
  * unit_note presence (honesty discipline — per-trade NOT portfolio P&L)
  * end-to-end JSONL round-trip + corrupt-line skipping
  * every CLI exit code path (0 benign / 2 actionable)
  * file-missing error path
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_trader.ml import kill_switch_arm_breakdown as ksab


# ────────────────────────── helpers / row builders ──────────────────────────

def _row(pred: float, realized: float, *, action: str = "BUY",
         kind: str | None = "killswitch") -> dict:
    """Build a minimal valid outcome row for the analyzer."""
    return {
        "action": action,
        "gate_abstention_kind": kind,
        "gate_scorer_pred": pred,
        "forward_return_5d": realized,
    }


def _arm_for(pred: float) -> str:
    """Map a prediction to the documented arm bucket (mirrors the analyzer)."""
    if pred < -10.0:
        return "strong_headwind"
    if pred < 0.0:
        return "mild_headwind"
    if pred > 10.0:
        return "strong_tailwind"
    if pred > 5.0:
        return "mild_tailwind"
    return "neutral"


# ────────────────────────── filter discipline tests ──────────────────────────

class TestFiltering:
    def test_drops_non_buy_rows(self):
        # SELL rows must never contribute — the gate is BUY-only.
        rows = [_row(15.0, 5.0, action="SELL")] * 300
        out = ksab.kill_switch_arm_report(rows)
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["n_killswitched"] == 0

    def test_drops_non_killswitch_kinds(self):
        # clamp / None / unknown kinds must be filtered out — this analyzer
        # is the kill-switch-only deep dive (sibling handles per-bucket
        # comparison).
        rows = ([_row(15.0, 5.0, kind="clamp")] * 100
                + [_row(15.0, 5.0, kind=None)] * 100
                + [_row(15.0, 5.0, kind="acted")] * 100
                + [_row(15.0, 5.0, kind="UNKNOWN_FUTURE_KIND")] * 100)
        out = ksab.kill_switch_arm_report(rows)
        assert out["n_killswitched"] == 0
        assert out["verdict"] == "INSUFFICIENT_DATA"

    def test_killswitch_case_insensitive(self):
        rows = [_row(15.0, 5.0, kind="KillSwitch") for _ in range(ksab.MIN_TOTAL_N)]
        out = ksab.kill_switch_arm_report(rows)
        # All rows kept (just below MIN_TOTAL_N boundary intentionally would
        # NOT clear; at MIN_TOTAL_N it does — so verdict moves past
        # INSUFFICIENT_DATA).
        assert out["n_killswitched"] == ksab.MIN_TOTAL_N

    def test_drops_missing_pred_or_realized(self):
        # _maybe_float (shared sibling helper) rejects None / bool / NaN /
        # ±Inf — but ACCEPTS numeric strings ("15.0" → 15.0) because the
        # corpus has historically tolerated stringified values. Test the
        # documented contract: 6/6 truly-invalid rows are dropped, the
        # string row passes through (its REAL field would have been
        # stringified upstream by a bug — analyzer is robust either way).
        rows = [
            {"action": "BUY", "gate_abstention_kind": "killswitch",
             "gate_scorer_pred": None, "forward_return_5d": 5.0},
            {"action": "BUY", "gate_abstention_kind": "killswitch",
             "gate_scorer_pred": 15.0, "forward_return_5d": None},
            {"action": "BUY", "gate_abstention_kind": "killswitch",
             "gate_scorer_pred": float("nan"), "forward_return_5d": 5.0},
            {"action": "BUY", "gate_abstention_kind": "killswitch",
             "gate_scorer_pred": float("inf"), "forward_return_5d": 5.0},
            {"action": "BUY", "gate_abstention_kind": "killswitch",
             "gate_scorer_pred": 15.0, "forward_return_5d": float("nan")},
            {"action": "BUY", "gate_abstention_kind": "killswitch",
             "gate_scorer_pred": True, "forward_return_5d": 5.0},  # bool
        ]
        out = ksab.kill_switch_arm_report(rows)
        assert out["n_killswitched"] == 0

    def test_unparseable_string_pred_dropped(self):
        rows = [
            {"action": "BUY", "gate_abstention_kind": "killswitch",
             "gate_scorer_pred": "not-a-number", "forward_return_5d": 5.0},
            {"action": "BUY", "gate_abstention_kind": "killswitch",
             "gate_scorer_pred": 15.0, "forward_return_5d": "garbage"},
            {"action": "BUY", "gate_abstention_kind": "killswitch",
             "gate_scorer_pred": [1, 2], "forward_return_5d": 5.0},
        ]
        out = ksab.kill_switch_arm_report(rows)
        assert out["n_killswitched"] == 0

    def test_skips_non_dict_rows(self):
        rows = [None, 42, "garbage", [1, 2, 3]] + [_row(15.0, 5.0)] * 5
        out = ksab.kill_switch_arm_report(rows)
        assert out["n_killswitched"] == 5

    def test_skips_input_when_iter_raises(self):
        class _RaisingIter:
            def __iter__(self):
                raise RuntimeError("simulated")
        # ``list(...)`` consumption raises; the analyzer swallows it.
        out = ksab.kill_switch_arm_report(_RaisingIter())  # type: ignore[arg-type]
        assert out["n_killswitched"] == 0


# ────────────────────────── arm decode + bucketing ──────────────────────────

class TestArmDecode:
    @pytest.mark.parametrize("pred,expected_arm", [
        (-25.0, "strong_headwind"),
        (-10.001, "strong_headwind"),
        (-10.0, "mild_headwind"),     # boundary: < -10.0 only
        (-5.0, "mild_headwind"),
        (-0.001, "mild_headwind"),
        (0.0, "neutral"),
        (5.0, "neutral"),
        (5.001, "mild_tailwind"),
        (10.0, "mild_tailwind"),       # boundary: > 10.0 only
        (10.001, "strong_tailwind"),
        (25.0, "strong_tailwind"),
    ])
    def test_arm_boundaries_match_ml_decide(self, pred, expected_arm):
        # Each boundary point MUST land in the documented arm so the
        # analyzer's report describes the same arm _ml_decide would have
        # fired at decision time. Test by feeding 60 rows at the boundary
        # so it clears MIN_ARM_N and shows up populated.
        rows = [_row(pred, 0.0)] * 60
        out = ksab.kill_switch_arm_report(rows)
        by = {a["arm"]: a for a in out["arms"]}
        assert by[expected_arm]["n"] == 60
        # All other arms zero.
        for arm_name, _ in ksab._ARMS:
            if arm_name == expected_arm:
                continue
            assert by[arm_name]["n"] == 0


# ────────────────────────── per-arm metric arithmetic ──────────────────────

class TestPerArmMetrics:
    def test_per_trade_tilt_formula(self):
        # strong_tailwind multiplier is 1.30, so a +10% mean realized →
        # tilt = (1.30 - 1.00) * 10.0 = +3.0pp.
        rows = [_row(15.0, 10.0) for _ in range(60)]
        out = ksab.kill_switch_arm_report(rows)
        by = {a["arm"]: a for a in out["arms"]}
        st = by["strong_tailwind"]
        assert st["mean_realized"] == 10.0
        assert st["per_trade_tilt_pp"] == pytest.approx(3.0)
        # Other arms have None for per_trade_tilt_pp when n=0.
        assert by["strong_headwind"]["per_trade_tilt_pp"] is None

    def test_aggregate_tilt_formula(self):
        # Mix two arms: 100 strong_tailwind @ +10% and 100 strong_headwind @ -10%.
        # Per-arm tilts: (1.30-1)*10 = +3.0, (0.60-1)*(-10) = +4.0
        # Aggregate: (100*3.0 + 100*4.0) / 200 = +3.5
        rows = ([_row(15.0, 10.0) for _ in range(100)]
                + [_row(-15.0, -10.0) for _ in range(100)])
        out = ksab.kill_switch_arm_report(rows)
        assert out["aggregate_per_trade_tilt_pp"] == pytest.approx(3.5)

    def test_empty_arms_aggregate_none(self):
        rows: list[dict] = []
        out = ksab.kill_switch_arm_report(rows)
        assert out["aggregate_per_trade_tilt_pp"] is None
        for a in out["arms"]:
            assert a["n"] == 0
            assert a["mean_realized"] is None
            assert a["per_trade_tilt_pp"] is None
            assert a["mean_pred"] is None
            assert a["median_realized"] is None

    def test_median_realized_correct(self):
        # Median of [-5, -1, 0, 3, 10] = 0
        # Mean = 1.4
        ps = [15.0] * 5
        rs = [-5.0, -1.0, 0.0, 3.0, 10.0]
        rows = [_row(p, r) for p, r in zip(ps, rs)]
        out = ksab.kill_switch_arm_report(rows)
        by = {a["arm"]: a for a in out["arms"]}
        st = by["strong_tailwind"]
        assert st["median_realized"] == 0.0
        assert st["mean_realized"] == pytest.approx(1.4)

    def test_arms_have_fixed_order(self):
        # Order matters for stable CLI output across Python versions.
        out = ksab.kill_switch_arm_report([])
        names = [a["arm"] for a in out["arms"]]
        assert names == ["strong_headwind", "mild_headwind", "neutral",
                         "mild_tailwind", "strong_tailwind"]

    def test_multiplier_per_arm_matches_canonical(self):
        out = ksab.kill_switch_arm_report([])
        mults = {a["arm"]: a["multiplier"] for a in out["arms"]}
        assert mults == {
            "strong_headwind": 0.60,
            "mild_headwind": 0.85,
            "neutral": 1.00,
            "mild_tailwind": 1.15,
            "strong_tailwind": 1.30,
        }


# ────────────────────────── verdict ladder tests ──────────────────────────

class TestVerdictLadder:
    def test_insufficient_data_below_min_total(self):
        # n_killswitched < MIN_TOTAL_N → INSUFFICIENT_DATA always.
        rows = [_row(15.0, 50.0) for _ in range(ksab.MIN_TOTAL_N - 1)]
        out = ksab.kill_switch_arm_report(rows)
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert "need >=" in out["hint"]

    def test_strong_tailwind_costly_at_threshold(self):
        # ≥ MIN_TOTAL_N total with strong_tailwind n>=MIN_ARM_N and mean
        # exactly at ARM_TOL_PCT → STRONG_TAILWIND_COSTLY.
        # 250 strong_tailwind @ exactly +ARM_TOL_PCT = +1.0%
        # All in same arm so MIN_ARM_N is cleared; sh has n=0 → not costly.
        rows = [_row(15.0, ksab.ARM_TOL_PCT) for _ in range(250)]
        out = ksab.kill_switch_arm_report(rows)
        assert out["n_killswitched"] == 250
        assert out["verdict"] == "STRONG_TAILWIND_COSTLY"
        # Hint mentions the arm name and threshold.
        assert "strong_tailwind" in out["hint"]
        assert "+1.30" in out["hint"] or "×1.30" in out["hint"]

    def test_strong_tailwind_below_threshold_neutral(self):
        # Just below ARM_TOL_PCT → not costly. Add neutral filler so we
        # clear MIN_TOTAL_N (strong_tailwind alone at 250 also clears).
        rows = ([_row(15.0, ksab.ARM_TOL_PCT - 0.01) for _ in range(60)]
                + [_row(0.0, 0.0) for _ in range(200)])
        out = ksab.kill_switch_arm_report(rows)
        assert out["n_killswitched"] == 260
        assert out["verdict"] == "ARMS_NEUTRAL"

    def test_strong_headwind_costly_at_threshold(self):
        rows = [_row(-15.0, -ksab.ARM_TOL_PCT) for _ in range(250)]
        out = ksab.kill_switch_arm_report(rows)
        assert out["verdict"] == "STRONG_HEADWIND_COSTLY"
        assert "strong_headwind" in out["hint"]

    def test_both_tails_costly(self):
        rows = ([_row(15.0, 5.0) for _ in range(100)]
                + [_row(-15.0, -5.0) for _ in range(100)])
        out = ksab.kill_switch_arm_report(rows)
        assert out["n_killswitched"] == 200
        assert out["verdict"] == "BOTH_TAILS_COSTLY"

    def test_min_arm_n_floor_prevents_tiny_arm_flip(self):
        # 1 row in strong_tailwind with massive realized return MUST NOT
        # flip the verdict (MIN_ARM_N guard against tail-noise verdicts).
        rows = ([_row(15.0, 100.0)]  # tiny arm with huge mean
                + [_row(0.0, 0.0) for _ in range(ksab.MIN_TOTAL_N)])
        out = ksab.kill_switch_arm_report(rows)
        by = {a["arm"]: a for a in out["arms"]}
        # Strong tailwind has n=1, mean=100.0, but should NOT trigger
        # STRONG_TAILWIND_COSTLY because n < MIN_ARM_N.
        assert by["strong_tailwind"]["n"] == 1
        assert by["strong_tailwind"]["mean_realized"] == 100.0
        assert out["verdict"] == "ARMS_NEUTRAL"

    def test_arms_neutral_when_extremes_inside_band(self):
        rows = ([_row(15.0, 0.5) for _ in range(60)]
                + [_row(-15.0, -0.5) for _ in range(60)]
                + [_row(0.0, 0.0) for _ in range(80)])
        out = ksab.kill_switch_arm_report(rows)
        assert out["verdict"] == "ARMS_NEUTRAL"


# ────────────────────────── unit_note + shape ──────────────────────────

class TestReportShape:
    def test_unit_note_always_present(self):
        # The honesty unit-note (per-trade NOT portfolio P&L) is
        # load-bearing — every report must carry it so a reader cannot
        # mistake tilt for portfolio return.
        for rows in ([], [_row(15.0, 10.0)] * 250):
            out = ksab.kill_switch_arm_report(rows)
            assert "per_trade_tilt_pp" in out["unit_note"]
            assert "NOT portfolio" in out["unit_note"]

    def test_fixed_top_level_keys(self):
        out = ksab.kill_switch_arm_report([_row(15.0, 5.0)] * 250)
        expected_keys = {
            "verdict", "n_buys", "n_killswitched", "arms",
            "aggregate_per_trade_tilt_pp", "min_total_n", "min_arm_n",
            "arm_tol_pct", "hint", "unit_note",
        }
        assert set(out.keys()) == expected_keys

    def test_per_arm_keys(self):
        out = ksab.kill_switch_arm_report([_row(15.0, 5.0)] * 60)
        for a in out["arms"]:
            assert set(a.keys()) == {
                "arm", "multiplier", "n", "mean_pred", "mean_realized",
                "median_realized", "per_trade_tilt_pp",
            }


# ────────────────────────── analyze() — file IO ──────────────────────────

class TestAnalyze:
    def test_missing_file_returns_error_status(self, tmp_path):
        missing = tmp_path / "nope.jsonl"
        out = ksab.analyze(missing)
        assert out["status"] == "error"
        assert "missing" in out["error"]
        # Even on error, the verdict shape is still present (callers can rely
        # on a stable top-level schema).
        assert out["verdict"] == "INSUFFICIENT_DATA"

    def test_jsonl_round_trip(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        rows = ([_row(15.0, 5.0) for _ in range(150)]
                + [_row(-15.0, -5.0) for _ in range(100)])
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        out = ksab.analyze(path)
        assert out["status"] == "ok"
        assert out["n_killswitched"] == 250
        assert out["verdict"] == "BOTH_TAILS_COSTLY"

    def test_jsonl_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        with path.open("w") as fh:
            for i in range(250):
                fh.write(json.dumps(_row(15.0, 5.0)) + "\n")
            fh.write("not-json\n")
            fh.write("\n")  # empty line skipped
            fh.write('{"truncated": ' + "\n")  # invalid JSON skipped
        out = ksab.analyze(path)
        assert out["status"] == "ok"
        # 250 valid rows survive — corrupt and empty lines dropped, not
        # raised.
        assert out["n_killswitched"] == 250

    def test_default_path_used_when_none_passed(self, monkeypatch, tmp_path):
        path = tmp_path / "default_outcomes.jsonl"
        with path.open("w") as fh:
            for _ in range(250):
                fh.write(json.dumps(_row(15.0, 5.0)) + "\n")
        monkeypatch.setattr(ksab, "DECISION_OUTCOMES", path)
        out = ksab.analyze(None)
        assert out["status"] == "ok"
        assert out["n_killswitched"] == 250


# ────────────────────────── CLI exit codes ──────────────────────────

class TestCLI:
    def test_cli_insufficient_data_exits_zero(self, tmp_path, capsys):
        path = tmp_path / "outcomes.jsonl"
        with path.open("w") as fh:
            for _ in range(10):
                fh.write(json.dumps(_row(15.0, 5.0)) + "\n")
        rc = ksab._cli(["--outcomes", str(path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "INSUFFICIENT_DATA" in out

    def test_cli_strong_tailwind_costly_exits_two(self, tmp_path, capsys):
        path = tmp_path / "outcomes.jsonl"
        with path.open("w") as fh:
            for _ in range(250):
                fh.write(json.dumps(_row(15.0, 5.0)) + "\n")
        rc = ksab._cli(["--outcomes", str(path)])
        out = capsys.readouterr().out
        assert rc == 2
        assert "STRONG_TAILWIND_COSTLY" in out
        # CLI table shows the multiplier and the per_trade_tilt_pp column.
        assert "1.30" in out
        assert "per_trade_tilt_pp" in out

    def test_cli_both_tails_costly_exits_two(self, tmp_path, capsys):
        path = tmp_path / "outcomes.jsonl"
        rows = ([_row(15.0, 5.0) for _ in range(100)]
                + [_row(-15.0, -5.0) for _ in range(100)])
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        rc = ksab._cli(["--outcomes", str(path)])
        out = capsys.readouterr().out
        assert rc == 2
        assert "BOTH_TAILS_COSTLY" in out

    def test_cli_arms_neutral_exits_zero(self, tmp_path, capsys):
        path = tmp_path / "outcomes.jsonl"
        with path.open("w") as fh:
            for _ in range(250):
                # Tail at +0.5pp — below ARM_TOL_PCT=1.0 → neutral.
                fh.write(json.dumps(_row(15.0, 0.5)) + "\n")
        rc = ksab._cli(["--outcomes", str(path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "ARMS_NEUTRAL" in out

    def test_cli_json_machine_readable(self, tmp_path, capsys):
        path = tmp_path / "outcomes.jsonl"
        with path.open("w") as fh:
            for _ in range(250):
                fh.write(json.dumps(_row(15.0, 5.0)) + "\n")
        rc = ksab._cli(["--outcomes", str(path), "--json"])
        out = capsys.readouterr().out
        # Output is valid JSON the operator/dashboard can parse.
        parsed = json.loads(out)
        assert parsed["verdict"] == "STRONG_TAILWIND_COSTLY"
        # Exit 2 for actionable verdict regardless of --json.
        assert rc == 2

    def test_cli_missing_file(self, tmp_path, capsys):
        # Missing file → INSUFFICIENT_DATA verdict → exit 0 (benign).
        rc = ksab._cli(["--outcomes", str(tmp_path / "nope.jsonl"), "--json"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["status"] == "error"
        assert rc == 0
