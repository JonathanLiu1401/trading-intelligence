"""Tests for paper_trader.ml.scorer_health.

Pin the verdict-precedence ladder and exit-code contract that an operator
(or cron) branches on. Each verdict represents a distinct operational state,
so a regression that silently re-derives one of them as another would
change the meaning of `if ! python3 -m paper_trader.ml.scorer_health`
without any other diagnostic flagging the drift.
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml import scorer_health
from paper_trader.ml.scorer_health import (
    GATE_THRESHOLD,
    MIN_OUTCOMES_FOR_VERDICT,
    _EXIT_NONZERO,
    _derive_verdict,
    report,
)


# ─────────────────────── _derive_verdict precedence ───────────────────────

class TestDeriveVerdictPrecedence:
    """The verdict ladder is the public contract — assert each rung."""

    def test_untrained_overrides_everything(self):
        # Even if downstream diagnostics happen to look "healthy", a missing
        # pickle means the gate is DARK and the operator must know.
        verdict, hint = _derive_verdict(
            scorer={"trained": False, "n_train": 0, "gate_active": False,
                    "error": None},
            gate_real={"verdict": "GATE_EFFECTIVE", "n_acted": 1000,
                       "hint": "", "status": "ok"},
            gate_calib={"verdict": "WELL_CALIBRATED", "n": 1000,
                        "hint": "", "status": "ok"},
        )
        assert verdict == "UNTRAINED"
        assert "gate is dark" in hint.lower() or "untrained" in hint.lower()

    def test_untrained_with_load_error_surfaces_error(self):
        verdict, hint = _derive_verdict(
            scorer={"trained": False, "n_train": 0, "gate_active": False,
                    "error": "EOFError: pickle ran out of input"},
            gate_real={"verdict": "GATE_EFFECTIVE", "n_acted": 1000,
                       "hint": "", "status": "ok"},
            gate_calib={"verdict": "WELL_CALIBRATED", "n": 1000,
                        "hint": "", "status": "ok"},
        )
        assert verdict == "UNTRAINED"
        assert "EOFError" in hint

    def test_gate_inactive_when_subthreshold(self):
        # n_train below GATE_THRESHOLD ⇒ gate doesn't engage, so any
        # downstream "noise / harmful" verdict is academic.
        verdict, hint = _derive_verdict(
            scorer={"trained": True, "n_train": GATE_THRESHOLD - 1,
                    "gate_active": False, "error": None},
            gate_real={"verdict": "GATE_HARMFUL", "n_acted": 1000,
                       "hint": "", "status": "ok"},
            gate_calib={"verdict": "MISCALIBRATED", "n": 1000,
                        "hint": "", "status": "ok"},
        )
        assert verdict == "GATE_INACTIVE"
        assert str(GATE_THRESHOLD) in hint

    def test_insufficient_data_when_n_below_min(self):
        verdict, hint = _derive_verdict(
            scorer={"trained": True, "n_train": GATE_THRESHOLD + 100,
                    "gate_active": True, "error": None},
            gate_real={"verdict": "INSUFFICIENT_DATA",
                       "n_acted": MIN_OUTCOMES_FOR_VERDICT - 1,
                       "hint": "", "status": "insufficient_data"},
            gate_calib={"verdict": "INSUFFICIENT_DATA",
                        "n": MIN_OUTCOMES_FOR_VERDICT - 1,
                        "hint": "", "status": "insufficient_data"},
        )
        assert verdict == "INSUFFICIENT_DATA"

    def test_gate_harmful_when_realized_inverted(self):
        verdict, hint = _derive_verdict(
            scorer={"trained": True, "n_train": GATE_THRESHOLD + 100,
                    "gate_active": True, "error": None},
            gate_real={"verdict": "GATE_HARMFUL", "n_acted": 1000,
                       "hint": "tail−head=-3.5pp", "status": "ok"},
            gate_calib={"verdict": "WELL_CALIBRATED", "n": 1000,
                        "hint": "", "status": "ok"},
        )
        assert verdict == "GATE_HARMFUL"

    def test_noise_when_ineffective_and_active(self):
        verdict, hint = _derive_verdict(
            scorer={"trained": True, "n_train": GATE_THRESHOLD + 100,
                    "gate_active": True, "error": None},
            gate_real={"verdict": "GATE_INEFFECTIVE", "n_acted": 1000,
                       "hint": "", "status": "ok"},
            gate_calib={"verdict": "MISCALIBRATED", "n": 1000,
                        "hint": "", "status": "ok"},
        )
        assert verdict == "NOISE_GATE_ACTIVE"
        assert "no edge" in hint.lower()

    def test_noise_when_only_realized_ineffective(self):
        verdict, _ = _derive_verdict(
            scorer={"trained": True, "n_train": GATE_THRESHOLD + 100,
                    "gate_active": True, "error": None},
            gate_real={"verdict": "GATE_INEFFECTIVE", "n_acted": 1000,
                       "hint": "", "status": "ok"},
            gate_calib={"verdict": "WELL_CALIBRATED", "n": 1000,
                        "hint": "", "status": "ok"},
        )
        # Realized GATE_INEFFECTIVE alone is enough to demote — magnitude
        # tracking doesn't compensate for "no realized edge".
        assert verdict == "NOISE_GATE_ACTIVE"

    def test_noise_when_only_calibration_miscalibrated(self):
        verdict, _ = _derive_verdict(
            scorer={"trained": True, "n_train": GATE_THRESHOLD + 100,
                    "gate_active": True, "error": None},
            gate_real={"verdict": "GATE_EFFECTIVE", "n_acted": 1000,
                       "hint": "", "status": "ok"},
            gate_calib={"verdict": "MISCALIBRATED", "n": 1000,
                        "hint": "", "status": "ok"},
        )
        # Even when arm allocation worked, a MISCALIBRATED magnitude
        # signal means we're sizing on noise.
        assert verdict == "NOISE_GATE_ACTIVE"

    def test_directional_but_biased_passes_through(self):
        # Rank skill works (gate effective) but the predicted % is biased
        # — operator should size on signal but discount the %.
        verdict, hint = _derive_verdict(
            scorer={"trained": True, "n_train": GATE_THRESHOLD + 100,
                    "gate_active": True, "error": None},
            gate_real={"verdict": "GATE_EFFECTIVE", "n_acted": 1000,
                       "hint": "", "status": "ok"},
            gate_calib={"verdict": "DIRECTIONAL_BUT_BIASED", "n": 1000,
                        "hint": "decile error 12pp", "status": "ok"},
        )
        assert verdict == "DIRECTIONAL_BUT_BIASED"

    def test_healthy_only_when_everything_positive(self):
        verdict, _ = _derive_verdict(
            scorer={"trained": True, "n_train": GATE_THRESHOLD + 100,
                    "gate_active": True, "error": None},
            gate_real={"verdict": "GATE_EFFECTIVE", "n_acted": 1000,
                       "hint": "", "status": "ok"},
            gate_calib={"verdict": "WELL_CALIBRATED", "n": 1000,
                        "hint": "", "status": "ok"},
        )
        assert verdict == "HEALTHY"

    def test_unrecognised_combo_degrades_to_noise(self):
        # If a future ``gate_realized`` adds a new verdict literal we
        # don't enumerate, we MUST NOT claim HEALTHY by default.
        verdict, hint = _derive_verdict(
            scorer={"trained": True, "n_train": GATE_THRESHOLD + 100,
                    "gate_active": True, "error": None},
            gate_real={"verdict": "FUTURE_NEW_VERDICT", "n_acted": 1000,
                       "hint": "", "status": "ok"},
            gate_calib={"verdict": "WELL_CALIBRATED", "n": 1000,
                        "hint": "", "status": "ok"},
        )
        assert verdict == "NOISE_GATE_ACTIVE"
        assert "unrecognised" in hint.lower()


# ─────────────────────── report() integration ───────────────────────

class TestReportIntegration:
    """End-to-end: scorer pickle (mocked) + outcomes file (synthetic JSONL)
    flowing through report() and producing a structured dict.
    """

    def _write_outcomes(self, tmp_path, rows):
        path = tmp_path / "outcomes.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return path

    def test_report_untrained_when_no_pickle(self, tmp_path, monkeypatch):
        # SCORER_PATH redirected to a non-existent file ⇒ DecisionScorer
        # never loads ⇒ trained=False ⇒ verdict=UNTRAINED.
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "no_such.pkl")
        path = self._write_outcomes(tmp_path, [])
        rep = report(path)
        assert rep["verdict"] == "UNTRAINED"
        assert rep["scorer"]["trained"] is False
        assert rep["gate_threshold"] == GATE_THRESHOLD

    def test_report_shape_is_json_safe(self, tmp_path, monkeypatch):
        # Every value must be JSON-serialisable so the --json CLI flag works.
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "no_such.pkl")
        path = self._write_outcomes(tmp_path, [])
        rep = report(path)
        # round-trip — raises if any value is non-serialisable
        s = json.dumps(rep, default=str)
        loaded = json.loads(s)
        assert loaded["verdict"] == "UNTRAINED"
        assert "scorer" in loaded
        assert "gate_realized" in loaded
        assert "gate_calibration" in loaded

    def test_report_never_raises_on_corrupt_outcomes(self, tmp_path, monkeypatch):
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "no_such.pkl")
        # Half-valid, half-corrupt JSONL — must NOT raise.
        path = tmp_path / "outcomes.jsonl"
        path.write_text(
            json.dumps({"gate_scorer_pred": 5.0, "forward_return_5d": 1.0,
                        "action": "BUY"}) + "\n"
            + "not json at all\n"
            + json.dumps({"gate_scorer_pred": -3.0,
                          "forward_return_5d": -0.5, "action": "BUY"}) + "\n"
        )
        rep = report(path)
        # The corrupt line is skipped; everything still works.
        assert "verdict" in rep
        assert rep["gate_calibration"].get("error") is None

    def test_report_missing_outcomes_file(self, tmp_path, monkeypatch):
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "no_such.pkl")
        rep = report(tmp_path / "definitely_not_here.jsonl")
        # The scorer side will say UNTRAINED — but ALSO the outcomes side
        # must degrade gracefully, not raise.
        assert rep["verdict"] in {"UNTRAINED", "INSUFFICIENT_DATA"}
        assert rep["gate_calibration"]["n"] == 0


# ─────────────────────── _safe_gate_calibration unit ───────────────────────

class TestSafeGateCalibration:
    """The gate calibration helper has its own contract: filter null
    gate_scorer_pred, flip SELL targets, drop unparseable rows."""

    def test_filters_null_gate_predictions(self, tmp_path):
        # Records where gate_scorer_pred is None (untrained / sub-gate
        # cycle, or SELL) must be filtered OUT of the calibration set —
        # otherwise the calibration would conflate "the gate said
        # nothing" with "the gate said 0".
        path = tmp_path / "outcomes.jsonl"
        path.write_text(
            json.dumps({"gate_scorer_pred": None,
                        "forward_return_5d": 5.0, "action": "BUY"}) + "\n"
            + json.dumps({"gate_scorer_pred": 5.0,
                          "forward_return_5d": 5.0, "action": "BUY"}) + "\n"
        )
        rep = scorer_health._safe_gate_calibration(path)
        # Only 1 record made it through; that's below MIN_PAIRS, so the
        # verdict will be INSUFFICIENT_DATA, but n is the PAIR count.
        assert rep["n"] == 1

    def test_sell_target_sign_is_flipped(self, tmp_path):
        # A SELL with gate_pred=+5 and forward=-5 (correct call) should
        # land in the pair as (+5, +5) — sign flip aligning with
        # _oos_rank_metrics / evaluate_scorer_oos.
        #
        # We exercise this indirectly via the calibration spearman: a
        # batch of "correctly-flipped" SELLs (pred high, realized low)
        # should produce a NON-negative correlation if the flip is on.
        path = tmp_path / "outcomes.jsonl"
        rows = []
        for i in range(40):
            # SELL where realized=-(pred): correct call sign-flipped =>
            # pair is (pred, +pred), perfectly positively correlated.
            pred = (i - 20) * 0.5
            rows.append({"gate_scorer_pred": pred,
                         "forward_return_5d": -pred,
                         "action": "SELL"})
        # gate_scorer_pred is never emitted on SELL by the live path,
        # but a test-injected SELL with the field exercises the flip
        # branch — gives us coverage that the flip is on the right side.
        for r in rows:
            path.open("a").write(json.dumps(r) + "\n")
        rep = scorer_health._safe_gate_calibration(path)
        assert rep["n"] == 40
        # If the flip is on, the pairs are (pred, +pred) ⇒ spearman ≈ 1.
        # If the flip is off, pairs would be (pred, -pred) ⇒ spearman ≈ -1.
        assert rep["spearman"] is not None
        assert rep["spearman"] > 0.5, (
            f"sign-flip appears inverted: spearman={rep['spearman']} "
            f"(expected ≈ +1 when SELL targets are flipped)"
        )

    def test_drops_unparseable_pred_or_target(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        rows = [
            {"gate_scorer_pred": "not a number",
             "forward_return_5d": 5.0, "action": "BUY"},
            {"gate_scorer_pred": 5.0,
             "forward_return_5d": "garbage", "action": "BUY"},
            {"gate_scorer_pred": 5.0,
             "forward_return_5d": 5.0, "action": "BUY"},  # the only valid one
        ]
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        rep = scorer_health._safe_gate_calibration(path)
        assert rep["n"] == 1

    def test_missing_outcomes_file_returns_insufficient(self, tmp_path):
        rep = scorer_health._safe_gate_calibration(tmp_path / "nope.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0
        assert rep["error"] is None  # missing file is NOT an error


# ─────────────────────── exit codes ───────────────────────

class TestExitCodes:
    """The shell branch contract — cron / loop wrappers gate on $? so
    actionable verdicts must exit 2 and informational ones must exit 0."""

    def test_exit_nonzero_set_pins_actionable(self):
        # GATE_HARMFUL / NOISE_GATE_ACTIVE / UNTRAINED are actionable
        # — an operator must know.
        assert "GATE_HARMFUL" in _EXIT_NONZERO
        assert "NOISE_GATE_ACTIVE" in _EXIT_NONZERO
        assert "UNTRAINED" in _EXIT_NONZERO

    def test_informational_verdicts_not_in_exit_nonzero(self):
        # GATE_INACTIVE and INSUFFICIENT_DATA are informational — a cron
        # should NOT page on them. Pin so a future tightening of the
        # contract doesn't accidentally start paging.
        assert "GATE_INACTIVE" not in _EXIT_NONZERO
        assert "INSUFFICIENT_DATA" not in _EXIT_NONZERO
        assert "HEALTHY" not in _EXIT_NONZERO
        assert "DIRECTIONAL_BUT_BIASED" not in _EXIT_NONZERO

    def test_cli_exits_nonzero_on_untrained(self, tmp_path, monkeypatch, capsys):
        # End-to-end exit-code check via the CLI. No pickle ⇒ UNTRAINED ⇒
        # exit code 2.
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "no_such.pkl")
        monkeypatch.setattr(scorer_health, "OUTCOMES_PATH",
                            tmp_path / "outcomes.jsonl")
        rc = scorer_health._cli([])
        out = capsys.readouterr().out
        assert rc == 2
        assert "VERDICT: UNTRAINED" in out

    def test_cli_json_mode_emits_parseable_json(
            self, tmp_path, monkeypatch, capsys):
        import paper_trader.ml.decision_scorer as ds
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "no_such.pkl")
        monkeypatch.setattr(scorer_health, "OUTCOMES_PATH",
                            tmp_path / "outcomes.jsonl")
        rc = scorer_health._cli(["--json"])
        out = capsys.readouterr().out
        # Must round-trip as JSON
        rep = json.loads(out)
        assert rep["verdict"] == "UNTRAINED"
        assert rc == 2
