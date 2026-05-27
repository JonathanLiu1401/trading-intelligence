"""Agent 2 (ML+backtests) HYBRID pass — 2026-05-28 Phase 2.

Locks the new ``paper_trader.ml.scorecard`` unified quant scorecard module.

Rationale (full discussion in scorecard.py docstring): until this landed,
getting a single "current ML health" picture required ten separate CLI
invocations against the per-cycle skill ledgers (``baseline_compare``,
``calibration_reliability``, ``stop_out_audit``, ``mfe_conversion``,
``gate_pnl``, ``gate_arm_historical``, ``persona_skill``,
``persona_regime_skill``, ``conviction_calibration``,
``llm_annotation_skill``). Each printed a different verdict format. An
unattended operator had to remember which CLI answers which question and
manually fuse them. ``scorecard`` collapses that to one consolidated
read-only view with HEALTHY / DEGRADED / CRITICAL / UNKNOWN verdict.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestScorecardCollect(unittest.TestCase):
    """``scorecard.collect`` reads the latest row of every per-cycle skill
    ledger and emits one consolidated dict."""

    def test_collect_returns_a_dict_with_expected_sections(self):
        from paper_trader.ml import scorecard
        out = scorecard.collect()
        self.assertIsInstance(out, dict)
        # These are the canonical sections every consumer relies on. A
        # future refactor that drops one silently would break the
        # dashboard / CLI / Discord summary.
        for section in ("scorer", "baseline", "gate_pnl",
                        "calibrated_reliability", "stop_out", "mfe",
                        "persona", "llm_annotation",
                        "conviction_calibration"):
            self.assertIn(section, out, f"missing section: {section}")

    def test_collect_uses_latest_row_of_each_ledger(self):
        from paper_trader.ml import scorecard
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "scorer_skill_log.jsonl"
            tmp_path.write_text(
                json.dumps({"cycle": 1, "status": "ok", "train_n": 500,
                            "oos_rmse": 99.0, "gate_active": True}) + "\n"
                + json.dumps({"cycle": 2, "status": "ok", "train_n": 750,
                              "oos_rmse": 11.5, "gate_active": True}) + "\n"
            )
            row = scorecard._tail_row(tmp_path)
            self.assertEqual(row.get("cycle"), 2)
            self.assertEqual(row.get("train_n"), 750)

    def test_collect_degrades_gracefully_when_a_ledger_is_missing(self):
        # A missing ledger file must NOT crash the scorecard — it must
        # surface honestly as ``None``. Same operational discipline as
        # every ``_append_*_skill_log`` ledger.
        from paper_trader.ml import scorecard
        with tempfile.TemporaryDirectory() as tmp:
            row = scorecard._tail_row(Path(tmp) / "absent.jsonl")
            self.assertIsNone(row)

    def test_collect_skips_unparseable_rows(self):
        # A torn JSONL row (process killed mid-write) must NOT crash
        # ``_tail_row`` — same defensive contract as ``_parse_scorer_status``.
        from paper_trader.ml import scorecard
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "torn.jsonl"
            tmp_path.write_text(
                json.dumps({"cycle": 1, "status": "ok"}) + "\n"
                + "{this is torn json"
            )
            row = scorecard._tail_row(tmp_path)
            self.assertIsNotNone(row)
            self.assertEqual(row.get("cycle"), 1)


class TestScorecardVerdict(unittest.TestCase):
    """The top-level verdict rolls up per-ledger states into one
    operator-readable summary (HEALTHY / DEGRADED / CRITICAL / UNKNOWN)."""

    def test_verdict_healthy_when_all_ledgers_green(self):
        from paper_trader.ml.scorecard import _compute_verdict
        sections = {
            "scorer": {"status": "ok", "gate_active": True,
                       "oos_buy_ic": 0.15, "oos_rmse_ratio": 0.9},
            "baseline": {"verdict": "MLP_BETTER_THAN_TRIVIAL",
                         "ic_gap": 0.03},
            "gate_pnl": {"verdict": "GATE_ADDS_RETURN",
                         "equal_weight_gate_contribution_pp": 1.5},
            "calibrated_reliability": {"calibrated_dark": False,
                                       "vs_raw_bias_reduction": 2.0},
            "stop_out": {"verdict": "STOP_HELPS", "stop_dark": False},
            "mfe": {"verdict": "TP_HELPS", "tp_dark": False},
            "persona": {"n_inverted": 0, "signal_dark": False},
            "llm_annotation": {"pipeline_dark": False, "n_endorsed": 50},
            "conviction_calibration": {"verdict": "CALIBRATED",
                                       "sizing_dark": False},
        }
        v = _compute_verdict(sections)
        self.assertEqual(v["overall"], "HEALTHY")

    def test_verdict_critical_when_mlp_worse_than_trivial_and_gate_active(self):
        from paper_trader.ml.scorecard import _compute_verdict
        # The single most-decisive operational concern: gate is sizing
        # real conviction on a model the data says is worse than a one-liner.
        sections = {
            "scorer": {"status": "ok", "gate_active": True,
                       "oos_buy_ic": -0.01, "oos_rmse_ratio": 1.7},
            "baseline": {"verdict": "MLP_WORSE_THAN_TRIVIAL",
                         "ic_gap": -0.05},
            "gate_pnl": {"verdict": "GATE_SUBTRACTS_RETURN"},
            "calibrated_reliability": {"calibrated_dark": False},
            "stop_out": {"stop_dark": True},
            "mfe": {"tp_dark": True},
            "persona": {"n_inverted": 0, "signal_dark": False},
            "llm_annotation": {"pipeline_dark": True, "n_endorsed": 0},
            "conviction_calibration": {"verdict": "MISCALIBRATED"},
        }
        v = _compute_verdict(sections)
        self.assertEqual(v["overall"], "CRITICAL")
        joined = " ".join(v["reasons"])
        self.assertIn("MLP_WORSE_THAN_TRIVIAL", joined)

    def test_verdict_degraded_when_some_ledgers_red(self):
        from paper_trader.ml.scorecard import _compute_verdict
        sections = {
            "scorer": {"status": "ok", "gate_active": True,
                       "oos_buy_ic": 0.08, "oos_rmse_ratio": 1.1},
            "baseline": {"verdict": "MLP_NO_BETTER_THAN_TRIVIAL",
                         "ic_gap": -0.01},
            "gate_pnl": {"verdict": "GATE_RETURN_NEUTRAL"},
            "calibrated_reliability": {"calibrated_dark": False},
            "stop_out": {"stop_dark": True},
            "mfe": {"tp_dark": True},
            "persona": {"n_inverted": 1, "signal_dark": False},
            "llm_annotation": {"pipeline_dark": True, "n_endorsed": 0},
            "conviction_calibration": {"verdict": "MISCALIBRATED"},
        }
        v = _compute_verdict(sections)
        # Not all-green, not the worst case (gate is at noise, not actively
        # subtracting). Operator should triage but not page on this.
        self.assertEqual(v["overall"], "DEGRADED")

    def test_verdict_unknown_when_scorer_ledger_missing(self):
        from paper_trader.ml.scorecard import _compute_verdict
        # If the scorer ledger is missing entirely (fresh install, or
        # a wiped data dir), we can't honestly verdict.
        sections = {"scorer": None, "baseline": None, "gate_pnl": None,
                    "calibrated_reliability": None, "stop_out": None,
                    "mfe": None, "persona": None, "llm_annotation": None,
                    "conviction_calibration": None}
        v = _compute_verdict(sections)
        self.assertEqual(v["overall"], "UNKNOWN")


class TestScorecardCLI(unittest.TestCase):
    """The CLI surface emits valid JSON when ``--json`` is passed AND a
    human-readable summary table otherwise. Locks both contracts."""

    def test_cli_json_emits_valid_payload(self):
        from paper_trader.ml.scorecard import main
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["--json"])
        out = buf.getvalue()
        payload = json.loads(out)
        self.assertIn("overall", payload)
        self.assertIn("sections", payload)
        self.assertIn("reasons", payload)
        # Exit code 0 for HEALTHY/DEGRADED/UNKNOWN; 1 for CRITICAL — a
        # shell consumer can gate on $?. Both are acceptable here; just
        # check the contract isn't broken.
        self.assertIn(rc, (0, 1))

    def test_cli_text_emits_a_summary_table(self):
        from paper_trader.ml.scorecard import main
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main([])
        out = buf.getvalue()
        # The text output must contain the scorecard header AND the overall
        # verdict at minimum.
        self.assertIn("scorecard", out.lower())
        self.assertTrue(any(v in out for v in
                            ("HEALTHY", "DEGRADED", "CRITICAL", "UNKNOWN")))


class TestScorecardOperatorSummary(unittest.TestCase):
    """``operator_summary`` returns a single line a Discord / Slack post
    can use verbatim."""

    def test_operator_summary_is_one_line(self):
        from paper_trader.ml.scorecard import operator_summary
        v = {"overall": "DEGRADED", "reasons": ["MLP at noise"],
             "sections": {}}
        s = operator_summary(v)
        self.assertNotIn("\n", s.strip("\n"))
        self.assertIn("DEGRADED", s)


if __name__ == "__main__":
    unittest.main()
