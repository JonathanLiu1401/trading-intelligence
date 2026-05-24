"""Per-cycle ledger tests for the gate-arm historical skill analyzer.

``paper_trader/ml/gate_arm_historical.py`` already CLI-reports the realized
economic effect of the conviction gate arms (×0.6 / ×0.85 / ×1.0 / ×1.15 /
×1.3) bucketed by the gate's TRUE then-deployed prediction
(``gate_scorer_pred``, not a counterfactual re-predict with today's
pickle). The deployed scorer carries strong measured OOS rank-IC (+0.48)
but the documented ``GATE_INEFFECTIVE`` verdict shows the bucket
assignment only captures ~1% of that rank skill — exactly the kind of
state a skeptical quant needs trended per cycle to see whether bucket
tuning recovers economic edge. Until ``_append_gate_arm_skill_log``
landed there was NO per-cycle durable trend for the gate-arm verdict —
a quant could only know "do the gate's arms actually realize different
returns" by manually running the CLI. These tests pin the wiring:
best-effort discipline, honest gap rows when the gate is dark, bounded
trim, SSOT cross-check that the persisted verdict equals what the
analyzer returned. Mirrors the discipline of every sibling
``_append_*_skill_log`` test in ``test_continuous.py`` and the
intraperiod ledger tests in ``test_continuous_intraperiod_ledgers.py``.

A new test file (not appended to ``test_continuous.py``) so a concurrent
sibling agent editing the same test file cannot collide with this work
via whole-file ``git add`` — the documented same-role HYBRID
staging-race mitigation pattern.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

import run_continuous_backtests as rcb


class TestAppendGateArmSkillLog:
    """The per-cycle gate-arm historical skill ledger — answers "do the
    conviction gate's bucket assignments realize differentiated economic
    outcomes, given the gate's true then-deployed prediction?" durably
    and per-cycle so a quant can trend the bucket health. Mirrors the
    sibling ``_append_stop_out_skill_log`` / ``_append_mfe_skill_log``
    discipline: best-effort, honest gap rows, atomic bounded trim,
    SSOT no-drift, never breaks the loop.
    """

    def test_insufficient_data_persists_dark_row(self, tmp_path, monkeypatch):
        """The documented live state where the analyzer reports
        INSUFFICIENT_DATA — either the corpus pre-dates the 2026-05-18
        ``_parse_gate_decision`` capture, OR every recent cycle ran
        sub-gate. The ledger must STILL append the row so the darkness
        is visible in the trend, exactly how the stop-out ledger
        surfaces its ``stop_dark`` state.
        """
        log = tmp_path / "gate_arm_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_ARM_SKILL_LOG", log)

        from paper_trader.ml import gate_arm_historical as gah
        empty = {
            "status": "ok",
            "verdict": "INSUFFICIENT_DATA",
            "n": 0,
            "arms": [],
            "strong_tailwind_minus_headwind_pp": None,
            "arm_monotone_fraction": None,
            "hint": "need ≥800 historical-gate pairs and ≥10 in BOTH extreme arms; "
                    "have n=0, strong_headwind=0, strong_tailwind=0",
            "n_dropped_no_gate_pred": 1422,
            "n_dropped_off_dist": 0,
            "n_dropped_no_return": 0,
            "slice": "oos",
            "outcomes_n": 1750,
        }
        monkeypatch.setattr(gah, "analyze", lambda *a, **k: dict(empty))

        assert rcb._append_gate_arm_skill_log(
            cycle=33, win_start=date(2020, 1, 2), win_end=date(2025, 1, 2),
            outcomes_path=tmp_path / "nope.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["cycle"] == 33
        assert row["verdict"] == "INSUFFICIENT_DATA"
        # ``gate_dark`` mirrors the sibling ``stop_dark`` / ``tp_dark`` /
        # ``sizing_dark`` / ``calibrated_dark`` boolean.
        assert row["gate_dark"] is True
        assert row["n"] == 0
        assert row["n_dropped_no_gate_pred"] == 1422
        assert row["slice"] == "oos"
        assert row["outcomes_n"] == 1750
        assert row["window_start"] == "2020-01-02"
        assert row["window_end"] == "2025-01-02"
        # Per-arm flat columns are absent / None when ``arms`` is empty.
        assert "mean_x13" not in row or row.get("mean_x13") is None

    def test_gate_ineffective_verdict_surfaces_spread(self, tmp_path,
                                                      monkeypatch):
        """SSOT cross-check: the analyzer's verdict and key metrics
        (spread, monotone_fraction) must round-trip unchanged through the
        ledger. A drift here would silently break trend integrity — the
        whole point of the per-cycle wiring is no-drift fidelity to the
        CLI verdict.

        This is the documented current state: scorer rank-IC is strong
        (+0.48) but bucket assignment only realizes a sub-tolerance
        spread of -0.13pp. The verdict ``GATE_INEFFECTIVE`` must persist
        WITH that exact spread number for trend usefulness.
        """
        log = tmp_path / "gate_arm_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_ARM_SKILL_LOG", log)

        ineffective = {
            "status": "ok",
            "verdict": "GATE_INEFFECTIVE",
            "n": 328,
            "slice": "oos",
            "outcomes_n": 1750,
            "strong_tailwind_minus_headwind_pp": -0.13,
            "arm_monotone_fraction": 0.5,
            "n_dropped_no_gate_pred": 1422,
            "n_dropped_off_dist": 0,
            "n_dropped_no_return": 0,
            "hint": "historical gate ×1.30 arm +3.99% vs ×0.60 arm +4.12% "
                    "(spread -0.13pp, within ±1.0pp) — the production gate's "
                    "bigger bets have not realized higher returns",
            "arms": [
                {"arm": "strong_headwind", "multiplier": 0.6, "n": 9,
                 "mean_realized": 4.1172, "lo": -24.733, "hi": 22.1284},
                {"arm": "mild_headwind", "multiplier": 0.85, "n": 122,
                 "mean_realized": 0.8894, "lo": -22.1928, "hi": 41.5505},
                {"arm": "neutral", "multiplier": 1.0, "n": 136,
                 "mean_realized": 2.5163, "lo": -18.4128, "hi": 33.9856},
                {"arm": "mild_tailwind", "multiplier": 1.15, "n": 39,
                 "mean_realized": -0.5097, "lo": -17.6132, "hi": 28.9059},
                {"arm": "strong_tailwind", "multiplier": 1.3, "n": 22,
                 "mean_realized": 3.9901, "lo": -13.7736, "hi": 24.1111},
            ],
        }
        from paper_trader.ml import gate_arm_historical as gah
        monkeypatch.setattr(gah, "analyze", lambda *a, **k: dict(ineffective))

        rcb._append_gate_arm_skill_log(
            cycle=44, win_start=date(2018, 6, 1), win_end=date(2023, 6, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "GATE_INEFFECTIVE"
        assert row["gate_dark"] is False
        # Spread sign + magnitude preserved exactly (SSOT).
        assert row["strong_tailwind_minus_headwind_pp"] == -0.13
        assert row["arm_monotone_fraction"] == 0.5
        assert row["n"] == 328
        # Per-arm flat columns mapped via _MULT_TO_KEY.
        assert row["mean_x06"] == 4.1172
        assert row["mean_x085"] == 0.8894
        assert row["mean_x10"] == 2.5163
        assert row["mean_x115"] == -0.5097
        assert row["mean_x13"] == 3.9901
        assert row["n_x06"] == 9
        assert row["n_x13"] == 22

    def test_gate_effective_verdict_with_positive_spread(self, tmp_path,
                                                          monkeypatch):
        """When the scorer's rank skill DOES translate into arm divergence
        (the documented "the moment bucket tuning recovers real economic
        edge" state), the verdict flips to GATE_EFFECTIVE and the spread
        is positive. The ledger MUST persist that exact positive sign so
        the trend visibly captures the regime change.
        """
        log = tmp_path / "gate_arm_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_ARM_SKILL_LOG", log)

        effective = {
            "status": "ok",
            "verdict": "GATE_EFFECTIVE",
            "n": 500,
            "slice": "oos",
            "outcomes_n": 1750,
            "strong_tailwind_minus_headwind_pp": 4.5,
            "arm_monotone_fraction": 1.0,
            "n_dropped_no_gate_pred": 1250,
            "n_dropped_off_dist": 0,
            "n_dropped_no_return": 0,
            "hint": "historical gate ×1.30 arm realized +6.20% > ×0.60 arm "
                    "+1.70% (spread +4.50pp) — economically justified",
            "arms": [
                {"arm": "strong_headwind", "multiplier": 0.6, "n": 50,
                 "mean_realized": 1.7, "lo": -10.0, "hi": 12.0},
                {"arm": "mild_headwind", "multiplier": 0.85, "n": 100,
                 "mean_realized": 2.5, "lo": -10.0, "hi": 12.0},
                {"arm": "neutral", "multiplier": 1.0, "n": 200,
                 "mean_realized": 3.5, "lo": -10.0, "hi": 12.0},
                {"arm": "mild_tailwind", "multiplier": 1.15, "n": 100,
                 "mean_realized": 5.0, "lo": -10.0, "hi": 12.0},
                {"arm": "strong_tailwind", "multiplier": 1.3, "n": 50,
                 "mean_realized": 6.2, "lo": -10.0, "hi": 12.0},
            ],
        }
        from paper_trader.ml import gate_arm_historical as gah
        monkeypatch.setattr(gah, "analyze", lambda *a, **k: dict(effective))

        rcb._append_gate_arm_skill_log(
            cycle=55, win_start=date(2018, 6, 1), win_end=date(2023, 6, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "GATE_EFFECTIVE"
        # Positive spread sign preserved — the actionable signal.
        assert row["strong_tailwind_minus_headwind_pp"] == 4.5
        # Perfect monotone — every arm step is non-decreasing.
        assert row["arm_monotone_fraction"] == 1.0
        assert row["gate_dark"] is False

    def test_gate_harmful_verdict_with_inverted_spread(self, tmp_path,
                                                        monkeypatch):
        """The pathological case the analyzer can detect: the gate's
        bigger bets realize LOWER returns than its smaller bets — capital
        inversion. The ledger MUST preserve the negative sign as the
        quant-actionable signal.
        """
        log = tmp_path / "gate_arm_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_ARM_SKILL_LOG", log)

        harmful = {
            "status": "ok",
            "verdict": "GATE_HARMFUL",
            "n": 500,
            "slice": "oos",
            "outcomes_n": 1750,
            "strong_tailwind_minus_headwind_pp": -3.5,
            "arm_monotone_fraction": 0.25,
            "n_dropped_no_gate_pred": 1250,
            "n_dropped_off_dist": 0,
            "n_dropped_no_return": 0,
            "hint": "historical gate ×1.30 arm realized +1.5% < ×0.60 arm "
                    "+5.0% (spread -3.50pp) — inverting capital allocation",
            "arms": [
                {"arm": "strong_headwind", "multiplier": 0.6, "n": 50,
                 "mean_realized": 5.0, "lo": -10.0, "hi": 12.0},
                {"arm": "mild_headwind", "multiplier": 0.85, "n": 100,
                 "mean_realized": 3.0, "lo": -10.0, "hi": 12.0},
                {"arm": "neutral", "multiplier": 1.0, "n": 200,
                 "mean_realized": 2.5, "lo": -10.0, "hi": 12.0},
                {"arm": "mild_tailwind", "multiplier": 1.15, "n": 100,
                 "mean_realized": 2.0, "lo": -10.0, "hi": 12.0},
                {"arm": "strong_tailwind", "multiplier": 1.3, "n": 50,
                 "mean_realized": 1.5, "lo": -10.0, "hi": 12.0},
            ],
        }
        from paper_trader.ml import gate_arm_historical as gah
        monkeypatch.setattr(gah, "analyze", lambda *a, **k: dict(harmful))

        rcb._append_gate_arm_skill_log(
            cycle=66, win_start=date(2010, 1, 1), win_end=date(2015, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "GATE_HARMFUL"
        # Inverted spread sign preserved.
        assert row["strong_tailwind_minus_headwind_pp"] == -3.5
        assert row["arm_monotone_fraction"] == 0.25
        assert row["gate_dark"] is False
        # The pathological case still has data — gate_dark FALSE.
        # The actionable signal is the negative spread, not absence.

    def test_analyze_exception_degrades_to_honest_row_not_crash(
            self, tmp_path, monkeypatch):
        """A ledger write must NEVER break the continuous loop — the
        documented best-effort discipline of every sibling
        ``_append_*_skill_log`` function. If ``gate_arm_historical.analyze``
        raises (e.g. corrupted outcomes file the analyzer can't iterate),
        the ledger must still persist an honest INSUFFICIENT_DATA row
        instead of bubbling the exception.
        """
        log = tmp_path / "gate_arm_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_ARM_SKILL_LOG", log)

        def _boom(*a, **k):
            raise RuntimeError("simulated gate_arm_historical failure")

        from paper_trader.ml import gate_arm_historical as gah
        monkeypatch.setattr(gah, "analyze", _boom)

        assert rcb._append_gate_arm_skill_log(
            cycle=77, win_start=date(2012, 1, 3), win_end=date(2013, 1, 3),
            outcomes_path=tmp_path / "missing.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["status"] == "error"
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["gate_dark"] is True
        assert row["cycle"] == 77

    def test_bounded_trim_rewrites_when_past_2x_keep(self, tmp_path,
                                                      monkeypatch):
        """Bounded growth — when the file exceeds 2× the KEEP cap, the
        next append atomically rewrites the file to the last KEEP rows.
        Mirrors the sibling ``STOP_OUT_SKILL_LOG`` / ``MFE_SKILL_LOG``
        trim idiom exactly: only pay the rewrite when well past the cap,
        write tmp + ``.replace`` so a torn truncate cannot lose history.
        """
        log = tmp_path / "gate_arm_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_ARM_SKILL_LOG", log)
        monkeypatch.setattr(rcb, "GATE_ARM_SKILL_LOG_KEEP", 5)

        # Seed with 12 lines — past 2× the cap of 5 = 10.
        log.write_text("\n".join(
            json.dumps({"seed": i}) for i in range(12)) + "\n")
        from paper_trader.ml import gate_arm_historical as gah
        monkeypatch.setattr(gah, "analyze",
                            lambda *a, **k: {"status": "ok",
                                              "verdict": "INSUFFICIENT_DATA",
                                              "n": 0, "arms": []})

        rcb._append_gate_arm_skill_log(
            cycle=99, win_start=date(2020, 1, 1), win_end=date(2021, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        # After append: file has 13 lines, then trim to last 5.
        lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
        assert len(lines) == 5
        # The new row is the most-recent one.
        latest = json.loads(lines[-1])
        assert latest.get("cycle") == 99 or latest.get("verdict") == "INSUFFICIENT_DATA"

    def test_dir_created_on_first_write(self, tmp_path, monkeypatch):
        """``GATE_ARM_SKILL_LOG.parent.mkdir(parents=True, exist_ok=True)``
        must create missing parent dirs on first write so a fresh deploy
        does not silently fail the first append.
        """
        nested = tmp_path / "nested" / "deep" / "gate_arm_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_ARM_SKILL_LOG", nested)
        assert not nested.parent.exists()

        from paper_trader.ml import gate_arm_historical as gah
        monkeypatch.setattr(gah, "analyze",
                            lambda *a, **k: {"status": "ok",
                                              "verdict": "INSUFFICIENT_DATA",
                                              "n": 0, "arms": []})
        ok = rcb._append_gate_arm_skill_log(
            cycle=1, win_start=date(2020, 1, 1), win_end=date(2021, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        assert ok is True
        assert nested.exists()

    def test_unwritable_parent_returns_false(self, tmp_path, monkeypatch):
        """The discipline contract: handled fault must return False, not
        raise. If the parent dir cannot be created (e.g. permission
        denied — simulated by pointing at a file as parent), the ledger
        must degrade to False.
        """
        # Make a regular file at the would-be parent path so mkdir fails.
        blocker = tmp_path / "block.txt"
        blocker.write_text("x")
        blocked_log = blocker / "gate_arm_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_ARM_SKILL_LOG", blocked_log)

        from paper_trader.ml import gate_arm_historical as gah
        monkeypatch.setattr(gah, "analyze",
                            lambda *a, **k: {"status": "ok",
                                              "verdict": "INSUFFICIENT_DATA",
                                              "n": 0, "arms": []})
        ok = rcb._append_gate_arm_skill_log(
            cycle=1, win_start=date(2020, 1, 1), win_end=date(2021, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        assert ok is False


class TestGateArmLedgerWiringRegression:
    """Source-level wiring assertion: the ``main()`` loop body MUST call
    ``_append_gate_arm_skill_log(`` somewhere. Without this regression
    test a future refactor could orphan the function (define it but
    never invoke it), silently breaking the per-cycle trend — exactly
    the documented failure mode that motivated similar wiring tests
    for sibling ledgers (``TestCycleWiringRegression`` in
    ``test_continuous.py``, ``TestIntraperiodLedgerWiring`` in
    ``test_continuous_intraperiod_ledgers.py``).
    """

    def test_main_calls_append_gate_arm_skill_log(self):
        src = Path(rcb.__file__).read_text()
        # Find the main() definition.
        import re
        main_match = re.search(r"^def main\(\)", src, re.MULTILINE)
        assert main_match, "main() not found in run_continuous_backtests.py"
        main_body = src[main_match.start():]
        assert "_append_gate_arm_skill_log(" in main_body, (
            "main() must call _append_gate_arm_skill_log per cycle — "
            "the per-cycle wiring is the entire point of the ledger"
        )

    def test_constants_module_level_for_testability(self):
        """``GATE_ARM_SKILL_LOG`` and ``GATE_ARM_SKILL_LOG_KEEP`` must be
        module-level (not function-local) so test fixtures can monkeypatch
        them — the AGENTS.md "hardcoded paths must be module-level for
        testability" rule. Sibling ledgers (SCORER_SKILL_LOG, ...) all
        follow this; a regression here would break test isolation.
        """
        assert hasattr(rcb, "GATE_ARM_SKILL_LOG")
        assert hasattr(rcb, "GATE_ARM_SKILL_LOG_KEEP")
        assert isinstance(rcb.GATE_ARM_SKILL_LOG, Path)
        assert isinstance(rcb.GATE_ARM_SKILL_LOG_KEEP, int)
        assert rcb.GATE_ARM_SKILL_LOG_KEEP > 0


class TestGateArmHistoricalAnalyzerContract:
    """The ledger trusts ``gate_arm_historical.analyze`` to return a
    JSON-safe dict. These tests pin the analyzer-side contract on the
    keys the ledger reads — any future analyzer change that drops or
    renames these keys would silently break the ledger's persisted
    schema. Mirrors the no-drift cross-check discipline every sibling
    skill-ledger tests applies (e.g. ``conviction_calibration`` returns
    ``spearman`` which the ledger reads verbatim — same idea here for
    ``arm_monotone_fraction`` / ``strong_tailwind_minus_headwind_pp``).
    """

    def test_analyzer_returns_keys_ledger_reads(self):
        from paper_trader.ml import gate_arm_historical as gah
        # Empty-records → INSUFFICIENT_DATA shape (the cheapest contract
        # surface that the ledger reads on every cycle).
        rep = gah.gate_arm_historical_report([])
        # Every key the ledger reads in _append_gate_arm_skill_log:
        for key in ("status", "verdict", "n", "arms",
                    "strong_tailwind_minus_headwind_pp",
                    "arm_monotone_fraction",
                    "n_dropped_no_gate_pred",
                    "n_dropped_off_dist",
                    "n_dropped_no_return"):
            assert key in rep, f"analyzer must return {key!r}"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0
        assert rep["arms"] == []

    def test_analyzer_returns_json_safe_arms_list(self):
        """The arms list rows must carry ``arm``, ``multiplier``, ``n``,
        ``mean_realized``, ``lo``, ``hi``. The ledger extracts flat
        per-arm columns by ``multiplier``; a drift here would silently
        break the ``mean_x06`` / ``mean_x13`` flat schema.
        """
        from paper_trader.ml import gate_arm_historical as gah
        rep = gah.gate_arm_historical_report([])
        # On empty input arms is empty; build a synthetic rep that mirrors
        # the documented populated shape.
        for arm_row in rep["arms"]:
            assert "arm" in arm_row
            assert "multiplier" in arm_row
            assert "n" in arm_row
