"""Per-cycle ledger tests for the gate economic-counterfactual analyzer.

``paper_trader/ml/gate_pnl.py`` already CLI-reports the realized economic
effect of the conviction-gate multiplier overlay (×0.6 / ×0.85 / ×1.0 /
×1.15 / ×1.3) rolled up across ALL five arms — the single quant-decisive
"does the reallocation pay" number the per-arm view structurally cannot
produce. The deployed gate's current OOS verdict on the live corpus
combines ``gate_arm_historical=GATE_INEFFECTIVE`` (per-arm spread
sub-tolerance) with a per-cycle aggregate effect that until now was
visible only by manually running ``python3 -m paper_trader.ml.gate_pnl``.
Until ``_append_gate_pnl_skill_log`` landed there was NO per-cycle
durable trend for the keep-or-kill verdict — exactly the same
operator-blind state ``_append_gate_arm_skill_log`` closed for the arms
breakdown. These tests pin the wiring: best-effort discipline, honest
gap rows when the analyzer is dark, bounded trim, SSOT cross-check that
the persisted verdict equals what the analyzer returned. Mirrors the
discipline of every sibling ``_append_*_skill_log`` test in
``test_continuous.py`` and ``test_continuous_gate_arm_ledger.py``.

A new test file (not appended to ``test_continuous.py`` or
``test_continuous_gate_arm_ledger.py``) so a concurrent sibling agent
editing the same test file cannot collide with this work via whole-file
``git add`` — the documented same-role HYBRID staging-race mitigation
pattern (memory ``pt-concurrent-samerole-staging-race``).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import run_continuous_backtests as rcb


class TestAppendGatePnlSkillLog:
    """The per-cycle gate economic-counterfactual ledger — answers "does
    the gate's multiplier overlay net ADD or SUBTRACT realized return on
    the sized fills, vs not gating at all?" durably and per-cycle so a
    quant can trend the keep-or-kill verdict. Mirrors the sibling
    ``_append_gate_arm_skill_log`` / ``_append_conviction_calibration_log``
    discipline: best-effort, honest gap rows, atomic bounded trim, SSOT
    no-drift, never breaks the loop.
    """

    def test_insufficient_data_persists_dark_row(self, tmp_path, monkeypatch):
        """The documented live state where ``gate_pnl.analyze`` reports
        INSUFFICIENT_DATA — typically the cycle is pre-``n_train>=500``
        so the gate never acted, OR the corpus has fewer than
        ``MIN_TOTAL=30`` aligned (pred, realized) pairs. The ledger must
        STILL append the row so the darkness is visible in the trend,
        exactly how the gate-arm ledger surfaces its ``gate_dark`` state.
        """
        log = tmp_path / "gate_pnl_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_PNL_SKILL_LOG", log)

        from paper_trader.ml import gate_pnl as gp
        empty = {
            "status": "ok",
            "verdict": "INSUFFICIENT_DATA",
            "n": 0,
            "gate_off_mean_pct": None,
            "gate_on_mean_pct": None,
            "equal_weight_gate_contribution_pp": None,
            "sized_gate_contribution_pp": None,
            "sized_n": 0,
            "avg_gate_multiplier": None,
            "hint": "need ≥30 usable pairs; have n=0",
            "slice": "oos",
            "n_records_considered": 1750,
            "n_train": 8417,
        }
        monkeypatch.setattr(gp, "analyze", lambda *a, **k: dict(empty))

        assert rcb._append_gate_pnl_skill_log(
            cycle=33, win_start=date(2020, 1, 2), win_end=date(2025, 1, 2),
            outcomes_path=tmp_path / "nope.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["cycle"] == 33
        assert row["verdict"] == "INSUFFICIENT_DATA"
        # ``gate_pnl_dark`` mirrors the sibling ``gate_dark`` / ``stop_dark``
        # / ``tp_dark`` / ``sizing_dark`` / ``calibrated_dark`` boolean.
        assert row["gate_pnl_dark"] is True
        assert row["n"] == 0
        assert row["slice"] == "oos"
        assert row["n_records_considered"] == 1750
        assert row["n_train"] == 8417
        assert row["window_start"] == "2020-01-02"
        assert row["window_end"] == "2025-01-02"
        # No contribution to surface when dark.
        assert row["equal_weight_gate_contribution_pp"] is None
        assert row["sized_gate_contribution_pp"] is None
        assert row["sized_n"] == 0

    def test_gate_adds_return_verdict_surfaces_positive_contribution(
            self, tmp_path, monkeypatch):
        """SSOT cross-check: the analyzer's verdict and the headline
        ``equal_weight_gate_contribution_pp`` must round-trip unchanged
        through the ledger. A drift here would silently break trend
        integrity — the whole point of the per-cycle wiring is no-drift
        fidelity to the CLI verdict.

        This is the "the gate's reallocation EARNS its keep" state — the
        actionable positive signal a quant needs trended so the moment
        the gate starts adding real return is visible, not buried in CLI.
        """
        log = tmp_path / "gate_pnl_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_PNL_SKILL_LOG", log)

        adds = {
            "status": "ok",
            "verdict": "GATE_ADDS_RETURN",
            "n": 512,
            "slice": "oos",
            "n_records_considered": 1750,
            "n_train": 8417,
            "gate_off_mean_pct": 2.30,
            "gate_on_mean_pct": 4.85,
            "equal_weight_gate_contribution_pp": 2.55,
            "sized_gate_contribution_pp": 3.10,
            "sized_n": 488,
            "avg_gate_multiplier": 1.07,
            "hint": "gate-on realized +4.85% > gate-off +2.30% "
                    "(contribution +2.55pp) — the multiplier overlay net "
                    "sizes toward the winners; the gate's reallocation "
                    "earns its keep on these fills",
        }
        from paper_trader.ml import gate_pnl as gp
        monkeypatch.setattr(gp, "analyze", lambda *a, **k: dict(adds))

        rcb._append_gate_pnl_skill_log(
            cycle=44, win_start=date(2018, 6, 1), win_end=date(2023, 6, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "GATE_ADDS_RETURN"
        assert row["gate_pnl_dark"] is False
        # Sign + magnitude preserved exactly (SSOT) — equal-weight is
        # the verdict-driving headline.
        assert row["equal_weight_gate_contribution_pp"] == 2.55
        # Sized contribution preserved as the secondary informational
        # number (the gate_pnl honesty pattern — reported but never
        # folded into the verdict).
        assert row["sized_gate_contribution_pp"] == 3.10
        assert row["gate_off_mean_pct"] == 2.30
        assert row["gate_on_mean_pct"] == 4.85
        assert row["avg_gate_multiplier"] == 1.07
        assert row["sized_n"] == 488
        assert row["n"] == 512

    def test_gate_subtracts_return_verdict_preserves_negative_sign(
            self, tmp_path, monkeypatch):
        """The pathological case: the multiplier overlay net sizes
        toward LOSERS — turning the gate off would have realized more.
        The ledger MUST preserve the negative sign as the quant-actionable
        signal so the trend visibly captures the regime.
        """
        log = tmp_path / "gate_pnl_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_PNL_SKILL_LOG", log)

        subtracts = {
            "status": "ok",
            "verdict": "GATE_SUBTRACTS_RETURN",
            "n": 328,
            "slice": "oos",
            "n_records_considered": 1750,
            "n_train": 8417,
            "gate_off_mean_pct": 3.50,
            "gate_on_mean_pct": 1.80,
            "equal_weight_gate_contribution_pp": -1.70,
            "sized_gate_contribution_pp": -1.20,
            "sized_n": 310,
            "avg_gate_multiplier": 0.93,
            "hint": "gate-on realized +1.80% < gate-off +3.50% "
                    "(contribution -1.70pp) — the multiplier overlay net "
                    "sizes toward the losers; not gating would have "
                    "realized more on these fills",
        }
        from paper_trader.ml import gate_pnl as gp
        monkeypatch.setattr(gp, "analyze", lambda *a, **k: dict(subtracts))

        rcb._append_gate_pnl_skill_log(
            cycle=66, win_start=date(2010, 1, 1), win_end=date(2015, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "GATE_SUBTRACTS_RETURN"
        # The actionable signal is the negative sign — preserved.
        assert row["equal_weight_gate_contribution_pp"] == -1.70
        assert row["sized_gate_contribution_pp"] == -1.20
        assert row["gate_pnl_dark"] is False
        # The pathological case still has data — gate_pnl_dark FALSE.

    def test_gate_return_neutral_verdict_within_band(
            self, tmp_path, monkeypatch):
        """The current expected live state: the gate's reallocation lands
        inside ``±EDGE_TOL_PP=1.0pp`` — neither adds nor subtracts net
        return, pure added sizing variance. Verdict and the (small,
        non-zero) contribution must round-trip exactly so a quant can see
        the magnitude even when the verdict is NEUTRAL.
        """
        log = tmp_path / "gate_pnl_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_PNL_SKILL_LOG", log)

        neutral = {
            "status": "ok",
            "verdict": "GATE_RETURN_NEUTRAL",
            "n": 421,
            "slice": "oos",
            "n_records_considered": 1750,
            "n_train": 8417,
            "gate_off_mean_pct": 2.80,
            "gate_on_mean_pct": 2.95,
            "equal_weight_gate_contribution_pp": 0.15,
            "sized_gate_contribution_pp": 0.22,
            "sized_n": 410,
            "avg_gate_multiplier": 1.01,
            "hint": "gate-on +2.95% vs gate-off +2.80% (contribution "
                    "+0.15pp, within ±1.0pp) — the overlay reallocates "
                    "capital with no net realized effect: pure added "
                    "sizing variance",
        }
        from paper_trader.ml import gate_pnl as gp
        monkeypatch.setattr(gp, "analyze", lambda *a, **k: dict(neutral))

        rcb._append_gate_pnl_skill_log(
            cycle=77, win_start=date(2019, 3, 1), win_end=date(2024, 3, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "GATE_RETURN_NEUTRAL"
        # The non-zero in-band magnitude is still informative for trend
        # — preserved exactly.
        assert row["equal_weight_gate_contribution_pp"] == 0.15
        assert row["gate_pnl_dark"] is False

    def test_analyze_exception_degrades_to_honest_row_not_crash(
            self, tmp_path, monkeypatch):
        """A ledger write must NEVER break the continuous loop — the
        documented best-effort discipline of every sibling
        ``_append_*_skill_log`` function. If ``gate_pnl.analyze`` raises
        (e.g. corrupted outcomes file the analyzer can't iterate, an
        unhandled type from a future schema change), the ledger must
        still persist an honest INSUFFICIENT_DATA row instead of
        bubbling the exception.
        """
        log = tmp_path / "gate_pnl_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_PNL_SKILL_LOG", log)

        def _boom(*a, **k):
            raise RuntimeError("simulated gate_pnl failure")

        from paper_trader.ml import gate_pnl as gp
        monkeypatch.setattr(gp, "analyze", _boom)

        assert rcb._append_gate_pnl_skill_log(
            cycle=77, win_start=date(2012, 1, 3), win_end=date(2013, 1, 3),
            outcomes_path=tmp_path / "missing.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["status"] == "error"
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["gate_pnl_dark"] is True
        assert row["cycle"] == 77

    def test_non_dict_return_degrades_to_honest_insufficient(
            self, tmp_path, monkeypatch):
        """Defense-in-depth against a future analyzer signature drift
        that returns ``None`` or a list. The ledger must coerce to an
        honest INSUFFICIENT_DATA row rather than crash on ``.get()``.
        Mirrors the sibling ``_append_*_skill_log`` non-dict guard.
        """
        log = tmp_path / "gate_pnl_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_PNL_SKILL_LOG", log)

        from paper_trader.ml import gate_pnl as gp
        monkeypatch.setattr(gp, "analyze", lambda *a, **k: None)

        assert rcb._append_gate_pnl_skill_log(
            cycle=88, win_start=date(2020, 1, 1), win_end=date(2021, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["gate_pnl_dark"] is True
        assert row["n"] == 0

    def test_bounded_trim_rewrites_when_past_2x_keep(self, tmp_path,
                                                     monkeypatch):
        """Bounded growth — when the file exceeds 2× the KEEP cap, the
        next append atomically rewrites the file to the last KEEP rows.
        Mirrors the sibling ``GATE_ARM_SKILL_LOG`` / ``STOP_OUT_SKILL_LOG``
        / ``MFE_SKILL_LOG`` trim idiom exactly: only pay the rewrite when
        well past the cap, write tmp + ``.replace`` so a torn truncate
        cannot lose history.
        """
        log = tmp_path / "gate_pnl_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_PNL_SKILL_LOG", log)
        monkeypatch.setattr(rcb, "GATE_PNL_SKILL_LOG_KEEP", 5)

        # Seed with 12 lines — past 2× the cap of 5 = 10.
        log.write_text("\n".join(
            json.dumps({"seed": i}) for i in range(12)) + "\n")
        from paper_trader.ml import gate_pnl as gp
        monkeypatch.setattr(gp, "analyze",
                            lambda *a, **k: {"status": "ok",
                                             "verdict": "INSUFFICIENT_DATA",
                                             "n": 0})

        rcb._append_gate_pnl_skill_log(
            cycle=99, win_start=date(2020, 1, 1), win_end=date(2021, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        # After append: file had 13 lines, then trim to last 5.
        lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
        assert len(lines) == 5
        # The new row is the most-recent one (latest by tail position).
        latest = json.loads(lines[-1])
        assert latest.get("cycle") == 99 or latest.get("verdict") == "INSUFFICIENT_DATA"

    def test_dir_created_on_first_write(self, tmp_path, monkeypatch):
        """``GATE_PNL_SKILL_LOG.parent.mkdir(parents=True, exist_ok=True)``
        must create missing parent dirs on first write so a fresh deploy
        does not silently fail the first append.
        """
        nested = tmp_path / "nested" / "deep" / "gate_pnl_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_PNL_SKILL_LOG", nested)
        assert not nested.parent.exists()

        from paper_trader.ml import gate_pnl as gp
        monkeypatch.setattr(gp, "analyze",
                            lambda *a, **k: {"status": "ok",
                                             "verdict": "INSUFFICIENT_DATA",
                                             "n": 0})
        ok = rcb._append_gate_pnl_skill_log(
            cycle=1, win_start=date(2020, 1, 1), win_end=date(2021, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        assert ok is True
        assert nested.exists()

    def test_unwritable_parent_returns_false(self, tmp_path, monkeypatch):
        """The discipline contract: handled fault must return False, not
        raise. If the parent dir cannot be created (e.g. permission denied
        — simulated by pointing at a regular file as parent), the ledger
        must degrade to False.
        """
        blocker = tmp_path / "block.txt"
        blocker.write_text("x")
        blocked_log = blocker / "gate_pnl_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_PNL_SKILL_LOG", blocked_log)

        from paper_trader.ml import gate_pnl as gp
        monkeypatch.setattr(gp, "analyze",
                            lambda *a, **k: {"status": "ok",
                                             "verdict": "INSUFFICIENT_DATA",
                                             "n": 0})
        ok = rcb._append_gate_pnl_skill_log(
            cycle=1, win_start=date(2020, 1, 1), win_end=date(2021, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        assert ok is False

    def test_default_outcomes_path_resolves_when_none(
            self, tmp_path, monkeypatch):
        """When ``outcomes_path=None`` the ledger must default to
        ``ROOT/data/decision_outcomes.jsonl`` so the wired ``main()``
        call (which passes no path) targets the production corpus.
        Mirrors the sibling ledgers' default-path contract.
        """
        log = tmp_path / "gate_pnl_skill_log.jsonl"
        monkeypatch.setattr(rcb, "GATE_PNL_SKILL_LOG", log)

        captured: dict = {}

        def _capture(outcomes_path, **kwargs):
            captured["path"] = outcomes_path
            return {"status": "ok", "verdict": "INSUFFICIENT_DATA", "n": 0}

        from paper_trader.ml import gate_pnl as gp
        monkeypatch.setattr(gp, "analyze", _capture)

        rcb._append_gate_pnl_skill_log(
            cycle=1, win_start=date(2020, 1, 1), win_end=date(2021, 1, 1),
            outcomes_path=None)
        # Default is ROOT/data/decision_outcomes.jsonl.
        from paper_trader.backtest import ROOT
        assert captured["path"] == ROOT / "data" / "decision_outcomes.jsonl"


class TestGatePnlLedgerWiringRegression:
    """Source-level wiring assertion: the ``main()`` loop body MUST call
    ``_append_gate_pnl_skill_log(`` somewhere. Without this regression
    test a future refactor could orphan the function (define it but
    never invoke it), silently breaking the per-cycle trend — exactly
    the documented failure mode that motivated similar wiring tests
    for sibling ledgers (``TestCycleWiringRegression`` in
    ``test_continuous.py``, ``TestGateArmLedgerWiringRegression`` in
    ``test_continuous_gate_arm_ledger.py``).
    """

    def test_main_calls_append_gate_pnl_skill_log(self):
        src = Path(rcb.__file__).read_text()
        import re
        main_match = re.search(r"^def main\(\)", src, re.MULTILINE)
        assert main_match, "main() not found in run_continuous_backtests.py"
        main_body = src[main_match.start():]
        assert "_append_gate_pnl_skill_log(" in main_body, (
            "main() must call _append_gate_pnl_skill_log per cycle — the "
            "per-cycle wiring is the entire point of the ledger"
        )

    def test_constants_module_level_for_testability(self):
        """``GATE_PNL_SKILL_LOG`` and ``GATE_PNL_SKILL_LOG_KEEP`` must be
        module-level (not function-local) so test fixtures can monkeypatch
        them — the AGENTS.md "hardcoded paths must be module-level for
        testability" rule. Sibling ledgers (SCORER_SKILL_LOG, ...) all
        follow this; a regression here would break test isolation.
        """
        assert hasattr(rcb, "GATE_PNL_SKILL_LOG")
        assert hasattr(rcb, "GATE_PNL_SKILL_LOG_KEEP")
        assert isinstance(rcb.GATE_PNL_SKILL_LOG, Path)
        assert isinstance(rcb.GATE_PNL_SKILL_LOG_KEEP, int)
        assert rcb.GATE_PNL_SKILL_LOG_KEEP > 0


class TestGatePnlAnalyzerContract:
    """The ledger trusts ``gate_pnl.analyze`` (and the underlying
    ``gate_pnl_report``) to return a JSON-safe dict with the exact keys
    the ledger reads. These tests pin the analyzer-side contract — any
    future analyzer change that drops or renames these keys would
    silently break the ledger's persisted schema. Mirrors the no-drift
    cross-check discipline every sibling skill-ledger test applies.
    """

    def test_report_returns_keys_ledger_reads_on_empty_input(self):
        """The cheapest contract surface that the ledger exercises on
        every cycle: the empty-input INSUFFICIENT_DATA branch. Pin the
        full key surface here so a regression that drops a key is caught
        at the analyzer level rather than discovered as a ``KeyError``
        / silent ``None`` in the ledger row.
        """
        from paper_trader.ml import gate_pnl as gp
        rep = gp.gate_pnl_report([])
        for key in ("status", "verdict", "n",
                    "gate_off_mean_pct", "gate_on_mean_pct",
                    "equal_weight_gate_contribution_pp",
                    "sized_gate_contribution_pp",
                    "sized_n", "avg_gate_multiplier", "hint"):
            assert key in rep, f"analyzer must return {key!r}"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0
        # Headline contribution is None on empty — the ledger's dark-row
        # contract relies on this exact sentinel.
        assert rep["equal_weight_gate_contribution_pp"] is None

    def test_report_below_min_total_still_insufficient(self):
        """The ``MIN_TOTAL=30`` floor must hold: a 5-pair input still
        reads INSUFFICIENT_DATA. Pin this so a future tighten/loosen of
        ``MIN_TOTAL`` is intentional, not accidental, and the ledger's
        ``gate_pnl_dark`` boolean stays semantically meaningful (it
        tracks the "no usable rollup data" state).
        """
        from paper_trader.ml import gate_pnl as gp
        # 5 finite triples — below MIN_TOTAL=30.
        triples = [(1.0, 0.5, 0.1), (2.0, -0.3, 0.1),
                   (-1.5, 0.2, 0.1), (0.5, 1.1, 0.1), (3.0, -0.8, 0.1)]
        rep = gp.gate_pnl_report(triples)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 5
        # Sentinel for the ledger's dark-row contract.
        assert rep["equal_weight_gate_contribution_pp"] is None
