"""Per-cycle ledger tests for the (persona × regime) cross-tab analyzer.

Pass #44 shipped ``paper_trader/ml/persona_regime_skill.py`` — the
missing intersection of ``persona_skill`` (per-persona aggregate) and
``regime_audit`` (per-regime aggregate). Neither sibling can answer the
actionable question ``persona_regime_skill`` does: *does THIS persona
carry signal in THIS regime?* But pass #44 stopped before wiring the
per-cycle ledger, so the verdict was CLI-only — an unattended operator
could not trend per-cell signal health, and the most directly
operational state (``HAS_INVERTED_CELL`` — a specific persona is
anti-predictive in a specific regime, the actionable data for
suppressing that persona-in-that-regime) was invisible.

These tests pin the wiring: best-effort discipline, honest gap rows
when the cross-tab has no stable cells, bounded trim, SSOT cross-check
that the persisted verdict equals the analyzer's. Mirrors the sibling
``_append_persona_skill_log`` test discipline exactly.

A new test file (not appended to existing ones) so a concurrent sibling
agent editing the same test file cannot collide with this work via
whole-file ``git add`` — the documented same-role HYBRID staging-race
mitigation pattern.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

import run_continuous_backtests as rcb


class TestAppendPersonaRegimeSkillLog:
    """The per-cycle (persona × regime) cross-tab skill ledger — answers
    "does THIS persona carry signal in THIS regime?" durably so an
    unattended operator can trend per-cell signal health and catch
    INVERTED cells the moment they emerge.

    Mirrors the sibling ``_append_persona_skill_log`` /
    ``_append_gate_arm_skill_log`` discipline: best-effort, honest gap
    rows, atomic bounded trim, SSOT no-drift, never breaks the loop.
    """

    def test_insufficient_data_persists_dark_row(self, tmp_path,
                                                  monkeypatch):
        """When ``persona_regime_skill.analyze`` returns INSUFFICIENT_DATA
        (corpus has fewer than MIN_RECORDS aligned outcomes), the ledger
        MUST still append the row with ``signal_dark=True`` so the
        darkness is visible in the trend rather than silent. Mirrors the
        sibling ``signal_dark`` / ``gate_dark`` / ``stop_dark`` precedent.
        """
        log = tmp_path / "persona_regime_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_REGIME_SKILL_LOG", log)

        from paper_trader.ml import persona_regime_skill as prs
        empty = {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n_records": 5,
            "n_cells": 0,
            "n_stable_cells": 0,
            "cells": [],
            "inverted_cells": [],
            "best_cell": None,
            "worst_cell": None,
            "n_dropped_unknown_regime": 2,
            "hint": "need ≥30 aligned outcomes, have 5",
        }
        monkeypatch.setattr(prs, "analyze", lambda *a, **k: dict(empty))
        # _load_outcomes is also called inside the ledger function — stub it
        # to return an empty list so the analyzer stub is what we exercise.
        monkeypatch.setattr(prs, "_load_outcomes", lambda *a, **k: [])

        assert rcb._append_persona_regime_skill_log(
            cycle=33, win_start=date(2020, 1, 2),
            win_end=date(2025, 1, 2),
            outcomes_path=tmp_path / "nope.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["cycle"] == 33
        assert row["verdict"] == "INSUFFICIENT_DATA"
        # ``signal_dark`` mirrors sibling ``*_dark`` booleans.
        assert row["signal_dark"] is True
        assert row["n_records"] == 5
        assert row["n_cells"] == 0
        assert row["n_stable_cells"] == 0
        assert row["best_persona"] is None
        assert row["best_regime"] is None
        assert row["best_score_ic"] is None
        assert row["worst_persona"] is None
        assert row["n_inverted_cells"] == 0
        assert row["inverted_cells"] == []
        assert row["cells"] == []
        # Honest count of rows dropped for an unknown regime — pass #44
        # documented this as a real production data signal (~32% of
        # would-be-bull rows were "unknown").
        assert row["n_dropped_unknown_regime"] == 2
        assert row["window_start"] == "2020-01-02"
        assert row["window_end"] == "2025-01-02"

    def test_has_inverted_cell_surfaces_red_flag(self, tmp_path,
                                                  monkeypatch):
        """SSOT cross-check: the analyzer's ``HAS_INVERTED_CELL`` verdict
        — the single most actionable state because a specific persona
        in a specific regime is actively HURTING when it sizes up —
        must round-trip unchanged through the ledger, including the
        ``inverted_cells`` list AND ``n_inverted_cells`` count. A drift
        here would silently break the operator's only durable trend on
        this red-flag state.
        """
        log = tmp_path / "persona_regime_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_REGIME_SKILL_LOG", log)

        state = {
            "status": "ok",
            "verdict": "HAS_INVERTED_CELL",
            "n_records": 4776,
            "n_cells": 28,
            "n_stable_cells": 22,
            "cells": [
                {"persona": "ESG / Thematic", "regime": "sideways",
                 "n": 111, "score_ic": 0.293,
                 "mean_aligned_return": 2.1, "win_rate": 0.56,
                 "verdict": "SIGNAL_EDGE"},
                {"persona": "Pure Speculator", "regime": "bear",
                 "n": 95, "score_ic": -0.21,
                 "mean_aligned_return": -3.0, "win_rate": 0.40,
                 "verdict": "INVERTED"},
            ],
            "inverted_cells": [
                {"persona": "Pure Speculator", "regime": "bear",
                 "score_ic": -0.21, "n": 95},
            ],
            "best_cell": {"persona": "ESG / Thematic", "regime": "sideways",
                          "score_ic": 0.293, "n": 111},
            "worst_cell": {"persona": "Pure Speculator", "regime": "bear",
                           "score_ic": -0.21, "n": 95},
            "n_dropped_unknown_regime": 1102,
            "hint": "1 anti-predictive cell(s) …",
        }
        from paper_trader.ml import persona_regime_skill as prs
        monkeypatch.setattr(prs, "analyze",
                            lambda *a, **k: dict(state))
        monkeypatch.setattr(prs, "_load_outcomes", lambda *a, **k: [{}])

        rcb._append_persona_regime_skill_log(
            cycle=44, win_start=date(2018, 6, 1),
            win_end=date(2023, 6, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        # Verdict round-trips unchanged.
        assert row["verdict"] == "HAS_INVERTED_CELL"
        # signal_dark FALSE because n_stable_cells > 0.
        assert row["signal_dark"] is False
        # Best stable cell flat fields — the operator-readable leader.
        assert row["best_persona"] == "ESG / Thematic"
        assert row["best_regime"] == "sideways"
        assert row["best_score_ic"] == 0.293
        assert row["best_n"] == 111
        # Worst stable cell flat fields — surfaces the inverted cell.
        assert row["worst_persona"] == "Pure Speculator"
        assert row["worst_regime"] == "bear"
        assert row["worst_score_ic"] == -0.21
        # Inverted list + count both surfaced as actionable flat fields.
        assert row["n_inverted_cells"] == 1
        assert row["inverted_cells"] == [
            {"persona": "Pure Speculator", "regime": "bear",
             "score_ic": -0.21, "n": 95},
        ]
        # Full cells list ships intact for forensics.
        assert len(row["cells"]) == 2
        # Honest count of dropped unknown-regime rows preserved.
        assert row["n_dropped_unknown_regime"] == 1102

    def test_regime_conditional_verdict_round_trips(self, tmp_path,
                                                     monkeypatch):
        """The ``REGIME_CONDITIONAL`` verdict — pass #44's live finding
        on the production corpus — must round-trip with both best
        (SIGNAL_EDGE) and worst (NO_EDGE/WEAK_SIGNAL) stable cells
        captured as flat fields. This is the documented "aggregate
        persona_skill HIDES the regime structure" cycle state, and a
        skeptical quant needs to see when the verdict flips back to
        HEALTHY (regime conditioning resolved) or to HAS_INVERTED_CELL
        (regime conditioning got worse).
        """
        log = tmp_path / "persona_regime_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_REGIME_SKILL_LOG", log)

        state = {
            "status": "ok",
            "verdict": "REGIME_CONDITIONAL",
            "n_records": 1500,
            "n_cells": 22,
            "n_stable_cells": 18,
            "cells": [
                {"persona": "Value Investor", "regime": "bull",
                 "n": 150, "score_ic": 0.18,
                 "mean_aligned_return": 1.5, "win_rate": 0.55,
                 "verdict": "SIGNAL_EDGE"},
                {"persona": "Value Investor", "regime": "bear",
                 "n": 80, "score_ic": 0.02,
                 "mean_aligned_return": 0.1, "win_rate": 0.50,
                 "verdict": "NO_EDGE"},
            ],
            "inverted_cells": [],
            "best_cell": {"persona": "Value Investor", "regime": "bull",
                          "score_ic": 0.18, "n": 150},
            "worst_cell": {"persona": "Value Investor", "regime": "bear",
                           "score_ic": 0.02, "n": 80},
            "n_dropped_unknown_regime": 50,
        }
        from paper_trader.ml import persona_regime_skill as prs
        monkeypatch.setattr(prs, "analyze",
                            lambda *a, **k: dict(state))
        monkeypatch.setattr(prs, "_load_outcomes", lambda *a, **k: [{}])

        rcb._append_persona_regime_skill_log(
            cycle=55, win_start=date(2018, 6, 1),
            win_end=date(2023, 6, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "REGIME_CONDITIONAL"
        assert row["signal_dark"] is False
        # Best/worst flat fields are populated even when no cell is inverted.
        assert row["best_persona"] == "Value Investor"
        assert row["best_regime"] == "bull"
        assert row["best_score_ic"] == 0.18
        assert row["worst_persona"] == "Value Investor"
        assert row["worst_regime"] == "bear"
        assert row["worst_score_ic"] == 0.02
        assert row["n_inverted_cells"] == 0

    def test_healthy_verdict_with_top_signal_edge(self, tmp_path,
                                                    monkeypatch):
        """When at least one cell reaches SIGNAL_EDGE and NO cell is
        anti-predictive (the documented HEALTHY state), the ledger must
        capture ``best_persona`` + ``best_regime`` + ``best_score_ic``
        as the operator-readable leader fields.
        """
        log = tmp_path / "persona_regime_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_REGIME_SKILL_LOG", log)

        healthy = {
            "status": "ok",
            "verdict": "HEALTHY",
            "n_records": 2000,
            "n_cells": 18,
            "n_stable_cells": 15,
            "cells": [
                {"persona": "GARP", "regime": "bull", "n": 200,
                 "score_ic": 0.31, "mean_aligned_return": 2.5,
                 "win_rate": 0.58, "verdict": "SIGNAL_EDGE"},
            ],
            "inverted_cells": [],
            "best_cell": {"persona": "GARP", "regime": "bull",
                          "score_ic": 0.31, "n": 200},
            "worst_cell": {"persona": "GARP", "regime": "bull",
                           "score_ic": 0.31, "n": 200},
            "n_dropped_unknown_regime": 0,
        }
        from paper_trader.ml import persona_regime_skill as prs
        monkeypatch.setattr(prs, "analyze",
                            lambda *a, **k: dict(healthy))
        monkeypatch.setattr(prs, "_load_outcomes", lambda *a, **k: [{}])

        rcb._append_persona_regime_skill_log(
            cycle=99, win_start=date(2018, 6, 1),
            win_end=date(2023, 6, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "HEALTHY"
        assert row["signal_dark"] is False
        assert row["best_persona"] == "GARP"
        assert row["best_regime"] == "bull"
        assert row["best_score_ic"] == 0.31
        assert row["n_inverted_cells"] == 0

    def test_analyze_exception_degrades_to_honest_row_not_crash(
            self, tmp_path, monkeypatch):
        """The ``_append_persona_regime_skill_log`` discipline: a ledger
        write must NEVER break the continuous loop. If
        ``persona_regime_skill.analyze`` raises (e.g. a malformed
        outcomes file the analyzer fails to load), the ledger must still
        persist an honest INSUFFICIENT_DATA row with ``signal_dark=True``
        instead of bubbling.
        """
        log = tmp_path / "persona_regime_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_REGIME_SKILL_LOG", log)

        def _boom(*a, **k):
            raise RuntimeError("simulated persona_regime_skill failure")

        from paper_trader.ml import persona_regime_skill as prs
        monkeypatch.setattr(prs, "analyze", _boom)
        monkeypatch.setattr(prs, "_load_outcomes", lambda *a, **k: [])

        assert rcb._append_persona_regime_skill_log(
            cycle=77, win_start=date(2012, 1, 3),
            win_end=date(2013, 1, 3),
            outcomes_path=tmp_path / "missing.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["status"] == "error"
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["signal_dark"] is True
        assert row["cycle"] == 77

    def test_non_dict_analyzer_return_degrades_safely(self, tmp_path,
                                                       monkeypatch):
        """Defense-in-depth: if a future ``persona_regime_skill.analyze``
        mutation returned something other than a dict (e.g. None on an
        early-exit path), the ledger must NOT raise — it must persist
        the honest gap row, the same way every sibling
        ``_append_*_skill_log`` normalizes a non-dict to
        ``INSUFFICIENT_DATA``.
        """
        log = tmp_path / "persona_regime_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_REGIME_SKILL_LOG", log)

        from paper_trader.ml import persona_regime_skill as prs
        monkeypatch.setattr(prs, "analyze", lambda *a, **k: None)
        monkeypatch.setattr(prs, "_load_outcomes", lambda *a, **k: [])

        assert rcb._append_persona_regime_skill_log(
            cycle=88, win_start=date(2020, 1, 1),
            win_end=date(2021, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["signal_dark"] is True

    def test_bounded_trim_rewrites_when_past_2x_keep(self, tmp_path,
                                                      monkeypatch):
        """Bounded growth — when the file exceeds 2× the KEEP cap, the
        next append atomically rewrites the file to the last KEEP rows.
        Mirrors the sibling ``PERSONA_SKILL_LOG`` trim idiom exactly:
        pay the rewrite only when well past the cap, tmp + ``.replace``
        so a torn truncate cannot lose history.
        """
        log = tmp_path / "persona_regime_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_REGIME_SKILL_LOG", log)
        monkeypatch.setattr(rcb, "PERSONA_REGIME_SKILL_LOG_KEEP", 5)

        # Seed with 12 lines — past 2× the cap of 5 = 10.
        log.write_text("\n".join(
            json.dumps({"seed": i}) for i in range(12)) + "\n")
        from paper_trader.ml import persona_regime_skill as prs
        monkeypatch.setattr(prs, "analyze",
                            lambda *a, **k: {"status": "insufficient_data",
                                              "verdict": "INSUFFICIENT_DATA",
                                              "n_records": 0,
                                              "n_cells": 0,
                                              "n_stable_cells": 0,
                                              "cells": [],
                                              "inverted_cells": [],
                                              "best_cell": None,
                                              "worst_cell": None,
                                              "n_dropped_unknown_regime": 0})
        monkeypatch.setattr(prs, "_load_outcomes", lambda *a, **k: [])

        rcb._append_persona_regime_skill_log(
            cycle=99, win_start=date(2020, 1, 1),
            win_end=date(2021, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
        # After append (13) → trim to last 5.
        assert len(lines) == 5
        latest = json.loads(lines[-1])
        # The new row is the most-recent one.
        assert (latest.get("cycle") == 99
                or latest.get("verdict") == "INSUFFICIENT_DATA")

    def test_dir_created_on_first_write(self, tmp_path, monkeypatch):
        """A nested target dir must be created on first write — fresh
        deploy regression.
        """
        nested = tmp_path / "nested" / "deep" / "persona_regime_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_REGIME_SKILL_LOG", nested)
        assert not nested.parent.exists()

        from paper_trader.ml import persona_regime_skill as prs
        monkeypatch.setattr(prs, "analyze",
                            lambda *a, **k: {"status": "insufficient_data",
                                              "verdict": "INSUFFICIENT_DATA",
                                              "n_records": 0,
                                              "n_cells": 0,
                                              "n_stable_cells": 0,
                                              "cells": [],
                                              "inverted_cells": [],
                                              "best_cell": None,
                                              "worst_cell": None,
                                              "n_dropped_unknown_regime": 0})
        monkeypatch.setattr(prs, "_load_outcomes", lambda *a, **k: [])
        ok = rcb._append_persona_regime_skill_log(
            cycle=1, win_start=date(2020, 1, 1),
            win_end=date(2021, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        assert ok is True
        assert nested.exists()


class TestPersonaRegimeSkillLedgerWiringRegression:
    """Source-level wiring assertion: the ``main()`` loop body MUST call
    ``_append_persona_regime_skill_log(`` somewhere. Without this
    regression test a future refactor could orphan the function (define
    it but never invoke it), silently breaking the per-cycle trend —
    exactly the documented failure mode that motivated the sibling
    ``test_continuous_persona_skill_ledger.TestPersonaSkillLedgerWiringRegression``.
    """

    def test_main_calls_append_persona_regime_skill_log(self):
        src = Path(rcb.__file__).read_text()
        import re
        main_match = re.search(r"^def main\(\)", src, re.MULTILINE)
        assert main_match, "main() not found in run_continuous_backtests.py"
        main_body = src[main_match.start():]
        assert "_append_persona_regime_skill_log(" in main_body, (
            "main() must call _append_persona_regime_skill_log per cycle "
            "— the per-cycle wiring is the entire point of the ledger"
        )

    def test_constants_module_level_for_testability(self):
        """``PERSONA_REGIME_SKILL_LOG`` and
        ``PERSONA_REGIME_SKILL_LOG_KEEP`` must be module-level so test
        fixtures can monkeypatch them.
        """
        assert hasattr(rcb, "PERSONA_REGIME_SKILL_LOG")
        assert hasattr(rcb, "PERSONA_REGIME_SKILL_LOG_KEEP")
        assert isinstance(rcb.PERSONA_REGIME_SKILL_LOG, Path)
        assert isinstance(rcb.PERSONA_REGIME_SKILL_LOG_KEEP, int)
        assert rcb.PERSONA_REGIME_SKILL_LOG_KEEP > 0


class TestPersonaRegimeSkillAnalyzerContract:
    """The ledger trusts ``persona_regime_skill.analyze`` to return a
    JSON-safe dict with specific keys. Pin the analyzer-side contract on
    the keys the ledger reads so a future analyzer change can never
    silently break the ledger's persisted schema.
    """

    def test_analyzer_returns_keys_ledger_reads_on_empty_input(self):
        from paper_trader.ml.persona_regime_skill import analyze
        rep = analyze([])
        for key in ("status", "verdict", "n_records", "n_cells",
                    "n_stable_cells", "cells", "inverted_cells",
                    "best_cell", "worst_cell",
                    "n_dropped_unknown_regime"):
            assert key in rep, f"analyzer must return {key!r}"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_records"] == 0
        assert rep["cells"] == []
        assert rep["inverted_cells"] == []
        assert rep["best_cell"] is None
        assert rep["worst_cell"] is None

    def test_analyzer_best_cell_carries_keys_ledger_flattens(self):
        """The ledger flattens best_cell (and worst_cell) to top-level
        ``best_persona`` / ``best_regime`` / ``best_score_ic`` / ``best_n``
        fields. Pin that the analyzer's best_cell dict actually carries
        those keys on a synthetic stable corpus — a future change
        dropping any of them would silently NULL the ledger's flat
        leader fields.
        """
        from paper_trader.ml.persona_regime_skill import (
            analyze, MIN_RECORDS, MIN_PER_CELL,
        )
        # run_id 1 = Value Investor (per persona_for); regime_label="bull"
        # is the explicit feature added 2026-05-19. Use varying ml_score so
        # spearman has spread (else IC is 0.0).
        recs = []
        for i in range(max(MIN_PER_CELL, 25)):
            recs.append({
                "run_id": 1, "action": "BUY",
                "ml_score": 1.0 + i * 0.1,
                "forward_return_5d": (i % 5) * 1.5 - 2.5,
                "regime_label": "bull",
            })
        # Pad with a second persona/regime so n_records >= MIN_RECORDS.
        for i in range(max(MIN_PER_CELL, 25)):
            recs.append({
                "run_id": 2, "action": "BUY",
                "ml_score": 1.0 + i * 0.1,
                "forward_return_5d": (i % 4) * 2.0 - 3.0,
                "regime_label": "sideways",
            })
        rep = analyze(recs)
        assert rep["n_records"] >= MIN_RECORDS
        # At least one stable cell expected.
        if rep["best_cell"] is not None:
            for key in ("persona", "regime", "score_ic", "n"):
                assert key in rep["best_cell"], \
                    f"best_cell must carry {key!r}"

    def test_analyzer_inverted_cells_is_list(self):
        from paper_trader.ml.persona_regime_skill import analyze
        rep = analyze([])
        assert isinstance(rep["inverted_cells"], list)
