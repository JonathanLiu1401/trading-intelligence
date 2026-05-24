"""Per-cycle ledger tests for the per-persona decision-signal skill analyzer.

``paper_trader/ml/persona_skill.py`` already CLI-reports each persona's
decision-level rank skill (the action-aligned Spearman between ``ml_score``
and ``forward_return_5d``) so a skeptical quant can tell whether a
persona's per-run vs_spy_pct dispersion is real signal skill or pure
leveraged-beta luck. The single most-decisive state is
``HAS_INVERTED_PERSONA``: at least one persona whose signal is
ANTI-predictive — the actionable red flag for pruning the
``_PERSONA_BOOSTS`` entry. Until ``_append_persona_skill_log`` landed,
that state was visible only via manual CLI invocation; an unattended
operator could not trend it. These tests pin the wiring: best-effort
discipline, honest gap rows when ``signal_dark``, bounded trim, SSOT
cross-check that the persisted verdict equals the analyzer's. Mirrors
the discipline of every sibling ``_append_*_skill_log`` test in
``test_continuous.py`` / ``test_continuous_gate_arm_ledger.py``.

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


class TestAppendPersonaSkillLog:
    """The per-cycle per-persona decision-signal skill ledger — answers
    "within each persona's own decisions, does the signal rank-predict
    realized outcomes, or is the return pure leveraged-beta noise?"
    durably and per-cycle so an unattended operator can trend per-persona
    signal health and catch INVERTED personas the moment they emerge.

    Mirrors the sibling ``_append_gate_arm_skill_log`` /
    ``_append_stop_out_skill_log`` discipline: best-effort, honest gap
    rows, atomic bounded trim, SSOT no-drift, never breaks the loop.
    """

    def test_insufficient_data_persists_dark_row(self, tmp_path,
                                                  monkeypatch):
        """When ``persona_skill`` returns INSUFFICIENT_DATA (corpus has
        fewer than MIN_RECORDS aligned outcomes), the ledger MUST still
        append the row with ``signal_dark=True`` so the darkness is
        visible in the trend rather than silent. Mirrors the
        ``gate_dark`` / ``stop_dark`` / ``tp_dark`` precedent.
        """
        log = tmp_path / "persona_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_SKILL_LOG", log)

        from paper_trader.ml import persona_skill as ps
        empty = {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n_records": 5,
            "n_personas": 0,
            "personas": [],
            "inverted_personas": [],
            "hint": "need ≥30 aligned outcomes, have 5",
        }
        monkeypatch.setattr(ps, "persona_skill", lambda *a, **k: dict(empty))
        # _load_outcomes is also called inside the ledger function — stub it
        # to return an empty list so the analyzer stub is what we exercise.
        monkeypatch.setattr(ps, "_load_outcomes", lambda *a, **k: [])

        assert rcb._append_persona_skill_log(
            cycle=33, win_start=date(2020, 1, 2),
            win_end=date(2025, 1, 2),
            outcomes_path=tmp_path / "nope.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["cycle"] == 33
        assert row["verdict"] == "INSUFFICIENT_DATA"
        # ``signal_dark`` mirrors sibling ``*_dark`` booleans.
        assert row["signal_dark"] is True
        assert row["n_records"] == 5
        assert row["n_personas"] == 0
        assert row["top_persona"] is None
        assert row["top_score_ic"] is None
        assert row["n_inverted"] == 0
        assert row["inverted_personas"] == []
        assert row["personas"] == []
        assert row["window_start"] == "2020-01-02"
        assert row["window_end"] == "2025-01-02"

    def test_has_inverted_persona_surfaces_red_flag(self, tmp_path,
                                                     monkeypatch):
        """SSOT cross-check: the analyzer's ``HAS_INVERTED_PERSONA``
        verdict — the single most actionable state because the
        anti-predictive persona is actively HURTING when it sizes up —
        must round-trip unchanged through the ledger, including the
        ``inverted_personas`` list AND ``n_inverted`` count. A drift here
        would silently break the operator's only durable trend on this
        red-flag state.
        """
        log = tmp_path / "persona_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_SKILL_LOG", log)

        inverted_state = {
            "status": "ok",
            "verdict": "HAS_INVERTED_PERSONA",
            "n_records": 1200,
            "n_personas": 10,
            "personas": [
                {"persona": "Momentum Trader", "n": 130, "score_ic": 0.22,
                 "mean_aligned_return": 1.5, "win_rate": 0.55,
                 "mean_signal": 1.4, "std_return": 8.0,
                 "verdict": "SIGNAL_EDGE"},
                {"persona": "Quant / Event-Driven", "n": 124,
                 "score_ic": 0.10, "mean_aligned_return": 0.5,
                 "win_rate": 0.52, "mean_signal": 1.2, "std_return": 7.5,
                 "verdict": "WEAK_SIGNAL_EDGE"},
                {"persona": "Contrarian", "n": 110, "score_ic": 0.02,
                 "mean_aligned_return": -0.1, "win_rate": 0.49,
                 "mean_signal": 1.0, "std_return": 7.0,
                 "verdict": "NO_SIGNAL_EDGE"},
                # Inverted — the actionable red flag.
                {"persona": "Pure Speculator", "n": 95, "score_ic": -0.18,
                 "mean_aligned_return": -2.0, "win_rate": 0.42,
                 "mean_signal": 1.8, "std_return": 10.0,
                 "verdict": "INVERTED_SIGNAL"},
            ],
            "inverted_personas": ["Pure Speculator"],
            "hint": "1 persona(s) have an anti-predictive signal …",
        }
        from paper_trader.ml import persona_skill as ps
        monkeypatch.setattr(ps, "persona_skill",
                            lambda *a, **k: dict(inverted_state))
        monkeypatch.setattr(ps, "_load_outcomes", lambda *a, **k: [{}])

        rcb._append_persona_skill_log(
            cycle=44, win_start=date(2018, 6, 1),
            win_end=date(2023, 6, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        # Verdict round-trips unchanged.
        assert row["verdict"] == "HAS_INVERTED_PERSONA"
        # signal_dark FALSE because n_personas > 0.
        assert row["signal_dark"] is False
        # Inverted list + count both surfaced as actionable flat fields.
        assert row["inverted_personas"] == ["Pure Speculator"]
        assert row["n_inverted"] == 1
        # Top persona is the leader by score_ic (Momentum @ +0.22),
        # NOT the inverted persona.
        assert row["top_persona"] == "Momentum Trader"
        assert row["top_score_ic"] == 0.22
        assert row["n_records"] == 1200
        assert row["n_personas"] == 10
        # Full personas list ships intact for forensics.
        assert len(row["personas"]) == 4

    def test_healthy_verdict_with_top_signal_edge(self, tmp_path,
                                                    monkeypatch):
        """When at least one persona reaches SIGNAL_EDGE and NO persona
        is anti-predictive (the documented HEALTHY state), the ledger
        must capture ``top_persona`` + ``top_score_ic`` as the
        operator-readable leader fields.
        """
        log = tmp_path / "persona_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_SKILL_LOG", log)

        healthy = {
            "status": "ok",
            "verdict": "HEALTHY",
            "n_records": 1500,
            "n_personas": 10,
            "personas": [
                {"persona": "Value Investor", "n": 150,
                 "score_ic": 0.31, "mean_aligned_return": 2.5,
                 "win_rate": 0.58, "mean_signal": 1.5,
                 "std_return": 6.5, "verdict": "SIGNAL_EDGE"},
                {"persona": "GARP", "n": 145,
                 "score_ic": 0.18, "mean_aligned_return": 1.8,
                 "win_rate": 0.54, "mean_signal": 1.3,
                 "std_return": 7.0, "verdict": "SIGNAL_EDGE"},
            ],
            "inverted_personas": [],
        }
        from paper_trader.ml import persona_skill as ps
        monkeypatch.setattr(ps, "persona_skill",
                            lambda *a, **k: dict(healthy))
        monkeypatch.setattr(ps, "_load_outcomes", lambda *a, **k: [{}])

        rcb._append_persona_skill_log(
            cycle=55, win_start=date(2018, 6, 1),
            win_end=date(2023, 6, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "HEALTHY"
        assert row["signal_dark"] is False
        # Top is the highest-IC persona, which the analyzer sorts to index 0.
        assert row["top_persona"] == "Value Investor"
        assert row["top_score_ic"] == 0.31
        assert row["n_inverted"] == 0

    def test_top_persona_skips_insufficient_entries(self, tmp_path,
                                                      monkeypatch):
        """When the top entry by analyzer-sort is INSUFFICIENT (small n,
        unstable IC), the ledger must skip past it to the first
        verdict-stable persona — otherwise an unstable IC value
        masquerades as the cycle's leader.
        """
        log = tmp_path / "persona_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_SKILL_LOG", log)

        mixed = {
            "status": "ok",
            "verdict": "HEALTHY",
            "n_records": 300,
            "n_personas": 3,
            # The analyzer sorts INSUFFICIENT entries last, so this list
            # is realistic: stable personas first, INSUFFICIENT at the
            # tail (the implementation order is asserted by persona_skill
            # itself). The ledger MUST pick the first non-INSUFFICIENT.
            "personas": [
                {"persona": "GARP", "n": 120, "score_ic": 0.21,
                 "mean_aligned_return": 1.9, "win_rate": 0.55,
                 "mean_signal": 1.3, "std_return": 6.5,
                 "verdict": "SIGNAL_EDGE"},
                {"persona": "Momentum Trader", "n": 100,
                 "score_ic": 0.05, "mean_aligned_return": 0.4,
                 "win_rate": 0.51, "mean_signal": 1.1, "std_return": 7.0,
                 "verdict": "WEAK_SIGNAL_EDGE"},
                {"persona": "Pure Speculator", "n": 10,
                 "score_ic": 0.99, "mean_aligned_return": 5.0,
                 "win_rate": 0.7, "mean_signal": 2.0, "std_return": 9.0,
                 "verdict": "INSUFFICIENT"},
            ],
            "inverted_personas": [],
        }
        from paper_trader.ml import persona_skill as ps
        monkeypatch.setattr(ps, "persona_skill",
                            lambda *a, **k: dict(mixed))
        monkeypatch.setattr(ps, "_load_outcomes", lambda *a, **k: [{}])

        rcb._append_persona_skill_log(
            cycle=60, win_start=date(2020, 1, 1),
            win_end=date(2021, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        # The INSUFFICIENT Pure Speculator's +0.99 IC must NOT win — it
        # is unstable. The first stable entry (GARP, IC +0.21) wins.
        assert row["top_persona"] == "GARP"
        assert row["top_score_ic"] == 0.21

    def test_analyze_exception_degrades_to_honest_row_not_crash(
            self, tmp_path, monkeypatch):
        """The ``_append_persona_skill_log`` discipline: a ledger write
        must NEVER break the continuous loop. If ``persona_skill`` raises
        (e.g. a malformed outcomes file the analyzer fails to load), the
        ledger must still persist an honest INSUFFICIENT_DATA row with
        ``signal_dark=True`` instead of bubbling.
        """
        log = tmp_path / "persona_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_SKILL_LOG", log)

        def _boom(*a, **k):
            raise RuntimeError("simulated persona_skill failure")

        from paper_trader.ml import persona_skill as ps
        monkeypatch.setattr(ps, "persona_skill", _boom)
        monkeypatch.setattr(ps, "_load_outcomes", lambda *a, **k: [])

        assert rcb._append_persona_skill_log(
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
        """Defense-in-depth: if a future ``persona_skill`` mutation
        returned something other than a dict (e.g. None on an early-exit
        path), the ledger must NOT raise — it must persist the honest
        gap row, the same way every sibling ``_append_*_skill_log``
        normalizes a non-dict to ``INSUFFICIENT_DATA``.
        """
        log = tmp_path / "persona_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_SKILL_LOG", log)

        from paper_trader.ml import persona_skill as ps
        monkeypatch.setattr(ps, "persona_skill", lambda *a, **k: None)
        monkeypatch.setattr(ps, "_load_outcomes", lambda *a, **k: [])

        assert rcb._append_persona_skill_log(
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
        Mirrors the sibling ``GATE_ARM_SKILL_LOG`` trim idiom exactly:
        pay the rewrite only when well past the cap, tmp + ``.replace``
        so a torn truncate cannot lose history.
        """
        log = tmp_path / "persona_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_SKILL_LOG", log)
        monkeypatch.setattr(rcb, "PERSONA_SKILL_LOG_KEEP", 5)

        # Seed with 12 lines — past 2× the cap of 5 = 10.
        log.write_text("\n".join(
            json.dumps({"seed": i}) for i in range(12)) + "\n")
        from paper_trader.ml import persona_skill as ps
        monkeypatch.setattr(ps, "persona_skill",
                            lambda *a, **k: {"status": "insufficient_data",
                                              "verdict": "INSUFFICIENT_DATA",
                                              "n_records": 0,
                                              "n_personas": 0,
                                              "personas": [],
                                              "inverted_personas": []})
        monkeypatch.setattr(ps, "_load_outcomes", lambda *a, **k: [])

        rcb._append_persona_skill_log(
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
        nested = tmp_path / "nested" / "deep" / "persona_skill_log.jsonl"
        monkeypatch.setattr(rcb, "PERSONA_SKILL_LOG", nested)
        assert not nested.parent.exists()

        from paper_trader.ml import persona_skill as ps
        monkeypatch.setattr(ps, "persona_skill",
                            lambda *a, **k: {"status": "insufficient_data",
                                              "verdict": "INSUFFICIENT_DATA",
                                              "n_records": 0,
                                              "n_personas": 0,
                                              "personas": [],
                                              "inverted_personas": []})
        monkeypatch.setattr(ps, "_load_outcomes", lambda *a, **k: [])
        ok = rcb._append_persona_skill_log(
            cycle=1, win_start=date(2020, 1, 1),
            win_end=date(2021, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        assert ok is True
        assert nested.exists()


class TestPersonaSkillLedgerWiringRegression:
    """Source-level wiring assertion: the ``main()`` loop body MUST call
    ``_append_persona_skill_log(`` somewhere. Without this regression
    test a future refactor could orphan the function (define it but
    never invoke it), silently breaking the per-cycle trend — exactly
    the documented failure mode that motivated the sibling
    ``test_continuous_gate_arm_ledger.TestGateArmLedgerWiringRegression``.
    """

    def test_main_calls_append_persona_skill_log(self):
        src = Path(rcb.__file__).read_text()
        import re
        main_match = re.search(r"^def main\(\)", src, re.MULTILINE)
        assert main_match, "main() not found in run_continuous_backtests.py"
        main_body = src[main_match.start():]
        assert "_append_persona_skill_log(" in main_body, (
            "main() must call _append_persona_skill_log per cycle — "
            "the per-cycle wiring is the entire point of the ledger"
        )

    def test_constants_module_level_for_testability(self):
        """``PERSONA_SKILL_LOG`` and ``PERSONA_SKILL_LOG_KEEP`` must be
        module-level so test fixtures can monkeypatch them.
        """
        assert hasattr(rcb, "PERSONA_SKILL_LOG")
        assert hasattr(rcb, "PERSONA_SKILL_LOG_KEEP")
        assert isinstance(rcb.PERSONA_SKILL_LOG, Path)
        assert isinstance(rcb.PERSONA_SKILL_LOG_KEEP, int)
        assert rcb.PERSONA_SKILL_LOG_KEEP > 0


class TestPersonaSkillAnalyzerContract:
    """The ledger trusts ``persona_skill.persona_skill`` to return a
    JSON-safe dict with specific keys. Pin the analyzer-side contract
    on the keys the ledger reads so a future analyzer change can never
    silently break the ledger's persisted schema.
    """

    def test_analyzer_returns_keys_ledger_reads_on_empty_input(self):
        from paper_trader.ml.persona_skill import persona_skill
        rep = persona_skill([])
        for key in ("status", "verdict", "n_records", "n_personas",
                    "personas", "inverted_personas"):
            assert key in rep, f"analyzer must return {key!r}"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_records"] == 0
        assert rep["n_personas"] == 0
        assert rep["personas"] == []
        assert rep["inverted_personas"] == []

    def test_analyzer_per_persona_row_carries_keys_ledger_reads(self):
        """Each entry in ``personas`` must carry ``persona``, ``n``,
        ``score_ic``, ``verdict`` — the keys the ledger picks the top
        leader from. A future change dropping any of these would silently
        break top-persona extraction.
        """
        from paper_trader.ml.persona_skill import (
            persona_skill, MIN_RECORDS, MIN_OUTCOMES_PER_PERSONA,
        )
        # Build a minimal synthetic corpus that produces a populated
        # ``personas`` list — at least MIN_OUTCOMES_PER_PERSONA rows per
        # persona and >= MIN_RECORDS total. run_id 1 = Value Investor
        # (per ``persona_for``). Use varying ml_score so spearman has
        # spread (else the analyzer returns 0.0 IC with no information).
        recs = []
        # run_id=1 → Value Investor, run_id=2 → Momentum
        for i in range(max(MIN_OUTCOMES_PER_PERSONA, 25)):
            recs.append({
                "run_id": 1, "action": "BUY",
                "ml_score": 1.0 + i * 0.1,
                "forward_return_5d": (i % 5) * 1.5 - 2.5,
            })
            recs.append({
                "run_id": 2, "action": "BUY",
                "ml_score": 1.0 + i * 0.1,
                "forward_return_5d": -((i % 5) * 1.5 - 2.5),
            })
        rep = persona_skill(recs)
        assert rep["n_records"] >= MIN_RECORDS
        assert rep["n_personas"] >= 1
        assert rep["personas"], "expected at least one persona entry"
        for entry in rep["personas"]:
            for key in ("persona", "n", "score_ic", "verdict"):
                assert key in entry, f"persona entry must carry {key!r}"
            assert isinstance(entry["persona"], str)
            assert isinstance(entry["n"], int)
            assert isinstance(entry["score_ic"], (int, float))
            assert entry["verdict"] in (
                "INSUFFICIENT", "INVERTED_SIGNAL", "SIGNAL_EDGE",
                "WEAK_SIGNAL_EDGE", "NO_SIGNAL_EDGE",
            )

    def test_analyzer_inverted_personas_is_list(self):
        from paper_trader.ml.persona_skill import persona_skill
        rep = persona_skill([])
        assert isinstance(rep["inverted_personas"], list)
