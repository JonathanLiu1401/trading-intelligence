"""Per-cycle ledger tests for the per-sector OOS skill analyzer.

``paper_trader/ml/sector_skill.py`` already CLI-reports the scorer's
per-sector OOS rank-IC + verdict (HEALTHY / HAS_INVERTED_SECTOR /
SECTOR_CONCENTRATED / NO_SECTOR_EDGE / INSUFFICIENT_DATA / SCORER_UNTRAINED)
so a skeptical quant can tell whether the headline scorer rank-IC is
uniform across the universe or carried by one fat sector. The live
``decision_outcomes.jsonl`` tail is ~89% tech, so an inverted non-tech
sector (rank_ic ≤ -0.15) means gating on that sector is actively
harmful. Until ``_append_sector_skill_log`` landed, that state was
visible only via manual CLI invocation; an unattended operator could
not trend per-sector signal health.

These tests pin the wiring: best-effort discipline, honest gap rows
when ``signal_dark``, bounded trim, SSOT cross-check that the
persisted verdict equals the analyzer's. Mirrors the discipline of
every sibling ``_append_*_skill_log`` test (persona / persona×regime /
gate-arm / stop-out / mfe / etc.).

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


class TestAppendSectorSkillLog:
    """The per-cycle per-sector OOS skill ledger — answers "is the
    scorer's rank skill uniform across sectors or carried by one fat
    sector?" durably and per-cycle so an unattended operator can trend
    per-sector signal health and catch INVERTED sectors the moment
    they emerge.

    Mirrors the sibling ``_append_persona_skill_log`` /
    ``_append_persona_regime_skill_log`` discipline: best-effort,
    honest gap rows, atomic bounded trim, SSOT no-drift, never breaks
    the loop.
    """

    def test_insufficient_data_persists_dark_row(self, tmp_path,
                                                 monkeypatch):
        """When ``sector_skill.analyze`` returns INSUFFICIENT_DATA
        (corpus has fewer than MIN_RECORDS aligned outcomes), the
        ledger MUST still append the row with ``signal_dark=True`` so
        the darkness is visible in the trend rather than silent.
        Mirrors the ``signal_dark`` / ``stop_dark`` / ``tp_dark``
        precedent.
        """
        log = tmp_path / "sector_skill_log.jsonl"
        monkeypatch.setattr(rcb, "SECTOR_SKILL_LOG", log)

        from paper_trader.ml import sector_skill as ss
        empty = {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n_train": 5,
            "n_oos": 3,
            "concentrated_sector": None,
            "sectors": [],
            "inverted_sectors": [],
            "hint": "need ≥30 aligned OOS outcomes, have 3",
        }
        monkeypatch.setattr(ss, "analyze", lambda **kw: dict(empty))

        assert rcb._append_sector_skill_log(
            cycle=33, win_start=date(2020, 1, 2),
            win_end=date(2025, 1, 2),
            outcomes_path=tmp_path / "nope.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["cycle"] == 33
        assert row["verdict"] == "INSUFFICIENT_DATA"
        # ``signal_dark`` mirrors sibling ``*_dark`` booleans.
        assert row["signal_dark"] is True
        assert row["n_train"] == 5
        assert row["n_oos"] == 3
        assert row["top_sector"] is None
        assert row["top_rank_ic"] is None
        assert row["n_inverted"] == 0
        assert row["inverted_sectors"] == []
        assert row["sectors"] == []
        assert row["concentrated_sector"] is None
        assert row["window_start"] == "2020-01-02"
        assert row["window_end"] == "2025-01-02"

    def test_has_inverted_sector_surfaces_red_flag(self, tmp_path,
                                                    monkeypatch):
        """SSOT cross-check: the analyzer's ``HAS_INVERTED_SECTOR``
        verdict — the single most actionable state because the
        anti-predictive sector is actively HURTING when the gate sizes
        up on it — must round-trip unchanged through the ledger,
        including the ``inverted_sectors`` list AND ``n_inverted`` count.
        A drift here would silently break the operator's only durable
        trend on this red-flag state.
        """
        log = tmp_path / "sector_skill_log.jsonl"
        monkeypatch.setattr(rcb, "SECTOR_SKILL_LOG", log)

        inverted_state = {
            "status": "ok",
            "verdict": "HAS_INVERTED_SECTOR",
            "n_train": 4000,
            "n_oos": 1000,
            "concentrated_sector": None,
            "sectors": [
                {"sector": "tech", "n_train": 3500, "n_oos": 850,
                 "mean_pred": 4.5, "mean_realized": 1.2,
                 "magnitude_bias": 3.3, "rmse": 11.0, "dir_acc": 0.55,
                 "rank_ic": 0.22, "verdict": "SIGNAL_EDGE"},
                {"sector": "financials", "n_train": 400, "n_oos": 100,
                 "mean_pred": 2.0, "mean_realized": 0.8,
                 "magnitude_bias": 1.2, "rmse": 10.0, "dir_acc": 0.51,
                 "rank_ic": 0.08, "verdict": "WEAK_SIGNAL_EDGE"},
                # Inverted — the actionable red flag.
                {"sector": "crypto", "n_train": 100, "n_oos": 50,
                 "mean_pred": 5.0, "mean_realized": -2.0,
                 "magnitude_bias": 7.0, "rmse": 16.0, "dir_acc": 0.40,
                 "rank_ic": -0.18, "verdict": "INVERTED_SIGNAL"},
            ],
            "inverted_sectors": ["crypto"],
            "hint": "1 sector(s) have anti-predictive rank skill …",
        }
        from paper_trader.ml import sector_skill as ss
        monkeypatch.setattr(ss, "analyze",
                            lambda **kw: dict(inverted_state))

        rcb._append_sector_skill_log(
            cycle=44, win_start=date(2018, 6, 1),
            win_end=date(2023, 6, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        # Verdict round-trips unchanged (SSOT no-drift).
        assert row["verdict"] == "HAS_INVERTED_SECTOR"
        # signal_dark FALSE because at least one non-SPARSE sector
        # surfaced.
        assert row["signal_dark"] is False
        # Inverted list + count both surfaced as actionable flat fields.
        assert row["inverted_sectors"] == ["crypto"]
        assert row["n_inverted"] == 1
        # Top sector is the leader by rank_ic (tech @ +0.22), NOT the
        # inverted sector.
        assert row["top_sector"] == "tech"
        assert row["top_rank_ic"] == 0.22
        assert row["n_train"] == 4000
        assert row["n_oos"] == 1000
        # Full sectors list ships intact for forensics.
        assert len(row["sectors"]) == 3

    def test_sector_concentrated_surfaces_in_row(self, tmp_path,
                                                  monkeypatch):
        """When the analyzer flags ``SECTOR_CONCENTRATED`` (one sector
        ≥ 70% of OOS rows — the headline oos_ic is essentially that
        sector's IC), the ledger must propagate both the verdict AND
        the ``concentrated_sector`` name so a quant can decode the
        caveat directly from the trend.
        """
        log = tmp_path / "sector_skill_log.jsonl"
        monkeypatch.setattr(rcb, "SECTOR_SKILL_LOG", log)

        concentrated_state = {
            "status": "ok",
            "verdict": "SECTOR_CONCENTRATED",
            "n_train": 5000,
            "n_oos": 1000,
            "concentrated_sector": "tech",
            "sectors": [
                {"sector": "tech", "n_train": 4500, "n_oos": 900,
                 "mean_pred": 3.0, "mean_realized": 0.8,
                 "magnitude_bias": 2.2, "rmse": 11.5, "dir_acc": 0.54,
                 "rank_ic": 0.18, "verdict": "SIGNAL_EDGE"},
                {"sector": "financials", "n_train": 500, "n_oos": 100,
                 "mean_pred": 1.0, "mean_realized": 0.4,
                 "magnitude_bias": 0.6, "rmse": 9.0, "dir_acc": 0.52,
                 "rank_ic": 0.07, "verdict": "WEAK_SIGNAL_EDGE"},
            ],
            "inverted_sectors": [],
            "hint": "sector 'tech' carries 900/1000 (90%) of OOS rows",
        }
        from paper_trader.ml import sector_skill as ss
        monkeypatch.setattr(ss, "analyze",
                            lambda **kw: dict(concentrated_state))

        rcb._append_sector_skill_log(
            cycle=55, win_start=date(2017, 1, 1),
            win_end=date(2022, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "SECTOR_CONCENTRATED"
        assert row["concentrated_sector"] == "tech"
        # signal_dark FALSE — both sectors are non-SPARSE.
        assert row["signal_dark"] is False
        # Top sector is the SIGNAL_EDGE leader.
        assert row["top_sector"] == "tech"
        assert row["top_rank_ic"] == 0.18
        # Inverted count is zero in a non-inverted concentrated state.
        assert row["n_inverted"] == 0

    def test_all_sparse_marks_signal_dark(self, tmp_path, monkeypatch):
        """When every sector that surfaced is SPARSE (n_oos <
        MIN_OUTCOMES_PER_SECTOR — Spearman not stable), signal_dark
        must be True even if n_oos overall reaches MIN_RECORDS.
        Mirrors the sibling ``persona_skill`` "every persona
        INSUFFICIENT ⇒ signal_dark" precedent."""
        log = tmp_path / "sector_skill_log.jsonl"
        monkeypatch.setattr(rcb, "SECTOR_SKILL_LOG", log)

        sparse_state = {
            "status": "ok",
            "verdict": "NO_SECTOR_EDGE",
            "n_train": 200,
            "n_oos": 50,
            "concentrated_sector": None,
            "sectors": [
                {"sector": "tech", "n_train": 100, "n_oos": 18,
                 "mean_pred": 2.0, "mean_realized": 0.5,
                 "magnitude_bias": 1.5, "rmse": 10.0, "dir_acc": 0.5,
                 "rank_ic": 0.04, "verdict": "SPARSE"},
                {"sector": "financials", "n_train": 50, "n_oos": 12,
                 "mean_pred": 1.0, "mean_realized": 0.3,
                 "magnitude_bias": 0.7, "rmse": 8.0, "dir_acc": 0.5,
                 "rank_ic": 0.02, "verdict": "SPARSE"},
            ],
            "inverted_sectors": [],
            "hint": "no sector reaches MIN_OUTCOMES_PER_SECTOR=20",
        }
        from paper_trader.ml import sector_skill as ss
        monkeypatch.setattr(ss, "analyze",
                            lambda **kw: dict(sparse_state))

        rcb._append_sector_skill_log(
            cycle=66, win_start=date(2019, 1, 1),
            win_end=date(2024, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        # signal_dark TRUE because no non-SPARSE sector surfaced.
        assert row["signal_dark"] is True
        # Top sector skips SPARSE entries even though they're sorted
        # first by name → so top_sector is None when every sector is
        # SPARSE.
        assert row["top_sector"] is None
        assert row["top_rank_ic"] is None
        # Sectors list is still preserved verbatim for forensics.
        assert len(row["sectors"]) == 2

    def test_analyzer_exception_falls_through_to_honest_gap(self,
                                                              tmp_path,
                                                              monkeypatch):
        """A crash inside ``sector_skill.analyze`` must NOT propagate
        out of the ledger — it must degrade to an honest
        ``status='error' verdict='INSUFFICIENT_DATA'`` row so the gap
        is visible in the trend rather than silently missing. Same
        discipline as every sibling ``_append_*_skill_log``.
        """
        log = tmp_path / "sector_skill_log.jsonl"
        monkeypatch.setattr(rcb, "SECTOR_SKILL_LOG", log)

        def _boom(**kw):
            raise RuntimeError("synthetic analyzer crash")
        from paper_trader.ml import sector_skill as ss
        monkeypatch.setattr(ss, "analyze", _boom)

        # The function must return True (a successful append, even
        # though the analyzer failed) — the row IS the honest signal.
        assert rcb._append_sector_skill_log(
            cycle=77, win_start=date(2020, 1, 1),
            win_end=date(2025, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["status"] == "error"
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["signal_dark"] is True
        assert row["n_train"] == 0
        assert row["n_oos"] == 0
        assert row["sectors"] == []

    def test_atomic_trim_when_file_exceeds_2x_keep(self, tmp_path,
                                                    monkeypatch):
        """When the ledger file exceeds 2× ``SECTOR_SKILL_LOG_KEEP``
        lines, the next append must atomically rewrite via
        tmp+``.replace`` keeping only the most-recent ``KEEP`` lines.
        Mirrors the sibling ledgers' bounded-growth discipline.
        """
        log = tmp_path / "sector_skill_log.jsonl"
        monkeypatch.setattr(rcb, "SECTOR_SKILL_LOG", log)
        monkeypatch.setattr(rcb, "SECTOR_SKILL_LOG_KEEP", 5)

        # Seed: 12 rows (> 2 × 5 = 10) so the next append triggers a
        # trim down to the last 5 — BUT keep in mind the new append
        # writes BEFORE the trim check, so we trim to 5 from the
        # post-append 13 rows.
        log.write_text("\n".join(json.dumps({"cycle": i})
                                 for i in range(12)) + "\n")

        from paper_trader.ml import sector_skill as ss
        monkeypatch.setattr(ss, "analyze", lambda **kw: {
            "status": "ok", "verdict": "HEALTHY",
            "n_train": 1000, "n_oos": 200,
            "concentrated_sector": None,
            "sectors": [], "inverted_sectors": [],
        })

        rcb._append_sector_skill_log(
            cycle=999, win_start=date(2021, 1, 1),
            win_end=date(2022, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        # Post-trim: 5 lines kept; the newest (cycle=999) is the last.
        lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
        assert len(lines) == 5
        last = json.loads(lines[-1])
        assert last["cycle"] == 999
        assert last["verdict"] == "HEALTHY"

    def test_module_level_constants_match_pattern(self):
        """The constants must follow the documented module-level
        testability convention (the same rule every sibling
        ``*_SKILL_LOG`` / ``*_LOG_KEEP`` constant follows). Without
        this, the ``monkeypatch.setattr(rcb, "SECTOR_SKILL_LOG", ...)``
        idiom every test above relies on would silently no-op.
        """
        assert isinstance(rcb.SECTOR_SKILL_LOG, Path)
        assert str(rcb.SECTOR_SKILL_LOG).endswith("sector_skill_log.jsonl")
        # Cap value must be a positive int — same as every sibling.
        assert isinstance(rcb.SECTOR_SKILL_LOG_KEEP, int)
        assert rcb.SECTOR_SKILL_LOG_KEEP > 0
