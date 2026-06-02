"""Per-cycle ledger tests for the outcome-corpus-health analyzer.

``paper_trader/ml/outcome_corpus_health.py`` (pass #2 / 2026-05-29) already
CLI-reports the *pre-model* data-quality verdict every downstream model
verdict implicitly assumes: *what is the training corpus actually made
of?* The verdict ladder is ``INSUFFICIENT_DATA`` / ``NEWS_FEATURES_DARK``
/ ``ACTION_IMBALANCED`` / ``REGIME_BUCKETS_SPARSE`` / ``TARGET_DEGENERATE``
/ ``HEALTHY``. The live production state — ``MLP_WORSE_THAN_TRIVIAL``
co-existing with ``news_urgency`` populated in only 1% of training rows
— means a reading quant interpreting the headline scorer/baseline ledger
without the corpus state is reading the model as "weak" when the actual
culprit is data starvation. Until ``_append_outcome_corpus_health_log``
landed, that pre-model state was visible only via manual CLI invocation;
an unattended operator could not trend per-cycle data quality.

These tests pin the wiring: best-effort discipline, honest gap rows when
``corpus_dark``, bounded trim, SSOT cross-check that the persisted
verdict equals the analyzer's. Mirrors the discipline of every sibling
``_append_*_skill_log`` test (sector / persona / persona×regime /
gate-arm / stop-out / mfe / etc.).

A new test file (not appended to ``test_continuous.py``) so a concurrent
sibling agent editing the same test file cannot collide with this work
via whole-file ``git add`` — the documented same-role HYBRID staging-
race mitigation pattern.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

import run_continuous_backtests as rcb


class TestAppendOutcomeCorpusHealthLog:
    """The per-cycle outcome-corpus-health ledger — answers "what is the
    training corpus actually made of?" durably and per-cycle so an
    unattended operator can read the data state next to the model state
    and tell whether ``MLP_WORSE_THAN_TRIVIAL`` reflects model weakness
    or data starvation.

    Mirrors the sibling ``_append_sector_skill_log`` /
    ``_append_persona_skill_log`` discipline: best-effort, honest gap
    rows, atomic bounded trim, SSOT no-drift, never breaks the loop.
    """

    def test_insufficient_data_persists_corpus_dark_row(
        self, tmp_path, monkeypatch
    ):
        """When ``outcome_corpus_health.analyze`` returns
        INSUFFICIENT_DATA (corpus < MIN_RECORDS), the ledger MUST still
        append the row with ``corpus_dark=True`` so the darkness is
        visible in the trend rather than silent. Mirrors the
        ``signal_dark`` / ``stop_dark`` / ``tp_dark`` precedent.
        """
        log = tmp_path / "outcome_corpus_health_log.jsonl"
        monkeypatch.setattr(rcb, "OUTCOME_CORPUS_HEALTH_LOG", log)

        from paper_trader.ml import outcome_corpus_health as och
        empty = {
            "status": "ok",
            "verdict": "INSUFFICIENT_DATA",
            "n_total": 25,
            "n_buys": 0, "n_sells": 0, "n_other_action": 0,
            "n_with_news": 0, "fraction_with_news": 0.0,
            "regime_counts": {}, "sector_counts": {},
            "persona_counts": {}, "feature_density": {},
            "target": {"n": 0, "mean": None, "std": None,
                       "min": None, "max": None,
                       "p5": None, "p95": None},
            "date_range_start": None, "date_range_end": None,
            "hints": ["corpus has 25 rows, need ≥ 100 for any verdict "
                      "beyond INSUFFICIENT_DATA"],
        }
        monkeypatch.setattr(och, "analyze", lambda **kw: dict(empty))

        assert rcb._append_outcome_corpus_health_log(
            cycle=33, win_start=date(2020, 1, 2),
            win_end=date(2025, 1, 2),
            outcomes_path=tmp_path / "nope.jsonl",
        ) is True
        row = json.loads(log.read_text().strip())
        assert row["cycle"] == 33
        assert row["verdict"] == "INSUFFICIENT_DATA"
        # ``corpus_dark`` mirrors sibling ``*_dark`` booleans.
        assert row["corpus_dark"] is True
        assert row["n_total"] == 25
        assert row["n_with_news"] == 0
        assert row["fraction_with_news"] == 0.0
        assert row["top_sector"] is None
        assert row["top_sector_share"] is None
        assert row["dominant_action"] is None
        assert row["dominant_action_share"] is None
        assert row["n_hints"] == 1
        assert row["window_start"] == "2020-01-02"
        assert row["window_end"] == "2025-01-02"

    def test_news_features_dark_round_trips_unchanged(
        self, tmp_path, monkeypatch
    ):
        """SSOT cross-check: the analyzer's ``NEWS_FEATURES_DARK``
        verdict — the single most directly operational pre-model state
        because the gate-feeding news features are effectively absent —
        must round-trip unchanged through the ledger, including the
        ``n_with_news`` / ``fraction_with_news`` flat fields AND the
        ``hints`` list. A drift here would silently break the
        operator's only durable trend on this red-flag state.
        """
        log = tmp_path / "outcome_corpus_health_log.jsonl"
        monkeypatch.setattr(rcb, "OUTCOME_CORPUS_HEALTH_LOG", log)

        dark_state = {
            "status": "ok",
            "verdict": "NEWS_FEATURES_DARK",
            "n_total": 6581,
            "n_buys": 4700, "n_sells": 1881, "n_other_action": 0,
            "n_with_news": 65, "fraction_with_news": 0.0099,
            "regime_counts": {"bull": 4200, "sideways": 2300,
                              "bear": 81},
            "sector_counts": {"tech": 5300, "crypto": 700,
                              "financials": 300, "other": 281},
            "persona_counts": {"Value Investor": 660,
                               "Momentum Trader": 660,
                               "Contrarian": 660},
            "feature_density": {
                "rsi": {"n_non_null": 6500, "fraction": 0.9877},
                "news_urgency": {"n_non_null": 65, "fraction": 0.0099},
            },
            "target": {"n": 6581, "mean": 0.85, "std": 6.21,
                       "min": -49.5, "max": 49.8,
                       "p5": -9.2, "p95": 11.0},
            "date_range_start": "2010-04-01",
            "date_range_end": "2024-11-15",
            "hints": [
                "news features in only 65/6581 (1.0%) — below 10% threshold",
            ],
        }
        from paper_trader.ml import outcome_corpus_health as och
        monkeypatch.setattr(och, "analyze", lambda **kw: dict(dark_state))

        assert rcb._append_outcome_corpus_health_log(
            cycle=11, win_start=date(2010, 1, 1),
            win_end=date(2020, 1, 1),
            outcomes_path=tmp_path / "x.jsonl",
        ) is True
        row = json.loads(log.read_text().strip())
        # SSOT cross-check: verdict + all the flat top-line fields a
        # quant queries.
        assert row["verdict"] == "NEWS_FEATURES_DARK"
        assert row["n_total"] == 6581
        assert row["n_buys"] == 4700
        assert row["n_sells"] == 1881
        assert row["n_with_news"] == 65
        assert row["fraction_with_news"] == 0.0099
        # ``corpus_dark`` is False — analyzer produced a real verdict
        # (NEWS_FEATURES_DARK is a "we can read this" verdict, not the
        # INSUFFICIENT_DATA / analyzer-failed bucket).
        assert row["corpus_dark"] is False
        assert row["top_sector"] == "tech"
        # tech 5300 / 6581 ≈ 0.8053
        assert row["top_sector_share"] == round(5300 / 6581, 4)
        assert row["dominant_action"] == "BUY"
        assert row["dominant_action_share"] == round(4700 / 6581, 4)
        assert row["n_hints"] == 1
        assert row["target_mean"] == 0.85
        assert row["target_std"] == 6.21
        assert row["date_range_start"] == "2010-04-01"
        assert row["date_range_end"] == "2024-11-15"
        # Forensic-bin payloads still ship.
        assert row["sector_counts"]["tech"] == 5300
        assert row["regime_counts"]["bear"] == 81
        assert row["hints"] == dark_state["hints"]
        # Nested ``target`` block preserved for forensics.
        assert row["target"]["p5"] == -9.2
        assert row["target"]["p95"] == 11.0

    def test_healthy_corpus_clears_corpus_dark(self, tmp_path, monkeypatch):
        """The HEALTHY verdict must surface as ``corpus_dark=False`` and
        ``n_hints=0`` — the all-clear signal a quant trends to confirm
        the corpus actually carries learnable signal.
        """
        log = tmp_path / "outcome_corpus_health_log.jsonl"
        monkeypatch.setattr(rcb, "OUTCOME_CORPUS_HEALTH_LOG", log)

        healthy = {
            "status": "ok", "verdict": "HEALTHY",
            "n_total": 4200,
            "n_buys": 2800, "n_sells": 1400, "n_other_action": 0,
            "n_with_news": 1200, "fraction_with_news": 0.2857,
            "regime_counts": {"bull": 2400, "sideways": 1500, "bear": 300},
            "sector_counts": {"tech": 2300, "financials": 900,
                              "healthcare": 600, "crypto": 400},
            "persona_counts": {},
            "feature_density": {},
            "target": {"n": 4200, "mean": 0.9, "std": 7.1,
                       "min": -42.0, "max": 45.0,
                       "p5": -10.5, "p95": 12.0},
            "date_range_start": "2015-01-01",
            "date_range_end": "2024-01-01",
            "hints": [],
        }
        from paper_trader.ml import outcome_corpus_health as och
        monkeypatch.setattr(och, "analyze", lambda **kw: dict(healthy))

        assert rcb._append_outcome_corpus_health_log(
            cycle=7, win_start=date(2015, 1, 1),
            win_end=date(2024, 1, 1),
            outcomes_path=tmp_path / "x.jsonl",
        ) is True
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "HEALTHY"
        assert row["corpus_dark"] is False
        assert row["n_hints"] == 0
        assert row["hints"] == []
        # Top sector = tech (2300 / 4200 ≈ 0.5476)
        assert row["top_sector"] == "tech"
        assert row["top_sector_share"] == round(2300 / 4200, 4)

    def test_analyzer_crash_falls_through_to_honest_gap(
        self, tmp_path, monkeypatch
    ):
        """When the analyzer itself raises (corrupt JSONL / unexpected
        path / etc.), the ledger MUST still emit a row marked
        ``corpus_dark=True`` so the gap is visible in the trend. The
        loop's "never break on a ledger fault" discipline.
        """
        log = tmp_path / "outcome_corpus_health_log.jsonl"
        monkeypatch.setattr(rcb, "OUTCOME_CORPUS_HEALTH_LOG", log)

        from paper_trader.ml import outcome_corpus_health as och

        def _boom(**kw):
            raise RuntimeError("synthetic analyzer failure")

        monkeypatch.setattr(och, "analyze", _boom)

        assert rcb._append_outcome_corpus_health_log(
            cycle=99, win_start=date(2021, 1, 1),
            win_end=date(2022, 1, 1),
            outcomes_path=tmp_path / "nope.jsonl",
        ) is True
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["corpus_dark"] is True
        assert row["status"] == "error"
        # The synthetic failure surface in hints so a reading quant can
        # tell WHY the row is dark.
        assert any("synthetic" in h.lower() or "runtimeerror" in h.lower()
                   or "outcome_corpus_health unavailable" in h
                   for h in row.get("hints") or [])

    def test_dominant_action_picks_majority(self, tmp_path, monkeypatch):
        """``dominant_action`` must select the larger of (n_buys, n_sells)
        and report the share against (n_buys + n_sells). When SELL
        outnumbers BUY (an unusual but possible regime — heavily-shorting
        cycle), the ledger must surface SELL, not silently default to BUY.
        """
        log = tmp_path / "outcome_corpus_health_log.jsonl"
        monkeypatch.setattr(rcb, "OUTCOME_CORPUS_HEALTH_LOG", log)

        sell_heavy = {
            "status": "ok", "verdict": "ACTION_IMBALANCED",
            "n_total": 1000,
            "n_buys": 100, "n_sells": 900, "n_other_action": 0,
            "n_with_news": 200, "fraction_with_news": 0.2,
            "regime_counts": {"bear": 800, "sideways": 200},
            "sector_counts": {"tech": 600, "financials": 400},
            "persona_counts": {}, "feature_density": {},
            "target": {"n": 1000, "mean": -1.2, "std": 8.0,
                       "min": -40, "max": 35, "p5": -12, "p95": 10},
            "date_range_start": "2008-09-01",
            "date_range_end": "2009-03-01",
            "hints": ["SELL 900/1000 (90.0%) — exceeds 85% imbalance "
                      "threshold; minority-action skill is data-limited"],
        }
        from paper_trader.ml import outcome_corpus_health as och
        monkeypatch.setattr(och, "analyze", lambda **kw: dict(sell_heavy))

        assert rcb._append_outcome_corpus_health_log(
            cycle=8, win_start=date(2008, 9, 1),
            win_end=date(2009, 3, 1),
            outcomes_path=tmp_path / "x.jsonl",
        ) is True
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "ACTION_IMBALANCED"
        assert row["dominant_action"] == "SELL"
        assert row["dominant_action_share"] == 0.9
        # ``corpus_dark`` is False — analyzer produced a verdict.
        assert row["corpus_dark"] is False

    def test_atomic_trim_at_2x_keep_caps_growth(self, tmp_path, monkeypatch):
        """When the ledger exceeds 2× the keep cap, the rewrite must
        atomically truncate to the most-recent ``KEEP`` lines via
        tmp+``.replace`` — same idiom as every sibling ledger trim.
        Without bounded trim the file grows unbounded across cycles.
        """
        log = tmp_path / "outcome_corpus_health_log.jsonl"
        monkeypatch.setattr(rcb, "OUTCOME_CORPUS_HEALTH_LOG", log)
        # Low cap so the test exercises trim quickly.
        monkeypatch.setattr(rcb, "OUTCOME_CORPUS_HEALTH_LOG_KEEP", 3)

        from paper_trader.ml import outcome_corpus_health as och
        healthy = {
            "status": "ok", "verdict": "HEALTHY",
            "n_total": 1000, "n_buys": 600, "n_sells": 400,
            "n_other_action": 0, "n_with_news": 200,
            "fraction_with_news": 0.2,
            "regime_counts": {"bull": 700, "sideways": 200, "bear": 100},
            "sector_counts": {"tech": 800, "financials": 200},
            "persona_counts": {}, "feature_density": {},
            "target": {"n": 1000, "mean": 0.5, "std": 6.0,
                       "min": -30, "max": 30, "p5": -10, "p95": 9},
            "date_range_start": "2015-01-01",
            "date_range_end": "2024-01-01",
            "hints": [],
        }
        monkeypatch.setattr(och, "analyze", lambda **kw: dict(healthy))

        # Write 7 rows so we exceed 2× the cap of 3 — trim fires once.
        for c in range(1, 8):
            assert rcb._append_outcome_corpus_health_log(
                cycle=c, win_start=date(2020, 1, 1),
                win_end=date(2024, 1, 1),
                outcomes_path=tmp_path / "x.jsonl",
            ) is True

        lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
        # After trim we keep exactly KEEP=3 lines (the most recent).
        assert len(lines) == 3
        cycles = [json.loads(ln)["cycle"] for ln in lines]
        # Most-recent 3 cycles preserved (5, 6, 7).
        assert cycles == [5, 6, 7]

    def test_module_level_constants_are_module_globals(self):
        """The path + keep constants MUST be module-level (not function
        locals) so tests can ``monkeypatch.setattr(rcb, …)`` to redirect
        the file under test — the documented "hardcoded paths must be
        module-level for testability" rule every sibling skill log
        follows. A regression to a local constant would silently break
        every test in this file and the ledger would write to a
        production path during test runs.
        """
        assert hasattr(rcb, "OUTCOME_CORPUS_HEALTH_LOG")
        assert hasattr(rcb, "OUTCOME_CORPUS_HEALTH_LOG_KEEP")
        # The default path must live under ``data/`` so it co-locates
        # with every sibling ledger.
        assert str(rcb.OUTCOME_CORPUS_HEALTH_LOG).endswith(
            "data/outcome_corpus_health_log.jsonl")
        assert rcb.OUTCOME_CORPUS_HEALTH_LOG_KEEP == 2000
