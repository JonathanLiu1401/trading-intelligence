"""Tests for paper_trader.reporter._decision_weekday_line and its wiring
into send_hourly_summary / send_daily_close.

Mirrors the structure of test_repeat_loser_reporter.py: the helper is
silence-by-default and only surfaces ``WEEKDAY_CONCENTRATION``; the
``EVEN_DISTRIBUTION`` / ``INSUFFICIENT_DATA`` verdicts stay quiet so the
summary never grows a lying green light.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from paper_trader.reporter import (
    _decision_weekday_line,
    send_daily_close,
    send_hourly_summary,
)


def _decisions_concentrated_friday(n_fri_starved: int = 6,
                                    n_other_filled: int = 8) -> list[dict]:
    """Build ``recent_decisions`` rows so that a Friday weekday bucket has
    >=50% NO_DECISION over >=3 samples and the rest of the week is healthy.
    Fridays in May 2026: 1, 8, 15, 22, 29. Mondays: 4, 11, 18, 25.
    """
    rows: list[dict] = []
    for i in range(n_fri_starved):
        # Spread NO_DECISIONs across Fridays so multiple buckets are loaded.
        fri_day = [1, 8, 15, 22, 29][i % 5]
        rows.append({
            "timestamp": f"2026-05-{fri_day:02d}T18:00:00+00:00",  # 14:00 ET Fri
            "action_taken": "NO_DECISION",
            "reasoning": "claude returned no response (timeout)",
        })
    for i in range(n_other_filled):
        # Spread filled decisions over Mon..Thu so other days look healthy.
        day = [4, 11, 18, 25, 5, 12, 19, 26][i % 8]
        rows.append({
            "timestamp": f"2026-05-{day:02d}T18:00:00+00:00",
            "action_taken": "BUY NVDA → FILLED",
            "reasoning": "{}",
        })
    return rows


def _store_with_decisions(decisions: list[dict]):
    store = MagicMock()
    store.recent_decisions.return_value = decisions
    return store


class TestDecisionWeekdayLineSuppression:
    def test_empty_decisions_returns_empty(self):
        store = _store_with_decisions([])
        # INSUFFICIENT_DATA must stay silent
        assert _decision_weekday_line(store) == ""

    def test_even_distribution_returns_empty(self):
        # Lots of filled decisions across all weekdays, no concentration
        decisions = []
        for day in range(1, 28):
            decisions.append({
                "timestamp": f"2026-05-{day:02d}T18:00:00+00:00",
                "action_taken": "BUY NVDA → FILLED",
                "reasoning": "{}",
            })
        store = _store_with_decisions(decisions)
        assert _decision_weekday_line(store) == ""

    def test_below_min_total_returns_empty(self):
        # only 3 decisions total → INSUFFICIENT_DATA (< MIN_TOTAL_DECISIONS=7)
        decisions = [
            {"timestamp": "2026-05-01T18:00:00+00:00",
             "action_taken": "NO_DECISION",
             "reasoning": "claude returned no response (timeout)"}
            for _ in range(3)
        ]
        store = _store_with_decisions(decisions)
        assert _decision_weekday_line(store) == ""


class TestDecisionWeekdayLineSurfaces:
    def test_friday_concentration_surfaces_block(self):
        decisions = _decisions_concentrated_friday()
        store = _store_with_decisions(decisions)
        line = _decision_weekday_line(store)
        # Output present, names the verdict, names the weekday
        assert line, f"expected non-empty line, got {line!r}"
        assert "WEEKDAY_CONCENTRATION" in line
        assert "Fri" in line
        assert line.startswith("⚠️")
        # Builder's headline carries the percent
        assert "%" in line


class TestDecisionWeekdayLineFailureContract:
    def test_store_raise_returns_empty(self):
        store = MagicMock()
        store.recent_decisions.side_effect = RuntimeError("db locked")
        assert _decision_weekday_line(store) == ""

    def test_non_dict_builder_result_returns_empty(self):
        store = _store_with_decisions(_decisions_concentrated_friday())
        with patch("paper_trader.reporter.build_decision_weekday",
                   create=True, return_value=None):
            # The builder is imported inside the helper; need to patch the
            # module path the helper uses.
            pass
        # Patch via the actual import location
        with patch(
            "paper_trader.analytics.decision_weekday.build_decision_weekday",
            return_value=None,
        ):
            assert _decision_weekday_line(store) == ""

    def test_missing_headline_returns_empty(self):
        store = _store_with_decisions(_decisions_concentrated_friday())
        with patch(
            "paper_trader.analytics.decision_weekday.build_decision_weekday",
            return_value={"verdict": "WEEKDAY_CONCENTRATION", "headline": ""},
        ):
            assert _decision_weekday_line(store) == ""

    def test_non_concentration_verdict_returns_empty(self):
        store = _store_with_decisions(_decisions_concentrated_friday())
        with patch(
            "paper_trader.analytics.decision_weekday.build_decision_weekday",
            return_value={"verdict": "EVEN_DISTRIBUTION", "headline": "x"},
        ):
            assert _decision_weekday_line(store) == ""


class TestDecisionWeekdayWiredIntoSummaries:
    def test_hourly_summary_includes_weekday_block(self):
        # Patch _send to capture body and assert weekday surface is present.
        captured = {}

        def fake_send(message):
            captured["body"] = message
            return True

        with patch("paper_trader.reporter._send", side_effect=fake_send), \
             patch("paper_trader.reporter._decision_weekday_line",
                   return_value="⚠️ **DECISION WEEKDAY** ◈ WEEKDAY_CONCENTRATION\n> Fri has 60% NO_DECISION"):
            assert send_hourly_summary() is True
        body = captured.get("body", "")
        assert "DECISION WEEKDAY" in body
        assert "60% NO_DECISION" in body

    def test_hourly_summary_silent_when_no_concentration(self):
        captured = {}

        def fake_send(message):
            captured["body"] = message
            return True

        with patch("paper_trader.reporter._send", side_effect=fake_send), \
             patch("paper_trader.reporter._decision_weekday_line",
                   return_value=""):
            assert send_hourly_summary() is True
        body = captured.get("body", "")
        assert "DECISION WEEKDAY" not in body

    def test_daily_close_includes_weekday_block(self):
        captured = {}

        def fake_send(message):
            captured["body"] = message
            return True

        # send_daily_close also reads benchmark_sp500 (network). Stub it out
        # so the test stays offline.
        with patch("paper_trader.reporter._send", side_effect=fake_send), \
             patch("paper_trader.market.benchmark_sp500", return_value=None), \
             patch("paper_trader.reporter._decision_weekday_line",
                   return_value="⚠️ **DECISION WEEKDAY** ◈ WEEKDAY_CONCENTRATION\n> Fri starved"):
            assert send_daily_close() is True
        body = captured.get("body", "")
        assert "DECISION WEEKDAY" in body
        assert "Fri starved" in body

    def test_summary_still_sends_when_weekday_line_faults(self):
        captured = {}

        def fake_send(message):
            captured["body"] = message
            return True

        # A helper raising must NOT take down the whole summary — the
        # reporter additive contract. The helper itself catches and returns
        # "", so a fault is the empty string upstream — re-verify here.
        with patch("paper_trader.reporter._send", side_effect=fake_send), \
             patch(
                 "paper_trader.analytics.decision_weekday.build_decision_weekday",
                 side_effect=RuntimeError("boom"),
             ):
            assert send_hourly_summary() is True
        body = captured.get("body", "")
        assert body, "hourly body should still be sent even if weekday block faults"
