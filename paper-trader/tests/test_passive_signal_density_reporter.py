"""Tests for paper_trader.reporter._passive_signal_density_line and its
wiring into send_hourly_summary / send_daily_close.

The helper is silence-by-default — it surfaces a line ONLY when the
``build_passive_signal_density`` verdict is ``DEAFENING_SILENCE``. Every
other verdict (NO_DATA / NO_PASSIVE_RUN / INSUFFICIENT / INFORMED_PASSIVE
/ SIGNAL_RICH_PASSIVE) returns ``""`` so the summary never grows a lying
green light.

The wiring discipline: the line MUST be wired into BOTH send_hourly_summary
and send_daily_close. Sibling lines (``_exit_only_streak_line`` /
``_repeat_loser_line`` / ``_rebuy_regret_line``) have always carried this
two-surface contract; the structural-sibling docstring says the same.
This test fails the instant the wiring is silently dropped from either
report — closing the bug where ``_passive_signal_density_line`` was
defined but never called (the DEAFENING_SILENCE Discord alert never fired
for any trader).
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from paper_trader.reporter import (
    _passive_signal_density_line,
    send_daily_close,
    send_hourly_summary,
)


def _store(decisions=None):
    store = MagicMock()
    store.recent_decisions.return_value = decisions or []
    return store


class TestPassiveSignalDensityLine:
    def test_returns_empty_on_no_data(self):
        # build_passive_signal_density returns NO_DATA on empty decisions —
        # silence-by-default keeps the hourly clean.
        assert _passive_signal_density_line(_store([])) == ""

    def test_returns_empty_on_no_passive_run(self):
        # Mock the builder to return a non-actionable verdict; the line
        # must stay silent.
        with patch(
            "paper_trader.analytics.passive_signal_density."
            "build_passive_signal_density"
        ) as mock_build:
            mock_build.return_value = {
                "verdict": "NO_PASSIVE_RUN",
                "headline": "Most recent decision was ACTIVE.",
            }
            assert _passive_signal_density_line(_store([{"x": 1}])) == ""

    def test_returns_empty_on_insufficient(self):
        with patch(
            "paper_trader.analytics.passive_signal_density."
            "build_passive_signal_density"
        ) as mock_build:
            mock_build.return_value = {
                "verdict": "INSUFFICIENT",
                "headline": "only 2 cycles since last active.",
            }
            assert _passive_signal_density_line(_store([{"x": 1}])) == ""

    def test_returns_empty_on_informed_passive(self):
        # Quiet news + quiet engine → no alert (the design intent).
        with patch(
            "paper_trader.analytics.passive_signal_density."
            "build_passive_signal_density"
        ) as mock_build:
            mock_build.return_value = {
                "verdict": "INFORMED_PASSIVE",
                "headline": "engine correctly quiet during quiet window.",
            }
            assert _passive_signal_density_line(_store([{"x": 1}])) == ""

    def test_returns_empty_on_signal_rich_passive(self):
        # Moderate news, watching not acting — still below the actionable
        # threshold. Only DEAFENING_SILENCE fires.
        with patch(
            "paper_trader.analytics.passive_signal_density."
            "build_passive_signal_density"
        ) as mock_build:
            mock_build.return_value = {
                "verdict": "SIGNAL_RICH_PASSIVE",
                "headline": "moderate news, watching but not acting.",
            }
            assert _passive_signal_density_line(_store([{"x": 1}])) == ""

    def test_returns_formatted_block_on_deafening_silence(self):
        with patch(
            "paper_trader.analytics.passive_signal_density."
            "build_passive_signal_density"
        ) as mock_build:
            mock_build.return_value = {
                "verdict": "DEAFENING_SILENCE",
                "headline": (
                    "DEAFENING_SILENCE — 12 passive cycles with median "
                    "15.0 signals/cycle (>10)."
                ),
            }
            out = _passive_signal_density_line(_store([{"x": 1}]))
            assert "SIGNAL-RICH PASSIVE" in out
            assert "DEAFENING_SILENCE" in out
            assert "12 passive cycles" in out
            # The Discord block prefix discipline — must start with the
            # bold header and contain the headline on a quote line.
            assert out.startswith("**SIGNAL-RICH PASSIVE**")
            assert "\n>" in out  # headline rendered as blockquote

    def test_silent_on_missing_headline(self):
        # Even with the right verdict, a missing headline must not emit a
        # half-rendered block (degrade-safe contract).
        with patch(
            "paper_trader.analytics.passive_signal_density."
            "build_passive_signal_density"
        ) as mock_build:
            mock_build.return_value = {"verdict": "DEAFENING_SILENCE",
                                       "headline": ""}
            assert _passive_signal_density_line(_store([{"x": 1}])) == ""

    def test_silent_on_non_dict_builder_result(self):
        # The builder's contract is dict, but a bug that returns None
        # must not crash the hourly — degrade to silent.
        with patch(
            "paper_trader.analytics.passive_signal_density."
            "build_passive_signal_density"
        ) as mock_build:
            mock_build.return_value = None
            assert _passive_signal_density_line(_store([{"x": 1}])) == ""

    def test_silent_on_builder_exception(self):
        # Any exception inside the builder must not propagate — the
        # additive contract says "drop this one line, never the whole
        # report".
        with patch(
            "paper_trader.analytics.passive_signal_density."
            "build_passive_signal_density"
        ) as mock_build:
            mock_build.side_effect = RuntimeError("simulated builder fault")
            assert _passive_signal_density_line(_store([{"x": 1}])) == ""


class TestWiringIntoHourlyAndDaily:
    """Source-level regression-lock: the line must be wired into BOTH
    reports. The bug this closes: ``_passive_signal_density_line`` was
    defined but never called from any send_* function, so the
    DEAFENING_SILENCE alert was unreachable from Discord (the documented
    primary surface). A typo or accidental removal would silently turn
    the alert off again."""

    def test_hourly_calls_passive_signal_density_line(self):
        src = inspect.getsource(send_hourly_summary)
        assert "_passive_signal_density_line(" in src, (
            "_passive_signal_density_line is not called from "
            "send_hourly_summary — the DEAFENING_SILENCE alert will not "
            "reach Discord on the hourly report"
        )

    def test_daily_close_calls_passive_signal_density_line(self):
        src = inspect.getsource(send_daily_close)
        assert "_passive_signal_density_line(" in src, (
            "_passive_signal_density_line is not called from "
            "send_daily_close — the DEAFENING_SILENCE alert will not "
            "reach Discord on the daily close report"
        )
