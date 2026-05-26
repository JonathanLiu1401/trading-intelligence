"""Tests for the alarm-latch one-liner injected into the hourly Discord summary.

A trader whose primary monitoring surface is Discord (per AGENTS.md) had no
way to see whether the consecutive-NO_DECISION breaker or the Claude quota
latch was CURRENTLY held between the FIRED alert and the eventual CLEARED
alert. The hourly summary is the natural surface for that state — on a
multi-hour wedge it answers "is it STILL latched now, hours later?".

These tests pin two contracts:

1. ``_alarm_latch_line`` silences itself when no latch is held (the
   silence-when-nothing-actionable precedent the rest of reporter.py
   follows — e.g. ``_drawdown_line`` / ``_mark_integrity_line``). A
   healthy engine must not introduce a noise line on every hourly.
2. When either latch is held, the line surfaces the runner-side headline
   **verbatim** (single source of truth — ``runner.alarm_latch_headline``
   owns the formatting), prefixed with the canonical bold tag
   ``**ENGINE LATCH** ◈`` so a trader scanning Discord recognises the
   block alongside the other status lines (``MARK INTEGRITY``,
   ``FEED HEALTH``, ``HOST``, ``CASH FIT``, etc.).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import reporter, runner


def _clear_runner_latches():
    """Reset every in-memory latch global the runner exposes so the line is
    tested against a deterministic clean slate."""
    runner._breaker_alert_active = False
    runner._quota_alert_active = False
    runner._consecutive_no_decisions = 0
    runner._no_decision_first_ts = None
    runner._quota_first_ts = None


class TestAlarmLatchLineSilent:
    """No latch held → no line in the hourly summary."""

    def teardown_method(self, _m):
        _clear_runner_latches()

    def test_clean_engine_returns_empty(self):
        _clear_runner_latches()
        assert reporter._alarm_latch_line() == ""

    def test_zero_consecutive_no_decisions_still_silent(self):
        # Latch booleans are False AND counter is zero — the most common case.
        _clear_runner_latches()
        runner._consecutive_no_decisions = 0
        assert reporter._alarm_latch_line() == ""


class TestAlarmLatchLineActive:
    """Either latch held → the line fires with the verbatim runner headline."""

    def teardown_method(self, _m):
        _clear_runner_latches()

    def test_breaker_latched_surfaces_headline(self):
        _clear_runner_latches()
        runner._breaker_alert_active = True
        line = reporter._alarm_latch_line()
        # Tag prefix is the canonical hourly-summary block marker.
        assert line.startswith("**ENGINE LATCH** ◈")
        # The runner-side headline body must be present (SSOT — reporter
        # never re-derives the verdict text).
        assert "CLAUDE BREAKER held" in line
        assert "operator alerted" in line

    def test_quota_latched_surfaces_headline(self):
        _clear_runner_latches()
        runner._quota_alert_active = True
        line = reporter._alarm_latch_line()
        assert line.startswith("**ENGINE LATCH** ◈")
        assert "QUOTA latch held" in line

    def test_both_latched_surfaces_both(self):
        _clear_runner_latches()
        runner._breaker_alert_active = True
        runner._quota_alert_active = True
        line = reporter._alarm_latch_line()
        assert "CLAUDE BREAKER held" in line
        assert "QUOTA latch held" in line

    def test_runner_import_failure_returns_empty(self, monkeypatch):
        """A degenerate runner import / state-read fault MUST degrade to ""
        — never raise into the hourly send path. Mirrors every other
        ``reporter`` helper's "never blocks the summary" contract."""
        _clear_runner_latches()
        # Force the helper to fail by replacing `alarm_latch_state` with a
        # raiser. The function should swallow the exception and return "".
        monkeypatch.setattr(
            runner, "alarm_latch_state",
            lambda: (_ for _ in ()).throw(RuntimeError("simulated fault")),
        )
        assert reporter._alarm_latch_line() == ""


class TestHourlySummaryIncludesLatchLine:
    """The line must be wired into ``send_hourly_summary`` — not just defined.
    The pin guards against a future refactor accidentally dropping the
    helper from the body composition (the exact failure mode the existing
    ``_session_block`` / ``_drawdown_line`` wiring tests guard against)."""

    def test_send_hourly_invokes_alarm_latch_helper(self, monkeypatch):
        # Capture the line content reporter._alarm_latch_line returns by
        # stubbing it. We verify it gets CALLED — the body composition is
        # exercised end-to-end in the existing test_decision_weekday_reporter
        # / test_blocked_reasons_reporter pattern.
        calls: list[int] = []

        def _stub():
            calls.append(1)
            return "**ENGINE LATCH** ◈ test-active-latch"

        monkeypatch.setattr(reporter, "_alarm_latch_line", _stub)
        # Neutralise the actual Discord send so the test stays offline.
        sent: list[str] = []
        monkeypatch.setattr(
            reporter, "_send",
            lambda body: sent.append(body) or True,
        )
        assert reporter.send_hourly_summary() is True
        assert calls == [1], "send_hourly_summary must invoke _alarm_latch_line"
        # And the latch text must land in the body the operator actually sees.
        assert any("**ENGINE LATCH** ◈ test-active-latch" in s for s in sent)
