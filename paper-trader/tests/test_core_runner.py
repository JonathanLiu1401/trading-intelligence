"""Tests for paper_trader.runner — the hourly + daily-close gating logic.

The runner is hard to unit-test as a whole because it runs an infinite loop,
but the gating helpers _maybe_hourly() and _maybe_daily_close() are pure
functions over module state + the wall clock. We patch the clock and the
reporter so each test deterministically reaches a single decision branch.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import runner

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _ny(year, month, day, hour, minute):
    """Build a UTC datetime corresponding to a given NY wall-clock time."""
    return datetime(year, month, day, hour, minute, tzinfo=NY).astimezone(UTC)


@pytest.fixture(autouse=True)
def _reset_runner_state(monkeypatch):
    """Each test starts with a fresh module state — no prior hourly/daily fired."""
    monkeypatch.setattr(runner, "_daily_close_sent_for", None)
    monkeypatch.setattr(runner, "_last_hourly", None)


def _patch_now(monkeypatch, when):
    """Patch datetime.now inside runner so the gating sees a fixed time."""
    class _FakeDT:
        @staticmethod
        def now(tz=None):
            return when.astimezone(tz) if tz else when

        # The module also calls datetime.fromisoformat etc. — preserve passthroughs.
        @staticmethod
        def fromisoformat(s):
            return datetime.fromisoformat(s)

    monkeypatch.setattr(runner, "datetime", _FakeDT)


class TestMaybeDailyClose:
    def test_does_not_fire_on_saturday(self, monkeypatch):
        # 2026-05-16 is a Saturday, 17:00 ET — past the trigger time but weekend.
        _patch_now(monkeypatch, _ny(2026, 5, 16, 17, 0))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        runner._maybe_daily_close()
        assert calls == []
        # Sent-for flag must NOT advance on weekends.
        assert runner._daily_close_sent_for is None

    def test_does_not_fire_on_sunday(self, monkeypatch):
        _patch_now(monkeypatch, _ny(2026, 5, 17, 17, 0))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        runner._maybe_daily_close()
        assert calls == []

    def test_does_not_fire_on_nyse_holiday(self, monkeypatch):
        # 2026-05-25 is Memorial Day — a Monday (weekday) full-market close.
        # 16:10 ET is past the trigger time, so only the holiday guard can
        # stop the spurious "DAILY CLOSE" post.
        _patch_now(monkeypatch, _ny(2026, 5, 25, 16, 10))
        assert _ny(2026, 5, 25, 16, 10).astimezone(NY).weekday() < 5  # is a weekday
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        runner._maybe_daily_close()
        assert calls == []
        # Holiday guard must not advance the sent-for flag.
        assert runner._daily_close_sent_for is None

    def test_does_not_fire_before_1605_ET(self, monkeypatch):
        # Thursday 16:04 NY — too early.
        _patch_now(monkeypatch, _ny(2026, 5, 14, 16, 4))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        runner._maybe_daily_close()
        assert calls == []

    def test_does_not_fire_at_1500_ET(self, monkeypatch):
        # 3 PM — well before close.
        _patch_now(monkeypatch, _ny(2026, 5, 14, 15, 0))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        runner._maybe_daily_close()
        assert calls == []

    def test_fires_at_1605_ET(self, monkeypatch):
        _patch_now(monkeypatch, _ny(2026, 5, 14, 16, 5))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        runner._maybe_daily_close()
        assert calls == [1]
        assert runner._daily_close_sent_for == "2026-05-14"

    def test_only_fires_once_per_day(self, monkeypatch):
        _patch_now(monkeypatch, _ny(2026, 5, 14, 16, 10))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        runner._maybe_daily_close()
        runner._maybe_daily_close()
        runner._maybe_daily_close()
        assert calls == [1]

    def test_fires_again_next_day(self, monkeypatch):
        calls = []
        monkeypatch.setattr(runner.reporter, "send_daily_close",
                            lambda: calls.append(1) or True)
        # Fire on Thursday.
        _patch_now(monkeypatch, _ny(2026, 5, 14, 16, 10))
        runner._maybe_daily_close()
        # Fire on Friday — different date → fires again.
        _patch_now(monkeypatch, _ny(2026, 5, 15, 16, 10))
        runner._maybe_daily_close()
        assert len(calls) == 2

    def test_send_failure_does_not_advance_sent_for(self, monkeypatch):
        _patch_now(monkeypatch, _ny(2026, 5, 14, 16, 10))
        # Simulate openclaw failure: send_daily_close returns False.
        monkeypatch.setattr(runner.reporter, "send_daily_close", lambda: False)
        runner._maybe_daily_close()
        # Must NOT mark today as sent, so we retry next cycle.
        assert runner._daily_close_sent_for is None


class TestMaybeHourly:
    def test_fires_when_last_hourly_none(self, monkeypatch):
        _patch_now(monkeypatch, _ny(2026, 5, 14, 10, 0))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_hourly_summary",
                            lambda: calls.append(1) or True)
        runner._last_hourly = None
        runner._maybe_hourly()
        assert calls == [1]

    def test_does_not_fire_within_3600s(self, monkeypatch):
        first_t = _ny(2026, 5, 14, 10, 0)
        runner._last_hourly = first_t
        # 30 minutes later — should NOT fire.
        _patch_now(monkeypatch, _ny(2026, 5, 14, 10, 30))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_hourly_summary",
                            lambda: calls.append(1) or True)
        runner._maybe_hourly()
        assert calls == []

    def test_fires_after_3600s(self, monkeypatch):
        first_t = _ny(2026, 5, 14, 10, 0)
        runner._last_hourly = first_t
        # 65 minutes later — should fire.
        _patch_now(monkeypatch, _ny(2026, 5, 14, 11, 5))
        calls = []
        monkeypatch.setattr(runner.reporter, "send_hourly_summary",
                            lambda: calls.append(1) or True)
        runner._maybe_hourly()
        assert calls == [1]

    def test_send_failure_does_not_advance_last_hourly(self, monkeypatch):
        runner._last_hourly = None
        _patch_now(monkeypatch, _ny(2026, 5, 14, 10, 0))
        monkeypatch.setattr(runner.reporter, "send_hourly_summary", lambda: False)
        runner._maybe_hourly()
        # If send failed, we want to retry on the next cycle, not skip an hour.
        assert runner._last_hourly is None

    def test_send_exception_swallowed(self, monkeypatch):
        runner._last_hourly = None
        _patch_now(monkeypatch, _ny(2026, 5, 14, 10, 0))

        def boom():
            raise RuntimeError("openclaw exploded")

        monkeypatch.setattr(runner.reporter, "send_hourly_summary", boom)
        # Must not raise; the runner is a daemon loop.
        runner._maybe_hourly()
        assert runner._last_hourly is None
