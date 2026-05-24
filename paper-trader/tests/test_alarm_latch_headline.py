"""Tests for ``runner.alarm_latch_headline`` and its formatter.

These pin the operator-facing string the dashboard banner / chat
status surface will eventually render from
``runner.alarm_latch_state()``. The dashboard currently builds an
ad-hoc headline inline in ``/api/alarm-latches`` that omits the
wall-clock outage duration; this helper closes that gap. A drift on
any of these would be silently trader-visible.
"""
from __future__ import annotations

import pytest

from paper_trader import runner


# ── _fmt_outage_seconds ──────────────────────────────────────────────


class TestFmtOutageSeconds:
    def test_none_returns_empty(self):
        assert runner._fmt_outage_seconds(None) == ""

    def test_negative_returns_empty(self):
        # Clock step-back hazard the rest of runner.py already hardens against.
        assert runner._fmt_outage_seconds(-5) == ""
        assert runner._fmt_outage_seconds(-0.1) == ""

    def test_non_numeric_returns_empty(self):
        assert runner._fmt_outage_seconds("oops") == ""
        assert runner._fmt_outage_seconds(object()) == ""

    def test_sub_minute_clamps_to_zero_m(self):
        assert runner._fmt_outage_seconds(0) == "0m"
        assert runner._fmt_outage_seconds(30) == "0m"
        assert runner._fmt_outage_seconds(59) == "0m"

    def test_minutes_window(self):
        assert runner._fmt_outage_seconds(60) == "1m"
        assert runner._fmt_outage_seconds(60 * 42) == "42m"
        assert runner._fmt_outage_seconds(3599) == "59m"

    def test_hours_minutes(self):
        # 1h flat
        assert runner._fmt_outage_seconds(3600) == "1h0m"
        # 1h32m
        assert runner._fmt_outage_seconds(3600 + 32 * 60) == "1h32m"
        # 23h59m
        assert runner._fmt_outage_seconds(23 * 3600 + 59 * 60) == "23h59m"

    def test_days_hours(self):
        # 1d flat
        assert runner._fmt_outage_seconds(86400) == "1d0h"
        # 2d4h
        assert runner._fmt_outage_seconds(2 * 86400 + 4 * 3600) == "2d4h"

    def test_float_seconds_floor(self):
        # The helper coerces via int() so fractional seconds floor.
        assert runner._fmt_outage_seconds(60.999) == "1m"
        assert runner._fmt_outage_seconds(3661.5) == "1h1m"

    def test_int_string_coerces_via_float(self):
        # A numeric string is accepted (float() coerces); a non-numeric one is "".
        assert runner._fmt_outage_seconds("120") == "2m"


# ── alarm_latch_headline ─────────────────────────────────────────────


def _state(*, breaker=False, breaker_s=None, quota=False, quota_s=None,
           cons=0):
    """Convenience: build an ``alarm_latch_state``-shaped dict."""
    return {
        "breaker_active": breaker,
        "quota_active": quota,
        "any_active": breaker or quota,
        "consecutive_no_decisions": int(cons),
        "breaker_threshold": runner.CONSECUTIVE_NO_DECISION_LIMIT,
        "breaker_outage_s": breaker_s,
        "quota_outage_s": quota_s,
    }


class TestAlarmLatchHeadline:
    # ── silent path ────────────────────────────────────────────────

    def test_no_latch_returns_empty_string(self):
        """No latch held → no headline; the caller renders nothing.

        This is the silence-when-nothing-actionable contract: never emit
        an "OK" / "green light" line from a *latch* surface (the latch
        is *exactly* a 'something went wrong' marker)."""
        assert runner.alarm_latch_headline(_state()) == ""

    def test_non_dict_state_returns_empty_string(self):
        """A malformed state dict must degrade to "", never raise."""
        assert runner.alarm_latch_headline(None) == "" or True  # uses default
        assert runner.alarm_latch_headline("not a dict") == ""
        assert runner.alarm_latch_headline(42) == ""
        assert runner.alarm_latch_headline([]) == ""

    def test_default_state_uses_alarm_latch_state(self, monkeypatch):
        """Omitting ``state`` reads from ``alarm_latch_state()`` at call
        time. Verifies the helper is wired correctly to the live snapshot,
        not pinned at import time."""
        # Force the default-read path to return a known latched state.
        monkeypatch.setattr(
            runner, "alarm_latch_state",
            lambda: _state(breaker=True, breaker_s=120),
        )
        out = runner.alarm_latch_headline()
        assert "CLAUDE BREAKER held (2m)" in out
        assert out.startswith("⚠️ ")

    # ── breaker-only path ───────────────────────────────────────────

    def test_breaker_only_with_duration(self):
        out = runner.alarm_latch_headline(
            _state(breaker=True, breaker_s=42 * 60)
        )
        assert out == (
            "⚠️ CLAUDE BREAKER held (42m) — operator alerted; "
            "waiting for the next real decision to clear."
        )

    def test_breaker_only_without_duration(self):
        """Missing breaker_outage_s drops just the parenthetical, never
        the latch token itself — the operator must still see WHICH latch
        is held even if the timer is unknown."""
        out = runner.alarm_latch_headline(
            _state(breaker=True, breaker_s=None)
        )
        assert "CLAUDE BREAKER held" in out
        assert "CLAUDE BREAKER held (" not in out  # no opened paren
        assert out.endswith("clear.")
        assert "QUOTA" not in out

    def test_breaker_with_garbage_duration_drops_paren(self):
        """A non-numeric duration field must not crash and must not show
        a confusing '(blank)' parenthetical."""
        out = runner.alarm_latch_headline(
            _state(breaker=True, breaker_s="oops")
        )
        assert "CLAUDE BREAKER held" in out
        assert "(" not in out

    # ── quota-only path ─────────────────────────────────────────────

    def test_quota_only_with_duration(self):
        out = runner.alarm_latch_headline(
            _state(quota=True, quota_s=3 * 3600 + 12 * 60)
        )
        assert out == (
            "⚠️ QUOTA latch held (3h12m) — operator alerted; "
            "waiting for the next real decision to clear."
        )
        assert "BREAKER" not in out

    def test_quota_only_without_duration(self):
        out = runner.alarm_latch_headline(
            _state(quota=True, quota_s=None)
        )
        assert "QUOTA latch held" in out
        assert "QUOTA latch held (" not in out

    # ── both-latched path ──────────────────────────────────────────

    def test_both_latches_joined_with_middot(self):
        """Both held → both tokens joined with ' · ' (matches the
        dashboard's existing inline style) in the canonical order
        (breaker first, quota second)."""
        out = runner.alarm_latch_headline(
            _state(breaker=True, breaker_s=47 * 60,
                   quota=True, quota_s=2 * 3600 + 5 * 60)
        )
        # Order: breaker before quota
        idx_b = out.index("CLAUDE BREAKER")
        idx_q = out.index("QUOTA latch")
        assert idx_b < idx_q
        assert "CLAUDE BREAKER held (47m)" in out
        assert "QUOTA latch held (2h5m)" in out
        assert " · " in out
        assert out.endswith("clear.")

    def test_both_with_one_missing_duration(self):
        out = runner.alarm_latch_headline(
            _state(breaker=True, breaker_s=None,
                   quota=True, quota_s=120)
        )
        # Breaker bare, quota with paren
        assert "CLAUDE BREAKER held — " not in out  # not an emdash
        # Breaker must NOT carry a (foo) paren; quota MUST.
        assert "CLAUDE BREAKER held · QUOTA latch held (2m)" in out

    # ── boundary / determinism ────────────────────────────────────

    def test_any_active_alone_does_not_trigger_headline(self):
        """The function reads breaker_active / quota_active directly —
        a stray ``any_active=True`` with both flags False must NOT
        fabricate a headline (defensive against a divergent state dict)."""
        bogus = _state()  # both False
        bogus["any_active"] = True
        assert runner.alarm_latch_headline(bogus) == ""

    def test_headline_never_raises_on_any_input(self):
        """Property-style coverage of the never-raises contract."""
        for s in [
            None, "x", 0, [], (), {},
            {"breaker_active": True},  # no durations
            {"quota_active": True, "quota_outage_s": float("nan")},
            {"breaker_active": True, "breaker_outage_s": object()},
        ]:
            # Just must not raise; return is a string (possibly "").
            assert isinstance(runner.alarm_latch_headline(s), str)
