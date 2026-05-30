"""Tests for paper_trader.analytics.last_fill + the _last_fill_line reporter helper.

Exercises the verdict ladder boundaries (FRESH/STATIC/FROZEN), the silence-by-
default reporter contract, the degrade-safe failure paths (empty ledger,
corrupt timestamp, non-dict ledger entries), and the wiring into the hourly
summary.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics import last_fill
from paper_trader.analytics.last_fill import (
    FRESH_HOURS,
    FROZEN_HOURS,
    build_last_fill,
)
from paper_trader import reporter


# Anchor "now" to a deterministic instant so the age math is repeatable.
NOW = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)


def _trade(ts: datetime, *, ticker="MU", action="BUY", qty=1.0,
           price=100.0, value=100.0, reason="news catalyst") -> dict:
    """Build a trade row in the ``store.recent_trades`` shape."""
    return {
        "timestamp": ts.isoformat(),
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "price": price,
        "value": value,
        "reason": reason,
    }


class TestBuildLastFillVerdicts:
    """Verdict ladder boundaries — FRESH/STATIC/FROZEN."""

    def test_no_data_on_empty_ledger(self):
        r = build_last_fill([], now=NOW)
        assert r["state"] == "NO_DATA"
        assert r["headline"]  # the no-data headline is informational
        assert r["secs_since"] is None
        assert r["age"] == ""
        assert r["ticker"] is None

    def test_no_data_on_none_ledger(self):
        r = build_last_fill(None, now=NOW)
        assert r["state"] == "NO_DATA"
        assert r["ticker"] is None

    def test_fresh_when_under_fresh_hours(self):
        # 2h ago — well within FRESH_HOURS (6)
        t = _trade(NOW - timedelta(hours=2), ticker="NVDA", action="BUY")
        r = build_last_fill([t], now=NOW)
        assert r["state"] == "FRESH"
        assert r["ticker"] == "NVDA"
        assert r["action"] == "BUY"
        assert r["age"] == "2h0m"
        assert "actively trading" in r["headline"]

    def test_fresh_boundary_just_under_threshold(self):
        # FRESH_HOURS - 1 minute — still FRESH
        t = _trade(NOW - timedelta(hours=FRESH_HOURS, minutes=-1))
        r = build_last_fill([t], now=NOW)
        assert r["state"] == "FRESH"

    def test_static_at_fresh_hours_boundary(self):
        # Exactly FRESH_HOURS — STATIC (hours >= fresh is STATIC)
        t = _trade(NOW - timedelta(hours=FRESH_HOURS), ticker="MU",
                   action="BUY")
        r = build_last_fill([t], now=NOW)
        assert r["state"] == "STATIC"
        assert r["ticker"] == "MU"
        assert "static" in r["headline"].lower()

    def test_static_in_window(self):
        # 12h ago — between FRESH_HOURS (6) and FROZEN_HOURS (36)
        t = _trade(NOW - timedelta(hours=12), ticker="AMD", action="SELL")
        r = build_last_fill([t], now=NOW)
        assert r["state"] == "STATIC"
        assert r["age"] == "12h0m"
        assert "AMD" in r["headline"]
        assert "SELL" in r["headline"]

    def test_frozen_at_threshold(self):
        # Exactly FROZEN_HOURS — FROZEN
        t = _trade(NOW - timedelta(hours=FROZEN_HOURS))
        r = build_last_fill([t], now=NOW)
        assert r["state"] == "FROZEN"

    def test_frozen_far_past_threshold(self):
        # 5 days ago — comfortably FROZEN
        t = _trade(NOW - timedelta(days=5), ticker="MU", action="BUY")
        r = build_last_fill([t], now=NOW)
        assert r["state"] == "FROZEN"
        assert r["age"] == "5d0h"
        assert "FROZEN" in r["headline"]

    def test_custom_thresholds_override_defaults(self):
        # A test could pass fresh=1h and a 3h-old trade is FROZEN.
        t = _trade(NOW - timedelta(hours=3))
        r = build_last_fill([t], now=NOW, fresh_hours=1.0, frozen_hours=2.0)
        assert r["state"] == "FROZEN"


class TestBuildLastFillShape:
    """Verify the returned dict carries every documented field with the
    expected types and values."""

    def test_returned_fields_for_fresh_buy(self):
        t = _trade(NOW - timedelta(hours=1), ticker="NVDA", action="BUY",
                   qty=3.0, price=200.50, value=601.50, reason="momentum")
        r = build_last_fill([t], now=NOW)
        assert r["state"] == "FRESH"
        assert r["ticker"] == "NVDA"
        assert r["action"] == "BUY"
        assert r["qty"] == 3.0
        assert r["price"] == 200.50
        assert r["value"] == 601.50
        assert r["reason"] == "momentum"
        assert r["secs_since"] == 3600.0
        assert r["age"] == "1h0m"
        assert r["last_fill_ts"] == (NOW - timedelta(hours=1)).isoformat()

    def test_option_action_is_upper(self):
        t = _trade(NOW - timedelta(hours=8), ticker="nvda", action="buy_call")
        r = build_last_fill([t], now=NOW)
        assert r["ticker"] == "NVDA"
        assert r["action"] == "BUY_CALL"

    def test_recent_trades_input_uses_index_zero(self):
        # The newest fill is index 0 — verify the builder doesn't
        # accidentally walk the whole list.
        new = _trade(NOW - timedelta(hours=2), ticker="MU")
        old = _trade(NOW - timedelta(days=10), ticker="ANCIENT")
        r = build_last_fill([new, old], now=NOW)
        assert r["ticker"] == "MU"
        assert r["state"] == "FRESH"


class TestBuildLastFillDegradeSafe:
    """Failure paths — every must degrade to NO_DATA without raising."""

    def test_unparseable_timestamp_degrades_to_no_data(self):
        t = {"timestamp": "not a real iso string", "ticker": "MU",
             "action": "BUY", "qty": 1.0, "price": 1.0, "value": 1.0}
        r = build_last_fill([t], now=NOW)
        assert r["state"] == "NO_DATA"
        assert r["ticker"] is None  # we can't surface fields from a torn row

    def test_missing_timestamp_field(self):
        t = {"ticker": "MU", "action": "BUY"}
        r = build_last_fill([t], now=NOW)
        assert r["state"] == "NO_DATA"

    def test_non_dict_ledger_entry_safe(self):
        r = build_last_fill(["not a dict"], now=NOW)
        assert r["state"] == "NO_DATA"

    def test_garbage_qty_does_not_raise(self):
        # Torn ledger row — qty is a string. Builder coerces to None.
        t = {"timestamp": NOW.isoformat(), "ticker": "MU", "action": "BUY",
             "qty": "garbage", "price": 100.0, "value": 100.0}
        r = build_last_fill([t], now=NOW)
        # Verdict still computed (timestamp parses); qty is None.
        assert r["state"] == "FRESH"
        assert r["qty"] is None

    def test_clock_step_back_clamps_to_zero(self):
        # Trade in the future (clock stepped back) — secs_since clamps to 0
        # rather than reporting a negative duration.
        t = _trade(NOW + timedelta(hours=1))
        r = build_last_fill([t], now=NOW)
        assert r["state"] == "FRESH"
        assert r["secs_since"] == 0.0
        assert r["age"] == "0m"

    def test_now_without_tz_treated_as_utc(self):
        # Defensive: a caller that passes a naive ``now`` must not crash.
        naive_now = NOW.replace(tzinfo=None)
        t = _trade(NOW - timedelta(hours=2))
        r = build_last_fill([t], now=naive_now)
        assert r["state"] == "FRESH"


class TestHumanize:
    """The age-formatter directly — covers the boundary cases that surface
    in the headline string."""

    def test_humanize_sub_minute_renders_zero_m(self):
        # Sub-minute clamps to "0m" so a 30s wedge reads cleanly,
        # not as the empty string (which would suppress the age token in
        # the headline).
        assert last_fill._humanize(30) == "0m"

    def test_humanize_negative_returns_empty(self):
        assert last_fill._humanize(-5) == ""

    def test_humanize_none_returns_empty(self):
        assert last_fill._humanize(None) == ""

    def test_humanize_garbage_returns_empty(self):
        assert last_fill._humanize("abc") == ""

    def test_humanize_one_hour(self):
        assert last_fill._humanize(3600) == "1h0m"

    def test_humanize_one_day(self):
        assert last_fill._humanize(86400) == "1d0h"


class _StubStore:
    """Test double for ``reporter._last_fill_line``. Only ``recent_trades``
    is used."""

    def __init__(self, trades):
        self._trades = trades

    def recent_trades(self, limit):
        return list(self._trades[:limit])


class TestLastFillReporterLine:
    """The reporter helper composes the builder + surfaces only the
    actionable verdicts."""

    def test_silent_on_fresh(self):
        t = _trade(datetime.now(timezone.utc) - timedelta(hours=1))
        line = reporter._last_fill_line(_StubStore([t]))
        assert line == ""

    def test_silent_on_no_data(self):
        line = reporter._last_fill_line(_StubStore([]))
        assert line == ""

    def test_surfaces_on_static(self):
        t = _trade(datetime.now(timezone.utc) - timedelta(hours=12),
                   ticker="NVDA", action="BUY")
        line = reporter._last_fill_line(_StubStore([t]))
        assert "LAST FILL" in line
        assert "STATIC" in line
        assert "NVDA" in line

    def test_surfaces_on_frozen_with_warning_prefix(self):
        t = _trade(datetime.now(timezone.utc) - timedelta(hours=48),
                   ticker="MU", action="BUY")
        line = reporter._last_fill_line(_StubStore([t]))
        assert "LAST FILL" in line
        assert "FROZEN" in line
        assert "⚠️" in line  # warning marker on FROZEN

    def test_degrades_to_empty_on_store_fault(self):
        class _BadStore:
            def recent_trades(self, n):
                raise RuntimeError("simulated lock contention")

        line = reporter._last_fill_line(_BadStore())
        assert line == ""

    def test_unrecognized_state_silent(self):
        # If the builder returns something unexpected, the reporter must
        # not surface garbage to Discord.
        import paper_trader.analytics.last_fill as lf_mod
        orig = lf_mod.build_last_fill
        try:
            lf_mod.build_last_fill = lambda *_a, **_k: {
                "state": "MOON",
                "headline": "garbage state",
            }
            line = reporter._last_fill_line(_StubStore([_trade(NOW)]))
            assert line == ""
        finally:
            lf_mod.build_last_fill = orig


class TestReporterWiring:
    """The line is wired into both hourly summary and daily close."""

    def test_helper_visible_in_reporter_module(self):
        assert hasattr(reporter, "_last_fill_line")

    def test_hourly_summary_invokes_last_fill(self):
        # Read the source of send_hourly_summary and assert the wiring
        # call is present — a structural assertion that survives
        # refactors but catches an accidental delete of the wiring.
        import inspect
        src = inspect.getsource(reporter.send_hourly_summary)
        assert "_last_fill_line(store)" in src

    def test_daily_close_invokes_last_fill(self):
        import inspect
        src = inspect.getsource(reporter.send_daily_close)
        assert "_last_fill_line(store)" in src
