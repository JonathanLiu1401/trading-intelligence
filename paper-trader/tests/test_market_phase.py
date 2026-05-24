"""Tests for ``paper_trader.market.market_phase`` and its wiring into the
live decide() prompt header via ``strategy._build_payload``.

Phase boundaries are tested at the exact tick because off-by-one in this
function silently mis-labels every header line the live trader feeds Opus —
a decision at 10:00:00 ET that read OPENING_BELL would let Opus think it
was in the volatile first 30-min when in fact it has crossed into normal
mid-session liquidity.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import market, strategy

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _ny(year, month, day, hour, minute):
    """Build the UTC datetime corresponding to a given NY wall-clock time."""
    return datetime(year, month, day, hour, minute, tzinfo=NY).astimezone(UTC)


class TestMarketPhaseTradingDay:
    # 2026-05-14 is a Thursday — a regular full-session trading day.

    def test_overnight_pre_dawn(self):
        # 03:59 ET — before the pre-market window opens.
        assert market.market_phase(_ny(2026, 5, 14, 3, 59)) == "OVERNIGHT"

    def test_pre_market_open_boundary_is_pre_market(self):
        # 04:00 ET exactly enters PRE_MARKET (inclusive lower bound).
        assert market.market_phase(_ny(2026, 5, 14, 4, 0)) == "PRE_MARKET"

    def test_pre_market_mid(self):
        assert market.market_phase(_ny(2026, 5, 14, 7, 30)) == "PRE_MARKET"

    def test_pre_market_just_before_open(self):
        # 09:29 ET — still PRE_MARKET, OPENING_BELL is at 09:30.
        assert market.market_phase(_ny(2026, 5, 14, 9, 29)) == "PRE_MARKET"

    def test_opening_bell_boundary_inclusive(self):
        # 09:30 ET exactly flips to OPENING_BELL.
        assert market.market_phase(_ny(2026, 5, 14, 9, 30)) == "OPENING_BELL"

    def test_opening_bell_mid(self):
        assert market.market_phase(_ny(2026, 5, 14, 9, 45)) == "OPENING_BELL"

    def test_opening_bell_last_minute(self):
        assert market.market_phase(_ny(2026, 5, 14, 9, 59)) == "OPENING_BELL"

    def test_mid_session_boundary_inclusive(self):
        # 10:00 ET exactly enters MID_SESSION.
        assert market.market_phase(_ny(2026, 5, 14, 10, 0)) == "MID_SESSION"

    def test_mid_session_mid_day(self):
        assert market.market_phase(_ny(2026, 5, 14, 12, 30)) == "MID_SESSION"

    def test_mid_session_just_before_close_window(self):
        # 15:29 ET — last minute of MID_SESSION (CLOSING_HALF_HOUR begins 15:30).
        assert market.market_phase(_ny(2026, 5, 14, 15, 29)) == "MID_SESSION"

    def test_closing_half_hour_boundary_inclusive(self):
        # 15:30 ET exactly enters CLOSING_HALF_HOUR.
        assert market.market_phase(_ny(2026, 5, 14, 15, 30)) == "CLOSING_HALF_HOUR"

    def test_closing_half_hour_just_before_close(self):
        assert market.market_phase(_ny(2026, 5, 14, 15, 59)) == "CLOSING_HALF_HOUR"

    def test_close_boundary_flips_to_after_close(self):
        # 16:00 ET exactly is no longer in CLOSING_HALF_HOUR — AFTER_CLOSE.
        assert market.market_phase(_ny(2026, 5, 14, 16, 0)) == "AFTER_CLOSE"

    def test_after_close_extended_hours(self):
        assert market.market_phase(_ny(2026, 5, 14, 18, 0)) == "AFTER_CLOSE"

    def test_after_close_last_minute(self):
        assert market.market_phase(_ny(2026, 5, 14, 19, 59)) == "AFTER_CLOSE"

    def test_overnight_after_extended_hours(self):
        # 20:00 ET exactly — after-hours window closes, overnight begins.
        assert market.market_phase(_ny(2026, 5, 14, 20, 0)) == "OVERNIGHT"

    def test_overnight_late_evening(self):
        assert market.market_phase(_ny(2026, 5, 14, 23, 59)) == "OVERNIGHT"


class TestMarketPhaseHalfDay:
    # 2026-11-27 is a known NYSE half-day (1:00 p.m. ET close — day after
    # Thanksgiving). MID_SESSION must end 30 min before that close (12:30 ET),
    # NOT 15:30 ET — otherwise the prompt would say MID_SESSION right through
    # the actual closing bell.

    def test_pre_market_still_normal(self):
        # Pre-market windows are unaffected by the early close.
        assert market.market_phase(_ny(2026, 11, 27, 8, 0)) == "PRE_MARKET"

    def test_opening_bell_still_930_to_10(self):
        # Half-days still open at 09:30 ET, OPENING_BELL semantics unchanged.
        assert market.market_phase(_ny(2026, 11, 27, 9, 30)) == "OPENING_BELL"

    def test_mid_session_after_open(self):
        assert market.market_phase(_ny(2026, 11, 27, 10, 30)) == "MID_SESSION"

    def test_mid_session_last_minute_is_1229(self):
        # MID_SESSION ends at close-30min = 12:30 ET on a half-day.
        assert market.market_phase(_ny(2026, 11, 27, 12, 29)) == "MID_SESSION"

    def test_closing_half_hour_begins_at_1230_on_half_day(self):
        assert market.market_phase(_ny(2026, 11, 27, 12, 30)) == "CLOSING_HALF_HOUR"

    def test_closing_half_hour_just_before_early_close(self):
        assert market.market_phase(_ny(2026, 11, 27, 12, 59)) == "CLOSING_HALF_HOUR"

    def test_after_close_begins_at_1pm_on_half_day(self):
        # The early bell rings at 13:00 ET — AFTER_CLOSE begins there, NOT
        # at the regular 16:00 close. This is the bug class this function
        # exists to surface to the prompt header.
        assert market.market_phase(_ny(2026, 11, 27, 13, 0)) == "AFTER_CLOSE"

    def test_after_close_three_hours_past_early_close(self):
        # 16:00 ET on a half-day — three hours past the actual bell, still
        # AFTER_CLOSE (never re-opens to CLOSING_HALF_HOUR).
        assert market.market_phase(_ny(2026, 11, 27, 16, 0)) == "AFTER_CLOSE"


class TestMarketPhaseNonTradingDays:
    def test_saturday_morning_is_weekend(self):
        # 2026-05-16 is a Saturday.
        assert market.market_phase(_ny(2026, 5, 16, 10, 0)) == "WEEKEND"

    def test_saturday_overnight_is_weekend(self):
        # Even at the post-after-hours window — WEEKEND takes priority.
        assert market.market_phase(_ny(2026, 5, 16, 22, 0)) == "WEEKEND"

    def test_sunday_mid_day_is_weekend(self):
        assert market.market_phase(_ny(2026, 5, 17, 14, 0)) == "WEEKEND"

    def test_thanksgiving_mid_day_is_holiday(self):
        # 2026-11-26 is Thanksgiving — full holiday.
        assert market.market_phase(_ny(2026, 11, 26, 10, 0)) == "HOLIDAY"

    def test_new_years_day_pre_market_window_is_holiday(self):
        # 2026-01-01 — 08:00 ET would normally be PRE_MARKET on a trading
        # day, but the holiday flag takes precedence over the phase windows.
        assert market.market_phase(_ny(2026, 1, 1, 8, 0)) == "HOLIDAY"

    def test_good_friday_session_window_is_holiday(self):
        # 2026-04-03 is Good Friday — 11:00 ET would be MID_SESSION on a
        # trading day; holiday must win.
        assert market.market_phase(_ny(2026, 4, 3, 11, 0)) == "HOLIDAY"


class TestMarketPhaseTimezoneInjection:
    """Verify that a tz-aware ``now`` from UTC, NY, or another zone all
    resolve to the same phase by walking back to NY wall-clock time."""

    def test_utc_naive_input_treated_as_utc(self):
        # Today's 14:30 UTC = 09:30 NY (one hour earlier in summer DST? May
        # is EDT = UTC-4, so 14:30 UTC = 10:30 ET → MID_SESSION). Verify the
        # function picks up the ambient tz.
        utc_dt = datetime(2026, 5, 14, 14, 30, tzinfo=UTC)
        assert market.market_phase(utc_dt) == "MID_SESSION"

    def test_default_now_does_not_raise(self):
        # The no-arg path uses real wall clock; it must always return one
        # of the documented phase tokens (the prompt header relies on it).
        valid = {"WEEKEND", "HOLIDAY", "PRE_MARKET", "OPENING_BELL",
                 "MID_SESSION", "CLOSING_HALF_HOUR", "AFTER_CLOSE",
                 "OVERNIGHT"}
        assert market.market_phase() in valid


class TestPhaseWiredIntoPrompt:
    """``strategy._build_payload`` must emit a ``MARKET_PHASE:`` line in the
    prompt header so Opus can calibrate conviction by phase."""

    def _snap(self):
        return {"cash": 1000.0, "total_value": 1000.0,
                "open_value": 0.0, "positions": []}

    def test_phase_line_present_with_label(self, monkeypatch):
        # Force phase to a known label so the assertion does not depend on
        # the wall clock at test time.
        monkeypatch.setattr(strategy.market, "market_phase",
                            lambda: "MID_SESSION")
        body = strategy._build_payload(self._snap(), [], [], {}, {}, None,
                                        True)
        assert "MARKET_PHASE: MID_SESSION" in body

    def test_phase_line_for_each_label(self, monkeypatch):
        for label in ("PRE_MARKET", "OPENING_BELL", "CLOSING_HALF_HOUR",
                      "AFTER_CLOSE", "WEEKEND", "HOLIDAY", "OVERNIGHT"):
            monkeypatch.setattr(strategy.market, "market_phase",
                                lambda lab=label: lab)
            body = strategy._build_payload(self._snap(), [], [], {}, {},
                                            None, True)
            assert f"MARKET_PHASE: {label}" in body, label

    def test_phase_failure_degrades_to_no_line(self, monkeypatch):
        # A market.market_phase() that raises must NOT abort the prompt —
        # the header drops the MARKET_PHASE line and ships the rest of the
        # payload, byte-identical to the pre-feature behaviour for the
        # MARKET_OPEN line that follows.
        def _boom():
            raise RuntimeError("simulated market.py fault")
        monkeypatch.setattr(strategy.market, "market_phase", _boom)
        body = strategy._build_payload(self._snap(), [], [], {}, {}, None,
                                        True)
        assert "MARKET_PHASE:" not in body
        # And the rest of the header still ships correctly.
        assert "MARKET_OPEN: True" in body
        assert "S&P 500 BENCHMARK:" in body

    def test_phase_line_sits_between_market_open_and_spy(self, monkeypatch):
        # Order is load-bearing: MARKET_OPEN then MARKET_PHASE then S&P, so
        # the trader (and Opus) reads the open/closed flag first and refines
        # with the phase. A swap would let the phase be misread as a
        # high-level state replacing the open/closed signal.
        monkeypatch.setattr(strategy.market, "market_phase",
                            lambda: "OPENING_BELL")
        body = strategy._build_payload(self._snap(), [], [], {}, {}, None,
                                        True)
        i_open = body.index("MARKET_OPEN:")
        i_phase = body.index("MARKET_PHASE:")
        i_sp = body.index("S&P 500 BENCHMARK:")
        assert i_open < i_phase < i_sp
