"""Tests for ``paper_trader.analytics.market_phase_block``.

Locks the additive heartbeat block contract: never raises, returns
phase + open boolean + countdown + headline, degrades to UNKNOWN on
fault. Plus the wiring lock: ``/api/runner-heartbeat`` actually carries
the ``market_phase`` block in its response.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import market as market_mod
from paper_trader.analytics.market_phase_block import (
    _fmt_secs,
    build_market_phase_block,
)

NY = ZoneInfo("America/New_York")


def _utc(yr, mo, day, hour, minute):
    """Convert (NY) Y/M/D/h/m to a UTC-aware datetime — readable test fixtures."""
    return datetime(yr, mo, day, hour, minute, tzinfo=NY).astimezone(timezone.utc)


class TestFmtSecs:
    def test_none_returns_empty(self):
        assert _fmt_secs(None) == ""

    def test_negative_returns_empty(self):
        assert _fmt_secs(-5) == ""

    def test_sub_minute_renders_zero_m(self):
        assert _fmt_secs(30) == "0m"

    def test_minutes(self):
        assert _fmt_secs(42 * 60) == "42m"

    def test_hours_and_minutes(self):
        assert _fmt_secs(1 * 3600 + 32 * 60) == "1h32m"

    def test_days_and_hours(self):
        assert _fmt_secs(2 * 86400 + 4 * 3600) == "2d4h"

    def test_non_numeric_returns_empty(self):
        assert _fmt_secs("nope") == ""


class TestBuildMarketPhaseBlock:
    def test_mid_session_weekday(self):
        # Wednesday 2026-06-03 11:00 ET — clean mid-session.
        now = _utc(2026, 6, 3, 11, 0)
        b = build_market_phase_block(market_mod, now=now)
        assert b["phase"] == "MID_SESSION"
        assert b["is_open"] is True
        assert b["is_half_day"] is False
        # 16:00 ET close → 5h to close.
        assert b["secs_to_close"] == 5 * 3600
        assert b["secs_to_open"] is None
        assert "MID_SESSION" in b["headline"]
        assert "5h0m until close" in b["headline"]

    def test_opening_bell(self):
        # Wednesday 09:35 ET — inside the opening bell window.
        now = _utc(2026, 6, 3, 9, 35)
        b = build_market_phase_block(market_mod, now=now)
        assert b["phase"] == "OPENING_BELL"
        assert b["is_open"] is True
        assert "OPENING_BELL" in b["headline"]
        # whipsaw blurb is present
        assert "whipsaw" in b["headline"]

    def test_closing_half_hour(self):
        # Wednesday 15:40 ET — last 30 min on a regular day.
        now = _utc(2026, 6, 3, 15, 40)
        b = build_market_phase_block(market_mod, now=now)
        assert b["phase"] == "CLOSING_HALF_HOUR"
        assert b["is_open"] is True
        # 20 min until 16:00 close.
        assert b["secs_to_close"] == 20 * 60
        assert "20m until close" in b["headline"]

    def test_pre_market_not_open(self):
        # Wednesday 08:00 ET — pre-market.
        now = _utc(2026, 6, 3, 8, 0)
        b = build_market_phase_block(market_mod, now=now)
        assert b["phase"] == "PRE_MARKET"
        assert b["is_open"] is False
        # secs_to_open should be ~1.5h (09:30 - 08:00) = 5400s.
        assert b["secs_to_open"] == 5400
        assert "opens in 1h30m" in b["headline"]

    def test_weekend(self):
        # Saturday — no session.
        now = _utc(2026, 6, 6, 11, 0)
        b = build_market_phase_block(market_mod, now=now)
        assert b["phase"] == "WEEKEND"
        assert b["is_open"] is False
        # Monday 09:30 ET → ~46h30m.
        assert b["secs_to_open"] is not None
        assert b["secs_to_open"] > 24 * 3600
        assert "WEEKEND" in b["headline"]
        assert "opens in" in b["headline"]

    def test_half_day_is_flagged_and_closing_half_hour_at_1230(self):
        # 2026-11-27 (Black Friday) is the documented NYSE half-day.
        # CLOSING_HALF_HOUR is 12:30-13:00 ET on a half-day, not 15:30-16:00.
        now = _utc(2026, 11, 27, 12, 40)
        b = build_market_phase_block(market_mod, now=now)
        assert b["phase"] == "CLOSING_HALF_HOUR"
        assert b["is_open"] is True
        assert b["is_half_day"] is True
        # 13:00 ET close → 20 min remaining.
        assert b["secs_to_close"] == 20 * 60
        # The headline carries the "(half-day)" hint so a trader sizing into
        # the closing window can immediately see the session ends 3h early.
        assert "(half-day)" in b["headline"]

    def test_holiday_returns_no_session_phase(self):
        # 2026-12-25 — Christmas Day.
        now = _utc(2026, 12, 25, 11, 0)
        b = build_market_phase_block(market_mod, now=now)
        assert b["phase"] == "HOLIDAY"
        assert b["is_open"] is False
        # Next session opens 2026-12-28 09:30 ET (Monday).
        assert b["secs_to_open"] is not None
        assert b["secs_to_open"] > 60 * 3600

    def test_after_close(self):
        # Wednesday 17:30 ET — after the bell but inside extended hours.
        now = _utc(2026, 6, 3, 17, 30)
        b = build_market_phase_block(market_mod, now=now)
        assert b["phase"] == "AFTER_CLOSE"
        assert b["is_open"] is False
        # Next open is Thursday 09:30 ET (~16h from 17:30).
        assert b["secs_to_open"] is not None
        assert b["secs_to_open"] > 12 * 3600

    def test_overnight_after_extended_hours(self):
        # Wednesday 23:30 ET — past 20:00 ET (extended-hours close),
        # before 04:00 ET pre-market open the next day.
        now = _utc(2026, 6, 3, 23, 30)
        b = build_market_phase_block(market_mod, now=now)
        assert b["phase"] == "OVERNIGHT"
        assert b["is_open"] is False

    def test_degrades_to_unknown_on_market_module_fault(self):
        """A monkeypatched market module that throws on every call still
        returns a well-formed block — the heartbeat endpoint can attach this
        block unconditionally without guarding."""
        class BrokenMarket:
            NY = NY  # for the is_half_day path's tz conversion attempt
            def market_phase(self, now): raise RuntimeError("boom")
            def is_market_open(self, now): raise RuntimeError("boom")
            def is_half_day(self, d): raise RuntimeError("boom")
            def seconds_until_close(self, now): raise RuntimeError("boom")
            def next_session_open(self, now): raise RuntimeError("boom")
        broken = BrokenMarket()
        b = build_market_phase_block(broken, now=_utc(2026, 6, 3, 11, 0))
        assert b["phase"] == "UNKNOWN"
        assert b["is_open"] is False
        # Both countdowns degrade to None.
        assert b["secs_to_close"] is None
        assert b["secs_to_open"] is None
        # The headline still renders something — the dashboard banner must
        # not see an empty string.
        assert b["headline"]
        assert "UNKNOWN" in b["headline"]


class TestHeartbeatWiring:
    """The block is actually attached to /api/runner-heartbeat's response."""

    def test_heartbeat_response_carries_market_phase_block(self,
                                                            tmp_path,
                                                            monkeypatch):
        # Lean smoke check via the Flask test client: a request to the real
        # endpoint returns a body that includes ``market_phase``.
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        import paper_trader.dashboard as d

        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        s.record_decision(True, 0, "HOLD NVDA → HOLD", "stand pat",
                           1000.0, 1000.0)
        # SWR disabled by default under pytest — that's fine, the handler
        # runs inline and returns the real block.
        d.app.config["TESTING"] = True
        try:
            with d.app.test_client() as client:
                j = client.get("/api/runner-heartbeat").get_json()
                assert "market_phase" in j
                mp = j["market_phase"]
                # Shape is the documented one.
                assert "phase" in mp
                assert "is_open" in mp
                assert "headline" in mp
                assert isinstance(mp["headline"], str)
                assert mp["headline"]  # never empty
        finally:
            s.close()
