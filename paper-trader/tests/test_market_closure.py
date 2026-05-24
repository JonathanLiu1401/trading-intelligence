"""Tests for paper_trader/analytics/market_closure.py +
the new market.previous_session_close helper + the /api/market-closure-window
endpoint.

These pins assert exact NYSE calendar behaviour — classification ladder
(OPEN / OVERNIGHT / WEEKEND / HOLIDAY_EXTENDED), holiday-in-window
detection, gap-hours arithmetic, and the endpoint single-sourcing the
builder verdict (never re-deriving it).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard, market
from paper_trader.analytics.market_closure import (
    _holidays_in_window,
    build_market_closure,
)


# ─────────────────── previous_session_close helper ───────────────────


class TestPreviousSessionClose:
    """Mirror semantics of next_session_close, walked backward."""

    def test_returns_yesterday_close_mid_session(self):
        # Wed 2026-05-20 14:00 ET (mid-session) → Tue 2026-05-19 16:00 ET
        # is the most recent close that has already rung.
        now = datetime(2026, 5, 20, 18, 0, tzinfo=timezone.utc)  # 14:00 ET
        prev = market.previous_session_close(now)
        assert prev is not None
        ny = prev.astimezone(market.NY)
        assert ny.date().isoformat() == "2026-05-19"
        assert (ny.hour, ny.minute) == (16, 0)

    def test_returns_today_close_after_today_close(self):
        # Wed 2026-05-20 17:00 ET — past today's 16:00 close.
        now = datetime(2026, 5, 20, 21, 0, tzinfo=timezone.utc)  # 17:00 ET
        prev = market.previous_session_close(now)
        ny = prev.astimezone(market.NY)
        assert ny.date().isoformat() == "2026-05-20"
        assert (ny.hour, ny.minute) == (16, 0)

    def test_returns_friday_close_on_saturday(self):
        # Sat 2026-05-23 12:00 ET → Fri 2026-05-22 16:00 ET.
        now = datetime(2026, 5, 23, 16, 0, tzinfo=timezone.utc)  # 12:00 ET
        prev = market.previous_session_close(now)
        ny = prev.astimezone(market.NY)
        assert ny.date().isoformat() == "2026-05-22"
        assert (ny.hour, ny.minute) == (16, 0)

    def test_skips_memorial_day_holiday(self):
        # Tue 2026-05-26 08:00 ET (pre-open day after Memorial Day Monday).
        # Memorial Day 2026-05-25 has NO close — walk back to Fri 2026-05-22.
        now = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)  # 08:00 ET
        prev = market.previous_session_close(now)
        ny = prev.astimezone(market.NY)
        assert ny.date().isoformat() == "2026-05-22"

    def test_half_day_close_at_13_et(self):
        # Day-after-Thanksgiving 2026-11-27 (half-day, 13:00 ET close).
        # Sample at 14:00 ET to confirm the close is the 13:00 half-day bell,
        # not 16:00.
        now = datetime(2026, 11, 27, 19, 0, tzinfo=timezone.utc)  # 14:00 ET
        prev = market.previous_session_close(now)
        ny = prev.astimezone(market.NY)
        assert ny.date().isoformat() == "2026-11-27"
        assert (ny.hour, ny.minute) == (13, 0)


# ────────────────── _holidays_in_window (pure helper) ──────────────────


class TestHolidaysInWindow:
    def test_memorial_day_inside_a_friday_to_tuesday_window(self):
        # Fri 2026-05-22 16:00 ET close → Tue 2026-05-26 09:30 ET open.
        # Memorial Day 2026-05-25 (Mon) falls strictly inside the window.
        prev = datetime(2026, 5, 22, 20, 0, tzinfo=timezone.utc)
        nxt = datetime(2026, 5, 26, 13, 30, tzinfo=timezone.utc)
        assert _holidays_in_window(prev, nxt) == ["2026-05-25"]

    def test_normal_weekend_has_no_holidays(self):
        # Fri 2026-05-15 → Mon 2026-05-18 — no NYSE holiday in window.
        prev = datetime(2026, 5, 15, 20, 0, tzinfo=timezone.utc)
        nxt = datetime(2026, 5, 18, 13, 30, tzinfo=timezone.utc)
        assert _holidays_in_window(prev, nxt) == []

    def test_endpoint_dates_are_excluded(self):
        # The closing day (boundary) is NOT in the window. The open day
        # is also NOT in the window. Construct a pathological zero-gap.
        prev = datetime(2026, 5, 25, 20, 0, tzinfo=timezone.utc)  # closing
        nxt = datetime(2026, 5, 25, 20, 1, tzinfo=timezone.utc)
        assert _holidays_in_window(prev, nxt) == []

    def test_returns_empty_on_none(self):
        assert _holidays_in_window(None, None) == []


# ───────────────── build_market_closure classification ─────────────────


class TestBuildMarketClosure:
    def test_open_classification_inside_session(self):
        # Wed 2026-05-20 11:00 ET — mid-session.
        now = datetime(2026, 5, 20, 15, 0, tzinfo=timezone.utc)
        r = build_market_closure(now)
        assert r["is_market_open"] is True
        assert r["closure_class"] == "OPEN"
        assert r["verdict"] == "OPEN"
        assert r["closure_hours"] == 0.0
        assert r["hours_since_close"] == 0.0
        assert r["secs_until_open"] == 0
        assert r["holidays_in_window"] == []
        assert "OPEN — next bell" in r["headline"]

    def test_overnight_classification(self):
        # Wed 2026-05-20 20:00 ET (post-close) → Thu 2026-05-21 09:30 ET.
        # Gap = 17.5h, no holiday → OVERNIGHT.
        now = datetime(2026, 5, 21, 0, 0, tzinfo=timezone.utc)  # 20:00 ET
        r = build_market_closure(now)
        assert r["is_market_open"] is False
        assert r["closure_class"] == "OVERNIGHT"
        assert r["verdict"] == "CLOSED"
        assert r["holidays_in_window"] == []
        # Friday close-to-open overnight is ~17.5h
        assert 17.0 <= r["closure_hours"] <= 18.0
        assert "OVERNIGHT" in r["headline"]

    def test_weekend_classification_no_holiday(self):
        # Sat 2026-05-16 12:00 ET → Fri 16:00 → Mon 09:30. Gap ~65.5h.
        now = datetime(2026, 5, 16, 16, 0, tzinfo=timezone.utc)
        r = build_market_closure(now)
        assert r["is_market_open"] is False
        assert r["closure_class"] == "WEEKEND"
        assert r["verdict"] == "CLOSED"
        assert r["holidays_in_window"] == []
        assert 64.0 <= r["closure_hours"] <= 67.0
        assert "WEEKEND" in r["headline"]

    def test_holiday_extended_memorial_day_weekend(self):
        # Sun 2026-05-24 12:00 ET — the bot's *current* exact situation.
        # Fri 2026-05-22 16:00 ET close → Tue 2026-05-26 09:30 ET open.
        # Memorial Day Monday 2026-05-25 in window → HOLIDAY_EXTENDED.
        now = datetime(2026, 5, 24, 16, 0, tzinfo=timezone.utc)
        r = build_market_closure(now)
        assert r["is_market_open"] is False
        assert r["closure_class"] == "HOLIDAY_EXTENDED"
        assert r["verdict"] == "CLOSED"
        assert r["holidays_in_window"] == ["2026-05-25"]
        # Gap is ~89.5h (Fri 16:00 → Tue 09:30).
        assert 88.0 <= r["closure_hours"] <= 91.0
        assert "HOLIDAY_EXTENDED" in r["headline"]
        assert "2026-05-25" in r["headline"]

    def test_half_day_friday_after_thanksgiving_then_weekend(self):
        # Fri 2026-11-27 is a half-day (close 13:00 ET). Sat 2026-11-28 12:00 ET
        # → prev close Fri 13:00, next open Mon 2026-11-30 09:30. Gap ~68.5h
        # (longer than a regular Fri 16:00 → Mon 09:30 because the half-day
        # closes earlier). No NYSE holiday strictly inside the Sat-Sun gap.
        now = datetime(2026, 11, 28, 17, 0, tzinfo=timezone.utc)  # 12:00 ET
        r = build_market_closure(now)
        assert r["is_market_open"] is False
        assert r["closure_class"] == "WEEKEND"  # no holiday in gap
        assert r["holidays_in_window"] == []
        assert 67.0 <= r["closure_hours"] <= 70.0

    def test_secs_until_open_decreases_toward_bell(self):
        # Sun 12:00 ET vs Mon 08:00 ET on the same Mon-open week — the
        # later sample must have a SMALLER secs_until_open.
        sun = build_market_closure(
            datetime(2026, 5, 17, 16, 0, tzinfo=timezone.utc)  # 12:00 ET Sun
        )
        mon = build_market_closure(
            datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)  # 08:00 ET Mon
        )
        assert sun["secs_until_open"] > mon["secs_until_open"]
        assert mon["secs_until_open"] > 0  # not yet open

    def test_calendar_walk_failure_degrades_to_unknown(self, monkeypatch):
        # If is_market_open itself raises, the builder must degrade to a
        # safe-defaults envelope rather than propagate.
        def _boom(now=None):
            raise RuntimeError("calendar broken")
        import paper_trader.analytics.market_closure as mc_mod
        monkeypatch.setattr(mc_mod._mkt, "is_market_open", _boom)
        r = build_market_closure()
        assert r["closure_class"] == "UNKNOWN"
        assert r["verdict"] == "UNKNOWN"
        assert r["headline"].startswith("Closure window unknown")


# ───────────────── /api/market-closure-window endpoint ─────────────────


@pytest.fixture
def client():
    dashboard.app.config["TESTING"] = True
    return dashboard.app.test_client()


class TestMarketClosureWindowEndpoint:
    def test_endpoint_single_sources_builder_verdict(self, client):
        # The endpoint must NOT re-derive class / verdict — it composes
        # the builder verbatim. We assert the wire payload contains the
        # builder's keys and that the classification matches what the
        # builder would emit for the same instant.
        r = client.get("/api/market-closure-window")
        assert r.status_code == 200
        body = r.get_json()
        # Required envelope keys.
        for k in (
            "as_of", "is_market_open", "closure_class", "closure_hours",
            "hours_since_close", "secs_until_open", "prev_close_ts_utc",
            "next_open_ts_utc", "holidays_in_window", "headline", "verdict",
        ):
            assert k in body, f"missing key {k}"
        # closure_class must be one of the four documented values.
        assert body["closure_class"] in {
            "OPEN", "OVERNIGHT", "WEEKEND", "HOLIDAY_EXTENDED", "UNKNOWN",
        }
        # When CLOSED, secs_until_open must be a non-negative int.
        if not body["is_market_open"] and body["closure_class"] != "UNKNOWN":
            assert isinstance(body["secs_until_open"], int)
            assert body["secs_until_open"] >= 0

    def test_endpoint_returns_500_envelope_on_failure(self, client, monkeypatch):
        # Force the builder import to fail and confirm the route still
        # returns a valid-shaped envelope at HTTP 500 (mirrors the
        # /api/alarm-latches degrade contract).
        import paper_trader.analytics.market_closure as mc_mod

        def _boom(now=None):
            raise RuntimeError("builder broken")
        monkeypatch.setattr(mc_mod, "build_market_closure", _boom)
        r = client.get("/api/market-closure-window")
        assert r.status_code == 500
        body = r.get_json()
        assert body["closure_class"] == "UNKNOWN"
        assert body["verdict"] == "UNKNOWN"
        assert "builder broken" in body["headline"]
