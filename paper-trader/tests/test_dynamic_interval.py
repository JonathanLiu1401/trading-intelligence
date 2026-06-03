"""Tests for analytics/dynamic_interval.py — the context-aware sleep
duration that replaces the hardcoded OPEN_INTERVAL_S / CLOSED_INTERVAL_S.

Discriminating regressions locked here:
* after-close earnings monitoring running faster than regular-session cadence,
* normal market hours not returning the documented 300s,
* held-name earnings days using fast cadence while the market is closed,
* a Saturday with no positions losing the QUIET_CLOSED optimisation and
  reverting to the MARKET_CLOSED default.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.dynamic_interval import compute_interval

_NY = ZoneInfo("America/New_York")


def _et(year, month, day, hour, minute=0) -> datetime:
    """Build a tz-aware UTC datetime corresponding to the given ET wall time."""
    return datetime(year, month, day, hour, minute, tzinfo=_NY).astimezone(timezone.utc)


def _write_calendar(path: Path, events: list[dict]) -> None:
    snap = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "horizon_days": 14,
        "n_events": len(events),
        "events": events,
    }
    path.write_text(json.dumps(snap))


# ─────────────────────── EARNINGS tiers ───────────────────────

def test_after_close_earnings_window_is_slower_than_market_open(tmp_path):
    """NVDA held + earnings today + it is 16:00 ET (inside the
    15:45-18:30 ET reaction window) → use the after-close monitor cadence,
    not a frantic loop faster than tradable market hours."""
    cal = tmp_path / "earnings_calendar.json"
    # 2026-05-19 (Tuesday) 16:00 ET — same date in ET as the earnings stamp.
    now_utc = _et(2026, 5, 19, 16, 0)
    # NVDA reports after-close today. Earnings calendars commonly store the
    # day at midnight UTC; using ET midnight here keeps the date match
    # unambiguous on both sides of the dateline.
    earnings_dt = datetime(2026, 5, 19, 0, 0, tzinfo=_NY)
    _write_calendar(cal, [
        {"ticker": "NVDA", "earnings_date": earnings_dt.isoformat()},
    ])

    sleep_s = compute_interval(
        positions=[{"ticker": "NVDA"}],
        now=now_utc,
        calendar_path=cal,
    )
    assert sleep_s == 1800, (
        f"after-close earnings monitor should return 1800s, got {sleep_s}"
    )


def test_held_earnings_today_regular_session_returns_fast_open_cadence(tmp_path):
    """Held-name earnings day during regular NYSE hours → fast market-open
    cadence. This keeps actual tradable sessions more closely watched than
    closed-market monitoring."""
    cal = tmp_path / "earnings_calendar.json"
    now_utc = _et(2026, 5, 19, 14, 0)  # Tuesday, regular session
    earnings_dt = datetime(2026, 5, 19, 0, 0, tzinfo=_NY)
    _write_calendar(cal, [
        {"ticker": "NVDA", "earnings_date": earnings_dt.isoformat()},
    ])

    sleep_s = compute_interval(
        positions=[{"ticker": "NVDA"}],
        now=now_utc,
        calendar_path=cal,
    )
    assert sleep_s == 300, (
        f"regular-session earnings day should return 300s, got {sleep_s}"
    )


def test_held_earnings_today_premarket_uses_closed_cadence(tmp_path):
    """Held-name earnings day before the regular session must not use the
    fast earnings-day tier. This was the bug behind frequent closed-market
    cycles and sparse open-market cycles."""
    cal = tmp_path / "earnings_calendar.json"
    now_utc = _et(2026, 5, 19, 8, 0)  # Tuesday, premarket closed
    earnings_dt = datetime(2026, 5, 19, 0, 0, tzinfo=_NY)
    _write_calendar(cal, [
        {"ticker": "NVDA", "earnings_date": earnings_dt.isoformat()},
    ])

    sleep_s = compute_interval(
        positions=[{"ticker": "NVDA"}],
        now=now_utc,
        calendar_path=cal,
    )
    assert sleep_s == 3600, (
        f"premarket earnings day should use closed cadence, got {sleep_s}"
    )


# ─────────────────────── MARKET_OPEN tier ───────────────────────

def test_normal_market_hours_no_earnings_returns_300s(tmp_path):
    """10:30 ET Tuesday, AAPL held, no earnings event for AAPL today
    → MARKET_OPEN tier (300s). Confirms the session-open window
    (9:30-10:00) is correctly exited at 10:00 and the empty calendar
    doesn't accidentally promote a tier."""
    cal = tmp_path / "earnings_calendar.json"
    _write_calendar(cal, [])  # no events at all

    now_utc = _et(2026, 5, 19, 10, 30)  # Tuesday 10:30 ET

    sleep_s = compute_interval(
        positions=[{"ticker": "AAPL"}],
        now=now_utc,
        calendar_path=cal,
    )
    assert sleep_s == 300, (
        f"normal market hours should return 300s, got {sleep_s}"
    )


# ─────────────────────── QUIET_CLOSED tier ───────────────────────

def test_quiet_closed_saturday_no_positions_returns_5400s(tmp_path):
    """Saturday with no open positions and no calendar events at all
    → QUIET_CLOSED tier (5400s). Confirms the weekend branch beats the
    plain MARKET_CLOSED default when there's nothing to watch."""
    cal = tmp_path / "earnings_calendar.json"
    _write_calendar(cal, [])

    # 2026-05-23 is a Saturday. 22:00 ET would also qualify on a weekday,
    # but Saturday additionally proves the weekend gate.
    now_utc = _et(2026, 5, 23, 12, 0)

    sleep_s = compute_interval(
        positions=[],
        now=now_utc,
        calendar_path=cal,
    )
    assert sleep_s == 5400, (
        f"quiet closed (no positions, weekend) should return 5400s, got {sleep_s}"
    )


# ─────────────────────── never-raise contract ───────────────────────

def test_missing_calendar_path_does_not_raise(tmp_path):
    """A nonexistent calendar path must NOT raise — the hot path
    contract is "degrade, never crash"."""
    bogus = tmp_path / "does_not_exist.json"
    now_utc = _et(2026, 5, 19, 10, 30)
    sleep_s = compute_interval(
        positions=[{"ticker": "AAPL"}],
        now=now_utc,
        calendar_path=bogus,
    )
    # Missing file → no earnings tier — falls through to MARKET_OPEN
    # for a weekday 10:30 ET.
    assert sleep_s == 300


# ─────────────────────── half-day / holiday correctness ───────────────────────

def test_half_day_afternoon_after_early_close_is_closed_cadence(tmp_path):
    """Day after Thanksgiving 2026 (2026-11-27) closes at 13:00 ET. At 14:30
    ET — three-quarters of an hour past the early bell — the trader must NOT
    still be on the fast MARKET_OPEN cadence; the bug was the simple
    9:30-16:00 rule kept firing OPEN-tier cycles for three hours on a closed
    market, doubling Opus subprocess load against a frozen book."""
    cal = tmp_path / "earnings_calendar.json"
    _write_calendar(cal, [])
    now_utc = _et(2026, 11, 27, 14, 30)  # half-day, post-close
    sleep_s = compute_interval(
        positions=[{"ticker": "NVDA"}],
        now=now_utc,
        calendar_path=cal,
    )
    # Held position → MARKET_CLOSED (3600s), not MARKET_OPEN (300s)
    assert sleep_s == 3600, (
        f"half-day post-close should use closed cadence, got {sleep_s}s"
    )


def test_half_day_before_early_close_still_open_cadence(tmp_path):
    """Day after Thanksgiving 2026 at 11:00 ET — pre-13:00 close, market IS
    open. Cadence must remain MARKET_OPEN (300s) so the fix does not over-
    correct and starve the genuine half-day morning session."""
    cal = tmp_path / "earnings_calendar.json"
    _write_calendar(cal, [])
    now_utc = _et(2026, 11, 27, 11, 0)  # half-day morning, market open
    sleep_s = compute_interval(
        positions=[{"ticker": "NVDA"}],
        now=now_utc,
        calendar_path=cal,
    )
    assert sleep_s == 300, (
        f"half-day morning should still be open cadence, got {sleep_s}s"
    )


def test_full_holiday_uses_closed_cadence(tmp_path):
    """Christmas 2026 at 10:00 ET — NYSE fully closed. The trader must NOT
    cycle on the OPEN cadence: the simple weekday/hour rule used to pick
    MARKET_OPEN, leaving the runner spawning Opus subprocesses against a
    closed market every 30 min on a major holiday."""
    cal = tmp_path / "earnings_calendar.json"
    _write_calendar(cal, [])
    now_utc = _et(2026, 12, 25, 10, 0)  # Christmas Day, full holiday
    sleep_s = compute_interval(
        positions=[{"ticker": "NVDA"}],
        now=now_utc,
        calendar_path=cal,
    )
    assert sleep_s == 3600, (
        f"full holiday should use closed cadence, got {sleep_s}s"
    )


def test_holiday_does_not_trigger_session_open_window(tmp_path):
    """MLK Day 2026 (Monday 2026-01-19) at 9:45 ET — without the holiday
    check the 9:30-10:00 SESSION_OPEN window would fire (300s), which is
    even more aggressive than MARKET_OPEN. On a closed market this is
    pure wasted Opus capacity."""
    cal = tmp_path / "earnings_calendar.json"
    _write_calendar(cal, [])
    now_utc = _et(2026, 1, 19, 9, 45)  # MLK Day, fully closed
    sleep_s = compute_interval(
        positions=[{"ticker": "NVDA"}],
        now=now_utc,
        calendar_path=cal,
    )
    assert sleep_s == 3600, (
        f"holiday 9:45 ET should NOT trigger SESSION_OPEN, got {sleep_s}s"
    )
