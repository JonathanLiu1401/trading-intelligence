"""Tests for analytics/dynamic_interval.py — the context-aware sleep
duration that replaces the hardcoded OPEN_INTERVAL_S / CLOSED_INTERVAL_S.

Discriminating regressions locked here:
* an earnings-window scenario falling back to MARKET_OPEN because the
  earnings calendar wasn't consulted,
* normal market hours not returning the documented 1800s,
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


# ─────────────────────── EARNINGS_WINDOW tier ───────────────────────

def test_earnings_window_held_name_today_returns_60_to_180s(tmp_path):
    """NVDA held + earnings today + it is 16:00 ET (inside the
    15:45-18:30 ET reaction window) → must pick EARNINGS_WINDOW, NOT
    fall back to MARKET_OPEN."""
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
    assert 60 <= sleep_s <= 180, (
        f"earnings-window tier should return 60-180s, got {sleep_s}"
    )


# ─────────────────────── MARKET_OPEN tier ───────────────────────

def test_normal_market_hours_no_earnings_returns_1800s(tmp_path):
    """10:30 ET Tuesday, AAPL held, no earnings event for AAPL today
    → MARKET_OPEN tier (1800s). Confirms the session-open window
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
    assert sleep_s == 1800, (
        f"normal market hours should return 1800s, got {sleep_s}"
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
    assert sleep_s == 1800
