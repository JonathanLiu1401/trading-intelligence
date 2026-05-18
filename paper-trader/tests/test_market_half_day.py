"""NYSE early-close (1:00 p.m. ET) half-day handling in market.is_market_open.

Before this feature the engine treated 13:00–16:00 ET on the day after
Thanksgiving and Christmas Eve as a normal open session: it ran the fast
30-min OPEN cadence and executed trades against frozen post-close yfinance
marks for three hours of a CLOSED market. These tests pin the corrected
behaviour AND assert full backward-compatibility on every regular day.
"""
from __future__ import annotations

from datetime import date, datetime

from zoneinfo import ZoneInfo

from paper_trader import market

NY = ZoneInfo("America/New_York")

# 2026-11-27 (Fri, day after Thanksgiving) and 2026-12-24 (Thu, Christmas Eve)
# are both weekday early closes; 2026-05-15 is an ordinary Friday.
HALF = date(2026, 11, 27)
HALF2 = date(2026, 12, 24)
REGULAR = date(2026, 5, 15)


def _ny(d: date, hh: int, mm: int) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=NY)


# ── helper-function correctness ───────────────────────────────────────────
def test_is_half_day_only_for_known_dates():
    assert market.is_half_day(HALF) is True
    assert market.is_half_day(HALF2) is True
    assert market.is_half_day(REGULAR) is False
    assert market.is_half_day(date(2026, 11, 26)) is False  # Thanksgiving itself


def test_close_minute_values():
    assert market.close_minute(HALF) == 13 * 60      # early close 13:00 ET
    assert market.close_minute(HALF2) == 13 * 60
    assert market.close_minute(REGULAR) == 16 * 60    # regular 16:00 ET


# ── the bug this feature fixes ────────────────────────────────────────────
def test_half_day_closed_after_early_bell():
    # 13:30 ET on a half-day — used to read OPEN, must now be CLOSED.
    assert market.is_market_open(_ny(HALF, 13, 30)) is False
    assert market.is_market_open(_ny(HALF, 15, 59)) is False
    assert market.is_market_open(_ny(HALF2, 14, 0)) is False


def test_half_day_open_before_early_bell():
    assert market.is_market_open(_ny(HALF, 10, 0)) is True
    assert market.is_market_open(_ny(HALF, 12, 59)) is True


def test_half_day_close_boundary_is_exclusive():
    # 13:00 ET sharp: the session is over (`< close_minute`).
    assert market.is_market_open(_ny(HALF, 13, 0)) is False
    # 12:59 ET: still the last open minute.
    assert market.is_market_open(_ny(HALF, 12, 59)) is True


def test_half_day_still_respects_open_bell():
    assert market.is_market_open(_ny(HALF, 9, 0)) is False   # pre-open
    assert market.is_market_open(_ny(HALF, 9, 29)) is False
    assert market.is_market_open(_ny(HALF, 9, 30)) is True   # open bell


# ── backward compatibility (regular sessions unchanged) ───────────────────
def test_regular_day_unchanged_at_2pm():
    # 14:00 ET on an ordinary weekday is still OPEN — the early close must
    # NOT leak into normal days.
    assert market.is_market_open(_ny(REGULAR, 14, 0)) is True
    assert market.is_market_open(_ny(REGULAR, 15, 59)) is True
    assert market.is_market_open(_ny(REGULAR, 16, 0)) is False  # regular close


def test_regular_day_open_and_weekend_unchanged():
    assert market.is_market_open(_ny(REGULAR, 10, 0)) is True
    # 2026-05-16/17 is Sat/Sun.
    assert market.is_market_open(_ny(date(2026, 5, 16), 11, 0)) is False
    assert market.is_market_open(_ny(date(2026, 5, 17), 11, 0)) is False


def test_full_holiday_still_fully_closed_and_not_a_half_day():
    # Thanksgiving (full close) at noon — closed, and is_half_day False.
    assert market.is_market_open(_ny(date(2026, 11, 26), 12, 0)) is False
    assert market.is_half_day(date(2026, 11, 26)) is False


def test_every_regular_minute_matches_pre_feature_rule():
    """Exhaustive backward-compat: for a normal weekday, is_market_open must
    equal the original `9:30 <= minutes < 16:00` rule for every minute."""
    for h in range(0, 24):
        for m in (0, 15, 29, 30, 31, 45, 59):
            minutes = h * 60 + m
            expected = (9 * 60 + 30) <= minutes < (16 * 60)
            assert market.is_market_open(_ny(REGULAR, h, m)) is expected, (
                f"regression at {h:02d}:{m:02d}"
            )
