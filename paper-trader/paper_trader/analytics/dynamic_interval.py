"""Dynamic decision interval — replaces hardcoded OPEN_INTERVAL_S / CLOSED_INTERVAL_S.

Returns the recommended sleep duration in seconds based on current market
context. Designed for hot-path use: never raises, never does network I/O.

Tiers (highest-priority first):
  EARNINGS_WINDOW  60-180s   Held name has same-day earnings AND
                             it is 3:45pm-6:30pm ET (captures after-close
                             announcement + initial reaction window)
  SESSION_OPEN     300s      First 30 min of regular session (9:30-10:00 ET)
  EARNINGS_DAY     600s      Held name has earnings today (all day)
  MARKET_OPEN      1800s     Normal market hours
  MARKET_CLOSED    3600s     No special event, market closed
  QUIET_CLOSED     5400s     Market closed + no positions + no imminent events

The earnings calendar is read from disk (digital-intern writes it) — the
freshest readable candidate path by mtime wins. All I/O is wrapped in
try/except: any failure degrades to the simple MARKET_OPEN/MARKET_CLOSED
fallback, never raises. ``now`` and ``calendar_path`` are injectable for
testing.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Ordered candidate locations for digital-intern's earnings snapshot. The
# freshest readable file (by mtime) wins. Only consulted when the caller
# does not pass an explicit ``calendar_path``. Mirrors the discovery
# pattern in analytics/event_calendar.py.
_CANDIDATE_PATHS = (
    Path("/media/zeph/projects/digital-intern/data/earnings_calendar.json"),
    Path("/home/zeph/trading-intelligence/digital-intern/data/earnings_calendar.json"),
    Path("/home/zeph/digital-intern/data/earnings_calendar.json"),
)

_NY = ZoneInfo("America/New_York")

# Tier sleep durations (seconds).
_EARNINGS_WINDOW_S = 120   # midpoint of 60-180 — single int per spec
_SESSION_OPEN_S = 300
_EARNINGS_DAY_S = 600
_MARKET_OPEN_S = 1800
_MARKET_CLOSED_S = 3600
_QUIET_CLOSED_S = 5400


def _pick_freshest(paths) -> Path | None:
    """Of the readable candidates, the one with the newest mtime. Returns
    ``None`` when none can be stat'd. Never raises."""
    best: Path | None = None
    best_mtime: float = -1.0
    for p in paths:
        try:
            mtime = Path(p).stat().st_mtime
        except (OSError, ValueError, TypeError):
            continue
        if mtime > best_mtime:
            best, best_mtime = Path(p), mtime
    return best


def _load_calendar_events(calendar_path) -> list[dict]:
    """Read the earnings snapshot and return its events list. Returns an
    empty list on any failure. Never raises."""
    try:
        path = Path(calendar_path) if calendar_path is not None else \
            _pick_freshest(_CANDIDATE_PATHS)
        if path is None:
            return []
        snap = json.loads(Path(path).read_text())
        if not isinstance(snap, dict):
            return []
        events = snap.get("events") or []
        return events if isinstance(events, list) else []
    except (OSError, ValueError, TypeError):
        return []


def _parse_dt(s) -> datetime | None:
    """Parse an ISO timestamp, treating a naive value as UTC. Returns
    ``None`` on anything unparseable."""
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _is_session_open_window(now_et: datetime) -> bool:
    """First 30 min of the regular NYSE session (9:30-10:00 ET, weekday,
    not a full holiday). Half-days still open at 9:30 ET so they fall in
    this window normally."""
    if now_et.weekday() >= 5:  # Saturday/Sunday
        return False
    try:
        from ..market import NYSE_HOLIDAYS_2026
        if now_et.date() in NYSE_HOLIDAYS_2026:
            return False
    except Exception:
        pass
    start = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now_et.replace(hour=10, minute=0, second=0, microsecond=0)
    return start <= now_et < end


def _is_market_hours(now_et: datetime) -> bool:
    """Regular NYSE hours, honoring weekends, full holidays, and half-day
    early closes (so cadence here never disagrees with the trader's own
    ``market.is_market_open`` gate — the documented half-day bug was that the
    simple 9:30-16:00 rule kept the OPEN cadence running for three hours past
    the 13:00 early bell, and fired SESSION/OPEN-tier cycles on full
    holidays). Falls back to the bare weekday/hour rule only if the market
    module is unavailable."""
    try:
        from ..market import is_market_open
        return is_market_open(now_et.astimezone(timezone.utc))
    except Exception:
        if now_et.weekday() >= 5:
            return False
        start = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        end = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return start <= now_et < end


def _is_earnings_window(now_et: datetime) -> bool:
    """15:45-18:30 ET: after-close print + initial reaction band."""
    if now_et.weekday() >= 5:
        return False
    start = now_et.replace(hour=15, minute=45, second=0, microsecond=0)
    end = now_et.replace(hour=18, minute=30, second=0, microsecond=0)
    return start <= now_et < end


def _held_has_earnings_today(
    positions: list[dict],
    events: list[dict],
    now_utc: datetime,
    now_et: datetime,
) -> bool:
    """True iff some ticker in ``positions`` has an earnings event whose
    date matches today's ET calendar date, OR is within the next 12 hours
    (handles after-close prints stored as midnight UTC)."""
    if not positions or not events:
        return False
    held = set()
    for p in positions:
        try:
            tk = (p.get("ticker") or "").upper()
        except (AttributeError, TypeError):
            continue
        if tk:
            held.add(tk)
    if not held:
        return False
    today_et = now_et.date()
    horizon = now_utc + timedelta(hours=12)
    for ev in events:
        try:
            tk = (ev.get("ticker") or "").upper()
        except (AttributeError, TypeError):
            continue
        if not tk or tk not in held:
            continue
        ed = _parse_dt(ev.get("earnings_date"))
        if ed is None:
            continue
        ed_et_date = ed.astimezone(_NY).date()
        if ed_et_date == today_et:
            return True
        if now_utc <= ed <= horizon:
            return True
    return False


def compute_interval(
    positions: list[dict],
    now: datetime | None = None,
    calendar_path: Path | None = None,
) -> int:
    """Return the recommended sleep duration in seconds for the next cycle.

    ``positions`` — open positions, each a dict containing at least
    ``ticker``. ``now`` — current time (UTC, tz-aware); defaults to real
    ``datetime.now(timezone.utc)``. ``calendar_path`` — explicit earnings
    snapshot path (tests); otherwise the freshest known candidate by mtime.

    Never raises. On any internal failure, falls back to MARKET_OPEN or
    MARKET_CLOSED based on a simple weekday/hour check.
    """
    try:
        if now is None:
            now_utc = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now_utc = now.replace(tzinfo=timezone.utc)
        else:
            now_utc = now.astimezone(timezone.utc)
        now_et = now_utc.astimezone(_NY)

        positions = positions or []
        events = _load_calendar_events(calendar_path)

        held_earnings_today = _held_has_earnings_today(
            positions, events, now_utc, now_et,
        )

        # Tier 1 (highest priority): earnings window on a held name.
        if held_earnings_today and _is_earnings_window(now_et):
            tier, sleep_s = "EARNINGS_WINDOW", _EARNINGS_WINDOW_S
        # Tier 2: first 30 min of the regular session.
        elif _is_session_open_window(now_et):
            tier, sleep_s = "SESSION_OPEN", _SESSION_OPEN_S
        # Tier 3: held name has earnings today (all day).
        elif held_earnings_today:
            tier, sleep_s = "EARNINGS_DAY", _EARNINGS_DAY_S
        # Tier 4: normal market hours.
        elif _is_market_hours(now_et):
            tier, sleep_s = "MARKET_OPEN", _MARKET_OPEN_S
        # Tier 6 (before tier 5): quiet closed — no positions, closed.
        elif not positions:
            tier, sleep_s = "QUIET_CLOSED", _QUIET_CLOSED_S
        # Tier 5: market closed, positions held, no special event.
        else:
            tier, sleep_s = "MARKET_CLOSED", _MARKET_CLOSED_S

        print(f"[interval] tier={tier} sleep={sleep_s}s")
        return sleep_s
    except Exception:
        # Last-resort fallback: never raise. Use a simple weekday/hour
        # check against UTC if even the timezone math went sideways.
        try:
            if now is None:
                ref = datetime.now(timezone.utc)
            elif now.tzinfo is None:
                ref = now.replace(tzinfo=timezone.utc)
            else:
                ref = now
            ref_et = ref.astimezone(_NY)
            if _is_market_hours(ref_et):
                tier, sleep_s = "MARKET_OPEN", _MARKET_OPEN_S
            else:
                tier, sleep_s = "MARKET_CLOSED", _MARKET_CLOSED_S
        except Exception:
            tier, sleep_s = "MARKET_CLOSED", _MARKET_CLOSED_S
        try:
            print(f"[interval] tier={tier} sleep={sleep_s}s (fallback)")
        except Exception:
            pass
        return sleep_s
