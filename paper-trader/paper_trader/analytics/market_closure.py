"""Market-closure-window planning surface — frames the gap between the
last NYSE close and the next open as a planning object the dashboard, chat
and Discord can read.

Every other "is the market open" surface in this stack is a *boolean*: the
runner cadence, the prompt's MARKET_OPEN flag, the briefing card. None of
them say "the bell has been silent for 62 hours and the next bell rings
Tuesday 09:30 ET". That distinction matters because:

* The Opus reasoning on the live trader currently writes things like
  "MRVL earnings in 2.8d gives a setup to size into Tuesday open" — the
  bot is implicitly reasoning about a 56-hour closure window without any
  analytical surface that names it. A panel that says
  ``HOLIDAY_EXTENDED — 64h closure (Fri 16:00 → Tue 09:30); Memorial Day
  2026-05-25 in window`` lets the operator confirm at a glance whether
  the bot is sized correctly for the gap.
* The closure *class* (OVERNIGHT / WEEKEND / HOLIDAY_EXTENDED) is itself
  the signal: a normal overnight is ~17h, a weekend is ~64h, a holiday-
  extended weekend can be ~88h. Surfaces that just print "market closed"
  collapse this critical structural difference.

Pure: takes ``now`` (defaulting to wall-clock UTC) and walks the
``paper_trader.market`` NYSE calendar via the existing
``previous_session_close`` / ``next_session_open`` helpers. No I/O. Never
raises — a calendar miss degrades to ``UNKNOWN`` with safe defaults so
the dashboard panel can still render.

Advisory only — it reports, never gates Opus, adds no caps (the
``benchmark`` / ``self_review`` observational precedent).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .. import market as _mkt

# Classification thresholds (hours of bell-to-bell closure):
#   - OVERNIGHT: a normal weeknight close → next-morning open (~17.5h)
#   - WEEKEND:   Friday close → Monday open (~65.5h regular) — a Monday
#     holiday or a holiday inside the work week pushes this longer.
#   - HOLIDAY_EXTENDED: any closure containing one or more NYSE holiday
#     dates; the existence of a holiday in the window is the trigger,
#     not the hours threshold itself (a single mid-week holiday with no
#     weekend can still hit OVERNIGHT_HOLIDAY ~41h territory).
_OVERNIGHT_MAX_H = 24.0


def _hours_between(earlier: datetime, later: datetime) -> float:
    return (later - earlier).total_seconds() / 3600.0


def _holidays_in_window(prev_close: datetime, next_open: datetime) -> list[str]:
    """NYSE holiday dates (ISO strings) whose NY-local calendar date falls
    strictly between the previous close and the next open. The closing
    date itself is excluded (the bell already rang); the opening date is
    excluded (its 09:30 open is the boundary). A holiday wholly inside
    the gap — Memorial Day Monday for a Fri→Tue closure — appears here.
    """
    if prev_close is None or next_open is None:
        return []
    start = prev_close.astimezone(_mkt.NY).date()
    end = next_open.astimezone(_mkt.NY).date()
    out: list[str] = []
    cursor = start + timedelta(days=1)
    while cursor < end:
        if cursor in _mkt.NYSE_HOLIDAYS_2026:
            out.append(cursor.isoformat())
        cursor = cursor + timedelta(days=1)
    return out


def build_market_closure(now: datetime | None = None) -> dict:
    """Closure-window snapshot — the structured object the
    ``/api/market-closure-window`` endpoint serializes verbatim.

    Returns keys:

      * ``is_market_open`` — bool, the existing ``market.is_market_open``
        verdict.
      * ``closure_class`` — ``OPEN`` (market currently open) /
        ``OVERNIGHT`` (≤ 24h gap, no NYSE holiday) / ``WEEKEND`` (> 24h
        gap, no NYSE holiday) / ``HOLIDAY_EXTENDED`` (any holiday inside
        the gap, regardless of length).
      * ``closure_hours`` — wall-clock hours between previous close and
        next open. ``0.0`` when the market is currently OPEN.
      * ``prev_close_ts_utc`` / ``next_open_ts_utc`` — ISO timestamps.
      * ``secs_until_open`` — int seconds, or ``0`` when already open.
      * ``hours_since_close`` — float hours since the previous close
        (advances even after the next open is "in view"); ``0.0`` when
        open.
      * ``holidays_in_window`` — list of ISO dates strictly inside the
        gap that match ``NYSE_HOLIDAYS_2026``.
      * ``headline`` — single-source one-liner the dashboard, chat and
        Discord render verbatim.
      * ``verdict`` — ``OPEN`` / ``CLOSED`` / ``UNKNOWN`` (calendar miss).
    """
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    base = {
        "as_of": now_utc.isoformat(timespec="seconds"),
        "is_market_open": False,
        "closure_class": "UNKNOWN",
        "closure_hours": 0.0,
        "hours_since_close": 0.0,
        "secs_until_open": None,
        "prev_close_ts_utc": None,
        "next_open_ts_utc": None,
        "holidays_in_window": [],
        "headline": "Closure window unknown — NYSE calendar walk returned no anchor.",
        "verdict": "UNKNOWN",
    }
    try:
        is_open = _mkt.is_market_open(now_utc)
    except Exception:
        return base

    base["is_market_open"] = is_open
    prev_close = _mkt.previous_session_close(now_utc)
    next_open = _mkt.next_session_open(now_utc)

    if prev_close is not None:
        base["prev_close_ts_utc"] = prev_close.isoformat(timespec="seconds")
    if next_open is not None:
        base["next_open_ts_utc"] = next_open.isoformat(timespec="seconds")

    if is_open:
        base["closure_class"] = "OPEN"
        base["verdict"] = "OPEN"
        base["closure_hours"] = 0.0
        base["hours_since_close"] = 0.0
        base["secs_until_open"] = 0
        # Mid-session — name the next bell so the panel still shows the
        # "next interruption" without a separate fetch.
        close_dt = _mkt.next_session_close(now_utc)
        if close_dt is not None:
            secs = max(0, int((close_dt - now_utc).total_seconds()))
            hrs, mins = divmod(secs // 60, 60)
            base["headline"] = (
                f"OPEN — next bell in {hrs}h{mins:02d}m "
                f"({close_dt.astimezone(_mkt.NY).strftime('%a %H:%M ET')})."
            )
        else:
            base["headline"] = "OPEN — market currently trading."
        return base

    # Closed: classify the gap.
    if prev_close is None or next_open is None:
        return base

    closure_h = _hours_between(prev_close, next_open)
    holidays = _holidays_in_window(prev_close, next_open)
    secs_until_open = max(0, int((next_open - now_utc).total_seconds()))
    hours_since_close = max(0.0, _hours_between(prev_close, now_utc))

    if holidays:
        klass = "HOLIDAY_EXTENDED"
    elif closure_h > _OVERNIGHT_MAX_H:
        klass = "WEEKEND"
    else:
        klass = "OVERNIGHT"

    base["closure_class"] = klass
    base["closure_hours"] = round(closure_h, 2)
    base["hours_since_close"] = round(hours_since_close, 2)
    base["secs_until_open"] = secs_until_open
    base["holidays_in_window"] = holidays
    base["verdict"] = "CLOSED"

    prev_ny = prev_close.astimezone(_mkt.NY).strftime("%a %H:%M ET")
    next_ny = next_open.astimezone(_mkt.NY).strftime("%a %H:%M ET")
    hrs_to_open = secs_until_open // 3600
    rest_min = (secs_until_open % 3600) // 60
    if klass == "HOLIDAY_EXTENDED":
        hol_str = ", ".join(holidays)
        base["headline"] = (
            f"HOLIDAY_EXTENDED — {closure_h:.1f}h closure "
            f"({prev_ny} → {next_ny}); NYSE holiday in window: {hol_str}; "
            f"next bell in {hrs_to_open}h{rest_min:02d}m."
        )
    elif klass == "WEEKEND":
        base["headline"] = (
            f"WEEKEND — {closure_h:.1f}h closure "
            f"({prev_ny} → {next_ny}); next bell in "
            f"{hrs_to_open}h{rest_min:02d}m."
        )
    else:
        base["headline"] = (
            f"OVERNIGHT — {closure_h:.1f}h closure "
            f"({prev_ny} → {next_ny}); next bell in "
            f"{hrs_to_open}h{rest_min:02d}m."
        )
    return base


if __name__ == "__main__":
    import json
    print(json.dumps(build_market_closure(), indent=2))
