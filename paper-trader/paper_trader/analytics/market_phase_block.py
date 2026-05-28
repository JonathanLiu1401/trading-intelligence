"""Build the ``market_phase`` operator block for /api/runner-heartbeat.

Why this exists
---------------
The runner-heartbeat already surfaces ``market_open: bool`` — but a trader
checking the heartbeat in the middle of a trading day needs more than a
boolean. ``OPENING_BELL`` (09:30-10:00 ET, whipsaw-prone, wide spreads),
``MID_SESSION`` (the calm liquidity regime), and ``CLOSING_HALF_HOUR``
(rebalance flow + earnings about to print after the bell) are three very
different states a portfolio manager calibrates conviction differently in.
``market.market_phase()`` already returns the granular label and Opus
sees it in the prompt header, but the operator looking at the dashboard
or polling the heartbeat from outside did NOT — silently the most
useful piece of context for "is this a sensible time to add risk?".

What the block carries
----------------------
A single composition over the pure functions already in ``market.py`` —
``market_phase`` / ``seconds_until_close`` / ``next_session_open`` /
``previous_session_close``. No new state, no I/O, no caps; observational
only (AGENTS.md invariants #2/#12 — same additive contract as the
``singleton_lock`` / ``notify`` / ``latches`` / ``claude_call`` blocks
the heartbeat already carries).

Returned shape:
  * ``phase`` — one of the strings ``market_phase`` returns (WEEKEND /
    HOLIDAY / PRE_MARKET / OPENING_BELL / MID_SESSION / CLOSING_HALF_HOUR
    / AFTER_CLOSE / OVERNIGHT)
  * ``is_open`` — bool — True only during the regular session window
    (mirrors ``market.is_market_open``); a trading day's PRE_MARKET /
    OPENING_BELL window with extended-hours equity activity is NOT
    ``is_open`` (the binary the runner's cycle cadence keys off)
  * ``is_half_day`` — bool — True iff today (NY) is a known early-close
    half-day (13:00 ET close, not 16:00); a trader sizing into the
    CLOSING_HALF_HOUR on a half-day cares whether they have minutes or
    hours of session left
  * ``secs_to_close`` — int|None — seconds until the next NYSE session
    close (None when no close is reachable in the calendar window — a
    defensive cap matching ``market.next_session_close``)
  * ``secs_to_open`` — int|None — seconds until the next regular-session
    09:30 ET open; ``None`` only when the calendar walk fails to find one
    inside 14 forward days (defensive)
  * ``headline`` — operator-facing single line; the only string a
    dashboard banner needs to render

Failure contract
----------------
Never raises. A monkeypatched market module that throws on every call
returns ``{"phase": "UNKNOWN", "is_open": False, "headline": "..."}``
so the heartbeat endpoint can attach the block unconditionally without
guarding (the same degrade-safe shape ``notify_health()`` returns).
"""
from __future__ import annotations

from datetime import datetime, timezone


def _fmt_secs(secs: int | None) -> str:
    """``42m`` / ``1h32m`` / ``2d4h`` — compact countdown label.

    Mirrors the ``_format_elapsed`` token already used elsewhere in
    reporter.py: ``None`` / negative → ``""`` so the caller suppresses
    the duration token entirely. Sub-minute clamps to ``0m`` so a
    30s countdown reads cleanly, not as ``""``."""
    if secs is None:
        return ""
    try:
        s = int(secs)
    except (TypeError, ValueError):
        return ""
    if s < 0:
        return ""
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        h, m = divmod(s, 3600)
        return f"{h}h{m // 60}m"
    d, rem = divmod(s, 86400)
    return f"{d}d{rem // 3600}h"


_PHASE_BLURB = {
    "WEEKEND":           "no session",
    "HOLIDAY":           "no session (NYSE holiday)",
    "PRE_MARKET":        "extended hours, no regular session yet",
    "OPENING_BELL":      "first 30 minutes — whipsaw-prone, wide spreads",
    "MID_SESSION":       "regular session, normal liquidity",
    "CLOSING_HALF_HOUR": "last 30 minutes — rebalance flow + post-close earnings imminent",
    "AFTER_CLOSE":       "regular session over, extended hours active",
    "OVERNIGHT":         "no extended-hours session active",
    "UNKNOWN":           "calendar resolution failed",
}


def _build_headline(phase: str, is_open: bool, is_half_day: bool,
                    secs_to_close: int | None,
                    secs_to_open: int | None) -> str:
    blurb = _PHASE_BLURB.get(phase, "")
    half_tok = " (half-day)" if is_half_day else ""
    if is_open:
        # Inside the regular session — emphasise close countdown.
        close_tok = _fmt_secs(secs_to_close)
        if close_tok:
            return f"{phase}{half_tok} — {close_tok} until close · {blurb}"
        return f"{phase}{half_tok} · {blurb}"
    # Outside the regular session — emphasise the next open.
    open_tok = _fmt_secs(secs_to_open)
    if open_tok:
        return f"{phase} — opens in {open_tok} · {blurb}"
    return f"{phase} · {blurb}"


def build_market_phase_block(market_module, now=None) -> dict:
    """Compose the heartbeat ``market_phase`` block.

    ``market_module`` is the ``paper_trader.market`` module (passed
    explicitly so tests can substitute a stub without monkey-patching the
    global import — the ``runner_heartbeat`` precedent that takes
    ``now=None`` for the same reason). ``now`` is a UTC-aware datetime
    (default real wall clock).

    Returns the additive block shape documented in the module docstring.
    Degrade-safe — any internal fault returns the ``UNKNOWN`` shape so the
    heartbeat endpoint can attach this block unconditionally."""
    if now is None:
        now = datetime.now(timezone.utc)
    try:
        phase = market_module.market_phase(now)
    except Exception:
        phase = "UNKNOWN"
    try:
        is_open = bool(market_module.is_market_open(now))
    except Exception:
        is_open = False
    try:
        is_half_day = bool(market_module.is_half_day(now.astimezone(market_module.NY).date()))
    except Exception:
        is_half_day = False
    secs_to_close: int | None = None
    secs_to_open: int | None = None
    if is_open:
        try:
            secs_to_close = market_module.seconds_until_close(now)
        except Exception:
            secs_to_close = None
    else:
        try:
            nxt_open = market_module.next_session_open(now)
        except Exception:
            nxt_open = None
        if nxt_open is not None:
            try:
                secs_to_open = max(0, int((nxt_open - now).total_seconds()))
            except Exception:
                secs_to_open = None
    headline = _build_headline(phase, is_open, is_half_day,
                                secs_to_close, secs_to_open)
    return {
        "phase": phase,
        "is_open": is_open,
        "is_half_day": is_half_day,
        "secs_to_close": secs_to_close,
        "secs_to_open": secs_to_open,
        "headline": headline,
    }
