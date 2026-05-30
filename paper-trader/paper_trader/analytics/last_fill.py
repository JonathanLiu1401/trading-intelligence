"""Last-fill timestamp — when did the engine last *execute* (not just *decide*)?

The runner heartbeat answers "is the decision loop alive?" by tracking
``decisions.timestamp`` cadence. ``/api/last-real-decision`` answers "did the
engine produce a HOLD/FILLED/BLOCKED row recently?". Neither answers the
trader's actual question on a multi-hour stale book:

    *"When did the engine last actually pull the trigger?"*

A HOLD is a real decision but the book is unchanged. A NO_DECISION is the
loop's failure mode. Only a FILLED (or HARD_SL / HARD_TP auto-exit) row in
the ``trades`` ledger represents the engine moving money. Two surfaces miss
the wall-clock figure today:

  * ``capital_paralysis.cycles_since_last_fill`` is **cycles**, not seconds.
    Under dynamic_interval a cycle is 60s..90min, so "12 cycles since last
    fill" is uninterpretable as a duration without re-deriving the cadence.
  * The hourly's "Recent trades" block prints the last 5 timestamps but
    leaves the operator to subtract by hand to learn the age of the newest.

This builder takes the trades ledger (newest-first; the
``store.recent_trades`` shape) and a wall clock, returns the latest fill's
age + a verdict ladder. Hourly/daily report consumers can suppress the
``FRESH`` arm so a deciding-and-acting desk produces no extra noise
(silence-when-nothing-actionable — the ``_hold_discipline_line`` precedent),
and surface only ``STATIC`` / ``FROZEN`` when the book has been static long
enough to warrant operator attention.

Pure & offline: the caller owns the ``store.recent_trades`` read and the
wall clock (the ``runner_heartbeat`` "network in the caller, builder is
pure" split). Same advisory-only invariant: a verdict on the trades ledger,
no caps, no gating, no path to ``_execute`` (AGENTS.md invariants #2/#12 —
the ``feed_health`` / ``self_review`` precedent).

**Verdict ladder** (advisory only, never gates):

  * ``NO_DATA``  — the ledger is empty (a fresh-boot book that has never
    traded). The hourly suppresses this — the engine hasn't had a chance to
    act yet, so it isn't an "engine frozen" signal.
  * ``FRESH``    — most recent fill is within ``FRESH_HOURS`` (6h). Engine
    is actively trading; the hourly suppresses this.
  * ``STATIC``   — fill age between ``FRESH_HOURS`` and ``FROZEN_HOURS``
    (6h..36h). The book has been static through at least one full trading
    session without action. The hourly surfaces this.
  * ``FROZEN``   — fill age past ``FROZEN_HOURS`` (36h+). The engine has
    sat on its current book for over a trading day-and-a-half. The hourly
    surfaces this with an attention marker — the engine may be alive
    (producing HOLD rows) but it is not *acting* on the market.

Never raises — an unparseable timestamp degrades to ``NO_DATA`` and the
hourly suppresses the line, byte-identical to a fresh book.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Cadence thresholds — owned by this module (the ``runner_heartbeat`` /
# ``feed_health`` precedent: the module is the spec, tests read these
# constants so a retune cannot false-fail). Tuned to one full NYSE
# trading session: 6h covers the bulk of a single session (09:30-16:00 ET
# is 6.5h), so a "FRESH" verdict means the engine has acted within today's
# session. 36h covers a full session + one closed overnight, so a "FROZEN"
# verdict means the engine has not traded for at least a full trading day
# (and a half) — the actionable threshold for an operator review.
FRESH_HOURS = 6.0
FROZEN_HOURS = 36.0


def _humanize(secs: float | None) -> str:
    """Compact age token. Mirrors ``runner_heartbeat._humanize`` /
    ``reporter._format_elapsed``. ``None`` / negative → ``""`` so callers
    can suppress the token entirely. Sub-minute → ``0m`` so a 30s gap
    reads as ``0m``, not ``""``."""
    if secs is None:
        return ""
    try:
        s = float(secs)
    except (TypeError, ValueError):
        return ""
    if s < 0:
        return ""
    s_i = int(s)
    if s_i < 3600:
        return f"{s_i // 60}m"
    if s_i < 86400:
        h, m = divmod(s_i, 3600)
        return f"{h}h{m // 60}m"
    d, rem = divmod(s_i, 86400)
    return f"{d}d{rem // 3600}h"


def _parse_iso(ts: str | None) -> datetime | None:
    """Best-effort ISO-8601 parse. Returns ``None`` on missing/garbage so
    the verdict ladder degrades to ``NO_DATA``."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def build_last_fill(
    trades_newest_first: list[dict] | None,
    now: datetime | None = None,
    *,
    fresh_hours: float = FRESH_HOURS,
    frozen_hours: float = FROZEN_HOURS,
) -> dict:
    """Compute the last-fill verdict + a compact headline for the trades
    ledger. ``trades_newest_first`` is the ``store.recent_trades`` shape
    (the most recent trade is index 0). ``now`` defaults to the real
    wall clock (UTC tz-aware).

    Returns a stable shape:
      * ``state``        — one of ``NO_DATA`` / ``FRESH`` / ``STATIC`` /
                            ``FROZEN``
      * ``headline``     — one-sentence summary (empty on ``NO_DATA``)
      * ``ticker``       — the symbol of the most recent fill (None on
                            ``NO_DATA``)
      * ``action``       — the action verb verbatim (BUY / SELL /
                            BUY_CALL / SELL_PUT / …) — None on ``NO_DATA``
      * ``qty``          — quantity from the trade row (None on ``NO_DATA``)
      * ``price``        — fill price (None on ``NO_DATA``)
      * ``value``        — fill notional (None on ``NO_DATA``)
      * ``last_fill_ts`` — ISO timestamp of the most recent fill
                            (None on ``NO_DATA``)
      * ``secs_since``   — wall-clock seconds since the fill (None on
                            ``NO_DATA`` or unparseable ts)
      * ``age``          — humanised age (``"6h32m"``); empty on ``NO_DATA``
      * ``reason``       — the ``reason`` field of the fill (a free-text
                            tag like ``HARD_TP``, useful to distinguish
                            engine-driven from auto-exit fills); None
                            when absent

    Failure contract: never raises. An empty / non-iterable / corrupt
    ledger degrades to ``NO_DATA``. An unparseable ``timestamp`` on the
    head trade ALSO degrades to ``NO_DATA`` — the trade clearly happened
    but we cannot say *when*, so a duration-based verdict would be
    arbitrary and operator-misleading.
    """
    base: dict = {
        "state": "NO_DATA",
        "headline": "",
        "ticker": None,
        "action": None,
        "qty": None,
        "price": None,
        "value": None,
        "last_fill_ts": None,
        "secs_since": None,
        "age": "",
        "reason": None,
    }

    if not trades_newest_first:
        base["headline"] = "no fills yet — the engine has never executed a trade"
        return base
    try:
        head = trades_newest_first[0]
    except (TypeError, IndexError):
        return base
    if not isinstance(head, dict):
        return base

    ts_str = head.get("timestamp")
    ts = _parse_iso(ts_str)
    if ts is None:
        return base

    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    secs = max(0.0, (now_utc - ts).total_seconds())
    age = _humanize(secs)

    # Coerce numeric fields defensively so a torn row never crashes the
    # builder. The dashboard reads these verbatim.
    def _coerce_float(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    qty = _coerce_float(head.get("qty"))
    price = _coerce_float(head.get("price"))
    value = _coerce_float(head.get("value"))
    ticker = (head.get("ticker") or "").upper() or None
    action = (head.get("action") or "").upper() or None
    reason = head.get("reason") or None

    hours = secs / 3600.0
    if hours < fresh_hours:
        state = "FRESH"
        headline = (
            f"last fill {age} ago — engine actively trading "
            f"(latest: {action} {ticker})"
        )
    elif hours < frozen_hours:
        state = "STATIC"
        headline = (
            f"book static {age} since last fill ({action} {ticker}) — "
            f"engine has not acted since"
        )
    else:
        state = "FROZEN"
        headline = (
            f"book FROZEN {age} since last fill ({action} {ticker}) — "
            f"engine has not executed in over {int(frozen_hours)}h"
        )

    return {
        **base,
        "state": state,
        "headline": headline,
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "price": price,
        "value": value,
        "last_fill_ts": ts_str if isinstance(ts_str, str) else ts.isoformat(),
        "secs_since": round(secs, 1),
        "age": age,
        "reason": reason,
    }
