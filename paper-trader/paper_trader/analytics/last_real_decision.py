"""Last-real-decision age — when did the engine last *produce* a decision?

The companion to ``last_fill`` one dimension over. ``last_fill`` answers
"when did the engine last *execute* (FILLED only)?"; this answers "when did
the engine last produce *any real* decision (FILLED / HOLD / BLOCKED)?". The
two surfaces split an actionable diagnostic apart:

  * ``last_fill`` STATIC + this FRESH → engine is producing HOLD decisions
    (intentional sit-out — healthy). The Discord operator on a pager sees
    "engine deciding to wait", not "engine wedged".
  * ``last_fill`` STATIC + this STALE → engine is producing only
    NO_DECISION rows (claude wedge / quota / host-saturation). The IDLE_STORM
    smoking gun: the loop is cycling (the decisions table grows) but real
    decisions are NOT, and the trader needs to either bounce the runner or
    wait out the quota.

The dashboard endpoint ``/api/last-real-decision`` already exposes this
verdict, but it never reaches the Discord operator — who lives in Discord
and never opens the dashboard panel (the documented dashboard→Discord gap
``_capital_pulse_line`` / ``_host_pulse_line`` / ``_last_fill_line`` each
closed one dimension over). This builder is the SSOT both surfaces consume
so the Discord line and the endpoint can never drift on the same row.

Pure & offline (the ``last_fill`` precedent): the caller owns the
``store.last_real_decision()`` read and the wall clock. Cadence multiples
mirror ``analytics.runner_heartbeat`` — re-imported here would create a
cycle hazard if the heartbeat module ever grows store-touching imports, so
this leaf re-declares the constants. They must STAY in lockstep with
``runner_heartbeat`` (a future divergence would be the bug); a tiny
regression test pins the equality.

Same advisory-only invariant: a verdict on the decisions table, no caps,
no gating, no path to ``_execute`` (AGENTS.md invariants #2/#12 — the
``last_fill`` precedent).

**Verdict ladder** (advisory only, never gates):

  * ``NEVER``       — ``last_real_decision()`` returned ``None`` (a
    fresh-boot book whose first 24h was all NO_DECISION storms; the engine
    has never produced a real decision). The hourly surfaces this — the
    book is alive but has decided nothing.
  * ``FRESH``       — gap ≤ ``LAGGING_MULT`` × expected cadence. Engine is
    deciding normally. The hourly suppresses this.
  * ``DELAYED``     — gap between ``LAGGING_MULT`` and ``STALLED_MULT`` ×
    expected. The engine has decided recently but slower than normal. The
    hourly suppresses this — a quiet hour is not actionable yet.
  * ``STALE``       — gap > ``STALLED_MULT`` × expected. The actionable
    "engine cycling but not producing real decisions" wedge — the hourly
    surfaces this.

Never raises — an unparseable timestamp degrades to ``STALE`` (the dashboard
endpoint precedent: the row clearly exists but we cannot say *when*, so the
operator-safe verdict is "treat as wedged" — false-negative-protective).
"""
from __future__ import annotations

from datetime import datetime, timezone


# Cadence baseline + verdict multipliers. KEEP IN SYNC with
# ``analytics.runner_heartbeat.OPEN_INTERVAL_S`` / ``CLOSED_INTERVAL_S`` /
# ``LAGGING_MULT`` / ``STALLED_MULT`` — a divergence here would silently
# desync the Discord verdict from the dashboard's. The regression test
# in ``tests/test_last_real_decision.py`` pins the equality.
OPEN_INTERVAL_S = 300.0    # mirrors runner.OPEN_INTERVAL_S (market open)
CLOSED_INTERVAL_S = 3600.0  # mirrors runner.CLOSED_INTERVAL_S (market closed)
LAGGING_MULT = 1.25
STALLED_MULT = 2.0


def _humanize(secs: float | None) -> str:
    """Compact age token. Mirrors ``last_fill._humanize`` / ``reporter._format_elapsed``.
    ``None`` / negative / non-numeric → ``""`` so callers can suppress the
    token entirely. Sub-minute clamps to ``0m`` so a 30s gap reads cleanly."""
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
    """Best-effort ISO-8601 parse. Returns ``None`` on missing / garbage."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _extract_verb_ticker_status(action_taken: str | None
                                ) -> tuple[str | None, str | None, str | None]:
    """Best-effort split of the free-text ``decisions.action_taken`` string
    (AGENTS.md invariant #11) into ``(verb, ticker, status)``.

    Shape: ``"<VERB> <TICKER> → <STATUS>"`` (e.g. ``"BUY NVDA → FILLED"``,
    ``"HOLD MU → HOLD"``). Returns ``(None, None, None)`` on any non-matching
    row — pure dict-string parse, never raises."""
    s = (action_taken or "").strip()
    if not s:
        return None, None, None
    head: str
    status: str | None
    if "→" in s:
        head, status_raw = s.split("→", 1)
        status = status_raw.strip() or None
        head = head.strip()
    else:
        head = s
        status = None
    parts = head.split()
    verb = parts[0].upper() if parts else None
    ticker = parts[1].upper() if len(parts) >= 2 else None
    return verb, ticker, status


def build_last_real_decision(
    row: dict | None,
    now: datetime | None = None,
    *,
    market_open: bool = False,
    lagging_mult: float = LAGGING_MULT,
    stalled_mult: float = STALLED_MULT,
    open_interval_s: float = OPEN_INTERVAL_S,
    closed_interval_s: float = CLOSED_INTERVAL_S,
) -> dict:
    """Compute the last-real-decision verdict + a compact headline for the
    given ``store.last_real_decision()`` row.

    Arguments:
      * ``row``         — the ``store.last_real_decision()`` payload (a
                            dict with ``timestamp``, ``action_taken``, …
                            keys; ``None`` ⇒ NEVER).
      * ``now``         — wall clock (UTC tz-aware). Defaults to real time.
      * ``market_open`` — whether the NYSE session is open *now*. Drives
                            which cadence baseline (open / closed) applies.

    Returns a stable shape:
      * ``state``               — ``NEVER`` / ``FRESH`` / ``DELAYED`` / ``STALE``
      * ``headline``            — one-sentence summary (empty on a future-only
                                   ``FRESH`` suppression path; the reporter
                                   suppresses ``FRESH`` and ``DELAYED`` so the
                                   headline is the actionable verdicts only)
      * ``verb``                — action verb (``BUY`` / ``SELL`` / ``HOLD`` /
                                   ``BUY_CALL`` / ``SELL_PUT`` / …) or ``None``
      * ``ticker``              — symbol or ``None``
      * ``status``              — terminal status (``FILLED`` / ``HOLD`` /
                                   ``BLOCKED``) or ``None``
      * ``last_real_ts``        — ISO timestamp of the row or ``None``
      * ``secs_since``          — wall-clock seconds since the row, or ``None``
      * ``age``                 — humanised age (``"42m"`` / ``"3h12m"``) or ``""``
      * ``expected_interval_s`` — applicable cadence baseline (open / closed)
      * ``market_open``         — echo of the input flag (for downstream
                                   diagnostics)

    Failure contract: never raises. A ``None`` row → ``NEVER``. An
    unparseable timestamp → ``STALE`` (operator-safe false-negative-
    protective: the row clearly happened but we can't time it, so treat
    as wedged rather than serve a misleading FRESH). Negative ``secs_since``
    (a wall-clock step-back) clamps to 0 so the verdict ladder is
    monotone."""
    expected = float(open_interval_s if market_open else closed_interval_s)
    base = {
        "state": "NEVER",
        "headline": "",
        "verb": None,
        "ticker": None,
        "status": None,
        "last_real_ts": None,
        "secs_since": None,
        "age": "",
        "expected_interval_s": expected,
        "market_open": bool(market_open),
    }

    if not isinstance(row, dict):
        base["headline"] = (
            "The engine has never produced a real decision "
            "(no FILLED/HOLD/BLOCKED row in history; only NO_DECISION cycles)."
        )
        return base

    ts_str = row.get("timestamp")
    ts = _parse_iso(ts_str)
    verb, ticker, status = _extract_verb_ticker_status(row.get("action_taken"))

    if ts is None:
        # Operator-safe degrade: a real row exists but we cannot time it.
        # Treat as STALE (false-negative-protective) — the dashboard endpoint
        # uses the same arm with the same wording.
        return {
            **base,
            "state": "STALE",
            "headline": (
                f"Real-decision row present but timestamp is unparseable "
                f"({ts_str!r}); treating as STALE."
            ),
            "verb": verb,
            "ticker": ticker,
            "status": status,
            "last_real_ts": ts_str if isinstance(ts_str, str) else None,
        }

    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    secs = max(0.0, (now_utc - ts).total_seconds())
    age = _humanize(secs)

    last_real_ts = ts.isoformat()

    if secs > stalled_mult * expected:
        state = "STALE"
        headline = (
            f"STALE — last real decision was {age} ago, past the "
            f"{_humanize(stalled_mult * expected)} cadence-stall "
            f"threshold. The loop may still be cycling but the engine "
            f"is not actually deciding."
        )
    elif secs > lagging_mult * expected:
        state = "DELAYED"
        headline = (
            f"DELAYED — last real decision was {age} ago, past the "
            f"{_humanize(lagging_mult * expected)} normal cadence. The "
            f"engine is still active but slower than expected."
        )
    else:
        state = "FRESH"
        headline = (
            f"FRESH — last real decision was {age} ago, within the "
            f"{_humanize(expected)} expected cadence."
        )

    return {
        **base,
        "state": state,
        "headline": headline,
        "verb": verb,
        "ticker": ticker,
        "status": status,
        "last_real_ts": last_real_ts,
        "secs_since": round(secs, 1),
        "age": age,
    }
