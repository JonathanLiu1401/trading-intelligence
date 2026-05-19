"""Event-readiness — will the live trader actually be able to react before
the next earnings print?

The system already exposes:

  - ``/api/earnings-risk`` → which held tickers report soon (days_away).
  - ``/api/decision-drought`` → whether the bot is currently paralyzed
    (ongoing NO_DECISION run, kind=PARALYSIS).
  - ``/api/empty-claude-rate`` → rolling Claude-empty fraction (the
    proximate driver of paralysis).

Each one answers a piece of the question, but none answers the operator's
actual question on the eve of a print:

  **"NVDA reports in 16h, my book is 44% NVDA, and the bot is in a 4.7h
  PARALYSIS streak. Is it going to be able to act in time, or am I going to
  walk into the print decision-blind?"**

This module composes the three primitives into a single per-event verdict
(``READY`` / ``DEGRADED`` / ``BLIND``) and a portfolio-level worst-case
verdict, plus an ``expected_decisions_before_event`` estimate (decisions per
hour × hours-until-event × (1 − empty-rate)). It is **pure**: feed it
``open_positions()``, ``recent_decisions(limit)``, and the earnings event
list from digital-intern's ``/api/earnings``; it does not touch the DB or the
network. ``now`` is injectable for tests.

The endpoint wrapper (``dashboard.py::event_readiness_api``) is the I/O seam
— it pulls earnings events from ``:8080/api/earnings`` (like
``/api/earnings-risk`` does) and the decision rows from the live store, then
calls this builder.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


# Per-event verdict thresholds. ``expected`` here is the *successful*
# decision count, not raw cycles — a successful decision is one that
# actually landed an action (FILLED/HOLD/BLOCKED with reasoning) rather
# than NO_DECISION. The thresholds are deliberately CONSERVATIVE because
# (a) most "successful" decisions are HOLDs and (b) a successful decision
# right before the print is far more useful than a successful decision
# at horizon-edge — the expectation alone understates how cornered the
# bot can be near the close. The downgrade rules below further penalize
# an ongoing NO_DECISION streak: a bot historically averaging fine but
# stuck *right now* should not get a READY pass.
_BLIND_EXPECTED_DECISIONS = 5.0
_DEGRADED_EXPECTED_DECISIONS = 30.0

# How far ahead an event must be to be considered "imminent" enough to
# matter for readiness. Past 72h there's plenty of room for the operator to
# manually restart the daemon, swap quota keys, etc.
_IMMINENT_HORIZON_DAYS = 3.0

# Decision-velocity window — how many recent cycles to inspect when
# estimating decisions/hr and Claude-empty rate.
_VELOCITY_WINDOW_HOURS = 6.0

# Current-streak downgrade rules. A bot that historically averages OK
# but is stuck in an *ongoing* NO_DECISION run is being mis-served by a
# pure expected-decisions verdict — that's exactly the live failure mode
# /api/decision-drought catches as PARALYSIS. These mirror the same idea:
# if the most-recent N cycles are all empty, downgrade the verdict.
_STREAK_DEGRADE_CYCLES = 10        # READY → DEGRADED
_STREAK_BLIND_CYCLES = 20          # DEGRADED → BLIND


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _classify(action_taken: str | None) -> str:
    """Mirror analytics/decision_drought.py::_classify — same SSOT bucketing."""
    raw = (action_taken or "").strip().upper()
    if not raw or raw == "NO_DECISION":
        return "NO_DECISION"
    if "FILLED" in raw:
        return "FILLED"
    if "BLOCKED" in raw:
        return "BLOCKED"
    if "HOLD" in raw:
        return "HOLD"
    return "OTHER"


def _decision_velocity(decisions: list[dict],
                       now: datetime,
                       window_hours: float = _VELOCITY_WINDOW_HOURS,
                       ) -> dict:
    """Cycles/hour, NO_DECISION fraction, current NO_DECISION streak, and
    total cycles over the window.

    Decisions are newest-first (``store.recent_decisions`` shape). A bot
    that hasn't logged any cycles in the window is reported as
    cycles_per_hour=0 — the readiness verdict must collapse to BLIND in
    that case (we can't *expect* decisions if none are happening).

    ``current_streak_no_decision`` counts the leading run of NO_DECISION
    rows from newest. This is the *active paralysis* sensor — a bot whose
    6h average looks fine but whose last 30 cycles are all empty is being
    mis-served by the average. Mirrors /api/decision-drought's
    "ongoing PARALYSIS" detection (SSOT-compatible: same _classify
    bucketing as analytics/decision_drought.py)."""
    cutoff = now - timedelta(hours=window_hours)
    n_cycles = 0
    n_empty = 0
    streak = 0
    streak_open = True
    last_non_empty_ts: datetime | None = None

    for d in decisions or []:
        cat = _classify(d.get("action_taken"))
        if streak_open:
            if cat == "NO_DECISION":
                streak += 1
            else:
                streak_open = False
                last_non_empty_ts = _parse_ts(d.get("timestamp"))
        elif last_non_empty_ts is None and cat != "NO_DECISION":
            last_non_empty_ts = _parse_ts(d.get("timestamp"))
        dt = _parse_ts(d.get("timestamp"))
        if dt is None or dt < cutoff:
            continue
        n_cycles += 1
        if cat == "NO_DECISION":
            n_empty += 1

    cycles_per_hour = round(n_cycles / window_hours, 2) if n_cycles else 0.0
    empty_rate = round(n_empty / n_cycles, 3) if n_cycles else 0.0
    streak_minutes: float | None = None
    if streak > 0 and last_non_empty_ts is not None:
        streak_minutes = round(
            (now - last_non_empty_ts).total_seconds() / 60.0, 2)
    return {
        "n_cycles": n_cycles,
        "n_empty": n_empty,
        "cycles_per_hour": cycles_per_hour,
        "empty_rate": empty_rate,
        "current_streak_no_decision": streak,
        "minutes_since_last_real_decision": streak_minutes,
        "window_hours": window_hours,
    }


def _apply_streak_downgrade(base_verdict: str, streak: int) -> str:
    """If the bot is in an active NO_DECISION streak, downgrade verdicts
    that the expected-decisions math would otherwise pass.

    The expected-decisions math is a historical average; the streak is the
    *current* state. They can disagree honestly (good average + bad
    streak), and when they do, the streak wins — that's the live
    PARALYSIS regime /api/decision-drought catches. ``BLIND`` and
    ``IMMINENT_OVERDUE`` are passthrough — they're already the worst the
    ladder allows."""
    if base_verdict in ("BLIND", "IMMINENT_OVERDUE"):
        return base_verdict
    if streak >= _STREAK_BLIND_CYCLES:
        return "BLIND"
    if streak >= _STREAK_DEGRADE_CYCLES and base_verdict == "READY":
        return "DEGRADED"
    return base_verdict


def _expected_decisions(velocity: dict, hours_until_event: float) -> float:
    """cycles/hr × hours_until_event × (1 − empty_rate). Floors at 0."""
    if hours_until_event <= 0:
        return 0.0
    cph = velocity.get("cycles_per_hour") or 0.0
    empty_rate = velocity.get("empty_rate") or 0.0
    success = max(0.0, 1.0 - empty_rate)
    return round(max(0.0, cph * hours_until_event * success), 2)


def _verdict_for(expected: float, hours_until: float) -> str:
    """Map expected_decisions_before_event → readiness verdict.

    A separate ``IMMINENT_OVERDUE`` bucket fires when the event is already
    past — we don't try to predict a verdict for it, just flag it so the
    operator sees the row didn't get filtered."""
    if hours_until <= 0:
        return "IMMINENT_OVERDUE"
    if expected < _BLIND_EXPECTED_DECISIONS:
        return "BLIND"
    if expected < _DEGRADED_EXPECTED_DECISIONS:
        return "DEGRADED"
    return "READY"


def _recommended_action(verdict: str, exposure_usd: float,
                        velocity: dict, hours_until: float) -> str:
    """One-line operator-actionable hint.

    Deliberately conservative — this is a readiness *diagnostic*, not a
    trade directive. ``risk_mirror`` and the system prompt already remind
    Opus that it retains autonomy."""
    empty_rate = velocity.get("empty_rate") or 0.0
    streak = velocity.get("current_streak_no_decision") or 0
    streak_min = velocity.get("minutes_since_last_real_decision")
    if verdict == "BLIND":
        if streak >= _STREAK_BLIND_CYCLES:
            mins = f"{streak_min:.0f}m" if streak_min else "unknown"
            return (
                f"ACTIVE PARALYSIS streak: {streak} consecutive NO_DECISION "
                f"cycles ({mins} since last real decision) — restart paper-"
                "trader (or check the 3-concurrent claude subprocess cap) "
                "before the print, or pre-trim exposure manually")
        if empty_rate >= 0.5:
            return (
                f"Claude is empty {int(empty_rate*100)}% of cycles — "
                "restart paper-trader (or check the 3-concurrent claude "
                "subprocess cap) before the print, or pre-trim exposure")
        return (
            f"only ~{hours_until:.1f}h to print with very low decision "
            "velocity — consider pre-trimming exposure manually")
    if verdict == "DEGRADED":
        if streak >= _STREAK_DEGRADE_CYCLES:
            return (
                f"streak of {streak} NO_DECISION cycles right now even "
                "though the 6h average looks OK — monitor closely; "
                "consider a daemon restart")
        return (
            f"thin decision budget before the print (Claude-empty "
            f"{int(empty_rate*100)}%) — monitor closely; manual trim "
            "is an option if the bot doesn't recover")
    if verdict == "IMMINENT_OVERDUE":
        return "event time has passed — verify earnings calendar freshness"
    return "decision pipeline healthy enough to react"


def _exposure_map(positions: list[dict]) -> dict[str, float]:
    """USD exposure per ticker — options at 100× multiplier (matches the
    existing /api/earnings-risk handler shape, AGENTS.md SSOT)."""
    out: dict[str, float] = {}
    for p in positions or []:
        t = (p.get("ticker") or "").upper()
        if not t:
            continue
        mult = 100 if p.get("type") in ("call", "put") else 1
        price = p.get("current_price")
        if price is None:
            price = p.get("avg_cost") or 0.0
        try:
            qty = float(p.get("qty") or 0.0)
            px = float(price or 0.0)
        except (TypeError, ValueError):
            continue
        out[t] = out.get(t, 0.0) + px * qty * mult
    return out


_VERDICT_RANK = {"BLIND": 0, "IMMINENT_OVERDUE": 1, "DEGRADED": 2, "READY": 3}


def build_event_readiness(positions: list[dict],
                          decisions: list[dict],
                          earnings_events: list[dict],
                          now: datetime | None = None,
                          horizon_days: float = _IMMINENT_HORIZON_DAYS,
                          ) -> dict:
    """Compose earnings-risk + decision-velocity into a single readiness view.

    Pure. ``positions`` matches ``store.open_positions()`` shape;
    ``decisions`` is newest-first per ``store.recent_decisions``;
    ``earnings_events`` is the ``events`` list from
    ``:8080/api/earnings`` (each row has at least ``ticker`` and
    ``days_away`` — ``earnings_date`` carried through verbatim if present)."""
    now = now or datetime.now(timezone.utc)
    velocity = _decision_velocity(decisions or [], now)
    exposure = _exposure_map(positions or [])

    rows: list[dict] = []
    for ev in earnings_events or []:
        if not isinstance(ev, dict):
            continue
        tk = (ev.get("ticker") or "").upper()
        if not tk:
            continue
        days = ev.get("days_away")
        if not isinstance(days, (int, float)) or isinstance(days, bool):
            continue
        if days > horizon_days:
            continue
        exp_usd = round(exposure.get(tk, 0.0), 2)
        if exp_usd <= 0:
            continue
        hours_until = round(days * 24.0, 2)
        expected = _expected_decisions(velocity, hours_until)
        base_verdict = _verdict_for(expected, hours_until)
        streak = velocity.get("current_streak_no_decision") or 0
        verdict = _apply_streak_downgrade(base_verdict, streak)
        rows.append({
            "ticker": tk,
            "earnings_date": ev.get("earnings_date"),
            "days_away": round(float(days), 3),
            "hours_until_event": hours_until,
            "exposure_usd": exp_usd,
            "expected_decisions_before_event": expected,
            "base_verdict": base_verdict,
            "verdict": verdict,
            "recommended_action": _recommended_action(
                verdict, exp_usd, velocity, hours_until),
        })

    rows.sort(key=lambda r: (_VERDICT_RANK.get(r["verdict"], 9),
                             r["hours_until_event"]))

    worst = (rows[0]["verdict"] if rows
             else ("NO_EVENTS" if velocity["n_cycles"] else "NO_DECISIONS"))
    exposure_at_risk = round(
        sum(r["exposure_usd"] for r in rows
            if r["verdict"] in ("BLIND", "DEGRADED")), 2)

    if worst == "NO_EVENTS":
        summary = "no held positions report within %.0fd — readiness moot" % horizon_days
    elif worst == "NO_DECISIONS":
        summary = ("no recent decisions in the velocity window — "
                   "the live trader is not making cycles")
    elif worst == "BLIND":
        summary = (
            "BLIND for %d event(s), $%.0f at risk — "
            "the bot is statistically unlikely to react before the print"
        ) % (sum(1 for r in rows if r["verdict"] == "BLIND"),
             sum(r["exposure_usd"] for r in rows if r["verdict"] == "BLIND"))
    elif worst == "DEGRADED":
        summary = (
            "DEGRADED for %d event(s), $%.0f at risk — "
            "thin decision budget before the print(s)"
        ) % (sum(1 for r in rows if r["verdict"] == "DEGRADED"),
             sum(r["exposure_usd"] for r in rows if r["verdict"] == "DEGRADED"))
    elif worst == "IMMINENT_OVERDUE":
        summary = "event time has passed for at least one held name — calendar may be stale"
    else:  # READY
        summary = (
            "READY for %d event(s) — decision pipeline healthy"
        ) % len(rows)

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "horizon_days": horizon_days,
        "velocity": velocity,
        "events": rows,
        "n_events": len(rows),
        "worst_verdict": worst,
        "exposure_at_risk_usd": exposure_at_risk,
        "summary": summary,
    }
