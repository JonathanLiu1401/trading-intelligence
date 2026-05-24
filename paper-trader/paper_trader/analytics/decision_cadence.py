"""Decision-cadence advisor — what tier is the dynamic-interval logic in
RIGHT NOW, and when is the bot's next decision cycle expected?

The runner's sleep cadence is computed dynamically by
``analytics.dynamic_interval.compute_interval``: six tiers from
``EARNINGS_WINDOW`` (120s) at the top through ``QUIET_CLOSED`` (5400s) at
the bottom, picked from the held book + earnings calendar + NY clock. The
tier is logged to stdout but is NOT structurally exposed anywhere — a
trader watching the dashboard sees ``/api/state`` (cash + positions) and
``/api/last-real-decision`` (when it last acted) but has no surface that
answers *"so when is it going to TRY again?"*. This builder closes that
gap by surfacing:

  * ``tier``       — the dynamic-interval tier name the runner would pick
                     for the NEXT cycle if asked NOW
                     (EARNINGS_WINDOW / SESSION_OPEN / EARNINGS_DAY /
                     MARKET_OPEN / MARKET_CLOSED / QUIET_CLOSED)
  * ``sleep_s``    — the cadence (seconds) that tier maps to
  * ``last_decision_ts``      — most recent decisions row's ISO timestamp
                                (any verb — NO_DECISION counts here; this
                                surface measures the LOOP cadence, not
                                the engine's hit rate)
  * ``since_last_decision_s`` — seconds since that timestamp (or None on
                                a fresh book / unparseable ts)
  * ``next_decision_eta_s``   — best-effort countdown to the expected
                                next cycle (last_ts + sleep_s − now,
                                clamped >= 0); ``None`` when no last_ts
  * ``next_decision_expected_at`` — ISO of (last_ts + sleep_s), or None
  * ``is_overdue``            — True iff since_last_decision_s exceeds
                                ``OVERDUE_MULT × sleep_s`` (loop wedged
                                well past the cadence window — the same
                                stallness signal ``/api/runner-heartbeat``
                                / ``/api/last-real-decision`` use, but
                                tier-aware so an EARNINGS_WINDOW cadence
                                is correctly tight and a QUIET_CLOSED is
                                correctly loose)
  * ``verdict``               — single-bucket panel signal:
                                ``NO_DATA`` (no decisions row),
                                ``ON_SCHEDULE`` (within sleep_s window),
                                ``ELAPSED_NORMAL`` (between 1× and
                                OVERDUE_MULT × sleep_s — the cycle is
                                imminent),
                                ``OVERDUE`` (past OVERDUE_MULT — wedged)
  * ``headline``              — human one-liner the dashboard renders

Pure: NO store writes, NO yfinance / Claude calls, NO subprocess. Reads
only the in-hand ``positions`` list (matching the runner's own
``compute_interval`` input) and the most recent decision timestamp. Uses
the same private predicates / tier constants ``compute_interval`` does so
the trader's view and the runner's actual sleep can NEVER drift.

Observational only — never gates a trade, adds no caps (invariants
#2/#12 — the ``decision_conditionals`` / ``last_real_decision`` precedent).
Failure contract mirrors the rest of analytics: any internal failure
degrades to an ``ERROR`` envelope (verdict + headline), never raises.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Reuse the runner's authoritative tier logic — same predicates, same
# constants — so this surface and the actual runner sleep can never drift.
# The leading-underscore symbols are deliberately internal to
# dynamic_interval, but importing them here is the only honest way to
# preserve the SSOT (re-implementing the logic would defeat the point of
# the surface).
from . import dynamic_interval as _di

# Loop is "overdue" once we are this many tiers past the expected sleep.
# Matches the spirit of ``runner_heartbeat``'s ``LAGGING_MULT`` (1.5) but
# is intentionally a touch looser (2.0): the cadence-vs-realised-cycle gap
# is dominated by host-saturation skips (the documented #1 NO_DECISION
# cause), which routinely add a 30-60s probe overhead even on a clean
# loop. A 2× boundary is the smallest factor that does not fire spuriously
# on a healthy-but-slow cycle. Adjust here, not in scattered constants.
OVERDUE_MULT = 2.0


def _parse_iso(ts) -> datetime | None:
    """Permissive ISO-8601 parser → tz-aware UTC datetime, or None on any
    fault. Treats a naive value as UTC (matches the convention
    ``store.last_real_decision`` / ``signals._age_hours`` use)."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _compute_tier(now_et, positions, events) -> tuple[str, int]:
    """Mirror the ``compute_interval`` ladder. Returns ``(tier, sleep_s)``.

    Re-uses the SAME private predicates dynamic_interval consults so the
    runner's actual sleep and this builder's reported tier cannot drift.
    Falls back to the simple MARKET_OPEN / MARKET_CLOSED arms on any
    internal failure — byte-identical to compute_interval's own fallback
    (never raises)."""
    try:
        held_earnings_today = _di._held_has_earnings_today(
            positions, events,
            now_utc=now_et.astimezone(timezone.utc),
            now_et=now_et,
        )
        if held_earnings_today and _di._is_earnings_window(now_et):
            return "EARNINGS_WINDOW", _di._EARNINGS_WINDOW_S
        if _di._is_session_open_window(now_et):
            return "SESSION_OPEN", _di._SESSION_OPEN_S
        if held_earnings_today:
            return "EARNINGS_DAY", _di._EARNINGS_DAY_S
        if _di._is_market_hours(now_et):
            return "MARKET_OPEN", _di._MARKET_OPEN_S
        if not positions:
            return "QUIET_CLOSED", _di._QUIET_CLOSED_S
        return "MARKET_CLOSED", _di._MARKET_CLOSED_S
    except Exception:
        try:
            if _di._is_market_hours(now_et):
                return "MARKET_OPEN", _di._MARKET_OPEN_S
        except Exception:
            pass
        return "MARKET_CLOSED", _di._MARKET_CLOSED_S


def build_decision_cadence(
    positions: list[dict] | None,
    last_decision_ts: str | None,
    *,
    now: datetime | None = None,
    calendar_path=None,
) -> dict:
    """Build the decision-cadence advisor payload.

    ``positions``           — open positions list (same shape
                              ``compute_interval`` reads — only ``ticker``
                              is required). ``None`` → empty.
    ``last_decision_ts``    — most recent decisions row's ISO timestamp,
                              or None on a fresh book.
    ``now``                 — injectable wall clock (UTC) for tests.
    ``calendar_path``       — explicit earnings snapshot path for tests
                              (else freshest known candidate by mtime).

    Returns the field set documented at module top. Always a dict —
    never raises, even on a malformed timestamp or a corrupt calendar.
    """
    try:
        if now is None:
            now_utc = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now_utc = now.replace(tzinfo=timezone.utc)
        else:
            now_utc = now.astimezone(timezone.utc)
        now_et = now_utc.astimezone(_di._NY)

        pos = positions or []
        events = _di._load_calendar_events(calendar_path)
        tier, sleep_s = _compute_tier(now_et, pos, events)

        last_dt = _parse_iso(last_decision_ts)
        secs_since: int | None = None
        next_eta_s: int | None = None
        next_expected_iso: str | None = None
        is_overdue = False
        verdict = "NO_DATA"
        headline = (
            "No decisions in store yet — the runner has not completed "
            f"its first cycle. Expected cadence: {sleep_s}s ({tier})."
        )

        if last_dt is not None:
            raw = (now_utc - last_dt).total_seconds()
            # Clamp to >=0 so a wall-clock step-back can never render as
            # "negative seconds since last decision" — same hardening
            # ``signals._age_hours`` and ``runner.alarm_latch_state``
            # apply to clock-skew events. A future ``last_dt`` (clock
            # stepped back AFTER the store write) reads as 0s ago.
            secs_since = max(0, int(raw))
            next_expected = last_dt.timestamp() + sleep_s
            next_expected_dt = datetime.fromtimestamp(
                next_expected, tz=timezone.utc,
            )
            next_expected_iso = next_expected_dt.isoformat(timespec="seconds")
            eta_raw = next_expected - now_utc.timestamp()
            next_eta_s = max(0, int(eta_raw))
            is_overdue = secs_since > sleep_s * OVERDUE_MULT
            if is_overdue:
                verdict = "OVERDUE"
                headline = (
                    f"OVERDUE — last decision {secs_since}s ago, past the "
                    f"{int(sleep_s * OVERDUE_MULT)}s ({OVERDUE_MULT:.1f}×) "
                    f"{tier} cadence boundary. The loop may be wedged."
                )
            elif secs_since > sleep_s:
                verdict = "ELAPSED_NORMAL"
                # `next_eta_s` clamps to 0 once we are past the expected
                # boundary — the cycle is imminent / running right now.
                headline = (
                    f"ELAPSED_NORMAL — last decision {secs_since}s ago "
                    f"({tier} cadence is {sleep_s}s). Next cycle imminent."
                )
            else:
                verdict = "ON_SCHEDULE"
                headline = (
                    f"ON_SCHEDULE — last decision {secs_since}s ago. "
                    f"Next {tier} cycle in {next_eta_s}s."
                )

        return {
            "as_of": now_utc.isoformat(timespec="seconds"),
            "verdict": verdict,
            "headline": headline,
            "tier": tier,
            "sleep_s": sleep_s,
            "overdue_mult": OVERDUE_MULT,
            "overdue_threshold_s": int(sleep_s * OVERDUE_MULT),
            "last_decision_ts": last_decision_ts,
            "since_last_decision_s": secs_since,
            "next_decision_expected_at": next_expected_iso,
            "next_decision_eta_s": next_eta_s,
            "is_overdue": is_overdue,
            "n_positions": len(pos),
            "market_open": _di._is_market_hours(now_et),
        }
    except Exception as e:
        # Never raise — the endpoint is a side-channel; degrading to ERROR
        # is the right move so a bad input cannot 500 the whole dashboard.
        return {
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "verdict": "ERROR",
            "headline": f"decision-cadence error: {e}",
            "tier": None,
            "sleep_s": None,
            "overdue_mult": OVERDUE_MULT,
            "overdue_threshold_s": None,
            "last_decision_ts": last_decision_ts,
            "since_last_decision_s": None,
            "next_decision_expected_at": None,
            "next_decision_eta_s": None,
            "is_overdue": False,
            "n_positions": None,
            "market_open": None,
        }


def is_cadence_overdue(report: dict | None) -> bool:
    """Single-bool surface (mirrors ``decision_conditionals.is_intents_stale``
    / ``failed_run_audit.is_failed_runs_hidden``): fires ONLY on the
    ``OVERDUE`` verdict. Used by future per-cycle ledger rows so the
    overdue state can be a single column without re-deriving the whole
    builder."""
    if not isinstance(report, dict):
        return False
    return report.get("verdict") == "OVERDUE"
