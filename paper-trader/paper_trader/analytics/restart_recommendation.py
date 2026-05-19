"""Single operator-actionable verdict: should I restart paper-trader RIGHT NOW?

Closes the false-HEALTHY gap that `desk_pulse.liveness` and the bare
`runner_heartbeat` cadence verdict cannot close on their own. Both of those
verdict ladders answer "is the loop cycling?" — neither composes the
*decision-integrity* signal (parse-fail / empty / host-skipped) with the
*book-at-risk-into-event* exposure into one bit the operator actually needs:
**should I restart the process NOW or later?**

Concrete pathology observed live (2026-05-19): the desk reads
`liveness.restart_recommended: false` because the runner is cycling every ~18m
in market-closed mode, but `/api/empty-claude-rate` shows 81% of the last 59
cycles never reached Opus and `/api/earnings-risk` shows $445 of NVDA exposure
into an imminent print. The operator surface is HEALTHY when the truth is
BLIND-and-EXPOSED.

This builder is pure — it takes already-computed scalars and returns the
composite. The endpoint owns the I/O (the documented thesis_drift split). The
verdict ladder is precedence-ordered: first match wins, so a single 81%+
empty rate with held-imminent exposure dominates every other state. Advisory
only — never gates Opus, adds no caps (AGENTS.md invariants #2 / #12).

Restart-recommendation precedence
---------------------------------
- RESTART_URGENT  — empty_rate >= URGENT_EMPTY_RATE and there is held-imminent
                   exposure within URGENT_HOURS hours. The exact "BLIND into
                   the print" wedge that prompted this surface.
- RESTART_RECOMMENDED — IDLE_STORM (>= IDLE_STORM_N consecutive NO_DECISION,
                   matching `runner_heartbeat`), OR moderate empty_rate with
                   any held-imminent exposure.
- MONITOR        — host_saturated, OR mild empty_rate with held exposure,
                   OR a shorter consecutive-no-decision streak. Operator
                   should look but a restart isn't yet justified.
- OK             — none of the above.

Thresholds are module constants so tests can pin them and the future
operator can tune without changing the structure.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Precedence thresholds. Tunable but tested for stability.
URGENT_EMPTY_RATE = 60.0          # %  empty-claude rate over the 24h window
RECOMMENDED_EMPTY_RATE = 50.0     # %  empty-claude rate with any held event
MONITOR_EMPTY_RATE = 30.0         # %  empty-claude rate with any held event
URGENT_HOURS = 24.0               # held-event proximity that flips URGENT
RECOMMENDED_HOURS = 48.0          # held-event proximity for RECOMMENDED
IDLE_STORM_N = 5                  # ties to runner_heartbeat's IDLE_STORM gate
MONITOR_NO_DECISION_N = 3         # softer streak that warrants attention

_NEXT_CHECK_BY_VERDICT = {
    "RESTART_URGENT": 60,
    "RESTART_RECOMMENDED": 180,
    "MONITOR": 300,
    "OK": 900,
}


def _coerce_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _coerce_int(x, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def build_restart_recommendation(
    empty_rate_pct: float | None,
    host_saturated: bool | None,
    held_imminent_exposure_usd: float | None,
    hours_to_nearest_held_event: float | None,
    consecutive_no_decision: int | None,
    now: datetime | None = None,
) -> dict:
    """Compose the single restart-now bit + reason bundle.

    Parameters
    ----------
    empty_rate_pct : float | None
        Trailing 24h "claude returned no response" rate as a percentage. From
        ``/api/empty-claude-rate`` (or equivalent caller-supplied scalar).
        ``None`` is treated as "unknown" — pulled out of every gate, never
        flips a verdict on its own.
    host_saturated : bool | None
        Whether the host has too many concurrent ``claude`` subprocesses.
        From ``/api/host-guard``. ``None`` ⇒ "unknown".
    held_imminent_exposure_usd : float | None
        Dollar exposure on held positions reporting earnings within the
        ``recommended`` horizon. From ``/api/earnings-risk``. ``None`` or
        non-finite ⇒ 0.
    hours_to_nearest_held_event : float | None
        Hours until the soonest held-imminent print. ``None`` ⇒ no held
        event in the horizon.
    consecutive_no_decision : int | None
        Trailing run of newest-first ``decisions.action_taken == NO_DECISION``
        (the ``runner_heartbeat`` decision-efficacy input).
    now : datetime, optional
        Test injection.

    Returns
    -------
    JSON-ready dict with the verdict, scalar urgency, reason bundle, the
    suggested re-poll cadence, and an ``inputs`` transparency block.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    er = _coerce_float(empty_rate_pct, default=0.0) if empty_rate_pct is not None else None
    exposure = _coerce_float(held_imminent_exposure_usd, default=0.0)
    hours_to_event: float | None
    if hours_to_nearest_held_event is None:
        hours_to_event = None
    else:
        try:
            hours_to_event = float(hours_to_nearest_held_event)
        except (TypeError, ValueError):
            hours_to_event = None
    cnd = _coerce_int(consecutive_no_decision, default=0)
    saturated = bool(host_saturated) if host_saturated is not None else None

    has_imminent = (exposure > 0.0
                    and hours_to_event is not None
                    and hours_to_event <= RECOMMENDED_HOURS)
    has_urgent_event = (exposure > 0.0
                        and hours_to_event is not None
                        and hours_to_event <= URGENT_HOURS)

    reasons: list[str] = []

    # Precedence ladder.
    verdict = "OK"
    restart_now = False
    urgency = 0.0

    if (er is not None and er >= URGENT_EMPTY_RATE and has_urgent_event):
        verdict = "RESTART_URGENT"
        restart_now = True
        urgency = 1.0
        reasons.append(
            f"bot empty {er:.0f}% of last 24h cycles "
            f"with ${exposure:.0f} into earnings in "
            f"{hours_to_event:.1f}h — likely BLIND into the print")
    elif cnd >= IDLE_STORM_N:
        verdict = "RESTART_RECOMMENDED"
        restart_now = True
        urgency = 0.85 if has_imminent else 0.75
        reasons.append(
            f"{cnd} consecutive NO_DECISION cycles — engine cycling "
            f"but not deciding, restart clears a wedged Claude CLI")
        if has_imminent:
            reasons.append(
                f"${exposure:.0f} held-imminent exposure in "
                f"{hours_to_event:.1f}h compounds the wedge")
    elif (er is not None and er >= RECOMMENDED_EMPTY_RATE and has_imminent):
        verdict = "RESTART_RECOMMENDED"
        restart_now = True
        urgency = 0.8
        reasons.append(
            f"bot empty {er:.0f}% of last 24h cycles with "
            f"${exposure:.0f} into earnings in {hours_to_event:.1f}h")
    elif (er is not None and er >= MONITOR_EMPTY_RATE and has_imminent):
        verdict = "MONITOR"
        urgency = 0.55
        reasons.append(
            f"bot empty {er:.0f}% of last 24h cycles with "
            f"${exposure:.0f} into earnings in {hours_to_event:.1f}h "
            f"— monitor for further degradation")
    elif saturated is True:
        verdict = "MONITOR"
        urgency = 0.45
        reasons.append(
            "host saturated by concurrent Opus subprocesses — "
            "live decisions are being skipped, kill out-of-band "
            "review/backtest jobs")
        if has_imminent:
            reasons.append(
                f"${exposure:.0f} held-imminent exposure in "
                f"{hours_to_event:.1f}h")
    elif cnd >= MONITOR_NO_DECISION_N:
        verdict = "MONITOR"
        urgency = 0.35
        reasons.append(
            f"{cnd} consecutive NO_DECISION cycles — still under the "
            f"{IDLE_STORM_N}-cycle restart trigger but worth a look")
    else:
        verdict = "OK"
        urgency = 0.0
        if er is not None:
            reasons.append(
                f"empty rate {er:.0f}% over 24h, no held-imminent "
                f"exposure that elevates this to a restart trigger")
        else:
            reasons.append(
                "no parse-fail or host-saturation signal exceeds "
                "monitor threshold")

    if verdict == "OK":
        headline = "OK — no restart trigger active."
    elif verdict == "MONITOR":
        headline = f"MONITOR — {reasons[0]}"
    elif verdict == "RESTART_RECOMMENDED":
        headline = f"RESTART RECOMMENDED — {reasons[0]}"
    else:  # RESTART_URGENT
        headline = f"URGENT — restart paper-trader: {reasons[0]}"

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "verdict": verdict,
        "restart_now": restart_now,
        "urgency_score": round(urgency, 2),
        "headline": headline,
        "reasons": reasons,
        "next_check_seconds": _NEXT_CHECK_BY_VERDICT[verdict],
        "inputs": {
            "empty_rate_pct": (round(er, 1) if er is not None else None),
            "host_saturated": saturated,
            "held_imminent_exposure_usd": round(exposure, 2),
            "hours_to_nearest_held_event": (
                round(hours_to_event, 2) if hours_to_event is not None else None),
            "consecutive_no_decision": cnd,
        },
        "thresholds": {
            "urgent_empty_rate_pct": URGENT_EMPTY_RATE,
            "recommended_empty_rate_pct": RECOMMENDED_EMPTY_RATE,
            "monitor_empty_rate_pct": MONITOR_EMPTY_RATE,
            "urgent_hours": URGENT_HOURS,
            "recommended_hours": RECOMMENDED_HOURS,
            "idle_storm_n": IDLE_STORM_N,
            "monitor_no_decision_n": MONITOR_NO_DECISION_N,
        },
    }
