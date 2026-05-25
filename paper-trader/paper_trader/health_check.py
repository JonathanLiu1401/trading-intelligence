"""Unified live-trader health check — fuse offline preflight + online
should-restart into the single command an operator actually wants to run.

The motivating gap
------------------
The repo has two adjacent but separate checks the operator must run in
sequence to get a complete picture:

  * ``python3 -m paper_trader.preflight`` (offline) — loop liveness,
    NO_DECISION reliability, news-feed freshness, runner-process presence.
    Works exactly when the Flask dashboard does NOT (the dashboard is the
    thing most likely to be broken — AGENTS.md "Common failure modes").
  * ``python3 -m paper_trader.should_restart`` (online) — dashboard-fed
    fusion of ``/api/supervision`` (stale code) + ``/api/runner-heartbeat``
    (wedged loop / dark Discord / degraded lock) + ``host_guard.pulse()``
    (host saturation). Answers "should I restart, or is this an ops fix?".

These overlap (heartbeat appears in both) and diverge (preflight sees
feed-freshness; should-restart sees supervision/host-pulse). A trader at
the open wants ONE command — not two — and wants the WORST verdict +
the union of remediation actions.

This module is that command. It is a **router, not a grader** (the
``trader_scorecard`` / ``preflight`` precedent — invariants #2/#12,
observational only, no path to ``_execute()``): forwards each child
verdict's wording verbatim, derives the overall state by precedence,
and emits the worst exit code. It mints no metric of its own.

Verdict / exit-code (matching the preflight + should_restart ladder so
``health_check`` is a strict superset, never a contradiction of either):

  * ``HEALTHY``  (0) — everything green
  * ``NO_DATA``  (0) — fresh boot / no signal yet, nothing actionable
  * ``OPS_ONLY`` (2) — should-restart says OPS_ONLY (host saturation —
                       restart alone will NOT fix this)
  * ``DEGRADED`` (2) — preflight DEGRADED (loop alive but a constituent
                       unhappy) and nothing higher reported
  * ``RESTART``  (1) — should-restart says RESTART recommended (loop
                       wedged, Discord dark, lock degraded, supervision
                       stale, …). Mapped to exit 1 — composable in shell
                       guards (``python3 -m paper_trader.health_check ||
                       systemctl --user restart paper-trader``).
  * ``DOWN``     (3) — preflight DOWN (no runner / heartbeat STALLED)

Precedence (most severe wins): ``DOWN`` > ``RESTART`` > ``OPS_ONLY`` ≈
``DEGRADED`` > ``HEALTHY`` > ``NO_DATA``. ``OPS_ONLY`` and ``DEGRADED``
both map to exit 2; the headline shows whichever fired first so the
operator sees the more actionable narrative (an OPS finding has a
concrete remediation; a generic DEGRADED is informational).

Never raises: every constituent is wrapped — a fault inside preflight
or should_restart degrades to a NO_DATA constituent, never an
exception. A health-check command that crashes is worse than useless
to a trader at the open (the ``preflight`` / ``should_restart``
docstrings carry the same contract).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from . import preflight, should_restart


# Exit-code map. Matches each child's own convention so the fused
# command never disagrees with the child it's strictly a superset of.
_EXIT = {
    "HEALTHY":  0,
    "NO_DATA":  0,
    "OPS_ONLY": 2,
    "DEGRADED": 2,
    "RESTART":  1,  # composes with shell guards
    "DOWN":     3,
}

# Precedence order — higher wins. DOWN > RESTART > {OPS_ONLY, DEGRADED} >
# HEALTHY > NO_DATA. OPS_ONLY and DEGRADED tied at 2; first-reported wins
# the headline so the operator sees the more specific narrative.
_RANK = {
    "NO_DATA":  0,
    "HEALTHY":  1,
    "DEGRADED": 2,
    "OPS_ONLY": 2,
    "RESTART":  3,
    "DOWN":     4,
}


def _preflight_to_state(pf_result: dict | None) -> str:
    """Map a preflight result dict to a health_check state token."""
    if not isinstance(pf_result, dict):
        return "NO_DATA"
    overall = pf_result.get("overall")
    if overall in ("DOWN", "DEGRADED", "HEALTHY", "NO_DATA"):
        return overall
    return "NO_DATA"


def _should_restart_to_state(sr_result: dict | None) -> str:
    """Map a should_restart verdict dict to a health_check state token.

    should_restart uses {OK, RESTART, OPS_ONLY, ERROR}; we map OK→HEALTHY,
    ERROR→NO_DATA, and pass RESTART / OPS_ONLY through unchanged."""
    if not isinstance(sr_result, dict):
        return "NO_DATA"
    state = sr_result.get("state")
    if state == "OK":
        return "HEALTHY"
    if state == "ERROR":
        return "NO_DATA"
    if state in ("RESTART", "OPS_ONLY"):
        return state
    return "NO_DATA"


def _worse(a: str, b: str) -> str:
    """Return the state with the higher (worse) rank; ties keep ``a`` so
    first-reported wins the headline (matches preflight._worse semantics)."""
    return a if _RANK.get(a, 0) >= _RANK.get(b, 0) else b


def build_health_check(*,
                       preflight_result: dict | None,
                       should_restart_result: dict | None,
                       now: datetime | None = None) -> dict:
    """Pure router. Forwards each child dict verbatim and maps the worst
    to a single (state, exit_code, headline) triple. Mints no metric.

    Args:
        preflight_result: ``preflight.build_preflight`` / ``run_preflight``
            output, or ``None`` when that probe itself failed.
        should_restart_result: ``should_restart.build_should_restart`` /
            ``gather()``+build output, or ``None`` when that probe failed.

    Returns:
        Dict with ``state``, ``exit_code``, ``headline``,
        ``recommended_action`` (the WORST child's action — restart wins
        over ops), ``preflight`` and ``should_restart`` (the child dicts,
        verbatim), and ``as_of`` (ISO timestamp).
    """
    ts = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")

    pf_state = _preflight_to_state(preflight_result)
    sr_state = _should_restart_to_state(should_restart_result)

    # First-reported wins ties — preflight is checked first.
    state = _worse(pf_state, sr_state)

    # Headline & action follow the state. For OPS_ONLY / RESTART the
    # should_restart child already wrote a concrete sentence — surface it
    # verbatim (single source of truth). For DOWN / DEGRADED the preflight
    # child owns the wording. HEALTHY / NO_DATA use the static lines.
    if state == "DOWN":
        headline = ((preflight_result or {}).get("headline")
                    or "DOWN — the trading loop is not running or is stalled.")
        action = ((preflight_result or {}).get("recommended_action")
                  or "Restart paper-trader.")
    elif state == "RESTART":
        headline = ((should_restart_result or {}).get("headline")
                    or "RESTART RECOMMENDED")
        action = "; ".join((should_restart_result or {}).get("actions") or
                            ["systemctl --user restart paper-trader"])
    elif state == "OPS_ONLY":
        headline = ((should_restart_result or {}).get("headline")
                    or "OPS — host saturation (a restart alone will not fix this).")
        action = "; ".join((should_restart_result or {}).get("actions") or
                            ["reduce concurrent Opus load"])
    elif state == "DEGRADED":
        headline = ((preflight_result or {}).get("headline")
                    or "DEGRADED — a check is unhappy.")
        action = ((preflight_result or {}).get("recommended_action")
                  or "Review the drivers; a restart may help.")
    elif state == "HEALTHY":
        headline = ("HEALTHY — preflight green and should-restart says "
                    "nothing to do. Trust the trader.")
        action = "None — the desk is healthy."
    else:  # NO_DATA
        headline = ("NO_DATA — neither check produced an actionable verdict "
                    "(fresh boot, dashboard unreachable, or DB empty).")
        action = ("Re-run health-check after the first decision cycle; "
                  "if the dashboard is unreachable, check that the runner "
                  "is running.")

    return {
        "as_of": ts,
        "state": state,
        "exit_code": _EXIT.get(state, 0),
        "headline": headline,
        "recommended_action": action,
        "preflight": preflight_result,
        "should_restart": should_restart_result,
    }


def run_health_check() -> dict:
    """Wire the two children and route. Never raises — a child fault
    becomes a ``None`` constituent and the router degrades cleanly."""
    try:
        pf = preflight.run_preflight()
    except Exception:
        pf = None
    try:
        sr_inputs = should_restart.gather()
        sr = should_restart.build_should_restart(
            supervision=sr_inputs["supervision"],
            heartbeat=sr_inputs["heartbeat"],
            host_pulse=sr_inputs["host_pulse"],
            dashboard_reachable=sr_inputs["dashboard_reachable"],
        )
    except Exception:
        sr = None
    return build_health_check(preflight_result=pf, should_restart_result=sr)


def _render(verdict: dict) -> str:
    """Compact CLI body — the operator's one-glance scan surface."""
    lines = [f"[health-check] {verdict['state']}: {verdict['headline']}"]
    pf = verdict.get("preflight") or {}
    sr = verdict.get("should_restart") or {}
    if pf:
        lines.append(f"  preflight     : {pf.get('overall', 'NO_DATA')}"
                     + (f" — {pf.get('headline', '')}" if pf.get('headline')
                        else ""))
        for d in (pf.get("drivers") or [])[:3]:
            lines.append(f"    • {d}")
    if sr:
        lines.append(f"  should-restart: {sr.get('state', 'ERROR')}"
                     + (f" — {sr.get('headline', '')}" if sr.get('headline')
                        else ""))
        for r in (sr.get("restart_reasons") or [])[:3]:
            lines.append(f"    • {r}")
        for r in (sr.get("ops_reasons") or [])[:3]:
            lines.append(f"    • (ops) {r}")
    lines.append(f"  action: {verdict['recommended_action']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    verdict = run_health_check()
    if "--json" in argv:
        print(json.dumps(verdict, indent=2, sort_keys=True, default=str))
    else:
        print(_render(verdict))
    return int(verdict["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
