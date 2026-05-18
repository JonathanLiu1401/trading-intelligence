"""One-shot operator verdict: "should I restart this trader right now?"

Composes the existing verdict builders the rest of the codebase already
produces — `/api/supervision` (stale code / no safety net),
`/api/runner-heartbeat` (wedged decision loop, dark Discord channel,
degraded singleton lock) and `host_guard.pulse()` (host saturation) — into
ONE actionable answer printable from a plain shell on the same box.

Why this exists
---------------
The codebase has ~30 endpoints diagnosing the live pathologies (stale boot,
host saturation, NO_DECISION storms, Discord dark, double-trade). The
operator who lives in Discord has to *fuse them* by hand:
  • the runner heartbeat says STALLED;
  • supervision says STALE (committed fix not deployed);
  • host-guard says SATURATED (review agents starving the box).
"Should I restart?" is the same answer every time — but the operator has to
ask three different endpoints to reach it. This is the fusion they keep
doing manually.

Pure ``build_should_restart`` consumes already-built verdict dicts; a tiny
CLI fetches them from the live dashboard (with degrade-safe fallbacks when
the dashboard is itself unreachable) and prints the single-line answer.
Exits ``1`` when a restart is recommended (composable in shell guards:
``python3 -m paper_trader.should_restart || systemctl --user restart paper-trader``).

The CLI is **read-only**: never restarts anything, never writes anywhere.
Operator retains every action — this just collapses the diagnosis. Mirrors
the established ``python3 -m paper_trader.host_guard`` / ``python3 -m
paper_trader.signals --check-freshness`` pattern.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

from . import host_guard

DASHBOARD_URL = "http://localhost:8090"
_HTTP_TIMEOUT_S = 5.0


def _fetch_json(url: str, timeout: float = _HTTP_TIMEOUT_S) -> dict | None:
    """GET ``url`` and decode JSON, or ``None`` on any failure.

    Used by the CLI to consult the live dashboard. Degrade-safe by design:
    the whole point of this command is to work when things are broken, so
    every failure path resolves to a sensible default (a dashboard that
    won't answer IS a signal — the runner is likely down)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            json.JSONDecodeError, UnicodeDecodeError):
        return None


def build_should_restart(*,
                          supervision: dict | None,
                          heartbeat: dict | None,
                          host_pulse: dict | None,
                          dashboard_reachable: bool,
                          now: datetime | None = None) -> dict:
    """Pure verdict over already-computed inputs.

    Args:
        supervision: ``/api/supervision`` payload (or builder output). May be
            ``None`` when the endpoint is unreachable.
        heartbeat:   ``/api/runner-heartbeat`` payload. May be ``None``.
        host_pulse:  ``host_guard.pulse()`` snapshot. May be ``None`` on a
            degrade-safe pulse fault (the contract is "never raise").
        dashboard_reachable: True when the dashboard answered any request.
            The dashboard runs inside the runner process — when it is dark
            for an extended period the runner is almost certainly down and
            a restart is the correct response regardless of what any other
            input would have said.

    Returns:
        ``{state, headline, restart_recommended, restart_reasons,
        ops_reasons, actions, exit_code, as_of}``.

      * ``state``: ``OK`` / ``RESTART`` / ``OPS_ONLY`` / ``ERROR``
      * ``restart_recommended``: True iff a restart will help
      * ``restart_reasons``: ordered list of human strings — why restart
      * ``ops_reasons``: ordered list — non-restart actions (host saturation,
        feed pipeline issues — a restart alone will not fix these)
      * ``actions``: ordered remediation steps (ops first when present)
      * ``exit_code``: 0 OK · 1 RESTART · 2 OPS_ONLY · 3 ERROR
    """
    ts = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    out = {
        "as_of": ts,
        "state": "OK",
        "headline": "",
        "restart_recommended": False,
        "restart_reasons": [],
        "ops_reasons": [],
        "actions": [],
        "exit_code": 0,
    }

    restart_reasons: list[str] = []
    ops_reasons: list[str] = []
    actions: list[str] = []

    # ── dashboard unreachable ⇒ trader almost certainly down ─────────────
    # The dashboard thread runs inside the runner process. If it can't be
    # reached at all, the runner is the most likely cause; recommend a
    # restart (systemd will boot it on current code).
    if not dashboard_reachable:
        restart_reasons.append(
            "dashboard at :8090 unreachable — the trader process is likely "
            "down (the dashboard runs inside the runner)")

    # ── supervision (stale / unsupervised) ────────────────────────────────
    if isinstance(supervision, dict):
        verdict = supervision.get("verdict")
        if verdict in ("STALE", "UNSUPERVISED", "UNSUPERVISED_STALE",
                       "UNKNOWN"):
            rec = supervision.get("recommendation") or ""
            restart_reasons.append(f"supervision {verdict} — {rec}")

    # ── runner heartbeat (wedged loop, dark Discord, degraded lock) ──────
    if isinstance(heartbeat, dict):
        if heartbeat.get("restart_recommended"):
            hb_headline = heartbeat.get("headline") or ""
            eff = heartbeat.get("decision_efficacy") or {}
            eff_headline = (eff.get("headline") or "") if isinstance(eff, dict) else ""
            tail = f" — {eff_headline}" if eff_headline else ""
            restart_reasons.append(
                f"runner-heartbeat restart_recommended — {hb_headline}{tail}")
        lock = heartbeat.get("singleton_lock") or {}
        if isinstance(lock, dict) and lock.get("degraded"):
            restart_reasons.append(
                "singleton lock DEGRADED — this trader booted without the "
                "single-instance guard and may be double-trading the paper "
                "book against another runner. Restart so one guarded "
                "instance owns the lock.")
        notify = heartbeat.get("notify") or {}
        if isinstance(notify, dict) and notify.get("restart_recommended"):
            err = notify.get("last_error") or "unknown"
            restart_reasons.append(
                "Discord channel DARK — recent sends failed "
                f"({notify.get('consecutive_failures', '?')} consecutive); "
                "operator monitoring surface is silent. Restart to re-resolve "
                f"openclaw/PATH. (last error: {err})")

    # ── host saturation (OPS — restart will NOT fix this) ────────────────
    if isinstance(host_pulse, dict):
        state = host_pulse.get("state")
        if state in ("SATURATED", "STARVED"):
            ops_reasons.append(host_pulse.get("headline")
                                or f"host_guard.pulse() state={state}")

    # ── compose actions in remediation order ──────────────────────────────
    # OPS first: when both apply, killing out-of-band Opus before the
    # restart prevents the freshly-booted runner from being immediately
    # re-starved on its very first decision cycle.
    if ops_reasons:
        actions.append(
            "(ops) reduce concurrent Opus load — review/backtest agents are "
            "starving the live trader's Opus call; killing them or letting "
            "them finish is the actual fix for the freeze")
    if restart_reasons:
        actions.append("systemctl --user restart paper-trader")

    # ── state ──────────────────────────────────────────────────────────────
    if restart_reasons:
        out["state"] = "RESTART"
        out["restart_recommended"] = True
        out["exit_code"] = 1
        out["headline"] = (
            f"RESTART RECOMMENDED — {restart_reasons[0]}"
            + (f" (+ {len(restart_reasons) - 1} more)"
               if len(restart_reasons) > 1 else "")
        )
    elif ops_reasons:
        out["state"] = "OPS_ONLY"
        out["restart_recommended"] = False
        out["exit_code"] = 2
        out["headline"] = (
            f"OPS — {ops_reasons[0]} (a restart alone will not fix this)"
        )
    elif supervision is None and heartbeat is None and host_pulse is None:
        out["state"] = "ERROR"
        out["restart_recommended"] = False
        out["exit_code"] = 3
        out["headline"] = (
            "ERROR — could not read any verdict source (dashboard, "
            "supervision, host probe all unavailable)")
    else:
        out["headline"] = (
            "OK — supervised, heartbeat healthy, Discord channel up, "
            "host clear; nothing to do.")

    out["restart_reasons"] = restart_reasons
    out["ops_reasons"] = ops_reasons
    out["actions"] = actions
    return out


def gather(dashboard_url: str = DASHBOARD_URL) -> dict:
    """Probe live system for the three verdict inputs.

    Pure I/O — degrade-safe everywhere. Returns the shape ``build_should_restart``
    consumes plus ``dashboard_reachable`` so the builder can tell the
    "dashboard unreachable" branch apart from "endpoint missing" / "ok but
    nothing actionable"."""
    sup = _fetch_json(f"{dashboard_url}/api/supervision")
    hb = _fetch_json(f"{dashboard_url}/api/runner-heartbeat")
    dashboard_reachable = (sup is not None) or (hb is not None)
    # host_guard.pulse() is degrade-safe but disabled under pytest unless
    # explicitly forced — match production by passing no injectors and
    # tolerating an empty dict.
    try:
        hp = host_guard.pulse()
    except Exception:
        hp = None
    return {
        "supervision": sup,
        "heartbeat": hb,
        "host_pulse": hp,
        "dashboard_reachable": dashboard_reachable,
    }


def _render(verdict: dict) -> str:
    """Compact, scannable text body for the CLI."""
    lines = [f"[should-restart] {verdict['headline']}"]
    if verdict["restart_reasons"]:
        lines.append("  why restart:")
        for r in verdict["restart_reasons"]:
            lines.append(f"    • {r}")
    if verdict["ops_reasons"]:
        lines.append("  ops (restart will NOT fix these alone):")
        for r in verdict["ops_reasons"]:
            lines.append(f"    • {r}")
    if verdict["actions"]:
        lines.append("  remediation:")
        for i, a in enumerate(verdict["actions"], 1):
            lines.append(f"    {i}. {a}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    inputs = gather()
    verdict = build_should_restart(
        supervision=inputs["supervision"],
        heartbeat=inputs["heartbeat"],
        host_pulse=inputs["host_pulse"],
        dashboard_reachable=inputs["dashboard_reachable"],
    )
    if "--json" in argv:
        print(json.dumps(verdict, indent=2, sort_keys=True))
    else:
        print(_render(verdict))
    return int(verdict["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
