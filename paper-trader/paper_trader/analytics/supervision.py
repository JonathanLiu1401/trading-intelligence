"""Deployment-recovery health — "if this trader exits, will anything bring it
back?" — as a PURE, single-source-of-truth verdict builder.

Why this module exists
----------------------
The supervision verdict (orphan / stale-code / no-restart-net) was the **#1
recurring HIGH live finding** across many review passes: the live trader runs
as an orphaned ``python3 runner.py`` (PPID 1) with the systemd unit
``disabled``/``inactive`` AND on stale code — i.e. no restart safety net, so
the moment its git-watcher / deadman does ``os._exit(0)`` the trader stays
**DOWN permanently**. The verdict was computed *inline* in
``dashboard.supervision_api`` and surfaced only on ``/api/supervision`` — a
dashboard the operator (who lives in Discord) does not open. There was no
pure builder and no test, so it could not be reused by the reporter without
re-deriving the verdict (an invariant-#10 single-source-of-truth violation
that would let the Discord line and the endpoint silently disagree).

This module extracts the verdict logic verbatim. The endpoint and the
hourly/daily Discord ``_supervision_line`` now both compose ``build_supervision``
so they can never tell different stories. The impure probes (``os.getppid``,
``systemctl --user``, git HEAD/behind) stay in each caller — the established
"network/process in the caller, builder is pure" split this codebase uses for
``build_runner_heartbeat`` / ``_heartbeat_line``.

Contract
--------
``build_supervision`` is pure, deterministic given its inputs (``now``
injectable), and **never raises** — a diagnostics fault must never sink a
live decision cycle or a Discord summary (the ``_safe`` contract shared by
``buying_power`` / ``runner_heartbeat``). Advisory / read-only only: it
observes process + unit state, never gates Opus, never restarts anything,
never adds a position cap (AGENTS.md invariants #2 / #12).
"""
from __future__ import annotations

from datetime import datetime, timezone

# Verdicts that the operator should act on. HEALTHY (supervised + current) is
# the ONLY non-actionable verdict — a healthy supervised trader adds no hourly
# Discord noise (the ``_heartbeat_line`` HEALTHY-suppression precedent).
# UNKNOWN is deliberately actionable: "I can't tell if I have a restart net"
# is operationally closer to UNSUPERVISED than to HEALTHY — the operator
# should manually verify, and the recommendation already states the exact
# `systemctl --user is-active/is-enabled` commands to run. Surfacing it (vs.
# staying silent on an unreadable user bus) is the deliberate call here.
_ACTIONABLE_VERDICTS = frozenset(
    {"STALE", "UNSUPERVISED", "UNSUPERVISED_STALE", "UNKNOWN"}
)


def is_actionable(verdict: str | None) -> bool:
    """Single source of truth for "should this reach Discord?" so the
    reporter never re-derives the suppression rule (invariant #10)."""
    return verdict in _ACTIONABLE_VERDICTS


def build_supervision(*, pid: int | None, ppid: int | None,
                       unit_active: str, unit_enabled: str,
                       boot_sha: str | None, head_sha: str | None,
                       behind: int, now: datetime | None = None,
                       unit_scope: str | None = None) -> dict:
    """Compose the deployment-recovery verdict from already-probed inputs.

    Args (all probed by the caller — this function does NO IO):
      pid / ppid          — ``os.getpid()`` / ``os.getppid()`` of the trader
                            process (the dashboard runs inside the runner, so
                            ``ppid`` is the trader's own parent).
      unit_active         — ``systemctl is-active paper-trader`` →
                            ``active|inactive|failed|unknown``.
      unit_enabled        — ``systemctl is-enabled paper-trader`` →
                            ``enabled|disabled|static|unknown``.
      boot_sha            — git SHA captured at process import time.
      head_sha / behind   — current git HEAD + commit distance from boot.
      unit_scope          — ``system`` / ``user`` when the caller can prove
                            this process lives inside the matching unit cgroup.
      now                 — injectable clock for the ``as_of`` stamp.

    Returns the exact dict shape ``/api/supervision`` has always returned,
    plus an ``actionable`` flag (single-sourced suppression rule). Verdict
    (advisory only — never gates or restarts anything):
      HEALTHY            — supervised and running current code
      STALE              — supervised but on old code (restart to deploy;
                           safety net present so it will recover)
      UNSUPERVISED       — no restart safety net: a clean exit (git-watcher
                           restart / deadman / crash) leaves the trader DOWN
      UNSUPERVISED_STALE — worst: unsupervised AND already on old code
      UNKNOWN            — could not determine (degrade-safe; never raises)

    A bare ``PPID==1`` is not enough to call the process orphaned: root
    systemd services also have PID 1 as parent. Treat PPID 1 as orphaned only
    when no active+enabled service owns the restart contract."""
    try:
        ts = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        unit_supervised = (unit_active == "active"
                           and unit_enabled == "enabled"
                           and (ppid != 1 or unit_scope in {"system", "user"}))
        orphan = (ppid == 1 and not unit_supervised)
        stale = bool(boot_sha and head_sha and head_sha != boot_sha)

        # "supervised" = this process WILL be auto-restarted if it exits.
        # A true orphan never will. Otherwise require an active+enabled unit
        # (Restart=always in force).
        # Unreadable systemctl on a non-orphan ⇒ supervision indeterminate.
        if orphan:
            supervised: bool | None = False
        elif unit_active == "unknown" or unit_enabled == "unknown":
            supervised = None
        else:
            supervised = unit_supervised

        if supervised is None:
            verdict = "UNKNOWN"
            recommendation = (
                "Could not read systemd user state from inside the process "
                "(user bus may be unreachable). Verify manually: "
                "`systemctl --user is-active paper-trader; "
                "systemctl --user is-enabled paper-trader`.")
        elif supervised:
            if stale:
                verdict = "STALE"
                recommendation = (
                    f"Supervised but running old code (boot {boot_sha} vs "
                    f"head {head_sha}, behind {behind}). `systemctl --user "
                    "restart paper-trader` to deploy the committed fixes.")
            else:
                verdict = "HEALTHY"
                recommendation = "Supervised and current — no action."
        else:
            if stale:
                verdict = "UNSUPERVISED_STALE"
                recommendation = (
                    f"NO restart safety net AND on old code (boot "
                    f"{boot_sha} vs head {head_sha}, behind {behind}). This "
                    "is an orphan / un-managed run; the moment its "
                    "git-watcher or deadman does os._exit(0) the trader "
                    "stays DOWN. Re-attach supervision: `systemctl --user "
                    "enable --now paper-trader` (it boots on current code).")
            else:
                verdict = "UNSUPERVISED"
                recommendation = (
                    "Running current code but with NO restart safety net "
                    "(orphan / unit not active+enabled). A clean exit "
                    "(git-watcher restart, deadman) or crash leaves the "
                    "trader DOWN. `systemctl --user enable --now "
                    "paper-trader`.")

        return {
            "as_of": ts,
            "service": "paper_trader",
            "pid": pid,
            "ppid": ppid,
            "orphan": orphan,
            "systemd": {"active": unit_active, "enabled": unit_enabled},
            "unit_scope": unit_scope,
            "boot_sha": boot_sha,
            "head_sha": head_sha,
            "behind": behind,
            "stale": stale,
            "supervised": supervised,
            "verdict": verdict,
            "recommendation": recommendation,
            "actionable": is_actionable(verdict),
        }
    except Exception as e:
        # The _safe contract: never raise. Degrade to an honest UNKNOWN that
        # the caller can suppress or surface like any other verdict. The
        # fallback as_of NEVER reuses the passed `now` — a malformed clock is
        # itself a way into this branch, so re-touching it would re-raise.
        try:
            fallback_as_of = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
        except Exception:
            fallback_as_of = ""
        return {
            "as_of": fallback_as_of,
            "service": "paper_trader",
            "pid": pid,
            "ppid": ppid,
            "orphan": None,
            "systemd": {"active": unit_active, "enabled": unit_enabled},
            "unit_scope": unit_scope,
            "boot_sha": boot_sha,
            "head_sha": head_sha,
            "behind": behind,
            "stale": None,
            "supervised": None,
            "verdict": "UNKNOWN",
            "recommendation": (
                f"supervision verdict computation failed ({e}); verify "
                "manually: `systemctl --user is-active paper-trader; "
                "systemctl --user is-enabled paper-trader`."),
            "actionable": True,
        }
