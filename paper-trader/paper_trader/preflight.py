"""Offline live-trader preflight — "should I trust my trader right now?"

The single morning question a live trader actually asks has no single
answer today. The pieces exist but are scattered and, worse, **most of
them live behind the Flask dashboard that is itself the thing most likely
to be broken**:

  * loop liveness        → ``/api/runner-heartbeat``  (Flask)
  * NO_DECISION health   → ``/api/decision-reliability`` (Flask)
  * news-feed freshness  → ``python3 -m paper_trader.signals --check-freshness``
  * code currency        → ``/api/build-info`` (Flask)

The documented, *recurring* live failure mode (AGENTS.md "Common failure
modes" — `behind:N`, `stale:true`) is precisely that the running ``:8090``
process is stale, in which case **every one of those Flask endpoints 404s
or serves legacy** — the dashboard cannot tell you it is broken because it
*is* broken. There is no offline command that answers "is my trader alive,
on-cadence, and seeing news?" when the dashboard can't.

This module is that command. It is the exact read-only sibling of
``ml/calibration.py`` / ``ml/label_audit.py`` / ``ml/persona_leaderboard.py``
and ``signals --check-freshness``: no Flask, no network, no write, opens
``paper_trader.db`` strictly ``?mode=ro`` (AGENTS.md invariant #7). It is a
**router, not a grader** (the ``trader_scorecard`` precedent — invariants
#2/#12, advisory only, no path to ``_execute()``): it composes the existing
pure builders **verbatim** (single source of truth, invariant #10) —

  * ``analytics.runner_heartbeat.build_runner_heartbeat``  (loop liveness)
  * ``analytics.decision_reliability.build_decision_reliability``  (NO_DEC)
  * ``signals.feed_status``  (feed freshness / split-brain)

— forwards each constituent's *own* verdict and headline unchanged, adds the
one fact none of them can observe offline (is a ``paper_trader.runner``
process actually running, by a Flask-free ``/proc`` scan), and maps the
worst constituent to one ``overall`` verdict + a shell exit code. It mints
no metric of its own.

Verdict / exit-code (the ``signals --check-freshness`` / ``label_audit``
exit-code convention — 0 = nothing to do, 2 = degraded, 3 = act now):

  * ``HEALTHY``  (0) — loop on-cadence, NO_DECISION rate healthy, feed fresh
  * ``NO_DATA``  (0) — no DB / no decisions yet (a fresh boot, not a fault)
  * ``DEGRADED`` (2) — loop LAGGING, or reliability DEGRADED/CRITICAL/
                       restart-recommended, or feed stale/split-brain
  * ``DOWN``     (3) — no runner process **or** heartbeat STALLED: the loop
                       is dead — restart paper-trader

Never raises: every read is wrapped and degrades to ``NO_DATA`` /
"process probe unavailable", never an exception (a preflight that crashes
is worse than useless to a trader at the open).
"""
from __future__ import annotations

import glob
import os
import sqlite3
import sys
from datetime import datetime, timezone

from .analytics.decision_reliability import build_decision_reliability
from .analytics.runner_heartbeat import build_runner_heartbeat
from .store import DB_PATH

# Precedence ranking: a higher number is worse. ``overall`` is the worst
# constituent — the router never invents a level the constituents didn't
# justify (the trader_scorecard "mints no opinion" discipline).
_RANK = {"HEALTHY": 0, "NO_DATA": 0, "DEGRADED": 2, "DOWN": 3}

# The runner is launched either as ``python3 -m paper_trader.runner`` or via
# the systemd unit's ``paper-trader/runner.py``; match either form. Mirrors
# the cmdline pattern AGENTS.md's "Common failure modes" row documents for
# locating the runner's true stdout.
_RUNNER_CMDLINE_MARKERS = ("paper_trader.runner", "paper-trader/runner.py",
                           "paper_trader/runner.py")


def _running_runner_pids() -> list[int] | None:
    """PIDs whose cmdline launches the paper-trader runner — a Flask-free
    ``/proc`` scan so it works exactly when the dashboard does not. Returns
    ``None`` (not ``[]``) when the probe itself is unavailable (no ``/proc``,
    e.g. macOS/CI) so the router can distinguish "provably no runner" from
    "could not tell" and never false-alarm ``DOWN`` on a platform without
    ``/proc``."""
    if not os.path.isdir("/proc"):
        return None
    pids: list[int] = []
    self_pid = os.getpid()
    for cmd_path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            pid = int(cmd_path.split("/")[2])
        except (ValueError, IndexError):
            continue
        if pid == self_pid:
            continue  # never count this preflight process itself
        try:
            with open(cmd_path, "rb") as fh:
                cmdline = fh.read().replace(b"\x00", b" ").decode(
                    "utf-8", "replace")
        except (OSError, IOError):
            continue  # process exited mid-scan / permission — skip, don't die
        if any(m in cmdline for m in _RUNNER_CMDLINE_MARKERS):
            pids.append(pid)
    return pids


def _read_db_ro(db_path=DB_PATH, decisions_limit: int = 3000,
                equity_limit: int = 5000) -> dict | None:
    """Read-only (``?mode=ro``, invariant #7) pull of the rows the two pure
    builders need: decisions newest-first, equity ascending. Returns ``None``
    on any failure (missing DB / locked / schema drift) — the caller renders
    ``NO_DATA``, never crashes."""
    p = str(db_path)
    if not os.path.exists(p):
        return None
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            dec = [dict(r) for r in conn.execute(
                "SELECT * FROM decisions "
                "ORDER BY timestamp DESC, id DESC LIMIT ?",
                (decisions_limit,)).fetchall()]
            eq_desc = [dict(r) for r in conn.execute(
                "SELECT timestamp, total_value, cash, sp500_price "
                "FROM equity_curve ORDER BY timestamp DESC, id DESC LIMIT ?",
                (equity_limit,)).fetchall()]
        finally:
            conn.close()
    except Exception:
        return None
    return {"decisions": dec, "equity_curve": list(reversed(eq_desc))}


def _feed_status_safe() -> dict | None:
    """``signals.feed_status()`` verbatim, but never fatal: a missing
    digital-intern mount must degrade to "feed unknown", not abort the whole
    preflight (the trader still wants the loop/reliability verdicts)."""
    try:
        from . import signals
        return signals.feed_status()
    except Exception:
        return None


def build_preflight(
    runner_pids: list[int] | None,
    heartbeat: dict | None,
    reliability: dict | None,
    feed: dict | None,
    now: datetime | None = None,
) -> dict:
    """Pure router. Forwards each constituent's *own* verdict verbatim and
    maps the worst to ``overall`` + ``exit_code``. Mints no metric. Never
    raises.

    ``runner_pids``: ``None`` ⇒ probe unavailable (do not penalise);
    ``[]`` ⇒ provably no runner process (a ``DOWN`` driver); non-empty ⇒
    alive. ``heartbeat`` / ``reliability`` / ``feed`` are the constituent
    builder dicts (or ``None`` when their input was unavailable).
    """
    now = now or datetime.now(timezone.utc)
    hb_verdict = (heartbeat or {}).get("verdict", "NO_DATA")
    rel_state = (reliability or {}).get("state", "NO_DATA")
    rel_restart = bool((reliability or {}).get("restart_recommended", False))

    drivers: list[str] = []
    level = "NO_DATA"

    # ── runner process liveness (the offline-only fact) ──────────────────
    if runner_pids is None:
        proc_note = "process probe unavailable (no /proc on this platform)"
    elif not runner_pids:
        proc_note = "no paper_trader.runner process found"
        level = "DOWN"
        drivers.append("runner process not running")
    else:
        proc_note = f"runner alive (pid {', '.join(map(str, runner_pids))})"

    # ── loop liveness (heartbeat builder, verbatim verdict) ──────────────
    if hb_verdict == "STALLED":
        level = _worse(level, "DOWN")
        drivers.append(f"heartbeat STALLED: {(heartbeat or {}).get('headline','')}")
    elif hb_verdict == "LAGGING":
        level = _worse(level, "DEGRADED")
        drivers.append(f"heartbeat LAGGING: {(heartbeat or {}).get('headline','')}")
    elif hb_verdict == "HEALTHY":
        level = _worse(level, "HEALTHY")

    # ── NO_DECISION reliability (decision_reliability, verbatim state) ────
    if rel_state in ("CRITICAL", "DEGRADED", "STALE_LEGACY_DOMINATED") or rel_restart:
        level = _worse(level, "DEGRADED")
        drivers.append(
            f"reliability {rel_state}: {(reliability or {}).get('headline','')}")
    elif rel_state == "HEALTHY":
        level = _worse(level, "HEALTHY")

    # ── news-feed freshness (signals.feed_status, verbatim flags) ────────
    if feed:
        if feed.get("split_brain"):
            level = _worse(level, "DEGRADED")
            drivers.append(
                "feed split-brain: a stale process would read a materially "
                "older feed — restart to apply the fresh resolver")
        elif feed.get("stale"):
            level = _worse(level, "DEGRADED")
            ca = feed.get("chosen_age_hours")
            drivers.append(
                f"feed stale: freshest live article is "
                f"{ca:.1f}h old" if isinstance(ca, (int, float))
                else "feed stale: freshest live article is old")
        else:
            level = _worse(level, "HEALTHY")

    headline, action = _summarize(level, drivers)
    return {
        "as_of": now.isoformat(timespec="seconds"),
        "overall": level,
        "exit_code": _RANK[level] if level != "NO_DATA" else 0,
        "headline": headline,
        "recommended_action": action,
        "runner_process": proc_note,
        "runner_pids": runner_pids,
        "drivers": drivers,
        "heartbeat": heartbeat,
        "reliability": reliability,
        "feed": feed,
    }


def _worse(a: str, b: str) -> str:
    return a if _RANK.get(a, 0) >= _RANK.get(b, 0) else b


def _summarize(level: str, drivers: list[str]) -> tuple[str, str]:
    if level == "DOWN":
        return ("DOWN — the trading loop is not running or is stalled; the "
                "trader is making no decisions.",
                "Restart paper-trader (systemctl --user restart paper-trader "
                "or python3 -m paper_trader.runner).")
    if level == "DEGRADED":
        return ("DEGRADED — the loop is alive but a check is unhappy; the "
                "trader is running on weaker footing than it should.",
                "Review the drivers below; a restart applies on-disk fixes "
                "(see /api/build-info `stale`).")
    if level == "NO_DATA":
        return ("NO_DATA — no decisions recorded yet (a fresh boot, not a "
                "fault).", "Re-run preflight after the first decision cycle.")
    return ("HEALTHY — loop on-cadence, NO_DECISION rate healthy, news feed "
            "fresh. Trust the trader.",
            "None — the desk is healthy.")


def run_preflight() -> dict:
    """Wire the read-only DB pull + pure builders + offline process probe.
    The IO/network lives here; ``build_preflight`` stays pure (the
    ``thesis_drift`` / ``runner_heartbeat`` "shell does IO, builder takes
    dicts" split)."""
    now = datetime.now(timezone.utc)
    db = _read_db_ro()
    runner_pids = _running_runner_pids()

    heartbeat = None
    reliability = None
    if db is not None and db["decisions"]:
        # market_open is a pure NYSE-calendar/clock call — no network — so
        # it is safe in this offline tool (mirrors the endpoint wiring).
        try:
            from . import market
            market_open = market.is_market_open(now)
        except Exception:
            market_open = False
        last_ts = db["decisions"][0].get("timestamp")
        try:
            heartbeat = build_runner_heartbeat(last_ts, market_open, now=now)
        except Exception:
            heartbeat = None
        try:
            reliability = build_decision_reliability(
                db["decisions"], db["equity_curve"], now=now)
        except Exception:
            reliability = None

    feed = _feed_status_safe()
    return build_preflight(runner_pids, heartbeat, reliability, feed, now=now)


def _print_report(rep: dict) -> None:
    print("=== paper-trader preflight ===")
    print(f"as of    : {rep['as_of']}")
    print(f"overall  : {rep['overall']}")
    print(f"process  : {rep['runner_process']}")
    hb = rep.get("heartbeat") or {}
    rl = rep.get("reliability") or {}
    fd = rep.get("feed") or {}
    print(f"heartbeat: {hb.get('verdict', 'NO_DATA')}"
          + (f" — {hb.get('headline')}" if hb.get("headline") else ""))
    print(f"reliab.  : {rl.get('state', 'NO_DATA')}"
          + (f" — {rl.get('headline')}" if rl.get("headline") else ""))
    if fd:
        fstate = ("split-brain" if fd.get("split_brain")
                  else "stale" if fd.get("stale") else "fresh")
        ca = fd.get("chosen_age_hours")
        print(f"feed     : {fstate}"
              + (f" (newest live {ca:.1f}h old)"
                 if isinstance(ca, (int, float)) else "")
              + f"  [{fd.get('chosen', '?')}]")
    else:
        print("feed     : unknown (digital-intern DB unreachable)")
    if rep.get("drivers"):
        print("\ndrivers:")
        for d in rep["drivers"]:
            print(f"  - {d}")
    print(f"\n{rep['headline']}")
    print(f"action : {rep['recommended_action']}")


if __name__ == "__main__":
    report = run_preflight()
    _print_report(report)
    sys.exit(int(report.get("exit_code", 0)))
