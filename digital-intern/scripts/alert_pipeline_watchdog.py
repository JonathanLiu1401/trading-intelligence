"""
External alert-pipeline watchdog — escalates a silently-stalled urgent-alert
pipeline to Discord so the consuming analyst is told the moment breaking-news
delivery stops.

WHY THIS EXISTS (live evidence, 2026-05-18)
-------------------------------------------
The daemon supervisor (``daemon.py``) only *respawns* worker threads that have
**exited** — its respawn path is gated on ``if t.is_alive(): continue``. A
worker that is alive but wedged indefinitely (blocked on the shared
``_store_lock`` / sqlite ``busy_timeout`` under the documented heavy
lock-contention + OOM-restart churn) is correctly flagged DEAD in
``logs/supervisor_state.json`` (``alive: false``) but is **never recovered and
never escalated** — it only produces one WARNING line in ``daemon.log``.

This was observed in production: the ``alert`` worker pinged once at daemon
boot, then never again for 15+ minutes while every other worker stayed
healthy. The supervisor could not act (the thread was ``is_alive()``), and the
analyst — who depends on the Bloomberg-style push for breaking events — had
**zero indication** their urgent-alert pipeline had gone dark. A crash-looping
daemon is even worse: it never lives long enough to write the 5-minute health
snapshot at all, so the silence is total.

Python cannot safely kill a wedged thread, and self-restarting the whole
daemon from inside a possibly-wedged supervisor is both unreliable and risky
(it can feed the very restart loop that caused the problem). The robust,
low-risk fix is an **independent process** that watches the artifact the
supervisor leaves behind and shouts when the analyst-critical workers go
silent — exactly the pattern the in-process supervisor already uses for
degraded/disabled transitions (``daemon._notify_state_transition``), but
hoisted out so it survives the supervisor itself being wedged.

INVARIANTS
----------
Read-only and DB-free: this touches NO ``articles`` rows and never reads or
writes ``ai_score`` / ``ml_score`` / ``score_source``. It reads only
``logs/supervisor_state.json`` (the snapshot the daemon already writes for the
dashboard) plus its own throttle file. All four load-bearing pipeline
invariants are untouched by construction.

USAGE
-----
    python3 scripts/alert_pipeline_watchdog.py            # check once + escalate
    python3 scripts/alert_pipeline_watchdog.py --dry-run  # print, do not post

Designed for a cron / systemd-timer cadence of ~2-5 min (independent of the
daemon process) or to be driven by ``monitor.py``. ``evaluate()`` is a pure
function so the decision logic is unit-tested without files, threads, or
Discord.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SUPERVISOR_STATE_PATH = BASE_DIR / "logs" / "supervisor_state.json"
WATCHDOG_STATE_PATH = BASE_DIR / "logs" / ".alert_watchdog_state.json"

# Workers whose silent stall is an analyst-visible outage:
#   alert     — the Bloomberg breaking-news push itself
#   scorer    — nothing gets scored urgent → nothing is ever queued to alert
#   heartbeat — the 5h Opus briefing (the analyst's scheduled digest)
CRITICAL_WORKERS = ("alert", "scorer", "heartbeat")

# The daemon rewrites supervisor_state.json every HEALTH_REPORT_INTERVAL_SECS
# (300s). If it is materially older than that the supervisor loop itself is
# wedged, or the daemon is crash-looping (it never lives long enough to write
# a snapshot) / is down entirely — all of which silence the alert pipeline.
# 3x the report interval tolerates one missed write + scheduling jitter
# without false-positiving on a momentarily slow host.
STALE_SNAPSHOT_S = 3 * 300  # 900s

# Do not re-page for the same still-active condition more often than this.
# Long enough not to spam during a sustained incident, short enough that the
# analyst is reminded the pipeline is still down.
ESCALATION_THROTTLE_S = 30 * 60  # 1800s

# Drop throttle entries this far past their last fire so the state file can
# never grow without bound across restarts.
_STATE_TTL_S = 4 * ESCALATION_THROTTLE_S


def evaluate(
    snapshot: dict | None,
    snapshot_age_s: float | None,
    now_epoch: float,
    throttle_state: dict | None,
) -> tuple[list[str], dict]:
    """Pure decision core. Returns ``(messages, new_throttle_state)``.

    ``snapshot``         parsed supervisor_state.json, or None if the file is
                         missing / unparseable.
    ``snapshot_age_s``   seconds since the file was last written, or None if
                         it does not exist.
    ``now_epoch``        ``time.time()`` (injected so tests are deterministic).
    ``throttle_state``   ``{condition_key: last_fire_epoch}`` from the prior
                         run; may be None on first ever run.

    A condition fires at most once per ``ESCALATION_THROTTLE_S``. When a
    previously-firing condition clears, a one-shot ``✅`` recovery line is
    emitted (mirrors ``daemon._notify_state_transition``) and the entry is
    dropped. Stale entries are TTL-pruned so the state file is bounded.
    """
    prev = dict(throttle_state or {})
    new_state: dict = {}
    messages: list[str] = []
    active: set[str] = set()

    def _fire(key: str, msg: str) -> None:
        active.add(key)
        last = prev.get(key, 0.0)
        if now_epoch - last >= ESCALATION_THROTTLE_S:
            messages.append(msg)
            new_state[key] = now_epoch
        else:
            # Still within the throttle window — keep the original fire time
            # so the throttle is anchored to when the incident *started*.
            new_state[key] = last

    # ── Tier 1: the snapshot itself is missing / stale ───────────────────────
    # This is the worst case for the analyst: a crash-looping or hung daemon
    # never writes a fresh snapshot, so per-worker liveness can't even be
    # assessed. Treat it as a single high-priority condition and stop — the
    # per-worker checks below would be reading stale (lying) data anyway.
    if snapshot is None or snapshot_age_s is None:
        _fire(
            "supervisor_state_missing",
            "🛑 ALERT-PIPELINE WATCHDOG: supervisor_state.json is missing or "
            "unreadable — the digital-intern daemon is down or has never "
            "written a health snapshot. Urgent Discord alerts are NOT being "
            "delivered. Check `systemctl --user status digital-intern`.",
        )
    elif snapshot_age_s > STALE_SNAPSHOT_S:
        mins = snapshot_age_s / 60.0
        _fire(
            "supervisor_state_stale",
            f"🛑 ALERT-PIPELINE WATCHDOG: health snapshot is {mins:.1f} min "
            f"stale (daemon rewrites it every 5 min). The supervisor is "
            f"wedged or the daemon is crash-looping — urgent alerts are NOT "
            f"firing. Check `journalctl --user -u digital-intern`.",
        )
    else:
        # ── Tier 2: snapshot is fresh — check the critical workers ───────────
        workers = {
            w.get("name"): w
            for w in snapshot.get("workers", [])
            if isinstance(w, dict)
        }
        for name in CRITICAL_WORKERS:
            w = workers.get(name)
            if w is None:
                continue
            # ``alive`` is the daemon's OWN computed liveness (last-OK age vs
            # the worker's cadence-scaled deadline) — trust it rather than
            # recomputing the deadline here.
            if w.get("alive", True):
                continue
            age = w.get("last_ok_age_s")
            age_txt = (
                f"{age / 60.0:.1f} min" if isinstance(age, (int, float))
                else "unknown"
            )
            exc = w.get("last_exception") or "none (blocked, not crashed)"
            _fire(
                f"worker:{name}",
                f"🛑 ALERT-PIPELINE WATCHDOG: critical worker `{name}` is "
                f"DEAD/hung (no success ping for {age_txt}, "
                f"state={w.get('state')}, last_exc={exc}). The supervisor "
                f"cannot respawn a still-alive wedged thread — urgent "
                f"breaking-news alerts via `{name}` are NOT being delivered.",
            )

    # ── Recovery notices for conditions that were firing and have cleared ────
    for key, last in prev.items():
        if key in active:
            continue
        # Only announce recovery for a condition we had actually paged about
        # (a fresh, non-zero last-fire time within the TTL); then drop it.
        if last and (now_epoch - last) < _STATE_TTL_S:
            label = key.replace("worker:", "worker `") + (
                "`" if key.startswith("worker:") else ""
            )
            messages.append(
                f"✅ ALERT-PIPELINE WATCHDOG: {label} recovered — alert "
                f"pipeline condition cleared."
            )
        # Either way the cleared condition is not carried forward.

    return messages, new_state


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _snapshot_age_s(path: Path, now_epoch: float) -> float | None:
    try:
        return now_epoch - path.stat().st_mtime
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run", action="store_true",
        help="print escalations instead of posting to Discord",
    )
    args = ap.parse_args(argv)

    now = time.time()
    snapshot = _load_json(SUPERVISOR_STATE_PATH)
    age = _snapshot_age_s(SUPERVISOR_STATE_PATH, now)
    throttle = _load_json(WATCHDOG_STATE_PATH) or {}

    messages, new_state = evaluate(snapshot, age, now, throttle)

    if not messages:
        print("[alert_watchdog] ok — alert pipeline healthy "
              f"(snapshot_age={age if age is None else round(age)}s)")
    for msg in messages:
        if args.dry_run:
            print(f"[alert_watchdog] (dry-run) would post: {msg}")
        else:
            print(f"[alert_watchdog] escalating: {msg}")
            try:
                from notifier.discord_notifier import send as discord_send
                discord_send(msg, is_alert=True)
            except Exception as e:  # never let a notifier failure crash the cron
                print(f"[alert_watchdog] discord send failed: {e}")

    if not args.dry_run:
        try:
            WATCHDOG_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            WATCHDOG_STATE_PATH.write_text(json.dumps(new_state))
        except Exception as e:
            print(f"[alert_watchdog] could not persist throttle state: {e}")

    # Exit non-zero when something is wrong so a systemd timer / cron wrapper
    # can surface it independently of the Discord path.
    return 1 if messages else 0


if __name__ == "__main__":
    sys.path.insert(0, str(BASE_DIR))
    raise SystemExit(main())
