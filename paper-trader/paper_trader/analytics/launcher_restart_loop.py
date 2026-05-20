"""Launcher restart-loop detector — pure builder over runner.log lines.

The systemd/launcher layer can crash-loop the paper-trader entrypoint
when a healthy trader already holds the singleton flock. Each failed
launch emits one ``[runner] starting paper trader`` line followed by a
``[runner] another paper trader is already running (pid=...); refusing
to start a second trader on the same paper book — exiting.`` line in
``logs/runner.log``. The actual trading loop is unaffected (the flock
is doing its job — see invariant #19), but the log churn hides real
events and burns inode/journald budget. No analytics surface tallied
those refusals, so an operator looking at /api/runner-heartbeat sees
HEALTHY (correct — the loop *is* alive) while the launcher is busy
re-spawning a no-op every few seconds.

This builder is pure: caller passes raw log lines (newest order does
not matter — we tally on substring match). The dashboard endpoint
owns the file read (the thesis_drift split). Never raises — a
garbage line is ignored, never sinks the verdict.

State ladder:
  * ``NO_DATA`` — no refusal lines in the window
  * ``QUIET``   — refusals < ``LOOP_FLOOR``
  * ``LOOP``    — refusals >= ``LOOP_FLOOR`` (the launcher is wedged)

Advisory only — never gates Opus, adds no caps (AGENTS.md #2/#12).
"""
from __future__ import annotations

from typing import Iterable

# Module-level thresholds (the live-constant discipline).
LOOP_FLOOR = 5
REFUSAL_MARK = "another paper trader is already running"
START_MARK = "[runner] starting paper trader"


def build_launcher_restart_loop(lines: Iterable[str],
                                *,
                                log_size_bytes: int | None = None,
                                log_age_seconds: float | None = None) -> dict:
    """Tally refusal lines and (optionally) price the log-churn cost.

    ``log_size_bytes`` and ``log_age_seconds`` are optional and pure
    pass-through facts the dashboard endpoint can hand in (``stat()``
    on the runner.log path). When present we surface them so an
    operator can see the *operational cost* of a wedged launcher — a
    multi-MB runner.log that grows by N MB/day is the symptom; the
    refusal count is the cause. Caller may omit either; missing fields
    degrade to ``None`` (the NO_DATA contract). Never raises."""
    starts = 0
    refusals = 0
    last_holder_pid: str | None = None
    for ln in lines:
        if not isinstance(ln, str):
            continue
        if START_MARK in ln:
            starts += 1
        if REFUSAL_MARK in ln:
            refusals += 1
            # Best-effort holder pid pull: "...(pid=3358429);..."
            i = ln.find("pid=")
            if i >= 0:
                j = i + 4
                k = j
                while k < len(ln) and ln[k].isdigit():
                    k += 1
                if k > j:
                    last_holder_pid = ln[j:k]
    if starts == 0 and refusals == 0:
        verdict = "NO_DATA"
        headline = "no launcher activity in window"
    elif refusals >= LOOP_FLOOR:
        verdict = "LOOP"
        holder = f" (current holder pid={last_holder_pid})" if last_holder_pid else ""
        headline = (
            f"LAUNCHER RESTART LOOP — {refusals} refusals across {starts} "
            f"launches{holder}; the trading loop itself is unaffected "
            f"(flock is holding) but the launcher is wedged."
        )
    else:
        verdict = "QUIET"
        headline = f"quiet — {refusals} refusal(s) across {starts} launch(es)"

    # Log-churn cost block — degrades cleanly when the caller omits the
    # stat() facts; never re-derives them (the _safe contract).
    log_size_mb: float | None = None
    bytes_per_day: float | None = None
    if isinstance(log_size_bytes, (int, float)) and log_size_bytes >= 0:
        log_size_mb = round(float(log_size_bytes) / (1024 * 1024), 3)
        if (isinstance(log_age_seconds, (int, float))
                and log_age_seconds and log_age_seconds > 0):
            # Wall-clock growth rate, not refusal-attributed. The whole
            # file is dominated by the refusal pair in a LOOP, so this
            # is a faithful upper bound for the operational cost.
            bytes_per_day = round(
                float(log_size_bytes) * 86400.0 / float(log_age_seconds), 1)

    return {
        "verdict": verdict,
        "headline": headline,
        "starts": starts,
        "refusals": refusals,
        "holder_pid": last_holder_pid,
        "loop_floor": LOOP_FLOOR,
        "log_size_bytes": (int(log_size_bytes)
                           if isinstance(log_size_bytes, (int, float))
                           and log_size_bytes >= 0 else None),
        "log_size_mb": log_size_mb,
        "log_age_seconds": (float(log_age_seconds)
                            if isinstance(log_age_seconds, (int, float))
                            and log_age_seconds and log_age_seconds > 0
                            else None),
        "log_bytes_per_day": bytes_per_day,
    }
