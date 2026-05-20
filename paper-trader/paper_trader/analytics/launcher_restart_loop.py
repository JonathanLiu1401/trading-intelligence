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


def build_launcher_restart_loop(lines: Iterable[str]) -> dict:
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
    return {
        "verdict": verdict,
        "headline": headline,
        "starts": starts,
        "refusals": refusals,
        "holder_pid": last_holder_pid,
        "loop_floor": LOOP_FLOOR,
    }
