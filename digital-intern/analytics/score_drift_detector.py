#!/usr/bin/env python3
"""Score drift detector.

Detects when the live ML scoring pipeline drifts: alerts when the average
ml_score over the most recent ~1h deviates by more than 1.5 standard
deviations from a rolling 7-day hourly baseline.

Design constraints (see workspace memory):
  * The articles.db is ~1.4 GB on a USB-backed volume and under constant
    write contention from the daemon. Full-table COUNT/aggregation queries
    time out. We therefore NEVER scan the whole table: a single bounded
    `ORDER BY first_seen DESC LIMIT N` query (served by idx_first_seen) is
    the only DB read, and the 7d baseline is accumulated incrementally in a
    small JSON state file rather than queried from the DB each run.

Artifacts:
  * State : /home/zeph/logs/.score_drift_state.json   (rolling hourly points)
  * Log   : /home/zeph/logs/score_drift.log           (human-readable trail)

Exit status is always 0; this is an observability tool, not a gate.
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics
import sys
from datetime import datetime, timezone, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "articles.db")
LOG_DIR = "/home/zeph/logs"
STATE_PATH = os.path.join(LOG_DIR, ".score_drift_state.json")
LOG_PATH = os.path.join(LOG_DIR, "score_drift.log")

SCAN_LIMIT = 6000          # bounded idx_first_seen scan, ~12h of live rows
WINDOW_MINUTES = 60        # "current" window for the drift comparison
MAX_HISTORY = 168          # keep 7 days of hourly baseline points
MIN_BASELINE_POINTS = 6    # need this many history points before alerting
SIGMA = 1.5                # drift threshold in std devs
ABS_FLOOR = 0.5            # min band (ml_score pts) so a near-constant
                           # baseline still catches large jumps


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(raw: str):
    """Parse the DB's first_seen ('...T...+00:00' or space-separated)."""
    if not raw:
        return None
    s = str(raw).replace("T", " ").split("+")[0].strip()[:26]
    try:
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def fetch_recent_scores():
    """Return (current_window_avg, current_n, sample_total) using one
    bounded, index-served query. Returns (None, 0, n) if no scored rows."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        rows = conn.execute(
            "SELECT ml_score, first_seen FROM articles "
            "ORDER BY first_seen DESC LIMIT ?",
            (SCAN_LIMIT,),
        ).fetchall()
    finally:
        conn.close()

    cutoff = _utcnow() - timedelta(minutes=WINDOW_MINUTES)
    window = []
    for ml_score, first_seen in rows:
        if ml_score is None:
            continue
        ts = _parse_ts(first_seen)
        if ts is not None and ts >= cutoff:
            window.append(float(ml_score))

    if not window:
        return None, 0, len(rows)
    return statistics.fmean(window), len(window), len(rows)


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {"history": []}


def save_state(state: dict) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, STATE_PATH)


def append_log(line: str) -> None:
    with open(LOG_PATH, "a") as fh:
        fh.write(line + "\n")


def main() -> int:
    now = _utcnow()
    cur_avg, cur_n, sample_total = fetch_recent_scores()

    if cur_avg is None:
        msg = (
            f"{now.isoformat()} | NO-DATA | no ml_score'd rows in last "
            f"{WINDOW_MINUTES}m of {sample_total}-row sample"
        )
        append_log(msg)
        print(msg)
        return 0

    state = load_state()
    history = state.get("history", [])
    baseline_avgs = [p["avg"] for p in history]

    status = "BASELINE-BUILD"
    detail = f"history={len(baseline_avgs)}/{MIN_BASELINE_POINTS}"
    if len(baseline_avgs) >= MIN_BASELINE_POINTS:
        b_mean = statistics.fmean(baseline_avgs)
        b_std = statistics.pstdev(baseline_avgs)
        # A perfectly stable pipeline yields std~0; without a floor the
        # detector would never fire. Keep it armed with an absolute band.
        band = max(SIGMA * b_std, ABS_FLOOR)
        deviation = abs(cur_avg - b_mean)
        if deviation > band:
            direction = "UP" if cur_avg > b_mean else "DOWN"
            status = f"DRIFT-{direction}"
        else:
            status = "OK"
        detail = (
            f"baseline_mean={b_mean:.4f} baseline_std={b_std:.4f} "
            f"dev={deviation:.4f} band={band:.4f}"
        )

    history.append({"ts": now.isoformat(), "avg": round(cur_avg, 6), "n": cur_n})
    state["history"] = history[-MAX_HISTORY:]
    save_state(state)

    msg = (
        f"{now.isoformat()} | {status} | cur_avg={cur_avg:.4f} "
        f"cur_n={cur_n} sample={sample_total} | {detail}"
    )
    append_log(msg)
    print(msg)
    if status.startswith("DRIFT"):
        print(
            f"ALERT: ml_score {status} — current {cur_avg:.4f} "
            f"vs 7d baseline ({detail})",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
