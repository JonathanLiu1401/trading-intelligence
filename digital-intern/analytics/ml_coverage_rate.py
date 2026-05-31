"""ML coverage rate monitor.

Answers: *What fraction of articles ingested in the last hour actually have
an ml_score?*  A drop-off here is the earliest warning that the ML scoring
worker has stalled, before score_drift_detector can notice (it needs enough
scored rows to compute a mean; if scoring stops entirely it emits NO-DATA
rather than an alert).

Distinct from existing tools:
  * ``scoring_backlog_audit``  — counts *all* unscored rows across the DB,
    dominated by backtest_run rows that are intentionally unscored.
  * ``scoring_funnel``         — 24h funnel snapshot, not run hourly.
  * ``score_drift_detector``   — detects mean-value drift, blind to a
    complete scoring outage (emits NO-DATA instead of ALERT).

Design constraints (inherited from workspace memory):
  * Single bounded LIMIT scan via idx_first_seen — no COUNT(*) on full table.
  * _LIVE_ONLY_CLAUSE applied — backtest rows excluded.
  * Read-only sqlite URI; busy_timeout 15 000 ms.
  * Graceful on empty window (market closed / holiday).

Output: /home/zeph/logs/ml_coverage.json
Log:    /home/zeph/logs/ml_coverage.log  (append; human-readable trail)

Standalone:  python3 -m analytics.ml_coverage_rate
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT_PATH = Path("/home/zeph/logs/ml_coverage.json")
LOG_PATH = Path("/home/zeph/logs/ml_coverage.log")

WINDOW_HOURS = 1          # look-back for liveness check
SCAN_LIMIT   = 2000       # bounded to keep USB-backed DB responsive
ALERT_BELOW  = 0.30       # coverage < 30% → ALERT
WARN_BELOW   = 0.60       # coverage < 60% → WARN


def _parse_ts(ts: str) -> datetime:
    """Parse ISO-8601 string (with or without +00:00 suffix) to UTC datetime."""
    ts = ts.replace("T", " ").rstrip("Z")
    if "+" in ts:
        ts = ts[: ts.index("+")]
    elif ts.count("-") > 2:
        # handle -00:00 suffix
        last = ts.rfind("-")
        if last > 10:
            ts = ts[:last]
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def run() -> dict:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=WINDOW_HOURS)

    db_path = _get_db_path()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15)
    con.execute("PRAGMA busy_timeout=15000")

    rows = con.execute(
        f"SELECT first_seen, ml_score FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        f"ORDER BY first_seen DESC LIMIT {SCAN_LIMIT}"
    ).fetchall()
    con.close()

    total = 0
    scored = 0
    for first_seen_raw, ml_score in rows:
        try:
            ts = _parse_ts(first_seen_raw)
        except Exception:
            continue
        if ts < cutoff:
            break
        total += 1
        if ml_score is not None and ml_score > 0:
            scored += 1

    coverage = (scored / total) if total else None

    if coverage is None:
        state = "NO-DATA"
        headline = f"No live articles in last {WINDOW_HOURS}h (market holiday / collector gap)."
    elif coverage < ALERT_BELOW:
        state = "ALERT"
        headline = (
            f"ML scoring appears stalled: only {scored}/{total} articles scored "
            f"({coverage*100:.1f}%) in the last {WINDOW_HOURS}h."
        )
    elif coverage < WARN_BELOW:
        state = "WARN"
        headline = (
            f"ML coverage low: {scored}/{total} articles scored "
            f"({coverage*100:.1f}%) in the last {WINDOW_HOURS}h."
        )
    else:
        state = "OK"
        headline = (
            f"ML scoring healthy: {scored}/{total} articles scored "
            f"({coverage*100:.1f}%) in the last {WINDOW_HOURS}h."
        )

    result = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "scan_limit": SCAN_LIMIT,
        "state": state,
        "headline": headline,
        "total_articles": total,
        "scored_articles": scored,
        "coverage_pct": round(coverage * 100, 2) if coverage is not None else None,
        "alert_threshold_pct": ALERT_BELOW * 100,
        "warn_threshold_pct": WARN_BELOW * 100,
    }

    OUT_PATH.write_text(json.dumps(result, indent=2))

    log_line = f"{now.isoformat()} | {state} | {headline}\n"
    with LOG_PATH.open("a") as fh:
        fh.write(log_line)

    print(log_line.rstrip())
    return result


main = run


if __name__ == "__main__":
    r = run()
    sys.exit(0)
