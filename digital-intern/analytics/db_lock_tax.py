"""DB lock-tax analyzer.

Reads journalctl to count 'database is locked' errors per collector over the
last hour, then computes a lock_tax: the estimated fraction of write attempts
that failed due to contention.

Why this exists: the intern DB on a USB-backed volume has no WAL mode and
suffers contention from 15+ concurrent collectors. Each "database is locked"
means an article was silently dropped. This module makes that cost visible.

Approach:
  * Parse journalctl lines matching 'database is locked' for the last WINDOW_MIN
    minutes; extract the collector name from the preceding bracket tag.
  * Count committed articles in the same window from the DB (bounded scan).
  * lock_tax = errors / (errors + committed) — fraction of write attempts lost.
  * Per-collector breakdown surfaces the worst offenders.

Output: /home/zeph/logs/db_lock_rate.json
Standalone: python3 -m analytics.db_lock_tax
"""
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/home/zeph/logs/db_lock_rate.json")
DB = Path(__file__).resolve().parents[1] / "data" / "articles.db"
WINDOW_MIN = 60
# Regex to pull the collector name from lines like:
#   [rss_collector] dedup row skipped ...: database is locked
#   [finnhub_worker] error: database is locked; backing off 10s
_COLLECTOR_RE = re.compile(r"\[([a-z_A-Z0-9]+)\]")
# Skip the PID tag (all digits) and severity tags W/I/E
_SKIP_TAGS = re.compile(r"^\d+$|^[WIE]$|^daemon$")


def _parse_lock_errors(window_min: int) -> dict[str, int]:
    """Count lock errors per collector from journalctl."""
    since = f"{window_min} minutes ago"
    try:
        result = subprocess.run(
            ["journalctl", "-u", "digital-intern",
             f"--since={since}", "--no-pager", "--output=short"],
            capture_output=True, text=True, timeout=20,
        )
        lines = result.stdout.splitlines()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    per_collector: dict[str, int] = defaultdict(int)
    for line in lines:
        if "database is locked" not in line:
            continue
        tags = _COLLECTOR_RE.findall(line)
        collector = None
        for tag in tags:
            if not _SKIP_TAGS.match(tag):
                collector = tag
                break
        if collector:
            per_collector[collector] += 1
        else:
            per_collector["unknown"] += 1
    return dict(per_collector)


def _committed_in_window(window_min: int) -> int:
    """Count articles committed to DB in the last window_min minutes."""
    try:
        conn = sqlite3.connect(str(DB), timeout=5)
        conn.execute("PRAGMA busy_timeout=4000")
        cutoff = f"2026-05-28T{(datetime.now(timezone.utc).hour):02d}"
        # Use string comparison on ISO first_seen — fastest index path
        from datetime import timedelta
        cutoff_dt = datetime.now(timezone.utc) - timedelta(minutes=window_min)
        cutoff_str = cutoff_dt.strftime("%Y-%m-%dT%H:%M")
        row = conn.execute(
            "SELECT COUNT(*) FROM articles WHERE first_seen >= ? ORDER BY first_seen DESC LIMIT 1",
            (cutoff_str,),
        ).fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return 0


def main() -> None:
    now = datetime.now(timezone.utc)
    per_collector = _parse_lock_errors(WINDOW_MIN)
    total_errors = sum(per_collector.values())
    committed = _committed_in_window(WINDOW_MIN)

    # lock_tax: fraction of write attempts that failed
    total_attempts = total_errors + committed
    lock_tax = round(total_errors / total_attempts, 4) if total_attempts > 0 else 0.0

    # Rank collectors by error count
    ranked = sorted(per_collector.items(), key=lambda x: x[1], reverse=True)

    status = "OK"
    if lock_tax > 0.30:
        status = "CRITICAL"
    elif lock_tax > 0.10:
        status = "WARN"

    result = {
        "generated_at": now.isoformat(),
        "window_minutes": WINDOW_MIN,
        "status": status,
        "lock_errors_total": total_errors,
        "articles_committed": committed,
        "total_write_attempts": total_attempts,
        "lock_tax": lock_tax,
        "lock_tax_pct": round(lock_tax * 100, 1),
        "worst_offenders": [{"collector": k, "errors": v} for k, v in ranked[:10]],
    }

    OUT.write_text(json.dumps(result, indent=2))
    print(
        f"lock_tax={lock_tax:.1%} | errors={total_errors} committed={committed} "
        f"status={status} | top_offender={ranked[0][0] if ranked else 'none'}"
    )


if __name__ == "__main__":
    main()
