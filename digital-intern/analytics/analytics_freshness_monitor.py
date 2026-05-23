"""Analytics freshness monitor.

Scans all JSON output files in /home/zeph/logs/, reports their age, and
identifies which analytics outputs are stale (older than STALE_THRESHOLD_HOURS).
Writes a summary to /home/zeph/logs/analytics_freshness_monitor.json.

Standalone: python3 -m analytics.analytics_freshness_monitor
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

LOGS_DIR = Path("/home/zeph/logs")
OUT_PATH = LOGS_DIR / "analytics_freshness_monitor.json"
STALE_THRESHOLD_HOURS = 24


def run() -> dict:
    now = datetime.now(timezone.utc)
    results = []

    for f in sorted(LOGS_DIR.glob("*.json")):
        if f.name == OUT_PATH.name:
            continue
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
            age_hours = (now - mtime).total_seconds() / 3600
            results.append({
                "file": f.name,
                "last_updated": mtime.strftime("%Y-%m-%d %H:%M UTC"),
                "age_hours": round(age_hours, 1),
                "stale": age_hours > STALE_THRESHOLD_HOURS,
            })
        except OSError:
            continue

    results.sort(key=lambda r: r["age_hours"], reverse=True)
    stale = [r for r in results if r["stale"]]
    fresh = [r for r in results if not r["stale"]]

    summary = {
        "generated_at": now.isoformat(),
        "total_files": len(results),
        "stale_count": len(stale),
        "fresh_count": len(fresh),
        "stale_threshold_hours": STALE_THRESHOLD_HOURS,
        "stale_files": stale,
        "fresh_files": fresh,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    summary = run()
    print(f"Scanned {summary['total_files']} analytics files")
    print(f"Fresh (<{summary['stale_threshold_hours']}h): {summary['fresh_count']}  "
          f"Stale (>{summary['stale_threshold_hours']}h): {summary['stale_count']}")
    if summary["stale_files"]:
        print("\nStale files (oldest first):")
        for f in summary["stale_files"]:
            print(f"  {f['age_hours']:6.1f}h  {f['file']}")
    else:
        print("All analytics files are fresh.")
    print(f"\nReport written to {OUT_PATH}")


if __name__ == "__main__":
    main()
