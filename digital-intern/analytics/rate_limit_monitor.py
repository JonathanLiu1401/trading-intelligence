"""Per-collector HTTP 429 / rate-limit monitor.

Scans the last ``WINDOW_HOURS`` of journald for the digital-intern unit,
extracts every line that signals a rate-limit response, attributes it to a
collector (the leading ``[name]`` tag the collectors already emit), tallies
counts plus rough per-hour rate, and writes a JSON snapshot to
``REPORT_PATH``. Any collector exceeding ``HOT_THRESHOLD_PER_HOUR`` is
flagged ``hot`` with a suggested back-off bucket so a future patch can pick
it up without re-grepping logs.

Standalone: ``python3 -m analytics.rate_limit_monitor``
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from core.logger import get_logger

log = get_logger("rate_limit_monitor")

REPORT_PATH = Path("/home/zeph/logs/rate_limit_report.json")
WINDOW_HOURS = 24
HOT_THRESHOLD_PER_HOUR = 5
UNIT = "digital-intern"

# Collectors prefix log lines with "[name] ...". Captures the leading tag.
_TAG_RE = re.compile(r"\[([a-zA-Z0-9_\-]+)\]")
_RATE_RE = re.compile(r"(?i)(429|rate.?limit|too.?many.?request)")


def _journal_lines(hours: int) -> list[str]:
    out = subprocess.run(
        ["journalctl", "-u", UNIT, "--since", f"{hours} hours ago", "--no-pager", "-o", "cat"],
        capture_output=True, text=True, timeout=60,
    )
    return out.stdout.splitlines()


def _suggested_backoff(per_hour: float) -> str:
    if per_hour >= 100: return "60s+jitter"
    if per_hour >= 30:  return "30s+jitter"
    if per_hour >= 10:  return "15s+jitter"
    return "5s+jitter"


def build_report(hours: int = WINDOW_HOURS) -> dict:
    counts: Counter[str] = Counter()
    total = 0
    for line in _journal_lines(hours):
        if not _RATE_RE.search(line):
            continue
        total += 1
        m = _TAG_RE.search(line)
        counts[m.group(1) if m else "unknown"] += 1

    per_collector = []
    for name, n in counts.most_common():
        rate = n / hours
        per_collector.append({
            "collector": name,
            "events": n,
            "per_hour": round(rate, 2),
            "hot": rate >= HOT_THRESHOLD_PER_HOUR,
            "suggested_backoff": _suggested_backoff(rate),
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_hours": hours,
        "total_events": total,
        "distinct_collectors": len(counts),
        "hot_collectors": [c["collector"] for c in per_collector if c["hot"]],
        "per_collector": per_collector,
    }


def main() -> None:
    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    hot = report["hot_collectors"]
    log.info(
        "rate_limit_monitor: %d events / %dh across %d collectors; hot=%s",
        report["total_events"], report["window_hours"],
        report["distinct_collectors"], hot or "none",
    )
    print(json.dumps({
        "total": report["total_events"],
        "distinct": report["distinct_collectors"],
        "hot": hot,
        "top": report["per_collector"][:5],
    }, indent=2))


if __name__ == "__main__":
    main()
