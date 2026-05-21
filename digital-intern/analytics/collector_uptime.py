"""Collector uptime tracker: gap analysis per source over a 24h window.

For each live collector source, fetch all article timestamps from the past
24h and compute silence gaps (intervals with no article). Reports sources
that had at least one gap >= GAP_THRESHOLD_MIN, sorted by longest single gap.

Distinguishes from stale_source_alerter (which only checks current staleness)
by tracking the FULL gap history within the window — a source that went quiet
for 3h at noon then recovered is caught here but invisible to a point-in-time
staleness check.

Constraints:
  * No full-table COUNT(*). Uses ORDER BY first_seen DESC LIMIT N (idx scan).
  * Read-only connection, PRAGMA busy_timeout=8000.
  * Ignores sources with <MIN_ARTICLES_PER_SOURCE articles in window
    (too sparse to gap-analyse reliably).

Output: /home/zeph/logs/collector_uptime.json

Standalone:  python3 -m analytics.collector_uptime
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT_PATH = Path("/home/zeph/logs/collector_uptime.json")
SCAN_LIMIT = 10_000
WINDOW_HOURS = 24
GAP_THRESHOLD_MIN = 120   # gaps >= this are reported
MIN_ARTICLES_PER_SOURCE = 5  # skip sources with too few articles


def _parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    # Normalise: replace T separator, strip tz suffix, parse as UTC.
    ts = ts.replace("T", " ")[:19]
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def compute() -> dict:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=WINDOW_HOURS)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    db_path = str(_get_db_path())
    conn = sqlite3.connect(db_path, timeout=8)
    conn.execute("PRAGMA busy_timeout=8000")

    rows = conn.execute(
        f"""
        SELECT source, first_seen
          FROM articles
         WHERE {_LIVE_ONLY_CLAUSE}
           AND replace(first_seen,'T',' ') >= ?
         ORDER BY first_seen DESC
         LIMIT ?
        """,
        (cutoff_str, SCAN_LIMIT),
    ).fetchall()
    conn.close()

    # Bucket timestamps per source.
    by_source: dict[str, list[datetime]] = defaultdict(list)
    for source, ts_raw in rows:
        dt = _parse_ts(ts_raw)
        if dt:
            by_source[source].append(dt)

    gap_reports: list[dict] = []

    for source, timestamps in by_source.items():
        if len(timestamps) < MIN_ARTICLES_PER_SOURCE:
            continue
        # Sort ascending so we can walk gaps in order.
        timestamps.sort()
        gaps: list[dict] = []
        for i in range(1, len(timestamps)):
            delta_min = (timestamps[i] - timestamps[i - 1]).total_seconds() / 60
            if delta_min >= GAP_THRESHOLD_MIN:
                gaps.append({
                    "gap_start": timestamps[i - 1].strftime("%Y-%m-%dT%H:%MZ"),
                    "gap_end": timestamps[i].strftime("%Y-%m-%dT%H:%MZ"),
                    "gap_min": round(delta_min, 1),
                })
        if gaps:
            max_gap = max(g["gap_min"] for g in gaps)
            gap_reports.append({
                "source": source,
                "articles_in_window": len(timestamps),
                "gap_count": len(gaps),
                "max_gap_min": max_gap,
                "gaps": gaps,
            })

    gap_reports.sort(key=lambda r: r["max_gap_min"], reverse=True)

    result = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_hours": WINDOW_HOURS,
        "gap_threshold_min": GAP_THRESHOLD_MIN,
        "sources_analysed": len(by_source),
        "sources_with_gaps": len(gap_reports),
        "top_gap_sources": gap_reports[:20],
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    return result


def main():
    result = compute()
    sources_with_gaps = result["sources_with_gaps"]
    sources_analysed = result["sources_analysed"]
    print(f"Analysed {sources_analysed} sources | {sources_with_gaps} had gaps >= {result['gap_threshold_min']}min")
    for r in result["top_gap_sources"][:5]:
        print(
            f"  {r['source']}: {r['gap_count']} gap(s), max {r['max_gap_min']}min"
            f" | {r['articles_in_window']} articles"
        )


if __name__ == "__main__":
    main()
