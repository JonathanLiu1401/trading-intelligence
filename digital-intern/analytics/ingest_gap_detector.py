"""Article ingestion gap detector.

Scans the last 12 hours of ``first_seen`` timestamps and flags any interval
between consecutive ingests that exceeds ``GAP_THRESHOLD_MINUTES``.  This
catches collection outages that a coarser hourly-bin view ([[hourly_ingestion]])
can miss when the surrounding hour still has enough volume to look normal.

  * ``status = "alert"`` if any gap exceeds ``ALERT_GAP_MINUTES`` (60 min)
  * ``status = "ok"`` otherwise (including the no-gaps case)

Design constraints:
  * No full COUNT(*) scan on the 1.4 GB USB-backed DB.
  * Single bounded ``idx_first_seen`` scan via LIMIT.
  * Read-only connection, busy_timeout=5 000 ms.
  * ``_LIVE_ONLY_CLAUSE`` applied so backtest rows never mask real gaps.

Standalone:  ``python3 -m analytics.ingest_gap_detector``
Output:      /home/zeph/logs/ingest_gaps.json
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

OUT = Path("/home/zeph/logs/ingest_gaps.json")
SCAN_LIMIT = 10_000
WINDOW_HOURS = 12
GAP_THRESHOLD_MINUTES = 30
ALERT_GAP_MINUTES = 60


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        normalised = raw[:19].replace("T", " ")
        return datetime.fromisoformat(normalised).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def compute() -> dict:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=WINDOW_HOURS)

    conn = sqlite3.connect(str(_get_db_path()))
    conn.execute("PRAGMA busy_timeout=5000")
    rows = conn.execute(
        f"""
        SELECT first_seen
          FROM articles
         WHERE {_LIVE_ONLY_CLAUSE}
         ORDER BY first_seen DESC
         LIMIT ?
        """,
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    timestamps: list[datetime] = []
    for (ts_raw,) in rows:
        ts = _parse_ts(ts_raw)
        if ts is None or ts < cutoff:
            continue
        timestamps.append(ts)

    timestamps.sort()

    gaps: list[dict] = []
    for prev, curr in zip(timestamps, timestamps[1:]):
        delta = curr - prev
        minutes = delta.total_seconds() / 60.0
        if minutes > GAP_THRESHOLD_MINUTES:
            gaps.append(
                {
                    "start": prev.isoformat(),
                    "end": curr.isoformat(),
                    "duration_minutes": round(minutes, 2),
                }
            )

    longest = max((g["duration_minutes"] for g in gaps), default=0.0)
    status = "alert" if longest > ALERT_GAP_MINUTES else "ok"

    payload = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "gap_threshold_minutes": GAP_THRESHOLD_MINUTES,
        "total_articles_scanned": len(timestamps),
        "gaps": gaps,
        "gap_count": len(gaps),
        "longest_gap_minutes": round(longest, 2),
        "status": status,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    p = compute()
    print(
        f"ingest_gap_detector: scanned={p['total_articles_scanned']} in {p['window_hours']}h"
        f"  threshold={p['gap_threshold_minutes']}min  status={p['status'].upper()}"
    )
    if p["gaps"]:
        print(f"  gap_count={p['gap_count']}  longest={p['longest_gap_minutes']} min")
        for g in p["gaps"]:
            print(f"  GAP {g['start']} → {g['end']} ({g['duration_minutes']} min)")
    else:
        print("  no gaps > threshold detected")
    print(f"  output={OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
