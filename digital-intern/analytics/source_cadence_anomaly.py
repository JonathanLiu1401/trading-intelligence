"""Source cadence anomaly detector.

For every live collector source with enough history, computes an expected
hourly publication rate from a 7-day baseline (mean ± std over hourly buckets).
The LAST full hour is then compared against that baseline via Z-score.

Anomalies flagged:
  - FLOOD : last-hour count > mean + Z_FLOOD * std  (sudden burst, possible spam/injection)
  - QUIET : last-hour count == 0 AND mean > QUIET_MIN_MEAN  (source went silent mid-stream)

Output: /home/zeph/logs/source_cadence_anomaly.json

Standalone:  python3 -m analytics.source_cadence_anomaly
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT = Path("/home/zeph/logs/source_cadence_anomaly.json")

# Z-score threshold above which a source is considered flooding
Z_FLOOD = 3.0

# Only flag QUIET for sources whose 7d mean exceeds this articles/hour
QUIET_MIN_MEAN = 0.5

# Only consider sources with at least this many hourly data points in 7d
MIN_HOURLY_BUCKETS = 6

# Only include sources with at least this many total articles in 7d
MIN_TOTAL_ARTICLES = 10


def _normalise_ts(ts: str) -> str:
    """Normalise ISO8601 first_seen to 'YYYY-MM-DD HH' hour bucket."""
    return ts[:13].replace("T", " ")


def main() -> None:
    db_path = str(_get_db_path())
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA busy_timeout=8000")

    now_utc = datetime.now(timezone.utc)
    window_start = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    # "last full hour" = the complete hour that just ended
    last_hour_start = now_utc.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    last_hour_end = last_hour_start + timedelta(hours=1)
    last_hour_bucket = last_hour_start.strftime("%Y-%m-%d %H")

    # Pull per-source, per-hour-bucket article counts over the last 7 days.
    # Use idx_first_seen via the WHERE date filter; SUBSTR gives us the hour bucket.
    rows = conn.execute(
        f"""
        SELECT
            source,
            SUBSTR(REPLACE(first_seen, 'T', ' '), 1, 13) AS hour_bucket,
            COUNT(*) AS n
        FROM articles
        WHERE REPLACE(first_seen, 'T', ' ') >= ?
          AND {_LIVE_ONLY_CLAUSE}
        GROUP BY source, hour_bucket
        """,
        (window_start,),
    ).fetchall()
    conn.close()

    # Build per-source hourly rate map
    source_hours: dict[str, dict[str, int]] = {}
    for source, bucket, n in rows:
        source_hours.setdefault(source, {})[bucket] = n

    anomalies: list[dict] = []
    summaries: list[dict] = []

    for source, hour_map in source_hours.items():
        # Exclude the current (incomplete) hour and last hour from baseline
        current_hour_bucket = now_utc.strftime("%Y-%m-%d %H")
        baseline_counts = [
            v for k, v in hour_map.items()
            if k != current_hour_bucket and k != last_hour_bucket
        ]

        if len(baseline_counts) < MIN_HOURLY_BUCKETS:
            continue
        if sum(baseline_counts) < MIN_TOTAL_ARTICLES:
            continue

        mean = sum(baseline_counts) / len(baseline_counts)
        variance = sum((x - mean) ** 2 for x in baseline_counts) / len(baseline_counts)
        std = math.sqrt(variance)

        last_hour_count = hour_map.get(last_hour_bucket, 0)
        z_score = (last_hour_count - mean) / std if std > 0.01 else 0.0

        entry: dict = {
            "source": source,
            "last_hour_count": last_hour_count,
            "baseline_mean": round(mean, 3),
            "baseline_std": round(std, 3),
            "z_score": round(z_score, 2),
            "baseline_hours": len(baseline_counts),
            "anomaly": None,
        }

        if z_score > Z_FLOOD:
            entry["anomaly"] = "FLOOD"
            anomalies.append(entry)
        elif last_hour_count == 0 and mean >= QUIET_MIN_MEAN:
            entry["anomaly"] = "QUIET"
            anomalies.append(entry)

        summaries.append(entry)

    # Sort anomalies: FLOOD by z_score desc, then QUIET by mean desc
    flood = sorted([a for a in anomalies if a["anomaly"] == "FLOOD"],
                   key=lambda x: -x["z_score"])
    quiet = sorted([a for a in anomalies if a["anomaly"] == "QUIET"],
                   key=lambda x: -x["baseline_mean"])

    result = {
        "generated_at": now_utc.isoformat(),
        "last_hour_utc": last_hour_bucket,
        "sources_checked": len(summaries),
        "flood_count": len(flood),
        "quiet_count": len(quiet),
        "flood_anomalies": flood,
        "quiet_anomalies": quiet[:20],  # cap quiet list — often many overnight
    }

    OUT.write_text(json.dumps(result, indent=2))
    print(
        f"checked {len(summaries)} sources | "
        f"FLOOD: {len(flood)} | QUIET: {len(quiet)} | "
        f"last_hour={last_hour_bucket}"
    )
    if flood:
        for a in flood[:3]:
            print(f"  FLOOD {a['source']}: {a['last_hour_count']} articles (z={a['z_score']}, mean={a['baseline_mean']})")
    if quiet:
        for a in quiet[:3]:
            print(f"  QUIET {a['source']}: 0 this hour (mean={a['baseline_mean']} art/hr)")


if __name__ == "__main__":
    main()
