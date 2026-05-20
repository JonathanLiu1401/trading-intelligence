"""Hourly article ingestion rate tracker.

Buckets the last 24 hours of ``first_seen`` timestamps into 24 one-hour bins,
computes mean and standard deviation across those bins, and flags any hour
whose count deviates by more than ``ANOMALY_STD`` standard deviations from
the mean.  Two anomaly classes:

  * ``spike``  — count > mean + N*std  (possible duplicate flood or news burst)
  * ``gap``    — count < mean - N*std  (collection outage / stale collectors)

Design constraints:
  * No full COUNT(*) scan on the 1.4 GB USB-backed DB.
  * Single bounded ``idx_first_seen`` scan via LIMIT.
  * Read-only connection, busy_timeout=5 000 ms.
  * ``_LIVE_ONLY_CLAUSE`` applied so backtest rows never inflate counts.

Standalone:  ``python3 -m analytics.hourly_ingestion``
Output:      /home/zeph/logs/hourly_ingestion.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, stdev

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT = Path("/home/zeph/logs/hourly_ingestion.json")
SCAN_LIMIT = 20_000   # bounded scan — enough to cover ~24 h at typical volumes
ANOMALY_STD = 2.0


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        # Normalise: replace 'T' separator and strip timezone suffix so
        # fromisoformat works on both "2026-05-20T14:02:03.123456+00:00"
        # and "2026-05-20 14:02:03".
        normalised = raw[:19].replace("T", " ")
        return datetime.fromisoformat(normalised).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def compute() -> dict:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)

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

    # Bucket into 24 UTC hourly slots (index 0 = oldest hour).
    counts: dict[int, int] = defaultdict(int)
    scanned = 0
    in_window = 0
    for (ts_raw,) in rows:
        scanned += 1
        ts = _parse_ts(ts_raw)
        if ts is None or ts < cutoff:
            continue
        in_window += 1
        # Hours ago from now: 0 = current hour, 23 = 23 h ago.
        hours_ago = int((now - ts).total_seconds() // 3600)
        if 0 <= hours_ago <= 23:
            counts[hours_ago] += 1

    # Build ordered list oldest→newest.
    hourly: list[dict] = []
    for h in range(23, -1, -1):
        slot_start = now - timedelta(hours=h + 1)
        hourly.append(
            {
                "hour_utc": slot_start.strftime("%Y-%m-%d %H:00"),
                "hours_ago": h,
                "count": counts.get(h, 0),
            }
        )

    volumes = [s["count"] for s in hourly]
    avg = mean(volumes) if volumes else 0.0
    std = stdev(volumes) if len(volumes) >= 2 else 0.0
    threshold_high = avg + ANOMALY_STD * std
    threshold_low = max(0.0, avg - ANOMALY_STD * std)

    anomalies: list[dict] = []
    for slot in hourly:
        c = slot["count"]
        if std > 0 and c > threshold_high:
            slot["anomaly"] = "spike"
            anomalies.append({"hour_utc": slot["hour_utc"], "type": "spike", "count": c})
        elif std > 0 and c < threshold_low:
            slot["anomaly"] = "gap"
            anomalies.append({"hour_utc": slot["hour_utc"], "type": "gap", "count": c})

    payload = {
        "generated_at": now.isoformat(),
        "scan_limit": SCAN_LIMIT,
        "scanned": scanned,
        "in_window_24h": in_window,
        "stats": {
            "mean": round(avg, 1),
            "std": round(std, 1),
            "threshold_high": round(threshold_high, 1),
            "threshold_low": round(threshold_low, 1),
            "total_24h": sum(volumes),
            "peak_hour": max(hourly, key=lambda s: s["count"])["hour_utc"],
            "trough_hour": min(hourly, key=lambda s: s["count"])["hour_utc"],
        },
        "anomalies": anomalies,
        "hourly": hourly,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    p = compute()
    s = p["stats"]
    print(
        f"hourly_ingestion: scanned={p['scanned']} in_window={p['in_window_24h']}"
        f"  mean={s['mean']}/hr  std={s['std']}"
    )
    print(f"  peak={s['peak_hour']}  trough={s['trough_hour']}  total_24h={s['total_24h']}")
    if p["anomalies"]:
        for a in p["anomalies"]:
            print(f"  ANOMALY [{a['type'].upper()}] {a['hour_utc']} count={a['count']}")
    else:
        print("  no anomalies detected")
    print(f"  output={OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
