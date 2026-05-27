"""Day-of-week ingest baseline.

Computes the average article ingest rate for each (day-of-week, hour-of-day)
slot over the past 14 days, then compares the current hour to its historical
baseline.  Useful for normalising anomaly thresholds in `hourly_ingestion`:
a "gap" at 3 AM Sunday is normal; a "gap" at 10 AM Tuesday is not.

Key fields in output:
  * ``current_dow``      — 0=Mon … 6=Sun
  * ``current_hod``      — 0–23 UTC
  * ``current_count``    — articles ingested in the current calendar hour
  * ``baseline_mean``    — historical mean for this DOW+HOD slot
  * ``baseline_std``     — historical std for this slot
  * ``z_score``          — (current - mean) / std  (null if std==0)
  * ``status``           — "ok" / "low" / "high" / "no_baseline"
  * ``slots``            — full 7×24 table for dashboard rendering
  * ``quietest_slots``   — top-5 historically quiet DOW+HOD slots

Design constraints:
  * No full COUNT(*) on the 1.4 GB USB-backed DB.
  * Single bounded DESC scan (LIMIT 50 000) on idx_first_seen — well under
    14-day article volume at typical rates (~4 k/hr live).
  * Read-only connection, busy_timeout=5 000 ms.
  * _LIVE_ONLY_CLAUSE applied — backtest rows never inflate counts.

Standalone:  python3 -m analytics.dow_baseline
Output:      /home/zeph/logs/dow_baseline.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean, pstdev

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT = Path("/home/zeph/logs/dow_baseline.json")
SCAN_LIMIT = 200_000
LOOKBACK_DAYS = 14
LOW_Z = -1.5
HIGH_Z = 2.0

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        normalised = raw[:19].replace("T", " ")
        return datetime.strptime(normalised, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def compute() -> dict:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=LOOKBACK_DAYS)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    current_hour_start = now.replace(minute=0, second=0, microsecond=0)
    current_hour_str = current_hour_start.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(str(_get_db_path()))
    conn.execute("PRAGMA busy_timeout=10000")

    # Use raw first_seen comparisons so idx_first_seen is used for range scans.
    # ISO-8601 timestamps ('2026-05-12T...' / '2026-05-12 ...') sort correctly
    # lexicographically, so >= / < comparisons against the index-stored values
    # are safe without any column transformation.
    grouped = conn.execute(
        f"""
        SELECT
            date(first_seen) AS day,
            CAST(strftime('%H', first_seen) AS INTEGER) AS hod,
            COUNT(*) AS cnt
          FROM articles
         WHERE {_LIVE_ONLY_CLAUSE}
           AND first_seen >= ?
         GROUP BY day, hod
        """,
        (cutoff_str,),
    ).fetchall()

    # Current hour count (lightweight bounded scan)
    (current_count,) = conn.execute(
        f"""
        SELECT COUNT(*)
          FROM articles
         WHERE {_LIVE_ONLY_CLAUSE}
           AND first_seen >= ?
        """,
        (current_hour_str,),
    ).fetchone()
    conn.close()

    # Build per-(dow, hod) day-instance buckets
    slot_days: dict[int, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for day_str, hod, cnt in grouped:
        try:
            dt = datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        dow = dt.weekday()  # 0=Mon … 6=Sun
        slot_days[dow][hod][day_str] += cnt

    # Build 7×24 summary table
    slots: list[dict] = []
    for dow in range(7):
        for hod in range(24):
            day_counts = list(slot_days[dow][hod].values())
            if not day_counts:
                mean_val = 0.0
                std_val = 0.0
                n_days = 0
            else:
                mean_val = round(fmean(day_counts), 1)
                std_val = round(pstdev(day_counts), 1) if len(day_counts) >= 2 else 0.0
                n_days = len(day_counts)
            slots.append({
                "dow": dow,
                "dow_name": DOW_NAMES[dow],
                "hod": hod,
                "mean": mean_val,
                "std": std_val,
                "n_days": n_days,
            })

    # Current slot comparison
    cur_dow = now.weekday()
    cur_hod = now.hour
    cur_slot = next((s for s in slots if s["dow"] == cur_dow and s["hod"] == cur_hod), None)
    if cur_slot and cur_slot["n_days"] >= 2:
        baseline_mean = cur_slot["mean"]
        baseline_std = cur_slot["std"]
        z = round((current_count - baseline_mean) / baseline_std, 2) if baseline_std > 0 else None
        if z is None:
            status = "ok"
        elif z < LOW_Z:
            status = "low"
        elif z > HIGH_Z:
            status = "high"
        else:
            status = "ok"
    else:
        baseline_mean = None
        baseline_std = None
        z = None
        status = "no_baseline"

    # Top-5 quietest slots (by mean, exclude zero-data slots)
    quietest = sorted(
        [s for s in slots if s["n_days"] >= 2 and s["mean"] > 0],
        key=lambda s: s["mean"],
    )[:5]

    # Top-5 busiest slots
    busiest = sorted(
        [s for s in slots if s["n_days"] >= 2],
        key=lambda s: -s["mean"],
    )[:5]

    payload = {
        "generated_at": now.isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "current_dow": cur_dow,
        "current_dow_name": DOW_NAMES[cur_dow],
        "current_hod": cur_hod,
        "current_count": current_count,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "z_score": z,
        "status": status,
        "quietest_slots": quietest,
        "busiest_slots": busiest,
        "slots": slots,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    p = compute()
    print(
        f"dow_baseline: {p['current_dow_name']} HOD={p['current_hod']:02d}  "
        f"current={p['current_count']}  "
        f"baseline={p['baseline_mean']} ± {p['baseline_std']}  "
        f"z={p['z_score']}  status={p['status']}"
    )
    if p["busiest_slots"]:
        print("  Busiest slots (historical avg):")
        for s in p["busiest_slots"]:
            print(f"    {s['dow_name']} {s['hod']:02d}:00  avg={s['mean']} std={s['std']} (n={s['n_days']})")
    if p["quietest_slots"]:
        print("  Quietest slots (historical avg):")
        for s in p["quietest_slots"]:
            print(f"    {s['dow_name']} {s['hod']:02d}:00  avg={s['mean']} std={s['std']} (n={s['n_days']})")
    print(f"  output={OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
