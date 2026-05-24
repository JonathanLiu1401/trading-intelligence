"""Ingest volume heatmap by hour-of-week (7 days × 24 hours).

Builds a 7×24 matrix of average articles/slot across the last 30 days of live
ingest data.  Each cell shows how many articles were collected in that
(weekday, UTC-hour) slot on average, normalised by how many weeks of data are
available.  Cells ±1.5 std from the row-mean are flagged as HOT or COLD.

Use case: spot structural collection blind spots — e.g. "Saturday 03:00 UTC
always runs dry" — so cron schedules and alert thresholds can account for
predictable quiet windows.

Design constraints:
  * Bounded SCAN_LIMIT idx_first_seen scan — no full-table scan on 1.4 GB DB.
  * Read-only sqlite URI.
  * _LIVE_ONLY_CLAUSE applied (no backtest rows).
  * busy_timeout 10 000 ms — USB-safe.

Output: /home/zeph/logs/ingest_heatmap_weekly.json
Standalone: python3 -m analytics.ingest_heatmap_weekly
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

OUT = Path("/home/zeph/logs/ingest_heatmap_weekly.json")
SCAN_LIMIT = 80_000   # ~30 days of typical volume
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FLAG_STD = 1.5


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw[:19].replace("T", " ")).replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def run() -> dict:
    db_path = _get_db_path()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    rows = con.execute(
        f"""
        SELECT first_seen
        FROM   articles
        WHERE  {_LIVE_ONLY_CLAUSE}
          AND  replace(first_seen,'T',' ') >= ?
        ORDER  BY first_seen DESC
        LIMIT  ?
        """,
        (cutoff_str, SCAN_LIMIT),
    ).fetchall()
    con.close()

    # Count articles per (weekday 0=Mon, hour) slot across distinct calendar dates
    slot_counts: dict[tuple[int, int], list[int]] = defaultdict(list)
    # date_slot[(date_str, weekday, hour)] = count
    date_slot: dict[tuple[str, int, int], int] = defaultdict(int)

    for (raw,) in rows:
        ts = _parse_ts(raw)
        if ts is None:
            continue
        day = ts.weekday()   # 0=Mon … 6=Sun
        hr = ts.hour
        date_str = ts.strftime("%Y-%m-%d")
        date_slot[(date_str, day, hr)] += 1

    # Aggregate: for each (day, hr) collect per-date counts so we can average
    agg: dict[tuple[int, int], list[int]] = defaultdict(list)
    for (date_str, day, hr), cnt in date_slot.items():
        agg[(day, hr)].append(cnt)

    # Fill zeros for slots that had no articles on some occurrences
    # Each weekday appears ~4–5 times in 30 days; count distinct dates per weekday
    weekday_dates: dict[int, set[str]] = defaultdict(set)
    for (date_str, day, _hr), _ in date_slot.items():
        weekday_dates[day].add(date_str)

    # Compute per-day averages and row stats for flagging
    matrix: list[dict] = []
    for d in range(7):
        n_weeks = max(len(weekday_dates[d]), 1)
        row_avgs = []
        cells = []
        for h in range(24):
            counts = agg.get((d, h), [])
            total = sum(counts)
            avg = total / n_weeks
            row_avgs.append(avg)
            cells.append({"hour": h, "avg": round(avg, 2), "total": total})

        row_mean = mean(row_avgs) if row_avgs else 0.0
        row_std = stdev(row_avgs) if len(row_avgs) > 1 else 0.0
        threshold_hot = row_mean + FLAG_STD * row_std
        threshold_cold = row_mean - FLAG_STD * row_std

        for cell in cells:
            if row_std > 0:
                if cell["avg"] >= threshold_hot:
                    cell["flag"] = "HOT"
                elif cell["avg"] <= threshold_cold and cell["avg"] < row_mean * 0.5:
                    cell["flag"] = "COLD"
                else:
                    cell["flag"] = "ok"
            else:
                cell["flag"] = "ok"

        cold_hours = [c["hour"] for c in cells if c["flag"] == "COLD"]
        hot_hours = [c["hour"] for c in cells if c["flag"] == "HOT"]
        matrix.append(
            {
                "weekday": DAYS[d],
                "weekday_idx": d,
                "n_dates": n_weeks,
                "row_mean": round(row_mean, 2),
                "cold_hours": cold_hours,
                "hot_hours": hot_hours,
                "cells": cells,
            }
        )

    scanned = len(rows)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": 30,
        "scan_limit": SCAN_LIMIT,
        "scanned": scanned,
        "flag_std_threshold": FLAG_STD,
        "matrix": matrix,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    return result


def _print_summary(result: dict) -> None:
    print(f"Ingest heatmap — scanned {result['scanned']} articles (last 30 days)")
    print(f"{'Day':<4}  {'Mean/hr':>8}  Cold hours                    Hot hours")
    print("-" * 64)
    for row in result["matrix"]:
        cold = ", ".join(f"{h:02d}:00" for h in row["cold_hours"][:4]) or "none"
        hot = ", ".join(f"{h:02d}:00" for h in row["hot_hours"][:4]) or "none"
        print(f"{row['weekday']:<4}  {row['row_mean']:>8.1f}  cold={cold:<28} hot={hot}")


if __name__ == "__main__":
    result = run()
    _print_summary(result)
    print(f"\nWritten → {OUT}")
