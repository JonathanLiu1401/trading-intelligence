#!/usr/bin/env python3
"""Per-source publish→ingest latency tracker.

For each source in the last `--hours` window, compute the median and p90
lag between the article's `published` timestamp and our `first_seen`
ingest timestamp. Surfaces feeds whose RSS is stale relative to others.

Writes to /home/zeph/logs/source_latency.json and prints the slowest 10
sources to stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "data" / "articles.db"
OUT = Path("/home/zeph/logs/source_latency.json")

MIN_SAMPLES = 3


def _parse_any(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        dt = parsedate_to_datetime(ts)
        if dt is not None:
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass
    s = ts.replace("T", " ").rstrip("Z")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def collect(hours: int) -> dict:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute(
        """
        SELECT source, published, first_seen
        FROM articles
        WHERE replace(first_seen,'T',' ') >= datetime('now', ?)
          AND published IS NOT NULL AND published != ''
          AND source NOT LIKE 'backtest_run_%'
          AND source NOT LIKE 'opus_annotation%'
        """,
        (f"-{hours} hours",),
    ).fetchall()
    con.close()

    buckets: dict[str, list[float]] = {}
    for source, pub, seen in rows:
        p = _parse_any(pub)
        s = _parse_any(seen)
        if p is None or s is None:
            continue
        lag = (s - p).total_seconds()
        if lag < -300 or lag > 7 * 86400:
            continue
        buckets.setdefault(source or "?", []).append(lag)

    out = []
    for source, lags in buckets.items():
        if len(lags) < MIN_SAMPLES:
            continue
        lags_sorted = sorted(lags)
        p90 = lags_sorted[max(0, math.ceil(0.9 * len(lags_sorted)) - 1)]
        out.append({
            "source": source,
            "n": len(lags),
            "median_s": round(statistics.median(lags), 1),
            "p90_s": round(p90, 1),
            "min_s": round(lags_sorted[0], 1),
            "max_s": round(lags_sorted[-1], 1),
        })
    out.sort(key=lambda r: r["median_s"], reverse=True)
    return {"window_hours": hours, "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_count": len(out), "sources": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=6)
    args = ap.parse_args()
    payload = collect(args.hours)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"sources tracked: {payload['source_count']} window: {args.hours}h")
    for row in payload["sources"][:10]:
        print(f"  {row['source']:30s} n={row['n']:4d}  median={row['median_s']:>8.1f}s  p90={row['p90_s']:>8.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
