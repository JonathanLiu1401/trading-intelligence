"""Score percentile snapshot.

Computes p50/p75/p90/p95/p99 of ml_score over last 1h, 24h, and 7d.
Useful for adaptive thresholding (e.g., "alert when ml_score > p95 of trailing 24h").
Output: /home/zeph/logs/score_percentiles.json
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

DB = Path("/home/zeph/digital-intern/data/articles.db")
OUT = Path("/home/zeph/logs/score_percentiles.json")
WINDOWS = {"1h": "-1 hours", "24h": "-24 hours", "7d": "-7 days"}
PCTS = [50, 75, 90, 95, 99]


def percentiles(values: list[float], pcts: list[int]) -> dict[str, float]:
    if not values:
        return {f"p{p}": None for p in pcts}
    s = sorted(values)
    n = len(s)
    out = {}
    for p in pcts:
        if n == 1:
            out[f"p{p}"] = s[0]
            continue
        k = (p / 100.0) * (n - 1)
        lo = int(k)
        hi = min(lo + 1, n - 1)
        frac = k - lo
        out[f"p{p}"] = round(s[lo] + (s[hi] - s[lo]) * frac, 4)
    return out


def main() -> int:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=10)
    con.execute("PRAGMA query_only=ON")
    snap = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "windows": {},
    }
    for label, modifier in WINDOWS.items():
        rows = con.execute(
            "SELECT ml_score FROM articles "
            "WHERE ml_score IS NOT NULL "
            "AND replace(first_seen,'T',' ') >= datetime('now', ?) "
            "ORDER BY first_seen DESC LIMIT 200000",
            (modifier,),
        ).fetchall()
        vals = [float(r[0]) for r in rows if r[0] is not None]
        entry = {"count": len(vals)}
        entry.update(percentiles(vals, PCTS))
        if vals:
            entry["mean"] = round(statistics.fmean(vals), 4)
            entry["stdev"] = round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0
            entry["max"] = round(max(vals), 4)
            entry["min"] = round(min(vals), 4)
        snap["windows"][label] = entry
    con.close()
    OUT.write_text(json.dumps(snap, indent=2))
    print(json.dumps(snap, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
