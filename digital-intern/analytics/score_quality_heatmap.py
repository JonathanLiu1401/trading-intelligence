"""Score quality heatmap by UTC hour of day.

Buckets the last 7 days of articles into 24 one-hour bins (by first_seen UTC
hour) and computes quality metrics per bin:
  * count       — articles in that hour bucket across 7 days
  * avg_ai      — mean ai_score (0–1)
  * avg_ml      — mean ml_score where not NULL
  * pct_urgent  — fraction with urgency >= 2
  * peak_day    — which weekday (0=Mon) sees most articles in this hour

Use case: understand *when* during the day the pipeline receives the
highest-quality signals, so alert thresholds and review windows can be
scheduled around actual information density.

Design constraints:
  * Bounded SCAN_LIMIT idx_first_seen scan — no full-table scans.
  * Read-only sqlite URI — no writer contention.
  * _LIVE_ONLY_CLAUSE — excludes backtest rows.
  * busy_timeout 10 000 ms — USB-safe.

Output: /home/zeph/logs/score_quality_heatmap.json
Standalone: python3 -m analytics.score_quality_heatmap
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT = Path("/home/zeph/logs/score_quality_heatmap.json")
SCAN_LIMIT = 50_000   # ~7 days at typical volume
LOOKBACK_DAYS = 7


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.strptime(s[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main() -> int:
    db = _get_db_path()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=10000")

    cutoff = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).isoformat()
    rows = conn.execute(
        "SELECT first_seen, ai_score, ml_score, urgency "
        "FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "  AND first_seen >= ? "
        "ORDER BY first_seen DESC "
        "LIMIT ?",
        (cutoff, SCAN_LIMIT),
    ).fetchall()
    conn.close()

    if not rows:
        print("score_quality_heatmap: no rows in range", file=sys.stderr)
        return 1

    # Per-hour buckets
    ai_scores:  dict[int, list[float]] = defaultdict(list)
    ml_scores:  dict[int, list[float]] = defaultdict(list)
    urgencies:  dict[int, list[int]]   = defaultdict(list)
    day_counts: dict[int, list[int]]   = defaultdict(list)  # hour -> list of weekdays

    scanned = 0
    for fs, ai, ml, urg in rows:
        ts = _parse_ts(fs)
        if ts is None:
            continue
        h = ts.hour
        scanned += 1
        if ai is not None:
            ai_scores[h].append(float(ai))
        if ml is not None:
            ml_scores[h].append(float(ml))
        urgencies[h].append(int(urg or 0))
        day_counts[h].append(ts.weekday())

    now = datetime.now(timezone.utc)
    bins = []
    for h in range(24):
        n = len(urgencies[h])
        if n == 0:
            bins.append({
                "hour_utc": h,
                "count": 0,
                "avg_ai": None,
                "avg_ml": None,
                "pct_urgent": None,
                "peak_weekday": None,
            })
            continue

        avg_ai = round(mean(ai_scores[h]), 4) if ai_scores[h] else None
        avg_ml = round(mean(ml_scores[h]), 4) if ml_scores[h] else None
        pct_urgent = round(sum(1 for u in urgencies[h] if u >= 2) / n, 4)

        # Most common weekday in this hour slot
        wd_counts: dict[int, int] = defaultdict(int)
        for wd in day_counts[h]:
            wd_counts[wd] += 1
        peak_wd = max(wd_counts, key=lambda k: wd_counts[k])
        wd_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        bins.append({
            "hour_utc": h,
            "count": n,
            "avg_ai": avg_ai,
            "avg_ml": avg_ml,
            "pct_urgent": pct_urgent,
            "peak_weekday": wd_names[peak_wd],
        })

    # Identify top-3 quality hours (by avg_ai)
    ranked = sorted(
        [b for b in bins if b["avg_ai"] is not None],
        key=lambda b: b["avg_ai"],
        reverse=True,
    )
    top_hours = [b["hour_utc"] for b in ranked[:3]]

    payload = {
        "generated_at": now.isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "scanned": scanned,
        "top_quality_hours_utc": top_hours,
        "bins": bins,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"score_quality_heatmap: scanned={scanned} lookback={LOOKBACK_DAYS}d top_hours={top_hours}")
    for b in ranked[:5]:
        print(f"  UTC {b['hour_utc']:02d}:00  count={b['count']:5d}  avg_ai={b['avg_ai']:.4f}  pct_urgent={b['pct_urgent']:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
