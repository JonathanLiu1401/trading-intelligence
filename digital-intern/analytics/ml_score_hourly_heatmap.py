"""ML score hourly heatmap: 24-bucket signal intensity timeline.

Computes the average ml_score for each UTC hour-of-day bucket across the
past 24 hours of live articles (ml_score IS NOT NULL AND ml_score > 0).
Identifies "peak signal hours" — UTC hours where the hourly avg ml_score
exceeds (daily_mean + 0.5 * daily_std), useful for understanding when the
highest-quality actionable news arrives.

Distinct from existing features:
  * ``score_drift_detector``: anomaly detection vs 7-day rolling baseline
  * ``market_session_analyzer``: groups by US trading session (pre/regular/after/overnight)
  * ``recency_decay``: time-decays individual article scores

This module answers a different question: *within the last 24 hours, which
clock-hour buckets produced the strongest ML signal on average?*

Output: /home/zeph/logs/ml_score_hourly_heatmap.json
Standalone: python3 -m analytics.ml_score_hourly_heatmap
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

try:
    from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path
    DB_PATH = _get_db_path()
except Exception:
    _LIVE_ONLY_CLAUSE = "source NOT LIKE 'backtest_run_%'"
    DB_PATH = BASE / "data" / "articles.db"

OUT_PATH = Path("/home/zeph/logs/ml_score_hourly_heatmap.json")
SCAN_LIMIT = 8_000
PEAK_ZSCORE_THRESHOLD = 0.5   # hours with avg > mean + 0.5*std are "peak"


def _parse_hour(raw: str) -> int | None:
    """Return UTC hour (0-23) from first_seen string, or None on failure."""
    if not raw:
        return None
    try:
        ts_str = raw[:19].replace("T", " ")
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt.hour
    except ValueError:
        return None


def build_heatmap(rows: list[tuple[str, float]]) -> dict:
    """Pure builder — no I/O. rows = list of (first_seen, ml_score)."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for first_seen, ml_score in rows:
        if ml_score is None or ml_score <= 0:
            continue
        hour = _parse_hour(first_seen)
        if hour is None:
            continue
        buckets[hour].append(float(ml_score))

    if not buckets:
        return {
            "hours_with_data": 0,
            "total_articles": 0,
            "daily_mean": None,
            "daily_std": None,
            "peak_hours": [],
            "heatmap": [],
        }

    # Build per-hour summary
    heatmap: list[dict] = []
    all_avgs: list[float] = []
    for hour in range(24):
        scores = buckets.get(hour, [])
        if scores:
            avg = round(mean(scores), 4)
            all_avgs.append(avg)
        else:
            avg = None
        heatmap.append({
            "hour_utc": hour,
            "article_count": len(scores),
            "avg_ml_score": avg,
        })

    if not all_avgs:
        return {
            "hours_with_data": 0,
            "total_articles": sum(len(v) for v in buckets.values()),
            "daily_mean": None,
            "daily_std": None,
            "peak_hours": [],
            "heatmap": heatmap,
        }

    daily_mean = round(mean(all_avgs), 4)
    daily_std = round(stdev(all_avgs), 4) if len(all_avgs) >= 2 else 0.0
    threshold = daily_mean + PEAK_ZSCORE_THRESHOLD * daily_std

    peak_hours: list[dict] = []
    for entry in heatmap:
        avg = entry["avg_ml_score"]
        if avg is not None and avg >= threshold:
            entry["is_peak"] = True
            peak_hours.append({
                "hour_utc": entry["hour_utc"],
                "avg_ml_score": avg,
                "article_count": entry["article_count"],
            })
        else:
            entry["is_peak"] = False

    peak_hours.sort(key=lambda x: -x["avg_ml_score"])

    return {
        "hours_with_data": len(all_avgs),
        "total_articles": sum(len(v) for v in buckets.values()),
        "daily_mean": daily_mean,
        "daily_std": daily_std,
        "peak_threshold": round(threshold, 4),
        "peak_hours": peak_hours,
        "heatmap": heatmap,
    }


def main() -> int:
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
        conn.execute("PRAGMA query_only=ON")
        # Use ISO 8601 string comparison so idx_first_seen is used directly.
        # first_seen is stored as 'YYYY-MM-DDTHH:MM:SS...' — ISO strings sort
        # lexicographically, so a cutoff string works without replace().
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
        rows = conn.execute(
            "SELECT first_seen, ml_score FROM articles "
            f"WHERE {_LIVE_ONLY_CLAUSE} "
            "  AND ml_score IS NOT NULL AND ml_score > 0 "
            "  AND first_seen >= ? "
            "ORDER BY first_seen DESC LIMIT ?",
            (cutoff, SCAN_LIMIT),
        ).fetchall()
        conn.close()
    except Exception as exc:
        print(f"[ml_score_hourly_heatmap] DB error: {exc}", file=sys.stderr)
        return 1

    result = build_heatmap(rows)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["scan_limit"] = SCAN_LIMIT
    result["rows_fetched"] = len(rows)

    OUT_PATH.write_text(json.dumps(result, indent=2))
    n_peak = len(result["peak_hours"])
    total = result["total_articles"]
    mean_val = result["daily_mean"]
    print(f"[ml_score_hourly_heatmap] {total} articles | mean={mean_val} | {n_peak} peak hours")
    if result["peak_hours"]:
        top = result["peak_hours"][0]
        print(f"  top peak: hour {top['hour_utc']:02d}:00 UTC → avg_ml={top['avg_ml_score']} ({top['article_count']} articles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
