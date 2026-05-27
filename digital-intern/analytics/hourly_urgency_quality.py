"""Hourly urgency quality index: combines urgency rate and avg ml_score per UTC hour.

Fills the gap between ``ml_score_hourly_heatmap`` (scores only, no urgency)
and ``market_session_analyzer`` (broad session groups, not individual hours).

For each of the last 24 UTC hour buckets:
  - article_count: total live articles ingested in that hour
  - urgent_count:  articles with urgency >= 2
  - urgency_rate:  urgent_count / article_count (0–1)
  - avg_ml:        mean ml_score of scored articles in that hour
  - quality_index: avg_ml * urgency_rate * 10  (0–100 composite)

``quality_index`` answers: *which clock-hours produce both high-scoring AND
urgent articles?*  Useful for calibrating alert thresholds (higher threshold
during low-qi off-hours, tighter during high-qi market hours).

HOT hours: quality_index > mean_qi + 1.0 * std_qi

Output: /home/zeph/logs/hourly_urgency_quality.json
Standalone: python3 -m analytics.hourly_urgency_quality
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

try:
    from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path
    DB_PATH = _get_db_path()
except Exception:
    _LIVE_ONLY_CLAUSE = "source NOT LIKE 'backtest_run_%'"
    DB_PATH = BASE / "data" / "articles.db"

OUT_PATH = Path("/home/zeph/logs/hourly_urgency_quality.json")
SCAN_LIMIT = 12_000
HOT_ZSCORE = 1.0


def _parse_hour(raw: str) -> int | None:
    if not raw:
        return None
    try:
        return int(raw[11:13])
    except (ValueError, IndexError):
        return None


def build_index(rows: list[tuple[str, float | None, int | None]]) -> dict:
    """Pure builder. rows = list of (first_seen, ml_score, urgency)."""
    buckets: dict[int, list[tuple[float | None, int]]] = defaultdict(list)
    for first_seen, ml_score, urgency in rows:
        hour = _parse_hour(first_seen)
        if hour is None:
            continue
        buckets[hour].append((ml_score, urgency or 0))

    if not buckets:
        return {
            "hours_with_data": 0, "total_articles": 0,
            "total_urgent": 0, "hot_hours": [], "index": [],
        }

    index: list[dict] = []
    qi_vals: list[float] = []

    for hour in range(24):
        entries = buckets.get(hour, [])
        if not entries:
            index.append({
                "hour_utc": hour, "article_count": 0,
                "urgent_count": 0, "urgency_rate": None,
                "avg_ml": None, "quality_index": None,
            })
            continue

        urgent_n = sum(1 for _, u in entries if u >= 2)
        scored = [s for s, _ in entries if s is not None and s > 0]
        urgency_rate = urgent_n / len(entries)
        avg_ml = round(mean(scored), 4) if scored else None
        qi = round(avg_ml * urgency_rate * 10, 2) if avg_ml is not None else 0.0
        qi_vals.append(qi)
        index.append({
            "hour_utc": hour,
            "article_count": len(entries),
            "urgent_count": urgent_n,
            "urgency_rate": round(urgency_rate, 4),
            "avg_ml": avg_ml,
            "quality_index": qi,
        })

    mean_qi = round(mean(qi_vals), 4) if qi_vals else 0.0
    std_qi = round(stdev(qi_vals), 4) if len(qi_vals) >= 2 else 0.0
    threshold = mean_qi + HOT_ZSCORE * std_qi

    hot_hours: list[dict] = []
    for entry in index:
        qi = entry["quality_index"]
        if qi is not None and qi >= threshold:
            entry["is_hot"] = True
            hot_hours.append({
                "hour_utc": entry["hour_utc"],
                "quality_index": qi,
                "article_count": entry["article_count"],
                "urgent_count": entry["urgent_count"],
                "avg_ml": entry["avg_ml"],
            })
        else:
            entry["is_hot"] = False

    hot_hours.sort(key=lambda x: -x["quality_index"])
    total = sum(e["article_count"] for e in index)
    total_urgent = sum(e["urgent_count"] for e in index)

    return {
        "hours_with_data": len(qi_vals),
        "total_articles": total,
        "total_urgent": total_urgent,
        "mean_quality_index": mean_qi,
        "std_quality_index": std_qi,
        "hot_threshold": round(threshold, 4),
        "hot_hours": hot_hours,
        "index": index,
    }


def main() -> int:
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
        conn.execute("PRAGMA query_only=ON")
        rows = conn.execute(
            "SELECT first_seen, ml_score, urgency FROM articles "
            f"WHERE {_LIVE_ONLY_CLAUSE} "
            "  AND first_seen >= ? "
            "ORDER BY first_seen DESC LIMIT ?",
            (cutoff, SCAN_LIMIT),
        ).fetchall()
        conn.close()
    except Exception as exc:
        print(f"[hourly_urgency_quality] DB error: {exc}", file=sys.stderr)
        return 1

    result = build_index(rows)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["scan_limit"] = SCAN_LIMIT
    result["rows_fetched"] = len(rows)

    OUT_PATH.write_text(json.dumps(result, indent=2))

    total = result["total_articles"]
    urgent = result["total_urgent"]
    hot_n = len(result["hot_hours"])
    mean_qi = result["mean_quality_index"]
    print(f"[hourly_urgency_quality] {total} articles | {urgent} urgent | mean_qi={mean_qi} | {hot_n} hot hours")
    for h in result["hot_hours"][:3]:
        print(
            f"  HOT {h['hour_utc']:02d}:00 UTC → qi={h['quality_index']}"
            f" | {h['urgent_count']}/{h['article_count']} urgent"
            f" | avg_ml={h['avg_ml']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
