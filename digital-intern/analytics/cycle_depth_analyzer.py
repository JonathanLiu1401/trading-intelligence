#!/usr/bin/env python3
"""Cycle depth analyzer.

The ``cycle`` column tracks how many reprocessing passes an article has
received (0 = processed once, 1 = reprocessed once, etc.).  High cycle
depth can indicate: LLM-retry loops, scorer failures on first pass, or
article content that required multiple enrichment rounds.

This script measures:
  * Global distribution of articles by cycle depth (0, 1-5, 6-10, 11+)
  * Average ml_score / ai_score / urgency by cycle bucket — does
    reprocessing actually improve signal quality?
  * Per-source avg cycle depth for the most-reprocessed sources
  * Overall reprocess_rate: fraction of articles that have cycle > 0

Design: uses GROUP BY aggregates for distribution (no full-table scan risk),
then a bounded per-source sample for source-level stats.
Output: /home/zeph/logs/cycle_depth_analysis.json
Standalone: python3 -m analytics.cycle_depth_analyzer
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/cycle_depth_analysis.json")
SOURCE_SCAN_LIMIT = 50_000  # bounded scan for per-source stats

# Cycle buckets: label -> (min_cycle, max_cycle inclusive)
BUCKETS = [
    ("0", 0, 0),
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21+", 21, 999999),
]

LIVE_EXCLUDE = "source NOT LIKE 'backtest%' AND url NOT LIKE 'backtest://%'"


def _bucket_label(cycle: int) -> str:
    for label, lo, hi in BUCKETS:
        if lo <= cycle <= hi:
            return label
    return "21+"


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=15)
    conn.row_factory = sqlite3.Row

    # --- Distribution via GROUP BY (fast, no full-scan needed for counts) ---
    dist_rows = conn.execute(
        f"""
        SELECT cycle, COUNT(*) AS n,
               AVG(CASE WHEN ml_score IS NOT NULL THEN ml_score END) AS avg_ml,
               AVG(CASE WHEN ai_score IS NOT NULL THEN ai_score END) AS avg_ai,
               AVG(CAST(urgency AS REAL)) AS avg_urgency
        FROM articles
        WHERE {LIVE_EXCLUDE} AND cycle IS NOT NULL
        GROUP BY cycle
        ORDER BY cycle
        """
    ).fetchall()

    # Aggregate into buckets
    bucket_data: dict[str, dict] = {
        label: {"count": 0, "ml_sum": 0.0, "ml_n": 0, "ai_sum": 0.0, "ai_n": 0, "urg_sum": 0.0}
        for label, *_ in BUCKETS
    }
    total = 0
    for row in dist_rows:
        label = _bucket_label(row["cycle"])
        b = bucket_data[label]
        n = row["n"]
        b["count"] += n
        total += n
        if row["avg_ml"] is not None:
            b["ml_sum"] += row["avg_ml"] * n
            b["ml_n"] += n
        if row["avg_ai"] is not None:
            b["ai_sum"] += row["avg_ai"] * n
            b["ai_n"] += n
        b["urg_sum"] += (row["avg_urgency"] or 0.0) * n

    depth_stats = []
    for label, *_ in BUCKETS:
        b = bucket_data[label]
        depth_stats.append({
            "cycle_bucket": label,
            "count": b["count"],
            "pct": round(b["count"] / max(total, 1) * 100, 1),
            "avg_ml_score": round(b["ml_sum"] / b["ml_n"], 4) if b["ml_n"] else None,
            "avg_ai_score": round(b["ai_sum"] / b["ai_n"], 4) if b["ai_n"] else None,
            "avg_urgency": round(b["urg_sum"] / max(b["count"], 1), 4),
        })

    reprocess_n = total - bucket_data["0"]["count"]
    reprocess_rate = reprocess_n / max(total, 1)

    # --- Per-source avg cycle from recent bounded scan ---
    src_rows = conn.execute(
        f"""
        SELECT source, cycle
        FROM articles INDEXED BY idx_first_seen
        WHERE {LIVE_EXCLUDE} AND cycle > 0 AND cycle IS NOT NULL
        ORDER BY first_seen DESC
        LIMIT ?
        """,
        (SOURCE_SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    source_sum: dict[str, list[int]] = {}
    for row in src_rows:
        src = row["source"] or "unknown"
        if src not in source_sum:
            source_sum[src] = []
        source_sum[src].append(row["cycle"])

    source_avg = [
        {"source": src, "avg_cycle": round(sum(cycs) / len(cycs), 2), "count": len(cycs)}
        for src, cycs in source_sum.items()
        if len(cycs) >= 5
    ]
    source_avg.sort(key=lambda r: r["avg_cycle"], reverse=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_articles_scanned": total,
        "reprocess_count": reprocess_n,
        "reprocess_rate_pct": round(reprocess_rate * 100, 1),
        "depth_distribution": depth_stats,
        "top_sources_by_avg_cycle": source_avg[:15],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    print(f"cycle_depth_analyzer: total={total} reprocessed={reprocess_n} ({reprocess_rate*100:.1f}%)")
    for d in depth_stats:
        ml = f"ml={d['avg_ml_score']:.3f}" if d["avg_ml_score"] is not None else "ml=n/a"
        print(f"  cycle={d['cycle_bucket']}: {d['count']} ({d['pct']}%) {ml} urgency={d['avg_urgency']:.3f}")
    print("  top reprocess-heavy sources (avg cycle of reprocessed rows):")
    for s in source_avg[:5]:
        print(f"    {s['source']}: avg_cycle={s['avg_cycle']} n={s['count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
