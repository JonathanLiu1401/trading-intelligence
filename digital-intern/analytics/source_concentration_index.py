#!/usr/bin/env python3
"""Source concentration index (Herfindahl-Hirschman Index).

Measures how concentrated recent article ingest is across sources.
A high HHI (near 10,000) means a few sources dominate the pipeline — a
fragility risk. Low HHI (near 0) means healthy diversity.

Formula: HHI = sum((share_i * 100)^2) for each source i
  * HHI > 2500 → CONCENTRATED (oligopoly-level dominance)
  * HHI 1000–2500 → MODERATE
  * HHI < 1000 → DIVERSE

Runs over the most recent SCAN_LIMIT live rows (excludes backtest/annotation
synthetics) and outputs the top dominant sources plus the HHI verdict.

Artifacts:
  * /home/zeph/logs/source_concentration.json

Standalone: python3 -m analytics.source_concentration_index
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "articles.db"
LOG_DIR = Path("/home/zeph/logs")
OUT_PATH = LOG_DIR / "source_concentration.json"

SCAN_LIMIT = 4000  # bounded scan via idx_first_seen


def _verdict(hhi: float) -> str:
    if hhi >= 2500:
        return "CONCENTRATED"
    if hhi >= 1000:
        return "MODERATE"
    return "DIVERSE"


def compute() -> dict:
    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro", uri=True, timeout=10
    )
    try:
        rows = conn.execute(
            """
            SELECT source
            FROM articles
            WHERE source NOT LIKE 'backtest%'
              AND source NOT LIKE 'opus_annotation%'
              AND source NOT LIKE 'backtest_run%'
              AND url NOT LIKE 'backtest://%'
            ORDER BY first_seen DESC
            LIMIT ?
            """,
            (SCAN_LIMIT,),
        ).fetchall()
    finally:
        conn.close()

    counts: Counter = Counter(row[0] for row in rows)
    total = sum(counts.values())

    if total == 0:
        return {"error": "no rows", "ts": datetime.now(timezone.utc).isoformat()}

    # HHI: sum of squared market shares (in percentage points)
    hhi = sum((count / total * 100) ** 2 for count in counts.values())
    hhi = round(hhi, 1)

    top5 = [
        {
            "source": src,
            "count": cnt,
            "share_pct": round(cnt / total * 100, 2),
        }
        for src, cnt in counts.most_common(5)
    ]

    result = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "scan_limit": SCAN_LIMIT,
        "total_articles": total,
        "unique_sources": len(counts),
        "hhi": hhi,
        "verdict": _verdict(hhi),
        "top5_dominant_sources": top5,
    }
    return result


def run() -> None:
    result = compute()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    verdict = result.get("verdict", "?")
    hhi = result.get("hhi", "?")
    n_sources = result.get("unique_sources", "?")
    total = result.get("total_articles", "?")
    top_src = result.get("top5_dominant_sources", [])
    top_line = ", ".join(
        f"{s['source']}({s['share_pct']}%)" for s in top_src[:3]
    )
    print(
        f"[source_concentration] HHI={hhi} {verdict} | "
        f"{n_sources} sources / {total} articles | "
        f"top3: {top_line}"
    )


if __name__ == "__main__":
    run()
