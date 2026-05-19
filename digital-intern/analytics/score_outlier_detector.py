"""Score outlier detector.

Flags articles from the past hour whose ml_score is >2 std above the trailing
24h mean. Writes top hits to /home/zeph/logs/score_outliers.json so other
operators (paper-trader, dashboards) can pick them up without re-querying
the full articles table.
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/score_outliers.json")


def _rows(con: sqlite3.Connection, sql: str) -> list[tuple[Any, ...]]:
    return list(con.execute(sql))


def detect(con: sqlite3.Connection, pct: float = 0.95, std_mult: float = 1.5, limit: int = 25) -> dict[str, Any]:
    baseline = sorted(
        r[0]
        for r in _rows(
            con,
            "SELECT ml_score FROM articles "
            "WHERE ml_score IS NOT NULL "
            "AND datetime(replace(first_seen,'T',' '))>=datetime('now','-24 hours')",
        )
        if r[0] is not None
    )
    if len(baseline) < 30:
        return {"status": "insufficient_baseline", "samples": len(baseline), "outliers": []}

    mean = statistics.fmean(baseline)
    std = statistics.pstdev(baseline) or 0.0
    pct_threshold = baseline[int(len(baseline) * pct)]
    std_threshold = mean + std_mult * std
    threshold = min(pct_threshold, std_threshold)

    recent = _rows(
        con,
        "SELECT id, source, ml_score, urgency, first_seen, substr(title,1,160) "
        "FROM articles "
        "WHERE ml_score IS NOT NULL "
        "AND datetime(replace(first_seen,'T',' '))>=datetime('now','-3 hours') "
        f"AND ml_score >= {threshold} "
        "ORDER BY ml_score DESC LIMIT " + str(limit),
    )
    outliers = [
        {
            "id": r[0],
            "source": r[1],
            "ml_score": r[2],
            "urgency": r[3],
            "first_seen": r[4],
            "title": r[5],
        }
        for r in recent
    ]
    return {
        "status": "ok",
        "baseline_n": len(baseline),
        "baseline_mean": round(mean, 4),
        "baseline_std": round(std, 4),
        "threshold": round(threshold, 4),
        "outliers": outliers,
    }


def main() -> dict[str, Any]:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10) as con:
        report = detect(con)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    rep = main()
    print(f"status={rep['status']} baseline_n={rep.get('baseline_n')} "
          f"threshold={rep.get('threshold')} outliers={len(rep.get('outliers', []))}")
    for o in rep.get("outliers", [])[:5]:
        print(f"  {o['ml_score']:.3f} u{o['urgency']} [{o['source']}] {o['title']}")
