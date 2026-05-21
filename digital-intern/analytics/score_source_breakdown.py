#!/usr/bin/env python3
"""Score source distribution report.

Analyzes how articles are scored: by 'llm', by 'ml' model, or not scored at all.
Reports coverage, avg ml_score, avg kw_score, avg ai_score per scoring method
over a bounded recent window.

Design: uses LIMIT-based idx_first_seen scan to avoid full-table timeout on
the 1.4GB USB-backed DB.

Artifacts:
  * /home/zeph/logs/score_source_breakdown.json
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "articles.db")
LOG_DIR = "/home/zeph/logs"
OUT_PATH = os.path.join(LOG_DIR, "score_source_breakdown.json")

SCAN_LIMIT = 10000  # bounded scan via idx_first_seen


def run() -> dict:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        """
        SELECT score_source, ml_score, kw_score, ai_score
        FROM articles
        ORDER BY first_seen DESC
        LIMIT ?
        """,
        (SCAN_LIMIT,),
    ).fetchall()
    con.close()

    buckets: dict[str, dict] = {
        "llm": {"count": 0, "ml_scores": [], "kw_scores": [], "ai_scores": []},
        "ml":  {"count": 0, "ml_scores": [], "kw_scores": [], "ai_scores": []},
        "unscored": {"count": 0, "ml_scores": [], "kw_scores": [], "ai_scores": []},
    }

    for row in rows:
        key = row["score_source"] if row["score_source"] in ("llm", "ml") else "unscored"
        b = buckets[key]
        b["count"] += 1
        if row["ml_score"] is not None:
            b["ml_scores"].append(row["ml_score"])
        if row["kw_score"] is not None:
            b["kw_scores"].append(row["kw_score"])
        if row["ai_score"] is not None:
            b["ai_scores"].append(row["ai_score"])

    def avg(lst: list) -> float | None:
        return round(sum(lst) / len(lst), 3) if lst else None

    total = len(rows)
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_limit": SCAN_LIMIT,
        "total_scanned": total,
        "breakdown": {},
    }

    for key, b in buckets.items():
        cnt = b["count"]
        report["breakdown"][key] = {
            "count": cnt,
            "pct": round(100 * cnt / total, 1) if total else 0,
            "avg_ml_score": avg(b["ml_scores"]),
            "avg_kw_score": avg(b["kw_scores"]),
            "avg_ai_score": avg(b["ai_scores"]),
        }

    return report


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    report = run()
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    bd = report["breakdown"]
    print(f"Scanned {report['total_scanned']} recent articles")
    for key in ("llm", "ml", "unscored"):
        b = bd[key]
        print(
            f"  {key:9s}: {b['count']:5d} ({b['pct']:5.1f}%)"
            f"  avg_ml={b['avg_ml_score']}"
            f"  avg_kw={b['avg_kw_score']}"
            f"  avg_ai={b['avg_ai_score']}"
        )
    print(f"Written: {OUT_PATH}")
