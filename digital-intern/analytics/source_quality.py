#!/usr/bin/env python3
"""Per-source quality tracker.

For every distinct live collector ``source`` in ``articles.db`` this computes
the article count, average ai_score / ml_score / kw_score (skipping NULLs),
and the fraction of urgent rows (``urgency >= 2``) over a bounded recent
window.

Design: bounded LIMIT-based scan via ``idx_first_seen`` to avoid full-table
timeout on the 1.4GB USB-backed DB. Synthetic ``backtest*`` /
``opus_annotation*`` rows and ``backtest://`` URLs are excluded — those are
training-only injections, not live collectors.

Artifacts:
  * /home/zeph/logs/source_quality.json
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "articles.db"
LOG_DIR = Path("/home/zeph/logs")
OUT_PATH = LOG_DIR / "source_quality.json"

SCAN_LIMIT = 8000  # bounded scan via idx_first_seen


def _avg(values: list[float]) -> float | None:
    """Average of non-NULL values, rounded to 4dp. None if list empty."""
    return round(sum(values) / len(values), 4) if values else None


def compute() -> dict:
    """Scan the most recent SCAN_LIMIT live rows and roll up per source."""
    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro", uri=True, timeout=10
    )
    try:
        rows = conn.execute(
            """
            SELECT source, ai_score, ml_score, kw_score, urgency
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

    # Per-source accumulators
    buckets: dict[str, dict] = defaultdict(
        lambda: {
            "count": 0,
            "ai_scores": [],
            "ml_scores": [],
            "kw_scores": [],
            "urgent": 0,
        }
    )

    for source, ai_score, ml_score, kw_score, urgency in rows:
        if source is None or source == "":
            continue
        b = buckets[source]
        b["count"] += 1
        if ai_score is not None:
            b["ai_scores"].append(ai_score)
        if ml_score is not None:
            b["ml_scores"].append(ml_score)
        if kw_score is not None:
            b["kw_scores"].append(kw_score)
        try:
            if urgency is not None and int(urgency) >= 2:
                b["urgent"] += 1
        except (TypeError, ValueError):
            pass

    sources: dict[str, dict] = {}
    for source, b in buckets.items():
        cnt = b["count"]
        sources[source] = {
            "count": cnt,
            "avg_ai_score": _avg(b["ai_scores"]),
            "avg_ml_score": _avg(b["ml_scores"]),
            "avg_kw_score": _avg(b["kw_scores"]),
            "pct_urgent": round(b["urgent"] / cnt, 4) if cnt else 0.0,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_limit": SCAN_LIMIT,
        "scanned": len(rows),
        "sources_reported": len(sources),
        "sources": sources,
    }


def write_snapshot(report: dict, path: Path = OUT_PATH) -> Path:
    """Atomically write the report as pretty JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


if __name__ == "__main__":
    report = compute()
    out = write_snapshot(report)
    print(
        f"source-quality: scanned {report['scanned']} rows "
        f"({report['sources_reported']} distinct sources); "
        f"snapshot={out}"
    )
