"""ML scoring coverage breakdown by source.

Answers: *Which sources are NOT getting their articles ML-scored?*

``ml_coverage_rate`` gives an aggregate coverage %, but when coverage drops
it's opaque — you can't tell if one dominant source is flooding the queue or
if the scorer is skipping a class of articles.  This module partitions the
last ``SCAN_LIMIT`` live articles by ``source``, computes:

  * ``total``       — article count in window
  * ``scored``      — rows where ml_score IS NOT NULL
  * ``coverage_pct`` — scored / total * 100
  * ``avg_ml``      — mean ml_score of scored rows (null if none)
  * ``avg_ai``      — mean ai_score of scored rows
  * ``share_pct``   — this source's fraction of all live articles in window

Sources are ranked by ``coverage_pct`` ASC so the worst-covered sources
surface first.  A top-level ``alert`` flag is set if any source with ≥
``MIN_ARTICLES`` has coverage < ``ALERT_THRESHOLD_PCT``.

Design constraints (workspace memory):
  * Single bounded LIMIT scan via ``idx_first_seen`` — no full-table scan.
  * ``_LIVE_ONLY_CLAUSE`` applied — backtest/opus rows excluded.
  * Read-only sqlite URI; busy_timeout 10 000 ms.

Output: /home/zeph/logs/ml_coverage_by_source.json

Standalone:  python3 -m analytics.ml_coverage_by_source
Importable:  from analytics.ml_coverage_by_source import main
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT_PATH = Path("/home/zeph/logs/ml_coverage_by_source.json")
SCAN_LIMIT = 4000          # ~2-3h of live articles at typical ingest rates
MIN_ARTICLES = 5           # ignore sources with fewer articles (noise)
ALERT_THRESHOLD_PCT = 40.0 # coverage below this triggers alert flag
WARN_THRESHOLD_PCT = 70.0  # coverage below this triggers warn flag


def _db_path() -> Path:
    try:
        return Path(_get_db_path())
    except Exception:
        return BASE / "data" / "articles.db"


def compute(scan_limit: int = SCAN_LIMIT) -> dict:
    """Scan recent live articles and return per-source coverage breakdown."""
    db = _db_path()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    conn.execute("PRAGMA busy_timeout=10000")

    try:
        rows = conn.execute(
            f"""
            SELECT source, ml_score, ai_score
            FROM articles INDEXED BY idx_first_seen
            WHERE {_LIVE_ONLY_CLAUSE}
            ORDER BY first_seen DESC
            LIMIT ?
            """,
            (scan_limit,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scan_limit": scan_limit,
            "state": "NO_DATA",
            "headline": "No live articles found in recent window.",
            "total_articles": 0,
            "sources": [],
            "alert": False,
        }

    # Accumulate per source
    buckets: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "scored": 0, "ml_scores": [], "ai_scores": []}
    )
    for source, ml_score, ai_score in rows:
        key = source or "(null)"
        b = buckets[key]
        b["total"] += 1
        if ml_score is not None:
            b["scored"] += 1
            b["ml_scores"].append(ml_score)
        if ai_score is not None:
            b["ai_scores"].append(float(ai_score))

    total_articles = len(rows)
    alert = False
    warn = False
    source_rows: list[dict] = []

    for src, b in buckets.items():
        total = b["total"]
        scored = b["scored"]
        cov = round(scored / total * 100, 1) if total else 0.0
        avg_ml = round(sum(b["ml_scores"]) / len(b["ml_scores"]), 4) if b["ml_scores"] else None
        avg_ai = round(sum(b["ai_scores"]) / len(b["ai_scores"]), 4) if b["ai_scores"] else None
        share = round(total / total_articles * 100, 1)

        if total >= MIN_ARTICLES:
            if cov < ALERT_THRESHOLD_PCT:
                alert = True
            elif cov < WARN_THRESHOLD_PCT:
                warn = True

        source_rows.append({
            "source": src,
            "total": total,
            "scored": scored,
            "coverage_pct": cov,
            "share_pct": share,
            "avg_ml": avg_ml,
            "avg_ai": avg_ai,
        })

    # Sort by coverage_pct ASC (worst first), then total DESC to break ties
    source_rows.sort(key=lambda r: (r["coverage_pct"], -r["total"]))

    # Aggregate coverage
    all_scored = sum(b["scored"] for b in buckets.values())
    overall_cov = round(all_scored / total_articles * 100, 1) if total_articles else 0.0

    if alert:
        state = "ALERT"
        headline = (
            f"ML coverage alert: {overall_cov}% overall; "
            f"{sum(1 for r in source_rows if r['total'] >= MIN_ARTICLES and r['coverage_pct'] < ALERT_THRESHOLD_PCT)} "
            f"source(s) below {ALERT_THRESHOLD_PCT}% threshold."
        )
    elif warn:
        state = "WARN"
        headline = f"ML coverage warning: {overall_cov}% overall across {len(source_rows)} sources."
    else:
        state = "OK"
        headline = f"ML coverage healthy: {overall_cov}% overall across {len(source_rows)} sources."

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_limit": scan_limit,
        "state": state,
        "headline": headline,
        "total_articles": total_articles,
        "overall_coverage_pct": overall_cov,
        "alert": alert,
        "sources": source_rows,
    }


def main() -> None:
    result = compute()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(result["headline"])
    # Print top 5 worst-covered sources with enough volume
    heavy = [r for r in result["sources"] if r["total"] >= MIN_ARTICLES][:5]
    for r in heavy:
        print(
            f"  {r['source']:40s}  {r['coverage_pct']:5.1f}%  "
            f"({r['scored']}/{r['total']} scored, share={r['share_pct']}%)"
        )


if __name__ == "__main__":
    main()
