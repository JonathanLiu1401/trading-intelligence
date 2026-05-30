"""Cross-scorer consensus detector.

Finds articles where the keyword scorer (kw_score) is confirmed by a
second independent scorer (ml_score or ai_score). Cross-scorer agreement
is stronger evidence of news significance than any single scorer alone.

Key insight from data: ml_score and ai_score are mutually exclusive paths
(LLM-scored articles have ml_score=0/None; ML-scored ones have ai_score=0).
We therefore define two consensus categories:

  * KW+ML  — kw_score >= KW_MIN AND ml_score >= ML_MIN
  * KW+AI  — kw_score >= KW_MIN AND ai_score >= AI_MIN  (ai_score > 0)

A composite "dual_score" = geometric mean of the two non-zero scores is
used to rank within each category.

Thresholds are calibrated to ~90th-percentile scores (not absolute), so
the detector adapts to pipeline output scale.

Artifacts:
  /home/zeph/logs/triple_score_consensus.json

Standalone: python3 -m analytics.triple_score_consensus
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DB_PATH = str(BASE / "data" / "articles.db")
OUT_PATH = Path("/home/zeph/logs/triple_score_consensus.json")

SCAN_LIMIT = 8000       # bounded idx_first_seen scan (~recent few hours)
KW_MIN = 4.0            # calibrated ~80th pct for live articles
ML_MIN = 5.0            # calibrated ~85th pct for ml-scored articles
AI_MIN = 4.0            # calibrated ~80th pct for ai-scored articles
TOP_N = 15


def _geometric_mean(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return math.sqrt(a * b)


def main() -> None:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA busy_timeout=8000")

    rows = conn.execute(
        """
        SELECT title, source, first_seen, kw_score, ai_score, ml_score, urgency
        FROM articles
        ORDER BY first_seen DESC
        LIMIT ?
        """,
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    kw_ml: list[dict] = []
    kw_ai: list[dict] = []
    total_ml_scored = 0
    total_ai_scored = 0

    for title, source, first_seen, kw, ai, ml, urgency in rows:
        src = source or ""
        # Skip synthetic rows
        if src.startswith("backtest") or src.startswith("opus_annotation"):
            continue

        kw = kw or 0.0

        if ml is not None and ml > 0:
            total_ml_scored += 1
            if kw >= KW_MIN and ml >= ML_MIN:
                kw_ml.append({
                    "title": (title or "")[:100],
                    "source": src,
                    "first_seen": first_seen,
                    "kw_score": round(kw, 2),
                    "ml_score": round(ml, 2),
                    "urgency": urgency,
                    "dual_score": round(_geometric_mean(kw, ml), 3),
                })

        if ai is not None and ai > 0:
            total_ai_scored += 1
            if kw >= KW_MIN and ai >= AI_MIN:
                kw_ai.append({
                    "title": (title or "")[:100],
                    "source": src,
                    "first_seen": first_seen,
                    "kw_score": round(kw, 2),
                    "ai_score": round(ai, 2),
                    "urgency": urgency,
                    "dual_score": round(_geometric_mean(kw, ai), 3),
                })

    kw_ml.sort(key=lambda r: r["dual_score"], reverse=True)
    kw_ai.sort(key=lambda r: r["dual_score"], reverse=True)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_limit": SCAN_LIMIT,
        "thresholds": {"kw": KW_MIN, "ml": ML_MIN, "ai": AI_MIN},
        "scanned": len(rows),
        "ml_scored_in_scan": total_ml_scored,
        "ai_scored_in_scan": total_ai_scored,
        "kw_ml_consensus_count": len(kw_ml),
        "kw_ai_consensus_count": len(kw_ai),
        "kw_ml_top": kw_ml[:TOP_N],
        "kw_ai_top": kw_ai[:TOP_N],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    total_consensus = len(kw_ml) + len(kw_ai)
    top_item = (kw_ml + kw_ai)
    top_item.sort(key=lambda r: r["dual_score"], reverse=True)
    top = top_item[0] if top_item else None

    print(
        f"cross_scorer_consensus: {total_consensus} consensus articles "
        f"({len(kw_ml)} KW+ML, {len(kw_ai)} KW+AI) from {SCAN_LIMIT} scanned; "
        + (f"best: dual={top['dual_score']} {top['source']}: {top['title'][:50]}" if top else "none found")
    )


if __name__ == "__main__":
    main()
