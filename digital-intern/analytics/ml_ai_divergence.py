"""ML vs LLM (AI) score divergence tracker.

Identifies articles where the ML model (ml_score 0-10) and the LLM scorer
(ai_score 0-10) disagree significantly, which is the primary signal for ML
model drift and calibration problems.

Two divergence regimes tracked over a bounded recent window:

* **ml_overscored** (high ml / low ai): ML scored >= ML_HIGH but LLM gave
  ai_score < AI_LOW. ML is tagging noise as signal. High counts indicate the
  model has overfit to surface patterns that the LLM rejects on content.

* **ml_underscored** (low ml / high ai): LLM gave ai_score >= AI_HIGH but
  ML scored < ML_LOW. ML is missing material that the LLM rated important.
  These are the most dangerous failures — real signals suppressed by the
  pipeline before they reach triage.

Per-source breakdowns show which collectors produce the most divergent rows.
Top-5 examples (by divergence magnitude) are included for each regime.

Design constraints (identical to all analytics siblings):
  * Bounded SCAN_LIMIT idx_first_seen read — never full-table scan.
  * Read-only sqlite URI — never contends with daemon writers.
  * USB-safe busy_timeout.
  * _LIVE_ONLY_CLAUSE discipline — synthetic backtest/opus rows excluded.

Artifacts:
  * /home/zeph/logs/ml_ai_divergence.json
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

try:
    from storage.article_store import _get_db_path as _resolve_db_path
except Exception:
    _resolve_db_path = None

DB_PATH = (
    Path(_resolve_db_path()) if _resolve_db_path is not None
    else BASE / "data" / "articles.db"
)
LOG_DIR = Path("/home/zeph/logs")
OUT_PATH = LOG_DIR / "ml_ai_divergence.json"

SCAN_LIMIT = 8000
# ml_score and ai_score are both on 0-10 scale
ML_HIGH = 6.0   # ML thinks it's important
ML_LOW  = 2.5   # ML thinks it's noise
AI_HIGH = 6.0   # LLM thinks it's important
AI_LOW  = 1.5   # LLM thinks it's noise
MAX_EXAMPLES = 5

_LIVE_ONLY = (
    "source NOT LIKE 'backtest%' "
    "AND source NOT LIKE 'opus_annotation%' "
    "AND source NOT LIKE 'backtest_run%' "
    "AND url NOT LIKE 'backtest://%'"
)


def compute() -> dict:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10)
    conn.execute("PRAGMA busy_timeout=8000")
    try:
        rows = conn.execute(
            f"""
            SELECT source, ml_score, ai_score, title, first_seen
              FROM articles
             WHERE {_LIVE_ONLY}
               AND ml_score IS NOT NULL
               AND ai_score IS NOT NULL
             ORDER BY first_seen DESC
             LIMIT ?
            """,
            (SCAN_LIMIT,),
        ).fetchall()
    finally:
        conn.close()

    total = len(rows)
    over_by_source: dict[str, int] = defaultdict(int)
    under_by_source: dict[str, int] = defaultdict(int)
    over_examples: list[dict] = []
    under_examples: list[dict] = []

    for source, ml, ai, title, first_seen in rows:
        ml = ml or 0.0
        ai = ai or 0.0
        gap = ml - ai

        if ml >= ML_HIGH and ai < AI_LOW:
            # ML overscored: ML thinks important, LLM says noise
            over_by_source[source or "unknown"] += 1
            over_examples.append({
                "title": (title or "")[:100],
                "source": source or "unknown",
                "ml_score": round(ml, 2),
                "ai_score": round(ai, 2),
                "gap": round(gap, 2),
                "first_seen": first_seen,
            })

        elif ai >= AI_HIGH and ml < ML_LOW:
            # ML underscored: LLM thinks important, ML says noise
            under_by_source[source or "unknown"] += 1
            under_examples.append({
                "title": (title or "")[:100],
                "source": source or "unknown",
                "ml_score": round(ml, 2),
                "ai_score": round(ai, 2),
                "gap": round(gap, 2),
                "first_seen": first_seen,
            })

    # Sort examples by divergence magnitude (largest gap first)
    over_examples.sort(key=lambda x: abs(x["gap"]), reverse=True)
    under_examples.sort(key=lambda x: abs(x["gap"]), reverse=True)

    over_by_src = sorted(over_by_source.items(), key=lambda kv: kv[1], reverse=True)
    under_by_src = sorted(under_by_source.items(), key=lambda kv: kv[1], reverse=True)

    total_over = sum(over_by_source.values())
    total_under = sum(under_by_source.values())

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_limit": SCAN_LIMIT,
        "scanned": total,
        "ml_overscored": {
            "description": "ML high (>=6) but LLM low (<1.5) — ML false positives",
            "count": total_over,
            "pct": round(100.0 * total_over / total, 2) if total else 0.0,
            "top_sources": [{"source": s, "count": n} for s, n in over_by_src[:10]],
            "examples": over_examples[:MAX_EXAMPLES],
        },
        "ml_underscored": {
            "description": "LLM high (>=6) but ML low (<2.5) — ML false negatives",
            "count": total_under,
            "pct": round(100.0 * total_under / total, 2) if total else 0.0,
            "top_sources": [{"source": s, "count": n} for s, n in under_by_src[:10]],
            "examples": under_examples[:MAX_EXAMPLES],
        },
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    import sys
    sys.path.insert(0, str(BASE))
    result = compute()
    over = result["ml_overscored"]
    under = result["ml_underscored"]
    print(
        f"ml_ai_divergence: scanned={result['scanned']} | "
        f"ml_overscored={over['count']} ({over['pct']}%) | "
        f"ml_underscored={under['count']} ({under['pct']}%)"
    )
    if under["examples"]:
        ex = under["examples"][0]
        print(f"  worst miss: [{ex['source']}] ai={ex['ai_score']} ml={ex['ml_score']} — {ex['title'][:70]}")
    if over["examples"]:
        ex = over["examples"][0]
        print(f"  worst overfit: [{ex['source']}] ml={ex['ml_score']} ai={ex['ai_score']} — {ex['title'][:70]}")


if __name__ == "__main__":
    main()
