"""ML score threshold calibration: precision/recall/F1 vs urgency labels.

Computes precision, recall, and F1 for a grid of ml_score (and ai_score)
decision thresholds against the ground-truth ``urgency >= 2`` labels.
Identifies the threshold that maximises F1 so the paper trader's signal
filter can be tuned to the actual data distribution rather than an arbitrary
cut-point.

Why this matters: the current paper trader uses a fixed ml_score threshold
(see config/paper_trader*.json).  If the ML model's score scale shifts after
retraining or fine-tuning, the optimal decision boundary shifts with it.  This
report makes the drift visible before it causes missed trades or false signals.

Design constraints (same as all analytics in this codebase):
  * No full COUNT(*): bounded idx_first_seen scan only.
  * Read-only connection, busy_timeout=5 000 ms.
  * ``_LIVE_ONLY_CLAUSE`` applied — backtest rows carry curated urgency labels
    and would inflate the apparent precision of any threshold.

Output: /home/zeph/logs/ml_score_calibration.json
Standalone: ``python3 -m analytics.ml_score_calibration``
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT = Path("/home/zeph/logs/ml_score_calibration.json")
SCAN_LIMIT = 8000
URGENCY_THRESHOLD = 2  # urgency >= this = positive class
# Grid of thresholds to evaluate (inclusive lower bound)
ML_GRID = [round(v * 0.5, 1) for v in range(4, 20)]   # 2.0 … 9.5
AI_GRID = [round(v * 0.5, 1) for v in range(4, 20)]


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _calibrate(rows: list[tuple], score_idx: int, grid: list[float], label: str) -> dict:
    """Return threshold sweep results for a given score column."""
    scored = [(r[score_idx], r[2]) for r in rows if r[score_idx] is not None]
    if not scored:
        return {"error": "no scored rows", "label": label}

    positives = sum(1 for _, u in scored if u >= URGENCY_THRESHOLD)
    total = len(scored)

    sweep = []
    best_f1 = -1.0
    best_thresh = None
    for thresh in grid:
        tp = sum(1 for s, u in scored if s >= thresh and u >= URGENCY_THRESHOLD)
        fp = sum(1 for s, u in scored if s >= thresh and u < URGENCY_THRESHOLD)
        fn = positives - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = _f1(prec, rec)
        sweep.append({
            "threshold": thresh,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f, 4),
            "flagged": tp + fp,
        })
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh

    return {
        "label": label,
        "total_rows": total,
        "positive_rows": positives,
        "positive_pct": round(100 * positives / total, 2) if total else 0,
        "best_threshold": best_thresh,
        "best_f1": round(best_f1, 4),
        "sweep": sweep,
    }


def compute() -> dict:
    db_path = _get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    conn.execute("PRAGMA busy_timeout=5000")

    rows = conn.execute(
        f"""
        SELECT ml_score, ai_score, urgency
          FROM articles INDEXED BY idx_first_seen
         WHERE ml_score IS NOT NULL AND ai_score IS NOT NULL AND urgency IS NOT NULL
           AND {_LIVE_ONLY_CLAUSE}
         ORDER BY first_seen DESC LIMIT ?
        """,
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    ml_cal = _calibrate(rows, score_idx=0, grid=ML_GRID, label="ml_score")
    ai_cal = _calibrate(rows, score_idx=1, grid=AI_GRID, label="ai_score")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_limit": SCAN_LIMIT,
        "scanned": len(rows),
        "urgency_threshold": URGENCY_THRESHOLD,
        "ml_score": ml_cal,
        "ai_score": ai_cal,
        "summary": {
            "optimal_ml_threshold": ml_cal.get("best_threshold"),
            "optimal_ml_f1": ml_cal.get("best_f1"),
            "optimal_ai_threshold": ai_cal.get("best_threshold"),
            "optimal_ai_f1": ai_cal.get("best_f1"),
        },
    }

    OUT.write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    result = compute()
    ml = result["ml_score"]
    ai = result["ai_score"]
    print(
        f"ml_score_calibration: scanned={result['scanned']} "
        f"positives={ml['positive_rows']} ({ml['positive_pct']}%)"
    )
    print(
        f"  ml_score  optimal threshold={ml['best_threshold']}  F1={ml['best_f1']:.4f}"
    )
    print(
        f"  ai_score  optimal threshold={ai['best_threshold']}  F1={ai['best_f1']:.4f}"
    )
    # Print top-5 rows of ml sweep near the optimal threshold
    optimal = ml.get("best_threshold")
    for row in ml.get("sweep", []):
        dist = abs(row["threshold"] - optimal) if optimal is not None else 999
        if dist <= 1.5:
            print(
                f"    ml≥{row['threshold']:.1f}: "
                f"prec={row['precision']:.3f} rec={row['recall']:.3f} "
                f"F1={row['f1']:.4f} flagged={row['flagged']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
