"""Scorer skew: per-source gap between ai_score and ml_score.

Surfaces collectors whose keyword/AI judgement diverges from the ML model.
Large positive ml-ai gap = ML is more bullish than ai_score; negative = ML rejects
items the ai_score thought were strong.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE

DB = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT = Path("/home/zeph/logs/scorer_skew.json")
SCAN_LIMIT = 4000
MIN_PER_SOURCE = 5


def compute(window_hours: int = 6):
    conn = sqlite3.connect(str(DB))
    conn.execute("PRAGMA busy_timeout=5000")
    # Canonical `_LIVE_ONLY_CLAUSE` — the previous `source NOT LIKE
    # 'backtest_run_%'`-only filter let `backtest_*` (e.g. `backtest_winner`)
    # and `opus_annotation*` rows through, polluting per-source ai-vs-ml gap
    # averages with synthetic training labels. Currently masked by the
    # `ml_score IS NOT NULL` predicate (synthetic rows never go through ML
    # scoring), but the partial filter is the same drift class as elsewhere.
    rows = conn.execute(
        f"""
        SELECT source, ai_score, ml_score
          FROM articles
         WHERE id IN (SELECT id FROM articles ORDER BY id DESC LIMIT ?)
           AND ai_score IS NOT NULL
           AND ml_score IS NOT NULL
           AND {_LIVE_ONLY_CLAUSE}
        """,
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    agg: dict[str, dict] = {}
    for source, ai, ml in rows:
        s = agg.setdefault(source, {"n": 0, "ai": 0.0, "ml": 0.0, "gap": 0.0})
        s["n"] += 1
        s["ai"] += ai
        s["ml"] += ml
        s["gap"] += (ml - ai)

    out = {}
    for src, s in agg.items():
        if s["n"] < MIN_PER_SOURCE:
            continue
        n = s["n"]
        out[src] = {
            "n": n,
            "avg_ai": round(s["ai"] / n, 3),
            "avg_ml": round(s["ml"] / n, 3),
            "avg_gap_ml_minus_ai": round(s["gap"] / n, 3),
        }

    ranked = sorted(out.items(), key=lambda kv: abs(kv[1]["avg_gap_ml_minus_ai"]), reverse=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_limit": SCAN_LIMIT,
        "scanned": len(rows),
        "sources_reported": len(ranked),
        "ranked": [{"source": k, **v} for k, v in ranked],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    return payload


def main():
    p = compute()
    print(f"scanned={p['scanned']} sources_reported={p['sources_reported']}")
    print(f"output={OUT}")
    for entry in p["ranked"][:5]:
        print(
            f"  {entry['source'][:48]:<48} n={entry['n']:>4}  "
            f"ai={entry['avg_ai']:>6.2f}  ml={entry['avg_ml']:>6.2f}  "
            f"gap={entry['avg_gap_ml_minus_ai']:+.2f}"
        )


if __name__ == "__main__":
    main()
