"""Source score volatility: per-source std-dev of ai_score.

Identifies collectors whose scores swing widely (noisy/inconsistent) vs sources
whose articles cluster tightly (predictable signal quality). Pairs with
[[scorer_skew]] which compares ai vs ml; this one is intra-source variance only.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE

DB = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT = Path("/home/zeph/logs/source_score_volatility.json")
SCAN_LIMIT = 5000
MIN_PER_SOURCE = 8


def compute():
    conn = sqlite3.connect(str(DB))
    conn.execute("PRAGMA busy_timeout=5000")
    # Use the canonical `_LIVE_ONLY_CLAUSE` rather than a partial filter:
    # `source NOT LIKE 'backtest_run_%'` alone lets `backtest://` URLs,
    # `backtest_winner`-style sources, and `opus_annotation*` synthetic rows
    # leak through and inflate the per-source ai_score variance with fractional
    # training labels. Same drift class as the publish_lag_audit /
    # source_diversity / trend_velocity fixes; pinned by
    # tests/test_analytics_backtest_isolation.py.
    #
    # ai_score > 0: the column defaults to 0 (REAL DEFAULT 0), never NULL,
    # so `ai_score IS NOT NULL` was a tautology that included every unscored
    # row at ai_score=0 in the per-source variance calculation. A source with
    # 100 LLM-scored rows (3..7) and 900 unscored zeros looks vastly more
    # "noisy" than its real LLM-label spread; urgency_scorer floors any
    # LLM-touched row at 0.01 so `> 0` is the canonical "the LLM graded this"
    # filter (same SSOT as ml/score_agreement._MIN_AI).
    rows = conn.execute(
        f"""
        SELECT source, ai_score
          FROM articles
         WHERE id IN (SELECT id FROM articles ORDER BY id DESC LIMIT ?)
           AND ai_score > 0
           AND {_LIVE_ONLY_CLAUSE}
        """,
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    buckets: dict[str, list[float]] = {}
    for source, ai in rows:
        buckets.setdefault(source, []).append(float(ai))

    out = {}
    for src, vals in buckets.items():
        if len(vals) < MIN_PER_SOURCE:
            continue
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = math.sqrt(var)
        out[src] = {
            "n": n,
            "mean": round(mean, 3),
            "std": round(std, 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
        }

    ranked = sorted(out.items(), key=lambda kv: kv[1]["std"], reverse=True)
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
    print("top noisiest sources (highest ai_score std):")
    for entry in p["ranked"][:5]:
        print(
            f"  {entry['source'][:44]:<44} n={entry['n']:>4}  "
            f"mean={entry['mean']:>6.2f}  std={entry['std']:>5.2f}  "
            f"range=[{entry['min']:.1f},{entry['max']:.1f}]"
        )


if __name__ == "__main__":
    main()
