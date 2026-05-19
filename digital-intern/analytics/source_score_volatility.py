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

DB = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT = Path("/home/zeph/logs/source_score_volatility.json")
SCAN_LIMIT = 5000
MIN_PER_SOURCE = 8


def compute():
    conn = sqlite3.connect(str(DB))
    conn.execute("PRAGMA busy_timeout=5000")
    rows = conn.execute(
        """
        SELECT source, ai_score
          FROM articles
         WHERE id IN (SELECT id FROM articles ORDER BY id DESC LIMIT ?)
           AND ai_score IS NOT NULL
           AND source NOT LIKE 'backtest_run_%'
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
