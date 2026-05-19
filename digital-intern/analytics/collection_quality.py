"""Collection quality report: per-source volume, avg score, urgent %.

Hourly snapshot of collector health so weak feeds become visible.
Reads the most recent SCAN_LIMIT rows (excluding backtest sources) and
aggregates by source. Output: /home/zeph/logs/collection_quality.json
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE

DB = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT = Path("/home/zeph/logs/collection_quality.json")
SCAN_LIMIT = 8000
MIN_PER_SOURCE = 3


def compute():
    conn = sqlite3.connect(str(DB))
    conn.execute("PRAGMA busy_timeout=5000")
    # Canonical `_LIVE_ONLY_CLAUSE` — synthetic backtest replay and
    # opus-annotation rows carry their own ai_score / urgency values and would
    # inflate per-source averages with training-pool magnitudes if they leaked
    # through the partial `source NOT LIKE 'backtest_run_%'` filter.
    rows = conn.execute(
        f"""
        SELECT source, ai_score, ml_score, urgency
          FROM articles
         WHERE id IN (SELECT id FROM articles ORDER BY id DESC LIMIT ?)
           AND {_LIVE_ONLY_CLAUSE}
        """,
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    agg: dict[str, dict] = {}
    for source, ai, ml, urg in rows:
        s = agg.setdefault(
            source,
            {"n": 0, "ai_sum": 0.0, "ai_n": 0, "ml_sum": 0.0, "ml_n": 0, "urgent": 0},
        )
        s["n"] += 1
        if ai is not None:
            s["ai_sum"] += ai
            s["ai_n"] += 1
        if ml is not None:
            s["ml_sum"] += ml
            s["ml_n"] += 1
        if urg is not None and urg >= 2:
            s["urgent"] += 1

    out = {}
    for src, s in agg.items():
        if s["n"] < MIN_PER_SOURCE:
            continue
        out[src] = {
            "n": s["n"],
            "avg_ai": round(s["ai_sum"] / s["ai_n"], 3) if s["ai_n"] else None,
            "avg_ml": round(s["ml_sum"] / s["ml_n"], 3) if s["ml_n"] else None,
            "pct_urgent": round(100.0 * s["urgent"] / s["n"], 2),
            "urgent": s["urgent"],
        }

    ranked = sorted(out.items(), key=lambda kv: kv[1]["n"], reverse=True)
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
    for entry in p["ranked"][:8]:
        ai = entry["avg_ai"]
        ml = entry["avg_ml"]
        print(
            f"  {entry['source'][:42]:<42} n={entry['n']:>4}  "
            f"ai={ai if ai is not None else '  n/a':>6}  "
            f"ml={ml if ml is not None else '  n/a':>6}  "
            f"urgent={entry['pct_urgent']:>5.2f}% ({entry['urgent']})"
        )


if __name__ == "__main__":
    main()
