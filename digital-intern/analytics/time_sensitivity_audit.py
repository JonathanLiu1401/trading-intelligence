"""Time-sensitivity feature effectiveness audit.

Answers the ML engineer question: **Does ``time_sensitivity`` actually predict
urgency, or is it decorative noise?**

No existing module analyses ``time_sensitivity`` directly. The column is set
by collectors / early ML stages to flag articles that are time-bound (e.g.
earnings releases, FDA decisions).  If the feature is working it should
correlate positively with ``urgency >= 2`` outcomes and with high ``ml_score``.

Analysis:
  1. Split last SCAN_LIMIT articles into ``ts_null`` (no value) vs ``ts_set``.
  2. For each group: count, avg ai_score, avg ml_score, urgent_pct.
  3. Bucket ``ts_set`` rows by value range: [0,2), [2,5), [5,10] to test
     monotonicity — a well-calibrated feature should show increasing urgency
     rate at higher values.
  4. Compute Pearson r between time_sensitivity and urgency for ``ts_set`` rows.

Design constraints (mirrors rest of analytics/ codebase):
  * Bounded LIMIT scan via idx_first_seen — no full-table COUNT.
  * Read-only sqlite uri mode, busy_timeout=8 000 ms.
  * _LIVE_ONLY_CLAUSE applied — backtest rows excluded.

Output: /home/zeph/logs/time_sensitivity_audit.json
Standalone: python3 -m analytics.time_sensitivity_audit
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT = Path("/home/zeph/logs/time_sensitivity_audit.json")
SCAN_LIMIT = 10_000

BUCKETS = [
    ("0.0-0.4", 0.0, 0.4),
    ("0.4-0.6", 0.4, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-1.0", 0.8, 1.01),
]


def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return round(num / (sx * sy), 4)


def _stats(rows: list[tuple]) -> dict:
    """rows: (ai_score, ml_score, urgency)"""
    if not rows:
        return {"count": 0, "avg_ai_score": None, "avg_ml_score": None, "urgent_pct": None}
    ai_vals  = [r[0] for r in rows if r[0] is not None]
    ml_vals  = [r[1] for r in rows if r[1] is not None]
    urgent   = sum(1 for r in rows if (r[2] or 0) >= 2)
    return {
        "count":        len(rows),
        "avg_ai_score": round(mean(ai_vals), 3) if ai_vals else None,
        "avg_ml_score": round(mean(ml_vals), 3) if ml_vals else None,
        "urgent_pct":   round(urgent / len(rows) * 100, 2),
    }


def main() -> int:
    db = _get_db_path()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA busy_timeout=8000")
    conn.execute("PRAGMA query_only=ON")

    rows = conn.execute(
        "SELECT time_sensitivity, ai_score, ml_score, urgency FROM articles "
        f"INDEXED BY idx_first_seen WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    ts_null, ts_set = [], []
    for ts, ai, ml, urg in rows:
        triple = (ai, ml, urg)
        if ts is None:
            ts_null.append(triple)
        else:
            ts_set.append((ts, ai, ml, urg))

    null_stats = _stats(ts_null)
    set_stats  = _stats([(r[1], r[2], r[3]) for r in ts_set])

    # Bucket breakdown for ts_set
    bucket_stats = {}
    for label, lo, hi in BUCKETS:
        subset = [(r[1], r[2], r[3]) for r in ts_set if lo <= (r[0] or 0) < hi]
        bucket_stats[label] = _stats(subset)

    # Pearson r(time_sensitivity, urgency)
    ts_vals  = [r[0] for r in ts_set if r[0] is not None]
    urg_vals = [r[3] or 0 for r in ts_set if r[0] is not None]
    pearson  = _pearson_r(ts_vals, urg_vals)

    # Lift: how much more urgent are ts_set rows vs ts_null?
    null_u = null_stats["urgent_pct"] or 0
    set_u  = set_stats["urgent_pct"] or 0
    lift   = round(set_u - null_u, 2) if null_stats["count"] and set_stats["count"] else None

    result = {
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "scan_limit":       SCAN_LIMIT,
        "rows_scanned":     len(rows),
        "ts_null_rows":     len(ts_null),
        "ts_set_rows":      len(ts_set),
        "ts_coverage_pct":  round(len(ts_set) / len(rows) * 100, 2) if rows else 0,
        "null_group":       null_stats,
        "set_group":        set_stats,
        "urgency_lift_pct": lift,
        "pearson_r_ts_urgency": pearson,
        "buckets":          bucket_stats,
        "verdict": (
            "EFFECTIVE"   if (pearson or 0) >= 0.05 and (lift or 0) > 1.0 else
            "WEAK"        if (pearson or 0) >= 0.01 or (lift or 0) > 0.5 else
            "INEFFECTIVE" if len(ts_set) >= 100 else
            "INSUFFICIENT_DATA"
        ),
    }

    OUT.write_text(json.dumps(result, indent=2))

    # Concise stdout summary
    print(f"rows_scanned={len(rows)}  ts_set={len(ts_set)} ({result['ts_coverage_pct']}%)")
    print(f"null_group  urgent_pct={null_stats['urgent_pct']}%  avg_ml={null_stats['avg_ml_score']}")
    print(f"set_group   urgent_pct={set_stats['urgent_pct']}%   avg_ml={set_stats['avg_ml_score']}")
    print(f"lift={lift}pp  pearson_r={pearson}  verdict={result['verdict']}")
    for bk, bv in bucket_stats.items():
        print(f"  bucket[{bk}]: n={bv['count']}  urgent%={bv['urgent_pct']}  avg_ml={bv['avg_ml_score']}")
    print(f"Written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
