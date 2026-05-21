"""Scoring funnel: article drop-off at each pipeline gate over the last 24h.

Answers the operator question: *Where are signals getting filtered out?*
Existing tools cover individual stages (scoring_backlog_audit: unscored count,
score_drift_detector: ml_score mean, daily_digest: urgent output). This module
is the only one that stitches all gates into a single conversion funnel,
showing absolute counts and per-stage retention rates.

Funnel stages (in order):
  1. ingested     — all live articles first_seen in the last WINDOW_HOURS
  2. kw_signal    — kw_score >= KW_MIN (keyword filter passed)
  3. ai_scored    — ai_score > 0 (LLM scoring ran)
  4. ai_high      — ai_score >= AI_HIGH (high-confidence LLM signal)
  5. urgency_1    — urgency >= 1 (queued for alerting)
  6. urgency_2    — urgency >= 2 (alert delivered)

Design constraints (mirrors score_drift_detector):
  * No full-table COUNT/aggregation — bounded LIMIT scan via idx_first_seen.
  * All timestamp parsing in Python; sqlite datetime() comparisons fail on
    the ISO-8601+tz format stored in first_seen (e.g. "2026-05-21T03:02:22+00:00").
  * Read-only (uri mode). Zero DB writes.

Output: /home/zeph/logs/scoring_funnel.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/scoring_funnel.json")

WINDOW_HOURS = 24
SCAN_LIMIT = 8000   # enough to cover 24h at typical ingest rates
KW_MIN = 1.0        # minimum kw_score to count as "has keyword signal"
AI_HIGH = 4.0       # ai_score threshold for "high confidence"


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = raw.strip()
    # Handle both space-separated and T-separated, with or without tz offset
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.strptime(s[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compute() -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT first_seen, kw_score, ai_score, urgency "
        f"FROM articles WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    # Filter to actual 24h window in Python (avoids ISO tz comparison mismatch)
    window_rows = []
    for r in rows:
        ts = _parse_ts(r["first_seen"])
        if ts and ts >= cutoff:
            window_rows.append(r)

    n_ingested = len(window_rows)
    n_kw_signal = sum(1 for r in window_rows if (r["kw_score"] or 0) >= KW_MIN)
    n_ai_scored = sum(1 for r in window_rows if (r["ai_score"] or 0) > 0)
    n_ai_high   = sum(1 for r in window_rows if (r["ai_score"] or 0) >= AI_HIGH)
    n_urgency_1 = sum(1 for r in window_rows if (r["urgency"] or 0) >= 1)
    n_urgency_2 = sum(1 for r in window_rows if (r["urgency"] or 0) >= 2)

    def pct(num: int, denom: int) -> float | None:
        return round(100.0 * num / denom, 1) if denom else None

    stages = [
        {"stage": "ingested",   "count": n_ingested,  "retention_pct": 100.0},
        {"stage": "kw_signal",  "count": n_kw_signal,  "retention_pct": pct(n_kw_signal,  n_ingested)},
        {"stage": "ai_scored",  "count": n_ai_scored,  "retention_pct": pct(n_ai_scored,  n_ingested)},
        {"stage": "ai_high",    "count": n_ai_high,    "retention_pct": pct(n_ai_high,    n_ingested)},
        {"stage": "urgency_1",  "count": n_urgency_1,  "retention_pct": pct(n_urgency_1,  n_ingested)},
        {"stage": "urgency_2",  "count": n_urgency_2,  "retention_pct": pct(n_urgency_2,  n_ingested)},
    ]

    # Bottleneck: biggest absolute drop between adjacent stages
    bottleneck = None
    max_drop = -1
    for i in range(1, len(stages)):
        drop = stages[i - 1]["count"] - stages[i]["count"]
        if drop > max_drop:
            max_drop = drop
            bottleneck = stages[i]["stage"]

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": WINDOW_HOURS,
        "scan_limit": SCAN_LIMIT,
        "rows_scanned": len(rows),
        "rows_in_window": n_ingested,
        "kw_min_threshold": KW_MIN,
        "ai_high_threshold": AI_HIGH,
        "stages": stages,
        "bottleneck_stage": bottleneck,
        "bottleneck_drop": max_drop,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    base = Path(__file__).resolve().parents[1]
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))

    data = compute()
    stages = data["stages"]
    print(f"Scoring funnel — last {data['window_hours']}h  "
          f"(scanned {data['rows_scanned']} rows, {data['rows_in_window']} in window)")
    print(f"{'Stage':<14} {'Count':>6}  {'Retention':>10}")
    print("-" * 36)
    for s in stages:
        ret = f"{s['retention_pct']:.1f}%" if s["retention_pct"] is not None else "  n/a"
        print(f"{s['stage']:<14} {s['count']:>6}  {ret:>10}")
    print(f"\nBottleneck: {data['bottleneck_stage']}  "
          f"(drop of {data['bottleneck_drop']} articles)")
    print(f"Written → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
