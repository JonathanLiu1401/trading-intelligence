"""Score pipeline coverage by collector source.

Answers: "Which collectors' articles are actually reaching the ML/LLM scoring
stage, and which are falling through as unscored?"

Existing tools track score *quality* (source_quality.py: avg scores per source)
or overall score *distribution* (score_source_breakdown.py: global ml/llm/unscored
buckets). Neither shows per-source coverage rates — i.e. which collectors are
getting their articles scored at all.

This module computes, per collector source over the last WINDOW_HOURS:
  * total articles ingested
  * ml_scored_pct   — fraction with score_source='ml'
  * llm_scored_pct  — fraction with score_source='llm'
  * unscored_pct    — fraction with score_source IS NULL
  * avg_ml_score    — among ml-scored rows (0.0 if none)
  * verdict         — WELL_COVERED / PARTIAL / UNSCORED

Design constraints (mirrors analytics/ patterns):
  * Bounded LIMIT scan via idx_first_seen — no full-table scan.
  * Read-only sqlite URI, busy_timeout=8000 ms.
  * _LIVE_ONLY_CLAUSE applied — backtest rows excluded.

Output: /home/zeph/logs/score_pipeline_coverage.json
Standalone: python3 -m analytics.score_pipeline_coverage
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = BASE / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/score_pipeline_coverage.json")

SCAN_LIMIT = 8000
WINDOW_HOURS = 24
MIN_ARTICLES = 10       # sources below this are reported but labeled 'low_volume'
TOP_N = 20              # top sources by article count shown in detail


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = raw.strip().replace("Z", "+00:00")
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


def run() -> dict:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=8)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        f"""
        SELECT source, score_source, ml_score, first_seen
        FROM articles
        WHERE {_LIVE_ONLY_CLAUSE}
        ORDER BY first_seen DESC
        LIMIT {SCAN_LIMIT}
        """,
    ).fetchall()
    con.close()

    now = datetime.now(timezone.utc)
    cutoff = now.replace(microsecond=0) - __import__("datetime").timedelta(hours=WINDOW_HOURS)

    # bucket per source
    buckets: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "ml": 0, "llm": 0, "unscored": 0, "ml_scores": []}
    )

    in_window = 0
    for row in rows:
        ts = _parse_ts(row["first_seen"])
        if ts is None or ts < cutoff:
            continue
        in_window += 1
        src = row["source"] or "unknown"
        b = buckets[src]
        b["total"] += 1
        ss = row["score_source"]
        if ss == "ml":
            b["ml"] += 1
            if row["ml_score"] is not None:
                b["ml_scores"].append(row["ml_score"])
        elif ss == "llm":
            b["llm"] += 1
        else:
            b["unscored"] += 1

    # compute verdicts
    sources_detail = []
    total_ml = total_llm = total_unscored = total_all = 0
    for src, b in buckets.items():
        n = b["total"]
        total_all += n
        total_ml += b["ml"]
        total_llm += b["llm"]
        total_unscored += b["unscored"]

        ml_pct = round(b["ml"] / n * 100, 1) if n else 0
        llm_pct = round(b["llm"] / n * 100, 1) if n else 0
        unscored_pct = round(b["unscored"] / n * 100, 1) if n else 0
        avg_ml = round(sum(b["ml_scores"]) / len(b["ml_scores"]), 3) if b["ml_scores"] else None

        if n < MIN_ARTICLES:
            verdict = "low_volume"
        elif unscored_pct >= 50:
            verdict = "UNSCORED"
        elif unscored_pct >= 20:
            verdict = "PARTIAL"
        else:
            verdict = "WELL_COVERED"

        sources_detail.append({
            "source": src,
            "total": n,
            "ml_pct": ml_pct,
            "llm_pct": llm_pct,
            "unscored_pct": unscored_pct,
            "avg_ml_score": avg_ml,
            "verdict": verdict,
        })

    sources_detail.sort(key=lambda x: x["total"], reverse=True)

    unscored_sources = [s for s in sources_detail if s["verdict"] == "UNSCORED" and s["total"] >= MIN_ARTICLES]
    partial_sources  = [s for s in sources_detail if s["verdict"] == "PARTIAL" and s["total"] >= MIN_ARTICLES]

    result = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "scanned": len(rows),
        "in_window": in_window,
        "global": {
            "total": total_all,
            "ml_pct": round(total_ml / total_all * 100, 1) if total_all else 0,
            "llm_pct": round(total_llm / total_all * 100, 1) if total_all else 0,
            "unscored_pct": round(total_unscored / total_all * 100, 1) if total_all else 0,
        },
        "n_sources": len(sources_detail),
        "n_unscored": len(unscored_sources),
        "n_partial": len(partial_sources),
        "unscored_sources": [s["source"] for s in unscored_sources[:10]],
        "partial_sources": [s["source"] for s in partial_sources[:10]],
        "top_sources": sources_detail[:TOP_N],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = run()
    g = r["global"]
    print(
        f"Score pipeline coverage | {r['in_window']} articles in {r['window_hours']}h "
        f"| ML: {g['ml_pct']}% LLM: {g['llm_pct']}% unscored: {g['unscored_pct']}%"
    )
    if r["unscored_sources"]:
        print(f"UNSCORED sources ({r['n_unscored']}): {', '.join(r['unscored_sources'][:5])}")
    if r["partial_sources"]:
        print(f"PARTIAL sources ({r['n_partial']}): {', '.join(r['partial_sources'][:5])}")
    top = r["top_sources"][:3]
    for s in top:
        print(f"  {s['source']}: {s['total']} arts | ML {s['ml_pct']}% unscored {s['unscored_pct']}% verdict={s['verdict']}")
