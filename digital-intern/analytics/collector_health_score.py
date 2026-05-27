"""Collector composite health score.

Aggregates four existing analytics snapshots into a single 0-100 health score
per collector, giving operators one ranked leaderboard instead of four separate
reports.

Sources consumed (all read-only, no DB access):
  * /home/zeph/logs/collection_quality.json  — volume, avg ml_score, urgent%
  * /home/zeph/logs/collector_uptime.json    — gap analysis
  * /home/zeph/logs/scoring_backlog.json     — ml_score coverage ratio
  * /home/zeph/logs/publish_lag.json         — ingest latency per collector

Scoring weights (sum = 100):
  quality   (30): avg ml_score / 10 * 30
  urgency   (15): pct_urgent * 15
  uptime    (25): 25 - gap penalty (max_gap_min / 120 capped at 1) * 25
  coverage  (20): (1 - unscored_ratio) * 20
  freshness (10): fresh_5m_pct / 100 * 10

Output: /home/zeph/logs/collector_health_score.json

Standalone: python3 -m analytics.collector_health_score
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

LOGS = Path("/home/zeph/logs")
OUT = LOGS / "collector_health_score.json"


def _load(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _source_key(name: str) -> str:
    """Normalize collector names for cross-file joins.

    collection_quality uses full source strings like 'gdelt_gkg/iheart.com'
    while publish_lag uses short collector keys like 'gdelt_gkg'. We normalise
    to the first path segment so the join works.
    """
    return name.split("/")[0].strip().lower()


def compute() -> list[dict]:
    quality_data = _load(LOGS / "collection_quality.json") or {}
    uptime_data = _load(LOGS / "collector_uptime.json") or {}
    backlog_data = _load(LOGS / "scoring_backlog.json") or {}
    lag_data = _load(LOGS / "publish_lag.json") or {}

    # --- quality: avg_ml and pct_urgent per source ---
    quality: dict[str, dict] = {}
    for row in quality_data.get("ranked", []):
        src = row.get("source", "")
        if not src or "backtest_run" in src:
            continue
        key = _source_key(src)
        if key not in quality:
            quality[key] = {"avg_ml": 0.0, "pct_urgent": 0.0, "n": 0, "display": src}
        # merge by volume-weighted average when the same base collector has many subfeeds
        prev = quality[key]
        n_total = prev["n"] + row.get("n", 0)
        if n_total > 0:
            prev["avg_ml"] = (prev["avg_ml"] * prev["n"] + row.get("avg_ml", 0) * row.get("n", 0)) / n_total
            prev["pct_urgent"] = (prev["pct_urgent"] * prev["n"] + row.get("pct_urgent", 0) * row.get("n", 0)) / n_total
        prev["n"] = n_total

    # --- uptime: max_gap_min per source (gaps list only has offenders) ---
    gap_max: dict[str, float] = {}
    for row in uptime_data.get("top_gap_sources", []):
        src = row.get("source", "")
        key = _source_key(src)
        existing = gap_max.get(key, 0.0)
        gap_max[key] = max(existing, row.get("max_gap_min", 0.0))

    # --- coverage: unscored_ratio per source ---
    coverage: dict[str, float] = {}
    for row in backlog_data.get("sources", []):
        src = row.get("source", "")
        if not src or "backtest_run" in src:
            continue
        key = _source_key(src)
        # take worst ratio for this collector base
        existing = coverage.get(key, 0.0)
        coverage[key] = max(existing, row.get("unscored_ratio", 0.0))

    # --- freshness: fresh_5m_pct per collector ---
    fresh: dict[str, float] = {}
    for coll, stats in lag_data.get("collectors", {}).items():
        key = _source_key(coll)
        fresh[key] = stats.get("fresh_5m_pct", 50.0)

    # --- build composite score for every source seen in quality ---
    results: list[dict] = []
    for key, q in quality.items():
        avg_ml = min(q["avg_ml"], 10.0)
        pct_urg = min(q["pct_urgent"], 1.0)
        max_gap = gap_max.get(key, 0.0)
        unscored = coverage.get(key, 0.0)
        fresh_pct = fresh.get(key, 50.0)  # default 50 when no lag data

        s_quality = avg_ml / 10.0 * 30.0
        s_urgency = pct_urg * 15.0
        gap_penalty = min(max_gap / 120.0, 1.0) * 25.0  # 2h gap = full penalty
        s_uptime = 25.0 - gap_penalty
        s_coverage = (1.0 - unscored) * 20.0
        s_freshness = fresh_pct / 100.0 * 10.0

        score = s_quality + s_urgency + s_uptime + s_coverage + s_freshness

        results.append({
            "collector": key,
            "display_source": q["display"],
            "score": round(score, 1),
            "breakdown": {
                "quality": round(s_quality, 1),
                "urgency": round(s_urgency, 1),
                "uptime": round(s_uptime, 1),
                "coverage": round(s_coverage, 1),
                "freshness": round(s_freshness, 1),
            },
            "raw": {
                "avg_ml": round(avg_ml, 3),
                "pct_urgent": round(pct_urg, 4),
                "max_gap_min": round(max_gap, 1),
                "unscored_ratio": round(unscored, 3),
                "fresh_5m_pct": round(fresh_pct, 1),
                "article_count": q["n"],
            },
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def main() -> int:
    results = compute()
    if not results:
        print("collector_health_score: no data", file=sys.stderr)
        return 1

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collectors_scored": len(results),
        "top": results[:10],
        "bottom": results[-5:],
        "all": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"collector_health_score: scored {len(results)} collectors")
    print(f"  TOP collectors:")
    for r in results[:5]:
        b = r["breakdown"]
        print(
            f"  [{r['score']:5.1f}] {r['collector']:<25}"
            f"  qual={b['quality']:.0f} urg={b['urgency']:.0f}"
            f" up={b['uptime']:.0f} cov={b['coverage']:.0f} fresh={b['freshness']:.0f}"
        )
    print(f"  BOTTOM collectors:")
    for r in results[-3:]:
        b = r["breakdown"]
        print(
            f"  [{r['score']:5.1f}] {r['collector']:<25}"
            f"  qual={b['quality']:.0f} urg={b['urgency']:.0f}"
            f" up={b['uptime']:.0f} cov={b['coverage']:.0f} fresh={b['freshness']:.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
