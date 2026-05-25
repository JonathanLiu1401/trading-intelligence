"""Collector health report: RED/AMBER/GREEN rollup from source_health.db.

Reads ``source_health.db`` (maintained by ``collectors/source_health.py``)
and produces a structured health summary for all 96+ collectors:

- Active sources: GREEN (0 consecutive failures), AMBER (1-2), RED (3+)
- Disabled sources: grouped by failure severity to surface which need fixes
- Success rate: fetch_successes / fetch_attempts
- Overall fleet health score (0-100)

Why this matters: 83/96 collectors are currently disabled with varying
failure severities. This report surfaces the worst offenders and tracks
recovery when fixes land.

Output: /home/zeph/logs/source_health_report.json
Standalone: python3 -m analytics.source_health_report
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

HEALTH_DB = BASE / "data" / "source_health.db"
OUT_PATH = Path("/home/zeph/logs/source_health_report.json")


def _status(consecutive_failures: int, disabled: bool) -> str:
    if disabled:
        return "DISABLED"
    if consecutive_failures == 0:
        return "GREEN"
    if consecutive_failures <= 2:
        return "AMBER"
    return "RED"


def _success_pct(fetch_successes: int, fetch_attempts: int) -> float:
    if not fetch_attempts:
        return 0.0
    return round(100.0 * fetch_successes / fetch_attempts, 1)


def main() -> dict:
    now = datetime.now(timezone.utc)

    if not HEALTH_DB.exists():
        result = {
            "generated_at": now.isoformat(),
            "error": f"source_health.db not found at {HEALTH_DB}",
        }
        OUT_PATH.write_text(json.dumps(result, indent=2))
        return result

    conn = sqlite3.connect(str(HEALTH_DB), timeout=15)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT source, consecutive_failures, total_articles, disabled,
               fetch_attempts, fetch_successes, last_seen, last_success
        FROM source_health
        ORDER BY disabled ASC, consecutive_failures DESC
    """).fetchall()
    conn.close()

    active: list[dict] = []
    disabled_sources: list[dict] = []
    green = amber = red = 0

    for r in rows:
        cf = r["consecutive_failures"]
        dis = bool(r["disabled"])
        status = _status(cf, dis)
        entry = {
            "source": r["source"],
            "status": status,
            "consecutive_failures": cf,
            "total_articles": r["total_articles"],
            "fetch_attempts": r["fetch_attempts"],
            "fetch_successes": r["fetch_successes"],
            "success_pct": _success_pct(r["fetch_successes"], r["fetch_attempts"]),
            "last_seen": r["last_seen"],
        }
        if dis:
            disabled_sources.append(entry)
        else:
            if status == "GREEN":
                green += 1
            elif status == "AMBER":
                amber += 1
            else:
                red += 1
            active.append(entry)

    total = len(rows)
    active_count = len(active)
    disabled_count = len(disabled_sources)

    # Fleet health score: weight active/total + green-rate among active
    # 100 = all active and all green; 0 = all disabled
    active_ratio = active_count / total if total else 0
    green_ratio = green / active_count if active_count else 0
    fleet_score = round((active_ratio * 0.5 + green_ratio * 0.5) * 100)

    # Top disabled by failure severity (those with some historical data)
    top_disabled = sorted(
        [s for s in disabled_sources if s["total_articles"] > 0],
        key=lambda s: s["consecutive_failures"],
        reverse=True,
    )[:10]

    result = {
        "generated_at": now.isoformat(),
        "total_sources": total,
        "active": active_count,
        "disabled": disabled_count,
        "active_green": green,
        "active_amber": amber,
        "active_red": red,
        "fleet_health_score": fleet_score,
        "active_sources": active,
        "top_disabled_by_failures": top_disabled,
    }

    OUT_PATH.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = main()
    print(f"Fleet health: {r['fleet_health_score']}/100 | "
          f"Active: {r['active']}/{r['total_sources']} | "
          f"GREEN={r['active_green']} AMBER={r['active_amber']} RED={r['active_red']}")
    print(f"Top failing disabled sources:")
    for s in r["top_disabled_by_failures"][:5]:
        print(f"  {s['source']:30s} failures={s['consecutive_failures']:4d}  "
              f"articles={s['total_articles']:6d}  success={s['success_pct']}%")
