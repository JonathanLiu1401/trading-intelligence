"""Sector pulse: news density + signal quality per sector, taxonomy-driven.

Complements ``ticker_comentions`` (pair-discovery axis) and
``sector_rotation`` (2h window deltas) by giving an absolute density +
quality snapshot per sector for the last hour vs prior hour.

Metrics per sector:
  * article_count_now / article_count_prev — raw volume
  * avg_ml_score, avg_ai_score — signal quality
  * urgency_rate — fraction of articles with urgency >= 2
  * pulse_score — composite: count_now * avg_score * (1 + urgency_rate)

Output: /home/zeph/logs/sector_pulse.json
Standalone: python3 -m analytics.sector_pulse
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

try:
    from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path
    DB_PATH = _get_db_path()
except Exception:
    _LIVE_ONLY_CLAUSE = "source NOT LIKE 'backtest_run_%'"
    DB_PATH = BASE / "data" / "articles.db"

from analytics.sector_rotation import SECTOR_MAP, _TICKER_TO_SECTOR
from analytics.trend_velocity import _parse_ts, extract_tickers

OUT = Path("/home/zeph/logs/sector_pulse.json")
WINDOW_HOURS = 1
FETCH_LIMIT = 2000
TOP_N = 8


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")

    cur = conn.execute(
        "SELECT first_seen, title, ml_score, ai_score, urgency "
        f"FROM articles WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (FETCH_LIMIT,),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("sector_pulse: no rows", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    cut_now = now - timedelta(hours=WINDOW_HOURS)
    cut_prev = now - timedelta(hours=WINDOW_HOURS * 2)

    # Buckets: sector -> list of (ml_score, ai_score, urgency) for now/prev windows
    now_bucket: dict[str, list[tuple]] = defaultdict(list)
    prev_bucket: dict[str, list[tuple]] = defaultdict(list)

    for fs, title, ml, ai, urg in rows:
        ts = _parse_ts(fs)
        if ts is None:
            continue
        tickers = extract_tickers(title)
        sectors = {_TICKER_TO_SECTOR[tk] for tk in tickers if tk in _TICKER_TO_SECTOR}
        if not sectors:
            continue
        entry = (ml or 0.0, ai or 0.0, int(urg or 0))
        if ts >= cut_now:
            for s in sectors:
                now_bucket[s].append(entry)
        elif ts >= cut_prev:
            for s in sectors:
                prev_bucket[s].append(entry)

    all_sectors = set(now_bucket) | set(prev_bucket)
    if not all_sectors:
        print("sector_pulse: no sector-tagged articles found", file=sys.stderr)
        OUT.write_text(json.dumps({"generated_at": now.isoformat(), "sectors": [], "window_hours": WINDOW_HOURS}))
        return 0

    results = []
    for sector in all_sectors:
        now_arts = now_bucket.get(sector, [])
        prev_arts = prev_bucket.get(sector, [])
        count_now = len(now_arts)
        count_prev = len(prev_arts)

        if now_arts:
            avg_ml = round(mean(e[0] for e in now_arts), 3)
            avg_ai = round(mean(e[1] for e in now_arts), 3)
            urg_rate = round(sum(1 for e in now_arts if e[2] >= 2) / count_now, 3)
        else:
            avg_ml = avg_ai = urg_rate = 0.0

        avg_score = avg_ml if avg_ml > 0 else avg_ai
        pulse = round(count_now * avg_score * (1.0 + urg_rate), 3)
        delta = count_now - count_prev

        results.append({
            "sector": sector,
            "count_now": count_now,
            "count_prev": count_prev,
            "delta": delta,
            "avg_ml_score": avg_ml,
            "avg_ai_score": avg_ai,
            "urgency_rate": urg_rate,
            "pulse_score": pulse,
        })

    results.sort(key=lambda r: r["pulse_score"], reverse=True)

    payload = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "scanned": len(rows),
        "sectors": results[:TOP_N],
        "all_sector_count": len(results),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"sector_pulse: scanned={len(rows)} sectors={len(results)} window={WINDOW_HOURS}h")
    for r in results[:5]:
        sign = "+" if r["delta"] >= 0 else ""
        print(
            f"  {r['sector']:<14} pulse={r['pulse_score']:.2f}  "
            f"n={r['count_now']} ({sign}{r['delta']})  "
            f"ml={r['avg_ml_score']:.2f}  urg={r['urgency_rate']:.0%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
