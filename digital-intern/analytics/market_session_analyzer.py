"""Market session analyzer: urgency/score distribution by US trading session.

Classifies last 24h of articles by US Eastern market session:
  * pre_market   04:00–09:30 ET
  * regular      09:30–16:00 ET
  * after_hours  16:00–20:00 ET
  * overnight    20:00–04:00 ET (next day)

Reports article count, avg ml_score, avg ai_score, and urgency rate per session.
EDT offset (UTC-4) used — correct for May–Nov; Nov–Mar would need UTC-5.

Output: /home/zeph/logs/market_session_analysis.json
Standalone: python3 -m analytics.market_session_analyzer
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

OUT = Path("/home/zeph/logs/market_session_analysis.json")
SCAN_LIMIT = 25_000
ET_OFFSET = timedelta(hours=-4)  # EDT (UTC-4), valid May–Nov


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _session(et_hour: float) -> str:
    """Map ET hour (0-24) to session label."""
    if 4.0 <= et_hour < 9.5:
        return "pre_market"
    if 9.5 <= et_hour < 16.0:
        return "regular"
    if 16.0 <= et_hour < 20.0:
        return "after_hours"
    return "overnight"


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=15)
    conn.execute("PRAGMA query_only=ON")

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    rows = conn.execute(
        "SELECT first_seen, ml_score, ai_score, urgency "
        "FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "AND first_seen >= ? "
        "ORDER BY first_seen DESC LIMIT ?",
        (cutoff, SCAN_LIMIT),
    ).fetchall()
    conn.close()

    if not rows:
        print("market_session_analyzer: no rows in last 24h", file=sys.stderr)
        return 1

    buckets: dict[str, dict] = {
        s: {"count": 0, "ml_scores": [], "ai_scores": [], "urgent": 0}
        for s in ("pre_market", "regular", "after_hours", "overnight")
    }

    for fs, ml, ai, urg in rows:
        ts = _parse_ts(fs)
        if ts is None:
            continue
        et = ts + ET_OFFSET
        et_hour = et.hour + et.minute / 60.0
        sess = _session(et_hour)
        b = buckets[sess]
        b["count"] += 1
        if ml is not None:
            b["ml_scores"].append(float(ml))
        if ai is not None:
            b["ai_scores"].append(float(ai))
        if urg is not None and urg >= 2:
            b["urgent"] += 1

    now_utc = datetime.now(timezone.utc)
    summary = {
        "generated_at": now_utc.isoformat(),
        "window_hours": 24,
        "total_articles": len(rows),
        "sessions": {},
    }
    SESSION_ORDER = ["pre_market", "regular", "after_hours", "overnight"]
    for sess in SESSION_ORDER:
        b = buckets[sess]
        n = b["count"]
        summary["sessions"][sess] = {
            "count": n,
            "avg_ml_score": round(mean(b["ml_scores"]), 4) if b["ml_scores"] else None,
            "avg_ai_score": round(mean(b["ai_scores"]), 4) if b["ai_scores"] else None,
            "urgent_count": b["urgent"],
            "urgency_rate": round(b["urgent"] / n, 4) if n else 0.0,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))

    print(f"market_session_analyzer: {len(rows)} articles in last 24h")
    for sess in SESSION_ORDER:
        s = summary["sessions"][sess]
        ml = f"{s['avg_ml_score']:.3f}" if s["avg_ml_score"] is not None else "n/a"
        ai = f"{s['avg_ai_score']:.3f}" if s["avg_ai_score"] is not None else "n/a"
        print(f"  {sess:15s}: n={s['count']:5d}  ml={ml}  ai={ai}  urgent={s['urgent_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
