"""Weekend sector strength scanner — Monday gap-open readiness by sector.

Scans the last 48 h of ML-scored articles, applies exponential recency decay
(half-life 8 h so Friday-close news still matters but Saturday noise decays),
groups articles by sector via ticker mentions, and ranks sectors by a
"weekend strength score" = sum of decay-weighted ml_scores.

Distinct from:
  * sector_pulse       — 1h window, absolute density snapshot
  * sector_rotation    — 2h delta, intraday rotation
  * weekend_catalyst_brief — article-level, no sector grouping

Output: /home/zeph/logs/weekend_sector_strength.json
Standalone: python3 -m analytics.weekend_sector_strength
"""
from __future__ import annotations

import json
import math
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

OUT = Path("/home/zeph/logs/weekend_sector_strength.json")
LOOKBACK_HOURS = 48
HALF_LIFE_HOURS = 8.0  # longer half-life: Friday news still counts on Sunday
FETCH_LIMIT = 10000
MIN_ARTICLES = 3  # minimum sector articles before reporting


def _decay(age_hours: float) -> float:
    return math.exp(-age_hours / HALF_LIFE_HOURS)


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")

    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=LOOKBACK_HOURS)).isoformat()

    cur = conn.execute(
        f"""
        SELECT first_seen, title, source, ml_score, ai_score, urgency
          FROM articles INDEXED BY idx_first_seen
         WHERE first_seen >= ?
           AND ml_score IS NOT NULL
           AND {_LIVE_ONLY_CLAUSE}
         ORDER BY first_seen DESC
         LIMIT ?
        """,
        (cutoff, FETCH_LIMIT),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("weekend_sector_strength: no ml-scored rows in 48h window", file=sys.stderr)
        return 1

    # Per-sector accumulators
    sector_score: dict[str, float] = defaultdict(float)
    sector_count: dict[str, int] = defaultdict(int)
    sector_ml_raw: dict[str, list[float]] = defaultdict(list)
    sector_urgent: dict[str, int] = defaultdict(int)
    sector_top: dict[str, list[dict]] = defaultdict(list)

    for fs, title, source, ml, ai, urg in rows:
        ts = _parse_ts(fs)
        if ts is None:
            continue
        age_h = max(0.0, (now - ts).total_seconds() / 3600.0)
        d = _decay(age_h)
        weighted = float(ml) * d

        tickers = extract_tickers(title or "")
        sectors = {_TICKER_TO_SECTOR[tk] for tk in tickers if tk in _TICKER_TO_SECTOR}
        if not sectors:
            continue

        for sector in sectors:
            sector_score[sector] += weighted
            sector_count[sector] += 1
            sector_ml_raw[sector].append(float(ml))
            if urg and urg >= 2:
                sector_urgent[sector] += 1
            if len(sector_top[sector]) < 3:
                sector_top[sector].append({
                    "title": (title or "")[:100],
                    "source": source,
                    "ml_score": round(float(ml), 3),
                    "age_h": round(age_h, 1),
                    "effective": round(weighted, 3),
                })

    ranked = []
    for sector in sorted(sector_score, key=sector_score.__getitem__, reverse=True):
        n = sector_count[sector]
        if n < MIN_ARTICLES:
            continue
        avg_ml = mean(sector_ml_raw[sector])
        ranked.append({
            "sector": sector,
            "weekend_strength": round(sector_score[sector], 3),
            "article_count": n,
            "avg_ml_score": round(avg_ml, 3),
            "urgent_count": sector_urgent[sector],
            "top_articles": sector_top[sector],
        })

    payload = {
        "generated_at": now.isoformat(timespec="seconds"),
        "lookback_hours": LOOKBACK_HOURS,
        "half_life_hours": HALF_LIFE_HOURS,
        "scanned": len(rows),
        "sectors": ranked,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"weekend_sector_strength: scanned={len(rows)} sectors={len(ranked)}")
    for r in ranked[:6]:
        print(
            f"  {r['sector']:<14} strength={r['weekend_strength']:>8.2f} "
            f"n={r['article_count']:>4}  avg_ml={r['avg_ml_score']:.3f}  "
            f"urgent={r['urgent_count']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
