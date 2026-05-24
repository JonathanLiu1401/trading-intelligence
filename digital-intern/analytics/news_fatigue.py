"""News fatigue detector: high-volume tickers where average score is declining.

A ticker is "fatigued" when it has accumulated many articles over the last
24 hours but its average ai_score in the most-recent 6-hour window is
meaningfully lower than the prior 18-hour window.  This pattern signals a
story being "burned out" — heavy coverage that the market has already
digested, so incremental articles are generating less urgency than the
initial wave.

Operator use: avoid chasing fatigued tickers as if they are breaking news.
The story is old; wait for a new catalyst before treating fresh articles
on that ticker as high-priority.

Thresholds:
  MIN_TOTAL_24H   = 15   articles in last 24h to qualify
  MIN_RECENT_6H   = 3    articles in last 6h (ticker must still be active)
  FATIGUE_DROP    = 1.5  ai_score points: recent_mean < prior_mean - 1.5
  SCAN_LIMIT      = 12000 rows (covers ~24h at typical ingest rate)

Output: /home/zeph/logs/news_fatigue.json
Standalone: ``python3 -m analytics.news_fatigue``
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

from analytics.trend_velocity import TICKER_RE, STOP, _parse_ts
from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT_PATH = Path("/home/zeph/logs/news_fatigue.json")

SCAN_LIMIT = 12_000
TOTAL_WINDOW_HOURS = 24
RECENT_HOURS = 6       # "is the story still fresh?" window
PRIOR_HOURS = 18       # baseline window (hours 6-24 ago)
MIN_TOTAL_24H = 15     # minimum total mentions to qualify
MIN_RECENT_6H = 3      # must still have recent coverage (not dead)
FATIGUE_DROP = 1.5     # ai_score drop to call fatigue
TOP_N = 10


def _extract_tickers(title: str) -> list[str]:
    out: list[str] = []
    for m in TICKER_RE.findall(title or ""):
        if m not in STOP and len(m) >= 2:
            out.append(m)
    return out


def main() -> int:
    db_path = _get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=5000")

    rows = conn.execute(
        "SELECT first_seen, title, ai_score FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "AND ai_score IS NOT NULL "
        "ORDER BY first_seen DESC LIMIT ?",
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    if not rows:
        print("news_fatigue: no rows", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    recent_cut = now - timedelta(hours=RECENT_HOURS)
    prior_cut = now - timedelta(hours=TOTAL_WINDOW_HOURS)

    # per-ticker buckets: list of ai_score values in each window
    recent_scores: dict[str, list[float]] = defaultdict(list)
    prior_scores: dict[str, list[float]] = defaultdict(list)

    for fs, title, ai_score in rows:
        ts = _parse_ts(fs)
        if ts is None:
            continue
        if ts < prior_cut:
            break  # rows are DESC; everything older than 24h is irrelevant
        tickers = _extract_tickers(title)
        if not tickers:
            continue
        score = float(ai_score)
        for tk in tickers:
            if ts >= recent_cut:
                recent_scores[tk].append(score)
            else:
                prior_scores[tk].append(score)

    fatigued: list[dict] = []
    all_tickers = set(recent_scores) | set(prior_scores)

    for tk in all_tickers:
        recent = recent_scores.get(tk, [])
        prior = prior_scores.get(tk, [])
        total = len(recent) + len(prior)

        if total < MIN_TOTAL_24H:
            continue
        if len(recent) < MIN_RECENT_6H:
            continue
        if not prior:
            continue

        recent_mean = mean(recent)
        prior_mean = mean(prior)
        drop = prior_mean - recent_mean

        if drop >= FATIGUE_DROP:
            fatigued.append({
                "ticker": tk,
                "total_24h": total,
                "recent_6h_count": len(recent),
                "prior_18h_count": len(prior),
                "recent_mean_score": round(recent_mean, 2),
                "prior_mean_score": round(prior_mean, 2),
                "score_drop": round(drop, 2),
            })

    fatigued.sort(key=lambda r: r["score_drop"], reverse=True)
    top = fatigued[:TOP_N]

    payload = {
        "generated_at": now.isoformat(),
        "scanned_rows": len(rows),
        "fatigue_threshold_drop": FATIGUE_DROP,
        "min_total_24h": MIN_TOTAL_24H,
        "min_recent_6h": MIN_RECENT_6H,
        "fatigued_count": len(fatigued),
        "tickers": top,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    print(f"news_fatigue: scanned={len(rows)} fatigued={len(fatigued)}")
    for r in top:
        print(
            f"  {r['ticker']}: total={r['total_24h']} "
            f"recent_avg={r['recent_mean_score']:.1f} "
            f"prior_avg={r['prior_mean_score']:.1f} "
            f"drop={r['score_drop']:.1f}"
        )
    if not top:
        print("  (no fatigued tickers in window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
