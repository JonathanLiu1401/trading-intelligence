"""Ticker cold-case detector.

Identifies tickers that had high mention volume (>=MIN_HOT_COUNT articles)
in the WARM window (2-6h ago) but complete silence in the HOT window (last 1h).

This "cooling story" signal is complementary to:
- trend_velocity (2h vs prior 2h, doesn't catch *zero* in recent window)
- breaking_news_detector (detects spikes, not cooling)
- stale_source_alerter (source-level, not ticker-level)

Use case: a ticker that was heavily discussed 3-5h ago but has gone silent
may indicate a story resolved, news embargo, or pre-announcement quiet period.

Output: /home/zeph/logs/ticker_cold_case.json
Standalone: python3 -m analytics.ticker_cold_case_detector
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from analytics.trend_velocity import STOP, TICKER_RE, _parse_ts, extract_tickers  # noqa: E402
from storage.article_store import _LIVE_ONLY_CLAUSE  # noqa: E402

DB_PATH = BASE / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/ticker_cold_case.json")

# Tuning
HOT_WINDOW_H = 1        # recent silence window (last N hours)
WARM_WINDOW_H = 6       # look-back window end (up to N hours ago)
WARM_START_H = 2        # look-back window start (skip most recent N hours)
MIN_HOT_COUNT = 5       # min articles in warm window to qualify
SCAN_LIMIT = 10_000     # bounded scan to avoid USB DB timeout
TOP_N = 10              # max cold cases to report


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def main() -> dict:
    now = _now_utc()

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=15)
    try:
        # Bounded scan of recent rows covering the full warm window
        rows = conn.execute(
            f"""
            SELECT title, first_seen
            FROM   articles
            WHERE  {_LIVE_ONLY_CLAUSE}
            ORDER  BY first_seen DESC
            LIMIT  {SCAN_LIMIT}
            """,
        ).fetchall()
    finally:
        conn.close()

    # Bucket articles into hot (0→1h) and warm (2→6h)
    hot_counts: dict[str, int] = defaultdict(int)
    warm_counts: dict[str, int] = defaultdict(int)
    warm_articles: dict[str, list[str]] = defaultdict(list)

    for title, first_seen_raw in rows:
        if not title or not first_seen_raw:
            continue
        ts = _parse_ts(first_seen_raw)
        if ts is None:
            continue
        age_h = (now - ts).total_seconds() / 3600.0
        if age_h < 0:
            continue  # future-dated row (clock skew)

        tickers = extract_tickers(title)
        if not tickers:
            continue

        if age_h <= HOT_WINDOW_H:
            for t in tickers:
                hot_counts[t] += 1
        elif WARM_START_H <= age_h <= WARM_WINDOW_H:
            for t in tickers:
                warm_counts[t] += 1
                if len(warm_articles[t]) < 3:  # keep up to 3 example headlines
                    warm_articles[t].append(title[:120])

    # Find cold cases: hot in warm window, silent in hot window
    cold_cases = []
    for ticker, warm_n in warm_counts.items():
        if warm_n < MIN_HOT_COUNT:
            continue
        hot_n = hot_counts.get(ticker, 0)
        if hot_n > 0:
            continue  # still active
        # Compute rate: mentions/hour in warm window
        warm_span_h = WARM_WINDOW_H - WARM_START_H  # 4h
        rate = round(warm_n / warm_span_h, 2)
        cold_cases.append(
            {
                "ticker": ticker,
                "warm_count": warm_n,
                "hot_count": hot_n,
                "warm_rate_per_hour": rate,
                "example_headlines": warm_articles[ticker],
                "silence_since_h": HOT_WINDOW_H,
            }
        )

    # Sort by warm_count descending (hottest stories that went cold first)
    cold_cases.sort(key=lambda x: x["warm_count"], reverse=True)
    cold_cases = cold_cases[:TOP_N]

    result = {
        "generated_at": now.isoformat(),
        "scan_limit": SCAN_LIMIT,
        "hot_window_h": HOT_WINDOW_H,
        "warm_window_h": f"{WARM_START_H}-{WARM_WINDOW_H}h",
        "min_hot_count": MIN_HOT_COUNT,
        "cold_cases_found": len(cold_cases),
        "cold_cases": cold_cases,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    # Print summary for verification
    print(f"Cold-case scan: {len(rows)} rows scanned, {len(cold_cases)} cold cases found")
    if cold_cases:
        for cc in cold_cases[:5]:
            print(f"  {cc['ticker']}: {cc['warm_count']} mentions 2-6h ago → 0 in last 1h "
                  f"({cc['warm_rate_per_hour']}/hr)")

    return result


if __name__ == "__main__":
    main()
