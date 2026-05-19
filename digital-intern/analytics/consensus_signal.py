"""Strong consensus signal: 5+ articles on same ticker in lookback window
with avg ai_score >= bullish threshold => strong_consensus event logged.

ai_score (0..10, Sonnet's combined relevance/urgency score) is the closest
proxy this codebase has to a directional importance signal. We treat a
sustained cluster of high-ai_score coverage as a "strong consensus" event:
many independent sources are simultaneously rating the same ticker as
high-importance. Distinct-source requirement prevents single-feed spam from
firing the signal.

Events append to /home/zeph/logs/strong_consensus.jsonl.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analytics.trend_velocity import _parse_ts, extract_tickers

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/strong_consensus.jsonl")
LOOKBACK_HOURS = 6
THRESHOLD = 5
HIGH_AI_CUTOFF = 6.0
MIN_DISTINCT_SOURCES = 3
FETCH_LIMIT = 6000


def detect(rows: list[tuple[str, str, str, float | None]]) -> list[dict]:
    # Only consider high-importance articles (ai_score >= cutoff) — the
    # baseline ai_score distribution is heavily zero-skewed, so an "avg
    # score" filter is the wrong shape. A consensus is N independent
    # sources all rating the same ticker as high-importance.
    by_ticker: dict[str, list[tuple[datetime, str, str, float]]] = defaultdict(list)
    for first_seen, title, source, ai_score in rows:
        if ai_score is None or ai_score < HIGH_AI_CUTOFF:
            continue
        ts = _parse_ts(first_seen)
        if ts is None:
            continue
        for tk in set(extract_tickers(title)):
            by_ticker[tk].append((ts, title, source or "", float(ai_score)))

    events: list[dict] = []
    for tk, items in by_ticker.items():
        if len(items) < THRESHOLD:
            continue
        sources = {it[2] for it in items if it[2]}
        if len(sources) < MIN_DISTINCT_SOURCES:
            continue
        avg_ai = sum(it[3] for it in items) / len(items)
        items.sort(key=lambda x: x[0])
        events.append({
            "ticker": tk,
            "count": len(items),
            "distinct_sources": len(sources),
            "avg_ai_score": round(avg_ai, 2),
            "window_start": items[0][0].isoformat(),
            "window_end": items[-1][0].isoformat(),
            "sample_title": items[-1][1][:160],
        })
    events.sort(key=lambda e: (-e["avg_ai_score"], -e["count"]))
    return events


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    since = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    cur = conn.execute(
        "SELECT first_seen, title, source, ai_score FROM articles INDEXED BY idx_first_seen "
        "WHERE first_seen >= ? AND source NOT LIKE 'backtest_run_%' "
        "ORDER BY first_seen DESC LIMIT ?",
        (since, FETCH_LIMIT),
    )
    rows = cur.fetchall()
    events = detect(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).isoformat()
    if events:
        with OUT_PATH.open("a") as f:
            for ev in events:
                f.write(json.dumps({"detected_at": run_ts, **ev}) + "\n")

    print(f"scanned {len(rows)} rows in last {LOOKBACK_HOURS}h; "
          f"strong_consensus events: {len(events)}")
    for ev in events[:10]:
        print(f"  {ev['ticker']:6s} n={ev['count']:3d} src={ev['distinct_sources']:2d} "
              f"avg_ai={ev['avg_ai_score']:.2f} :: {ev['sample_title']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
