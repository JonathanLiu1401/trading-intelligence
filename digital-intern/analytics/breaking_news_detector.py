"""Breaking news detector: 3+ articles on same ticker within a 5-min window.

Scans the most recent articles, extracts tickers from titles, and flags any
ticker that received 3 or more distinct articles inside a rolling 5-minute
window in the lookback period. Writes events to
/home/zeph/logs/breaking_news.jsonl (append-only) and prints the events for
the current run.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analytics.trend_velocity import TICKER_RE, STOP, _parse_ts, extract_tickers

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/breaking_news.jsonl")
LOOKBACK_HOURS = 2
WINDOW_MINUTES = 5
THRESHOLD = 3
FETCH_LIMIT = 4000


def detect(rows: list[tuple[str, str, str]]) -> list[dict]:
    by_ticker: dict[str, list[tuple[datetime, str, str]]] = defaultdict(list)
    for first_seen, title, source in rows:
        ts = _parse_ts(first_seen)
        if ts is None:
            continue
        for tk in set(extract_tickers(title)):
            by_ticker[tk].append((ts, title, source or ""))

    window = timedelta(minutes=WINDOW_MINUTES)
    events: list[dict] = []
    for tk, items in by_ticker.items():
        items.sort(key=lambda x: x[0])
        n = len(items)
        for i in range(n):
            j = i
            while j + 1 < n and items[j + 1][0] - items[i][0] <= window:
                j += 1
            count = j - i + 1
            if count >= THRESHOLD:
                sources = sorted({items[k][2] for k in range(i, j + 1)})
                if len(sources) < 2:
                    continue  # single-source burst, not breaking
                events.append({
                    "ticker": tk,
                    "count": count,
                    "window_start": items[i][0].isoformat(),
                    "window_end": items[j][0].isoformat(),
                    "sources": sources,
                    "sample_title": items[i][1][:160],
                })
                break  # one event per ticker per run
    events.sort(key=lambda e: (-e["count"], e["ticker"]))
    return events


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    since = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    cur = conn.execute(
        "SELECT first_seen, title, source FROM articles INDEXED BY idx_first_seen "
        "WHERE first_seen >= ? AND source NOT LIKE 'backtest_run_%' "
        "ORDER BY first_seen DESC LIMIT ?",
        (since, FETCH_LIMIT),
    )
    rows = cur.fetchall()
    events = detect(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with OUT_PATH.open("a") as fh:
        for ev in events:
            fh.write(json.dumps({"run_at": stamp, **ev}) + "\n")

    print(f"breaking_news: scanned={len(rows)} events={len(events)} window={WINDOW_MINUTES}m threshold={THRESHOLD}")
    for ev in events[:10]:
        print(f"  BREAKING {ev['ticker']}: {ev['count']} articles in "
              f"{ev['window_start']}..{ev['window_end']} | sources={','.join(ev['sources'][:4])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
