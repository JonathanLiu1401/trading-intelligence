"""Breaking news detector: 3+ articles on same ticker within a 5-min window.

Scans the most recent articles, extracts tickers from titles, and flags any
ticker that received 3 or more distinct articles inside a rolling 5-minute
window in the lookback period. Writes events to
/home/zeph/logs/breaking_news.jsonl (append-only), marks matching rows via
articles.breaking_news=1, and prints the events for the current run.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analytics.trend_velocity import TICKER_RE, STOP, _parse_ts, extract_tickers
from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/breaking_news.jsonl")
LOOKBACK_HOURS = 2
WINDOW_MINUTES = 5
THRESHOLD = 3
FETCH_LIMIT = 4000


def _unpack_row(row: tuple) -> tuple[str | None, str, str, str]:
    """Accept legacy 3-tuples plus the new id-bearing 4-tuples."""
    if len(row) == 4:
        aid, first_seen, title, source = row
        return aid, first_seen, title, source
    first_seen, title, source = row
    return None, first_seen, title, source


def detect(rows: list[tuple]) -> list[dict]:
    by_ticker: dict[str, list[tuple[datetime, str, str, str | None]]] = defaultdict(list)
    for row in rows:
        aid, first_seen, title, source = _unpack_row(row)
        ts = _parse_ts(first_seen)
        if ts is None:
            continue
        for tk in set(extract_tickers(title)):
            by_ticker[tk].append((ts, title, source or "", aid))

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
                    "article_ids": [
                        items[k][3] for k in range(i, j + 1) if items[k][3]
                    ],
                })
                break  # one event per ticker per run
    events.sort(key=lambda e: (-e["count"], e["ticker"]))
    return events


def _ensure_breaking_news_column(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(articles)").fetchall()}
    if "breaking_news" in cols:
        return
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN breaking_news INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise


def mark_breaking_articles(conn: sqlite3.Connection, events: list[dict]) -> int:
    ids = sorted({aid for ev in events for aid in ev.get("article_ids", []) if aid})
    if not ids:
        return 0
    _ensure_breaking_news_column(conn)
    placeholders = ",".join("?" for _ in ids)
    conn.execute(
        f"UPDATE articles SET breaking_news=1 WHERE id IN ({placeholders})",
        ids,
    )
    conn.commit()
    return int(conn.execute("SELECT changes()").fetchone()[0])


def main() -> int:
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.execute("PRAGMA busy_timeout=20000")
    since = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    # Canonical `_LIVE_ONLY_CLAUSE` — a partial `source NOT LIKE 'backtest_run_%'`
    # lets backtest:// URLs and opus_annotation* rows contribute fake "breaking"
    # bursts (multiple synthetic rows on one ticker in a small replay window).
    cur = conn.execute(
        "SELECT id, first_seen, title, source FROM articles INDEXED BY idx_first_seen "
        f"WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (since, FETCH_LIMIT),
    )
    rows = cur.fetchall()
    events = detect(rows)
    marked = mark_breaking_articles(conn, events)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with OUT_PATH.open("a") as fh:
        for ev in events:
            fh.write(json.dumps({"run_at": stamp, **ev}) + "\n")

    print(
        f"breaking_news: scanned={len(rows)} events={len(events)} "
        f"marked={marked} window={WINDOW_MINUTES}m threshold={THRESHOLD}"
    )
    for ev in events[:10]:
        print(f"  BREAKING {ev['ticker']}: {ev['count']} articles in "
              f"{ev['window_start']}..{ev['window_end']} | sources={','.join(ev['sources'][:4])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
