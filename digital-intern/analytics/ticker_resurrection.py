"""Ticker resurrection detector: tickers silent for 24h+ that reappear in last 2h.

A ticker is "resurrected" when:
  - It has >= MIN_RECENT mentions in the last 2 hours, AND
  - It had zero mentions in the 22 hours before that window (hours 2-24 ago).

This pattern flags re-emerging stories — often when new information surfaces on
a previously dormant ticker, which can signal a material event.

Output: /home/zeph/logs/ticker_resurrection.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analytics.trend_velocity import TICKER_RE, STOP, _parse_ts
from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/ticker_resurrection.json")

RECENT_HOURS = 2       # "active again" window
SILENCE_HOURS = 24     # silence window before that
FETCH_LIMIT = 20_000   # rows to scan (covers ~24h at typical ingest rate)
MIN_RECENT = 2         # minimum mentions in recent window to count


def extract_tickers(title: str) -> list[str]:
    out = []
    for m in TICKER_RE.findall(title or ""):
        if m not in STOP and len(m) >= 2:
            out.append(m)
    return out


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")

    rows = conn.execute(
        "SELECT first_seen, title FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (FETCH_LIMIT,),
    ).fetchall()
    conn.close()

    if not rows:
        print("ticker_resurrection: no rows", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(hours=RECENT_HOURS)
    silence_start = now - timedelta(hours=SILENCE_HOURS)
    # silence window: silence_start <= t < recent_cutoff

    recent_counts: dict[str, int] = defaultdict(int)
    silence_counts: dict[str, int] = defaultdict(int)

    for raw_ts, title in rows:
        ts = _parse_ts(raw_ts)
        if ts is None:
            continue

        tickers = extract_tickers(title)
        if not tickers:
            continue

        if ts >= recent_cutoff:
            for t in tickers:
                recent_counts[t] += 1
        elif ts >= silence_start:
            for t in tickers:
                silence_counts[t] += 1
        # older than silence window: skip

    resurrected = []
    for ticker, cnt in recent_counts.items():
        if cnt >= MIN_RECENT and silence_counts.get(ticker, 0) == 0:
            resurrected.append({"ticker": ticker, "recent_mentions": cnt})

    resurrected.sort(key=lambda x: x["recent_mentions"], reverse=True)

    output = {
        "generated_at": now.isoformat(),
        "recent_window_hours": RECENT_HOURS,
        "silence_window_hours": SILENCE_HOURS,
        "min_recent_mentions": MIN_RECENT,
        "total_resurrected": len(resurrected),
        "tickers": resurrected,
    }

    OUT_PATH.write_text(json.dumps(output, indent=2))

    if resurrected:
        print(f"ticker_resurrection: {len(resurrected)} resurrected tickers")
        for item in resurrected[:5]:
            print(f"  ${item['ticker']} — {item['recent_mentions']} mentions in last {RECENT_HOURS}h (silent prior {SILENCE_HOURS}h)")
    else:
        print(f"ticker_resurrection: no resurrected tickers in this window")

    return 0


if __name__ == "__main__":
    sys.exit(main())
