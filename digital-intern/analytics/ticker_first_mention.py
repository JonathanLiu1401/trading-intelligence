"""Ticker first-mention detector.

Flags tickers that appeared in the last MENTION_WINDOW_MIN minutes but had
zero mentions in the prior LOOKBACK_HOURS window — i.e. newly-emerging
symbols. Complements trend_velocity (which ranks accelerating tickers that
were already on the radar) by surfacing cold-start signals.

Output: /home/zeph/logs/ticker_first_mention.json
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/ticker_first_mention.json")
MENTION_WINDOW_MIN = 60
LOOKBACK_HOURS = 6
FETCH_LIMIT = 8000
TOP_N = 20

TICKER_RE = re.compile(r"\b\$?([A-Z]{2,5})\b")
STOP = {
    "CEO", "CFO", "CTO", "USA", "USD", "EUR", "GBP", "EU", "UK", "US",
    "AI", "ML", "API", "IPO", "ETF", "SEC", "FOMC", "FED", "GDP", "CPI",
    "PPI", "ECB", "BOJ", "PBOC", "OPEC", "NYSE", "NASDAQ", "AMEX",
    "Q1", "Q2", "Q3", "Q4", "YTD", "YOY", "EPS", "PE", "EV", "ESG",
    "BUY", "SELL", "HOLD", "ON", "AT", "IN", "TO", "OF", "FOR", "THE",
    "AND", "OR", "BY", "AS", "IS", "WAS", "ARE", "BE", "AN", "A",
    "NEW", "OLD", "TOP", "LOW", "HIGH", "BIG", "DAY", "WEEK", "MONTH",
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
    "II", "III", "IV", "VI", "IT", "ITS", "WE", "OUR", "YOU", "HE", "SHE",
    "THIS", "THAT", "WITH", "FROM", "INTO", "OVER", "AFTER", "BEFORE",
    "AGO", "NOW", "AM", "PM", "ET", "PT", "UTC", "GMT",
}


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00").replace(" ", "T"))
    except ValueError:
        return None


def _extract(title: str | None) -> set[str]:
    if not title:
        return set()
    return {m for m in TICKER_RE.findall(title) if m not in STOP and len(m) >= 2}


def run() -> dict:
    now = datetime.now(timezone.utc)
    recent_cut = now - timedelta(minutes=MENTION_WINDOW_MIN)
    history_cut = now - timedelta(hours=LOOKBACK_HOURS)

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    # Canonical `_LIVE_ONLY_CLAUSE` — synthetic backtest/opus rows carry
    # tickers in their title (the same tickers as live news) and would otherwise
    # mask a genuinely cold-start "first mention" by appearing in the LOOKBACK
    # history window.
    rows = con.execute(
        "SELECT first_seen, title, source, url, ml_score "
        "FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY rowid DESC LIMIT ?",
        (FETCH_LIMIT,),
    ).fetchall()
    con.close()

    recent: dict[str, list[tuple[datetime, str, str, str, float | None]]] = defaultdict(list)
    history: set[str] = set()

    for first_seen, title, source, url, ml_score in rows:
        ts = _parse(first_seen)
        if ts is None or ts < history_cut:
            continue
        ticks = _extract(title)
        if not ticks:
            continue
        if ts >= recent_cut:
            for t in ticks:
                recent[t].append((ts, title, source, url, ml_score))
        else:
            history.update(ticks)

    new_tickers = []
    for t, mentions in recent.items():
        if t in history:
            continue
        mentions.sort(key=lambda r: r[0])
        first = mentions[0]
        new_tickers.append({
            "ticker": t,
            "count": len(mentions),
            "first_seen": first[0].isoformat(),
            "title": first[1],
            "source": first[2],
            "url": first[3],
            "avg_ml_score": (
                round(sum(m[4] for m in mentions if m[4] is not None) / max(1, sum(1 for m in mentions if m[4] is not None)), 4)
                if any(m[4] is not None for m in mentions) else None
            ),
        })

    new_tickers.sort(key=lambda r: (-r["count"], r["first_seen"]))
    report = {
        "generated_at": now.isoformat(),
        "window_min": MENTION_WINDOW_MIN,
        "lookback_hours": LOOKBACK_HOURS,
        "scanned_rows": len(rows),
        "new_ticker_count": len(new_tickers),
        "new_tickers": new_tickers[:TOP_N],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    report = run()
    print(f"scanned={report['scanned_rows']} new_tickers={report['new_ticker_count']}")
    for nt in report["new_tickers"][:10]:
        print(f"  {nt['ticker']}: count={nt['count']} src={nt['source']} :: {nt['title'][:80]}")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
