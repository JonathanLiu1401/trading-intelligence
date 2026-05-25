"""Earnings Pre-heat Ranker.

For each ticker with earnings in the next 7 days (sourced from the
``nasdaq/earnings_calendar`` articles already in the DB), measures recent
article coverage (volume + quality) from NON-earnings sources in the last
48 hours.  A high "preheat score" (mention_count × avg_ml_score) suggests
the market is actively discussing the stock ahead of its print, which often
precedes larger gap moves.

Design constraints:
  * Two bounded idx_first_seen scans — no full-table scan.
  * Read-only connection, busy_timeout 5000 ms.
  * _LIVE_ONLY_CLAUSE applied — no backtest rows.
  * Ticker extraction from titles only (no tickers column).

Output: /home/zeph/logs/earnings_preheat.json

Standalone::

    python3 -m analytics.earnings_preheat
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
OUT_PATH = Path("/home/zeph/logs/earnings_preheat.json")

CALENDAR_SCAN = 500      # how many earnings_calendar rows to pull
ARTICLE_SCAN = 8000      # bounded scan for recent live articles
LOOKBACK_DAYS = 7        # earnings within this many days are "upcoming"
ARTICLE_WINDOW_H = 48    # how far back to look for pre-earnings articles
TOP_N = 15

EARNINGS_PAT = re.compile(
    r"^([A-Z]{1,5}) \(.*?\) reports earnings (\d{4}-\d{2}-\d{2})"
)
TICKER_RE = re.compile(r"\b\$?([A-Z]{2,5})\b")
STOP = {
    "CEO", "CFO", "CTO", "USA", "USD", "EUR", "GBP", "EU", "UK", "US",
    "AI", "ML", "API", "IPO", "ETF", "SEC", "FOMC", "FED", "GDP", "CPI",
    "PPI", "ECB", "BOJ", "PBOC", "OPEC", "NYSE", "NASDAQ", "AMEX",
    "Q1", "Q2", "Q3", "Q4", "YTD", "YOY", "EPS", "PE", "EV", "ESG",
    "BUY", "SELL", "HOLD", "ON", "AT", "IN", "TO", "OF", "FOR", "THE",
    "AND", "OR", "BY", "AS", "IS", "WAS", "ARE", "BE", "AN", "A",
    "NEW", "OLD", "TOP", "LOW", "HIGH", "BIG", "DAY", "WEEK", "MONTH",
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
    "NOV", "DEC", "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
    "II", "III", "IV", "VI", "NEWS", "INC", "LLC", "LTD", "CORP", "CO",
    "PLC", "MSN", "CNN", "BBC", "WSJ", "NYT", "FT", "AP", "AFP",
    "MONEY", "STOCK", "STOCKS", "MARKET", "DEAL", "DEALS", "JUNE", "JULY",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = str(raw).replace("T", " ").split("+")[0].strip()[:19]
    try:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _extract_tickers(title: str) -> set[str]:
    return {
        t for t in TICKER_RE.findall(title or "")
        if t not in STOP and len(t) >= 2
    }


def main() -> int:
    now = _utcnow()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout=5000")

    # --- Step 1: collect upcoming earnings tickers ---
    horizon = now + timedelta(days=LOOKBACK_DAYS)
    cal_rows = conn.execute(
        "SELECT DISTINCT title FROM articles "
        "WHERE source='nasdaq/earnings_calendar' "
        "ORDER BY first_seen DESC LIMIT ?",
        (CALENDAR_SCAN,),
    ).fetchall()

    earnings: dict[str, str] = {}  # ticker -> earnings_date string
    for (title,) in cal_rows:
        m = EARNINGS_PAT.match(title or "")
        if not m:
            continue
        ticker, date_str = m.group(1), m.group(2)
        try:
            earn_dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if now <= earn_dt <= horizon:
            if ticker not in earnings or date_str < earnings[ticker]:
                earnings[ticker] = date_str

    if not earnings:
        print("earnings_preheat: no upcoming earnings found in DB", file=sys.stderr)
        result = {"ts": now.isoformat(), "earnings_count": 0, "top": []}
        OUT_PATH.write_text(json.dumps(result, indent=2))
        return 0

    # --- Step 2: scan recent live articles for ticker mentions ---
    cutoff = now - timedelta(hours=ARTICLE_WINDOW_H)
    article_rows = conn.execute(
        f"SELECT title, ml_score, urgency, source, first_seen FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        f"AND source != 'nasdaq/earnings_calendar' "
        f"ORDER BY first_seen DESC LIMIT ?",
        (ARTICLE_SCAN,),
    ).fetchall()
    conn.close()

    # Filter to ARTICLE_WINDOW_H window
    mentions: dict[str, list[float]] = defaultdict(list)
    for title, ml_score, urgency, source, first_seen in article_rows:
        ts = _parse_ts(first_seen)
        if ts is None or ts < cutoff:
            continue
        for ticker in _extract_tickers(title):
            if ticker in earnings:
                score = float(ml_score or 0.0)
                mentions[ticker].append(score)

    # --- Step 3: compute preheat score and rank ---
    results = []
    for ticker, earn_date in earnings.items():
        scores = mentions.get(ticker, [])
        count = len(scores)
        avg_ml = sum(scores) / count if scores else 0.0
        preheat = round(count * avg_ml, 2)
        try:
            earn_dt = datetime.fromisoformat(earn_date).replace(tzinfo=timezone.utc)
            days_away = round((earn_dt - now).total_seconds() / 86400, 1)
        except ValueError:
            days_away = None
        results.append({
            "ticker": ticker,
            "earnings_date": earn_date,
            "days_away": days_away,
            "mention_count": count,
            "avg_ml_score": round(avg_ml, 3),
            "preheat_score": preheat,
        })

    results.sort(key=lambda x: x["preheat_score"], reverse=True)
    top = results[:TOP_N]

    # --- Step 4: print and write output ---
    total_with_mentions = sum(1 for r in results if r["mention_count"] > 0)
    print(
        f"earnings_preheat: {len(earnings)} upcoming earnings | "
        f"{total_with_mentions} have recent coverage | "
        f"scanned={len(article_rows)} articles"
    )
    for r in top:
        if r["mention_count"] == 0:
            break
        print(
            f"  {r['ticker']:<6} preheat={r['preheat_score']:>7.1f} "
            f"count={r['mention_count']:>3}  avg_ml={r['avg_ml_score']:.3f}  "
            f"earns={r['earnings_date']} (+{r['days_away']}d)"
        )

    output = {
        "ts": now.isoformat(),
        "earnings_count": len(earnings),
        "article_window_h": ARTICLE_WINDOW_H,
        "total_with_mentions": total_with_mentions,
        "scanned_articles": len(article_rows),
        "top": top,
    }
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"output={OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
