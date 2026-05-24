"""Sentiment reversal detector.

Identifies tickers whose average ml_score flips sign (negative→positive or
positive→negative) between consecutive 2-hour windows.  A flip combined with
meaningful article volume in both windows can indicate a news-driven sentiment
shift worth attention.

Design:
  * Two 2h windows: PREV  = [4h ago, 2h ago)  and  CURR = [2h ago, now)
  * Ticker extraction via the same regex/stopword set as trend_velocity.
  * Only considers tickers with >= MIN_ARTICLES in each window (avoids noise).
  * A reversal requires:
      - sign(avg_curr) != sign(avg_prev)  (actual flip)
      - abs(avg_curr - avg_prev) >= MIN_DELTA  (meaningful magnitude change)
  * Output written to SNAPSHOT_PATH; summary logged to stdout.

Standalone:  python3 -m analytics.sentiment_reversal
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
SNAPSHOT_PATH = Path("/home/zeph/logs/sentiment_reversal.json")

WINDOW_HOURS = 2
FETCH_LIMIT = 5000
MIN_ARTICLES = 2      # per ticker per window
MIN_DELTA = 0.15      # min score swing to count as a reversal
TOP_N = 10

TICKER_RE = re.compile(r"\b\$?([A-Z]{2,5})\b")
STOP = {
    "CEO", "CFO", "CTO", "USA", "USD", "EUR", "GBP", "EU", "UK", "US",
    "AI", "ML", "API", "IPO", "ETF", "SEC", "FOMC", "FED", "GDP", "CPI",
    "PPI", "ECB", "BOJ", "PBOC", "OPEC", "NYSE", "NASDAQ", "AMEX",
    "Q1", "Q2", "Q3", "Q4", "YTD", "YOY", "EPS", "PE", "EV", "ESG",
    "BUY", "SELL", "HOLD", "ON", "AT", "IN", "TO", "OF", "FOR", "THE",
    "AND", "OR", "BY", "AS", "IS", "WAS", "ARE", "BE", "AN", "A",
    "NEW", "OLD", "TOP", "LOW", "HIGH", "BIG", "DAY", "WEEK", "MONTH",
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP",
    "OCT", "NOV", "DEC", "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
    "II", "III", "IV", "VI",
    "NEWS", "INC", "LLC", "LTD", "CORP", "CO", "PLC",
    "MSN", "CNN", "BBC", "WSJ", "NYT", "FT", "AP", "AFP",
    "MONEY", "STOCK", "STOCKS", "MARKET", "DEAL", "DEALS",
    "JUNE", "JULY",
}


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.strptime(s[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_tickers(title: str) -> list[str]:
    return [m for m in TICKER_RE.findall(title or "") if m not in STOP and len(m) >= 2]


def main() -> int:
    now = datetime.now(timezone.utc)
    cutoff_curr = now - timedelta(hours=WINDOW_HOURS)       # 2h ago
    cutoff_prev = now - timedelta(hours=WINDOW_HOURS * 2)   # 4h ago

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    try:
        rows = conn.execute(
            "SELECT first_seen, title, ml_score FROM articles "
            f"WHERE {_LIVE_ONLY_CLAUSE} AND ml_score IS NOT NULL "
            "ORDER BY first_seen DESC LIMIT ?",
            (FETCH_LIMIT,),
        ).fetchall()
    finally:
        conn.close()

    # Bucket articles into PREV and CURR windows
    curr_scores: dict[str, list[float]] = defaultdict(list)
    prev_scores: dict[str, list[float]] = defaultdict(list)
    skipped = 0

    for raw_ts, title, ml_score in rows:
        ts = _parse_ts(raw_ts)
        if ts is None or ml_score is None:
            skipped += 1
            continue
        if ts >= cutoff_curr:
            window = curr_scores
        elif ts >= cutoff_prev:
            window = prev_scores
        else:
            break  # rows are DESC by first_seen; beyond 4h, stop

        for ticker in _extract_tickers(title):
            window[ticker].append(float(ml_score))

    # Find reversals
    reversals = []
    all_tickers = set(curr_scores) | set(prev_scores)
    for ticker in all_tickers:
        c_scores = curr_scores.get(ticker, [])
        p_scores = prev_scores.get(ticker, [])
        if len(c_scores) < MIN_ARTICLES or len(p_scores) < MIN_ARTICLES:
            continue

        avg_curr = sum(c_scores) / len(c_scores)
        avg_prev = sum(p_scores) / len(p_scores)
        delta = avg_curr - avg_prev

        if (avg_curr > 0) == (avg_prev > 0):
            continue  # no sign flip
        if abs(delta) < MIN_DELTA:
            continue  # too small

        direction = "neg→pos" if avg_curr > 0 else "pos→neg"
        reversals.append({
            "ticker": ticker,
            "direction": direction,
            "avg_prev": round(avg_prev, 4),
            "avg_curr": round(avg_curr, 4),
            "delta": round(delta, 4),
            "articles_prev": len(p_scores),
            "articles_curr": len(c_scores),
        })

    reversals.sort(key=lambda x: abs(x["delta"]), reverse=True)
    reversals = reversals[:TOP_N]

    snapshot = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "rows_scanned": len(rows),
        "skipped": skipped,
        "min_articles_per_window": MIN_ARTICLES,
        "min_delta": MIN_DELTA,
        "reversals_found": len(reversals),
        "reversals": reversals,
    }
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2))

    print(f"Scanned {len(rows)} rows | reversals: {len(reversals)}")
    if reversals:
        for r in reversals[:5]:
            print(
                f"  {r['ticker']:6s} {r['direction']}  "
                f"prev={r['avg_prev']:+.3f} ({r['articles_prev']}art) "
                f"→ curr={r['avg_curr']:+.3f} ({r['articles_curr']}art)  "
                f"Δ={r['delta']:+.3f}"
            )
    else:
        print("  No reversals detected in current windows.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
