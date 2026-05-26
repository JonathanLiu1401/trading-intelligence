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

Public surfaces:
  * ``build_sentiment_reversal(articles, now=None)`` — pure builder.
    Takes an iterable of ``{"first_seen", "title", "ml_score"}`` dicts and
    returns the full payload. No DB access, no I/O — used by the Flask
    endpoint and unit tests.
  * ``compute()`` — opens articles.db, runs the builder, writes a snapshot
    JSON to /home/zeph/logs/. Used by the standalone CLI.

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
from typing import Iterable

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


def _parse_ts(raw) -> datetime | None:
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    s = str(raw).strip().replace("Z", "+00:00")
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


def build_sentiment_reversal(
    articles: Iterable[dict],
    now: datetime | None = None,
) -> dict:
    """Pure builder.

    ``articles`` is any iterable of dicts with at least ``first_seen``,
    ``title``, and ``ml_score`` keys. Rows with missing/None ``ml_score``
    or unparseable ``first_seen`` are skipped (counted as ``skipped``).

    Returns a stable shape so the endpoint can ``jsonify`` it directly and
    chat helpers can assume the keys exist:

      {
        "generated_at": iso str,
        "window_hours": int,
        "rows_scanned": int,
        "skipped": int,
        "min_articles_per_window": int,
        "min_delta": float,
        "reversals_found": int,
        "reversals": [
            {ticker, direction, avg_prev, avg_curr, delta,
             articles_prev, articles_curr}, ...
        ],
      }
    """
    now = (now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cutoff_curr = now - timedelta(hours=WINDOW_HOURS)
    cutoff_prev = now - timedelta(hours=WINDOW_HOURS * 2)

    curr_scores: dict[str, list[float]] = defaultdict(list)
    prev_scores: dict[str, list[float]] = defaultdict(list)
    rows_scanned = 0
    skipped = 0

    for art in articles:
        rows_scanned += 1
        ml_score = art.get("ml_score")
        ts = _parse_ts(art.get("first_seen"))
        if ts is None or ml_score is None:
            skipped += 1
            continue
        if ts >= cutoff_curr:
            window = curr_scores
        elif ts >= cutoff_prev:
            window = prev_scores
        else:
            continue
        try:
            score = float(ml_score)
        except (TypeError, ValueError):
            skipped += 1
            continue
        for ticker in _extract_tickers(art.get("title") or ""):
            window[ticker].append(score)

    reversals: list[dict] = []
    for ticker in set(curr_scores) | set(prev_scores):
        c = curr_scores.get(ticker, [])
        p = prev_scores.get(ticker, [])
        if len(c) < MIN_ARTICLES or len(p) < MIN_ARTICLES:
            continue
        avg_curr = sum(c) / len(c)
        avg_prev = sum(p) / len(p)
        delta = avg_curr - avg_prev
        if (avg_curr > 0) == (avg_prev > 0):
            continue
        if abs(delta) < MIN_DELTA:
            continue
        reversals.append({
            "ticker": ticker,
            "direction": "neg→pos" if avg_curr > 0 else "pos→neg",
            "avg_prev": round(avg_prev, 4),
            "avg_curr": round(avg_curr, 4),
            "delta": round(delta, 4),
            "articles_prev": len(p),
            "articles_curr": len(c),
        })

    reversals.sort(key=lambda x: abs(x["delta"]), reverse=True)
    reversals = reversals[:TOP_N]

    return {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "rows_scanned": rows_scanned,
        "skipped": skipped,
        "min_articles_per_window": MIN_ARTICLES,
        "min_delta": MIN_DELTA,
        "reversals_found": len(reversals),
        "reversals": reversals,
    }


def _fetch_articles_from_db(db_path: Path) -> list[dict]:
    """Read live articles from the DB (DESC by first_seen, capped)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=20)
    try:
        rows = conn.execute(
            "SELECT first_seen, title, ml_score FROM articles "
            f"WHERE {_LIVE_ONLY_CLAUSE} AND ml_score IS NOT NULL "
            "ORDER BY first_seen DESC LIMIT ?",
            (FETCH_LIMIT,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {"first_seen": r[0], "title": r[1], "ml_score": r[2]}
        for r in rows
    ]


def compute(now: datetime | None = None, db_path: Path | None = None) -> dict:
    """CLI entry — fetch from DB, build the payload, write snapshot, return it."""
    db_path = db_path or DB_PATH
    articles = _fetch_articles_from_db(db_path)
    payload = build_sentiment_reversal(articles, now=now)
    try:
        SNAPSHOT_PATH.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass
    return payload


def main() -> int:
    payload = compute()
    reversals = payload.get("reversals") or []
    print(
        f"Scanned {payload.get('rows_scanned', 0)} rows | "
        f"reversals: {len(reversals)}"
    )
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
