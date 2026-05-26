"""Source echo detector — distinguishes genuine multi-source coverage from wire syndication.

For the top trending tickers in the last 2 hours, computes:
  * ``total``       — raw article count
  * ``unique_src``  — count of distinct source values
  * ``echo_ratio``  — total / unique_src  (>3 = mostly syndication echo)
  * ``verdict``     — GENUINE | SYNDICATED | WEAK

  GENUINE    → unique_src >= 3 AND echo_ratio <= 2.5
  SYNDICATED → unique_src < 3 OR echo_ratio > 3.0
  WEAK       → total < 3 (not enough signal either way)

Writes to /home/zeph/logs/source_echo.json and prints a brief to stdout.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/source_echo.json")
WINDOW_HOURS = 2
SCAN_LIMIT = 4000
TOP_N = 8
MIN_ARTICLES = 3  # ignore tickers with fewer mentions

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
    return dt


def _extract_tickers(text: str) -> list[str]:
    return [t for t in TICKER_RE.findall(text or "") if t not in STOP and len(t) >= 2]


def _verdict(total: int, unique_src: int) -> str:
    if total < MIN_ARTICLES:
        return "WEAK"
    echo = total / max(unique_src, 1)
    if unique_src >= 3 and echo <= 2.5:
        return "GENUINE"
    return "SYNDICATED"


def main() -> None:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    conn.execute("PRAGMA busy_timeout=5000")
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=WINDOW_HOURS)).isoformat()

    rows = conn.execute(
        """
        SELECT title, source, first_seen
        FROM articles INDEXED BY idx_first_seen
        WHERE first_seen >= ?
          AND (source NOT LIKE 'backtest_run_%' OR source IS NULL)
        ORDER BY first_seen DESC
        LIMIT ?
        """,
        (cutoff, SCAN_LIMIT),
    ).fetchall()
    conn.close()

    # Per ticker: collect article count and set of unique sources
    ticker_total: Counter[str] = Counter()
    ticker_sources: dict[str, set[str]] = defaultdict(set)

    for title, source, first_seen in rows:
        ts = _parse_ts(first_seen)
        if ts is None:
            continue
        tickers = _extract_tickers(title or "")
        for t in tickers:
            ticker_total[t] += 1
            ticker_sources[t].add(source or "unknown")

    # Rank by total mentions, filter low-count
    top = [
        (tk, cnt) for tk, cnt in ticker_total.most_common(50)
        if cnt >= MIN_ARTICLES
    ][:TOP_N]

    results = []
    for ticker, total in top:
        srcs = ticker_sources[ticker]
        unique_src = len(srcs)
        echo_ratio = round(total / max(unique_src, 1), 2)
        verdict = _verdict(total, unique_src)
        results.append({
            "ticker": ticker,
            "total": total,
            "unique_src": unique_src,
            "echo_ratio": echo_ratio,
            "verdict": verdict,
            "top_sources": sorted(srcs)[:5],
        })

    out = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "scanned": len(rows),
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))

    print(f"source_echo: scanned={len(rows)} tickers={len(results)}")
    for r in results:
        verdict_tag = r["verdict"]
        print(
            f"  {r['ticker']}: articles={r['total']} srcs={r['unique_src']} "
            f"echo={r['echo_ratio']}x [{verdict_tag}]  "
            f"({', '.join(r['top_sources'][:3])})"
        )


if __name__ == "__main__":
    main()
