"""Ticker mention velocity runner.

Discovers the top tickers from live articles in the past 4 hours, then
uses ``ArticleStore.ticker_mention_velocity`` (correct live-only filter,
whole-word matching) to compute recent vs prior-window counts and ratios.

Writes to /home/zeph/logs/ticker_velocity.json:
  {
    "generated_at": "<ISO>",
    "window_min": 120,
    "tickers": [
      {"ticker": "NVDA", "recent": 5, "prior": 2, "ratio": 2.0,
       "newest_age_s": 340.1},
      ...
    ]
  }

Complements analytics/trend_velocity.py (which uses a custom regex with a
partial live-only filter) with a version backed by the validated store method.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from storage.article_store import ArticleStore, _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/ticker_velocity.json")

DISCOVER_WINDOW_HOURS = 4
DISCOVER_LIMIT = 6000
TOP_N = 20
VELOCITY_WINDOW_MIN = 120

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
    "II", "III", "IV", "VI", "NEWS", "INC", "LLC", "LTD", "CORP", "CO",
    "PLC", "MSN", "CNN", "BBC", "WSJ", "NYT", "FT", "AP", "AFP",
    "MONEY", "STOCK", "STOCKS", "MARKET", "DEAL", "DEALS", "JUNE", "JULY",
    "DRAM", "NAND", "HDD", "SSD", "RAM", "CPU", "GPU", "PCB", "PCIe",
    "AG", "TD", "AT", "BE",
}


def _discover_tickers(conn: sqlite3.Connection) -> list[str]:
    """Return top TOP_N tickers by raw mention count from recent live articles."""
    rows = conn.execute(
        f"""
        SELECT title FROM articles
        WHERE replace(first_seen, 'T', ' ') >= datetime('now', '-{DISCOVER_WINDOW_HOURS} hours')
          AND {_LIVE_ONLY_CLAUSE}
        LIMIT {DISCOVER_LIMIT}
        """
    ).fetchall()

    counts: Counter[str] = Counter()
    for (title,) in rows:
        for m in TICKER_RE.findall(title or ""):
            if m not in STOP and len(m) >= 2:
                counts[m] += 1

    return [t for t, _ in counts.most_common(TOP_N)]


def compute() -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout=5000")

    tickers = _discover_tickers(conn)
    if not tickers:
        return []

    store = ArticleStore()
    results = store.ticker_mention_velocity(tickers, window_min=VELOCITY_WINDOW_MIN)

    now_iso = datetime.now(timezone.utc).isoformat()
    out = {
        "generated_at": now_iso,
        "window_min": VELOCITY_WINDOW_MIN,
        "tickers": results,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    return results


def main() -> None:
    results = compute()
    print(f"ticker_velocity_runner: {len(results)} tickers scored")
    for row in results[:10]:
        age = f"{row['newest_age_s']:.0f}s ago" if row["newest_age_s"] is not None else "no hits"
        print(
            f"  {row['ticker']:6s}  recent={row['recent']:3d}  prior={row['prior']:3d}"
            f"  ratio={row['ratio']:.2f}  ({age})"
        )


if __name__ == "__main__":
    main()
