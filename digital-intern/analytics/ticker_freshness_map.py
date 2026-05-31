"""Ticker coverage freshness map.

For the top-N most-mentioned tickers in the last 24h, shows how recent the
coverage actually is — complementing velocity (ratio) with an absolute
recency signal useful for pre-open watchlist preparation.

Freshness tiers (from most-recent article mentioning the ticker):
  HOT   — within last 1h
  WARM  — 1–6h
  COOL  — 6–24h
  COLD  — >24h (still in the 24h window tail)

Why this differs from siblings:
  * trend_velocity / ticker_velocity_runner  — momentum ratio, not raw recency
  * held_ticker_news_silence                 — only held book tickers
  * overnight_gap_scanner                   — urgency-gated, market-hours filter

Design: single bounded idx_first_seen scan (USB-safe), read-only, no full scan.

Output: /home/zeph/logs/ticker_freshness_map.json
Standalone: python3 -m analytics.ticker_freshness_map
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = os.path.join(BASE, "data", "articles.db")
OUT_PATH = "/home/zeph/logs/ticker_freshness_map.json"
SCAN_LIMIT = 5000
TOP_N = 25

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
    return dt.astimezone(timezone.utc)


def _extract_tickers(title: str) -> list[str]:
    return [
        m for m in TICKER_RE.findall(title or "")
        if m not in STOP and len(m) >= 2
    ]


def _freshness_tier(age_hours: float) -> str:
    if age_hours <= 1.0:
        return "HOT"
    if age_hours <= 6.0:
        return "WARM"
    if age_hours <= 24.0:
        return "COOL"
    return "COLD"


def build_ticker_freshness_map(
    rows: list[tuple[str, str]],
    now: datetime,
    top_n: int = TOP_N,
) -> dict:
    """Pure builder — no I/O. rows = [(first_seen, title), ...]."""
    count_24h: dict[str, int] = defaultdict(int)
    count_6h: dict[str, int] = defaultdict(int)
    count_1h: dict[str, int] = defaultdict(int)
    latest_ts: dict[str, datetime] = {}

    cutoff_24h = now.timestamp() - 86400
    cutoff_6h = now.timestamp() - 21600
    cutoff_1h = now.timestamp() - 3600

    for raw_ts, title in rows:
        dt = _parse_ts(raw_ts)
        if dt is None:
            continue
        ts = dt.timestamp()
        if ts < cutoff_24h:
            continue
        tickers = _extract_tickers(title)
        for tk in tickers:
            count_24h[tk] += 1
            if ts >= cutoff_6h:
                count_6h[tk] += 1
            if ts >= cutoff_1h:
                count_1h[tk] += 1
            if tk not in latest_ts or dt > latest_ts[tk]:
                latest_ts[tk] = dt

    top = sorted(count_24h, key=lambda t: count_24h[t], reverse=True)[:top_n]
    result = []
    for tk in top:
        lt = latest_ts[tk]
        age_h = (now.timestamp() - lt.timestamp()) / 3600
        result.append({
            "ticker": tk,
            "count_24h": count_24h[tk],
            "count_6h": count_6h[tk],
            "count_1h": count_1h[tk],
            "latest_ts": lt.isoformat(),
            "age_hours": round(age_h, 2),
            "freshness": _freshness_tier(age_h),
        })
    return result


def main() -> int:
    if not os.path.exists(DB_PATH):
        print("db missing", file=sys.stderr)
        return 1

    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=15)
    conn.execute("PRAGMA busy_timeout=10000")

    rows = conn.execute(
        f"SELECT first_seen, title FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        f"ORDER BY first_seen DESC LIMIT ?",
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    now = datetime.now(timezone.utc)
    tickers = build_ticker_freshness_map(rows, now)

    payload = {
        "generated_at": now.isoformat(),
        "scan_limit": SCAN_LIMIT,
        "rows_scanned": len(rows),
        "top_n": TOP_N,
        "tickers": tickers,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    hot = [t for t in tickers if t["freshness"] == "HOT"]
    warm = [t for t in tickers if t["freshness"] == "WARM"]
    cool = [t for t in tickers if t["freshness"] == "COOL"]

    print(f"Ticker freshness map — scanned {len(rows)} articles, {len(tickers)} tickers ranked")
    print(f"  HOT ({len(hot)}): {', '.join(t['ticker'] for t in hot[:5])}")
    print(f"  WARM ({len(warm)}): {', '.join(t['ticker'] for t in warm[:5])}")
    print(f"  COOL ({len(cool)}): {', '.join(t['ticker'] for t in cool[:5])}")
    top5 = ", ".join(f"{t['ticker']}({t['count_24h']})" for t in tickers[:5])
    print(f"Top 5: {top5}")
    print(f"Saved → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
