"""Kalshi prediction market collector — CFTC-regulated real-money probabilities.

Fetches active economic/financial markets from Kalshi (api.elections.kalshi.com)
and emits synthetic articles encoding market-implied probabilities. Unlike
Polymarket (crypto-native), Kalshi is CFTC-regulated and attracts more
institutional participation, giving its probabilities higher signal quality
for macro events (Fed decisions, CPI, unemployment, recession odds).

Targeted series:
  KXFED    — Fed funds rate after each FOMC meeting
  KXCPI    — CPI YoY outcome buckets
  KXUNRATE — Unemployment rate outcomes
  KXGDP    — GDP growth outcomes
  KXRECESSION — US recession probability

Dedup: each event_ticker + date → at most one article/day (probability updates
surface daily without spam).

No auth required for read-only market data.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
SEEN_DB = BASE_DIR / "data" / "seen_articles.db"

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
FETCH_TIMEOUT = 15

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Economic series to monitor
TARGET_SERIES = [
    "KXFED",       # Fed funds rate
    "KXCPI",       # CPI
    "KXUNRATE",    # Unemployment rate
    "KXGDP",       # GDP
]

# Also fetch by category for broader macro coverage
CATEGORIES = ["Economics", "Finance"]

# Min open interest to skip dust markets
MIN_OI = 10.0


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _ensure_db() -> sqlite3.Connection:
    SEEN_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SEEN_DB), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY,
            link TEXT, title TEXT, source TEXT, first_seen TEXT
        )
    """)
    conn.commit()
    return conn


def _article_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _fetch_events(series_ticker: str) -> list[dict]:
    """Fetch open events for a given series."""
    try:
        r = requests.get(
            f"{API_BASE}/events",
            params={"status": "open", "series_ticker": series_ticker, "limit": 20},
            headers={"Accept": "application/json", "User-Agent": _UA},
            timeout=FETCH_TIMEOUT,
        )
        if r.status_code != 200:
            return []
        return r.json().get("events", [])
    except Exception:
        return []


def _fetch_markets(event_ticker: str) -> list[dict]:
    """Fetch active markets for an event."""
    try:
        r = requests.get(
            f"{API_BASE}/markets",
            params={"status": "open", "event_ticker": event_ticker, "limit": 30},
            headers={"Accept": "application/json", "User-Agent": _UA},
            timeout=FETCH_TIMEOUT,
        )
        if r.status_code != 200:
            return []
        return r.json().get("markets", [])
    except Exception:
        return []


def _fetch_category_events(category: str) -> list[dict]:
    """Fetch events by category for broad macro coverage."""
    try:
        r = requests.get(
            f"{API_BASE}/events",
            params={"status": "open", "category": category, "limit": 50},
            headers={"Accept": "application/json", "User-Agent": _UA},
            timeout=FETCH_TIMEOUT,
        )
        if r.status_code != 200:
            return []
        return r.json().get("events", [])
    except Exception:
        return []


def _format_probability(price_str: str | None) -> str:
    """Convert dollar price string like '0.6500' → '65%'."""
    if not price_str:
        return "N/A"
    try:
        return f"{float(price_str) * 100:.0f}%"
    except (ValueError, TypeError):
        return "N/A"


def _summarize_event(event: dict, markets: list[dict]) -> str | None:
    """Build a human-readable probability summary for an event."""
    title = event.get("title", "")
    if not title or not markets:
        return None

    # Filter to markets with meaningful open interest
    active = [
        m for m in markets
        if float(m.get("open_interest_fp", "0") or 0) >= MIN_OI
    ]
    if not active:
        # fall back to all markets if none meet OI threshold
        active = markets[:5]

    lines = [f"Kalshi: {title}"]
    for m in active[:6]:
        yes_bid = m.get("yes_bid_dollars") or m.get("last_price_dollars", "0")
        pct = _format_probability(yes_bid)
        subtitle = m.get("subtitle") or m.get("yes_sub_title") or m.get("ticker", "")
        lines.append(f"  • {subtitle}: {pct}")

    close_time = event.get("strike_date") or markets[0].get("close_time", "")
    if close_time:
        lines.append(f"  Resolves: {close_time[:10]}")

    return "\n".join(lines)


def collect() -> list[dict]:
    conn = _ensure_db()
    articles: list[dict] = []
    today = _today()

    # Gather events from targeted series + category sweep
    seen_event_tickers: set[str] = set()
    all_events: list[dict] = []

    for series in TARGET_SERIES:
        for ev in _fetch_events(series):
            t = ev.get("event_ticker", "")
            if t and t not in seen_event_tickers:
                seen_event_tickers.add(t)
                all_events.append(ev)

    for cat in CATEGORIES:
        for ev in _fetch_category_events(cat):
            t = ev.get("event_ticker", "")
            if t and t not in seen_event_tickers:
                seen_event_tickers.add(t)
                all_events.append(ev)

    for event in all_events:
        event_ticker = event.get("event_ticker", "")
        if not event_ticker:
            continue

        dedup_key = f"kalshi:{event_ticker}:{today}"
        aid = _article_id(dedup_key)

        existing = conn.execute(
            "SELECT id FROM seen_articles WHERE id=?", (aid,)
        ).fetchone()
        if existing:
            continue

        markets = _fetch_markets(event_ticker)
        summary = _summarize_event(event, markets)
        if not summary:
            continue

        title = event.get("title", event_ticker)
        link = f"https://kalshi.com/markets/{event_ticker.lower()}"
        now_iso = datetime.now(timezone.utc).isoformat()

        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?,?,?,?,?)",
            (aid, link, title, "kalshi", now_iso),
        )
        conn.commit()

        articles.append({
            "id": aid,
            "title": title,
            "link": link,
            "summary": summary,
            "source": "kalshi",
            "published": now_iso,
        })

    conn.close()
    return articles


if __name__ == "__main__":
    results = collect()
    print(f"Kalshi: fetched {len(results)} new market probability articles")
    for a in results[:10]:
        print(f"\n[{a['source']}] {a['title']}")
        print(a["summary"])
