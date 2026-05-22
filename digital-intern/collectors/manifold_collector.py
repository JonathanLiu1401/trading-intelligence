"""Manifold Markets prediction market collector — crowd-sourced probabilities.

Fetches active markets from Manifold Markets' public API and filters for
financial/macro relevance. Complements polymarket_collector.py — Manifold has
different markets, users, and time horizons.

Why this matters for trading:
  Manifold aggregates dispersed information into probabilities on topics like
  Fed rate decisions, recession likelihood, and company milestones. These
  often capture consensus before mainstream news.

Dedup strategy:
  Each market is keyed by its Manifold market ID + today's date, so the same
  market re-emits at most once per day (catching daily probability updates).

No API key required. Manifold's public API has no published rate limit;
we fetch once per daemon cycle which is well within safe use.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from core.logger import get_logger
    _log = get_logger("manifold")
except Exception:
    _log = logging.getLogger("manifold")

BASE_DIR = Path(__file__).resolve().parent.parent
SEEN_DB = BASE_DIR / "data" / "seen_articles.db"

API_BASE = "https://api.manifold.markets/v0"
SEARCH_URL = f"{API_BASE}/search-markets"
SOURCE_NAME = "manifold"
FETCH_TIMEOUT = 15

FIN_KEYWORDS = {
    "fed", "rate", "inflation", "cpi", "pce", "gdp", "recession", "tariff",
    "unemployment", "treasury", "interest", "stock", "bitcoin", "btc",
    "dollar", "economy", "trade", "deficit", "debt", "earnings", "ipo",
    "crypto", "oil", "gold", "market", "s&p", "nasdaq", "dow", "trump",
    "china", "nvidia", "apple", "tesla", "bank", "fomc", "jobs", "payroll",
    "yields", "bonds", "sanctions", "imf", "world bank", "opec",
    "etf", "merger", "acquisition", "bankruptcy", "default", "housing",
    "mortgage", "semiconductor", "ai", "tariff", "iran", "russia", "ukraine",
    "election", "congress", "biden", "harris", "powell",
}

# Search terms to sweep across Manifold for financial/macro markets
SEARCH_QUERIES = [
    "fed rate interest recession",
    "inflation cpi gdp economy",
    "stock market crash bull bear",
    "bitcoin crypto dollar",
    "tariff trade china",
    "unemployment jobs payroll",
    "nvidia apple tesla earnings",
    "treasury bond yield curve",
]

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _ensure_db() -> sqlite3.Connection:
    SEEN_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SEEN_DB), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY, link TEXT, title TEXT,
            source TEXT, first_seen TEXT
        )"""
    )
    conn.commit()
    return conn


def _seen_id(market_id: str, date: str) -> str:
    return hashlib.sha256(f"manifold:{market_id}:{date}".encode()).hexdigest()


def _is_seen(conn: sqlite3.Connection, sid: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM seen_articles WHERE id=?", (sid,)
    ).fetchone() is not None


def _mark_seen(conn: sqlite3.Connection, sid: str, link: str, title: str):
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
        "VALUES (?, ?, ?, ?, ?)",
        (sid, link, title, SOURCE_NAME, datetime.now(timezone.utc).isoformat()),
    )


def _is_relevant(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in FIN_KEYWORDS)


def _fetch_markets_for_query(query: str, limit: int = 20) -> list[dict]:
    try:
        r = requests.get(
            SEARCH_URL,
            params={"term": query, "limit": limit, "sort": "score"},
            headers={"User-Agent": _UA},
            timeout=FETCH_TIMEOUT,
        )
        r.raise_for_status()
        return r.json() if isinstance(r.json(), list) else []
    except Exception as e:
        _log.warning("manifold fetch failed for %r: %s", query, e)
        return []


def collect_manifold() -> list[dict]:
    """Return article-like dicts for relevant, active Manifold markets."""
    conn = _ensure_db()
    today = _today()
    articles: list[dict] = []
    seen_market_ids: set[str] = set()

    for query in SEARCH_QUERIES:
        markets = _fetch_markets_for_query(query)
        for m in markets:
            mid = m.get("id", "")
            if not mid or mid in seen_market_ids:
                continue
            if m.get("isResolved"):
                continue
            prob = m.get("probability")
            if prob is None:
                continue

            question = m.get("question", "")
            if not _is_relevant(question):
                continue

            seen_market_ids.add(mid)
            sid = _seen_id(mid, today)
            if _is_seen(conn, sid):
                continue

            pct = round(prob * 100, 1)
            url = m.get("url", f"https://manifold.markets/market/{mid}")
            volume = m.get("volume", 0)
            bettors = m.get("uniqueBettorCount", 0)

            title = f"[Manifold] {question} — {pct}% probability"
            summary = (
                f"Crowd probability: {pct}%. "
                f"Volume: {volume:.0f} MANA, {bettors} bettors. "
                f"Source: Manifold Markets (play-money prediction market)."
            )

            close_ts = m.get("closeTime")
            if close_ts:
                try:
                    published = datetime.fromtimestamp(
                        close_ts / 1000, tz=timezone.utc
                    ).strftime("%a, %d %b %Y %H:%M:%S +0000")
                except Exception:
                    published = datetime.now(timezone.utc).strftime(
                        "%a, %d %b %Y %H:%M:%S +0000"
                    )
            else:
                published = datetime.now(timezone.utc).strftime(
                    "%a, %d %b %Y %H:%M:%S +0000"
                )

            articles.append({
                "title": title,
                "link": url,
                "summary": summary,
                "published": published,
                "source": SOURCE_NAME,
                "_tickers": [],
            })
            _mark_seen(conn, sid, url, title)

    conn.commit()
    conn.close()
    _log.info("manifold: %d new market articles", len(articles))
    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    results = collect_manifold()
    print(f"\nFetched {len(results)} new Manifold market articles:\n")
    for a in results[:10]:
        print(f"  {a['title']}")
        print(f"  {a['link']}")
        print()
