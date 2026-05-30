"""Crypto news RSS collector — CoinTelegraph, CoinDesk, Decrypt.

Covers major crypto news outlets not captured by the price/sentiment collectors
(coingecko, binance_funding, crypto_fear_greed). Valuable for COIN, MSTR,
BTC-adjacent equities, and broader DeFi/regulatory signals.

No API key required. Returns article dicts in daemon-standard format
(link/title/source/published/summary) for _ingest → ArticleStore.
Parallel fetch across all three feeds per call.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import feedparser

log = logging.getLogger("crypto_news_rss")

FEEDS = [
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Decrypt", "https://decrypt.co/feed"),
]

USER_AGENT = "Mozilla/5.0 (Digital Intern Daemon)"


def _fetch_feed(source_name: str, feed_url: str) -> list[dict]:
    """Fetch one RSS feed; return list of article dicts in daemon format."""
    try:
        feed = feedparser.parse(
            feed_url,
            request_headers={"User-Agent": USER_AGENT},
        )
        if feed.get("status", 200) >= 400:
            log.warning("%s: HTTP %s", source_name, feed.get("status"))
            return []
        articles = []
        for entry in feed.entries:
            title = (entry.get("title") or "").strip()
            link = (entry.get("link") or "").strip()
            if not title or not link:
                continue
            pub = ""
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
                except Exception:
                    pass
            summary = (entry.get("summary") or "")[:500]
            articles.append(
                {
                    "title": title,
                    "link": link,
                    "source": f"CryptoNews/{source_name}",
                    "published": pub,
                    "summary": summary,
                }
            )
        return articles
    except Exception as e:
        log.error("%s: fetch failed: %s", source_name, e)
        return []


def collect_crypto_news() -> list[dict]:
    """Fetch all feeds in parallel; return combined article list for _ingest."""
    all_articles: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(FEEDS)) as pool:
        futures = {
            pool.submit(_fetch_feed, name, url): name for name, url in FEEDS
        }
        for fut in as_completed(futures):
            results = fut.result()
            all_articles.extend(results)
    log.info("crypto_news_rss: %d articles fetched", len(all_articles))
    return all_articles


if __name__ == "__main__":
    import sqlite3
    import hashlib
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    articles = collect_crypto_news()
    print(f"\nFetched {len(articles)} articles total")
    for src in ["CoinTelegraph", "CoinDesk", "Decrypt"]:
        subset = [a for a in articles if src in a["source"]]
        print(f"\n--- {src} ({len(subset)}) ---")
        for a in subset[:3]:
            print(f"  {a['title'][:80]}")
            print(f"  {a['link'][:70]}")

    # Standalone insert for testing (bypasses ArticleStore scoring)
    DB_PATH = Path(__file__).resolve().parent.parent / "data" / "articles.db"
    conn = sqlite3.connect(str(DB_PATH), timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    new_count = 0
    with conn:
        for art in articles:
            aid = hashlib.sha256(f"{art['link']}|{art['title']}".encode()).hexdigest()[:32]
            conn.execute(
                "INSERT OR IGNORE INTO articles (id, url, title, source, published, first_seen) VALUES (?,?,?,?,?,?)",
                (aid, art["link"], art["title"], art["source"], art["published"], now),
            )
            if conn.execute("SELECT changes()").fetchone()[0]:
                new_count += 1
    conn.close()
    print(f"\nInserted {new_count} new articles to DB")
