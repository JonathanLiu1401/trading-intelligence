"""InvestorPlace, Motley Fool, and Nasdaq RSS collector.

High-volume financial editorial sources not covered by the existing 302-feed
RSS collector config. Fetches in parallel; deduplicates against seen_articles.

Feeds (~50 entries each per cycle):
  InvestorPlace  — main feed + stocks-to-buy category
  Motley Fool    — main feed + investing/podcasts feed
  Nasdaq         — markets, stocks, and ETFs news feeds
"""
import hashlib
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

REQUEST_TIMEOUT = 10
MAX_WORKERS = 6

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

FEEDS = [
    ("InvestorPlace", "https://investorplace.com/feed/"),
    ("InvestorPlace/Stocks", "https://investorplace.com/category/stocks-to-buy/feed/"),
    ("MotleyFool", "https://www.fool.com/feeds/index.aspx"),
    ("MotleyFool/Investing", "https://www.fool.com/feeds/index.aspx?id=podcasts"),
    ("Nasdaq/Markets", "https://www.nasdaq.com/feed/rssoutbound?category=Markets"),
    ("Nasdaq/Stocks", "https://www.nasdaq.com/feed/rssoutbound?category=Stocks"),
    ("Nasdaq/ETFs", "https://www.nasdaq.com/feed/rssoutbound?category=ETFs+and+Funds"),
]


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
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


def _article_id(link: str, title: str) -> str:
    return hashlib.sha256(f"{link}|{title}".encode()).hexdigest()[:16]


def _fetch_feed(source: str, url: str) -> list[dict]:
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return []

    d = feedparser.parse(resp.content)
    results = []
    for entry in d.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue
        summary = (entry.get("summary") or "")[:500]
        published = entry.get("published") or entry.get("updated") or ""
        results.append({
            "title": title,
            "link": link,
            "summary": summary,
            "published": published,
            "source": source,
        })
    return results


def collect_financial_blogs() -> list[dict]:
    conn = _ensure_db()
    raw: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_feed, src, url): src for src, url in FEEDS}
        for fut in as_completed(futures):
            raw.extend(fut.result())

    now_iso = datetime.now(timezone.utc).isoformat()
    new_articles: list[dict] = []

    for art in raw:
        aid = _article_id(art["link"], art["title"])
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
            continue
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (aid, art["link"], art["title"], art["source"], now_iso),
        )
        new_articles.append(art)

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    t0 = time.time()
    items = collect_financial_blogs()
    dt = time.time() - t0
    print(f"[financial_blogs] {len(items)} new items in {dt:.1f}s")
    for a in items[:5]:
        print(f"  [{a['source']}] {a['title'][:85]}")
