"""Short-seller research report collector.

Monitors prominent short-seller research firms for new reports — these
are major market-moving events (stocks can drop 20-50%+ on publication).

Sources:
  RSS  — Hindenburg Research, Grizzly Research, Muddy Waters Research
  HTML — Spruce Point Capital (research page scrape)

Uses local seen_articles.db for dedup. Polls every ~30 min; short-seller
reports are rare but high-priority signals.
"""
from __future__ import annotations

import hashlib
import sqlite3
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup

log = logging.getLogger("short_seller_collector")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

REQUEST_TIMEOUT = 12
MAX_WORKERS = 6

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}

RSS_SOURCES = [
    ("Hindenburg Research", "https://hindenburgresearch.com/feed/"),
    ("Grizzly Research",    "https://grizzlyreports.com/feed/"),
    ("Muddy Waters Research", "https://muddywatersresearch.com/feed/"),
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


def _fetch_rss(source_name: str, feed_url: str) -> list[dict]:
    """Fetch and parse an RSS/Atom feed from a short-seller firm."""
    try:
        resp = requests.get(feed_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"[short_seller] {source_name} RSS fetch failed: {e}")
        return []

    feed = feedparser.parse(resp.content)
    articles = []
    for entry in feed.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        summary = (entry.get("summary") or entry.get("description") or "").strip()
        if len(summary) > 500:
            summary = summary[:500] + "…"
        published = entry.get("published") or entry.get("updated") or ""

        if not title or not link:
            continue

        articles.append({
            "title": title,
            "link": link,
            "summary": f"[Short-Seller Report] {source_name}: {summary}",
            "source": source_name,
            "published": published,
            "_relevance_score": 0.95,  # short-seller reports are always high-priority
        })

    return articles


def _fetch_spruce_point() -> list[dict]:
    """Scrape Spruce Point Capital's research page for short reports."""
    source_name = "Spruce Point Capital"
    base_url = "https://www.sprucepointcap.com"
    try:
        resp = requests.get(f"{base_url}/research", headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"[short_seller] {source_name} scrape failed: {e}")
        return []

    soup = BeautifulSoup(resp.content, "html.parser")
    articles = []
    seen_hrefs: set[str] = set()

    for link_tag in soup.find_all("a", href=True):
        href = link_tag["href"]
        text = link_tag.get_text(strip=True)

        if not href.startswith("/research/") or href == "/research":
            continue
        if href in seen_hrefs:
            continue
        if not text or text.lower() in {"view report", "read more", "learn more"}:
            continue

        seen_hrefs.add(href)
        full_url = f"{base_url}{href}"
        articles.append({
            "title": text,
            "link": full_url,
            "summary": f"[Short-Seller Report] {source_name} research report: {text}",
            "source": source_name,
            "published": "",
            "_relevance_score": 0.90,
        })

    return articles


def collect_short_sellers() -> list[dict]:
    """Collect short-seller research reports from all sources. Returns new-only articles."""
    conn = _ensure_db()
    all_articles: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for name, url in RSS_SOURCES:
            futures[pool.submit(_fetch_rss, name, url)] = name
        futures[pool.submit(_fetch_spruce_point)] = "Spruce Point Capital"

        for fut in as_completed(futures):
            try:
                all_articles.extend(fut.result())
            except Exception as e:
                log.warning(f"[short_seller] future error: {e}")

    new_articles: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()

    for art in all_articles:
        aid = _article_id(art["link"], art["title"])
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
            continue
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (aid, art["link"], art["title"], art["source"], now),
        )
        new_articles.append(art)

    conn.commit()
    conn.close()

    if new_articles:
        log.info(f"[short_seller] {len(new_articles)} new reports found")

    return new_articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = collect_short_sellers()
    print(f"\nFound {len(results)} new short-seller reports:")
    for r in results[:10]:
        print(f"  [{r['source']}] {r['title'][:80]}")
        print(f"    {r['link']}")
