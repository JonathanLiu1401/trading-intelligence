"""TrendForce press center collector — semiconductor & memory market intelligence.

TrendForce is the leading market research firm for DRAM, NAND, foundry, and
display supply chains. Their public press center publishes free research notes
on memory pricing, supply/demand, fab utilisation, and quarterly revenue data
— high-signal for MU, NVDA, AMD, ASML, LRCX, TSM, and related semis.

No API key. Scrapes the press center listing for article URLs and extracts
title + description from og: meta tags (no JS rendering needed). Rate-limits
to one article fetch per pass with a per-article cooldown.

Dedup: seen_articles.db keyed by article URL (same scheme as finviz_collector).
"""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

log = logging.getLogger("trendforce_collector")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

BASE_URL = "https://www.trendforce.com"
LISTING_URL = f"{BASE_URL}/presscenter/"
SOURCE = "TrendForce"
REQUEST_TIMEOUT = 12
SLEEP_BETWEEN_FETCHES = 1.5   # polite crawl rate

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# URL pattern for press release articles (YYYYMMDD-NNNNN.html)
_ARTICLE_RE = re.compile(r"/presscenter/news/(\d{8})-\d+\.html$")

# High-signal keywords — emit even for non-portfolio companies when these appear
_PRIORITY_KEYWORDS = {
    "dram", "hbm", "nand", "memory", "ddr", "lpddr", "gddr",
    "tsmc", "micron", "samsung", "sk hynix", "kioxia",
    "foundry", "wafer", "capacity", "bit shipment", "spot price",
    "amd", "nvidia", "ai chip", "hpc", "data center", "server",
    "shortage", "oversupply", "inventory", "pricing",
}


def _article_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY,
            link TEXT,
            title TEXT,
            source TEXT,
            first_seen TEXT
        )
    """)
    conn.commit()


def _is_seen(conn: sqlite3.Connection, url: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM seen_articles WHERE id=?", (_article_id(url),)
    ).fetchone()
    return row is not None


def _mark_seen(conn: sqlite3.Connection, url: str, title: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
        (_article_id(url), url, title, SOURCE, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _fetch_listing() -> list[dict]:
    """Fetch the press center listing and return (url, date_str) pairs."""
    try:
        r = requests.get(LISTING_URL, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        log.warning("TrendForce listing fetch failed: %s", e)
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    articles = []
    seen_urls: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        m = _ARTICLE_RE.search(href)
        if not m:
            continue
        url = urljoin(BASE_URL, href)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        date_str = m.group(1)  # YYYYMMDD
        articles.append({"url": url, "date_str": date_str})

    return articles


def _fetch_article(url: str, date_str: str) -> dict | None:
    """Fetch a single article and return a collector dict, or None on failure."""
    try:
        r = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        log.warning("TrendForce article fetch failed %s: %s", url, e)
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    def _meta(prop: str) -> str:
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        return (tag.get("content", "") if tag else "").strip()

    title = _meta("og:title") or _meta("twitter:title")
    summary = _meta("og:description") or _meta("description") or _meta("twitter:description")

    if not title:
        return None

    # Parse date from URL string YYYYMMDD
    try:
        published = datetime(
            int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]),
            tzinfo=timezone.utc,
        ).isoformat()
    except (ValueError, IndexError):
        published = datetime.now(timezone.utc).isoformat()

    return {
        "title": title,
        "link": url,
        "summary": summary,
        "published": published,
        "source": SOURCE,
    }


def _is_relevant(article: dict) -> bool:
    """Keep articles about semiconductor/memory topics."""
    text = (article.get("title", "") + " " + article.get("summary", "")).lower()
    return any(kw in text for kw in _PRIORITY_KEYWORDS)


def collect_trendforce() -> list[dict]:
    """Main entry point — returns list of new article dicts."""
    conn = sqlite3.connect(str(DB_PATH))
    _ensure_db(conn)

    listing = _fetch_listing()
    if not listing:
        return []

    results: list[dict] = []
    fetched = 0

    for item in listing:
        url = item["url"]
        if _is_seen(conn, url):
            continue

        time.sleep(SLEEP_BETWEEN_FETCHES)
        article = _fetch_article(url, item["date_str"])
        fetched += 1

        if article and _is_relevant(article):
            _mark_seen(conn, url, article["title"])
            results.append(article)
        elif article:
            # Mark seen so we don't re-fetch irrelevant articles
            _mark_seen(conn, url, article["title"])

    conn.close()
    log.info("TrendForce: fetched %d new, kept %d relevant", fetched, len(results))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    articles = collect_trendforce()
    print(f"\nTrendForce — {len(articles)} new relevant articles:\n")
    for a in articles:
        print(f"  [{a['published'][:10]}] {a['title']}")
        print(f"  {a['summary'][:120]}...")
        print(f"  {a['link']}")
        print()
