"""RSS collector with SQLite-based deduplication. Parallel fetch across feeds."""
import json
import os
import sqlite3
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

import feedparser

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCES_PATH = BASE_DIR / "config" / "sources.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

MAX_WORKERS = 24  # parallel feed fetches


def _ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY,
            link TEXT,
            title TEXT,
            source TEXT,
            first_seen TEXT
        )
        """
    )
    conn.commit()
    return conn


def _article_id(link: str, title: str) -> str:
    return hashlib.sha256(f"{link}||{title}".encode("utf-8")).hexdigest()


def _load_sources():
    with open(SOURCES_PATH, "r") as f:
        return json.load(f)


def _fetch_feed(feed: dict) -> list:
    name = feed.get("name", "unknown")
    url = feed.get("url")
    if not url:
        return []
    try:
        parsed = feedparser.parse(url)
    except Exception as e:
        print(f"[rss_collector] Error fetching {name}: {e}")
        return []
    out: list = []
    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue
        summary = entry.get("summary") or entry.get("description") or ""
        published = entry.get("published") or entry.get("updated") or ""
        out.append({
            "title": title,
            "link": link,
            "summary": summary,
            "published": published,
            "source": name,
        })
    return out


def collect_rss():
    """Collect deduplicated articles from configured RSS feeds (parallel).

    Returns a list of dicts: {title, link, summary, published, source}.
    """
    sources = _load_sources()
    feeds = sources.get("rss_feeds", [])

    # Fetch in parallel — feedparser is HTTP-bound and benefits from threads.
    fetched: list[list] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_feed, f): f for f in feeds}
        for future in as_completed(futures):
            try:
                fetched.append(future.result())
            except Exception as e:
                print(f"[rss_collector] worker error: {e}")

    # Dedup in a single SQLite pass after parallel I/O.
    conn = _ensure_db()
    new_articles: list = []
    seen_in_run: set = set()
    for batch in fetched:
        for art in batch:
            link, title = art["link"], art["title"]
            aid = _article_id(link, title)
            if aid in seen_in_run:
                continue
            seen_in_run.add(aid)
            if conn.execute(
                "SELECT 1 FROM seen_articles WHERE id = ?", (aid,)
            ).fetchone():
                continue
            new_articles.append(art)
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles "
                "(id, link, title, source, first_seen) VALUES (?, ?, ?, ?, ?)",
                (aid, link, title, art["source"], datetime.utcnow().isoformat()),
            )

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    items = collect_rss()
    print(f"Collected {len(items)} new articles")
    for a in items[:5]:
        print(f" - [{a['source']}] {a['title']}")
