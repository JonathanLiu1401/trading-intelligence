"""RSS collector with SQLite-based deduplication."""
import json
import os
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime

import feedparser

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCES_PATH = BASE_DIR / "config" / "sources.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"


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


def _is_seen(conn, article_id: str) -> bool:
    cur = conn.execute("SELECT 1 FROM seen_articles WHERE id = ?", (article_id,))
    return cur.fetchone() is not None


def _mark_seen(conn, article_id: str, link: str, title: str, source: str):
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?, ?, ?, ?, ?)",
        (article_id, link, title, source, datetime.utcnow().isoformat()),
    )


def _load_sources():
    with open(SOURCES_PATH, "r") as f:
        return json.load(f)


def collect_rss():
    """Collect deduplicated articles from configured RSS feeds.

    Returns a list of dicts: {title, link, summary, published, source}.
    """
    sources = _load_sources()
    feeds = sources.get("rss_feeds", [])

    conn = _ensure_db()
    new_articles = []

    for feed in feeds:
        name = feed.get("name", "unknown")
        url = feed.get("url")
        if not url:
            continue
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries:
                title = (entry.get("title") or "").strip()
                link = (entry.get("link") or "").strip()
                if not title or not link:
                    continue
                aid = _article_id(link, title)
                if _is_seen(conn, aid):
                    continue
                summary = entry.get("summary") or entry.get("description") or ""
                published = entry.get("published") or entry.get("updated") or ""
                new_articles.append(
                    {
                        "title": title,
                        "link": link,
                        "summary": summary,
                        "published": published,
                        "source": name,
                    }
                )
                _mark_seen(conn, aid, link, title, name)
        except Exception as e:
            print(f"[rss_collector] Error fetching {name}: {e}")
            continue

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    items = collect_rss()
    print(f"Collected {len(items)} new articles")
    for a in items[:5]:
        print(f" - [{a['source']}] {a['title']}")
