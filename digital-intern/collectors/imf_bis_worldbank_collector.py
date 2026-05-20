"""BIS (Bank for International Settlements) collector.

Pulls research, press releases, and central-bank speeches from BIS's public
RSS feeds. BIS content is a leading signal for systemic banking risk, Basel
regulatory changes, and global monetary-policy coordination — material that
moves credit spreads and EM currencies before it shows up in mainstream news.

Feeds (verified working 2026-05-20):
    bis_all       — all categories combined (papers, speeches, press)
    bis_press     — press releases only (CPMI, IOSCO, FSI, Basel Committee)
    bis_speeches  — central bankers' speeches (G10 + BIS governors)

Same two-layer dedup as every other collector:
  1. data/seen_articles.db keyed by sha256(link||title)
  2. articles.db PRIMARY KEY = sha256(url||title) inside insert_batch.
"""
import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

FETCH_TIMEOUT = 12
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

BIS_FEEDS = [
    ("bis_press",    "https://www.bis.org/doclist/all_pressrels.rss"),
    ("bis_speeches", "https://www.bis.org/doclist/cbspeeches.rss?paging_length=25"),
    ("bis_research", "https://www.bis.org/doclist/rss_all_categories.rss"),
]


def _article_id(link: str, title: str) -> str:
    return hashlib.sha256(f"{link}||{title}".encode()).hexdigest()


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


def _fetch_feed(source: str, url: str) -> list[dict]:
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA})
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception as e:
        print(f"[bis_collector] fetch failed ({source}): {e}")
        return []

    articles = []
    for entry in feed.entries:
        link = getattr(entry, "link", "") or ""
        title = getattr(entry, "title", "") or ""
        if not link or not title:
            continue

        summary = (
            getattr(entry, "summary", "")
            or getattr(entry, "description", "")
            or ""
        )
        summary = re.sub(r"<[^>]+>", " ", summary).strip()
        summary = " ".join(summary.split())[:500]

        published = (
            getattr(entry, "published", "")
            or getattr(entry, "updated", "")
            or ""
        )

        articles.append({
            "title": title.strip(),
            "link": link.strip(),
            "summary": summary,
            "published": published,
            "source": source,
        })
    return articles


def collect_imf_bis_worldbank() -> list[dict]:
    conn = _ensure_db()
    new_articles: list[dict] = []
    seen_in_run: set = set()

    for name, url in BIS_FEEDS:
        for art in _fetch_feed(name, url):
            link = art["link"]
            title = art["title"]
            aid = _article_id(link, title)
            if aid in seen_in_run:
                continue
            seen_in_run.add(aid)
            try:
                if conn.execute(
                    "SELECT 1 FROM seen_articles WHERE id = ?", (aid,)
                ).fetchone():
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO seen_articles "
                    "(id, link, title, source, first_seen) VALUES (?, ?, ?, ?, ?)",
                    (aid, link, title, name,
                     datetime.now(timezone.utc).isoformat()),
                )
            except sqlite3.Error as e:
                print(f"[bis_collector] dedup row skipped ({name}): {e}")
                continue
            new_articles.append(art)

    conn.commit()
    conn.close()
    return new_articles


collect = collect_imf_bis_worldbank


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))
    print("=== BIS Collector (live fetch) ===")
    eg_line = None
    for _name, _url in BIS_FEEDS:
        raw = _fetch_feed(_name, _url)
        print(f"  {_name:16s} {len(raw):3d} entries")
        for e in raw[:3]:
            print(f"    - {e['title'][:80]}")
        if eg_line is None and raw:
            eg_line = f"{_name}: {raw[0]['title']}"

    items = collect_imf_bis_worldbank()
    inserted = 0
    if items:
        from storage.article_store import ArticleStore
        store = ArticleStore()
        inserted = store.insert_batch(items)

    print("\n=== Summary ===")
    print(f"New deduped items        : {len(items)}")
    print(f"Inserted into articles.db: {inserted}")
    if eg_line:
        print(f"DISCORD_EG: {eg_line}")
    for a in items[:5]:
        print(f"  + [{a['source']}] {a['title']}")
