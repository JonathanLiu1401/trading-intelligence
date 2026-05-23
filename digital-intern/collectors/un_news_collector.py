"""UN News collector — macro-relevant feeds from news.un.org.

Covers economic development, climate policy, regional economic events, and
geopolitical disruptions that move commodities, currencies, and EM equities.
No API key needed; all feeds are public RSS 2.0.

Working feeds (verified 2026-05-23):
    un_econ_dev   — Economic Development topic
    un_climate    — Climate Change topic (carbon policy, energy transition)
    un_health     — Health topic (supply-chain / pharma relevance)
    un_americas   — Americas region
    un_africa     — Africa region (commodities / EM)
    un_europe     — Europe region

Same two-layer dedup as ecb_press_collector / fed_press_collector:
  1. data/seen_articles.db keyed by sha256(link||title)
  2. articles.db PRIMARY KEY = sha256(url||title) inside insert_batch
"""
import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
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

UN_FEEDS = [
    ("un_econ_dev", "https://news.un.org/feed/subscribe/en/news/topic/economic-development/feed/rss.xml"),
    ("un_climate",  "https://news.un.org/feed/subscribe/en/news/topic/climate-change/feed/rss.xml"),
    ("un_health",   "https://news.un.org/feed/subscribe/en/news/topic/health/feed/rss.xml"),
    ("un_americas", "https://news.un.org/feed/subscribe/en/news/region/americas/feed/rss.xml"),
    ("un_africa",   "https://news.un.org/feed/subscribe/en/news/region/africa/feed/rss.xml"),
    ("un_europe",   "https://news.un.org/feed/subscribe/en/news/region/europe/feed/rss.xml"),
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
    return hashlib.sha256(f"{link}||{title}".encode("utf-8")).hexdigest()


def _fetch_feed(name: str, url: str) -> list[dict]:
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA})
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as e:
        print(f"[un_news] Error fetching {name}: {e}")
        return []
    out: list[dict] = []
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


def collect_un_news() -> list[dict]:
    """Collect deduplicated UN News items across all topic/region feeds.

    Returns list of {title, link, summary, published, source} dicts.
    """
    conn = _ensure_db()
    new_articles: list[dict] = []
    seen_in_run: set = set()

    with ThreadPoolExecutor(max_workers=len(UN_FEEDS)) as pool:
        futures = {pool.submit(_fetch_feed, name, url): name for name, url in UN_FEEDS}
        for fut in as_completed(futures):
            for art in fut.result():
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
                        (aid, link, title, art["source"],
                         datetime.now(timezone.utc).isoformat()),
                    )
                except sqlite3.Error as e:
                    print(f"[un_news] dedup row skipped: {e}")
                    continue
                new_articles.append(art)

    conn.commit()
    conn.close()
    return new_articles


collect = collect_un_news


if __name__ == "__main__":
    print("=== UN News feeds (live fetch) ===")
    eg_line = None
    totals: dict[str, int] = {}
    for name, url in UN_FEEDS:
        raw = _fetch_feed(name, url)
        totals[name] = len(raw)
        print(f"  {name:16s} {len(raw):3d} entries")
        if eg_line is None and raw:
            eg_line = f"{name}: {raw[0]['title']}"

    items = collect_un_news()
    inserted = 0
    if items:
        from storage.article_store import ArticleStore
        store = ArticleStore()
        inserted = store.insert_batch(items)

    print("\n=== Summary ===")
    print(f"New deduped items built  : {len(items)}")
    print(f"Inserted into articles.db: {inserted}")
    if eg_line:
        print(f"DISCORD_EG: {eg_line}")
    for a in items[:8]:
        print(f"  + [{a['source']:16s}] {a['title']}")
