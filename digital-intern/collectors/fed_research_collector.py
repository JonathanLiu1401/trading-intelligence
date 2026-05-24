"""Federal Reserve research papers and full press-release collector.

Extends fed_press_collector (which covers monetary policy / speeches / testimony)
with three additional public Fed feeds that are NOT in the existing collector:

    press_all.xml   — every category of Fed press release, including Beige
                      Book publication notices, enforcement orders, and
                      regulatory approval that can move bank stocks.
    feds.xml        — Finance and Economics Discussion Series working papers:
                      the Fed's internal economic research that previews
                      methodological shifts and future policy thinking.
    ifdp.xml        — International Finance Discussion Papers: covers global
                      capital flows, FX models, EM contagion — leading
                      indicators for USD moves and cross-border risk.

All three feeds are public, no API key required.

Dedup layers identical to fed_press_collector:
  1. data/seen_articles.db keyed by sha256(link||title)
  2. articles.db PRIMARY KEY = sha256(url||title) inside insert_batch.
"""
import hashlib
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

FED_RESEARCH_FEEDS = {
    "fed_all_press": "https://www.federalreserve.gov/feeds/press_all.xml",
    "fed_feds_papers": "https://www.federalreserve.gov/feeds/feds.xml",
    "fed_ifdp_papers": "https://www.federalreserve.gov/feeds/ifdp.xml",
}


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
    """Fetch and parse one Fed RSS feed. Any error returns [] — one bad feed
    never aborts the whole pass."""
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA})
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as e:
        print(f"[fed_research_collector] Error fetching {name}: {e}")
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


def collect_fed_research() -> list[dict]:
    """Collect deduplicated Fed research papers and full press-release items.

    Returns list of dicts: {title, link, summary, published, source}.
    Compatible with the standard collect_<x>() contract used by the daemon.
    """
    conn = _ensure_db()
    new_articles: list[dict] = []
    seen_in_run: set = set()

    for name, url in FED_RESEARCH_FEEDS.items():
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
                print(f"[fed_research_collector] dedup row skipped ({name}): {e}")
                continue
            new_articles.append(art)

    conn.commit()
    conn.close()
    return new_articles


collect = collect_fed_research


if __name__ == "__main__":
    print("=== Fed Research + Full Press feeds (live fetch) ===")
    eg_line = None
    for _name, _url in FED_RESEARCH_FEEDS.items():
        raw = _fetch_feed(_name, _url)
        print(f"  {_name:20s} {len(raw):3d} entries")
        if raw:
            print(f"    sample: {raw[0]['title'][:70]}")
            if eg_line is None:
                eg_line = f"{_name}: {raw[0]['title']}"

    items = collect_fed_research()
    inserted = 0
    if items:
        from storage.article_store import ArticleStore
        store = ArticleStore()
        inserted = store.insert_batch(items)

    print("\n=== Summary ===")
    print(f"New deduped items     : {len(items)}")
    print(f"Inserted into articles.db : {inserted}")
    if eg_line:
        print(f"DISCORD_EG: {eg_line}")
    for a in items[:8]:
        print(f"  + [{a['source']}] {a['title']}")
