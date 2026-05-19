"""U.S. Energy Information Administration (EIA) news collector.

EIA's Today in Energy and Weekly Petroleum Status Report drive energy
equities (XOM, CVX, OXY), refiners (VLO, MPC), and crude/gasoline futures.
Weekly oil & gas inventory surprises routinely move the sector 2-5% on
release. No existing collector covers EIA — fred pulls macro series, not
the agency's own news/release wire.

Public RSS, no API key:

    eia_today        Today in Energy daily explainer articles.
    eia_press        EIA press releases (inventory reports, STEO,
                     Annual Energy Outlook, oil/gas/coal/electricity).

Standard collector contract: returns {title, link, summary, published,
source} dicts; daemon `_ingest()` runs them through `ArticleStore.insert_batch`.

Two dedup layers, matching boe_press_collector / fed_press_collector:
  1. shared `data/seen_articles.db` (WAL, busy_timeout=30000) keyed by
     sha256(link||title).
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

EIA_FEEDS = {
    "eia_today": "https://www.eia.gov/rss/todayinenergy.xml",
    "eia_press": "https://www.eia.gov/rss/press_rss.xml",
}

FETCH_TIMEOUT = 12
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


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
        print(f"[eia_collector] Error fetching {name}: {e}")
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


def collect_eia() -> list[dict]:
    conn = _ensure_db()
    new_articles: list[dict] = []
    seen_in_run: set = set()

    for name, url in EIA_FEEDS.items():
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
                print(f"[eia_collector] dedup row skipped ({name}): {e}")
                continue
            new_articles.append(art)

    conn.commit()
    conn.close()
    return new_articles


collect = collect_eia


if __name__ == "__main__":
    print("=== EIA feeds (live fetch) ===")
    eg_line = None
    for _name, _url in EIA_FEEDS.items():
        raw = _fetch_feed(_name, _url)
        print(f"  {_name:18s} {len(raw):3d} entries")
        if eg_line is None and raw:
            eg_line = f"{_name}: {raw[0]['title']}"

    items = collect_eia()
    inserted = 0
    if items:
        from storage.article_store import ArticleStore
        store = ArticleStore()
        inserted = store.insert_batch(items)

    print("\n=== Summary ===")
    print(f"New deduped items built : {len(items)}")
    print(f"Inserted into articles.db : {inserted}")
    if eg_line:
        print(f"DISCORD_EG: {eg_line}")
    for a in items[:8]:
        print(f"  + [{a['source']}] {a['title']}")
