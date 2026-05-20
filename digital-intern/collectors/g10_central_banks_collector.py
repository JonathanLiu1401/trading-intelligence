"""G10 central banks collector — Bank of Canada and Reserve Bank of Australia.

Fifth and sixth legs of the central-bank set after:
  fed_press_collector (USD wire), ecb_press_collector (EUR wire),
  boj_press_collector (JPY wire), boe_press_collector (GBP wire).

Adds coverage for:
  BoC — Bank of Canada (CAD wire): rate decisions, MPR, speeches.
        CAD is a commodity currency; BoC policy drives oil-correlated
        equity sectors and the USD/CAD pair that affects US multinationals.
  RBA — Reserve Bank of Australia (AUD wire): rate decisions, SMP,
        speeches. AUD is the proxy for China's commodity demand; RBA
        decisions move mining stocks, iron ore, and copper.

Public RSS, no API key.

Two dedup layers matching boe_press_collector:
  1. shared data/seen_articles.db keyed by sha256(link||title)
  2. articles.db PRIMARY KEY = sha256(url||title) inside insert_batch
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

# source_name → feed_url. source_name becomes the `source` column in articles.db.
# Prefixes follow the cb_ naming convention so dashboard grouping works.
CB_FEEDS = {
    "boc_press":     "https://www.bankofcanada.ca/category/press-releases/feed/",
    "boc_speeches":  "https://www.bankofcanada.ca/category/publications/speeches/feed/",
    "boc_news":      "https://www.bankofcanada.ca/feed/",
    "rba_releases":  "https://www.rba.gov.au/rss/rss-cb-media-releases.xml",
    "rba_speeches":  "https://www.rba.gov.au/rss/rss-cb-speeches.xml",
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
    """Fetch and parse one central bank RSS feed. Any error returns [] so one
    bad feed never aborts the whole pass."""
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA})
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as e:
        print(f"[g10_cb] Error fetching {name}: {e}")
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


def collect_g10_central_banks() -> list[dict]:
    """Collect deduplicated BoC + RBA central bank communications.

    Returns a list of dicts: {title, link, summary, published, source}.
    Consistent with collect_boe_press / collect_boj_press — the caller
    (daemon _ingest or __main__) inserts via ArticleStore.insert_batch.
    """
    conn = _ensure_db()
    new_articles: list[dict] = []
    seen_in_run: set = set()

    for name, url in CB_FEEDS.items():
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
                print(f"[g10_cb] dedup row skipped ({name}): {e}")
                continue
            new_articles.append(art)

    conn.commit()
    conn.close()
    return new_articles


collect = collect_g10_central_banks


if __name__ == "__main__":
    print("=== G10 Central Banks (BoC + RBA) — live fetch ===")
    eg_line = None
    for _name, _url in CB_FEEDS.items():
        raw = _fetch_feed(_name, _url)
        print(f"  {_name:16s} {len(raw):3d} entries")
        if eg_line is None and raw:
            eg_line = f"{_name}: {raw[0]['title']}"

    items = collect_g10_central_banks()
    inserted = 0
    if items:
        from storage.article_store import ArticleStore
        store = ArticleStore()
        inserted = store.insert_batch(items)

    print("\n=== Summary ===")
    print(f"New deduped items: {len(items)}")
    print(f"Inserted into articles.db: {inserted}")
    if eg_line:
        print(f"DISCORD_EG: {eg_line}")
    for a in items[:8]:
        print(f"  + [{a['source']}] {a['title']}")
