"""Global financial regulators RSS collector.

Pulls high-quality, low-noise research and regulatory signals from four
international financial-stability bodies whose output frequently moves
credit markets and shapes central-bank reaction functions:

    FSB   — Financial Stability Board: systemic-risk warnings, global
            financial-stability reports, crypto/private-credit/NBFI regs.
    FCA   — Financial Conduct Authority (UK): consumer finance, market
            structure, tokenisation, and supervisory policy.
    FEDS  — Federal Reserve FEDS Notes: working-level Fed research notes;
            early read on areas the Board is studying (liquidity, credit).
    FEDS-WP — Federal Reserve Working Papers: formal academic papers that
            often pre-signal future policy thinking.

None of these four feeds are covered by the existing fed_press_collector
(which pulls press_monetary / speeches / testimony) or by the main RSS
sources.json (which is filtered to headline-news feeds).

Same two-layer dedup as every other collector:
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

# (source_name, feed_url). source_name becomes the `source` column.
REGULATOR_FEEDS = [
    ("fsb_news",     "https://www.fsb.org/feed/"),
    ("fca_news",     "https://www.fca.org.uk/news/rss.xml"),
    ("fed_feds_notes", "https://www.federalreserve.gov/feeds/feds_notes.xml"),
    ("fed_working_papers", "https://www.federalreserve.gov/feeds/working_papers.xml"),
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
    """Fetch and parse one regulator RSS feed. Any error returns [] so one
    bad feed never aborts the whole pass."""
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA})
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as e:
        print(f"[global_regulators] Error fetching {name}: {e}")
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


def collect_global_regulators() -> list[dict]:
    """Collect deduplicated items from global financial regulator feeds.

    Returns [{title, link, summary, published, source}, ...].
    Caller (daemon _ingest or __main__) inserts via ArticleStore.insert_batch.
    """
    conn = _ensure_db()
    new_articles: list[dict] = []
    seen_in_run: set = set()

    for name, url in REGULATOR_FEEDS:
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
                print(f"[global_regulators] dedup row skipped ({name}): {e}")
                continue
            new_articles.append(art)

    conn.commit()
    conn.close()
    return new_articles


collect = collect_global_regulators


if __name__ == "__main__":
    print("=== Global Financial Regulators (live fetch) ===")
    eg_line = None
    for _name, _url in REGULATOR_FEEDS:
        raw = _fetch_feed(_name, _url)
        print(f"  {_name:24s} {len(raw):3d} entries")
        for e in raw[:3]:
            print(f"    - {e['title'][:80]}")
        if eg_line is None and raw:
            eg_line = f"{_name}: {raw[0]['title']}"

    items = collect_global_regulators()
    inserted = 0
    if items:
        from storage.article_store import ArticleStore
        store = ArticleStore()
        inserted = store.insert_batch(items)

    print("\n=== Summary ===")
    print(f"New deduped items      : {len(items)}")
    print(f"Inserted into articles.db: {inserted}")
    if eg_line:
        print(f"DISCORD_EG: {eg_line}")
    for a in items[:8]:
        print(f"  + [{a['source']}] {a['title']}")
