"""Energy / oil / natural-gas / clean-energy news aggregator.

Patterned on fed_research_collector.py — pulls public RSS feeds from four
energy-trade publications and routes them through the standard collector
contract (dedup via data/seen_articles.db, list[dict] return shape).

Feeds:
    oilprice         — https://oilprice.com/rss/main
    rigzone          — https://www.rigzone.com/news/rss/rigzone_latest.aspx
    naturalgasintel  — https://www.naturalgasintel.com/rss/
    energymonitor    — https://www.energymonitor.ai/rss

Each emitted article is tagged with up to 5 tickers in the ``_tickers`` key
based on case-insensitive keyword matching against title + summary.
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

ENERGY_FEEDS = {
    "oilprice":        "https://oilprice.com/rss/main",
    "rigzone":         "https://www.rigzone.com/news/rss/rigzone_latest.aspx",
    "naturalgasintel": "https://www.naturalgasintel.com/rss/",
    "energymonitor":   "https://www.energymonitor.ai/rss",
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


def _tag_tickers(title: str, summary: str) -> list[str]:
    """Return up to 5 tickers based on keyword match. XLE is always included."""
    text = f"{title} {summary}".lower()
    tickers: list[str] = ["XLE"]

    if any(k in text for k in ("oil", "crude", "wti", "brent")):
        tickers.extend(["XOM", "CVX", "COP", "OXY"])
    if any(k in text for k in ("natural gas", "lng", "henry hub")):
        tickers.extend(["UNG", "LNG", "EQT", "AR"])
    if any(k in text for k in ("renewable", "solar", "wind", "transition", "clean energy")):
        tickers.extend(["TAN", "ICLN", "ENPH"])

    # Dedupe preserving order, then cap at 5
    seen: set = set()
    deduped: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped[:5]


def _fetch_feed(name: str, url: str) -> list[dict]:
    """Fetch and parse one energy RSS feed. Any error returns [] — one bad
    feed never aborts the whole pass."""
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA})
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as e:
        print(f"[energy_news_collector] Error fetching {name}: {e}")
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


def collect_energy_news() -> list[dict]:
    """Collect deduplicated energy-trade news items.

    Returns list of dicts: {title, link, summary, published, source, _tickers}.
    Compatible with the standard collect_<x>() contract used by the daemon.
    """
    conn = _ensure_db()
    new_articles: list[dict] = []
    seen_in_run: set = set()

    for name, url in ENERGY_FEEDS.items():
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
                print(f"[energy_news_collector] dedup row skipped ({name}): {e}")
                continue
            art["_tickers"] = _tag_tickers(title, art.get("summary", ""))
            new_articles.append(art)

    conn.commit()
    conn.close()
    return new_articles


collect = collect_energy_news


if __name__ == "__main__":
    print("=== Energy News feeds (live fetch) ===")
    eg_line = None
    for _name, _url in ENERGY_FEEDS.items():
        raw = _fetch_feed(_name, _url)
        print(f"  {_name:20s} {len(raw):3d} entries")
        if raw:
            print(f"    sample: {raw[0]['title'][:70]}")
            if eg_line is None:
                eg_line = f"{_name}: {raw[0]['title']}"

    items = collect_energy_news()
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
        print(f"  + [{a['source']}] {a['title']} tickers={a.get('_tickers')}")
