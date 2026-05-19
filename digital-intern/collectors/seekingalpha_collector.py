"""Seeking Alpha breaking-news RSS collector.

Pulls https://seekingalpha.com/market_currents.xml — a high-signal breaking
news feed: M&A, earnings beats/misses, executive moves, reverse splits,
analyst calls. Each item carries one or more <category> tags whose value is
the ticker, which we extract into _tickers for downstream relevance scoring.

No API key. Polite User-Agent. Dedups via shared seen_articles.db.
"""
import hashlib
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

FEED_URL = "https://seekingalpha.com/market_currents.xml"
USER_AGENT = "Mozilla/5.0 (Digital Intern Daemon; contact@digital-intern.local)"


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


def _extract_tickers(entry) -> list[str]:
    tickers: list[str] = []
    for tag in entry.get("tags", []) or []:
        term = (tag.get("term") or "").strip().upper()
        if term and term.isalnum() and len(term) <= 6:
            tickers.append(term)
    return tickers


def collect_seekingalpha(max_items: int = 60) -> list:
    try:
        parsed = feedparser.parse(FEED_URL, agent=USER_AGENT)
    except Exception as e:
        print(f"[seekingalpha] fetch error: {e}")
        return []

    if getattr(parsed, "bozo", 0) and not parsed.entries:
        print(f"[seekingalpha] bozo parse, no entries")
        return []

    conn = _ensure_db()
    new_articles: list = []
    seen_in_run: set = set()

    for entry in parsed.entries[:max_items]:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue
        aid = _article_id(link, title)
        if aid in seen_in_run:
            continue
        seen_in_run.add(aid)
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
            continue

        tickers = _extract_tickers(entry)
        published = entry.get("published") or entry.get("updated") or ""
        art = {
            "title": title,
            "link": link,
            "summary": entry.get("summary") or "",
            "published": published,
            "source": "SeekingAlpha/BreakingNews",
            "_tickers": tickers,
        }
        new_articles.append(art)
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (aid, link, title, "SeekingAlpha", datetime.now(timezone.utc).isoformat()),
        )

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    t0 = time.time()
    items = collect_seekingalpha()
    dt = time.time() - t0
    print(f"[seekingalpha] {len(items)} new items in {dt:.1f}s")
    for a in items[:10]:
        tk = ",".join(a["_tickers"]) or "-"
        print(f"  [{tk}] {a['title'][:90]}")
        print(f"     {a['link'][:110]}")
