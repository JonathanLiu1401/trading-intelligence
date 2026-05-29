"""Quality investment research and finance blog collector.

Eight high-signal blogs not covered by the main RSS config:
  CMT Association         — technical analysis research (high volume)
  Real Investment Advice  — macro/technical commentary
  Lyn Alden               — macro research & balance-of-payments analysis
  The Felder Report       — contrarian macro
  A Wealth of Common Sense — behavioral finance (Ben Carlson)
  Alpha Architect         — quantitative factor research
  Humble Dollar           — long-term personal finance
  Philosophical Economics — deep macro valuation essays

Parallel fetch; deduplicates via seen_articles.db.
"""
import hashlib
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

REQUEST_TIMEOUT = 12
MAX_WORKERS = 8

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

FEEDS = [
    ("CMT Association",         "https://cmtassociation.org/feed/"),
    ("Real Investment Advice",  "https://realinvestmentadvice.com/feed/"),
    ("Lyn Alden",               "https://www.lynalden.com/feed/"),
    ("The Felder Report",       "https://thefelderreport.com/feed/"),
    ("A Wealth of Common Sense","https://awealthofcommonsense.com/feed/"),
    ("Alpha Architect",         "https://alphaarchitect.com/feed/"),
    ("Humble Dollar",           "https://humbledollar.com/feed/"),
    ("Philosophical Economics", "https://www.philosophicaleconomics.com/feed/"),
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
    return hashlib.sha256(f"{link}|{title}".encode()).hexdigest()[:16]


def _fetch_feed(source: str, url: str) -> list[dict]:
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[investment_research_blogs] fetch error {source}: {exc}")
        return []

    d = feedparser.parse(resp.content)
    results = []
    for entry in d.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue
        summary = (entry.get("summary") or "")[:600]
        published = entry.get("published") or entry.get("updated") or ""
        results.append({
            "title": title,
            "link": link,
            "summary": summary,
            "published": published,
            "source": source,
        })
    return results


def collect_investment_research_blogs() -> list[dict]:
    conn = _ensure_db()
    raw: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_feed, src, url): src for src, url in FEEDS}
        for fut in as_completed(futures):
            raw.extend(fut.result())

    now_iso = datetime.now(timezone.utc).isoformat()
    new_articles: list[dict] = []

    for art in raw:
        aid = _article_id(art["link"], art["title"])
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
            continue
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (aid, art["link"], art["title"], art["source"], now_iso),
        )
        new_articles.append(art)

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    t0 = time.time()
    items = collect_investment_research_blogs()
    dt = time.time() - t0
    print(f"[investment_research_blogs] {len(items)} new items in {dt:.1f}s")
    for a in items[:10]:
        print(f"  [{a['source']}] {a['title'][:85]}")
