"""FDA press releases + MedWatch safety alerts RSS collector.

Biotech catalyst gap: drug/device approvals, recalls, safety alerts.
Two endpoints:
  * Press Releases — drug approvals, voucher grants, policy actions
  * MedWatch — safety alerts, recalls, withdrawals

Both are zero-auth, polite User-Agent, dedup via shared seen_articles.db.
"""
import hashlib
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

FEEDS = {
    "FDA/PressReleases": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
    "FDA/MedWatch":      "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/medwatch/rss.xml",
}
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


def collect_fda(max_per_feed: int = 40) -> list:
    new_articles: list = []
    conn = _ensure_db()
    seen_in_run: set = set()

    for source, url in FEEDS.items():
        try:
            parsed = feedparser.parse(url, agent=USER_AGENT)
        except Exception as e:
            print(f"[fda] {source} fetch error: {e}")
            continue
        if getattr(parsed, "bozo", 0) and not parsed.entries:
            print(f"[fda] {source} bozo parse, skipping")
            continue

        for entry in parsed.entries[:max_per_feed]:
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

            art = {
                "title": title,
                "link": link,
                "summary": entry.get("summary") or entry.get("description") or "",
                "published": entry.get("published") or entry.get("updated") or "",
                "source": source,
            }
            new_articles.append(art)
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                "VALUES (?, ?, ?, ?, ?)",
                (aid, link, title, source, datetime.now(timezone.utc).isoformat()),
            )

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    t0 = time.time()
    items = collect_fda()
    dt = time.time() - t0
    print(f"[fda] {len(items)} new items in {dt:.1f}s")
    for a in items[:15]:
        print(f"  [{a['source']}] {a['title'][:90]}")
        print(f"     {a['link'][:110]}")
