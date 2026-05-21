"""White House executive orders, proclamations, and press briefings collector.

The White House publishes RSS feeds for all presidential actions and press
briefings. These are among the highest-signal streams for market-moving policy:
tariffs, sanctions, executive orders, trade proclamations, and regulatory
direction all land here before they hit financial media.

Four feeds, all public, no API key:
    whitehouse_eo          Executive orders
    whitehouse_proc        Proclamations (tariffs, trade actions)
    whitehouse_briefings   Press briefings and statements
    whitehouse_actions     All presidential actions (catch-all)

Like other collectors, returns standard {title, link, summary, published, source}
dicts for ArticleStore.insert_batch.
"""
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import feedparser

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

WH_FEEDS = {
    "whitehouse_eo":        "https://www.whitehouse.gov/presidential-actions/executive-orders/feed/",
    "whitehouse_proc":      "https://www.whitehouse.gov/presidential-actions/proclamations/feed/",
    "whitehouse_briefings": "https://www.whitehouse.gov/briefings-statements/feed/",
    "whitehouse_actions":   "https://www.whitehouse.gov/presidential-actions/feed/",
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
    """Fetch one WH RSS feed. Any error returns [] so one bad feed
    never aborts the whole pass."""
    try:
        feed = feedparser.parse(
            url,
            agent=_UA,
            request_headers={"Accept": "application/rss+xml, application/xml, */*"},
        )
        if feed.bozo and not feed.entries:
            return []
        articles = []
        for entry in feed.entries:
            link  = getattr(entry, "link", "") or ""
            title = getattr(entry, "title", "") or ""
            if not link or not title:
                continue
            summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
            # Strip HTML tags from summary (simple approach)
            import re
            summary = re.sub(r"<[^>]+>", " ", summary).strip()
            summary = " ".join(summary.split())[:500]

            pub_struct = getattr(entry, "published_parsed", None)
            if pub_struct:
                published = datetime(*pub_struct[:6], tzinfo=timezone.utc).isoformat()
            else:
                published = datetime.now(timezone.utc).isoformat()

            articles.append({
                "title":     title,
                "link":      link,
                "summary":   summary,
                "published": published,
                "source":    name,
            })
        return articles
    except Exception as e:
        print(f"[whitehouse_collector] {name} fetch error: {e}")
        return []


def collect_whitehouse() -> list[dict]:
    """Fetch all WH feeds, deduplicate via seen_articles.db, return new articles."""
    conn = _ensure_db()
    new_articles: list[dict] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for name, url in WH_FEEDS.items():
        fetched = _fetch_feed(name, url)
        for art in fetched:
            aid = _article_id(art["link"], art["title"])
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO seen_articles(id, link, title, source, first_seen) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (aid, art["link"], art["title"], name, now_iso),
                )
                if conn.execute(
                    "SELECT changes()"
                ).fetchone()[0]:
                    new_articles.append(art)
            except sqlite3.Error:
                pass
    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    articles = collect_whitehouse()
    print(f"[whitehouse_collector] {len(articles)} new articles")
    for a in articles[:10]:
        print(f"  [{a['source']}] {a['title']}")
        print(f"    {a['link']}")
        print(f"    {a['summary'][:120]}...")
        print()
    if not articles:
        print("  (no new articles — all already seen or feeds returned nothing)")
        # Force print some current feed entries for verification
        print("\n[verification] Checking feed freshness:")
        import feedparser as fp
        for name, url in WH_FEEDS.items():
            f = fp.parse(url, agent=_UA)
            entries = f.entries[:2]
            print(f"  {name}: {len(f.entries)} entries, latest: {entries[0].title if entries else 'none'}")
