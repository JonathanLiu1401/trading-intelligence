"""Commodity Futures Trading Commission press / enforcement collector.

Sits alongside ``fed_press_collector`` / ``ecb_press_collector`` /
``boj_press_collector`` / ``boe_press_collector`` (central-bank wires) but
covers the US derivatives regulator's wire — a different, complementary feed
that none of the existing collectors capture:

  * Enforcement actions against banks, exchanges, swap dealers, and crypto
    venues (regularly move the named ticker and adjacent sector — e.g. a
    CFTC fine against a US-listed crypto exchange tags COIN / HOOD).
  * Position-limit rules and swap-rule changes (move energy / ag / rates
    futures and the leveraged-ETF complexes built on top of them).
  * Crypto-asset oversight statements (move BTC / ETH and crypto-adjacent
    equities directly via the daemon's leveraged-ETF book lens).
  * Weekly Commitments-of-Traders (COT) release pointers — institutional
    positioning data that the briefing already aggregates context around.

CFTC publishes a public press-release RSS feed at
``https://www.cftc.gov/rss/PressReleases.xml`` — no API key, no auth.

Like every other collector, ``collect_cftc_press()`` returns the standard
``{title, link, summary, published, source}`` dicts and the daemon's
``_ingest()`` hands them to ``ArticleStore.insert_batch`` — the canonical
articles.db insert path.

Two dedup layers, matching rss_collector / fed_press_collector / boj_press_collector:
  1. shared ``data/seen_articles.db`` (WAL, busy_timeout=30000) keyed by
     sha256(link||title) so a re-run never re-emits the same release.
  2. ``articles.db`` PRIMARY KEY = sha256(url||title) inside insert_batch.
"""
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

# name -> feed URL. The `name` becomes the article `source` column. Short,
# grep-friendly, parallel to fed_* / ecb_* / boj_* / boe_* so dashboard /
# briefing code that already groups on "fed_" / "ecb_" can trivially also
# group on "cftc_".
CFTC_FEEDS = {
    "cftc_press": "https://www.cftc.gov/RSS/RSSGP/rssgp.xml",
    "cftc_enforcement": "https://www.cftc.gov/RSS/RSSENF/rssenf.xml",
}

FETCH_TIMEOUT = 12  # seconds; bounds a slow/dead feed so it can't starve the worker
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Hardened seen_articles.db connection — mirrors rss_collector._ensure_db /
    # fed_press_collector._ensure_db / boj_press_collector._ensure_db. Many
    # collectors share this one file; SQLite's default busy_timeout=0 turns any
    # transient cross-writer lock into an immediate OperationalError that aborts
    # the pass and drops the fetched batch. WAL + 30s timeout lets the write
    # wait out the contention.
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
    """Fetch and parse one CFTC RSS feed. Any error returns [] so one bad feed
    never aborts the whole pass (mirrors rss_collector._fetch_feed)."""
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA})
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as e:
        print(f"[cftc_press_collector] Error fetching {name}: {e}")
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


def collect_cftc_press() -> list[dict]:
    """Collect deduplicated CFTC press / enforcement / rule items.

    Returns a list of dicts: {title, link, summary, published, source}.
    Consistent with collect_fed_press / collect_ecb_press / collect_boj_press /
    collect_rss — the caller (daemon _ingest or __main__) inserts via
    ArticleStore.insert_batch.
    """
    conn = _ensure_db()
    new_articles: list[dict] = []
    seen_in_run: set = set()

    for name, url in CFTC_FEEDS.items():
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
                print(f"[cftc_press_collector] dedup row skipped ({name}): {e}")
                continue
            new_articles.append(art)

    conn.commit()
    conn.close()
    return new_articles


# Alias matching the collect_<source>() convention used across collectors/.
collect = collect_cftc_press


if __name__ == "__main__":
    # Live fetch proves the public feed returned real entries (not placeholders),
    # then collect (deduped) and insert via the canonical shared article store.
    print("=== CFTC feeds (live fetch) ===")
    eg_line = None
    for _name, _url in CFTC_FEEDS.items():
        raw = _fetch_feed(_name, _url)
        print(f"  {_name:14s} {len(raw):3d} entries")
        if eg_line is None and raw:
            eg_line = f"{_name}: {raw[0]['title']}"

    items = collect_cftc_press()
    inserted = 0
    if items:
        from storage.article_store import ArticleStore  # canonical insert path
        store = ArticleStore()
        inserted = store.insert_batch(items)

    print("\n=== Summary ===")
    print(f"New deduped items built : {len(items)}")
    print(f"Inserted into articles.db : {inserted}")
    if eg_line:
        print(f"DISCORD_EG: {eg_line}")
    for a in items[:8]:
        print(f"  + [{a['source']}] {a['title']}")
