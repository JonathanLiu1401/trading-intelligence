"""PR Newswire press release collector.

Polls PR Newswire's free public RSS feeds for corporate press releases.
Extracts exchange-listed ticker symbols from titles/summaries so releases
surface in per-ticker urgency scoring and briefings.

Feeds polled:
  - news-releases-list  → all categories, ~20 items
  - financial-services  → finance-focused releases, ~20 items

Ticker extraction: matches ``(NYSE: XXX)``, ``(NASDAQ: XXX)`` etc. patterns
that PR Newswire embeds in release text.

Dedup: keyed by sha256(link||title) in seen_articles.db — same DB used by
other collectors; WAL + busy_timeout=30s handles write contention.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import feedparser

log = logging.getLogger("prnewswire_collector")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

REQUEST_TIMEOUT = 12
MAX_WORKERS = 4

# PR Newswire needs a real browser UA; default feedparser UA sometimes gets 403.
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

SOURCE_NAME = "PR Newswire"

# Public RSS feeds (no auth required; financial feed follows 301 redirect).
_FEEDS = [
    ("prnewswire_general", "https://www.prnewswire.com/rss/news-releases-list.rss"),
    ("prnewswire_financial", "https://www.prnewswire.com/rss/financial-services-latest-news.rss"),
]

# Matches (NYSE: ABC), (NASDAQ: ABC), (TSX: ABC) etc. embedded in PR text.
_TICKER_RE = re.compile(
    r"\((?:NYSE|NASDAQ|Nasdaq|TSX|AMEX|OTC|OTCQB|OTCQX|NYSEMKT|NYSE American):\s*"
    r"([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\)"
)

# Keywords that indicate high market relevance — used to boost urgency score.
_HIGH_RELEVANCE_KEYWORDS = [
    "earnings", "quarterly results", "financial results", "revenue", "guidance",
    "acquisition", "merger", "buyout", "takeover", "deal", "agreement",
    "dividend", "buyback", "repurchase", "share repurchase",
    "ipo", "public offering", "secondary offering", "debt offering",
    "fda", "drug approval", "clearance", "clinical trial",
    "bankruptcy", "chapter 11", "restructuring", "delisting",
    "sec", "investigation", "lawsuit", "settlement",
    "ceo", "cfo", "president", "appointment", "resignation", "departure",
    "partnership", "joint venture", "licensing",
]


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY,
            link TEXT,
            title TEXT,
            source TEXT,
            first_seen TEXT
        )"""
    )
    conn.commit()
    return conn


def _article_id(link: str, title: str) -> str:
    return hashlib.sha256(f"{link}||{title}".encode("utf-8")).hexdigest()[:16]


def _extract_tickers(text: str) -> list[str]:
    """Extract exchange-listed ticker symbols from release text."""
    return list(dict.fromkeys(_TICKER_RE.findall(text)))  # dedup, preserve order


def _is_high_relevance(title: str, summary: str) -> bool:
    combined = (title + " " + summary).lower()
    return any(kw in combined for kw in _HIGH_RELEVANCE_KEYWORDS)


def _fetch_feed(feed_id: str, url: str) -> list[dict]:
    """Fetch and parse a single RSS feed; return list of article dicts."""
    try:
        d = feedparser.parse(url, agent=_UA)
    except Exception as exc:
        log.warning("[prnewswire] fetch error %s: %s", feed_id, exc)
        return []

    if d.get("status", 200) not in (200, 301, 302):
        log.warning("[prnewswire] HTTP %s for %s", d.get("status"), feed_id)
        return []

    articles = []
    for entry in d.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        summary = (entry.get("summary") or entry.get("description") or "").strip()
        published = entry.get("published") or entry.get("updated") or ""

        if not title or not link:
            continue

        tickers = _extract_tickers(title + " " + summary)
        high_relevance = _is_high_relevance(title, summary)

        articles.append({
            "id": _article_id(link, title),
            "title": title,
            "link": link,
            "summary": summary[:500],
            "source": SOURCE_NAME,
            "feed_id": feed_id,
            "tickers": tickers,
            "high_relevance": high_relevance,
            "published": published,
            "first_seen": datetime.now(timezone.utc).isoformat(),
        })

    return articles


def collect() -> list[dict]:
    """Fetch all PR Newswire feeds, dedup, and insert new articles into DB."""
    conn = _ensure_db()
    t0 = time.monotonic()

    # Fetch feeds in parallel
    all_articles: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_feed, fid, url): fid for fid, url in _FEEDS}
        for fut in as_completed(futures):
            fid = futures[fut]
            try:
                articles = fut.result()
                all_articles.extend(articles)
                log.debug("[prnewswire] %s: %d items", fid, len(articles))
            except Exception as exc:
                log.warning("[prnewswire] error in %s: %s", fid, exc)

    # Dedup across feeds (same release appears in both general + financial)
    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for a in all_articles:
        if a["id"] not in seen_ids:
            seen_ids.add(a["id"])
            deduped.append(a)

    if not deduped:
        log.info("[prnewswire] no articles fetched")
        return []

    # Insert new articles (skip already-seen)
    new_articles: list[dict] = []
    try:
        with conn:
            for a in deduped:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (a["id"], a["link"], a["title"], a["source"], a["first_seen"]),
                    )
                    if conn.execute(
                        "SELECT changes()"
                    ).fetchone()[0]:
                        new_articles.append(a)
                except sqlite3.Error as exc:
                    log.warning("[prnewswire] DB insert error: %s", exc)
    except sqlite3.Error as exc:
        log.error("[prnewswire] DB transaction error: %s", exc)
    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    log.info(
        "[prnewswire] %d new / %d total in %.1fs",
        len(new_articles),
        len(deduped),
        elapsed,
    )
    return new_articles


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    results = collect()
    print(f"\n=== PR Newswire Collector: {len(results)} new articles ===\n")
    for a in results[:10]:
        ticker_str = f"  tickers={a['tickers']}" if a["tickers"] else ""
        hr_str = " [HIGH RELEVANCE]" if a["high_relevance"] else ""
        print(f"  {a['title'][:80]}{hr_str}")
        print(f"  {a['link'][:80]}")
        if a["tickers"]:
            print(f"  tickers: {a['tickers']}")
        print()
    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more")
