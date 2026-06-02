"""Motley Fool earnings call transcript collector.

Polls the Motley Fool earnings-transcripts RSS feed which publishes full-text
earnings call transcripts shortly after each call ends. Entries include a
ticker tag (e.g. 'ADSK', 'GAP') making them directly filterable.

Two filter modes:
  1. Portfolio/watchlist tickers — any transcript for a held or watched ticker
  2. Broad capture — all transcripts regardless of ticker (high-signal events)

The full-text transcript summary is preserved in the article body so the ML
pipeline can score CEO/CFO guidance language, margin commentary, and outlook.

RSS: https://www.fool.com/feeds/index.aspx?id=fool-rss-earnings-transcripts
No API key required. Returns ~50 most recent entries per poll.

Dedup: seen_articles.db keyed by sha256(link || title).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import feedparser

try:
    from core.logger import get_logger
    _log = get_logger("earnings_transcript")
except Exception:
    _log = logging.getLogger("earnings_transcript")

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

FEED_URL = "https://www.fool.com/feeds/index.aspx?id=fool-rss-earnings-transcripts"
SOURCE = "MotleyFool/EarningsTranscripts"
FETCH_TIMEOUT = 12
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# Tag term used by Motley Fool to mark transcript entries
_TRANSCRIPT_TAG = "earningscall-transcripts"
# Strip HTML tags from summary for clean text storage
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _load_tracked_tickers() -> set[str]:
    tickers: set[str] = set()
    try:
        pf = json.loads(PORTFOLIO_PATH.read_text())
        for pos in pf.get("positions", []):
            t = pos.get("ticker", "").strip().upper()
            if t:
                tickers.add(t)
        for opt in pf.get("options", []):
            t = opt.get("underlying", "").strip().upper()
            if t:
                tickers.add(t)
        for t in pf.get("sector_watchlist", []):
            if t.strip():
                tickers.add(t.strip().upper())
    except Exception:
        pass
    try:
        wl = json.loads(WATCHLIST_PATH.read_text())
        for key in ("memory_core", "semis_equipment", "portfolio"):
            for t in wl.get(key, []):
                if t.strip():
                    tickers.add(t.strip().upper())
    except Exception:
        pass
    return tickers


def _article_id(link: str, title: str) -> str:
    return hashlib.sha256(f"{link}||{title}".encode()).hexdigest()


def _entry_tickers(entry) -> list[str]:
    """Extract ticker symbols from feedparser entry tags."""
    tickers = []
    for tag in entry.get("tags", []):
        term = tag.get("term", "").strip().upper()
        # Filter: 1-5 uppercase letters, not a UUID-like tag or category word
        if re.match(r"^[A-Z]{1,5}$", term) and term not in {
            "AI", "US", "UK", "EU", "CEO", "CFO", "ETF", "IPO", "EPS",
            "Q1", "Q2", "Q3", "Q4", "BUY", "SELL", "HOLD", "NYSE",
        }:
            tickers.append(term)
    return tickers


def _is_transcript(entry) -> bool:
    return any(
        tag.get("term", "").lower() == _TRANSCRIPT_TAG
        for tag in entry.get("tags", [])
    )


def _clean_html(text: str) -> str:
    return _HTML_TAG_RE.sub("", text).strip()


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY, link TEXT, title TEXT,
            source TEXT, first_seen TEXT
        )"""
    )
    conn.commit()


def collect_earnings_transcripts() -> list[dict]:
    """Fetch Motley Fool earnings transcripts and return new articles."""
    feedparser.USER_AGENT = _UA
    feed = feedparser.parse(FEED_URL)
    if not feed.entries:
        _log.warning("earnings_transcript: empty feed (status=%s)", feed.get("status"))
        return []

    tracked = _load_tracked_tickers()
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    _ensure_db(conn)

    articles = []
    for entry in feed.entries:
        link = entry.get("link", "")
        title = entry.get("title", "").strip()
        if not link or not title:
            continue

        # Accept if it's a transcript OR if it mentions a portfolio ticker
        entry_tickers = _entry_tickers(entry)
        is_transcript = _is_transcript(entry)
        matches_portfolio = bool(tracked & set(entry_tickers))

        if not is_transcript and not matches_portfolio:
            continue

        aid = _article_id(link, title)
        try:
            if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
                continue  # already seen
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                "VALUES (?, ?, ?, ?, ?)",
                (aid, link, title, SOURCE, datetime.now(timezone.utc).isoformat()),
            )
        except sqlite3.Error as e:
            _log.debug("earnings_transcript dedup error: %s", e)
            continue

        raw_summary = entry.get("summary", "") or ""
        summary = _clean_html(raw_summary)[:1000]
        published_raw = entry.get("published", "")
        try:
            from email.utils import parsedate_to_datetime
            pub_dt = parsedate_to_datetime(published_raw)
            published = pub_dt.astimezone(timezone.utc).isoformat()
        except Exception:
            published = datetime.now(timezone.utc).isoformat()

        label = "Transcript" if is_transcript else "Article"
        ticker_str = " ".join(entry_tickers) if entry_tickers else "N/A"

        articles.append({
            "id": aid,
            "url": link,
            "title": title,
            "source": SOURCE,
            "published": published,
            "summary": f"[{label}] Tickers: {ticker_str}\n\n{summary}",
            "_tickers": entry_tickers,
        })

    conn.commit()
    conn.close()
    _log.info("earnings_transcript: %d new articles from %d entries", len(articles), len(feed.entries))
    return articles
