"""Credit rating change collector — Moody's, S&P Global, Fitch Ratings.

Polls Google News RSS for headline credit rating actions: downgrades, upgrades,
affirmations, and outlook changes from the three major rating agencies. Bond
credit rating changes are major market-moving events affecting equity prices,
credit spreads, and debt costs of the rated companies.

Actions tracked: downgrade, upgrade, affirm, review, watchlist, outlook change.

No API key required. Dedup via seen_articles.db. Source: CreditRatings.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import feedparser

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

USER_AGENT = "Mozilla/5.0 (compatible; DigitalInternBot/1.0)"

# Google News RSS queries: one per agency for better recall
_AGENCY_QUERIES = [
    (
        "moodys.com (downgrade OR upgrade OR affirm OR outlook OR watchlist OR \"rating action\")",
        "Moody's",
    ),
    (
        "\"S&P Global\" OR \"S&P Ratings\" (downgrade OR upgrade OR affirm OR outlook OR watchlist OR \"credit rating\")",
        "S&P",
    ),
    (
        "fitchratings.com OR \"Fitch Ratings\" (downgrade OR upgrade OR affirm OR outlook OR watchlist OR \"rating action\")",
        "Fitch",
    ),
]

_GN_BASE = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

# Stoplist: uppercase tokens that aren't tickers
_STOPWORDS = frozenset({
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BY", "FOR", "FROM",
    "HAS", "IN", "IS", "IT", "NOT", "OF", "ON", "OR", "THE", "TO",
    "UP", "WAS", "WITH", "ALL", "INC", "LTD", "LLC", "PLC", "LP",
    "BBB", "BB", "CCC", "AAA", "AA", "BB+", "BB-", "BBB+", "BBB-",
    "AA+", "AA-", "AAA", "CCC+", "CCC-", "CC", "SD", "NR", "WR",
    "US", "UK", "EU", "FED", "ECB", "GDP", "CPI", "IPO", "CEO", "CFO",
    "MOODY", "MOODYS", "FITCH", "GLOBAL", "RATINGS", "RATING", "CREDIT",
    "DOWNGRADE", "UPGRADE", "AFFIRM", "OUTLOOK", "WATCH", "REVIEW",
    "STABLE", "NEGATIVE", "POSITIVE", "DEVELOPING", "SPECULATIVE",
    "GRADE", "INVESTMENT", "JUNK", "DEFAULT", "DEBT", "BOND", "NOTE",
    "SENIOR", "UNSECURED", "SECURED", "CORPORATE", "MUNICIPAL",
    "SOVEREIGN", "GOVERNMENT", "BANK", "FINANCE", "FINANCIAL",
    "NEW", "OLD", "Q1", "Q2", "Q3", "Q4", "FY", "YTD", "YOY",
    "USD", "EUR", "GBP", "JPY", "CNN", "FOX", "NBC", "CBS",
})

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")


def _article_id(url: str, title: str) -> str:
    return hashlib.sha256(f"{url}||{title}".encode()).hexdigest()


def _ensure_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
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


def _extract_tickers(title: str) -> list[str]:
    candidates = _TICKER_RE.findall(title)
    return [t for t in candidates if t not in _STOPWORDS and len(t) >= 2]


def _clean_title(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip()


def _detect_action(title: str) -> str:
    low = title.lower()
    if "downgrade" in low:
        return "DOWNGRADE"
    if "upgrade" in low:
        return "UPGRADE"
    if "affirm" in low or "affirmed" in low:
        return "AFFIRM"
    if "watch" in low or "watchlist" in low:
        return "WATCHLIST"
    if "outlook" in low:
        return "OUTLOOK"
    if "review" in low:
        return "REVIEW"
    return "RATING_ACTION"


def collect_credit_ratings(max_items_per_agency: int = 30) -> list:
    conn = _ensure_db()
    new_articles: list = []
    seen_in_run: set = set()

    for query, agency_label in _AGENCY_QUERIES:
        feed_url = _GN_BASE.format(q=urllib.parse.quote_plus(query))
        try:
            parsed = feedparser.parse(feed_url, agent=USER_AGENT)
        except Exception as e:
            print(f"[credit_rating] fetch error ({agency_label}): {e}")
            continue

        if getattr(parsed, "bozo", 0) and not parsed.entries:
            print(f"[credit_rating] bozo parse, no entries ({agency_label})")
            continue

        for entry in parsed.entries[:max_items_per_agency]:
            title = _clean_title(entry.get("title") or "")
            link = (entry.get("link") or "").strip()
            if not title or not link:
                continue

            aid = _article_id(link, title)
            if aid in seen_in_run:
                continue
            seen_in_run.add(aid)
            if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
                continue

            action = _detect_action(title)
            tickers = _extract_tickers(title)
            published = entry.get("published") or entry.get("updated") or ""

            art = {
                "title": f"[{agency_label} {action}] {title}",
                "link": link,
                "summary": entry.get("summary") or title,
                "published": published,
                "source": f"CreditRatings/{agency_label}",
                "_tickers": tickers,
                "_rating_action": action,
                "_agency": agency_label,
            }
            new_articles.append(art)
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                "VALUES (?, ?, ?, ?, ?)",
                (aid, link, title, f"CreditRatings/{agency_label}",
                 datetime.now(timezone.utc).isoformat()),
            )

        time.sleep(0.3)  # polite between agency queries

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    t0 = time.time()
    items = collect_credit_ratings()
    dt = time.time() - t0
    print(f"[credit_rating] {len(items)} new items in {dt:.1f}s")
    for a in items[:8]:
        tk = ",".join(a["_tickers"]) or "-"
        print(f"  [{a['_agency']}][{a['_rating_action']}][{tk}] {a['title'][:100]}")
