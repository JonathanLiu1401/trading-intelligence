"""Benzinga analyst ratings RSS collector.

Pulls Benzinga's stream of analyst upgrades / downgrades / initiations /
price-target changes. Titles routinely encode the affected ticker(s) inline
("Goldman Upgrades AAPL", "Morgan Stanley Downgrades TSLA To Equal-Weight"),
so we extract uppercase 1-5 letter tokens from the headline and filter
against a stoplist of common English-word / acronym false positives.

NOTE ON FEED URL
----------------
The historically-published feed https://www.benzinga.com/rss/analyst-ratings
now returns HTTP 404, and Benzinga no longer exposes a dedicated public RSS
feed for the analyst-ratings category (their main /feed.xml is a 10-item mixed
stream where only ~10% of entries carry an "Analyst Ratings" category tag —
not enough volume to be useful).

As a working substitute we query Google News RSS for site:benzinga.com
restricted to analyst-rating verbs (upgrades / downgrades / maintains /
initiates / reiterates / price target). The resulting headlines are real
Benzinga articles; the link is a Google News redirect URL that resolves to
the underlying benzinga.com article on click. Source field is labelled
"Benzinga/AnalystRatings" so downstream consumers attribute correctly; the
Google News indirection is purely a delivery mechanism.

No API key. Polite User-Agent. Dedups via shared seen_articles.db.
"""
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

# Google News RSS, site-restricted to benzinga.com analyst-rating headlines.
_GN_QUERY = (
    "site:benzinga.com "
    "(upgrades OR downgrades OR maintains OR initiates OR reiterates "
    "OR \"price target\" OR \"raises price\" OR \"lowers price\")"
)
FEED_URL = (
    "https://news.google.com/rss/search?q="
    + urllib.parse.quote_plus(_GN_QUERY)
    + "&hl=en-US&gl=US&ceid=US:en"
)
USER_AGENT = "Mozilla/5.0 (Digital Intern Daemon; contact@digital-intern.local)"

# Strip the trailing " - Benzinga" Google News appends to every title.
_BENZINGA_SUFFIX_RE = re.compile(r"\s*-\s*Benzinga\s*$", re.IGNORECASE)

# Tokens that LOOK like tickers (1-5 uppercase letters) but are not — common
# English words and acronyms that surface in analyst-rating headlines.
_TICKER_STOPLIST = {
    "BUY", "SELL", "HOLD", "NEW", "CEO", "CFO", "ETF", "IPO", "USD", "EPS",
    "Q1", "Q2", "Q3", "Q4", "AI", "US", "UK", "EU", "FDA", "SEC", "NYSE",
    "NASDAQ", "AND", "OR", "THE", "FOR", "ON", "TO", "OF", "AT", "IS", "BY",
    "A", "AN", "IN", "IT", "ITS", "AS",
    # Additional analyst-headline noise.
    "PT", "TGT", "UP", "DOWN", "FROM", "WITH", "OUT", "NEUTRAL", "OVER",
    "UNDER", "EQUAL", "WEIGHT", "RAISE", "RAISES", "RAISED", "CUT", "CUTS",
    "LOWERS", "LOWERED", "BE", "INC", "CO", "CORP", "LLC", "LTD", "PLC",
    "SA", "AG", "NV", "ON", "OFF", "ALL", "TOP", "BIG", "NOW", "HOW", "WHY",
    "WHEN", "WHAT", "STOCK", "STOCKS", "MARKET", "BUYS", "SELLS", "HOLDS",
    "MORE", "LESS", "BEST", "WORST", "VS", "PER", "AFTER", "BEFORE", "WEEK",
    "DAY", "DAYS", "YEAR", "YEARS", "MONTH", "BPS", "PE", "PEG", "ROE", "ROI",
    "CPI", "PPI", "GDP", "OPEC", "ETFS", "REIT", "REITS", "BDC", "BDCS",
    "EBITDA", "GAAP", "ESG", "NEW",
}

_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")


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


def _clean_title(title: str) -> str:
    """Strip the Google News " - Benzinga" suffix from a headline."""
    return _BENZINGA_SUFFIX_RE.sub("", (title or "").strip()).strip()


def _extract_tickers(title: str, entry=None) -> list[str]:
    """Extract probable tickers from a Benzinga analyst headline.

    Pulls uppercase 1-5 letter tokens from the title and filters against a
    stoplist of common English words / sector acronyms. Also harvests any
    <category>/<tags> terms exposed by feedparser, mirroring the
    seekingalpha_collector style.
    """
    tickers: list[str] = []
    seen: set[str] = set()

    # 1) tags / categories (rare for Google News RSS but cheap to check)
    if entry is not None:
        for tag in entry.get("tags", []) or []:
            term = (tag.get("term") or "").strip().upper()
            if term and term.isalnum() and len(term) <= 6 and term not in _TICKER_STOPLIST:
                if term not in seen:
                    seen.add(term)
                    tickers.append(term)

    # 2) headline tokens
    for match in _TICKER_RE.findall(title or ""):
        tok = match.strip().upper()
        if not tok or tok in _TICKER_STOPLIST:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        tickers.append(tok)

    return tickers


def collect_benzinga_analyst(max_items: int = 60) -> list:
    try:
        parsed = feedparser.parse(FEED_URL, agent=USER_AGENT)
    except Exception as e:
        print(f"[benzinga_analyst] fetch error: {e}")
        return []

    if getattr(parsed, "bozo", 0) and not parsed.entries:
        print(f"[benzinga_analyst] bozo parse, no entries")
        return []

    conn = _ensure_db()
    new_articles: list = []
    seen_in_run: set = set()

    for entry in parsed.entries[:max_items]:
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

        tickers = _extract_tickers(title, entry)
        published = entry.get("published") or entry.get("updated") or ""
        art = {
            "title": title,
            "link": link,
            "summary": entry.get("summary") or "",
            "published": published,
            "source": "Benzinga/AnalystRatings",
            "_tickers": tickers,
        }
        new_articles.append(art)
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (aid, link, title, "Benzinga", datetime.now(timezone.utc).isoformat()),
        )

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    t0 = time.time()
    items = collect_benzinga_analyst()
    dt = time.time() - t0
    print(f"[benzinga_analyst] {len(items)} new items in {dt:.1f}s")
    for a in items[:5]:
        tk = ",".join(a["_tickers"]) or "-"
        print(f"  [{tk}] {a['title'][:90]}")
        print(f"     {a['link'][:110]}")
