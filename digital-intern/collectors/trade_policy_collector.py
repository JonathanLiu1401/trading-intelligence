"""Trade policy & tariff action collector.

Aggregates official US trade-policy announcements from multiple zero-auth
government sources:

  * USTR (US Trade Representative) — tariffs, trade deals, USMCA, 301 actions
  * CBP (Customs and Border Protection) — trade rulings, AD/CVD enforcement
  * ITC (International Trade Commission) — antidumping, countervailing duty
  * Commerce/ITA (International Trade Administration) — anti-dumping orders

These are among the highest-signal streams for supply-chain and tariff news
that directly move sectors (semis, autos, steel, agriculture).

All feeds are public, no API key required.
Dedup: keyed on article_id in seen_articles.db.

Standalone smoke test:
    python3 collectors/trade_policy_collector.py
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

log = logging.getLogger("trade_policy_collector")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

REQUEST_TIMEOUT = 12
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Official trade policy RSS/Atom feeds
FEEDS: dict[str, str] = {
    "USTR": "https://ustr.gov/rss.xml",
    # Federal Register: anti-dumping + countervailing duty orders from Commerce
    "FedReg/Commerce-AD": (
        "https://www.federalregister.gov/api/v1/documents.json"
        "?fields[]=title&fields[]=html_url&fields[]=publication_date"
        "&fields[]=abstract&fields[]=agencies&conditions[agencies][]=international-trade-administration"
        "&conditions[type][]=RULE&conditions[type][]=NOTICE"
        "&per_page=20&order=newest"
    ),
    # Federal Register: CBP trade enforcement
    "FedReg/CBP": (
        "https://www.federalregister.gov/api/v1/documents.json"
        "?fields[]=title&fields[]=html_url&fields[]=publication_date"
        "&fields[]=abstract&fields[]=agencies&conditions[agencies][]=u-s-customs-and-border-protection"
        "&conditions[type][]=RULE&conditions[type][]=NOTICE"
        "&per_page=20&order=newest"
    ),
    # Federal Register: USTR Section 301 actions
    "FedReg/USTR": (
        "https://www.federalregister.gov/api/v1/documents.json"
        "?fields[]=title&fields[]=html_url&fields[]=publication_date"
        "&fields[]=abstract&fields[]=agencies&conditions[agencies][]=trade-representative-office-of-united-states"
        "&conditions[type][]=RULE&conditions[type][]=NOTICE&conditions[type][]=PRORULE"
        "&per_page=20&order=newest"
    ),
}

# Keywords that elevate relevance — tariffs, supply chain, semiconductor export
PRIORITY_KEYWORDS = [
    "tariff", "antidumping", "countervailing", "section 301", "section 232",
    "section 201", "trade agreement", "export control", "entity list",
    "semiconductor", "chip", "steel", "aluminum", "solar panel", "electric vehicle",
    "china", "usmca", "nafta", "free trade agreement",
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


def _article_id(source: str, url: str) -> str:
    return hashlib.sha256(f"tradepol:{source}:{url}".encode()).hexdigest()


def _is_priority(title: str, abstract: str = "") -> bool:
    text = (title + " " + abstract).lower()
    return any(kw in text for kw in PRIORITY_KEYWORDS)


def _fetch_ustr(source_name: str, url: str) -> list[dict]:
    """Fetch USTR RSS feed via feedparser."""
    articles = []
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        for entry in feed.entries:
            link = entry.get("link", "")
            title = entry.get("title", "").strip()
            if not link or not title:
                continue
            published = entry.get("published", "")
            summary = entry.get("summary", "")
            articles.append({
                "id": _article_id(source_name, link),
                "title": title,
                "link": link,
                "source": source_name,
                "summary": summary,
                "published": published,
            })
        log.debug("[%s] fetched %d entries", source_name, len(articles))
    except Exception as exc:
        log.warning("[%s] fetch error: %s", source_name, exc)
    return articles


def _fetch_fedreg(source_name: str, url: str) -> list[dict]:
    """Fetch Federal Register API JSON endpoint."""
    articles = []
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        for doc in results:
            link = doc.get("html_url", "")
            title = doc.get("title", "").strip()
            if not link or not title:
                continue
            pub_date = doc.get("publication_date", "")
            abstract = doc.get("abstract", "") or ""
            articles.append({
                "id": _article_id(source_name, link),
                "title": title,
                "link": link,
                "source": source_name,
                "summary": abstract[:400],
                "published": pub_date,
            })
        log.debug("[%s] fetched %d docs", source_name, len(articles))
    except Exception as exc:
        log.warning("[%s] fetch error: %s", source_name, exc)
    return articles


def _insert(conn: sqlite3.Connection, articles: list[dict]) -> int:
    """Insert articles, skipping duplicates. Returns new count."""
    now = datetime.now(timezone.utc).isoformat()
    new_count = 0
    for a in articles:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                "VALUES (?, ?, ?, ?, ?)",
                (a["id"], a["link"], a["title"], a["source"], now),
            )
            if conn.total_changes > 0:
                new_count += 1
        except sqlite3.Error as exc:
            log.warning("DB insert error: %s", exc)
    conn.commit()
    return new_count


def collect_trade_policy() -> int:
    """Run one collection cycle. Returns count of new articles inserted."""
    conn = _ensure_db()
    total_new = 0
    all_articles = []

    for source_name, url in FEEDS.items():
        if source_name == "USTR":
            articles = _fetch_ustr(source_name, url)
        else:
            articles = _fetch_fedreg(source_name, url)
        all_articles.extend(articles)

    # Filter: keep all USTR entries + FedReg entries matching priority keywords
    filtered = []
    for a in all_articles:
        if a["source"] == "USTR" or _is_priority(a["title"], a.get("summary", "")):
            filtered.append(a)

    total_new = _insert(conn, filtered)
    conn.close()

    log.info("[trade_policy] %d new articles from %d candidates", total_new, len(filtered))
    return total_new


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)

    print("=== Trade Policy Collector Smoke Test ===")
    conn = _ensure_db()
    all_articles: list[dict] = []

    for sname, url in FEEDS.items():
        print(f"\n--- {sname} ---")
        if sname == "USTR":
            articles = _fetch_ustr(sname, url)
        else:
            articles = _fetch_fedreg(sname, url)
        print(f"  fetched: {len(articles)}")
        for a in articles[:3]:
            prio = " [PRIORITY]" if _is_priority(a["title"], a.get("summary","")) else ""
            print(f"  [{a['source']}]{prio} {a['title'][:80]}")
            print(f"    {a['link'][:70]}")
        all_articles.extend(articles)

    filtered = [a for a in all_articles if a["source"] == "USTR" or _is_priority(a["title"], a.get("summary", ""))]
    print(f"\nTotal fetched: {len(all_articles)}  |  Priority-filtered: {len(filtered)}")

    new = _insert(conn, filtered)
    conn.close()
    print(f"New articles inserted: {new}")
