"""Wikipedia recent-changes collector — filters mainspace edits for finance/company pages.

Uses the public MediaWiki API. No API key. Returns a small filtered stream of
edits to articles whose title matches a tracked ticker name or a financial keyword.

API: GET https://en.wikipedia.org/w/api.php?action=query&list=recentchanges&...
"""
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

API_URL = "https://en.wikipedia.org/w/api.php"
HTTP_TIMEOUT = 10
USER_AGENT = "Digital-Intern-Daemon (sealai215j@gmail.com)"
LIMIT = 500  # recent-changes returns the most recent N edits

# Keywords whose presence in a page title makes the edit worth ingesting.
FINANCIAL_KEYWORDS = {
    "earnings", "stock", "stocks", "shares", "ipo", "merger", "acquisition",
    "bankruptcy", "fed", "federal reserve", "interest rate", "inflation",
    "recession", "gdp", "tariff", "central bank", "ecb",
    "semiconductor", "chip", "foundry", "dram", "nand", "hbm",
    "bitcoin", "ethereum", "cryptocurrency", "blockchain",
    "oil", "gas", "opec", "commodity", "treasury", "bond yield",
    "trade war", "sanctions", "export controls",
}

# Company / ticker mapping — case-insensitive title contains.
COMPANY_TITLES = {
    "nvidia", "advanced micro devices", "amd ", "intel corporation",
    "qualcomm", "tsmc", "taiwan semiconductor", "samsung electronics",
    "sk hynix", "micron", "lam research", "applied materials", "kla corporation",
    "asml", "broadcom", "marvell", "western digital", "seagate", "kioxia",
    "oracle corporation", "microsoft", "apple inc", "amazon", "alphabet",
    "meta platforms", "tesla", "ford motor", "general motors",
    "boeing", "lockheed", "raytheon",
    "lumentum", "axcelis",
}


def _load_tickers() -> set[str]:
    tickers: set[str] = set()
    try:
        with open(PORTFOLIO_PATH, "r") as f:
            pf = json.load(f)
        for pos in pf.get("positions", []):
            t = (pos.get("ticker") or "").upper()
            if t:
                tickers.add(t)
        for opt in pf.get("options", []):
            u = (opt.get("underlying") or "").upper()
            if u:
                tickers.add(u)
        for t in pf.get("sector_watchlist", []):
            if t:
                tickers.add(t.upper())
    except Exception:
        pass

    try:
        with open(WATCHLIST_PATH, "r") as f:
            wl = json.load(f)
        for key in ("memory_core", "semis_equipment", "broader_semis", "portfolio"):
            for t in wl.get(key, []):
                if t:
                    tickers.add(t.upper())
    except Exception:
        pass
    return tickers


def _is_relevant(title: str, tickers: set[str]) -> bool:
    t_lower = title.lower()
    for kw in FINANCIAL_KEYWORDS:
        if kw in t_lower:
            return True
    for ct in COMPANY_TITLES:
        if ct in t_lower:
            return True
    # Ticker symbol exact-match: "(NVDA)" or " NVDA " in title
    for tkr in tickers:
        if len(tkr) >= 3 and (f"({tkr})" in title or f" {tkr} " in title):
            return True
    return False


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Hardened seen_articles.db connection — mirrors google_news._ensure_db /
    # source_health.py / article_store.py. 11 collectors share this one file;
    # SQLite's default busy_timeout=0 turns any transient cross-writer lock
    # into an immediate OperationalError that aborts the whole pass and drops
    # the fetched batch. WAL + 30s timeout lets the write wait out contention.
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


def collect_wikipedia() -> list:
    """Fetch recent changes, filter for finance/company pages, return article dicts."""
    params = {
        "action": "query",
        "list": "recentchanges",
        "rcnamespace": 0,           # mainspace only
        "rctype": "edit|new",
        "rcprop": "title|timestamp|comment|user",
        "rclimit": LIMIT,
        "format": "json",
    }
    try:
        r = requests.get(API_URL, params=params, timeout=HTTP_TIMEOUT,
                         headers={"User-Agent": USER_AGENT})
    except Exception as e:
        print(f"[wikipedia] fetch error: {e}")
        return []
    if r.status_code != 200:
        return []
    try:
        data = r.json()
    except Exception:
        return []

    changes = data.get("query", {}).get("recentchanges", [])
    if not changes:
        return []

    tickers = _load_tickers()
    conn = _ensure_db()
    new_articles: list = []
    seen_in_run: set = set()

    for ch in changes:
        title = (ch.get("title") or "").strip()
        if not title or not _is_relevant(title, tickers):
            continue
        comment = (ch.get("comment") or "").strip()
        user = (ch.get("user") or "").strip()
        ts = (ch.get("timestamp") or "").strip()
        # Wikipedia page URL — stable, dedup-friendly per-edit by appending timestamp
        page_slug = title.replace(" ", "_")
        link = f"https://en.wikipedia.org/wiki/{page_slug}"
        # Make link unique per edit so each meaningful edit can register
        dedup_key = f"{link}#{ts}"

        aid = _article_id(dedup_key, title)
        if aid in seen_in_run:
            continue
        seen_in_run.add(aid)
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
            continue

        summary = comment if comment else f"Edit by {user}" if user else ""
        new_articles.append({
            "title": f"[Wikipedia] {title}",
            "link": link,
            "summary": summary,
            "published": ts,
            "source": "Wikipedia",
        })
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (aid, dedup_key, title, "Wikipedia", datetime.now(timezone.utc).isoformat()),
        )

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    items = collect_wikipedia()
    print(f"Got {len(items)} new Wikipedia edits")
    for a in items[:10]:
        print(f"  {a['title'][:80]}")
