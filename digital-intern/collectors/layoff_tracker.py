"""Tech & semiconductor layoff tracker.

Monitors multiple sources for layoff news affecting portfolio/watchlist
companies. Layoffs are high-signal events for semiconductor stocks — Intel,
Micron, and AMD cuts directly affect supply/demand and margin outlooks.

Sources:
  - TechCrunch layoffs tag RSS (https://techcrunch.com/tag/layoffs/feed/)
  - Google News RSS: "semiconductor layoffs" broad search
  - Google News RSS: "chip maker layoffs" search
  - Google News RSS per high-priority ticker + "layoff"

Dedupes via shared seen_articles.db.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import feedparser

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"

SOURCE_NAME = "layoff_tracker"
FETCH_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; Digital-Intern-Daemon)"

# Always-watched companies — add ADR/alt names for better headline matching
PRIORITY_COMPANIES: set[str] = {
    "micron", "intel", "nvidia", "amd", "tsmc", "asml", "applied materials",
    "lam research", "kla", "qualcomm", "broadcom", "marvell", "skyworks",
    "qorvo", "on semi", "onsemi", "maxim", "analog devices", "texas instruments",
    "wolfspeed", "globalfoundries", "samsung", "sk hynix", "hynix",
    "western digital", "seagate", "axcelis", "axti", "tsem", "lite-on",
    "lite on", "ii-vi", "coherent", "qbts", "d-wave", "microsoft",
    "oracle", "arm", "synopsys", "cadence", "ansys",
}

# Priority tickers for per-ticker Google News queries
PRIORITY_TICKERS = [
    "MU", "INTC", "NVDA", "AMD", "ASML", "AMAT", "LRCX", "KLAC",
    "QCOM", "AVGO", "AXTI", "TSEM", "LITE", "QBTS",
]


def _load_watched_companies() -> set[str]:
    """Supplement PRIORITY_COMPANIES with names from portfolio + watchlist."""
    names: set[str] = set(PRIORITY_COMPANIES)
    try:
        with open(PORTFOLIO_PATH) as f:
            pf = json.load(f)
        for pos in pf.get("positions", []):
            t = (pos.get("ticker") or "").upper()
            if t:
                names.add(t.lower())
    except Exception:
        pass
    try:
        with open(WATCHLIST_PATH) as f:
            wl = json.load(f)
        for key in ("memory_core", "semis_equipment", "portfolio"):
            for t in wl.get(key, []):
                if t:
                    names.add(t.lower())
    except Exception:
        pass
    return names


def _article_id(link: str, title: str) -> str:
    return hashlib.sha256(f"{link}||{title}".encode()).hexdigest()[:32]


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS seen_articles "
        "(id TEXT PRIMARY KEY, link TEXT, title TEXT, source TEXT, first_seen TEXT)"
    )
    conn.commit()
    return conn


def _is_relevant(title: str, summary: str, watched: set[str]) -> tuple[bool, str]:
    """Return (relevant, matched_company). Checks headline for layoff + company."""
    text = (title + " " + summary).lower()
    layoff_terms = {"layoff", "layoffs", "lay off", "laid off", "job cut",
                    "job cuts", "workforce reduction", "headcount reduction",
                    "redundanc", "restructur", "downsiz", "rif "}
    has_layoff = any(term in text for term in layoff_terms)
    if not has_layoff:
        return False, ""
    for co in watched:
        cl = co.lower()
        # Short tokens (≤4 chars) need word-boundary match to avoid substring noise
        if len(cl) <= 4:
            if re.search(r'\b' + re.escape(cl) + r'\b', text):
                return True, co
        else:
            if cl in text:
                return True, co
    return False, ""


def _fetch_feed(url: str, source_label: str, watched: set[str], max_items: int = 40) -> list[dict]:
    """Fetch an RSS feed and return relevant new articles."""
    try:
        feed = feedparser.parse(url, agent=USER_AGENT, request_headers={"timeout": str(FETCH_TIMEOUT)})
    except Exception as e:
        print(f"[layoff_tracker] fetch error {source_label}: {e}")
        return []

    if getattr(feed, "bozo", 0) and not feed.entries:
        return []

    results: list[dict] = []
    for entry in feed.entries[:max_items]:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        summary = (entry.get("summary") or entry.get("description") or "").strip()
        if not title or not link:
            continue

        relevant, matched = _is_relevant(title, summary, watched)
        if not relevant:
            continue

        results.append({
            "title": title,
            "link": link,
            "summary": summary[:500],
            "published": entry.get("published") or entry.get("updated") or "",
            "source": source_label,
            "_matched_company": matched,
        })
    return results


def collect_layoffs(max_items: int = 50) -> list[dict]:
    watched = _load_watched_companies()
    conn = _ensure_db()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Build feed list: broad RSS sources + per-ticker Google News
    feeds: list[tuple[str, str]] = [
        (
            "https://techcrunch.com/tag/layoffs/feed/",
            "TechCrunch/Layoffs",
        ),
        (
            "https://news.google.com/rss/search?q=semiconductor+layoffs+OR+%22chip+maker+layoffs%22&hl=en-US&gl=US&ceid=US:en",
            "GoogleNews/SemiLayoffs",
        ),
        (
            "https://news.google.com/rss/search?q=tech+company+layoffs+2025+OR+2026&hl=en-US&gl=US&ceid=US:en",
            "GoogleNews/TechLayoffs",
        ),
    ]
    # Add per-ticker queries for highest-priority holdings
    for ticker in PRIORITY_TICKERS[:6]:  # keep to 6 to avoid rate-limit
        url = (
            f"https://news.google.com/rss/search?"
            f"q={ticker}+layoff+OR+layoffs&hl=en-US&gl=US&ceid=US:en"
        )
        feeds.append((url, f"GoogleNews/{ticker}Layoff"))

    # Fetch in parallel
    all_candidates: list[dict] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_feed, url, label, watched): label for url, label in feeds}
        for fut in as_completed(futures):
            try:
                all_candidates.extend(fut.result())
            except Exception as e:
                print(f"[layoff_tracker] feed error: {e}")

    # Dedup + persist new articles
    new_articles: list[dict] = []
    seen_in_run: set[str] = set()

    for art in all_candidates:
        aid = _article_id(art["link"], art["title"])
        if aid in seen_in_run:
            continue
        seen_in_run.add(aid)
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
            continue

        new_articles.append(art)
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (aid, art["link"], art["title"], SOURCE_NAME, now_iso),
        )
        if len(new_articles) >= max_items:
            break

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    t0 = time.time()
    items = collect_layoffs()
    dt = time.time() - t0
    print(f"[layoff_tracker] {len(items)} new items in {dt:.1f}s")
    for a in items[:8]:
        co = a.get("_matched_company", "?")
        print(f"  [{a['source']} / {co}] {a['title'][:90]}")
        print(f"     {a['link'][:100]}")
