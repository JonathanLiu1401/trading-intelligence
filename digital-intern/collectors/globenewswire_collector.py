"""GlobeNewswire financial press release collector.

Pulls corporate press releases from GlobeNewswire's public RSS feeds, filtered
to financially-relevant subject codes. Each release includes the issuing
company's exchange-listed ticker(s) in category tags, so we can route items to
the right _tickers list for relevance scoring downstream.

No API key required. Up to 20 items per subject-code feed; we poll several
financial subject codes in parallel.

Subject codes covered (selected for ticker density):
  02 — corporate governance / M&A (board elections, mergers, acquisitions)
  06 — debt/bond offerings (fixed income capital markets)
  12 — dividends (quarterly / special cash dividends)
  13 — financial results / quarterly filings
  17 — equity grants / stock option plans
  20 — clinical / pharma trial results (biotech catalyst)
  25 — licensing / IP agreements
  30 — patents granted (technology / biotech)
"""
import hashlib
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

REQUEST_TIMEOUT = 12
MAX_WORKERS = 8

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

SUBJECT_CODES = [2, 6, 12, 13, 17, 20, 25, 30]
BASE_URL = "https://www.globenewswire.com/RssFeed/subjectcode/{code}"

_STOCK_DOMAIN = "https://www.globenewswire.com/rss/stock"
_EXCHANGE_RE = re.compile(r"^[A-Z0-9\-]+:(.+)$")


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
    return hashlib.sha256(f"{link}|{title}".encode()).hexdigest()[:16]


def _strip_exchange(cat_text: str) -> str:
    """'NYSE:BRC' → 'BRC', 'Nasdaq:NVDA' → 'NVDA'"""
    m = _EXCHANGE_RE.match(cat_text or "")
    return m.group(1).strip() if m else (cat_text or "").strip()


def _fetch_feed(code: int) -> list[dict]:
    url = BASE_URL.format(code=code)
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return []

    articles = []
    for item in root.findall(".//item"):
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        pub_el = item.find("pubDate")

        if title_el is None or link_el is None:
            continue

        title = (title_el.text or "").strip()
        link = (link_el.text or "").strip()
        if not title or not link:
            continue

        summary = ""
        if desc_el is not None and desc_el.text:
            summary = re.sub(r"<[^>]+>", "", desc_el.text).strip()[:500]

        pub = (pub_el.text or "").strip() if pub_el is not None else ""

        tickers = []
        for cat in item.findall("category"):
            domain = cat.get("domain", "")
            if domain.endswith("/stock") and cat.text:
                sym = _strip_exchange(cat.text)
                if sym and 1 <= len(sym) <= 6 and sym not in tickers:
                    tickers.append(sym)

        articles.append({
            "title": title,
            "link": link,
            "summary": summary,
            "published": pub,
            "source": "GlobeNewswire",
            "_tickers": tickers,
            "_gnw_code": code,
        })

    return articles


def collect_globenewswire() -> list[dict]:
    conn = _ensure_db()
    new_articles: list[dict] = []

    # Fetch all subject code feeds in parallel
    raw: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_feed, code): code for code in SUBJECT_CODES}
        for fut in as_completed(futures):
            raw.extend(fut.result())

    now_iso = datetime.now(timezone.utc).isoformat()
    for art in raw:
        aid = _article_id(art["link"], art["title"])
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
            continue
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (aid, art["link"], art["title"], "GlobeNewswire", now_iso),
        )
        new_articles.append(art)

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    t0 = time.time()
    items = collect_globenewswire()
    dt = time.time() - t0
    print(f"[globenewswire] {len(items)} new items in {dt:.1f}s")
    for a in items[:10]:
        tk = ",".join(a["_tickers"]) or "-"
        print(f"  [{tk}] {a['title'][:90]}")
