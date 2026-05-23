"""FTC and DOJ Antitrust Division press-release collector.

Sits alongside ``cftc_press_collector`` / ``fed_press_collector`` in the
regulatory-wire tier. Covers two US antitrust/competition enforcers whose
actions directly move individual tickers and sectors:

  * FTC press releases — merger challenges, consent decrees, Big Tech
    enforcement, privacy enforcement that names public companies.
  * DOJ Antitrust Division (ATR) press releases — criminal cartel actions,
    merger review clearances/blocks, civil enforcement.

Sources:
  - FTC: https://www.ftc.gov/feeds/press-release.xml  (official RSS, no auth)
  - DOJ ATR: https://www.justice.gov/atr/news (HTML scrape; no working RSS)

Why this matters:
  - A merger block or DOJ investigation announcement often moves the target
    ticker 5–20% intraday and rattles sector peers.
  - FTC consent orders that name a tech/platform company (AAPL, GOOGL, META,
    AMZN) are high-urgency signals for the leveraged-ETF + sector book.
  - Cartel actions in semiconductor supply chain, chemicals, or aerospace show
    up here before appearing in standard news wires.

Dedup: shared ``data/seen_articles.db`` keyed by sha256(link||title) —
identical to cftc_press_collector / fed_press_collector.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

FTC_FEED = "https://www.ftc.gov/feeds/press-release.xml"
DOJ_ATR_URL = "https://www.justice.gov/atr/press-releases"

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


def _fetch_ftc() -> list[dict]:
    """Fetch FTC press releases via official RSS feed."""
    try:
        resp = requests.get(
            FTC_FEED, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA}
        )
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as e:
        print(f"[ftc_doj_collector] FTC RSS error: {e}")
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
            "source": "ftc_press",
        })
    return out


def _fetch_doj_atr() -> list[dict]:
    """Scrape DOJ Antitrust Division news page (no working RSS feed)."""
    try:
        resp = requests.get(
            DOJ_ATR_URL,
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": _UA},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"[ftc_doj_collector] DOJ ATR scrape error: {e}")
        return []

    out: list[dict] = []
    # DOJ pages render news items as <article> or <div class="views-row"> with
    # an <a> link and a date. Try multiple selectors gracefully.
    items = (
        soup.select("div.views-row")
        or soup.select("article")
        or soup.select("li.views-row")
    )
    for item in items[:20]:
        a_tag = item.find("a")
        if not a_tag:
            continue
        title = a_tag.get_text(strip=True)
        href = a_tag.get("href", "")
        if not title or not href:
            continue
        # Resolve relative URLs
        if href.startswith("/"):
            href = "https://www.justice.gov" + href
        elif not href.startswith("http"):
            continue

        # Extract date if present
        date_el = item.find(class_=re.compile(r"date|time|views-field-created", re.I))
        published = date_el.get_text(strip=True) if date_el else ""

        # Summary from paragraph or span
        p = item.find("p") or item.find("span", class_=re.compile(r"summary|body|desc", re.I))
        summary = p.get_text(strip=True) if p else ""

        out.append({
            "title": title,
            "link": href,
            "summary": summary,
            "published": published,
            "source": "doj_atr",
        })
    return out


def collect_ftc_doj() -> list[dict]:
    """Collect deduplicated FTC + DOJ ATR press-release items.

    Returns a list of dicts: {title, link, summary, published, source}.
    Consistent with collect_cftc_press / collect_fed_press — the caller
    (daemon _ingest or __main__) inserts via ArticleStore.insert_batch.
    """
    conn = _ensure_db()
    new_articles: list[dict] = []
    seen_in_run: set[str] = set()

    all_items = _fetch_ftc() + _fetch_doj_atr()
    for art in all_items:
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
                (aid, link, title, art["source"],
                 datetime.now(timezone.utc).isoformat()),
            )
        except sqlite3.Error as e:
            print(f"[ftc_doj_collector] dedup row skipped ({art['source']}): {e}")
            continue
        new_articles.append(art)

    conn.commit()
    conn.close()
    return new_articles


# Alias matching the collect_<source>() convention.
collect = collect_ftc_doj


if __name__ == "__main__":
    print("=== FTC / DOJ ATR live fetch ===")
    ftc_raw = _fetch_ftc()
    doj_raw = _fetch_doj_atr()
    print(f"  ftc_press   {len(ftc_raw):3d} entries")
    print(f"  doj_atr     {len(doj_raw):3d} entries")

    eg_line = None
    for art in ftc_raw[:3]:
        print(f"  [FTC] {art['title'][:80]}")
        if eg_line is None:
            eg_line = f"ftc_press: {art['title']}"
    for art in doj_raw[:3]:
        print(f"  [DOJ] {art['title'][:80]}")
        if eg_line is None:
            eg_line = f"doj_atr: {art['title']}"

    items = collect_ftc_doj()
    inserted = 0
    if items:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from storage.article_store import ArticleStore
        store = ArticleStore()
        inserted = store.insert_batch(items)

    print("\n=== Summary ===")
    print(f"New deduped items     : {len(items)}")
    print(f"Inserted articles.db  : {inserted}")
    if eg_line:
        print(f"DISCORD_EG: {eg_line}")
    for a in items[:8]:
        print(f"  + [{a['source']}] {a['title']}")
