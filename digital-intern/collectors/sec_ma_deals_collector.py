"""SEC EDGAR M&A deal detector — market-wide merger/acquisition signal.

Scans all 8-K filings via EDGAR Full-Text Search (EFTS) for deal-specific
language: "definitive agreement", "tender offer", "merger consideration",
"per share in cash", etc. Covers the entire market (not just portfolio tickers),
so it surfaces deals involving third parties that may affect portfolio holdings.

Distinct from sec_edgar.py which only searches EFTS per portfolio ticker.
This collector is keyword-driven and market-wide.

API: https://efts.sec.gov/LATEST/search-index (no auth; SEC public service)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

log = logging.getLogger("sec_ma_deals")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
REQUEST_TIMEOUT = 15
SOURCE_NAME = "SEC EDGAR M&A"
MAX_RESULTS = 20  # per query; keep modest to avoid hammering EDGAR

EDGAR_UA = os.environ.get(
    "SEC_USER_AGENT",
    "Digital-Intern-Daemon sealai215j@gmail.com",
)

# Keyword searches that indicate a genuine M&A deal in an 8-K.
# We use EFTS phrase search; each entry is run as a separate query.
# Prioritise phrases that appear in actual deal announcements.
_DEAL_QUERIES = [
    '"definitive agreement" "per share"',
    '"tender offer" "per share"',
    '"merger consideration" "stockholders"',
    '"agreement and plan of merger"',
    '"acquisition" "definitive agreement" "cash consideration"',
]

# 8-K item codes that are most deal-relevant
_DEAL_ITEMS = {"1.01", "2.01", "8.01"}


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS seen_articles "
        "(id TEXT PRIMARY KEY, link TEXT, title TEXT, source TEXT, first_seen TEXT)"
    )
    conn.commit()


def _article_id(adsh: str, query_tag: str) -> str:
    return hashlib.sha1(f"sec_ma|{adsh}|{query_tag}".encode()).hexdigest()


def _fetch_deals(query: str, days_back: int = 1) -> list[dict]:
    """Query EFTS for recent 8-K filings matching deal keywords."""
    today = datetime.now(timezone.utc)
    start = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    params = {
        "q": query,
        "dateRange": "custom",
        "startdt": start,
        "enddt": end,
        "forms": "8-K",
    }
    try:
        r = requests.get(
            EFTS_URL,
            params=params,
            headers={"User-Agent": EDGAR_UA},
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 429:
            log.warning("[sec_ma] EDGAR EFTS rate-limited")
            return []
        if r.status_code != 200:
            log.warning(f"[sec_ma] EFTS status {r.status_code}")
            return []
        return r.json().get("hits", {}).get("hits", [])
    except Exception as e:
        log.warning(f"[sec_ma] fetch error for '{query}': {e}")
        return []


def _hit_to_article(hit: dict, query_tag: str) -> dict | None:
    src = hit.get("_source", {})
    adsh = src.get("adsh", "")
    if not adsh:
        return None

    # Filter: require at least one deal-relevant item code
    items = set(src.get("items", []))
    if not (items & _DEAL_ITEMS):
        return None

    names = src.get("display_names", ["Unknown Company"])
    company = names[0].split("(CIK")[0].strip() if names else "Unknown"
    ciks = src.get("ciks", ["0"])
    cik = ciks[0].lstrip("0") if ciks else "0"
    file_date = src.get("file_date", "")
    items_str = ", ".join(sorted(items))

    art_id = _article_id(adsh, query_tag)
    link = (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include&count=5"
    )
    title = f"[M&A Signal] {company} filed 8-K — {query_tag} ({file_date})"
    summary = (
        f"SEC EDGAR 8-K filing by {company} (CIK {cik}) on {file_date}. "
        f"Items: {items_str}. Query matched: {query_tag}. "
        f"Accession: {adsh}. "
        f"Full filing: https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-','')}/{adsh}-index.htm"
    )

    return {
        "id": art_id,
        "link": link,
        "title": title,
        "summary": summary,
        "source": SOURCE_NAME,
        "published": file_date,
        "_adsh": adsh,
        "_company": company,
        "_items": list(items),
        "_query": query_tag,
    }


def collect_sec_ma_deals() -> list[dict]:
    """Scan EDGAR EFTS for market-wide M&A deal 8-K filings."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    _ensure_db(conn)

    new_articles: list[dict] = []
    seen_adsh: set[str] = set()

    for query in _DEAL_QUERIES:
        query_tag = query.replace('"', "").replace(" ", "_")[:40]
        hits = _fetch_deals(query, days_back=2)
        log.debug(f"[sec_ma] query '{query_tag}' → {len(hits)} hits")

        for hit in hits[:MAX_RESULTS]:
            art = _hit_to_article(hit, query_tag)
            if not art:
                continue
            adsh = art["_adsh"]
            if adsh in seen_adsh:
                continue  # dedupe cross-query
            seen_adsh.add(adsh)

            try:
                if conn.execute(
                    "SELECT 1 FROM seen_articles WHERE id=?", (art["id"],)
                ).fetchone():
                    continue
                now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                conn.execute(
                    "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                    "VALUES (?,?,?,?,?)",
                    (art["id"], art["link"], art["title"], SOURCE_NAME, now),
                )
                conn.commit()
                new_articles.append(art)
                log.info(f"[sec_ma] NEW: {art['title']}")
            except sqlite3.Error as e:
                log.warning(f"[sec_ma] db error: {e}")

        time.sleep(0.5)  # gentle pacing between EFTS queries

    conn.close()
    log.info(f"[sec_ma] {len(new_articles)} new M&A signal articles")
    return new_articles


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    articles = collect_sec_ma_deals()
    if not articles:
        print("No new M&A filings (all already seen or none in last 2 days)")
        sys.exit(0)
    print(f"\n=== {len(articles)} NEW M&A DEAL SIGNALS ===\n")
    for a in articles:
        print(f"COMPANY: {a['_company']}")
        print(f"TITLE:   {a['title']}")
        print(f"ITEMS:   {a['_items']}")
        print(f"LINK:    {a['link']}")
        print()
