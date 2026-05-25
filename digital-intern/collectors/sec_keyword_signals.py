"""SEC EDGAR full-text keyword signal collector.

Polls the EFTS full-text search API for high-signal keywords that appear
across ALL SEC filings (not just 8-K). Each keyword is a distinct
trading/risk signal:

  - "going concern"       → bankruptcy / liquidity risk
  - "material weakness"   → accounting fraud / restatement risk
  - "strategic alternatives" → M&A / activist pressure / buyout signal
  - "cybersecurity incident" → breach disclosure (SEC rules mandate 4-day filing)
  - "workforce reduction" → layoffs / cost-cutting
  - "restatement"         → financial fraud / accounting errors
  - "executive departure" → C-suite leadership change

API: https://efts.sec.gov/LATEST/search-index
No auth required; public EDGAR full-text search.

Dedup: keyed by sha256(adsh + keyword) in seen_articles.db — same filing
won't re-emit for the same keyword even across restarts.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

log = logging.getLogger("sec_keyword_signals")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
FETCH_TIMEOUT = 15
SOURCE = "sec_keyword_signals"

_UA = "Digital-Intern/1.0 (research; contact@example.com)"

# High-signal keywords with urgency labels
KEYWORDS: list[tuple[str, str]] = [
    ("going concern", "GOING-CONCERN"),
    ("material weakness", "MATERIAL-WEAKNESS"),
    ("strategic alternatives", "STRATEGIC-ALT"),
    ("cybersecurity incident", "CYBER-INCIDENT"),
    ("workforce reduction", "LAYOFFS"),
    ("restatement", "RESTATEMENT"),
    ("executive departure", "EXEC-DEPARTURE"),
    ("hostile takeover", "HOSTILE-TAKEOVER"),
    ("short seller report", "SHORT-ATTACK"),
    ("debt covenant", "COVENANT-BREACH"),
]

# Only include filings from these forms for relevance
PRIORITY_FORMS = {
    "8-K", "10-K", "10-Q", "NT 10-K", "NT 10-Q",
    "8-K/A", "S-1", "SC TO-T", "SC 13D", "SC 13G",
}

LOOKBACK_DAYS = 7  # scan the last 7 days; dedup layer prevents re-emitting


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


def _seen(conn: sqlite3.Connection, uid: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM seen_articles WHERE id=?", (uid,)
    ).fetchone()
    return row is not None


def _mark_seen(conn: sqlite3.Connection, uid: str, link: str, title: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
            (uid, link, title, SOURCE, now),
        )
        conn.commit()
    except Exception as e:
        log.warning(f"[sec_keyword_signals] mark_seen error: {e}")


def _fetch_keyword(
    keyword: str, label: str, start_date: str, end_date: str
) -> list[dict]:
    params = {
        "q": f'"{keyword}"',
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
    }
    headers = {
        "User-Agent": _UA,
        "Accept": "application/json",
    }
    try:
        r = requests.get(EFTS_URL, params=params, headers=headers, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        return r.json().get("hits", {}).get("hits", [])
    except Exception as e:
        log.warning(f"[sec_keyword_signals] fetch error for '{keyword}': {e}")
        return []


def collect_sec_keyword_signals() -> list[dict]:
    conn = _ensure_db()
    articles: list[dict] = []

    now = datetime.now(timezone.utc)
    start_date = (now - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    for keyword, label in KEYWORDS:
        hits = _fetch_keyword(keyword, label, start_date, end_date)
        time.sleep(0.3)  # gentle rate limiting

        for hit in hits:
            src = hit.get("_source", {})
            adsh = src.get("adsh", "")
            form = src.get("form", "")
            file_date = src.get("file_date", "")
            display_names = src.get("display_names", [])

            # Filter to priority form types
            if form not in PRIORITY_FORMS:
                continue

            # Build entity string
            if isinstance(display_names, list) and display_names:
                entity = display_names[0]
                # Strip CIK from "Company Name (TICK) (CIK 0001234567)"
                if "(CIK" in entity:
                    entity = entity.split("(CIK")[0].strip()
            elif isinstance(display_names, str):
                entity = display_names.split("(CIK")[0].strip()
            else:
                entity = "Unknown"

            if not adsh:
                continue

            uid = hashlib.sha256(f"{adsh}|{keyword}".encode()).hexdigest()
            if _seen(conn, uid):
                continue

            link = f"https://www.sec.gov/Archives/edgar/data/{adsh.replace('-', '')}/{adsh}.txt"
            filing_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&filenum=&State=0&SIC=&dateb=&owner=include&count=1&search_text=&action=getcompany&company={adsh}"
            edgar_link = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&filenum={adsh}"

            title = f"[{label}] {entity} — {form} filing ({file_date})"
            summary = (
                f"SEC EDGAR full-text alert: keyword '{keyword}' found in "
                f"{form} filing by {entity} on {file_date}. "
                f"Filing accession: {adsh}. "
                f"Signal type: {label}."
            )

            article_link = f"https://www.sec.gov/Archives/edgar/data/{adsh.replace('-', '/')}"

            articles.append(
                {
                    "title": title,
                    "link": article_link,
                    "summary": summary,
                    "published": file_date or now.isoformat(),
                    "source": SOURCE,
                }
            )
            _mark_seen(conn, uid, article_link, title)

        log.debug(
            f"[sec_keyword_signals] '{keyword}' → {len(hits)} hits from EFTS"
        )

    log.info(f"[sec_keyword_signals] collected {len(articles)} new keyword signals")
    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    results = collect_sec_keyword_signals()
    print(f"\nTotal new signals: {len(results)}")
    for r in results[:10]:
        print(f"  {r['title']}")
        print(f"    {r['link']}")
