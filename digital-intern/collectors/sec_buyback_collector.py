"""SEC Buyback Announcement Collector — scans EDGAR EFTS for share repurchase
program announcements in 8-K filings over the past 3 days.

Corporate buyback authorizations are market-moving signals: they reflect
management confidence, provide price support, and often precede sustained
outperformance. This collector isolates buyback-specific 8-Ks that the
broader sec_edgar.py sweep doesn't tag distinctly.

No API key required. Uses the same EDGAR EFTS endpoint as sec_edgar.py.
"""
import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
UA = "Digital-Intern-Daemon contact@digital-intern.local"
TIMEOUT = 15
LOOKBACK_DAYS = 3
SOURCE_TAG = "sec_buyback"

_QUERIES = [
    '"share repurchase program"',
    '"stock repurchase program"',
    '"share buyback"',
    '"stock buyback"',
    '"repurchase authorization"',
]


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=60, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY, link TEXT, title TEXT,
            source TEXT, first_seen TEXT
        )"""
    )
    conn.commit()
    return conn


def _article_id(link: str, title: str) -> str:
    return hashlib.sha256(f"{link}|{title}".encode()).hexdigest()


def _is_seen(conn: sqlite3.Connection, aid: str) -> bool:
    return conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone() is not None


def _mark_seen(conn: sqlite3.Connection, aid: str, link: str, title: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles(id,link,title,source,first_seen) VALUES(?,?,?,?,?)",
        (aid, link, title, SOURCE_TAG, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _efts_query(query: str, since: str, until: str) -> list[dict]:
    params = {
        "q": query,
        "forms": "8-K",
        "dateRange": "custom",
        "startdt": since,
        "enddt": until,
        "_source": "file_date,period_of_report,entity_name,file_num,period_of_report,form_type",
        "hits.hits.total.relation": "eq",
    }
    resp = requests.get(EFTS_URL, params=params, headers={"User-Agent": UA}, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data.get("hits", {}).get("hits", [])


def collect_sec_buybacks() -> list[dict]:
    """Return list of article dicts for recent buyback announcements."""
    conn = _ensure_db()
    since = (datetime.now(timezone.utc).date() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    until = datetime.now(timezone.utc).date().isoformat()

    seen_filing_ids: set[str] = set()
    articles: list[dict] = []

    for query in _QUERIES:
        try:
            hits = _efts_query(query, since, until)
        except Exception as e:
            print(f"[sec_buyback] error querying {query!r}: {e}")
            continue

        for hit in hits:
            src = hit.get("_source", {})
            filing_id = hit.get("_id", "")

            # Deduplicate across queries by filing doc id
            if filing_id in seen_filing_ids:
                continue
            seen_filing_ids.add(filing_id)

            # display_names is a list of strings like "Company, Inc. (TICK) (CIK 0001234567)"
            display_names = src.get("display_names", [])
            entity = display_names[0].split("(CIK")[0].strip() if display_names else "Unknown"
            file_date = src.get("file_date", until)
            accession = src.get("adsh", filing_id.split(":")[0] if ":" in filing_id else filing_id)

            # Stable direct link to the specific filing document
            doc_filename = filing_id.split(":")[-1] if ":" in filing_id else ""
            cik = src.get("ciks", ["0"])[0].lstrip("0") or "0"
            accession_nodash = accession.replace("-", "")
            if doc_filename:
                doc_link = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{doc_filename}"
            else:
                doc_link = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include&count=5"

            title = f"{entity} — Share Repurchase Announcement ({file_date})"
            summary = (
                f"{entity} filed an 8-K on {file_date} disclosing a share repurchase / buyback program. "
                f"Search match: {query}. Accession: {accession}."
            )

            aid = _article_id(doc_link, title)
            if _is_seen(conn, aid):
                continue

            _mark_seen(conn, aid, doc_link, title)
            articles.append({
                "title": title,
                "link": doc_link,
                "summary": summary,
                "published": file_date,
                "source": SOURCE_TAG,
            })

    return articles
