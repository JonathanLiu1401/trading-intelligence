"""SEC activist & M&A event collector.

Polls EDGAR current-filings atom feeds for high-impact special form types:
  SC 13D / SC 13D/A  — activist investors taking/amending >5% stakes
  SC TO-T            — tender offers (acquisition bids)
  DEFM14A / PREM14A  — definitive/preliminary merger proxy statements

These filings are broadly market-moving (any stock, not just portfolio
tickers) and are not covered by the portfolio-scoped sec_edgar.py EFTS sweep.

Dedup is keyed on EDGAR accession number so amendments to the same filing
each emit their own article (material new information may be in each version).
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import requests

try:
    from core.logger import get_logger
    _log = get_logger("sec_activist")
except Exception:
    _log = logging.getLogger("sec_activist")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

EDGAR_USER_AGENT = "Digital-Intern-Daemon contact@digital-intern.local"
ATOM_BASE = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type={form}&dateb=&owner=include&count=40&output=atom"
)
FORMS = [
    ("SC 13D",   "SEC/activist-13D",    "Activist stake >5%"),
    ("SC 13D/A", "SEC/activist-13D-amend", "Activist stake amendment"),
    ("SC TO-T",  "SEC/tender-offer",    "Tender offer (acquisition bid)"),
    ("DEFM14A",  "SEC/merger-proxy",    "Definitive merger proxy"),
    ("PREM14A",  "SEC/merger-prelim",   "Preliminary merger proxy"),
]
ATOM_NS = "http://www.w3.org/2005/Atom"
HTTP_TIMEOUT = 15
INTER_FORM_DELAY = 0.5  # be polite to SEC servers


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
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


def _article_id(link: str, title: str) -> str:
    return hashlib.sha256(f"{link}||{title}".encode()).hexdigest()


def _already_seen(conn: sqlite3.Connection, art_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM seen_articles WHERE id=?", (art_id,)
    ).fetchone()
    return row is not None


def _mark_seen(conn: sqlite3.Connection, art_id: str, link: str, title: str, source: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles(id,link,title,source,first_seen) VALUES(?,?,?,?,?)",
        (art_id, link, title, source, now),
    )
    conn.commit()


def _fetch_form_feed(form: str) -> list[dict]:
    url = ATOM_BASE.format(form=requests.utils.quote(form, safe=""))
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": EDGAR_USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as exc:
        _log.warning("sec_activist: fetch failed for %s: %s", form, exc)
        return []

    try:
        root = ElementTree.fromstring(resp.content)
    except ElementTree.ParseError as exc:
        _log.warning("sec_activist: XML parse failed for %s: %s", form, exc)
        return []

    ns = {"atom": ATOM_NS}
    entries = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        link_el = entry.find("atom:link", ns)
        updated_el = entry.find("atom:updated", ns)
        summary_el = entry.find("atom:summary", ns)

        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        link = link_el.get("href", "") if link_el is not None else ""
        updated = updated_el.text.strip() if updated_el is not None and updated_el.text else ""
        summary = summary_el.text.strip() if summary_el is not None and summary_el.text else ""

        if title and link:
            entries.append(
                {"title": title, "link": link, "updated": updated, "summary": summary}
            )
    return entries


def collect() -> list[dict]:
    conn = _ensure_db()
    articles: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()

    for form, source, label in FORMS:
        entries = _fetch_form_feed(form)
        for entry in entries:
            title = f"[{label}] {entry['title']}"
            link = entry["link"]
            art_id = _article_id(link, entry["title"])

            if _already_seen(conn, art_id):
                continue

            summary = entry.get("summary", "")
            # Strip any HTML tags from EDGAR summary
            summary = summary.replace("<b>", "").replace("</b>", "").strip()

            articles.append(
                {
                    "id": art_id,
                    "title": title,
                    "link": link,
                    "summary": f"{form} filing. {summary}",
                    "source": source,
                    "first_seen": now,
                }
            )
            _mark_seen(conn, art_id, link, title, source)
            _log.info("sec_activist: new %s filing: %s", form, entry["title"][:80])

        time.sleep(INTER_FORM_DELAY)

    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = collect()
    print(f"\nFetched {len(results)} new SEC activist/M&A filings:\n")
    for a in results:
        print(f"  [{a['source']}] {a['title']}")
        print(f"    {a['link']}")
        print(f"    {a['summary'][:120]}")
        print()
