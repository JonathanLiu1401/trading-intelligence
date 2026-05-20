"""SEC EDGAR 13F-HR institutional holdings collector.

Pulls the EDGAR RSS feed for 13F-HR filings (quarterly institutional holdings
disclosures). Q1 2026 filings are due mid-May, making this especially timely.

Each 13F filing becomes an article so the urgency scorer can surface notable
institutions. Deduplicates via seen_articles.db. For priority institutions,
fetches the filing index to extract top holdings details.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

EDGAR_13F_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=13F-HR&dateb=&owner=include&count=40&output=atom"
)
EDGAR_USER_AGENT = "Digital-Intern-Daemon contact@digital-intern.local"
SOURCE = "sec_13f_holdings"
FETCH_TIMEOUT = 12

# Well-known institutions whose 13Fs are always high-signal.
PRIORITY_INSTITUTIONS = {
    "berkshire", "blackrock", "vanguard", "state street", "fidelity",
    "bridgewater", "citadel", "renaissance", "two sigma", "aqr",
    "point72", "millennium", "third point", "pershing square", "baupost",
    "druckenmiller", "soros", "greenlight", "loeb", "icahn",
    "tiger global", "coatue", "d1 capital", "lone pine",
    "maverick capital", "viking global", "jana partners",
    "starboard", "valueact", "tci fund", "harris associates",
    "alaska permanent", "calpers", "calstrs", "norway",
}

# Tickers that make any 13F more interesting (portfolio relevance).
WATCHLIST_TICKERS = {
    "NVDA", "AMD", "ASML", "TSM", "INTC", "MU", "MSFT", "AAPL",
    "META", "GOOGL", "AMZN", "TSLA", "AVGO", "QCOM", "AMAT",
    "LRCX", "KLAC", "LITE", "AXTI", "TSEM", "QBTS", "ORCL",
}


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


def _article_id(accession: str) -> str:
    return hashlib.sha256(f"13f:{accession}".encode()).hexdigest()


def _extract_accession(link: str) -> str:
    """Pull accession number from EDGAR link (last path component before .htm)."""
    m = re.search(r"([0-9]{18}|[0-9]{10}-[0-9]{2}-[0-9]{6})", link)
    return m.group(1) if m else link[-40:]


def _is_priority(filer_name: str) -> bool:
    name_lower = filer_name.lower()
    return any(kw in name_lower for kw in PRIORITY_INSTITUTIONS)


def _fetch_index_holdings(filing_link: str, filer_name: str) -> tuple[list[str], int | None]:
    """Try to fetch top-5 holdings from the 13F filing index XML.

    Returns (tickers_list, table_row_count). Non-critical: returns ([], None) on any error.
    """
    try:
        # Derive the index URL from the filing link.
        # EDGAR archive links look like: /Archives/edgar/data/CIK/ACCESSION-NODASH/
        m = re.search(r"(/Archives/edgar/data/\d+/[^/]+)", filing_link)
        if not m:
            return [], None
        base = "https://www.sec.gov" + m.group(1)
        idx_url = base.rstrip("/") + "/index.json"
        r = requests.get(
            idx_url,
            headers={"User-Agent": EDGAR_USER_AGENT},
            timeout=FETCH_TIMEOUT,
        )
        if r.status_code != 200:
            return [], None

        data = r.json()
        files = [f.get("name", "") for f in data.get("directory", {}).get("item", [])]

        # Find the primary 13F XML document (infotable.xml).
        xml_file = next(
            (f for f in files if "infotable" in f.lower() and f.endswith(".xml")),
            None,
        )
        if not xml_file:
            return [], None

        xml_url = base.rstrip("/") + "/" + xml_file
        xr = requests.get(
            xml_url,
            headers={"User-Agent": EDGAR_USER_AGENT},
            timeout=FETCH_TIMEOUT,
        )
        if xr.status_code != 200:
            return [], None

        xml = xr.text
        # Extract ticker/CUSIP and value from infoTable entries.
        entries = re.findall(
            r"<nameOfIssuer>(.*?)</nameOfIssuer>.*?<value>(\d+)</value>",
            xml, re.DOTALL
        )
        if not entries:
            return [], None

        # Sort by value (thousands USD), descending.
        entries_sorted = sorted(entries, key=lambda x: int(x[1]), reverse=True)
        top_names = [n.strip() for n, _ in entries_sorted[:5]]
        total_rows = len(entries)
        return top_names, total_rows
    except Exception:
        return [], None


def collect_13f_filings() -> list[dict]:
    conn = _ensure_db()
    try:
        feed = feedparser.parse(
            EDGAR_13F_URL,
            agent=EDGAR_USER_AGENT,
            request_headers={"User-Agent": EDGAR_USER_AGENT},
        )
    except Exception as exc:
        print(f"[sec_13f] feed fetch failed: {exc}")
        return []

    articles: list[dict] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for entry in feed.entries:
        raw_title = entry.get("title", "")
        link = entry.get("link", "")
        accession = _extract_accession(link)
        art_id = _article_id(accession)

        # Already seen?
        if conn.execute(
            "SELECT 1 FROM seen_articles WHERE id=?", (art_id,)
        ).fetchone():
            continue

        # Extract filer name from title: "13F-HR - Filer Name (CIK) (Filer)"
        m = re.match(r"13F-HR(?:/A)?\s*-\s*(.+?)\s*\(\d+\)", raw_title, re.IGNORECASE)
        filer_name = m.group(1).strip() if m else raw_title
        is_amendment = "/A" in raw_title

        priority = _is_priority(filer_name)
        top_holdings: list[str] = []
        holding_count: int | None = None

        if priority:
            top_holdings, holding_count = _fetch_index_holdings(link, filer_name)
            if top_holdings:
                time.sleep(0.3)  # SEC rate-limit courtesy

        # Build title.
        amendment_str = " (amendment)" if is_amendment else ""
        if top_holdings:
            holdings_str = ", ".join(top_holdings[:3])
            title = (
                f"13F{amendment_str}: {filer_name} Q1 2026 filing"
                f" — top holdings include {holdings_str}"
            )
        elif holding_count:
            title = (
                f"13F{amendment_str}: {filer_name} Q1 2026 filing"
                f" ({holding_count} positions)"
            )
        else:
            title = f"13F{amendment_str}: {filer_name} files Q1 2026 institutional holdings"

        # Build tickers list for relevance scoring.
        tickers: list[str] = []
        title_upper = title.upper()
        for tkr in WATCHLIST_TICKERS:
            if tkr in title_upper:
                tickers.append(tkr)

        article: dict = {
            "id": art_id,
            "title": title,
            "url": link or f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=13F-HR",
            "source": SOURCE,
            "published": now_iso,
            "content": title,
            "_tickers": tickers,
            "priority": priority,
        }

        try:
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen)"
                " VALUES (?, ?, ?, ?, ?)",
                (art_id, link, title, SOURCE, now_iso),
            )
            conn.commit()
            articles.append(article)
        except sqlite3.Error:
            pass

    conn.close()
    return articles
