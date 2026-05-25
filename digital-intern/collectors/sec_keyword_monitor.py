"""SEC EDGAR keyword event monitor.

Scans all 8-K filings (past 7 days) for high-signal keywords that indicate
material corporate events — layoffs, M&A, going-concern warnings, data
breaches, material weaknesses, restructurings. Unlike sec_edgar.py
(which scans by portfolio ticker), this catches events from ANY company.

Uses a dedicated lightweight local DB (sec_keyword_monitor.db) rather than
the shared seen_articles.db on the USB drive to avoid lock contention.

API: https://efts.sec.gov/LATEST/search-index (SEC full-text search, no auth)
Rate-limit: polite 0.35s between searches; well under SEC's 10 req/sec.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
# Dedicated local DB — not the USB-backed seen_articles.db — to avoid contention.
_DEDUP_DB = BASE_DIR / "data" / "sec_keyword_monitor.db"

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
FETCH_TIMEOUT = 15
LOOKBACK_HOURS = 168  # 7 days — EFTS index has ~2-3 day ingestion lag
SOURCE_TAG = "sec_keyword_monitor"
MAX_NEW_PER_RUN = 50  # cap to avoid flooding on first run

_UA = "Mozilla/5.0 (compatible; Digital-Intern research-bot/1.0; +mailto:research@digital-intern.local)"

# High-signal keyword groups: (phrase, label, urgency_prefix)
KEYWORD_GROUPS: list[tuple[str, str, str]] = [
    # Workforce / restructuring
    ("workforce reduction", "Layoffs/Workforce Reduction", "LAYOFFS"),
    ("reduction in force", "Layoffs/RIF", "LAYOFFS"),
    ("going concern", "Going Concern Warning", "GOING CONCERN"),
    # M&A / strategic
    ("strategic alternatives", "Strategic Alternatives Explored", "M&A SIGNAL"),
    ("definitive agreement to acquire", "Acquisition Announced", "ACQUISITION"),
    ("merger agreement", "Merger Agreement", "MERGER"),
    # Financial distress
    ("material weakness", "Material Weakness Disclosed", "MATERIAL WEAKNESS"),
    ("restatement of financial", "Financial Restatement", "RESTATEMENT"),
    ("chapter 11", "Bankruptcy Filing", "BANKRUPTCY"),
    # Cyber / security
    ("cybersecurity incident", "Cybersecurity Incident", "CYBER BREACH"),
    ("ransomware", "Ransomware Attack", "CYBER BREACH"),
    # Macro / political
    ("tariff", "Tariff Impact Disclosure", "TARIFF"),
]


def _ensure_db() -> sqlite3.Connection:
    _DEDUP_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DEDUP_DB), timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen (
            id TEXT PRIMARY KEY,
            first_seen TEXT
        )"""
    )
    conn.commit()
    return conn


def _article_id(adsh: str, label: str) -> str:
    return hashlib.sha256(f"{adsh}|{label}".encode()).hexdigest()[:16]


def _extract_ticker(display_name: str) -> str:
    m = re.search(r"\(([A-Z]{1,5}(?:-[A-Z]{1,2})?)\)", display_name)
    return m.group(1) if m else ""


def _extract_company(display_name: str) -> str:
    return display_name.split("(")[0].strip()


def _fetch_keyword(keyword: str, since: str, until: str) -> list[dict]:
    params = {
        "q": f'"{keyword}"',
        "forms": "8-K",
        "dateRange": "custom",
        "startdt": since,
        "enddt": until,
    }
    try:
        resp = requests.get(
            EFTS_URL, params=params, timeout=FETCH_TIMEOUT,
            headers={"User-Agent": _UA},
        )
        resp.raise_for_status()
        return resp.json().get("hits", {}).get("hits", [])
    except Exception as e:
        print(f"[sec_keyword_monitor] fetch error for {keyword!r}: {e}")
        return []


def collect_sec_keyword_events() -> list[dict]:
    """Scan recent 8-Ks for high-signal keywords. Returns article dicts."""
    conn = _ensure_db()
    since = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).strftime("%Y-%m-%d")
    until = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_articles: list[dict] = []
    seen_in_run: set[str] = set()

    for keyword, label, urgency in KEYWORD_GROUPS:
        if len(new_articles) >= MAX_NEW_PER_RUN:
            break
        hits = _fetch_keyword(keyword, since, until)
        for hit in hits:
            if len(new_articles) >= MAX_NEW_PER_RUN:
                break
            src = hit.get("_source", {})
            display_names = src.get("display_names") or []
            file_date = src.get("file_date", "")
            adsh = src.get("adsh", "")
            items = src.get("items") or []

            if not adsh:
                continue

            link = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&filenum=&State=0&SIC=&dateb=&owner=include&count=1&search_text=&accession={adsh}"
            raw_name = display_names[0] if display_names else "Unknown Company"
            company = _extract_company(raw_name)
            ticker = _extract_ticker(raw_name)
            ticker_str = f" ({ticker})" if ticker else ""
            items_str = f" [Items: {', '.join(items[:3])}]" if items else ""

            title = f"[{urgency}] {company}{ticker_str} — {label}{items_str}"
            aid = _article_id(adsh, label)

            if aid in seen_in_run:
                continue
            seen_in_run.add(aid)

            try:
                if conn.execute("SELECT 1 FROM seen WHERE id=?", (aid,)).fetchone():
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO seen (id, first_seen) VALUES (?,?)",
                    (aid, datetime.now(timezone.utc).isoformat()),
                )
            except sqlite3.Error as e:
                print(f"[sec_keyword_monitor] db error: {e}")
                continue

            new_articles.append({
                "title": title,
                "link": link,
                "summary": (
                    f"SEC 8-K filed {file_date}: {company}{ticker_str} disclosed "
                    f"'{keyword}'. Items: {', '.join(items)}. "
                    f"EDGAR full-text search."
                ),
                "published": file_date,
                "source": SOURCE_TAG,
            })

        time.sleep(0.35)

    conn.commit()
    conn.close()
    return new_articles


collect = collect_sec_keyword_events


if __name__ == "__main__":
    print("=== SEC EDGAR Keyword Event Monitor (live fetch) ===")
    items = collect_sec_keyword_events()
    print(f"New events found: {len(items)}")
    for art in items[:10]:
        print(f"  {art['published']:10s}  {art['title']}")
