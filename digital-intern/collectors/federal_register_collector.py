"""Federal Register regulatory action collector.

Monitors high-impact regulatory filings from agencies that directly move
semiconductor, tech, and financial markets:

  - BIS (Bureau of Industry and Security): export controls, entity list,
    chip/AI restrictions, EAR rule changes
  - OFAC (Treasury): sanctions designations — Russia, China, Iran affect semis
  - FTC (antitrust): tech M&A reviews, market-competition rules
  - FCC: telecom/spectrum rules
  - Commerce/NTIA: broadband, AI policy, chip act implementations

Uses the free, unauthenticated Federal Register API v1.
Dedup: per article-id in seen_articles.db so each rule emits once.

Standalone smoke test:
    python3 collectors/federal_register_collector.py
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger("federal_register_collector")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

API_BASE = "https://www.federalregister.gov/api/v1/documents.json"
REQUEST_TIMEOUT = 12
_UA = "Mozilla/5.0 (Digital-Intern/1.0; research bot; contact@digital-intern.local)"

# Agency slugs → source label. Order = priority.
AGENCIES: dict[str, str] = {
    "industry-and-security-bureau": "FedReg/BIS",          # export controls
    "foreign-assets-control-office": "FedReg/OFAC",        # sanctions
    "federal-trade-commission": "FedReg/FTC",              # antitrust
    "federal-communications-commission": "FedReg/FCC",     # telecom
    "national-telecommunications-and-information-administration": "FedReg/NTIA",
    "national-institute-of-standards-and-technology": "FedReg/NIST",
    "treasury-department": "FedReg/Treasury",
}

# Only emit Final Rules, Proposed Rules, and Notices (skip corrections/corrections)
DOCUMENT_TYPES = ["RULE", "PRORULE", "NOTICE"]

# Look back this many days per fetch pass (keeps each pass bounded)
LOOKBACK_DAYS = 3

# Fields to request from the API
FIELDS = [
    "title", "type", "publication_date", "html_url",
    "abstract", "agencies", "document_number",
]

# Keywords that elevate relevance for semis/tech focus
_HIGH_SIGNAL_KEYWORDS = [
    "export control", "entity list", "eas", "ear", "advanced semiconductor",
    "artificial intelligence", "chip", "nvidia", "amd", "tsmc", "huawei",
    "sanction", "designation", "sdn", "blocked", "russia", "china", "iran",
    "merger", "acquisition", "antitrust", "competition",
    "chips act", "broadband", "spectrum", "5g", "6g",
]


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


def _article_id(doc_number: str, source: str) -> str:
    return hashlib.sha256(f"{source}||{doc_number}".encode()).hexdigest()


def _is_high_signal(title: str, abstract: str) -> bool:
    text = (title + " " + (abstract or "")).lower()
    return any(kw in text for kw in _HIGH_SIGNAL_KEYWORDS)


def _fetch_agency(
    agency_slug: str,
    source_name: str,
    since_date: str,
    conn: sqlite3.Connection,
    seen_in_run: set[str],
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "conditions[agencies][]": agency_slug,
        "conditions[publication_date][gte]": since_date,
        "conditions[type][]": DOCUMENT_TYPES,
        "per_page": 20,
        "order": "newest",
        "fields[]": FIELDS,
    }
    try:
        resp = requests.get(
            API_BASE, params=params,
            headers={"User-Agent": _UA},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning(f"[federal_register] {source_name} fetch error: {exc}")
        return []

    articles: list[dict[str, Any]] = []
    for doc in data.get("results", []):
        doc_num = doc.get("document_number", "")
        title = (doc.get("title") or "").strip()
        link = doc.get("html_url", "").strip()
        if not title or not link or not doc_num:
            continue

        aid = _article_id(doc_num, source_name)
        if aid in seen_in_run:
            continue
        seen_in_run.add(aid)
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
            continue

        abstract = (doc.get("abstract") or "").strip()
        doc_type = doc.get("type", "Notice")
        pub_date = doc.get("publication_date", "")

        # Prepend doc type so urgency scoring can weight Final Rules higher
        display_title = f"[{doc_type}] {title}"

        art: dict[str, Any] = {
            "title": display_title,
            "link": link,
            "summary": abstract[:600] if abstract else f"Federal Register {doc_type}: {title}",
            "published": pub_date,
            "source": source_name,
        }

        # Tag high-signal items so downstream urgency scorer can pick them up
        if _is_high_signal(title, abstract):
            art["high_signal"] = True

        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                "VALUES (?,?,?,?,?)",
                (aid, link, display_title, source_name, now),
            )
            conn.commit()
        except sqlite3.Error as e:
            log.warning(f"[federal_register] db insert error: {e}")
            continue

        articles.append(art)

    return articles


def collect_federal_register(lookback_days: int = LOOKBACK_DAYS) -> list[dict[str, Any]]:
    """Collect recent Federal Register regulatory actions. Returns list of article dicts."""
    since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    conn = _ensure_db()
    seen_in_run: set[str] = set()
    all_articles: list[dict[str, Any]] = []

    for agency_slug, source_name in AGENCIES.items():
        arts = _fetch_agency(agency_slug, source_name, since, conn, seen_in_run)
        if arts:
            log.info(f"[federal_register] {source_name}: +{len(arts)} new")
        all_articles.extend(arts)
        time.sleep(0.3)  # polite rate limiting

    conn.close()
    return all_articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("Testing Federal Register collector...")
    arts = collect_federal_register(lookback_days=7)
    print(f"\nFetched {len(arts)} new regulatory actions\n")
    for a in arts[:10]:
        sig = " [HIGH SIGNAL]" if a.get("high_signal") else ""
        print(f"  [{a['source']}]{sig} {a['title'][:90]}")
        print(f"    {a['link']}")
        print()
    if not arts:
        print("  (no new documents — all already seen or none in window)")
