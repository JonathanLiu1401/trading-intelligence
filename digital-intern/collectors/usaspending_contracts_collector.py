"""USASpending.gov federal contract awards collector.

Fetches large federal contract awards (≥$10M) from USASpending.gov's public API
(no key required). Surfaces DoD/DoE/NASA semiconductor, tech, and defense awards
that are catalysts for portfolio holdings (AXTI, LRCX, NVDA, MSFT, ORCL, etc.).

Dedup strategy:
    seen_articles.db keyed by sha256(usaspending_award_id), refreshed daily.
    Each award emitted at most once per day so incremental value updates surface.

Run cadence: every daemon cycle; USASpending asks max 30 req/min — we fetch
one page of 25 per cycle, well within limits.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

log = logging.getLogger("usaspending_contracts")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"

SOURCE_NAME = "usaspending_contracts"
API_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
FETCH_TIMEOUT = 25
MIN_AWARD_USD = 10_000_000  # $10M floor — below this, too noisy

# NAICS codes relevant to portfolio holdings
_RELEVANT_NAICS = {
    "334413",  # Semiconductors
    "334411",  # Electron tubes
    "334419",  # Electronic components
    "334210",  # Telephone apparatus
    "334220",  # Radio/TV broadcast equipment
    "334290",  # Communications equipment
    "334510",  # Electromedical / electro-optical instruments
    "334516",  # Analytical instruments
    "541511",  # Custom computer programming services
    "541512",  # Computer systems design services
    "541519",  # Other computer-related services
    "541715",  # R&D in physical sciences (incl. semiconductor)
    "336411",  # Aircraft manufacturing (aerospace)
    "336414",  # Guided missiles & space vehicles
    "927110",  # Space research & tech
    "928110",  # National security
}

# Keywords in recipient name or description that flag relevance
_RELEVANCE_KEYWORDS = {
    "semiconductor", "laser", "photon", "optic", "optical", "lidar", "radar",
    "quantum", "microelectron", "silicon", "gallium", "arsenic", "nitride",
    "wafer", "fab", "foundry", "chip", "integrated circuit", "sensor",
    "missile", "drone", "unmanned", "satellite", "space launch",
    "cloud", "cybersecurity", "artificial intelligence", "machine learning",
    "software", "computing", "data center",
}


def _load_watchlist_tickers() -> set[str]:
    tickers: set[str] = set()
    try:
        with open(PORTFOLIO_PATH) as f:
            p = json.load(f)
        for pos in p.get("positions", []):
            if t := pos.get("ticker", ""):
                tickers.add(t.upper())
        for t in p.get("sector_watchlist", []):
            if t:
                tickers.add(t.upper())
    except Exception:
        pass
    return tickers


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


def _article_id(award_id: str, today: str) -> str:
    return hashlib.sha256(f"{award_id}||{today}".encode()).hexdigest()


def _is_relevant(recipient: str, description: str, naics: str) -> bool:
    if naics in _RELEVANT_NAICS:
        return True
    combined = (recipient + " " + description).lower()
    return any(kw in combined for kw in _RELEVANCE_KEYWORDS)


def _fetch_awards(start_date: str, end_date: str) -> list[dict]:
    payload = {
        "filters": {
            "time_period": [{"start_date": start_date, "end_date": end_date}],
            "award_type_codes": ["A", "B", "C", "D"],  # contracts only
            "award_amounts": [{"lower_bound": MIN_AWARD_USD}],
        },
        "fields": [
            "Award ID",
            "Recipient Name",
            "Award Amount",
            "Awarding Agency",
            "Funding Agency",
            "Description",
            "Award Date",
            "NAICS Description",
            "NAICS Code",
            "Place of Performance State Code",
        ],
        "sort": "Award Amount",
        "order": "desc",
        "limit": 25,
        "page": 1,
    }
    try:
        resp = requests.post(
            API_URL,
            json=payload,
            timeout=FETCH_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as exc:
        log.warning("[usaspending] fetch error: %s", exc)
        return []


def collect_usaspending_contracts() -> list[dict]:
    today = date.today()
    today_str = today.isoformat()
    # Look back 2 days so weekend gaps don't miss Friday awards
    start_str = (today - timedelta(days=2)).isoformat()

    raw = _fetch_awards(start_str, today_str)
    if not raw:
        return []

    conn = _ensure_db()
    new_articles: list[dict] = []

    for award in raw:
        award_id = award.get("Award ID") or ""
        recipient = (award.get("Recipient Name") or "").strip()
        description = (award.get("Description") or "").strip()
        amount = award.get("Award Amount") or 0
        agency = award.get("Awarding Agency") or ""
        naics_code = str(award.get("NAICS Code") or "")
        naics_desc = award.get("NAICS Description") or ""
        award_date = award.get("Award Date") or today_str
        state = award.get("Place of Performance State Code") or ""

        if not award_id or not recipient:
            continue
        if not _is_relevant(recipient, description, naics_code):
            continue

        aid = _article_id(award_id, today_str)
        if conn.execute("SELECT 1 FROM seen_articles WHERE id = ?", (aid,)).fetchone():
            continue

        amount_m = amount / 1_000_000
        title = (
            f"USASpending: ${amount_m:.0f}M contract — {recipient}"
            f" ({agency[:40]})"
        )
        summary = (
            f"Award ID: {award_id} | Amount: ${amount_m:.1f}M | "
            f"Agency: {agency} | NAICS: {naics_code} {naics_desc} | "
            f"State: {state} | Description: {description[:200]}"
        )
        link = f"https://www.usaspending.gov/award/{award_id}/"

        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (aid, link, title, SOURCE_NAME, datetime.now(timezone.utc).isoformat()),
        )
        new_articles.append({
            "title": title,
            "link": link,
            "summary": summary,
            "published": award_date,
            "source": SOURCE_NAME,
        })

    conn.commit()
    conn.close()
    log.info("[usaspending] %d new relevant contract awards", len(new_articles))
    return new_articles


collect = collect_usaspending_contracts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("=== USASpending.gov contract awards (live) ===")
    items = collect_usaspending_contracts()
    if items:
        from storage.article_store import ArticleStore
        store = ArticleStore()
        inserted = store.insert_batch(items)
        print(f"Fetched: {len(items)}  |  Inserted to articles.db: {inserted}")
        for it in items[:5]:
            print(f"  - {it['title']}")
            print(f"    {it['summary'][:120]}")
    else:
        print("No new relevant awards found (all seen or no matching awards).")
