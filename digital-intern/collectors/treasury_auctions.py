"""Treasury auction announcements collector.

Pulls upcoming UST auctions from TreasuryDirect's public JSON endpoint.
Auction announcements (size, term, rate) move the yield curve and are a
direct macro signal the rest of the pipeline can react to.

Endpoint: https://www.treasurydirect.gov/TA_WS/securities/announced?format=json
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
ENDPOINT = "https://www.treasurydirect.gov/TA_WS/securities/announced?format=json"
SOURCE = "treasury_auctions"


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


def _article_id(cusip: str, announcement_date: str) -> str:
    # CUSIP+announcement_date is stable across reannouncements of the same auction.
    return hashlib.sha256(f"ust:{cusip}:{announcement_date}".encode()).hexdigest()


def _fmt_title(rec: dict) -> str:
    term = (rec.get("securityTerm") or "").strip()
    stype = (rec.get("securityType") or "").strip()
    auc = (rec.get("auctionDate") or "")[:10]
    offered = rec.get("offeringAmount") or rec.get("totalAccepted") or ""
    cusip = rec.get("cusip") or ""
    size_str = f" ${offered}" if offered else ""
    return f"Treasury announces {term} {stype}{size_str} auction on {auc} (CUSIP {cusip})"


def collect_treasury_auctions(limit: int = 40) -> list[dict]:
    try:
        r = requests.get(
            ENDPOINT,
            headers={"User-Agent": "Digital-Intern/1.0 (+macro-collector)"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[treasury_auctions] fetch error: {e}")
        return []

    if not isinstance(data, list):
        return []

    conn = _ensure_db()
    out: list[dict] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for rec in data[:limit]:
        cusip = (rec.get("cusip") or "").strip()
        announcement = (rec.get("announcementDate") or "")[:10]
        if not cusip or not announcement:
            continue
        aid = _article_id(cusip, announcement)
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
            continue

        title = _fmt_title(rec)
        link = f"https://www.treasurydirect.gov/instit/annceresult/press/preanre/{cusip}.pdf"

        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (aid, link, title, SOURCE, now_iso),
        )
        out.append({
            "id": aid,
            "title": title,
            "link": link,
            "source": SOURCE,
            "first_seen": now_iso,
            "cusip": cusip,
            "auction_date": (rec.get("auctionDate") or "")[:10],
            "security_type": rec.get("securityType"),
            "security_term": rec.get("securityTerm"),
        })

    conn.commit()
    conn.close()
    return out


if __name__ == "__main__":
    items = collect_treasury_auctions()
    print(f"[treasury_auctions] new items: {len(items)}")
    for it in items[:10]:
        print(f"  {it['auction_date']}  {it['title']}")
        print(f"    -> {it['link']}")
