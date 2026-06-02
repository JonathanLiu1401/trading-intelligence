"""Daily Treasury Statement (DTS) cash balance collector.

Fetches the US Treasury's daily operating cash balance and key fiscal flows
from the FiscalData API. The Treasury General Account (TGA) balance is a
macro signal: a rapidly-shrinking TGA can precede emergency borrowing or
debt-ceiling drama; surges follow tax seasons.

API: https://api.fiscaldata.treasury.gov/services/api/v1/accounting/dts/dts_table_1
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
SOURCE = "dts_cash_balance"

# Table 1, line item "Federal Reserve Account: Total Operating Balance"
ENDPOINT = (
    "https://api.fiscaldata.treasury.gov/services/api/v1/accounting/dts/dts_table_1"
    "?fields=record_date,account_type,open_today_bal,close_today_bal"
    "&filter=account_type:eq:Federal Reserve Account"
    "&sort=-record_date&page[size]=10"
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


def _article_id(record_date: str, account_type: str) -> str:
    return hashlib.sha256(f"dts:{record_date}:{account_type}".encode()).hexdigest()


def _fmt_title(row: dict) -> str:
    date = row.get("record_date", "")[:10]
    close_bal = row.get("close_today_bal", "")
    open_bal = row.get("open_today_bal", "")
    try:
        close_b = float(close_bal) / 1000  # millions → billions
        open_b = float(open_bal) / 1000
        change = close_b - open_b
        sign = "+" if change >= 0 else ""
        return (
            f"Treasury Cash Balance {date}: ${close_b:,.1f}B "
            f"({sign}{change:,.1f}B vs prior day)"
        )
    except (TypeError, ValueError):
        return f"Treasury Cash Balance {date}: close={close_bal}M open={open_bal}M"


def collect_dts_cash_balance() -> list[dict]:
    try:
        r = requests.get(
            ENDPOINT,
            headers={"User-Agent": "Digital-Intern/1.0 (+macro-dts)"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:
        print(f"[dts_cash_balance] fetch error: {e}")
        return []

    if not data:
        print("[dts_cash_balance] no data returned")
        return []

    conn = _ensure_db()
    now = datetime.now(timezone.utc).isoformat()
    inserted: list[dict] = []

    for row in data:
        record_date = row.get("record_date", "")[:10]
        account_type = row.get("account_type", "Federal Reserve Account")
        art_id = _article_id(record_date, account_type)

        exists = conn.execute(
            "SELECT 1 FROM seen_articles WHERE id=?", (art_id,)
        ).fetchone()
        if exists:
            continue

        title = _fmt_title(row)
        link = f"https://fiscaldata.treasury.gov/datasets/daily-treasury-statement/operating-cash-balance?startDate={record_date}&endDate={record_date}"

        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
            (art_id, link, title, SOURCE, now),
        )
        inserted.append({"id": art_id, "title": title, "link": link, "date": record_date})

    conn.commit()
    conn.close()

    if inserted:
        print(f"[dts_cash_balance] +{len(inserted)} new records")
    else:
        print("[dts_cash_balance] no new records (already seen)")

    return inserted


if __name__ == "__main__":
    results = collect_dts_cash_balance()
    print(f"\nFetched {len(results)} new DTS records:")
    for r in results:
        print(f"  [{r['date']}] {r['title']}")
