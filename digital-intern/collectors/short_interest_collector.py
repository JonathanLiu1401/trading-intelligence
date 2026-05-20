"""High short interest collector — scrapes highshortinterest.com.

Fetches the top ~80 stocks with >20% short interest, updated twice monthly.
Each stock is stored as a structured article so the labeler and dashboard
can surface potential short squeeze candidates.

No API key required. Uses a browser User-Agent to avoid 403s.
Runs at most once per 6 hours (data updates every ~2 weeks, but hourly
re-fetching is wasteful).
"""
import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
CURSOR_PATH = BASE_DIR / "data" / "short_interest_cursor.json"

SOURCE = "short_interest_highshortinterest"
BASE_URL = "https://www.highshortinterest.com/"
COOLDOWN_HOURS = 6
REQUEST_TIMEOUT = 15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _article_id(url: str, title: str) -> str:
    return hashlib.sha256(f"{url}|{title}".encode()).hexdigest()


def _seen_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS seen_articles (id TEXT PRIMARY KEY)"
    )
    conn.commit()
    return conn


def _is_seen(conn: sqlite3.Connection, aid: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM seen_articles WHERE id=?", (aid,)
    ).fetchone() is not None


def _mark_seen(conn: sqlite3.Connection, aid: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id) VALUES (?)", (aid,)
    )
    conn.commit()


def _load_cursor() -> dict:
    try:
        return json.loads(CURSOR_PATH.read_text())
    except Exception:
        return {}


def _save_cursor(data: dict) -> None:
    CURSOR_PATH.write_text(json.dumps(data))


def _parse_stocks(html: str) -> list[dict]:
    """Extract stock rows from the highshortinterest.com table."""
    soup = BeautifulSoup(html, "html.parser")
    stocks = []

    # Find the data table — it has header row: Ticker, Company, Exchange,
    # ShortInt, Float, Outstd, Industry
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        header_found = False
        for row in rows:
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if not header_found:
                if "Ticker" in cells and "ShortInt" in cells:
                    header_found = True
                continue
            # Each data row has exactly 7 cols; skip empties / nav rows
            clean = [c for c in cells if c]
            if len(clean) < 7:
                continue
            ticker, company, exchange, short_int, float_sh, outstd, industry = (
                clean[0], clean[1], clean[2], clean[3], clean[4], clean[5], clean[6]
            )
            # Skip if ticker looks invalid
            if not re.match(r'^[A-Z]{1,5}$', ticker):
                continue
            try:
                si_pct = float(short_int.rstrip("%"))
            except ValueError:
                continue
            stocks.append({
                "ticker": ticker,
                "company": company,
                "exchange": exchange,
                "short_interest_pct": si_pct,
                "float_shares": float_sh,
                "outstanding_shares": outstd,
                "industry": industry,
            })
        if stocks:
            break

    return stocks


def collect_short_interest() -> list[dict]:
    """Fetch and return new short-interest articles (max once per COOLDOWN_HOURS)."""
    cursor = _load_cursor()
    last_run = cursor.get("last_run", 0)
    now_ts = time.time()

    if now_ts - last_run < COOLDOWN_HOURS * 3600:
        return []

    try:
        resp = requests.get(BASE_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[short_interest] fetch error: {exc}")
        return []

    stocks = _parse_stocks(resp.text)
    if not stocks:
        print("[short_interest] no stocks parsed")
        return []

    now_iso = datetime.now(timezone.utc).isoformat()
    conn = _seen_db()
    articles = []

    for s in stocks:
        ticker = s["ticker"]
        title = (
            f"Short Interest Alert: {ticker} ({s['company']}) "
            f"— {s['short_interest_pct']:.1f}% short interest"
        )
        url = f"{BASE_URL}#{ticker}"
        aid = _article_id(url, title)

        if _is_seen(conn, aid):
            continue

        summary = (
            f"{s['company']} ({ticker}) on {s['exchange']} has "
            f"{s['short_interest_pct']:.1f}% short interest. "
            f"Float: {s['float_shares']}, Outstanding: {s['outstanding_shares']}. "
            f"Industry: {s['industry']}. "
            f"High short interest stocks are potential short squeeze candidates."
        )

        articles.append(
            {
                "id": aid,
                "title": title,
                "url": url,
                "summary": summary,
                "source": SOURCE,
                "published": now_iso,
                "tickers": [ticker],
                "extra": {
                    "short_interest_pct": s["short_interest_pct"],
                    "float_shares": s["float_shares"],
                    "outstanding_shares": s["outstanding_shares"],
                    "industry": s["industry"],
                    "exchange": s["exchange"],
                },
            }
        )
        _mark_seen(conn, aid)

    conn.close()
    _save_cursor({"last_run": now_ts, "stocks_found": len(stocks)})
    print(f"[short_interest] fetched {len(stocks)} stocks, {len(articles)} new articles")
    return articles
