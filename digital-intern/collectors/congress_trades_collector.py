"""Congressional stock trading disclosure collector via Quiver Quantitative.

Fetches the latest congressional trading disclosures from the public
Quiver Quant API. Each trade is surfaced as an article-like record with
the ticker in _tickers for relevance scoring downstream.

No API key required. Deduplicates via seen_articles.db.
"""
import hashlib
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

API_URL = "https://api.quiverquant.com/beta/live/congresstrading"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
LOOKBACK_DAYS = 30


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


def _article_id(rep: str, ticker: str, transaction: str, tx_date: str) -> str:
    key = f"congress||{rep}||{ticker}||{transaction}||{tx_date}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def collect_congress_trades(lookback_days: int = LOOKBACK_DAYS) -> list:
    try:
        resp = requests.get(
            API_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        trades = resp.json()
    except Exception as e:
        print(f"[congress_trades] fetch error: {e}")
        return []

    if not isinstance(trades, list):
        print(f"[congress_trades] unexpected response type: {type(trades)}")
        return []

    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    conn = _ensure_db()
    new_articles: list = []

    for trade in trades:
        report_date = trade.get("ReportDate") or ""
        tx_date = trade.get("TransactionDate") or report_date
        if report_date < cutoff:
            continue

        rep = trade.get("Representative") or "Unknown"
        ticker = (trade.get("Ticker") or "").upper().strip()
        transaction = trade.get("Transaction") or "Unknown"
        amount_range = trade.get("Range") or "Unknown"
        party = trade.get("Party") or "?"
        house = trade.get("House") or "Congress"
        excess_return = trade.get("ExcessReturn")

        if not ticker:
            continue

        aid = _article_id(rep, ticker, transaction, tx_date)
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
            continue

        action_word = "buys" if "purchase" in transaction.lower() else "sells"
        excess_str = ""
        if excess_return is not None:
            excess_str = f" (excess return: {excess_return:+.1f}%)"
        title = (
            f"Congress: {rep} ({party}) {action_word} {ticker} "
            f"{amount_range} — reported {report_date}{excess_str}"
        )
        link = f"https://efts.sec.gov/LATEST/search-index?q={ticker}&forms=4"

        art = {
            "title": title,
            "link": link,
            "summary": (
                f"{rep} ({party}, {house}) disclosed a {transaction} of {ticker} "
                f"worth {amount_range} on {tx_date}. Filed {report_date}."
            ),
            "published": report_date,
            "source": "CongressTrades/QuiverQuant",
            "_tickers": [ticker] if ticker else [],
        }
        new_articles.append(art)
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (aid, link, title, "CongressTrades", datetime.now(timezone.utc).isoformat()),
        )

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    t0 = time.time()
    items = collect_congress_trades()
    dt = time.time() - t0
    print(f"[congress_trades] {len(items)} new items in {dt:.1f}s")
    for a in items[:10]:
        tk = ",".join(a["_tickers"]) or "-"
        print(f"  [{tk}] {a['title'][:100]}")
