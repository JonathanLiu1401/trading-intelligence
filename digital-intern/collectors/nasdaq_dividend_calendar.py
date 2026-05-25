"""Nasdaq dividend & stock-split calendar collector.

Fetches upcoming ex-dividend dates and stock splits from the Nasdaq public
calendar API for the next 5 trading days. Generates synthetic article rows
for high-value events:

  - Large regular dividends (>= $0.50/share) — large payers signal financial
    health and are tracked by income-focused institutions
  - Special / one-time dividends — always notable, often accompany M&A or
    capital returns
  - Stock splits — historically precede short-term price momentum
  - All dividends / splits for portfolio tickers regardless of size

No API key required. Nasdaq's calendar endpoints are public:
  https://api.nasdaq.com/api/calendar/dividends?date=YYYY-MM-DD
  https://api.nasdaq.com/api/calendar/splits?date=YYYY-MM-DD

Like every other collector, ``collect_nasdaq_dividends()`` returns the
standard ``{title, link, summary, published, source}`` dicts.

Two dedup layers:
  1. ``data/seen_articles.db`` (WAL, busy_timeout=30000) keyed by
     sha256(symbol||ex_date||type).
  2. ``articles.db`` PRIMARY KEY = sha256(url||title) inside insert_batch.
"""
import hashlib
import sqlite3
from datetime import date, timedelta, timezone, datetime
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
CONFIG_DIR = BASE_DIR / "config"

FETCH_TIMEOUT = 10
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

NASDAQ_DIV_URL = "https://api.nasdaq.com/api/calendar/dividends"
NASDAQ_SPLIT_URL = "https://api.nasdaq.com/api/calendar/splits"

# Emit an article for dividends >= this threshold (all portfolio tickers always included)
DIV_RATE_THRESHOLD = 0.50
# Look ahead N calendar days
LOOKAHEAD_DAYS = 5

SOURCE_DIV = "nasdaq_dividend_calendar"
SOURCE_SPLIT = "nasdaq_split_calendar"


def _load_portfolio_tickers() -> set[str]:
    """Load tickers from config/tickers.json or config/portfolio.json if present."""
    tickers: set[str] = set()
    for fname in ("tickers.json", "portfolio.json"):
        p = CONFIG_DIR / fname
        if p.exists():
            try:
                import json
                data = json.loads(p.read_text())
                if isinstance(data, list):
                    tickers.update(str(t).upper() for t in data)
                elif isinstance(data, dict):
                    tickers.update(str(t).upper() for t in data.keys())
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


def _article_id(symbol: str, ex_date: str, event_type: str) -> str:
    return hashlib.sha256(f"{symbol}|{ex_date}|{event_type}".encode()).hexdigest()


def _fetch_dividends(dt: date) -> list[dict]:
    try:
        r = requests.get(
            NASDAQ_DIV_URL, params={"date": dt.strftime("%Y-%m-%d")},
            timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA}
        )
        r.raise_for_status()
        rows = (r.json().get("data") or {}).get("calendar", {}).get("rows", []) or []
        return rows
    except Exception as e:
        print(f"[nasdaq_dividend_calendar] fetch dividends {dt}: {e}")
        return []


def _fetch_splits(dt: date) -> list[dict]:
    try:
        r = requests.get(
            NASDAQ_SPLIT_URL, params={"date": dt.strftime("%Y-%m-%d")},
            timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA}
        )
        r.raise_for_status()
        rows = (r.json().get("data") or {}).get("calendar", {}).get("rows", []) or []
        return rows
    except Exception as e:
        print(f"[nasdaq_dividend_calendar] fetch splits {dt}: {e}")
        return []


def collect_nasdaq_dividends() -> list[dict]:
    """Collect upcoming ex-dividend and stock-split events for the next LOOKAHEAD_DAYS."""
    conn = _ensure_db()
    portfolio = _load_portfolio_tickers()
    new_articles: list[dict] = []
    seen_in_run: set[str] = set()
    today = date.today()

    for offset in range(LOOKAHEAD_DAYS + 1):
        dt = today + timedelta(days=offset)

        # --- Dividends ---
        for row in _fetch_dividends(dt):
            symbol = (row.get("symbol") or "").strip().upper()
            company = (row.get("companyName") or symbol).strip()
            ex_date = (row.get("dividend_Ex_Date") or "").strip()
            rate = row.get("dividend_Rate") or 0.0
            payment_date = (row.get("payment_Date") or "").strip()
            announced = (row.get("announcement_Date") or "").strip()

            if not symbol or not ex_date:
                continue

            try:
                rate_f = float(rate)
            except (TypeError, ValueError):
                rate_f = 0.0

            # Only emit if dividend is large or ticker is in portfolio
            if rate_f < DIV_RATE_THRESHOLD and symbol not in portfolio:
                continue

            aid = _article_id(symbol, ex_date, "dividend")
            if aid in seen_in_run:
                continue
            seen_in_run.add(aid)

            try:
                if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
                    (aid, f"https://www.nasdaq.com/market-activity/stocks/{symbol.lower()}/dividend-history",
                     f"[Dividend] {symbol}", SOURCE_DIV,
                     datetime.now(timezone.utc).isoformat()),
                )
            except sqlite3.Error as e:
                print(f"[nasdaq_dividend_calendar] dedup skip: {e}")
                continue

            link = f"https://www.nasdaq.com/market-activity/stocks/{symbol.lower()}/dividend-history"
            title = f"[Dividend] {symbol} — ${rate_f:.4g}/sh ex-date {ex_date}"
            summary = (
                f"{company} ({symbol}) dividend: ${rate_f:.4g}/share. "
                f"Ex-date: {ex_date}. Payment: {payment_date}. "
                f"Announced: {announced}."
            )
            new_articles.append({
                "title": title,
                "link": link,
                "summary": summary,
                "published": ex_date,
                "source": SOURCE_DIV,
            })

        # --- Splits ---
        for row in _fetch_splits(dt):
            symbol = (row.get("symbol") or "").strip().upper()
            company = (row.get("companyName") or symbol).strip()
            ex_date = (row.get("executionDate") or row.get("splitDate") or "").strip()
            ratio = (row.get("optionAdjustmentFactor") or row.get("splitRatio") or "").strip()

            if not symbol:
                continue
            if not ex_date:
                ex_date = dt.strftime("%Y-%m-%d")

            aid = _article_id(symbol, ex_date, "split")
            if aid in seen_in_run:
                continue
            seen_in_run.add(aid)

            try:
                if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
                    (aid, f"https://www.nasdaq.com/market-activity/stocks/{symbol.lower()}",
                     f"[Split] {symbol}", SOURCE_SPLIT,
                     datetime.now(timezone.utc).isoformat()),
                )
            except sqlite3.Error as e:
                print(f"[nasdaq_dividend_calendar] dedup skip split: {e}")
                continue

            link = f"https://www.nasdaq.com/market-activity/stocks/{symbol.lower()}"
            ratio_str = f" {ratio}" if ratio else ""
            title = f"[Stock Split]{ratio_str} {symbol} — {ex_date}"
            summary = f"{company} ({symbol}) stock split{ratio_str} on {ex_date}."
            new_articles.append({
                "title": title,
                "link": link,
                "summary": summary,
                "published": ex_date,
                "source": SOURCE_SPLIT,
            })

    conn.commit()
    conn.close()
    return new_articles


collect = collect_nasdaq_dividends


if __name__ == "__main__":
    print("=== Nasdaq Dividend & Split Calendar (live fetch) ===")
    items = collect_nasdaq_dividends()
    print(f"New items: {len(items)}")
    for art in items[:8]:
        print(f"  {art['published']:12s} {art['title'][:80]}")
    if items:
        print(f"\nExample: {items[0]['title']}")
        print(f"Summary: {items[0]['summary']}")
