"""NASDAQ Earnings Calendar collector — broad market upcoming earnings events.

Fetches the next 5 trading days of earnings reports from Nasdaq's public
JSON endpoint and emits synthetic article rows so the ML pipeline sees
upcoming EPS events for the full market (not just portfolio tickers).

Complements earnings_calendar.py (portfolio/yfinance, returns dict) with:
 - Full market coverage (50-200 companies per day)
 - EPS forecast vs. prior year comparison in the article summary
 - Pre/after-market timing label
 - Stored as articles so scoring + signals pipelines see them

Dedup: seen_articles.db keyed by sha256(symbol||reportDate) so reruns
across the same day don't repeat rows, but a ticker with a date correction
(Nasdaq updates their calendar) still emits again.

API: https://api.nasdaq.com/api/calendar/earnings?date=YYYY-MM-DD
No API key required. Rate-limit: 1 request per trading day fetched.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

SOURCE = "nasdaq/earnings_calendar"
FETCH_TIMEOUT = 12
TRADING_DAYS_AHEAD = 5  # fetch Mon-Fri for the next N trading days

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}

_NASDAQ_EARNINGS_URL = "https://api.nasdaq.com/api/calendar/earnings?date={date}"


def _next_trading_days(n: int) -> list[date]:
    """Return next n weekdays (Mon-Fri) including today if it's a weekday."""
    days: list[date] = []
    d = date.today()
    while len(days) < n:
        if d.weekday() < 5:  # Mon=0 .. Fri=4
            days.append(d)
        d += timedelta(days=1)
    return days


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


def _seen_key(symbol: str, report_date: str) -> str:
    raw = f"nasdaq_earnings|{symbol}|{report_date}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _already_seen(conn: sqlite3.Connection, key: str) -> bool:
    row = conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (key,)).fetchone()
    return row is not None


def _mark_seen(conn: sqlite3.Connection, key: str, link: str, title: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
        (key, link, title, SOURCE, now),
    )


def _parse_market_cap(raw: str) -> str:
    """Format raw market cap string (e.g. '$1,234,567,890') to '$1.23B'."""
    if not raw or raw.strip() in ("", "N/A"):
        return ""
    digits = raw.replace("$", "").replace(",", "").strip()
    try:
        val = float(digits)
        if val >= 1e12:
            return f"${val/1e12:.2f}T"
        if val >= 1e9:
            return f"${val/1e9:.2f}B"
        if val >= 1e6:
            return f"${val/1e6:.2f}M"
        return raw
    except ValueError:
        return raw


def _timing_label(time_field: str) -> str:
    mapping = {
        "time-pre-market": "before open",
        "time-after-hours": "after close",
        "time-not-supplied": "time TBD",
    }
    return mapping.get(time_field, time_field)


def _build_article(row: dict, report_date: str) -> dict:
    symbol = row.get("symbol", "").upper()
    name = row.get("name", symbol)
    timing = _timing_label(row.get("time", ""))
    eps_forecast = row.get("epsForecast", "") or ""
    last_eps = row.get("lastYearEPS", "") or ""
    mktcap = _parse_market_cap(row.get("marketCap", ""))
    fiscal_qtr = row.get("fiscalQuarterEnding", "")
    num_ests = row.get("noOfEsts", "")

    # Build a descriptive title
    title_parts = [f"{symbol} ({name}) reports earnings {report_date} {timing}"]
    detail_parts = []
    if eps_forecast and eps_forecast not in ("", "N/A"):
        detail_parts.append(f"EPS forecast {eps_forecast}")
    if last_eps and last_eps not in ("", "N/A"):
        detail_parts.append(f"vs {last_eps} last year")
    if fiscal_qtr:
        detail_parts.append(f"Q: {fiscal_qtr}")
    if mktcap:
        detail_parts.append(f"mktcap {mktcap}")
    if num_ests:
        detail_parts.append(f"{num_ests} estimates")

    title = " | ".join(title_parts)
    summary = "; ".join(detail_parts) if detail_parts else f"{name} upcoming earnings"
    link = f"https://www.nasdaq.com/market-activity/earnings?date={report_date}#{symbol}"

    return {
        "title": title,
        "link": link,
        "summary": summary,
        "published": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE,
        "tickers": [symbol] if symbol else [],
    }


def collect() -> list[dict]:
    """Fetch earnings for next TRADING_DAYS_AHEAD weekdays, return new articles."""
    conn = _ensure_db()
    results: list[dict] = []

    for trading_day in _next_trading_days(TRADING_DAYS_AHEAD):
        date_str = trading_day.strftime("%Y-%m-%d")
        url = _NASDAQ_EARNINGS_URL.format(date=date_str)

        try:
            resp = requests.get(url, headers=_HEADERS, timeout=FETCH_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[nasdaq_earnings_calendar] {date_str}: fetch error {e}")
            continue

        rows = (data.get("data") or {}).get("rows") or []
        if not rows:
            # Weekend or holiday — NASDAQ returns null rows
            continue

        day_new = 0
        for row in rows:
            symbol = (row.get("symbol") or "").upper()
            if not symbol:
                continue
            key = _seen_key(symbol, date_str)
            if _already_seen(conn, key):
                continue

            article = _build_article(row, date_str)
            _mark_seen(conn, key, article["link"], article["title"])
            results.append(article)
            day_new += 1

        conn.commit()
        print(f"[nasdaq_earnings_calendar] {date_str}: {len(rows)} companies, {day_new} new")

    conn.close()
    return results


if __name__ == "__main__":
    from storage.article_store import ArticleStore

    items = collect()
    print(f"\nCollected {len(items)} new earnings events")
    if items:
        print("\nSample articles:")
        for art in items[:5]:
            print(f"  {art['title']}")
            print(f"    {art['summary']}")
        store = ArticleStore()
        inserted = store.insert_batch(items)
        print(f"\nInserted into articles.db: {inserted}")
    else:
        print("No new events (all already seen or no data)")
