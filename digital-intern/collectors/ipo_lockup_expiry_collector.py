"""IPO Lockup Expiry Collector — alerts when insider lockup windows expire.

After a company IPOs, insiders and early investors are locked up for ~180 days.
When that window expires, there's often selling pressure as insiders can sell.
This collector fetches priced IPOs from the past 6 months via Nasdaq's public
API, computes the standard 180-day lockup expiry date, and emits synthetic
alert articles for expirations within the next 14 days.

Data source: api.nasdaq.com/api/ipo/calendar (no API key, public JSON)
Dedup: seen_articles.db keyed by (ticker, expiry_date) — one alert per deal.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

log = logging.getLogger("ipo_lockup_expiry")

BASE_DIR = Path(__file__).resolve().parent.parent
SEEN_DB = BASE_DIR / "data" / "seen_articles.db"

SOURCE_NAME = "ipo/lockup_expiry"
FETCH_TIMEOUT = 15
LOCKUP_DAYS = 180       # standard IPO lockup
ALERT_WINDOW_DAYS = 14  # alert this many days before expiry

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}

NASDAQ_IPO_URL = "https://api.nasdaq.com/api/ipo/calendar?date={ym}"


def _ensure_db() -> sqlite3.Connection:
    SEEN_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SEEN_DB), timeout=30, check_same_thread=False)
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


def _seen_id(ticker: str, expiry_date: str) -> str:
    raw = f"ipo_lockup:{ticker}:{expiry_date}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_seen(conn: sqlite3.Connection, sid: str) -> bool:
    cur = conn.execute("SELECT 1 FROM seen_articles WHERE id=? LIMIT 1", (sid,))
    return cur.fetchone() is not None


def _mark_seen(conn: sqlite3.Connection, sid: str, ticker: str, expiry_date: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    title = f"IPO Lockup Expiry: {ticker} on {expiry_date}"
    link = "https://www.nasdaq.com/market-activity/ipos"
    try:
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles(id, link, title, source, first_seen) "
            "VALUES (?,?,?,?,?)",
            (sid, link, title, SOURCE_NAME, now),
        )
        conn.commit()
    except sqlite3.OperationalError as exc:
        log.warning("[ipo_lockup_expiry] _mark_seen db error (will retry next pass): %s", exc)


def _parse_date(s: str) -> date | None:
    """Parse MM/DD/YYYY or YYYY-MM-DD strings."""
    if not s:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    return None


def _fetch_priced_ipos(months: list[str]) -> list[dict]:
    """Return priced IPO rows from Nasdaq calendar for given YYYY-MM months."""
    all_deals: list[dict] = []
    for ym in months:
        url = NASDAQ_IPO_URL.format(ym=ym)
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=FETCH_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            rows = (
                data.get("data", {})
                    .get("priced", {})
                    .get("rows") or []
            )
            all_deals.extend(rows)
        except Exception as exc:
            log.warning("[ipo_lockup_expiry] fetch error for %s: %s", ym, exc)
    return all_deals


def collect_ipo_lockup_expiry() -> list[dict]:
    """Collect upcoming IPO lockup expiry alerts.

    Returns a list of article dicts compatible with ArticleStore.insert_batch.
    """
    today = date.today()
    alert_cutoff = today + timedelta(days=ALERT_WINDOW_DAYS)

    # Fetch the last 7 months (covers ~210 days = lockup window + buffer)
    months: list[str] = []
    pivot = today.replace(day=1)
    for _ in range(7):
        months.append(pivot.strftime("%Y-%m"))
        # Step back one month
        if pivot.month == 1:
            pivot = pivot.replace(year=pivot.year - 1, month=12)
        else:
            pivot = pivot.replace(month=pivot.month - 1)

    priced = _fetch_priced_ipos(months)
    log.info("[ipo_lockup_expiry] fetched %d priced IPO rows", len(priced))

    conn = _ensure_db()
    articles: list[dict] = []

    for row in priced:
        ticker = (row.get("proposedTickerSymbol") or "").strip().upper()
        company = (row.get("companyName") or "").strip()
        priced_date_str = (row.get("pricedDate") or "").strip()
        if not ticker or not priced_date_str:
            continue

        priced_dt = _parse_date(priced_date_str)
        if not priced_dt:
            continue

        expiry_dt = priced_dt + timedelta(days=LOCKUP_DAYS)

        # Skip if already expired or beyond our alert window
        if expiry_dt < today or expiry_dt > alert_cutoff:
            continue

        expiry_str = expiry_dt.strftime("%Y-%m-%d")
        days_left = (expiry_dt - today).days

        sid = _seen_id(ticker, expiry_str)
        if _is_seen(conn, sid):
            continue

        urgency = "TODAY" if days_left == 0 else f"in {days_left}d"
        title = (
            f"IPO Lockup Expiry {urgency}: {ticker} ({company}) "
            f"— {LOCKUP_DAYS}-day window closes {expiry_str}"
        )
        summary = (
            f"{company} ({ticker}) IPO priced on {priced_date_str}. "
            f"Standard {LOCKUP_DAYS}-day insider lockup expires {expiry_str} "
            f"({days_left} days from today). Insiders and early investors become "
            f"free to sell. Watch for elevated selling pressure and volume near expiry."
        )
        link = f"https://www.nasdaq.com/market-activity/stocks/{ticker.lower()}"

        articles.append({
            "title": title,
            "link": link,
            "summary": summary,
            "published": datetime.now(timezone.utc).isoformat(),
            "source": SOURCE_NAME,
        })

        _mark_seen(conn, sid, ticker, expiry_str)
        log.info("[ipo_lockup_expiry] alert: %s expiry=%s", ticker, expiry_str)

    conn.close()
    log.info("[ipo_lockup_expiry] emitting %d lockup expiry alerts", len(articles))
    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = collect_ipo_lockup_expiry()
    print(f"\n=== IPO Lockup Expiry Collector ===")
    print(f"Alerts found: {len(results)}")
    for a in results:
        print(f"\n  TICKER: {a['source']}")
        print(f"  TITLE:  {a['title']}")
        print(f"  LINK:   {a['link']}")
        print(f"  SUMMARY: {a['summary']}")
