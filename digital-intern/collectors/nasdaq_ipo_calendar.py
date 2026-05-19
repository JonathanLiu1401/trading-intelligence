"""Nasdaq IPO calendar collector — synthetic 'article' rows for upcoming,
priced, and filed IPOs from Nasdaq's public JSON endpoint.

    https://api.nasdaq.com/api/ipo/calendar?date=YYYY-MM

No API key. Fetches the current month + next month, emits one article per
deal across the upcoming / priced / filed / withdrawn buckets. Same
pipeline as every other collector: returns standard
{title, link, summary, published, source} dicts and the daemon's
_ingest (or __main__ here) inserts via ArticleStore.insert_batch.

Dedup follows the rss_collector / sec_edgar / fred_collector convention:
  1. seen_articles.db (WAL, busy_timeout=30000) keyed by
     dealID|bucket so a deal moving from filed -> priced still emits once
     per status transition.
  2. articles.db PRIMARY KEY = sha256(url||title) inside insert_batch.
"""
import hashlib
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

NASDAQ_IPO_URL = "https://api.nasdaq.com/api/ipo/calendar?date={ym}"
SOURCE = "nasdaq/ipo_calendar"
FETCH_TIMEOUT = 15

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}

# Bucket -> (display label, date field) for Nasdaq's response shape.
BUCKETS = {
    "upcoming": ("Upcoming", "expectedPriceDate"),
    "priced": ("Priced", "pricedDate"),
    "filed": ("Filed", "filedDate"),
    "withdrawn": ("Withdrawn", "withdrawDate"),
}


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


def _seen_id(deal_id: str, bucket: str) -> str:
    raw = f"nasdaq_ipo:{deal_id}:{bucket}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_seen(conn: sqlite3.Connection, sid: str) -> bool:
    cur = conn.execute("SELECT 1 FROM seen_articles WHERE id=? LIMIT 1", (sid,))
    return cur.fetchone() is not None


def _mark_seen_batch(conn, rows: list[tuple]) -> None:
    if not rows:
        return
    now = datetime.now(timezone.utc).isoformat()
    rows_with_ts = [(sid, link, title, SOURCE, now) for (sid, link, title) in rows]
    # Single transaction + retry — seen_articles.db is shared across collectors;
    # WAL+busy_timeout handles most contention but burst writes from other
    # collectors can still trip locks, so we retry briefly.
    import time as _t
    for attempt in range(6):
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                "VALUES (?, ?, ?, ?, ?)",
                rows_with_ts,
            )
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < 5:
                _t.sleep(0.5 * (attempt + 1))
                continue
            raise


def _months_to_fetch() -> list[str]:
    today = date.today()
    ym1 = today.strftime("%Y-%m")
    nxt_month = today.month + 1
    nxt_year = today.year
    if nxt_month > 12:
        nxt_month = 1
        nxt_year += 1
    ym2 = f"{nxt_year:04d}-{nxt_month:02d}"
    return [ym1, ym2]


def _fetch_month(ym: str) -> dict:
    url = NASDAQ_IPO_URL.format(ym=ym)
    r = requests.get(url, headers=_HEADERS, timeout=FETCH_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    return j.get("data") or {}


def _format_article(row: dict, bucket: str, ym: str) -> dict | None:
    label, date_field = BUCKETS[bucket]
    ticker = (row.get("proposedTickerSymbol") or "").strip()
    name = (row.get("companyName") or "").strip()
    if not name:
        return None
    exch = (row.get("proposedExchange") or "").strip()
    price = (row.get("proposedSharePrice") or "").strip()
    shares = (row.get("sharesOffered") or "").strip()
    offer = (row.get("dollarValueOfSharesOffered") or "").strip()
    when = (row.get(date_field) or "").strip()

    tk = f"[{ticker}] " if ticker else ""
    title = f"IPO {label}: {tk}{name}"
    extras = []
    if when:
        extras.append(when)
    if exch:
        extras.append(exch)
    if price:
        extras.append(f"${price}")
    if shares:
        extras.append(f"{shares} sh")
    if offer:
        extras.append(offer)
    if extras:
        title += " (" + ", ".join(extras) + ")"

    summary = (
        f"Nasdaq IPO calendar — bucket={label}. Company: {name}"
        f"{' (' + ticker + ')' if ticker else ''}"
        f"{', exchange ' + exch if exch else ''}"
        f"{', proposed price $' + price if price else ''}"
        f"{', shares offered ' + shares if shares else ''}"
        f"{', offer ' + offer if offer else ''}"
        f"{', date ' + when if when else ''}."
    )

    link = f"https://www.nasdaq.com/market-activity/ipos?date={ym}"
    if ticker:
        link = f"https://www.nasdaq.com/market-activity/stocks/{ticker.lower()}"

    return {
        "title": title,
        "link": link,
        "summary": summary,
        "published": when or ym,
        "source": SOURCE,
        "_deal_id": str(row.get("dealID") or ""),
        "_bucket": bucket,
    }


def collect_nasdaq_ipo() -> list[dict]:
    conn = _ensure_db()
    new_articles: list[dict] = []
    pending_seen: list[tuple] = []
    for ym in _months_to_fetch():
        try:
            data = _fetch_month(ym)
        except Exception as e:
            print(f"  fetch {ym} failed: {e}")
            continue
        for bucket in BUCKETS:
            block = data.get(bucket) or {}
            rows = block.get("rows") or []
            for row in rows:
                deal_id = str(row.get("dealID") or "").strip()
                if not deal_id:
                    continue
                sid = _seen_id(deal_id, bucket)
                if _is_seen(conn, sid):
                    continue
                art = _format_article(row, bucket, ym)
                if not art:
                    continue
                new_articles.append(art)
                pending_seen.append((sid, art["link"], art["title"]))
    _mark_seen_batch(conn, pending_seen)
    return new_articles


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))

    print("=== Nasdaq IPO calendar collector ===")
    months = _months_to_fetch()
    print(f"Months: {months}")
    for ym in months:
        try:
            data = _fetch_month(ym)
            counts = {b: len((data.get(b) or {}).get("rows") or []) for b in BUCKETS}
            print(f"  {ym}: {counts}")
        except Exception as e:
            print(f"  {ym}: fetch failed: {e}")

    items = collect_nasdaq_ipo()
    inserted = 0
    if items:
        from storage.article_store import ArticleStore
        store = ArticleStore()
        inserted = store.insert_batch(items)

    print("\n=== Summary ===")
    print(f"New synthetic articles built : {len(items)}")
    print(f"Total new items inserted into articles.db : {inserted}")
    eg_line = None
    for a in items[:8]:
        print(f"  + {a['title']}")
        if eg_line is None:
            eg_line = a["title"]
    if eg_line:
        print(f"DISCORD_EG: {eg_line}")
