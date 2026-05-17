"""Finviz quote-page news scraper — per-ticker, round-robin.

Finviz publishes a curated headline table on every quote page
(https://finviz.com/quote.ashx?t=NVDA, ``<table id="news-table">``). It
aggregates wires, blogs and majors with tighter ticker relevance than a raw
RSS search. No API key; Finviz 403s the default Python UA so a real browser
User-Agent + Accept headers are mandatory.

Mirrors the google_news / yahoo_ticker_rss collector contract exactly:
dedup against the shared ``data/seen_articles.db`` via
``_article_id(link, title)`` and return the net-new ``list[dict]``. The
daemon's ``_ingest()`` does the articles.db insert and
``source_health.record_result`` is recorded by the worker, not here.
"""
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
CURSOR_PATH = BASE_DIR / "data" / "finviz_cursor.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

# Finviz is more rate-sensitive than RSS feeds — fetch serially with a small
# sleep between tickers and keep the per-pass fan-out small.
BATCH_PER_PASS = 6
PER_TICKER_COOLDOWN_SEC = 600
SLEEP_BETWEEN_TICKERS_SEC = 1.0
REQUEST_TIMEOUT = 10

# Default Python UA gets a 403 from Finviz — a modern browser UA + Accept
# headers are required.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _load_tickers() -> list[str]:
    """Positions + options underlyings + sector watchlist + watchlist.json,
    dedup-preserved. Identical sourcing to yahoo_ticker_rss._load_tickers."""
    tickers: list[str] = []
    seen: set[str] = set()

    def _add(t: str):
        u = (t or "").strip().upper()
        if u and u not in seen:
            seen.add(u)
            tickers.append(u)

    try:
        with open(PORTFOLIO_PATH, "r") as f:
            pf = json.load(f)
        for pos in pf.get("positions", []):
            _add(pos.get("ticker", ""))
        for opt in pf.get("options", []):
            _add(opt.get("underlying", ""))
        for t in pf.get("sector_watchlist", []):
            _add(t)
    except Exception:
        pass

    try:
        with open(WATCHLIST_PATH, "r") as f:
            wl = json.load(f)
        for key in ("memory_core", "semis_equipment", "broader_semis", "portfolio",
                    "korean", "japanese", "etfs"):
            for t in wl.get(key, []):
                _add(t)
    except Exception:
        pass

    return tickers


def _load_cursor() -> dict:
    if CURSOR_PATH.exists():
        try:
            with open(CURSOR_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"index": 0, "last_polled": {}}


def _save_cursor(state: dict):
    CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CURSOR_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(CURSOR_PATH)


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Hardened seen_articles.db connection — mirrors google_news._ensure_db /
    # yahoo_ticker_rss._ensure_db / source_health.py / article_store.py. Many
    # collectors share this one file; SQLite's default busy_timeout=0 turns
    # any transient cross-writer lock into an immediate OperationalError that
    # aborts the whole pass and drops the fetched batch. WAL + 30s timeout
    # lets the write wait out contention.
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


def _article_id(link: str, title: str) -> str:
    return hashlib.sha256(f"{link}||{title}".encode("utf-8")).hexdigest()


def _parse_published(raw: str, last_date: str) -> tuple[str, str]:
    """Finviz timestamps are sticky: a day-change row carries the full
    ``"May-16-26 10:00PM"``; later rows the same day show only ``"10:00AM"``.
    The newest rows use relative ``"Today 10:00AM"`` / ``"Yesterday ..."``.
    Resolve the relative labels and carry the last ``Mon-DD-YY`` prefix into
    time-only rows.

    Returns (iso_published, new_last_date). On any parse failure the ISO
    string is empty (never crash a row) but the date carry-forward is kept.
    """
    raw = (raw or "").strip()
    if not raw:
        return "", last_date
    parts = raw.split()
    if len(parts) == 2:           # "May-16-26 10:00PM" / "Today 10:00AM"
        lead, time_part = parts[0], parts[1]
        if lead == "Today":
            date_part = datetime.now().strftime("%b-%d-%y")
        elif lead == "Yesterday":
            date_part = (datetime.now() - timedelta(days=1)).strftime("%b-%d-%y")
        else:
            date_part = lead      # expected "Mon-DD-YY"
        last_date = date_part
    elif len(parts) == 1:         # "10:00AM" — reuse the carried date
        date_part, time_part = last_date, parts[0]
    else:
        return "", last_date
    if not date_part:
        return "", last_date
    try:
        dt = datetime.strptime(f"{date_part} {time_part}", "%b-%d-%y %I:%M%p")
        return dt.isoformat(), last_date
    except (ValueError, TypeError):
        return "", last_date


def _fetch_ticker(ticker: str) -> list:
    """Fetch + parse the Finviz news table for one ticker. Returns raw
    entries (no dedup yet). Never raises — returns [] on any failure."""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            print(f"[finviz] {ticker} HTTP {resp.status_code}")
            return []
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"[finviz] {ticker} fetch error: {e}")
        return []

    table = soup.find("table", id="news-table")
    if table is None:
        print(f"[finviz] {ticker} no news-table in markup")
        return []

    out: list = []
    last_date = ""
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        published, last_date = _parse_published(cells[0].get_text(), last_date)
        a = cells[1].find("a", class_="tab-link-news")
        if a is None:
            continue
        title = a.get_text(strip=True)
        href = (a.get("href") or "").strip()
        if not title or not href:
            continue
        link = urljoin("https://finviz.com/", href)
        out.append({
            "title": title,
            "link": link,
            "summary": "",
            "published": published,
            "source": f"finviz/{ticker}",
            "_ticker": ticker,
        })
    return out


def collect_finviz(batch: int = BATCH_PER_PASS) -> list:
    """Collect one round-robin batch of Finviz quote-page headlines.

    Serial fetch (Finviz is rate-sensitive) with a small inter-ticker sleep;
    one ticker failing never aborts the pass. Dedup against seen_articles.db
    and return only net-new articles, matching the google_news / yahoo
    collector contract.
    """
    tickers = _load_tickers()
    if not tickers:
        return []

    state = _load_cursor()
    idx = int(state.get("index", 0)) % len(tickers)
    last_polled: dict = state.get("last_polled") or {}
    now = time.time()

    selected: list[str] = []
    attempts = 0
    max_attempts = max(batch * 3, len(tickers))
    while len(selected) < batch and attempts < max_attempts:
        ticker = tickers[idx]
        idx = (idx + 1) % len(tickers)
        attempts += 1
        last = float(last_polled.get(ticker, 0))
        if now - last < PER_TICKER_COOLDOWN_SEC:
            continue
        selected.append(ticker)
        last_polled[ticker] = now

    raw: list[list] = []
    for i, ticker in enumerate(selected):
        try:
            raw.append(_fetch_ticker(ticker))
        except Exception as e:
            print(f"[finviz] {ticker} worker error: {e}")
        if i < len(selected) - 1:
            time.sleep(SLEEP_BETWEEN_TICKERS_SEC)

    conn = _ensure_db()
    new_articles: list = []
    seen_in_run: set = set()
    for entries in raw:
        for art in entries:
            aid = _article_id(art["link"], art["title"])
            if aid in seen_in_run:
                continue
            seen_in_run.add(aid)
            if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
                continue
            new_articles.append(art)
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                "VALUES (?, ?, ?, ?, ?)",
                (aid, art["link"], art["title"], "Finviz",
                 datetime.now(timezone.utc).isoformat()),
            )

    state["index"] = idx
    state["last_polled"] = last_polled
    _save_cursor(state)

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    items = collect_finviz(batch=3)
    print(f"Got {len(items)} new Finviz articles")
    for a in items[:10]:
        print(f"  [{a['_ticker']}] {a['title'][:90]}")
        print(f"      {a['link']}  ({a['published']})")
