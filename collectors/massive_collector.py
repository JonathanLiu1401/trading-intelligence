"""Massive.com news collector — per-ticker, round-robin.

Massive.com offers a Polygon-compatible REST API for stocks/options/futures
news + market data. We poll its news endpoint per portfolio/watchlist ticker.

Endpoint: GET https://api.massive.com/v2/reference/news?ticker=NVDA&apiKey=KEY

Skips silently if MASSIVE_API_KEY is unset.
"""
import hashlib
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
CURSOR_PATH = BASE_DIR / "data" / "massive_cursor.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

API_URL = "https://api.massive.com/v2/reference/news"
# 18 tickers per 10-min pass = ~108 polls/hour, comfortably under typical
# 1000 req/min plans; cycles through ~50 portfolio+watchlist tickers in ~30min.
BATCH_PER_PASS = 18
PER_TICKER_COOLDOWN_SEC = 600
HTTP_TIMEOUT = 10
_KEY_WARNED = False


def _load_tickers() -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()

    def _add(t: str):
        u = (t or "").strip().upper()
        if u and u not in seen and "." not in u and "=" not in u and "^" not in u:
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
        for key in ("memory_core", "semis_equipment", "broader_semis", "portfolio"):
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
    # source_health.py / article_store.py. 11 collectors share this one file;
    # SQLite's default busy_timeout=0 turns any transient cross-writer lock
    # into an immediate OperationalError that aborts the whole pass and drops
    # the fetched batch. WAL + 30s timeout lets the write wait out contention.
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


def _fetch_ticker(key: str, ticker: str) -> list:
    params = {"ticker": ticker, "limit": 50, "order": "desc",
              "sort": "published_utc", "apiKey": key}
    try:
        r = requests.get(API_URL, params=params, timeout=HTTP_TIMEOUT)
    except Exception as e:
        print(f"[massive] {ticker} fetch error: {e}")
        return []
    if r.status_code == 429:
        print(f"[massive] {ticker} rate-limited (429)")
        return []
    if r.status_code != 200:
        return []
    try:
        data = r.json()
    except Exception:
        return []

    out: list = []
    for it in data.get("results", []):
        title = (it.get("title") or "").strip()
        link = (it.get("article_url") or "").strip()
        if not title or not link:
            continue
        summary = (it.get("description") or "").strip()
        published = (it.get("published_utc") or "").strip()
        pub = it.get("publisher") or {}
        publisher = pub.get("name", "unknown") if isinstance(pub, dict) else "unknown"
        out.append({
            "title": title,
            "link": link,
            "summary": summary,
            "published": published,
            "source": f"Massive/{publisher}",
            "_ticker": ticker,
        })
    return out


def collect_massive(batch: int = BATCH_PER_PASS) -> list:
    global _KEY_WARNED
    key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if not key:
        if not _KEY_WARNED:
            print("[massive] MASSIVE_API_KEY not set — skipping. "
                  "Get a key at https://massive.com")
            _KEY_WARNED = True
        return []

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
    if selected:
        with ThreadPoolExecutor(max_workers=min(len(selected), 12)) as ex:
            futures = {ex.submit(_fetch_ticker, key, t): t for t in selected}
            for fut in as_completed(futures):
                try:
                    raw.append(fut.result())
                except Exception as e:
                    print(f"[massive] worker error: {e}")

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
                (aid, art["link"], art["title"], "Massive", datetime.utcnow().isoformat()),
            )

    state["index"] = idx
    state["last_polled"] = last_polled
    _save_cursor(state)

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    items = collect_massive(batch=3)
    print(f"Got {len(items)} new Massive articles")
    for a in items[:10]:
        print(f"  [{a['_ticker']}] {a['title'][:80]}")
