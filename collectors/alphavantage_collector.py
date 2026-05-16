"""Alpha Vantage NEWS_SENTIMENT collector — per-ticker, round-robin.

Free tier: 25 calls/day total. Keep BATCH_PER_PASS small and PER_TICKER_COOLDOWN_SEC
high so a 24h window comfortably fits in the quota.

Endpoint: GET https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=NVDA&apikey=KEY

Skips silently if ALPHA_VANTAGE_KEY is unset.
"""
import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
CURSOR_PATH = BASE_DIR / "data" / "alphavantage_cursor.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

API_URL = "https://www.alphavantage.co/query"
# Free tier = 25 req/day → 1 ticker per call, every ~1h means 24 calls/day. Conservative.
BATCH_PER_PASS = 1
PER_TICKER_COOLDOWN_SEC = 6 * 3600
HTTP_TIMEOUT = 12
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
        for key in ("memory_core", "semis_equipment", "portfolio"):
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
    params = {"function": "NEWS_SENTIMENT", "tickers": ticker, "apikey": key, "limit": 50}
    try:
        r = requests.get(API_URL, params=params, timeout=HTTP_TIMEOUT)
    except Exception as e:
        print(f"[alphavantage] {ticker} fetch error: {e}")
        return []
    if r.status_code != 200:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    # Quota-exceeded responses come back as {"Information": "..."} or {"Note": "..."}
    if "feed" not in data:
        msg = data.get("Information") or data.get("Note") or ""
        if msg:
            print(f"[alphavantage] {ticker}: {msg[:120]}")
        return []

    out: list = []
    for it in data.get("feed", []):
        title = (it.get("title") or "").strip()
        link = (it.get("url") or "").strip()
        if not title or not link:
            continue
        summary = (it.get("summary") or "").strip()
        published = (it.get("time_published") or "").strip()
        # Alpha Vantage format: 20260513T143000 — leave as-is or attempt to parse
        out.append({
            "title": title,
            "link": link,
            "summary": summary,
            "published": published,
            "source": f"AlphaVantage/{it.get('source', 'unknown')}",
            "_ticker": ticker,
        })
    return out


def collect_alphavantage(batch: int = BATCH_PER_PASS) -> list:
    global _KEY_WARNED
    key = os.environ.get("ALPHA_VANTAGE_KEY", "").strip()
    if not key:
        if not _KEY_WARNED:
            print("[alphavantage] ALPHA_VANTAGE_KEY not set — skipping. "
                  "Get a free key at https://www.alphavantage.co/support/#api-key")
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
    # Single-threaded — quota is very tight, no parallelism gain.
    for t in selected:
        raw.append(_fetch_ticker(key, t))

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
                (aid, art["link"], art["title"], "AlphaVantage", datetime.utcnow().isoformat()),
            )

    state["index"] = idx
    state["last_polled"] = last_polled
    _save_cursor(state)

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    items = collect_alphavantage(batch=1)
    print(f"Got {len(items)} new Alpha Vantage articles")
    for a in items[:10]:
        print(f"  [{a['_ticker']}] {a['title'][:80]}")
