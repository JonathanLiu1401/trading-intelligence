"""Yahoo Finance per-ticker RSS collector.

No API key. Pull https://finance.yahoo.com/rss/headline?s=NVDA for every portfolio
+ watchlist ticker on a round-robin schedule.
"""
import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
CURSOR_PATH = BASE_DIR / "data" / "yahoo_ticker_cursor.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

BATCH_PER_PASS = 10
PER_TICKER_COOLDOWN_SEC = 240
REQUEST_TIMEOUT = 8
MAX_FETCH_ATTEMPTS = 3
RETRY_BACKOFF_SEC = 1.0
RETRY_BACKOFF_CAP_SEC = 8.0
USER_AGENT = "Mozilla/5.0 (Digital Intern Daemon)"


def _load_tickers() -> list[str]:
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


def _retry_after_seconds(resp, fallback: float) -> float:
    raw = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
    try:
        seconds = float(raw) if raw is not None else fallback
    except (TypeError, ValueError):
        seconds = fallback
    if seconds <= 0:
        seconds = fallback
    return min(seconds, RETRY_BACKOFF_CAP_SEC)


def _parse_entries(ticker: str, parsed) -> list:
    out: list = []
    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue
        summary = entry.get("summary") or entry.get("description") or ""
        published = entry.get("published") or entry.get("updated") or ""
        out.append({
            "title": title,
            "link": link,
            "summary": summary,
            "published": published,
            "source": f"YahooFinance/{ticker}",
            "_ticker": ticker,
        })
    return out


def _fetch_ticker(ticker: str) -> list:
    url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    backoff = RETRY_BACKOFF_SEC
    last_error = "unknown"
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        retry_delay = backoff
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                retry_delay = _retry_after_seconds(resp, backoff)
                raise RuntimeError(f"HTTP {resp.status_code}")
            parsed = feedparser.parse(resp.content)
            if getattr(parsed, "bozo", 0) and not parsed.entries:
                bozo_exc = getattr(parsed, "bozo_exception", None)
                detail = f": {bozo_exc}" if bozo_exc else ""
                raise RuntimeError(f"feedparser bozo parse with no entries{detail}")
            return _parse_entries(ticker, parsed)
        except Exception as e:
            last_error = str(e)
            if attempt >= MAX_FETCH_ATTEMPTS:
                break
            print(
                f"[yahoo_ticker_rss] {ticker} attempt {attempt}/"
                f"{MAX_FETCH_ATTEMPTS} failed ({last_error}); "
                f"retrying in {retry_delay:g}s"
            )
            time.sleep(retry_delay)
            backoff = min(backoff * 2, RETRY_BACKOFF_CAP_SEC)

    print(
        f"[yahoo_ticker_rss] {ticker} failed after "
        f"{MAX_FETCH_ATTEMPTS} attempts: {last_error}"
    )
    return []


def collect_yahoo_ticker_rss(batch: int = BATCH_PER_PASS) -> list:
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
        with ThreadPoolExecutor(max_workers=min(len(selected), 16)) as ex:
            futures = {ex.submit(_fetch_ticker, t): t for t in selected}
            for fut in as_completed(futures):
                try:
                    raw.append(fut.result())
                except Exception as e:
                    ticker = futures.get(fut, "?")
                    print(f"[yahoo_ticker_rss] {ticker} worker error: {e}")

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
                (aid, art["link"], art["title"], "YahooFinance", datetime.now(timezone.utc).isoformat()),
            )

    state["index"] = idx
    state["last_polled"] = last_polled
    _save_cursor(state)

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    items = collect_yahoo_ticker_rss(batch=5)
    print(f"Got {len(items)} new Yahoo per-ticker articles")
    for a in items[:10]:
        print(f"  [{a['_ticker']}] {a['title'][:80]}")
