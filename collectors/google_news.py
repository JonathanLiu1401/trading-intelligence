"""Google News RSS collector — per-ticker, round-robin to keep fan-out small.

Each call to ``collect_google_news()`` advances through the ticker list one or
two tickers per pass (configurable). A persistent cursor lives in
data/google_news_cursor.json so progress survives restarts. The full universe
gets covered over ``len(tickers) / BATCH`` passes.
"""
import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import feedparser

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
CURSOR_PATH = BASE_DIR / "data" / "google_news_cursor.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

# How many tickers to fetch per call.
BATCH_PER_PASS = 8
# Skip a ticker if we polled it within this many seconds (prevents thrash on restart).
PER_TICKER_COOLDOWN_SEC = 120
USER_AGENT = "Mozilla/5.0 (Digital Intern Daemon)"


def _load_tickers() -> list[str]:
    """Positions + sector watchlist + watchlist.json memory_core, dedup-preserved."""
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
    conn = sqlite3.connect(DB_PATH)
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


def _is_seen(conn, aid: str) -> bool:
    return conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone() is not None


def _mark_seen(conn, aid: str, link: str, title: str, source: str):
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
        "VALUES (?, ?, ?, ?, ?)",
        (aid, link, title, source, datetime.utcnow().isoformat()),
    )


def _build_url(ticker: str) -> str:
    q = quote_plus(f"{ticker} stock")
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def _fetch_ticker_feed(ticker: str) -> list:
    """Parse the Google News RSS for one ticker. Returns raw entries (no dedup yet)."""
    url = _build_url(ticker)
    try:
        parsed = feedparser.parse(url, agent=USER_AGENT)
    except Exception as e:
        print(f"[google_news] {ticker} fetch error: {e}")
        return []
    if getattr(parsed, "bozo", 0) and not parsed.entries:
        return []

    out: list = []
    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue
        summary = entry.get("summary") or entry.get("description") or ""
        published = entry.get("published") or entry.get("updated") or ""
        source_name = (entry.get("source", {}) or {}).get("title", "Google News")
        out.append({
            "title": title,
            "link": link,
            "summary": summary,
            "published": published,
            "source": f"GoogleNews/{source_name}",
            "_ticker": ticker,
        })
    return out


def collect_google_news(batch: int = BATCH_PER_PASS) -> list:
    """Collect one round-robin batch of Google News articles (parallel fetch)."""
    tickers = _load_tickers()
    if not tickers:
        return []

    state = _load_cursor()
    idx = int(state.get("index", 0)) % len(tickers)
    last_polled: dict = state.get("last_polled") or {}
    now = time.time()

    # Walk forward and select up to `batch` due-for-poll tickers.
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

    # Fetch all selected tickers in parallel.
    raw_per_ticker: list[list] = []
    if selected:
        with ThreadPoolExecutor(max_workers=min(len(selected), 16)) as ex:
            futures = {ex.submit(_fetch_ticker_feed, t): t for t in selected}
            for fut in as_completed(futures):
                try:
                    raw_per_ticker.append(fut.result())
                except Exception as e:
                    print(f"[google_news] worker error: {e}")

    # Dedup against the seen_articles DB in a single serial pass.
    conn = _ensure_db()
    new_articles: list = []
    seen_in_run: set = set()
    for entries in raw_per_ticker:
        for art in entries:
            aid = _article_id(art["link"], art["title"])
            if aid in seen_in_run:
                continue
            seen_in_run.add(aid)
            if _is_seen(conn, aid):
                continue
            new_articles.append(art)
            _mark_seen(conn, aid, art["link"], art["title"], "Google News")

    state["index"] = idx
    state["last_polled"] = last_polled
    _save_cursor(state)

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    items = collect_google_news(batch=5)
    print(f"Got {len(items)} new Google News articles")
    for a in items[:10]:
        print(f"  [{a['_ticker']}] {a['title'][:80]}")
