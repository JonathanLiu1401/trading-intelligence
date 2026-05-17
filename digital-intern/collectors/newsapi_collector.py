"""NewsAPI.org collector — query-based, round-robin across themes.

Free tier: 100 requests/day. Each call uses a query like "NVDA semiconductor".
We rotate through a small set of high-value queries.

Endpoint: GET https://newsapi.org/v2/everything?q=...&apiKey=KEY

Skips silently if NEWS_API_KEY is unset.
"""
import hashlib
import os
import sqlite3
import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
CURSOR_PATH = BASE_DIR / "data" / "newsapi_cursor.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

API_URL = "https://newsapi.org/v2/everything"
BATCH_PER_PASS = 1
# 100 req/day / 24h ~ 4 req/hour. Set 25min cooldown per query (~57 q/day).
PER_QUERY_COOLDOWN_SEC = 1500
HTTP_TIMEOUT = 10
LOOKBACK_HOURS = 12
_KEY_WARNED = False

# Fixed thematic queries — survive even when ticker config is empty.
STATIC_QUERIES = [
    "semiconductor OR \"chip shortage\" OR \"AI chips\"",
    "DRAM OR HBM OR NAND memory",
    "TSMC OR ASML OR Samsung foundry",
    "\"Federal Reserve\" OR \"interest rate\" OR inflation",
    "earnings beat OR earnings miss OR guidance",
    "bitcoin OR ethereum OR crypto regulation",
    "oil prices OR OPEC OR \"natural gas\"",
    "\"export controls\" OR tariff OR China sanctions",
]


def _ticker_queries() -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()

    def _add(t: str):
        u = (t or "").strip().upper()
        if u and u not in seen and "." not in u and "=" not in u and "^" not in u and len(u) <= 5:
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

    return [f"{t} stock" for t in tickers]


def _all_queries() -> list[str]:
    return STATIC_QUERIES + _ticker_queries()


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


def _fetch_query(key: str, query: str) -> list:
    since = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).isoformat(timespec="seconds")
    params = {
        "q": query,
        "from": since,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": key,
    }
    try:
        r = requests.get(API_URL, params=params, timeout=HTTP_TIMEOUT)
    except Exception as e:
        print(f"[newsapi] {query[:30]} fetch error: {e}")
        return []
    if r.status_code == 429:
        print(f"[newsapi] rate-limited (429) on '{query[:30]}'")
        return []
    if r.status_code != 200:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    if data.get("status") != "ok":
        msg = data.get("message", "")
        if msg:
            print(f"[newsapi] api error: {msg[:120]}")
        return []

    out: list = []
    for it in data.get("articles", []):
        title = (it.get("title") or "").strip()
        link = (it.get("url") or "").strip()
        if not title or not link or title == "[Removed]":
            continue
        summary = (it.get("description") or it.get("content") or "").strip()
        published = (it.get("publishedAt") or "").strip()
        src_name = (it.get("source") or {}).get("name") or "unknown"
        out.append({
            "title": title,
            "link": link,
            "summary": summary,
            "published": published,
            "source": f"NewsAPI/{src_name}",
            "_query": query,
        })
    return out


def collect_newsapi(batch: int = BATCH_PER_PASS) -> list:
    global _KEY_WARNED
    key = os.environ.get("NEWS_API_KEY", "").strip()
    if not key:
        if not _KEY_WARNED:
            print("[newsapi] NEWS_API_KEY not set — skipping. "
                  "Get a free key at https://newsapi.org/register")
            _KEY_WARNED = True
        return []

    queries = _all_queries()
    if not queries:
        return []

    state = _load_cursor()
    idx = int(state.get("index", 0)) % len(queries)
    last_polled: dict = state.get("last_polled") or {}
    now = time.time()

    selected: list[str] = []
    attempts = 0
    max_attempts = max(batch * 3, len(queries))
    while len(selected) < batch and attempts < max_attempts:
        q = queries[idx]
        idx = (idx + 1) % len(queries)
        attempts += 1
        last = float(last_polled.get(q, 0))
        if now - last < PER_QUERY_COOLDOWN_SEC:
            continue
        selected.append(q)
        last_polled[q] = now

    raw: list[list] = []
    for q in selected:
        raw.append(_fetch_query(key, q))

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
                (aid, art["link"], art["title"], "NewsAPI", datetime.now(timezone.utc).isoformat()),
            )

    state["index"] = idx
    state["last_polled"] = last_polled
    _save_cursor(state)

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    items = collect_newsapi(batch=2)
    print(f"Got {len(items)} new NewsAPI articles")
    for a in items[:10]:
        print(f"  [{a.get('_query','')[:30]}] {a['title'][:80]}")
