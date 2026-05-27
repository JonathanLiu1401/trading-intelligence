"""StockTwits trending symbols rank detector — no API key required.

Distinct from stocktwits_collector.py (message stream) and
stocktwits_sentiment.py (per-ticker sentiment). This fetches the ranked
*symbol* list (api/2/trending/symbols.json) which returns the 30 tickers
with the highest trending_score right now, plus an AI-generated community
sentiment summary for each.

Emits one article per trending ticker per day, deduped by (ticker, date) in
seen_articles.db. High-score tickers (score >= HIGH_SCORE_THRESH) emit with
slightly higher source priority in the title so scoring picks them up.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger("stocktwits_trending_symbols")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

SYMBOLS_URL = "https://api.stocktwits.com/api/2/trending/symbols.json"
SOURCE_NAME = "StockTwits Trending"
REQUEST_TIMEOUT = 10
HIGH_SCORE_THRESH = 15.0   # top-tier buzz — call out explicitly in title
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY,
            link TEXT,
            title TEXT,
            source TEXT,
            first_seen TEXT
        )"""
    )
    conn.commit()


def _article_id(ticker: str, date_str: str) -> str:
    return hashlib.sha256(f"st_trend:{ticker}:{date_str}".encode()).hexdigest()


def collect_trending_symbols() -> list[dict]:
    """Fetch StockTwits trending symbols and emit article-shaped dicts."""
    try:
        resp = requests.get(
            SYMBOLS_URL,
            headers={"User-Agent": _UA},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("[stocktwits_trending] fetch failed: %s", exc)
        return []

    symbols = data.get("symbols", [])
    if not symbols:
        log.debug("[stocktwits_trending] no symbols returned")
        return []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now_iso = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    _ensure_db(conn)

    articles: list[dict] = []
    inserted = 0

    for rank, sym in enumerate(symbols, 1):
        ticker = (sym.get("symbol") or "").strip().upper()
        if not ticker:
            continue

        art_id = _article_id(ticker, today)

        # Skip if already seen today
        row = conn.execute(
            "SELECT 1 FROM seen_articles WHERE id=?", (art_id,)
        ).fetchone()
        if row:
            continue

        title_name = sym.get("title") or ticker
        score = sym.get("trending_score") or 0.0
        wc = sym.get("watchlist_count") or 0
        sector = sym.get("sector") or ""
        industry = sym.get("industry") or ""

        trends = sym.get("trends") or {}
        summary = (trends.get("summary") or "").strip()
        rank_all = trends.get("all", rank)

        # Build title
        tier = "🔥 HOT" if score >= HIGH_SCORE_THRESH else "Trending"
        title = (
            f"[{tier}] ${ticker} #{rank_all} on StockTwits "
            f"(score {score:.1f}, {wc:,} followers)"
        )

        # Build summary text
        parts = [f"${ticker} ({title_name}) is trending #{rank_all} on StockTwits."]
        if sector:
            parts.append(f"Sector: {sector}" + (f" / {industry}" if industry else "") + ".")
        if summary:
            parts.append(summary)
        parts.append(f"Trending score: {score:.1f} | Watchlist followers: {wc:,}")

        link = f"https://stocktwits.com/symbol/{ticker}"

        article = {
            "id": art_id,
            "title": title,
            "link": link,
            "summary": " ".join(parts),
            "published": now_iso,
            "source": SOURCE_NAME,
            "_tickers": [ticker],
        }
        articles.append(article)

        try:
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                "VALUES (?,?,?,?,?)",
                (art_id, link, title, SOURCE_NAME, now_iso),
            )
            inserted += 1
        except sqlite3.OperationalError as exc:
            log.warning("[stocktwits_trending] db write error: %s", exc)

    conn.commit()
    conn.close()

    log.info(
        "[stocktwits_trending] %d symbols fetched, %d new articles emitted",
        len(symbols),
        inserted,
    )
    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = collect_trending_symbols()
    print(f"\nFetched {len(results)} new trending symbol articles:")
    for a in results[:10]:
        print(f"  {a['title']}")
        print(f"    {a['summary'][:120]}...")
