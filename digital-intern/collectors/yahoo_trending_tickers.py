"""Yahoo Finance trending tickers collector.

Fetches the real-time trending tickers list from Yahoo Finance (no API key).
Emits one article per trending ticker surge, capturing retail attention flow
that the day-gainers/losers screener misses — a ticker can trend without
being a top mover by percent change.

Each article title: "[YF/trending] {SYMBOL} - {name} trending on Yahoo Finance"
Dedup: per-symbol cooldown (default 60 min) so the same ticker doesn't re-emit
every pass if it stays on the trending list all day.

Source: https://query1.finance.yahoo.com/v1/finance/trending/US
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from core.logger import get_logger
    _log = get_logger("yahoo_trending")
except Exception:
    _log = logging.getLogger("yahoo_trending")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

TRENDING_URL = (
    "https://query1.finance.yahoo.com/v1/finance/trending/US"
    "?count=25&fields=symbol,shortName,regularMarketPrice,regularMarketChangePercent"
)
QUOTE_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?interval=1d&range=1d&fields=shortName,regularMarketPrice,regularMarketChangePercent,regularMarketVolume"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
HTTP_TIMEOUT = 10

SOURCE_NAME = "YF/trending"
# Re-emit same ticker only after this many minutes off the trending list
TRENDING_COOLDOWN_MIN = 60

# Skip pure index/crypto/forex tickers that add noise but not actionable signal.
_SKIP_PREFIXES = ("^", "BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "BNB", "=X")


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
    conn.execute(
        """CREATE TABLE IF NOT EXISTS trending_cooldown (
            symbol TEXT PRIMARY KEY,
            last_emit_iso TEXT NOT NULL
        )"""
    )
    conn.commit()


def _article_id(link: str, title: str) -> str:
    return hashlib.sha256(f"{link}|{title}".encode()).hexdigest()


def _within_cooldown(conn: sqlite3.Connection, symbol: str, now: datetime) -> bool:
    row = conn.execute(
        "SELECT last_emit_iso FROM trending_cooldown WHERE symbol=?", (symbol,)
    ).fetchone()
    if not row or not row[0]:
        return False
    try:
        last = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (now - last).total_seconds() / 60.0 < TRENDING_COOLDOWN_MIN


def _update_cooldown(conn: sqlite3.Connection, symbol: str, now: datetime) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO trending_cooldown (symbol, last_emit_iso) VALUES (?,?)",
        (symbol, now.strftime("%Y-%m-%dT%H:%M:%SZ")),
    )


def _fetch_trending() -> list[dict]:
    try:
        r = requests.get(TRENDING_URL, headers=HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json().get("finance", {}).get("result", [{}])[0].get("quotes", [])
    except Exception as e:
        _log.warning(f"[yahoo_trending] trending fetch failed: {type(e).__name__}: {e}")
        return []


def _fetch_quote(symbol: str) -> dict:
    """Fetch current price/change for a symbol; returns {} on failure."""
    try:
        r = requests.get(
            QUOTE_URL.format(symbol=symbol), headers=HEADERS, timeout=HTTP_TIMEOUT
        )
        r.raise_for_status()
        meta = r.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
        return meta
    except Exception:
        return {}


def collect_yahoo_trending() -> list[dict]:
    """Fetch Yahoo Finance trending tickers. Returns net-new article dicts."""
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    results: list[dict] = []

    quotes = _fetch_trending()
    if not quotes:
        return results

    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    _ensure_db(conn)

    try:
        for q in quotes:
            symbol = (q.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            # Skip non-equity symbols (indices, crypto, forex)
            if any(symbol.startswith(p) for p in _SKIP_PREFIXES):
                continue

            if _within_cooldown(conn, symbol, now_dt):
                continue

            # Try to get a display name — Yahoo trending endpoint sometimes
            # returns sparse records; fall back to a quote call only when needed.
            name = q.get("shortName") or q.get("longName") or ""
            price = q.get("regularMarketPrice")
            chg_pct = q.get("regularMarketChangePercent")

            if not name or price is None:
                meta = _fetch_quote(symbol)
                name = name or meta.get("shortName") or meta.get("longName") or symbol
                price = price if price is not None else meta.get("regularMarketPrice")
                chg_pct = chg_pct if chg_pct is not None else meta.get("regularMarketChangePercent")

            # Build human-readable change string
            if price is not None and chg_pct is not None:
                direction = "+" if chg_pct >= 0 else ""
                detail = f" @ ${price:.2f} ({direction}{chg_pct:.2f}%)"
            elif price is not None:
                detail = f" @ ${price:.2f}"
            else:
                detail = ""

            title = f"[YF/trending] {symbol} - {name} trending on Yahoo Finance{detail}"
            link = f"https://finance.yahoo.com/quote/{symbol}/"
            art_id = _article_id(link, title)

            # Skip if exact title already emitted (price changed → new title → new emit)
            if conn.execute(
                "SELECT 1 FROM seen_articles WHERE id=?", (art_id,)
            ).fetchone():
                continue

            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                "VALUES (?,?,?,?,?)",
                (art_id, link, title, SOURCE_NAME, now_iso),
            )
            _update_cooldown(conn, symbol, now_dt)
            conn.commit()

            results.append({
                "id": art_id,
                "title": title,
                "link": link,
                "source": SOURCE_NAME,
                "published": now_iso,
                "summary": (
                    f"{symbol} is trending on Yahoo Finance. "
                    f"Retail attention indicator for {name or symbol}."
                ),
            })

    finally:
        conn.close()

    if results:
        _log.info(f"[yahoo_trending] {len(results)} new trending tickers emitted")
    return results


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    articles = collect_yahoo_trending()
    print(f"\nFetched {len(articles)} new trending ticker articles:")
    for a in articles:
        print(f"  {a['title']}")
