"""Robinhood Popular Stocks collector — retail sentiment via position popularity.

Fetches Robinhood's "100 Most Popular" list (public, no auth) and emits a
synthetic article summarising the top retail-held tickers. This is a unique
retail-sentiment signal: high Robinhood popularity → high retail concentration
→ potential squeeze/crash risk when sentiment shifts.

Batch-resolves instrument IDs to tickers using the public /instruments/?ids=
endpoint (up to 50 per request). Deduped daily via seen_articles.db.

No API key required; Robinhood's public market-data endpoints have no auth.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger("robinhood_popular")

BASE_DIR = Path(__file__).resolve().parent.parent
SEEN_DB = BASE_DIR / "data" / "seen_articles.db"

SOURCE_NAME = "Robinhood Popular"
RH_POPULAR_URL = "https://api.robinhood.com/midlands/tags/tag/100-most-popular/"
RH_INSTRUMENTS_URL = "https://api.robinhood.com/instruments/"
FETCH_TIMEOUT = 12
BATCH_SIZE = 50  # max IDs per /instruments/?ids= call

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
_SESS = requests.Session()
_SESS.headers.update({"User-Agent": _UA})


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


def _extract_id(instrument_url: str) -> str | None:
    """Extract the UUID from a Robinhood instrument URL."""
    parts = urlparse(instrument_url).path.strip("/").split("/")
    return parts[-1] if parts else None


def _resolve_instruments(ids: list[str]) -> dict[str, dict]:
    """Batch-resolve instrument IDs to {id: {symbol, name}} via public API."""
    result: dict[str, dict] = {}
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i : i + BATCH_SIZE]
        try:
            r = _SESS.get(
                RH_INSTRUMENTS_URL,
                params={"ids": ",".join(batch)},
                timeout=FETCH_TIMEOUT,
            )
            r.raise_for_status()
            for item in r.json().get("results", []):
                iid = _extract_id(item.get("url", ""))
                if iid:
                    result[iid] = {
                        "symbol": item.get("symbol", ""),
                        "name": item.get("simple_name") or item.get("name", ""),
                    }
        except Exception as e:
            log.warning(f"[robinhood_popular] instrument batch resolve error: {e}")
    return result


def collect_robinhood_popular() -> list[dict]:
    """Collect Robinhood's 100 most-popular stocks and emit a daily summary article."""
    try:
        r = _SESS.get(RH_POPULAR_URL, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"[robinhood_popular] fetch error: {e}")
        return []

    instrument_urls: list[str] = data.get("instruments", [])
    if not instrument_urls:
        log.warning("[robinhood_popular] empty instruments list")
        return []

    ids = [_extract_id(u) for u in instrument_urls if u]
    ids = [i for i in ids if i]

    resolved = _resolve_instruments(ids)

    tickers: list[str] = []
    names: list[str] = []
    for iid in ids:
        info = resolved.get(iid)
        if info and info.get("symbol"):
            tickers.append(info["symbol"])
            names.append(info.get("name", info["symbol"]))

    if not tickers:
        log.warning("[robinhood_popular] could not resolve any tickers")
        return []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dedup_key = f"rh_popular_{today}"
    aid = hashlib.sha256(dedup_key.encode()).hexdigest()

    conn = _ensure_db()
    try:
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
            log.debug("[robinhood_popular] already emitted today, skipping")
            conn.close()
            return []
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen)"
            " VALUES (?,?,?,?,?)",
            (aid, RH_POPULAR_URL, f"[RH/popular] Top 100 — {today}", SOURCE_NAME,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    except sqlite3.Error as e:
        log.warning(f"[robinhood_popular] dedup DB error: {e}")
    finally:
        conn.close()

    top10 = tickers[:10]
    top_str = ", ".join(top10)
    all_str = ", ".join(tickers[:50])
    summary = (
        f"Robinhood 100 Most Popular Stocks ({today}): Top 10 = {top_str}. "
        f"Full list (top 50): {all_str}. "
        f"Total tracked: {len(tickers)}. "
        "High retail concentration in these tickers may indicate crowded trades, "
        "squeeze risk, or meme-stock sentiment build-up."
    )

    return [{
        "title": f"[RH/popular] Robinhood Top 100: {top_str}",
        "link": "https://robinhood.com/collections/100-most-popular",
        "summary": summary,
        "published": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": SOURCE_NAME,
    }]


collect = collect_robinhood_popular


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("=== Robinhood Popular Stocks (live fetch) ===")
    items = collect_robinhood_popular()
    print(f"Articles emitted: {len(items)}")
    for art in items:
        print(f"  Title: {art['title']}")
        print(f"  Summary: {art['summary'][:300]}")
