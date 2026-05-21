"""Unusual volume screener — stocks trading at 3x+ their average daily volume.

High relative volume (RVOL) often precedes news catalysts, earnings beats,
M&A activity, or institutional accumulation. Distinct from market_movers which
tracks absolute price change; a stock can have RVOL=5x while price is flat.

Emits one article row per stock when RVOL crosses a threshold for the first
time that session. Dedup key: symbol|date|rvol_bucket (integer) so a stock
that moves from 3x→5x fires once per integer-RVOL level per day.
"""
import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from core.logger import get_logger
    _log = get_logger("unusual_volume")
except Exception:
    _log = logging.getLogger("unusual_volume")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

MOST_ACTIVE_URL = (
    "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    "?formatted=false&lang=en-US&region=US&scrIds=most_actives&count=100"
)

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
HTTP_TIMEOUT = 12

MIN_RVOL = 3.0        # minimum relative volume to emit
MIN_AVG_VOL = 500_000  # skip illiquid tickers
SOURCE = "YF/unusual_volume"


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


def _art_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _fetch_screener_quotes() -> list[dict]:
    """Return quote dicts directly from most-actives screener (includes avg vol)."""
    try:
        resp = requests.get(MOST_ACTIVE_URL, headers=HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("finance", {})
            .get("result", [{}])[0]
            .get("quotes", [])
        )
    except Exception as e:
        _log.warning(f"[unusual_vol] screener fetch failed: {e}")
        return []


def collect_unusual_volume() -> list[dict]:
    """Find stocks with unusual relative volume; return new article dicts."""
    quotes = _fetch_screener_quotes()
    if not quotes:
        return []

    now_utc = datetime.now(timezone.utc)
    today = now_utc.strftime("%Y-%m-%d")
    now_str = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    _ensure_db(conn)

    results = []
    try:
        for q in quotes:
            sym = q.get("symbol", "")
            avg_vol = q.get("averageDailyVolume3Month", 0) or 0
            cur_vol = q.get("regularMarketVolume", 0) or 0
            price = q.get("regularMarketPrice", 0) or 0
            chg_pct = q.get("regularMarketChangePercent", 0) or 0
            name = q.get("shortName") or sym

            if avg_vol < MIN_AVG_VOL or cur_vol <= 0:
                continue

            rvol = cur_vol / avg_vol
            if rvol < MIN_RVOL:
                continue

            rvol_bucket = int(rvol)  # 3x, 4x, 5x … unique per day
            dedup_key = f"unusual_vol|{sym}|{today}|{rvol_bucket}"
            art_id = _art_id(dedup_key)

            if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (art_id,)).fetchone():
                continue

            direction = "▲" if chg_pct >= 0 else "▼"
            link = f"https://finance.yahoo.com/quote/{sym}"
            title = (
                f"🔥 Unusual Volume: {sym} ({name}) — "
                f"RVOL {rvol:.1f}x | Vol {cur_vol:,} vs avg {avg_vol:,} | "
                f"Price ${price:.2f} {direction}{abs(chg_pct):.2f}%"
            )
            summary = (
                f"{sym} ({name}) is trading at {rvol:.1f}x its 3-month average daily volume "
                f"({cur_vol:,} vs avg {avg_vol:,}). "
                f"Current price: ${price:.2f} ({chg_pct:+.2f}%). "
                f"High relative volume often precedes significant news catalysts, "
                f"institutional activity, or breakout moves. Date: {today}."
            )

            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                "VALUES (?,?,?,?,?)",
                (art_id, link, title, SOURCE, now_str),
            )
            conn.commit()

            results.append({
                "id": art_id,
                "link": link,
                "title": title,
                "summary": summary,
                "source": SOURCE,
                "first_seen": now_str,
                "rvol": rvol,
                "symbol": sym,
                "current_volume": cur_vol,
                "avg_volume": avg_vol,
                "price_change_pct": chg_pct,
            })

        results.sort(key=lambda x: x["rvol"], reverse=True)
    finally:
        conn.close()

    if results:
        _log.info(f"[unusual_vol] {len(results)} new high-RVOL stocks")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    articles = collect_unusual_volume()
    if articles:
        print(f"Found {len(articles)} unusual volume stocks:")
        for a in articles:
            print(f"  {a['title']}")
    else:
        print("No new unusual volume stocks (all already seen today or market closed)")
