"""Yahoo Finance market movers collector — top gainers, losers, most-active.

Polls the Yahoo Finance predefined screener API (no key required) every pass
and emits structured article rows so the briefing pipeline can surface
significant price movers alongside news.

Each article row title encodes: symbol, name, price, % change, volume.
Source tags: "YF/day_gainers", "YF/day_losers", "YF/most_actives".
"""
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

SCREENER_URL = (
    "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    "?formatted=false&lang=en-US&region=US&scrIds={scr_id}&count=25"
)
SCREENERS = [
    ("day_gainers",  "YF/day_gainers"),
    ("day_losers",   "YF/day_losers"),
    ("most_actives", "YF/most_actives"),
]
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
HTTP_TIMEOUT = 10

# Minimum move thresholds to avoid noise on low-volatility days.
MIN_GAINER_PCT = 3.0
MIN_LOSER_PCT = -3.0


def _ensure_db(conn: sqlite3.Connection) -> None:
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


def _article_id(link: str, title: str) -> str:
    return hashlib.sha256(f"{link}|{title}".encode()).hexdigest()


def _fetch_screener(scr_id: str) -> list[dict]:
    url = SCREENER_URL.format(scr_id=scr_id)
    try:
        r = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
    except Exception:
        return []


def collect_market_movers() -> list[dict]:
    """Fetch gainers/losers/most-active. Returns net-new article dicts."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results: list[dict] = []

    conn = sqlite3.connect(str(DB_PATH))
    _ensure_db(conn)

    try:
        for scr_id, source_tag in SCREENERS:
            quotes = _fetch_screener(scr_id)
            for q in quotes:
                symbol = q.get("symbol", "")
                name = q.get("shortName") or q.get("longName") or symbol
                price = q.get("regularMarketPrice")
                chg_pct = q.get("regularMarketChangePercent", 0.0)
                volume = q.get("regularMarketVolume", 0)
                avg_vol = q.get("averageDailyVolume3Month") or q.get("averageDailyVolume10Day") or 0

                if not symbol or price is None:
                    continue

                # Filter noise: skip small movers for gainers/losers
                if scr_id == "day_gainers" and chg_pct < MIN_GAINER_PCT:
                    continue
                if scr_id == "day_losers" and chg_pct > MIN_LOSER_PCT:
                    continue

                vol_str = f"{volume/1e6:.1f}M" if volume >= 1_000_000 else f"{volume/1e3:.0f}K"
                vol_rel = f" ({volume/avg_vol:.1f}x avg)" if avg_vol else ""

                sign = "+" if chg_pct >= 0 else ""
                title = (
                    f"[{source_tag}] {symbol} ({name}) "
                    f"{sign}{chg_pct:.1f}% @ ${price:.2f} | vol {vol_str}{vol_rel}"
                )
                link = f"https://finance.yahoo.com/quote/{symbol}"

                aid = _article_id(link, title)
                if conn.execute(
                    "SELECT 1 FROM seen_articles WHERE id=?", (aid,)
                ).fetchone():
                    continue

                conn.execute(
                    "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                    "VALUES (?,?,?,?,?)",
                    (aid, link, title, source_tag, now),
                )
                results.append({
                    "id": aid,
                    "link": link,
                    "title": title,
                    "source": source_tag,
                    "first_seen": now,
                    "symbol": symbol,
                    "chg_pct": chg_pct,
                    "price": price,
                    "volume": volume,
                })

        conn.commit()
    finally:
        conn.close()

    return results


if __name__ == "__main__":
    articles = collect_market_movers()
    print(f"Fetched {len(articles)} new mover articles")
    for a in articles[:10]:
        print(f"  {a['title']}")
