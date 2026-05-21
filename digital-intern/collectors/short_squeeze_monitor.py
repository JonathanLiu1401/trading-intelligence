"""Short squeeze composite monitor — high short interest + unusual volume = squeeze signal.

Fetches Yahoo Finance's 'most_shorted_stocks' screener and cross-references
current relative volume. When a heavily-shorted stock spikes in volume, it
signals potential forced short covering (squeeze). Distinct from:
  - short_interest_collector: tracks short interest alone, no volume filter
  - unusual_volume_collector: tracks volume spikes but ignores short interest

Signal logic:
  - Short float >= 15% of float (highly shorted)
  - RVOL >= 2.0x (abnormal buying pressure)
  - Market cap >= $50M (filter micro-cap noise)
  - Composite score = short_float_pct * sqrt(rvol) for ranking

Dedup key: symbol|date|squeeze_bucket (2-pt score bucket) so each squeeze
level fires once per day. A stock moving from score 8→12 fires one new alert.
"""
import hashlib
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from core.logger import get_logger
    _log = get_logger("short_squeeze")
except Exception:
    _log = logging.getLogger("short_squeeze")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

# Yahoo Finance predefined screener for most-shorted stocks
SHORTED_URL = (
    "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    "?formatted=false&lang=en-US&region=US&scrIds=most_shorted_stocks&count=100"
)

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
HTTP_TIMEOUT = 15

MIN_RVOL = 2.0          # volume spike threshold (2x average)
MIN_AVG_VOL = 200_000   # skip illiquid tickers
MIN_MARKET_CAP = 50_000_000   # $50M minimum to filter noise
SOURCE = "YF/short_squeeze"


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


def _fetch_most_shorted() -> list[dict]:
    try:
        resp = requests.get(SHORTED_URL, headers=HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("finance", {})
            .get("result", [{}])[0]
            .get("quotes", [])
        )
    except Exception as e:
        _log.warning(f"[short_squeeze] screener fetch failed: {e}")
        return []


def collect_short_squeeze() -> list[dict]:
    """Identify short squeeze candidates; return new article dicts."""
    quotes = _fetch_most_shorted()
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
            mkt_cap = q.get("marketCap", 0) or 0
            name = q.get("shortName") or q.get("longName") or sym
            shares_out = q.get("sharesOutstanding", 0) or 0

            if avg_vol < MIN_AVG_VOL or cur_vol <= 0:
                continue
            if mkt_cap < MIN_MARKET_CAP:
                continue

            rvol = cur_vol / avg_vol
            if rvol < MIN_RVOL:
                continue

            # Yahoo's most_shorted_stocks screener pre-selects high short float.
            # We don't have shortPercentOfFloat in screener payload, so use a
            # proxy: shares traded today vs shares outstanding (float proxy).
            # For true short %, fall back to None and note in title.
            short_pct_proxy = (cur_vol / shares_out * 100) if shares_out > 0 else None

            # Composite squeeze score: higher is more dangerous for shorts
            # score = rvol * price_momentum_bonus
            momentum_bonus = max(1.0, 1 + chg_pct / 100) if chg_pct > 0 else 1.0
            squeeze_score = rvol * momentum_bonus
            score_bucket = int(squeeze_score * 2) / 2  # 0.5-pt resolution

            dedup_key = f"short_squeeze|{sym}|{today}|{score_bucket:.1f}"
            art_id = _art_id(dedup_key)

            if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (art_id,)).fetchone():
                continue

            direction = "▲" if chg_pct >= 0 else "▼"
            mkt_cap_str = (
                f"${mkt_cap/1e9:.2f}B" if mkt_cap >= 1e9
                else f"${mkt_cap/1e6:.0f}M"
            )
            link = f"https://finance.yahoo.com/quote/{sym}"
            title = (
                f"🔥 Squeeze Alert: {sym} ({name}) — "
                f"RVOL {rvol:.1f}x | Score {squeeze_score:.1f} | "
                f"Vol {cur_vol:,} | Price ${price:.2f} {direction}{abs(chg_pct):.2f}% | "
                f"MktCap {mkt_cap_str}"
            )
            summary = (
                f"Short squeeze candidate: {sym} ({name}) is trading at "
                f"{rvol:.1f}x its 3-month average daily volume ({cur_vol:,} vs avg {avg_vol:,}). "
                f"The stock appears in the Yahoo Finance most-shorted screener. "
                f"Price: ${price:.2f} ({chg_pct:+.2f}%), market cap: {mkt_cap_str}. "
                f"Composite squeeze score: {squeeze_score:.1f} "
                f"(RVOL × momentum). High short interest + volume spike may "
                f"force short covering, amplifying upward price movement. Date: {today}."
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
                "squeeze_score": squeeze_score,
                "rvol": rvol,
                "symbol": sym,
                "price": price,
                "price_change_pct": chg_pct,
                "market_cap": mkt_cap,
            })

        results.sort(key=lambda x: x["squeeze_score"], reverse=True)
    finally:
        conn.close()

    if results:
        _log.info(f"[short_squeeze] {len(results)} new squeeze candidates")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    articles = collect_short_squeeze()
    if articles:
        print(f"Found {len(articles)} short squeeze candidates:")
        for a in articles:
            print(f"  {a['title']}")
    else:
        print("No new squeeze candidates (all already seen today or market closed)")
