"""Market breadth collector — Finviz screener-based.

Tracks the percentage of US-listed stocks above key moving averages,
plus new 52-week high/low counts. These are leading indicators of
broad market health: when breadth diverges from price (e.g. SPX makes
new highs but fewer stocks are above 200MA), it signals fragility.

Emits a synthetic article per session when breadth metrics shift by
more than a threshold, deduped by (date, breadth-bucket).

Data source: Finviz public screener (no API key required).
"""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger("market_breadth_collector")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

SOURCE_NAME = "Market Breadth"
REQUEST_TIMEOUT = 12
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Finviz screener base + filter for each breadth dimension
SCREENER_BASE = "https://finviz.com/screener.ashx?v=111"
FILTERS = {
    "above_200ma": "ta_sma200_pa",
    "above_50ma": "ta_sma50_pa",
    "above_20ma": "ta_sma20_pa",
    "new_52w_highs": "ta_highlow52w_nh",
    "new_52w_lows": "ta_highlow52w_nl",
    "rsi_overbought": "ta_rsi_ob70",
    "rsi_oversold": "ta_rsi_os30",
}

# Emit a new article when breadth (% above 200MA) shifts by this many pct pts
EMIT_THRESHOLD_PCT = 2.0
# Minimum cooldown between emits regardless of shift (seconds)
EMIT_COOLDOWN_SEC = 3600


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


def _article_id(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _fetch_count(filter_param: str | None = None, retries: int = 2) -> int | None:
    """Fetch the stock count from Finviz screener for a given filter."""
    url = SCREENER_BASE
    if filter_param:
        url += f"&f={filter_param}"
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            # Finviz embeds the filtered count as result_count in a JS call;
            # the base (no-filter) page uses "#1 / N Total" in screener-total.
            m = re.search(r'"result_count"\s*:\s*(\d+)', r.text)
            if not m:
                m = re.search(r"#\d+\s*/\s*([\d,]+)\s*Total", r.text)
            if m:
                return int(m.group(1).replace(",", ""))
            log.warning("market_breadth: count pattern not found for filter=%s", filter_param)
            return None
        except requests.RequestException as exc:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            log.warning("market_breadth: fetch failed filter=%s err=%s", filter_param, exc)
            return None
    return None


def _breadth_label(pct: float) -> str:
    if pct >= 70:
        return "strong"
    if pct >= 55:
        return "healthy"
    if pct >= 45:
        return "neutral"
    if pct >= 30:
        return "weak"
    return "washed-out"


def collect_market_breadth() -> list[dict]:
    """Fetch Finviz breadth counts and emit synthetic articles on notable shifts."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    # Fetch total universe first
    total = _fetch_count(None)
    if not total or total < 1000:
        log.warning("market_breadth: implausible total=%s, aborting", total)
        return []

    # Small delay between requests to be polite to Finviz
    results: dict[str, int | None] = {}
    for name, filt in FILTERS.items():
        results[name] = _fetch_count(filt)
        time.sleep(0.5)

    above_200 = results.get("above_200ma")
    above_50 = results.get("above_50ma")
    above_20 = results.get("above_20ma")
    new_highs = results.get("new_52w_highs")
    new_lows = results.get("new_52w_lows")
    rsi_ob = results.get("rsi_overbought")
    rsi_os = results.get("rsi_oversold")

    if above_200 is None:
        log.warning("market_breadth: missing 200MA count, skipping emit")
        return []

    pct_200 = round(above_200 / total * 100, 1)
    pct_50 = round(above_50 / total * 100, 1) if above_50 else None
    pct_20 = round(above_20 / total * 100, 1) if above_20 else None
    label = _breadth_label(pct_200)

    # Dedup key: date + 2-pct-point bucket of the 200MA breadth reading
    bucket = int(pct_200 // 2) * 2
    dedup_key = f"market_breadth:{date_str}:{bucket}"
    article_id = _article_id(dedup_key)

    for _attempt in range(5):
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=60, check_same_thread=False)
            _ensure_db(conn)
            break
        except sqlite3.OperationalError:
            time.sleep(3)
    else:
        log.warning("market_breadth: could not open DB after retries")
        return []

    try:
        existing = conn.execute(
            "SELECT id FROM seen_articles WHERE id=?", (article_id,)
        ).fetchone()
        if existing:
            log.debug("market_breadth: dedup hit %s", dedup_key)
            return []

        highs_lows = ""
        if new_highs is not None and new_lows is not None:
            hl_ratio = new_highs / max(new_lows, 1)
            highs_lows = f" | New highs: {new_highs}, new lows: {new_lows} (H/L ratio: {hl_ratio:.1f})"

        ma_parts = [f"Above 200MA: {pct_200}%"]
        if pct_50 is not None:
            ma_parts.append(f"50MA: {pct_50}%")
        if pct_20 is not None:
            ma_parts.append(f"20MA: {pct_20}%")
        ma_str = " | ".join(ma_parts)

        rsi_str = ""
        if rsi_ob is not None and rsi_os is not None:
            rsi_str = f" | RSI overbought: {rsi_ob}, oversold: {rsi_os}"

        title = (
            f"Market breadth ({date_str}): {label} — {ma_str}"
            f"{highs_lows}{rsi_str} (universe: {total:,})"
        )
        body = (
            f"Finviz market breadth snapshot: {ma_str}{highs_lows}{rsi_str}. "
            f"Total US-listed stocks tracked: {total:,}. "
            f"Breadth condition: {label}."
        )
        link = f"https://finviz.com/screener.ashx?v=111&f=ta_sma200_pa"

        article = {
            "id": article_id,
            "title": title,
            "body": body,
            "link": link,
            "source": SOURCE_NAME,
            "first_seen": now.isoformat(),
        }

        for _w in range(10):
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (article_id, link, title, SOURCE_NAME, now.isoformat()),
                )
                conn.commit()
                break
            except sqlite3.OperationalError:
                time.sleep(2)
        else:
            log.warning("market_breadth: failed to insert after retries")
            return []
        log.info(
            "market_breadth: emitted breadth article pct_200=%.1f%% label=%s",
            pct_200,
            label,
        )
        return [article]

    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    arts = collect_market_breadth()
    for a in arts:
        print(a["title"])
        print(a["body"])
        print()
    if not arts:
        print("(dedup: no new article emitted this bucket)")
