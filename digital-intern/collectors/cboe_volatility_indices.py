"""CBOE cross-asset volatility index collector.

Fetches OVX (crude oil vol), GVZ (gold vol), and SKEW (tail risk) directly
from CBOE's public CDN.  Unlike vix_term_structure.py which polls Yahoo
Finance for VIX/VVIX, these three indices are not reliably quoted on Yahoo
and must be sourced from the authoritative CBOE CDN.

Signals:
  OVX  — CBOE Crude Oil ETF Volatility Index. Spikes accompany energy supply
          shocks, geopolitical events, and commodity demand collapses. OVX > 60
          is historically elevated; > 80 coincides with major energy crises.
  GVZ  — CBOE Gold ETF Volatility Index. Elevated readings signal flight-to-
          safety demand or hedge positioning. GVZ > 25 warrants attention.
  SKEW — CBOE Skew Index. Measures the tail-risk premium priced into S&P 500
          OTM puts.  Values above 140 indicate unusual demand for downside
          protection (institutional hedging).  Recent reading 139.5 is near
          the historical alert threshold.

Emits one article per index per NEW value (deduped by index + date). Alert
prefix added when a threshold is breached.

Data source: https://cdn.cboe.com/api/global/us_indices/daily_prices/<IDX>_History.csv
No authentication required; CBOE public data.
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
    log = get_logger("cboe_vol_indices")
except Exception:
    log = logging.getLogger("cboe_vol_indices")

BASE_DIR = Path(__file__).resolve().parent.parent
SEEN_DB = BASE_DIR / "data" / "seen_articles.db"
SOURCE = "cboe_volatility_indices"

CBOE_CDN = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{idx}_History.csv"
FETCH_TIMEOUT = 15
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# Index definitions: (symbol, description, alert_threshold, direction)
# direction='above' = alert when value > threshold
INDICES = [
    ("OVX",  "CBOE Crude Oil Volatility", 60.0,  "above",
     ["USO", "XOM", "CVX", "OXY", "SLB", "HAL"]),
    ("GVZ",  "CBOE Gold Volatility",      25.0,  "above",
     ["GLD", "GDX", "GDXJ", "NEM", "GOLD"]),
    ("SKEW", "CBOE Skew Index",          140.0, "above",
     ["SPY", "QQQ", "IWM", "SQQQ", "SPXU", "VIX"]),
]


def _ensure_seen_db(conn: sqlite3.Connection) -> None:
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


def _article_id(idx: str, date_str: str) -> str:
    raw = f"cboe_vol:{idx}:{date_str}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _fetch_latest(idx: str) -> tuple[str, float] | None:
    """Return (date_str, value) for the most recent available day."""
    url = CBOE_CDN.format(idx=idx)
    try:
        r = requests.get(url, headers={"User-Agent": _UA}, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
    except Exception as exc:
        log.warning("[cboe_vol] fetch %s failed: %s", idx, exc)
        return None

    last_line = ""
    for line in r.text.splitlines():
        line = line.strip()
        if line and not line.startswith("Trade") and "," in line:
            last_line = line
    if not last_line:
        return None

    parts = last_line.split(",")
    if len(parts) < 2:
        return None
    date_str = parts[0].strip()
    # CSV columns: Date, Open, High, Low, Close  (or Date, Close for SKEW)
    # Use Close (last column with data)
    close_str = parts[-1].strip()
    try:
        return date_str, float(close_str)
    except ValueError:
        return None


def collect_cboe_volatility_indices() -> list[dict]:
    """Fetch latest OVX, GVZ, SKEW; return new article dicts."""
    seen_conn = sqlite3.connect(str(SEEN_DB), timeout=30, check_same_thread=False)
    _ensure_seen_db(seen_conn)

    now_iso = datetime.now(timezone.utc).isoformat()
    articles: list[dict] = []

    for idx, desc, threshold, direction, tickers in INDICES:
        result = _fetch_latest(idx)
        if not result:
            continue
        date_str, value = result

        aid = _article_id(idx, date_str)
        existing = seen_conn.execute(
            "SELECT id FROM seen_articles WHERE id=?", (aid,)
        ).fetchone()
        if existing:
            log.debug("[cboe_vol] %s %s already seen", idx, date_str)
            continue

        is_alert = (direction == "above" and value > threshold)
        prefix = f"ALERT: {idx} ELEVATED — " if is_alert else ""

        ticker_str = " ".join(tickers)
        link = f"https://www.cboe.com/tradable_products/vix/vix_historical_data/"

        if idx == "OVX":
            summary = (
                f"{desc} ({idx}) closed at {value:.2f} on {date_str}. "
                f"OVX tracks 30-day implied vol of crude oil (USO options). "
                f"{'ELEVATED: above alert threshold of '+str(threshold)+'. ' if is_alert else ''}"
                f"Rising OVX accompanies energy supply shocks and geopolitical risk spikes. "
                f"Related: {ticker_str}"
            )
        elif idx == "GVZ":
            summary = (
                f"{desc} ({idx}) closed at {value:.2f} on {date_str}. "
                f"GVZ tracks 30-day implied vol of gold (GLD options). "
                f"{'ELEVATED: above alert threshold of '+str(threshold)+'. ' if is_alert else ''}"
                f"Rising GVZ signals flight-to-safety demand or positioning in gold. "
                f"Related: {ticker_str}"
            )
        else:
            summary = (
                f"{desc} ({idx}) closed at {value:.2f} on {date_str}. "
                f"SKEW measures the premium for S&P 500 OTM put protection. "
                f"{'ELEVATED: above threshold '+str(threshold)+', unusual tail-risk hedging. ' if is_alert else ''}"
                f"Values > 140 indicate institutional demand for downside protection. "
                f"Related: {ticker_str}"
            )

        title = (
            f"{prefix}{desc} ({idx}): {value:.2f} on {date_str}"
            + (" [ABOVE ALERT THRESHOLD]" if is_alert else "")
        )

        seen_conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
            (aid, link, title, SOURCE, now_iso),
        )
        seen_conn.commit()

        articles.append({
            "title": title,
            "link": link,
            "summary": summary,
            "published": date_str,
            "source": SOURCE,
        })
        log.info("[cboe_vol] new: %s = %.2f on %s", idx, value, date_str)

    seen_conn.close()
    return articles


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = collect_cboe_volatility_indices()
    print(f"\nFetched {len(results)} new items:")
    for a in results:
        print(f"  [{a['source']}] {a['title']}")
        print(f"    {a['summary'][:120]}...")
