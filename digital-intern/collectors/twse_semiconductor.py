"""TWSE (Taiwan Stock Exchange) semiconductor pre-market tracker.

Taiwan's market closes ~4-6 hours before US markets open, providing
advance signal for semiconductor stocks. TSMC (2330→TSM), MediaTek
(2454), UMC (2303→UMC), etc. often predict their US ADR movements.

Emits a synthetic article:
  - When any tracked stock moves >ALERT_THRESHOLD % (high urgency)
  - Daily summary of Taiwan semiconductor sector performance

TWSE public API: https://www.twse.com.tw/exchangeReport/STOCK_DAY
No API key required. Rate limit: ~1 req/sec per endpoint.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

SOURCE_NAME = "TWSE Semiconductor"
REQUEST_TIMEOUT = 12
SLEEP_BETWEEN_TICKERS = 0.8  # TWSE rate-limit buffer
ALERT_THRESHOLD = 2.0  # % move that triggers an alert article

# Taiwan time is UTC+8; Taiwan date for YYYYMMDD API param
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# Key Taiwan semiconductor stocks: TWSE ticker → description + US equivalent
TICKERS: dict[str, dict] = {
    "2330": {"name": "TSMC",           "us": "TSM",    "sector": "foundry"},
    "2454": {"name": "MediaTek",       "us": "MDTKF",  "sector": "fabless"},
    "2303": {"name": "UMC",            "us": "UMC",    "sector": "foundry"},
    "2317": {"name": "Foxconn",        "us": "HNHPF",  "sector": "EMS"},
    "3034": {"name": "Novatek",        "us": "NOVKY",  "sector": "IC design"},
    "2382": {"name": "Quanta Computer","us": "QUCPF",  "sector": "ODM"},
    "2311": {"name": "ASE Technology", "us": "ASX",    "sector": "packaging"},
    "2357": {"name": "ASUS",           "us": "ASUUY",  "sector": "OEM"},
}

log = logging.getLogger("twse_semiconductor")


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("""CREATE TABLE IF NOT EXISTS seen_articles (
        id TEXT PRIMARY KEY,
        link TEXT,
        title TEXT,
        source TEXT,
        first_seen TEXT
    )""")
    conn.commit()


def _article_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _taiwan_date() -> str:
    """Return today's date in YYYYMMDD for TWSE API (Taiwan = UTC+8)."""
    from datetime import timezone, timedelta
    tw = datetime.now(timezone(timedelta(hours=8)))
    return tw.strftime("%Y%m%d")


def _fetch_ticker(twse_id: str, date_str: str) -> dict | None:
    """Fetch the most recent trading day row for a TWSE ticker."""
    url = (
        f"https://www.twse.com.tw/exchangeReport/STOCK_DAY"
        f"?response=json&date={date_str}&stockNo={twse_id}"
    )
    try:
        r = requests.get(url, headers={"User-Agent": _UA}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if data.get("stat") != "OK" or not data.get("data"):
            return None
        row = data["data"][-1]  # most recent trading day
        # Fields: 日期, 成交股數, 成交金額, 開盤價, 最高價, 最低價, 收盤價, 漲跌價差, 成交筆數
        date, vol, amount, open_, high, low, close, change, trades = row[:9]

        def _clean(s: str) -> float:
            return float(s.replace(",", "").replace("+", ""))

        close_f = _clean(close)
        change_f = _clean(change)
        prev_close = close_f - change_f
        pct = (change_f / prev_close * 100) if prev_close else 0.0

        return {
            "twse_id": twse_id,
            "date": date,
            "close": close_f,
            "change": change_f,
            "pct": pct,
            "high": _clean(high),
            "low": _clean(low),
            "open": _clean(open_),
            "volume": int(vol.replace(",", "")),
        }
    except Exception as e:
        log.warning("[twse] %s fetch failed: %s", twse_id, e)
        return None


def collect_twse_semiconductor() -> list[dict]:
    """Fetch TWSE data for tracked semis; return new synthetic article rows."""
    date_str = _taiwan_date()
    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    _ensure_db(conn)

    results: list[dict] = []
    snapshots: list[dict] = []

    for twse_id, meta in TICKERS.items():
        snap = _fetch_ticker(twse_id, date_str)
        if snap:
            snap["meta"] = meta
            snapshots.append(snap)
        time.sleep(SLEEP_BETWEEN_TICKERS)

    if not snapshots:
        conn.close()
        return []

    # Emit alert articles for large movers
    for s in snapshots:
        meta = s["meta"]
        pct = s["pct"]
        if abs(pct) < ALERT_THRESHOLD:
            continue

        direction = "surges" if pct > 0 else "drops"
        title = (
            f"TWSE: {meta['name']} ({meta['us']}) {direction} "
            f"{pct:+.1f}% to TWD {s['close']:.0f} in Taiwan trading"
        )
        link = f"https://tw.stock.yahoo.com/quote/{s['twse_id']}"
        art_id = _article_id(f"twse|{s['twse_id']}|{s['date']}|{pct:.1f}")

        try:
            conn.execute(
                "INSERT INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
                (art_id, link, title, SOURCE_NAME, now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            continue  # already seen

        summary = (
            f"{meta['name']} ({s['twse_id']}.TW / US: {meta['us']}) "
            f"closed at TWD {s['close']:.0f} ({pct:+.1f}%), "
            f"open {s['open']:.0f} high {s['high']:.0f} low {s['low']:.0f}. "
            f"Taiwan market date: {s['date']}."
        )
        results.append({
            "id": art_id,
            "link": link,
            "title": title,
            "summary": summary,
            "source": SOURCE_NAME,
            "first_seen": now,
            "published": now,
        })

    # Daily summary article (once per date)
    if snapshots:
        movers = sorted(snapshots, key=lambda x: abs(x["pct"]), reverse=True)
        top = movers[:3]
        top_str = ", ".join(
            f"{s['meta']['name']} {s['pct']:+.1f}%" for s in top
        )
        tsmc = next((s for s in snapshots if s["twse_id"] == "2330"), None)
        tsmc_str = f"TSMC {tsmc['pct']:+.1f}% (TWD {tsmc['close']:.0f})" if tsmc else "TSMC N/A"

        summary_title = f"Taiwan Semis Daily ({date_str}): {tsmc_str} | Top movers: {top_str}"
        summary_link = "https://www.twse.com.tw/en/"
        summary_id = _article_id(f"twse|daily_summary|{date_str}")

        try:
            conn.execute(
                "INSERT INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
                (summary_id, summary_link, summary_title, SOURCE_NAME, now),
            )
            conn.commit()
            full_lines = [
                f"{s['meta']['name']} ({s['twse_id']}): TWD {s['close']:.0f} {s['pct']:+.1f}%"
                for s in snapshots
            ]
            results.append({
                "id": summary_id,
                "link": summary_link,
                "title": summary_title,
                "summary": "Taiwan semiconductor sector performance: " + "; ".join(full_lines),
                "source": SOURCE_NAME,
                "first_seen": now,
                "published": now,
            })
        except sqlite3.IntegrityError:
            pass  # daily summary already emitted

    conn.close()
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"Fetching TWSE semiconductor data ({_taiwan_date()})...")
    articles = collect_twse_semiconductor()
    if articles:
        for a in articles:
            print(f"\nNEW: {a['title']}")
            print(f"     {a['summary'][:300]}")
    else:
        print("No new articles (all seen or no significant moves)")
