"""CFTC Commitment of Traders (COT) — Traders in Financial Futures (TFF) collector.

Pulls the weekly TFF report from the CFTC, which shows positioning by:
  - Dealer/Intermediary (banks/broker-dealers)
  - Asset Manager/Institutional (pension funds, mutual funds — "smart money")
  - Leveraged Funds (hedge funds — often leading/contrarian)
  - Other Large Traders + Non-reportable (retail)

Published every Friday at 3:30 PM ET covering the previous Tuesday.
Emits one article per tracked contract when a new report is available.
No API key required.
"""
import csv
import hashlib
import io
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from core.logger import get_logger
    log = get_logger("cftc_cot")
except Exception:
    log = logging.getLogger("cftc_cot")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

TFF_URL = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"
HTTP_TIMEOUT = 20
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

# Contracts to track: (exact name fragment to match, short label, asset class)
# Source: CFTC FinFutWk.txt (Traders in Financial Futures format).
# This file covers equity index, FX, rates, and crypto futures only.
# Commodities (Gold, Oil) are in the separate Legacy COT file; not tracked here.
TRACKED = [
    ("E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",         "E-Mini S&P 500",    "equity"),
    ("NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE",             "Nasdaq Mini",        "equity"),
    ("RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE",          "Russell E-Mini",     "equity"),
    ("MICRO E-MINI S&P 500 INDEX - CHICAGO MERCANTILE EXCHANGE", "Micro E-Mini S&P", "equity"),
    ("EURO FX - CHICAGO MERCANTILE EXCHANGE",                 "EUR/USD",            "fx"),
    ("JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",            "JPY/USD",            "fx"),
    ("BRITISH POUND - CHICAGO MERCANTILE EXCHANGE",           "GBP/USD",            "fx"),
    ("BITCOIN - CHICAGO MERCANTILE EXCHANGE",                 "Bitcoin (CME)",      "crypto"),
    ("ETHER CASH SETTLED - CHICAGO MERCANTILE EXCHANGE",      "Ether (CME)",        "crypto"),
]

# TFF column indices (0-based, from CFTC TFF layout):
# 0=Name, 2=Date, 7=OI
# 8=Dealer_L, 9=Dealer_S, 10=Dealer_Spread
# 11=AssetMgr_L, 12=AssetMgr_S, 13=AssetMgr_Spread
# 14=LevFund_L (actually col 14 is Other_L in some docs, let's verify via actual col count)
# For TFF specifically: 8-10=Dealer, 11-13=AssetMgr, 14-16=LevFund, 17-19=Other, 20-22=Non-report
COL_OI        = 7
COL_DEALER_L  = 8
COL_DEALER_S  = 9
COL_ASSETMGR_L = 11
COL_ASSETMGR_S = 12
COL_LEVFUND_L  = 14
COL_LEVFUND_S  = 15


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
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


def _seen(conn: sqlite3.Connection, sid: str) -> bool:
    return conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (sid,)).fetchone() is not None


def _mark(conn: sqlite3.Connection, sid: str, link: str, title: str, source: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
        (sid, link, title, source, datetime.now(timezone.utc).isoformat()),
    )


def _int(s: str) -> int:
    try:
        return int(s.strip().replace(",", ""))
    except (ValueError, AttributeError):
        return 0


def _net_signal(longs: int, shorts: int) -> str:
    if longs + shorts == 0:
        return "flat"
    ratio = longs / (longs + shorts)
    if ratio > 0.70:
        return "strongly bullish"
    if ratio > 0.55:
        return "bullish"
    if ratio < 0.30:
        return "strongly bearish"
    if ratio < 0.45:
        return "bearish"
    return "neutral"


def _fetch_tff() -> list[list[str]]:
    r = requests.get(TFF_URL, headers={"User-Agent": _UA}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    reader = csv.reader(io.StringIO(r.text))
    return list(reader)


def collect_cftc_cot() -> list[dict]:
    try:
        rows = _fetch_tff()
    except Exception as e:
        log.warning("cftc_cot: fetch failed: %s", e)
        return []

    conn = _ensure_db()
    articles: list[dict] = []

    for row in rows:
        if len(row) < 23:
            continue
        name = row[0].strip('"').strip()
        report_date = row[2].strip()

        for search_term, label, asset_class in TRACKED:
            if search_term.upper() != name.upper():
                continue

            # Dedup: one article per (contract, report_date)
            sid = hashlib.sha256(f"cftc_cot:{name}:{report_date}".encode()).hexdigest()
            if _seen(conn, sid):
                break

            oi           = _int(row[COL_OI])
            dealer_l     = _int(row[COL_DEALER_L])
            dealer_s     = _int(row[COL_DEALER_S])
            assetmgr_l   = _int(row[COL_ASSETMGR_L])
            assetmgr_s   = _int(row[COL_ASSETMGR_S])
            levfund_l    = _int(row[COL_LEVFUND_L])
            levfund_s    = _int(row[COL_LEVFUND_S])

            dealer_net   = dealer_l - dealer_s
            assetmgr_net = assetmgr_l - assetmgr_s
            levfund_net  = levfund_l - levfund_s

            hf_signal  = _net_signal(levfund_l, levfund_s)
            inst_signal = _net_signal(assetmgr_l, assetmgr_s)

            title = (
                f"[COT {report_date}] {label}: HedgeFunds {hf_signal} "
                f"(net {levfund_net:+,}), Institutions {inst_signal} "
                f"(net {assetmgr_net:+,}), OI={oi:,}"
            )
            summary = (
                f"CFTC COT Traders in Financial Futures report dated {report_date}. "
                f"Contract: {name}. Open Interest: {oi:,}. "
                f"Dealer/Intermediary: L={dealer_l:,} S={dealer_s:,} net={dealer_net:+,}. "
                f"Asset Manager/Institutional: L={assetmgr_l:,} S={assetmgr_s:,} net={assetmgr_net:+,} ({inst_signal}). "
                f"Leveraged Funds (hedge funds): L={levfund_l:,} S={levfund_s:,} net={levfund_net:+,} ({hf_signal}). "
                f"Asset class: {asset_class}."
            )
            link = TFF_URL

            art = {
                "title": title,
                "link": link,
                "summary": summary,
                "published": datetime.now(timezone.utc).isoformat(),
                "source": "cftc_cot",
                "_id": sid,
                "_asset_class": asset_class,
            }
            _mark(conn, sid, link, title, "cftc_cot")
            articles.append(art)
            log.info("cftc_cot: new %s %s", report_date, label)
            break  # matched, move to next row

    conn.commit()
    conn.close()
    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.insert(0, str(BASE_DIR))

    articles = collect_cftc_cot()
    print(f"\n=== CFTC COT Collector ===")
    print(f"New articles: {len(articles)}")
    for a in articles:
        print(f"  {a['title']}")
    if articles:
        print(f"\nSample summary:\n{articles[0]['summary']}")
