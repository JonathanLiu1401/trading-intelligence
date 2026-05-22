"""Intraday sector rotation detector.

Groups tickers by sector and compares mention velocity in the current 2h window
vs the prior 2h window.  Rising sectors indicate capital rotation or macro
catalysts; falling sectors flag fading interest.

Output: /home/zeph/logs/sector_rotation.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analytics.trend_velocity import _parse_ts, extract_tickers
from storage.article_store import _LIVE_ONLY_CLAUSE
import sqlite3

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/sector_rotation.json")
WINDOW_HOURS = 2
FETCH_LIMIT = 6000
TOP_N = 5

# Sector → set of representative tickers
SECTOR_MAP: dict[str, set[str]] = {
    "Technology": {
        "AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AMD", "INTC", "TSLA",
        "AVGO", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ADI",
        "NXPI", "ON", "SNPS", "CDNS", "ANSS", "PANW", "CRWD", "ZS", "OKTA",
        "NET", "DDOG", "SNOW", "PLTR", "RBLX", "U", "UBER", "LYFT", "DASH",
        "SHOP", "TWLO", "ZM", "DOCU", "DOCN", "GTLB", "MDB", "ESTC", "CFLT",
        "ORCL", "CRM", "SAP", "ADBE", "NOW", "WDAY", "INTU", "TEAM", "ATLASSIAN",
        "IBM", "HPQ", "DELL", "STX", "WDC", "SMCI", "PSTG", "NTNX",
    },
    "Finance": {
        "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF",
        "AXP", "DFS", "SYF", "ALLY", "KEY", "FITB", "HBAN", "RF", "MTB", "CFG",
        "BK", "STT", "SCHW", "IBKR", "RJF", "AMTD", "TD", "RY", "BMO", "BNS",
        "V", "MA", "PYPL", "FIS", "FI", "GPN", "WEX", "FLYW",
        "BLK", "SPGI", "MCO", "ICE", "CME", "CBOE", "NDAQ",
        "MET", "PRU", "AIG", "HIG", "TRV", "ALL", "CB", "PGR", "AJG", "WTW",
        "BRK", "BRKB",
    },
    "Healthcare": {
        "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "BMY", "AMGN", "GILD",
        "BIIB", "REGN", "VRTX", "SGEN", "MRNA", "BNTX", "ALNY", "INCY",
        "HZNP", "JAZZ", "EXEL", "RARE", "SRPT", "BLUE", "EDIT", "CRSP",
        "ABT", "TMO", "DHR", "A", "BIO", "IDXX", "IQV", "CRL", "MTD", "WAT",
        "CVS", "MCK", "ABC", "CAH", "HCA", "THC", "UHS", "CNC", "HUM", "MOH",
        "MDT", "EW", "BSX", "SYK", "ZBH", "DXCM", "ISRG", "RMD", "HOLX",
    },
    "Energy": {
        "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "BKR", "DVN", "MPC",
        "PSX", "VLO", "PBF", "HES", "APA", "OXY", "FANG", "PXD", "CTRA",
        "AR", "EQT", "RRC", "SWN", "COG", "CHX", "NOV",
        "NEE", "DUK", "SO", "AEP", "EXC", "XEL", "WEC", "ES", "PPL", "CNP",
        "LNG", "NFG", "KMI", "WMB", "ET", "TRGP", "OKE", "EPD",
    },
    "Consumer": {
        "AMZN", "WMT", "TGT", "COST", "KR", "DG", "DLTR", "BJ", "SFM",
        "HD", "LOW", "TJX", "ROST", "BURL", "M", "KSS", "GAP", "ANF", "AEO",
        "NKE", "LULU", "VFC", "PVH", "RL", "TPR", "CPRI",
        "MCD", "SBUX", "YUM", "QSR", "WING", "SHAK", "JACK", "TXRH", "DINE",
        "TSCO", "AAP", "AZO", "GPC", "O",
        "PG", "CL", "KMB", "CHD", "EL", "COTY", "REVG",
        "KO", "PEP", "MNST", "CELH", "KHC", "GIS", "K", "CPB", "SJM", "MKC",
    },
    "Industrials": {
        "GE", "HON", "MMM", "CAT", "DE", "EMR", "ETN", "PH", "ROK", "AME",
        "ITW", "IR", "XYL", "GNRC", "AOS", "REXR", "FTV", "ROP", "TDY",
        "GD", "LMT", "RTX", "NOC", "BA", "HII", "TXT", "L3H",
        "UPS", "FDX", "EXPD", "XPO", "SAIA", "ODFL", "JBHT", "KNX",
        "URI", "GATX", "AGCO", "PNR", "GWW", "WGO",
    },
    "Real Estate": {
        "SPG", "PLD", "AMT", "CCI", "EQIX", "DLR", "EXR", "PSA", "WELL",
        "VTR", "PEAK", "ARE", "BXP", "KIM", "REG", "FRT", "UDR", "EQR",
        "AVB", "AIV", "CPT", "MAA", "NNN", "O", "WPC", "STOR",
        "VICI", "MPW", "OHI", "SBAC", "SBA",
    },
    "Crypto": {
        "COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF", "BTBT",
        "BTDR", "HOOD", "WULF", "IREN", "CORZ", "GRPN",
        "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT",
        "MATIC", "LINK", "UNI", "LTC", "BCH", "ATOM", "ALGO", "XLM",
        "QUBT", "IONQ", "RGTI",
    },
    "Materials": {
        "LIN", "APD", "ECL", "PPG", "SHW", "NEM", "FCX", "SCCO", "AA",
        "ALB", "MP", "LTHM", "SQM", "LAC", "NOVT", "CF", "MOS", "NTR",
        "VMC", "MLM", "SUM", "EXP", "BECN",
    },
    "Biotech": {
        "SANA", "RXRX", "BEAM", "NTLA", "ACMR", "FOLD", "MGTX", "FATE",
        "ARVN", "KYMR", "KRTX", "IMVT", "PRAX", "ARQT", "ROIV", "GRTS",
        "TGTX", "IDYA", "PMV", "AGIO", "ATEA", "PRCT", "SAGE", "DAWN",
        "PRGO", "CRVS", "IMCR", "ZYMV", "KROS", "ACRS", "IOVA",
    },
}

# Build reverse lookup: ticker -> sector
_TICKER_TO_SECTOR: dict[str, str] = {}
for _sector, _tickers in SECTOR_MAP.items():
    for _tk in _tickers:
        _TICKER_TO_SECTOR[_tk] = _sector


def classify_tickers(tickers: list[str]) -> list[str]:
    """Return list of sectors for the given tickers (may have duplicates)."""
    return [_TICKER_TO_SECTOR[tk] for tk in tickers if tk in _TICKER_TO_SECTOR]


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")

    cur = conn.execute(
        "SELECT first_seen, title FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (FETCH_LIMIT,),
    )
    rows = cur.fetchall()

    if not rows:
        print("sector_rotation: no rows", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    cur_cut = now - timedelta(hours=WINDOW_HOURS)
    prev_cut = now - timedelta(hours=WINDOW_HOURS * 2)

    cur_c: Counter[str] = Counter()
    prev_c: Counter[str] = Counter()
    article_cur = 0
    article_prev = 0

    for fs, title in rows:
        ts = _parse_ts(fs)
        if ts is None:
            continue
        tickers = extract_tickers(title)
        sectors = classify_tickers(tickers)
        if not sectors:
            continue
        if ts >= cur_cut:
            cur_c.update(sectors)
            article_cur += 1
        elif ts >= prev_cut:
            prev_c.update(sectors)
            article_prev += 1

    all_sectors = set(cur_c) | set(prev_c)
    movers = []
    for s in all_sectors:
        c = cur_c.get(s, 0)
        p = prev_c.get(s, 0)
        delta = c - p
        ratio = (c + 1) / (p + 1)
        movers.append((s, c, p, delta, ratio))

    movers.sort(key=lambda r: (r[3], r[4]), reverse=True)
    gainers = movers[:TOP_N]
    losers = sorted(movers, key=lambda r: (r[3], r[4]))[:TOP_N]

    payload = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "scanned": len(rows),
        "articles_with_sector_cur": article_cur,
        "articles_with_sector_prev": article_prev,
        "gainers": [
            {
                "sector": s,
                "now": c,
                "prev": p,
                "delta": d,
                "ratio": round(r, 2),
            }
            for s, c, p, d, r in gainers
        ],
        "losers": [
            {
                "sector": s,
                "now": c,
                "prev": p,
                "delta": d,
                "ratio": round(r, 2),
            }
            for s, c, p, d, r in losers if d < 0
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    print(
        f"sector_rotation: scanned={len(rows)} "
        f"cur_articles={article_cur} prev_articles={article_prev}"
    )
    print("  TOP GAINERS:")
    for s, c, p, d, r in gainers:
        sign = "+" if d >= 0 else ""
        print(f"    {s}: now={c} prev={p} delta={sign}{d} ratio={r:.2f}x")
    print("  TOP LOSERS:")
    for s, c, p, d, r in losers:
        if d < 0:
            print(f"    {s}: now={c} prev={p} delta={d} ratio={r:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
