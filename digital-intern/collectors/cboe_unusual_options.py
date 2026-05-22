"""CBOE unusual options flow detector.

Uses CBOE's free delayed-quote API to detect unusual options activity
across portfolio and watchlist tickers:

  - Volume/OI ratio > 5 on contracts with > 50 volume (fresh positioning)
  - Top-volume contracts with >500 volume regardless of OI ratio
  - Significant call/put skew by volume

API: https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json
No auth required; data is delayed ~15 min during market hours.

Emits a single synthetic article per ticker when unusual flow is detected,
deduped by (ticker, date, top-contract) so we don't repeat the same signal.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger("cboe_unusual_options")

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

SOURCE_NAME = "CBOE Unusual Options"
CBOE_API = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
REQUEST_TIMEOUT = 10
MAX_WORKERS = 8

# Thresholds for unusual activity
MIN_VOLUME = 50          # ignore tiny contracts
VOL_OI_RATIO = 5.0      # volume > 5x open interest → fresh positioning
HIGH_VOL_ABSOLUTE = 500  # report even if OI unknown/high

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Extra liquid names always worth watching
_CORE_TICKERS = [
    "SPY", "QQQ", "IWM", "NVDA", "AMD", "TSLA", "AAPL", "MSFT",
    "AMZN", "META", "GOOGL", "SMCI", "SOFI",
]


def _load_tickers() -> list[str]:
    tickers: set[str] = set(_CORE_TICKERS)
    try:
        with open(PORTFOLIO_PATH) as f:
            pf = json.load(f)
        positions = pf.get("positions", [])
        if isinstance(positions, list):
            for p in positions:
                t = p.get("ticker", "")
                if t:
                    tickers.add(t.upper())
        for t in pf.get("sector_watchlist", []):
            tickers.add(t.upper())
    except Exception:
        pass
    try:
        with open(WATCHLIST_PATH) as f:
            wl = json.load(f)
        for key in ("memory_core", "semis_equipment", "broader_semis", "portfolio"):
            for t in wl.get(key, []):
                tickers.add(t.upper())
    except Exception:
        pass
    return sorted(tickers)


def _parse_option_symbol(sym: str) -> dict | None:
    """Parse OCC option symbol like NVDA260522C00050000."""
    m = re.match(r"^([A-Z]{1,5})(\d{6})([CP])(\d{8})$", sym)
    if not m:
        return None
    ticker, expiry_raw, opt_type, strike_raw = m.groups()
    try:
        expiry = datetime.strptime(expiry_raw, "%y%m%d").strftime("%Y-%m-%d")
        strike = int(strike_raw) / 1000.0
    except (ValueError, OverflowError):
        return None
    return {"ticker": ticker, "expiry": expiry, "type": opt_type, "strike": strike}


def _fetch_cboe(symbol: str) -> dict | None:
    url = CBOE_API.format(symbol=symbol)
    try:
        r = requests.get(url, headers={"User-Agent": _UA}, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        log.debug(f"[cboe] {symbol} fetch error: {e}")
        return None


def _find_unusual(cboe_data: dict) -> list[dict]:
    """Extract unusual contracts from CBOE options data."""
    data = cboe_data.get("data", {})
    if not isinstance(data, dict):
        return []

    current_price = data.get("current_price")
    iv30 = data.get("iv30")
    options = data.get("options", [])
    if not options:
        return []

    unusual = []
    for opt in options:
        sym = opt.get("option", "")
        parsed = _parse_option_symbol(sym)
        if not parsed:
            continue

        volume = opt.get("volume") or 0
        oi = opt.get("open_interest") or 0
        iv = opt.get("iv") or 0
        bid = opt.get("bid") or 0
        ask = opt.get("ask") or 0

        if volume < MIN_VOLUME:
            continue

        vol_oi_ratio = (volume / oi) if oi > 1 else 99.0
        is_unusual = vol_oi_ratio >= VOL_OI_RATIO or volume >= HIGH_VOL_ABSOLUTE

        if not is_unusual:
            continue

        unusual.append({
            **parsed,
            "volume": int(volume),
            "open_interest": int(oi),
            "vol_oi_ratio": round(vol_oi_ratio, 1),
            "iv_pct": round(iv * 100, 1) if iv else None,
            "bid": bid,
            "ask": ask,
            "current_price": current_price,
            "iv30": iv30,
        })

    # Sort by volume descending
    unusual.sort(key=lambda x: x["volume"], reverse=True)
    return unusual[:10]  # top 10 unusual contracts per ticker


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
    return hashlib.sha256(key.encode()).hexdigest()


def _build_article(ticker: str, unusual: list[dict]) -> dict | None:
    if not unusual:
        return None

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Top contract drives the dedup key
    top = unusual[0]
    dedup_key = f"cboe_unusual|{ticker}|{today}|{top['type']}|{top['expiry']}|{top['strike']}"
    art_id = _article_id(dedup_key)

    current_price = top.get("current_price")
    price_str = f" (stock @ ${current_price:.2f})" if current_price else ""

    # Summarize top contracts
    call_count = sum(1 for u in unusual if u["type"] == "C")
    put_count = sum(1 for u in unusual if u["type"] == "P")
    total_vol = sum(u["volume"] for u in unusual)
    skew = "CALL HEAVY" if call_count > put_count else ("PUT HEAVY" if put_count > call_count else "NEUTRAL")

    lines = []
    for u in unusual[:5]:
        opt_type = "CALL" if u["type"] == "C" else "PUT"
        ratio_str = f"vol/OI={u['vol_oi_ratio']:.1f}x" if u["open_interest"] > 1 else "vol/OI=NEW"
        iv_str = f" IV={u['iv_pct']:.0f}%" if u["iv_pct"] else ""
        lines.append(
            f"  {opt_type} ${u['strike']:.0f} {u['expiry']}: "
            f"vol={u['volume']:,} OI={u['open_interest']:,} {ratio_str}{iv_str}"
        )

    title = (
        f"UNUSUAL OPTIONS: {ticker}{price_str} — {skew} | "
        f"{total_vol:,} unusual vol | {today}"
    )
    summary = (
        f"CBOE unusual options flow detected for {ticker}{price_str}. "
        f"Skew: {skew} ({call_count} call / {put_count} put unusual contracts). "
        f"Top unusual contracts (vol/OI > {VOL_OI_RATIO}x or vol > {HIGH_VOL_ABSOLUTE}):\n"
        + "\n".join(lines)
    )

    link = f"https://www.cboe.com/delayed_quotes/options/{ticker}"

    return {
        "id": art_id,
        "link": link,
        "title": title,
        "summary": summary,
        "source": SOURCE_NAME,
        "published": now,
        "_unusual_contracts": unusual,
        "_ticker": ticker,
        "_skew": skew,
        "_call_count": call_count,
        "_put_count": put_count,
    }


def collect_cboe_unusual_options() -> list[dict]:
    """Fetch and return unusual options flow articles."""
    tickers = _load_tickers()
    log.info(f"[cboe_unusual] scanning {len(tickers)} tickers")

    results: list[dict] = []
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    _ensure_db(conn)

    def _process(ticker: str) -> dict | None:
        data = _fetch_cboe(ticker)
        if not data:
            return None
        unusual = _find_unusual(data)
        if not unusual:
            return None
        return _build_article(ticker, unusual)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_process, t): t for t in tickers}
        for fut in as_completed(futures):
            art = fut.result()
            if not art:
                continue
            art_id = art["id"]
            try:
                if conn.execute(
                    "SELECT 1 FROM seen_articles WHERE id=?", (art_id,)
                ).fetchone():
                    log.debug(f"[cboe_unusual] already seen: {art['_ticker']}")
                    continue
                now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                conn.execute(
                    "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                    "VALUES (?,?,?,?,?)",
                    (art_id, art["link"], art["title"], SOURCE_NAME, now),
                )
                conn.commit()
                results.append(art)
                log.info(f"[cboe_unusual] NEW: {art['title']}")
            except sqlite3.Error as e:
                log.warning(f"[cboe_unusual] db error: {e}")

    conn.close()
    log.info(f"[cboe_unusual] {len(results)} new unusual-flow articles")
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    articles = collect_cboe_unusual_options()
    if not articles:
        print("No new unusual options flow detected (all already seen or no unusual activity)")
        sys.exit(0)
    print(f"\n=== {len(articles)} UNUSUAL OPTIONS SIGNALS ===\n")
    for a in articles:
        print(f"TICKER: {a['_ticker']}")
        print(f"TITLE:  {a['title']}")
        print(f"DETAIL: {a['summary'][:300]}")
        print()
