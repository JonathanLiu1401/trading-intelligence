"""DTCC / SEC Fails-to-Deliver (FTD) data collector.

SEC publishes consolidated NSCC fails-to-deliver data twice per month:
  - Period A: settlement dates 1–15  (released ~4 business days after period end)
  - Period B: settlement dates 16–EOM (released ~4 business days after period end)

Data URL: https://www.sec.gov/files/data/fails-deliver-data/cnsfails{YYYYMM}{a|b}.zip
No API key required.

Format (pipe-delimited):
  SETTLEMENT DATE | CUSIP | SYMBOL | QUANTITY (FAILS) | DESCRIPTION | PRICE

What we emit:
  - An article per ticker in portfolio/watchlist with any FTD data
  - An article per ticker whose FTD value (qty × price) exceeds $5M (market-wide signal)
  - A summary article for the full period ("Top FTD stocks: X, Y, Z")

FTD signals matter because:
  - Persistently high FTD = potential short squeeze fuel
  - Large FTD in a specific ticker often precedes SEC enforcement or forced buy-ins
  - Spikes in FTD coincide with extreme borrowing demand / negative rebate rates

Dedup: (symbol, period_tag) keyed in seen_articles.db so each half-month
period fires at most once per ticker.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import sqlite3
import zipfile
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import requests

log = logging.getLogger("dtcc_ftd")

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

SOURCE = "SEC/DTCC FTD"
REQUEST_TIMEOUT = 20
USER_AGENT = "Digital-Intern-Research contact@digital-intern.local"

# Minimum FTD dollar value to flag market-wide (qty × price)
MIN_DOLLAR_THRESHOLD = 5_000_000  # $5M

# How many periods back to look (in case latest isn't published yet)
LOOKBACK_PERIODS = 4


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


def _already_seen(conn: sqlite3.Connection, key: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM seen_articles WHERE id=?", (key,)
    ).fetchone()
    return row is not None


def _mark_seen(conn: sqlite3.Connection, key: str, title: str, link: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
        (key, link, title, SOURCE, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _dedup_key(symbol: str, period_tag: str) -> str:
    raw = f"ftd:{symbol}:{period_tag}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_tracked_tickers() -> set[str]:
    tickers: set[str] = set()
    for path in (PORTFOLIO_PATH, WATCHLIST_PATH):
        try:
            with open(path) as f:
                data = json.load(f)
            # portfolio: positions list with 'ticker' key
            for pos in data.get("positions", []):
                t = (pos.get("ticker") or "").strip().upper()
                if t:
                    tickers.add(t)
            # options underlying
            for opt in data.get("options", []):
                t = (opt.get("underlying") or "").strip().upper()
                if t:
                    tickers.add(t)
            # watchlist: flat list or dict with 'tickers'
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        tickers.add(item.strip().upper())
            for t in data.get("tickers", []):
                tickers.add(str(t).strip().upper())
        except (FileNotFoundError, KeyError, ValueError):
            pass
    return tickers


def _candidate_periods(today: date) -> list[tuple[int, int, str]]:
    """Return (year, month, half) tuples to try, most recent first.

    half = 'b' for second half of current month if past mid-month,
    'a' for first half of current month, etc.
    """
    candidates = []
    # Generate last LOOKBACK_PERIODS half-months
    cur = today
    for _ in range(LOOKBACK_PERIODS):
        if cur.day > 15:
            candidates.append((cur.year, cur.month, "b"))
            candidates.append((cur.year, cur.month, "a"))
        else:
            candidates.append((cur.year, cur.month, "a"))
        # Step back one month
        if cur.month == 1:
            cur = cur.replace(year=cur.year - 1, month=12, day=28)
        else:
            cur = cur.replace(month=cur.month - 1, day=28)
    return candidates


def _fetch_ftd_zip(year: int, month: int, half: str) -> bytes | None:
    """Fetch the zip file bytes for a given period, or None on failure."""
    yyyymm = f"{year}{month:02d}"
    url = (
        f"https://www.sec.gov/files/data/fails-deliver-data/"
        f"cnsfails{yyyymm}{half}.zip"
    )
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            log.info("FTD: fetched %s (%d bytes)", url, len(resp.content))
            return resp.content
        log.debug("FTD: %s → HTTP %d", url, resp.status_code)
    except Exception as exc:
        log.warning("FTD: fetch error %s: %s", url, exc)
    return None


def _parse_ftd_zip(data: bytes, year: int, month: int, half: str) -> dict[str, dict]:
    """Parse zip → dict mapping symbol → {qty, price, dollar_value, date_str, description}."""
    records: dict[str, dict] = {}
    yyyymm = f"{year}{month:02d}"
    fname = f"cnsfails{yyyymm}{half}.txt"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            # The file name may vary slightly; find the matching member
            members = zf.namelist()
            target = next((m for m in members if m.lower().endswith(".txt")), None)
            if not target:
                log.warning("FTD: no .txt in zip, members: %s", members)
                return records
            raw = zf.read(target).decode("utf-8", errors="replace")
    except zipfile.BadZipFile as exc:
        log.warning("FTD: bad zip: %s", exc)
        return records

    for line in raw.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 6:
            continue
        date_str, cusip, symbol, qty_str, description, price_str = parts[:6]
        if symbol.upper() == "SYMBOL":
            continue  # header row
        try:
            qty = int(qty_str.replace(",", ""))
            price = float(price_str.replace(",", "")) if price_str.strip() else 0.0
        except ValueError:
            continue
        dollar_val = qty * price
        symbol = symbol.strip().upper()
        if symbol not in records or records[symbol]["dollar_value"] < dollar_val:
            records[symbol] = {
                "qty": qty,
                "price": price,
                "dollar_value": dollar_val,
                "date_str": date_str.strip(),
                "description": description.strip(),
                "cusip": cusip.strip(),
            }

    log.info("FTD: parsed %d unique symbols from %s%s", len(records), yyyymm, half)
    return records


def _build_article(
    symbol: str,
    info: dict,
    period_tag: str,
    reason: str,
) -> dict:
    qty = info["qty"]
    price = info["price"]
    dval = info["dollar_value"]
    desc = info["description"] or symbol
    date_str = info["date_str"]

    dval_str = f"${dval/1_000_000:.1f}M" if dval >= 1_000_000 else f"${dval:,.0f}"
    title = (
        f"FTD Alert {symbol}: {qty:,} shares failed (≈{dval_str}) — "
        f"{period_tag} [{reason}]"
    )
    summary = (
        f"SEC/DTCC Fails-to-Deliver for {symbol} ({desc}): "
        f"{qty:,} shares undelivered as of {date_str}, "
        f"at ${price:.2f}/sh = {dval_str} total FTD value. "
        f"Period: {period_tag}. "
        f"High FTD can indicate naked short pressure, forced buy-in risk, "
        f"or squeeze setup. Source: SEC EDGAR FTD dataset."
    )
    link = (
        f"https://www.sec.gov/data/foiadocsfailsdatahtm#{period_tag}-{symbol}"
    )
    return {
        "title": title,
        "link": link,
        "summary": summary,
        "source": SOURCE,
        "published": datetime.now(timezone.utc).isoformat(),
    }


def collect_dtcc_ftd() -> list[dict]:
    """Main entry point. Returns list of article dicts."""
    today = date.today()
    tracked = _load_tracked_tickers()

    conn = sqlite3.connect(str(DB_PATH))
    _ensure_db(conn)

    articles: list[dict] = []

    for year, month, half in _candidate_periods(today):
        period_tag = f"{year}{month:02d}{half}"

        # Check if we already processed this period's summary
        summary_key = _dedup_key("__SUMMARY__", period_tag)
        if _already_seen(conn, summary_key):
            log.debug("FTD: period %s already processed, skipping", period_tag)
            continue

        raw = _fetch_ftd_zip(year, month, half)
        if raw is None:
            continue  # not published yet or network error

        records = _parse_ftd_zip(raw, year, month, half)
        if not records:
            continue

        period_articles: list[dict] = []

        # 1. Tracked tickers
        for sym in tracked:
            if sym in records:
                key = _dedup_key(sym, period_tag)
                if not _already_seen(conn, key):
                    art = _build_article(sym, records[sym], period_tag, "portfolio/watchlist")
                    period_articles.append(art)
                    _mark_seen(conn, key, art["title"], art["link"])

        # 2. Large FTD by dollar value (market-wide signals)
        large_ftd = [
            (sym, info) for sym, info in records.items()
            if info["dollar_value"] >= MIN_DOLLAR_THRESHOLD
        ]
        large_ftd.sort(key=lambda x: x[1]["dollar_value"], reverse=True)

        for sym, info in large_ftd[:20]:  # top 20 large FTD
            key = _dedup_key(sym, period_tag)
            if not _already_seen(conn, key):
                art = _build_article(sym, info, period_tag, f">$5M FTD")
                period_articles.append(art)
                _mark_seen(conn, key, art["title"], art["link"])

        # 3. Summary article for top FTD stocks
        if large_ftd:
            top_names = ", ".join(s for s, _ in large_ftd[:5])
            top_val = sum(i["dollar_value"] for _, i in large_ftd[:10])
            summary_title = (
                f"SEC FTD Report {period_tag}: Top failures — {top_names} "
                f"(top-10 total ≈${top_val/1_000_000:.0f}M)"
            )
            summary_art = {
                "title": summary_title,
                "link": "https://www.sec.gov/data/foiadocsfailsdatahtm",
                "summary": (
                    f"DTCC Fails-to-Deliver summary for period {period_tag}. "
                    f"{len(large_ftd)} tickers exceeded $5M in FTD value. "
                    f"Top symbols: {top_names}. "
                    f"Top-10 combined FTD ≈${top_val/1_000_000:.0f}M. "
                    f"FTD data lags ~4 business days from period end."
                ),
                "source": SOURCE,
                "published": datetime.now(timezone.utc).isoformat(),
            }
            period_articles.append(summary_art)
            _mark_seen(conn, summary_key, summary_title, summary_art["link"])

        articles.extend(period_articles)
        log.info(
            "FTD: period %s → %d new articles (%d tracked, %d large)",
            period_tag, len(period_articles), len(tracked & records.keys()), len(large_ftd),
        )

        # Once we find a valid period, stop (don't re-process older ones each cycle)
        break

    conn.close()
    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = collect_dtcc_ftd()
    print(f"\n=== DTCC FTD Collector: {len(results)} articles ===\n")
    for art in results[:10]:
        print(f"  TITLE:   {art['title']}")
        print(f"  LINK:    {art['link']}")
        print(f"  SUMMARY: {art['summary'][:200]}...")
        print()
    if not results:
        print("  (no new FTD data — may already be seen or not yet published)")
