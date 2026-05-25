"""Earnings expected-move calculator — IV-implied pre-earnings price range.

For each ticker with earnings in the next N days, fetches the CBOE delayed
options chain and computes the ATM straddle price as a % of stock price.
This is the market's implied expected move: if the 5-DTE straddle costs $10
on a $100 stock, the market implies ±10% by expiry.

Why it matters:
  • Identifies high-IV pre-earnings setups worth watching
  • Gives context for position sizing / hedging decisions
  • Abnormally high or low expected moves are signals in themselves

Data sources:
  • Earnings dates: NASDAQ earnings calendar API (no key) + local earnings_calendar.json
  • Options data: CBOE delayed-quote API (15-min delay, no key)

Collector contract:
  Returns list[dict] with standard {title, link, summary, published, source,
  _tickers} keys — same shape as every other collector.

Dedup:
  seen_articles.db keyed by sha256(ticker + earnings_date) so the same signal
  only emits once per calendar day. A revised expected move (e.g. if IV spikes
  the day before) re-emits by using (ticker + earnings_date + int(pct/2)*2)
  to allow ~2-pct-point updates to surface without flooding.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

try:
    from core.logger import get_logger
    log = get_logger("earnings_iv_move")
except Exception:
    log = logging.getLogger("earnings_iv_move")

BASE_DIR = Path(__file__).resolve().parent.parent
EARNINGS_SNAP = BASE_DIR / "data" / "earnings_calendar.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

CBOE_API = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
NASDAQ_CAL_API = (
    "https://api.nasdaq.com/api/calendar/earnings?date={date}"
)
SOURCE_NAME = "earnings_iv_move"
REQUEST_TIMEOUT = 12
# Emit an article only when expected move >= this threshold (avoid noise on illiquid names)
MIN_EXPECTED_MOVE_PCT = 4.0
# Look ahead this many calendar days for earnings
EARNINGS_LOOKBACK_DAYS = 5

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_NASDAQ_HEADERS = {
    "User-Agent": _UA,
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nasdaq.com/",
}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

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


def _article_id(ticker: str, earnings_date: str, move_bucket: int) -> str:
    key = f"{SOURCE_NAME}||{ticker}||{earnings_date}||{move_bucket}"
    return hashlib.sha256(key.encode()).hexdigest()


def _already_seen(conn: sqlite3.Connection, aid: str) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM seen_articles WHERE id=?", (aid,)
    ).fetchone())


def _mark_seen(conn: sqlite3.Connection, aid: str, ticker: str, title: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles(id,link,title,source,first_seen)"
            " VALUES(?,?,?,?,?)",
            (aid, f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json",
             title, SOURCE_NAME, now),
        )
        conn.commit()
    except sqlite3.OperationalError as e:
        # DB lock under daemon contention — non-fatal; article is still returned
        log.debug(f"[earnings_iv_move] seen_db mark failed ({e}), continuing")


# ---------------------------------------------------------------------------
# Option symbol parsing
# ---------------------------------------------------------------------------

_OPT_RE = re.compile(r'^([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d+)$')


def _parse_opt(sym: str) -> dict | None:
    m = _OPT_RE.match(sym)
    if not m:
        return None
    ticker, yy, mm, dd, otype, strike_raw = m.groups()
    return {
        "expiry": f"20{yy}-{mm}-{dd}",
        "type": otype,
        "strike": float(strike_raw) / 1000.0,
    }


# ---------------------------------------------------------------------------
# CBOE expected-move calculation
# ---------------------------------------------------------------------------

def _calc_expected_move(symbol: str) -> dict | None:
    """Return expected-move dict for *symbol* or None on any failure."""
    url = CBOE_API.format(symbol=symbol)
    try:
        r = requests.get(url, headers={"User-Agent": _UA}, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        log.debug(f"[earnings_iv_move] {symbol} fetch error: {e}")
        return None
    if r.status_code != 200:
        log.debug(f"[earnings_iv_move] {symbol} HTTP {r.status_code}")
        return None

    raw = r.json().get("data", {})
    stock_price = float(raw.get("current_price") or 0)
    if stock_price <= 0:
        return None

    # Group options by expiry
    by_expiry: dict[str, dict[str, dict]] = defaultdict(
        lambda: {"calls": {}, "puts": {}}
    )
    for opt in raw.get("options", []):
        parsed = _parse_opt(opt.get("option", ""))
        if not parsed:
            continue
        bid = float(opt.get("bid") or 0)
        ask = float(opt.get("ask") or 0)
        mid = (bid + ask) / 2.0
        if mid <= 0:
            continue
        expiry = parsed["expiry"]
        strike = parsed["strike"]
        side = "calls" if parsed["type"] == "C" else "puts"
        by_expiry[expiry][side][strike] = mid

    today_str = str(date.today())
    # Nearest expiry that has both calls and puts and is after today
    valid = sorted(
        e for e in by_expiry
        if e > today_str
        and by_expiry[e]["calls"]
        and by_expiry[e]["puts"]
    )
    if not valid:
        return None

    nearest = valid[0]
    calls = by_expiry[nearest]["calls"]
    puts = by_expiry[nearest]["puts"]

    common = set(calls) & set(puts)
    if not common:
        return None

    atm = min(common, key=lambda s: abs(s - stock_price))
    straddle = calls[atm] + puts[atm]
    move_pct = (straddle / stock_price) * 100.0

    return {
        "symbol": symbol,
        "price": round(stock_price, 2),
        "expiry": nearest,
        "atm_strike": atm,
        "straddle": round(straddle, 2),
        "expected_move_pct": round(move_pct, 2),
    }


# ---------------------------------------------------------------------------
# Earnings ticker collection
# ---------------------------------------------------------------------------

def _get_earnings_tickers(days_ahead: int = EARNINGS_LOOKBACK_DAYS) -> dict[str, str]:
    """Return {ticker: earnings_date_str} for companies reporting in next N days."""
    tickers: dict[str, str] = {}
    today = date.today()

    # 1. Load from local portfolio snapshot (most important for us)
    if EARNINGS_SNAP.exists():
        try:
            snap = json.loads(EARNINGS_SNAP.read_text())
            for ev in snap.get("events", []):
                sym = ev.get("ticker", "").upper().strip()
                ed = ev.get("earnings_date", "")
                if sym and ed:
                    tickers[sym] = ed[:10]
        except Exception as e:
            log.debug(f"[earnings_iv_move] snapshot load error: {e}")

    # 2. Pull from NASDAQ broader calendar for each upcoming trading day
    for offset in range(days_ahead):
        target = today + timedelta(days=offset)
        if target.weekday() >= 5:  # skip weekends
            continue
        date_str = target.strftime("%Y-%m-%d")
        try:
            resp = requests.get(
                NASDAQ_CAL_API.format(date=date_str),
                headers=_NASDAQ_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                continue
            payload = resp.json()
            rows = (
                payload.get("data", {}).get("rows")
                or payload.get("data", {}).get("earnings", {}).get("rows")
                or []
            )
            if not rows:
                continue
            for row in rows:
                sym = (row.get("symbol") or row.get("ticker") or "").upper().strip()
                if sym and sym not in tickers:
                    tickers[sym] = date_str
        except Exception as e:
            log.debug(f"[earnings_iv_move] NASDAQ cal {date_str} error: {e}")

    return tickers


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

def collect_earnings_iv_move(max_tickers: int = 30) -> list[dict]:
    """
    Fetch expected moves for upcoming earnings tickers.

    Returns a list of standard article dicts, one per ticker where the
    expected move >= MIN_EXPECTED_MOVE_PCT and not already seen today.
    """
    conn = _ensure_db()
    articles: list[dict] = []

    earnings_map = _get_earnings_tickers()
    if not earnings_map:
        log.debug("[earnings_iv_move] no upcoming earnings found")
        return []

    log.info(f"[earnings_iv_move] checking {len(earnings_map)} upcoming earnings tickers")

    # Sort by closeness of earnings date (soonest first), limit to max_tickers
    today = date.today()

    def days_away(d_str: str) -> int:
        try:
            return (date.fromisoformat(d_str[:10]) - today).days
        except Exception:
            return 999

    sorted_tickers = sorted(earnings_map.items(), key=lambda kv: days_away(kv[1]))[:max_tickers]

    for ticker, earnings_date in sorted_tickers:
        result = _calc_expected_move(ticker)
        if not result:
            continue

        move_pct = result["expected_move_pct"]
        if move_pct < MIN_EXPECTED_MOVE_PCT:
            continue

        # Bucket by 2% intervals to allow re-emit on big IV spikes
        move_bucket = int(move_pct / 2) * 2
        aid = _article_id(ticker, earnings_date, move_bucket)
        if _already_seen(conn, aid):
            continue

        days = days_away(earnings_date)
        days_label = "tomorrow" if days <= 1 else f"in {days}d"

        title = (
            f"{ticker} earnings {days_label} — implied move ±{move_pct:.1f}% "
            f"(ATM straddle ${result['straddle']:.2f} @ ${result['price']:.2f})"
        )
        summary = (
            f"{ticker} reports earnings on {earnings_date}. "
            f"The options market implies a ±{move_pct:.1f}% move by "
            f"{result['expiry']} expiry. "
            f"Front-month ATM straddle (${result['atm_strike']:.0f} strike): "
            f"${result['straddle']:.2f} on a ${result['price']:.2f} stock. "
            f"High implied volatility = market expecting a large post-earnings reaction."
            if move_pct >= 10 else
            f"{ticker} reports earnings on {earnings_date}. "
            f"Options market implies ±{move_pct:.1f}% move by {result['expiry']} expiry. "
            f"ATM straddle ${result['straddle']:.2f} @ strike ${result['atm_strike']:.0f}."
        )

        now_utc = datetime.now(timezone.utc)
        article = {
            "title": title,
            "link": f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json",
            "summary": summary,
            "published": now_utc.isoformat(),
            "source": SOURCE_NAME,
            "_tickers": [ticker],
        }
        _mark_seen(conn, aid, ticker, title)
        articles.append(article)
        log.info(f"[earnings_iv_move] {ticker}: ±{move_pct:.1f}% expected move ({earnings_date})")

    conn.close()
    log.info(f"[earnings_iv_move] emitted {len(articles)} articles")
    return articles


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = collect_earnings_iv_move()
    if not results:
        print("No new expected-move signals (already seen or below threshold).")
    for a in results:
        print(f"\n[{a['source']}] {a['title']}")
        print(f"  Summary: {a['summary'][:200]}")
        print(f"  Tickers: {a['_tickers']}")
    print(f"\nTotal: {len(results)} articles")
