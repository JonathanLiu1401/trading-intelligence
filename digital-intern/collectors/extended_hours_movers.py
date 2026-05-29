"""Extended-hours (pre-market + after-hours) movers collector.

Detects stocks with significant pre-market or after-hours price moves by
pulling 1-minute OHLCV data with prepost=True from yfinance and comparing
the extended-hours close to the prior regular-session close.

Universe: portfolio + watchlist tickers, plus a broad large-cap supplement.

Emits one article row per ticker per session once the move crosses the
threshold. Dedup key: symbol|session_date|direction so re-checks don't
re-fire for the same move, but a reversal (pre-mkt gap-up then gap-down)
re-arms.

Source tags: "extended_hours/pre_market" or "extended_hours/after_hours"
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import NamedTuple

import yfinance as yf
import pandas as pd

try:
    from core.logger import get_logger
    _log = get_logger("extended_hours_movers")
except Exception:
    _log = logging.getLogger("extended_hours_movers")

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

# Large-cap supplement to catch broad market catalysts beyond the personal universe
SUPPLEMENT_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "TSLA", "AVGO",
    "ORCL", "AMD", "INTC", "QCOM", "MU", "ASML", "TSM",
    "JPM", "GS", "MS", "BAC", "WFC",
    "XOM", "CVX", "OXY",
    "UNH", "JNJ", "PFE", "LLY",
    "BRK-B", "V", "MA",
]

# Threshold: minimum abs(change%) to emit an article
MIN_MOVE_PCT = 3.0
# Batch size for yfinance download to avoid hammering API
BATCH_SIZE = 30
# Maximum tickers to scan per run (keep runtime under ~45s)
MAX_TICKERS = 120

SOURCE_PRE = "extended_hours/pre_market"
SOURCE_POST = "extended_hours/after_hours"


def _load_tickers() -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()

    def add(t: str) -> None:
        s = (t or "").strip().upper()
        if s and s not in seen:
            seen.add(s)
            tickers.append(s)

    try:
        with open(WATCHLIST_PATH) as f:
            wl = json.load(f)
        for key in ("memory_core", "semis_equipment", "broader_semis", "memory_options_focus",
                    "korean", "japanese", "portfolio", "indices"):
            for t in wl.get(key, []) or []:
                add(t)
    except Exception as e:
        _log.warning("watchlist load error: %s", e)

    try:
        with open(PORTFOLIO_PATH) as f:
            pf = json.load(f)
        positions = pf.get("positions", [])
        if isinstance(positions, dict):
            for t in positions.keys():
                add(t)
        else:
            for pos in positions:
                if isinstance(pos, dict):
                    add(pos.get("ticker", ""))
                else:
                    add(str(pos))
        for t in pf.get("sector_watchlist", []) or []:
            add(t)
    except Exception as e:
        _log.warning("portfolio load error: %s", e)

    for t in SUPPLEMENT_TICKERS:
        add(t)

    return tickers[:MAX_TICKERS]


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


class MoverResult(NamedTuple):
    symbol: str
    change_pct: float
    ext_price: float
    reg_close: float
    session: str  # "pre_market" or "after_hours"
    as_of: str


def _detect_movers(tickers: list[str]) -> list[MoverResult]:
    """Download 1-day 1-minute data with prepost=True for the given batch.

    Returns tickers whose latest extended-hours price deviates >= MIN_MOVE_PCT
    from the most recent regular-session close.
    """
    results: list[MoverResult] = []
    now_utc = datetime.now(timezone.utc)
    today = now_utc.date()
    hour_utc = now_utc.hour

    # Decide which session we're in (UTC times):
    #   Pre-market:  08:00–13:30 UTC (4:00–9:30 AM ET)
    #   After-hours: 20:00–00:00 UTC (4:00–8:00 PM ET)
    #   Regular:     13:30–20:00 UTC — we still check for post data left from yesterday

    if 8 <= hour_utc < 14:
        target_session = "pre_market"
        source_tag = SOURCE_PRE
    elif hour_utc >= 20 or hour_utc < 2:
        target_session = "after_hours"
        source_tag = SOURCE_POST
    else:
        # Regular hours — check for after-hours data from last close
        target_session = "after_hours"
        source_tag = SOURCE_POST

    try:
        df = yf.download(
            tickers,
            period="2d",
            interval="1m",
            prepost=True,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        _log.error("yfinance download failed: %s", e)
        return results

    if df.empty:
        return results

    # Handle single-ticker vs multi-ticker column structure
    single = len(tickers) == 1

    for sym in tickers:
        try:
            if single:
                sym_df = df.copy()
            else:
                if sym not in df.columns.get_level_values(0):
                    continue
                sym_df = df[sym].copy()

            sym_df = sym_df.dropna(subset=["Close"])
            if sym_df.empty:
                continue

            # Index is tz-aware; convert to UTC
            sym_df.index = sym_df.index.tz_convert("UTC")

            # Regular session: 13:30–20:00 UTC (M–F)
            reg_mask = (sym_df.index.hour * 60 + sym_df.index.minute >= 13 * 60 + 30) & \
                       (sym_df.index.hour * 60 + sym_df.index.minute < 20 * 60)
            ext_mask = ~reg_mask

            reg_rows = sym_df[reg_mask]
            ext_rows = sym_df[ext_mask]

            if reg_rows.empty or ext_rows.empty:
                continue

            reg_close = float(reg_rows["Close"].iloc[-1])
            # Latest extended-hours price
            ext_latest = ext_rows.iloc[-1]
            ext_price = float(ext_latest["Close"])
            ext_time = ext_latest.name  # Timestamp

            if reg_close <= 0:
                continue

            change_pct = (ext_price - reg_close) / reg_close * 100.0

            if abs(change_pct) >= MIN_MOVE_PCT:
                results.append(MoverResult(
                    symbol=sym,
                    change_pct=round(change_pct, 2),
                    ext_price=round(ext_price, 4),
                    reg_close=round(reg_close, 4),
                    session=target_session,
                    as_of=ext_time.strftime("%Y-%m-%d %H:%M UTC"),
                ))

        except Exception as e:
            _log.debug("error processing %s: %s", sym, e)
            continue

    return results


def collect() -> list[dict]:
    """Main entry point — called by the daemon worker loop."""
    tickers = _load_tickers()
    if not tickers:
        _log.warning("no tickers loaded")
        return []

    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    _ensure_db(conn)

    today_str = date.today().isoformat()
    all_results: list[MoverResult] = []

    # Process in batches
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        batch_results = _detect_movers(batch)
        all_results.extend(batch_results)

    articles: list[dict] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for mover in all_results:
        direction = "UP" if mover.change_pct > 0 else "DOWN"
        source = SOURCE_PRE if mover.session == "pre_market" else SOURCE_POST
        session_label = "Pre-Market" if mover.session == "pre_market" else "After-Hours"

        dedup_key = f"{mover.symbol}|{today_str}|{direction}|{mover.session}"
        art_id = _art_id(dedup_key)

        try:
            exists = conn.execute(
                "SELECT 1 FROM seen_articles WHERE id=?", (art_id,)
            ).fetchone()
        except Exception:
            exists = None

        if exists:
            continue

        sign = "+" if mover.change_pct > 0 else ""
        title = (
            f"[{session_label}] {mover.symbol} {direction} {sign}{mover.change_pct}% "
            f"@ ${mover.ext_price} (reg close ${mover.reg_close}) — {mover.as_of}"
        )
        link = f"https://finance.yahoo.com/quote/{mover.symbol}"

        article = {
            "id": art_id,
            "title": title,
            "link": link,
            "source": source,
            "first_seen": now_iso,
            "_tickers": [mover.symbol],
            "_meta": {
                "symbol": mover.symbol,
                "change_pct": mover.change_pct,
                "ext_price": mover.ext_price,
                "reg_close": mover.reg_close,
                "session": mover.session,
                "as_of": mover.as_of,
            },
        }

        try:
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles(id,link,title,source,first_seen) VALUES(?,?,?,?,?)",
                (art_id, link, title, source, now_iso),
            )
            conn.commit()
            articles.append(article)
        except Exception as e:
            _log.warning("db insert error for %s: %s", mover.symbol, e)

    conn.close()
    _log.info("extended_hours_movers: %d movers found from %d tickers", len(articles), len(tickers))
    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = collect()
    print(f"\n{'='*60}")
    print(f"Extended Hours Movers: {len(results)} found")
    print(f"{'='*60}")
    for a in results:
        print(f"  {a['title']}")
    if not results:
        print("  (no significant movers above threshold right now)")
