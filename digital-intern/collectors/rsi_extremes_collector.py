"""RSI Extremes Screener — fires when portfolio/watchlist tickers hit RSI < 30 or > 70.

Computes 14-period RSI (Wilder's EMA method) using 30 days of daily bars from
yfinance. Emits one article per ticker per RSI level-cross (oversold ↔ neutral ↔
overbought) so the signal fires on entry, not on every pass.

Dedup key: symbol|rsi_bucket|date — where rsi_bucket is:
  "oversold"   RSI < 30
  "overbought" RSI > 70
  "neutral"    RSI 30-70 (tracked only to reset the crossing so a ticker can fire again)

Source tags: "technical/rsi_oversold", "technical/rsi_overbought"
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, date, timezone
from pathlib import Path

import yfinance as yf

try:
    from core.logger import get_logger
    _log = get_logger("rsi_extremes")
except Exception:
    _log = logging.getLogger("rsi_extremes")

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
STATE_PATH = BASE_DIR / "data" / "rsi_extremes_state.json"

RSI_PERIOD = 14
OVERSOLD_THRESHOLD = 30.0
OVERBOUGHT_THRESHOLD = 70.0
HISTORY_PERIOD = "30d"


def _load_tickers() -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        u = (t or "").strip().upper()
        if u and u not in seen:
            seen.add(u)
            tickers.append(u)

    try:
        with open(PORTFOLIO_PATH) as f:
            pf = json.load(f)
        for pos in pf.get("positions", []):
            _add(pos.get("ticker", ""))
        for opt in pf.get("options", []):
            _add(opt.get("underlying", ""))
        for t in pf.get("sector_watchlist", []):
            _add(t)
    except Exception:
        pass

    try:
        with open(WATCHLIST_PATH) as f:
            wl = json.load(f)
        for key in ("memory_core", "semis_equipment", "broader_semis", "portfolio",
                    "korean", "japanese", "etfs"):
            for t in wl.get(key, []):
                _add(t)
    except Exception:
        pass

    return tickers


def _calc_rsi(closes) -> float:
    """14-period RSI using Wilder's EMA (com=period-1)."""
    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = gain / loss
    rsi_series = 100 - 100 / (1 + rs)
    return float(rsi_series.iloc[-1])


def _rsi_bucket(rsi: float) -> str:
    if rsi < OVERSOLD_THRESHOLD:
        return "oversold"
    if rsi > OVERBOUGHT_THRESHOLD:
        return "overbought"
    return "neutral"


def _load_state() -> dict:
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


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


def _article_id(symbol: str, bucket: str, today: str) -> str:
    return hashlib.sha256(f"{symbol}|{bucket}|{today}".encode()).hexdigest()[:16]


def collect_rsi_extremes() -> list[dict]:
    tickers = _load_tickers()
    if not tickers:
        _log.warning("[rsi_extremes] no tickers loaded")
        return []

    state = _load_state()
    conn = _ensure_db()
    today = date.today().isoformat()
    articles: list[dict] = []

    for symbol in tickers:
        try:
            hist = yf.Ticker(symbol).history(period=HISTORY_PERIOD, interval="1d")
            if len(hist) < RSI_PERIOD + 2:
                continue
            rsi = _calc_rsi(hist["Close"])
            if not (0 < rsi < 100):
                continue

            bucket = _rsi_bucket(rsi)
            prev_bucket = state.get(symbol, {}).get("bucket", "neutral")
            state.setdefault(symbol, {})["bucket"] = bucket
            state[symbol]["rsi"] = round(rsi, 1)
            state[symbol]["updated"] = today

            if bucket == "neutral":
                continue
            if bucket == prev_bucket:
                continue

            article_id = _article_id(symbol, bucket, today)
            already = conn.execute(
                "SELECT 1 FROM seen_articles WHERE id=?", (article_id,)
            ).fetchone()
            if already:
                continue

            source = f"technical/rsi_{bucket}"
            label = "OVERSOLD" if bucket == "oversold" else "OVERBOUGHT"
            title = f"[RSI/{label}] {symbol} RSI={rsi:.1f} — crossed {bucket} threshold"
            link = f"https://finance.yahoo.com/chart/{symbol}"
            summary = (
                f"{symbol} daily RSI hit {rsi:.1f} ({label}). "
                f"Threshold: oversold <{OVERSOLD_THRESHOLD}, overbought >{OVERBOUGHT_THRESHOLD}. "
                f"Previous zone: {prev_bucket}."
            )
            published = datetime.now(timezone.utc).isoformat()

            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
                (article_id, link, title, source, published),
            )
            conn.commit()

            articles.append({
                "title": title,
                "link": link,
                "summary": summary,
                "published": published,
                "source": source,
                "_tickers": [symbol],
            })
            _log.info(f"[rsi_extremes] {title}")

        except Exception as e:
            _log.debug(f"[rsi_extremes] {symbol}: {e}")

    conn.close()
    _save_state(state)
    _log.info(f"[rsi_extremes] cycle done — {len(articles)} new signals")
    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = collect_rsi_extremes()
    print(f"\nRSI extreme signals found: {len(results)}")
    for a in results:
        print(f"  {a['title']}")
    if not results:
        # Show current RSI values anyway
        state = _load_state()
        print("\nCurrent RSI state (sample):")
        for sym, info in list(state.items())[:15]:
            print(f"  {sym}: RSI={info.get('rsi','?')} ({info.get('bucket','?')})")
