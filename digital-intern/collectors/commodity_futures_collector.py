"""Commodity Futures Price Monitor.

Tracks major commodity futures via yfinance and emits synthetic articles
when a contract posts a significant daily move. Covers the macro-relevant
commodity complex that no other collector tracks:

  Energy:       CL=F (WTI Crude), BZ=F (Brent Crude), NG=F (Natural Gas)
  Metals:       GC=F (Gold), SI=F (Silver), HG=F (Copper)
  Agriculture:  ZW=F (Wheat), ZC=F (Corn), ZS=F (Soybeans)

Each commodity has an asset-class-specific move threshold. A move article is
emitted at most once per (symbol, date, direction) so rapid re-polls don't
spam the feed.

Follows dxy_collector.py end-to-end: direct articles.db insert, state table
for tracking last-emit price, INSERT OR IGNORE dedup.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "articles.db"
SOURCE = "commodity_futures"

log = logging.getLogger("commodity_futures")

# (yfinance symbol, human name, unit, move threshold %, urgency base score)
COMMODITIES = [
    ("CL=F", "WTI Crude Oil", "$/bbl", 2.0, 5.0),
    ("BZ=F", "Brent Crude Oil", "$/bbl", 2.0, 5.0),
    ("NG=F", "Natural Gas", "$/MMBtu", 3.0, 4.5),
    ("GC=F", "Gold", "$/oz", 1.0, 4.5),
    ("SI=F", "Silver", "$/oz", 2.0, 4.0),
    ("HG=F", "Copper", "$/lb", 2.0, 5.0),
    ("ZW=F", "Wheat", "¢/bu", 2.5, 4.5),
    ("ZC=F", "Corn", "¢/bu", 2.5, 4.0),
    ("ZS=F", "Soybeans", "¢/bu", 2.0, 4.5),
]


def _article_id(symbol: str, date_str: str, direction: str) -> str:
    key = f"{SOURCE}:{symbol}:{date_str}:{direction}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS commodity_state (
            symbol TEXT PRIMARY KEY,
            last_price REAL,
            last_emit_date TEXT
        )"""
    )
    conn.commit()


def _get_state(conn: sqlite3.Connection, symbol: str) -> tuple[float | None, str | None]:
    row = conn.execute(
        "SELECT last_price, last_emit_date FROM commodity_state WHERE symbol=?",
        (symbol,),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def _set_state(conn: sqlite3.Connection, symbol: str, price: float, date_str: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO commodity_state(symbol, last_price, last_emit_date) VALUES (?,?,?)",
        (symbol, price, date_str),
    )
    conn.commit()


def _fetch_commodity(symbol: str) -> tuple[float, float | None] | None:
    """Return (latest_close, prev_close) or None on failure."""
    try:
        h = yf.Ticker(symbol).history(period="5d")
        if h.empty:
            return None
        closes = h["Close"].dropna().tolist()
        if not closes:
            return None
        latest = float(closes[-1])
        prev = float(closes[-2]) if len(closes) >= 2 else None
        return (latest, prev)
    except Exception as exc:
        log.warning("commodity_futures: failed to fetch %s: %s", symbol, exc)
        return None


def _kw_score(base: float, pct: float, threshold: float) -> float:
    """Scale score with move size: base + extra per threshold multiple."""
    extra = min(abs(pct) / threshold, 3.0)
    return min(base + extra, 9.5)


def collect(conn: sqlite3.Connection | None = None) -> list[dict]:
    """Fetch all commodity futures and emit articles on significant moves."""
    close_conn = conn is None
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)

    try:
        _ensure_schema(conn)
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        ts = now.strftime("%Y-%m-%d %H:%M:%S")
        articles: list[dict] = []

        for symbol, name, unit, threshold, base_score in COMMODITIES:
            result = _fetch_commodity(symbol)
            if result is None:
                continue
            latest, prev = result
            if prev is None or prev == 0:
                continue

            pct = (latest - prev) / prev * 100.0
            if abs(pct) < threshold:
                continue

            direction = "up" if pct > 0 else "down"
            article_id = _article_id(symbol, date_str, direction)

            already = conn.execute(
                "SELECT 1 FROM articles WHERE id=? LIMIT 1", (article_id,)
            ).fetchone()
            if already:
                log.debug("commodity_futures: %s %s already emitted today", symbol, direction)
                continue

            score = _kw_score(base_score, pct, threshold)
            arrow = "▲" if pct > 0 else "▼"
            title = (
                f"{name} {arrow}{abs(pct):.1f}% to {latest:.2f} {unit} "
                f"— commodity futures alert"
            )
            body_lines = [
                f"{name} ({symbol})",
                f"  Latest close: {latest:.2f} {unit}",
                f"  Prior close:  {prev:.2f} {unit}",
                f"  Daily change: {pct:+.2f}%",
                "",
                f"Move exceeds {threshold:.1f}% threshold — significant commodity price shift.",
            ]
            context = {
                "CL=F": "WTI crude is the US benchmark for oil pricing; large moves affect energy stocks, inflation expectations, and consumer spending.",
                "BZ=F": "Brent crude is the global benchmark; divergence vs WTI signals refinery/transport dislocations.",
                "NG=F": "Natural gas drives utility costs and winter heating demand; moves affect XLE and utility ETFs.",
                "GC=F": "Gold is the primary safe-haven asset; large up-moves signal risk-off positioning or inflation fears.",
                "SI=F": "Silver has dual industrial/monetary demand; often leads gold on breakouts.",
                "HG=F": "Copper is the leading macro demand indicator ('Dr. Copper'); large moves signal global growth shifts.",
                "ZW=F": "Wheat affects food inflation globally; supply shocks propagate to EM food-import economies.",
                "ZC=F": "Corn is the broadest US agricultural benchmark; ethanol demand links it to energy prices.",
                "ZS=F": "Soybeans are a key US export; China demand and weather drive the largest moves.",
            }
            if symbol in context:
                body_lines.extend(["", context[symbol]])

            full_text = "\n".join(body_lines)
            url = f"internal://commodity_futures/{symbol}/{date_str}/{direction}"
            compressed = zlib.compress(full_text.encode("utf-8"))

            conn.execute(
                """INSERT OR IGNORE INTO articles
                   (id, url, title, source, published, kw_score, urgency, full_text, first_seen)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    article_id, url, title, SOURCE, ts,
                    score, 1 if score >= 6.0 else 0,
                    compressed, ts,
                ),
            )
            conn.commit()
            log.info("commodity_futures: emitted — %s", title)
            _set_state(conn, symbol, latest, date_str)
            articles.append({"title": title, "score": score, "pct": pct, "symbol": symbol, "price": latest})

        return articles

    finally:
        if close_conn:
            conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = collect()
    if results:
        print(f"\nEmitted {len(results)} commodity alert(s):")
        for r in results:
            print(f"  {r['symbol']:6s} {r['pct']:+.2f}%  ${r['price']:.2f}  → {r['title']}")
    else:
        print("No significant commodity moves today (all within thresholds).")
        # Show current prices for reference
        print("\nCurrent prices:")
        for symbol, name, unit, threshold, _ in COMMODITIES:
            result = _fetch_commodity(symbol)
            if result:
                latest, prev = result
                pct = (latest - prev) / prev * 100.0 if prev and prev != 0 else 0
                print(f"  {name:20s} {latest:>10.2f} {unit:<8s} ({pct:+.2f}%)")
