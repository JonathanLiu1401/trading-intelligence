"""US Dollar Index (DXY) + major-pair FX snapshot collector.

Fetches DXY (ICE Dollar Index) plus the three highest-weight bilateral pairs
(EUR/USD, USD/JPY, USD/CNH) from Yahoo Finance and emits a synthetic article
when the dollar regime shifts:

  - Band change: DXY crossing the 95 / 100 / 105 / 110 psychological levels
  - Big move: |daily % change| >= ``DAILY_MOVE_THRESHOLD_PCT`` (0.5%)
    — DXY moves slowly, so 0.5% in a day is a real signal
  - Intraday spike: |DXY since last emit| >= ``INTRADAY_MOVE_THRESHOLD`` (0.30 pts)

DXY is currently absent from the collector set despite being a primary macro
driver: rate-cycle implication, EM stress signal, USD-funding-stress proxy
for the leveraged-ETF book, and the dominant variable behind multinational
earnings translation. Adding it gives the briefing layer a "USD broke 100" /
"DXY weakening — risk-on tailwind" line that previously had to come from
GDELT keyword scraping.

Mirrors ``collectors/vix_term_structure.py`` end-to-end (direct articles.db
insert, ``vix_ts_state``-style state table for the intraday tripwire, dedup
on ``(SOURCE, date, regime_band)``).
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
SOURCE = "dxy"

# DXY moves slowly: 0.5% intraday is roughly one trading day's typical range,
# so a daily move of >= 0.5% is the threshold for "actually noteworthy".
DAILY_MOVE_THRESHOLD_PCT = 0.5
# Absolute-point delta since the last emit. 0.30 pts ~= 0.3% — catches an
# intraday spike between scheduled emits even if the daily % bar isn't hit.
INTRADAY_MOVE_THRESHOLD = 0.30

# Psychological levels. Crossing one of these flips the regime band, which
# is the dedup key, so the first cross of the day always re-emits.
BAND_BOUNDARIES = (95.0, 100.0, 105.0, 110.0)

# yfinance symbols. ``DX-Y.NYB`` is the ICE Dollar Index (verified empirically
# 2026-05-19; ``DX=F`` and ``^DXY`` both return EMPTY from yfinance). The three
# bilateral pairs are context only — DXY is the only required series.
TICKERS = {
    "dxy": "DX-Y.NYB",
    "eurusd": "EURUSD=X",
    "usdjpy": "JPY=X",
    "usdcnh": "CNH=X",
}

log = logging.getLogger("dxy_collector")


def _fetch_latest() -> dict[str, tuple[float, float | None]] | None:
    """Return {key: (latest_close, prev_close_or_None)} for each pair.

    DXY is required (returns None if missing); the bilateral pairs are
    best-effort context (omitted from output if their fetch fails).
    """
    values: dict[str, tuple[float, float | None]] = {}
    for key, symbol in TICKERS.items():
        try:
            h = yf.Ticker(symbol).history(period="5d")
            if h.empty:
                log.warning("dxy_collector: empty history for %s", symbol)
                continue
            closes = h["Close"].dropna().tolist()
            if not closes:
                continue
            latest = float(closes[-1])
            prev = float(closes[-2]) if len(closes) >= 2 else None
            values[key] = (latest, prev)
        except Exception as exc:  # noqa: BLE001
            log.warning("dxy_collector: failed to fetch %s: %s", symbol, exc)
    if "dxy" not in values:
        return None
    return values


def _classify_band(dxy: float) -> str:
    """Return a stable band key for the current DXY level.

    The boundaries themselves become the dedup boundaries: crossing into a
    new band always emits, staying in the same band does not.
    """
    if dxy < BAND_BOUNDARIES[0]:
        return "weak_lt95"
    if dxy < BAND_BOUNDARIES[1]:
        return "soft_95_100"
    if dxy < BAND_BOUNDARIES[2]:
        return "strong_100_105"
    if dxy < BAND_BOUNDARIES[3]:
        return "very_strong_105_110"
    return "extreme_gte110"


def _band_label(band: str) -> str:
    return {
        "weak_lt95": "weak (DXY<95)",
        "soft_95_100": "soft (DXY 95-100)",
        "strong_100_105": "strong (DXY 100-105)",
        "very_strong_105_110": "very strong (DXY 105-110)",
        "extreme_gte110": "extreme (DXY>=110)",
    }.get(band, band)


def _article_id(date_str: str, suffix: str) -> str:
    return hashlib.sha256(f"{SOURCE}:{date_str}:{suffix}".encode()).hexdigest()[:16]


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS dxy_state (
            key TEXT PRIMARY KEY,
            value REAL
        )"""
    )
    conn.commit()


def _get_state(conn: sqlite3.Connection, key: str) -> float | None:
    row = conn.execute(
        "SELECT value FROM dxy_state WHERE key=?", (key,)
    ).fetchone()
    return row[0] if row else None


def _set_state(conn: sqlite3.Connection, key: str, value: float) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO dxy_state(key,value) VALUES (?,?)", (key, value)
    )
    conn.commit()


def _urgency_score(
    dxy: float, daily_pct: float | None, band: str, big_move: bool
) -> float:
    """0..9.5. Higher for extreme bands or large moves."""
    score = 2.5
    if band in ("extreme_gte110", "weak_lt95"):
        score += 2.5
    elif band in ("very_strong_105_110",):
        score += 1.5
    if daily_pct is not None and abs(daily_pct) >= DAILY_MOVE_THRESHOLD_PCT:
        # Each additional 0.5% adds 1 point, capped.
        score += min(abs(daily_pct) / 0.5, 3.0)
    if big_move:
        score += 0.5
    return min(score, 9.5)


def collect(conn: sqlite3.Connection | None = None) -> list[dict]:
    """Fetch DXY + bilateral pairs and emit article rows on regime shifts."""
    close_conn = conn is None
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)

    try:
        _ensure_schema(conn)
        vals = _fetch_latest()
        if not vals:
            log.warning("dxy_collector: could not fetch DXY")
            return []

        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")

        dxy, dxy_prev = vals["dxy"]
        daily_pct: float | None = None
        if dxy_prev and dxy_prev != 0:
            daily_pct = (dxy - dxy_prev) / dxy_prev * 100.0

        band = _classify_band(dxy)

        # Intraday tripwire — separate from the daily-bar move so an emit can
        # fire between scheduled passes if DXY shifts fast.
        last_dxy = _get_state(conn, "last_dxy")
        big_move = (
            last_dxy is not None and abs(dxy - last_dxy) >= INTRADAY_MOVE_THRESHOLD
        )
        _set_state(conn, "last_dxy", dxy)

        band_article_id = _article_id(date_str, f"band_{band}")
        already_band = conn.execute(
            "SELECT 1 FROM articles WHERE id=? LIMIT 1", (band_article_id,)
        ).fetchone()
        big_daily = (
            daily_pct is not None and abs(daily_pct) >= DAILY_MOVE_THRESHOLD_PCT
        )

        # Decide whether to emit and which article variant
        if not already_band:
            # First time in this band today: emit a regime-band article
            article_id = band_article_id
            title = (
                f"USD regime: {_band_label(band)} | DXY {dxy:.2f}"
                + (f" ({daily_pct:+.2f}% d/d)" if daily_pct is not None else "")
            )
            variant = "band"
        elif big_move:
            # Intraday tripwire fired — emit a move article keyed by the hour
            move_dir = "up" if dxy > (last_dxy or dxy) else "down"
            article_id = _article_id(
                now.strftime("%Y-%m-%d-%H"), f"move_{move_dir}"
            )
            already_move = conn.execute(
                "SELECT 1 FROM articles WHERE id=? LIMIT 1", (article_id,)
            ).fetchone()
            if already_move:
                log.debug(
                    "dxy_collector: intraday %s already emitted this hour", move_dir
                )
                return []
            title = (
                f"DXY intraday {move_dir}: {last_dxy:.2f} -> {dxy:.2f} "
                f"({_band_label(band)})"
            )
            variant = "intraday_move"
        elif big_daily and daily_pct is not None:
            # Same band as a prior emit today, but the daily bar is large —
            # emit a daily-move article keyed by the date so it fires once.
            move_dir = "up" if daily_pct > 0 else "down"
            article_id = _article_id(date_str, f"daily_{move_dir}")
            already_daily = conn.execute(
                "SELECT 1 FROM articles WHERE id=? LIMIT 1", (article_id,)
            ).fetchone()
            if already_daily:
                log.debug("dxy_collector: daily %s already emitted today", move_dir)
                return []
            title = (
                f"DXY {move_dir} {daily_pct:+.2f}% d/d to {dxy:.2f} "
                f"({_band_label(band)})"
            )
            variant = "daily_move"
        else:
            log.debug(
                "dxy_collector: band %s unchanged, no big move — skipping", band
            )
            return []

        body_lines = [
            f"DXY:    {dxy:.2f}"
            + (f"  ({daily_pct:+.2f}% d/d)" if daily_pct is not None else ""),
        ]
        for key, label in (
            ("eurusd", "EUR/USD"),
            ("usdjpy", "USD/JPY"),
            ("usdcnh", "USD/CNH"),
        ):
            if key in vals:
                latest, prev = vals[key]
                if prev:
                    pct = (latest - prev) / prev * 100.0
                    body_lines.append(f"{label}: {latest:.4f}  ({pct:+.2f}% d/d)")
                else:
                    body_lines.append(f"{label}: {latest:.4f}")
        body_lines.append("")
        body_lines.append(f"Regime: {_band_label(band)}")
        if band == "extreme_gte110":
            body_lines.append(
                "Extreme USD strength — historic EM stress trigger; multinational"
                " earnings translation drag; commodity / risk-asset headwind."
            )
        elif band == "weak_lt95":
            body_lines.append(
                "USD weakening — typically a tailwind for risk assets,"
                " commodities, and EM equities; loosening financial conditions."
            )

        url = f"internal://dxy/{date_str}/{variant}"
        ts = now.strftime("%Y-%m-%d %H:%M:%S")
        full_text = "\n".join(body_lines)
        kw = _urgency_score(dxy, daily_pct, band, big_move)

        row = {
            "id": article_id,
            "url": url,
            "title": title,
            "full_text": full_text,
            "source": SOURCE,
            "published": ts,
            "first_seen": ts,
            "kw_score": kw,
            "urgency": 1 if kw >= 6.0 else 0,
        }

        compressed = zlib.compress(full_text.encode("utf-8"))
        conn.execute(
            """INSERT OR IGNORE INTO articles
               (id, url, title, source, published, kw_score, urgency, full_text, first_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (article_id, url, title, SOURCE, ts, kw, row["urgency"], compressed, ts),
        )
        conn.commit()
        log.info("dxy_collector: emitted — %s", title)
        return [row]

    finally:
        if close_conn:
            conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    results = collect()
    if results:
        for r in results:
            print(f"\nTitle:  {r['title']}")
            print(f"Body:\n{r['full_text']}")
            print(f"Score:  {r['kw_score']}")
            print(f"ID:     {r['id']}")
    else:
        print("No new articles emitted (band unchanged, no big move).")
        import sqlite3 as _sq
        conn2 = _sq.connect(str(DB_PATH), timeout=10)
        row = conn2.execute(
            "SELECT title, kw_score, first_seen FROM articles "
            "WHERE source=? ORDER BY first_seen DESC LIMIT 1",
            (SOURCE,),
        ).fetchone()
        if row:
            print(f"Last emitted: {row[0]} (score={row[1]}, at={row[2]})")
        conn2.close()
