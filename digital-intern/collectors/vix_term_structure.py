"""VIX term structure collector.

Fetches VIX, VIX3M, VIX6M, VIX1Y, VVIX, and SKEW from Yahoo Finance and emits
a synthetic article when the structure changes meaningfully:
  - Backwardation (VIX > VIX3M): near-term fear spike, often a short-term bottom
  - Steep contango: complacent market, potential complacency risk
  - VVIX > 100: vol-of-vol spike, options market pricing extreme uncertainty
  - SKEW spike (> 145): tail-risk hedging accelerating

Dedup key: date + regime bucket so at most one article per regime per day.
A new article is also emitted intraday if VIX moves >= 1.5 points since last emit.
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
SOURCE = "vix_term_structure"

VIX_MOVE_THRESHOLD = 1.5
BACKWARDATION_THRESHOLD = 0.0
STEEP_CONTANGO_THRESHOLD = -5.0
VVIX_ALERT = 100.0
SKEW_ALERT = 145.0

TICKERS = {
    "vix": "^VIX",
    "vix3m": "^VIX3M",
    "vix6m": "^VIX6M",
    "vix1y": "^VIX1Y",
    "vvix": "^VVIX",
    "skew": "^SKEW",
}

log = logging.getLogger("vix_term_structure")


def _fetch_latest() -> dict[str, float] | None:
    """Return latest close values for all VIX-family tickers."""
    values: dict[str, float] = {}
    for key, symbol in TICKERS.items():
        try:
            h = yf.Ticker(symbol).history(period="2d")
            if not h.empty:
                values[key] = float(h["Close"].iloc[-1])
        except Exception as exc:  # noqa: BLE001
            log.warning("vix_term_structure: failed to fetch %s: %s", symbol, exc)
    # require at minimum VIX and VIX3M
    if "vix" not in values or "vix3m" not in values:
        return None
    return values


def _classify_regime(vals: dict[str, float]) -> tuple[str, str]:
    """Return (regime_key, human_label) from term structure."""
    spread = vals["vix"] - vals["vix3m"]
    vvix = vals.get("vvix", 0)
    skew = vals.get("skew", 0)

    if spread > BACKWARDATION_THRESHOLD:
        regime = "backwardation"
        label = "backwardation"
    elif spread < STEEP_CONTANGO_THRESHOLD:
        regime = "steep_contango"
        label = "steep contango"
    else:
        regime = "normal_contango"
        label = "normal contango"

    alerts = []
    if vvix >= VVIX_ALERT:
        alerts.append(f"VVIX={vvix:.0f}")
    if skew >= SKEW_ALERT:
        alerts.append(f"SKEW={skew:.0f}")

    return regime, label + (" [" + ", ".join(alerts) + "]" if alerts else "")


def _article_id(date_str: str, regime: str) -> str:
    return hashlib.sha256(f"{SOURCE}:{date_str}:{regime}".encode()).hexdigest()[:16]


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS vix_ts_state (
            key TEXT PRIMARY KEY,
            value REAL
        )"""
    )
    conn.commit()


def _get_state(conn: sqlite3.Connection, key: str) -> float | None:
    row = conn.execute(
        "SELECT value FROM vix_ts_state WHERE key=?", (key,)
    ).fetchone()
    return row[0] if row else None


def _set_state(conn: sqlite3.Connection, key: str, value: float) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO vix_ts_state(key,value) VALUES (?,?)", (key, value)
    )
    conn.commit()


def collect(conn: sqlite3.Connection | None = None) -> list[dict]:
    """Fetch VIX term structure and emit article rows if regime changed."""
    close_conn = conn is None
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)

    try:
        _ensure_schema(conn)
        vals = _fetch_latest()
        if not vals:
            log.warning("vix_term_structure: could not fetch data")
            return []

        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        regime, regime_label = _classify_regime(vals)

        vix = vals["vix"]
        vix3m = vals["vix3m"]
        spread = vix - vix3m
        vvix = vals.get("vvix")
        skew = vals.get("skew")

        # Check intraday move
        last_vix = _get_state(conn, "last_vix")
        big_move = last_vix is not None and abs(vix - last_vix) >= VIX_MOVE_THRESHOLD
        _set_state(conn, "last_vix", vix)

        # Determine if we should emit
        article_id = _article_id(date_str, regime)
        already_exists = conn.execute(
            "SELECT 1 FROM articles WHERE id=? LIMIT 1", (article_id,)
        ).fetchone()

        if already_exists and not big_move:
            log.debug("vix_term_structure: no new regime, no big move — skipping")
            return []

        if big_move and already_exists:
            # emit a move article instead
            move_dir = "spike" if vix > (last_vix or vix) else "drop"
            article_id = _article_id(
                now.strftime("%Y-%m-%d-%H"), f"move_{move_dir}"
            )
            title = (
                f"VIX intraday {move_dir}: {last_vix:.1f} → {vix:.1f} "
                f"(spread vs VIX3M: {spread:+.1f})"
            )
        else:
            spread_desc = "backwardation" if spread > 0 else f"contango ({abs(spread):.1f}pt)"
            title = (
                f"VIX term structure: {regime_label} | "
                f"VIX {vix:.1f} vs VIX3M {vix3m:.1f} ({spread_desc})"
            )

        vix6m = vals.get("vix6m", "N/A")
        vix1y = vals.get("vix1y", "N/A")
        body_lines = [
            f"VIX:   {vix:.2f}",
            f"VIX3M: {vix3m:.2f}  (spread: {spread:+.2f})",
            f"VIX6M: {vix6m:.2f}" if isinstance(vix6m, float) else "VIX6M: N/A",
            f"VIX1Y: {vix1y:.2f}" if isinstance(vix1y, float) else "VIX1Y: N/A",
            f"VVIX:  {vvix:.2f}" if vvix else "VVIX: N/A",
            f"SKEW:  {skew:.2f}" if skew else "SKEW: N/A",
            "",
            f"Regime: {regime_label}",
        ]
        if spread > 0:
            body_lines.append(
                "Backwardation signals near-term fear; historically precedes short-term market bottoms."
            )
        elif vvix and vvix >= VVIX_ALERT:
            body_lines.append(
                "VVIX elevated: options market pricing extreme uncertainty; gamma hedging flows likely."
            )

        url = f"internal://vix_term_structure/{date_str}/{regime}"
        ts = now.strftime("%Y-%m-%d %H:%M:%S")
        full_text = "\n".join(body_lines)
        kw = _urgency_score(vals, regime)

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
        log.info("vix_term_structure: emitted — %s", title)
        return [row]

    finally:
        if close_conn:
            conn.close()


def _urgency_score(vals: dict[str, float], regime: str) -> float:
    """Higher score for more extreme / actionable readings."""
    score = 3.0
    spread = vals["vix"] - vals["vix3m"]
    if regime == "backwardation":
        score += min(spread * 0.5, 3.0)  # more backwardation = higher urgency
    vvix = vals.get("vvix", 0)
    if vvix >= VVIX_ALERT:
        score += 2.0
    skew = vals.get("skew", 0)
    if skew >= SKEW_ALERT:
        score += 1.0
    return min(score, 9.5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    results = collect()
    if results:
        for r in results:
            print(f"\nTitle:  {r['title']}")
            print(f"Body:\n{r['body']}")
            print(f"Score:  {r['kw_score']}")
            print(f"ID:     {r['id']}")
    else:
        print("No new articles emitted (regime unchanged, no big move).")
        # Still print current state
        import sqlite3 as _sq
        conn2 = _sq.connect(str(DB_PATH), timeout=10)
        row = conn2.execute(
            "SELECT title, kw_score, first_seen FROM articles "
            "WHERE source=? ORDER BY first_seen DESC LIMIT 1",
            (SOURCE,)
        ).fetchone()
        if row:
            print(f"Last emitted: {row[0]} (score={row[1]}, at={row[2]})")
        conn2.close()
