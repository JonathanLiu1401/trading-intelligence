"""Fed Liquidity Monitor — tracks system liquidity via FRED.

Four key metrics that together describe how much cash is sloshing through
the financial system. All sourced from FRED public CSV (no API key required).

  WALCL    — Fed total assets / balance sheet ($M → reported as $B)
             Rising = QE/liquidity injection; falling = QT/tightening
  RRPONTSYD — Overnight Reverse Repo usage ($B)
             High = excess liquidity parked at Fed; near-zero = drained
  WTREGEN   — Treasury General Account ($M → $B)
             High TGA = Treasury hoard; spending it = liquidity injection
  WRESBAL   — Bank reserve balances at Fed ($M → $B)
             Proxy for banking system excess liquidity

Signals emitted when meaningful week-over-week changes are detected:
  - RRP crosses 50B, 100B, 500B, 1T thresholds (regime shifts)
  - WALCL changes ≥ 1% WoW (balance sheet acceleration)
  - WTREGEN drops > $100B in a week (Treasury spending spree)
  - WRESBAL drops < $2T (historical warning level for repo stress)
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from core.logger import get_logger
    log = get_logger("fed_liquidity")
except Exception:
    log = logging.getLogger("fed_liquidity")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "fed_liquidity_seen.db"

FREDGRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FETCH_TIMEOUT = 20
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

SOURCE_NAME = "fed_liquidity"

# Series definitions: (label, unit_divisor, unit_label)
# WALCL and WRESBAL and WTREGEN are in $M; divide by 1000 to get $B.
SERIES: dict[str, tuple[str, float, str]] = {
    "WALCL":      ("Fed Balance Sheet Total Assets", 1_000.0, "B"),
    "RRPONTSYD":  ("Overnight Reverse Repo (RRP) Usage", 1.0, "B"),
    "WTREGEN":    ("Treasury General Account (TGA)", 1_000.0, "B"),
    "WRESBAL":    ("Bank Reserves at Fed", 1_000.0, "B"),
}

# Fetch last N observations for WoW delta calculation
FETCH_N = 10


def _ensure_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS seen (key TEXT PRIMARY KEY, ts TEXT)"
    )
    conn.commit()
    return conn


def _seen(conn: sqlite3.Connection, key: str) -> bool:
    return conn.execute("SELECT 1 FROM seen WHERE key=?", (key,)).fetchone() is not None


def _mark_seen(conn: sqlite3.Connection, key: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen(key, ts) VALUES(?,?)",
        (key, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _article_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _fetch(series: str) -> list[tuple[str, float]]:
    url = FREDGRAPH_CSV.format(series=series)
    resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA})
    resp.raise_for_status()
    rows: list[tuple[str, float]] = []
    for line in resp.text.splitlines()[1:]:
        parts = line.strip().split(",")
        if len(parts) != 2:
            continue
        date_str, val_str = parts[0].strip(), parts[1].strip()
        if val_str in (".", ""):
            continue
        try:
            rows.append((date_str, float(val_str)))
        except ValueError:
            continue
    return rows[-FETCH_N:] if len(rows) >= FETCH_N else rows


def _signal(series: str, label: str, divisor: float, unit: str,
            obs_date: str, val: float, prev_val: float | None) -> str | None:
    """Return a signal string if the observation warrants an alert, else None."""
    val_b = val / divisor
    prev_b = prev_val / divisor if prev_val is not None else None

    if series == "RRPONTSYD":
        # Threshold crossings (significant regime changes)
        thresholds = [1_000, 500, 100, 50, 10]
        if prev_b is not None:
            for t in thresholds:
                if (prev_b > t) != (val_b > t):
                    direction = "above" if val_b > t else "below"
                    return f"RRP crossed {direction} ${t}B: now ${val_b:.1f}B (prev ${prev_b:.1f}B)"
        # Also report weekly change if > 10%
        if prev_b and prev_b > 0:
            pct = (val_b - prev_b) / prev_b * 100
            if abs(pct) >= 20:
                return f"RRP {'+' if pct>0 else ''}{pct:.1f}% WoW: ${val_b:.1f}B"

    elif series == "WALCL":
        if prev_b and prev_b > 0:
            pct = (val_b - prev_b) / prev_b * 100
            if abs(pct) >= 1.0:
                direction = "expanded" if pct > 0 else "contracted"
                return f"Fed balance sheet {direction} {abs(pct):.2f}% WoW: ${val_b/1000:.2f}T"

    elif series == "WTREGEN":
        if prev_b is not None:
            delta = val_b - prev_b
            if delta < -100:
                return f"TGA dropped ${abs(delta):.0f}B WoW: ${val_b:.0f}B → liquidity injection"
            if delta > 150:
                return f"TGA rose ${delta:.0f}B WoW: ${val_b:.0f}B → liquidity drain"

    elif series == "WRESBAL":
        if val_b < 2_000:
            return f"Bank reserves fell below $2T: ${val_b:.0f}B — watch for repo stress"

    return None


def collect_fed_liquidity() -> list[dict]:
    """Fetch Fed liquidity series from FRED and return new article dicts."""
    conn = _ensure_db()
    articles: list[dict] = []
    now_utc = datetime.now(timezone.utc)

    # Build a combined snapshot article each week (keyed by most recent obs date)
    snapshot_rows: list[str] = []
    snapshot_date: str | None = None

    for series, (label, divisor, unit) in SERIES.items():
        try:
            rows = _fetch(series)
        except Exception as exc:
            log.warning("fed_liquidity: fetch failed %s: %s", series, exc)
            continue

        if not rows:
            continue

        obs_date, val = rows[-1]
        prev_val = rows[-2][1] if len(rows) >= 2 else None
        val_b = val / divisor

        if snapshot_date is None or obs_date > snapshot_date:
            snapshot_date = obs_date

        # Format for snapshot
        if val_b >= 1_000:
            snapshot_rows.append(f"{label}: ${val_b/1000:.2f}T ({obs_date})")
        else:
            snapshot_rows.append(f"{label}: ${val_b:.1f}B ({obs_date})")

        # Check for signal-worthy moves
        sig = _signal(series, label, divisor, unit, obs_date, val, prev_val)
        if sig:
            key = f"fed_liquidity|signal|{series}|{obs_date}|{sig[:40]}"
            if not _seen(conn, key):
                articles.append({
                    "title": f"[FED LIQUIDITY] {sig}",
                    "link": f"https://fred.stlouisfed.org/series/{series}",
                    "summary": (
                        f"{label} ({obs_date}): {sig}\n\n"
                        f"Current value: ${val_b:.2f}{'T' if val_b>=1000 else 'B'}\n"
                        "Source: FRED (Federal Reserve Economic Data)"
                    ),
                    "published": now_utc.isoformat(),
                    "source": SOURCE_NAME,
                    "_tickers": [],
                    "_article_id": _article_id(key),
                })
                _mark_seen(conn, key)
                log.info("fed_liquidity: signal emitted — %s", sig)

    # Emit weekly snapshot article
    if snapshot_date and snapshot_rows:
        snap_key = f"fed_liquidity|snapshot|{snapshot_date}"
        if not _seen(conn, snap_key):
            snapshot_text = "\n".join(snapshot_rows)
            articles.append({
                "title": f"[FED LIQUIDITY SNAPSHOT] Week of {snapshot_date}",
                "link": "https://fred.stlouisfed.org/series/WALCL",
                "summary": (
                    f"Weekly Fed liquidity snapshot ({snapshot_date}):\n\n"
                    f"{snapshot_text}\n\n"
                    "Tracks balance sheet size, RRP drain, TGA level, and bank reserves "
                    "as proxies for system liquidity available to risk assets."
                ),
                "published": now_utc.isoformat(),
                "source": SOURCE_NAME,
                "_tickers": [],
                "_article_id": _article_id(snap_key),
            })
            _mark_seen(conn, snap_key)
            log.info("fed_liquidity: snapshot emitted for %s", snapshot_date)

    conn.close()
    return articles
