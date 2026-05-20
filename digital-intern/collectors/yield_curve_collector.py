"""Yield curve inversion monitor.

Computes the 10Y-2Y Treasury spread from FRED's public CSV graph endpoint
and emits a synthetic article when the inversion state changes (entry/exit)
or when the spread crosses a new 25-bp bucket on a given day. This gives
the briefing pipeline an explicit recession-signal feature distinct from
the per-observation rows in fred_collector (which already pulls DGS2/DGS10
but doesn't compute or alert on the spread).

Dedup keys (date-scoped, so a revised value never re-emits):
  - state:    "yc|<date>|inverted|<bool>"
  - bucket:   "yc|<date>|bp_<bucket>"  (25-bp buckets)
"""
import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
SOURCE_NAME = "Yield Curve Monitor"
HTTP_TIMEOUT = 15
BUCKET_BP = 25  # bp

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
}

log = logging.getLogger("yield_curve_collector")


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
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _latest_value(series_id: str) -> tuple[str, float] | None:
    """Return (date_str, value) for the most recent non-dot row, else None."""
    try:
        r = requests.get(
            FRED_CSV.format(sid=series_id),
            headers=HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
    except Exception as e:
        log.warning(f"[yield_curve] fetch {series_id} failed: {e}")
        return None

    # CSV: "DATE,<SERIES>" then rows; missing values are '.'
    rows = [ln.strip() for ln in StringIO(r.text).read().splitlines() if ln.strip()]
    for ln in reversed(rows[1:]):  # skip header
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        date_str, val_str = parts[0], parts[1].strip()
        if not val_str or val_str == ".":
            continue
        try:
            return date_str, float(val_str)
        except ValueError:
            continue
    return None


def collect_yield_curve() -> list[dict]:
    """Compute 10Y-2Y spread; emit net-new articles on state/bucket change."""
    y10 = _latest_value("DGS10")
    y2 = _latest_value("DGS2")
    if not y10 or not y2:
        return []

    # Use whichever common date both series share (FRED publishes same day).
    date_str = y10[0] if y10[0] <= y2[0] else y2[0]
    spread_pct = round(y10[1] - y2[1], 3)  # percentage points
    spread_bp = int(round(spread_pct * 100))
    inverted = spread_pct < 0

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    state_key = f"yc|{date_str}|inverted|{int(inverted)}"
    state_id = _article_id(state_key)

    # 25-bp buckets, signed (handles negative correctly via floor division).
    bucket = (spread_bp // BUCKET_BP) * BUCKET_BP
    bucket_key = f"yc|{date_str}|bp_{bucket}"
    bucket_id = _article_id(bucket_key)

    link = "https://fred.stlouisfed.org/graph/?g=T10Y2Y"
    sign = "+" if spread_pct >= 0 else ""
    flag = "🚨 INVERTED" if inverted else "✅ NORMAL"
    title = (
        f"{flag} 10Y-2Y spread: {sign}{spread_pct:.2f}pp "
        f"({sign}{spread_bp}bp) | 10Y {y10[1]:.2f}% / 2Y {y2[1]:.2f}% [{date_str}]"
    )
    summary = (
        f"US Treasury 10Y-2Y yield spread is {spread_pct:.3f} percentage points "
        f"({spread_bp} bp) on {date_str}. 10Y={y10[1]:.3f}%, 2Y={y2[1]:.3f}%. "
        f"State: {'inverted (negative spread, historical recession signal)' if inverted else 'normal (positive spread)'}."
    )

    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    _ensure_db(conn)
    results: list[dict] = []
    try:
        for art_id, _key in [(state_id, state_key), (bucket_id, bucket_key)]:
            if conn.execute(
                "SELECT 1 FROM seen_articles WHERE id=?", (art_id,)
            ).fetchone():
                continue
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                "VALUES (?,?,?,?,?)",
                (art_id, link, title, SOURCE_NAME, now),
            )
            results.append({
                "id": art_id,
                "link": link,
                "title": title,
                "summary": summary,
                "source": SOURCE_NAME,
                "first_seen": now,
                "yc_spread_pp": spread_pct,
                "yc_spread_bp": spread_bp,
                "yc_inverted": inverted,
                "yc_date": date_str,
            })
        conn.commit()
    finally:
        conn.close()

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arts = collect_yield_curve()
    if arts:
        for a in arts:
            print(f"NEW: {a['title']}")
            print(f"     {a['summary']}")
    else:
        print("No new yield-curve articles (already seen for this date/state/bucket)")
