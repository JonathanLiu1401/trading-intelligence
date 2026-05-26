"""Treasury yield curve spread collector — 2Y-10Y and 3M-10Y inversion tracker.

Fetches daily spread data from FRED (no API key required):
  T10Y2Y  — 10-Year minus 2-Year Treasury Constant Maturity Spread
  T10Y3M  — 10-Year minus 3-Month Treasury Constant Maturity Spread

Emits synthetic articles on:
  - Inversion / un-inversion events (spread crosses 0)
  - Key threshold crossings: -50bp, -25bp, 0, +25bp, +50bp, +100bp
  - Daily update article when spread changes by >=5bp vs prior day

The 2Y-10Y spread is widely used as a recession leading indicator; an
inversion (negative value) has preceded every U.S. recession since 1955.
The 3M-10Y spread is arguably more accurate (NY Fed recession model uses it).

Dedup: one article per (series, date, threshold_bucket) in seen_articles.db
so the daemon never re-emits the same signal.
"""
from __future__ import annotations

import hashlib
import logging
import ssl
import sqlite3
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
_UA = "curl/7.88.1"
HTTP_TIMEOUT = 15

SOURCE_BASE = "yield_curve_spread"

# FRED series → human label
SERIES = {
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "T10Y3M": "10Y-3M Treasury Spread",
}

# Bucket size for threshold-crossing dedup (basis points)
BUCKET_BP = 25
# Minimum daily move (bp) to emit a move-alert article
MOVE_ALERT_BP = 5

log = logging.getLogger("yield_curve_spread")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

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


def _seen(conn: sqlite3.Connection, key: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM seen_articles WHERE id=?", (key,)
    ).fetchone() is not None


def _mark(conn: sqlite3.Connection, key: str, link: str, title: str, src: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id,link,title,source,first_seen) VALUES (?,?,?,?,?)",
        (key, link, title, src, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# FRED fetch
# ---------------------------------------------------------------------------

def _fetch_series(sid: str) -> list[tuple[str, float]]:
    """Return [(date_str, value), ...] ascending, dropping '.' entries."""
    import urllib.request
    url = FRED_CSV_URL.format(sid=sid)
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT, context=ctx) as resp:
        text = resp.read().decode()
    rows: list[tuple[str, float]] = []
    for line in StringIO(text):
        line = line.strip()
        if not line or line.startswith("DATE"):
            continue
        parts = line.split(",")
        if len(parts) != 2 or parts[1].strip() == ".":
            continue
        try:
            rows.append((parts[0].strip(), float(parts[1].strip())))
        except ValueError:
            continue
    return rows


# ---------------------------------------------------------------------------
# Article builders
# ---------------------------------------------------------------------------

def _inversion_state(val: float) -> str:
    if val < 0:
        return "inverted"
    elif val < 0.10:
        return "near-flat"
    elif val > 1.00:
        return "steep"
    else:
        return "normal"


def _build_articles(
    sid: str,
    label: str,
    rows: list[tuple[str, float]],
    conn: sqlite3.Connection,
) -> list[dict]:
    if len(rows) < 2:
        return []

    articles: list[dict] = []
    link = f"https://fred.stlouisfed.org/series/{sid}"
    src = f"{SOURCE_BASE}/{sid.lower()}"

    date_str, val = rows[-1]
    prev_date, prev_val = rows[-2]
    move_bp = (val - prev_val) * 100
    bucket = int(val / (BUCKET_BP / 100)) * BUCKET_BP  # e.g. 25 means 0.25%-0.49%

    # --- Inversion crossing ---
    crossed_zero = (prev_val >= 0 and val < 0) or (prev_val < 0 and val >= 0)
    if crossed_zero:
        direction = "INVERTED" if val < 0 else "UN-INVERTED"
        inv_key = _sha256(f"{sid}|{date_str}|zero_cross|{direction}")
        if not _seen(conn, inv_key):
            title = (
                f"ALERT: {label} {direction} — {val:+.2f}% ({date_str})"
            )
            summary = (
                f"The {label} has {'turned negative' if val < 0 else 'returned to positive'} "
                f"as of {date_str}: {val:+.3f}% (prior: {prev_val:+.3f}% on {prev_date}, "
                f"move: {move_bp:+.1f}bp). "
                f"{'An inverted yield curve is a historically reliable recession indicator.' if val < 0 else 'The yield curve un-inverting may signal a pre-recession peak has passed.'}"
            )
            articles.append({"title": title, "link": link, "summary": summary,
                              "published": date_str, "source": f"{src}_alert"})
            _mark(conn, inv_key, link, title, src)

    # --- Threshold bucket crossing ---
    thresh_key = _sha256(f"{sid}|{date_str}|bucket|{bucket}")
    if not _seen(conn, thresh_key):
        state = _inversion_state(val)
        title = (
            f"{label}: {val:+.2f}% ({move_bp:+.0f}bp vs {prev_date}) — "
            f"curve {state}"
        )
        summary = (
            f"{label} as of {date_str}: {val:+.3f}% "
            f"(prior: {prev_val:+.3f}% on {prev_date}, Δ {move_bp:+.1f}bp). "
            f"Yield curve state: {state}. "
            f"FRED series {sid} — daily yield spread between "
            f"{'10Y and 2Y' if '2Y' in sid else '10Y and 3-Month'} Treasuries. "
            f"Negative spreads historically precede recessions by 6-18 months."
        )
        articles.append({"title": title, "link": link, "summary": summary,
                          "published": date_str, "source": src})
        _mark(conn, thresh_key, link, title, src)

    # --- Large daily move alert ---
    if abs(move_bp) >= MOVE_ALERT_BP:
        move_key = _sha256(f"{sid}|{date_str}|move|{int(move_bp)}")
        if not _seen(conn, move_key):
            direction_word = "widens" if move_bp > 0 else "narrows"
            title = (
                f"{label} {direction_word} {abs(move_bp):.0f}bp to {val:+.2f}% "
                f"({date_str})"
            )
            summary = (
                f"Daily move in {label}: {prev_val:+.3f}% → {val:+.3f}% "
                f"({move_bp:+.1f}bp on {date_str}). "
                f"Current curve state: {_inversion_state(val)}. "
                f"Large spread moves may reflect shifts in rate-cut expectations, "
                f"growth outlook, or flight-to-safety flows."
            )
            articles.append({"title": title, "link": link, "summary": summary,
                              "published": date_str, "source": f"{src}_move"})
            _mark(conn, move_key, link, title, src)

    return articles


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def collect_yield_curve_spreads() -> list[dict]:
    """Fetch Treasury yield curve spread data and return synthetic articles."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    _ensure_db(conn)

    all_articles: list[dict] = []

    for sid, label in SERIES.items():
        try:
            rows = _fetch_series(sid)
        except Exception as exc:
            log.warning("yield_curve_spread %s fetch failed: %s", sid, exc)
            continue

        articles = _build_articles(sid, label, rows, conn)
        all_articles.extend(articles)
        if articles:
            log.info("yield_curve_spread %s: %d new articles", sid, len(articles))

    conn.close()
    log.info("yield_curve_spread: %d total new articles", len(all_articles))
    return all_articles


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    _ensure_db(conn)

    print("\n=== Treasury Yield Curve Spread Collector — Live Test ===\n")

    for sid, label in SERIES.items():
        try:
            rows = _fetch_series(sid)
        except Exception as e:
            print(f"  {sid}: FETCH ERROR — {e}")
            continue

        if not rows:
            print(f"  {sid}: no data returned")
            continue

        date_str, val = rows[-1]
        prev_date, prev_val = rows[-2] if len(rows) >= 2 else (date_str, val)
        move_bp = (val - prev_val) * 100
        state = _inversion_state(val)

        print(f"  {label} ({sid})")
        print(f"    Latest:  {val:+.3f}%  ({date_str})")
        print(f"    Prior:   {prev_val:+.3f}%  ({prev_date})")
        print(f"    Daily Δ: {move_bp:+.1f}bp")
        print(f"    State:   {state}")
        print()

    conn.close()

    print("Running collect_yield_curve_spreads()...")
    results = collect_yield_curve_spreads()
    print(f"\nNew articles generated: {len(results)}")
    for a in results:
        print(f"  [{a['source']}] {a['title']}")
    if not results:
        print("  (none — all thresholds already seen for today's data)")
    sys.exit(0)
