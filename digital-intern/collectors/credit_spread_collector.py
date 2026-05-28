"""Credit spread monitor — IG / HY / CCC OAS from ICE BofA / FRED.

Fetches daily Option-Adjusted Spread (OAS) data from FRED's public CSV
endpoint (no API key required):

  BAMLC0A0CM     — ICE BofA US Corporate Bond Index OAS (IG aggregate)
  BAMLC0A4CBBBEY — ICE BofA BBB US Corporate OAS (lowest IG tier)
  BAMLH0A0HYM2   — ICE BofA US High Yield Index OAS (HY aggregate)
  BAMLH0A3HYC    — ICE BofA CCC & Lower US High Yield OAS (distressed)

All values are in percentage points (e.g. 1.20 = 120bp above Treasuries).

Emits synthetic articles on:
  - Threshold bucket crossings (25bp buckets for IG, 50bp for HY/CCC)
  - Daily moves ≥ 10bp (IG) or ≥ 25bp (HY/CCC)
  - Stress-level transitions: normal / elevated / stress / crisis

Stress levels (HY OAS):
  < 300bp  — normal (post-GFC baseline)
  300-400  — elevated caution
  400-600  — stress (recession risk)
  > 600bp  — crisis (GFC / COVID peak)

Dedup: one article per (series, date, threshold_bucket) in seen_articles.db.
"""
from __future__ import annotations

import hashlib
import logging
import ssl
import sqlite3
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import urllib.request

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
_UA = "curl/7.88.1"
HTTP_TIMEOUT = 20

SOURCE_BASE = "credit_spread"

# FRED series → (human label, asset class, bucket_bp, move_alert_bp)
SERIES = {
    "BAMLC0A0CM":     ("US IG Corporate OAS",    "ig",       25, 10),
    "BAMLC0A4CBBB":   ("US BBB Corporate OAS",   "ig_bbb",   25, 10),
    "BAMLH0A0HYM2":   ("US HY Corporate OAS",    "hy",       50, 25),
    "BAMLH0A3HYC":    ("US CCC & Lower OAS",     "hy_ccc",   50, 25),
}

log = logging.getLogger("credit_spread")


# ---------------------------------------------------------------------------
# DB helpers (mirror yield_curve_spread pattern)
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
    """Return [(date_str, value_pct), ...] ascending, dropping '.' entries."""
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
# Stress classification
# ---------------------------------------------------------------------------

def _hy_stress_level(oas_pct: float) -> str:
    bp = oas_pct * 100
    if bp < 300:
        return "normal"
    elif bp < 400:
        return "elevated"
    elif bp < 600:
        return "stress"
    else:
        return "crisis"


def _ig_stress_level(oas_pct: float) -> str:
    bp = oas_pct * 100
    if bp < 100:
        return "tight"
    elif bp < 150:
        return "normal"
    elif bp < 200:
        return "elevated"
    else:
        return "stress"


def _stress_level(oas_pct: float, asset_class: str) -> str:
    if asset_class.startswith("hy"):
        return _hy_stress_level(oas_pct)
    return _ig_stress_level(oas_pct)


# ---------------------------------------------------------------------------
# Article builder
# ---------------------------------------------------------------------------

def _build_articles(
    sid: str,
    label: str,
    asset_class: str,
    bucket_bp: int,
    move_alert_bp: int,
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
    bp_val = val * 100

    # Round to nearest bucket
    bucket = int(bp_val / bucket_bp) * bucket_bp
    stress = _stress_level(val, asset_class)

    # --- Threshold bucket crossing ---
    thresh_key = _sha256(f"cs|{sid}|{date_str}|bucket|{bucket}")
    if not _seen(conn, thresh_key):
        direction_word = "widens" if move_bp > 0 else "tightens"
        title = (
            f"{label}: {bp_val:.0f}bp ({move_bp:+.0f}bp vs {prev_date}) — {stress}"
        )
        summary = (
            f"{label} (FRED: {sid}) as of {date_str}: {bp_val:.1f}bp "
            f"(prior: {prev_val * 100:.1f}bp on {prev_date}, Δ {move_bp:+.1f}bp). "
            f"Stress level: {stress}. "
            f"Credit spreads measure the extra yield investors demand over Treasuries; "
            f"widening spreads signal rising default risk and risk-off sentiment. "
            f"HY crisis threshold: 600bp; IG stress threshold: 200bp."
        )
        articles.append({
            "title": title, "link": link, "summary": summary,
            "published": date_str, "source": src,
        })
        _mark(conn, thresh_key, link, title, src)

    # --- Large daily move alert ---
    if abs(move_bp) >= move_alert_bp:
        move_key = _sha256(f"cs|{sid}|{date_str}|move|{int(move_bp)}")
        if not _seen(conn, move_key):
            direction_word = "widens" if move_bp > 0 else "tightens"
            title = (
                f"ALERT: {label} {direction_word} {abs(move_bp):.0f}bp to {bp_val:.0f}bp "
                f"({date_str})"
            )
            summary = (
                f"Significant credit spread move in {label}: "
                f"{prev_val * 100:.1f}bp → {bp_val:.1f}bp "
                f"({move_bp:+.1f}bp on {date_str}). "
                f"Current stress level: {stress}. "
                f"Sharp spread widening may reflect flight-to-quality, "
                f"deteriorating credit conditions, or systemic risk concerns."
            )
            articles.append({
                "title": title, "link": link, "summary": summary,
                "published": date_str, "source": f"{src}_alert",
            })
            _mark(conn, move_key, link, title, src)

    return articles


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def collect_credit_spreads() -> list[dict]:
    """Fetch ICE BofA credit spread OAS data and return synthetic articles."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    _ensure_db(conn)

    all_articles: list[dict] = []

    for sid, (label, asset_class, bucket_bp, move_alert_bp) in SERIES.items():
        try:
            rows = _fetch_series(sid)
        except Exception as exc:
            log.warning("credit_spread %s fetch failed: %s", sid, exc)
            continue

        articles = _build_articles(sid, label, asset_class, bucket_bp, move_alert_bp, rows, conn)
        all_articles.extend(articles)
        if articles:
            log.info("credit_spread %s: %d new articles", sid, len(articles))

    conn.close()
    log.info("credit_spread: %d total new articles", len(all_articles))
    return all_articles


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    articles = collect_credit_spreads()
    print(f"\nTotal new articles: {len(articles)}")
    for a in articles:
        print(f"  [{a['source']}] {a['title']}")
    # Show live values even if all already seen
    print("\n--- Live OAS values ---")
    for sid, (label, asset_class, _, _) in SERIES.items():
        try:
            rows = _fetch_series(sid)
            if rows:
                date_str, val = rows[-1]
                prev_date, prev_val = rows[-2] if len(rows) > 1 else (date_str, val)
                move_bp = (val - prev_val) * 100
                stress = _stress_level(val, asset_class)
                print(f"  {label}: {val * 100:.1f}bp ({move_bp:+.1f}bp) — {stress} [{date_str}]")
        except Exception as exc:
            print(f"  {label}: FETCH ERROR — {exc}")
