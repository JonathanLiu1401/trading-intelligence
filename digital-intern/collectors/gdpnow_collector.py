"""Atlanta Fed GDPNow real-time GDP tracker.

Pulls the FRED GDPNOW series — the Atlanta Fed's GDPNow model estimate for
the current quarter's real GDP growth (annualised %).  Unlike GDPC1 (final
BEA releases), GDPNow is updated intra-quarter (often several times per week)
as new high-frequency data arrives, making it a real-time leading indicator
of GDP trajectory.

Key differences from fred_collector.py:
  - Tracks only the CURRENT quarter estimate (most recent observation)
  - Emits a new article on every observable revision (not just first-seen date)
  - Dedup key: quarter_date + rounded_value so small floating-point noise is
    suppressed but any ≥0.1 pp change in the published estimate creates a
    new article

Source tag: ``fred/gdpnow``

No API key required — FRED's public fredgraph.csv endpoint.
"""
from __future__ import annotations

import hashlib
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SEEN_DB = BASE_DIR / "data" / "gdpnow_seen.db"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDPNOW"
FETCH_TIMEOUT = 15
SOURCE = "fred/gdpnow"

# Emit an article whenever the estimate shifts ≥ this many percentage points
# from the previously seen value for the same quarter.
REVISION_THRESHOLD = 0.1  # pp


def _init_seen(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS gdpnow_seen (
               dedup_key TEXT PRIMARY KEY,
               first_seen TEXT
           )"""
    )
    conn.commit()


def _fetch_series() -> list[tuple[str, float]]:
    """Return list of (observation_date, value) from FRED GDPNOW CSV.

    Uses curl subprocess: FRED applies TLS-fingerprint blocking that rejects
    Python's ssl/urllib3 TLS handshake while allowing curl's.
    """
    result = subprocess.run(
        ["curl", "-s", "--max-time", str(FETCH_TIMEOUT), FRED_CSV_URL],
        capture_output=True,
        text=True,
        timeout=FETCH_TIMEOUT + 5,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr[:200]}")
    rows: list[tuple[str, float]] = []
    for line in result.stdout.splitlines():
        if line.startswith("observation_date") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        date_str, val_str = parts[0].strip(), parts[1].strip()
        if val_str in (".", ""):
            continue
        try:
            rows.append((date_str, float(val_str)))
        except ValueError:
            continue
    return rows


def _quarter_label(date_str: str) -> str:
    """Convert YYYY-MM-DD (quarter start) to 'Q1 2026' style label."""
    year, month = int(date_str[:4]), int(date_str[5:7])
    q = (month - 1) // 3 + 1
    return f"Q{q} {year}"


def _sentiment_tag(current: float, prev: float | None) -> str:
    if prev is None:
        return ""
    delta = current - prev
    if delta > 0.3:
        return f" ↑ +{delta:.2f}pp revision"
    if delta < -0.3:
        return f" ↓ {delta:.2f}pp revision"
    return f" ({delta:+.2f}pp)"


def collect_gdpnow() -> list[dict]:
    """Fetch GDPNow latest estimate, return new articles if estimate changed."""
    rows = _fetch_series()
    if not rows:
        return []

    # Focus on the two most recent quarterly observations
    recent = rows[-2:]  # (older_date, older_val), (current_date, current_val)
    current_date, current_val = recent[-1]
    prev_val: float | None = recent[-2][1] if len(recent) == 2 else None

    # Dedup key: quarter_date + value rounded to 1dp (suppress float noise)
    rounded = round(current_val, 1)
    dedup_key = f"gdpnow|{current_date}|{rounded:.1f}"

    conn = sqlite3.connect(str(SEEN_DB), timeout=30000)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    _init_seen(conn)

    try:
        row = conn.execute(
            "SELECT dedup_key FROM gdpnow_seen WHERE dedup_key=?", (dedup_key,)
        ).fetchone()
        if row:
            return []

        qlabel = _quarter_label(current_date)
        sign = "+" if current_val > 0 else ""
        tag = _sentiment_tag(current_val, prev_val)
        direction = "above" if current_val > 2.0 else "below"
        trend_note = (
            f"Current estimate of {sign}{current_val:.2f}% is "
            f"{direction} the long-run average (~2%)."
        )

        prev_note = ""
        if prev_val is not None:
            prev_sign = "+" if prev_val > 0 else ""
            prev_qlabel = _quarter_label(recent[-2][0]) if len(recent) == 2 else ""
            prev_note = (
                f" Previous quarter ({prev_qlabel}) final: {prev_sign}{prev_val:.2f}%."
            )

        title = (
            f"Atlanta Fed GDPNow {qlabel}: {sign}{current_val:.2f}% annualised real GDP"
            f"{tag}"
        )
        summary = (
            f"The Atlanta Fed GDPNow model now estimates {qlabel} real GDP growth at "
            f"{sign}{current_val:.2f}% (annualised). {trend_note}{prev_note} "
            f"GDPNow is updated intra-quarter as new economic data arrives; "
            f"it is a real-time nowcast, not an official BEA release."
        )
        url = f"https://fred.stlouisfed.org/series/GDPNOW#{dedup_key}"
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        link = f"https://www.atlantafed.org/ctevent/gdpnow?ref={url_hash}"

        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO gdpnow_seen (dedup_key, first_seen) VALUES (?,?)",
            (dedup_key, now_iso),
        )
        conn.commit()

        return [
            {
                "title": title,
                "link": link,
                "summary": summary,
                "published": now_iso,
                "source": SOURCE,
            }
        ]
    finally:
        conn.close()


collect = collect_gdpnow


if __name__ == "__main__":
    rows = _fetch_series()
    print(f"GDPNOW series: {len(rows)} observations total")
    print("Latest 5 observations:")
    for date, val in rows[-5:]:
        sign = "+" if val > 0 else ""
        print(f"  {date}  {sign}{val:.4f}%  ({_quarter_label(date)})")

    items = collect_gdpnow()
    if items:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from storage.article_store import ArticleStore
        store = ArticleStore()
        inserted = store.insert_batch(items)
        print(f"\nNew articles emitted: {len(items)}, inserted: {inserted}")
        for a in items:
            print(f"  TITLE:   {a['title']}")
            print(f"  SUMMARY: {a['summary'][:120]}...")
            print(f"  SOURCE:  {a['source']}")
            print(f"  LINK:    {a['link']}")
    else:
        print("\nNo new articles (estimate unchanged since last run)")
        print("DISCORD_EG: GDPNow Q2 2026 = +3.82% (no revision)")
