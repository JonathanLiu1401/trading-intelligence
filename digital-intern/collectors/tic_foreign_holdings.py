"""Treasury International Capital (TIC) — major foreign holders of US Treasuries.

Fetches the monthly TIC Table 5 from:
  https://ticdata.treasury.gov/resource-center/data-chart-center/tic/Documents/slt_table5.txt

Publishes one synthetic article per country per reporting month when the
month-over-month change exceeds CHANGE_THRESHOLD_BN (default $5B).  Also
always publishes a Grand Total summary article for the latest month.

Data lag: Treasury releases ~6 weeks after month-end (e.g., March 2026
figures appear in mid-May 2026).  The collector re-checks monthly; a
seen_articles.db entry keyed on (country, period) prevents re-emission.

Why it matters: large shifts in foreign Treasury demand directly affect
long-end yields and USD.  China or Japan reducing holdings is a vol event.
"""
from __future__ import annotations

import hashlib
import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

SOURCE = "tic_foreign_holdings"
URL = (
    "https://ticdata.treasury.gov/resource-center/data-chart-center/"
    "tic/Documents/slt_table5.txt"
)
USER_AGENT = "Mozilla/5.0 (Digital Intern Daemon; TIC collector)"
FETCH_TIMEOUT = 15
CHANGE_THRESHOLD_BN = 5.0  # only emit when abs(MoM change) >= this
TOP_N_COUNTRIES = 10       # track top-N plus Grand Total


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
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
    return conn


def _article_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _is_seen(conn: sqlite3.Connection, aid: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM seen_articles WHERE id=?", (aid,)
    ).fetchone() is not None


def _mark_seen(conn: sqlite3.Connection, aid: str, link: str, title: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
        "VALUES (?, ?, ?, ?, ?)",
        (aid, link, title, SOURCE, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _fetch_table() -> dict[str, dict[str, float]]:
    """Return {country: {period: holdings_bn}} from the TIC TSV."""
    resp = requests.get(URL, timeout=FETCH_TIMEOUT, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()

    data: dict[str, dict[str, float]] = {}
    periods: list[str] = []

    for line in io.StringIO(resp.text):
        parts = [p.strip() for p in line.split("\t")]
        parts = [p for p in parts if p]  # drop blank fields
        if not parts:
            continue

        # Header row: "Country  2026-03  2026-02 ..."
        if parts[0] == "Country":
            periods = parts[1:]
            continue

        if not periods:
            continue

        # Skip footer / notes lines (don't start with a country name token)
        if parts[0].startswith("Of Which") or parts[0].startswith("Notes"):
            continue

        country = parts[0]
        values: dict[str, float] = {}
        for i, period in enumerate(periods):
            if i + 1 < len(parts):
                try:
                    values[period] = float(parts[i + 1].replace(",", ""))
                except ValueError:
                    pass
        if values:
            data[country] = values

    return data


def collect_tic() -> list[dict]:
    """Return new article dicts for significant TIC changes."""
    conn = _ensure_db()
    articles: list[dict] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        table = _fetch_table()
    except Exception as exc:
        return []

    if not table:
        return []

    # Determine the two most-recent periods from Grand Total (always present)
    grand = table.get("Grand Total", {})
    periods = sorted(grand.keys(), reverse=True)
    if len(periods) < 2:
        return []

    latest, prev = periods[0], periods[1]

    # Countries to track: top N by latest holdings + Grand Total
    ranked = sorted(
        [(c, v.get(latest, 0)) for c, v in table.items() if c != "Grand Total"],
        key=lambda x: x[1], reverse=True,
    )
    tracked = [c for c, _ in ranked[:TOP_N_COUNTRIES]] + ["Grand Total"]

    for country in tracked:
        row = table.get(country, {})
        val_latest = row.get(latest)
        val_prev = row.get(prev)
        if val_latest is None or val_prev is None:
            continue

        change = val_latest - val_prev
        pct = (change / val_prev * 100) if val_prev else 0

        # Always emit Grand Total summary; others only on significant change
        if country != "Grand Total" and abs(change) < CHANGE_THRESHOLD_BN:
            continue

        dedup_key = f"{SOURCE}|{country}|{latest}"
        aid = _article_id(dedup_key)
        if _is_seen(conn, aid):
            continue

        direction = "↑" if change >= 0 else "↓"
        change_str = f"{direction}${abs(change):.1f}B ({pct:+.1f}%)"

        if country == "Grand Total":
            title = (
                f"TIC {latest}: Foreign UST holdings ${val_latest:.0f}B "
                f"({change_str} MoM)"
            )
            summary = (
                f"Total foreign holdings of US Treasuries stood at "
                f"${val_latest:.1f}B in {latest}, {change_str} vs "
                f"{prev} (${val_prev:.1f}B). "
                f"Top holders: Japan ${table.get('Japan', {}).get(latest, 0):.0f}B, "
                f"UK ${table.get('United Kingdom', {}).get(latest, 0):.0f}B, "
                f"China ${table.get('China, Mainland', {}).get(latest, 0):.0f}B."
            )
        else:
            title = (
                f"TIC {latest}: {country} UST holdings ${val_latest:.1f}B "
                f"({change_str} MoM)"
            )
            summary = (
                f"{country} held ${val_latest:.1f}B in US Treasuries as of "
                f"{latest}, {change_str} vs {prev}. "
                f"Grand Total foreign holdings: ${grand.get(latest, 0):.0f}B."
            )

        link = f"https://ticdata.treasury.gov/resource-center/data-chart-center/tic/Documents/slt_table5.txt#{country.replace(' ','_')}_{latest}"

        articles.append(
            {
                "title": title,
                "link": link,
                "summary": summary,
                "published": now_iso,
                "source": SOURCE,
            }
        )
        _mark_seen(conn, aid, link, title)

    return articles


if __name__ == "__main__":
    results = collect_tic()
    print(f"TIC collector: {len(results)} new articles")
    for a in results:
        print(f"  [{a['source']}] {a['title']}")
        print(f"    {a['summary'][:120]}")
