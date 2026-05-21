"""G10 sovereign 10-year yield tracker via FRED.

Tracks 10-year government bond yields for G10 economies to monitor global
rate differentials — a key driver of FX flows, equity valuations, and
cross-border capital allocation. Emits a synthetic article when any
country's yield moves more than 10bp since the last observation or
crosses a new 25bp bucket on a given day.

FRED series used (all Monthly/Daily depending on availability):
  US      DGS10            10-Year Treasury Constant Maturity Rate
  DE      IRLTLT01DEM156N  Germany Long-Term Gov't Bond Yield (monthly)
  JP      IRLTLT01JPM156N  Japan Long-Term Gov't Bond Yield (monthly)
  GB      IRLTLT01GBM156N  UK Long-Term Gov't Bond Yield (monthly)
  CA      IRLTLT01CAM156N  Canada Long-Term Gov't Bond Yield (monthly)
  AU      IRLTLT01AUM156N  Australia Long-Term Gov't Bond Yield (monthly)
  CH      IRLTLT01CHM156N  Switzerland Long-Term Gov't Bond Yield (monthly)
  SE      IRLTLT01SEM156N  Sweden Long-Term Gov't Bond Yield (monthly)
  NO      IRLTLT01NOM156N  Norway Long-Term Gov't Bond Yield (monthly)
  NZ      IRLTLT01NZM156N  New Zealand Long-Term Gov't Bond Yield (monthly)

Rate-differential articles: when US-DE or US-JP spread crosses a
50bp bucket, a cross-market spread article is also emitted.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import ssl
import urllib.request

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
SOURCE_NAME = "G10 Sovereign Yields"
HTTP_TIMEOUT = 15
MOVE_THRESHOLD_BP = 10   # emit article if move >= this
BUCKET_BP = 25           # bucket size for level-crossing articles
SPREAD_BUCKET_BP = 50    # bucket for cross-market spread articles

# FRED blocks Python requests UA; urllib + curl UA works
_UA = "curl/7.88.1"

G10_SERIES = {
    "US": ("DGS10",            "United States 10Y Treasury Yield"),
    "DE": ("IRLTLT01DEM156N",  "Germany 10Y Bund Yield"),
    "JP": ("IRLTLT01JPM156N",  "Japan 10Y JGB Yield"),
    "GB": ("IRLTLT01GBM156N",  "UK 10Y Gilt Yield"),
    "CA": ("IRLTLT01CAM156N",  "Canada 10Y Gov't Bond Yield"),
    "AU": ("IRLTLT01AUM156N",  "Australia 10Y Gov't Bond Yield"),
    "CH": ("IRLTLT01CHM156N",  "Switzerland 10Y Gov't Bond Yield"),
    "SE": ("IRLTLT01SEM156N",  "Sweden 10Y Gov't Bond Yield"),
    "NO": ("IRLTLT01NOM156N",  "Norway 10Y Gov't Bond Yield"),
    "NZ": ("IRLTLT01NZM156N",  "New Zealand 10Y Gov't Bond Yield"),
}

# Key differentials to monitor as separate signals
SPREAD_PAIRS = [
    ("US", "DE", "USD-EUR rate differential"),
    ("US", "JP", "USD-JPY rate differential"),
    ("US", "GB", "USD-GBP rate differential"),
]

log = logging.getLogger("g10_sovereign_yields")


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


def _mark(conn: sqlite3.Connection, key: str, link: str, title: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id,link,title,source,first_seen) VALUES (?,?,?,?,?)",
        (key, link, title, SOURCE_NAME, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _fetch_fred_series(sid: str) -> list[tuple[str, float]]:
    """Return [(date_str, value), ...] sorted ascending, dropping . values."""
    url = FRED_CSV.format(sid=sid)
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


def _article_key(country: str, date: str, kind: str) -> str:
    raw = f"g10yield|{country}|{date}|{kind}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def collect_g10_yields() -> list[dict]:
    """Fetch G10 10Y yields and return synthetic article dicts."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    _ensure_db(conn)

    articles: list[dict] = []
    latest: dict[str, tuple[str, float]] = {}  # country -> (date, value)

    for country, (sid, label) in G10_SERIES.items():
        try:
            rows = _fetch_fred_series(sid)
        except Exception as exc:
            log.warning("g10 %s (%s) fetch failed: %s", country, sid, exc)
            continue

        if len(rows) < 2:
            continue

        latest[country] = rows[-1]
        date_str, val = rows[-1]
        prev_date, prev_val = rows[-2]
        move_bp = (val - prev_val) * 100
        bucket = int(val / (BUCKET_BP / 100)) * BUCKET_BP  # e.g. 425 for 4.25-4.49

        # Level crossing article
        level_key = _article_key(country, date_str, f"level_{bucket}")
        if not _seen(conn, level_key):
            direction = "rose" if move_bp >= 0 else "fell"
            link = f"https://fred.stlouisfed.org/series/{sid}"
            title = (
                f"{label}: {val:.2f}% ({move_bp:+.0f}bp vs {prev_date}) "
                f"— {country} 10Y yield {direction} to {val:.2f}%"
            )
            summary = (
                f"{label} as of {date_str}: {val:.2f}% "
                f"(prior: {prev_val:.2f}% on {prev_date}, move: {move_bp:+.1f}bp). "
                f"G10 sovereign yield monitoring — tracks cross-border rate differentials "
                f"and their impact on FX and capital flows."
            )
            articles.append({
                "title": title,
                "link": link,
                "summary": summary,
                "published": date_str,
                "source": f"g10_sovereign_yields/{country}",
            })
            _mark(conn, level_key, link, title)

        # Large-move article (>= MOVE_THRESHOLD_BP)
        if abs(move_bp) >= MOVE_THRESHOLD_BP:
            move_key = _article_key(country, date_str, f"move_{int(move_bp)}")
            if not _seen(conn, move_key):
                direction = "surge" if move_bp >= 0 else "drop"
                link = f"https://fred.stlouisfed.org/series/{sid}"
                title = (
                    f"ALERT: {label} {direction}s {abs(move_bp):.0f}bp to {val:.2f}% "
                    f"({date_str})"
                )
                summary = (
                    f"Significant move in {country} 10-year sovereign yield: "
                    f"{prev_val:.2f}% → {val:.2f}% ({move_bp:+.1f}bp). "
                    f"Large sovereign yield moves can signal shifts in central-bank "
                    f"expectations, inflation regime, or risk appetite."
                )
                articles.append({
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "published": date_str,
                    "source": f"g10_sovereign_yields/{country}_alert",
                })
                _mark(conn, move_key, link, title)

    # Rate differential articles
    for c1, c2, label in SPREAD_PAIRS:
        if c1 not in latest or c2 not in latest:
            continue
        d1, v1 = latest[c1]
        d2, v2 = latest[c2]
        spread_bp = (v1 - v2) * 100
        date_str = d1  # use the more recent date
        bucket = int(spread_bp / SPREAD_BUCKET_BP) * SPREAD_BUCKET_BP

        spread_key = _article_key(f"{c1}_{c2}", date_str, f"spread_{bucket}")
        if not _seen(conn, spread_key):
            link = f"https://fred.stlouisfed.org/series/{G10_SERIES[c1][0]}"
            title = (
                f"{label}: {c1} 10Y at {v1:.2f}% vs {c2} at {v2:.2f}% "
                f"(spread: {spread_bp:+.0f}bp)"
            )
            summary = (
                f"Cross-market rate differential — {label}: "
                f"{c1} 10Y = {v1:.2f}% ({d1}), {c2} 10Y = {v2:.2f}% ({d2}). "
                f"Spread = {spread_bp:+.1f}bp. "
                f"Wide differentials drive carry-trade flows and FX pressure."
            )
            articles.append({
                "title": title,
                "link": link,
                "summary": summary,
                "published": date_str,
                "source": f"g10_sovereign_yields/spread_{c1}_{c2}",
            })
            _mark(conn, spread_key, link, title)

    conn.close()
    log.info("g10_sovereign_yields: %d new articles", len(articles))
    return articles


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = collect_g10_yields()
    print(f"\nFetched {len(results)} new articles:\n")
    for a in results[:10]:
        print(f"  [{a['source']}] {a['title']}")
        print(f"    {a['summary'][:100]}...")
        print()
    if not results:
        print("  (no new articles — all already seen or no FRED data available)")
    sys.exit(0)
