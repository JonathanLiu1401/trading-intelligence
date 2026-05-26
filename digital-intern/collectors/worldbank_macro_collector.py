"""World Bank macro indicator collector — synthetic 'article' rows from the
World Bank Open Data API (no API key required).

Covers major developed and emerging-market economies across key macro
dimensions that FRED does not provide:
  - GDP growth (annual %)
  - CPI inflation (annual %)
  - Unemployment rate (% labour force)
  - Current account balance (% GDP)
  - Government gross debt (% GDP)

Each combination of (country, indicator, year) is emitted as one synthetic
article row, following the exact same pattern as fred_collector.py.  Dedup
is keyed by ``wb|{iso3}|{indicator}|{year}`` so re-runs or backfills never
produce duplicates, and a corrected/revised value for the same year is
silently suppressed (same behaviour as FRED).

API endpoint (JSON, no auth):
    https://api.worldbank.org/v2/country/{codes}/indicator/{indicator}
        ?format=json&mrv={N}&per_page=200

Rate limits: WB docs say "no strict limit" but recommend ≤100 req/min.
We batch all countries in a single request per indicator (semicolon-separated
ISO3 codes) so total calls = len(WB_INDICATORS) = 5.
"""

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "data" / "seen_articles.db"

FETCH_TIMEOUT = 15
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Recent observations to surface per (country, indicator).
RECENT_N = 2

# Countries: G20 major economies + South Korea, Netherlands, Spain.
# Semicolon-separated for the WB batch endpoint.
COUNTRY_CODES = [
    "US", "CN", "JP", "DE", "GB", "IN", "BR", "CA", "AU",
    "KR", "MX", "ID", "SA", "TR", "ZA", "AR", "NL", "ES",
]
COUNTRY_CODES_PARAM = ";".join(COUNTRY_CODES)

# Indicator → (source_tag, human_label, unit_suffix)
WB_INDICATORS: dict[str, tuple[str, str, str]] = {
    "NY.GDP.MKTP.KD.ZG": ("wb/gdp_growth",   "GDP growth (annual %)",              "%"),
    "FP.CPI.TOTL.ZG":    ("wb/cpi_inflation", "CPI inflation (annual %)",           "%"),
    "SL.UEM.TOTL.ZS":    ("wb/unemployment",  "Unemployment rate (% labour force)", "%"),
    "BN.CAB.XOKA.GD.ZS": ("wb/current_acct",  "Current account balance (% GDP)",   "% GDP"),
    "GC.DOD.TOTL.GD.ZS": ("wb/govt_debt",     "Government gross debt (% GDP)",      "% GDP"),
}

WB_API = (
    "https://api.worldbank.org/v2/country/{codes}/indicator/{indicator}"
    "?format=json&mrv={mrv}&per_page=500"
)


def _article_id(iso3: str, indicator: str, year: str) -> str:
    key = f"wb|{iso3}|{indicator}|{year}"
    return hashlib.sha256(key.encode()).hexdigest()


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY, link TEXT, title TEXT,
            source TEXT, first_seen TEXT
        )"""
    )
    conn.commit()
    return conn


def _fetch_indicator(indicator: str, mrv: int = 4) -> list[dict]:
    """Return all rows for the indicator across all tracked countries, sorted
    by (country_iso3, date desc)."""
    url = WB_API.format(
        codes=COUNTRY_CODES_PARAM, indicator=indicator, mrv=mrv
    )
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA})
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[worldbank_collector] {indicator}: fetch error — {exc}")
        return []

    payload = resp.json()
    if not isinstance(payload, list) or len(payload) < 2:
        print(f"[worldbank_collector] {indicator}: unexpected payload shape")
        return []

    rows = payload[1] or []
    # Filter out nulls and sort descending by date within each country
    valid = [r for r in rows if r.get("value") is not None]
    valid.sort(key=lambda r: (r["countryiso3code"], r["date"]), reverse=True)
    return valid


def _sign(delta: float) -> str:
    return "+" if delta >= 0 else "−"


def collect_worldbank() -> list[dict]:
    """Collect deduplicated synthetic macro articles from World Bank Open Data.

    Returns list of dicts: {title, link, summary, published, source}.
    """
    conn = _ensure_db()
    new_articles: list[dict] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for indicator, (source_tag, label, unit) in WB_INDICATORS.items():
        rows = _fetch_indicator(indicator, mrv=RECENT_N + 1)
        if not rows:
            continue

        # Group by country
        by_country: dict[str, list[dict]] = {}
        for r in rows:
            iso3 = r.get("countryiso3code", "??")
            by_country.setdefault(iso3, []).append(r)

        for iso3, obs in by_country.items():
            # obs already sorted desc; take up to RECENT_N + 1 for delta
            obs = obs[: RECENT_N + 1]
            if not obs:
                continue

            latest = obs[0]
            year   = latest["date"]
            value  = latest["value"]
            country_name = latest.get("country", {}).get("value", iso3)

            aid = _article_id(iso3, indicator, year)
            try:
                if conn.execute(
                    "SELECT 1 FROM seen_articles WHERE id = ?", (aid,)
                ).fetchone():
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO seen_articles "
                    "(id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
                    (aid, f"https://data.worldbank.org/indicator/{indicator}?locations={iso3}",
                     f"WB {country_name} {label} {year}", source_tag, now_iso),
                )
            except sqlite3.Error as e:
                print(f"[worldbank_collector] dedup error ({iso3}/{indicator}): {e}")
                continue

            # Build title
            prev_val  = obs[1]["value"] if len(obs) > 1 else None
            prev_year = obs[1]["date"]  if len(obs) > 1 else None
            if prev_val is not None:
                delta = value - prev_val
                sign  = _sign(delta)
                title = (
                    f"World Bank: {country_name} {label} {year}: "
                    f"{value:.2f}{unit} ({sign}{abs(delta):.2f} vs {prev_year})"
                )
            else:
                title = (
                    f"World Bank: {country_name} {label} {year}: {value:.2f}{unit}"
                )

            # Build summary
            trend_lines = []
            for o in obs[:RECENT_N]:
                trend_lines.append(f"  {o['date']}: {o['value']:.2f}{unit}")
            summary = (
                f"{country_name} — {label}\n"
                + "\n".join(trend_lines)
            )
            if prev_val is not None:
                direction = "up" if value > prev_val else "down"
                summary += (
                    f"\nYear-on-year change: {_sign(value - prev_val)}{abs(value - prev_val):.2f} "
                    f"({direction} from {prev_year})"
                )

            link = (
                f"https://data.worldbank.org/indicator/{indicator}"
                f"?locations={iso3}"
            )
            new_articles.append({
                "title":     title,
                "link":      link,
                "summary":   summary,
                "published": now_iso,
                "source":    source_tag,
            })

    conn.commit()
    conn.close()
    return new_articles


collect = collect_worldbank


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))

    print("=== World Bank Macro Collector (live fetch) ===\n")
    results = collect_worldbank()

    if not results:
        print("  (no new articles — all already seen in dedup DB)")
    else:
        for ind_key in WB_INDICATORS:
            src = WB_INDICATORS[ind_key][0]
            subset = [a for a in results if a["source"] == src]
            print(f"[{src}] {len(subset)} new articles")
            for a in subset[:3]:
                print(f"  - {a['title']}")
        print(f"\nTotal new: {len(results)}")

        from storage.article_store import ArticleStore
        store = ArticleStore()
        inserted = store.insert_batch(results)
        print(f"Inserted into articles.db: {inserted}")
