"""S&P 500 market valuation collector — multpl.com scraper.

Fetches S&P 500 trailing P/E, Shiller CAPE (cyclically adjusted P/E),
and earnings yield from multpl.com (no API key, public HTML).

Emits a synthetic article when the valuation regime changes or crosses
a meaningful threshold (daily dedup so at most one per regime per day):
  - CAPE > 40: extreme overvaluation (dot-com bubble territory)
  - CAPE > 30: expensive (above historical median 2x)
  - CAPE < 20: fair value
  - CAPE < 15: undervalued

Dedup key: date + regime bucket so reruns are idempotent.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "articles.db"
SOURCE = "market_valuation"

HTTP_TIMEOUT = 10
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

# Historical Shiller CAPE reference points (from multpl.com)
CAPE_MEAN = 17.38
CAPE_MEDIAN = 16.09
CAPE_MAX = 44.19  # Dec 1999 peak

CAPE_REGIMES = [
    (40.0, "extreme_overvalued", "Extreme Overvaluation"),
    (30.0, "expensive",         "Expensive"),
    (20.0, "fair_value",        "Fair Value"),
    (15.0, "undervalued",       "Undervalued"),
    (0.0,  "deeply_undervalued","Deep Undervaluation"),
]

log = logging.getLogger("market_valuation_collector")


def _fetch_multpl(slug: str) -> tuple[float | None, str | None]:
    """Fetch current value from a multpl.com page. Returns (value, date_str)."""
    url = f"https://www.multpl.com/{slug}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        div = soup.find(id="current")
        if not div:
            return None, None
        text = div.get_text(separator=" ")
        # Extract numeric value (first float after the title)
        nums = re.findall(r"[-+]?\d+\.\d+", text)
        if not nums:
            return None, None
        # The main value is the first standalone number (not a small change)
        value = float(nums[0])
        # Extract date
        ts_div = div.find(id="timestamp")
        date_str = ts_div.get_text(strip=True) if ts_div else None
        return value, date_str
    except Exception as exc:
        log.warning("market_valuation: fetch %s failed: %s", slug, exc)
        return None, None


def _classify_cape(cape: float) -> tuple[str, str]:
    for threshold, key, label in CAPE_REGIMES:
        if cape >= threshold:
            return key, label
    return "deeply_undervalued", "Deep Undervaluation"


def _percentile_vs_history(cape: float) -> str:
    """Rough percentile description vs historical CAPE range."""
    pct_of_mean = cape / CAPE_MEAN
    pct_of_peak = cape / CAPE_MAX * 100
    return f"{pct_of_mean:.1f}x historical mean ({pct_of_peak:.0f}% of {CAPE_MAX} all-time peak)"


def _urgency_score(cape: float | None, pe: float | None) -> float:
    """Higher urgency for more extreme readings."""
    score = 2.0
    if cape is not None:
        if cape >= 40:
            score += 3.5
        elif cape >= 35:
            score += 2.0
        elif cape >= 30:
            score += 1.0
        elif cape <= 15:
            score += 2.5
    if pe is not None and pe >= 30:
        score += 1.0
    return min(score, 8.0)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS market_valuation_state (
            key TEXT PRIMARY KEY,
            value REAL
        )"""
    )
    conn.commit()


def _article_id(date_str: str, regime: str) -> str:
    return hashlib.sha256(f"{SOURCE}:{date_str}:{regime}".encode()).hexdigest()[:16]


def collect() -> list[dict]:
    """Fetch S&P 500 valuation metrics and emit article if regime changed."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    try:
        _ensure_schema(conn)

        cape, cape_date = _fetch_multpl("shiller-pe")
        pe, pe_date = _fetch_multpl("s-p-500-pe-ratio")
        ey, ey_date = _fetch_multpl("s-p-500-earnings-yield")

        if cape is None:
            log.warning("market_valuation: failed to fetch CAPE")
            return []

        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        regime, regime_label = _classify_cape(cape)

        article_id = _article_id(date_str, regime)
        already_exists = conn.execute(
            "SELECT 1 FROM articles WHERE id=? LIMIT 1", (article_id,)
        ).fetchone()
        if already_exists:
            log.debug("market_valuation: %s already emitted today", regime)
            return []

        hist_desc = _percentile_vs_history(cape)
        pe_str = f"{pe:.2f}" if pe is not None else "N/A"
        ey_str = f"{ey:.2f}%" if ey is not None else "N/A"

        title = (
            f"S&P 500 valuation: {regime_label} — "
            f"CAPE {cape:.2f} ({hist_desc}), P/E {pe_str}"
        )
        url = f"internal://market_valuation/{date_str}/{regime}"

        body_lines = [
            f"Shiller CAPE (P/E10): {cape:.2f}",
            f"Trailing P/E:         {pe_str}",
            f"Earnings Yield:       {ey_str}",
            "",
            f"Regime: {regime_label}",
            f"vs. CAPE mean ({CAPE_MEAN}): {hist_desc}",
            "",
        ]
        if cape >= 40:
            body_lines.append(
                "CAPE above 40: approaching dot-com bubble levels (all-time peak: 44.19). "
                "Historically associated with poor 10-year forward returns."
            )
        elif cape >= 30:
            body_lines.append(
                f"CAPE of {cape:.1f} is {cape/CAPE_MEAN:.1f}x the historical mean. "
                "Elevated valuation; market typically produces below-average 10-year returns at this level."
            )
        elif cape <= 15:
            body_lines.append(
                "CAPE below 15: historically associated with above-average 10-year forward returns."
            )

        if pe is not None and pe >= 25:
            body_lines.append(
                f"Trailing P/E of {pe:.1f} also elevated vs. long-run ~17x average."
            )

        full_text = "\n".join(body_lines)
        kw = _urgency_score(cape, pe)
        ts = now.strftime("%Y-%m-%d %H:%M:%S")
        compressed = zlib.compress(full_text.encode("utf-8"))

        conn.execute(
            """INSERT OR IGNORE INTO articles
               (id, url, title, source, published, kw_score, urgency, full_text, first_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                article_id, url, title, SOURCE, ts,
                kw, 1 if kw >= 6.0 else 0,
                compressed, ts,
            ),
        )
        conn.commit()
        log.info("market_valuation: emitted — %s", title)

        return [{
            "id": article_id,
            "url": url,
            "title": title,
            "full_text": full_text,
            "source": SOURCE,
            "published": ts,
            "kw_score": kw,
        }]

    finally:
        conn.close()


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=logging.INFO)

    print("=== Market Valuation Collector — Live Test ===")
    cape, cape_date = _fetch_multpl("shiller-pe")
    pe, pe_date = _fetch_multpl("s-p-500-pe-ratio")
    ey, ey_date = _fetch_multpl("s-p-500-earnings-yield")
    print(f"Shiller CAPE: {cape}  (as of: {cape_date})")
    print(f"Trailing P/E: {pe}  (as of: {pe_date})")
    print(f"Earnings Yield: {ey}%  (as of: {ey_date})")

    if cape:
        regime, label = _classify_cape(cape)
        print(f"Regime: {label} ({regime})")
        print(f"vs mean ({CAPE_MEAN}): {_percentile_vs_history(cape)}")

    print("\n--- Running collect() ---")
    results = collect()
    if results:
        for r in results:
            print(f"Title:  {r['title']}")
            print(f"Score:  {r['kw_score']}")
            print(f"Body:\n{r['full_text']}")
    else:
        print("No new article (already emitted today or fetch failed).")
        conn = sqlite3.connect(str(DB_PATH), timeout=10)
        row = conn.execute(
            "SELECT title, kw_score, first_seen FROM articles "
            "WHERE source=? ORDER BY first_seen DESC LIMIT 1",
            (SOURCE,)
        ).fetchone()
        conn.close()
        if row:
            print(f"Last emitted: {row[0]!r} (score={row[1]}, at={row[2]})")
