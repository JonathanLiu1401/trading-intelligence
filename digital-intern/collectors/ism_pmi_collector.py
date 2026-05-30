"""ISM / S&P Global PMI collector.

Monitors press releases and news for ISM Manufacturing PMI, ISM Services PMI,
and S&P Global (Markit) flash PMI estimates via Google News RSS.

PMI readings are major market-moving macro events:
  - ISM Manufacturing PMI: released first business day of each month
  - ISM Services PMI: released third business day of each month
  - S&P Global US Manufacturing Flash PMI: ~22nd-23rd of month
  - S&P Global US Services/Composite Flash PMI: ~23rd of month

Extracts the PMI reading from the headline when present and encodes it in the
article title/summary for urgency scoring. Reading > 50 = expansion, < 50 =
contraction; prints crossing the 50-line get an EXPANSION/CONTRACTION tag.

Deduplication: keyed by ``pmi|source_label|YYYY-MM`` in seen_articles.db so
the same month's reading is inserted once per data source. A revised final
reading (e.g. S&P Global flash → final) gets its own row since source_label
includes "flash" vs "final".
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

SOURCE = "ism_pmi"
FETCH_TIMEOUT = 12

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# PMI-specific Google News RSS queries.  Keep narrow so we don't
# duplicate articles already captured by the general RSS collector.
_FEEDS: list[dict[str, str]] = [
    {
        "label": "ISM Manufacturing PMI",
        "url": (
            "https://news.google.com/rss/search?q=ISM+%22Manufacturing+PMI%22"
            "+%22Report+on+Business%22&hl=en-US&gl=US&ceid=US:en"
        ),
    },
    {
        "label": "ISM Services PMI",
        "url": (
            "https://news.google.com/rss/search?q=ISM+%22Services+PMI%22"
            "+%22Report+on+Business%22&hl=en-US&gl=US&ceid=US:en"
        ),
    },
    {
        "label": "S&P Global Manufacturing PMI",
        "url": (
            "https://news.google.com/rss/search?q=%22S%26P+Global%22"
            "+%22Manufacturing+PMI%22&hl=en-US&gl=US&ceid=US:en"
        ),
    },
    {
        "label": "S&P Global Services PMI",
        "url": (
            "https://news.google.com/rss/search?q=%22S%26P+Global%22"
            "+%22Services+PMI%22&hl=en-US&gl=US&ceid=US:en"
        ),
    },
    {
        "label": "S&P Global Composite PMI",
        "url": (
            "https://news.google.com/rss/search?q=%22S%26P+Global%22"
            "+%22Composite+PMI%22&hl=en-US&gl=US&ceid=US:en"
        ),
    },
]

# Regex to pull the PMI reading from a headline.
# Matches patterns like: "at 52.7%", "at 52.7", "PMI 52.7", "PMI® 52.7"
_PMI_RE = re.compile(
    r"(?:PMI[®™]?\s*(?:at|:)?\s*|at\s+)(\d{2}(?:\.\d)?)"
    r"(?:%|\b)",
    re.IGNORECASE,
)

# Month-year pattern in headlines: "April 2026", "May 2026", etc.
_MONTH_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(20\d{2})\b",
    re.IGNORECASE,
)


def _ensure_seen_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_pmi (
            key TEXT PRIMARY KEY,
            inserted_at TEXT
        )"""
    )
    conn.commit()
    return conn


def _seen_key(label: str, month_tag: str) -> str:
    return f"pmi|{label}|{month_tag}"


def _extract_reading(text: str) -> float | None:
    m = _PMI_RE.search(text)
    if m:
        val = float(m.group(1))
        if 20 < val < 90:  # sanity-check: PMI is always 0-100 in practice 25-75
            return val
    return None


def _extract_month_tag(text: str) -> str:
    """Return 'YYYY-MM' from headline; fall back to current month."""
    m = _MONTH_RE.search(text)
    if m:
        month_str = m.group(1)[:3].lower()
        year = int(m.group(2))
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        return f"{year}-{month_map.get(month_str, 0):02d}"
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _signal_tag(reading: float | None) -> str:
    if reading is None:
        return ""
    if reading >= 55:
        return " [STRONG EXPANSION]"
    if reading >= 50:
        return " [EXPANSION]"
    if reading >= 45:
        return " [CONTRACTION]"
    return " [SHARP CONTRACTION]"


def collect_pmi() -> list[dict[str, Any]]:
    conn = _ensure_seen_db()
    articles: list[dict[str, Any]] = []

    for feed_cfg in _FEEDS:
        label = feed_cfg["label"]
        try:
            feed = feedparser.parse(
                feed_cfg["url"],
                request_headers={"User-Agent": _UA},
            )
        except Exception as exc:
            print(f"[ism_pmi] fetch error for {label}: {exc}")
            continue

        for entry in feed.entries[:8]:
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            if not title:
                continue

            reading = _extract_reading(title)
            month_tag = _extract_month_tag(title)

            # Use a per-(label, month) key so flash and final don't collide
            flash_tag = "flash" if "flash" in title.lower() else "final"
            seen_key = _seen_key(f"{label}:{flash_tag}", month_tag)

            try:
                existing = conn.execute(
                    "SELECT 1 FROM seen_pmi WHERE key=?", (seen_key,)
                ).fetchone()
            except sqlite3.OperationalError:
                existing = None

            if existing:
                continue

            signal = _signal_tag(reading)
            reading_str = f" — PMI {reading:.1f}" if reading else ""
            enriched_title = f"[PMI] {title}{reading_str}{signal}"

            summary_parts = [f"Source: {label}"]
            if reading:
                direction = "expansion" if reading >= 50 else "contraction"
                summary_parts.append(
                    f"PMI reading {reading:.1f} — "
                    f"{'above' if reading >= 50 else 'below'} the 50-mark "
                    f"({direction})."
                )
            summary_parts.append(f"Period: {month_tag}.")
            summary = " ".join(summary_parts)

            published = entry.get("published", "")

            articles.append(
                {
                    "title": enriched_title,
                    "link": link,
                    "summary": summary,
                    "published": published,
                    "source": SOURCE,
                }
            )

            try:
                conn.execute(
                    "INSERT OR IGNORE INTO seen_pmi (key, inserted_at) VALUES (?, ?)",
                    (seen_key, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
            except sqlite3.OperationalError:
                pass

    conn.close()
    return articles


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(BASE_DIR))
    from storage.article_store import ArticleStore

    print("[ism_pmi] Fetching PMI data...")
    items = collect_pmi()
    print(f"[ism_pmi] {len(items)} new PMI articles found")
    for a in items:
        print(f"  {a['source']} | {a['title'][:100]}")
        print(f"    link: {a['link'][:80]}")

    if items:
        store = ArticleStore()
        inserted = store.insert_batch(items)
        print(f"[ism_pmi] Inserted {inserted} rows into articles.db")
    else:
        print("[ism_pmi] No new articles (all already seen or no data)")
        # Still show what feeds returned to confirm they work
        print("\n[ism_pmi] Live feed check (bypassing dedup):")
        import feedparser

        _UA2 = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "Chrome/124.0 Safari/537.36"
        )
        for feed_cfg in _FEEDS:
            feed = feedparser.parse(
                feed_cfg["url"],
                request_headers={"User-Agent": _UA2},
            )
            print(f"\n  {feed_cfg['label']} ({len(feed.entries)} entries):")
            for e in feed.entries[:3]:
                t = e.get("title", "")
                reading = _extract_reading(t)
                print(f"    [{reading or '?':>5}] {t[:85]}")
