"""SEC Enforcement Actions collector — litigation releases, admin proceedings,
and trading suspensions from the Securities and Exchange Commission.

Three distinct RSS feeds, each covering a separate enforcement channel:

  - Litigation releases: SEC sues companies/individuals in federal court.
    Fraud charges, insider trading, market manipulation. These name companies
    and move their stock prices significantly on announcement.

  - Administrative proceedings: SEC internal enforcement via hearings.
    Typically investment advisers, broker-dealers, accountants. Less
    dramatic but still market-signal (e.g. UPS, Foot Locker appeared May 2026).

  - Trading suspensions: SEC halts trading in a stock for up to 10 days.
    Immediate, binary market event — the stock cannot be traded while suspended.

All feeds are public, no API key required. SEC requires a real identifying
User-Agent per https://www.sec.gov/os/accessing-edgar-data.

Dedup: keyed on guid (SEC-assigned UUID per filing) in seen_articles.db.
One article per enforcement action; re-run safe.

Standalone smoke test:
    python3 collectors/sec_enforcement_collector.py
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import feedparser

log = logging.getLogger("sec_enforcement_collector")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

# SEC policy requires an identifying User-Agent; otherwise 403/blocked.
_UA = "Digital-Intern/1.0 data-collector (contact@digital-intern.local)"

# RSS feeds for SEC enforcement channels.
FEEDS = [
    {
        "url": "https://www.sec.gov/enforcement-litigation/litigation-releases/rss",
        "source": "SEC/LitigationRelease",
        "label": "SEC Litigation Release",
        "priority": "high",   # federal court charges — direct company/individual naming
    },
    {
        "url": "https://www.sec.gov/enforcement-litigation/administrative-proceedings/rss",
        "source": "SEC/AdminProceeding",
        "label": "SEC Administrative Proceeding",
        "priority": "medium",
    },
    {
        "url": "https://www.sec.gov/enforcement-litigation/trading-suspensions/rss",
        "source": "SEC/TradingSuspension",
        "label": "SEC TRADING SUSPENSION",
        "priority": "critical",  # immediate market halt
    },
]

FETCH_TIMEOUT = 15


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


def _seen_id(guid: str) -> str:
    return hashlib.sha256(f"sec_enforcement:{guid}".encode()).hexdigest()


def _is_seen(conn: sqlite3.Connection, sid: str) -> bool:
    return bool(
        conn.execute("SELECT 1 FROM seen_articles WHERE id=? LIMIT 1", (sid,)).fetchone()
    )


def _mark_seen(conn: sqlite3.Connection, sid: str, link: str, title: str, source: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
        "VALUES (?, ?, ?, ?, ?)",
        (sid, link.strip(), title.strip(), source, now),
    )


def collect_sec_enforcement() -> list[dict]:
    """Fetch all three SEC enforcement RSS feeds, return new articles."""
    conn = _ensure_db()
    results: list[dict] = []

    for feed_cfg in FEEDS:
        url = feed_cfg["url"]
        source = feed_cfg["source"]
        label = feed_cfg["label"]

        try:
            parsed = feedparser.parse(url, agent=_UA, request_headers={"User-Agent": _UA})
        except Exception as e:
            log.warning("[sec_enforcement] fetch failed for %s: %s", source, e)
            continue

        if parsed.bozo and not parsed.entries:
            log.warning("[sec_enforcement] malformed feed from %s: %s", source, parsed.bozo_exception)
            continue

        feed_new = 0
        for entry in parsed.entries:
            title = (entry.get("title") or "").strip()
            link = (entry.get("link") or "").strip()
            guid = (entry.get("id") or entry.get("guid") or link).strip()
            pub = entry.get("published") or entry.get("updated") or ""
            # Creator field = docket number (e.g. "LR-26557", "34-105550")
            docket = (entry.get("dc_creator") or entry.get("author") or "").strip()

            if not title or not link:
                continue

            sid = _seen_id(guid)
            if _is_seen(conn, sid):
                continue

            # Build informative summary
            if source == "SEC/TradingSuspension":
                summary = (
                    f"⚠️ SEC TRADING SUSPENSION: {title}. "
                    f"Docket: {docket}. Trading halted for up to 10 days per Section 12(k) of the Exchange Act. "
                    f"Link: {link.strip()}"
                )
            elif source == "SEC/LitigationRelease":
                summary = (
                    f"SEC Litigation Release ({docket}): {title}. "
                    f"Federal court action filed. See: {link.strip()}"
                )
            else:
                summary = (
                    f"SEC Administrative Proceeding ({docket}): {title}. "
                    f"See: {link.strip()}"
                )

            results.append({
                "title": f"[{label}] {title}" + (f" ({docket})" if docket else ""),
                "link": link.strip(),
                "summary": summary,
                "published": pub,
                "source": source,
            })
            _mark_seen(conn, sid, link, title, source)
            feed_new += 1

        log.info("[sec_enforcement] %s: %d new", source, feed_new)

    conn.commit()
    conn.close()
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    articles = collect_sec_enforcement()
    if not articles:
        print("No new SEC enforcement actions (all already seen or feeds empty).")
        sys.exit(0)

    print(f"\nFetched {len(articles)} new SEC enforcement actions:\n")
    for a in articles:
        print(f"  [{a['source']}] {a['title']}")
        print(f"    {a['link'].strip()}")
        print(f"    {a['published']}")
        print()
