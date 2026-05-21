"""Collector rate monitor — detects when high-volume sources go silent.

Queries articles.db to compare each source's last-3h throughput against
its 7-day hourly baseline. When a source with ≥50 avg articles/day has had
zero articles for 3+ consecutive hours it emits a synthetic alert article
so the failure surfaces in briefings and urgency scoring.

Excluded sources:
  - gdelt_*: irregular batch cadence, not hour-by-hour
  - SEC-EDGAR/8-K: filing volume is non-uniform (market-hours only)
  - backtest*: synthetic rows

Dedup: one alert per (source, UTC-date) via seen_articles.db so the
same source doesn't spam on every daemon pass.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("collector_rate_monitor")

BASE_DIR = Path(__file__).resolve().parent.parent
ARTICLES_DB = BASE_DIR / "data" / "articles.db"
SEEN_DB = BASE_DIR / "data" / "seen_articles.db"

SOURCE_NAME = "collector_monitor"

# Minimum daily average (articles/day, 7-day window) to be worth monitoring.
MIN_DAILY_AVG = 50.0
# How many silent hours trigger an alert.
SILENT_HOURS = 3
# Sources with highly irregular cadence that produce too many false positives.
EXCLUDED_PREFIXES = ("gdelt", "backtest", "SEC-EDGAR")


def _article_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _ensure_seen_db(conn: sqlite3.Connection) -> None:
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


def _load_source_stats() -> list[tuple[str, float, int]]:
    """Return (source, daily_avg_7d, count_last_Nh) for monitored sources."""
    if not ARTICLES_DB.exists():
        return []
    try:
        conn = sqlite3.connect(str(ARTICLES_DB), timeout=20, check_same_thread=False)
        conn.execute("PRAGMA query_only=ON")
        rows = conn.execute(
            f"""
            SELECT
                source,
                COUNT(*) / 7.0                                                AS daily_avg,
                SUM(CASE WHEN first_seen >= datetime('now', '-{SILENT_HOURS} hours')
                         THEN 1 ELSE 0 END)                                   AS cnt_window
            FROM articles
            WHERE first_seen >= datetime('now', '-7 days')
              AND source NOT LIKE 'backtest%'
            GROUP BY source
            HAVING daily_avg >= {MIN_DAILY_AVG}
            ORDER BY daily_avg DESC
            """
        ).fetchall()
        conn.close()
        return [(r[0], float(r[1]), int(r[2])) for r in rows]
    except Exception as exc:
        log.warning("[collector_monitor] stats query failed: %s", exc)
        return []


def collect_rate_alerts() -> list[dict]:
    """Return synthetic alert dicts for sources that have gone silent."""
    stats = _load_source_stats()
    if not stats:
        return []

    now_utc = datetime.now(timezone.utc)
    today = now_utc.strftime("%Y-%m-%d")
    now_str = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    alerts: list[dict] = []

    seen_conn = sqlite3.connect(str(SEEN_DB), timeout=30, check_same_thread=False)
    _ensure_seen_db(seen_conn)

    try:
        for source, daily_avg, cnt_window in stats:
            # Skip excluded prefixes.
            if any(source.lower().startswith(p.lower()) for p in EXCLUDED_PREFIXES):
                continue
            # Only alert when truly silent in the window.
            if cnt_window > 0:
                continue

            dedup_key = f"collector_monitor|silent|{source}|{today}"
            art_id = _article_id(dedup_key)

            if seen_conn.execute(
                "SELECT 1 FROM seen_articles WHERE id=?", (art_id,)
            ).fetchone():
                continue  # already alerted today

            title = (
                f"⚠️ COLLECTOR SILENT: [{source}] — "
                f"0 articles in {SILENT_HOURS}h (avg {daily_avg:.0f}/day)"
            )
            summary = (
                f"Source '{source}' has produced 0 articles in the last "
                f"{SILENT_HOURS} hours. Its 7-day baseline is "
                f"{daily_avg:.1f} articles/day. Possible collector failure "
                f"or upstream outage."
            )
            link = f"internal://collector_monitor/{source.replace('/', '_')}/{today}"

            seen_conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                "VALUES (?,?,?,?,?)",
                (art_id, link, title, SOURCE_NAME, now_str),
            )
            alerts.append({
                "id": art_id,
                "link": link,
                "title": title,
                "summary": summary,
                "source": SOURCE_NAME,
                "first_seen": now_str,
                "silent_source": source,
                "daily_avg": daily_avg,
            })
            log.warning("[collector_monitor] SILENT SOURCE: %s (avg %.0f/day)", source, daily_avg)

        seen_conn.commit()
    finally:
        seen_conn.close()

    return alerts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = collect_rate_alerts()
    if results:
        print(f"\nFound {len(results)} silent source alert(s):\n")
        for a in results:
            print(f"  {a['title']}")
    else:
        print("All monitored sources are active (no silent sources detected).")
