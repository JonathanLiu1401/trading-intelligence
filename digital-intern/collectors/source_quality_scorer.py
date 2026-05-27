"""Source quality scorer — ranks data sources by urgency rate and signal density.

Unlike source_health (which tracks consecutive zero-article failures), this
collector measures *quality*: what fraction of each source's articles score
high kw_score (≥5.0) and what the average scores are over a rolling 24h window.

Output: one synthetic article per run with a ranked table, source="source_quality_report".
Deduped by UTC-date so it fires once per day unless forced.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("source_quality_scorer")

BASE_DIR = Path(__file__).resolve().parent.parent
ARTICLES_DB = BASE_DIR / "data" / "articles.db"
SEEN_DB = BASE_DIR / "data" / "seen_articles.db"

SOURCE_NAME = "source_quality_report"
WINDOW_HOURS = 24
MIN_ARTICLES = 10  # ignore sources with too few samples
HIGH_SCORE_THRESHOLD = 5.0


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


def _load_source_quality() -> list[dict]:
    """Return per-source quality stats for the last WINDOW_HOURS."""
    if not ARTICLES_DB.exists():
        return []
    try:
        conn = sqlite3.connect(str(ARTICLES_DB), timeout=20, check_same_thread=False)
        conn.execute("PRAGMA query_only=ON")
        rows = conn.execute(
            f"""
            SELECT
                source,
                COUNT(*)                                                        AS total,
                ROUND(AVG(COALESCE(kw_score, 0)), 2)                           AS avg_kw,
                ROUND(AVG(COALESCE(ai_score, 0)), 2)                           AS avg_ai,
                ROUND(
                    100.0 * SUM(CASE WHEN kw_score >= {HIGH_SCORE_THRESHOLD}
                                     THEN 1 ELSE 0 END) / COUNT(*),
                    1
                )                                                               AS high_pct,
                SUM(CASE WHEN urgency=1 THEN 1 ELSE 0 END)                     AS urgent_cnt
            FROM articles
            WHERE first_seen >= datetime('now', '-{WINDOW_HOURS} hours')
              AND source NOT LIKE 'backtest%'
              AND source NOT LIKE 'source_quality%'
            GROUP BY source
            HAVING total >= {MIN_ARTICLES}
            ORDER BY high_pct DESC, avg_kw DESC
            """
        ).fetchall()
        conn.close()
        return [
            {
                "source": r[0],
                "total": r[1],
                "avg_kw": r[2] or 0.0,
                "avg_ai": r[3] or 0.0,
                "high_pct": r[4] or 0.0,
                "urgent_cnt": r[5] or 0,
            }
            for r in rows
        ]
    except Exception as e:
        log.error("source_quality_scorer: DB query failed: %s", e)
        return []


def _already_seen(seen_conn: sqlite3.Connection, article_id: str) -> bool:
    row = seen_conn.execute(
        "SELECT 1 FROM seen_articles WHERE id=?", (article_id,)
    ).fetchone()
    return row is not None


def _mark_seen(seen_conn: sqlite3.Connection, article_id: str, link: str, title: str) -> None:
    seen_conn.execute(
        "INSERT OR IGNORE INTO seen_articles(id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
        (article_id, link, title, SOURCE_NAME, datetime.now(timezone.utc).isoformat()),
    )
    seen_conn.commit()


def _write_article(title: str, summary: str, link: str) -> bool:
    """Insert the quality report into articles.db.

    ``full_text`` is declared as BLOB and the rest of the pipeline expects
    a zlib-compressed payload (see ``storage.article_store.compress`` /
    ``decompress``). Writing the raw ``summary`` string here leaves a
    TEXT-affinity value behind a BLOB column (SQLite is dynamically
    typed) — and the next call to ``store.get_unscored`` then crashes the
    *entire* scorer batch with ``a bytes-like object is required, not
    'str'`` because ``decompress`` is wired for bytes. Live regression
    (2026-05-27): one of these str-typed rows blocked the scorer 300+×
    /day. Compressing here matches every other ingestion path."""
    try:
        conn = sqlite3.connect(str(ARTICLES_DB), timeout=20, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        now = datetime.now(timezone.utc).isoformat()
        blob = zlib.compress(
            (summary or "").encode("utf-8", errors="replace"), level=6,
        ) if summary else None
        conn.execute(
            """INSERT OR IGNORE INTO articles
               (id, url, title, source, published, kw_score, full_text, first_seen)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                _article_id(link),
                link,
                title,
                SOURCE_NAME,
                now,
                0.0,
                blob,
                now,
            ),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        log.error("source_quality_scorer: write failed: %s", e)
        return False


def collect() -> int:
    """Generate a source quality report. Returns 1 if a new report was emitted."""
    stats = _load_source_quality()
    if not stats:
        log.info("source_quality_scorer: no data or DB unavailable")
        return 0

    utc_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dedup_key = f"source_quality_report:{utc_date}"
    article_id = _article_id(dedup_key)
    link = f"internal://source_quality_report/{utc_date}"

    seen_conn = sqlite3.connect(str(SEEN_DB), timeout=20, check_same_thread=False)
    _ensure_seen_db(seen_conn)

    if _already_seen(seen_conn, article_id):
        log.debug("source_quality_scorer: already reported for %s", utc_date)
        seen_conn.close()
        return 0

    top = stats[:20]
    lines = [f"Source Quality Report — {utc_date} ({WINDOW_HOURS}h window)\n"]
    lines.append(f"{'Source':<35} {'Articles':>8} {'AvgKW':>6} {'High%':>6} {'Urgent':>7}")
    lines.append("-" * 65)
    for s in top:
        lines.append(
            f"{s['source'][:35]:<35} {s['total']:>8} {s['avg_kw']:>6.2f} "
            f"{s['high_pct']:>5.1f}% {s['urgent_cnt']:>7}"
        )

    # Bottom 5 (noisy/low-quality)
    if len(stats) > 20:
        bottom = stats[-5:]
        lines.append("\nLowest quality sources:")
        for s in bottom:
            lines.append(
                f"  {s['source'][:35]:<35} {s['total']:>8} {s['avg_kw']:>6.2f} "
                f"{s['high_pct']:>5.1f}%"
            )

    summary = "\n".join(lines)
    title = (
        f"[Source Quality] Top: {top[0]['source'][:25]} ({top[0]['high_pct']}% high-score) "
        f"| {len(stats)} sources analyzed"
        if top else f"[Source Quality] {len(stats)} sources analyzed"
    )

    wrote = _write_article(title, summary, link)
    if wrote:
        _mark_seen(seen_conn, article_id, link, title)
        log.info("source_quality_scorer: emitted report — %d sources", len(stats))
        print(f"[source_quality_scorer] Report emitted: {len(stats)} sources analyzed")
        print(summary[:1500])

    seen_conn.close()
    return 1 if wrote else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect()
