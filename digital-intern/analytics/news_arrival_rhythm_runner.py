"""Operational runner for news_arrival_rhythm — writes hourly JSON output.

Wires the pure-library ``analytics.news_arrival_rhythm.build_news_arrival_rhythm``
to the live DB and writes ``/home/zeph/logs/news_arrival_rhythm.json`` so the
per-source hour-of-day urgent-article heatmap is queryable like the other log
snapshots (collection_quality, trend_velocity, etc.).

Standalone: ``python3 -m analytics.news_arrival_rhythm_runner``
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analytics.news_arrival_rhythm import build_news_arrival_rhythm
from storage.article_store import _LIVE_ONLY_CLAUSE

DB = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT = Path("/home/zeph/logs/news_arrival_rhythm.json")
WINDOW_HOURS = 24
MIN_URGENCY = 1
FETCH_LIMIT = 5000


def _fetch_articles(conn: sqlite3.Connection) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)).isoformat(timespec="seconds")
    rows = conn.execute(
        f"""SELECT source, urgency, first_seen
              FROM articles
             WHERE {_LIVE_ONLY_CLAUSE}
               AND urgency >= ?
               AND first_seen >= ?
             ORDER BY first_seen DESC
             LIMIT ?""",
        (MIN_URGENCY, cutoff, FETCH_LIMIT),
    ).fetchall()
    return [{"source": r[0], "urgency": r[1], "first_seen": r[2]} for r in rows]


def main() -> int:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    articles = _fetch_articles(conn)
    conn.close()

    report = build_news_arrival_rhythm(articles, hours=WINDOW_HOURS, min_urgency=MIN_URGENCY)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))

    n_arts = sum(s["total"] for s in report.get("sources", []))
    n_sources = len(report.get("sources", []))
    qw = report.get("quiet_window", {})
    qw_len = qw.get("length_hours", 0)

    print(f"news_arrival_rhythm: {n_arts} urgent articles across {n_sources} sources (24h)")
    print(f"  global quiet window: {qw_len}h starting UTC hour {qw.get('start_hour', 'n/a')}")

    totals: list[tuple[int, int]] = list(enumerate(report.get("hour_of_day_totals", [])))
    if totals:
        peak_h, peak_n = max(totals, key=lambda x: x[1])
        print(f"  peak hour UTC {peak_h:02d}:00 — {peak_n} urgent articles")

    print("  top sources:")
    for src in report.get("sources", [])[:5]:
        peak = src.get("peak_hour")
        print(
            f"    {src['source'][:40]:<40}  total={src['total']:>4}"
            + (f"  peak=UTC{peak:02d}" if peak is not None else "")
        )

    print(f"  written → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
