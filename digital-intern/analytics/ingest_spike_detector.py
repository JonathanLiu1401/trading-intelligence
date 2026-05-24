"""Ingest spike detector.

Alerts when total article ingestion rate in the past 5 minutes exceeds
3× the rolling 60-minute average. A broad volume spike (across all sources)
often precedes or coincides with a breaking macro event, circuit-breaker
halt, or data-feed anomaly before any single-ticker detector fires.

This is NOT the same as breaking_news_detector (same ticker, 5 min window).
This fires on TOTAL pipeline volume — the fire-hose reading that says
"something big is happening, go look".

Design constraints:
  * Single bounded idx_first_seen scan, no full-table COUNT.
  * Read-only sqlite URI, busy_timeout=12000 ms (USB-backed DB).
  * _LIVE_ONLY_CLAUSE excludes backtest rows.

Output: /home/zeph/logs/ingest_spike.json
Standalone: python3 -m analytics.ingest_spike_detector
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/ingest_spike.json")

SCAN_LIMIT = 3000          # ~1h of rows at typical ingest rate
SPIKE_WINDOW_MIN = 5       # "current" burst window
BASELINE_WINDOW_MIN = 60   # rolling baseline window
SPIKE_MULTIPLIER = 3.0     # fire when burst_rate > N × baseline_rate
MIN_BASELINE_ARTICLES = 10 # need at least this many in the hour to avoid
                           # false positives at pipeline startup


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.strptime(s[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def run() -> dict:
    now = datetime.now(timezone.utc)
    spike_cutoff = now - timedelta(minutes=SPIKE_WINDOW_MIN)
    baseline_cutoff = now - timedelta(minutes=BASELINE_WINDOW_MIN)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=12)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT first_seen, source FROM articles "
            f"WHERE {_LIVE_ONLY_CLAUSE} "
            f"ORDER BY first_seen DESC LIMIT {SCAN_LIMIT}"
        ).fetchall()
    finally:
        conn.close()

    baseline_count = 0
    spike_count = 0
    spike_sources: dict[str, int] = {}

    for row in rows:
        ts = _parse_ts(row["first_seen"])
        if ts is None or ts < baseline_cutoff:
            break
        baseline_count += 1
        if ts >= spike_cutoff:
            spike_count += 1
            src = row["source"] or "unknown"
            spike_sources[src] = spike_sources.get(src, 0) + 1

    # articles per minute
    baseline_rate = baseline_count / BASELINE_WINDOW_MIN
    spike_rate = spike_count / SPIKE_WINDOW_MIN

    ratio = (spike_rate / baseline_rate) if baseline_rate > 0 else 0.0

    is_spike = (
        ratio >= SPIKE_MULTIPLIER
        and baseline_count >= MIN_BASELINE_ARTICLES
        and spike_count >= 3
    )

    top_sources = sorted(spike_sources.items(), key=lambda x: -x[1])[:5]

    result = {
        "generated_at": now.isoformat(),
        "spike_window_min": SPIKE_WINDOW_MIN,
        "baseline_window_min": BASELINE_WINDOW_MIN,
        "spike_count": spike_count,
        "baseline_count": baseline_count,
        "spike_rate_per_min": round(spike_rate, 2),
        "baseline_rate_per_min": round(baseline_rate, 2),
        "ratio": round(ratio, 2),
        "is_spike": is_spike,
        "threshold_multiplier": SPIKE_MULTIPLIER,
        "top_spike_sources": [{"source": s, "count": n} for s, n in top_sources],
        "status": "SPIKE" if is_spike else "NORMAL",
    }

    OUT_PATH.write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    r = run()
    status = r["status"]
    print(
        f"[ingest_spike] {status} | "
        f"5-min rate: {r['spike_rate_per_min']:.1f}/min ({r['spike_count']} articles) | "
        f"60-min baseline: {r['baseline_rate_per_min']:.1f}/min ({r['baseline_count']} articles) | "
        f"ratio: {r['ratio']:.2f}x"
    )
    if r["is_spike"]:
        top = ", ".join(f"{s['source']}({s['count']})" for s in r["top_spike_sources"])
        print(f"  TOP SOURCES: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
