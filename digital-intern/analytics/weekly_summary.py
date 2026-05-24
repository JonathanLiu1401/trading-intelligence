"""Weekly summary: 7-day rollup of article volume, scores, sources, tickers."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analytics.trend_velocity import _parse_ts, extract_tickers
from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/weekly_summary.json")
WINDOW_DAYS = 7
SCAN_LIMIT = 50_000
TOP_TICKERS_N = 10
TOP_SOURCES_N = 5


def fetch_recent(conn: sqlite3.Connection, limit: int) -> list[tuple]:
    # Bounded idx_first_seen scan — never COUNT(*) the full 1.4GB USB-backed
    # articles table (see CLAUDE.md §intern DB count timeout). _LIVE_ONLY_CLAUSE
    # filters backtest replays and Opus annotation rows from the rollup.
    cur = conn.execute(
        "SELECT first_seen, title, source, ai_score, urgency FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (limit,),
    )
    return cur.fetchall()


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    rows = fetch_recent(conn, SCAN_LIMIT)
    if not rows:
        print("weekly_summary: no rows", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=WINDOW_DAYS)

    total = 0
    daily_counts: Counter[str] = Counter()
    ticker_counts: Counter[str] = Counter()
    source_family_counts: Counter[str] = Counter()
    ai_score_sum = 0.0
    ai_score_n = 0
    urgent = 0
    daily_score_sum: dict[str, float] = defaultdict(float)
    daily_score_n: dict[str, int] = defaultdict(int)

    for first_seen, title, source, ai_score, urgency in rows:
        ts = _parse_ts(first_seen)
        if ts is None or ts < cutoff:
            continue
        total += 1
        date_key = ts.date().isoformat()
        daily_counts[date_key] += 1

        for tk in extract_tickers(title):
            ticker_counts[tk] += 1

        if source:
            family = source.split("/")[0]
            if family:
                source_family_counts[family] += 1

        try:
            score = float(ai_score) if ai_score is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0
        if score > 0:
            ai_score_sum += score
            ai_score_n += 1
            daily_score_sum[date_key] += score
            daily_score_n[date_key] += 1

        try:
            u = int(urgency) if urgency is not None else 0
        except (TypeError, ValueError):
            u = 0
        if u >= 2:
            urgent += 1

    if total == 0:
        print("weekly_summary: no rows in 7d window", file=sys.stderr)
        return 1

    peak_day = max(daily_counts.items(), key=lambda kv: kv[1])[0]
    trough_day = min(daily_counts.items(), key=lambda kv: kv[1])[0]
    avg_ai_score = round(ai_score_sum / ai_score_n, 3) if ai_score_n else 0.0

    score_trend = [
        {
            "date": d,
            "avg_ai_score": round(daily_score_sum[d] / daily_score_n[d], 3),
            "count": daily_score_n[d],
        }
        for d in sorted(daily_score_n.keys())
        if daily_score_n[d] > 0
    ]

    top_tickers = ticker_counts.most_common(TOP_TICKERS_N)
    top_sources = source_family_counts.most_common(TOP_SOURCES_N)

    payload = {
        "generated_at": now.isoformat(),
        "total_articles": total,
        "daily_counts": dict(sorted(daily_counts.items())),
        "peak_day": peak_day,
        "trough_day": trough_day,
        "top_tickers": [[tk, c] for tk, c in top_tickers],
        "avg_ai_score": avg_ai_score,
        "urgent_count": urgent,
        "top_sources": [[sf, c] for sf, c in top_sources],
        "score_trend": score_trend,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    peak_n = daily_counts[peak_day]
    print(
        f"weekly_summary: {total} articles over 7 days | {urgent} urgent | "
        f"peak: {peak_day} ({peak_n})"
    )
    if top_tickers:
        tick_str = ", ".join(f"{tk} ({c})" for tk, c in top_tickers[:5])
    else:
        tick_str = "(none)"
    print(f"top tickers: {tick_str}")
    if top_sources:
        top_src_name, top_src_n = top_sources[0]
    else:
        top_src_name, top_src_n = "(none)", 0
    print(
        f"avg ai_score: {avg_ai_score} | top source: {top_src_name} "
        f"({top_src_n} articles)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
