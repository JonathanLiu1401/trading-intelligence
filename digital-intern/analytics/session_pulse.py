"""Session pulse: 30-minute rolling ticker heat map for real-time awareness.

Scans the last WINDOW_MIN minutes of live articles and produces a ranked
ticker heat map with mention counts, average ml/ai scores, and urgency flags.
Complements trend_velocity (2h window) and breaking_news_detector (burst
detection) by answering "what is hot *right now* in the last half hour?"

Design constraints (mirrors other analytics):
  * No full-table scan — bounded LIMIT scan via idx_first_seen.
  * _LIVE_ONLY_CLAUSE applied to exclude backtest/synthetic rows.
  * Read-only connection, busy_timeout=5000.

Output: /home/zeph/logs/session_pulse.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analytics.trend_velocity import STOP, TICKER_RE, _parse_ts
from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/session_pulse.json")

WINDOW_MIN = 30
FETCH_LIMIT = 2000
TOP_N = 15


def compute() -> dict:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=5000")

    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(minutes=WINDOW_MIN)).strftime("%Y-%m-%dT%H:%M:%S")

    rows = conn.execute(
        "SELECT first_seen, title, ml_score, ai_score, urgency, source "
        "FROM articles INDEXED BY idx_first_seen "
        f"WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (cutoff, FETCH_LIMIT),
    ).fetchall()

    # Aggregate per ticker
    counts: dict[str, int] = defaultdict(int)
    ml_scores: dict[str, list[float]] = defaultdict(list)
    ai_scores: dict[str, list[float]] = defaultdict(list)
    urgency_flags: dict[str, int] = defaultdict(int)
    sources: dict[str, set] = defaultdict(set)

    total_articles = len(rows)
    for first_seen, title, ml_score, ai_score, urgency, source in rows:
        ts = _parse_ts(first_seen)
        if ts is None:
            continue
        tickers = [
            m for m in TICKER_RE.findall(title or "")
            if m not in STOP and len(m) >= 2
        ]
        for tk in set(tickers):
            counts[tk] += 1
            if ml_score is not None:
                ml_scores[tk].append(float(ml_score))
            if ai_score is not None:
                ai_scores[tk].append(float(ai_score))
            if urgency and int(urgency) >= 2:
                urgency_flags[tk] += 1
            if source:
                sources[tk].add(source)

    # Build ranked list
    ranked = []
    for tk, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        avg_ml = sum(ml_scores[tk]) / len(ml_scores[tk]) if ml_scores[tk] else None
        avg_ai = sum(ai_scores[tk]) / len(ai_scores[tk]) if ai_scores[tk] else None
        ranked.append({
            "ticker": tk,
            "mentions": cnt,
            "avg_ml_score": round(avg_ml, 2) if avg_ml is not None else None,
            "avg_ai_score": round(avg_ai, 2) if avg_ai is not None else None,
            "urgent_count": urgency_flags[tk],
            "source_count": len(sources[tk]),
        })

    top = ranked[:TOP_N]

    result = {
        "generated_at": now.isoformat(),
        "window_min": WINDOW_MIN,
        "total_articles_scanned": total_articles,
        "ticker_count": len(counts),
        "top": top,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    result = compute()
    print(
        f"session_pulse [{result['window_min']}min]: "
        f"{result['total_articles_scanned']} articles, "
        f"{result['ticker_count']} tickers"
    )
    for entry in result["top"][:10]:
        urg = f" ⚠{entry['urgent_count']}" if entry["urgent_count"] else ""
        ml = f"ml={entry['avg_ml_score']:.1f}" if entry["avg_ml_score"] is not None else "ml=n/a"
        print(
            f"  {entry['ticker']:6s}  x{entry['mentions']:3d}  {ml}  "
            f"srcs={entry['source_count']}{urg}"
        )


if __name__ == "__main__":
    main()
