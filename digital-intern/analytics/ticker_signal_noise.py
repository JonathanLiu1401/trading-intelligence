"""Ticker syndication noise analyser.

For each ticker mentioned in the last ``WINDOW_HOURS`` of live articles,
counts both the raw article volume *and* the number of *unique stories*
(after Jaccard near-dedup collapse).  The difference is the syndication
noise: the same story reprinted across feeds.

This answers the question: "Is NVDA getting 10 independent signals, or
is it one press-release reprinted 9 times?"

  noise_ratio = 1 - (unique_stories / total_articles)

  * noise_ratio ~0  → genuine multi-source coverage
  * noise_ratio ~1  → mostly syndication of a single story

Tickers are extracted from article titles using the same TICKER_RE /
STOP list as ``analytics.trend_velocity`` to stay consistent.

Output: ``/home/zeph/logs/ticker_signal_noise.json``

  {
    "generated_at": "...",
    "window_hours": 6,
    "scanned": 3000,
    "tickers": [
      {
        "ticker": "NVDA",
        "total": 18,
        "unique": 4,
        "noise_ratio": 0.78,
        "top_stories": ["Nvidia posts record $81.6B ..."]
      }, ...
    ]
  }

  Tickers with fewer than MIN_ARTICLES are omitted to avoid noise from
  single-article tickers where the metric is meaningless.

Standalone::

    python3 -m analytics.ticker_signal_noise
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in BASE.parts:
    sys.path.insert(0, str(BASE))

from analytics.trend_velocity import TICKER_RE, STOP, _parse_ts  # noqa: E402
from ml.dedup import dedupe_articles  # noqa: E402
from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path  # noqa: E402

OUT_PATH = Path("/home/zeph/logs/ticker_signal_noise.json")
WINDOW_HOURS = 6
SCAN_LIMIT = 4000
MIN_ARTICLES = 3      # ignore tickers with fewer mentions
TOP_STORIES = 3       # how many unique story titles to include per ticker
JACCARD_THRESHOLD = 0.55  # slightly looser than default to catch near-syndicates


def _extract_tickers(title: str | None) -> list[str]:
    if not title:
        return []
    return [m for m in TICKER_RE.findall(title) if m not in STOP and len(m) >= 2]


def main() -> None:
    db_path = _get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row

    cutoff = f"-{WINDOW_HOURS} hours"
    rows = conn.execute(
        f"""
        SELECT id, title, source, ai_score, ml_score, first_seen
          FROM articles
         WHERE {_LIVE_ONLY_CLAUSE}
           AND first_seen >= datetime('now', ?)
         ORDER BY first_seen DESC
         LIMIT ?
        """,
        (cutoff, SCAN_LIMIT),
    ).fetchall()
    conn.close()

    # Group articles by ticker
    ticker_articles: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        tickers = _extract_tickers(row["title"])
        art = {
            "id": row["id"],
            "title": row["title"],
            "source": row["source"],
            "ai_score": row["ai_score"],
            "ml_score": row["ml_score"],
            "first_seen": row["first_seen"],
        }
        for ticker in tickers:
            ticker_articles[ticker].append(art)

    results: list[dict] = []
    for ticker, articles in ticker_articles.items():
        if len(articles) < MIN_ARTICLES:
            continue

        # Collapse near-duplicates; prefer ml_score then ai_score
        # dedupe_articles uses ai_score by default; pass ml_score when available
        unique = dedupe_articles(
            articles,
            threshold=JACCARD_THRESHOLD,
            score_key="ml_score",
        )
        total = len(articles)
        n_unique = len(unique)
        noise_ratio = round(1.0 - n_unique / total, 3) if total > 0 else 0.0

        top_stories = [a["title"] for a in unique[:TOP_STORIES] if a.get("title")]

        results.append(
            {
                "ticker": ticker,
                "total": total,
                "unique": n_unique,
                "noise_ratio": noise_ratio,
                "top_stories": top_stories,
            }
        )

    # Sort: most total coverage first, then by noise (low = better signal)
    results.sort(key=lambda r: (-r["total"], r["noise_ratio"]))

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": WINDOW_HOURS,
        "scanned": len(rows),
        "tickers": results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(snapshot, indent=2))

    # Human-readable summary
    print(
        f"ticker_signal_noise: scanned={len(rows)} "
        f"tickers={len(results)} window={WINDOW_HOURS}h"
    )
    for r in results[:8]:
        bar = "█" * int(r["noise_ratio"] * 10) + "░" * (10 - int(r["noise_ratio"] * 10))
        print(
            f"  {r['ticker']:6s} total={r['total']:3d}  unique={r['unique']:3d}  "
            f"noise={r['noise_ratio']:.0%} [{bar}]"
        )
        if r["top_stories"]:
            print(f"           → {r['top_stories'][0][:80]}")


if __name__ == "__main__":
    main()
