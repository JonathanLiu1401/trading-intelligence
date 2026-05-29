"""Premium source spotlight: institutional-quality news ranked by ML signal.

Filters the last WINDOW_HOURS of live articles to premium institutional
sources (Bloomberg, Reuters, CNBC, Financial Times, MarketWatch, Barron's)
and surfaces the highest-scoring tickers mentioned across those sources.

Distinct from existing tools:
  * ``composite_signal_strength`` — uses all sources, 2h window; no source
    authority filter.
  * ``source_quality``           — per-source score averages, not per-ticker.
  * ``source_credibility_audit`` — tracks defaulting tags, not signal quality.
  * ``sector_pulse``             — sector-level, not ticker-level with source
    authority dimension.

Operational value:
  * Answers "what are Bloomberg/Reuters/CNBC reporting right now?" without
    needing to read individual articles.
  * Premium-source coverage of a ticker is a stronger standalone signal than
    general coverage — one Bloomberg article beats ten StockTwits posts.
  * The ``first_premium_seen`` field shows who broke the story first among
    premium sources.

Design constraints:
  * Bounded SCAN_LIMIT idx_first_seen scan — no full-table scan.
  * Read-only sqlite URI, busy_timeout 15 000 ms.
  * _LIVE_ONLY_CLAUSE applied — backtest rows excluded.
  * Graceful on empty window (overnight, weekend, market closed).

Output: /home/zeph/logs/premium_source_spotlight.json

Standalone::

    python3 -m analytics.premium_source_spotlight
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from analytics.trend_velocity import STOP, TICKER_RE, _parse_ts, extract_tickers
from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT = Path("/home/zeph/logs/premium_source_spotlight.json")

WINDOW_HOURS = 3
SCAN_LIMIT = 5_000
TOP_N = 10
MIN_SCORE = 2.0   # only tickers with avg premium ml_score above this

# Source name fragments that indicate institutional/premium publishers.
# Matched case-insensitively against the source field.
# Additional stop tokens specific to premium source names leaking into titles
_EXTRA_STOP = frozenset({"CNBC", "BBC", "FT", "WSJ", "AP", "FOX", "CNN", "NBC", "ABC", "CBS"})

PREMIUM_FRAGMENTS: tuple[str, ...] = (
    "bloomberg",
    "reuters",
    "cnbc",
    "financial times",
    "marketwatch",
    "barron",
    "wsj",
    "wall street journal",
    "ft.com",
    "ft ",
)


def _is_premium(source: str) -> bool:
    s = source.lower()
    return any(frag in s for frag in PREMIUM_FRAGMENTS)


def compute() -> dict:
    now = datetime.now(timezone.utc)
    db_path = str(_get_db_path())
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=15000")

    rows = conn.execute(
        f"SELECT first_seen, source, title, ml_score "
        f"FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        f"ORDER BY first_seen DESC LIMIT ?",
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    # Filter to the time window and premium sources only
    cutoff = now.timestamp() - WINDOW_HOURS * 3600

    # Per-ticker aggregation: {ticker -> {sources, scores, first_seen, titles}}
    ticker_data: dict[str, dict] = {}

    for first_seen_raw, source, title, ml_score in rows:
        ts = _parse_ts(first_seen_raw)
        if ts is None or ts.timestamp() < cutoff:
            continue
        if not _is_premium(source):
            continue
        if ml_score is None:
            ml_score = 0.0

        tickers = [t for t in extract_tickers(title) if t not in _EXTRA_STOP]
        for ticker in tickers:
            if ticker not in ticker_data:
                ticker_data[ticker] = {
                    "sources": set(),
                    "scores": [],
                    "first_seen": ts,
                    "first_source": source,
                    "sample_title": title,
                    "article_count": 0,
                }
            td = ticker_data[ticker]
            td["sources"].add(source)
            td["scores"].append(ml_score)
            td["article_count"] += 1
            if ts < td["first_seen"]:
                td["first_seen"] = ts
                td["first_source"] = source

    # Build ranked output
    ranked = []
    for ticker, td in ticker_data.items():
        avg_score = mean(td["scores"]) if td["scores"] else 0.0
        if avg_score < MIN_SCORE:
            continue
        ranked.append({
            "ticker": ticker,
            "avg_ml_score": round(avg_score, 3),
            "article_count": td["article_count"],
            "distinct_sources": len(td["sources"]),
            "sources": sorted(td["sources"]),
            "first_premium_seen": td["first_seen"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "first_source": td["first_source"],
            "sample_title": td["sample_title"],
        })

    ranked.sort(key=lambda x: (x["avg_ml_score"], x["distinct_sources"]), reverse=True)
    top = ranked[:TOP_N]

    out = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_hours": WINDOW_HOURS,
        "scan_limit": SCAN_LIMIT,
        "premium_sources_matched": len({
            r["first_source"] for r in top
        }),
        "tickers_found": len(ranked),
        "top": top,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    result = compute()
    n = result["tickers_found"]
    print(
        f"premium_source_spotlight: {n} tickers in last {result['window_hours']}h "
        f"| scan_limit={result['scan_limit']}"
    )
    for entry in result["top"][:5]:
        print(
            f"  {entry['ticker']:6s} score={entry['avg_ml_score']:.2f} "
            f"arts={entry['article_count']} "
            f"sources={entry['distinct_sources']} "
            f"first={entry['first_source']!r:.40s}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
