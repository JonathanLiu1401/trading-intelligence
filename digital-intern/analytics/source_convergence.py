"""Multi-source convergence detector.

Flags tickers where 3+ *distinct* source domains publish articles within a
2-hour rolling window.  Article volume can be inflated by a single feed
posting multiple items; source *diversity* is a stronger editorial signal —
when Reuters, Bloomberg, and Seeking Alpha all cover the same ticker within
2h, that's independently sourced convergence, not spam.

Output: /home/zeph/logs/source_convergence.json
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analytics.trend_velocity import STOP, TICKER_RE, _parse_ts, extract_tickers
from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/source_convergence.json")

WINDOW_HOURS = 2
FETCH_LIMIT = 8000
MIN_DISTINCT_SOURCES = 3   # need at least this many distinct source domains
MIN_ARTICLES = 2           # per-ticker minimum before we even score it


def _source_domain(raw: str) -> str:
    """Normalise source to a root domain / feed name."""
    if not raw:
        return "unknown"
    # strip sub-paths: "yfinance/Zacks" -> "yfinance"
    base = raw.split("/")[0].strip().lower()
    # strip common prefixes
    for prefix in ("www.", "feeds.", "rss."):
        if base.startswith(prefix):
            base = base[len(prefix):]
    return base or raw.lower()


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")

    rows = conn.execute(
        "SELECT first_seen, title, source, ai_score FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (FETCH_LIMIT,),
    ).fetchall()
    conn.close()

    if not rows:
        print("source_convergence: no articles found", flush=True)
        return 0

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=WINDOW_HOURS)

    # ticker -> list of (ts, source_domain, ai_score, title)
    by_ticker: dict[str, list[tuple[datetime, str, float, str]]] = defaultdict(list)

    for first_seen, title, source, ai_score in rows:
        ts = _parse_ts(first_seen)
        if ts is None or ts < cutoff:
            continue
        domain = _source_domain(source or "")
        score = float(ai_score) if ai_score is not None else 0.0
        for tk in set(extract_tickers(title or "")):
            by_ticker[tk].append((ts, domain, score, title or ""))

    events: list[dict] = []
    for ticker, items in by_ticker.items():
        if len(items) < MIN_ARTICLES:
            continue
        domains = {it[1] for it in items}
        if len(domains) < MIN_DISTINCT_SOURCES:
            continue
        avg_ai = sum(it[2] for it in items) / len(items)
        items_sorted = sorted(items, key=lambda x: x[0])
        first_ts = items_sorted[0][0]
        last_ts = items_sorted[-1][0]
        span_min = (last_ts - first_ts).total_seconds() / 60

        events.append({
            "ticker": ticker,
            "distinct_sources": len(domains),
            "source_list": sorted(domains),
            "article_count": len(items),
            "avg_ai_score": round(avg_ai, 3),
            "span_minutes": round(span_min, 1),
            "first_seen": first_ts.isoformat(),
            "last_seen": last_ts.isoformat(),
            "top_title": items_sorted[-1][3][:120],
        })

    events.sort(key=lambda e: (-e["distinct_sources"], -e["avg_ai_score"]))

    payload = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "min_distinct_sources": MIN_DISTINCT_SOURCES,
        "total_tickers_seen": len(by_ticker),
        "convergence_count": len(events),
        "events": events,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    if events:
        print(f"source_convergence: {len(events)} tickers with multi-source convergence", flush=True)
        for e in events[:5]:
            srcs = ", ".join(e["source_list"][:4])
            print(
                f"  {e['ticker']:6s}  {e['distinct_sources']} sources [{srcs}]  "
                f"{e['article_count']} arts  ai={e['avg_ai_score']:.2f}  "
                f"span={e['span_minutes']:.0f}min",
                flush=True,
            )
    else:
        print(
            f"source_convergence: 0 convergence events "
            f"({len(by_ticker)} tickers checked, window={WINDOW_HOURS}h)",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
