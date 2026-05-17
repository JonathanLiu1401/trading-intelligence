#!/usr/bin/env python3
"""Historical news backfill — populates articles.db for every trading day in the backtest range.

Runs as a long-lived background job. Fetches one trading day at a time from GDELT, writes
each article into digital-intern/data/articles.db with the correct `published` date, then
moves to the next date. Already-cached GDELT responses and already-inserted articles are
skipped so the job is fully resumable.

Usage:
    cd /home/zeph/paper-trader
    python3 backfill_news.py                         # fill default window (May 2025–May 2026)
    python3 backfill_news.py --from 2025-05-01       # override start
    python3 backfill_news.py --status                # print coverage stats and exit

Rate: ~5.5s per GDELT request × 20 keyword groups × 260 trading days = ~8 hours.
The job can be interrupted and restarted at any time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sqlite3
import sys
import time
import zlib
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_trader.backtest import (
    BacktestEngine,
    KEYWORD_GROUPS,
    GDELT_CACHE,
    GDELT_RATE_LIMIT_S,
    LOCAL_ARTICLES_DB,
)

# Default backfill window when --from/--to are not supplied. Matches the previous
# hardcoded BacktestEngine window so existing operator muscle memory keeps working;
# override via CLI for variable-window backfills.
DEFAULT_BACKFILL_START = date(2025, 5, 1)
DEFAULT_BACKFILL_END = date(2026, 5, 13)

_STOP = False


def _handle_sig(_signum, _frame) -> None:
    global _STOP
    _STOP = True
    print("\n[backfill] signal received — finishing current date then stopping")


signal.signal(signal.SIGINT, _handle_sig)
signal.signal(signal.SIGTERM, _handle_sig)


def _compress(text: str) -> bytes:
    return zlib.compress(text.encode("utf-8", errors="replace"), level=6)


def _article_id(url: str, title: str) -> str:
    return hashlib.sha256(f"{url}||{title}".encode()).hexdigest()[:20]


def _coverage_stats(conn: sqlite3.Connection, start: date, end: date) -> dict:
    rows = conn.execute(
        """SELECT substr(published,1,10) as day, count(*) as n
           FROM articles
           WHERE published >= ? AND published <= ?
             AND url NOT LIKE 'backtest://%'
             AND source NOT LIKE 'backtest_%'
             AND source NOT LIKE 'opus_annotation%'
           GROUP BY day ORDER BY day""",
        (start.isoformat(), end.isoformat()),
    ).fetchall()
    days_covered = len(rows)
    total_articles = sum(r[1] for r in rows)
    return {"days": days_covered, "articles": total_articles,
            "avg_per_day": total_articles / days_covered if days_covered else 0}


def _gdelt_cached_articles(d: date) -> list[dict]:
    """Collect all GDELT cached articles for date d across all keyword groups."""
    articles: list[dict] = []
    seen_urls: set[str] = set()
    for kw in KEYWORD_GROUPS:
        slug = hashlib.md5(kw.encode()).hexdigest()[:8]
        path = GDELT_CACHE / f"{d.isoformat()}_{slug}.json"
        if path.exists():
            try:
                for a in json.loads(path.read_text()):
                    url = a.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        articles.append(a)
            except Exception:
                pass
    return articles


def _insert_articles(conn: sqlite3.Connection, articles: list[dict],
                     pub_date: str) -> int:
    inserted = 0
    now = date.today().isoformat()
    for a in articles:
        url = a.get("url") or ""
        title = a.get("title") or ""
        if not url or not title:
            continue
        source = a.get("source") or a.get("domain") or "gdelt_backfill"
        aid = _article_id(url, title)
        full_text = f"{title}. {source}"
        try:
            conn.execute(
                "INSERT OR IGNORE INTO articles "
                "(id,url,title,source,published,kw_score,ai_score,urgency,first_seen,full_text) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (aid, url, title, f"gdelt_{pub_date[:7]}", pub_date,
                 1.0, 1.0, 0, now, _compress(full_text)),
            )
            if conn.execute("SELECT changes()").fetchone()[0]:
                inserted += 1
        except Exception:
            pass
    return inserted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_date", default=None)
    ap.add_argument("--to", dest="to_date", default=None)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    start = date.fromisoformat(args.from_date) if args.from_date else DEFAULT_BACKFILL_START
    end = date.fromisoformat(args.to_date) if args.to_date else DEFAULT_BACKFILL_END

    if not LOCAL_ARTICLES_DB.exists():
        print(f"[backfill] articles DB not found at {LOCAL_ARTICLES_DB}")
        sys.exit(1)

    conn = sqlite3.connect(str(LOCAL_ARTICLES_DB), timeout=30)
    conn.row_factory = sqlite3.Row

    if args.status:
        stats = _coverage_stats(conn, start, end)
        print(f"[backfill] coverage {start} → {end}:")
        print(f"  days with articles : {stats['days']}")
        print(f"  total articles     : {stats['articles']}")
        print(f"  avg per day        : {stats['avg_per_day']:.1f}")
        conn.close()
        return

    # Need BacktestEngine for trading_days list and GDELTFetcher
    print("[backfill] initialising engine (PriceCache + GDELT)…")
    engine = BacktestEngine(start=start, end=end)
    trading_days = [d for d in engine.prices.trading_days if start <= d <= end]
    print(f"[backfill] {len(trading_days)} trading days in {start} → {end}")

    # Show current coverage before starting
    stats = _coverage_stats(conn, start, end)
    print(f"[backfill] current coverage: {stats['days']} days / "
          f"{stats['articles']} articles / {stats['avg_per_day']:.1f} avg/day")

    total_inserted = 0
    t0 = time.time()

    for idx, d in enumerate(trading_days):
        if _STOP:
            break

        day_str = d.isoformat()

        # First: insert any already-cached GDELT articles for this date (free, instant)
        cached = _gdelt_cached_articles(d)
        if cached:
            n = _insert_articles(conn, cached, day_str)
            conn.commit()
            if n:
                total_inserted += n

        # Then: fetch any uncached keyword groups for this date
        for kw in KEYWORD_GROUPS:
            if _STOP:
                break
            slug = hashlib.md5(kw.encode()).hexdigest()[:8]
            cache_path = GDELT_CACHE / f"{day_str}_{slug}.json"
            if cache_path.exists():
                continue  # already fetched
            # Live GDELT fetch
            articles = engine.gdelt.fetch(d, kw)
            if articles:
                n = _insert_articles(conn, articles, day_str)
                conn.commit()
                total_inserted += n

        if (idx + 1) % 10 == 0 or idx == len(trading_days) - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed * 60
            remaining = len(trading_days) - idx - 1
            eta_min = remaining / rate if rate > 0 else 0
            print(f"[backfill] {idx+1}/{len(trading_days)} days  "
                  f"+{total_inserted} articles  "
                  f"elapsed={elapsed/60:.1f}m  eta≈{eta_min:.0f}m")

    conn.close()
    elapsed = time.time() - t0
    print(f"[backfill] done — inserted {total_inserted} new articles "
          f"in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
