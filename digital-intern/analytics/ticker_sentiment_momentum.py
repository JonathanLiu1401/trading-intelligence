"""Ticker sentiment momentum: detect per-ticker ai_score shifts between windows.

Compares the average ai_score for each ticker in the current 2h window against
the prior 2h window (2-4 hours ago).  Tickers with >= MIN_ARTICLES in either
window and a score delta >= MIN_DELTA are flagged as momentum events.

Output: /home/zeph/logs/ticker_sentiment_momentum.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

from analytics.trend_velocity import TICKER_RE, STOP, _parse_ts
from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/ticker_sentiment_momentum.json")
WINDOW_HOURS = 2
FETCH_LIMIT = 6000
MIN_ARTICLES = 2   # minimum mentions per window to be included
MIN_DELTA = 0.15   # minimum abs(score swing) to flag as momentum


def extract_tickers(title: str) -> list[str]:
    out = []
    for m in TICKER_RE.findall(title or ""):
        if m not in STOP and len(m) >= 2:
            out.append(m)
    return out


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")

    rows = conn.execute(
        "SELECT first_seen, title, ai_score FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (FETCH_LIMIT,),
    ).fetchall()
    conn.close()

    if not rows:
        print("ticker_sentiment_momentum: no rows", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    cur_start = now - timedelta(hours=WINDOW_HOURS)
    prev_start = now - timedelta(hours=WINDOW_HOURS * 2)

    # per-ticker score lists for each window
    cur_scores: dict[str, list[float]] = defaultdict(list)
    prev_scores: dict[str, list[float]] = defaultdict(list)

    for fs, title, ai_score in rows:
        if ai_score is None:
            continue
        ts = _parse_ts(fs)
        if ts is None:
            continue
        tickers = set(extract_tickers(title))
        if ts >= cur_start:
            for t in tickers:
                cur_scores[t].append(float(ai_score))
        elif ts >= prev_start:
            for t in tickers:
                prev_scores[t].append(float(ai_score))

    # compute momentum for tickers present in both windows
    momentum: list[dict] = []
    all_tickers = set(cur_scores) | set(prev_scores)
    for ticker in sorted(all_tickers):
        c = cur_scores.get(ticker, [])
        p = prev_scores.get(ticker, [])
        if len(c) < MIN_ARTICLES and len(p) < MIN_ARTICLES:
            continue
        c_avg = mean(c) if c else None
        p_avg = mean(p) if p else None
        if c_avg is None or p_avg is None:
            continue
        delta = c_avg - p_avg
        if abs(delta) < MIN_DELTA:
            continue
        direction = "bullish" if delta > 0 else "bearish"
        momentum.append({
            "ticker": ticker,
            "cur_avg": round(c_avg, 3),
            "prev_avg": round(p_avg, 3),
            "delta": round(delta, 3),
            "direction": direction,
            "cur_articles": len(c),
            "prev_articles": len(p),
        })

    # sort by absolute delta descending
    momentum.sort(key=lambda x: abs(x["delta"]), reverse=True)

    out = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "cur_window": f"{cur_start.isoformat()} to {now.isoformat()}",
        "prev_window": f"{prev_start.isoformat()} to {cur_start.isoformat()}",
        "min_articles": MIN_ARTICLES,
        "min_delta": MIN_DELTA,
        "tickers": momentum,
        "top_bullish": [t for t in momentum if t["direction"] == "bullish"][:3],
        "top_bearish": [t for t in momentum if t["direction"] == "bearish"][:3],
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"ticker_sentiment_momentum: {len(momentum)} tickers with momentum shift")
    for entry in momentum[:5]:
        print(
            f"  {entry['ticker']:6s}  {entry['direction']:7s}  "
            f"prev={entry['prev_avg']:.3f} -> cur={entry['cur_avg']:.3f}  "
            f"delta={entry['delta']:+.3f}  "
            f"(n_cur={entry['cur_articles']} n_prev={entry['prev_articles']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
