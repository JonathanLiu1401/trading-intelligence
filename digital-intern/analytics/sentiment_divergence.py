"""Sentiment divergence detector.

Flags tickers where multiple *sources* report conflicting signals (one clearly
bullish, another clearly bearish) for the same ticker within a rolling 2h
window.  Divergence is a trading signal in itself: the market hasn't reached
consensus and the next catalyst is likely to break one side.

Scoring:
  * bullish article  : ai_score >= BULL_THRESH  (default 0.65)
  * bearish article  : ai_score <= BEAR_THRESH  (default 0.35)
  * divergence event : same ticker has >= MIN_BULL bullish article(s) from
                       one or more sources AND >= MIN_BEAR bearish article(s)
                       from one or more *different* sources

The divergence_score is bull_mean - bear_mean (abs = magnitude of split).
A high score means strong bullish + strong bearish disagreement.

Design constraints (same as all other analytics):
  * Bounded SCAN_LIMIT idx_first_seen read — never full-table scans the 1.4 GB DB
  * Read-only sqlite URI — never contends with daemon writers
  * USB-safe busy_timeout

Output: /home/zeph/logs/sentiment_divergence.json
Standalone: python3 -m analytics.sentiment_divergence
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
OUT_PATH = Path("/home/zeph/logs/sentiment_divergence.json")

WINDOW_HOURS = 2
SCAN_LIMIT = 6000
# ai_score is 0-10; baseline is heavily zero-skewed (median ~1.25)
# "high signal" = sources treating this ticker as urgent/important
# "low signal"  = sources treating it as routine noise
HIGH_THRESH = 5.0    # ai_score >= this = high-signal (urgent) coverage
LOW_THRESH  = 1.5    # ai_score <= this = low-signal (noise) coverage
BULL_THRESH = HIGH_THRESH   # alias kept for internal use
BEAR_THRESH = LOW_THRESH
MIN_BULL = 1
MIN_BEAR = 1
TOP_N = 15


def _extract_tickers(title: str) -> list[str]:
    out: list[str] = []
    for m in TICKER_RE.findall(title or ""):
        if m not in STOP and len(m) >= 2:
            out.append(m)
    return out


def compute() -> dict:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")

    rows = conn.execute(
        "SELECT first_seen, title, source, ai_score FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=WINDOW_HOURS)

    # per ticker: list of (ai_score, source, title) for each camp
    bull: dict[str, list[dict]] = defaultdict(list)
    bear: dict[str, list[dict]] = defaultdict(list)

    for fs, title, source, ai_score in rows:
        if ai_score is None:
            continue
        ts = _parse_ts(fs)
        if ts is None or ts < window_start:
            continue
        for ticker in set(_extract_tickers(title)):
            entry = {"score": float(ai_score), "source": source or "", "title": (title or "")[:80]}
            if ai_score >= BULL_THRESH:
                bull[ticker].append(entry)
            elif ai_score <= BEAR_THRESH:
                bear[ticker].append(entry)

    # find tickers with both bullish and bearish coverage
    divergent: list[dict] = []
    all_tickers = set(bull) & set(bear)
    for ticker in all_tickers:
        b_articles = bull[ticker]
        s_articles = bear[ticker]
        if len(b_articles) < MIN_BULL or len(s_articles) < MIN_BEAR:
            continue
        bull_sources = {a["source"] for a in b_articles}
        bear_sources = {a["source"] for a in s_articles}
        # require at least one source appearing on only one side (genuine disagreement)
        if not (bull_sources - bear_sources or bear_sources - bull_sources):
            continue
        b_mean = round(mean(a["score"] for a in b_articles), 4)
        s_mean = round(mean(a["score"] for a in s_articles), 4)
        divergent.append({
            "ticker": ticker,
            "divergence_score": round(b_mean - s_mean, 4),
            "bull_count": len(b_articles),
            "bear_count": len(s_articles),
            "bull_mean": b_mean,
            "bear_mean": s_mean,
            "bull_sources": sorted(bull_sources),
            "bear_sources": sorted(bear_sources),
            "top_bull": b_articles[0]["title"],
            "top_bear": s_articles[0]["title"],
        })

    divergent.sort(key=lambda x: x["divergence_score"], reverse=True)
    top = divergent[:TOP_N]

    result = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "scanned_rows": len(rows),
        "divergent_tickers": len(divergent),
        "top": top,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    result = compute()
    top = result["top"]
    print(f"sentiment_divergence: scanned={result['scanned_rows']} "
          f"divergent={result['divergent_tickers']} top={len(top)}")
    for d in top[:5]:
        print(f"  {d['ticker']:6s} div={d['divergence_score']:+.3f} "
              f"bull={d['bull_count']}×{d['bull_mean']:.2f} "
              f"bear={d['bear_count']}×{d['bear_mean']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
