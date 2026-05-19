"""Ticker co-mention graph: top ticker pairs co-occurring in recent articles.

Complements ``trend_velocity`` (per-ticker volume) by surfacing *sector* moves:
when two tickers light up together repeatedly in a short window, it's usually
a sector ETF rip, a peer-readthrough, or an M&A pairing rather than a single-
name story. Operators can use the list to decide whether a velocity signal is
idiosyncratic or part of a broader basket move.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

from analytics.trend_velocity import extract_tickers, _parse_ts
from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/ticker_comentions.json")
WINDOW_HOURS = 2
FETCH_LIMIT = 4000
TOP_N = 10
MIN_PAIR_COUNT = 2


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    # Canonical `_LIVE_ONLY_CLAUSE` — a partial `source NOT LIKE 'backtest_run_%'`
    # leaks `backtest://` URLs and `opus_annotation*` synthetic rows into the
    # co-mention graph, inflating pair counts with training-only artefacts.
    rows = conn.execute(
        "SELECT first_seen, title FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (FETCH_LIMIT,),
    ).fetchall()
    if not rows:
        print("ticker_comentions: no rows", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    cut = now - timedelta(hours=WINDOW_HOURS)

    pair_counts: Counter[tuple[str, str]] = Counter()
    solo_counts: Counter[str] = Counter()
    articles_in_window = 0
    for fs, title in rows:
        ts = _parse_ts(fs)
        if ts is None or ts < cut:
            continue
        articles_in_window += 1
        tix = sorted(set(extract_tickers(title)))
        for t in tix:
            solo_counts[t] += 1
        if len(tix) < 2:
            continue
        for a, b in combinations(tix, 2):
            pair_counts[(a, b)] += 1

    filtered = [(p, c) for p, c in pair_counts.items() if c >= MIN_PAIR_COUNT]
    filtered.sort(key=lambda x: x[1], reverse=True)
    top = filtered[:TOP_N]

    payload = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "scanned": len(rows),
        "articles_in_window": articles_in_window,
        "unique_pairs": len(pair_counts),
        "qualified_pairs": len(filtered),
        "top": [
            {
                "pair": [a, b],
                "co_count": c,
                "a_total": solo_counts[a],
                "b_total": solo_counts[b],
                "lift": round(c / min(solo_counts[a], solo_counts[b]), 2),
            }
            for (a, b), c in top
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    print(
        f"ticker_comentions: scanned={len(rows)} in_window={articles_in_window} "
        f"pairs={len(pair_counts)} qualified={len(filtered)}"
    )
    for (a, b), c in top:
        print(f"  {a}+{b}: co={c} a={solo_counts[a]} b={solo_counts[b]}")
    if not top:
        print("  (no qualifying ticker pairs in window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
