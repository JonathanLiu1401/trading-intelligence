"""Ticker score acceleration detector.

Splits the last 2 hours into four 30-minute sub-windows and computes the
average ml_score per ticker in each.  A ticker is "accelerating" when its
per-window scores form a monotonically (or near-monotonically) increasing
trend.  This catches building momentum *before* the signal saturates and
the simple 2-window comparator in ticker_sentiment_momentum flags it.

Acceleration is quantified as the linear slope (score-units per sub-window)
estimated via least-squares over the four windows.  Only tickers with >=2
articles across all windows and a slope >= MIN_SLOPE are reported.

Output: /home/zeph/logs/ticker_score_acceleration.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from analytics.trend_velocity import TICKER_RE, STOP, _parse_ts  # noqa: E402
from storage.article_store import _LIVE_ONLY_CLAUSE  # noqa: E402

DB_PATH = BASE / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/ticker_score_acceleration.json")

LOOKBACK_HOURS = 2
SUB_WINDOWS = 4           # 30-min slices
FETCH_LIMIT = 8000
MIN_TOTAL_ARTICLES = 2    # across all windows combined
MIN_ABS_SLOPE = 0.03      # min |slope| (score units / sub-window) to report
TOP_N = 5                 # top N for each direction


def _extract_tickers(title: str) -> list[str]:
    return [m for m in TICKER_RE.findall(title or "") if m not in STOP and len(m) >= 2]


def _linslope(ys: list[float]) -> float:
    """Least-squares slope of ys against x = 0,1,...,n-1."""
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = (n - 1) / 2
    y_mean = mean(ys)
    num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den else 0.0


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")

    rows = conn.execute(
        "SELECT first_seen, title, ml_score FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} AND ml_score IS NOT NULL "
        "ORDER BY first_seen DESC LIMIT ?",
        (FETCH_LIMIT,),
    ).fetchall()
    conn.close()

    if not rows:
        print("ticker_score_acceleration: no rows", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    window_len = timedelta(hours=LOOKBACK_HOURS) / SUB_WINDOWS  # 30 min each

    # bin[w][ticker] = list of ml_scores
    bins: list[dict[str, list[float]]] = [defaultdict(list) for _ in range(SUB_WINDOWS)]

    for fs, title, ml_score in rows:
        ts = _parse_ts(fs)
        if ts is None:
            continue
        age = (now - ts).total_seconds()
        if age < 0 or age > LOOKBACK_HOURS * 3600:
            continue
        w = int(age / window_len.total_seconds())
        if w >= SUB_WINDOWS:
            continue
        # w=0 is most recent, reverse so w=0 → slot 3 (ascending time order)
        slot = SUB_WINDOWS - 1 - w
        for ticker in set(_extract_tickers(title)):
            bins[slot][ticker].append(float(ml_score))

    # collect all tickers with enough coverage
    all_tickers: set[str] = set()
    for b in bins:
        all_tickers.update(b.keys())

    results: list[dict] = []
    for ticker in sorted(all_tickers):
        total = sum(len(bins[s].get(ticker, [])) for s in range(SUB_WINDOWS))
        if total < MIN_TOTAL_ARTICLES:
            continue

        window_avgs: list[float | None] = []
        for s in range(SUB_WINDOWS):
            scores = bins[s].get(ticker, [])
            window_avgs.append(mean(scores) if scores else None)

        # only compute slope over windows that have data
        filled = [(i, v) for i, v in enumerate(window_avgs) if v is not None]
        if len(filled) < 2:
            continue

        xs_f = [f[0] for f in filled]
        ys_f = [f[1] for f in filled]
        # normalize xs to 0..1 for comparability when some windows missing
        x_range = xs_f[-1] - xs_f[0] or 1
        xs_norm = [(x - xs_f[0]) / x_range for x in xs_f]
        n = len(ys_f)
        x_mean = mean(xs_norm)
        y_mean = mean(ys_f)
        num = sum((xs_norm[i] - x_mean) * (ys_f[i] - y_mean) for i in range(n))
        den = sum((x - x_mean) ** 2 for x in xs_norm)
        slope = (num / den) if den else 0.0

        if abs(slope) < MIN_ABS_SLOPE:
            continue

        results.append({
            "ticker": ticker,
            "slope": round(slope, 4),
            "direction": "accelerating" if slope > 0 else "decelerating",
            "window_avgs": [round(v, 3) if v is not None else None for v in window_avgs],
            "total_articles": total,
            "latest_avg": round(window_avgs[-1], 3) if window_avgs[-1] is not None else None,
        })

    results.sort(key=lambda x: x["slope"], reverse=True)
    top_bull = [r for r in results if r["slope"] > 0][:TOP_N]
    top_bear = [r for r in results if r["slope"] < 0][-TOP_N:][::-1]  # most negative first

    out = {
        "generated_at": now.isoformat(),
        "lookback_hours": LOOKBACK_HOURS,
        "sub_window_minutes": int(window_len.total_seconds() / 60),
        "sub_windows": [
            f"{now - window_len * (SUB_WINDOWS - s):%H:%M}-{now - window_len * (SUB_WINDOWS - s - 1):%H:%M} UTC"
            for s in range(SUB_WINDOWS)
        ],
        "min_abs_slope": MIN_ABS_SLOPE,
        "accelerating": top_bull,
        "decelerating": top_bear,
        "total_candidates": len(results),
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    total_bull = len(top_bull)
    total_bear = len(top_bear)
    print(f"ticker_score_acceleration: {total_bull} accelerating, {total_bear} decelerating (of {len(results)} total)")
    if top_bear:
        for r in top_bear[:3]:
            print(f"  BEAR {r['ticker']:6s}  slope={r['slope']:+.3f}  windows={r['window_avgs']}  n={r['total_articles']}")
    if top_bull:
        for r in top_bull[:3]:
            print(f"  BULL {r['ticker']:6s}  slope={r['slope']:+.3f}  windows={r['window_avgs']}  n={r['total_articles']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
