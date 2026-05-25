"""Sentiment streak tracker: tickers with 3+ consecutive bullish hours.

Scans the last ``LOOKBACK_HOURS`` of articles, buckets each ticker's average
ml_score into 1-hour windows, then counts the longest unbroken run of "bullish"
windows (avg >= ``BULL_THRESHOLD``). Tickers with a current streak of
``MIN_STREAK`` or more are written to ``SNAPSHOT_PATH``.

Useful for separating one-off spikes (trend_velocity) from sustained momentum:
a ticker that has been consistently net-bullish for 4+ hours is a different
signal than a 5-minute burst of articles.

Standalone:  python3 -m analytics.sentiment_streak
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analytics.trend_velocity import _parse_ts, extract_tickers
from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
SNAPSHOT_PATH = Path("/home/zeph/logs/sentiment_streak.json")

LOOKBACK_HOURS = 12
FETCH_LIMIT = 8000
BULL_THRESHOLD = 6.0  # avg ml_score to count a window as bullish
MIN_ARTICLES_PER_WINDOW = 1  # ignore windows with no articles
MIN_STREAK = 3  # minimum consecutive bullish hours to report
TOP_N = 10


def _hour_bucket(dt: datetime) -> str:
    """Return ISO hour string, e.g. '2026-05-24T18'."""
    return dt.strftime("%Y-%m-%dT%H")


def compute(fetch_limit: int = FETCH_LIMIT) -> list[dict]:
    """Return list of tickers with sustained bullish streaks, desc by streak len."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    conn.execute("PRAGMA busy_timeout=5000")
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=LOOKBACK_HOURS)).isoformat()

    cur = conn.execute(
        f"""
        SELECT first_seen, title, ml_score
        FROM articles INDEXED BY idx_first_seen
        WHERE first_seen >= ?
          AND ml_score IS NOT NULL
          AND {_LIVE_ONLY_CLAUSE}
        ORDER BY first_seen DESC
        LIMIT ?
        """,
        (cutoff, fetch_limit),
    )

    # ticker -> hour_bucket -> [ml_scores]
    ticker_hours: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for first_seen, title, ml_score in cur:
        ts = _parse_ts(first_seen)
        if ts is None:
            continue
        bucket = _hour_bucket(ts)
        for tk in set(extract_tickers(title)):
            ticker_hours[tk][bucket].append(float(ml_score))

    conn.close()

    # Build ordered list of the last LOOKBACK_HOURS buckets (newest last)
    all_buckets: list[str] = []
    t = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=LOOKBACK_HOURS - 1)
    while t <= now:
        all_buckets.append(_hour_bucket(t))
        t += timedelta(hours=1)

    results: list[dict] = []

    for tk, hours in ticker_hours.items():
        # Per-bucket: compute avg ml_score if enough articles, else None
        window_avgs: list[float | None] = []
        for b in all_buckets:
            scores = hours.get(b, [])
            if len(scores) >= MIN_ARTICLES_PER_WINDOW:
                window_avgs.append(sum(scores) / len(scores))
            else:
                window_avgs.append(None)

        # Find the current streak: walk backwards, skip empty trailing windows
        # (the current partial hour may have no data yet), then count bullish run.
        avgs_rev = list(reversed(window_avgs))
        # Skip initial None entries (empty/partial trailing windows)
        i = 0
        while i < len(avgs_rev) and avgs_rev[i] is None:
            i += 1
        current_streak = 0
        for avg in avgs_rev[i:]:
            if avg is not None and avg >= BULL_THRESHOLD:
                current_streak += 1
            else:
                break  # streak broken by populated bearish window or gap

        if current_streak < MIN_STREAK:
            continue

        # Also track overall best avg across streak windows
        streak_avgs = [
            window_avgs[-(i + 1)]
            for i in range(current_streak)
            if window_avgs[-(i + 1)] is not None
        ]
        mean_in_streak = round(sum(streak_avgs) / len(streak_avgs), 3) if streak_avgs else 0.0

        results.append({
            "ticker": tk,
            "streak_hours": current_streak,
            "mean_ml_score": mean_in_streak,
            "window_avgs": {
                b: (round(sum(hours.get(b, [])) / len(hours[b]), 3) if hours.get(b) else None)
                for b in all_buckets[-current_streak:]
            },
        })

    results.sort(key=lambda x: (-x["streak_hours"], -x["mean_ml_score"]))
    return results[:TOP_N]


def top_bullish_hours(fetch_limit: int = FETCH_LIMIT) -> list[dict]:
    """Fallback: rank tickers by total bullish windows (for slow/overnight periods)."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    conn.execute("PRAGMA busy_timeout=5000")
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=LOOKBACK_HOURS)).isoformat()
    cur = conn.execute(
        f"""
        SELECT first_seen, title, ml_score
        FROM articles INDEXED BY idx_first_seen
        WHERE first_seen >= ?
          AND ml_score IS NOT NULL
          AND {_LIVE_ONLY_CLAUSE}
        ORDER BY first_seen DESC
        LIMIT ?
        """,
        (cutoff, fetch_limit),
    )
    ticker_hours: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for first_seen, title, ml_score in cur:
        ts = _parse_ts(first_seen)
        if ts is None:
            continue
        bucket = _hour_bucket(ts)
        for tk in set(extract_tickers(title)):
            ticker_hours[tk][bucket].append(float(ml_score))
    conn.close()

    results = []
    for tk, hours in ticker_hours.items():
        total_windows = 0
        bull_windows = 0
        best_avg = 0.0
        for b, scores in hours.items():
            if not scores:
                continue
            avg = sum(scores) / len(scores)
            total_windows += 1
            if avg >= BULL_THRESHOLD:
                bull_windows += 1
                best_avg = max(best_avg, avg)
        if bull_windows > 0:
            results.append({
                "ticker": tk,
                "bullish_windows": bull_windows,
                "total_windows": total_windows,
                "best_avg_ml": round(best_avg, 3),
            })
    results.sort(key=lambda x: (-x["bullish_windows"], -x["best_avg_ml"]))
    return results[:TOP_N]


def main() -> None:
    streaks = compute()
    now = datetime.now(timezone.utc)
    out: dict = {
        "generated_at": now.isoformat(),
        "lookback_hours": LOOKBACK_HOURS,
        "bull_threshold": BULL_THRESHOLD,
        "min_streak": MIN_STREAK,
        "streaks": streaks,
    }

    if not streaks:
        fallback = top_bullish_hours()
        out["note"] = "no sustained streaks; showing top tickers by bullish window count"
        out["top_bullish"] = fallback
        SNAPSHOT_PATH.write_text(json.dumps(out, indent=2))
        print(f"sentiment_streak: no {MIN_STREAK}h streaks (slow period); top bullish tickers in last {LOOKBACK_HOURS}h:")
        for tb in fallback[:5]:
            print(f"  {tb['ticker']:<6} bullish={tb['bullish_windows']}/{tb['total_windows']}h  best_ml={tb['best_avg_ml']:.3f}")
        return

    SNAPSHOT_PATH.write_text(json.dumps(out, indent=2))
    print(f"sentiment_streak: {len(streaks)} tickers with {MIN_STREAK}+ consecutive bullish hours")
    for s in streaks:
        print(f"  {s['ticker']:<6} streak={s['streak_hours']}h  avg_ml={s['mean_ml_score']:.3f}")


if __name__ == "__main__":
    main()
