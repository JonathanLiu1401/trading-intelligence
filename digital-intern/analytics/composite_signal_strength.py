"""Composite signal strength scorer.

For each ticker mentioned in the last 2 hours, combine three dimensions into
a single 0-10 signal_strength score:

  1. velocity_z  — mention count z-score vs 7-day hourly baseline (clipped 0-4)
  2. avg_ml      — mean ml_score for those articles (0-10 scale)
  3. urgency_rate— fraction of articles with urgency>=2 (0-1, scaled to 0-10)

  signal_strength = (velocity_z/4 * 4) + (avg_ml/10 * 4) + (urgency_rate * 2)
                  = max 10

Outputs top-10 to /home/zeph/logs/composite_signal_strength.json and prints a
ranked table to stdout.
"""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/composite_signal_strength.json")

WINDOW_HOURS = 2        # recent window to score
BASELINE_DAYS = 7       # days to compute hourly baseline
FETCH_LIMIT = 4000
BASELINE_LIMIT = 20000
TOP_N = 10

TICKER_RE = re.compile(r"\b\$?([A-Z]{2,5})\b")
STOP = {
    "CEO", "CFO", "CTO", "COO", "USA", "USD", "EUR", "GBP", "JPY", "CNY",
    "EU", "UK", "US", "AI", "ML", "API", "IPO", "ETF", "SEC", "FOMC",
    "FED", "GDP", "CPI", "PPI", "ECB", "BOJ", "PBOC", "OPEC", "NYSE",
    "NASDAQ", "AMEX", "Q1", "Q2", "Q3", "Q4", "YTD", "YOY", "EPS", "PE",
    "EV", "ESG", "BUY", "SELL", "HOLD", "ON", "AT", "IN", "TO", "OF",
    "FOR", "THE", "AND", "OR", "BY", "AS", "IS", "WAS", "ARE", "BE",
    "AN", "A", "NEW", "OLD", "TOP", "LOW", "HIGH", "BIG", "DAY", "WEEK",
    "MONTH", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
    "SEP", "OCT", "NOV", "DEC", "NO", "NOT", "IF", "WE", "IT", "HE",
    "SHE", "PM", "AM", "EST", "PST", "UTC", "ET", "PT", "CT", "MT",
    "AD", "RE", "VS", "EX", "UP", "DOWN",
}

def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    raw = raw.replace("T", " ").split(".")[0]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def extract_tickers(text: str | None) -> list[str]:
    if not text:
        return []
    return [t for t in TICKER_RE.findall(text) if t not in STOP and len(t) >= 2]


def run() -> list[dict]:
    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(hours=WINDOW_HOURS)
    baseline_cutoff = now - timedelta(days=BASELINE_DAYS)

    con = sqlite3.connect(DB_PATH)
    try:
        # Fetch recent articles (last 2h)
        rows_recent = con.execute(
            f"""
            SELECT first_seen, title, ml_score, urgency
            FROM articles
            WHERE replace(first_seen,'T',' ') >= ?
              AND {_LIVE_ONLY_CLAUSE}
            ORDER BY first_seen DESC
            LIMIT {FETCH_LIMIT}
            """,
            (recent_cutoff.strftime("%Y-%m-%d %H:%M:%S"),),
        ).fetchall()

        # Fetch baseline articles (last 7d) — timestamps only for hourly counts
        rows_baseline = con.execute(
            f"""
            SELECT first_seen, title
            FROM articles
            WHERE replace(first_seen,'T',' ') >= ?
              AND replace(first_seen,'T',' ') < ?
              AND {_LIVE_ONLY_CLAUSE}
            ORDER BY first_seen DESC
            LIMIT {BASELINE_LIMIT}
            """,
            (
                baseline_cutoff.strftime("%Y-%m-%d %H:%M:%S"),
                recent_cutoff.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        ).fetchall()
    finally:
        con.close()

    # ── Build hourly baseline per ticker ─────────────────────────────────────
    # bucket = (ticker, hour_bucket) -> count
    baseline_hourly: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for raw_ts, title in rows_baseline:
        ts = _parse_ts(raw_ts)
        if ts is None:
            continue
        bucket = ts.strftime("%Y-%m-%d-%H")
        tickers = extract_tickers(title)
        if not tickers:
            continue
        for t in tickers:
            baseline_hourly[t][bucket] += 1

    # per-ticker: list of hourly counts over the 7d baseline
    def hourly_stats(ticker: str) -> tuple[float, float]:
        counts = list(baseline_hourly[ticker].values())
        if not counts:
            return 0.0, 0.0
        mean = statistics.mean(counts)
        stdev = statistics.stdev(counts) if len(counts) > 1 else 0.0
        return mean, stdev

    # ── Score recent articles per ticker ─────────────────────────────────────
    ticker_mentions: dict[str, list[dict]] = defaultdict(list)
    for raw_ts, title, ml_score, urgency in rows_recent:
        ts = _parse_ts(raw_ts)
        if ts is None:
            continue
        tickers = extract_tickers(title)
        for t in tickers:
            if t in STOP or len(t) < 2:
                continue
            ticker_mentions[t].append({
                "ml_score": float(ml_score or 0.0),
                "urgency": int(urgency or 0),
            })

    if not ticker_mentions:
        print("No ticker mentions in recent window.")
        return []

    # ── Compute composite score ───────────────────────────────────────────────
    results = []
    for ticker, articles in ticker_mentions.items():
        if len(articles) < 2:
            continue  # skip singletons — too noisy
        mention_count = len(articles)
        mean_b, std_b = hourly_stats(ticker)
        # velocity z-score: how many std above the baseline hourly rate?
        # We compare current-window count against expected count for WINDOW_HOURS
        expected = mean_b * WINDOW_HOURS
        if std_b > 0:
            z = (mention_count - expected) / std_b
        elif mention_count > expected:
            z = 2.0  # some signal even without std
        else:
            z = 0.0
        z = max(0.0, min(4.0, z))  # clip to [0,4]

        avg_ml = statistics.mean(a["ml_score"] for a in articles)
        urgency_rate = sum(1 for a in articles if a["urgency"] >= 2) / mention_count

        # weighted composite: velocity 40%, ml 40%, urgency 20%
        signal_strength = (z / 4.0 * 4.0) + (avg_ml / 10.0 * 4.0) + (urgency_rate * 2.0)
        signal_strength = round(signal_strength, 2)

        results.append({
            "ticker": ticker,
            "signal_strength": signal_strength,
            "mentions_2h": mention_count,
            "velocity_z": round(z, 2),
            "avg_ml_score": round(avg_ml, 2),
            "urgency_rate": round(urgency_rate, 3),
            "baseline_hourly_mean": round(mean_b, 2),
        })

    results.sort(key=lambda x: x["signal_strength"], reverse=True)
    top = results[:TOP_N]

    # ── Output ────────────────────────────────────────────────────────────────
    payload = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "baseline_days": BASELINE_DAYS,
        "top_signals": top,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    print(f"Composite Signal Strength — top {TOP_N} tickers (last {WINDOW_HOURS}h)")
    print(f"{'Ticker':<8} {'Strength':>8} {'Mentions':>9} {'VelZ':>6} {'AvgML':>7} {'UrgRt':>7}")
    print("-" * 54)
    for r in top:
        print(
            f"{r['ticker']:<8} {r['signal_strength']:>8.2f} {r['mentions_2h']:>9}"
            f" {r['velocity_z']:>6.2f} {r['avg_ml_score']:>7.2f} {r['urgency_rate']:>7.3f}"
        )
    print(f"\nWrote {len(top)} signals → {OUT_PATH}")
    return top


if __name__ == "__main__":
    run()
