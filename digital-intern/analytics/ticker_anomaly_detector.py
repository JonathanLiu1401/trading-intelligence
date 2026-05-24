"""Ticker mention anomaly detector: z-score spike detection vs 7-day baseline.

Finds tickers whose mention count in the past 1 hour deviates by more than
Z_THRESHOLD standard deviations from their hourly mean over the prior 7 days.
This catches low-volume tickers that suddenly spike (e.g., a normally quiet
stock mentioned once a week that appears 20 times in an hour) which
trend_velocity would miss because the raw count isn't in the top 5.

Algorithm:
  * Fetch last 7d+1h of articles (bounded SCAN_LIMIT)
  * Group mentions per ticker per hour-bucket
  * For each ticker with MIN_BASELINE_HOURS of history, compute mean+std
  * Flag current-hour count with z-score > Z_THRESHOLD

Output:
  /home/zeph/logs/ticker_anomaly.json

Standalone: python3 -m analytics.ticker_anomaly_detector
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/ticker_anomaly.json")

SCAN_LIMIT = 20_000
BASELINE_DAYS = 7
CURRENT_WINDOW_H = 1
Z_THRESHOLD = 3.0
MIN_BASELINE_HOURS = 12  # need enough history to compute meaningful std
MIN_BASELINE_MEAN = 0.5  # baseline avg must be >0 to flag (avoids 0→1 noise)
TOP_N = 10

TICKER_RE = re.compile(r"\b\$?([A-Z]{2,5})\b")
STOP = {
    "A", "I", "AM", "IS", "BE", "DO", "GO", "HE", "IT", "ME", "MY",
    "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE", "AT", "BY",
    "IN", "IF", "AS", "AN", "AND", "ARE", "FOR", "THE", "WAS", "BUT",
    "HAS", "HAD", "NOT", "ITS", "YOU", "NEW", "CAN", "ALL", "WILL",
    "WITH", "FROM", "HAVE", "BEEN", "THEY", "THIS", "THAT", "THAN",
    "INTO", "MORE", "OVER", "SAID", "ALSO", "MOST", "LAST", "JUST",
    "YEAR", "SAYS", "SAYS", "AFTER", "ABOUT", "THEIR", "WHICH",
    "WHEN", "WHAT", "SOME", "WERE", "THEN", "EACH", "BOTH", "WOULD",
    "COULD", "SHOULD", "BEFORE", "BETWEEN", "DURING", "WHILE",
    "CEO", "CFO", "IPO", "GDP", "CPI", "FED", "SEC", "NYSE", "ETF",
    "LLC", "INC", "LTD", "PLC", "ADR", "ESG", "NAV", "EPS", "YTD",
    "QoQ", "YoY", "MoM", "OTC", "AUM", "ROI", "EPS", "PE", "MA",
    "AI", "ML", "US", "UK", "EU", "EM", "DM", "VC", "PE", "M&A",
    "BUY", "SELL", "HOLD", "LONG", "SHORT", "CALL", "PUT",
    "HIGH", "BEST", "NEXT", "ONLY", "WELL", "BACK", "EVEN", "MUCH",
    "MANY", "FIRST", "THIRD", "HALF", "FULL", "GOOD", "LIKE", "MAKE",
}


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.strptime(s[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_tickers(title: str) -> list[str]:
    out = []
    for m in TICKER_RE.findall(title or ""):
        if m in STOP or len(m) < 2:
            continue
        out.append(m)
    return out


def _hour_bucket(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:00")


def detect_anomalies(conn: sqlite3.Connection, now: datetime | None = None) -> dict:
    if now is None:
        now = datetime.now(timezone.utc)

    cutoff = now - timedelta(days=BASELINE_DAYS, hours=CURRENT_WINDOW_H)
    current_cutoff = now - timedelta(hours=CURRENT_WINDOW_H)

    rows = conn.execute(
        "SELECT first_seen, title FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (SCAN_LIMIT,),
    ).fetchall()

    # ticker → hour_bucket → count
    hourly: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    current_counts: dict[str, int] = defaultdict(int)

    for raw_ts, title in rows:
        ts = _parse_ts(raw_ts)
        if ts is None or ts < cutoff:
            continue
        tickers = _extract_tickers(title)
        if not tickers:
            continue
        bucket = _hour_bucket(ts)
        for tk in tickers:
            hourly[tk][bucket] += 1
            if ts >= current_cutoff:
                current_counts[tk] += 1

    current_hour_bucket = _hour_bucket(current_cutoff)
    anomalies = []

    for tk, current_count in current_counts.items():
        if current_count == 0:
            continue
        # Baseline: all hours except the current window
        baseline_counts = [
            v for bucket, v in hourly[tk].items()
            if bucket != current_hour_bucket
        ]
        if len(baseline_counts) < MIN_BASELINE_HOURS:
            continue
        baseline_mean = statistics.mean(baseline_counts)
        if baseline_mean < MIN_BASELINE_MEAN:
            continue
        baseline_std = statistics.stdev(baseline_counts) if len(baseline_counts) > 1 else 0.0
        if baseline_std < 0.01:
            baseline_std = 0.01  # prevent division by zero for perfectly steady tickers
        z_score = (current_count - baseline_mean) / baseline_std
        if z_score >= Z_THRESHOLD:
            anomalies.append({
                "ticker": tk,
                "current_count": current_count,
                "baseline_mean": round(baseline_mean, 2),
                "baseline_std": round(baseline_std, 2),
                "z_score": round(z_score, 2),
                "baseline_hours": len(baseline_counts),
            })

    anomalies.sort(key=lambda r: r["z_score"], reverse=True)
    top = anomalies[:TOP_N]

    result = {
        "generated_at": now.isoformat(),
        "window_hours": CURRENT_WINDOW_H,
        "baseline_days": BASELINE_DAYS,
        "z_threshold": Z_THRESHOLD,
        "rows_scanned": len(rows),
        "tickers_evaluated": len(current_counts),
        "anomalies_found": len(anomalies),
        "top": top,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    result = detect_anomalies(conn)
    conn.close()

    print(f"ticker_anomaly_detector: scanned={result['rows_scanned']} "
          f"evaluated={result['tickers_evaluated']} "
          f"anomalies={result['anomalies_found']}")
    if result["top"]:
        for r in result["top"][:5]:
            print(f"  {r['ticker']:6s} z={r['z_score']:6.1f}  "
                  f"now={r['current_count']}  baseline_mean={r['baseline_mean']:.2f}")
    else:
        print("  (no anomalies above threshold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
