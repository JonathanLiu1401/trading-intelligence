"""Urgency spike rate detector.

Detects when the rate of urgency>=2 articles in the last 15 minutes
is significantly above the 6-hour hourly baseline. Surfaces which
tickers are driving the spike.

Design constraints:
  * Bounded idx_first_seen scan only — no full-table COUNT(*).
  * Read-only, busy_timeout=5000 ms.
  * State file accumulates hourly urgency-rate baseline.
  * _LIVE_ONLY_CLAUSE applied to exclude backtest rows.

Artifacts:
  * State : /home/zeph/logs/.urgency_spike_state.json
  * Log   : /home/zeph/logs/urgency_spike.log
  * Out   : /home/zeph/logs/urgency_spike.json

Exit status always 0.
"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
LOG_DIR = Path("/home/zeph/logs")
STATE_PATH = LOG_DIR / ".urgency_spike_state.json"
LOG_PATH = LOG_DIR / "urgency_spike.log"
OUT_PATH = LOG_DIR / "urgency_spike.json"

SCAN_LIMIT = 4000       # recent rows via idx_first_seen, ~8h of live data
WINDOW_MIN = 15         # "current" spike window in minutes
BASELINE_HOURS = 6      # how many hourly buckets to keep for baseline
MIN_BASELINE_POINTS = 3 # minimum history before alerting
SIGMA = 2.0             # threshold: current rate > mean + SIGMA*std
TOP_TICKERS = 5         # tickers to surface in output

TICKER_RE = re.compile(r"\b\$?([A-Z]{2,5})\b")
STOP = {
    "CEO", "CFO", "CTO", "USA", "USD", "EUR", "GBP", "EU", "UK", "US",
    "AI", "ML", "API", "IPO", "ETF", "SEC", "FOMC", "FED", "GDP", "CPI",
    "PPI", "ECB", "BOJ", "PBOC", "OPEC", "NYSE", "NASDAQ", "AMEX",
    "Q1", "Q2", "Q3", "Q4", "YTD", "YOY", "EPS", "PE", "EV", "ESG",
    "BUY", "SELL", "HOLD", "ON", "AT", "IN", "TO", "OF", "FOR", "THE",
    "AND", "OR", "BY", "AS", "IS", "WAS", "ARE", "BE", "AN", "A",
    "NEW", "OLD", "TOP", "LOW", "HIGH", "BIG", "DAY", "WEEK", "MONTH",
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP",
    "OCT", "NOV", "DEC", "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
    "II", "III", "IV", "VI",
    "NEWS", "INC", "LLC", "LTD", "CORP", "CO", "PLC",
    "MSN", "CNN", "BBC", "WSJ", "NYT", "FT", "AP", "AFP",
    "MONEY", "STOCK", "STOCKS", "MARKET", "DEAL", "DEALS",
    "JUNE", "JULY",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = str(raw).replace("T", " ").split("+")[0].strip()[:26]
    try:
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _extract_tickers(title: str) -> list[str]:
    return [
        m for m in TICKER_RE.findall(title or "")
        if m not in STOP and len(m) >= 2
    ]


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"hourly_rates": []}


def _save_state(state: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def run() -> dict:
    now = _utcnow()
    spike_cutoff = now - timedelta(minutes=WINDOW_MIN)
    scan_cutoff = now - timedelta(hours=BASELINE_HOURS + 1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row

    # Use ORDER BY first_seen DESC LIMIT (served by idx_first_seen) — fast on
    # the USB-backed DB, avoids the slow id-subquery pattern.
    rows = conn.execute(
        f"""
        SELECT title, urgency, first_seen
          FROM articles
         WHERE {_LIVE_ONLY_CLAUSE}
         ORDER BY first_seen DESC
         LIMIT ?
        """,
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    # Bucket rows into hourly slots for baseline and current 15-min window
    urgent_in_window: list[sqlite3.Row] = []
    hourly_counts: dict[str, int] = {}

    for row in rows:
        ts = _parse_ts(row["first_seen"])
        if ts is None:
            continue
        if row["urgency"] < 2:
            continue

        if ts >= spike_cutoff:
            urgent_in_window.append(row)

        # Bucket by hour key (exclude current partial hour from baseline)
        hour_key = ts.strftime("%Y-%m-%dT%H:00")
        current_hour = now.strftime("%Y-%m-%dT%H:00")
        if hour_key != current_hour:
            hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1

    # Convert current 15-min count to per-hour rate for fair comparison
    current_rate = len(urgent_in_window) * (60.0 / WINDOW_MIN)

    # Load and update rolling baseline
    state = _load_state()
    hourly_rates: list[dict] = state.get("hourly_rates", [])

    # Add new hourly data points from this scan
    cutoff_key = scan_cutoff.strftime("%Y-%m-%dT%H:00")
    for hkey, cnt in hourly_counts.items():
        if hkey >= cutoff_key:
            # Update or insert this hour's count
            existing = next((h for h in hourly_rates if h["hour"] == hkey), None)
            if existing:
                existing["rate"] = cnt
            else:
                hourly_rates.append({"hour": hkey, "rate": cnt})

    # Trim to BASELINE_HOURS most recent hours
    hourly_rates.sort(key=lambda h: h["hour"])
    hourly_rates = hourly_rates[-BASELINE_HOURS:]
    state["hourly_rates"] = hourly_rates
    _save_state(state)

    baseline_rates = [h["rate"] for h in hourly_rates]
    n = len(baseline_rates)

    if n < MIN_BASELINE_POINTS:
        status = "insufficient_baseline"
        threshold = None
        sigma_val = None
        is_spike = False
    else:
        mean = statistics.mean(baseline_rates)
        std = statistics.pstdev(baseline_rates)
        threshold = mean + SIGMA * std
        sigma_val = (current_rate - mean) / std if std > 0 else 0.0
        is_spike = current_rate > threshold and current_rate > 0

        if is_spike:
            status = "spike"
        elif current_rate == 0:
            status = "quiet"
        else:
            status = "normal"

    # Surface top tickers from spike window
    ticker_counts: Counter = Counter()
    for row in urgent_in_window:
        for t in _extract_tickers(row["title"]):
            ticker_counts[t] += 1
    top_tickers = [
        {"ticker": t, "count": c}
        for t, c in ticker_counts.most_common(TOP_TICKERS)
    ]

    result = {
        "ts": now.isoformat(),
        "status": status,
        "window_min": WINDOW_MIN,
        "urgent_in_window": len(urgent_in_window),
        "current_rate_per_hour": round(current_rate, 1),
        "baseline_n_hours": n,
        "baseline_mean": round(statistics.mean(baseline_rates), 2) if n > 0 else None,
        "baseline_std": round(statistics.pstdev(baseline_rates), 2) if n > 0 else None,
        "threshold_rate": round(threshold, 2) if threshold is not None else None,
        "sigma": round(sigma_val, 2) if sigma_val is not None else None,
        "top_tickers": top_tickers,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    log_line = (
        f"[{now.strftime('%H:%M')}] status={status} "
        f"rate={current_rate:.1f}/h "
        f"(baseline {result['baseline_mean']}/{result['baseline_std']} σ={result['sigma']}) "
        f"window_urgent={len(urgent_in_window)} "
        f"tickers={[t['ticker'] for t in top_tickers[:3]]}"
    )
    with LOG_PATH.open("a") as f:
        f.write(log_line + "\n")

    return result


def main() -> None:
    result = run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
