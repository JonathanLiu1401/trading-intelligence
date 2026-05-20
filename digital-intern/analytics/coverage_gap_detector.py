"""Coverage gap detector: tickers active 30–120min ago but silent in last 30min.

A ticker that generated ≥ MIN_PRIOR_MENTIONS headlines in the 30–120 min window
but zero in the last 30 min has "gone quiet". That transition can precede a
halt, post-announcement silence, or a lull before the next wave — all
operationally interesting to know about.

The inverse (appeared only in the last 30 min) is handled by breaking_news_detector;
this module captures the quiet-after-noise signal that complements it.

Standalone:  python3 -m analytics.coverage_gap_detector
Output:      /home/zeph/logs/coverage_gaps.json
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT = Path("/home/zeph/logs/coverage_gaps.json")
FETCH_LIMIT = 3000
PRIOR_WINDOW_MIN = 120
QUIET_WINDOW_MIN = 30
MIN_PRIOR_MENTIONS = 3  # need at least this many in the prior window

TICKER_RE = re.compile(r"\b\$?([A-Z]{2,5})\b")
STOP = {
    "CEO", "CFO", "CTO", "USA", "USD", "EUR", "GBP", "EU", "UK", "US",
    "AI", "ML", "API", "IPO", "ETF", "SEC", "FOMC", "FED", "GDP", "CPI",
    "PPI", "ECB", "BOJ", "PBOC", "OPEC", "NYSE", "NASDAQ", "AMEX",
    "Q1", "Q2", "Q3", "Q4", "YTD", "YOY", "EPS", "PE", "EV", "ESG",
    "BUY", "SELL", "HOLD", "ON", "AT", "IN", "TO", "OF", "FOR", "THE",
    "AND", "OR", "BY", "AS", "IS", "WAS", "ARE", "BE", "AN", "A",
    "NEW", "OLD", "TOP", "LOW", "HIGH", "BIG", "DAY", "WEEK", "MONTH",
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN", "II", "III", "IV", "VI",
    "NEWS", "INC", "LLC", "LTD", "CORP", "CO", "PLC", "MSN", "CNN", "BBC",
    "WSJ", "NYT", "FT", "AP", "AFP", "MONEY", "STOCK", "STOCKS", "MARKET",
    "DEAL", "DEALS", "JUNE", "JULY",
}


def _extract_tickers(title: str) -> list[str]:
    return [m for m in TICKER_RE.findall(title or "") if m not in STOP and len(m) >= 2]


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


def compute() -> dict:
    now = datetime.now(timezone.utc)
    cutoff_prior = now - timedelta(minutes=PRIOR_WINDOW_MIN)

    conn = sqlite3.connect(f"file:{_get_db_path()}?mode=ro", uri=True)
    conn.execute("PRAGMA busy_timeout=8000")
    rows = conn.execute(
        f"SELECT first_seen, title, ml_score FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        f"ORDER BY first_seen DESC LIMIT ?",
        (FETCH_LIMIT,),
    ).fetchall()
    conn.close()

    quiet_cutoff = now - timedelta(minutes=QUIET_WINDOW_MIN)

    # Bucket mentions into recent (<30m) and prior (30–120m)
    recent: Counter[str] = Counter()
    prior: Counter[str] = Counter()
    prior_scores: dict[str, list[float]] = defaultdict(list)

    for raw_ts, title, ml in rows:
        ts = _parse_ts(raw_ts)
        if ts is None or ts < cutoff_prior:
            continue
        tickers = _extract_tickers(title)
        if ts >= quiet_cutoff:
            recent.update(tickers)
        else:
            prior.update(tickers)
            if ml is not None:
                for t in tickers:
                    prior_scores[t].append(float(ml))

    # Tickers with enough prior activity but gone quiet
    gaps = []
    for ticker, prior_n in prior.most_common():
        if prior_n < MIN_PRIOR_MENTIONS:
            break  # Counter.most_common is sorted; anything below is also < MIN
        if recent.get(ticker, 0) == 0:
            scores = prior_scores.get(ticker, [])
            gaps.append({
                "ticker": ticker,
                "prior_mentions": prior_n,
                "recent_mentions": 0,
                "avg_ml_score": round(sum(scores) / len(scores), 2) if scores else None,
            })

    gaps.sort(key=lambda x: -x["prior_mentions"])

    result = {
        "generated_at": now.isoformat(timespec="seconds"),
        "prior_window_min": PRIOR_WINDOW_MIN,
        "quiet_window_min": QUIET_WINDOW_MIN,
        "min_prior_mentions": MIN_PRIOR_MENTIONS,
        "rows_scanned": len(rows),
        "gaps_found": len(gaps),
        "gaps": gaps[:20],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    r = compute()
    print(
        f"scanned={r['rows_scanned']} gaps={r['gaps_found']} "
        f"(prior {r['prior_window_min']}m / quiet {r['quiet_window_min']}m)"
    )
    for g in r["gaps"][:5]:
        print(
            f"  {g['ticker']}: prior={g['prior_mentions']} recent=0 "
            f"ml_avg={g['avg_ml_score']}"
        )
    if not r["gaps"]:
        print("  (no coverage gaps detected)")


if __name__ == "__main__":
    main()
