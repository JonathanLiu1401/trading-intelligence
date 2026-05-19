"""Source diversity index: per-ticker breadth of independent outlets.

Existing analytics measure *how much* a ticker is mentioned (trend_velocity)
or whether held names dominate (ticker_concentration). They don't separate
broad-market interest from single-feed echo: 30 mentions of $XYZ from one
RSS feed looks the same as 30 mentions across 30 outlets, but only the
latter is a real signal.

This analytic ranks tickers by *distinct sources* covering them in the
last WINDOW_HOURS, and flags the low-diversity-but-high-volume cases as
likely echo. Output complements consensus_signal: consensus says "many
articles agree", diversity says "those articles are not all the same
voice".

Output: /home/zeph/logs/source_diversity.json
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/source_diversity.json")
WINDOW_HOURS = 4
FETCH_LIMIT = 6000
TOP_N = 15
ECHO_MIN_MENTIONS = 5  # mentions threshold for the echo-warning lane

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
    "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
    "II", "III", "IV", "VI",
    "NEWS", "INC", "LLC", "LTD", "CORP", "CO", "PLC",
    "MSN", "CNN", "BBC", "WSJ", "NYT", "FT", "AP", "AFP",
    "MONEY", "STOCK", "STOCKS", "MARKET", "DEAL", "DEALS",
    "JUNE", "JULY",
}


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = str(raw).strip().replace("Z", "+00:00")
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
    return [m for m in TICKER_RE.findall(title or "")
            if m not in STOP and len(m) >= 2]


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    rows = conn.execute(
        "SELECT first_seen, source, title FROM articles "
        "WHERE source NOT LIKE 'backtest_run_%' "
        "ORDER BY first_seen DESC LIMIT ?",
        (FETCH_LIMIT,),
    ).fetchall()
    conn.close()

    if not rows:
        print("source_diversity: no rows", file=sys.stderr)
        return 1

    cutoff = datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)
    # ticker -> {source -> mention_count}
    tk_src: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for fs, source, title in rows:
        ts = _parse_ts(fs)
        if ts is None or ts < cutoff:
            continue
        src = (source or "unknown").strip().lower()
        for tk in _extract_tickers(title):
            tk_src[tk][src] += 1

    if not tk_src:
        print("source_diversity: no in-window mentions", file=sys.stderr)
        return 1

    records = []
    for tk, srcs in tk_src.items():
        mentions = sum(srcs.values())
        distinct = len(srcs)
        top_src, top_n = max(srcs.items(), key=lambda kv: kv[1])
        records.append({
            "ticker": tk,
            "mentions": mentions,
            "distinct_sources": distinct,
            "diversity_ratio": round(distinct / mentions, 3),
            "top_source": top_src,
            "top_source_share": round(top_n / mentions, 3),
            "echo_risk": (
                mentions >= ECHO_MIN_MENTIONS
                and (top_n / mentions) >= 0.7
                and distinct <= 2
            ),
        })

    records.sort(key=lambda r: (-r["distinct_sources"], -r["mentions"]))
    diverse = records[:TOP_N]
    echo = [r for r in records if r["echo_risk"]][:TOP_N]

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_hours": WINDOW_HOURS,
        "fetch_limit": FETCH_LIMIT,
        "tickers_scored": len(records),
        "top_diverse": diverse,
        "echo_warnings": echo,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    # Stdout summary (operator-facing)
    print(f"source_diversity: scored {len(records)} tickers, "
          f"{len(echo)} echo-risk")
    for r in diverse[:5]:
        print(f"  diverse {r['ticker']:>5} mentions={r['mentions']:<3} "
              f"sources={r['distinct_sources']} top={r['top_source']}"
              f"({r['top_source_share']})")
    for r in echo[:5]:
        print(f"  echo    {r['ticker']:>5} mentions={r['mentions']:<3} "
              f"sources={r['distinct_sources']} top={r['top_source']}"
              f"({r['top_source_share']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
