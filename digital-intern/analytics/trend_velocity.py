"""Trend velocity: tickers gaining mentions fastest in last 2h vs prior 2h."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/trend_velocity.json")
WINDOW_HOURS = 2
FETCH_LIMIT = 4000
TOP_N = 5

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


def extract_tickers(title: str) -> list[str]:
    out = []
    for m in TICKER_RE.findall(title or ""):
        if m in STOP or len(m) < 2:
            continue
        out.append(m)
    return out


def fetch_recent(conn: sqlite3.Connection, limit: int) -> list[tuple[str, str]]:
    cur = conn.execute(
        "SELECT first_seen, title FROM articles "
        "WHERE source NOT LIKE 'backtest_run_%' "
        "ORDER BY first_seen DESC LIMIT ?",
        (limit,),
    )
    return cur.fetchall()


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    rows = fetch_recent(conn, FETCH_LIMIT)
    if not rows:
        print("trend_velocity: no rows", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    cur_cut = now - timedelta(hours=WINDOW_HOURS)
    prev_cut = now - timedelta(hours=WINDOW_HOURS * 2)

    cur_c: Counter[str] = Counter()
    prev_c: Counter[str] = Counter()
    for fs, title in rows:
        ts = _parse_ts(fs)
        if ts is None:
            continue
        tix = extract_tickers(title)
        if not tix:
            continue
        if ts >= cur_cut:
            cur_c.update(tix)
        elif ts >= prev_cut:
            prev_c.update(tix)

    movers = []
    for tk, c in cur_c.items():
        p = prev_c.get(tk, 0)
        velocity = c - p
        ratio = (c + 1) / (p + 1)
        movers.append((tk, c, p, velocity, ratio))

    movers.sort(key=lambda r: (r[3], r[4]), reverse=True)
    top = movers[:TOP_N]

    payload = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "fetched": len(rows),
        "tickers_now": sum(cur_c.values()),
        "tickers_prev": sum(prev_c.values()),
        "top": [
            {"ticker": tk, "now": c, "prev": p, "delta": v, "ratio": round(r, 2)}
            for tk, c, p, v, r in top
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    print(f"trend_velocity: scanned={len(rows)} now_mentions={sum(cur_c.values())} prev={sum(prev_c.values())}")
    for tk, c, p, v, r in top:
        print(f"  {tk}: now={c} prev={p} delta=+{v} ratio={r:.2f}x")
    if not top:
        print("  (no ticker movers in window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
