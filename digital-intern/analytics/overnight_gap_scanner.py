"""Overnight Gap Scanner.

Identifies tickers with high-urgency (urgency >= 2) or high-ml_score articles
published during market-closed hours (after 4:00 PM ET / before 9:30 AM ET).
These are the articles most likely to cause gap moves at the next open.

Design: bounded idx_first_seen scan, read-only, USB-safe.

Two consumers share the same logic:
  * the CLI (``python3 -m analytics.overnight_gap_scanner``) writes the ranked
    digest to ``/home/zeph/logs/overnight_gaps.json``;
  * the dashboard endpoint ``/api/overnight-gaps`` calls ``build_overnight_gaps``
    directly on a live ``_ro_query`` so the operator sees pre-open gap risk
    without waiting for the next CLI run.

``build_overnight_gaps`` is the single source of truth — pure, never raises,
no DB and no file I/O. ``main()`` owns the DB read + JSON write and delegates
the ranking to the builder so the CLI and the endpoint can never disagree.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/overnight_gaps.json")
SCAN_LIMIT = 5000
TOP_N = 10
# An article needs at least this much signal to count toward a gap candidate;
# below it the row is overnight noise (urgency 0, near-zero ml_score).
MIN_URGENCY = 1
MIN_ML_SCORE = 0.3
ET = ZoneInfo("America/New_York")

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
_LIVE_ONLY = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)


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


def _is_overnight(dt_utc: datetime) -> bool:
    """Return True if dt falls outside regular market hours in ET."""
    dt_et = dt_utc.astimezone(ET)
    t = dt_et.time()
    # Market hours: 9:30 AM – 4:00 PM ET weekdays
    market_open = dt_et.replace(hour=9, minute=30, second=0, microsecond=0).time()
    market_close = dt_et.replace(hour=16, minute=0, second=0, microsecond=0).time()
    is_weekend = dt_et.weekday() >= 5  # Sat=5, Sun=6
    return is_weekend or t < market_open or t >= market_close


def extract_tickers(title: str) -> list[str]:
    out = []
    for m in TICKER_RE.findall(title or ""):
        if m not in STOP and len(m) >= 2:
            out.append(m)
    return out


def _coerce_urgency(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_ml(value) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def build_overnight_gaps(rows, now: datetime | None = None,
                         top_n: int = TOP_N) -> dict:
    """Pure: rank overnight gap candidates from raw article rows.

    ``rows`` is an iterable of ``(first_seen, title, urgency, ml_score,
    source)`` tuples — the exact projection ``main()`` and the
    ``/api/overnight-gaps`` endpoint both read. Callers are responsible for
    applying the live-only SQL filter; the builder does not see ``url``.

    Returns the JSON-ready digest: ``generated_at``, ``scanned`` (rows in),
    ``overnight_articles_24h`` (rows inside the 24h window that fell in a
    market-closed ET slot), and ``gap_candidates`` (top-N tickers ranked by
    ``max_urgency*2 + count + max_ml``). Pure — no DB, no file I/O, never
    raises on malformed rows (a bad row is skipped, not fatal)."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)
    rows = list(rows or [])

    # ticker -> {count, max_urgency, max_ml, articles[]}
    ticker_data: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "max_urgency": 0, "max_ml": 0.0, "articles": []}
    )
    overnight_count = 0

    for row in rows:
        try:
            first_seen, title, urgency, ml_score, source = row
        except (TypeError, ValueError):
            # A malformed row is skipped — a diagnostic must never raise.
            continue

        ts = _parse_ts(first_seen)
        if ts is None or ts < cutoff:
            continue
        if not _is_overnight(ts):
            continue

        overnight_count += 1
        urg = _coerce_urgency(urgency)
        ml = _coerce_ml(ml_score)

        # Only track articles with some signal — an overnight row at urgency 0
        # and near-zero ml_score is noise, not a gap catalyst.
        if urg < MIN_URGENCY and ml < MIN_ML_SCORE:
            continue

        for tk in extract_tickers(title or ""):
            d = ticker_data[tk]
            d["count"] += 1
            d["max_urgency"] = max(d["max_urgency"], urg)
            d["max_ml"] = max(d["max_ml"], ml)
            if len(d["articles"]) < 3:
                d["articles"].append({
                    "title": title,
                    "source": source,
                    "first_seen": first_seen,
                    "urgency": urg,
                    "ml_score": round(ml, 4),
                })

    # Rank by (max_urgency * 2 + count + max_ml).
    ranked = sorted(
        ticker_data.items(),
        key=lambda kv: (kv[1]["max_urgency"] * 2 + kv[1]["count"] + kv[1]["max_ml"]),
        reverse=True,
    )[:top_n]

    return {
        "generated_at": now.isoformat(),
        "scanned": len(rows),
        "overnight_articles_24h": overnight_count,
        "gap_candidates": [
            {
                "ticker": tk,
                "article_count": d["count"],
                "max_urgency": d["max_urgency"],
                "max_ml_score": round(d["max_ml"], 4),
                "top_articles": d["articles"],
            }
            for tk, d in ranked
        ],
    }


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")

    rows = conn.execute(
        "SELECT first_seen, title, urgency, ml_score, source FROM articles "
        f"WHERE {_LIVE_ONLY} "
        "ORDER BY first_seen DESC LIMIT ?",
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    if not rows:
        print("overnight_gap_scanner: no rows", file=sys.stderr)
        return 1

    result = build_overnight_gaps(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    print(f"overnight_gap_scanner: {result['overnight_articles_24h']} "
          f"overnight articles in last 24h")
    print(f"gap candidates ({len(result['gap_candidates'])}):")
    for item in result["gap_candidates"]:
        print(
            f"  {item['ticker']:6s} count={item['article_count']} "
            f"urgency={item['max_urgency']} ml={item['max_ml_score']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
