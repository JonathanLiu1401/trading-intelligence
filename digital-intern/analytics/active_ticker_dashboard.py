"""Active ticker dashboard — real-time per-ticker signal summary.

For every ticker mentioned in live articles from the last WINDOW_HOURS, builds
a compact summary row: mention count, avg ml_score, avg ai_score, distinct
source count, urgency hits, and a composite heat score.

Heat score = (mentions * 0.4) + (avg_ml * 0.3) + (source_diversity * 0.2) +
             (urgent_hits * 1.5) — each term normalised to [0, 1] across the
             result set, so the score is relative, not absolute.

Why this module when trend_velocity / ticker_alert_ranker already exist?
  - trend_velocity reports a velocity *ratio* (recent vs prior window) but
    no scores, source diversity, or urgency.
  - ticker_alert_ranker reads *stale* JSON outputs from multiple modules
    (written at different times), so it lags by up to 1h.
  - This module reads the DB directly in one pass, producing a live snapshot
    that the command-centre dashboard and paper-trader can query.

Design constraints (mirrors trend_velocity / source_cadence_anomaly):
  - Never full-table scan.  Reads at most FETCH_LIMIT rows via idx_first_seen.
  - busy_timeout=8000 ms; read-only URI connection — zero write-lock pressure.
  - DB is USB-backed (~1.7 GB); query is bounded + indexed.

Output: /home/zeph/logs/active_tickers.json
Standalone: python3 -m analytics.active_ticker_dashboard
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT_PATH = Path("/home/zeph/logs/active_tickers.json")
WINDOW_HOURS = 2
FETCH_LIMIT = 6000
TOP_N = 20

TICKER_RE = re.compile(r"\b\$?([A-Z]{2,5})\b")
_STOP = frozenset({
    "CEO", "CFO", "CTO", "COO", "USA", "USD", "EUR", "GBP", "EU", "UK", "US",
    "AI", "ML", "API", "IPO", "ETF", "SEC", "FOMC", "FED", "GDP", "CPI",
    "PPI", "ECB", "BOJ", "PBOC", "OPEC", "NYSE", "NASDAQ", "AMEX",
    "Q1", "Q2", "Q3", "Q4", "YTD", "YOY", "EPS", "PE", "EV", "ESG",
    "BUY", "SELL", "HOLD", "ON", "AT", "IN", "TO", "OF", "FOR", "THE",
    "AND", "OR", "BY", "AS", "IS", "WAS", "ARE", "BE", "AN", "A",
    "NEW", "OLD", "TOP", "LOW", "HIGH", "BIG", "DAY", "WEEK", "MONTH",
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
    "II", "III", "IV", "VI", "VS", "PM", "AM",
    "NEWS", "INC", "LLC", "LTD", "CORP", "CO", "PLC",
    "MSN", "CNN", "BBC", "WSJ", "NYT", "FT", "AP", "AFP",
    "MONEY", "STOCK", "STOCKS", "MARKET", "DEAL", "DEALS",
    "JUNE", "JULY", "REPORT", "UPDATE", "WATCH", "LIVE", "ALERT",
    "DATA", "IT", "NOT", "NO", "YES", "UP", "DOWN", "OUT", "IF",
})


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
    tickers = []
    for m in TICKER_RE.findall(title or ""):
        if m not in _STOP and len(m) >= 2:
            tickers.append(m)
    return tickers


def _normalise(values: list[float]) -> list[float]:
    """Min-max normalise to [0, 1]."""
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def build(rows: list[tuple]) -> list[dict]:
    """Pure builder — rows = (first_seen, title, source, ml_score, ai_score, urgency).

    Returns sorted list of per-ticker dicts, most active first.
    """
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - WINDOW_HOURS * 3600

    stats: dict[str, dict] = defaultdict(lambda: {
        "mentions": 0,
        "ml_scores": [],
        "ai_scores": [],
        "sources": set(),
        "urgent_hits": 0,
    })

    for first_seen_raw, title, source, ml_score, ai_score, urgency in rows:
        dt = _parse_ts(first_seen_raw)
        if dt is None or dt.timestamp() < cutoff:
            continue
        tickers = _extract_tickers(title)
        for ticker in tickers:
            s = stats[ticker]
            s["mentions"] += 1
            if ml_score is not None:
                s["ml_scores"].append(float(ml_score))
            if ai_score is not None:
                s["ai_scores"].append(float(ai_score))
            if source:
                s["sources"].add(source)
            if (urgency or 0) >= 2:
                s["urgent_hits"] += 1

    if not stats:
        return []

    # Build rows sorted by mention count
    tickers_sorted = sorted(stats.items(), key=lambda x: x[1]["mentions"], reverse=True)
    tickers_sorted = tickers_sorted[:TOP_N]

    records = []
    for ticker, s in tickers_sorted:
        ml_list = s["ml_scores"]
        ai_list = s["ai_scores"]
        records.append({
            "ticker": ticker,
            "mentions": s["mentions"],
            "avg_ml_score": round(sum(ml_list) / len(ml_list), 4) if ml_list else None,
            "avg_ai_score": round(sum(ai_list) / len(ai_list), 4) if ai_list else None,
            "source_diversity": len(s["sources"]),
            "urgent_hits": s["urgent_hits"],
            "_mentions_raw": s["mentions"],
            "_avg_ml_raw": sum(ml_list) / len(ml_list) if ml_list else 0.0,
            "_src_raw": len(s["sources"]),
            "_urg_raw": s["urgent_hits"],
        })

    # Heat score — normalise each dimension then weight
    n_mentions = _normalise([r["_mentions_raw"] for r in records])
    n_ml = _normalise([r["_avg_ml_raw"] for r in records])
    n_src = _normalise([r["_src_raw"] for r in records])
    n_urg = _normalise([r["_urg_raw"] for r in records])

    for i, r in enumerate(records):
        heat = (
            n_mentions[i] * 0.40
            + n_ml[i] * 0.25
            + n_src[i] * 0.20
            + n_urg[i] * 0.15
        )
        r["heat_score"] = round(heat, 4)

    # Strip private keys
    for r in records:
        for k in list(r.keys()):
            if k.startswith("_"):
                del r[k]

    records.sort(key=lambda r: r["heat_score"], reverse=True)
    return records


def main() -> None:
    db_path = str(_get_db_path())
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15)
    conn.execute("PRAGMA busy_timeout=8000")

    cur = conn.execute(
        "SELECT first_seen, title, source, ml_score, ai_score, urgency "
        f"FROM articles WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (FETCH_LIMIT,),
    )
    rows = cur.fetchall()
    conn.close()

    result = build(rows)

    now_str = datetime.now(timezone.utc).isoformat()
    snapshot = {
        "generated_at": now_str,
        "window_hours": WINDOW_HOURS,
        "articles_scanned": len(rows),
        "tickers_found": len(result),
        "tickers": result,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(snapshot, indent=2))

    if result:
        top = result[0]
        print(
            f"active_tickers: {len(result)} tickers in {WINDOW_HOURS}h window "
            f"({len(rows)} articles scanned)"
        )
        print(f"  top: {top['ticker']}  mentions={top['mentions']}  "
              f"heat={top['heat_score']}  "
              f"avg_ml={top['avg_ml_score']}  sources={top['source_diversity']}")
        for r in result[1:6]:
            print(f"  {r['ticker']:6s}  mentions={r['mentions']}  "
                  f"heat={r['heat_score']}  ml={r['avg_ml_score']}")
    else:
        print(f"active_tickers: no tickers found in last {WINDOW_HOURS}h "
              f"({len(rows)} articles scanned)")

    print(f"  written → {OUT_PATH}")


if __name__ == "__main__":
    main()
