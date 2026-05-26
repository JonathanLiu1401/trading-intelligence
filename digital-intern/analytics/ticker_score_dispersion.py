"""Ticker score dispersion — per-ticker consensus stability detector.

The sibling diagnostics are:

  * ``sentiment_reversal``         — did a ticker's avg ml_score *flip sign*
    between two 2h windows? (cross-window directional change)
  * ``source_score_volatility``    — std-dev of ai_score *per collector* —
    is a SOURCE noisy? (intra-source magnitude variance)

Neither answers "is the news on this ticker *internally consistent* right now,
or are different articles scoring it wildly differently?" A ticker with five
articles all scoring 7.5–8.0 is consensus-bullish; a ticker with five articles
spread 1.0–9.5 has the same mean but is *contested* news — a fundamentally
different desk read.

Design:
  * Look back ``window_hours`` (default 24h) over live articles only.
  * For each ticker mentioned in ≥ ``min_articles`` titles with a non-NULL
    ml_score, compute n / mean / std-dev / min / max.
  * Bin into TIGHT / MIXED / CONFLICTED by std-dev thresholds.
  * Sort CONFLICTED first (the actionable verdict) then by std-dev DESC.

Public surfaces:
  * ``build_ticker_score_dispersion(articles, window_hours=..., now=...)`` —
    pure builder. ``articles`` is any iterable of dicts with at least
    ``first_seen``, ``title``, ``ml_score``.
  * ``compute()`` — opens articles.db, runs the builder, writes a snapshot.
    Standalone:  ``python3 -m analytics.ticker_score_dispersion``.

Verdict ladder (top-level):
  * NO_DATA       — no rows scanned in window
  * NO_DISPERSION — at least one ticker scanned but none meet min_articles
  * CONSENSUS     — every qualifying ticker is TIGHT
  * MIXED_BOOK    — some tickers MIXED but none CONFLICTED
  * CONFLICTED_NEWS — at least one ticker is CONFLICTED
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
SNAPSHOT_PATH = Path("/home/zeph/logs/ticker_score_dispersion.json")

DEFAULT_WINDOW_HOURS = 24
FETCH_LIMIT = 8000
MIN_ARTICLES = 4         # require enough samples to make std-dev meaningful
TIGHT_STD = 0.75         # std-dev <= this → TIGHT
CONFLICTED_STD = 1.75    # std-dev >  this → CONFLICTED ; in-between → MIXED
TOP_N = 15

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


def _parse_ts(raw) -> datetime | None:
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
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
    return [m for m in TICKER_RE.findall(title or "") if m not in STOP and len(m) >= 2]


def _classify_per_ticker(std: float) -> str:
    if std <= TIGHT_STD:
        return "TIGHT"
    if std > CONFLICTED_STD:
        return "CONFLICTED"
    return "MIXED"


def build_ticker_score_dispersion(
    articles: Iterable[dict],
    window_hours: int = DEFAULT_WINDOW_HOURS,
    now: datetime | None = None,
) -> dict:
    """Pure builder.

    Returns a stable shape:

      {
        "generated_at": iso str,
        "window_hours": int,
        "rows_scanned": int,
        "rows_in_window": int,
        "min_articles_per_ticker": int,
        "tight_std_threshold": float,
        "conflicted_std_threshold": float,
        "verdict": str,
        "n_tickers_qualified": int,
        "n_tight": int,
        "n_mixed": int,
        "n_conflicted": int,
        "tickers": [
            {ticker, n, mean, std, min, max, range, verdict}, ...
        ],
      }
    """
    now = (now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    window_hours = max(1, int(window_hours))
    cutoff = now - timedelta(hours=window_hours)

    per_ticker: dict[str, list[float]] = defaultdict(list)
    rows_scanned = 0
    rows_in_window = 0

    for art in articles:
        rows_scanned += 1
        ts = _parse_ts(art.get("first_seen"))
        if ts is None or ts < cutoff:
            continue
        score = art.get("ml_score")
        if score is None:
            continue
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        rows_in_window += 1
        for ticker in _extract_tickers(art.get("title") or ""):
            per_ticker[ticker].append(score_f)

    tickers_out: list[dict] = []
    n_tight = n_mixed = n_conflicted = 0
    for ticker, scores in per_ticker.items():
        n = len(scores)
        if n < MIN_ARTICLES:
            continue
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / n
        std = math.sqrt(var)
        lo, hi = min(scores), max(scores)
        verdict_t = _classify_per_ticker(std)
        if verdict_t == "TIGHT":
            n_tight += 1
        elif verdict_t == "MIXED":
            n_mixed += 1
        else:
            n_conflicted += 1
        tickers_out.append({
            "ticker": ticker,
            "n": n,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(lo, 4),
            "max": round(hi, 4),
            "range": round(hi - lo, 4),
            "verdict": verdict_t,
        })

    rank = {"CONFLICTED": 0, "MIXED": 1, "TIGHT": 2}
    tickers_out.sort(key=lambda r: (rank[r["verdict"]], -r["std"]))
    tickers_out = tickers_out[:TOP_N]

    if rows_in_window == 0:
        verdict = "NO_DATA"
    elif not tickers_out:
        verdict = "NO_DISPERSION"
    elif n_conflicted > 0:
        verdict = "CONFLICTED_NEWS"
    elif n_mixed > 0:
        verdict = "MIXED_BOOK"
    else:
        verdict = "CONSENSUS"

    return {
        "generated_at": now.isoformat(),
        "window_hours": window_hours,
        "rows_scanned": rows_scanned,
        "rows_in_window": rows_in_window,
        "min_articles_per_ticker": MIN_ARTICLES,
        "tight_std_threshold": TIGHT_STD,
        "conflicted_std_threshold": CONFLICTED_STD,
        "verdict": verdict,
        "n_tickers_qualified": len(tickers_out),
        "n_tight": n_tight,
        "n_mixed": n_mixed,
        "n_conflicted": n_conflicted,
        "tickers": tickers_out,
    }


def _fetch_articles_from_db(
    db_path: Path,
    window_hours: int,
    now: datetime,
) -> list[dict]:
    """Read live articles within the window."""
    cutoff = (now - timedelta(hours=window_hours)).isoformat()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=20)
    try:
        rows = conn.execute(
            "SELECT first_seen, title, ml_score FROM articles "
            f"WHERE {_LIVE_ONLY_CLAUSE} "
            "AND ml_score IS NOT NULL "
            "AND first_seen >= ? "
            "ORDER BY first_seen DESC LIMIT ?",
            (cutoff, FETCH_LIMIT),
        ).fetchall()
    finally:
        conn.close()
    return [
        {"first_seen": r[0], "title": r[1], "ml_score": r[2]}
        for r in rows
    ]


def compute(
    window_hours: int = DEFAULT_WINDOW_HOURS,
    now: datetime | None = None,
    db_path: Path | None = None,
) -> dict:
    """CLI entry — fetch DB, build payload, write snapshot, return it."""
    now = now or datetime.now(timezone.utc)
    db_path = db_path or DB_PATH
    articles = _fetch_articles_from_db(db_path, window_hours, now)
    payload = build_ticker_score_dispersion(
        articles, window_hours=window_hours, now=now)
    try:
        SNAPSHOT_PATH.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass
    return payload


def main() -> int:
    payload = compute()
    print(
        f"verdict={payload['verdict']} qualified={payload['n_tickers_qualified']} "
        f"(tight={payload['n_tight']} mixed={payload['n_mixed']} "
        f"conflicted={payload['n_conflicted']}) "
        f"rows_scanned={payload['rows_scanned']} "
        f"rows_in_window={payload['rows_in_window']}"
    )
    for t in payload["tickers"][:8]:
        print(
            f"  {t['ticker']:6s} {t['verdict']:10s} "
            f"n={t['n']:3d} mean={t['mean']:+.2f} std={t['std']:.2f} "
            f"range=[{t['min']:+.2f}, {t['max']:+.2f}]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
