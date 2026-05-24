"""OpEx-news overlap detector.

Cross-references upcoming options-expiration events (next 30 days) with the
most-mentioned tickers in the last 24h of articles.db, emitting a JSON
signal feed of tickers that have BOTH recent news momentum (>= 3 mentions)
AND an OpEx event within 7 days.

SCHEMA NOTE (read this — important):
    The task spec assumed a `tickers` column on the `articles` table.
    The live DB does NOT have that column (PRAGMA table_info(articles)
    returns: id,url,title,source,published,kw_score,ai_score,urgency,
    full_text,first_seen,cycle,time_sensitivity,ml_score,score_source).
    We attempt the spec query first inside a try/except — when it raises
    OperationalError("no such column: tickers") we transparently fall
    back to extracting tickers from `title` via the same TICKER_RE regex
    used by analytics/trend_velocity.py. The "JSON list vs CSV" parsing
    branch is therefore dead in the current schema but kept in case the
    column is added later.

OpEx date logic is the collector's own _upcoming_opex_dates() — imported
directly from collectors/opex_calendar_collector.py so the two stay in
lock-step (3rd Friday detection for monthly + triple witching in
Mar/Jun/Sep/Dec; other Fridays = weekly).
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

# Re-use the collector's exact date logic so the two stay in lock-step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collectors.opex_calendar_collector import _upcoming_opex_dates  # noqa: E402

try:
    from storage.article_store import _LIVE_ONLY_CLAUSE  # type: ignore
except Exception:  # pragma: no cover — fail-soft if storage layout shifts
    _LIVE_ONLY_CLAUSE = "source NOT LIKE 'backtest_run_%'"

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/opex_news_overlap.json")

HORIZON_DAYS = 30           # OpEx look-ahead (per spec)
OPEX_PROXIMITY_DAYS = 7     # only report tickers with OpEx within this window
MIN_MENTIONS = 3            # spec threshold
TOP_FETCH_LIMIT = 200       # spec LIMIT 200 (only used on the tickers-column path)

# Same TICKER_RE / stoplist as trend_velocity.py.
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
    "JUNE", "JULY", "MARCH", "APRIL", "AUGUST", "OCTOBER",
    "NOVEMBER", "DECEMBER", "JANUARY", "FEBRUARY",
}


def _parse_tickers_field(raw: str) -> list[str]:
    """Parse the `tickers` column value — handles JSON list OR CSV.

    Kept for forward-compat: if the schema is later extended to include a
    `tickers` column populated by the labeller, this function does the
    parsing the spec asked for.
    """
    if not raw:
        return []
    s = raw.strip()
    if s.startswith("["):
        try:
            data = json.loads(s)
            if isinstance(data, list):
                return [str(x).strip().upper() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return [tok.strip().upper() for tok in s.split(",") if tok.strip()]


def _extract_from_title(title: str) -> list[str]:
    out: list[str] = []
    for m in TICKER_RE.findall(title or ""):
        if m in STOP or len(m) < 2:
            continue
        out.append(m)
    return out


def _read_only_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    return conn


def _query_tickers_column(conn: sqlite3.Connection) -> tuple[bool, list[tuple]]:
    """Attempt the literal spec query. Returns (success, rows).

    Two corrections vs the literal spec (preserved here, not silently
    swallowed): SQLite's modifier syntax is '-24 hours' not '-24h' (the
    latter returns NULL), and `first_seen` mixes 'T' and space separators
    so we normalise with replace() — same pattern as MEMORY.md
    "Intern healthcheck false negative".
    """
    sql = (
        "SELECT tickers, COUNT(*), AVG(CAST(ai_score AS REAL)) "
        "FROM articles "
        "WHERE replace(first_seen,'T',' ') >= datetime('now','-24 hours') "
        "  AND tickers IS NOT NULL AND tickers != '' "
        f"  AND {_LIVE_ONLY_CLAUSE} "
        "GROUP BY tickers "
        "ORDER BY COUNT(*) DESC "
        f"LIMIT {TOP_FETCH_LIMIT}"
    )
    try:
        rows = conn.execute(sql).fetchall()
        return True, rows
    except sqlite3.OperationalError as e:
        print(f"[opex_news_overlap] tickers-column path unavailable: {e}", file=sys.stderr)
        return False, []


def _aggregate_from_tickers_column(rows: list[tuple]) -> dict[str, dict]:
    """Per-individual-ticker aggregation from the (tickers, count, avg_ai)
    rows produced by the spec query."""
    counts: Counter[str] = Counter()
    score_sum: dict[str, float] = defaultdict(float)
    score_n: dict[str, int] = defaultdict(int)
    for raw_tk, cnt, avg_ai in rows:
        cnt = int(cnt or 0)
        for tk in _parse_tickers_field(raw_tk):
            if tk in STOP or len(tk) < 2:
                continue
            counts[tk] += cnt
            # ai_score default is 0 (not NULL); treat 0 as "unscored" so we
            # don't drag the average down with un-AI-scored rows.
            if avg_ai is not None and float(avg_ai) > 0:
                score_sum[tk] += float(avg_ai) * cnt
                score_n[tk] += cnt
    return {
        tk: {
            "mention_count": c,
            "avg_ai_score": round(score_sum[tk] / score_n[tk], 3) if score_n[tk] else None,
        }
        for tk, c in counts.items()
    }


def _aggregate_from_titles(conn: sqlite3.Connection) -> dict[str, dict]:
    """Fallback aggregation: extract tickers from `title` via regex, count
    per-individual-ticker, average ai_score (skipping NULL and the 0
    sentinel that means 'never scored')."""
    sql = (
        "SELECT title, ai_score FROM articles "
        "WHERE replace(first_seen,'T',' ') >= datetime('now','-24 hours') "
        f"  AND {_LIVE_ONLY_CLAUSE} "
        "  AND title IS NOT NULL AND title != ''"
    )
    counts: Counter[str] = Counter()
    score_sum: dict[str, float] = defaultdict(float)
    score_n: dict[str, int] = defaultdict(int)
    for title, ai in conn.execute(sql):
        tix = _extract_from_title(title)
        if not tix:
            continue
        # de-dup within a single title so "NVDA NVDA earnings" counts once
        for tk in set(tix):
            counts[tk] += 1
            if ai is None:
                continue
            try:
                val = float(ai)
            except (TypeError, ValueError):
                continue
            if val > 0:
                score_sum[tk] += val
                score_n[tk] += 1
    return {
        tk: {
            "mention_count": c,
            "avg_ai_score": round(score_sum[tk] / score_n[tk], 3) if score_n[tk] else None,
        }
        for tk, c in counts.items()
    }


def _nearest_opex(events: list[dict], today: date) -> dict | None:
    """Return the soonest OpEx event within OPEX_PROXIMITY_DAYS, or None."""
    for ev in events:  # collector helper returns them sorted ascending
        delta = (ev["date"] - today).days
        if 0 <= delta <= OPEX_PROXIMITY_DAYS:
            return {
                "type": ev["opex_type"],
                "date": ev["date"].isoformat(),
                "days_away": delta,
            }
    return None


def main() -> int:
    now = datetime.now(timezone.utc)
    today = now.date()

    events = _upcoming_opex_dates(today, HORIZON_DAYS)
    nearest = _nearest_opex(events, today)

    conn = _read_only_conn()
    try:
        used_tickers_column, rows = _query_tickers_column(conn)
        if used_tickers_column:
            per_ticker = _aggregate_from_tickers_column(rows)
            source_path = "tickers_column"
        else:
            per_ticker = _aggregate_from_titles(conn)
            source_path = "title_extraction_fallback"
    finally:
        conn.close()

    # A ticker becomes a signal iff:
    #   - mentioned >= MIN_MENTIONS in last 24h
    #   - there's an OpEx event within OPEX_PROXIMITY_DAYS (any qualifies
    #     for now — the collector treats all 3 OpEx classes as macro-level
    #     events; per-ticker expiry filters could be added later).
    signals: list[dict] = []
    if nearest is not None:
        for tk, stat in per_ticker.items():
            if stat["mention_count"] < MIN_MENTIONS:
                continue
            signals.append({
                "ticker": tk,
                "mention_count": stat["mention_count"],
                "avg_ai_score": stat["avg_ai_score"],
                "opex_event": nearest,
            })

    signals.sort(
        key=lambda s: (s["mention_count"], s["avg_ai_score"] or 0.0),
        reverse=True,
    )

    payload = {
        "generated_at": now.isoformat(),
        "horizon_days": HORIZON_DAYS,
        "opex_proximity_days": OPEX_PROXIMITY_DAYS,
        "min_mentions": MIN_MENTIONS,
        "source_path": source_path,
        "upcoming_opex_count": len(events),
        "nearest_opex": nearest,
        "tickers_considered": len(per_ticker),
        "signals": signals,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    # Human-readable summary
    print(
        f"opex_news_overlap: source={source_path} "
        f"opex_events={len(events)} "
        f"tickers_considered={len(per_ticker)} "
        f"signals={len(signals)}"
    )
    if nearest:
        print(
            f"  nearest OpEx: {nearest['type']} on {nearest['date']} "
            f"(in {nearest['days_away']}d)"
        )
    else:
        print(f"  no OpEx event within {OPEX_PROXIMITY_DAYS}d (none of {len(events)} events qualify)")

    if not signals:
        print("  (no tickers met threshold)")
        return 0

    print(f"\n{'TICKER':<8} {'MENTIONS':>9} {'AVG_AI':>8}   OPEX")
    print("-" * 60)
    for s in signals[:25]:
        ai = f"{s['avg_ai_score']:.2f}" if s["avg_ai_score"] is not None else "  n/a"
        ev = s["opex_event"]
        print(
            f"{s['ticker']:<8} {s['mention_count']:>9} {ai:>8}   "
            f"{ev['type']} {ev['date']} (+{ev['days_away']}d)"
        )
    if len(signals) > 25:
        print(f"... and {len(signals) - 25} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
