"""Market-closed watchlist: ranked high-score articles accumulated while the
US equity market was closed (overnight + weekends), so we can hit the next
open with a pre-prioritised list.

Run window: from the last 4:00 PM ET regular-session close through now.
  - If now (ET) is past 4:00 PM on a weekday, "last close" is today 4:00 PM ET.
  - If now is before 4:00 PM on a weekday, "last close" is the prior trading
    weekday's 4:00 PM ET.
  - If now is on the weekend, "last close" is the most-recent Friday 4:00 PM ET.

Selects articles with ai_score >= 6 first_seen since the last close, groups by
ticker (extracted from the title), and writes a ranked JSON briefing to
/home/zeph/logs/market_closed_watchlist.json.

Standalone: ``python3 -m analytics.market_closed_watchlist``
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from analytics.trend_velocity import STOP, extract_tickers, _parse_ts
from storage.article_store import _LIVE_ONLY_CLAUSE

REPO = Path(__file__).resolve().parents[1]
DB_PATH = REPO / "data" / "articles.db"
WATCHLIST_PATH = REPO / "config" / "watchlist.json"
PORTFOLIO_PATH = REPO / "config" / "portfolio.json"
OUT_PATH = Path("/home/zeph/logs/market_closed_watchlist.json")

ET = ZoneInfo("America/New_York")
CLOSE_TIME_ET = dtime(16, 0)   # 4:00 PM ET regular-session close
OPEN_TIME_ET = dtime(9, 30)    # 9:30 AM ET regular-session open

MIN_AI_SCORE = 6
FETCH_LIMIT = 200
TOP_ARTICLES_N = 10

# Fallback universe used only when both config files are missing / empty.
DEFAULT_TICKERS = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "AMD", "AVGO", "NFLX", "SPY", "QQQ",
}

# Equity-ticker shape: 1-5 uppercase letters, optionally $-prefixed.
EQUITY_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


def _last_market_close(now_et: datetime) -> datetime:
    """Return the most-recent 4:00 PM ET regular-session close, in ET."""
    today_close = now_et.replace(
        hour=CLOSE_TIME_ET.hour, minute=CLOSE_TIME_ET.minute,
        second=0, microsecond=0,
    )
    wd = now_et.weekday()  # Mon=0 ... Sun=6

    if wd == 5:  # Saturday -> last Friday's close
        return today_close - timedelta(days=1)
    if wd == 6:  # Sunday -> last Friday's close
        return today_close - timedelta(days=2)

    # Weekday
    if now_et >= today_close:
        return today_close

    # Before today's close: step back one weekday (skip weekend)
    days_back = 3 if wd == 0 else 1   # Mon AM -> last Fri
    return today_close - timedelta(days=days_back)


def _next_market_open(after_et: datetime) -> datetime:
    """Return the next 9:30 AM ET regular-session open strictly after ``after_et``."""
    candidate = after_et.replace(
        hour=OPEN_TIME_ET.hour, minute=OPEN_TIME_ET.minute,
        second=0, microsecond=0,
    )
    if candidate <= after_et:
        candidate += timedelta(days=1)
    # Skip Sat (5) and Sun (6)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _load_ticker_universe() -> set[str]:
    tix: set[str] = set()

    try:
        wl = json.loads(WATCHLIST_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        wl = {}
    for v in wl.values() if isinstance(wl, dict) else []:
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str) and EQUITY_TICKER_RE.match(item):
                    tix.add(item.upper())

    try:
        pf = json.loads(PORTFOLIO_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        pf = {}
    for pos in pf.get("positions", []) if isinstance(pf, dict) else []:
        t = (pos or {}).get("ticker")
        if isinstance(t, str) and EQUITY_TICKER_RE.match(t):
            tix.add(t.upper())
    for opt in pf.get("options", []) if isinstance(pf, dict) else []:
        t = (opt or {}).get("underlying")
        if isinstance(t, str) and EQUITY_TICKER_RE.match(t):
            tix.add(t.upper())

    if not tix:
        tix = set(DEFAULT_TICKERS)
    # Always allow large index ETFs even if config drops them.
    tix.update({"SPY", "QQQ"})
    # Remove any stopwords accidentally captured.
    tix -= STOP
    return tix


def _extract_article_tickers(title: str, universe: set[str]) -> list[str]:
    """Return tickers from a title.

    Accepts (a) explicit $TICKER cashtags (regardless of universe) and
    (b) bareword TICKERs that appear in our known universe (to avoid the
    sea of false positives from STOP-style 2-3-letter words).
    """
    out: set[str] = set()
    # Cashtags: $NVDA, $TSLA
    for m in re.findall(r"\$([A-Z]{1,5})\b", title or ""):
        if m in STOP:
            continue
        out.add(m)
    # Bareword tickers via shared trend_velocity extractor (already strips STOP)
    for tk in extract_tickers(title or ""):
        if tk in universe:
            out.add(tk)
    return sorted(out)


def main() -> int:
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)
    last_close_et = _last_market_close(now_et)
    last_close_utc = last_close_et.astimezone(timezone.utc)
    next_open_et = _next_market_open(now_et)

    # SQL boundary uses naive UTC 'YYYY-MM-DD HH:MM:SS' so it lines up with
    # replace(first_seen,'T',' ') string-comparison semantics (the ISO 'T' vs
    # space mismatch the project memory flagged).
    since_sql = last_close_utc.strftime("%Y-%m-%d %H:%M:%S")

    universe = _load_ticker_universe()

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, url, title, source, ai_score, first_seen "
        "FROM articles "
        "WHERE replace(first_seen,'T',' ') >= ? "
        "  AND ai_score >= ? "
        f"  AND {_LIVE_ONLY_CLAUSE} "
        "ORDER BY ai_score DESC, first_seen DESC "
        "LIMIT ?",
        (since_sql, MIN_AI_SCORE, FETCH_LIMIT),
    ).fetchall()
    conn.close()

    # Group by ticker
    by_ticker: dict[str, list[sqlite3.Row]] = defaultdict(list)
    untagged = 0
    for r in rows:
        tix = _extract_article_tickers(r["title"], universe)
        if not tix:
            untagged += 1
            continue
        for tk in tix:
            by_ticker[tk].append(r)

    tickers_out = []
    for tk, items in by_ticker.items():
        scores = [float(it["ai_score"]) for it in items]
        # Pick the single highest-score article as the "top" for that ticker
        top_item = max(items, key=lambda it: (float(it["ai_score"]),
                                              it["first_seen"]))
        # Earliest first_seen across this ticker in the window
        first_seen_min = min(
            (_parse_ts(it["first_seen"]) for it in items),
            default=None,
            key=lambda d: d if d is not None else datetime.max.replace(tzinfo=timezone.utc),
        )
        tickers_out.append({
            "ticker": tk,
            "article_count": len(items),
            "max_score": round(max(scores), 2),
            "avg_score": round(sum(scores) / len(scores), 2),
            "top_title": top_item["title"],
            "top_source": top_item["source"] or "",
            "first_seen": (first_seen_min.isoformat()
                           if first_seen_min is not None else None),
        })
    tickers_out.sort(key=lambda x: (-x["max_score"], -x["article_count"], x["ticker"]))

    top_articles = []
    for r in sorted(rows,
                    key=lambda x: (-float(x["ai_score"]), x["first_seen"]))[:TOP_ARTICLES_N]:
        top_articles.append({
            "id": r["id"],
            "title": r["title"],
            "source": r["source"] or "",
            "ai_score": round(float(r["ai_score"]), 2),
            "first_seen": r["first_seen"],
            "url": r["url"] or "",
        })

    payload = {
        "generated_at": now_utc.isoformat(),
        "market_closed_since": last_close_et.isoformat(),
        "next_market_open": next_open_et.isoformat(),
        "article_count_since_close": len(rows),
        "untagged_article_count": untagged,
        "ticker_universe_size": len(universe),
        "tickers": tickers_out,
        "top_articles": top_articles,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    print(
        f"market_closed_watchlist: since={last_close_et.isoformat()} "
        f"articles={len(rows)} tickers={len(tickers_out)} "
        f"untagged={untagged} next_open={next_open_et.isoformat()}"
    )
    for t in tickers_out[:10]:
        print(f"  {t['ticker']}: max={t['max_score']} avg={t['avg_score']} "
              f"n={t['article_count']} top='{t['top_title'][:60]}'")
    if not tickers_out:
        print("  (no tagged tickers in window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
