"""OpEx-news overlap detector.

For each upcoming options expiration within 30 days, find tickers with
heavy recent news coverage (>= 3 mentions in last 24h) that could face
amplified volatility around expiration.

Output: /home/zeph/logs/opex_news_overlap.json
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/opex_news_overlap.json")
HORIZON_DAYS = 30
LOOKBACK_HOURS = 24
MIN_MENTIONS = 3
FETCH_LIMIT = 6000

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

_TRIPLE_WITCHING_MONTHS = {3, 6, 9, 12}


def _third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    days_until_friday = (4 - d.weekday()) % 7
    first_fri = d + timedelta(days=days_until_friday)
    return first_fri + timedelta(weeks=2)


def _opex_type(d: date) -> str:
    if d.weekday() != 4:
        return ""
    if d.month in _TRIPLE_WITCHING_MONTHS and d == _third_friday(d.year, d.month):
        return "triple_witching"
    if d == _third_friday(d.year, d.month):
        return "monthly"
    return "weekly"


def upcoming_opex(today: date, horizon: int) -> list[dict]:
    end = today + timedelta(days=horizon)
    events = []
    seen: set[date] = set()
    d = today + timedelta(days=1)
    while d <= end:
        if d.weekday() == 4 and d not in seen:
            t = _opex_type(d)
            if t:
                events.append({"date": d, "opex_type": t, "days_away": (d - today).days})
                seen.add(d)
        d += timedelta(days=1)
    return sorted(events, key=lambda x: x["date"])


def _parse_ts(raw: str | None) -> datetime | None:
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


def extract_tickers(text: str | None) -> list[str]:
    if not text:
        return []
    return [
        m for m in TICKER_RE.findall(text)
        if m not in STOP and len(m) >= 2
    ]


def main() -> int:
    today = date.today()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=LOOKBACK_HOURS)

    opex_events = upcoming_opex(today, HORIZON_DAYS)
    if not opex_events:
        print("opex_news_overlap: no upcoming OpEx events found")
        return 0

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")

    # Fetch recent articles
    cur = conn.execute(
        "SELECT first_seen, title, ai_score "
        "FROM articles "
        "WHERE source NOT LIKE 'backtest_run_%' "
        "ORDER BY first_seen DESC LIMIT ?",
        (FETCH_LIMIT,),
    )
    rows = cur.fetchall()
    conn.close()

    # Count mentions and accumulate scores per ticker
    mentions: Counter[str] = Counter()
    score_sum: dict[str, float] = defaultdict(float)
    score_cnt: dict[str, int] = defaultdict(int)

    for fs, title, ai_score in rows:
        ts = _parse_ts(fs)
        if ts is None or ts < cutoff:
            continue
        tickers = extract_tickers(title)
        for tk in tickers:
            mentions[tk] += 1
            if ai_score is not None:
                try:
                    score_sum[tk] += float(ai_score)
                    score_cnt[tk] += 1
                except (TypeError, ValueError):
                    pass

    # Find tickers with enough mentions
    hot_tickers = {
        tk: cnt for tk, cnt in mentions.items() if cnt >= MIN_MENTIONS
    }

    if not hot_tickers:
        print("opex_news_overlap: no hot tickers in last 24h")
        payload = {
            "generated_at": now.isoformat(),
            "lookback_hours": LOOKBACK_HOURS,
            "horizon_days": HORIZON_DAYS,
            "signals": [],
            "opex_events": [
                {"date": str(e["date"]), "opex_type": e["opex_type"], "days_away": e["days_away"]}
                for e in opex_events
            ],
        }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(payload, indent=2))
        return 0

    # Cross-reference: for each hot ticker, find nearest OpEx within 7 days
    signals = []
    for tk, cnt in sorted(hot_tickers.items(), key=lambda x: -x[1]):
        avg_score = round(score_sum[tk] / score_cnt[tk], 2) if score_cnt[tk] else None
        # nearest OpEx within 7 days
        near = [e for e in opex_events if e["days_away"] <= 7]
        # or just closest overall
        closest = opex_events[0] if opex_events else None
        if near:
            event = near[0]
        elif closest:
            event = closest
        else:
            continue
        signals.append({
            "ticker": tk,
            "mention_count": cnt,
            "avg_ai_score": avg_score,
            "opex_event": {
                "type": event["opex_type"],
                "date": str(event["date"]),
                "days_away": event["days_away"],
            },
            "risk_flag": event["opex_type"] == "triple_witching" or event["days_away"] <= 3,
        })

    payload = {
        "generated_at": now.isoformat(),
        "lookback_hours": LOOKBACK_HOURS,
        "horizon_days": HORIZON_DAYS,
        "hot_tickers_found": len(hot_tickers),
        "signals": signals[:20],
        "opex_events": [
            {"date": str(e["date"]), "opex_type": e["opex_type"], "days_away": e["days_away"]}
            for e in opex_events
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    print(f"opex_news_overlap: {len(hot_tickers)} hot tickers | next OpEx: {opex_events[0]['date']} ({opex_events[0]['opex_type']}, {opex_events[0]['days_away']}d away)")
    for s in signals[:5]:
        flag = " [RISK]" if s["risk_flag"] else ""
        print(f"  {s['ticker']}: {s['mention_count']} mentions | score={s['avg_ai_score']} | {s['opex_event']['type']} in {s['opex_event']['days_away']}d{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
