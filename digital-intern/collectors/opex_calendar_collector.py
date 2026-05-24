"""Options Expiration Calendar collector.

Computes and emits synthetic article rows for upcoming options expiration
dates so they surface in briefings and urgency scoring:

  - Weekly expirations: every Friday
  - Monthly standard expirations: 3rd Friday of each month
  - Quarterly Triple Witching: 3rd Friday of Mar/Jun/Sep/Dec
    (simultaneous expiry of index options, index futures, stock options)

Proximity-based urgency:
  - Triple Witching within 3d or day-of → kw_score 8.0 (high urgency)
  - Monthly OpEx within 2d or day-of   → kw_score 7.0
  - Weekly OpEx day-of or tomorrow     → kw_score 5.5
  - All others                         → kw_score 4.0

Dedup: keyed by (opex_type, YYYY-MM-DD, day_class) in seen_articles.db so
the same event re-emits as proximity sharpens (same contract as
macro_calendar_collector).
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import zlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

SOURCE_NAME = "opex_calendar"
HORIZON_DAYS = 30  # look ahead this many days

log = logging.getLogger("opex_calendar_collector")

# Quarterly Triple Witching months
_TRIPLE_WITCHING_MONTHS = {3, 6, 9, 12}


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY,
            link TEXT,
            title TEXT,
            source TEXT,
            first_seen TEXT
        )"""
    )
    conn.commit()


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of the given year/month."""
    d = date(year, month, 1)
    # Advance to first Friday
    days_until_friday = (4 - d.weekday()) % 7
    first_fri = d + timedelta(days=days_until_friday)
    return first_fri + timedelta(weeks=2)


def _is_triple_witching(d: date) -> bool:
    if d.month not in _TRIPLE_WITCHING_MONTHS:
        return False
    return d == _third_friday(d.year, d.month)


def _is_monthly_opex(d: date) -> bool:
    return d == _third_friday(d.year, d.month)


def _is_weekly_friday(d: date) -> bool:
    return d.weekday() == 4  # Friday


def _day_class(event_date: date, today: date) -> str:
    """Bucket for dedup key — same as macro_calendar_collector contract."""
    delta = (event_date - today).days
    if delta <= 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if delta <= 7:
        return "upcoming"
    return "far"


def _day_prefix(event_date: date, today: date) -> str:
    delta = (event_date - today).days
    if delta <= 0:
        return "TODAY"
    if delta == 1:
        return "TOMORROW"
    if delta <= 7:
        return f"UPCOMING ({delta}d)"
    return f"IN {delta}d"


def _urgency(opex_type: str, delta_days: int) -> float:
    if opex_type == "triple_witching":
        if delta_days <= 3:
            return 8.0
        return 6.0
    if opex_type == "monthly":
        if delta_days <= 2:
            return 7.0
        return 5.5
    # weekly
    if delta_days <= 1:
        return 5.5
    return 4.0


def _body(opex_type: str, event_date: date, delta: int) -> str:
    date_str = event_date.strftime("%A, %B %d, %Y")
    lines: list[str] = [f"Date: {date_str}  (T-{delta} days)" if delta > 0 else f"Date: {date_str}  (TODAY)"]
    if opex_type == "triple_witching":
        lines += [
            "",
            "Triple Witching: simultaneous expiration of",
            "  • S&P 500 index options (SPX)",
            "  • S&P 500 index futures (ES)",
            "  • Individual equity options and futures",
            "",
            "Effect: one of the highest-volume sessions of the year. Large",
            "gamma and delta hedging flows near key strikes. Intraday",
            "volatility typically elevated, especially in the final hour.",
        ]
    elif opex_type == "monthly":
        lines += [
            "",
            "Standard Monthly Options Expiration (3rd Friday).",
            "All standard monthly contracts for equities and ETFs expire.",
            "Expect elevated volume and potential pinning near high OI strikes.",
        ]
    else:
        lines += [
            "",
            "Weekly Options Expiration (Friday).",
            "Short-dated weekly contracts expire. Watch for gamma exposure",
            "near key strikes in SPY, QQQ, and large-cap names.",
        ]
    return "\n".join(lines)


def _upcoming_opex_dates(today: date, horizon_days: int) -> list[dict]:
    """Return sorted list of {date, opex_type} within the horizon."""
    end = today + timedelta(days=horizon_days)
    events: list[dict] = []
    seen_dates: set[date] = set()

    d = today
    while d <= end:
        if d.weekday() == 4:  # Friday
            if d in seen_dates:
                d += timedelta(days=1)
                continue
            if _is_triple_witching(d):
                events.append({"date": d, "opex_type": "triple_witching"})
            elif _is_monthly_opex(d):
                events.append({"date": d, "opex_type": "monthly"})
            else:
                events.append({"date": d, "opex_type": "weekly"})
            seen_dates.add(d)
        d += timedelta(days=1)

    events.sort(key=lambda x: x["date"])
    return events


def collect_opex_calendar() -> list[dict]:
    """Compute upcoming OpEx dates and return net-new article dicts."""
    now = datetime.now(timezone.utc)
    today = now.date()

    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    _ensure_db(conn)

    articles: list[dict] = []
    events = _upcoming_opex_dates(today, HORIZON_DAYS)

    for ev in events:
        ev_date: date = ev["date"]
        opex_type: str = ev["opex_type"]
        delta = (ev_date - today).days
        day_cls = _day_class(ev_date, today)

        # Skip far-out weekly expirations (only emit within 7 days)
        if opex_type == "weekly" and delta > 7:
            continue

        seen_key = f"opex|{opex_type}|{ev_date.isoformat()}|{day_cls}"
        sid = _sha256(seen_key)
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=? LIMIT 1", (sid,)).fetchone():
            continue

        prefix = _day_prefix(ev_date, today)
        type_label = {
            "triple_witching": "Quarterly Triple Witching",
            "monthly": "Monthly Options Expiration",
            "weekly": "Weekly Options Expiration",
        }[opex_type]

        title = f"{prefix}: {type_label} — {ev_date.strftime('%B %d, %Y')}"
        body = _body(opex_type, ev_date, delta)
        score = _urgency(opex_type, delta)
        ts = now.strftime("%Y-%m-%d %H:%M:%S")
        url = f"internal://opex_calendar/{ev_date.isoformat()}/{opex_type}"

        compressed = zlib.compress(body.encode("utf-8"))

        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
            (sid, url, title, SOURCE_NAME, ts),
        )

        articles.append({
            "title": title,
            "link": url,
            "summary": body,
            "source": SOURCE_NAME,
            "published": ts,
            "_relevance_score": score,
        })
        log.info("opex_calendar: emitted — %s", title)

    conn.commit()
    conn.close()
    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("[opex_calendar] Computing upcoming options expiration dates...")
    results = collect_opex_calendar()
    if results:
        for r in results:
            print(f"  [{r['_relevance_score']:.1f}] {r['title']}")
            print(f"       {r['summary'][:120]}...")
    else:
        print("  (all events already seen or none within horizon)")
    print(f"\nTotal new articles: {len(results)}")
