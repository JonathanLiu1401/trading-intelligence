"""Macro economic calendar collector.

Fetches upcoming high-impact economic events and stores them as synthetic
article rows so they surface in urgency scoring and briefings:

  - FOMC meeting dates from federalreserve.gov
  - BLS release schedule (CPI, Jobs/Employment Situation, PPI) from bls.gov

Events within the next 7 days get an 'UPCOMING' prefix so they pass keyword
urgency. Events today or tomorrow get 'TODAY' / 'TOMORROW' prefix.

Dedup: keyed by ``macro_event|YYYY-MM-DD|event_type`` in seen_articles.db so
the same event is only inserted once until the date passes.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent
# Use a dedicated lightweight DB to avoid contention on seen_articles.db
# (which is heavily written to by the daemon's collector workers).
_MACRO_DB_PATH = BASE_DIR / "data" / "macro_calendar_seen.db"

FETCH_TIMEOUT = 15

# BLS requires a mailto-style UA; generic Chrome UA returns 403.
_UA_BLS = "Mozilla/5.0 (compatible; research-bot/1.0; +mailto:research@university.edu)"
# Fed website is fine with standard browser UA.
_UA_BROWSER = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

SOURCE_NAME = "macro_calendar"
HORIZON_DAYS = 30
UPCOMING_WINDOW_DAYS = 7

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4,
    "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}


def _ensure_db() -> sqlite3.Connection:
    _MACRO_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_MACRO_DB_PATH), timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_events (
            id TEXT PRIMARY KEY,
            event_type TEXT,
            event_date TEXT,
            title TEXT,
            first_seen TEXT
        )"""
    )
    conn.commit()
    return conn


def _seen_id(event_type: str, date_str: str) -> str:
    return hashlib.sha256(f"macro_event|{date_str}|{event_type}".encode()).hexdigest()


def _is_seen(conn: sqlite3.Connection, sid: str) -> bool:
    return conn.execute("SELECT 1 FROM seen_events WHERE id=?", (sid,)).fetchone() is not None


def _mark_seen(conn: sqlite3.Connection, sid: str, event_type: str, date_str: str, title: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO seen_events (id, event_type, event_date, title, first_seen) VALUES (?,?,?,?,?)",
        (sid, event_type, date_str, title, now),
    )
    conn.commit()


def _day_prefix(event_date: datetime, now: datetime) -> str:
    delta = (event_date.date() - now.date()).days
    if delta == 0:
        return "TODAY"
    if delta == 1:
        return "TOMORROW"
    if delta <= UPCOMING_WINDOW_DAYS:
        return f"UPCOMING ({delta}d)"
    return f"IN {delta}d"


def _parse_month(s: str) -> int | None:
    return _MONTH_MAP.get(s.lower().rstrip("."))


def _fetch_fomc_dates(now: datetime, horizon: datetime) -> list[dict[str, Any]]:
    """Scrape upcoming FOMC meeting dates from federalreserve.gov.

    The page renders meeting blocks as:
      <div class="fomc-meeting__month"><strong>March</strong></div>
      <div class="fomc-meeting__date">17-18*</div>
    grouped under h4 headings like "2026 FOMC Meetings".
    """
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    try:
        resp = requests.get(url, headers={"User-Agent": _UA_BROWSER}, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        print(f"[macro_calendar] FOMC fetch failed: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    events: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for h4 in soup.find_all("h4"):
        text = h4.get_text(strip=True)
        m = re.search(r'\b(20\d{2})\b', text)
        if not m:
            continue
        year = int(m.group(1))

        # Structure: div.panel.panel-default > div.panel-heading > h4
        # Meeting rows are direct children of div.panel.panel-default, NOT of panel-heading.
        # Use exact string "panel" to match only the outer panel div (not panel-heading).
        panel = h4.find_parent("div", class_="panel")
        if panel is None:
            panel = h4.find_parent() or h4
        for meeting_div in panel.find_all("div", class_="fomc-meeting"):
            month_div = meeting_div.find("div", class_=re.compile(r"fomc-meeting__month"))
            date_div = meeting_div.find("div", class_=re.compile(r"fomc-meeting__date"))
            if not month_div or not date_div:
                continue

            month_str = month_div.get_text(strip=True)
            date_str = date_div.get_text(strip=True)

            month_num = _parse_month(month_str)
            if month_num is None:
                continue

            # date_str can be "17-18*" or "28-29" — take the last day (decision day)
            day_m = re.search(r'(\d{1,2})\*?$', date_str.split("-")[-1].strip())
            if not day_m:
                continue
            day = int(day_m.group(1))

            try:
                dt = datetime(year, month_num, day, 14, 0, 0, tzinfo=timezone.utc)
            except ValueError:
                continue

            if now <= dt <= horizon:
                key = dt.strftime("%Y-%m-%d")
                if key not in seen_keys:
                    seen_keys.add(key)
                    events.append({"type": "FOMC Meeting", "date": dt, "url": url})

    return sorted(events, key=lambda x: x["date"])


def _fetch_bls_schedule(
    release_name: str,
    bls_url: str,
    now: datetime,
    horizon: datetime,
) -> list[dict[str, Any]]:
    """Fetch a BLS release schedule page and extract upcoming dates.

    BLS tables typically have rows like:
      Reference Month | Release Date | Release Time
      "January 2026"  | "Feb. 13, 2026" | "08:30 AM"
    """
    try:
        resp = requests.get(
            bls_url,
            headers={"User-Agent": _UA_BLS, "Accept": "text/html,application/xhtml+xml"},
            timeout=FETCH_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[macro_calendar] BLS fetch failed for {release_name}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    events: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    # BLS tables list "Release Date" column as "Mon. DD, YYYY" or "Month DD, YYYY"
    date_pat = re.compile(
        r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'\.?\s+(\d{1,2}),\s+(20\d{2})',
        re.I,
    )

    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in row.find_all(["td", "th"])]
            for cell in cells:
                m = date_pat.search(cell)
                if not m:
                    continue
                month_num = _parse_month(m.group(1))
                if month_num is None:
                    continue
                try:
                    dt = datetime(int(m.group(3)), month_num, int(m.group(2)),
                                  8, 30, 0, tzinfo=timezone.utc)
                except ValueError:
                    continue
                if now <= dt <= horizon:
                    key = dt.strftime("%Y-%m-%d")
                    if key not in seen_keys:
                        seen_keys.add(key)
                        events.append({"type": release_name, "date": dt, "url": bls_url})

    return sorted(events, key=lambda x: x["date"])


def collect_macro_calendar() -> list[dict[str, Any]]:
    """Fetch all upcoming macro events and return article dicts for ingestion."""
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=HORIZON_DAYS)
    conn = _ensure_db()
    articles: list[dict[str, Any]] = []

    all_events: list[dict[str, Any]] = []
    all_events.extend(_fetch_fomc_dates(now, horizon))
    all_events.extend(_fetch_bls_schedule(
        "CPI Release",
        "https://www.bls.gov/schedule/news_release/cpi.htm",
        now, horizon,
    ))
    all_events.extend(_fetch_bls_schedule(
        "Jobs Report (Employment Situation)",
        "https://www.bls.gov/schedule/news_release/empsit.htm",
        now, horizon,
    ))
    all_events.extend(_fetch_bls_schedule(
        "PPI Release",
        "https://www.bls.gov/schedule/news_release/ppi.htm",
        now, horizon,
    ))
    all_events.sort(key=lambda x: x["date"])

    for event in all_events:
        date_str = event["date"].strftime("%Y-%m-%d")
        sid = _seen_id(event["type"], date_str)
        if _is_seen(conn, sid):
            continue

        prefix = _day_prefix(event["date"], now)
        title = f"{prefix}: {event['type']} — {event['date'].strftime('%B %d, %Y')}"
        summary = (
            f"Scheduled {event['type']} on {event['date'].strftime('%A, %B %d, %Y')} "
            f"at {event['date'].strftime('%H:%M')} UTC. "
            f"High-impact macro event. Days until: {(event['date'].date() - now.date()).days}."
        )

        article: dict[str, Any] = {
            "title": title,
            "link": event["url"],
            "summary": summary,
            "published": event["date"].isoformat(),
            "source": SOURCE_NAME,
        }
        articles.append(article)
        _mark_seen(conn, sid, event["type"], date_str, title)

    conn.close()
    return articles


if __name__ == "__main__":
    print("[macro_calendar] Fetching upcoming FOMC and BLS macro events...")
    results = collect_macro_calendar()
    print(f"[macro_calendar] Found {len(results)} new events:")
    for a in results:
        print(f"  {a['title']}")
        print(f"    {a['summary'][:140]}")
    if not results:
        print("  (all events already seen or none within horizon)")
