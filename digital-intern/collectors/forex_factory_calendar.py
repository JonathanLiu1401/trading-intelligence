"""Forex Factory economic calendar collector.

Fetches the current-week economic calendar from Forex Factory's public JSON
endpoint. High-impact events are the primary target; Medium-impact events are
also captured but given lower urgency.

  * FOMC speakers, rate decisions, NFP, CPI, PCE, GDP, PMI, retail sales …
  * Currency tagged to the source country (USD, EUR, GBP, JPY, …)

Each event is stored as a synthetic article row so it surfaces in urgency
scoring and the Opus briefing. Events that have already fired (``actual``
present) are stored with their actual vs. forecast surprise, which is
directly tradeable signal. Upcoming events get an "UPCOMING" prefix so the
urgency scorer promotes them.

Dedup: keyed on ``ff_cal|{country}|{date_utc}|{title}`` in seen_articles.db.
Re-emits when ``actual`` appears (the event has now fired) — a fresh row is
inserted with the actual value so the briefing picks up realized surprises.
"""
from __future__ import annotations

import hashlib
import ssl
import sqlite3
import urllib.request
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("forex_factory_calendar")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "ff_calendar_seen.db"

ENDPOINT = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
SOURCE = "forex_factory_calendar"
FETCH_TIMEOUT = 15

# Only surface High and Medium impact by default; Low is noise for trading.
IMPACT_INCLUDE = {"High", "Medium"}

# High-impact country codes that matter most to equity/macro traders.
PRIORITY_COUNTRIES = {"USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD", "CHF"}


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY, link TEXT, title TEXT,
            source TEXT, first_seen TEXT
        )"""
    )
    conn.commit()
    return conn


def _event_id(country: str, date_str: str, title: str, has_actual: bool) -> str:
    # Separate dedup key for pending vs. realized — lets us re-emit once actual fires.
    suffix = "actual" if has_actual else "pending"
    raw = f"ff_cal|{country}|{date_str}|{title}|{suffix}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _day_prefix(event_dt: datetime, now: datetime) -> str:
    delta = (event_dt.date() - now.date()).days
    if delta < 0:
        return "PAST"
    if delta == 0:
        return "TODAY"
    if delta == 1:
        return "TOMORROW"
    return f"UPCOMING ({delta}d)"


def _build_title(event: dict, prefix: str) -> str:
    country = event.get("country", "")
    title = event.get("title", "")
    forecast = event.get("forecast", "")
    previous = event.get("previous", "")
    actual = event.get("actual", "")
    impact = event.get("impact", "")

    parts = [f"{prefix}: [{country}] {title}"]
    if actual:
        parts.append(f"Actual: {actual}")
        if forecast:
            parts.append(f"Forecast: {forecast}")
    else:
        if forecast:
            parts.append(f"Forecast: {forecast}")
    if previous:
        parts.append(f"Prev: {previous}")
    parts.append(f"[{impact}]")
    return " | ".join(parts)


def _build_summary(event: dict) -> str:
    actual = event.get("actual", "")
    forecast = event.get("forecast", "")
    previous = event.get("previous", "")
    country = event.get("country", "")
    title = event.get("title", "")
    date_str = event.get("date", "")

    lines = [f"Forex Factory Economic Event: {country} {title}"]
    lines.append(f"Scheduled: {date_str}")
    if actual:
        lines.append(f"Actual: {actual}")
    if forecast:
        lines.append(f"Forecast: {forecast}")
    if previous:
        lines.append(f"Previous: {previous}")
    return "\n".join(lines)


def collect() -> list[dict]:
    """Fetch this week's economic calendar and return new article dicts."""
    ctx = ssl.create_default_context()
    req = urllib.request.Request(
        ENDPOINT,
        headers={"User-Agent": "Mozilla/5.0 research-bot/1.0; +mailto:research@university.edu"},
    )
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=FETCH_TIMEOUT) as resp:
            events = json.loads(resp.read())
    except Exception as exc:
        log.warning("forex_factory_calendar: fetch failed: %s", exc)
        return []

    now = datetime.now(timezone.utc)
    conn = _ensure_db()
    articles: list[dict] = []

    for event in events:
        impact = event.get("impact", "")
        if impact not in IMPACT_INCLUDE:
            continue

        country = event.get("country", "")
        title_raw = event.get("title", "")
        date_str = event.get("date", "")
        actual = event.get("actual", "")

        # Parse event datetime
        try:
            event_dt = datetime.fromisoformat(date_str).astimezone(timezone.utc)
        except Exception:
            continue

        has_actual = bool(actual)
        eid = _event_id(country, date_str, title_raw, has_actual)

        # Check dedup
        row = conn.execute("SELECT id FROM seen_articles WHERE id=?", (eid,)).fetchone()
        if row:
            continue

        prefix = _day_prefix(event_dt, now)
        # Skip past events that haven't fired yet in our data (no actual, past date)
        if prefix == "PAST" and not has_actual:
            continue

        art_title = _build_title(event, prefix)
        art_summary = _build_summary(event)
        # Synthetic link keyed by event identity
        link = f"https://www.forexfactory.com/calendar#{country.lower()}_{event_dt.strftime('%Y%m%d')}_{hashlib.md5(title_raw.encode()).hexdigest()[:8]}"

        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
            (eid, link, art_title, SOURCE, now.isoformat()),
        )
        conn.commit()

        # Pre-score: heuristic_scorer returns 0 ("no_keywords") for most
        # macro-event titles like "[USD] CPI m/m" so they get dropped by the
        # 0.5 noise gate in daemon._ingest. Same class of bug as 66ac656
        # fixed for collector_rate_monitor — opt into _relevance_score.
        # High impact → 2.5 (clears gate, below 4.0 urgent threshold).
        # Medium → 1.0. Realized events (+1.0): surprise is the tradeable bit.
        score = 2.5 if impact == "High" else 1.0
        if has_actual:
            score += 1.0
        articles.append(
            {
                "title": art_title,
                "link": link,
                "summary": art_summary,
                "source": SOURCE,
                "published": event_dt.isoformat(),
                "_relevance_score": score,
            }
        )

    conn.close()
    log.info("forex_factory_calendar: %d new events", len(articles))
    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    results = collect()
    print(f"Fetched {len(results)} new events")
    for r in results[:10]:
        print(f"  [{r['source']}] {r['title']}")
        print(f"    {r['link']}")
