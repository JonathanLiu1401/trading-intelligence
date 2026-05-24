"""AAII Investor Sentiment Survey collector.

Weekly survey of individual investor sentiment toward the stock market over
the next six months. Published by the American Association of Individual
Investors (AAII) since 1987. A widely-used contrarian indicator.

Scrapes https://www.aaii.com/sentimentsurvey for the current week's:
  - Bullish %, Neutral %, Bearish %
  - Survey week-ending date
  - Latest article headline

Emits one synthetic article row per survey week, deduped by date so re-runs
don't emit duplicates. Extreme readings (Bullish <25% or Bearish >50%)
get an 'EXTREME SENTIMENT' prefix so they pass urgency scoring.

No API key required. Weekly cadence: published each Thursday.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

SOURCE = "aaii/sentiment_survey"
SURVEY_URL = "https://www.aaii.com/sentimentsurvey"
FETCH_TIMEOUT = 12
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Contrarian thresholds for elevated urgency prefix
EXTREME_BULL_LOW = 25.0   # below this → extreme bearish crowd = contrarian buy
EXTREME_BEAR_HIGH = 50.0  # above this → extreme fear = contrarian buy signal
EXTREME_BULL_HIGH = 55.0  # above this → extreme greed = contrarian sell signal


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


def _article_id(survey_date: str) -> str:
    return hashlib.sha256(f"aaii_sentiment|{survey_date}".encode()).hexdigest()


def _parse_sentiment(html: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")

    # Parse bar-chart div structure:
    # <div class="date">MM/DD/YYYY</div>
    # <div class="bar bullish" style="width:XX%">XX%</div>
    # <div class="bar neutral" style="width:XX%">XX%</div>
    # <div class="bar bearish" style="width:XX%">XX%</div>
    # First div.date is a column header "Week Ending"; actual dates follow.
    date_divs = soup.find_all("div", class_="date")
    survey_date = None
    for d in date_divs:
        text = d.get_text(strip=True)
        if re.match(r"\d{1,2}/\d{1,2}/\d{4}", text):
            survey_date = text
            break
    if not survey_date:
        return None

    def _pct(class_name: str) -> float | None:
        el = soup.find("div", class_=lambda c: c and class_name in c.split())
        if not el:
            return None
        m = re.search(r"([\d.]+)%", el.get_text())
        return float(m.group(1)) if m else None

    bullish = _pct("bullish")
    neutral = _pct("neutral")
    bearish = _pct("bearish")

    if None in (bullish, neutral, bearish):
        return None

    # Latest article headline
    headline = ""
    for tag in soup.find_all(["h2", "h3"]):
        t = tag.get_text(strip=True)
        if "aaii sentiment survey" in t.lower() and ":" in t:
            headline = t
            break

    return {
        "survey_date": survey_date,
        "bullish": bullish,
        "neutral": neutral,
        "bearish": bearish,
        "headline": headline,
    }


def _build_article(data: dict) -> dict:
    bull = data["bullish"]
    bear = data["bearish"]
    neut = data["neutral"]
    date = data["survey_date"]
    headline = data["headline"] or f"AAII Sentiment Survey week of {date}"

    prefix = ""
    if bull < EXTREME_BULL_LOW or bear > EXTREME_BEAR_HIGH:
        prefix = "EXTREME BEARISH SENTIMENT (Contrarian Buy Signal): "
    elif bull > EXTREME_BULL_HIGH:
        prefix = "EXTREME BULLISH SENTIMENT (Contrarian Sell Signal): "

    title = (
        f"{prefix}AAII Sentiment {date}: "
        f"Bullish {bull:.1f}% | Neutral {neut:.1f}% | Bearish {bear:.1f}% — {headline}"
    )

    spread = bull - bear
    bull_bear_label = "BULL" if spread > 0 else "BEAR"
    summary = (
        f"Weekly AAII Investor Sentiment Survey (week ending {date}). "
        f"Bullish: {bull:.1f}%, Neutral: {neut:.1f}%, Bearish: {bear:.1f}%. "
        f"Bull-Bear Spread: {spread:+.1f}pp ({bull_bear_label}). "
        f"Historical avg: ~37.5% bull / 31.5% bear. "
        f"Source: {headline}"
    )

    return {
        "id": _article_id(date),
        "title": title,
        "link": SURVEY_URL,
        "source": SOURCE,
        "summary": summary,
        "first_seen": datetime.now(timezone.utc).isoformat(),
        "_tickers": [],
        "_sentiment_data": {
            "bullish": bull,
            "neutral": neut,
            "bearish": bear,
            "bull_bear_spread": round(spread, 1),
            "survey_date": date,
        },
    }


def collect() -> list[dict]:
    try:
        resp = requests.get(
            SURVEY_URL,
            headers=_HEADERS,
            timeout=FETCH_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[aaii_sentiment] fetch error: {exc}")
        return []

    data = _parse_sentiment(resp.text)
    if not data:
        print("[aaii_sentiment] failed to parse sentiment data")
        return []

    article = _build_article(data)
    art_id = article["id"]

    conn = _ensure_db()
    try:
        exists = conn.execute(
            "SELECT 1 FROM seen_articles WHERE id=?", (art_id,)
        ).fetchone()
        if exists:
            print(
                f"[aaii_sentiment] already seen week of {data['survey_date']}, skipping"
            )
            return []

        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (art_id, article["link"], article["title"], article["source"], article["first_seen"]),
        )
        conn.commit()
    finally:
        conn.close()

    print(
        f"[aaii_sentiment] {data['survey_date']}: "
        f"Bull={data['bullish']:.1f}% Neut={data['neutral']:.1f}% Bear={data['bearish']:.1f}%"
    )
    return [article]


if __name__ == "__main__":
    results = collect()
    print(f"Collected {len(results)} item(s)")
    for r in results:
        print(f"  Title: {r['title']}")
        print(f"  Sentiment: {r['_sentiment_data']}")
