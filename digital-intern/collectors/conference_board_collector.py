"""Conference Board Economic Indicators collector.

Monitors two high-impact monthly US economic releases:
  - US Leading Economic Index (LEI) — recession predictor; market-moving
  - Consumer Confidence Index (CCI) — sentiment; moves retail/consumer stocks

Both are scraped from conference-board.org's public indicator pages.
No API key required. Monthly cadence, so very low request volume.

Emits one article per indicator per release date (deduped by indicator+date).
A release date change vs. prior run signals new data → emit article.

Dedup: seen_articles.db keyed by sha256(indicator_key|release_date).
State: data/conference_board_state.json tracks last seen release date per indicator.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:
    from core.logger import get_logger
    log = get_logger("conference_board")
except Exception:
    log = logging.getLogger("conference_board")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
STATE_PATH = BASE_DIR / "data" / "conference_board_state.json"

REQUEST_TIMEOUT = 20
SOURCE_NAME = "Conference Board"
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Indicators to monitor: key → (url, label, related_tickers)
INDICATORS = {
    "lei": (
        "https://www.conference-board.org/topics/us-leading-indicators",
        "US Leading Economic Index (LEI)",
        ["SPY", "QQQ", "IWM", "DIA"],
    ),
    "cci": (
        "https://www.conference-board.org/topics/consumer-confidence",
        "US Consumer Confidence Index (CCI)",
        ["XLY", "XLP", "SPY", "AMZN", "WMT"],
    ),
}

# Thresholds for escalating headline urgency
LEI_WARN_CONSECUTIVE = 6   # ≥6 consecutive monthly declines = recession warning
CCI_DROP_WARN = 5.0        # ≥5 pts single-month CCI drop = stress signal


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


def _article_id(key: str, release_date: str) -> str:
    return hashlib.sha256(f"{key}|{release_date}".encode()).hexdigest()


def _load_state() -> dict:
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_PATH)


def _fetch_lei(soup: BeautifulSoup) -> dict | None:
    """Parse LEI page → {change_pct, release_date, label}."""
    # Change value: looks like "+ 0.1%" or "- 0.3%"
    change = None
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        text = a.get_text().strip()
        if "us-leading-indicators" in href and re.search(r"[+-]\s*\d+\.\d+\s*%", text):
            m = re.search(r"([+-])\s*(\d+\.\d+)\s*%", text)
            if m:
                sign = 1 if m.group(1) == "+" else -1
                change = sign * float(m.group(2))
            break

    # Release date from page body
    release_date = None
    for el in soup.find_all(["div", "section"]):
        text = el.get_text(" ", strip=True)
        if "Full Press Release" in text or "Technical Notes" in text:
            m = re.search(
                r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+2\d{3}",
                text,
            )
            if m:
                release_date = m.group(0)
                break

    if change is None or release_date is None:
        return None
    return {"change_pct": change, "release_date": release_date}


def _fetch_cci(soup: BeautifulSoup) -> dict | None:
    """Parse CCI page → {change_pts, release_date, label}."""
    change = None
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        text = a.get_text().strip()
        if "consumer-confidence" in href and re.search(r"[+-]\s*\d+\.\d+\s*pts", text, re.I):
            m = re.search(r"([+-])\s*(\d+\.\d+)\s*pts", text, re.I)
            if m:
                sign = 1 if m.group(1) == "+" else -1
                change = sign * float(m.group(2))
            break

    # Release date
    release_date = None
    for el in soup.find_all(["div", "section"]):
        text = el.get_text(" ", strip=True)
        if "Download Release" in text or "PRESS RELEASE" in text:
            m = re.search(
                r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+2\d{3}",
                text,
            )
            if m:
                release_date = m.group(0)
                break

    if change is None or release_date is None:
        return None
    return {"change_pts": change, "release_date": release_date}


def _build_lei_article(data: dict, url: str) -> dict:
    chg = data["change_pct"]
    arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "─")
    direction = "rises" if chg > 0 else ("falls" if chg < 0 else "unchanged")
    urgency = ""
    if abs(chg) >= 0.5:
        urgency = " [NOTABLE]"
    elif chg < 0:
        urgency = " [DECLINE]"

    title = (
        f"Conference Board US LEI {direction} {arrow}{abs(chg):.1f}% "
        f"({data['release_date']}){urgency}"
    )
    summary = (
        f"The Conference Board US Leading Economic Index (LEI) "
        f"{'increased' if chg > 0 else 'decreased'} by {abs(chg):.1f}% "
        f"in the latest release ({data['release_date']}). "
        "The LEI is a composite of 10 forward-looking indicators and one of "
        "the most widely watched recession predictors."
        f" Tickers: SPY, QQQ, IWM, DIA."
    )
    return {
        "title": title,
        "link": url,
        "summary": summary,
        "published": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE_NAME,
        "_tickers": ["SPY", "QQQ", "IWM", "DIA"],
    }


def _build_cci_article(data: dict, url: str) -> dict:
    chg = data["change_pts"]
    arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "─")
    direction = "rises" if chg > 0 else ("falls" if chg < 0 else "unchanged")
    urgency = ""
    if abs(chg) >= CCI_DROP_WARN:
        urgency = f" [{'DROP' if chg < 0 else 'JUMP'} ALERT]"

    title = (
        f"Conference Board Consumer Confidence {direction} {arrow}{abs(chg):.1f} pts "
        f"({data['release_date']}){urgency}"
    )
    summary = (
        f"The Conference Board Consumer Confidence Index (CCI) "
        f"{'gained' if chg > 0 else 'dropped'} {abs(chg):.1f} points "
        f"in the latest reading ({data['release_date']}). "
        "Consumer confidence reflects household spending intentions and is a "
        "key leading indicator for consumer discretionary stocks and retail."
        " Tickers: XLY, XLP, SPY, AMZN, WMT."
    )
    return {
        "title": title,
        "link": url,
        "summary": summary,
        "published": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE_NAME,
        "_tickers": ["XLY", "XLP", "SPY", "AMZN", "WMT"],
    }


def collect_conference_board() -> list[dict]:
    """Fetch Conference Board LEI and CCI pages; emit articles for new releases."""
    state = _load_state()
    new_articles: list[dict] = []

    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    _ensure_db(conn)

    headers = {
        "User-Agent": _UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    parsers = {
        "lei": (_fetch_lei, _build_lei_article),
        "cci": (_fetch_cci, _build_cci_article),
    }

    for key, (url, label, _tickers) in INDICATORS.items():
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            parse_fn, build_fn = parsers[key]
            data = parse_fn(soup)
            if not data:
                log.warning(f"[conference_board] {key}: could not parse data from page")
                continue

            release_date = data["release_date"]
            prev_date = state.get(key, {}).get("release_date")

            # Only emit if release date changed (new monthly data)
            if release_date == prev_date:
                log.debug(f"[conference_board] {key}: no new release ({release_date})")
                continue

            aid = _article_id(key, release_date)
            already_seen = conn.execute(
                "SELECT 1 FROM seen_articles WHERE id=?", (aid,)
            ).fetchone()
            if already_seen:
                # Update state so we don't check again
                state.setdefault(key, {})["release_date"] = release_date
                continue

            article = build_fn(data, url)

            # Mark seen
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
                (
                    aid,
                    article["link"],
                    article["title"],
                    article["source"],
                    article["published"],
                ),
            )
            conn.commit()

            state.setdefault(key, {})["release_date"] = release_date
            new_articles.append(article)
            log.info(f"[conference_board] new {label}: {article['title']}")

        except Exception as e:
            log.warning(f"[conference_board] {key} error: {e}")

    conn.close()
    _save_state(state)
    return new_articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    articles = collect_conference_board()
    print(f"\nArticles emitted: {len(articles)}")
    for a in articles:
        print(f"  [{a['source']}] {a['title']}")
        print(f"    {a['summary'][:120]}...")
    if not articles:
        # Force-print current state for smoke test
        import requests
        from bs4 import BeautifulSoup
        print("\n--- Smoke test (current values) ---")
        headers = {"User-Agent": _UA}
        for key, (url, label, _) in INDICATORS.items():
            try:
                resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
                soup = BeautifulSoup(resp.text, "html.parser")
                parse_fn, _ = {"lei": (_fetch_lei, None), "cci": (_fetch_cci, None)}[key]
                data = parse_fn(soup)
                print(f"  {label}: {data}")
            except Exception as e:
                print(f"  {label}: ERROR {e}")
