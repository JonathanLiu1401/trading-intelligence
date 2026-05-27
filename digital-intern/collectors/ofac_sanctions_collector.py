"""OFAC sanctions & enforcement action collector.

The US Treasury's Office of Foreign Assets Control (OFAC) publishes
sanctions designations and enforcement actions that are among the
highest-volatility market signals available — energy stocks, defense
contractors, banks, and crypto platforms move 5-20% intraday on
major sanctions news.

Source: https://ofac.treasury.gov/recent-actions (HTML scrape, no API key)

Each scraped entry carries:
  - Date of action (YYYYMMDD slug → ISO date)
  - Action type (title): e.g. "Russia-related Designations", "Civil Penalty"
  - Direct link to the OFAC action detail page

Tickers most affected by OFAC signals:
  Energy:  XOM, CVX, OXY, MPC, VLO, PSX, SLB  (Russia/Iran/Venezuela)
  Defense: LMT, RTX, NOC, GD, BA               (arms embargo violations)
  Finance: JPM, BAC, C, WFC, GS, MS            (bank penalty risk)
  Crypto:  COIN, MSTR                           (OFAC compliance risk)
  Shipping:ZIM, MATX, GOGL                      (vessel sanctions)
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

OFAC_ACTIONS_URL = "https://ofac.treasury.gov/recent-actions"
FETCH_TIMEOUT = 15
SOURCE = "ofac_recent_actions"

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Tickers with OFAC exposure (injected as metadata for the scorer)
_SECTOR_TICKERS = [
    "XOM", "CVX", "OXY", "MPC", "VLO", "PSX", "SLB",  # Energy
    "LMT", "RTX", "NOC", "GD", "BA",                    # Defense
    "JPM", "BAC", "C", "WFC", "GS", "MS",               # Finance
    "COIN", "MSTR",                                       # Crypto
    "ZIM", "MATX", "GOGL",                               # Shipping
]

# High-signal action keywords that warrant urgency=1
_URGENT_KEYWORDS = {
    "russia", "iran", "north korea", "dprk", "china",
    "counter terrorism", "nuclear", "proliferation",
    "settlement", "civil penalty", "enforcement",
    "cuba", "venezuela", "syria", "myanmar",
}


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


def _article_id(link: str, title: str) -> str:
    return hashlib.sha256(f"{link}||{title}".encode()).hexdigest()


def _parse_date(slug: str) -> str:
    """Convert '20260521' date slug to ISO timestamp string."""
    try:
        d = datetime.strptime(slug[:8], "%Y%m%d")
        return d.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return datetime.now(timezone.utc).isoformat()


def _is_urgent(title: str) -> bool:
    tl = title.lower()
    return any(kw in tl for kw in _URGENT_KEYWORDS)


def _scrape_ofac_actions() -> list[dict]:
    """Scrape https://ofac.treasury.gov/recent-actions, return list of action dicts."""
    try:
        resp = requests.get(
            OFAC_ACTIONS_URL,
            headers={"User-Agent": _UA, "Accept": "text/html,*/*"},
            timeout=FETCH_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[ofac_sanctions_collector] fetch error: {e}")
        return []

    html = resp.text
    # Find all /recent-actions/{date slug} links with their anchor text
    # Pattern: href="/recent-actions/20260521" or /recent-actions/20260521_33
    pattern = re.compile(
        r'href="(/recent-actions/(\d{8}(?:_\d+)?)[^"]*)"[^>]*>([\s\S]*?)</a>',
        re.IGNORECASE,
    )

    seen_paths: set[str] = set()
    articles = []

    for m in pattern.finditer(html):
        path, date_slug, raw_title = m.groups()
        title = re.sub(r"<[^>]+>", " ", raw_title)
        title = " ".join(title.split()).strip()
        if not title or len(title) < 5:
            continue
        if path in seen_paths:
            continue
        seen_paths.add(path)

        link = f"https://ofac.treasury.gov{path}"
        published = _parse_date(date_slug)
        urgency = 1 if _is_urgent(title) else 0

        articles.append({
            "title":     f"OFAC: {title}",
            "link":      link,
            "summary":   f"OFAC action ({date_slug[:8]}): {title}",
            "published": published,
            "source":    SOURCE,
            "_tickers":  _SECTOR_TICKERS,
            "urgency":   urgency,
        })

    return articles


def collect_ofac_sanctions() -> list[dict]:
    """Scrape OFAC recent-actions, deduplicate, return only new articles."""
    conn = _ensure_db()
    new_articles: list[dict] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    fetched = _scrape_ofac_actions()
    for art in fetched:
        aid = _article_id(art["link"], art["title"])
        try:
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles(id, link, title, source, first_seen) "
                "VALUES (?, ?, ?, ?, ?)",
                (aid, art["link"], art["title"], SOURCE, now_iso),
            )
            if conn.execute("SELECT changes()").fetchone()[0]:
                new_articles.append(art)
        except sqlite3.Error:
            pass

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    import sys

    # --fresh: wipe prior seen state so we can see real data
    if "--fresh" in sys.argv:
        try:
            conn2 = sqlite3.connect(str(DB_PATH), timeout=5)
            conn2.execute(f"DELETE FROM seen_articles WHERE source='{SOURCE}'")
            conn2.commit()
            conn2.close()
            print(f"[ofac] wiped seen state for source='{SOURCE}'")
        except Exception as e:
            print(f"[ofac] could not wipe (DB busy?): {e} — running anyway")

    # Fetch and show
    print("[ofac] scraping https://ofac.treasury.gov/recent-actions ...")
    raw = _scrape_ofac_actions()
    print(f"[ofac] scraped {len(raw)} actions from page")
    for a in raw[:15]:
        print(f"\n  {a['published'][:10]}  {a['title']}")
        print(f"    {a['link']}")
        print(f"    urgency={a['urgency']}")

    print(f"\n[ofac] running collect_ofac_sanctions() (dedup filter)...")
    new = collect_ofac_sanctions()
    print(f"[ofac] {len(new)} new (not-yet-seen) articles")
