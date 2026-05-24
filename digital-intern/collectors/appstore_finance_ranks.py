"""Apple App Store Finance category rank tracker.

Fetches the top-100 free Finance apps from iTunes RSS daily and surfaces
rank changes in key trading/brokerage/crypto apps as market-sentiment signals.

When Robinhood jumps from #30 → #5 in the Finance charts, retail is active.
When Coinbase spikes to the top, crypto heat is real. These rank moves often
precede news coverage by hours.

API: https://itunes.apple.com/us/rss/topfreeapplications/limit=100/genre=6015/json
No auth, no rate limit.  Data refreshes once daily.

Deduplication: one article per (app_id, date) via seen_articles.db.
Rank history persisted in data/appstore_finance_ranks.json for delta tracking.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger("appstore_finance_ranks")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
RANKS_PATH = BASE_DIR / "data" / "appstore_finance_ranks.json"

SOURCE = "appstore/finance"
ITUNES_URL = (
    "https://itunes.apple.com/us/rss/topfreeapplications"
    "/limit=100/genre=6015/json"
)
HTTP_TIMEOUT = 15

# Apps we care about — by iTunes app ID.  Add more as needed.
TRACKED_APPS: dict[str, dict] = {
    "938003185":  {"name": "Robinhood",    "tickers": ["HOOD"]},
    "1488764897": {"name": "Coinbase",     "tickers": ["COIN"]},
    "883324671":  {"name": "Coinbase (alt)","tickers": ["COIN"]},
    "886427730":  {"name": "Coinbase",     "tickers": ["COIN"]},
    "1576588253": {"name": "Bloom Invest", "tickers": []},
    "348177453":  {"name": "Fidelity",     "tickers": []},
    "407358186":  {"name": "Schwab Mobile","tickers": ["SCHW"]},
    "1191985736": {"name": "SoFi",         "tickers": ["SOFI"]},
    "454558592":  {"name": "IBKR",         "tickers": ["IBKR"]},
    "1632713844": {"name": "Kalshi",       "tickers": []},
    "519817714":  {"name": "Credit Karma", "tickers": []},
    "1554623825": {"name": "Alinea Invest","tickers": []},
    "65244":      {"name": "Webull",       "tickers": []},
    "1642611278": {"name": "Webull",       "tickers": []},
    "1288339409": {"name": "Trust Wallet", "tickers": []},
    "1576577312": {"name": "Crypto.com",   "tickers": ["CRO"]},
    "1086287267": {"name": "eToro",        "tickers": []},
    "86217439":   {"name": "TD Ameritrade","tickers": []},
    "1409169287": {"name": "Lenme",        "tickers": []},
    "335186209":  {"name": "Vanguard",     "tickers": []},
    "1130616675": {"name": "Rocket Money", "tickers": []},
}

# Emit an article when a tracked app moves this many ranks or more.
SIGNIFICANT_MOVE = 5

# Always emit a daily snapshot article summarising the top-10 finance apps.
DAILY_SUMMARY_THRESHOLD = 10


def _load_prev_ranks() -> dict[str, int]:
    if RANKS_PATH.exists():
        try:
            return json.loads(RANKS_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_ranks(ranks: dict[str, int]) -> None:
    RANKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RANKS_PATH.write_text(json.dumps(ranks, indent=2))


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


def _article_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _insert_if_new(conn: sqlite3.Connection, article: dict) -> bool:
    try:
        conn.execute(
            "INSERT INTO seen_articles (id, link, title, source, first_seen)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                article["id"],
                article.get("link", ""),
                article.get("title", ""),
                article.get("source", SOURCE),
                article.get("published", datetime.now(timezone.utc).isoformat()),
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def collect_appstore_finance_ranks() -> list[dict]:
    """Fetch Finance top-100, diff against yesterday's ranks, emit articles."""
    try:
        resp = requests.get(ITUNES_URL, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("appstore_finance_ranks: fetch failed: %s", e)
        return []

    entries = data.get("feed", {}).get("entry", [])
    if not entries:
        log.warning("appstore_finance_ranks: empty feed")
        return []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Build rank map {app_id: rank}
    current_ranks: dict[str, int] = {}
    top10: list[str] = []
    for rank, entry in enumerate(entries, 1):
        app_id = entry.get("id", {}).get("attributes", {}).get("im:id", "")
        name = entry.get("im:name", {}).get("label", "?")
        if rank <= DAILY_SUMMARY_THRESHOLD:
            artist = entry.get("im:artist", {}).get("label", "")
            top10.append(f"#{rank} {name}")
        if app_id:
            current_ranks[app_id] = rank

    prev_ranks = _load_prev_ranks()
    _save_ranks(current_ranks)

    conn = _ensure_db()
    articles: list[dict] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    # Daily top-10 snapshot
    snap_id = _article_id("appstore_daily", today)
    snap_summary = "Top-10 Finance apps today: " + ", ".join(top10)
    snap = {
        "id": snap_id,
        "title": f"Apple App Store Finance Top-10 [{today}]: {top10[0]} leads",
        "link": "https://apps.apple.com/us/charts/iphone/finance-apps/6015",
        "summary": snap_summary,
        "published": now_iso,
        "source": SOURCE,
        "_tickers": [],
    }
    if _insert_if_new(conn, snap):
        articles.append(snap)
        log.info("appstore_finance_ranks: daily snapshot emitted")

    # Rank-change alerts for tracked apps
    for app_id, meta in TRACKED_APPS.items():
        current = current_ranks.get(app_id)
        if current is None:
            continue  # not in top 100 today
        prev = prev_ranks.get(app_id)
        if prev is None:
            # First time we see this app in the chart — emit an entry note
            delta_str = "newly in top-100"
            delta_mag = SIGNIFICANT_MOVE + 1  # always emit on first appearance
        else:
            delta = prev - current  # positive = climbed
            delta_mag = abs(delta)
            if delta_mag < SIGNIFICANT_MOVE:
                continue
            direction = "↑" if delta > 0 else "↓"
            delta_str = f"{direction}{delta_mag} ranks (#{prev} → #{current})"

        app_name = meta["name"]
        tickers = meta.get("tickers", [])
        art_id = _article_id("appstore_move", app_id, today)
        title = f"App Store: {app_name} {delta_str} in Finance charts (now #{current})"
        summary = (
            f"{app_name} moved {delta_str} in the Apple App Store Finance category "
            f"(top-100 free). App Store rank changes in trading/brokerage apps "
            f"often reflect shifts in retail investor activity."
        )
        article = {
            "id": art_id,
            "title": title,
            "link": f"https://apps.apple.com/us/app/id{app_id}",
            "summary": summary,
            "published": now_iso,
            "source": SOURCE,
            "_tickers": tickers,
        }
        if _insert_if_new(conn, article):
            articles.append(article)
            log.info("appstore_finance_ranks: rank move: %s", title)

    conn.close()
    log.info(
        "appstore_finance_ranks: %d articles emitted (top-100 fetched, %d tracked apps in chart)",
        len(articles),
        sum(1 for a in TRACKED_APPS if a in current_ranks),
    )
    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = collect_appstore_finance_ranks()
    print(f"\nEmitted {len(results)} articles:")
    for a in results:
        print(f"  - {a['title']}")
        print(f"    {a['summary'][:120]}")
