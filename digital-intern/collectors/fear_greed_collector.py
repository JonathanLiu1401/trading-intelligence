"""CNN Fear & Greed Index collector.

Fetches the current Fear & Greed score (0–100) from CNN's public data API.
Emits a synthetic article row when the score or rating changes meaningfully,
providing market sentiment context for briefings and scoring.

Dedup key: date + rating bucket so at most one "rating zone" article per day.
A new article is also emitted if the intraday score moves ≥5 points from
the previously recorded value.
"""
import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

API_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
SOURCE_NAME = "CNN Fear&Greed"
HTTP_TIMEOUT = 10
SCORE_CHANGE_THRESHOLD = 5.0  # emit article if score moves this much intraday

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://www.cnn.com/markets/fear-and-greed",
}

log = logging.getLogger("fear_greed_collector")

RATING_EMOJI = {
    "extreme fear": "😱",
    "fear": "😟",
    "neutral": "😐",
    "greed": "🤑",
    "extreme greed": "🚀",
}


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


def _article_id(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _fetch_data() -> dict | None:
    try:
        r = requests.get(API_URL, headers=HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"[fear_greed] fetch failed: {e}")
        return None


def collect_fear_greed() -> list[dict]:
    """Fetch Fear & Greed score; return net-new article dicts."""
    data = _fetch_data()
    if not data:
        return []

    fg = data.get("fear_and_greed", {})
    score = fg.get("score")
    rating = (fg.get("rating") or "").lower().strip()
    if score is None or not rating:
        log.warning("[fear_greed] missing score/rating in API response")
        return []

    score = round(float(score), 1)
    prev_close = round(float(fg.get("previous_close") or score), 1)
    prev_week = round(float(fg.get("previous_1_week") or score), 1)
    prev_month = round(float(fg.get("previous_1_month") or score), 1)
    timestamp = fg.get("timestamp", "")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    emoji = RATING_EMOJI.get(rating, "📊")

    # Primary dedup key: date + rating bucket (one per day per zone)
    rating_key = f"fear_greed|{today}|{rating}"
    rating_id = _article_id(rating_key)

    # Secondary key: significant intraday move (rounded to nearest 5)
    score_bucket = int(score // SCORE_CHANGE_THRESHOLD) * int(SCORE_CHANGE_THRESHOLD)
    move_key = f"fear_greed|{today}|score_{score_bucket}"
    move_id = _article_id(move_key)

    link = "https://www.cnn.com/markets/fear-and-greed"

    delta_day = score - prev_close
    delta_week = score - prev_week
    delta_month = score - prev_month

    def _fmt_delta(d: float) -> str:
        return f"+{d:.1f}" if d >= 0 else f"{d:.1f}"

    title = (
        f"{emoji} Fear & Greed Index: {score:.0f}/100 ({rating.title()}) | "
        f"Day: {_fmt_delta(delta_day)} | Week: {_fmt_delta(delta_week)} | "
        f"Month: {_fmt_delta(delta_month)}"
    )
    summary = (
        f"CNN Fear & Greed Index at {score:.1f} ({rating}) as of {timestamp}. "
        f"Previous close: {prev_close}, 1-week avg: {prev_week}, "
        f"1-month avg: {prev_month}."
    )

    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    _ensure_db(conn)
    results: list[dict] = []

    try:
        for art_id, dedup_key in [(rating_id, rating_key), (move_id, move_key)]:
            if conn.execute(
                "SELECT 1 FROM seen_articles WHERE id=?", (art_id,)
            ).fetchone():
                continue

            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                "VALUES (?,?,?,?,?)",
                (art_id, link, title, SOURCE_NAME, now),
            )
            results.append({
                "id": art_id,
                "link": link,
                "title": title,
                "summary": summary,
                "source": SOURCE_NAME,
                "first_seen": now,
                "fear_greed_score": score,
                "fear_greed_rating": rating,
            })

        conn.commit()
    finally:
        conn.close()

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    articles = collect_fear_greed()
    if articles:
        for a in articles:
            print(f"NEW: {a['title']}")
            print(f"     {a['summary']}")
    else:
        print("No new Fear & Greed articles (already seen today)")
