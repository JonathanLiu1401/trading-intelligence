"""Crypto Fear & Greed Index collector (alternative.me).

Fetches the daily Crypto Fear & Greed score (0–100) from alternative.me's
free public API — a distinct sentiment signal from CNN's equity Fear & Greed.
Crypto sentiment correlates with risk-on/risk-off flows that often bleed into
equity markets, making it a useful leading indicator even for stock-focused
portfolios.

API: https://api.alternative.me/fng/?limit=N  — no key, no auth.

Emits a synthetic article row when:
  - The rating classification changes (Extreme Fear → Fear → Neutral → …)
  - The score shifts ≥7 points from the previously recorded bucket

Dedup key: date + rating bucket (one per zone per day) and date + score bucket.
Mirrors fear_greed_collector.py patterns exactly.
"""
import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

API_URL = "https://api.alternative.me/fng/"
SOURCE_NAME = "Crypto Fear&Greed"
HTTP_TIMEOUT = 10
SCORE_CHANGE_THRESHOLD = 7.0  # emit on ≥7pt intraday bucket shift

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
}

log = logging.getLogger("crypto_fear_greed_collector")

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


def _fetch_data() -> list[dict] | None:
    try:
        r = requests.get(API_URL, params={"limit": "3"}, headers=HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        d = r.json()
        return d.get("data", [])
    except Exception as e:
        log.warning(f"[crypto_fear_greed] fetch failed: {e}")
        return None


def collect_crypto_fear_greed() -> list[dict]:
    """Fetch Crypto Fear & Greed score; return net-new article dicts."""
    data = _fetch_data()
    if not data:
        return []

    latest = data[0]
    score_raw = latest.get("value")
    rating = (latest.get("value_classification") or "").lower().strip()

    if score_raw is None or not rating:
        log.warning("[crypto_fear_greed] missing value/classification in response")
        return []

    score = round(float(score_raw), 1)

    # Compute deltas from prior days if available
    prev_day_score = round(float(data[1].get("value", score)), 1) if len(data) > 1 else score
    prev_2d_score = round(float(data[2].get("value", score)), 1) if len(data) > 2 else score

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    emoji = RATING_EMOJI.get(rating, "📊")

    delta_day = score - prev_day_score
    delta_2d = score - prev_2d_score

    def _fmt_delta(d: float) -> str:
        return f"+{d:.1f}" if d >= 0 else f"{d:.1f}"

    title = (
        f"{emoji} Crypto Fear & Greed: {score:.0f}/100 ({rating.title()}) | "
        f"1d: {_fmt_delta(delta_day)} | 2d: {_fmt_delta(delta_2d)}"
    )
    summary = (
        f"Alternative.me Crypto Fear & Greed Index at {score:.0f} ({rating}) as of {today}. "
        f"Yesterday: {prev_day_score:.0f}, 2 days ago: {prev_2d_score:.0f}. "
        f"Note: extreme fear often precedes crypto bounces (contrarian signal); "
        f"extreme greed may precede corrections."
    )
    link = "https://alternative.me/crypto/fear-and-greed-index/"

    # Dedup keys
    rating_key = f"crypto_fg|{today}|{rating}"
    rating_id = _article_id(rating_key)

    score_bucket = int(score // SCORE_CHANGE_THRESHOLD) * int(SCORE_CHANGE_THRESHOLD)
    move_key = f"crypto_fg|{today}|score_{score_bucket}"
    move_id = _article_id(move_key)

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
                "crypto_fear_greed_score": score,
                "crypto_fear_greed_rating": rating,
            })

        conn.commit()
    finally:
        conn.close()

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    articles = collect_crypto_fear_greed()
    if articles:
        for a in articles:
            print(f"NEW: {a['title']}")
            print(f"     {a['summary']}")
    else:
        print("No new Crypto Fear & Greed articles (already seen today)")
