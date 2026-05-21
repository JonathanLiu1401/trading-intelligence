"""StockTwits per-ticker sentiment collector.

Unlike stocktwits_collector.py (global trending stream), this fetches the
symbol-specific message stream for portfolio/watchlist tickers and computes
a bullish/bearish ratio. When sentiment is extreme (>70% bull or >60% bear)
it emits an article-like signal row so the scoring pipeline can act on it.

No API key required. Rate-limited to ~1 req/s; cycles through tickers with
a cooldown cursor so we don't hammer the same ticker every pass.
"""
import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
CURSOR_PATH = BASE_DIR / "data" / "stocktwits_sentiment_cursor.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

BATCH_PER_PASS = 12           # tickers fetched per daemon cycle
PER_TICKER_COOLDOWN_SEC = 900 # 15min between re-fetching same ticker
MSG_LIMIT = 30                # messages per ticker fetch
REQUEST_TIMEOUT = 8
MAX_WORKERS = 4               # keep parallel requests low — no key = stricter limits

# Emit an article when sentiment is this extreme
BULL_THRESHOLD = 0.70  # ≥70% bullish
BEAR_THRESHOLD = 0.60  # ≥60% bearish (lower bar since fear spreads fast)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
_ST_STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json?limit={limit}"


def _load_tickers() -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        sym = (t or "").strip().upper()
        if sym and sym not in seen and "." not in sym and "^" not in sym:
            seen.add(sym)
            tickers.append(sym)

    try:
        with open(PORTFOLIO_PATH) as f:
            pf = json.load(f)
        for pos in pf.get("positions", []):
            _add(pos.get("ticker", ""))
        for opt in pf.get("options", []):
            _add(opt.get("underlying", ""))
        for t in pf.get("sector_watchlist", []):
            _add(t)
    except Exception:
        pass

    try:
        with open(WATCHLIST_PATH) as f:
            wl = json.load(f)
        for key in ("memory_core", "semis_equipment", "broader_semis", "portfolio",
                    "korean", "japanese", "etfs"):
            for t in wl.get(key, []):
                _add(t)
    except Exception:
        pass

    return tickers


def _load_cursor() -> dict:
    try:
        with open(CURSOR_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cursor(cursor: dict) -> None:
    CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CURSOR_PATH, "w") as f:
        json.dump(cursor, f)


def _article_id(ticker: str, direction: str, window: str) -> str:
    key = f"stocktwits_sentiment|{ticker}|{direction}|{window}"
    return hashlib.sha256(key.encode()).hexdigest()


def _ensure_seen_db() -> sqlite3.Connection:
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


def _fetch_sentiment(ticker: str) -> dict | None:
    """Fetch symbol stream and return {ticker, bull, bear, neutral, total, ratio, msgs}."""
    url = _ST_STREAM_URL.format(ticker=ticker, limit=MSG_LIMIT)
    try:
        r = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code == 429:
            print(f"[stocktwits_sentiment] rate-limited fetching {ticker}")
            return None
        if r.status_code != 200:
            return None
        data = r.json()
        messages = data.get("messages") or []
        bull = bear = neutral = 0
        samples: list[str] = []
        for m in messages:
            sentiment = (m.get("entities") or {}).get("sentiment") or {}
            basic = (sentiment.get("basic") or "").lower()
            if basic == "bullish":
                bull += 1
            elif basic == "bearish":
                bear += 1
            else:
                neutral += 1
            body = (m.get("body") or "").strip()
            if body and len(samples) < 3:
                samples.append(body[:120])
        total = bull + bear + neutral
        if total == 0:
            return None
        ratio = bull / (bull + bear) if (bull + bear) else 0.5
        return {
            "ticker": ticker,
            "bull": bull,
            "bear": bear,
            "neutral": neutral,
            "total": total,
            "ratio": ratio,
            "samples": samples,
        }
    except Exception as e:
        print(f"[stocktwits_sentiment] error fetching {ticker}: {e}")
        return None


def collect_stocktwits_sentiment() -> list[dict]:
    """Main entry point. Returns list of article-like dicts for extreme sentiment."""
    all_tickers = _load_tickers()
    if not all_tickers:
        print("[stocktwits_sentiment] no tickers loaded")
        return []

    cursor = _load_cursor()
    now_ts = time.time()

    # Filter tickers due for a refresh
    due = [
        t for t in all_tickers
        if now_ts - cursor.get(t, 0) >= PER_TICKER_COOLDOWN_SEC
    ]
    batch = due[:BATCH_PER_PASS]
    if not batch:
        print(f"[stocktwits_sentiment] all {len(all_tickers)} tickers on cooldown")
        return []

    print(f"[stocktwits_sentiment] fetching sentiment for {len(batch)} tickers: {batch}")

    results: list[dict] = []
    conn = _ensure_seen_db()

    def _process(ticker: str):
        time.sleep(0.3)  # light rate-limit: ~3 req/s max across workers
        return _fetch_sentiment(ticker)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_process, t): t for t in batch}
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                s = fut.result()
            except Exception:
                s = None

            cursor[ticker] = now_ts

            if s is None:
                continue

            bull_pct = s["bull"] / s["total"] * 100
            bear_pct = s["bear"] / s["total"] * 100
            ratio = s["ratio"]

            print(
                f"[stocktwits_sentiment] {ticker}: "
                f"{bull_pct:.0f}% bull / {bear_pct:.0f}% bear "
                f"({s['bull']}B {s['bear']}B̄ {s['neutral']}N of {s['total']})"
            )

            # Emit an article only when sentiment is extreme
            if ratio >= BULL_THRESHOLD:
                direction = "Bullish"
            elif ratio <= (1 - BEAR_THRESHOLD):
                direction = "Bearish"
            else:
                continue  # neutral — no signal

            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            title = (
                f"[StockTwits Sentiment] {ticker} {direction}: "
                f"{bull_pct:.0f}% Bullish / {bear_pct:.0f}% Bearish "
                f"({s['bull']}↑ {s['bear']}↓ of {s['total']} msgs)"
            )
            link = f"https://stocktwits.com/symbol/{ticker}"
            summary_parts = s["samples"]
            summary = " | ".join(summary_parts)
            article_id = _article_id(ticker, direction, now_str[:13])  # hourly window

            try:
                conn.execute(
                    "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
                    (article_id, link, title, "stocktwits/sentiment", now_str),
                )
                conn.commit()
            except Exception:
                pass

            results.append({
                "id": article_id,
                "title": title,
                "link": link,
                "summary": summary,
                "source": "stocktwits/sentiment",
                "published": now_str,
                "_tickers": [ticker],
                "_sentiment": direction.lower(),
                "_bull_pct": round(bull_pct, 1),
                "_bear_pct": round(bear_pct, 1),
            })

    conn.close()
    _save_cursor(cursor)
    print(f"[stocktwits_sentiment] done — {len(results)} extreme-sentiment signals emitted")
    return results


if __name__ == "__main__":
    articles = collect_stocktwits_sentiment()
    print(f"\n=== {len(articles)} signals ===")
    for a in articles:
        print(f"  {a['title']}")
        if a.get("summary"):
            print(f"    Sample: {a['summary'][:100]}")
