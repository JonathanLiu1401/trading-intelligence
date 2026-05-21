"""Market-wide equity put/call ratio collector.

Aggregates options volume across major ETFs (SPY, QQQ, IWM, DIA, GLD, TLT)
to produce a market-wide put/call ratio — a key contrarian sentiment indicator.
A high ratio (>1.2) signals fear/bearish positioning; low (<0.7) signals greed.

Emits a synthetic article row when:
  - The hourly ratio bucket changes (e.g. 0.8→0.9 range)
  - The ratio crosses a named sentiment threshold (extreme fear, fear, neutral,
    greed, extreme greed)

Dedup key: date + ratio-bucket so at most one article per 0.1 increment per day.
"""
import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

SOURCE_NAME = "CBOE Put/Call"
# Major liquid ETFs + indices to compute aggregate P/C ratio
TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD", "TLT"]
# Number of nearest expirations to include
MAX_EXPIRATIONS = 2

log = logging.getLogger("putcall_ratio_collector")


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


def _sentiment(ratio: float) -> tuple[str, str]:
    """Return (label, emoji) for a put/call ratio."""
    if ratio >= 1.3:
        return "Extreme Fear", "😱"
    if ratio >= 1.0:
        return "Fear", "😟"
    if ratio >= 0.8:
        return "Neutral", "😐"
    if ratio >= 0.6:
        return "Greed", "🤑"
    return "Extreme Greed", "🚀"


def _fetch_options_volume(ticker: str) -> tuple[float, float]:
    """Return (total_call_volume, total_put_volume) across near-term expirations."""
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return 0.0, 0.0

        calls_total = 0.0
        puts_total = 0.0
        for exp in exps[:MAX_EXPIRATIONS]:
            try:
                chain = t.option_chain(exp)
                calls_total += float(chain.calls["volume"].fillna(0).sum())
                puts_total += float(chain.puts["volume"].fillna(0).sum())
            except Exception:
                continue
        return calls_total, puts_total
    except Exception as e:
        log.debug(f"[putcall] {ticker} fetch error: {e}")
        return 0.0, 0.0


def collect_putcall_ratio() -> list[dict]:
    """Aggregate put/call ratio across major ETFs; return net-new article dicts."""
    total_calls = 0.0
    total_puts = 0.0
    ticker_data: dict[str, dict] = {}

    for ticker in TICKERS:
        c, p = _fetch_options_volume(ticker)
        if c > 0 or p > 0:
            ticker_data[ticker] = {"calls": c, "puts": p}
            total_calls += c
            total_puts += p

    if total_calls < 1000:
        log.warning(f"[putcall] insufficient data: calls={total_calls:.0f}")
        return []

    ratio = total_puts / total_calls if total_calls > 0 else 0.0
    sentiment, emoji = _sentiment(ratio)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Dedup bucket: 0.1-wide ratio band per day
    bucket = int(ratio * 10) / 10.0
    dedup_key = f"putcall|{today}|{bucket:.1f}"
    art_id = _article_id(dedup_key)

    ticker_summary = " | ".join(
        f"{t}: {v['puts']/v['calls']:.2f}"
        for t, v in ticker_data.items()
        if v["calls"] > 0
    )
    link = "https://www.cboe.com/us/options/market_statistics/daily/"

    title = (
        f"{emoji} Market Put/Call Ratio: {ratio:.3f} ({sentiment}) | "
        f"Puts: {total_puts:,.0f}  Calls: {total_calls:,.0f} | "
        f"{today}"
    )
    summary = (
        f"Aggregate equity options put/call ratio across SPY/QQQ/IWM/DIA/GLD/TLT "
        f"stands at {ratio:.3f} ({sentiment}). "
        f"Total puts: {total_puts:,.0f}, total calls: {total_calls:,.0f} "
        f"(nearest {MAX_EXPIRATIONS} expirations). "
        f"Per-ticker ratios — {ticker_summary}. "
        f"A ratio >1.0 signals bearish hedging; <0.7 signals bullish complacency."
    )

    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    _ensure_db(conn)
    results: list[dict] = []

    try:
        if conn.execute(
            "SELECT 1 FROM seen_articles WHERE id=?", (art_id,)
        ).fetchone():
            log.debug(f"[putcall] already seen bucket {bucket:.1f} today")
            return []

        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?,?,?,?,?)",
            (art_id, link, title, SOURCE_NAME, now),
        )
        conn.commit()

        results.append({
            "id": art_id,
            "link": link,
            "title": title,
            "summary": summary,
            "source": SOURCE_NAME,
            "first_seen": now,
            "putcall_ratio": ratio,
            "putcall_sentiment": sentiment,
            "total_puts": total_puts,
            "total_calls": total_calls,
        })
    finally:
        conn.close()

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    articles = collect_putcall_ratio()
    if articles:
        for a in articles:
            print(f"NEW: {a['title']}")
            print(f"     {a['summary'][:200]}")
            print(f"     P/C ratio: {a['putcall_ratio']:.4f}")
    else:
        print("No new put/call ratio articles (bucket already seen today)")
