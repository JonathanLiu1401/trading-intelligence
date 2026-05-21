"""Polymarket prediction market collector — market-implied probabilities.

Fetches active prediction markets from Polymarket's Gamma API, filters for
financial/macro relevance, and emits synthetic articles so the rest of the
pipeline can label and act on crowd-sourced probability signals.

Why this matters for trading:
  Prediction markets aggregate dispersed information into probabilities that
  often lead news coverage. A market pricing Fed-cut at 85% before the FOMC
  announcement or a tariff-escalation at 60% are actionable signals.

Dedup strategy:
  Each market is keyed by its Polymarket slug + today's date, so the same
  market is re-emitted at most once per day (allowing daily probability
  updates to surface as new articles).

No API key required. Rate-limit: Polymarket public API has no published limit;
we fetch once per daemon cycle (60s cadence) which is well within safe use.
"""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
SEEN_DB   = BASE_DIR / "data" / "seen_articles.db"

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
POLY_BASE = "https://polymarket.com/event/"

FETCH_TIMEOUT  = 15
MAX_MARKETS    = 200   # top by volume
MIN_LIQUIDITY  = 100   # skip dust markets
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

FIN_KEYWORDS = {
    "fed", "rate", "inflation", "cpi", "pce", "gdp", "recession", "tariff",
    "unemployment", "treasury", "interest", "stock", "bitcoin", "btc",
    "dollar", "economy", "trade", "deficit", "debt", "earnings", "ipo",
    "crypto", "oil", "gold", "market", "s&p", "nasdaq", "dow", "trump",
    "china", "nvidia", "apple", "tesla", "bank", "fomc", "jobs", "payroll",
    "yields", "bonds", "sanctions", "imf", "world bank", "opec",
    "etf", "merger", "acquisition", "bankruptcy", "default",
}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _ensure_db() -> sqlite3.Connection:
    SEEN_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SEEN_DB), timeout=30, check_same_thread=False)
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


def _seen_id(slug: str, date: str) -> str:
    return hashlib.sha256(f"polymarket:{slug}:{date}".encode()).hexdigest()


def _is_seen(conn: sqlite3.Connection, sid: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM seen_articles WHERE id=?", (sid,)
    ).fetchone() is not None


def _mark_seen(conn: sqlite3.Connection, sid: str, link: str, title: str):
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
        "VALUES (?, ?, ?, ?, ?)",
        (sid, link, title, "polymarket", datetime.now(timezone.utc).isoformat()),
    )


def _fetch_markets() -> list[dict]:
    params = {
        "active": "true",
        "closed": "false",
        "limit": MAX_MARKETS,
        "order": "volume",
        "ascending": "false",
    }
    r = requests.get(GAMMA_URL, params=params, headers={"User-Agent": _UA},
                     timeout=FETCH_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _is_financial(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in FIN_KEYWORDS)


def _parse_json_field(raw) -> list:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return []
    return raw or []


def _format_probs(outcomes: list, prices: list) -> str:
    pairs = []
    for outcome, price in zip(outcomes, prices):
        pct = round(float(price) * 100, 1)
        pairs.append(f"{outcome}: {pct}%")
    return " | ".join(pairs)


def _build_article(market: dict, today: str) -> dict:
    question  = market.get("question", "Unknown market")
    slug      = market.get("slug", "")
    outcomes  = _parse_json_field(market.get("outcomes", "[]"))
    prices    = _parse_json_field(market.get("outcomePrices", "[]"))
    vol       = float(market.get("volume") or 0)
    liq       = float(market.get("liquidity") or 0)
    end_date  = market.get("endDate", "")[:10] if market.get("endDate") else "?"
    description = market.get("description", "") or ""

    prob_str = _format_probs(outcomes, prices)

    title = f"[Polymarket] {question} → {prob_str}"
    link  = f"{POLY_BASE}{slug}" if slug else POLY_BASE

    summary_parts = [
        f"Probabilities: {prob_str}",
        f"Volume: ${vol:,.0f} | Liquidity: ${liq:,.0f}",
        f"Resolves: {end_date}",
    ]
    if description:
        summary_parts.append(description[:400])

    return {
        "title":     title,
        "link":      link,
        "summary":   "\n".join(summary_parts),
        "published": today,
        "source":    "polymarket",
    }


def collect() -> list[dict]:
    """Fetch top Polymarket markets, filter for financial relevance, return articles."""
    today = _today()
    conn  = _ensure_db()
    items: list[dict] = []

    try:
        markets = _fetch_markets()
    except Exception as e:
        print(f"[polymarket] fetch failed: {e}")
        return []

    financial = [
        m for m in markets
        if _is_financial(m.get("question", ""))
        and float(m.get("liquidity") or 0) >= MIN_LIQUIDITY
    ]

    for m in financial:
        slug = m.get("slug", "")
        sid  = _seen_id(slug, today)
        if _is_seen(conn, sid):
            continue

        article = _build_article(m, today)
        _mark_seen(conn, sid, article["link"], article["title"])
        items.append(article)

    conn.commit()
    conn.close()

    print(f"[polymarket] {len(financial)} financial markets found, {len(items)} new")
    return items


if __name__ == "__main__":
    articles = collect()
    print(f"\nTotal new articles: {len(articles)}")
    for a in articles:
        print(f"  + {a['title'][:100]}")

    if articles:
        from storage.article_store import ArticleStore
        store = ArticleStore()
        n = store.insert_batch(articles)
        print(f"\nInserted {n} into articles.db")
