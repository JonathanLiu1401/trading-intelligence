"""Think tank and academic economics research collector.

Aggregates high-quality economic and policy research from sources not already
covered by the main RSS feed config. These sources publish analysis that can
move markets: rate forecasts, trade policy, geopolitical risk, monetary policy,
and macroeconomic outlooks.

Sources:
  ProMarket (Chicago Booth)  — antitrust, SEC, market regulation
  Phenomenal World           — political economy analysis
  Apricitas Economics        — macro data deep-dives (widely read by traders)
  Employ America             — labor market + monetary policy analysis
  VoxEU/CEPR                 — European economic research (200+ CEPR Fellows)
"""
import hashlib
import sqlite3
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "articles.db"

REQUEST_TIMEOUT = 12
MAX_WORKERS = 8

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# All feeds verified working — not already in config/sources.json
FEEDS = [
    ("ProMarket/ChicagoBooth", "https://promarket.org/feed/"),
    ("PhenomenalWorld", "https://www.phenomenalworld.org/feed/"),
    ("ApritasEconomics", "https://www.apricitas.io/feed"),
    ("EmployAmerica", "https://employamerica.org/researchreports/feed/"),
    ("VoxEU/CEPR", "https://voxeu.org/rss.xml"),
]

FINANCE_KEYWORDS = {
    "inflation", "deflation", "gdp", "recession", "growth", "employment",
    "unemployment", "jobs", "fed", "federal reserve", "ecb", "rate",
    "interest", "bond", "yield", "equity", "stock", "market", "trade",
    "tariff", "china", "dollar", "currency", "debt", "deficit", "surplus",
    "fiscal", "monetary", "policy", "banking", "credit", "default",
    "housing", "mortgage", "energy", "oil", "commodity", "geopolit",
    "sanction", "war", "supply chain", "earnings", "profit", "revenue",
    "technology", "ai", "semiconductor", "crypto", "bitcoin", "central bank",
    "treasury", "imf", "g7", "g20", "oecd", "forecast", "outlook", "risk",
    "volatility", "invest", "antitrust", "regulation", "sec", "labor",
    "wage", "price", "consumer", "spending", "retail", "manufacturing",
}


def _kw_score(text: str) -> float:
    low = text.lower()
    hits = sum(1 for kw in FINANCE_KEYWORDS if kw in low)
    return min(hits / 5.0, 1.0)


def _article_id(url: str, title: str) -> str:
    return hashlib.sha256(f"{url}|{title}".encode()).hexdigest()[:32]


def _fetch_feed(source: str, url: str) -> list[dict]:
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return []

    d = feedparser.parse(resp.content)
    results = []
    for entry in d.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue
        summary = (entry.get("summary") or entry.get("description") or "")[:1000]
        published = entry.get("published") or entry.get("updated") or ""
        results.append({
            "title": title,
            "url": link,
            "summary": summary,
            "published": published,
            "source": source,
        })
    return results


def collect_think_tank_research() -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")

    raw: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_feed, src, url): src for src, url in FEEDS}
        for fut in as_completed(futures):
            raw.extend(fut.result())

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    new_articles: list[dict] = []

    for art in raw:
        aid = _article_id(art["url"], art["title"])
        if conn.execute("SELECT 1 FROM articles WHERE id=?", (aid,)).fetchone():
            continue
        score = _kw_score(art["title"] + " " + art["summary"])
        full_text_blob = zlib.compress(
            (art["title"] + "\n\n" + art["summary"]).encode("utf-8", errors="replace")
        )
        conn.execute(
            """INSERT OR IGNORE INTO articles
               (id, url, title, source, published, kw_score, full_text, first_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                aid,
                art["url"],
                art["title"],
                art["source"],
                art["published"],
                score,
                full_text_blob,
                now_iso,
            ),
        )
        new_articles.append({**art, "kw_score": score})

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    t0 = time.time()
    items = collect_think_tank_research()
    dt = time.time() - t0
    print(f"[think_tank_research] {len(items)} new items in {dt:.1f}s")
    for a in items[:10]:
        score_str = f"kw={a.get('kw_score', 0):.2f}"
        print(f"  [{a['source']}] ({score_str}) {a['title'][:80]}")
