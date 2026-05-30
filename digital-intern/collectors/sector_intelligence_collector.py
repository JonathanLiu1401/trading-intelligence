"""Sector intelligence collector — mining, offshore energy, utilities, solar.

Adds six RSS feeds not covered by energy_news_collector.py or sources.json:

    offshore_energy  — offshore-energy.biz (oil & gas exploration/production)
    power_magazine   — powermag.com (power generation, grid infrastructure)
    mining_com       — mining.com (copper, lithium, gold, critical minerals)
    ns_energy        — nsenergybusiness.com (energy sector deals/projects)
    utility_dive     — utilitydive.com (utility regulation, grid modernization)
    pv_magazine      — pv-magazine.com (solar, storage, clean energy)

Each feed drives distinct market sectors:
  - mining_com → critical minerals (EV, batteries, AI data-center supply chains)
  - offshore_energy → oil services (SLB, HAL, RIG, DO, OII)
  - power_magazine / utility_dive → utilities (NEE, AEP, DUK, SO, EXC)
  - pv_magazine → solar (FSLR, ENPH, SEDG, CSIQ, RUN)

Standard collector contract: returns list[dict] with {title, link, summary,
published, source, _tickers}. Dedup via data/seen_articles.db (WAL).
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

log = logging.getLogger("sector_intelligence")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

FETCH_TIMEOUT = 12
MAX_WORKERS = 6
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

SECTOR_FEEDS: dict[str, str] = {
    "offshore_energy": "https://www.offshore-energy.biz/feed/",
    "power_magazine":  "https://www.powermag.com/feed/",
    "mining_com":      "https://www.mining.com/feed/",
    "ns_energy":       "https://www.nsenergybusiness.com/feed/",
    "utility_dive":    "https://www.utilitydive.com/feeds/news/",
    "pv_magazine":     "https://www.pv-magazine.com/feed/",
}

# sector → (keywords, tickers)
_SECTOR_MAP: list[tuple[list[str], list[str]]] = [
    # Mining / critical minerals
    (["copper", "lithium", "cobalt", "nickel", "rare earth", "mining", "ore", "gold",
      "silver", "uranium", "tungsten", "manganese", "graphite"],
     ["FCX", "VALE", "RIO", "BHP", "ALB", "MP", "GOLD", "NEM", "RGLD", "LAC"]),
    # Offshore / oil services
    (["offshore", "deepwater", "subsea", "fpso", "drillship", "jack-up", "oil services",
      "exploration", "slb", "halliburton", "transocean"],
     ["SLB", "HAL", "RIG", "DO", "OII", "PTEN", "VAL", "FTI"]),
    # Power generation / utilities
    (["utility", "utilities", "grid", "transmission", "power plant", "generation",
      "nuclear", "natural gas power", "coal plant", "load growth", "data center power"],
     ["NEE", "AEP", "DUK", "SO", "EXC", "PCG", "D", "EVRG", "VST", "NRG"]),
    # Solar / storage / clean energy
    (["solar", "photovoltaic", "pv", "battery storage", "bess", "wind farm",
      "renewable energy", "clean energy", "energy storage"],
     ["FSLR", "ENPH", "SEDG", "CSIQ", "JKS", "ARRY", "RUN", "FLNC", "ICLN"]),
]


def _tag_tickers(title: str, summary: str) -> list[str]:
    text = (title + " " + summary).lower()
    tags: list[str] = []
    seen: set[str] = set()
    for keywords, tickers in _SECTOR_MAP:
        if any(kw in text for kw in keywords):
            for t in tickers:
                if t not in seen:
                    seen.add(t)
                    tags.append(t)
            if len(tags) >= 5:
                break
    return tags[:5]


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
    return hashlib.sha256(f"{link}||{title}".encode("utf-8")).hexdigest()


def _fetch_feed(name: str, url: str) -> list[dict]:
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA})
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as e:
        log.warning("[sector_intelligence] fetch error %s: %s", name, e)
        return []

    out: list[dict] = []
    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue
        summary = entry.get("summary") or entry.get("description") or ""
        published = entry.get("published") or entry.get("updated") or ""
        out.append({
            "title": title,
            "link": link,
            "summary": summary,
            "published": published,
            "source": name,
        })
    return out


def collect_sector_intelligence() -> list[dict]:
    """Fetch all sector feeds in parallel; return new deduplicated articles."""
    conn = _ensure_db()
    new_articles: list[dict] = []
    seen_in_run: set[str] = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_feed, name, url): name
                   for name, url in SECTOR_FEEDS.items()}
        for fut in as_completed(futures):
            for art in (fut.result() or []):
                aid = _article_id(art["link"], art["title"])
                if aid in seen_in_run:
                    continue
                seen_in_run.add(aid)
                try:
                    if conn.execute(
                        "SELECT 1 FROM seen_articles WHERE id = ?", (aid,)
                    ).fetchone():
                        continue
                    conn.execute(
                        "INSERT OR IGNORE INTO seen_articles "
                        "(id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
                        (aid, art["link"], art["title"], art["source"],
                         datetime.now(timezone.utc).isoformat()),
                    )
                except sqlite3.Error as e:
                    log.debug("[sector_intelligence] dedup skip %s: %s", art["source"], e)
                    continue
                art["_tickers"] = _tag_tickers(art["title"], art.get("summary", ""))
                new_articles.append(art)

    conn.commit()
    conn.close()
    return new_articles


collect = collect_sector_intelligence


if __name__ == "__main__":
    print("=== Sector Intelligence feeds (live fetch) ===")
    first_eg: str | None = None
    feed_counts: dict[str, int] = {}
    for _name, _url in SECTOR_FEEDS.items():
        raw = _fetch_feed(_name, _url)
        feed_counts[_name] = len(raw)
        print(f"  {_name:20s} {len(raw):3d} entries")
        if raw:
            sample = raw[0]["title"]
            print(f"    sample: {sample[:72]}")
            if first_eg is None:
                first_eg = f"{_name}: {sample}"

    items = collect_sector_intelligence()
    inserted = 0
    if items:
        try:
            from storage.article_store import ArticleStore
            store = ArticleStore()
            inserted = store.insert_batch(items)
        except Exception as e:
            print(f"  [article_store] {e}")

    print("\n=== Summary ===")
    print(f"New deduped items        : {len(items)}")
    print(f"Inserted into articles.db: {inserted}")
    if first_eg:
        print(f"DISCORD_EG: {first_eg}")
    for a in items[:8]:
        print(f"  + [{a['source']:18s}] {a['title'][:65]}  tickers={a.get('_tickers')}")
