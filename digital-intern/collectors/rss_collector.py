"""RSS collector with SQLite-based deduplication. Parallel fetch across feeds.

Per-feed backoff: a feed that 404s, 429s, or times out is parked in
data/rss_feed_backoff.json until its next_retry timestamp. With 302 feeds
re-polled every 60s, a handful of dead URLs were generating thousands of
identical error lines per day and burning worker slots; skipping them until
they're due to re-probe kills that noise without disabling them permanently.
"""
import json
import os
import sqlite3
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone

import feedparser
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCES_PATH = BASE_DIR / "config" / "sources.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
BACKOFF_PATH = BASE_DIR / "data" / "rss_feed_backoff.json"

MAX_WORKERS = int(os.environ.get("RSS_MAX_WORKERS", "96"))  # high-throughput default: parallel feed fetches
FETCH_TIMEOUT = 8  # seconds; bounds dead/slow feeds so they don't starve workers
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Cooldown ceilings per failure class (seconds). A feed re-probes after the
# cooldown so a fixed sources.json / recovered host heals itself.
_PERMANENT_COOLDOWN = 7 * 24 * 3600  # 404/410: gone, re-probe weekly
_RATELIMIT_BASE = 1800              # 429: 30m, exponential
_RATELIMIT_CAP = 6 * 3600          # ...capped at 6h
_TRANSIENT_BASE = 300              # timeout/5xx/conn: 5m, exponential
_TRANSIENT_CAP = 3600              # ...capped at 1h


def _load_backoff() -> dict:
    try:
        with open(BACKOFF_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _save_backoff(state: dict) -> None:
    try:
        BACKOFF_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = BACKOFF_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, separators=(",", ":"))
        os.replace(tmp, BACKOFF_PATH)
    except OSError as e:
        print(f"[rss_collector] backoff persist failed: {e}")


def _cooldown(outcome: str, fails: int, retry_after: float | None) -> float:
    """Seconds until the next probe for a feed with `fails` consecutive failures."""
    if outcome == "permanent":
        return _PERMANENT_COOLDOWN
    if outcome == "ratelimited":
        if retry_after and retry_after > 0:
            return min(retry_after, _RATELIMIT_CAP)
        return min(_RATELIMIT_CAP, _RATELIMIT_BASE * (2 ** max(0, fails - 1)))
    return min(_TRANSIENT_CAP, _TRANSIENT_BASE * (2 ** max(0, fails - 1)))


def _ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Hardened seen_articles.db connection — mirrors google_news._ensure_db /
    # source_health.py / article_store.py. 11 collectors share this one file;
    # SQLite's default busy_timeout=0 turns any transient cross-writer lock
    # into an immediate OperationalError that aborts the whole pass and drops
    # the fetched batch. WAL + 30s timeout lets the write wait out contention.
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY,
            link TEXT,
            title TEXT,
            source TEXT,
            first_seen TEXT
        )
        """
    )
    conn.commit()
    return conn


def _article_id(link: str, title: str) -> str:
    return hashlib.sha256(f"{link}||{title}".encode("utf-8")).hexdigest()


def _load_sources():
    with open(SOURCES_PATH, "r") as f:
        return json.load(f)


def _fetch_feed(feed: dict):
    """Fetch one feed. Returns (name, articles, outcome, retry_after) where
    outcome is one of: "ok" | "permanent" | "ratelimited" | "transient".
    The caller uses the outcome to drive per-feed backoff."""
    name = feed.get("name", "unknown")
    url = feed.get("url")
    if not url:
        return name, [], "ok", None
    try:
        resp = requests.get(
            url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA}
        )
        if resp.status_code in (404, 410):
            print(f"[rss_collector] {name} gone (HTTP {resp.status_code}) — parking")
            return name, [], "permanent", None
        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            try:
                ra = float(ra) if ra is not None else None
            except (TypeError, ValueError):
                ra = None
            print(f"[rss_collector] {name} rate-limited (HTTP 429) — backing off")
            return name, [], "ratelimited", ra
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as e:
        print(f"[rss_collector] Error fetching {name}: {e}")
        return name, [], "transient", None
    out: list = []
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
    return name, out, "ok", None


def collect_rss():
    """Collect deduplicated articles from configured RSS feeds (parallel).

    Returns a list of dicts: {title, link, summary, published, source}.
    """
    sources = _load_sources()
    feeds = sources.get("rss_feeds", []) if isinstance(sources, dict) else []

    # Pre-filter: skip feeds that are in backoff until their next_retry.
    backoff_state = _load_backoff()
    now = datetime.now(timezone.utc).timestamp()
    active_feeds, skipped = [], 0
    for feed in feeds:
        name = feed.get("name", "unknown")
        entry = backoff_state.get(name)
        if entry and entry.get("next_retry", 0) > now:
            skipped += 1
        else:
            active_feeds.append(feed)
    if skipped:
        print(f"[rss_collector] skipping {skipped} backed-off feeds, "
              f"fetching {len(active_feeds)}/{len(feeds)}")

    # Fetch in parallel — feedparser is HTTP-bound and benefits from threads.
    fetched: list = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_feed, f): f for f in active_feeds}
        for future in as_completed(futures):
            try:
                fetched.append(future.result())
            except Exception as e:
                print(f"[rss_collector] worker error: {e}")

    # Layer 1 — unpack the (name, articles, outcome, retry_after) tuple
    # defensively. A result that ever changes shape must skip that one feed,
    # never abort the whole pass. (Regression guard: _fetch_feed's return
    # contract changed from list -> 4-tuple but this loop still iterated each
    # result as a list of article dicts, so `art` became the `name` str and
    # `art["link"]` raised "string indices must be integers", which backed the
    # rss_worker off 300s and collapsed RSS throughput.)
    batches: list = []
    for result in fetched:
        try:
            name, arts, outcome, retry_after = result
        except (ValueError, TypeError):
            print(f"[rss_collector] skipping malformed feed result: "
                  f"{repr(result)[:120]}")
            continue
        if outcome == "ok":
            backoff_state.pop(name, None)
        else:
            prev = backoff_state.get(name, {})
            fails = prev.get("fails", 0) + 1
            cooldown = _cooldown(outcome, fails, retry_after)
            backoff_state[name] = {
                "next_retry": now + cooldown,
                "fails": fails,
                "outcome": outcome,
            }
        if isinstance(arts, list):
            batches.append(arts)

    _save_backoff(backoff_state)

    # Dedup in a single SQLite pass after parallel I/O. Layer 2 (per-article)
    # and Layer 3 (per-row DB) ensure one malformed entry or row-level hiccup
    # is skipped, not allowed to drop the whole fetched batch.
    conn = _ensure_db()
    new_articles: list = []
    seen_in_run: set = set()
    for batch in batches:
        for art in batch:
            try:
                link = art["link"]
                title = art["title"]
                source = art["source"]
            except (TypeError, KeyError):
                continue
            aid = _article_id(link, title)
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
                    "(id, link, title, source, first_seen) VALUES (?, ?, ?, ?, ?)",
                    (aid, link, title, source,
                     datetime.now(timezone.utc).isoformat()),
                )
            except sqlite3.Error as e:
                print(f"[rss_collector] dedup row skipped ({source}): {e}")
                continue
            new_articles.append(art)

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    items = collect_rss()
    print(f"Collected {len(items)} new articles")
    for a in items[:5]:
        print(f" - [{a['source']}] {a['title']}")
