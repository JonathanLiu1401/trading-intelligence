"""Hacker News collector — front-page + targeted finance/business stories.

Uses the public HN Algolia API (https://hn.algolia.com/api), which needs **no
auth and no API key**. Hacker News routinely surfaces tech-business breaking
news — outages, breaches, layoffs, earnings reactions, M&A, regulatory action,
product launches — often faster than mainstream RSS and not covered by any
existing collector (StockTwits is cashtag social sentiment; this is
news/discussion). Each story becomes an article-like dict in the same shape
the other collectors emit, so the daemon's ``_ingest()`` path can
score/store it unchanged.

Standalone usage / smoke test:

    python3 collectors/hackernews_collector.py

To wire into the daemon (intentionally NOT done here to avoid colliding with a
concurrent editor of daemon.py), register a worker mirroring the other
collectors::

    from collectors.hackernews_collector import collect_hackernews
    # in main(): _spawn("hackernews", collect_hackernews, interval=180)
"""
from __future__ import annotations

import html
import re
import time
from datetime import datetime, timezone

import requests

REQUEST_TIMEOUT = 8
MAX_ATTEMPTS = 2
RETRY_BACKOFF = 2.0  # seconds; doubled each retry

API_BASE = "https://hn.algolia.com/api/v1"
FRONT_PAGE_URL = f"{API_BASE}/search?tags=front_page&hitsPerPage=50"
SEARCH_URL = f"{API_BASE}/search_by_date"

# Targeted queries — HN's full-text search over recent stories. Kept small on
# purpose; the front page already catches broad tech-business momentum.
SEARCH_QUERIES = ("earnings", "acquisition", "layoffs", "SEC", "IPO")

HITS_PER_QUERY = 25
MAX_ARTICLES = 60

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0 Safari/537.36"
)

# A title is finance/markets/tech-business relevant if it mentions any of these.
# Applied to front-page hits (broad firehose); query hits are already topical.
_RELEVANCE_RE = re.compile(
    r"\b("
    r"earning|revenue|guidance|profit|loss|acqui|merger|m&a|buyout|takeover|"
    r"ipo|spac|stock|share|equit|market|nasdaq|nyse|dow|s&p|index|"
    r"layoff|hiring freeze|restructur|bankrupt|chapter 11|default|"
    r"sec|ftc|doj|antitrust|regulat|lawsuit|settle|fine|probe|investigat|"
    r"fed|federal reserve|rate cut|rate hike|inflation|recession|tariff|"
    r"valuation|funding|raise|series [a-e]|venture|billion|trillion|"
    r"outage|breach|hack|ransomware|data leak|recall|"
    r"earnings call|quarterly|fiscal|dividend|buyback|short seller|"
    r"chip|semiconductor|oil|crude|gold|bond|yield|treasury|crypto|bitcoin"
    r")\b",
    re.IGNORECASE,
)


def _fetch(url: str) -> list:
    """GET an Algolia endpoint with retry-and-backoff. Returns the ``hits``
    list, or [] on persistent failure / bad payload."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    backoff = RETRY_BACKOFF
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            payload = r.json()
            hits = payload.get("hits") if isinstance(payload, dict) else None
            return hits or []
        except Exception as e:  # network, JSON, non-200 — all non-fatal
            if attempt >= MAX_ATTEMPTS:
                print(f"[hackernews] fetch failed after {attempt} attempts "
                      f"({url.split('?')[0]}): {e}")
                return []
            print(f"[hackernews] attempt {attempt} failed ({e}); "
                  f"retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff *= 2
    return []


def _published_iso(created_at_i) -> str:
    """HN ``created_at_i`` (epoch seconds) -> tz-aware UTC ISO string.

    Uses ``datetime.now(timezone.utc)`` semantics (never the deprecated naive
    ``utcnow()``) so the stamp round-trips through paper_trader's parser.
    Falls back to *now* if the epoch is missing/garbage."""
    try:
        return datetime.fromtimestamp(int(created_at_i), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return datetime.now(timezone.utc).isoformat()


def _hit_to_article(hit: dict, require_relevant: bool) -> dict | None:
    """Convert one Algolia hit to the standard collector dict, or None to
    skip (missing title/id, or front-page hit that isn't market-relevant)."""
    if not isinstance(hit, dict):
        return None
    oid = hit.get("objectID")
    title = html.unescape((hit.get("title") or "").strip())
    if not oid or not title:
        return None

    if require_relevant and not _RELEVANCE_RE.search(title):
        return None

    discussion = f"https://news.ycombinator.com/item?id={oid}"
    url = (hit.get("url") or "").strip()
    link = url if url.startswith(("http://", "https://")) else discussion

    points = hit.get("points") or 0
    comments = hit.get("num_comments") or 0
    story_text = html.unescape((hit.get("story_text") or "").strip())
    summary = story_text[:1000] if story_text else (
        f"Hacker News story — {points} points, {comments} comments. "
        f"Discussion: {discussion}"
    )

    return {
        "title": title[:200],
        "link": link,
        "summary": summary,
        "published": _published_iso(hit.get("created_at_i")),
        "source": "hackernews",
        "_hn_id": str(oid),
        "_hn_points": int(points) if isinstance(points, (int, float)) else 0,
        "_hn_comments": int(comments) if isinstance(comments, (int, float)) else 0,
        "_hn_discussion": discussion,
    }


def collect_hackernews() -> list:
    """Pull HN front page + targeted finance/business searches as
    article-like dicts. Front-page hits are relevance-filtered; query hits
    are kept as-is (the query already makes them topical). Deduped by HN
    objectID and by resolved link, capped at ``MAX_ARTICLES``."""
    print("[hackernews] Fetching front page + targeted searches...")
    t0 = time.time()

    sources: list = [(FRONT_PAGE_URL, True)]
    for q in SEARCH_QUERIES:
        sources.append(
            (f"{SEARCH_URL}?tags=story&query={q}&hitsPerPage={HITS_PER_QUERY}",
             False)
        )

    articles: list = []
    seen_ids: set = set()
    seen_links: set = set()

    for url, require_relevant in sources:
        for hit in _fetch(url):
            art = _hit_to_article(hit, require_relevant)
            if art is None:
                continue
            if art["_hn_id"] in seen_ids or art["link"] in seen_links:
                continue
            seen_ids.add(art["_hn_id"])
            seen_links.add(art["link"])
            articles.append(art)
            if len(articles) >= MAX_ARTICLES:
                break
        if len(articles) >= MAX_ARTICLES:
            break

    elapsed = time.time() - t0
    print(f"[hackernews] Got {len(articles)} stories in {elapsed:.1f}s")
    return articles


if __name__ == "__main__":
    items = collect_hackernews()
    print(f"Total: {len(items)}")
    for a in items[:5]:
        print(f"  [{a['source']}] {a['title'][:100]}")
        print(f"    {a['link']}  ({a['_hn_points']}pts, {a['_hn_comments']}c)")
