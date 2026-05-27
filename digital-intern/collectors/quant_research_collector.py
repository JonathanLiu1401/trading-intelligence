"""Quant research blog aggregator — systematic/factor investing signal.

Fetches posts from curated quantitative finance research blogs not covered
by the main RSS collector (sources.json) or arxiv_qfin_collector. These
sources publish systematic strategy research, factor investing analysis, and
options pricing studies — complementary to fast news flow.

Sources:
  quantocracy      - Curated weekly roundup of 300+ quant blogs (no auth)
  alpha_architect  - Factor investing research from Alpha Architect (Wes Gray)
  newfound         - Systematic investing / factor timing (Newfound Research)
  predicting_alpha - Options and alpha research blog

Dedup: shared data/seen_articles.db (same schema as all other collectors).
No API key required. Public RSS feeds.

Standalone smoke test:
    python3 collectors/quant_research_collector.py
"""
from __future__ import annotations

import html
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

REQUEST_TIMEOUT = 12
MAX_ATTEMPTS = 2
RETRY_BACKOFF = 2.0
MAX_PER_FEED = 30
SUMMARY_CAP = 800

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

FEEDS: list[tuple[str, str]] = [
    ("quantocracy",     "https://quantocracy.com/feed/"),
    ("alpha_architect", "https://alphaarchitect.com/feed/"),
    ("newfound_research", "https://blog.thinknewfound.com/feed/"),
    ("predicting_alpha", "https://predictingalpha.com/feed"),
]


def _fetch(url: str) -> bytes:
    last_err: Exception | None = None
    delay = RETRY_BACKOFF
    for _ in range(MAX_ATTEMPTS):
        try:
            r = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={
                    "User-Agent": _UA,
                    "Accept": "application/rss+xml,application/xml,text/xml,*/*",
                },
            )
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(delay)
            delay *= 2
    print(f"[quant_research] fetch failed {url}: {last_err}")
    return b""


def _published_iso(entry) -> str:
    pp = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if pp:
        try:
            return datetime(*pp[:6], tzinfo=timezone.utc).isoformat()
        except Exception:  # noqa: BLE001
            pass
    return datetime.now(timezone.utc).isoformat()


def _strip_html(raw: str) -> str:
    import re
    clean = re.sub(r"<[^>]+>", " ", raw)
    clean = html.unescape(clean)
    return " ".join(clean.split())


def _entry_to_article(entry, source: str) -> dict | None:
    title = html.unescape((getattr(entry, "title", "") or "").strip())
    link = (getattr(entry, "link", "") or "").strip()
    if not title or not link:
        return None
    if not link.startswith(("http://", "https://")):
        return None

    raw_summary = (
        getattr(entry, "summary", "")
        or getattr(entry, "description", "")
        or ""
    )
    summary = _strip_html(raw_summary)[:SUMMARY_CAP] or f"Quant research: {title}"

    return {
        "title": title[:300],
        "link": link,
        "summary": summary,
        "published": _published_iso(entry),
        "source": source,
    }


def collect_quant_research() -> list[dict]:
    print("[quant_research] Fetching quant research blogs...")
    t0 = time.time()
    out: list[dict] = []
    seen_links: set[str] = set()
    seen_titles: set[str] = set()

    for source, url in FEEDS:
        body = _fetch(url)
        if not body:
            print(f"[quant_research] {source}: empty response")
            continue
        parsed = feedparser.parse(body)
        entries = getattr(parsed, "entries", []) or []
        kept = 0
        for entry in entries:
            if kept >= MAX_PER_FEED:
                break
            art = _entry_to_article(entry, source)
            if art is None:
                continue
            link_key = art["link"].rstrip("/").lower()
            title_key = art["title"].lower()[:80]
            if link_key in seen_links or title_key in seen_titles:
                continue
            seen_links.add(link_key)
            seen_titles.add(title_key)
            out.append(art)
            kept += 1
        print(f"[quant_research] {source}: {kept} items (feed had {len(entries)} entries)")

    elapsed = time.time() - t0
    print(f"[quant_research] Total {len(out)} articles in {elapsed:.1f}s")
    return out


if __name__ == "__main__":
    items = collect_quant_research()
    print(f"\nTotal: {len(items)}")
    for a in items[:6]:
        print(f"  [{a['source']}] {a['title'][:90]}")
        print(f"    {a['link']}")
        print(f"    {a['summary'][:100]}")
        print()
