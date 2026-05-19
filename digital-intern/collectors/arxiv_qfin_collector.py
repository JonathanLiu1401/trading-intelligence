"""arXiv q-fin + econ RSS collector — fresh quantitative-finance/economics
papers.

Source of structural research signal not covered elsewhere: new papers on
market microstructure, risk, trading strategies, and econometrics. Useful as
slow-moving context (e.g., novel anomalies, methodologies) alongside the
fast news flow.

Feeds:
  https://export.arxiv.org/rss/q-fin
  https://export.arxiv.org/rss/econ

Each item -> standard collector article dict consumed by daemon._ingest().

Standalone smoke test:
    python3 collectors/arxiv_qfin_collector.py
"""
from __future__ import annotations

import html
import time
from datetime import datetime, timezone

import feedparser
import requests

REQUEST_TIMEOUT = 12
MAX_ATTEMPTS = 2
RETRY_BACKOFF = 2.0

FEEDS = (
    ("arxiv_qfin", "https://export.arxiv.org/rss/q-fin"),
    ("arxiv_econ", "https://export.arxiv.org/rss/econ"),
)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) digital-intern/arxiv-collector"
)

MAX_PER_FEED = 60
MAX_ARTICLES = 120
SUMMARY_CAP = 1200


def _fetch(url: str) -> str:
    last_err: Exception | None = None
    delay = RETRY_BACKOFF
    for _ in range(MAX_ATTEMPTS):
        try:
            r = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml,application/xml,text/xml,*/*"},
            )
            r.raise_for_status()
            return r.text
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(delay)
            delay *= 2
    print(f"[arxiv] fetch failed {url}: {last_err}")
    return ""


def _published_iso(entry) -> str:
    pp = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if pp:
        try:
            return datetime(*pp[:6], tzinfo=timezone.utc).isoformat()
        except Exception:  # noqa: BLE001
            pass
    return datetime.now(timezone.utc).isoformat()


def _entry_to_article(entry, source: str) -> dict | None:
    title = html.unescape((getattr(entry, "title", "") or "").strip())
    link = (getattr(entry, "link", "") or "").strip()
    if not title or not link:
        return None
    if not link.startswith(("http://", "https://")):
        return None

    raw_summary = html.unescape((getattr(entry, "summary", "") or "").strip())
    # Strip arXiv's "Announce Type" prefix noise if present.
    if "Abstract:" in raw_summary:
        raw_summary = raw_summary.split("Abstract:", 1)[1].strip()
    summary = raw_summary[:SUMMARY_CAP]

    authors = ""
    if getattr(entry, "authors", None):
        authors = ", ".join(
            (a.get("name") if isinstance(a, dict) else str(a))
            for a in entry.authors
        )[:300]
    elif getattr(entry, "author", None):
        authors = str(entry.author)[:300]

    categories: list[str] = []
    tags = getattr(entry, "tags", None) or []
    for t in tags:
        term = t.get("term") if isinstance(t, dict) else None
        if term:
            categories.append(term)

    return {
        "title": title[:300],
        "link": link,
        "summary": summary or f"arXiv paper: {title}",
        "published": _published_iso(entry),
        "source": source,
        "_arxiv_authors": authors,
        "_arxiv_categories": ",".join(categories[:8]),
    }


def collect_arxiv_qfin() -> list:
    print("[arxiv] Fetching q-fin + econ feeds...")
    t0 = time.time()
    out: list = []
    seen_links: set = set()
    seen_titles: set = set()

    for source, url in FEEDS:
        body = _fetch(url)
        if not body:
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
            if art["link"] in seen_links or art["title"] in seen_titles:
                continue
            seen_links.add(art["link"])
            seen_titles.add(art["title"])
            out.append(art)
            kept += 1
            if len(out) >= MAX_ARTICLES:
                break
        print(f"[arxiv] {source}: {kept} items")
        if len(out) >= MAX_ARTICLES:
            break

    elapsed = time.time() - t0
    print(f"[arxiv] Total {len(out)} papers in {elapsed:.1f}s")
    return out


if __name__ == "__main__":
    items = collect_arxiv_qfin()
    print(f"Total: {len(items)}")
    for a in items[:5]:
        print(f"  [{a['source']}] {a['title'][:110]}")
        print(f"    {a['link']}")
        print(f"    cats={a['_arxiv_categories']}")
