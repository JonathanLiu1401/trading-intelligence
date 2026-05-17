"""Substack newsletter collector — pulls public posts from financial Substacks.

Tries the Substack public JSON API (`/api/v1/posts/?limit=...`) first per
publication, then falls back to the standard `/feed` RSS. Many financial
newsletters publish at least their headline & teaser publicly even when the
full post is paywalled — both endpoints are enough to catch headlines for
relevance scoring.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from urllib.parse import urlparse

import feedparser
import requests

REQUEST_TIMEOUT = 12
MAX_WORKERS = 12
USER_AGENT = "Mozilla/5.0 DigitalIntern/1.0 (financial-news-aggregator)"

# Each entry is a Substack publication subdomain or full domain. The collector
# handles both `<handle>.substack.com` and custom domains transparently.
SUBSTACK_PUBS = [
    # Macro / markets
    "kobeissiletter.substack.com",       # The Kobeissi Letter
    "thekobeissiletter.substack.com",    # alt spelling — kept just in case
    "doomberg.substack.com",
    "themacrocompass.substack.com",
    "thelastbearstanding.substack.com",
    "concoda.substack.com",
    "epsilontheory.substack.com",
    "wifeyalpha.substack.com",
    "noahpinion.blog",
    "themargin.substack.com",
    "alphapicks.substack.com",
    "alphainsider.substack.com",
    "markethuddle.substack.com",
    "financepoweruser.substack.com",
    "themacroresearch.substack.com",
    # Tech / semis / AI
    "www.platformer.news",               # tech / tech policy
    "stratechery.com",                   # subscription mostly, headlines still flow
    "semianalysis.substack.com",         # AI hardware & datacenter
    "www.thealgorithmicbridge.com",      # AI commentary
    "garymarcus.substack.com",           # AI hype reality check
    "www.interconnects.ai",              # AI / ML insider
    "tomtunguz.com",                     # VC, infra
    "www.bigtechnology.com",             # Big Tech
    # Quant / analysis
    "www.netinterest.co",
    "thedailyupside.com",
    "themargin.substack.com",
    "americanaffairs.substack.com",
]


def _normalize_pub(pub: str) -> str:
    """Strip trailing slash, return bare host."""
    if "://" in pub:
        return urlparse(pub).netloc
    return pub.rstrip("/")


def _api_posts(pub: str) -> list:
    """Try Substack JSON API. Returns parsed articles or [] on failure."""
    host = _normalize_pub(pub)
    url = f"https://{host}/api/v1/posts/?limit=25&offset=0"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                         timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        data = r.json()
        if not isinstance(data, list):
            return []
    except Exception:
        return []

    out = []
    for post in data:
        title = (post.get("title") or "").strip()
        link = post.get("canonical_url") or post.get("url") or ""
        if not title or not link:
            continue
        subtitle = post.get("subtitle") or ""
        desc = post.get("description") or ""
        body_text = unescape((subtitle + " " + desc).strip())
        published = post.get("post_date") or post.get("published_at") or ""
        out.append({
            "title": title[:200],
            "link": link,
            "summary": body_text[:1000],
            "published": published,
            "source": f"substack/{host}",
        })
    return out


def _rss_posts(pub: str) -> list:
    """Fallback to /feed RSS."""
    host = _normalize_pub(pub)
    url = f"https://{host}/feed"
    try:
        parsed = feedparser.parse(url, agent=USER_AGENT)
        if not getattr(parsed, "entries", None):
            return []
    except Exception:
        return []

    out = []
    for entry in parsed.entries[:30]:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue
        summary = entry.get("summary") or entry.get("description") or ""
        published = entry.get("published") or entry.get("updated") or ""
        out.append({
            "title": title[:200],
            "link": link,
            "summary": summary[:1000],
            "published": published,
            "source": f"substack/{host}",
        })
    return out


def _fetch_pub(pub: str) -> list:
    items = _api_posts(pub)
    if items:
        return items
    return _rss_posts(pub)


def collect_substack() -> list:
    """Pull recent posts from all configured Substack publications in parallel."""
    print(f"[substack] {len(SUBSTACK_PUBS)} publications")
    t0 = time.time()
    all_articles: list = []
    seen_links: set = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_pub, p): p for p in SUBSTACK_PUBS}
        for fut in as_completed(futures):
            try:
                batch = fut.result()
            except Exception:
                batch = []
            for art in batch:
                link = art["link"]
                if link and link not in seen_links:
                    seen_links.add(link)
                    all_articles.append(art)

    elapsed = time.time() - t0
    print(f"[substack] {len(all_articles)} posts in {elapsed:.1f}s")
    return all_articles


if __name__ == "__main__":
    items = collect_substack()
    print(f"Total: {len(items)}")
    for a in items[:10]:
        print(f"  [{a['source']}] {a['title'][:90]}")
