"""Nitter (Twitter mirror) collector — no API key.

Scrapes tweets from financial accounts and ticker-cashtag searches via the
public Nitter web mirrors. Iterates a list of instances per request so that
when one is down/throttled we transparently fail over to the next.
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

REQUEST_TIMEOUT = 12
MAX_WORKERS = 12
PER_DOMAIN_DELAY = 2.0  # seconds between consecutive requests to the same instance

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0 Safari/537.36"
)

# Rotated in order; if one returns non-200 or empty, fall through to the next.
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.1d4.us",
    "https://nitter.kavin.rocks",
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
    "https://nitter.fdn.fr",
    "https://nitter.unixfox.eu",
]

# Financial / market-moving accounts. Mix of insiders, journalists, and famous traders.
ACCOUNTS = [
    "KobeissiLetter", "jimcramer", "elonmusk", "DeItaone", "elerianm",
    "WSJ", "Reuters", "markets", "FinancialTimes", "CNBC", "business",
    "WSJmarkets", "zerohedge", "Stocktwits", "unusual_whales",
    "SquawkCNBC", "MarketWatch", "CNBCnow", "FirstSquawk",
    "PiQSuite", "LiveSquawk", "DiMartinoBooth", "PeterSchiff",
    "RaoulGMI", "biancoresearch", "MichaelKantro", "SethCL",
    "michaeljburry", "WatcherGuru", "DougKass",
]

# Cashtag searches — capped so we don't hit instance rate limits.
SEARCH_TICKERS = [
    "NVDA", "MU", "AMD", "ASML", "TSM", "INTC", "ORCL", "MSFT", "META",
    "AAPL", "AVGO", "QCOM", "SMCI", "LITE", "AMAT", "LRCX", "KLAC",
    "AXTI", "TSEM", "QBTS",
]

_last_hit: dict[str, float] = {}


def _throttle(instance: str):
    """Per-instance courtesy delay so we don't get rate-limited."""
    now = time.time()
    last = _last_hit.get(instance, 0.0)
    wait = (last + PER_DOMAIN_DELAY) - now
    if wait > 0:
        time.sleep(wait)
    _last_hit[instance] = time.time()


def _get_with_failover(path: str) -> tuple[str, str] | None:
    """Fetch the given path (e.g. "/jimcramer" or "/search?...") trying each
    Nitter instance in order. Returns (html, instance_base) on success."""
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html"}
    for base in NITTER_INSTANCES:
        url = base + path
        _throttle(base)
        try:
            r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                continue
            html = r.text
            # Some Nitter mirrors return a landing page when rate-limited.
            if "tweet-content" not in html and "timeline-item" not in html:
                continue
            return html, base
        except Exception:
            continue
    return None


def _parse_tweets(html: str, source_tag: str, base_url: str) -> list:
    """Extract tweet objects from a Nitter timeline or search-results page."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return []

    out: list = []
    seen = set()
    for item in soup.select("div.timeline-item"):
        content = item.select_one(".tweet-content")
        if not content:
            continue
        text = content.get_text(" ", strip=True)
        if not text or len(text) < 10:
            continue

        # @username on the tweet (may differ from the page we fetched, e.g. retweets)
        user_node = item.select_one("a.username")
        user = user_node.get_text(strip=True) if user_node else ""

        # Link to the canonical tweet (Nitter rewrites paths to its own domain).
        link_node = item.select_one("a.tweet-link")
        href = link_node.get("href") if link_node else ""
        if href:
            link = "https://twitter.com" + href.replace("#m", "")
        else:
            link = base_url

        # Timestamp — Nitter renders absolute time in title="..."
        date_node = item.select_one("span.tweet-date a")
        published = ""
        if date_node:
            published = date_node.get("title") or date_node.get_text(strip=True)

        title = text[:200]
        if title in seen:
            continue
        seen.add(title)

        out.append({
            "title": f"{user}: {title}" if user else title,
            "link": link,
            "summary": text[:1000],
            "published": published,
            "source": source_tag,
        })
    return out


def _fetch_account(username: str) -> list:
    result = _get_with_failover(f"/{username}")
    if not result:
        return []
    html, base = result
    return _parse_tweets(html, f"twitter_nitter/@{username}", base)


def _fetch_search(query: str) -> list:
    path = f"/search?q={quote_plus(query)}&f=tweets"
    result = _get_with_failover(path)
    if not result:
        return []
    html, base = result
    return _parse_tweets(html, f"twitter_nitter/search:{query}", base)


def collect_nitter() -> list:
    """Pull tweets from financial accounts + cashtag searches in parallel."""
    print(f"[nitter] {len(ACCOUNTS)} accounts + {len(SEARCH_TICKERS)} cashtags")
    t0 = time.time()

    all_articles: list = []
    seen_links: set = set()

    jobs: list = []
    for u in ACCOUNTS:
        jobs.append(("account", u))
    for t in SEARCH_TICKERS:
        jobs.append(("search", f"${t}"))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for kind, arg in jobs:
            fn = _fetch_account if kind == "account" else _fetch_search
            futures[ex.submit(fn, arg)] = (kind, arg)
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
    print(f"[nitter] {len(all_articles)} tweets in {elapsed:.1f}s")
    return all_articles


if __name__ == "__main__":
    items = collect_nitter()
    print(f"Total: {len(items)}")
    for a in items[:10]:
        print(f"  [{a['source']}] {a['title'][:100]}")
