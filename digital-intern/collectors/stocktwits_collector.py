"""StockTwits collector — trending social messages, no API key needed.

Pulls the public trending stream (https://api.stocktwits.com/api/2/streams/
trending.json). No auth required. Each StockTwits message becomes an
article-like dict in the same shape the other collectors emit, so the daemon's
_ingest() path can score/store it unchanged.
"""
import html
import time

import requests

REQUEST_TIMEOUT = 8
MAX_ATTEMPTS = 2
RETRY_BACKOFF = 2.0  # seconds; doubled each retry

TRENDING_URL = "https://api.stocktwits.com/api/2/streams/trending.json"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0 Safari/537.36"
)


def _extract_tickers(msg: dict) -> list:
    """Cashtags for a message. Prefer the tokenized_body cashTag tokens, fall
    back to the prices[] symbols."""
    tickers: list = []
    seen: set = set()
    for tok in msg.get("tokenized_body") or []:
        if tok.get("type") == "cashTag":
            sym = (tok.get("data") or {}).get("symbol")
            if sym and sym not in seen:
                seen.add(sym)
                tickers.append(sym)
    if not tickers:
        for p in msg.get("prices") or []:
            sym = p.get("symbol")
            if sym and sym not in seen:
                seen.add(sym)
                tickers.append(sym)
    return tickers


def _fetch_trending() -> list:
    """GET the trending stream with retry-and-backoff. Returns the raw
    messages[] list, or [] on persistent failure."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    backoff = RETRY_BACKOFF
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.get(TRENDING_URL, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            return r.json().get("messages", []) or []
        except Exception as e:
            if attempt >= MAX_ATTEMPTS:
                print(f"[stocktwits] fetch failed after {attempt} attempts: {e}")
                return []
            print(f"[stocktwits] attempt {attempt} failed ({e}); retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff *= 2
    return []


def collect_stocktwits() -> list:
    """Pull trending StockTwits social messages as article-like dicts."""
    print("[stocktwits] Fetching trending stream...")
    t0 = time.time()

    messages = _fetch_trending()

    all_articles: list = []
    seen_urls: set = set()

    for msg in messages:
        mid = msg.get("id")
        if not mid:
            continue
        body = html.unescape((msg.get("body") or "").strip())
        if not body:
            continue

        link = f"https://stocktwits.com/message/{mid}"
        if link in seen_urls:
            continue
        seen_urls.add(link)

        tickers = _extract_tickers(msg)
        sentiment = (
            ((msg.get("entities") or {}).get("sentiment") or {}).get("basic") or ""
        )

        all_articles.append({
            "title": body[:200],
            "link": link,
            "summary": body[:1000],
            "published": msg.get("created_at", ""),
            "source": "stocktwits",
            "_tickers": tickers,
            "_stocktwits_sentiment": sentiment,
        })

    elapsed = time.time() - t0
    print(f"[stocktwits] Got {len(all_articles)} messages in {elapsed:.1f}s")
    return all_articles


if __name__ == "__main__":
    articles = collect_stocktwits()
    print(f"Total: {len(articles)}")
    for a in articles[:5]:
        print(f"  [{a['source']}] {a['title'][:100]}")
        print(f"    {a['link']}")
