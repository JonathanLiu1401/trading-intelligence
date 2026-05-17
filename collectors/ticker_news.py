"""Per-ticker news scraper using yfinance."""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
MAX_WORKERS = 15


def _load_all_tickers() -> list:
    with open(WATCHLIST_PATH) as f:
        wl = json.load(f)
    tickers = []
    for key in ("memory_core", "semis_equipment", "broader_semis", "korean",
                "japanese", "etfs", "portfolio", "indices"):
        tickers.extend(wl.get(key, []))
    seen = set()
    return [t for t in tickers if not (t in seen or seen.add(t))]


def _fetch_ticker_news(ticker: str) -> list:
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        results = []
        for item in news:
            content = item.get("content", {})
            title = content.get("title") or item.get("title", "")
            link = (content.get("canonicalUrl", {}) or {}).get("url") or item.get("link", "")
            summary = content.get("summary") or ""
            pub = content.get("pubDate") or item.get("providerPublishTime", "")
            provider = (content.get("provider") or {}).get("displayName") or item.get("publisher", "")
            if title and link:
                results.append({
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "published": str(pub),
                    "source": f"yfinance/{provider or ticker}",
                    "_ticker": ticker,
                })
        return results
    except Exception:
        return []


def collect_ticker_news() -> list:
    """Fetch news for all watchlist tickers in parallel."""
    tickers = _load_all_tickers()
    print(f"[ticker_news] Fetching news for {len(tickers)} tickers...")
    t0 = time.time()

    all_articles = []
    seen_urls = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_ticker_news, t): t for t in tickers}
        for future in as_completed(futures):
            for art in future.result():
                url = art["link"]
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_articles.append(art)

    elapsed = time.time() - t0
    print(f"[ticker_news] Got {len(all_articles)} unique articles in {elapsed:.1f}s")
    return all_articles


if __name__ == "__main__":
    articles = collect_ticker_news()
    print(f"Total: {len(articles)}")
    for a in articles[:5]:
        print(f"  [{a['_ticker']}] {a['title'][:80]}")
