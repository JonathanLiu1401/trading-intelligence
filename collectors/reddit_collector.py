"""Reddit collector — pulls top posts from financial subreddits via RSS (no API key needed)."""
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REQUEST_TIMEOUT = 10
MAX_WORKERS = 10

# Financial subreddits — RSS feeds require no API key
SUBREDDITS = [
    "stocks", "investing", "wallstreetbets", "SecurityAnalysis",
    "StockMarket", "options", "algotrading", "Economics",
    "finance", "personalfinance", "investing", "ValueInvesting",
    "dividends", "ETFs", "CryptoCurrency", "Bitcoin", "ethereum",
    "MacroEconomics", "GlobalMarkets", "emgmarket",
    "technology", "hardware", "Semiconductors", "AIstocks",
    "Micron", "nvidia", "AMD_Stock",
]


def _fetch_subreddit(subreddit: str) -> list:
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=100"
    headers = {"User-Agent": "DigitalIntern/1.0 financial-news-aggregator"}
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        posts = r.json().get("data", {}).get("children", [])
        results = []
        for post in posts:
            d = post.get("data", {})
            title = d.get("title", "").strip()
            link = d.get("url", "")
            permalink = f"https://reddit.com{d.get('selftext', '')}"
            score = d.get("score", 0)
            selftext = d.get("selftext", "")[:500]
            if title and score > 10:  # filter low-engagement posts
                results.append({
                    "title": title,
                    "link": link or f"https://reddit.com{d.get('permalink', '')}",
                    "summary": selftext,
                    "published": str(d.get("created_utc", "")),
                    "source": f"reddit/r/{subreddit}",
                    "_reddit_score": score,
                })
        return results
    except Exception:
        return []


def collect_reddit() -> list:
    """Pull top posts from all financial subreddits in parallel."""
    print(f"[reddit] Fetching from {len(SUBREDDITS)} subreddits...")
    t0 = time.time()

    all_articles = []
    seen_urls = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_subreddit, sub): sub for sub in SUBREDDITS}
        for future in as_completed(futures):
            for art in future.result():
                url = art["link"]
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_articles.append(art)

    elapsed = time.time() - t0
    print(f"[reddit] Got {len(all_articles)} posts in {elapsed:.1f}s")
    return all_articles


if __name__ == "__main__":
    articles = collect_reddit()
    print(f"Total: {len(articles)}")
    for a in sorted(articles, key=lambda x: x.get("_reddit_score", 0), reverse=True)[:5]:
        print(f"  [{a['source']}] score={a['_reddit_score']} {a['title'][:70]}")
