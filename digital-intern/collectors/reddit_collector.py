"""Reddit collector — pulls top posts from financial subreddits via RSS (no API key needed)."""
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REQUEST_TIMEOUT = 10
MAX_WORKERS = 24
MIN_SCORE = 3  # very loose — capture more discussion

# Financial / tech subreddits — JSON API, no key needed
SUBREDDITS = [
    # Core investing
    "stocks", "investing", "wallstreetbets", "SecurityAnalysis",
    "StockMarket", "options", "algotrading", "Economics",
    "finance", "personalfinance", "ValueInvesting",
    "dividends", "ETFs", "financialindependence",
    "Bogleheads", "FinancialPlanning", "FIREyFemmes",
    "thetagang", "Vitards", "Daytrading", "swingtrading",
    "pennystocks", "smallstreetbets", "Forex",
    # Macro / geopolitical
    "MacroEconomics", "GlobalMarkets", "emgmarket",
    "geopolitics", "EconomicHistory", "AusEcon",
    # Crypto
    "CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets",
    "BitcoinMarkets", "ethfinance", "solana",
    # Tech / semis / AI
    "technology", "hardware", "Semiconductors", "Semiconductor",
    "AIstocks", "AIComputing", "MachineLearning", "LocalLLaMA",
    "gpu", "buildapc", "intel", "nvidia", "AMD_Stock", "amd",
    "ASML", "MicronTechnology", "Micron", "TSMC", "AMD",
    "singularity", "ArtificialInteligence", "OpenAI", "ChatGPT",
    "GPT3", "Robotics", "datacenter", "selfhosted",
    # Quant / data
    "quant", "datascience", "DeFi",
    # Trading / analysis
    "TheRaceTo10Million", "Wallstreetsilver", "Superstonk",
    "GME", "AMCSTOCK", "EVStocks", "Biotechplays",
    "shortsqueeze", "RobinhoodPennystocks", "WallStreetbetsELITE",
    "stockstobuytoday", "trakstocks", "RealDayTrading",
    # Industry / professional
    "AskEconomics", "BusinessNews", "stockmarketnews",
    "EconomicSignals", "tradingview", "TradingEducation",
    "FuturesTrading", "Bonds", "GoldandSilverStackers",
    # Global / regional equity discussion
    "EuropeanStocks", "ChinaStocks", "IndiaInvestments", "Commodities",
    # Sector / income / FIRE
    "energyinvestors", "REITs", "fatFIRE", "HFEA", "mutualfunds",
    "dividendgang", "passive_income", "AsianStocks",
    # Hardware / chip engineering
    "chipdesign", "FPGA", "ASIC_Design", "ElectricalEngineering",
    # Macro / business news
    "economy", "business", "Layoffs", "geoeconomics",
    # Trading discussion (extra)
    "WallStreetbetsCrypto", "wallstreetbetsHUZZAH", "investing_discussion",
    "stockanalysis", "EuropeFIRE",
]


def _fetch_listing(subreddit: str, listing: str) -> list:
    url = f"https://www.reddit.com/r/{subreddit}/{listing}.json?limit=100"
    headers = {"User-Agent": "DigitalIntern/1.0 financial-news-aggregator"}
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        posts = r.json().get("data", {}).get("children", [])
        results = []
        # /new has no score gate — most "new" posts have score 1 or 0 by design.
        min_score = MIN_SCORE if listing == "hot" else 1
        for post in posts:
            d = post.get("data", {})
            title = d.get("title", "").strip()
            link = d.get("url", "")
            score = d.get("score", 0)
            selftext = d.get("selftext", "")[:500]
            if title and score >= min_score:
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


def _fetch_subreddit(subreddit: str) -> list:
    """Fetch both /hot and /new so we get both quality and freshness."""
    out = _fetch_listing(subreddit, "hot")
    out.extend(_fetch_listing(subreddit, "new"))
    return out


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
