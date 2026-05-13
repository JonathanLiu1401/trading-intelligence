"""
Async web scraper — directly crawls 100+ financial news sites and article feeds.
Never stops. Feeds URLs into a shared queue.
"""
import asyncio
import hashlib
import re
import time
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=12)
MAX_CONCURRENT = 40
USER_AGENT = "Mozilla/5.0 (compatible; FinancialIntelBot/1.0)"

# Direct article list pages to scrape (HTML, not RSS)
SCRAPE_TARGETS = [
    # US Financial News
    "https://finance.yahoo.com/news/",
    "https://www.marketwatch.com/latest-news",
    "https://www.cnbc.com/finance/",
    "https://www.cnbc.com/technology/",
    "https://www.cnbc.com/world-markets/",
    "https://www.reuters.com/finance/",
    "https://www.reuters.com/technology/",
    "https://www.bloomberg.com/markets",
    "https://www.bloomberg.com/technology",
    "https://www.wsj.com/news/markets",
    "https://www.ft.com/markets",
    "https://www.barrons.com/market-data",
    "https://seekingalpha.com/market-news/all",
    "https://www.investors.com/news/technology/",
    "https://www.thestreet.com/markets",
    "https://www.benzinga.com/news/",
    "https://www.benzinga.com/trading-ideas/",
    "https://finviz.com/news.ashx",
    "https://stockanalysis.com/news/",
    "https://www.zacks.com/stock-news/",
    "https://www.fool.com/investing-news/",
    "https://www.schaeffersresearch.com/content/news",
    "https://www.barchart.com/news",
    "https://www.marketbeat.com/financial-news/",
    # Semis / Tech specific
    "https://www.tomshardware.com/news",
    "https://www.anandtech.com/",
    "https://www.techpowerup.com/news/",
    "https://www.eetimes.com/category/semiconductors/",
    "https://semianalysis.com/",
    "https://www.electronicsweekly.com/news/",
    "https://www.electronicdesign.com/markets/",
    "https://www.edn.com/news/",
    "https://spectrum.ieee.org/semiconductors",
    # Options / Derivatives
    "https://www.unusualwhales.com/news",
    "https://optionstockwatchlist.com/",
    # Crypto
    "https://www.coindesk.com/markets/",
    "https://cointelegraph.com/",
    "https://decrypt.co/news",
    "https://www.theblock.co/latest",
    "https://cryptonews.com/news/",
    # Asia
    "https://asia.nikkei.com/Business/Markets",
    "https://asia.nikkei.com/Business/Technology",
    "https://www.scmp.com/business/markets",
    "https://www.koreaherald.com/list.php?ct=020100000000",
    "https://en.yna.co.kr/economy",
    "https://www.japantimes.co.jp/business/",
    "https://www.chinadaily.com.cn/business/",
    "https://www.globaltimes.cn/business/",
    # Europe
    "https://www.euronews.com/business",
    "https://www.dw.com/en/economy-and-business/s-1440",
    "https://www.theguardian.com/business/economics",
    "https://www.independent.co.uk/topic/business",
    # Macro / Economics
    "https://www.calculatedriskblog.com/",
    "https://econbrowser.com/",
    "https://www.nakedcapitalism.com/",
    "https://wolfstreet.com/",
    "https://www.axios.com/markets",
    "https://www.project-syndicate.org/",
    # Supply chain
    "https://www.supplychaindive.com/news/",
    "https://www.logisticsmgmt.com/news/",
]

# Finviz real-time news scraper (no API needed)
FINVIZ_NEWS = "https://finviz.com/news.ashx"

# SEC EDGAR real-time filings (8-K = material events)
SEC_EDGAR = "https://efts.sec.gov/LATEST/search-index?q=%22memory%22+%22semiconductor%22&dateRange=custom&startdt={}&enddt={}&forms=8-K"

# Patterns to skip
SKIP_PATTERNS = re.compile(
    r"/(author|about|contact|privacy|terms|login|register|subscribe|advertis|careers|help)/",
    re.I
)


def _is_article_url(url: str) -> bool:
    if SKIP_PATTERNS.search(url):
        return False
    path = urlparse(url).path
    return len(path) > 10 and path.count("/") >= 2


def _extract_articles(html: str, base_url: str) -> list:
    """Extract article links and titles from a page."""
    try:
        soup = BeautifulSoup(html, "lxml")
        articles = []
        seen = set()
        for tag in soup.find_all(["a", "h1", "h2", "h3"]):
            href = tag.get("href") or ""
            text = tag.get_text(strip=True)
            if not text or len(text) < 15:
                continue
            url = urljoin(base_url, href) if href else ""
            if url and url not in seen and _is_article_url(url):
                seen.add(url)
                articles.append({
                    "title": text[:200],
                    "link": url,
                    "summary": "",
                    "source": f"scraped/{urlparse(base_url).netloc}",
                    "published": "",
                })
        return articles[:50]  # cap per page
    except Exception:
        return []


async def _fetch_page(session: aiohttp.ClientSession, url: str) -> list:
    try:
        async with session.get(url, timeout=REQUEST_TIMEOUT,
                               headers={"User-Agent": USER_AGENT},
                               ssl=False) as r:
            if r.status != 200:
                return []
            html = await r.text(errors="replace")
            return _extract_articles(html, url)
    except Exception:
        return []


async def scrape_all_async() -> list:
    """Scrape all targets concurrently."""
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_fetch_page(session, url) for url in SCRAPE_TARGETS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    articles = []
    seen_urls = set()
    for batch in results:
        if isinstance(batch, list):
            for art in batch:
                url = art["link"]
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    articles.append(art)
    return articles


def scrape_web() -> list:
    """Sync wrapper for async scraper."""
    t0 = time.time()
    articles = asyncio.run(scrape_all_async())
    print(f"[web_scraper] {len(articles)} articles from {len(SCRAPE_TARGETS)} sites in {time.time()-t0:.1f}s")
    return articles


if __name__ == "__main__":
    arts = scrape_web()
    print(f"Total: {len(arts)}")
    for a in arts[:5]:
        print(f"  [{a['source']}] {a['title'][:80]}")
