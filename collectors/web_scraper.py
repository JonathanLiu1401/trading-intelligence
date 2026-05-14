"""
Async web scraper — directly crawls 100+ financial news sites and article feeds.
Never stops. Feeds URLs into a shared queue.
"""
import asyncio
import re
import time
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=12)
MAX_CONCURRENT = 80
USER_AGENT = "Mozilla/5.0 (compatible; FinancialIntelBot/1.0)"

# Direct article list pages to scrape (HTML, not RSS)
SCRAPE_TARGETS = [
    # ── US Financial News ──────────────────────────────────────────────────
    "https://finance.yahoo.com/news/",
    "https://finance.yahoo.com/topic/stock-market-news/",
    "https://finance.yahoo.com/topic/economic-news/",
    "https://finance.yahoo.com/topic/earnings/",
    "https://www.marketwatch.com/latest-news",
    "https://www.marketwatch.com/markets",
    "https://www.marketwatch.com/economy-politics",
    "https://www.cnbc.com/finance/",
    "https://www.cnbc.com/technology/",
    "https://www.cnbc.com/world-markets/",
    "https://www.cnbc.com/markets/",
    "https://www.cnbc.com/economy/",
    "https://www.cnbc.com/earnings/",
    "https://www.reuters.com/finance/",
    "https://www.reuters.com/technology/",
    "https://www.reuters.com/markets/",
    "https://www.reuters.com/business/",
    "https://www.bloomberg.com/markets",
    "https://www.bloomberg.com/technology",
    "https://www.bloomberg.com/economics",
    "https://www.wsj.com/news/markets",
    "https://www.wsj.com/news/technology",
    "https://www.wsj.com/news/business",
    "https://www.ft.com/markets",
    "https://www.ft.com/technology",
    "https://www.ft.com/companies",
    "https://www.ft.com/global-economy",
    "https://www.barrons.com/market-data",
    "https://www.barrons.com/topics/technology",
    "https://seekingalpha.com/market-news/all",
    "https://seekingalpha.com/earnings/earnings-news",
    "https://seekingalpha.com/dividends/news",
    "https://www.investors.com/news/technology/",
    "https://www.investors.com/news/",
    "https://www.investors.com/research/ibd-50/",
    "https://www.thestreet.com/markets",
    "https://www.thestreet.com/investing",
    "https://www.thestreet.com/markets/stocks",
    "https://www.benzinga.com/news/",
    "https://www.benzinga.com/trading-ideas/",
    "https://www.benzinga.com/movers/",
    "https://www.benzinga.com/analyst-ratings/",
    "https://www.benzinga.com/premarket/",
    "https://www.benzinga.com/markets/",
    "https://finviz.com/news.ashx",
    "https://stockanalysis.com/news/",
    "https://www.zacks.com/stock-news/",
    "https://www.fool.com/investing-news/",
    "https://www.fool.com/earnings/",
    "https://www.schaeffersresearch.com/content/news",
    "https://www.barchart.com/news",
    "https://www.marketbeat.com/financial-news/",
    "https://www.streetinsider.com/",
    "https://www.smarteranalyst.com/",
    "https://www.tipranks.com/news",
    # ── Financial blogs / macro ────────────────────────────────────────────
    "https://www.zerohedge.com/",
    "https://www.zerohedge.com/markets",
    "https://www.zerohedge.com/economics",
    "https://www.calculatedriskblog.com/",
    "https://www.pragcap.com/blog/",
    "https://wolfstreet.com/",
    "https://www.nakedcapitalism.com/",
    "https://thehustle.co/news/",
    "https://thehustle.co/category/business/",
    "https://marginalrevolution.com/",
    "https://mishtalk.com/",
    "https://ritholtz.com/",
    "https://www.bespokepremium.com/",
    "https://econbrowser.com/",
    "https://www.calculatedriskblog.com/p/economic-news.html",
    "https://stratechery.com/",
    "https://www.axios.com/markets",
    "https://www.axios.com/business",
    "https://www.axios.com/economy",
    "https://www.project-syndicate.org/",
    "https://www.epi.org/blog/",
    # ── Semis / Hardware deep ──────────────────────────────────────────────
    "https://www.tomshardware.com/news",
    "https://www.tomshardware.com/tag/semiconductors",
    "https://www.tomshardware.com/tag/gpu",
    "https://www.anandtech.com/",
    "https://www.anandtech.com/tag/cpus",
    "https://www.anandtech.com/tag/gpus",
    "https://www.techpowerup.com/news/",
    "https://www.eetimes.com/category/semiconductors/",
    "https://www.eetimes.com/category/manufacturing/",
    "https://semianalysis.com/",
    "https://semiengineering.com/category/news/",
    "https://semiwiki.com/",
    "https://chipsandcheese.com/",
    "https://www.electronicsweekly.com/news/",
    "https://www.electronicdesign.com/markets/",
    "https://www.edn.com/news/",
    "https://spectrum.ieee.org/semiconductors",
    "https://wccftech.com/category/news/",
    "https://videocardz.com/",
    "https://www.phoronix.com/",
    "https://www.hardwareluxx.com/",
    "https://www.theregister.com/Tag/Hardware/",
    # ── Company IR pages — portfolio + watchlist ──────────────────────────
    "https://investor.lumentum.com/news-events/press-releases",
    "https://investors.micron.com/news-releases",
    "https://nvidianews.nvidia.com/news",
    "https://ir.amd.com/news-events/press-releases",
    "https://www.oracle.com/news/",
    "https://news.microsoft.com/category/press-releases/",
    "https://newsroom.intel.com/news-releases/",
    "https://ir.appliedmaterials.com/news-events/press-releases",
    "https://investor.lamresearch.com/news-events/press-releases",
    "https://ir.kla.com/news-events/press-releases",
    "https://www.asml.com/en/news/press-releases",
    "https://pr.tsmc.com/english/news",
    "https://www.qualcomm.com/news/releases",
    "https://news.samsung.com/global/",
    # ── Options / Derivatives ─────────────────────────────────────────────
    "https://www.unusualwhales.com/news",
    "https://optionstockwatchlist.com/",
    "https://www.cboe.com/insights/",
    # ── Crypto ────────────────────────────────────────────────────────────
    "https://www.coindesk.com/markets/",
    "https://www.coindesk.com/business/",
    "https://www.coindesk.com/policy/",
    "https://cointelegraph.com/",
    "https://cointelegraph.com/category/markets",
    "https://cointelegraph.com/tags/bitcoin",
    "https://decrypt.co/news",
    "https://decrypt.co/markets",
    "https://www.theblock.co/latest",
    "https://www.theblock.co/learn/categories/markets",
    "https://cryptonews.com/news/",
    "https://bitcoinmagazine.com/articles",
    "https://cryptoslate.com/news/",
    # ── Asia ──────────────────────────────────────────────────────────────
    "https://asia.nikkei.com/Business/Markets",
    "https://asia.nikkei.com/Business/Technology",
    "https://asia.nikkei.com/Business/Semiconductors",
    "https://asia.nikkei.com/Economy",
    "https://www.scmp.com/business/markets",
    "https://www.scmp.com/tech",
    "https://www.scmp.com/business",
    "https://www.koreaherald.com/list.php?ct=020100000000",
    "https://www.koreaherald.com/list.php?ct=020200000000",
    "https://en.yna.co.kr/economy",
    "https://en.yna.co.kr/business",
    "https://www.japantimes.co.jp/business/",
    "https://www.chinadaily.com.cn/business/",
    "https://www.globaltimes.cn/business/",
    # ── Europe ────────────────────────────────────────────────────────────
    "https://www.euronews.com/business",
    "https://www.dw.com/en/economy-and-business/s-1440",
    "https://www.theguardian.com/business/economics",
    "https://www.theguardian.com/business",
    "https://www.theguardian.com/business/stock-markets",
    "https://www.independent.co.uk/topic/business",
    # ── Supply chain ─────────────────────────────────────────────────────
    "https://www.supplychaindive.com/news/",
    "https://www.logisticsmgmt.com/news/",
    "https://www.freightwaves.com/news",
    # ── Macro / fintech blogs (extended) ─────────────────────────────────
    "https://abnormalreturns.com/",
    "https://thereformedbroker.com/",
    "https://www.thekobeissiletter.com/",
    "https://noahpinion.blog/",
    "https://www.platformer.news/",
    "https://www.bigtechnology.com/",
    "https://www.interconnects.ai/",
    "https://thedailyupside.com/",
    "https://www.netinterest.co/",
    "https://doomberg.substack.com/",
    "https://themacrocompass.substack.com/",
    "https://epsilontheory.substack.com/",
    "https://garymarcus.substack.com/",
    # ── More semis ───────────────────────────────────────────────────────
    "https://www.tomshardware.com/tag/cpus",
    "https://www.anandtech.com/tag/smartphones",
    "https://www.eetimes.com/category/data-center/",
    "https://semiengineering.com/category/manufacturing-news/",
]

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


def _nearby_snippet(tag) -> str:
    """Look for paragraph/description text near an anchor tag (siblings or parent)."""
    try:
        # Check the anchor's title attribute first
        title_attr = tag.get("title") or ""
        if title_attr and len(title_attr) > 30:
            return title_attr[:500]
        # Walk up to find a container, then look for p/desc text
        node = tag
        for _ in range(3):
            node = node.parent
            if node is None:
                break
            for child in node.find_all(["p", "span", "div"], limit=4):
                t = child.get_text(" ", strip=True)
                if t and 40 <= len(t) <= 600 and t != tag.get_text(strip=True):
                    return t[:500]
    except Exception:
        pass
    return ""


def _extract_articles(html: str, base_url: str) -> list:
    """Extract article links, titles, and nearby snippet text from a page."""
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
                    "summary": _nearby_snippet(tag),
                    "source": f"scraped/{urlparse(base_url).netloc}",
                    "published": "",
                })
        return articles[:100]  # cap per page
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
