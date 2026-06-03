"""
Async web scraper — directly crawls 100+ financial news sites and article feeds.
``scrape_web()`` fetches every target concurrently and returns a deduplicated
list of article dicts; the daemon's web_worker calls it on a fixed interval.
"""
import asyncio
import re
import time
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=12)
MAX_CONCURRENT = 180
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
    "https://www.reuters.com/markets/asia/",
    "https://www.reuters.com/markets/europe/",
    "https://www.reuters.com/markets/global-market-data/",
    "https://www.reuters.com/business/aerospace-defense/",
    "https://www.reuters.com/business/autos-transportation/",
    "https://www.reuters.com/business/energy/",
    "https://www.reuters.com/business/finance/",
    "https://www.reuters.com/business/healthcare-pharmaceuticals/",
    "https://www.reuters.com/business/media-telecom/",
    "https://www.reuters.com/business/retail-consumer/",
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
    "https://www.digitimes.com/",
    "https://www.digitimes.com/news/",
    "https://www.taipeitimes.com/News/biz",
    "https://focustaiwan.tw/business",
    "https://www.straitstimes.com/business",
    "https://www.channelnewsasia.com/business",
    "https://www.thestar.com.my/business",
    "https://www.bangkokpost.com/business",
    "https://www.theedgemalaysia.com/section/business",
    "https://www.businesstimes.com.sg/",
    "https://www.businesstimes.com.sg/companies-markets",
    "https://www.thehindubusinessline.com/markets/",
    "https://www.livemint.com/market",
    "https://economictimes.indiatimes.com/markets",
    "https://www.moneycontrol.com/news/business/markets/",
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
    "https://www.france24.com/en/business-tech/",
    "https://www.lemonde.fr/en/economy/",
    "https://www.politico.eu/section/economy/",
    "https://www.euractiv.com/sections/economy-jobs/",
    "https://www.thelocal.de/business",
    "https://www.theguardian.com/business/economics",
    "https://www.theguardian.com/business",
    "https://www.theguardian.com/business/stock-markets",
    "https://www.independent.co.uk/topic/business",
    # ── Middle East / Africa / LatAm ─────────────────────────────────────
    "https://www.aljazeera.com/economy/",
    "https://www.thenationalnews.com/business/",
    "https://www.arabnews.com/business-economy",
    "https://www.zawya.com/en/markets",
    "https://www.arabianbusiness.com/markets",
    "https://www.businesslive.co.za/bd/",
    "https://www.moneyweb.co.za/",
    "https://www.reuters.com/world/africa/",
    "https://www.reuters.com/world/middle-east/",
    "https://www.reuters.com/world/americas/",
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

# ── Quote-widget pseudo-article rejection ────────────────────────────────────
# Yahoo Finance / Bloomberg / Seeking Alpha list pages embed a live ticker-tape
# sidebar whose every entry is an <a href="/quote/NVDA"> wrapping the rendered
# quote string with no inter-field spaces, e.g.
# "NVDANVIDIA Corporation227.13-8.61(-3.65%)". The generic anchor scan in
# _extract_articles treated each of those as a fresh "article"; because the
# price changes on every poll the title (and therefore the article id) is
# unique every cycle, so a single widget manufactured an unbounded stream of
# fake breaking news. Live evidence (2026-05-18): 3,476 of 5,847 sampled
# scraped/* rows were these, the ML relevance head scored them up to 9.99, and
# one ("NVDANVIDIA Corporation227.13-8.61(-3.65%)") was Sonnet-scored 8.0 and
# fired a "🚨 BREAKING" Discord push — the consuming analyst's single biggest
# noise complaint. Two independent fingerprints (either is sufficient):
#   1. a letter glued directly to a multi-digit decimal price — real prose
#      always has a space ("rises 22% to $35.1 billion" / "5,123.41 record
#      high" do NOT match; "Corporation227.13" / "USD2,169.83" do);
#   2. a parenthesised signed percent change "(-3.65%)" — the quote-tape
#      change column; real headlines write percentages without the parens.
# A defense-in-depth twin lives in watchers.alert_agent._looks_like_quote_widget
# (kept duplicated rather than cross-imported — collectors must not pull the
# watchers/ml import graph, same rationale as article_store._briefing_domain_key).
_QW_PRICE_GLUE = re.compile(r"[A-Za-z]\$?\d{1,4}[.,]\d{2,3}")
_QW_PCT_PAREN = re.compile(r"\([+-]?\d{1,3}(?:\.\d+)?%\)")
# A Yahoo quote *landing* page ("/quote/NVDA" or "/quote/NVDA/") — the widget's
# own href. Anchored to end-of-path so real quote-scoped articles
# ("/quote/NVDA/news/some-headline-123") are NOT caught.
_QW_QUOTE_PATH = re.compile(r"/quote/[^/]+/?$", re.I)
# Quote-aggregator *share-card / listing-page* pseudo-article. Google News (and
# other aggregators) index the Moomoo/Futu/Webull "share this quote" landing
# pages whose <title> is the rendered share card: "$NVIDIA (NVDA.US)$ - Moomoo"
# / "$Tencent (00700.HK)$ - Futu". These are NOT articles — they are a live
# quote page, the same pseudo-article class as the ticker-tape widget above
# (same noise complaint), but with a distinct surface the two fingerprints
# above don't catch (no glued price, no parenthesised % change, no Yahoo
# /quote/ URL). Live evidence (2026-05-18 + recurring across ≥6 prior passes):
# "$NVIDIA (NVDA.US)$ - Moomoo" arrived via the `GN: Nvidia` collector,
# ML-relevance-scored 9.77, and fired a 🚨 BREAKING urgent alert (urgency=2).
# The fingerprint is the leading "$" share-card lead glued to a
# "(SYMBOL.EXCH)$" close — real prose never ends a parenthetical with
# ".EXCH)$", and a real "$TICKER ..." headline ("$NVDA breaks out (NYSE)",
# "$MU upgraded to Buy (price target $150.00)") has no such close. Bounded
# ({0,60}) so there is no catastrophic backtracking. Validated zero false
# positives against the live $+paren headline corpus
# (e.g. "Zscaler (NASDAQ:ZS) Price Target Cut to $223.00").
_QW_LISTING = re.compile(
    r"^\s*\$[^$\n]{0,60}\([A-Za-z0-9.\-]{1,8}\.[A-Za-z]{1,4}\)\$"
)
# Image-credit pseudo-article — the news-page hero image is wrapped in the
# article's own <a href="..."> link, so when this scraper falls back to anchor
# text it picks up the photo credit line as the "title". Live evidence
# (2026-05-21 16:30:49Z, alert_recency.db): "Angela Weiss/AFP/Getty Images"
# fired a real 🚨 BREAKING Discord push from ``scraped/www.bloomberg.com``
# (cred=0.90 — well above the 0.45 lone-source bar; content type IS the
# failure). The ML urgency head scored it 10.0 because the bloomberg.com URL +
# proper-noun tokens + the implied semis-quantum subject (the hero image was
# beneath a "Trump pledges $2B for quantum firms" article) triggered the
# urgency head's high-relevance pattern recognition. Other live samples:
# "Tomohiro Ohsumi/Getty Images", "Timorthy A. Clary/AFP/Getty Images".
#
# Fingerprint: anchored ^...$ so the WHOLE title is the credit (never a
# mid-headline use). Title-Case photographer name (≥2 tokens, allowing
# initials like ``A.``), then one or more ``/Agency`` slugs with NO space
# around the slash, ending in a recognised image agency from a closed list.
# Validated zero false positives against the must-survive corpus including
# "Reuters/Yahoo Finance reports", "Sam Altman/OpenAI says", "MU drops 5%/Yahoo".
# Defense-in-depth: a byte-identical twin lives in
# ``watchers.alert_agent._QW_IMAGE_CREDIT`` and
# ``analysis.claude_analyst._QW_IMAGE_CREDIT`` — the documented lockstep
# triple-gate (collectors must not pull the watchers/ml import graph).
_QW_IMAGE_CREDIT = re.compile(
    # Name-token alternation: a hyphenated form (I-Hwa, O-Lin, Jean-Pierre)
    # OR a plain Capitalized run (Tomohiro, Cheng). The hyphenated branch
    # exists because Asian / French given-name conventions place the hyphen
    # *inside* the first token, and the prior `[A-Z][a-zA-Z]+` requirement
    # silently missed every such name — live evidence (2026-05-23 urgency=2
    # set): "I-Hwa Cheng/Bloomberg" reached alerted state un-suppressed
    # because the name-token regex demanded ≥1 letter immediately after the
    # capital and hit `-` first. The hyphen branch is anchored on a
    # second uppercase letter so a stray "I-foo" prose token cannot match.
    r"^\s*(?:[A-Z][a-zA-Z]*(?:-[A-Z][a-zA-Z]+)+|[A-Z][a-zA-Z]+)"
    r"(?:\s+(?:[A-Z]\.?|(?:[A-Z][a-zA-Z]*(?:-[A-Z][a-zA-Z]+)+|[A-Z][a-zA-Z]+)))+"
    r"(?:/(?:AFP|Reuters|Getty\s+Images|AP|Bloomberg|EPA|TASS|"
    r"WireImage|Shutterstock|Polaris|Bloomberg\s+News))+"
    r"\s*$"
)


def _looks_like_quote_widget(title: str, url: str) -> bool:
    """True for live quote-tape / quote-listing / image-credit entries
    masquerading as articles. See the block comments above for the live
    evidence and the four title fingerprints (price-glue, parenthesised %,
    share-card listing, image credit)."""
    t = title or ""
    if (_QW_PRICE_GLUE.search(t) or _QW_PCT_PAREN.search(t)
            or _QW_LISTING.search(t) or _QW_IMAGE_CREDIT.search(t)):
        return True
    try:
        if _QW_QUOTE_PATH.search(urlparse(url or "").path):
            return True
    except Exception:
        pass
    return False


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
        for tag in soup.find_all("a"):
            href = tag.get("href") or ""
            if not href:
                continue
            text = tag.get_text(strip=True)
            if not text or len(text) < 15:
                continue
            url = urljoin(base_url, href)
            if (url and url not in seen and _is_article_url(url)
                    and not _looks_like_quote_widget(text, url)):
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
                url = art.get("link") or ""
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
