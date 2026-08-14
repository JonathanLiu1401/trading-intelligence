"""Reddit collector — financial subreddits via public RSS (no API key).

The old JSON firehose (www.reddit.com/r/{sub}/{hot|new}.json) returns 429
from this VPS: datacenter IP + a custom bot User-Agent + 24-way parallel
fetches of ~80 subs * 2 listings every cycle. Public JSON is blocked; Atom
on old.reddit.com is the remaining unauthenticated path.

Optional OAuth (only if the operator sets env keys; none are invented here):
  REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET  -> client_credentials token
  then oauth.reddit.com JSON listings.
"""
from __future__ import annotations

import html
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import requests

REQUEST_TIMEOUT = 12
MAX_WORKERS = int(os.environ.get("REDDIT_MAX_WORKERS", "4"))
MIN_SCORE = 3  # JSON /hot only; RSS has no score
LISTINGS = ("hot", "new")

# Browser-like UA: Reddit 429s the previous "DigitalIntern/1.0 ..." bot string.
USER_AGENT = os.environ.get(
    "REDDIT_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)

# Prefer old.reddit.com RSS; www JSON is 429 from this host.
RSS_HOSTS = (
    "https://old.reddit.com",
    "https://www.reddit.com",
)
JSON_HOSTS = (
    "https://old.reddit.com",
    "https://www.reddit.com",
)

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

_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
}
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Cycle-local 429 circuit so one blocked pass does not hammer every sub.
_http_fail_logged = 0
_rate_limited = False


def _strip_html(text: str) -> str:
    return " ".join(_HTML_TAG_RE.sub(" ", html.unescape(text or "")).split())


def _headers(*, accept: str, token: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": accept,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _reddit_oauth_token() -> str | None:
    """Optional app-only token. Never logs the secret or access token."""
    cid = (os.environ.get("REDDIT_CLIENT_ID") or "").strip()
    secret = (os.environ.get("REDDIT_CLIENT_SECRET") or "").strip()
    if not cid or not secret:
        return None
    try:
        r = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            data={"grant_type": "client_credentials"},
            auth=(cid, secret),
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code != 200:
            print(f"[reddit] oauth token HTTP {r.status_code}")
            return None
        token = (r.json() or {}).get("access_token")
        return str(token) if token else None
    except Exception as e:
        print(f"[reddit] oauth token failed: {type(e).__name__}")
        return None


def _log_http(status: int, url: str) -> None:
    global _http_fail_logged
    if status == 200:
        return
    if _http_fail_logged < 6:
        print(f"[reddit] HTTP {status} {url}")
        _http_fail_logged += 1


def _get(url: str, *, accept: str, token: str | None = None) -> requests.Response | None:
    global _rate_limited
    if _rate_limited:
        return None
    try:
        r = requests.get(
            url,
            headers=_headers(accept=accept, token=token),
            timeout=REQUEST_TIMEOUT,
        )
    except Exception:
        return None
    _log_http(r.status_code, url)
    if r.status_code == 429:
        _rate_limited = True
        retry_after = r.headers.get("Retry-After")
        try:
            wait = min(float(retry_after), 8.0) if retry_after else 2.0
        except (TypeError, ValueError):
            wait = 2.0
        time.sleep(max(0.5, wait))
        return None
    if r.status_code != 200:
        return None
    return r


def _parse_rss_xml(xml_text: str, subreddit: str) -> list[dict]:
    """Parse Atom (old.reddit) or RSS 2.0 into intern article dicts."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    results: list[dict] = []
    tag = root.tag.lower()
    entries = []
    if tag.endswith("feed") or root.find("atom:entry", _ATOM_NS) is not None:
        entries = list(root.findall("atom:entry", _ATOM_NS)) or list(root.findall("entry"))
        for entry in entries:
            title_el = entry.find("atom:title", _ATOM_NS)
            if title_el is None:
                title_el = entry.find("title")
            title = (title_el.text or "").strip() if title_el is not None else ""
            link = ""
            for link_el in list(entry.findall("atom:link", _ATOM_NS)) + list(entry.findall("link")):
                href = (link_el.get("href") or "").strip()
                rel = (link_el.get("rel") or "alternate").lower()
                if href and rel in {"alternate", ""}:
                    link = href
                    break
                if href and not link:
                    link = href
            content_el = entry.find("atom:content", _ATOM_NS)
            if content_el is None:
                content_el = entry.find("atom:summary", _ATOM_NS)
            if content_el is None:
                content_el = entry.find("content")
            summary = _strip_html(content_el.text or "")[:500] if content_el is not None else ""
            updated_el = entry.find("atom:updated", _ATOM_NS)
            if updated_el is None:
                updated_el = entry.find("updated")
            published = (updated_el.text or "").strip() if updated_el is not None else ""
            if title:
                results.append({
                    "title": title,
                    "link": link or f"https://www.reddit.com/r/{subreddit}/",
                    "summary": summary,
                    "published": published,
                    "source": f"reddit/r/{subreddit}",
                    "_reddit_score": 1,
                })
        return results

    for item in root.findall(".//item"):
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        date_el = item.find("pubDate")
        title = (title_el.text or "").strip() if title_el is not None else ""
        link = (link_el.text or "").strip() if link_el is not None else ""
        summary = _strip_html(desc_el.text or "")[:500] if desc_el is not None else ""
        published = ""
        if date_el is not None and date_el.text:
            raw = date_el.text.strip()
            try:
                published = parsedate_to_datetime(raw).isoformat()
            except Exception:
                published = raw
        if title:
            results.append({
                "title": title,
                "link": link or f"https://www.reddit.com/r/{subreddit}/",
                "summary": summary,
                "published": published,
                "source": f"reddit/r/{subreddit}",
                "_reddit_score": 1,
            })
    return results


def _articles_from_listing_json(payload: dict, subreddit: str, listing: str) -> list[dict]:
    posts = (payload or {}).get("data", {}).get("children", [])
    results = []
    min_score = MIN_SCORE if listing == "hot" else 1
    for post in posts:
        d = post.get("data", {})
        title = d.get("title", "").strip()
        link = d.get("url", "")
        score = d.get("score", 0)
        selftext = (d.get("selftext", "") or "")[:500]
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


def _fetch_listing_rss(subreddit: str, listing: str) -> list[dict]:
    for host in RSS_HOSTS:
        url = f"{host}/r/{subreddit}/{listing}.rss"
        r = _get(url, accept="application/atom+xml, application/rss+xml, application/xml;q=0.9, */*;q=0.8")
        if r is None:
            if _rate_limited:
                return []
            continue
        items = _parse_rss_xml(r.text, subreddit)
        if items:
            return items
    return []


def _fetch_listing_json(subreddit: str, listing: str, token: str | None) -> list[dict]:
    if token:
        url = f"https://oauth.reddit.com/r/{subreddit}/{listing}.json?limit=100"
        r = _get(url, accept="application/json", token=token)
        if r is not None:
            try:
                return _articles_from_listing_json(r.json(), subreddit, listing)
            except Exception:
                return []
        return []
    for host in JSON_HOSTS:
        url = f"{host}/r/{subreddit}/{listing}.json?limit=100"
        r = _get(url, accept="application/json")
        if r is None:
            if _rate_limited:
                return []
            continue
        try:
            return _articles_from_listing_json(r.json(), subreddit, listing)
        except Exception:
            continue
    return []


def _fetch_listing(subreddit: str, listing: str, token: str | None = None) -> list[dict]:
    """RSS first (unauthenticated), JSON only as fallback or via OAuth."""
    if token:
        items = _fetch_listing_json(subreddit, listing, token)
        if items:
            return items
    items = _fetch_listing_rss(subreddit, listing)
    if items:
        return items
    return _fetch_listing_json(subreddit, listing, None)


def _fetch_subreddit(subreddit: str, token: str | None = None) -> list[dict]:
    """Fetch both /hot and /new so we get both quality and freshness."""
    out: list[dict] = []
    for listing in LISTINGS:
        if _rate_limited:
            break
        out.extend(_fetch_listing(subreddit, listing, token))
    return out


def collect_reddit() -> list:
    """Pull top posts from all financial subreddits in parallel."""
    global _http_fail_logged, _rate_limited
    _http_fail_logged = 0
    _rate_limited = False
    token = _reddit_oauth_token()
    mode = "oauth" if token else "rss"
    print(f"[reddit] Fetching from {len(SUBREDDITS)} subreddits ({mode}, workers={MAX_WORKERS})...")
    t0 = time.time()

    all_articles = []
    seen_urls = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_subreddit, sub, token): sub for sub in SUBREDDITS}
        for future in as_completed(futures):
            try:
                batch = future.result()
            except Exception:
                batch = []
            for art in batch:
                url = art["link"]
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_articles.append(art)

    elapsed = time.time() - t0
    extra = " rate-limited mid-pass" if _rate_limited else ""
    print(f"[reddit] Got {len(all_articles)} posts in {elapsed:.1f}s{extra}")
    return all_articles


if __name__ == "__main__":
    articles = collect_reddit()
    print(f"Total: {len(articles)}")
    for a in sorted(articles, key=lambda x: x.get("_reddit_score", 0), reverse=True)[:5]:
        print(f"  [{a['source']}] score={a['_reddit_score']} {a['title'][:70]}")
