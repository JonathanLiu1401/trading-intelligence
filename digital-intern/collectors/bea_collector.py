"""BEA (Bureau of Economic Analysis) economic data collector.

Fetches the BEA RSS feed (https://apps.bea.gov/rss/rss.xml) which contains
structured XML data for key US macro releases: GDP, Personal Income, Trade
Balance, PCE inflation, etc. Each release becomes one article-like record
with current and previous values so the downstream scorer can react to
economic surprises.

The BEA RSS uses custom <data> elements with <current>/<previous> sub-nodes
that include percentChange values — significantly richer than a plain RSS feed.

Dedup is handled by the daemon's ArticleStore / _ingest path; this collector
only fetches and parses. BEA releases monthly, so an hourly interval is fine.

Standalone usage:
    python3 collectors/bea_collector.py
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
BEA_RSS_URL = "https://apps.bea.gov/rss/rss.xml"
REQUEST_TIMEOUT = 12
SOURCE = "bea"
USER_AGENT = (
    "DigitalInternBot/1.0 (research; contact: sealai215j@gmail.com) "
    "python-requests"
)

# Map BEA release title keywords → relevant market tickers for scorer
_MACRO_KEYWORDS: dict[str, list[str]] = {
    "GDP":               ["SPY", "QQQ", "DIA"],
    "Personal Income":   ["XLY", "XLP", "SPY"],
    "Trade":             ["SPY", "FXI", "EWZ"],
    "International Trade": ["SPY", "FXI", "EWZ"],
    "Corporate Profits": ["SPY", "QQQ"],
    "Housing":           ["XHB", "ITB"],
    "Direct Investment": ["SPY"],
}


def _keywords_for(title: str) -> list[str]:
    for key, tickers in _MACRO_KEYWORDS.items():
        if key.lower() in title.lower():
            return tickers
    return ["SPY"]


def _parse_value(node: ET.Element | None, tag: str) -> str | None:
    if node is None:
        return None
    el = node.find(tag)
    return el.text.strip() if el is not None and el.text else None


def _build_article(item: ET.Element) -> dict | None:
    title_el = item.find("title")
    link_el = item.find("link")
    desc_el = item.find("description")
    pub_el = item.find("pubDate")

    if title_el is None or link_el is None:
        return None

    title = (title_el.text or "").strip()
    link = (link_el.text or "").strip()
    if not title or not link:
        return None

    desc = (desc_el.text or "").strip() if desc_el is not None else ""
    pub_str = (pub_el.text or "").strip() if pub_el is not None else ""

    # Parse structured data block
    data_node = item.find("data")
    current_node = data_node.find("main/current") if data_node is not None else None
    previous_node = data_node.find("main/previous") if data_node is not None else None

    current_val: float | None = None
    previous_val: float | None = None
    info_date: str | None = None
    change_str = ""

    if current_node is not None:
        raw_curr = _parse_value(current_node, "percentChange")
        info_date = _parse_value(current_node, "infoDate")
        if raw_curr is not None:
            try:
                current_val = float(raw_curr)
            except ValueError:
                pass

    if previous_node is not None:
        raw_prev = _parse_value(previous_node, "percentChange")
        if raw_prev is not None:
            try:
                previous_val = float(raw_prev)
            except ValueError:
                pass

    if current_val is not None and previous_val is not None:
        delta = current_val - previous_val
        sign = "+" if delta >= 0 else ""
        change_str = f" | prev: {previous_val:g} | Δ {sign}{delta:.1f}"

    # Parse published date
    published = datetime.now(timezone.utc).isoformat()
    if pub_str:
        for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
            try:
                dt = datetime.strptime(pub_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                published = dt.isoformat()
                break
            except ValueError:
                continue

    val_label = f"{current_val:g}" if current_val is not None else ""
    enhanced_title = (
        f"BEA: {title} → {val_label}{change_str}"
        if current_val is not None
        else f"BEA: {title}"
    )

    summary_parts: list[str] = []
    if desc:
        summary_parts.append(desc[:400])
    if info_date:
        summary_parts.append(f"Period: {info_date}.")
    if current_val is not None:
        summary_parts.append(f"Current: {current_val:g}")
    if previous_val is not None:
        summary_parts.append(f"Previous: {previous_val:g}")
    summary_parts.append("Source: U.S. Bureau of Economic Analysis.")

    return {
        "title": enhanced_title[:250],
        "link": link,
        "summary": " ".join(summary_parts)[:800],
        "published": published,
        "source": SOURCE,
        "_tickers": _keywords_for(title),
        "_bea_current": current_val,
        "_bea_previous": previous_val,
        "_bea_info_date": info_date,
    }


def collect_bea() -> list[dict]:
    """Fetch BEA RSS and return macro release articles (dedup handled by caller)."""
    print("[bea] Fetching BEA economic releases from apps.bea.gov...")
    t0 = time.time()

    try:
        resp = requests.get(
            BEA_RSS_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[bea] fetch failed: {e}")
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        print(f"[bea] XML parse error: {e}")
        return []

    channel = root.find("channel")
    if channel is None:
        print("[bea] no <channel> in RSS")
        return []

    items = channel.findall("item")
    articles: list[dict] = []
    for item in items:
        article = _build_article(item)
        if article is not None:
            articles.append(article)

    elapsed = time.time() - t0
    print(f"[bea] fetched {len(articles)} releases in {elapsed:.1f}s")
    return articles


if __name__ == "__main__":
    results = collect_bea()
    print(f"\n=== BEA COLLECTOR: {len(results)} articles fetched ===")
    for a in results[:15]:
        print(f"  [{a.get('_bea_info_date', '?')}] {a['title']}")
        if a.get("_bea_current") is not None:
            print(f"    current={a['_bea_current']}  prev={a['_bea_previous']}")
        print(f"    {a['link']}")
        print()
