"""NASDAQ trading-halt collector — live regulatory / volatility halt feed.

Pulls the public Nasdaq Trader trade-halts RSS feed
(https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts), which needs **no auth
and no API key**. It carries every UTP trading halt and resumption across
NASDAQ / NYSE / regional venues: LULD volatility pauses, ``T1`` news-pending
halts, ``H10`` SEC trading suspensions, ``H11`` regulatory-concern halts,
market-wide circuit breakers, etc.

For an urgency-focused daemon this is one of the highest-signal sources there
is: a halt on a held ticker is a hard, exchange-issued market event that
almost always *precedes* the news article explaining it — none of the existing
collectors (RSS / GDELT / Finnhub / SEC 8-K / social) surface it. Each halt
becomes an article-like dict in the exact shape the other collectors emit, so
the daemon's ``_ingest()`` path scores/stores it unchanged.

Standalone usage / smoke test:

    python3 collectors/nasdaq_halts_collector.py

To wire into the daemon (intentionally NOT done here to avoid colliding with a
concurrent editor of daemon.py), register a worker mirroring the other
collectors::

    from collectors.nasdaq_halts_collector import collect_nasdaq_halts
    # in main(): _spawn("nasdaq_halts", collect_nasdaq_halts, interval=120)
"""
from __future__ import annotations

import html
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import requests

try:  # stdlib >=3.9; present on the daemon host (verified 3.12)
    from zoneinfo import ZoneInfo

    _ET_TZ = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - zoneinfo always available here
    _ET_TZ = None

FEED_URL = "https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts"
REQUEST_TIMEOUT = 8
MAX_ATTEMPTS = 2
RETRY_BACKOFF = 2.0  # seconds; doubled each retry
MAX_ARTICLES = 80

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0 Safari/537.36"
)

# Curated NASDAQ/UTP halt reason codes -> human text. Anything unmapped falls
# back to ``Trading halt (<code>)`` so a new code never breaks ingestion.
_REASON_CODES = {
    "T1": "News pending",
    "T2": "News released",
    "T3": "News and resumption times",
    "T5": "Single-stock trading pause in effect",
    "T6": "Extraordinary market activity",
    "T7": "Single-stock trading pause / quotation-only period",
    "T8": "Exchange-traded fund halt",
    "T12": "Additional information requested by exchange",
    "H4": "Non-compliance with listing requirements",
    "H9": "Filing delinquent",
    "H10": "SEC trading suspension",
    "H11": "Regulatory concern",
    "O1": "Operations halt — contact market operations",
    "IPO1": "IPO issue not yet trading",
    "IPOQ": "IPO issue quotation period",
    "M1": "Corporate action",
    "M2": "Quotation not available",
    "LUDP": "Volatility trading pause (LULD)",
    "LUDS": "Volatility trading pause — straddle condition",
    "MWC0": "Market-wide circuit breaker — carry-over from prior day",
    "MWC1": "Market-wide circuit breaker — Level 1",
    "MWC2": "Market-wide circuit breaker — Level 2",
    "MWC3": "Market-wide circuit breaker — Level 3",
    "MWCQ": "Market-wide circuit breaker — resumption",
    "R1": "New issue available",
    "R4": "Qualifications issues resolved; trading to resume",
    "R9": "Filing requirements satisfied; trading to resume",
}

# Pure listing-mechanics noise, not market events — dropped unless the caller
# asks for them. Everything else (every real halt/resumption) is kept.
_LOW_SIGNAL_CODES = {"IPO1", "IPOQ", "M2"}


def _fetch(url: str) -> bytes:
    """GET the feed with retry-and-backoff. Returns the raw XML body, or
    ``b""`` on persistent failure / non-200 (caller degrades to []),
    mirroring ``hackernews_collector._fetch``."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml"}
    backoff = RETRY_BACKOFF
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            return r.content or b""
        except Exception as e:  # network / non-200 — all non-fatal
            if attempt >= MAX_ATTEMPTS:
                print(f"[nasdaq_halts] fetch failed after {attempt} attempts: {e}")
                return b""
            print(f"[nasdaq_halts] attempt {attempt} failed ({e}); "
                  f"retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff *= 2
    return b""


def _local(tag: str) -> str:
    """ElementTree gives ``{namespace-uri}LocalName`` for namespaced tags
    (the NASDAQ feed puts every payload field in the ``ndaq:`` namespace).
    Strip the namespace and lower-case so lookups are URI-agnostic — the feed
    has historically shipped the namespace under more than one URI."""
    return tag.rsplit("}", 1)[-1].strip().lower()


def _parse_halts_xml(xml_bytes: bytes) -> list[dict]:
    """Parse the feed body into a list of raw per-item field dicts (keys are
    lower-cased local tag names). Pure — no network — so it is unit-testable
    against a static fixture. Returns [] on empty/malformed XML."""
    if not xml_bytes:
        return []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        print(f"[nasdaq_halts] XML parse error: {e}")
        return []
    items: list[dict] = []
    for item in root.iter():
        if _local(item.tag) != "item":
            continue
        fields: dict = {}
        for child in item:
            key = _local(child.tag)
            if key and key not in fields:
                fields[key] = (child.text or "").strip()
        if fields:
            items.append(fields)
    return items


def _published_iso(halt_date: str, halt_time: str, pubdate: str) -> str:
    """Best-effort tz-aware UTC ISO timestamp for the halt.

    Preferred source is the feed's own ``HaltDate`` (MM/DD/YYYY) + ``HaltTime``
    (HH:MM:SS), interpreted in US/Eastern (the exchange clock) and converted to
    UTC. Falls back to the RSS ``pubDate`` (RFC-822), then to *now* — never the
    deprecated naive ``utcnow()`` — so the stamp always round-trips through
    paper_trader's parser."""
    if halt_date and halt_time and _ET_TZ is not None:
        # The live feed ships HaltTime with fractional seconds
        # (``14:32:01.056``); the ``.%f`` form must be tried first.
        for fmt in ("%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S",
                    "%m/%d/%Y %H:%M"):
            try:
                dt = datetime.strptime(f"{halt_date} {halt_time}", fmt)
                return dt.replace(tzinfo=_ET_TZ).astimezone(timezone.utc).isoformat()
            except ValueError:
                continue
    if pubdate:
        try:
            dt = parsedate_to_datetime(pubdate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError):
            pass
    return datetime.now(timezone.utc).isoformat()


def _item_to_article(fields: dict, include_low_signal: bool) -> dict | None:
    """Convert one parsed feed item to the standard collector dict, or None to
    skip (missing symbol, or a low-signal listing-mechanics code)."""
    symbol = html.unescape(fields.get("issuesymbol", "")).strip().upper()
    if not symbol:
        return None

    code = fields.get("reasoncode", "").strip().upper()
    if code in _LOW_SIGNAL_CODES and not include_low_signal:
        return None
    reason = _REASON_CODES.get(code, f"Trading halt ({code})" if code else "Trading halt")

    company = html.unescape(fields.get("companyname", "")).strip()
    market = fields.get("market", "").strip()
    halt_date = fields.get("haltdate", "").strip()
    halt_time = fields.get("halttime", "").strip()
    res_date = fields.get("resumptiondate", "").strip()
    res_quote = fields.get("resumptionquotetime", "").strip()
    res_trade = fields.get("resumptiontradetime", "").strip()
    resumed = bool(res_trade or res_quote)

    verb = "HALT / RESUME" if resumed else "HALT"
    name = f" {company}" if company else ""
    title = f"{verb} — {symbol}{name}: {reason}"[:200]

    when = " ".join(p for p in (halt_date, halt_time) if p) + " ET"
    if resumed:
        parts = []
        if res_quote:
            parts.append(f"quote {res_quote} ET")
        if res_trade:
            parts.append(
                f"trade {res_date + ' ' if res_date else ''}{res_trade} ET")
        tail = ("Resumption: " + ", ".join(parts) + "."
                if parts else "Resumption scheduled.")
    else:
        tail = "Not yet resumed."
    venue = f" on {market}" if market else ""
    summary = (
        f"NASDAQ/UTP trading halt: {symbol}"
        f"{(' (' + company + ')') if company else ''}{venue}. "
        f"Reason {code or 'n/a'} — {reason}. Halted {when}. {tail}"
    )

    # Stable, clickable link; the per-event fragment keeps article_store's
    # sha256(url||title) id unique across repeated halts of the same symbol.
    stamp = "".join(ch for ch in f"{halt_date}{halt_time}" if ch.isdigit())
    if symbol.isalnum():
        link = f"https://www.nasdaq.com/market-activity/stocks/{symbol.lower()}"
        if stamp:
            link += f"#halt-{stamp}"
    else:
        link = "https://www.nasdaqtrader.com/Trader.aspx?id=TradeHalts"

    return {
        "title": title,
        "link": link,
        "summary": summary[:1000],
        "published": _published_iso(halt_date, halt_time, fields.get("pubdate", "")),
        "source": "nasdaq_halts",
        "_halt_symbol": symbol,
        "_halt_reason_code": code,
        "_halt_market": market,
        "_halt_resumed": resumed,
    }


def collect_nasdaq_halts(include_low_signal: bool = False) -> list:
    """Pull the live NASDAQ trade-halt feed as article-like dicts. Low-signal
    listing-mechanics codes (IPO-not-trading, quote-unavailable) are dropped
    unless ``include_low_signal=True``. Deduped by (symbol, reason, halt
    timestamp) and capped at ``MAX_ARTICLES``. Returns [] on any fetch/parse
    failure (never raises — the daemon worker loop expects a list)."""
    print("[nasdaq_halts] Fetching trade-halt feed...")
    t0 = time.time()

    items = _parse_halts_xml(_fetch(FEED_URL))
    articles: list = []
    seen: set = set()
    for fields in items:
        art = _item_to_article(fields, include_low_signal)
        if art is None:
            continue
        key = (art["_halt_symbol"], art["_halt_reason_code"], art["link"])
        if key in seen:
            continue
        seen.add(key)
        articles.append(art)
        if len(articles) >= MAX_ARTICLES:
            break

    elapsed = time.time() - t0
    print(f"[nasdaq_halts] Got {len(articles)} halt events in {elapsed:.1f}s")
    return articles


if __name__ == "__main__":
    items = collect_nasdaq_halts()
    print(f"Total: {len(items)}")
    for a in items[:8]:
        print(f"  [{a['source']}] {a['title']}")
        print(f"    {a['link']}")
        print(f"    {a['summary']}")
