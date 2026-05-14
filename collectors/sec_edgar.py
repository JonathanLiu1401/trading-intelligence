"""SEC EDGAR collector — scrapes 8-K filings RSS for portfolio + watchlist tickers.

EDGAR rejects requests without an identifying User-Agent header per
https://www.sec.gov/os/accessing-edgar-data — we set one explicitly via
feedparser's `agent=` kwarg. Falls back gracefully on any error.

In addition to the recent 8-K RSS feed, we also query the EDGAR full-text
search API (efts.sec.gov) per portfolio ticker to catch S-1, 10-Q, DEF 14A,
SC 13G etc. filings that the 8-K stream misses.
"""
import hashlib
import json
import sqlite3
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import feedparser
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

EDGAR_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&dateb=&owner=include&count=40&output=atom"
)
# SEC requires a real contact in User-Agent; pull from env, fall back to a generic one.
EDGAR_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "Digital-Intern-Daemon contact@digital-intern.local",
)


def _load_relevant_tickers() -> set[str]:
    """Active positions + sector watchlist from portfolio.json (upper-cased)."""
    try:
        with open(PORTFOLIO_PATH, "r") as f:
            data = json.load(f)
    except Exception:
        return set()
    tickers: set[str] = set()
    for pos in data.get("positions", []):
        t = pos.get("ticker") or ""
        if t:
            tickers.add(t.upper())
    for opt in data.get("options", []):
        u = opt.get("underlying") or ""
        if u:
            tickers.add(u.upper())
    for t in data.get("sector_watchlist", []):
        if t:
            tickers.add(t.upper())
    return tickers


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY, link TEXT, title TEXT,
            source TEXT, first_seen TEXT
        )"""
    )
    conn.commit()
    return conn


def _article_id(link: str, title: str) -> str:
    return hashlib.sha256(f"{link}||{title}".encode("utf-8")).hexdigest()


def _is_seen(conn, aid: str) -> bool:
    return conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone() is not None


def _mark_seen(conn, aid: str, link: str, title: str, source: str):
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
        "VALUES (?, ?, ?, ?, ?)",
        (aid, link, title, source, datetime.utcnow().isoformat()),
    )


def collect_sec_edgar() -> list:
    """Scrape the EDGAR 8-K RSS feed, return only filings matching tracked tickers."""
    tickers = _load_relevant_tickers()
    if not tickers:
        return []

    parsed = feedparser.parse(EDGAR_URL, agent=EDGAR_USER_AGENT)
    if getattr(parsed, "bozo", 0) and not parsed.entries:
        return []

    conn = _ensure_db()
    new_articles: list = []

    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        summary = entry.get("summary") or entry.get("description") or ""
        if not title or not link:
            continue

        haystack = f"{title} {summary}".upper()
        matched = None
        for tkr in tickers:
            if len(tkr) <= 2:
                # Use simple boundary check for very short tickers (e.g. "MU")
                if f"({tkr})" in haystack or f" {tkr} " in haystack:
                    matched = tkr
                    break
            else:
                if tkr in haystack:
                    matched = tkr
                    break
        if not matched:
            continue

        aid = _article_id(link, title)
        if _is_seen(conn, aid):
            continue

        published = entry.get("updated") or entry.get("published") or ""
        new_articles.append({
            "title": f"[8-K {matched}] {title}",
            "link": link,
            "summary": summary,
            "published": published,
            "source": "SEC EDGAR",
            "_ticker": matched,
        })
        _mark_seen(conn, aid, link, title, "SEC EDGAR")

    conn.commit()
    conn.close()
    return new_articles


# ── Full-text search ────────────────────────────────────────────────────────
EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
EFTS_FORMS = "8-K,10-Q,10-K,S-1,DEF 14A,SC 13G,SC 13D,424B"
EFTS_DAYS_BACK = 14  # how far back to scan each pass


def _efts_search(ticker: str, since: str, until: str) -> list:
    """Hit EDGAR full-text JSON API for a ticker over the date window."""
    params = {
        "q": f'"{ticker}"',
        "dateRange": "custom",
        "startdt": since,
        "enddt": until,
        "forms": EFTS_FORMS,
    }
    headers = {"User-Agent": EDGAR_USER_AGENT, "Accept": "application/json"}
    try:
        r = requests.get(EFTS_URL, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            return []
        hits = r.json().get("hits", {}).get("hits", [])
    except Exception:
        return []

    # Keep only filings where the *filer itself* has the ticker — i.e. the
    # company's own filing, not some unrelated 8-K that mentions the ticker.
    # display_names entries look like "NVIDIA CORP  (NVDA)  (CIK 0001045810)".
    ticker_marker = f"({ticker})"

    out = []
    for hit in hits[:50]:
        src = hit.get("_source", {})
        adsh = src.get("adsh", "")
        forms = src.get("form", "")
        display_names = src.get("display_names", []) or []
        filer = display_names[0] if display_names else ""
        if ticker_marker not in filer:
            continue
        filed = src.get("file_date") or ""
        cik = src.get("ciks", [""])[0]
        if cik and adsh:
            adsh_nodash = adsh.replace("-", "")
            link = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{adsh_nodash}/{adsh}-index.htm"
        else:
            link = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
        title = f"[{forms} {ticker}] {filer}".strip()
        out.append({
            "title": title[:200],
            "link": link,
            "summary": f"{filer} filed {forms} on {filed}",
            "published": filed,
            "source": "SEC EDGAR full-text",
            "_ticker": ticker,
        })
    return out


def collect_sec_edgar_fulltext() -> list:
    """Per-ticker full-text EDGAR scan with deduplication via seen_articles DB."""
    tickers = _load_relevant_tickers()
    if not tickers:
        return []

    until = datetime.utcnow().date().isoformat()
    since = (datetime.utcnow().date() - timedelta(days=EFTS_DAYS_BACK)).isoformat()

    conn = _ensure_db()
    new_articles: list = []

    for tkr in sorted(tickers):
        try:
            results = _efts_search(tkr, since, until)
        except Exception:
            results = []
        for art in results:
            aid = _article_id(art["link"], art["title"])
            if _is_seen(conn, aid):
                continue
            new_articles.append(art)
            _mark_seen(conn, aid, art["link"], art["title"], art["source"])
        # courtesy delay — SEC requires <=10 req/sec; we're well under but be polite
        time.sleep(0.3)

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    items = collect_sec_edgar()
    print(f"Got {len(items)} new EDGAR 8-K filings")
    for a in items[:10]:
        print(f"  [{a['_ticker']}] {a['title'][:100]}")
    ft = collect_sec_edgar_fulltext()
    print(f"Got {len(ft)} new EDGAR full-text filings")
    for a in ft[:10]:
        print(f"  [{a['_ticker']}] {a['title'][:100]}")
