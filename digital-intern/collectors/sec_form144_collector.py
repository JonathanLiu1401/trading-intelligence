"""SEC EDGAR Form 144 (insider intent-to-sell) collector.

Form 144 = notice of proposed sale of restricted or control securities.
Filed before insiders sell under Rule 144 — a bearish leading indicator
that precedes actual Form 4 sale disclosures by days or weeks.

Distinct from:
  sec_insider_form4.py   — records completed buy/sell transactions
  openinsider_cluster.py — detects buy clusters from Form 4

This collector watches for INTENT to sell (Form 144), which is often
missed but signals that an insider has already decided to exit.

Dedup: one article per accession number via seen_articles.db.
"""
import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

try:
    from core.logger import get_logger
    _log = get_logger("sec_form144")
except Exception:
    _log = logging.getLogger("sec_form144")

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
SEEN_DB = BASE_DIR / "data" / "seen_articles.db"
CIK_CACHE_PATH = BASE_DIR / "data" / "sec_cik_to_ticker.json"
CIK_CACHE_TTL_SEC = 7 * 24 * 3600

FORM144_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=144&dateb=&owner=include&count=100&output=atom"
)
EDGAR_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "Digital-Intern-Daemon contact@digital-intern.local",
)

_CIK_RE = re.compile(r"\((\d{7,10})\)")
_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")


def _load_portfolio_tickers() -> set[str]:
    try:
        with open(PORTFOLIO_PATH) as f:
            data = json.load(f)
    except Exception:
        return set()
    tickers: set[str] = set()
    for pos in data.get("positions", []):
        t = (pos.get("ticker") or "").upper()
        if t:
            tickers.add(t)
    for opt in data.get("options", []):
        u = (opt.get("underlying") or "").upper()
        if u:
            tickers.add(u)
    for t in data.get("sector_watchlist", []):
        if t:
            tickers.add(t.upper())
    return tickers


def _load_cik_map() -> dict:
    """Return int(CIK) → ticker str. Weekly-cached from SEC company_tickers.json."""
    try:
        st = CIK_CACHE_PATH.stat()
        if (datetime.now().timestamp() - st.st_mtime) < CIK_CACHE_TTL_SEC:
            with open(CIK_CACHE_PATH) as f:
                raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
    except FileNotFoundError:
        pass
    except Exception:
        pass
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": EDGAR_USER_AGENT, "Accept": "application/json"},
            timeout=15,
        )
        if r.status_code == 200:
            raw = r.json()
            mapping = {v["cik_str"]: v["ticker"].upper() for v in raw.values() if "cik_str" in v and "ticker" in v}
            CIK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CIK_CACHE_PATH, "w") as f:
                json.dump(mapping, f)
            return {int(k): v for k, v in mapping.items()}
    except Exception as e:
        _log.warning("cik map fetch failed: %s", e)
    return {}


def _ensure_seen_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SEEN_DB), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY,
            link TEXT,
            title TEXT,
            source TEXT,
            first_seen TEXT
        )"""
    )
    conn.commit()
    return conn


def _article_id(link: str, title: str) -> str:
    return hashlib.sha256(f"{link}|{title}".encode()).hexdigest()


def collect_sec_form144() -> list:
    """Fetch Form 144 RSS and return new articles for portfolio/watchlist tickers."""
    tickers = _load_portfolio_tickers()
    cik_map = _load_cik_map()
    conn = _ensure_seen_db()
    new_articles = []

    parsed = feedparser.parse(FORM144_URL, agent=EDGAR_USER_AGENT)
    if getattr(parsed, "bozo", 0) and not parsed.entries:
        _log.warning("Form 144 RSS parse error")
        conn.close()
        return []

    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        summary = entry.get("summary") or entry.get("description") or ""
        if not title or not link:
            continue

        # Only "144 -" or "144/A -" type entries (Subject = the company)
        if not (title.startswith("144 -") or title.startswith("144/A -")):
            continue
        # Filter to Subject entries (the company being sold, not the filer)
        if "(Subject)" not in title:
            continue

        # Try CIK → ticker lookup
        matched = None
        m = _CIK_RE.search(title)
        if m and cik_map:
            tkr = cik_map.get(int(m.group(1)))
            if tkr and tkr in tickers:
                matched = tkr

        if not matched:
            # Fallback: scan summary for known tickers
            haystack = (title + " " + summary).upper()
            for cand in _TICKER_RE.findall(haystack):
                if cand in tickers:
                    matched = cand
                    break

        if not matched:
            # Alert on large-cap well-known names even if not in portfolio
            # by scanning the company name portion for known tickers
            company_part = re.sub(r"\(.*?\)", "", title).upper()
            for cand in _TICKER_RE.findall(company_part):
                if len(cand) >= 2 and cand in tickers:
                    matched = cand
                    break

        if not matched:
            continue

        aid = _article_id(link, title)
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
            continue

        published = entry.get("updated") or entry.get("published") or ""
        company_name = re.split(r"\(", title.replace("144 - ", "").replace("144/A - ", ""), 1)[0].strip()

        new_articles.append({
            "title": f"[Form 144 SELL INTENT {matched}] {company_name} — insider Rule 144 sale notice",
            "link": link,
            "summary": f"SEC Form 144 filed: {title}. {summary}",
            "published": published,
            "source": "SEC EDGAR Form 144",
            "_ticker": matched,
        })
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?, ?, ?, ?, ?)",
            (aid, link, title, "SEC EDGAR Form 144", datetime.now(timezone.utc).isoformat()),
        )

    conn.commit()
    conn.close()
    _log.info("form144: %d new insider sell-intent filings", len(new_articles))
    return new_articles


if __name__ == "__main__":
    items = collect_sec_form144()
    print(f"Got {len(items)} new Form 144 insider sell-intent notices")
    for a in items[:5]:
        print(" ", a["title"])
        print("  ->", a["link"])
