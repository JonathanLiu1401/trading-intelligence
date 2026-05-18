"""SEC EDGAR Form 4 (insider transactions) collector.

Pulls the recent Form 4 RSS feed and keeps only filings where the issuer
matches a tracked ticker. Form 4 = officer/director/10%+ owner buys & sells —
strong directional signal not covered by the 8-K / full-text collectors.
"""
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
CIK_CACHE_PATH = BASE_DIR / "data" / "sec_cik_to_ticker.json"
CIK_CACHE_TTL_SEC = 7 * 24 * 3600  # weekly refresh

FORM4_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=4&dateb=&owner=include&count=100&output=atom"
)
EDGAR_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "Digital-Intern-Daemon contact@digital-intern.local",
)


def _load_relevant_tickers() -> set[str]:
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
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
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


_ISSUER_RE = re.compile(r"\(([A-Z]{1,6})\)")
_CIK_RE = re.compile(r"\((\d{10})\)\s*\(Issuer\)")


def _load_cik_to_ticker() -> dict:
    """Return CIK(int)→ticker(str). Cached on disk, refreshed weekly."""
    try:
        st = CIK_CACHE_PATH.stat()
        if (datetime.now().timestamp() - st.st_mtime) < CIK_CACHE_TTL_SEC:
            with open(CIK_CACHE_PATH, "r") as f:
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
        if r.status_code != 200:
            return {}
        data = r.json()
        out = {}
        for row in data.values():
            cik = int(row.get("cik_str", 0))
            tkr = (row.get("ticker") or "").upper()
            if cik and tkr:
                out[cik] = tkr
        CIK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CIK_CACHE_PATH, "w") as f:
            json.dump({str(k): v for k, v in out.items()}, f)
        return out
    except Exception:
        return {}


def collect_sec_form4() -> list:
    """Form 4 RSS — keep only filings whose issuer matches a tracked ticker."""
    tickers = _load_relevant_tickers()
    if not tickers:
        return []

    parsed = feedparser.parse(FORM4_URL, agent=EDGAR_USER_AGENT)
    if getattr(parsed, "bozo", 0) and not parsed.entries:
        return []

    cik_map = _load_cik_to_ticker()
    conn = _ensure_db()
    new_articles: list = []

    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        summary = entry.get("summary") or entry.get("description") or ""
        if not title or not link:
            continue
        # EDGAR getcurrent does prefix-matching on type=, so "type=4" returns
        # 4, 4/A, 40-F, 424B2 ... — require an exact "4 -" / "4/A -" head.
        if not (title.startswith("4 -") or title.startswith("4/A -")):
            continue

        # EDGAR Form 4 titles look like:
        # "4 - SMITH JOHN A (0001234567) (Reporting)"
        # The issuer ticker is NOT in the title — it's in the summary/category.
        # Form 4 RSS titles only carry the filer name + CIK, not a ticker.
        # Resolve issuer CIK → ticker via cached SEC company_tickers.json.
        matched = None
        m = _CIK_RE.search(title)
        if m and cik_map:
            tkr = cik_map.get(int(m.group(1)))
            if tkr and tkr in tickers:
                matched = tkr
        if not matched:
            # Fallback: parenthesised ticker anywhere in title/summary
            haystack = f"{title} {summary}".upper()
            for cand in _ISSUER_RE.findall(haystack):
                if cand in tickers:
                    matched = cand
                    break
        if not matched:
            continue

        aid = _article_id(link, title)
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
            continue

        published = entry.get("updated") or entry.get("published") or ""
        new_articles.append({
            "title": f"[Form 4 {matched}] {title}"[:240],
            "link": link,
            "summary": summary,
            "published": published,
            "source": "SEC EDGAR Form 4",
            "_ticker": matched,
        })
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (aid, link, title, "SEC EDGAR Form 4", datetime.now(timezone.utc).isoformat()),
        )

    conn.commit()
    conn.close()
    return new_articles


if __name__ == "__main__":
    items = collect_sec_form4()
    print(f"Got {len(items)} new Form 4 insider filings")
    for a in items[:10]:
        print(f"  [{a['_ticker']}] {a['title'][:120]}")
        print(f"     {a['link']}")
