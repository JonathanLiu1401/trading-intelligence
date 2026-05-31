"""SEC EDGAR prospectus filing monitor.

Tracks 424B-series filings (IPO final prospectuses, secondary offerings,
convertible notes, shelf takedowns) and S-1/S-1A (IPO registration
amendments). These filings signal:

  424B4 — final IPO prospectus (stock about to list)
  424B3 — secondary offering / resale prospectus (dilution signal)
  424B2 — structured product / bank notes (less common signal)
  424B5 — shelf offering pricing supplement
  S-1/A — IPO registration amendment (IPO approaching effective date)

All events are kept regardless of portfolio membership because they represent
net new supply of shares hitting the market.

No API key. EDGAR ATOM feed. Rate-limit: 1s sleep between fetches.
"""
import hashlib
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

EDGAR_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "Digital-Intern-Daemon contact@digital-intern.local",
)

FETCH_TIMEOUT = 15

# Form types to monitor, each with a short human label.
FORM_CONFIGS = [
    ("424B4", "IPO-Prospectus"),
    ("424B3", "Secondary-Offering"),
    ("424B5", "Shelf-Pricing"),
    ("424B2", "Structured-Notes"),
    ("S-1%2FA", "S-1-Amendment"),   # URL-encoded "S-1/A"
]

_EDGAR_ATOM = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type={form_type}&dateb=&owner=include"
    "&count=40&search_text=&output=atom"
)

# Regex to extract company name from title like:
#   "424B4 - Acme Corp (0001234567) (Filer)"
_TITLE_RE = re.compile(
    r"^[A-Z0-9/]+ - (.+?)\s*\(\d+\)\s*\(Filer\)",
    re.IGNORECASE,
)

# Uppercase 1–6 letter tokens that often represent tickers in company names.
# We skip common English words / known false positives.
_TICKER_STOPLIST = {
    "THE", "AND", "INC", "CORP", "LLC", "LTD", "PLC", "CO", "LP",
    "NA", "NV", "SA", "AG", "SE", "ETF", "FUND", "TRUST", "GROUP",
    "BANK", "GLOBAL", "CAPITAL", "MANAGEMENT", "HOLDINGS", "ACQUISITION",
    "INTERNATIONAL", "FINANCE", "FINANCIAL", "EQUITY", "PARTNERS",
    "PROPERTIES", "TECHNOLOGIES", "SOLUTIONS", "SYSTEMS", "SERVICES",
    "A", "B", "C", "D", "I", "II", "III", "IV", "US", "USA",
}
_TICKER_PAT = re.compile(r"\b([A-Z]{1,6})\b")


def _extract_ticker_hint(company_name: str) -> str:
    """Best-effort: pull candidate ticker from company name."""
    tokens = _TICKER_PAT.findall(company_name.upper())
    for t in tokens:
        if t not in _TICKER_STOPLIST and len(t) >= 2:
            return t
    return ""


def _ensure_db(conn: sqlite3.Connection) -> None:
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


def _is_seen(conn: sqlite3.Connection, article_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM seen_articles WHERE id = ?", (article_id,)
    ).fetchone()
    return row is not None


def _mark_seen(conn: sqlite3.Connection, article_id: str, title: str, link: str, source: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) VALUES (?,?,?,?,?)",
        (article_id, link, title, source, now),
    )
    conn.commit()


def _fetch_form(form_type: str, label: str) -> list[dict]:
    url = _EDGAR_ATOM.format(form_type=form_type)
    feed = feedparser.parse(url, agent=EDGAR_USER_AGENT)
    status = feed.get("status", 0)
    if status not in (200, 0):
        return []

    articles = []
    for entry in feed.entries:
        title_raw = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        summary = entry.get("summary", "")

        # Parse company name
        m = _TITLE_RE.match(title_raw)
        company = m.group(1).strip() if m else title_raw

        # Extract filed date from summary HTML
        filed = ""
        date_m = re.search(r"<b>Filed:</b>\s*([\d-]+)", summary)
        if date_m:
            filed = date_m.group(1)

        # Extract accession number for dedup
        acc_m = re.search(r"AccNo:</b>\s*([\d-]+)", summary)
        accno = acc_m.group(1) if acc_m else link

        ticker_hint = _extract_ticker_hint(company)
        ticker_tag = f" [{ticker_hint}]" if ticker_hint else ""

        article_title = f"[SEC {label}]{ticker_tag} {company}"
        if filed:
            article_title += f" — filed {filed}"

        article_id = hashlib.sha256(f"sec_prospectus:{accno}:{form_type}".encode()).hexdigest()[:32]
        source = f"sec_prospectus/{label.lower()}"

        articles.append({
            "id": article_id,
            "title": article_title,
            "link": link,
            "summary": summary,
            "source": source,
            "published": filed or datetime.now(timezone.utc).isoformat(),
            "_ticker": ticker_hint,
            "_form_type": form_type,
            "_company": company,
        })

    return articles


def collect_sec_prospectus() -> list[dict]:
    """Fetch all configured form types and return new (unseen) articles."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    _ensure_db(conn)

    new_articles = []
    for form_type, label in FORM_CONFIGS:
        try:
            candidates = _fetch_form(form_type, label)
            for art in candidates:
                if not _is_seen(conn, art["id"]):
                    _mark_seen(conn, art["id"], art["title"], art["link"], art["source"])
                    new_articles.append(art)
            time.sleep(1.0)
        except Exception as e:
            print(f"[sec_prospectus] error fetching {form_type}: {e}")

    conn.close()
    return new_articles


if __name__ == "__main__":
    print("Running SEC prospectus collector test…")
    results = collect_sec_prospectus()
    print(f"\nNew filings found: {len(results)}")
    for r in results[:10]:
        print(f"  [{r['source']}] {r['title']}")
        print(f"    {r['link'][:80]}")
    if not results:
        print("  (all already seen — run again from a fresh state to verify)")
        # Re-run without dedup to show sample data
        print("\nSample feed data (ignoring dedup):")
        DB_PATH_tmp = Path("/tmp/_sec_prospectus_test.db")
        import os as _os
        if DB_PATH_tmp.exists():
            _os.unlink(DB_PATH_tmp)
        conn2 = sqlite3.connect(str(DB_PATH_tmp))
        import collectors.sec_prospectus_collector as _self
        _self.DB_PATH = DB_PATH_tmp
        for form_type, label in FORM_CONFIGS[:3]:
            items = _fetch_form(form_type, label)
            print(f"\n  {form_type} ({label}): {len(items)} entries")
            for it in items[:3]:
                print(f"    {it['title']}")
        conn2.close()
