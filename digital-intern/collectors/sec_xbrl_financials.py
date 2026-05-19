"""SEC XBRL financial facts collector.

Fetches quarterly EPS, revenue, and net income from SEC's free XBRL API for
tracked portfolio tickers. Complements sec_edgar.py (filing announcements) and
earnings_calendar.py (upcoming dates) with the *actual reported values* from
SEC filings — ground truth fundamental data.

API: https://data.sec.gov/api/xbrl/companyfacts/CIK{padded_cik}.json
CIK map: reuses data/sec_cik_to_ticker.json written by sec_insider_form4.py
         (refreshed weekly by that collector; falls back to a fresh fetch here).

Only emits articles for quarterly reports filed within the last 90 days that
haven't been seen before. One article per ticker per quarter per filing.
"""
import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
CIK_CACHE_PATH = BASE_DIR / "data" / "sec_cik_to_ticker.json"
XBRL_SEEN_PATH = BASE_DIR / "data" / "xbrl_financials_seen.json"

EDGAR_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "Digital-Intern-Daemon contact@digital-intern.local",
)
XBRL_BASE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_FILING_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcompany&CIK={cik}&type={form}&dateb=&owner=include&count=5"
)

# How far back to look for new filings
LOOKBACK_DAYS = 90
# XBRL concepts to fetch (try in order, use first that returns data)
REVENUE_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
EPS_CONCEPT = "EarningsPerShareDiluted"
NETINCOME_CONCEPT = "NetIncomeLoss"

# Rate limiting: SEC asks for ≤10 req/s, we stay conservative
_REQUEST_DELAY = 0.15  # seconds between XBRL API calls


def _load_portfolio_tickers() -> set[str]:
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


def _load_ticker_to_cik() -> dict[str, int]:
    """Return ticker→CIK map. Reads the shared cache written by sec_insider_form4."""
    try:
        with open(CIK_CACHE_PATH, "r") as f:
            raw = json.load(f)
        # raw is {str(cik): ticker} → invert to {ticker: cik}
        return {v.upper(): int(k) for k, v in raw.items()}
    except FileNotFoundError:
        pass
    except Exception:
        pass
    # Cache missing or corrupt — fetch fresh
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": EDGAR_USER_AGENT, "Accept": "application/json"},
            timeout=15,
        )
        if r.status_code != 200:
            return {}
        data = r.json()
        out: dict[str, int] = {}
        for row in data.values():
            cik = int(row.get("cik_str", 0))
            tkr = (row.get("ticker") or "").upper()
            if cik and tkr:
                out[tkr] = cik
        return out
    except Exception:
        return {}


def _load_seen() -> set[str]:
    """Load set of accession numbers already emitted."""
    try:
        with open(XBRL_SEEN_PATH, "r") as f:
            return set(json.load(f))
    except Exception:
        return set()


def _save_seen(seen: set[str]) -> None:
    XBRL_SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(XBRL_SEEN_PATH, "w") as f:
        json.dump(sorted(seen), f)


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


def _fmt_large(val: float | None) -> str:
    """Format large dollar values: 1234567890 → $1.23B"""
    if val is None:
        return "n/a"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e9:
        return f"{sign}${abs_val/1e9:.2f}B"
    if abs_val >= 1e6:
        return f"{sign}${abs_val/1e6:.1f}M"
    return f"{sign}${abs_val:,.0f}"


def _get_recent_quarterly(facts: dict, concept: str, cutoff: datetime) -> list[dict]:
    """Extract quarterly-period entries for a concept filed after cutoff."""
    units_map = facts.get("facts", {}).get("us-gaap", {}).get(concept, {}).get("units", {})
    # Try USD first, then USD/shares
    entries = units_map.get("USD") or units_map.get("USD/shares") or []
    results = []
    for e in entries:
        # Quarterly = has 'frame' like CY2025Q1, or fp like Q1/Q2/Q3
        # Exclude annual (fp=FY) and entries without filed date
        filed_str = e.get("filed", "")
        fp = e.get("fp", "")
        if not filed_str or fp == "FY":
            continue
        try:
            filed_dt = datetime.strptime(filed_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if filed_dt < cutoff:
            continue
        # Require quarterly-period: fp in Q1..Q4 or frame like CY2025Q1
        frame = e.get("frame", "")
        is_quarterly = fp in ("Q1", "Q2", "Q3", "Q4") or (
            frame and "Q" in frame and len(frame) >= 8
        )
        if not is_quarterly:
            continue
        results.append(e)
    # Deduplicate by accn, keep latest filed
    by_accn: dict[str, dict] = {}
    for e in results:
        accn = e.get("accn", "")
        if accn not in by_accn or e.get("filed", "") > by_accn[accn].get("filed", ""):
            by_accn[accn] = e
    return sorted(by_accn.values(), key=lambda x: x.get("filed", ""), reverse=True)


def collect_sec_xbrl_financials() -> list[dict]:
    """Fetch recent quarterly financial data for portfolio tickers via SEC XBRL API."""
    tickers = _load_portfolio_tickers()
    if not tickers:
        return []

    ticker_to_cik = _load_ticker_to_cik()
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    seen_accns = _load_seen()
    conn = _ensure_db()
    new_articles: list[dict] = []
    new_seen: set[str] = set()

    for ticker in sorted(tickers):
        cik = ticker_to_cik.get(ticker)
        if not cik:
            continue

        cik_padded = str(cik).zfill(10)
        url = XBRL_BASE.format(cik=cik_padded)
        try:
            r = requests.get(
                url,
                headers={"User-Agent": EDGAR_USER_AGENT, "Accept": "application/json"},
                timeout=20,
            )
            time.sleep(_REQUEST_DELAY)
            if r.status_code != 200:
                continue
            facts = r.json()
        except Exception:
            continue

        # Fetch quarterly data for each metric
        eps_entries = _get_recent_quarterly(facts, EPS_CONCEPT, cutoff)

        # Revenue: try concepts in order
        rev_entries: list[dict] = []
        for concept in REVENUE_CONCEPTS:
            rev_entries = _get_recent_quarterly(facts, concept, cutoff)
            if rev_entries:
                break

        ni_entries = _get_recent_quarterly(facts, NETINCOME_CONCEPT, cutoff)

        # Group by accession number — emit one article per unique quarterly filing
        all_accns: set[str] = set()
        for entry_list in (eps_entries, rev_entries, ni_entries):
            for e in entry_list:
                all_accns.add(e.get("accn", ""))
        all_accns.discard("")

        for accn in all_accns:
            if accn in seen_accns:
                continue

            # Find entries matching this accn
            eps_e = next((e for e in eps_entries if e.get("accn") == accn), None)
            rev_e = next((e for e in rev_entries if e.get("accn") == accn), None)
            ni_e = next((e for e in ni_entries if e.get("accn") == accn), None)

            # Use whichever entry exists for metadata
            meta = eps_e or rev_e or ni_e
            if not meta:
                continue

            fp = meta.get("fp", "?")
            fy = meta.get("fy", "?")
            filed = meta.get("filed", "?")
            form = meta.get("form", "10-Q")

            eps_val = eps_e.get("val") if eps_e else None
            rev_val = rev_e.get("val") if rev_e else None
            ni_val = ni_e.get("val") if ni_e else None

            eps_str = f"EPS ${eps_val:.2f}" if eps_val is not None else None
            rev_str = f"Rev {_fmt_large(rev_val)}" if rev_val is not None else None
            ni_str = f"NI {_fmt_large(ni_val)}" if ni_val is not None else None

            metrics = " | ".join(x for x in [eps_str, rev_str, ni_str] if x)
            title = f"[XBRL {ticker}] {fp} FY{fy}: {metrics} (filed {filed})"[:240]

            filing_url = EDGAR_FILING_URL.format(cik=cik_padded, form=form.replace("/", "%2F"))
            summary = (
                f"SEC XBRL financial facts: {ticker} {fp} FY{fy} ({form}, filed {filed}). "
                f"{metrics}. Accession: {accn}."
            )

            aid = _article_id(filing_url, title)
            if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
                new_seen.add(accn)
                continue

            new_articles.append({
                "title": title,
                "link": filing_url,
                "summary": summary,
                "published": filed,
                "source": "SEC XBRL Financial Facts",
                "_ticker": ticker,
            })
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                "VALUES (?, ?, ?, ?, ?)",
                (aid, filing_url, title, "SEC XBRL Financial Facts",
                 datetime.now(timezone.utc).isoformat()),
            )
            new_seen.add(accn)

    conn.commit()
    conn.close()
    seen_accns.update(new_seen)
    _save_seen(seen_accns)
    return new_articles


if __name__ == "__main__":
    items = collect_sec_xbrl_financials()
    print(f"Got {len(items)} new XBRL financial fact articles")
    for a in items[:10]:
        print(f"  [{a['_ticker']}] {a['title']}")
        print(f"     {a['link']}")
        print(f"     {a['summary'][:120]}")
