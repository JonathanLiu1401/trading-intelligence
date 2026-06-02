"""Treasury auction results collector.

Fetches completed UST auction results from TreasuryDirect's public API.
Auction announcements are already covered by treasury_auctions.py; this
collector focuses on POST-AUCTION results — the signals that matter:

  bid_to_cover     — total tendered / total accepted (demand strength)
  indirect_pct     — indirect bidders % of competitive accepted
                     (proxy for foreign central bank / overseas demand)
  direct_pct       — direct bidders % (hedge funds, asset managers)
  primary_dealer_pct — primary dealer absorption (residual buyer)
  awarded_rate     — high discount rate (Bills) or high yield (Notes/Bonds)

A bid-to-cover below the recent average signals weak demand; a surge in
primary dealer share (at the expense of indirect/direct) indicates the
Fed's primary dealers had to absorb excess supply — bearish for yields.

Endpoint: https://www.treasurydirect.gov/TA_WS/securities/auctioned?format=json
Returns ~250 most recent completed auctions.

Two-layer dedup matching other collectors:
  1. data/seen_articles.db keyed by cusip||auctionDate
  2. articles.db PRIMARY KEY = sha256(url||title) inside insert_batch.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
SOURCE = "treasury/auction_results"
ENDPOINT = "https://www.treasurydirect.gov/TA_WS/securities/auctioned?format=json"
LOOKBACK_DAYS = 7  # only emit results from the last 7 days
_UA = "Digital-Intern/1.0 (+macro-treasury-results)"


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


def _result_id(cusip: str, auction_date: str) -> str:
    return hashlib.sha256(f"ust_result:{cusip}:{auction_date}".encode()).hexdigest()


def _is_seen(conn: sqlite3.Connection, rid: str) -> bool:
    return bool(conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (rid,)).fetchone())


def _mark_seen(conn: sqlite3.Connection, rid: str, link: str, title: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles(id,link,title,source,first_seen) VALUES(?,?,?,?,?)",
        (rid, link, title, SOURCE, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _pct(num, denom) -> float | None:
    try:
        n, d = float(num), float(denom)
        return round(n / d * 100, 1) if d else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _fmt_rate(v) -> str:
    try:
        return f"{float(v):.3f}%"
    except (TypeError, ValueError):
        return "N/A"


def _build_article(rec: dict) -> dict | None:
    cusip = (rec.get("cusip") or "").strip()
    auction_date = (rec.get("auctionDate") or "")[:10]
    if not cusip or not auction_date:
        return None

    sec_type = (rec.get("securityType") or "").strip()
    term = (rec.get("securityTerm") or rec.get("term") or "").strip()
    issue_date = (rec.get("issueDate") or "")[:10]

    btc = rec.get("bidToCoverRatio")
    try:
        btc_f = round(float(btc), 2) if btc else None
    except (TypeError, ValueError):
        btc_f = None

    comp_accepted = rec.get("competitiveAccepted") or 0
    indirect_acc = rec.get("indirectBidderAccepted") or 0
    direct_acc = rec.get("directBidderAccepted") or 0
    primary_acc = rec.get("primaryDealerAccepted") or 0

    indirect_pct = _pct(indirect_acc, comp_accepted)
    direct_pct = _pct(direct_acc, comp_accepted)
    primary_pct = _pct(primary_acc, comp_accepted)

    # Awarded rate: Bills use highDiscountRate; Notes/Bonds use highYield
    awarded_rate = rec.get("highYield") or rec.get("highDiscountRate")
    tips = str(rec.get("tips") or "").lower() == "yes"

    # Build title with most actionable metrics
    btc_str = f"B/C {btc_f:.2f}" if btc_f else "B/C N/A"
    indir_str = f"indirect {indirect_pct:.0f}%" if indirect_pct is not None else ""
    rate_str = f"yield {_fmt_rate(awarded_rate)}" if awarded_rate else ""

    parts = [p for p in [btc_str, indir_str, rate_str] if p]
    title = (
        f"UST {term} {sec_type} auction result {auction_date}: "
        + ", ".join(parts)
        + (f" (CUSIP {cusip})" if cusip else "")
    )
    if tips:
        title = title.replace(f"UST {term}", f"UST {term} TIPS")

    # Body with full breakdown
    offering = rec.get("offeringAmount") or rec.get("totalAccepted") or ""
    body_parts = [
        f"Auction date: {auction_date}. Issue date: {issue_date}.",
        f"Security: {term} {sec_type}{' (TIPS)' if tips else ''}. CUSIP: {cusip}.",
    ]
    if offering:
        body_parts.append(f"Offering: ${int(float(offering)):,}")
    if btc_f:
        body_parts.append(f"Bid-to-cover: {btc_f:.2f}x")
    if awarded_rate:
        body_parts.append(f"Awarded rate: {_fmt_rate(awarded_rate)}")
    if indirect_pct is not None:
        body_parts.append(
            f"Indirect bidders: {indirect_pct:.1f}% of competitive (foreign demand proxy)"
        )
    if direct_pct is not None:
        body_parts.append(f"Direct bidders: {direct_pct:.1f}%")
    if primary_pct is not None:
        body_parts.append(f"Primary dealers: {primary_pct:.1f}% (residual absorber)")

    # Demand signal interpretation
    if btc_f is not None:
        if btc_f >= 3.0:
            body_parts.append("Signal: strong demand (B/C >= 3.0x).")
        elif btc_f <= 2.0:
            body_parts.append("Signal: weak demand (B/C <= 2.0x) — bearish for yields.")

    link = f"https://www.treasurydirect.gov/instit/annceresult/press/preanre/{auction_date[:4]}/{rec.get('pdfFilenameCompetitiveResults', '')}"

    return {
        "title": title,
        "link": link,
        "summary": " ".join(body_parts),
        "published": auction_date,
        "source": SOURCE,
        "_cusip": cusip,
        "_btc": btc_f,
        "_indirect_pct": indirect_pct,
    }


def collect_treasury_auction_results(lookback_days: int = LOOKBACK_DAYS) -> list[dict]:
    """Fetch recent completed UST auction results.

    Returns standard article dicts for insertion via ArticleStore.insert_batch.
    Only emits results from the past ``lookback_days`` days that haven't been
    seen before.
    """
    try:
        resp = requests.get(ENDPOINT, headers={"User-Agent": _UA}, timeout=12)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[treasury_auction_results] fetch error: {e}")
        return []

    if not isinstance(data, list):
        print("[treasury_auction_results] unexpected response format")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    conn = _ensure_db()
    out: list[dict] = []

    for rec in data:
        auction_date = (rec.get("auctionDate") or "")[:10]
        if not auction_date or auction_date < cutoff_str:
            continue
        # Skip if auction hasn't happened yet (no bid-to-cover)
        if not rec.get("bidToCoverRatio"):
            continue

        cusip = (rec.get("cusip") or "").strip()
        rid = _result_id(cusip, auction_date)
        if _is_seen(conn, rid):
            continue

        article = _build_article(rec)
        if not article:
            continue

        out.append(article)
        _mark_seen(conn, rid, article["link"], article["title"])

    print(f"[treasury_auction_results] {len(out)} new results from last {lookback_days} days")
    return out


if __name__ == "__main__":
    from storage.article_store import ArticleStore

    items = collect_treasury_auction_results(lookback_days=14)
    if not items:
        print("No new auction results found.")
    else:
        for item in items:
            print(f"\n  TITLE: {item['title']}")
            print(f"  SUMMARY: {item['summary'][:200]}...")
        store = ArticleStore()
        inserted = store.insert_batch(items)
        print(f"\nInserted {inserted} articles into articles.db")
