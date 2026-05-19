"""CISA Known Exploited Vulnerabilities (KEV) catalog collector.

When CISA adds a CVE to KEV it means active in-the-wild exploitation has been
confirmed and federal agencies must patch by `dueDate`. Two distinct trade
signals:

  1. **Vendor risk** — Microsoft / Cisco / Oracle / Fortinet / etc. landing in
     KEV is a forward-looking reputational / liability hit on the named vendor.
  2. **Cybersecurity sector tailwind** — a surge of KEV adds (especially the
     ransomware-flagged ones) lifts pure-play cybersec names (CRWD, S, ZS, PANW,
     FTNT, OKTA) as enterprise spend forecasts firm.

Public JSON, no auth. Polled once an hour — the upstream file refreshes ~daily
but multiple adds in a single calendar day are common during active campaigns.
Dedup via shared `seen_articles.db` (same pattern as FDA / SEC collectors).
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
SOURCE_TAG = "CISA/KEV"
USER_AGENT = "Mozilla/5.0 (Digital Intern Daemon; contact@digital-intern.local)"
HTTP_TIMEOUT = 20

# vendorProject (as CISA writes it) → primary listed ticker. Lower-cased keys;
# match is case-insensitive. Private / acquired / foundation-owned vendors are
# intentionally omitted — the title still names them, but no ticker is tagged.
VENDOR_TICKERS: dict[str, str] = {
    "microsoft":       "MSFT",
    "cisco":           "CSCO",
    "oracle":          "ORCL",
    "adobe":           "ADBE",
    "apple":           "AAPL",
    "google":          "GOOGL",
    "alphabet":        "GOOGL",
    "ibm":             "IBM",
    "fortinet":        "FTNT",
    "palo alto networks": "PANW",
    "cloudflare":      "NET",
    "f5":              "FFIV",
    "f5 networks":     "FFIV",
    "atlassian":       "TEAM",
    "sap":             "SAP",
    "salesforce":      "CRM",
    "servicenow":      "NOW",
    "zoom":            "ZM",
    "netgear":         "NTGR",
    "progress":        "PRGS",
    "progress software": "PRGS",
    "broadcom":        "AVGO",
    "vmware":          "AVGO",  # Broadcom acquired VMware
    "mitel":           "MITL",
    "solarwinds":      "SWI",
    "okta":            "OKTA",
    "splunk":          "CSCO",  # Cisco acquired Splunk
    "crowdstrike":     "CRWD",
    "sentinelone":     "S",
    "zscaler":         "ZS",
    "akamai":          "AKAM",
    "juniper":         "JNPR",
    "juniper networks": "JNPR",
    "qualcomm":        "QCOM",
    "intel":           "INTC",
    "nvidia":          "NVDA",
    "hewlett packard enterprise": "HPE",
    "hp":              "HPQ",
    "dell":            "DELL",
    "amazon":          "AMZN",
    "amazon web services": "AMZN",
    "meta":            "META",
    "facebook":        "META",
    "github":          "MSFT",
    "linkedin":        "MSFT",
}


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


def _ticker_for(vendor: str) -> str | None:
    if not vendor:
        return None
    return VENDOR_TICKERS.get(vendor.strip().lower())


def _entry_to_article(entry: dict) -> dict | None:
    cve = (entry.get("cveID") or "").strip()
    vendor = (entry.get("vendorProject") or "").strip()
    product = (entry.get("product") or "").strip()
    name = (entry.get("vulnerabilityName") or "").strip()
    short = (entry.get("shortDescription") or "").strip()
    ransomware = (entry.get("knownRansomwareCampaignUse") or "").strip()
    date_added = (entry.get("dateAdded") or "").strip()
    due_date = (entry.get("dueDate") or "").strip()
    if not cve or not vendor:
        return None

    ticker = _ticker_for(vendor)
    ticker_tag = f"${ticker} " if ticker else ""
    rw_tag = "[RANSOMWARE] " if ransomware.lower() == "known" else ""

    # Build a title the heuristic scorer + ArticleNet can latch onto. Ticker
    # symbol and the word "exploit" are both salient keyword features.
    title = (
        f"{rw_tag}{ticker_tag}CISA KEV: {vendor} {product} actively exploited "
        f"({cve}) — {name}"
    ).strip()
    if len(title) > 280:
        title = title[:277] + "..."

    summary_parts = [short]
    if ticker:
        summary_parts.append(f"Affected vendor ticker: {ticker}.")
    if ransomware.lower() == "known":
        summary_parts.append("CISA flags known ransomware-campaign use.")
    if due_date:
        summary_parts.append(f"Federal patch deadline: {due_date}.")
    summary = " ".join(p for p in summary_parts if p)

    # NVD link is stable + public, makes a good canonical URL.
    link = f"https://nvd.nist.gov/vuln/detail/{cve}"

    published = ""
    if date_added:
        try:
            dt = datetime.strptime(date_added, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            published = dt.isoformat()
        except ValueError:
            published = date_added

    return {
        "title": title,
        "link": link,
        "summary": summary,
        "published": published,
        "source": SOURCE_TAG,
    }


def collect_cisa_kev(max_items: int = 50) -> list[dict]:
    """Fetch the KEV catalog and return the newest unseen entries.

    `max_items` caps the *post-dedup* yield per cycle so a backfill of an
    empty seen_articles.db doesn't blast 1500 rows into the pipeline at once.
    """
    try:
        r = requests.get(KEV_URL, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[cisa_kev] fetch error: {e}")
        return []

    vulns = data.get("vulnerabilities") or []
    if not isinstance(vulns, list):
        return []

    # Sort newest first so the cap surfaces the most actionable items.
    vulns_sorted = sorted(vulns, key=lambda v: v.get("dateAdded") or "", reverse=True)

    conn = _ensure_db()
    out: list[dict] = []
    seen_in_run: set[str] = set()
    try:
        for entry in vulns_sorted:
            art = _entry_to_article(entry)
            if not art:
                continue
            aid = _article_id(art["link"], art["title"])
            if aid in seen_in_run:
                continue
            seen_in_run.add(aid)
            if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
                continue
            out.append(art)
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
                "VALUES (?, ?, ?, ?, ?)",
                (aid, art["link"], art["title"], SOURCE_TAG,
                 datetime.now(timezone.utc).isoformat()),
            )
            if len(out) >= max_items:
                break
        conn.commit()
    finally:
        conn.close()
    return out


if __name__ == "__main__":
    t0 = time.time()
    items = collect_cisa_kev()
    dt = time.time() - t0
    print(f"[cisa_kev] {len(items)} new items in {dt:.1f}s")
    for a in items[:10]:
        print(f"  {a['title'][:110]}")
        print(f"     {a['link']}")
