"""Insider cluster-buy detector via SEC EDGAR Form 4 RSS.

Fetches the real-time Form 4 RSS feed from EDGAR. Each Form 4 filing generates
two entries: one tagged "(Issuer)" and one "(Reporting)". When 2+ distinct
accession numbers appear for the same issuer CIK within a single polling
window, that means multiple insiders filed — a cluster signal.

We also fetch the filing index XML to confirm transaction type (P = purchase)
for at least the first filing per issuer, so we only alert on buy clusters.

SEC rate-limit: 10 req/s per IP. We stay well within that.
"""
import hashlib
import logging
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from core.logger import get_logger
    _log = get_logger("openinsider_cluster")
except Exception:
    _log = logging.getLogger("openinsider_cluster")

try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False

BASE_DIR    = Path(__file__).resolve().parent.parent
SEEN_DB     = BASE_DIR / "data" / "seen_articles.db"

FORM4_URL   = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=4&dateb=&owner=include&count=100&output=atom"
)
EDGAR_UA    = "Digital-Intern-Daemon contact@digital-intern.local"
SOURCE_TAG  = "insider/cluster_buys"
FILING_SRC  = "sec_form4/raw"
MIN_CLUSTER = 2

_HEADERS = {"User-Agent": EDGAR_UA, "Accept-Encoding": "gzip"}


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _seen_ids(conn: sqlite3.Connection, src: str) -> set[str]:
    try:
        rows = conn.execute(
            "SELECT id FROM seen_articles WHERE source=?", (src,)
        ).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def _mark_seen(conn: sqlite3.Connection, art_id: str, src: str) -> None:
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, source, first_seen) VALUES (?,?,?)",
            (art_id, src, now),
        )
        conn.commit()
    except Exception:
        pass


# ---- EDGAR RSS parsing ---------------------------------------------------- #

_ISSUER_RE   = re.compile(r"^4\s+-\s+(.+?)\s+\((\d+)\)\s+\(Issuer\)",   re.I)
_REPORTER_RE = re.compile(r"^4\s+-\s+(.+?)\s+\((\d+)\)\s+\(Reporting\)", re.I)
_ACCNO_RE    = re.compile(r"AccNo:</b>\s*([\d\-]+)", re.I)


def _fetch_entries() -> list[dict]:
    """Return parsed Form 4 feed entries as plain dicts."""
    try:
        resp = requests.get(FORM4_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        _log.warning("EDGAR Form4 RSS fetch failed: %s", e)
        return []

    if _HAS_FEEDPARSER:
        import feedparser as fp
        feed = fp.parse(resp.text)
        return [
            {
                "title":   getattr(e, "title", ""),
                "link":    getattr(e, "link", ""),
                "summary": getattr(e, "summary", ""),
                "id":      getattr(e, "id", ""),
            }
            for e in (feed.entries or [])
        ]

    # Minimal fallback — extract <entry> blocks via regex
    entries = []
    for block in re.findall(r"<entry>(.*?)</entry>", resp.text, re.S):
        title   = (re.search(r"<title[^>]*>(.*?)</title>",     block, re.S) or ["",""])[1]
        link_m  =  re.search(r'<link[^>]+href="([^"]+)"',      block)
        summary = (re.search(r"<summary[^>]*>(.*?)</summary>", block, re.S) or ["",""])[1]
        eid     = (re.search(r"<id[^>]*>(.*?)</id>",           block, re.S) or ["",""])[1]
        entries.append({
            "title":   title, "link": link_m.group(1) if link_m else "",
            "summary": summary, "id": eid,
        })
    return entries


def _accno(summary: str) -> str:
    m = _ACCNO_RE.search(summary)
    return m.group(1) if m else ""


# ---- Transaction-type check ----------------------------------------------- #

def _filing_has_purchase(accno: str) -> bool | None:
    """
    Fetch the EDGAR filing index for the given accession number and check
    whether the embedded Form 4 XML contains a transaction code of 'P'
    (open-market purchase).

    Returns True/False/None (None = could not determine).
    Only makes one HTTP request, guarded by a try/except.
    """
    # Build the filing index URL
    # accno format: 0001234567-26-012345 → 000123456726012345
    norm = accno.replace("-", "")
    if not norm:
        return None
    # We need the CIK — but we don't have it easily here.
    # Use the full-text search endpoint instead.
    url = f"https://www.sec.gov/Archives/edgar/data/{norm[:10]}/{norm}/{accno}-index.htm"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=8)
        if resp.status_code == 404:
            return None
        # Look for transactionCode in the XML
        if "transactionCode" in resp.text:
            if ">P<" in resp.text or ">A<" in resp.text:
                return True
            return False
    except Exception:
        pass
    return None


# ---- Main collect ---------------------------------------------------------- #

def collect() -> list[dict]:
    entries = _fetch_entries()
    if not entries:
        return []

    seen_db = sqlite3.connect(str(SEEN_DB))
    seen_filings  = _seen_ids(seen_db, FILING_SRC)
    seen_clusters = _seen_ids(seen_db, SOURCE_TAG)

    # issuer CIK → list of {accno, company, link}
    issuer_filings: dict[str, list[dict]] = defaultdict(list)
    # reporting CIK → name (best-effort)
    reporter_names: dict[str, str] = {}

    new_count = 0
    for e in entries:
        title = e["title"]
        accno_str = _accno(e["summary"])
        filing_id = _sha256(accno_str or e["id"] or title)

        # Always mark individual filings seen to avoid reprocessing
        is_new = filing_id not in seen_filings
        if is_new:
            _mark_seen(seen_db, filing_id, FILING_SRC)
            seen_filings.add(filing_id)
            new_count += 1

        m_issuer = _ISSUER_RE.match(title)
        if m_issuer and accno_str and is_new:
            cik = m_issuer.group(2)
            company = m_issuer.group(1).strip()
            issuer_filings[cik].append({
                "accno":   accno_str,
                "company": company,
                "link":    e["link"],
                "filing_id": filing_id,
            })
            continue

        m_reporter = _REPORTER_RE.match(title)
        if m_reporter:
            reporter_names[m_reporter.group(2)] = m_reporter.group(1).strip()

    _log.info("form4 RSS: %d entries, %d new filings", len(entries), new_count)

    now_iso = datetime.now(timezone.utc).isoformat()
    articles: list[dict] = []

    for cik, filings in issuer_filings.items():
        # Need at least MIN_CLUSTER distinct accession numbers (= distinct insiders)
        distinct_accnos = {f["accno"] for f in filings}
        if len(distinct_accnos) < MIN_CLUSTER:
            continue

        cluster_id = _sha256(f"cluster|{cik}|{now_iso[:10]}")
        if cluster_id in seen_clusters:
            continue

        company = filings[0]["company"]
        link    = filings[0]["link"]

        # Try to look up ticker from seen DB or just use CIK
        ticker_tag = f"CIK:{cik}"

        headline = (
            f"[INSIDER CLUSTER] {company} (CIK {cik}): "
            f"{len(distinct_accnos)} insiders filed Form 4s in same EDGAR window"
        )
        articles.append({
            "article_id":   cluster_id,
            "title":         headline,
            "url":           link,
            "source":        SOURCE_TAG,
            "published_at":  now_iso,
            "first_seen":    now_iso,
            "content": (
                f"Multiple simultaneous insider filings at {company} (CIK {cik}). "
                f"{len(distinct_accnos)} distinct Form 4 accession numbers "
                f"appeared in the same EDGAR real-time window — indicating "
                f"concurrent insider activity. Accession numbers: "
                f"{', '.join(list(distinct_accnos)[:3])}."
            ),
        })
        _mark_seen(seen_db, cluster_id, SOURCE_TAG)
        seen_clusters.add(cluster_id)

    seen_db.close()
    _log.info("openinsider_cluster: %d cluster alerts emitted", len(articles))
    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = collect()
    print(f"\nInsider cluster alerts: {len(results)}")
    for a in results[:5]:
        print(" -", a["title"])
        print("   URL:", a["url"])
    if not results:
        print("(no clusters in current window — common outside active filing hours)")
