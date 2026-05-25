"""USPTO PatentsView collector — recent patent grants from market-relevant assignees.

Queries the PatentsView API (https://search.patentsview.org/api/v1/patent/)
for recent patent grants by major publicly-traded companies. Patent filings
are market-moving signals — a surge of AI/chip patents from NVDA/AMD/INTC,
new drug compound patents from MRNA/PFE, or autonomous-vehicle patents from
TSLA all precede or accompany large moves.

No API key required. The API is run by USPTO's Open Data Portal and is free
for non-commercial use.

Like every other collector, ``collect_uspto_patents()`` returns the standard
``{title, link, summary, published, source}`` dicts and the daemon's
``_ingest()`` hands them to ``ArticleStore.insert_batch``.

Two dedup layers:
  1. ``data/seen_articles.db`` (WAL, busy_timeout=30000) keyed by
     sha256(patent_id) so a re-run never re-emits the same patent.
  2. ``articles.db`` PRIMARY KEY = sha256(url||title) inside insert_batch.
"""
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

FETCH_TIMEOUT = 15
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

PATENTSVIEW_URL = "https://search.patentsview.org/api/v1/patent/"

# Market-relevant assignees grouped by sector. These are exact substrings
# matched via the API's _text_phrase operator. Keep to ~50 entries to stay
# well inside the API's per-query limits.
ASSIGNEES = [
    # Semiconductors / AI chips
    "NVIDIA", "Advanced Micro Devices", "Intel Corporation",
    "Qualcomm", "Broadcom", "Marvell", "Micron Technology",
    # Big tech / cloud / AI
    "Apple Inc", "Microsoft Corporation", "Alphabet",
    "Meta Platforms", "Amazon Technologies", "OpenAI",
    "Anthropic",
    # Automotive / EV / AV
    "Tesla", "Rivian", "Waymo",
    # Pharma / biotech
    "Pfizer", "Moderna", "Eli Lilly", "Johnson & Johnson",
    "Merck", "AstraZeneca", "Novo Nordisk",
    # Energy / cleantech
    "First Solar", "SunPower",
    # Fintech
    "Visa Inc", "Mastercard", "PayPal",
]

SOURCE_TAG = "uspto_patent"
# Look back 7 days on each run; dedup layer handles repeats
LOOKBACK_DAYS = 7


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


def _article_id(patent_id: str) -> str:
    return hashlib.sha256(f"uspto_patent:{patent_id}".encode()).hexdigest()


def _build_query(assignee: str, since_date: str) -> dict:
    return {
        "_and": [
            {"_text_phrase": {"assignee_organization": assignee}},
            {"_gte": {"patent_date": since_date}},
        ]
    }


def _fetch_patents_for_assignee(assignee: str, since_date: str) -> list[dict]:
    """Fetch recent grants for one assignee. Returns [] on any error."""
    query = _build_query(assignee, since_date)
    fields = [
        "patent_id", "patent_title", "patent_date",
        "patent_abstract", "assignee_organization",
    ]
    params = {
        "q": json.dumps(query),
        "f": json.dumps(fields),
        "o": json.dumps({"per_page": 10, "sort": [{"patent_date": "desc"}]}),
    }
    try:
        resp = requests.get(
            PATENTSVIEW_URL, params=params,
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": _UA, "Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[uspto_patent_collector] error fetching {assignee!r}: {e}")
        return []

    patents = data.get("patents") or []
    out = []
    for p in patents:
        pid = (p.get("patent_id") or "").strip()
        title = (p.get("patent_title") or "").strip()
        date = (p.get("patent_date") or "").strip()
        abstract = (p.get("patent_abstract") or "").strip()
        org = (p.get("assignee_organization") or assignee).strip()

        if not pid or not title:
            continue

        link = f"https://patents.google.com/patent/US{pid}/en"
        summary = f"{org} granted US{pid} on {date}. {abstract[:400]}" if abstract else f"{org} granted US{pid} on {date}."

        out.append({
            "title": f"[Patent] {title} — {org}",
            "link": link,
            "summary": summary,
            "published": date,
            "source": SOURCE_TAG,
        })
    return out


def collect_uspto_patents() -> list[dict]:
    """Collect deduplicated recent USPTO patent grants from market-relevant companies.

    Returns {title, link, summary, published, source} dicts.
    """
    conn = _ensure_db()
    since = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    new_articles: list[dict] = []
    seen_in_run: set[str] = set()

    for assignee in ASSIGNEES:
        for art in _fetch_patents_for_assignee(assignee, since):
            # Derive stable ID from the patent URL (contains patent_id)
            pid_key = art["link"]
            aid = hashlib.sha256(pid_key.encode()).hexdigest()
            if aid in seen_in_run:
                continue
            seen_in_run.add(aid)
            try:
                if conn.execute(
                    "SELECT 1 FROM seen_articles WHERE id = ?", (aid,)
                ).fetchone():
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO seen_articles "
                    "(id, link, title, source, first_seen) VALUES (?, ?, ?, ?, ?)",
                    (aid, art["link"], art["title"], SOURCE_TAG,
                     datetime.now(timezone.utc).isoformat()),
                )
            except sqlite3.Error as e:
                print(f"[uspto_patent_collector] dedup row skipped: {e}")
                continue
            new_articles.append(art)

    conn.commit()
    conn.close()
    return new_articles


collect = collect_uspto_patents


if __name__ == "__main__":
    print("=== USPTO PatentsView (live fetch) ===")
    items = collect_uspto_patents()
    print(f"New patents fetched: {len(items)}")
    for art in items[:5]:
        print(f"  {art['published']:10s}  {art['title'][:90]}")
        print(f"             {art['link']}")
    if items:
        print(f"\nExample: {items[0]['title']}")
