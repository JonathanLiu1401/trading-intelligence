"""SEC EDGAR 8-K bulk importer — 1993 to present.

Downloads quarterly full-text indexes from EDGAR, filters for 8-K filings
from S&P 500-class companies, fetches the filing header (not full text — the
header contains the subject company name, form type, date, and a URL to the
actual filing). Inserts into ArticleStore as financial news with the
filing URL and a synthetic title like "8-K: Apple Inc (AAPL) 2019-01-15".

EDGAR rate limit: 10 req/sec. We stay at 5 req/sec to be polite.

Usage:
    python scripts/sec_edgar_bulk.py        # 1994 Q1 → now
    python scripts/sec_edgar_bulk.py 2015   # custom start year
"""
from __future__ import annotations

import os
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from io import StringIO

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from storage.article_store import ArticleStore, _get_db_path

for line in (BASE_DIR / ".env").read_text().splitlines():
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

USER_AGENT = os.environ.get("SEC_USER_AGENT", "Digital-Intern contact@digital-intern.local")
SLEEP_PER_REQ = 0.2      # 5 req/sec — well under 10 req/sec limit
CHECKPOINT_PATH = BASE_DIR / "data" / "edgar_sweep_checkpoint.json"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# Companies to track — symbol → rough name patterns that appear in EDGAR
# We only keep 8-Ks from companies in this set (by CIK or name match)
# Using major S&P 100 + our watchlist companies
TARGET_NAMES = {
    "APPLE", "MICROSOFT", "AMAZON", "ALPHABET", "GOOGLE", "META", "NVIDIA",
    "TESLA", "BERKSHIRE", "JPMORGAN", "JOHNSON", "UNITEDHEALTH", "EXXON",
    "VISA", "MASTERCARD", "PROCTER", "WALMART", "CHEVRON", "HOME DEPOT",
    "ABBVIE", "PFIZER", "MERCK", "JOHNSON & JOHNSON", "ELI LILLY",
    "INTEL", "AMD", "ADVANCED MICRO", "QUALCOMM", "BROADCOM", "MICRON",
    "APPLIED MATERIALS", "LAM RESEARCH", "KLA", "TEXAS INSTRUMENTS",
    "ASML", "TAIWAN SEMICONDUCTOR", "TSMC", "SK HYNIX", "SAMSUNG",
    "LUMENTUM", "AXT", "TOWER SEMICONDUCTOR", "D-WAVE", "ORACLE",
    "BANK OF AMERICA", "GOLDMAN SACHS", "MORGAN STANLEY", "WELLS FARGO",
    "CITIGROUP", "BLACKROCK", "CHARLES SCHWAB", "COINBASE",
    "NETFLIX", "SALESFORCE", "ADOBE", "SERVICENOW", "SNOWFLAKE",
    "PALANTIR", "DATADOG", "CROWDSTRIKE", "FORTINET", "PALO ALTO",
    "AMDOCS", "OPENAI",  # AMDOCS for AXTI adjacent
}


def _quarter_indexes(start_year: int) -> list[tuple[int, int]]:
    today = date.today()
    out = []
    for y in range(start_year, today.year + 1):
        for q in range(1, 5):
            # Skip future quarters
            q_start = date(y, (q - 1) * 3 + 1, 1)
            if q_start > today:
                break
            out.append((y, q))
    return out


def _fetch_quarter_index(year: int, quarter: int) -> list[dict]:
    """Download and parse the EDGAR full-index company.idx for one quarter."""
    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/company.idx"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return []
    except Exception:
        return []

    lines = r.text.splitlines()
    # Detect column positions from the header line — EDGAR format has shifted
    # over time. Locate "edgar/data/" in a non-header data line to anchor the
    # filename column, then use the date (YYYY-MM-DD) and CIK (digits) before it.
    filings = []
    for line in lines[8:]:
        if len(line) < 50:
            continue
        # Robust parse: find "edgar/data/" for filename, then scan backwards for
        # the YYYY-MM-DD date, then the preceding numeric CIK, then form type.
        fi = line.find("edgar/data/")
        if fi < 0:
            continue
        filename = line[fi:].strip()
        prefix = line[:fi]
        # Date: last YYYY-MM-DD pattern before filename
        import re as _re
        m = list(_re.finditer(r"\d{4}-\d{2}-\d{2}", prefix))
        if not m:
            continue
        dm = m[-1]
        date_str = dm.group()
        # CIK: digits immediately before date field
        pre_date = prefix[:dm.start()].strip()
        cik_m = _re.search(r"(\d+)\s*$", pre_date)
        cik = cik_m.group(1) if cik_m else ""
        # Form type + company: everything before the CIK, split at last word boundary
        pre_cik = prefix[:cik_m.start()].strip() if cik_m else prefix.strip()
        # Company is at 0-61 (62 chars), form type follows — split on 2+ spaces
        parts = _re.split(r"  +", pre_cik.strip())
        if len(parts) < 2:
            continue
        form_type = parts[-1].strip()
        company = " ".join(parts[:-1]).strip()

        if form_type not in ("8-K", "8-K/A"):
            continue

        # Filter for target companies
        co_upper = company.upper()
        if not any(t in co_upper for t in TARGET_NAMES):
            continue

        try:
            published = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            published = None

        filings.append({
            "company": company,
            "form_type": form_type,
            "date": date_str,
            "cik": cik,
            "filename": filename,
            "published_dt": published,
        })

    return filings


def run(start_year: int = 1994):
    import json
    done: set[str] = set()
    if CHECKPOINT_PATH.exists():
        try:
            done = set(json.loads(CHECKPOINT_PATH.read_text()).get("done", []))
        except Exception:
            pass

    store = ArticleStore()
    quarters = _quarter_indexes(start_year)
    inserted_total = 0
    print(f"[edgar_bulk] {len(quarters)} quarters from {start_year} | writing to {_get_db_path()}")

    for i, (year, q) in enumerate(quarters):
        qkey = f"{year}Q{q}"
        if qkey in done:
            continue

        filings = _fetch_quarter_index(year, q)
        time.sleep(SLEEP_PER_REQ)

        to_insert = []
        for f in filings:
            filing_url = f"https://www.sec.gov/Archives/{f['filename']}"
            title = f"8-K: {f['company']} ({f['form_type']}) filed {f['date']}"
            published = f["published_dt"].isoformat() if f["published_dt"] else f["date"]
            to_insert.append({
                "link": filing_url,
                "title": title,
                "source": f"SEC-EDGAR/8-K",
                "published": published,
                "summary": f"SEC 8-K filing by {f['company']}. CIK {f['cik']}. "
                           f"Form type: {f['form_type']}.",
                "_relevance_score": 5.0,  # 8-Ks are always high-relevance financial events
            })

        inserted = store.insert_batch(to_insert) if to_insert else 0
        inserted_total += inserted
        done.add(qkey)

        if (i + 1) % 10 == 0:
            CHECKPOINT_PATH.write_text(json.dumps({"done": list(done)}))
            print(f"[edgar_bulk] {i+1}/{len(quarters)} quarters | "
                  f"+{inserted_total} filings | Q{year}Q{q}: {len(filings)} 8-Ks → {inserted} new")

    CHECKPOINT_PATH.write_text(json.dumps({"done": list(done)}))
    print(f"[edgar_bulk] DONE — {inserted_total} new 8-K filings inserted")


if __name__ == "__main__":
    y = int(sys.argv[1]) if len(sys.argv) > 1 else 1994
    run(y)
