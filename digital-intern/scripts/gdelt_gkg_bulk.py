"""GDELT GKG v2 bulk file importer — no rate limit, static file downloads.

Downloads GDELT Global Knowledge Graph (GKG) 15-minute interval ZIP files,
parses the CSV, filters rows for finance/economics/business themes or
company mentions, generates synthetic titles from org + source + date,
and inserts into ArticleStore.

No API rate limit applies — these are static files on GDELT's CDN.
We sample 1 file per day (midnight snapshot) to balance coverage and bandwidth.

File format: http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.gkg.csv.zip
GKG CSV columns (tab-delimited, no header row):
  [0]  GKGRECORDID
  [1]  V21DATE        (YYYYMMDDHHMMSS)
  [2]  V2SOURCECOLLECTIONIDENTIFIER (1=WEB 2=CITATION ONLY)
  [3]  V2SOURCECOMMONNAME  (site name like "Wall Street Journal")
  [4]  V2DOCUMENTIDENTIFIER (article URL)
  [5]  V2COUNTS
  [6]  V21COUNTS
  [7]  V2THEMES       (semicolon-separated themes like ECON_RECESSION;INDUSTRY_TECH)
  [8]  V2ENHANCEDTHEMES
  [9]  V2LOCATIONS
  [10] V21LOCATIONS
  [11] V2PERSONS
  [12] V21PERSONS
  [13] V2ORGANIZATIONS (semicolon-separated company names)
  [14] V21ORGANIZATIONS
  [15] V2TONE         (comma: tone,pos,neg,polarity,activity_ref,self_ref,wordcount)
  ...

Usage:
    python scripts/gdelt_gkg_bulk.py          # 2015 → now, 1 file/day
    python scripts/gdelt_gkg_bulk.py 2018     # custom start year
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from storage.article_store import ArticleStore, _get_db_path

GDELT_GKG_BASE = "http://data.gdeltproject.org/gdeltv2/"
MAX_WORKERS = 20          # parallel downloads — no API rate limit on these
REQUEST_TIMEOUT = 60
CHECKPOINT_PATH = BASE_DIR / "data" / "gkg_bulk_checkpoint.json"

# Finance-related theme prefixes to filter rows
FINANCE_THEMES = {
    "ECON_", "BUS_", "INDUSTRY_", "COMPANY_", "MARKET_",
    "INVESTOR_", "FINANCIAL_", "BANKING_", "STOCK_", "TRADE_",
    "UNEMPLOYMENT_", "DEBT_", "INFLATION_", "TAX_", "PROFIT_",
    "MERGER_", "ACQUISITION_", "IPO_", "DIVIDEND_", "EARNINGS_",
    "TARIFF_", "REGULATION_", "FED_", "CENTRAL_BANK_",
    "CRYPTOCURRENCY_", "BITCOIN_",
}

# Finance keywords in themes (partial match)
FINANCE_KEYWORDS = {
    "ECONOMY", "ECONOMIC", "RECESSION", "GROWTH", "GDP", "CPI", "INFLATION",
    "INTEREST_RATE", "FEDERAL_RESERVE", "STOCK_MARKET", "WALL_STREET",
    "NASDAQ", "NYSE", "S&P", "DOW_JONES", "HEDGE_FUND", "PRIVATE_EQUITY",
    "VENTURE_CAPITAL", "IPO", "MERGER", "ACQUISITION", "EARNINGS",
    "REVENUE", "PROFIT", "DEFICIT", "SURPLUS", "UNEMPLOYMENT", "JOBS",
    "TRADE", "TARIFF", "EXPORTS", "IMPORTS", "CURRENCY", "DOLLAR",
}

# Companies in our watchlist (for boosted relevance score)
WATCHLIST = {
    "NVIDIA", "AMD", "INTEL", "MICRON", "QUALCOMM", "BROADCOM",
    "APPLIED MATERIALS", "LAM RESEARCH", "KLA", "TSMC", "TAIWAN SEMICONDUCTOR",
    "APPLE", "MICROSOFT", "GOOGLE", "AMAZON", "META", "TESLA",
    "JPMORGAN", "GOLDMAN SACHS", "MORGAN STANLEY", "BANK OF AMERICA",
    "ORACLE", "SALESFORCE", "NVIDIA CORP", "ADVANCED MICRO",
}


def _is_finance(themes: str, orgs: str) -> tuple[bool, float]:
    """Return (is_finance, score) for a GKG row."""
    themes_up = themes.upper()
    orgs_up = orgs.upper()

    score = 0.0
    for prefix in FINANCE_THEMES:
        if prefix in themes_up:
            score += 1.0
            break
    for kw in FINANCE_KEYWORDS:
        if kw in themes_up:
            score += 0.5
            break
    # Boost if watchlist company is mentioned
    for co in WATCHLIST:
        if co in orgs_up:
            score += 2.0
            break

    return score > 0, min(score, 5.0)


def _parse_gkg_date(raw: str) -> str:
    raw = (raw or "").strip()
    if len(raw) >= 14:
        try:
            dt = datetime.strptime(raw[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            pass
    return ""


def _download_and_parse(url: str) -> list[dict]:
    """Download one GKG zip file and parse for finance rows."""
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        if r.status_code == 404:
            return []  # file doesn't exist for this timestamp
        if r.status_code != 200:
            return []
        content = r.content
    except Exception:
        return []

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_name = zf.namelist()[0]
            csv_bytes = zf.read(csv_name)
    except Exception:
        return []

    articles = []
    try:
        text = csv_bytes.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text), delimiter="\t")
        for row in reader:
            if len(row) < 16:
                continue
            url_art = row[4].strip()
            if not url_art or not url_art.startswith("http"):
                continue
            themes = row[7] if len(row) > 7 else ""
            orgs_raw = row[13] if len(row) > 13 else ""
            orgs = orgs_raw.replace(";", ", ")
            source_name = row[3].strip() if len(row) > 3 else ""
            date_raw = row[1].strip() if len(row) > 1 else ""
            tone_raw = row[15].strip() if len(row) > 15 else ""

            is_fin, score = _is_finance(themes, orgs_raw)
            if not is_fin:
                continue

            # Tone: first field is overall tone (-100 to 100)
            try:
                tone = float(tone_raw.split(",")[0])
                kw_score = min(max(score + tone / 20.0, 0.5), 5.0)
            except (ValueError, IndexError):
                kw_score = score

            published = _parse_gkg_date(date_raw)

            # Synthetic title from orgs + source + themes preview
            theme_tags = [t.split(",")[0] for t in themes.split(";") if t][:3]
            theme_str = ", ".join(theme_tags) if theme_tags else "financial news"
            org_str = orgs[:120] if orgs else source_name
            title = f"{source_name}: {org_str} [{theme_str}]"[:200]
            if not title.strip() or len(title.strip()) < 12:
                title = f"Financial news: {url_art[:80]}"

            articles.append({
                "link": url_art,
                "title": title,
                "source": f"gdelt_gkg/{source_name}" if source_name else "gdelt_gkg",
                "published": published,
                "summary": f"Themes: {themes[:300]}. Organizations: {orgs[:200]}.",
                "_relevance_score": float(kw_score),
            })
    except Exception:
        pass

    return articles


def _build_urls(start_year: int) -> list[tuple[str, str]]:
    """One GKG URL per day from start_year to today (midnight UTC snapshot)."""
    today = date.today()
    start = date(start_year, 1, 1)
    # GDELT GKG v2 started Feb 19 2015
    if start < date(2015, 2, 19):
        start = date(2015, 2, 19)

    urls = []
    d = start
    while d <= today:
        dt_str = d.strftime("%Y%m%d") + "000000"
        url = f"{GDELT_GKG_BASE}{dt_str}.gkg.csv.zip"
        urls.append((url, dt_str))
        d += timedelta(days=1)
    return urls


def run(start_year: int = 2015):
    import json
    done: set[str] = set()
    if CHECKPOINT_PATH.exists():
        try:
            done = set(json.loads(CHECKPOINT_PATH.read_text()).get("done", []))
        except Exception:
            pass

    store = ArticleStore()
    all_urls = _build_urls(start_year)
    pending = [(u, k) for u, k in all_urls if k not in done]

    total = len(pending)
    inserted_total = 0
    failed = 0
    start_time = time.time()
    print(f"[gkg_bulk] {total} daily GKG files to download | "
          f"{len(done)} already done | writing to {_get_db_path()}")

    lock = threading.Lock()

    def worker(item):
        nonlocal inserted_total, failed
        url, key = item
        articles = _download_and_parse(url)
        inserted = store.insert_batch(articles) if articles else 0

        with lock:
            inserted_total += inserted
            if url == "404":
                failed += 1
            done.add(key)

        return inserted

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(worker, item): item for item in pending}
        completed = 0
        try:
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    pass
                completed += 1
                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed * 60
                    eta_min = (total - completed) / (completed / elapsed) / 60
                    print(
                        f"[gkg_bulk] {completed}/{total} files | "
                        f"+{inserted_total:,} articles | "
                        f"{rate:.0f} files/min | "
                        f"ETA {eta_min:.0f}min"
                    )
                    if completed % 500 == 0:
                        # Atomic write — write_text() truncates first, so an
                        # OOM-kill mid-write would empty the checkpoint and
                        # restart the whole multi-hour sweep from scratch.
                        _tmp = CHECKPOINT_PATH.with_suffix(".tmp")
                        _tmp.write_text(json.dumps({"done": list(done)}))
                        os.replace(_tmp, CHECKPOINT_PATH)
        except KeyboardInterrupt:
            print("\n[gkg_bulk] interrupted — saving checkpoint")

    _tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    _tmp.write_text(json.dumps({"done": list(done)}))
    os.replace(_tmp, CHECKPOINT_PATH)
    elapsed = time.time() - start_time
    print(f"\n[gkg_bulk] DONE — {inserted_total:,} new articles "
          f"from {total} files in {elapsed/60:.1f} min")


if __name__ == "__main__":
    y = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    run(y)
