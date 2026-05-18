"""BLS (Bureau of Labor Statistics) macro data collector.

Pulls key US economic series from the BLS public API v2 — no key required for
modest query volumes. Each series' latest data point becomes one article-like
dict so the daemon's ingest path scores/stores it like any other source. The
title encodes month-over-month change so downstream LLM/ML can react to
unexpected prints (CPI, payrolls, unemployment, PPI).

Series tracked:
  CUUR0000SA0   CPI-U All Items (headline inflation, NSA)
  CUUR0000SA0L1E CPI-U Less Food & Energy (core inflation, NSA)
  LNS14000000   Civilian Unemployment Rate (SA, %)
  CES0000000001 Total Nonfarm Payroll Employment (SA, thousands)
  WPSFD49207    PPI Final Demand (SA)
  LNS11300000   Labor Force Participation Rate

Standalone usage / smoke test:
    python3 collectors/bls_collector.py

To wire into the daemon, register a worker (interval=3600 is plenty — BLS
releases monthly):
    from collectors.bls_collector import collect_bls
    # _spawn("bls", collect_bls, interval=3600)
"""
from __future__ import annotations

import calendar
import time
from datetime import datetime, timezone

import requests

REQUEST_TIMEOUT = 10
MAX_ATTEMPTS = 2
API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/{series}"

SERIES = {
    "CUUR0000SA0":     ("CPI Headline",        "index",     "CPI-U All Items NSA"),
    "CUUR0000SA0L1E":  ("CPI Core",            "index",     "CPI-U ex Food & Energy NSA"),
    "LNS14000000":     ("Unemployment Rate",   "%",         "Civilian Unemployment Rate SA"),
    "CES0000000001":   ("Nonfarm Payrolls",    "k jobs",    "Total Nonfarm Employment SA"),
    "WPSFD49207":      ("PPI Final Demand",    "index",     "PPI Final Demand SA"),
    "LNS11300000":     ("Labor Force Part.",   "%",         "Labor Force Participation Rate"),
}

USER_AGENT = (
    "DigitalInternBot/1.0 (research; contact: sealai215j@gmail.com) "
    "python-requests"
)


def _fetch_series(series_id: str) -> list[dict]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    backoff = 2.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.get(API_URL.format(series=series_id),
                             headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            payload = r.json()
            if payload.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError(f"BLS status {payload.get('status')}")
            results = payload.get("Results", {}).get("series", [])
            if not results:
                return []
            return results[0].get("data", []) or []
        except Exception as e:
            if attempt >= MAX_ATTEMPTS:
                print(f"[bls] fetch failed for {series_id}: {e}")
                return []
            time.sleep(backoff)
            backoff *= 2
    return []


def _period_to_iso(year: str, period: str) -> str:
    """BLS uses M01..M12 for months; map to end-of-month UTC ISO timestamp."""
    try:
        y = int(year)
        if period.startswith("M") and period != "M13":
            m = int(period[1:])
            last_day = calendar.monthrange(y, m)[1]
            return datetime(y, m, last_day, 12, 0, 0,
                            tzinfo=timezone.utc).isoformat()
    except (ValueError, TypeError):
        pass
    return datetime.now(timezone.utc).isoformat()


def _format_change(curr: float, prev: float, unit: str) -> str:
    if prev == 0:
        return ""
    delta = curr - prev
    pct = (delta / prev) * 100
    sign = "+" if delta >= 0 else ""
    if unit == "%":
        return f"{sign}{delta:.2f}pp m/m"
    return f"{sign}{pct:.2f}% m/m"


def collect_bls() -> list[dict]:
    """Fetch latest BLS prints for tracked series."""
    print("[bls] Fetching macro series from BLS public API...")
    t0 = time.time()
    articles: list[dict] = []

    for series_id, (label, unit, full_name) in SERIES.items():
        data = _fetch_series(series_id)
        if len(data) < 2:
            continue
        curr, prev = data[0], data[1]
        try:
            curr_val = float(curr.get("value"))
            prev_val = float(prev.get("value"))
        except (TypeError, ValueError):
            continue

        change = _format_change(curr_val, prev_val, unit)
        period_name = curr.get("periodName", "")
        year = curr.get("year", "")
        title = (f"BLS {label}: {curr_val:g} {unit} ({period_name} {year})"
                 + (f" {change}" if change else ""))
        summary = (f"{full_name}. Latest: {curr_val:g} {unit} for "
                   f"{period_name} {year}. Previous: {prev_val:g}. "
                   f"Source: U.S. Bureau of Labor Statistics public API.")
        link = (f"https://data.bls.gov/timeseries/{series_id}")
        articles.append({
            "title": title[:200],
            "link": link,
            "summary": summary,
            "published": _period_to_iso(year, curr.get("period", "")),
            "source": "bls",
            "_bls_series": series_id,
            "_bls_value": curr_val,
            "_bls_prev": prev_val,
            "_bls_unit": unit,
        })

    elapsed = time.time() - t0
    print(f"[bls] Got {len(articles)} prints in {elapsed:.1f}s")
    return articles


if __name__ == "__main__":
    items = collect_bls()
    print(f"Total: {len(items)}")
    for a in items:
        print(f"  [{a['source']}] {a['title']}")
        print(f"    {a['link']}")
