"""Pre-warm historical data for a backtest window.

Called before BacktestEngine is created for a new window. Fetches:
  1. yfinance prices         — already handled by PriceCache; not duplicated here
  2. GDELT news              — for windows overlapping 2015-02-19 onwards
  3. SEC EDGAR 8-K/10-Q/10-K — for all windows back to 1996
  4. Claude relevance labels — applied to historical articles in batches

Data is cached to disk so subsequent backtests over the same window are instant.
All collectors are best-effort: a failing fetch logs and continues; an empty
window falls back to quant-only signals (already supported by BacktestEngine).

The functions here run inline by default. Heavy GDELT / SEC scans should be
dispatched with `background=True` because a multi-year window touching every
date+keyword would easily exceed any per-cycle time budget.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from .backtest import (
    CACHE_DIR,
    GDELT_CACHE,
    GDELT_RATE_LIMIT_S,
    KEYWORD_GROUPS,
    GDELTFetcher,
)

# GDELT 2.0 article-search API has documented coverage from 2015-02-19.
# Earlier windows fall back to SEC + price/quant signals only.
GDELT_COVERAGE_START = date(2015, 2, 19)

# SEC EDGAR full-text search covers ~1994+; we use 1996 as the floor matching
# _pick_window's EARLIEST_WINDOW_START.
SEC_COVERAGE_START = date(1996, 1, 1)

# Cache directories — co-located with backtest caches so the engine can read
# them with no extra wiring.
SEC_CACHE = CACHE_DIR / "sec_edgar"
HISTORICAL_LABEL_CACHE = CACHE_DIR / "historical_labels"

# Tickers we bother fetching SEC filings for. A 30-year window × 120 tickers ×
# many filings/quarter is unbounded — restrict to the names where filings most
# usefully feed the signal pipeline.
SEC_TICKERS = [
    "NVDA", "AMD", "INTC", "MU", "TSM", "AAPL", "MSFT", "META", "GOOGL",
    "AMZN", "TSLA", "JPM", "BAC", "GS", "XOM", "LLY", "UNH",
]

# Subject prefix Claude sees per batch when labeling. Tuned for short response.
_LABEL_SYSTEM = (
    "You score news headlines for a stock-trading model. For each headline, "
    "return ONE LINE in the format: INDEX|RELEVANCE|URGENCY where RELEVANCE is "
    "0-10 (10=major market-moving) and URGENCY is 0 or 1 (1=immediate action). "
    "Be concise. No prose, no markdown."
)


# ────────────────────────── public entry point ──────────────────────────

def prewarm_window(start: date, end: date, tickers: list[str] | None = None,
                   background: bool = False) -> None:
    """Pre-warm caches for a backtest window.

    `background=True` dispatches all heavy fetches to daemon threads and returns
    immediately so the caller can proceed with backtesting. Each collector
    handles its own caching and rate limiting; calling `prewarm_window` again
    for the same dates is cheap (cache hits).
    """
    SEC_CACHE.mkdir(parents=True, exist_ok=True)
    HISTORICAL_LABEL_CACHE.mkdir(parents=True, exist_ok=True)

    if background:
        threading.Thread(
            target=_run_all_collectors,
            args=(start, end, tickers),
            daemon=True,
            name=f"prewarm-{start.isoformat()}",
        ).start()
        return

    _run_all_collectors(start, end, tickers)


def _run_all_collectors(start: date, end: date,
                        tickers: list[str] | None) -> None:
    try:
        if end >= GDELT_COVERAGE_START:
            warm_gdelt_weekly(max(start, GDELT_COVERAGE_START), end)
    except Exception as exc:
        print(f"[prewarm] gdelt weekly warm failed: {exc}")
    try:
        if end >= SEC_COVERAGE_START:
            target = tickers or SEC_TICKERS
            fetch_sec_historical(target, start, end)
    except Exception as exc:
        print(f"[prewarm] sec historical fetch failed: {exc}")


# ────────────────────────── GDELT weekly pre-warm ──────────────────────────

def warm_gdelt_weekly(start: date, end: date) -> int:
    """Fetch one GDELT query per (week-start, keyword) pair across the window.

    Daily resolution would be ~ days × keywords = 1825 × 20 ≈ 36500 calls for a
    5yr window at >5s/call → ~50 hours, prohibitively long. Weekly resolution
    drops that to ~260 × 20 = 5200 calls / ~8 hours, still long but tractable
    in the background. The per-day lookup in BacktestEngine._fetch_signals will
    fall back to the live (uncached) path on a miss — pre-warming is purely
    a performance optimisation.

    Each cached file is written under the same key the engine reads from,
    keyed by the Monday of the week so consumers find it by either Monday
    itself or by their per-day lookup falling through to the weekly entry.

    Returns the number of (week, keyword) pairs newly fetched.
    """
    GDELT_CACHE.mkdir(parents=True, exist_ok=True)
    fetcher = GDELTFetcher()

    # Walk by week starting at the Monday of `start`.
    cur = start - timedelta(days=start.weekday())
    weeks: list[date] = []
    while cur <= end:
        weeks.append(cur)
        cur += timedelta(days=7)

    pairs = [(w, kw) for w in weeks for kw in KEYWORD_GROUPS]
    uncached = [(w, kw) for w, kw in pairs
                if not fetcher._cache_key(w, kw).exists()]
    if not uncached:
        print(f"[gdelt_weekly] all {len(pairs)} (week, kw) pairs cached")
        return 0

    print(f"[gdelt_weekly] warming {len(uncached)}/{len(pairs)} pairs "
          f"({start} → {end}, ~{GDELT_RATE_LIMIT_S * len(uncached) / 60:.1f}min budget)")

    n_fetched = 0
    for week, kw in uncached:
        try:
            fetcher.fetch(week, kw)
            n_fetched += 1
        except Exception as exc:
            print(f"[gdelt_weekly] {week} {kw[:30]!r} failed: {exc}")
        # The fetcher itself rate-limits via _last_request_ts; no extra sleep
        # here, but yield to other threads every 25 calls.
        if n_fetched % 25 == 0 and n_fetched:
            print(f"[gdelt_weekly] progress {n_fetched}/{len(uncached)}")
    print(f"[gdelt_weekly] done — {n_fetched} new entries")
    return n_fetched


# ────────────────────────── SEC EDGAR historical ──────────────────────────

_SEC_USER_AGENT = "paper-trader research collector contact@example.com"
_SEC_RATE_S = 0.15  # SEC EDGAR rate limit is 10 req/s; stay well under


def _sec_cache_path(ticker: str, start: date, end: date) -> Path:
    return SEC_CACHE / f"{ticker}_{start.isoformat()}_{end.isoformat()}.json"


def fetch_sec_historical(tickers: list[str], start: date, end: date) -> list[dict]:
    """Fetch 8-K / 10-Q / 10-K filings for each ticker in [start, end].

    Uses the EDGAR full-text search endpoint. Results are cached per
    (ticker, start, end) so subsequent calls for the same window are free.
    Returns a flat list of article-like dicts:
      {title, url, published, source: "SEC/{form}/{ticker}", full_text}

    Failures degrade gracefully — empty list is returned for tickers with
    no filings or network errors, never raised.
    """
    SEC_CACHE.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    forms = ["8-K", "10-Q", "10-K"]
    for ticker in tickers:
        path = _sec_cache_path(ticker, start, end)
        if path.exists():
            try:
                out.extend(json.loads(path.read_text()))
                continue
            except Exception:
                pass

        items: list[dict] = []
        for form in forms:
            try:
                items.extend(_sec_search(ticker, form, start, end))
            except Exception as exc:
                print(f"[sec] {ticker} {form} fetch failed: {exc}")
            time.sleep(_SEC_RATE_S)

        try:
            path.write_text(json.dumps(items))
        except Exception:
            pass
        out.extend(items)
        # Modest per-ticker pause so a wide tickers list doesn't burst.
        time.sleep(_SEC_RATE_S)

    return out


def _sec_search(ticker: str, form: str, start: date, end: date) -> list[dict]:
    """One full-text search query against EDGAR. Returns filing metadata."""
    params = {
        "q": f'"{ticker}"',
        "dateRange": "custom",
        "startdt": start.isoformat(),
        "enddt": end.isoformat(),
        "forms": form,
    }
    url = "https://efts.sec.gov/LATEST/search-index?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _SEC_USER_AGENT})
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read()
    try:
        data = json.loads(body)
    except Exception:
        return []
    hits = (data.get("hits") or {}).get("hits") or []
    results: list[dict] = []
    for h in hits:
        src = h.get("_source") or {}
        display = src.get("display_names") or [ticker]
        filing_date = src.get("file_date") or ""
        accession = h.get("_id") or src.get("adsh") or ""
        title = f"{form} filing — {display[0]} on {filing_date}"
        results.append({
            "title": title,
            "url": (f"https://www.sec.gov/Archives/edgar/data/{src.get('ciks',[''])[0]}/"
                    f"{accession.replace('-','')}/" if accession else ""),
            "published": filing_date,
            "source": f"SEC/{form}/{ticker}",
            "full_text": title,
        })
    return results


# ────────────────────────── Claude relevance labels ──────────────────────────

def label_historical_articles(articles: list[dict],
                              start: date, end: date,
                              batch_size: int = 10) -> list[dict]:
    """Use Claude Sonnet to assign (relevance, urgency) to historical articles.

    Articles already carrying `ai_score` keep it. Results are cached per
    (start, end) window on disk so re-labeling is free across cycles.
    `batch_size` headlines per Claude call keeps prompts cheap; default 10.
    """
    if not articles:
        return articles
    cache_path = HISTORICAL_LABEL_CACHE / f"{start.isoformat()}_{end.isoformat()}.json"
    labels: dict[str, tuple[float, int]] = {}
    if cache_path.exists():
        try:
            labels = {k: tuple(v) for k, v in json.loads(cache_path.read_text()).items()}
        except Exception:
            labels = {}

    todo = [a for a in articles
            if not a.get("ai_score")
            and _label_key(a) not in labels]
    if not todo:
        return _apply_labels(articles, labels)

    if not shutil.which("claude"):
        # No Claude available — just return articles unchanged so the engine
        # falls back to its keyword heuristic.
        return _apply_labels(articles, labels)

    for chunk_start in range(0, len(todo), batch_size):
        chunk = todo[chunk_start:chunk_start + batch_size]
        try:
            parsed = _label_batch(chunk)
            for art, (rel, urg) in zip(chunk, parsed):
                labels[_label_key(art)] = (rel, urg)
            # Persist incrementally so a Ctrl-C doesn't lose work.
            cache_path.write_text(
                json.dumps({k: list(v) for k, v in labels.items()})
            )
        except Exception as exc:
            print(f"[label] batch {chunk_start//batch_size} failed: {exc}")
            # Continue with the next batch; partial labels are fine.

    return _apply_labels(articles, labels)


def _label_key(article: dict) -> str:
    """Stable per-article cache key (title is stable; url may be empty)."""
    title = article.get("title") or ""
    src = article.get("source") or ""
    return hashlib.sha1(f"{title}|{src}".encode()).hexdigest()[:16]


def _label_batch(articles: list[dict]) -> list[tuple[float, int]]:
    """Call Claude on a batch; return [(relevance, urgency), …] aligned with input."""
    lines = "\n".join(f"{i}: {a.get('title','')[:200]}"
                      for i, a in enumerate(articles))
    prompt = f"{_LABEL_SYSTEM}\n\n{lines}"
    r = subprocess.run(
        ["claude", "--model", "claude-sonnet-4-6", "--print",
         "--permission-mode", "bypassPermissions"],
        input=prompt, capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(f"claude rc={r.returncode}: {r.stderr.strip()[:200]}")
    return _parse_labels(r.stdout, expected=len(articles))


def _parse_labels(raw: str, expected: int) -> list[tuple[float, int]]:
    """Parse 'INDEX|RELEVANCE|URGENCY' lines into aligned (rel, urg) tuples.

    Missing or malformed lines fall back to (0.0, 0) so the array length is
    always `expected` — keeps the caller's zip-pairing safe.
    """
    out: list[tuple[float, int]] = [(0.0, 0)] * expected
    for line in raw.splitlines():
        m = re.match(r"\s*(\d+)\s*[|,]\s*([0-9.]+)\s*[|,]\s*([01])\s*$", line)
        if not m:
            continue
        idx, rel, urg = int(m.group(1)), float(m.group(2)), int(m.group(3))
        if 0 <= idx < expected:
            out[idx] = (max(0.0, min(10.0, rel)), 1 if urg else 0)
    return out


def _apply_labels(articles: list[dict],
                  labels: dict[str, tuple[float, int]]) -> list[dict]:
    """Return a new list with ai_score/urgency populated from `labels` where missing."""
    out: list[dict] = []
    for a in articles:
        key = _label_key(a)
        if key in labels and not a.get("ai_score"):
            rel, urg = labels[key]
            a = dict(a)
            a["ai_score"] = rel
            a["urgency"] = urg
        out.append(a)
    return out
