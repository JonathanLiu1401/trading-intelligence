"""Finnhub company-news historical sweep.

Uses /api/v1/company-news?symbol=X&from=YYYY-MM-DD&to=YYYY-MM-DD to pull
years of headline-level news per ticker. Finnhub rate limit: 60 req/min free
tier; we stay at 50 req/min (1 req/1.2s) to avoid 429s.

Sweeps the full watchlist + portfolio universe from 2018-01-01 to today in
90-day windows (Finnhub returns more complete results for shorter windows).

Usage:
    python scripts/finnhub_historical_news.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from storage.article_store import ArticleStore, _get_db_path

# Load env
for line in (BASE_DIR / ".env").read_text().splitlines():
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ.get("FINNHUB_API_KEY", "")
BASE_URL = "https://finnhub.io/api/v1/company-news"
WINDOW_DAYS = 90       # Finnhub returns fuller results with shorter windows
SLEEP_PER_REQ = 1.2    # 50 req/min, under 60 free-tier limit
START_DATE = date(2018, 1, 1)
CHECKPOINT_PATH = BASE_DIR / "data" / "finnhub_sweep_checkpoint.json"


def _load_tickers() -> list[str]:
    tickers = set()
    for cfg in ["config/portfolio.json", "config/watchlist.json"]:
        p = BASE_DIR / cfg
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if "positions" in data:
            for pos in data["positions"]:
                t = (pos.get("ticker") or "").strip().upper()
                if t and "." not in t:
                    tickers.add(t)
        if "sector_watchlist" in data:
            for t in data["sector_watchlist"]:
                tickers.add(t.strip().upper())
        for key in ("memory_core", "semis_equipment", "broader_semis", "portfolio"):
            for t in data.get(key, []):
                tickers.add(t.strip().upper())
    # Add extended semiconductor / macro universe
    tickers.update([
        "NVDA", "AMD", "INTC", "MU", "TSMC", "AVGO", "QCOM", "AMAT", "LRCX", "KLAC",
        "ASML", "TXN", "ADI", "MRVL", "ON", "SWKS", "SLAB", "WOLF", "CREE",
        "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
        "JPM", "BAC", "GS", "MS", "WFC", "C",
        "XOM", "CVX", "OXY", "SLB", "HAL",
        "BTC-USD", "ETH-USD",  # these will 404 but fail silently
    ])
    return sorted(tickers)


def _load_checkpoint() -> set[str]:
    if CHECKPOINT_PATH.exists():
        try:
            return set(json.loads(CHECKPOINT_PATH.read_text()).get("done", []))
        except Exception:
            pass
    return set()


def _save_checkpoint(done: set[str]):
    # Atomic write: write_text() truncates first, so an OOM-kill (this box
    # peaks ~8G / 6G swap) between truncate and flush would leave an empty
    # checkpoint and silently restart the whole multi-hour sweep.
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"done": list(done)}))
    os.replace(tmp, CHECKPOINT_PATH)


def run():
    if not API_KEY:
        print("[finnhub_sweep] FINNHUB_API_KEY not set — skipping")
        return

    store = ArticleStore()
    tickers = _load_tickers()
    done = _load_checkpoint()

    # Build (ticker, from, to) windows
    tasks = []
    today = date.today()
    for ticker in tickers:
        d = START_DATE
        while d < today:
            end = min(d + timedelta(days=WINDOW_DAYS - 1), today)
            task_key = f"{ticker}|{d}|{end}"
            if task_key not in done:
                tasks.append((ticker, d, end, task_key))
            d = end + timedelta(days=1)

    total = len(tasks)
    inserted_total = 0
    print(f"[finnhub_sweep] {total} windows | {len(tickers)} tickers | "
          f"writing to {_get_db_path()}")

    for i, (ticker, frm, to, key) in enumerate(tasks):
        try:
            r = requests.get(
                BASE_URL,
                params={"symbol": ticker, "from": str(frm), "to": str(to), "token": API_KEY},
                timeout=15,
            )
            if r.status_code == 429:
                print(f"[finnhub_sweep] rate limited — sleeping 30s")
                time.sleep(30)
                continue
            if r.status_code != 200:
                done.add(key)
                continue
            news = r.json() or []
            to_insert = []
            for item in news:
                url = item.get("url") or ""
                headline = item.get("headline") or ""
                if not url or not headline:
                    continue
                ts = item.get("datetime")
                published = ""
                if ts:
                    from datetime import datetime, timezone
                    try:
                        published = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                    except Exception:
                        pass
                to_insert.append({
                    "link": url,
                    "title": headline,
                    "source": f"Finnhub/{item.get('source', ticker)}",
                    "published": published,
                    "summary": item.get("summary") or "",
                    "_relevance_score": 4.0,  # financial news, high relevance
                })
            inserted = store.insert_batch(to_insert) if to_insert else 0
            inserted_total += inserted
            done.add(key)

            if (i + 1) % 50 == 0:
                _save_checkpoint(done)
                print(f"[finnhub_sweep] {i+1}/{total} | +{inserted_total} articles")

        except Exception as e:
            print(f"[finnhub_sweep] {ticker} {frm}→{to}: {e}")
            done.add(key)  # skip broken windows to avoid infinite retry

        time.sleep(SLEEP_PER_REQ)

    _save_checkpoint(done)
    print(f"[finnhub_sweep] DONE — {inserted_total} new articles")


if __name__ == "__main__":
    run()
