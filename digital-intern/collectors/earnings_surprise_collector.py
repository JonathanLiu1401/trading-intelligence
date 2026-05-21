"""Earnings surprise collector — tracks EPS beats/misses for watchlist tickers.

Fetches recent earnings history via yfinance and emits synthetic article rows
when a ticker reports earnings with a meaningful beat or miss vs analyst estimates.

Dedup key: ticker + fiscal quarter end date so each surprise emits at most once.

Standalone usage / smoke test:
    python3 collectors/earnings_surprise_collector.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
CURSOR_PATH = BASE_DIR / "data" / "earnings_surprise_cursor.json"

SURPRISE_THRESHOLD = 0.05   # 5% beat/miss vs estimate to qualify
LOOKBACK_DAYS = 30           # only emit surprises from past N days
MAX_TICKERS_PER_PASS = 20   # yfinance rate-limit guard
SOURCE_NAME = "earnings_surprise"

log = logging.getLogger("earnings_surprise_collector")


def _load_tickers() -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()

    def _add(t: str):
        u = (t or "").strip().upper()
        # skip index/crypto/forex symbols
        if u and not u.startswith("^") and "=" not in u and u not in seen:
            seen.add(u)
            tickers.append(u)

    try:
        with open(WATCHLIST_PATH) as f:
            wl = json.load(f)
        for key in ("memory_core", "semis_equipment", "broader_semis",
                    "memory_options_focus", "portfolio"):
            for t in wl.get(key, []):
                _add(t)
    except Exception as e:
        log.warning(f"[earnings_surprise] watchlist load error: {e}")

    try:
        with open(PORTFOLIO_PATH) as f:
            pf = json.load(f)
        for pos in pf.get("positions", []):
            _add(pos.get("ticker", ""))
        for t in pf.get("sector_watchlist", []):
            _add(t)
    except Exception as e:
        log.warning(f"[earnings_surprise] portfolio load error: {e}")

    return tickers


def _load_cursor() -> set[str]:
    """Return set of already-emitted article IDs."""
    if CURSOR_PATH.exists():
        try:
            with open(CURSOR_PATH) as f:
                data = json.load(f)
                return set(data.get("emitted", []))
        except Exception:
            pass
    return set()


def _save_cursor(emitted: set[str]) -> None:
    CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CURSOR_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"emitted": sorted(emitted)}, f, indent=2)
    tmp.replace(CURSOR_PATH)


def _article_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _surprise_label(pct: float) -> str:
    if pct >= 0.20:
        return "HUGE BEAT"
    elif pct >= 0.10:
        return "strong beat"
    elif pct >= 0.05:
        return "beat"
    elif pct <= -0.20:
        return "HUGE MISS"
    elif pct <= -0.10:
        return "strong miss"
    else:
        return "miss"


def _fetch_surprise(ticker: str, cutoff: datetime) -> list[dict]:
    """Return list of article dicts for recent earnings surprises."""
    articles = []
    try:
        t = yf.Ticker(ticker)
        hist = t.earnings_history
        if hist is None or hist.empty:
            return []

        for _, row in hist.iterrows():
            quarter_date = row.name  # DatetimeIndex
            if hasattr(quarter_date, "to_pydatetime"):
                quarter_date = quarter_date.to_pydatetime()
            if quarter_date.tzinfo is None:
                quarter_date = quarter_date.replace(tzinfo=timezone.utc)
            if quarter_date < cutoff:
                continue

            try:
                eps_actual = float(row.get("epsActual") or 0)
                eps_estimate = float(row.get("epsEstimate") or 0)
            except (TypeError, ValueError):
                continue

            # prefer pre-computed surprisePercent when available
            if "surprisePercent" in row.index and row.get("surprisePercent") is not None:
                try:
                    surprise_pct = float(row["surprisePercent"])
                except (TypeError, ValueError):
                    surprise_pct = None
            else:
                surprise_pct = None

            if surprise_pct is None:
                if abs(eps_estimate) < 0.01:
                    continue
                surprise_pct = (eps_actual - eps_estimate) / abs(eps_estimate)

            if abs(surprise_pct) < SURPRISE_THRESHOLD:
                continue

            surprise_pct = (eps_actual - eps_estimate) / abs(eps_estimate)
            if abs(surprise_pct) < SURPRISE_THRESHOLD:
                continue

            label = _surprise_label(surprise_pct)
            pct_str = f"{'+' if surprise_pct >= 0 else ''}{surprise_pct*100:.1f}%"
            date_str = quarter_date.strftime("%Y-%m-%d")

            title = (
                f"{ticker} EPS {label}: reported ${eps_actual:.2f} vs "
                f"${eps_estimate:.2f} est ({pct_str}) — Q ending {date_str}"
            )
            summary = (
                f"{ticker} earnings surprise: actual EPS ${eps_actual:.2f} vs "
                f"estimate ${eps_estimate:.2f} ({pct_str} {label}). "
                f"Fiscal quarter ended {date_str}."
            )
            link = f"https://finance.yahoo.com/quote/{ticker}/financials/"
            dedup_key = f"earnings_surprise|{ticker}|{date_str}"

            articles.append({
                "id": _article_id(dedup_key),
                "link": link,
                "title": title,
                "summary": summary,
                "source": SOURCE_NAME,
                "published": quarter_date.isoformat(),
                "_dedup_key": dedup_key,
                "_surprise_pct": round(surprise_pct * 100, 1),
                "_ticker": ticker,
            })

    except Exception as e:
        log.warning(f"[earnings_surprise] {ticker} fetch error: {e}")

    return articles


def collect_earnings_surprises() -> list[dict]:
    """Main entry point — returns net-new surprise articles."""
    tickers = _load_tickers()
    emitted = _load_cursor()
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    new_articles: list[dict] = []

    # round-robin: process up to MAX_TICKERS_PER_PASS per call
    cursor_file = CURSOR_PATH.with_suffix(".offset.json")
    offset = 0
    if cursor_file.exists():
        try:
            with open(cursor_file) as f:
                offset = json.load(f).get("offset", 0)
        except Exception:
            offset = 0

    batch = tickers[offset: offset + MAX_TICKERS_PER_PASS]
    next_offset = (offset + MAX_TICKERS_PER_PASS) % max(len(tickers), 1)
    with open(cursor_file, "w") as f:
        json.dump({"offset": next_offset}, f)

    for ticker in batch:
        for art in _fetch_surprise(ticker, cutoff):
            key = art["_dedup_key"]
            if key not in emitted:
                new_articles.append(art)
                emitted.add(key)
        time.sleep(0.2)  # be polite to yfinance

    if new_articles:
        _save_cursor(emitted)
        log.info(
            f"[earnings_surprise] +{len(new_articles)} surprises "
            f"(scanned {len(batch)} tickers)"
        )

    return new_articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(f"Scanning earnings surprises (last {LOOKBACK_DAYS} days)...")
    articles = collect_earnings_surprises()
    if articles:
        for a in articles:
            pct = a.get("_surprise_pct", "?")
            print(f"  [{a['_ticker']}] {pct:+.1f}%  {a['title']}")
        print(f"\nTotal: {len(articles)} new surprise(s)")
    else:
        print("No new surprises found in this batch.")
