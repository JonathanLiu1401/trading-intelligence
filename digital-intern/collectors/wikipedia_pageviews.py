"""Wikipedia pageviews collector — surfaces z-score spikes on tracked
companies' Wikipedia pages.

A 2.5σ surge on $NVDA's Wikipedia article reliably tracks (and often
precedes by hours) breaking news on the name — retail interest leaks into
encyclopedia traffic before it shows up in headlines. Free public REST API,
no key.

Endpoint:
    https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/
        en.wikipedia.org/all-access/all-agents/{article}/daily/{start}/{end}

Returns the canonical ``{title, link, summary, published, source}`` dicts
so the daemon's ``_ingest()`` path (or the ``__main__`` smoke test here)
hands them to ``ArticleStore.insert_batch`` — same shape as every other
collector.

Dedup via ``data/seen_articles.db`` keyed by ``ticker|YYYY-MM-DD`` so a
later revision of the same pageview count can never re-emit the same day.

Standalone usage:
    python3 collectors/wikipedia_pageviews.py

Wire into the daemon as a worker (interval ~3600s is plenty — the API
itself only refreshes once per UTC day):
    from collectors.wikipedia_pageviews import collect_wikipedia_pageviews
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

USER_AGENT = "Digital-Intern-Daemon (sealai215j@gmail.com)"
API_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia.org/all-access/all-agents/{article}/daily/{start}/{end}"
)
LOOKBACK_DAYS = 7
SPIKE_Z = 2.5
HTTP_TIMEOUT = 10
SOURCE = "wikipedia/pageviews"

# Map our tracked tickers to canonical Wikipedia article slugs. ETFs without
# a dedicated wiki entity (SMH/SOXX/NVDL/MUU/LNOK) are intentionally omitted —
# pageview signal on a leveraged-ETF wrapper page is too thin to surface
# anything useful, and the underlying (NVDA/MU/LMT) is already covered here.
TICKER_TO_WIKI: dict[str, str] = {
    "NVDA": "Nvidia",
    "AMD":  "Advanced_Micro_Devices",
    "MU":   "Micron_Technology",
    "MUU":  "Micron_Technology",   # leveraged wrapper → same underlying
    "NVDL": "Nvidia",
    "LITE": "Lumentum",
    "LRCX": "Lam_Research",
    "AMAT": "Applied_Materials",
    "KLAC": "KLA_Corporation",
    "WDC":  "Western_Digital",
    "STX":  "Seagate_Technology",
    "AXTI": "AXT_(company)",
    "INTC": "Intel",
    "QCOM": "Qualcomm",
    "TSM":  "TSMC",
    "ASML": "ASML_Holding",
    "AVGO": "Broadcom",
    "DRAM": "Dynamic_random-access_memory",
}


def _ensure_db() -> sqlite3.Connection:
    """Same hardened seen_articles.db pattern as rss_collector / fred_collector."""
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


def _seen_id(ticker: str, date_ymd: str) -> str:
    return hashlib.sha256(
        f"wiki_pv:{ticker}:{date_ymd}".encode("utf-8")
    ).hexdigest()


def _portfolio_tickers() -> set[str]:
    """Tickers we have a wiki mapping for AND that appear somewhere in
    portfolio.json (positions, sector_watchlist, or option underlyings).
    Missing/unreadable portfolio file falls back to the full map — being
    over-broad on signal sources is better than going silently dark."""
    try:
        cfg = json.loads(PORTFOLIO_PATH.read_text())
    except Exception:
        return set(TICKER_TO_WIKI)
    referenced: set[str] = set()
    for p in cfg.get("positions", []) or []:
        t = (p.get("ticker") or "").upper()
        if t:
            referenced.add(t)
    for o in cfg.get("options", []) or []:
        u = (o.get("underlying") or "").upper()
        if u:
            referenced.add(u)
    for t in cfg.get("sector_watchlist", []) or []:
        if isinstance(t, str):
            referenced.add(t.upper())
    mapped = {t for t in TICKER_TO_WIKI if t in referenced}
    return mapped or set(TICKER_TO_WIKI)


def _fetch_views(article: str, start_ymd: str, end_ymd: str
                 ) -> list[tuple[str, int]]:
    """Returns [(YYYYMMDD, views), ...] oldest→newest. Empty on any
    HTTP / parse error so a single broken slug never poisons the batch.

    Article slug is URL-encoded — Wikimedia REST returns 400 on raw
    parentheses (e.g. AXT_(company)) without quoting."""
    url = API_URL.format(
        article=quote(article, safe=""),
        start=start_ymd,
        end=end_ymd,
    )
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT},
                         timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        return []
    items = data.get("items") or []
    out: list[tuple[str, int]] = []
    for it in items:
        # API timestamps are YYYYMMDDHH; we only ever ask for daily so HH=00.
        ts = (it.get("timestamp") or "")[:8]
        try:
            v = int(it.get("views"))
        except (TypeError, ValueError):
            continue
        if ts and v >= 0:
            out.append((ts, v))
    out.sort()
    return out


def _z_score(today: int, prior: list[int]) -> float | None:
    """Sample z-score of ``today`` vs ``prior``. ``None`` when the prior
    window has <3 observations or zero variance (no signal to report).
    Sample (not population) stdev matches statistics.stdev so callers
    can verify against numpy.std(ddof=1)."""
    if len(prior) < 3:
        return None
    mu = statistics.fmean(prior)
    try:
        sigma = statistics.stdev(prior)
    except statistics.StatisticsError:
        return None
    if sigma == 0:
        return None
    return (today - mu) / sigma


def build_spike_articles(rows: list[tuple[str, int]], ticker: str,
                          wiki: str, threshold: float = SPIKE_Z
                          ) -> list[dict]:
    """Pure builder: chronological [(YYYYMMDD, views)] → list of article
    dicts for every day whose z-score vs the preceding (up to
    LOOKBACK_DAYS-1) days exceeds ``threshold`` in absolute value.

    Pure-function: no I/O, no dedup state. Same `rows` → same output.
    All edge cases (insufficient priors, zero variance) covered in
    tests/test_wikipedia_pageviews.py.
    """
    if len(rows) < 4:
        return []
    out: list[dict] = []
    for i in range(3, len(rows)):
        date_ymd, today_views = rows[i]
        prior_window = [v for _, v in rows[max(0, i - (LOOKBACK_DAYS - 1)):i]]
        z = _z_score(today_views, prior_window)
        if z is None or abs(z) < threshold:
            continue
        baseline = round(statistics.fmean(prior_window))
        ratio = today_views / baseline if baseline else 0.0
        direction = "SURGE" if z >= 0 else "DROP"
        iso_date = f"{date_ymd[:4]}-{date_ymd[4:6]}-{date_ymd[6:8]}"
        link = f"https://en.wikipedia.org/wiki/{wiki}"
        title = (
            f"Wiki pageview {direction} {ticker} ({wiki}): "
            f"{today_views:,} vs {baseline:,} baseline "
            f"(z={z:+.1f}, x{ratio:.1f}) {iso_date}"
        )
        summary = (
            f"Wikipedia pageviews for {wiki} on {iso_date}: "
            f"{today_views:,} (prior {LOOKBACK_DAYS - 1}-day mean: "
            f"{baseline:,}). Z-score = {z:+.2f} "
            f"(|z|>={threshold} threshold). Pageview spikes on company "
            f"pages frequently coincide with or precede breaking news on "
            f"the underlying name."
        )
        out.append({
            "title": title[:200],
            "link": link,
            "summary": summary,
            "published": iso_date,
            "source": SOURCE,
            "_ticker": ticker,
            "_z": round(z, 2),
            "_views": today_views,
            "_baseline": baseline,
        })
    return out


def collect_wikipedia_pageviews() -> list[dict]:
    """Collect deduplicated pageview-spike articles for portfolio-referenced
    tickers. Standard collector return shape."""
    conn = _ensure_db()
    tickers = sorted(_portfolio_tickers())
    if not tickers:
        conn.close()
        return []
    # Wikimedia daily counts publish on a ~24h delay, so end at yesterday UTC.
    now = datetime.now(timezone.utc)
    end_dt = now - timedelta(days=1)
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS - 1)
    end_ymd = end_dt.strftime("%Y%m%d")
    start_ymd = start_dt.strftime("%Y%m%d")

    items: list[dict] = []
    for ticker in tickers:
        wiki = TICKER_TO_WIKI.get(ticker)
        if not wiki:
            continue
        rows = _fetch_views(wiki, start_ymd=start_ymd, end_ymd=end_ymd)
        if not rows:
            continue
        for art in build_spike_articles(rows, ticker, wiki):
            sid = _seen_id(ticker, art["published"])
            if conn.execute(
                "SELECT 1 FROM seen_articles WHERE id=?", (sid,)
            ).fetchone():
                continue
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles "
                "(id, link, title, source, first_seen) "
                "VALUES (?, ?, ?, ?, ?)",
                (sid, art["link"], art["title"], art["source"],
                 datetime.now(timezone.utc).isoformat()),
            )
            items.append(art)
    conn.commit()
    conn.close()
    return items


# Alias matching collectors/__init__.py / daemon worker naming convention.
collect = collect_wikipedia_pageviews


if __name__ == "__main__":
    arts = collect_wikipedia_pageviews()
    print(f"=== Wikipedia pageviews collector ===")
    print(f"Tracked tickers: {sorted(_portfolio_tickers())}")
    print(f"Spike-articles built: {len(arts)}")
    for a in arts:
        print(f"  + {a['title']}")
    if arts:
        from storage.article_store import ArticleStore
        store = ArticleStore()
        inserted = store.insert_batch(arts)
        print(f"Inserted into articles.db: {inserted}")
