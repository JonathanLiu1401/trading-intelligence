"""FRED macro-indicator collector — synthetic 'article' rows from Federal
Reserve Economic Data.

Pulls a handful of headline macro series straight from FRED's public CSV
graph endpoint (no API key needed):

    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>

For each series we take the most recent observations and synthesise one
article per observation — title carries the latest print + change vs. the
prior observation, the body summarises the recent trend. These feed the same
pipeline as every other collector: ``collect_fred()`` returns the standard
``{title, link, summary, published, source}`` dicts and the daemon's
``_ingest()`` (or the ``__main__`` block here) hands them to
``ArticleStore.insert_batch`` — the canonical articles.db insert path shared
by all collectors.

Two dedup layers, matching rss_collector / sec_edgar:
  1. ``data/seen_articles.db`` (WAL, busy_timeout=30000) keyed by
     ``series|date`` so a revised value never re-emits the same observation
     and re-runs don't duplicate.
  2. ``articles.db`` PRIMARY KEY = sha256(url||title) inside insert_batch.
"""
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

# Headline macro series. Order is the print order in __main__.
FRED_SERIES = {
    "CPIAUCSL": "CPI (all urban consumers, SA)",
    "UNRATE": "Unemployment rate",
    "DFF": "Effective federal funds rate",
    "DGS10": "10-year Treasury constant maturity",
    "GDPC1": "Real GDP (chained 2017 $)",
    "PAYEMS": "Total nonfarm payrolls",
}

FREDGRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
RECENT_N = 3  # most recent observations to synthesise per series
FETCH_TIMEOUT = 15  # seconds

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Hardened seen_articles.db connection — mirrors rss_collector._ensure_db /
    # sec_edgar._ensure_db / article_store.py. Many collectors share this one
    # file; SQLite's default busy_timeout=0 turns any transient cross-writer
    # lock into an immediate OperationalError that aborts the pass and drops
    # the fetched batch. WAL + 30s timeout lets the write wait out contention.
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


def _seen_id(series: str, obs_date: str) -> str:
    """Dedup key is series+date only (NOT the value) so a later FRED revision
    of the same observation does not re-emit a near-duplicate article."""
    return hashlib.sha256(f"fred:{series}:{obs_date}".encode("utf-8")).hexdigest()


def _is_seen(conn, sid: str) -> bool:
    return conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (sid,)).fetchone() is not None


def _mark_seen(conn, sid: str, link: str, title: str, source: str):
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
        "VALUES (?, ?, ?, ?, ?)",
        (sid, link, title, source, datetime.now(timezone.utc).isoformat()),
    )


def _fmt_num(x: float) -> str:
    """Trim trailing zeros: 320.400 -> 320.4, 4.0 -> 4."""
    return f"{x:g}"


def _fetch_series(series: str) -> list[tuple[str, float]]:
    """Return [(observation_date 'YYYY-MM-DD', value), ...] oldest→newest,
    with FRED's missing-value marker '.' filtered out."""
    url = FREDGRAPH_CSV.format(series=series)
    resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA})
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    rows: list[tuple[str, float]] = []
    # Skip the header line regardless of its name (modern FRED uses
    # 'observation_date,<SERIES>', older exports use 'DATE,<SERIES>').
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        date_s = parts[0].strip()
        val_s = parts[1].strip()
        # FRED encodes missing observations as '.' (e.g. DGS10 on holidays).
        if not date_s or val_s in ("", ".", "NaN", "ND"):
            continue
        try:
            val = float(val_s)
        except ValueError:
            continue
        rows.append((date_s, val))
    return rows


def collect_fred() -> list[dict]:
    """Collect deduplicated synthetic macro articles from FRED.

    Returns a list of dicts: {title, link, summary, published, source, _series}.
    Consistent with collect_rss / collect_sec_edgar — the caller (daemon
    _ingest or __main__) inserts via ArticleStore.insert_batch.
    """
    conn = _ensure_db()
    new_articles: list[dict] = []

    for series in FRED_SERIES:
        try:
            rows = _fetch_series(series)
        except Exception as e:
            print(f"[fred_collector] Error fetching {series}: {e}")
            continue
        if not rows:
            print(f"[fred_collector] {series}: no usable observations")
            continue

        url = FREDGRAPH_CSV.format(series=series)
        # Need one extra older obs to compute the change for the oldest of the
        # N we surface.
        window = rows[-(RECENT_N + 1):]
        recent = window[-RECENT_N:]

        # Body summarises the recent trend across the surfaced observations.
        # recent occupies window indices [len(window)-len(recent) .. end];
        # prev for window[i] is window[i-1].
        trend_bits = []
        first_idx = len(window) - len(recent)
        for i in range(first_idx, len(window)):
            d, v = window[i]
            prev = window[i - 1][1] if i > 0 else None
            if prev not in (None, 0):
                pct = (v - prev) / prev * 100.0
                trend_bits.append(f"{d} {_fmt_num(v)} ({pct:+.2f}%)")
            else:
                trend_bits.append(f"{d} {_fmt_num(v)}")
        body = (
            f"FRED series {series} ({FRED_SERIES[series]}). "
            f"Recent observations: " + "; ".join(trend_bits) + ". "
            f"Source: FRED (Federal Reserve Economic Data), St. Louis Fed."
        )

        for idx in range(len(window) - 1, len(window) - 1 - len(recent), -1):
            obs_date, val = window[idx]
            prev_val = window[idx - 1][1] if idx > 0 else None
            sid = _seen_id(series, obs_date)
            if _is_seen(conn, sid):
                continue
            if prev_val not in (None, 0):
                pct = (val - prev_val) / prev_val * 100.0
                change = f"prev {_fmt_num(prev_val)}, {pct:+.2f}%"
            elif prev_val is not None:
                change = f"prev {_fmt_num(prev_val)}"
            else:
                change = "no prior obs"
            title = f"FRED {series} {obs_date}: {_fmt_num(val)} ({change})"
            new_articles.append({
                "title": title,
                "link": url,
                "summary": body,
                "published": obs_date,  # ISO-parseable YYYY-MM-DD
                "source": f"fred/{series}",
                "_series": series,
            })
            _mark_seen(conn, sid, url, title, f"fred/{series}")

    conn.commit()
    conn.close()
    return new_articles


# Alias matching the task's requested name.
collect = collect_fred


if __name__ == "__main__":
    # 1) Fetch + show the real latest data point for every series (proves the
    #    public CSV endpoint returned real numbers, not placeholders).
    print("=== FRED latest observations (live fetch) ===")
    obs_count = 0
    eg_line = None
    for series in FRED_SERIES:
        try:
            rows = _fetch_series(series)
        except Exception as e:
            print(f"  {series:9s} FETCH FAILED: {e}")
            continue
        if not rows:
            print(f"  {series:9s} no observations")
            continue
        last_date, last_val = rows[-1]
        obs_count += min(RECENT_N, len(rows))
        ym = last_date[:7]  # YYYY-MM for the Discord example string
        print(f"  {series:9s} latest {last_date} = {_fmt_num(last_val)}  "
              f"({len(rows)} obs total)")
        if eg_line is None:
            eg_line = f"{series} {ym} = {_fmt_num(last_val)}"

    # 2) Collect (deduped) and insert via the canonical shared article store.
    items = collect_fred()
    inserted = 0
    if items:
        from storage.article_store import ArticleStore  # canonical insert path
        store = ArticleStore()
        inserted = store.insert_batch(items)

    print("\n=== Summary ===")
    print(f"Series fetched OK : {sum(1 for s in FRED_SERIES)}")
    print(f"Real observations : {obs_count} data points across series")
    print(f"New synthetic articles built : {len(items)}")
    print(f"Total new items inserted into articles.db : {inserted}")
    if eg_line:
        print(f"DISCORD_EG: {eg_line}")
    for a in items[:8]:
        print(f"  + {a['title']}")
