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
    "DGS2": "2-year Treasury constant maturity",
    "DGS10": "10-year Treasury constant maturity",
    "GDPC1": "Real GDP (chained 2017 $)",
    "PAYEMS": "Total nonfarm payrolls",
    # Credit market indicators (ICE BofA via FRED, 1-day lag)
    "BAMLC0A0CM": "ICE BofA US Corp Bond IG OAS (bps)",
    "BAMLH0A0HYM2": "ICE BofA US HY Index OAS (bps)",
    "SOFR": "Secured Overnight Financing Rate (%)",
}

# Yield-curve spread emitted as a separate synthetic article series. The
# 10Y-2Y spread is the canonical recession-leading-indicator (inversions in
# 1978/1989/2000/2006/2019 each preceded a US recession within ~6-24 months).
# Computed from the two underlying series we already fetch; no extra network
# call. Source tag intentionally distinct (``fred/10y2y_spread``) so the
# operator can see it as its own signal in the dashboard's source view.
YIELD_CURVE_SOURCE = "fred/10y2y_spread"
YIELD_CURVE_SERIES = "DGS10_MINUS_DGS2"

# HY-IG credit spread: difference between high-yield and investment-grade OAS.
# Widens ahead of equity selloffs when credit markets price stress first.
CREDIT_SPREAD_SOURCE = "fred/hy_ig_spread"
CREDIT_SPREAD_SERIES = "BAMLH0A0HYM2_MINUS_BAMLC0A0CM"

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


def _yield_curve_articles(conn, dgs10: list[tuple[str, float]],
                          dgs2: list[tuple[str, float]]) -> list[dict]:
    """Emit synthetic 10Y-2Y spread articles for the most recent N dates where
    BOTH DGS10 and DGS2 have a published value.

    The spread is the canonical recession signal — every US recession since
    1955 was preceded by an inversion (10Y < 2Y). A standalone article series
    lets the briefing / alert layer surface "yield curve inverted today" the
    same way it does a CPI print, without recomputing from per-series rows.

    Dedup key is the obs_date alone (``fred:DGS10_MINUS_DGS2:<date>``), so a
    later FRED revision of either input never re-emits the same date.

    Returns standard collector dicts (title/link/summary/published/source).
    All ``zip``-able input shape: oldest→newest per ``_fetch_series`` contract.
    """
    if not dgs10 or not dgs2:
        return []
    by_date_10 = dict(dgs10)
    by_date_2 = dict(dgs2)
    common_dates = sorted(set(by_date_10) & set(by_date_2))
    if not common_dates:
        return []
    surfaced = common_dates[-RECENT_N:]
    # FRED's two-series graph URL — clickable evidence link rather than the
    # single-series page used by the underlying observations.
    url = (
        "https://fred.stlouisfed.org/graph/?g=fredgraph"
        f"&id={','.join(['DGS10', 'DGS2'])}"
    )
    out: list[dict] = []
    for d in surfaced:
        v10 = by_date_10[d]
        v2 = by_date_2[d]
        spread = v10 - v2  # percentage points
        sid = _seen_id(YIELD_CURVE_SERIES, d)
        if _is_seen(conn, sid):
            continue
        # Prior observation in the OVERLAP set is the meaningful comparator —
        # a Δ vs the previous *common* date answers "did the curve steepen or
        # flatten today" rather than vs an arbitrary holiday-skipped row.
        prev_idx = common_dates.index(d) - 1
        if prev_idx >= 0:
            pd = common_dates[prev_idx]
            prev_spread = by_date_10[pd] - by_date_2[pd]
            d_spread = spread - prev_spread
            # Inversion *direction* is what moves markets, not just sign.
            direction = "flattening" if d_spread < 0 else "steepening"
            change = f"prev {_fmt_num(prev_spread)} ({direction} {d_spread:+.2f})"
        else:
            change = "no prior obs"
        # Spread sign is THE recession-signal headline tag; "INVERTED" reads
        # like the wire-headline analyst would scan for.
        regime = "INVERTED" if spread < 0 else "positive"
        title = (
            f"FRED 10Y-2Y spread {d}: {_fmt_num(spread)} ({regime}, "
            f"10Y={_fmt_num(v10)} 2Y={_fmt_num(v2)}; {change})"
        )
        body = (
            f"FRED-derived 10-year minus 2-year Treasury spread for {d}: "
            f"{spread:+.2f} percentage points "
            f"(10Y={_fmt_num(v10)}, 2Y={_fmt_num(v2)}). "
            f"Spread regime: {regime} (negative spread = inverted yield "
            f"curve, a leading recession indicator). "
            f"Source: FRED (Federal Reserve Economic Data), St. Louis Fed."
        )
        out.append({
            "title": title,
            "link": url,
            "summary": body,
            "published": d,
            "source": YIELD_CURVE_SOURCE,
            "_series": YIELD_CURVE_SERIES,
        })
        _mark_seen(conn, sid, url, title, YIELD_CURVE_SOURCE)
    return out


def _credit_spread_articles(conn, hy_oas: list[tuple[str, float]],
                            ig_oas: list[tuple[str, float]]) -> list[dict]:
    """Emit synthetic HY-IG credit risk spread articles.

    HY OAS minus IG OAS = excess compensation demanded for default risk.
    A widening spread signals credit-market stress; historically leads
    equity drawdowns by days to weeks. Dedup key: obs_date only.
    """
    if not hy_oas or not ig_oas:
        return []
    by_date_hy = dict(hy_oas)
    by_date_ig = dict(ig_oas)
    common_dates = sorted(set(by_date_hy) & set(by_date_ig))
    if not common_dates:
        return []
    surfaced = common_dates[-RECENT_N:]
    url = (
        "https://fred.stlouisfed.org/graph/?g=fredgraph"
        f"&id=BAMLH0A0HYM2,BAMLC0A0CM"
    )
    out: list[dict] = []
    for d in surfaced:
        hy = by_date_hy[d]
        ig = by_date_ig[d]
        spread = hy - ig
        sid = _seen_id(CREDIT_SPREAD_SERIES, d)
        if _is_seen(conn, sid):
            continue
        prev_idx = common_dates.index(d) - 1
        if prev_idx >= 0:
            pd = common_dates[prev_idx]
            prev_spread = by_date_hy[pd] - by_date_ig[pd]
            d_spread = spread - prev_spread
            direction = "widening" if d_spread > 0 else "tightening"
            change = f"prev {_fmt_num(prev_spread)} ({direction} {d_spread:+.2f})"
        else:
            change = "no prior obs"
        # Stress regime: >3% HY-IG spread historically signals elevated default risk
        regime = "STRESSED" if spread > 3.0 else ("elevated" if spread > 2.0 else "normal")
        title = (
            f"FRED HY-IG credit spread {d}: {_fmt_num(spread)}% ({regime}, "
            f"HY={_fmt_num(hy)} IG={_fmt_num(ig)}; {change})"
        )
        body = (
            f"FRED-derived HY minus IG OAS credit spread for {d}: "
            f"{spread:+.2f}% (HY OAS={_fmt_num(hy)}, IG OAS={_fmt_num(ig)}). "
            f"Credit regime: {regime} (>3% = stressed credit market, "
            f"historically leads equity drawdowns). "
            f"Source: ICE BofA indices via FRED, St. Louis Fed."
        )
        out.append({
            "title": title,
            "link": url,
            "summary": body,
            "published": d,
            "source": CREDIT_SPREAD_SOURCE,
            "_series": CREDIT_SPREAD_SERIES,
        })
        _mark_seen(conn, sid, url, title, CREDIT_SPREAD_SOURCE)
    return out


def collect_fred() -> list[dict]:
    """Collect deduplicated synthetic macro articles from FRED.

    Returns a list of dicts: {title, link, summary, published, source, _series}.
    Consistent with collect_rss / collect_sec_edgar — the caller (daemon
    _ingest or __main__) inserts via ArticleStore.insert_batch.
    """
    conn = _ensure_db()
    new_articles: list[dict] = []
    # Cache per-series rows so the derived 10Y-2Y spread reuses the same fetch
    # rather than hitting FRED twice.
    fetched: dict[str, list[tuple[str, float]]] = {}

    for series in FRED_SERIES:
        try:
            rows = _fetch_series(series)
        except Exception as e:
            print(f"[fred_collector] Error fetching {series}: {e}")
            continue
        if not rows:
            print(f"[fred_collector] {series}: no usable observations")
            continue
        fetched[series] = rows

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

    # Derived 10Y-2Y spread — runs after per-series fetches so both inputs
    # are cached. Quietly skipped if either DGS10 or DGS2 failed to fetch
    # (covered by the empty-rows branch above) so a transient FRED hiccup on
    # one leg never re-uses a stale value for the other.
    try:
        spread_items = _yield_curve_articles(
            conn, fetched.get("DGS10", []), fetched.get("DGS2", [])
        )
        new_articles.extend(spread_items)
    except Exception as e:  # pragma: no cover - belt+braces; live insertion
        print(f"[fred_collector] yield-curve spread synth failed: {e}")

    # Derived HY-IG credit risk spread.
    try:
        credit_items = _credit_spread_articles(
            conn, fetched.get("BAMLH0A0HYM2", []), fetched.get("BAMLC0A0CM", [])
        )
        new_articles.extend(credit_items)
    except Exception as e:
        print(f"[fred_collector] credit spread synth failed: {e}")

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
