"""FRED production & inflation depth collector.

Fills the gaps in fred_collector.py by tracking:

  PCEPILFE  — Core PCE price index (Fed's *preferred* inflation gauge, ex food/energy)
  WPSFD4131 — PPI Final Demand (upstream price pressure, monthly)
  PPIACO    — PPI All Commodities (broad commodity price pressure)
  INDPRO    — Industrial Production Index (real output of mfg+mining+utilities)
  TCU       — Capacity Utilization, Total Industry (% of capacity in use)
  PERMIT    — Privately-owned housing units authorised (SAAR, thousands)
  DGORDER   — Manufacturers' New Orders: Durable Goods ($ millions)
  PSAVERT   — Personal Savings Rate (% of disposable income)
  DSPIC96   — Real Disposable Personal Income (chained $, monthly)

Why these matter:
  - Core PCE is the inflation measure the FOMC explicitly targets (2% goal);
    when it deviates from 2%, monetary policy is most directly affected.
  - PPI leads CPI by ~2 months; a PPI spike is the earliest inflation warning.
  - Industrial Production + Capacity Utilization together define the
    "output gap": high utilization (>80%) implies inflationary pressure;
    low (<74%) implies slack and potential easing.
  - Building Permits lead housing starts by ~30 days and GDP by ~6 months.
  - Durable Goods Orders are a leading indicator of capital investment.
  - Personal Savings Rate and Real Disposable Income signal consumer health.

Signals emitted per-series (in addition to raw observation articles):
  PCEPILFE  — alert when YoY rate is above 2.5% (hawkish) or below 1.5%
  TCU       — alert when utilization crosses 80% (inflationary) or 74% (slack)
  PERMIT    — alert when MoM change ≥ |10%| (housing cycle turning point)
  DGORDER   — alert when MoM change ≥ |5%| (investment cycle signal)
  PSAVERT   — alert when rate < 3% (consumer stretched) or > 8% (defensive)

All dedup via seen_articles.db (keyed series+date), mirroring fred_collector.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
# Use a dedicated DB to avoid WAL-lock contention with the daemon's heavy writers.
# Mirrors the pattern used by macro_calendar_collector, earnings_calendar, etc.
DB_PATH = BASE_DIR / "data" / "fred_production_seen.db"

FREDGRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
RECENT_N = 2          # observations per series for article generation
FETCH_TIMEOUT = 45    # seconds — FRED returns full history CSV; allow extra time
SOURCE_PREFIX = "fred_production"

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

log = logging.getLogger("fred_production_collector")

# ---------------------------------------------------------------------------
# Series registry — (label, unit_hint, yoy_capable)
# unit_hint is used in article titles for human readability.
# yoy_capable=True means 12-month YoY % change is meaningful.
# ---------------------------------------------------------------------------
SERIES: dict[str, dict] = {
    "PCEPILFE": {
        "label": "Core PCE Price Index (ex food/energy)",
        "unit": "index pts",
        "yoy": True,
        "hawk_threshold_yoy": 2.5,   # % → hawkish alert
        "dove_threshold_yoy": 1.5,   # % → dovish alert
    },
    "WPSFD4131": {
        "label": "PPI Final Demand",
        "unit": "index",
        "yoy": True,
    },
    "PPIACO": {
        "label": "PPI All Commodities",
        "unit": "index",
        "yoy": True,
    },
    "INDPRO": {
        "label": "Industrial Production Index",
        "unit": "index",
        "yoy": True,
    },
    "TCU": {
        "label": "Capacity Utilization - Total Industry",
        "unit": "%",
        "yoy": False,
        "high_threshold": 80.0,   # → inflationary pressure
        "low_threshold": 74.0,    # → economic slack
    },
    "PERMIT": {
        "label": "Building Permits (SAAR, thousands)",
        "unit": "k units",
        "yoy": False,
        "mom_alert_pct": 10.0,    # |MoM %| trigger
    },
    "DGORDER": {
        "label": "Durable Goods New Orders",
        "unit": "$M",
        "yoy": False,
        "mom_alert_pct": 5.0,
    },
    "PSAVERT": {
        "label": "Personal Savings Rate",
        "unit": "%",
        "yoy": False,
        "stretched_threshold": 3.0,   # below → consumer stretched
        "defensive_threshold": 8.0,   # above → consumer defensive
    },
    "DSPIC96": {
        "label": "Real Disposable Personal Income",
        "unit": "$ billions",
        "yoy": True,
    },
    "PCE": {
        "label": "Personal Consumption Expenditures",
        "unit": "$ billions",
        "yoy": True,
    },
}

# Number of historical observations to fetch for YoY calculation (13 = 12 mo + current).
YOY_FETCH_N = 13


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


def _seen_id(series: str, obs_date: str, suffix: str = "") -> str:
    key = f"fred_prod:{series}:{obs_date}{suffix}"
    return hashlib.sha256(key.encode()).hexdigest()


def _is_seen(conn: sqlite3.Connection, sid: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM seen_articles WHERE id=?", (sid,)
    ).fetchone() is not None


def _mark_seen(conn: sqlite3.Connection, sid: str, link: str, title: str, source: str):
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
        "VALUES (?, ?, ?, ?, ?)",
        (sid, link, title, source, datetime.now(timezone.utc).isoformat()),
    )


def _fmt(x: float) -> str:
    return f"{x:g}"


def _fetch_series(series: str, n: int) -> list[tuple[str, float]]:
    """Return the last *n* valid observations as [(date, value), ...] oldest→newest."""
    url = FREDGRAPH_CSV.format(series=series)
    resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA})
    resp.raise_for_status()
    rows: list[tuple[str, float]] = []
    for line in resp.text.splitlines()[1:]:  # skip header
        parts = line.strip().split(",")
        if len(parts) != 2:
            continue
        date_str, val_str = parts
        val_str = val_str.strip()
        if val_str == "." or not val_str:
            continue
        try:
            rows.append((date_str.strip(), float(val_str)))
        except ValueError:
            continue
    # Return the most recent n observations.
    return rows[-n:] if len(rows) >= n else rows


def _build_article(
    series: str,
    obs_date: str,
    val: float,
    prev_val: float | None,
    cfg: dict,
    yoy_pct: float | None,
    signal: str | None,
) -> dict:
    label = cfg["label"]
    unit = cfg.get("unit", "")
    source = f"{SOURCE_PREFIX}/{series}"

    # Build title
    if prev_val is not None:
        change = val - prev_val
        change_str = f" (Δ{_fmt(change):+})" if abs(change) >= 0.01 else ""
    else:
        change_str = ""

    yoy_str = f", YoY {yoy_pct:+.2f}%" if yoy_pct is not None else ""
    title = f"FRED {series} {obs_date}: {_fmt(val)} {unit}{change_str}{yoy_str}"

    # Build summary
    parts = [f"{label} for {obs_date}: {_fmt(val)} {unit}."]
    if prev_val is not None:
        parts.append(f"Prior period: {_fmt(prev_val)} {unit}.")
    if yoy_pct is not None:
        parts.append(f"Year-over-year change: {yoy_pct:+.2f}%.")
    if signal:
        parts.append(signal)

    summary = " ".join(parts)
    link = f"https://fred.stlouisfed.org/series/{series}"

    return {
        "title": title,
        "link": link,
        "summary": summary,
        "published": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "_series": series,
        "_obs_date": obs_date,
    }


def _yoy_pct(rows: list[tuple[str, float]]) -> float | None:
    """Compute YoY % change from rows if we have >= 13 observations."""
    if len(rows) < 13:
        return None
    current = rows[-1][1]
    year_ago = rows[-13][1]
    if year_ago == 0:
        return None
    return (current - year_ago) / abs(year_ago) * 100.0


def collect_fred_production() -> list[dict]:
    """Fetch production/inflation series from FRED and return new article dicts."""
    conn = _ensure_db()
    articles: list[dict] = []

    for series, cfg in SERIES.items():
        try:
            # Fetch enough rows for YoY if needed
            need_n = YOY_FETCH_N if cfg.get("yoy") else max(RECENT_N, 3)
            rows = _fetch_series(series, need_n)
        except Exception as exc:
            log.warning("fred_production: fetch failed %s: %s", series, exc)
            continue

        if not rows:
            continue

        obs_date, val = rows[-1]
        prev_val = rows[-2][1] if len(rows) >= 2 else None
        yoy_pct = _yoy_pct(rows) if cfg.get("yoy") else None

        # Build contextual signal string for noteworthy readings
        signal: str | None = None

        # Core PCE hawkish/dovish signal
        if series == "PCEPILFE" and yoy_pct is not None:
            hawk = cfg.get("hawk_threshold_yoy", 2.5)
            dove = cfg.get("dove_threshold_yoy", 1.5)
            if yoy_pct >= hawk:
                signal = (
                    f"HAWKISH SIGNAL: Core PCE at {yoy_pct:.2f}% YoY is above the "
                    f"Fed's {hawk}% alert threshold — monetary tightening pressure elevated."
                )
            elif yoy_pct <= dove:
                signal = (
                    f"DOVISH SIGNAL: Core PCE at {yoy_pct:.2f}% YoY is below {dove}% "
                    "— opens the door for Fed easing."
                )
            else:
                signal = (
                    f"Core PCE at {yoy_pct:.2f}% YoY is within the Fed's target band."
                )

        # Capacity Utilization thresholds
        elif series == "TCU":
            high = cfg.get("high_threshold", 80.0)
            low = cfg.get("low_threshold", 74.0)
            if val >= high:
                signal = (
                    f"INFLATIONARY PRESSURE: Capacity utilization at {_fmt(val)}% "
                    f"is above {high}% — supply-side constraints building."
                )
            elif val <= low:
                signal = (
                    f"ECONOMIC SLACK: Capacity utilization at {_fmt(val)}% "
                    f"is below {low}% — significant idle capacity, disinflationary."
                )

        # Building Permits MoM spike
        elif series == "PERMIT" and prev_val is not None:
            alert_pct = cfg.get("mom_alert_pct", 10.0)
            if prev_val > 0:
                mom = (val - prev_val) / prev_val * 100.0
                if abs(mom) >= alert_pct:
                    direction = "surge" if mom > 0 else "collapse"
                    signal = (
                        f"HOUSING CYCLE SIGNAL: Building permits {direction} "
                        f"{mom:+.1f}% MoM — leading indicator for construction activity."
                    )

        # Durable Goods MoM swing
        elif series == "DGORDER" and prev_val is not None:
            alert_pct = cfg.get("mom_alert_pct", 5.0)
            if prev_val > 0:
                mom = (val - prev_val) / prev_val * 100.0
                if abs(mom) >= alert_pct:
                    direction = "jump" if mom > 0 else "drop"
                    signal = (
                        f"CAPEX SIGNAL: Durable goods orders {direction} "
                        f"{mom:+.1f}% MoM — signals shift in business investment."
                    )

        # Personal Savings Rate stress signals
        elif series == "PSAVERT":
            stretched = cfg.get("stretched_threshold", 3.0)
            defensive = cfg.get("defensive_threshold", 8.0)
            if val <= stretched:
                signal = (
                    f"CONSUMER STRESS: Savings rate at {_fmt(val)}% "
                    f"≤ {stretched}% — households drawing down savings to fund spending."
                )
            elif val >= defensive:
                signal = (
                    f"CONSUMER DEFENSIVE: Savings rate at {_fmt(val)}% "
                    f"≥ {defensive}% — households pulling back on discretionary spending."
                )

        # Dedup check — emit one article per (series, obs_date)
        sid = _seen_id(series, obs_date)
        if _is_seen(conn, sid):
            continue

        article = _build_article(
            series=series,
            obs_date=obs_date,
            val=val,
            prev_val=prev_val,
            cfg=cfg,
            yoy_pct=yoy_pct,
            signal=signal,
        )
        articles.append(article)
        _mark_seen(conn, sid, article["link"], article["title"], article["source"])

    conn.commit()
    conn.close()
    return articles


# Alias for daemon/task compatibility
collect = collect_fred_production


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    print("=== FRED Production & Inflation Depth Collector (live fetch) ===\n")

    results = []
    for series, cfg in SERIES.items():
        need_n = YOY_FETCH_N if cfg.get("yoy") else 3
        try:
            rows = _fetch_series(series, need_n)
        except Exception as exc:
            print(f"  {series:12s} FETCH FAILED: {exc}")
            continue

        if not rows:
            print(f"  {series:12s} no observations")
            continue

        obs_date, val = rows[-1]
        yoy_pct = _yoy_pct(rows) if cfg.get("yoy") else None
        yoy_str = f"  YoY {yoy_pct:+.2f}%" if yoy_pct is not None else ""
        print(f"  {series:12s}  {obs_date}  {_fmt(val):>12} {cfg.get('unit',''):8s}{yoy_str}")
        results.append((series, obs_date, val, yoy_pct))

    print(f"\nTotal series fetched: {len(results)}")

    # Run collect and show new articles
    items = collect_fred_production()
    print(f"New articles (not yet seen): {len(items)}")
    for a in items[:8]:
        print(f"  + {a['title']}")
        if a.get("summary") and len(a["summary"]) > len(a["title"]) + 10:
            # Print signal portion only
            summary = a["summary"]
            signal_start = max(summary.find("SIGNAL"), summary.find("STRESS"),
                               summary.find("PRESSURE"), summary.find("SLACK"),
                               summary.find("HOUSING"), summary.find("CAPEX"))
            if signal_start > 0:
                print(f"    → {summary[signal_start:signal_start+120]}")

    if results:
        # Print example Discord string
        s, d, v, yoy = results[0]
        yoy_str = f" YoY {yoy:+.2f}%" if yoy else ""
        eg = f"{s} {d[:7]}: {_fmt(v)}{yoy_str}"
        print(f"\nDISCORD_EG: {eg}")
