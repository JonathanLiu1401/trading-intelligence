"""Financial stress and regional Fed survey collector.

Tracks systemic financial stress indices and regional Fed manufacturing surveys
via FRED — free, no API key required.  These indicators are not covered by
fred_collector.py or fred_production_collector.py.

Series tracked:

  Financial Stress Indices (weekly/monthly):
    NFCI     — Chicago Fed National Financial Conditions Index
                (0 = historical average; positive = tighter/stressed)
    ANFCI    — Adjusted NFCI (removes macro effects; pure financial-sector stress)
    STLFSI2  — St. Louis Fed Financial Stress Index (same sign convention)
    KCFSI    — Kansas City Fed Financial Stress Index (negative = loose)
    CFNAIMA3 — Chicago Fed National Activity Index 3-month MA
                (positive = above-trend growth; below -0.7 = recession risk)

  Regional Manufacturing Surveys (monthly):
    GACDFSA066MSFRBPHI — Philadelphia Fed Manufacturing Business Outlook
                          (positive = expansion; negative = contraction)
    DRTSCILM  — Fed Senior Loan Officer Survey: C&I loans tightening %
                 (positive = tightening; a sharp jump precedes credit crunches)

Signals emitted:
  NFCI      — alert when index > 0.1 (stress rising) or < -0.6 (very loose)
  ANFCI     — alert when index > 0.5 (elevated pure financial stress)
  CFNAIMA3  — alert when index < -0.7 (recession flag) or > +0.7 (boom)
  GACDFSA..  — alert on any sign flip (expansion ↔ contraction)
  DRTSCILM  — alert when quarterly change > +10 pp (credit tightening surge)

Two-layer dedup matching all other collectors:
  1. data/seen_articles.db keyed by sha256(series+date)
  2. articles.db PRIMARY KEY = sha256(url||title) inside insert_batch
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "financial_stress_seen.db"

FREDGRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FETCH_TIMEOUT = 30
SOURCE_PREFIX = "financial_stress"

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

log = logging.getLogger("financial_stress_collector")

# ---------------------------------------------------------------------------
# Series registry
# ---------------------------------------------------------------------------
SERIES: dict[str, dict] = {
    "NFCI": {
        "label": "Chicago Fed National Financial Conditions Index",
        "unit": "index",
        "freq": "weekly",
        "high_threshold": 0.1,
        "low_threshold": -0.6,
        "high_signal": "FINANCIAL STRESS RISING: NFCI above 0.1 signals tightening conditions — credit spreads widening, volatility elevated.",
        "low_signal": "FINANCIAL CONDITIONS VERY LOOSE: NFCI below -0.6 signals historically accommodative conditions.",
    },
    "ANFCI": {
        "label": "Chicago Fed Adjusted National Financial Conditions Index",
        "unit": "index",
        "freq": "weekly",
        "high_threshold": 0.5,
        "high_signal": "ELEVATED PURE FINANCIAL STRESS: Adjusted NFCI above 0.5 signals financial sector stress independent of macro cycle.",
    },
    "STLFSI2": {
        "label": "St. Louis Fed Financial Stress Index",
        "unit": "index",
        "freq": "weekly",
        "high_threshold": 0.5,
        "low_threshold": -1.0,
        "high_signal": "STRESS SIGNAL: St. Louis FSI above 0.5 — cross-asset stress indicators elevated.",
        "low_signal": "LOW STRESS: St. Louis FSI below -1.0 — very benign financial conditions.",
    },
    "KCFSI": {
        "label": "Kansas City Fed Financial Stress Index",
        "unit": "index",
        "freq": "monthly",
        "high_threshold": 0.0,
        "low_threshold": -1.0,
        "high_signal": "KCFSI STRESS: Index turned positive — Kansas City Fed sees above-normal financial stress.",
        "low_signal": "KCFSI LOOSE: Index below -1.0 — very accommodative financial conditions in KC district.",
    },
    "CFNAIMA3": {
        "label": "Chicago Fed National Activity Index (3-month MA)",
        "unit": "index",
        "freq": "monthly",
        "high_threshold": 0.7,
        "low_threshold": -0.7,
        "high_signal": "ABOVE-TREND GROWTH: CFNAI-MA3 above +0.7 signals above-trend economic growth, potential inflationary pressure.",
        "low_signal": "RECESSION FLAG: CFNAI-MA3 below -0.7 — historically associated with recession onset. High vigilance warranted.",
    },
    "GACDFSA066MSFRBPHI": {
        "label": "Philadelphia Fed Manufacturing Business Outlook",
        "unit": "diffusion index",
        "freq": "monthly",
        "track_sign_flip": True,
    },
    "DRTSCILM": {
        "label": "Senior Loan Officers: C&I Loan Standards Tightening (%)",
        "unit": "%",
        "freq": "quarterly",
        "high_threshold": 20.0,
        "high_signal": "CREDIT TIGHTENING SURGE: >20% of senior loan officers tightening C&I standards — precedes credit crunch / business loan contraction.",
    },
}

FETCH_N = 4  # most recent observations (enough for sign-flip detection)


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


def _seen_id(series: str, obs_date: str) -> str:
    return hashlib.sha256(f"fstress:{series}:{obs_date}".encode()).hexdigest()


def _fetch_series(series: str, n: int) -> list[tuple[str, float]]:
    url = FREDGRAPH_CSV.format(series=series)
    resp = httpx.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA}, follow_redirects=True)
    resp.raise_for_status()
    rows: list[tuple[str, float]] = []
    for line in resp.text.splitlines()[1:]:
        parts = line.strip().split(",")
        if len(parts) != 2:
            continue
        date_str, val_str = parts
        val_str = val_str.strip()
        if val_str in (".", "") or not val_str:
            continue
        try:
            rows.append((date_str.strip(), float(val_str)))
        except ValueError:
            continue
    return rows[-n:] if len(rows) >= n else rows


def _fmt(x: float) -> str:
    return f"{x:g}"


def _build_signal(series: str, val: float, prev_val: float | None, cfg: dict) -> str | None:
    if cfg.get("track_sign_flip") and prev_val is not None:
        if prev_val < 0 and val >= 0:
            return f"EXPANSION SIGNAL: {cfg['label']} flipped positive ({_fmt(val)}) from {_fmt(prev_val)} — manufacturing activity returned to expansion territory."
        elif prev_val >= 0 and val < 0:
            return f"CONTRACTION SIGNAL: {cfg['label']} turned negative ({_fmt(val)}) from {_fmt(prev_val)} — manufacturing activity entered contraction territory."
        return None

    high = cfg.get("high_threshold")
    low = cfg.get("low_threshold")
    if high is not None and val > high:
        return cfg.get("high_signal")
    if low is not None and val < low:
        return cfg.get("low_signal")
    return None


def collect_financial_stress() -> list[dict]:
    conn = _ensure_db()
    articles: list[dict] = []

    for series, cfg in SERIES.items():
        try:
            rows = _fetch_series(series, FETCH_N)
        except Exception as exc:
            log.warning("financial_stress: fetch failed %s: %s", series, exc)
            continue

        if not rows:
            continue

        obs_date, val = rows[-1]
        prev_val = rows[-2][1] if len(rows) >= 2 else None

        sid = _seen_id(series, obs_date)
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (sid,)).fetchone():
            continue

        signal = _build_signal(series, val, prev_val, cfg)
        label = cfg["label"]
        unit = cfg.get("unit", "")
        freq = cfg.get("freq", "")

        change_str = ""
        if prev_val is not None:
            delta = val - prev_val
            change_str = f" (Δ{delta:+.3g})"

        title = f"[{freq.upper()}] {label}: {_fmt(val)} {unit} ({obs_date}){change_str}"
        if signal:
            title = f"[ALERT] {title}"

        summary_parts = [f"{label} for {obs_date}: {_fmt(val)} {unit}."]
        if prev_val is not None:
            summary_parts.append(f"Prior: {_fmt(prev_val)} {unit}.")
        if signal:
            summary_parts.append(signal)

        articles.append({
            "title": title,
            "link": f"https://fred.stlouisfed.org/series/{series}",
            "summary": " ".join(summary_parts),
            "published": datetime.now(timezone.utc).isoformat(),
            "source": f"{SOURCE_PREFIX}/{series}",
        })

        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (sid, f"https://fred.stlouisfed.org/series/{series}",
             title, f"{SOURCE_PREFIX}/{series}",
             datetime.now(timezone.utc).isoformat()),
        )

    conn.commit()
    conn.close()
    return articles


collect = collect_financial_stress


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Financial Stress & Regional Fed Survey Collector ===\n")
    items = collect_financial_stress()
    print(f"New articles: {len(items)}")
    for a in items:
        print(f"  {a['title'][:90]}")
        summary = a.get("summary", "")
        sig_start = max(summary.find("SIGNAL"), summary.find("STRESS"),
                        summary.find("FLAG"), summary.find("ALERT"),
                        summary.find("EXPANSION"), summary.find("CONTRACTION"),
                        summary.find("LOOSE"), summary.find("TIGHTENING"))
        if sig_start > 0:
            print(f"    → {summary[sig_start:sig_start+100]}")
