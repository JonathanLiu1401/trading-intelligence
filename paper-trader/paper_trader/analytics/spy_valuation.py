"""S&P 500 structural-valuation regime read — the macro backdrop nothing else surfaces.

The digital-intern ``market_valuation_collector`` scrapes multpl.com daily and
writes the latest Shiller CAPE, trailing P/E, and earnings yield as a synthetic
article with ``source='market_valuation'`` (URL ``internal://market_valuation/…``).
That article reaches Opus only IF it happens to win the briefing's top-N
ranking on any given day — most cycles it gets crowded out by the day's news,
so the trader is effectively blind to the most slow-moving but highest-stakes
context: "what regime am I trading INTO?"

Every existing macro surface (event_calendar, macro_calendar, sector_pulse,
sector_exposure, sector_signal_fit, earnings_*) describes the *event* or the
*book*, not the structural starting point. A CAPE of 42 (extreme overvaluation,
2.4× historical mean) materially changes the prior on every leveraged-ETF
buy in the watchlist; the operator deserves a discrete read.

This module is a **pure parser** over the latest ``market_valuation`` article's
title. The DB read lives in the dashboard endpoint (same split as
``trade_attribution`` / ``correlation``); the math is unit-testable without Flask.

Pure, **never raises**. Advisory only — never gates Opus, adds no caps
(AGENTS.md #2/#12).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

# Mirrors the collector's bands (digital-intern/collectors/market_valuation_collector.py).
# Keep these in sync; the collector is the SSOT for the regime classification
# itself — these labels are reproduced here so the endpoint can re-classify
# defensively when only raw CAPE is available and so the verdict surface is
# self-contained (no cross-repo import).
_CAPE_BANDS = [
    (40.0, "EXTREME_OVERVALUED", "Extreme Overvaluation"),
    (30.0, "EXPENSIVE",          "Expensive"),
    (20.0, "FAIR_VALUE",         "Fair Value"),
    (15.0, "UNDERVALUED",        "Undervalued"),
    (0.0,  "DEEPLY_UNDERVALUED", "Deep Undervaluation"),
]

# Historical reference (multpl.com / Shiller). Used for the "× mean" and
# "% of all-time peak" derived metrics so the endpoint can answer those
# without re-parsing the title's parenthetical (which the collector may
# eventually re-phrase).
_CAPE_MEAN = 17.38
_CAPE_PEAK = 44.19  # December 1999

_TITLE_CAPE_RE = re.compile(r"CAPE\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_TITLE_PE_RE = re.compile(r"P/E\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _classify_cape(cape: float) -> tuple[str, str]:
    for threshold, key, label in _CAPE_BANDS:
        if cape >= threshold:
            return key, label
    return "DEEPLY_UNDERVALUED", "Deep Undervaluation"


def _f(v) -> float | None:
    try:
        if v is None or isinstance(v, bool):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_title(title: str) -> tuple[float | None, float | None]:
    """Pull CAPE and P/E out of the collector's standard title format.
    Format: "S&P 500 valuation: <LABEL> — CAPE 42.04 (…), P/E 32.19"
    Returns ``(cape, pe)``, either may be None on parse miss."""
    if not title:
        return None, None
    m_cape = _TITLE_CAPE_RE.search(title)
    m_pe = _TITLE_PE_RE.search(title)
    cape = _f(m_cape.group(1)) if m_cape else None
    pe = _f(m_pe.group(1)) if m_pe else None
    return cape, pe


def _iso_age_hours(iso: str | None, now: datetime) -> float | None:
    if not iso:
        return None
    try:
        # The collector writes two ISO formats: ``YYYY-MM-DD HH:MM:SS`` for
        # ``first_seen`` (space separator, naive UTC) and a full ISO for
        # ``published``. Accept either.
        s = iso.strip().replace("Z", "+00:00")
        if " " in s and "T" not in s:
            s = s.replace(" ", "T")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - dt).total_seconds() / 3600.0)


def build_spy_valuation(article: dict | None, *, now: datetime | None = None) -> dict:
    """Parse the latest market_valuation article into a regime read.

    Args:
      ``article`` — a dict shaped like ``signals.get_top_signals()`` output
                    OR a minimal ``{"title": …, "first_seen": …, "url": …}``;
                    None / missing → ``state: "NO_DATA"`` verdict.
      ``now``     — optional UTC clock injection for deterministic tests.

    Returns a JSON-ready dict with ``state``, ``regime``, ``regime_label``,
    ``cape``, ``pe``, ``cape_vs_mean``, ``cape_pct_of_peak``,
    ``article_age_hours``, ``stale`` (article older than 36h — collector
    runs daily so >36h means the source has been silent for >a cycle and
    a half; the reading may be hours/days behind the actual market), and
    a short ``headline`` for direct chat use.

    States:
      * ``NO_DATA``        — no article passed in (collector silent for the
                              full DB retention window, or read failed)
      * ``PARSE_FAILED``   — article exists but title CAPE didn't parse
                              (collector format drift — surfaces honestly,
                              never raises)
      * ``REGIME_READ``    — parsed cleanly; ``regime`` carries the band
    """
    now = now or datetime.now(timezone.utc)

    if not isinstance(article, dict) or not article:
        return {
            "state": "NO_DATA",
            "regime": None,
            "regime_label": None,
            "headline": "S&P valuation read unavailable — no recent market_valuation article in news DB.",
            "cape": None,
            "pe": None,
            "cape_vs_mean": None,
            "cape_pct_of_peak": None,
            "article_age_hours": None,
            "stale": True,
            "source_url": None,
            "as_of": None,
        }

    title = str(article.get("title") or "")
    cape, pe = _parse_title(title)
    age_h = _iso_age_hours(article.get("first_seen") or article.get("published"), now)
    stale = (age_h is None) or (age_h > 36.0)

    if cape is None:
        return {
            "state": "PARSE_FAILED",
            "regime": None,
            "regime_label": None,
            "headline": "S&P valuation article present but title format did not parse — collector drift?",
            "cape": None,
            "pe": pe,
            "cape_vs_mean": None,
            "cape_pct_of_peak": None,
            "article_age_hours": round(age_h, 1) if age_h is not None else None,
            "stale": stale,
            "source_url": article.get("url"),
            "as_of": article.get("first_seen") or article.get("published"),
        }

    regime, regime_label = _classify_cape(cape)
    vs_mean = round(cape / _CAPE_MEAN, 2) if _CAPE_MEAN else None
    pct_peak = round(cape / _CAPE_PEAK * 100, 1) if _CAPE_PEAK else None

    headline = (
        f"S&P 500 regime: {regime_label} (CAPE {cape:.2f} = "
        f"{vs_mean:.2f}× historical mean, {pct_peak:.0f}% of 1999 peak)"
        if vs_mean is not None and pct_peak is not None
        else f"S&P 500 regime: {regime_label} (CAPE {cape:.2f})"
    )

    return {
        "state": "REGIME_READ",
        "regime": regime,
        "regime_label": regime_label,
        "headline": headline,
        "cape": round(cape, 2),
        "pe": round(pe, 2) if pe is not None else None,
        "cape_vs_mean": vs_mean,
        "cape_pct_of_peak": pct_peak,
        "article_age_hours": round(age_h, 1) if age_h is not None else None,
        "stale": stale,
        "source_url": article.get("url"),
        "as_of": article.get("first_seen") or article.get("published"),
    }
