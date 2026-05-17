"""Per-ticker 24h trend tracker — writes data/sentiment_trends.json.

This codebase has no dedicated sentiment classifier; the closest proxy is
``ai_score`` (Sonnet's combined relevance/urgency score, 0..10). We treat
``ai_score`` as a directional importance signal and track the rolling average
per ticker over a 24-hour window. Higher avg → more high-importance coverage
of that ticker. Stored alongside ``count`` and ``urgent_count`` so the dashboard
can show both.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "sentiment_trends.json"
PORTFOLIO_PATH = BASE_DIR / "config" / "portfolio.json"
WATCHLIST_PATH = BASE_DIR / "config" / "watchlist.json"

# How many hours back to aggregate.
WINDOW_HOURS = 24


def _load_tracked_tickers() -> list[str]:
    """Return the union of portfolio positions + sector watchlist + memory_core."""
    tickers: list[str] = []
    seen: set[str] = set()

    def _add(t: str):
        u = (t or "").strip().upper()
        if u and u not in seen:
            seen.add(u)
            tickers.append(u)

    try:
        with open(PORTFOLIO_PATH, "r") as f:
            pf = json.load(f)
        for pos in pf.get("positions", []):
            _add(pos.get("ticker", ""))
        for opt in pf.get("options", []):
            _add(opt.get("underlying", ""))
        for t in pf.get("sector_watchlist", []):
            _add(t)
    except Exception:
        pass

    try:
        with open(WATCHLIST_PATH, "r") as f:
            wl = json.load(f)
        for key in ("memory_core", "semis_equipment", "portfolio"):
            for t in wl.get(key, []):
                _add(t)
    except Exception:
        pass

    return tickers


def _build_ticker_regex(tickers: Iterable[str]) -> re.Pattern:
    return re.compile(
        r"\b(" + "|".join(re.escape(t) for t in tickers) + r")\b",
        re.I,
    )


def compute_trends(store) -> dict:
    """Aggregate per-ticker stats over the last WINDOW_HOURS. Returns the dict
    that gets written to OUTPUT_PATH."""
    tickers = _load_tracked_tickers()
    if not tickers:
        return {"as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "window_hours": WINDOW_HOURS, "tickers": {}}

    pattern = _build_ticker_regex(tickers)
    sums: dict[str, float] = {t: 0.0 for t in tickers}
    counts: dict[str, int] = {t: 0 for t in tickers}
    urgent: dict[str, int] = {t: 0 for t in tickers}
    max_score: dict[str, float] = {t: 0.0 for t in tickers}

    cutoff = (datetime.now(timezone.utc).timestamp() - WINDOW_HOURS * 3600)
    cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()

    try:
        from storage.article_store import _LIVE_ONLY_CLAUSE
        rows = store.conn.execute(
            "SELECT title, ai_score, kw_score, urgency, source "
            f"FROM articles WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (cutoff_iso,),
        ).fetchall()
    except Exception:
        rows = []

    for title, ai, kw, urg, src in rows:
        ai = float(ai or 0)
        kw = float(kw or 0)
        # Prefer ai_score; fall back to kw_score for unscored items.
        score = ai if ai > 0 else kw
        if score <= 0:
            continue
        haystack = f"{title or ''} {src or ''}"
        matched: set[str] = set()
        for m in pattern.finditer(haystack):
            matched.add(m.group(1).upper())
        for t in matched:
            sums[t] += score
            counts[t] += 1
            if score > max_score[t]:
                max_score[t] = score
            if int(urg or 0) >= 1:
                urgent[t] += 1

    out_tickers: dict[str, dict] = {}
    for t in tickers:
        n = counts[t]
        out_tickers[t] = {
            "count": n,
            "urgent_count": urgent[t],
            "avg_score": round(sums[t] / n, 2) if n else 0.0,
            "max_score": round(max_score[t], 2),
        }

    return {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_hours": WINDOW_HOURS,
        "tickers": out_tickers,
        "note": "avg_score is ai_score (Sonnet relevance/urgency, 0-10) — there is no dedicated sentiment model.",
    }


def write_trends(store) -> dict:
    data = compute_trends(store)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(OUTPUT_PATH)
    return data


def read_trends() -> dict | None:
    """Read the most recently written trends file (or None if missing)."""
    if not OUTPUT_PATH.exists():
        return None
    try:
        with open(OUTPUT_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None
