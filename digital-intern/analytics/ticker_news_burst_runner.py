"""Ticker news-burst — per-ticker volume spike vs per-hour baseline.

Pure builder + Flask endpoint surface mirroring the ``ticker_velocity_runner``
template. Answers a different question than ``ticker_velocity``:

  ``ticker_velocity`` DISCOVERS top-N tickers by mention count across a
  full ``2 * window_min`` window and emits a recent/prior RATIO.

  ``ticker_news_burst`` TAKES a known ticker universe (the held + watched
  book) and asks per-ticker: is this name's last-hour mention rate an
  unusual spike vs the prior 24h per-hour baseline? It SURVIVES with a
  small held universe where ``ticker_velocity`` would return ``NO_DATA``
  (because no held name made the top-N discover set).

Live evidence (2026-05-26, 1h vs 23h prior): SOXX 18×, MU 12.67×,
QBTS 12.55×, DRAM 10.31× — none surfaced anywhere else in the system.

The pure builder mirrors the verdict ladder of
``ArticleStore.ticker_news_burst`` (the daemon's in-process counterpart) byte
for byte — same thresholds, same baseline_per_h floor at 0.5, same sort,
same shape. Tests against ``ArticleStore.ticker_news_burst`` already pin
the verdict semantics; this runner is the endpoint-side reuse so the
dashboard read does not race the writer connection.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Iterable

from analytics.trend_velocity import _parse_ts

# Verdict thresholds — must mirror ``ArticleStore.ticker_news_burst``.
BLAZING_SPIKE = 10.0
BLAZING_MIN_COUNT = 5
HOT_SPIKE = 5.0
HOT_MIN_COUNT = 3
WARMING_SPIKE = 2.0
WARMING_MIN_COUNT = 2

BASELINE_PER_H_FLOOR = 0.5
TICKER_MIN_LEN = 2
TICKER_MAX_LEN = 8
FETCH_LIMIT = 6000


def _normalise_tickers(raw: Iterable[str] | None) -> list[str]:
    out: list[str] = []
    for r in raw or []:
        if not r:
            continue
        t = str(r).strip().upper()
        if len(t) < TICKER_MIN_LEN or len(t) > TICKER_MAX_LEN:
            continue
        if t not in out:
            out.append(t)
    return out


def _classify(spike: float | None, count_window: int) -> str:
    if count_window == 0:
        return "COLD"
    if spike is not None and spike >= BLAZING_SPIKE and count_window >= BLAZING_MIN_COUNT:
        return "BLAZING"
    if spike is not None and spike >= HOT_SPIKE and count_window >= HOT_MIN_COUNT:
        return "HOT"
    if spike is not None and spike >= WARMING_SPIKE and count_window >= WARMING_MIN_COUNT:
        return "WARMING"
    return "NORMAL"


def build_ticker_news_burst(
    articles: Iterable[dict],
    tickers: Iterable[str],
    window_h: float = 1.0,
    baseline_h: float = 24.0,
    now: datetime | None = None,
) -> dict:
    """Pure builder.

    ``articles`` is any iterable of dicts with at least ``first_seen`` and
    ``title`` keys. ``tickers`` is the universe to evaluate (no auto-discover
    — the held + watched book is the canonical input). Rows are bucketed
    into [window_start, now) and [baseline_start, window_start) and counted
    per ticker via word-boundary regex (``$TICKER`` and bare ``TICKER``
    both match).

    Returns a stable shape so the endpoint can ``jsonify`` it directly and
    chat helpers can rely on every key being present.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    window_h = max(float(window_h), 0.05)
    baseline_h = max(float(baseline_h), window_h * 1.5)

    cutoff_window = now - timedelta(hours=window_h)
    cutoff_baseline = now - timedelta(hours=baseline_h + window_h)

    clean = _normalise_tickers(tickers)
    if not clean:
        return {
            "generated_at": now.isoformat(),
            "window_h": round(window_h, 2),
            "baseline_h": round(baseline_h, 2),
            "n_window": 0,
            "n_baseline": 0,
            "by_ticker": [],
            "hottest": None,
            "n_hot": 0,
            "verdict": "NO_DATA",
            "headline": "no tickers supplied",
        }

    patterns = {
        t: re.compile(rf"\b\${{0,1}}{re.escape(t)}\b") for t in clean
    }

    c_win: dict[str, int] = {t: 0 for t in clean}
    c_base: dict[str, int] = {t: 0 for t in clean}
    n_window = 0
    n_baseline = 0

    for art in articles:
        ts = _parse_ts(art.get("first_seen"))
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= now:
            # future-dated rows are excluded from both windows; cannot count
            continue
        if ts >= cutoff_window:
            n_window += 1
            in_window = True
        elif ts >= cutoff_baseline:
            n_baseline += 1
            in_window = False
        else:
            continue
        title = art.get("title") or ""
        if not title:
            continue
        for t, pat in patterns.items():
            if pat.search(title):
                if in_window:
                    c_win[t] += 1
                else:
                    c_base[t] += 1

    out: list[dict] = []
    n_hot = 0
    for t in clean:
        cw = c_win[t]
        cb = c_base[t]
        base_per_h = cb / baseline_h if baseline_h > 0 else 0.0
        if cb == 0 and cw == 0:
            spike: float | None = None
        elif cb == 0:
            # treat zero baseline as ≤0.5/h to avoid divide-by-near-zero
            spike = float(cw) / BASELINE_PER_H_FLOOR
        else:
            spike = cw / max(base_per_h, BASELINE_PER_H_FLOOR)

        verdict = _classify(spike, cw)
        if verdict in ("HOT", "BLAZING"):
            n_hot += 1

        out.append({
            "ticker": t,
            "count_window": cw,
            "count_baseline": cb,
            "baseline_per_h": round(base_per_h, 2),
            "spike": round(spike, 2) if spike is not None else None,
            "verdict": verdict,
        })

    out.sort(
        key=lambda r: (
            -(r["spike"] if r["spike"] is not None else -1.0),
            -r["count_window"],
            r["ticker"],
        )
    )

    hottest: str | None = None
    for r in out:
        if r["verdict"] in ("BLAZING", "HOT", "WARMING"):
            hottest = r["ticker"]
            break

    if n_window == 0 and n_baseline == 0:
        verdict = "NO_DATA"
        headline = "no live articles in window or baseline"
    elif any(r["verdict"] == "BLAZING" for r in out):
        verdict = "BLAZING"
        top = next(r for r in out if r["verdict"] == "BLAZING")
        headline = (
            f"BLAZING: {top['ticker']} {top['count_window']} in {window_h:.1f}h "
            f"vs {top['baseline_per_h']:.2f}/h baseline "
            f"(spike {top['spike']:.1f}×)"
        )
    elif any(r["verdict"] == "HOT" for r in out):
        verdict = "HOT"
        top = next(r for r in out if r["verdict"] == "HOT")
        headline = (
            f"HOT: {top['ticker']} {top['count_window']} in {window_h:.1f}h "
            f"vs {top['baseline_per_h']:.2f}/h baseline "
            f"(spike {top['spike']:.1f}×)"
        )
    elif any(r["verdict"] == "WARMING" for r in out):
        verdict = "WARMING"
        top = next(r for r in out if r["verdict"] == "WARMING")
        headline = (
            f"WARMING: {top['ticker']} {top['count_window']} in {window_h:.1f}h "
            f"(spike {top['spike']:.1f}×)"
        )
    else:
        verdict = "NORMAL"
        headline = "no held ticker breaking out of its baseline rate"

    return {
        "generated_at": now.isoformat(),
        "window_h": round(window_h, 2),
        "baseline_h": round(baseline_h, 2),
        "n_window": n_window,
        "n_baseline": n_baseline,
        "by_ticker": out,
        "hottest": hottest,
        "n_hot": n_hot,
        "verdict": verdict,
        "headline": headline,
    }
