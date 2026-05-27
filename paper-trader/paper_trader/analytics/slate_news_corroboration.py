"""Slate × news corroboration — does the live news pulse agree with the
ML scorer's WATCHLIST slate?

The desk already exposes the two ingredients separately:

  * ``/api/scorer-opportunities`` ranks the watchlist by
    ``pred_5d_return_pct`` (a quant verdict).
  * ``/api/sector-pulse`` / ``/api/news-themes`` / ``/api/sector-coherence``
    report what the live wire is saying (a narrative verdict).

But a trader sitting in cash with the scorer pointing at 12 names needs
a *single* surface that answers the next question they actually have:
"for each name the model is recommending, is the news flow corroborating
or silent?". Today the per-opportunity ``news_count`` field embedded in
``/api/scorer-opportunities`` is baked into the model's input FEATURES at
scoring time — it isn't a fresh live read and it gives no headline, no
``ai_score``, no urgent flag, and no overall ladder over the slate.

This module is the missing join. Given the scorer slate and a per-ticker
news pulse (the same dict shape returned by
``dashboard._ticker_news_pulse``: ``{ticker → {n, urgent, top_title,
top_url, top_score}}``), classify each opportunity into a fixed ladder
and emit an overall verdict over the buy-candidate set:

| Per-name verdict       | Trigger                                                                |
|------------------------|------------------------------------------------------------------------|
| ``HOT_CONVERGENT``     | pred ≥ floor AND (urgent ≥ 1 OR (n ≥ hot_min_count AND max_score ≥ hot_min_score)) |
| ``CONVERGENT``         | pred ≥ floor AND n ≥ convergent_min_count AND max_score ≥ convergent_min_score      |
| ``THIN_NEWS``          | pred ≥ floor AND 0 < n < convergent_min_count                         |
| ``QUANT_ONLY``         | pred ≥ floor AND n == 0                                               |
| ``SUB_THRESHOLD``      | pred < floor                                                          |

| Overall verdict           | Meaning                                                                |
|---------------------------|------------------------------------------------------------------------|
| ``NO_SLATE``              | empty opportunities list                                              |
| ``STRONG_CORROBORATION``  | ≥ 1 HOT_CONVERGENT AND ≥ 50% of buy-candidates are CONVERGENT or HOT_CONVERGENT |
| ``QUANT_LEAD``            | ≥ 50% of buy-candidates are QUANT_ONLY (model leads, narrative quiet)  |
| ``THIN``                  | ≥ 50% of buy-candidates are THIN_NEWS                                  |
| ``MIXED_CORROBORATION``   | everything else with at least one buy-candidate                       |

Pure & offline. No DB, no network. Walks the precomputed inputs and
returns a JSON-ready dict. Same ``_safe`` discipline as adjacent
analytics modules: a garbage row contributes nothing, an empty slate
degrades to a deterministic ``NO_SLATE`` verdict. Never raises.

Advisory only — never gates Opus, no caps (AGENTS.md invariants #2 / #12).
"""
from __future__ import annotations

from typing import Any

DEFAULT_MIN_PRED_PCT = 1.0
DEFAULT_HOT_MIN_COUNT = 3
DEFAULT_HOT_MIN_SCORE = 6.0
DEFAULT_CONVERGENT_MIN_COUNT = 2
DEFAULT_CONVERGENT_MIN_SCORE = 4.0

# Soft band so a slightly noisy hot_min_score isn't unreachable on a fast wire.
MIN_HOT_MIN_SCORE = 1.0
MAX_HOT_MIN_SCORE = 10.0

_BUY_VERDICTS = frozenset({"STRONG_HOLD", "HOLD"})


def _f(x, default: float = 0.0) -> float:
    """Float coercion; garbage degrades to ``default``, never raises."""
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _i(x, default: int = 0) -> int:
    """Int coercion; garbage degrades to ``default``, never raises."""
    try:
        if x is None:
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def _z(v, ndigits: int = 2):
    """Round; fold ``-0.0 → 0.0``; None / non-numeric → None."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _classify_one(
    pred_pct: float,
    n: int,
    urgent: int,
    max_score: float,
    *,
    min_pred_pct: float,
    hot_min_count: int,
    hot_min_score: float,
    convergent_min_count: int,
    convergent_min_score: float,
) -> str:
    """Per-name verdict. Pred-floor first (so a name the planner would
    skip never counts toward the news ladder), then the news bands."""
    if pred_pct < min_pred_pct:
        return "SUB_THRESHOLD"
    if n <= 0:
        return "QUANT_ONLY"
    if urgent >= 1 or (n >= hot_min_count and max_score >= hot_min_score):
        return "HOT_CONVERGENT"
    if n >= convergent_min_count and max_score >= convergent_min_score:
        return "CONVERGENT"
    return "THIN_NEWS"


def build_slate_news_corroboration(
    opportunities: Any,
    news_pulse: Any,
    *,
    min_pred_pct: float = DEFAULT_MIN_PRED_PCT,
    hot_min_count: int = DEFAULT_HOT_MIN_COUNT,
    hot_min_score: float = DEFAULT_HOT_MIN_SCORE,
    convergent_min_count: int = DEFAULT_CONVERGENT_MIN_COUNT,
    convergent_min_score: float = DEFAULT_CONVERGENT_MIN_SCORE,
) -> dict:
    """Pure builder. ``opportunities`` is the scorer slate (same shape as
    ``/api/scorer-opportunities``'s ``opportunities`` list).
    ``news_pulse`` is the per-ticker dict from
    ``dashboard._ticker_news_pulse``. Returns a deterministic JSON-ready
    report with the per-name and overall verdict ladder. Never raises."""

    # ── Sanitise inputs ────────────────────────────────────────────────
    floor = _clamp(_f(min_pred_pct, DEFAULT_MIN_PRED_PCT), 0.0, 100.0)
    hot_n = max(_i(hot_min_count, DEFAULT_HOT_MIN_COUNT), 1)
    hot_s = _clamp(_f(hot_min_score, DEFAULT_HOT_MIN_SCORE),
                   MIN_HOT_MIN_SCORE, MAX_HOT_MIN_SCORE)
    conv_n = max(_i(convergent_min_count, DEFAULT_CONVERGENT_MIN_COUNT), 1)
    conv_s = _clamp(_f(convergent_min_score, DEFAULT_CONVERGENT_MIN_SCORE),
                    MIN_HOT_MIN_SCORE, MAX_HOT_MIN_SCORE)

    effective = {
        "min_pred_pct": _z(floor),
        "hot_min_count": hot_n,
        "hot_min_score": _z(hot_s),
        "convergent_min_count": conv_n,
        "convergent_min_score": _z(conv_s),
    }

    # ── Walk the slate ────────────────────────────────────────────────
    pulse: dict[str, dict] = {}
    if isinstance(news_pulse, dict):
        # Normalise: callers may pass {TICKER: row}; row may carry stale
        # keys. We touch only n/urgent/top_title/top_url/top_score.
        for k, v in news_pulse.items():
            if not isinstance(v, dict):
                continue
            ku = str(k).upper().strip()
            if ku:
                pulse[ku] = v

    per_name: list[dict] = []
    counts = {
        "HOT_CONVERGENT": 0,
        "CONVERGENT": 0,
        "THIN_NEWS": 0,
        "QUANT_ONLY": 0,
        "SUB_THRESHOLD": 0,
    }
    total_articles = 0
    total_urgent = 0

    if not isinstance(opportunities, list):
        opportunities = []

    for opp in opportunities:
        if not isinstance(opp, dict):
            continue
        ticker = str(opp.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        pred = _f(opp.get("pred_5d_return_pct"), 0.0)
        scorer_verdict = str(opp.get("verdict") or "").upper()
        row = pulse.get(ticker) or {}
        n = max(_i(row.get("n"), 0), 0)
        urgent = max(_i(row.get("urgent"), 0), 0)
        max_score = max(_f(row.get("top_score"), 0.0), 0.0)
        top_title = row.get("top_title")
        if top_title is not None and not isinstance(top_title, str):
            top_title = None
        top_url = row.get("top_url")
        if top_url is not None and not isinstance(top_url, str):
            top_url = None

        total_articles += n
        total_urgent += urgent

        verdict = _classify_one(
            pred, n, urgent, max_score,
            min_pred_pct=floor,
            hot_min_count=hot_n,
            hot_min_score=hot_s,
            convergent_min_count=conv_n,
            convergent_min_score=conv_s,
        )
        counts[verdict] += 1

        per_name.append({
            "ticker": ticker,
            "scorer_verdict": scorer_verdict or None,
            "pred_5d_return_pct": _z(pred),
            "news_n": n,
            "news_urgent": urgent,
            "news_max_score": _z(max_score),
            "news_top_title": top_title,
            "news_top_url": top_url,
            "verdict": verdict,
        })

    n_total = len(per_name)
    if n_total == 0:
        return {
            "verdict": "NO_SLATE",
            "headline": "no scorer opportunities — nothing to corroborate",
            "constraints": effective,
            "n_total": 0,
            "n_buy_candidates": 0,
            "counts": counts,
            "totals": {
                "articles": 0,
                "urgent": 0,
            },
            "by_name": [],
        }

    # ── Buy-candidate cohort: scorer says BUY (STRONG_HOLD/HOLD) AND
    # pred ≥ floor. Mirrors the deployment_plan filter; the operator
    # wants to know what the *actionable* names look like, not what the
    # SCREENed-out tail looks like.
    buy_candidates = [
        r for r in per_name
        if (r["scorer_verdict"] in _BUY_VERDICTS)
        and (r["verdict"] != "SUB_THRESHOLD")
    ]
    n_buy = len(buy_candidates)

    # Per-name verdict counts within the buy cohort (a separate dict so
    # the overall verdict ladder reasons over the actionable cohort,
    # not the full slate which may include EXIT/NEUTRAL names).
    cohort_counts = {
        "HOT_CONVERGENT": 0,
        "CONVERGENT": 0,
        "THIN_NEWS": 0,
        "QUANT_ONLY": 0,
    }
    for r in buy_candidates:
        v = r["verdict"]
        if v in cohort_counts:
            cohort_counts[v] += 1

    # ── Overall verdict ladder ────────────────────────────────────────
    if n_buy == 0:
        verdict = "NO_SLATE"
        headline = (
            "scorer publishes %d name(s) but 0 actionable buy-candidates "
            "(no STRONG_HOLD/HOLD above pred floor %.1f%%)"
            % (n_total, floor)
        )
    else:
        hot = cohort_counts["HOT_CONVERGENT"]
        conv = cohort_counts["CONVERGENT"]
        thin = cohort_counts["THIN_NEWS"]
        quant = cohort_counts["QUANT_ONLY"]
        converged_pct = 100.0 * (hot + conv) / n_buy
        thin_pct = 100.0 * thin / n_buy
        quant_pct = 100.0 * quant / n_buy

        if hot >= 1 and converged_pct >= 50.0:
            verdict = "STRONG_CORROBORATION"
        elif quant_pct >= 50.0:
            verdict = "QUANT_LEAD"
        elif thin_pct >= 50.0:
            verdict = "THIN"
        else:
            verdict = "MIXED_CORROBORATION"

        # Headline is built from the cohort counts so the operator sees
        # the actionable mix at a glance.
        headline = (
            "%d buy-candidate(s): %d HOT, %d CONVERGENT, %d THIN, %d QUANT_ONLY"
            % (n_buy, hot, conv, thin, quant)
        )

    return {
        "verdict": verdict,
        "headline": headline,
        "constraints": effective,
        "n_total": n_total,
        "n_buy_candidates": n_buy,
        "counts": counts,
        "cohort_counts": cohort_counts,
        "totals": {
            "articles": total_articles,
            "urgent": total_urgent,
        },
        "by_name": per_name,
    }
