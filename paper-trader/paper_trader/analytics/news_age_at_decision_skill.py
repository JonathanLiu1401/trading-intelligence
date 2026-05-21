"""News-age-at-decision skill — does fresher news at trade time → better outcome?

The desk question this answers: **when Opus pulled the trigger, was the most
recent live article on that ticker fresh (minutes old) or stale (hours old) —
and does that gap predict the realized return?**

Existing neighbours each see a *different* slice:

* ``/api/news-to-trade-lag`` — measures the gap from the *first* article on a
  catalyst to the *first* trade on it. A "we're slow off the line" metric.
* ``/api/news-edge`` / ``/api/source-edge`` — per-source predictive edge of
  scored headlines. Says nothing about freshness at decision time.
* ``/api/decision-context-completeness`` / ``/api/position-rationale`` —
  whether the trade had any news vs none. Binary, not a gradient.

None answer the *gradient* question: among trades where news existed, did
trades on fresh news outperform trades on stale news? That's the verdict
this endpoint produces.

Pure builder. Pre-joined samples in, dict out, never raises. Observational
only — never gates Opus, no caps (AGENTS.md #2/#12).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

# Age buckets in minutes. The boundaries are deliberately wide because at
# N=11 trades any finer cut buries the signal in single-sample noise. They
# map to operator-readable spans: "minutes-fresh", "hours-fresh",
# "day-old", "stale", "no-news".
BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("FRESH_LT_60M", 0.0, 60.0),
    ("HOURS_1_TO_6", 60.0, 360.0),
    ("HOURS_6_TO_24", 360.0, 1440.0),
    ("STALE_GT_24H", 1440.0, float("inf")),
)
NO_NEWS_BUCKET = "NO_NEWS"

# Verdict thresholds. The mean-return gap between fresh and stale must
# exceed this for a directional verdict to fire. 2.0% is loose enough to
# trigger at N≈10 per bucket without being trivially within sampling
# noise; tighter (e.g. 0.5) flips on a single outlier round-trip.
VERDICT_GAP_PCT = 2.0
# Minimum samples in *each* of the FRESH and STALE compare buckets before
# a directional verdict will fire. Below this, the verdict stays
# INSUFFICIENT_DATA and only the per-bucket cards are emitted.
MIN_PER_BUCKET = 3

# Verdict labels — module constants so tests and callers never depend on
# string literals at the call site.
FRESH_NEWS_BETTER = "FRESH_NEWS_BETTER"
STALE_NEWS_BETTER = "STALE_NEWS_BETTER"
NO_PATTERN = "NO_PATTERN"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# Action verbs counted as buys / sells for the entry-direction filter.
# Option verbs are included so leveraged-ETF and option round-trips are
# both visible. HOLD / NO_DECISION / BLOCKED never reach this builder
# (the route filters to FILLED only).
_BUY_VERBS = ("BUY", "BUY_CALL", "BUY_PUT")
_SELL_VERBS = ("SELL", "SELL_CALL", "SELL_PUT")


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if x != x:  # NaN
        return None
    return float(x)


def _bucket_for(age_min: float | None) -> str:
    """Map an article-age-in-minutes to its bucket label.

    ``None`` → NO_NEWS. A non-finite or negative age also → NO_NEWS
    (defensive — a negative age means the joined article is *after* the
    trade, which is a bug in the joiner, not a valid sample)."""
    if age_min is None:
        return NO_NEWS_BUCKET
    a = _num(age_min)
    if a is None or a < 0 or a != a or a == float("inf") and False:
        return NO_NEWS_BUCKET
    for label, lo, hi in BUCKETS:
        if lo <= a < hi:
            return label
    return NO_NEWS_BUCKET


def _summarise(samples: list[dict]) -> dict:
    """Per-bucket aggregates over a list of {realized_pct} dicts.

    Returns ``{n, mean_pct, median_pct, win_rate}``. All keys are always
    present — missing data degrades to ``None`` (n is always int)."""
    n = len(samples)
    if n == 0:
        return {
            "n": 0,
            "mean_pct": None,
            "median_pct": None,
            "win_rate": None,
        }
    rets = sorted([s["realized_pct"] for s in samples])
    mean = sum(rets) / n
    if n % 2 == 1:
        med = rets[n // 2]
    else:
        med = (rets[n // 2 - 1] + rets[n // 2]) / 2.0
    wins = sum(1 for r in rets if r > 0)
    return {
        "n": n,
        "mean_pct": round(mean, 2),
        "median_pct": round(med, 2),
        "win_rate": round(wins / n * 100.0, 1),
    }


def build_news_age_at_decision_skill(
    samples: Sequence[dict] | None,
    *,
    now: datetime | None = None,
    min_per_bucket: int = MIN_PER_BUCKET,
    verdict_gap_pct: float = VERDICT_GAP_PCT,
) -> dict:
    """Build the news-age-vs-realized-return verdict.

    Each sample is shaped::

        {
            "trade_id": int | None,
            "trade_ts": iso str,
            "ticker": str,
            "action": str,                  # BUY / SELL / BUY_CALL / ...
            "freshest_article_age_min": float | None,
            "realized_pct": float,          # mark-to-current OR exit return
            "closed": bool,                 # True if exit price, False if mark
        }

    The route is responsible for the join: it reads ``trades``, pulls the
    newest article per ticker before each trade ts from ``articles.db``,
    and computes realized_pct either from a matched SELL (closed round-trip)
    or from the current market mark (open position).

    Returns a stable envelope::

        {
            as_of, verdict, headline,
            n_samples, n_with_news, n_no_news,
            buckets: {LABEL: {n, mean_pct, median_pct, win_rate}, ...},
            verdict_gap_pct, min_per_bucket,
            samples: [first 50 raw rows, mark-then-closed],
        }

    Pure — never raises. Malformed samples are silently dropped.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # --- Normalise and bucket samples --------------------------------
    norm: list[dict] = []
    for s in samples or ():
        if not isinstance(s, dict):
            continue
        realized = _num(s.get("realized_pct"))
        if realized is None:
            continue
        action = s.get("action")
        if not isinstance(action, str):
            continue
        verb = action.split(None, 1)[0].upper() if action else ""
        if verb not in _BUY_VERBS and verb not in _SELL_VERBS:
            # Only entries (buys) and exits (sells) are valid samples —
            # HOLDs have no realized return to attribute.
            continue
        age = _num(s.get("freshest_article_age_min"))
        bucket = _bucket_for(age)
        norm.append({
            "trade_id": s.get("trade_id"),
            "trade_ts": s.get("trade_ts"),
            "ticker": s.get("ticker"),
            "action": action,
            "verb": verb,
            "freshest_article_age_min": age,
            "realized_pct": realized,
            "closed": bool(s.get("closed", False)),
            "bucket": bucket,
        })

    n_total = len(norm)
    n_no_news = sum(1 for x in norm if x["bucket"] == NO_NEWS_BUCKET)
    n_with_news = n_total - n_no_news

    # --- Per-bucket aggregates ---------------------------------------
    bucket_labels = [b[0] for b in BUCKETS] + [NO_NEWS_BUCKET]
    bucket_agg: dict[str, dict] = {}
    for label in bucket_labels:
        rows = [x for x in norm if x["bucket"] == label]
        bucket_agg[label] = _summarise(rows)

    # --- Verdict logic -----------------------------------------------
    # Compare the freshest news bucket (FRESH_LT_60M) against the
    # stale-news bucket (STALE_GT_24H). HOURS_1_TO_6 and HOURS_6_TO_24
    # are surfaced in the cards but do not move the verdict — the
    # operator question is the *extremes*: does minute-fresh news beat
    # day-old news?
    fresh = bucket_agg["FRESH_LT_60M"]
    stale = bucket_agg["STALE_GT_24H"]
    if (
        n_total == 0
        or fresh["n"] < min_per_bucket
        or stale["n"] < min_per_bucket
    ):
        verdict = INSUFFICIENT_DATA
        headline = (
            f"INSUFFICIENT_DATA — need ≥{min_per_bucket} samples in both "
            f"FRESH_LT_60M ({fresh['n']}) and STALE_GT_24H ({stale['n']}) "
            f"buckets to call the gradient."
        )
    else:
        gap = fresh["mean_pct"] - stale["mean_pct"]
        if gap >= verdict_gap_pct:
            verdict = FRESH_NEWS_BETTER
            headline = (
                f"FRESH_NEWS_BETTER — trades on <60m-old news returned "
                f"{fresh['mean_pct']:+.2f}% mean ({fresh['n']} trades) vs "
                f"{stale['mean_pct']:+.2f}% on day-old news "
                f"({stale['n']} trades); +{gap:.2f}pp edge to speed."
            )
        elif gap <= -verdict_gap_pct:
            verdict = STALE_NEWS_BETTER
            headline = (
                f"STALE_NEWS_BETTER — trades on <60m-old news returned "
                f"{fresh['mean_pct']:+.2f}% mean ({fresh['n']} trades) vs "
                f"{stale['mean_pct']:+.2f}% on day-old news "
                f"({stale['n']} trades); {gap:.2f}pp; speed is hurting."
            )
        else:
            verdict = NO_PATTERN
            headline = (
                f"NO_PATTERN — fresh-news mean {fresh['mean_pct']:+.2f}% "
                f"({fresh['n']}) vs stale {stale['mean_pct']:+.2f}% "
                f"({stale['n']}); gap {gap:+.2f}pp within "
                f"±{verdict_gap_pct:.1f}pp tolerance."
            )

    # Cards: emit at most 50 raw samples (closed first, then open marks)
    # so the operator can spot-check the join.
    by_priority = sorted(
        norm,
        key=lambda x: (0 if x["closed"] else 1, x.get("trade_ts") or ""),
    )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "verdict": verdict,
        "headline": headline,
        "n_samples": n_total,
        "n_with_news": n_with_news,
        "n_no_news": n_no_news,
        "buckets": bucket_agg,
        "thresholds": {
            "min_per_bucket": min_per_bucket,
            "verdict_gap_pct": verdict_gap_pct,
            "bucket_edges_min": [
                {"label": b[0], "lo_min": b[1], "hi_min": (b[2] if b[2] != float("inf") else None)}
                for b in BUCKETS
            ],
        },
        "samples": [
            {
                "trade_id": x["trade_id"],
                "trade_ts": x["trade_ts"],
                "ticker": x["ticker"],
                "action": x["action"],
                "freshest_article_age_min": (
                    round(x["freshest_article_age_min"], 1)
                    if x["freshest_article_age_min"] is not None
                    else None
                ),
                "realized_pct": round(x["realized_pct"], 2),
                "closed": x["closed"],
                "bucket": x["bucket"],
            }
            for x in by_priority[:50]
        ],
    }
