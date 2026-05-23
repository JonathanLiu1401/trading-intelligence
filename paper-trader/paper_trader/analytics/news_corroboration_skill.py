"""News-corroboration skill — does Opus need a chorus or pick winners on a single signal?

The desk question this answers: **at the moment of each trade, how many
distinct live articles were already on this ticker — and did the trades
entered into a LOUD chorus outperform the trades entered on a SINGLE
fresh signal?**

Existing neighbours each see a *different* slice:

* ``/api/news-age-at-decision-skill`` — *freshness* of the newest article
  at decision time. Says nothing about how many articles agreed.
* ``/api/news-themes`` — the per-ticker decayed-score rollup *right now*.
  Snapshot, not a per-trade outcome attribution.
* ``/api/signal-followthrough`` — high-score signals you ACTED ON vs the
  ones you IGNORED. Selection edge, not a count-of-corroboration vs
  outcome curve.
* ``/api/news-velocity`` — per-ticker Poisson z-score vs baseline.
  Anomaly detection, not a calibration check.
* ``/api/decision-context-completeness`` — news present or absent.
  Binary, not a gradient.

The endpoint answers an asymmetric question the existing surface ducks:
*the bot may need 5 articles agreeing to make a good call (so single-
signal trades systematically lose), OR the bot may pick winners on a
single fresh signal (so corroborating articles just add latency without
edge).* Either pattern is operationally actionable — the first says
"slow the bot down on thin evidence", the second says "stop waiting for
confirmation".

Buckets — count of distinct articles mentioning the ticker in the
``lookback_hours`` window strictly before the trade timestamp:

* ``NO_NEWS`` — 0 articles.
* ``SINGLE`` — 1 article.
* ``SMALL_CHORUS`` — 2..3 articles.
* ``CHORUS`` — 4..9 articles.
* ``FLOOD`` — 10+ articles.

Verdict matrix (compares ``CHORUS+`` vs ``SINGLE`` — the two operational
extremes; ``NO_NEWS`` is reported for inspection but does not move the
verdict):

* ``CORROBORATION_HELPS`` — (CHORUS ∪ FLOOD) mean ≥ SINGLE mean +
  verdict_gap_pct. Multiple agreeing articles is a real edge.
* ``SINGLE_HELPS`` — (CHORUS ∪ FLOOD) mean ≤ SINGLE mean -
  verdict_gap_pct. Chorus adds latency without edge — first-mover trades
  win.
* ``NO_PATTERN`` — both buckets full but gap within tolerance. Article
  count uninformative.
* ``INSUFFICIENT_DATA`` — either compare-bucket below ``min_per_bucket``.

Pure builder. Pre-joined samples in, dict out, never raises.
Observational only — never gates Opus, no caps (AGENTS.md #2/#12).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

NO_NEWS = "NO_NEWS"
SINGLE = "SINGLE"
SMALL_CHORUS = "SMALL_CHORUS"
CHORUS = "CHORUS"
FLOOD = "FLOOD"

# Bucket edges. lo inclusive, hi exclusive. SINGLE is exactly 1.
# A 1-article count rolls into SINGLE; a 2-count rolls into SMALL_CHORUS;
# a 4-count rolls into CHORUS; a 10-count rolls into FLOOD.
_BUCKETS = (
    (NO_NEWS, 0, 1),
    (SINGLE, 1, 2),
    (SMALL_CHORUS, 2, 4),
    (CHORUS, 4, 10),
    (FLOOD, 10, float("inf")),
)
_BUCKET_ORDER = [b[0] for b in _BUCKETS]

# Verdict thresholds.
VERDICT_GAP_PCT = 2.0  # mean-return gap (%) for a directional verdict
MIN_PER_BUCKET = 3      # min samples in EACH of SINGLE and CHORUS+ to fire

CORROBORATION_HELPS = "CORROBORATION_HELPS"
SINGLE_HELPS = "SINGLE_HELPS"
NO_PATTERN = "NO_PATTERN"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if x != x:
        return None
    return float(x)


def _bucket_for(count: Any) -> str:
    """Map an article count to a bucket label.

    Non-int / negative / NaN counts route to NO_NEWS — a join failure
    that says "we couldn't count" is informationally equivalent to
    "no news found" for the verdict. Boolean is rejected (True isn't
    a count of 1)."""
    if isinstance(count, bool) or not isinstance(count, (int, float)):
        return NO_NEWS
    if count != count or count < 0:
        return NO_NEWS
    c = float(count)
    for label, lo, hi in _BUCKETS:
        if lo <= c < hi:
            return label
    return FLOOD


def _summarise(samples: list[dict]) -> dict:
    n = len(samples)
    if n == 0:
        return {"n": 0, "mean_pct": None, "median_pct": None, "win_rate": None}
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


def build_news_corroboration_skill(
    samples: Sequence[dict] | None,
    *,
    now: datetime | None = None,
    min_per_bucket: int = MIN_PER_BUCKET,
    verdict_gap_pct: float = VERDICT_GAP_PCT,
) -> dict:
    """Build the corroboration-count-vs-realized-return verdict.

    Each sample is shaped::

        {
            "trade_id": int | None,
            "trade_ts": iso str,
            "ticker": str,
            "action": str,                  # BUY / SELL / ...
            "article_count": int | None,    # corroborating articles
            "realized_pct": float,          # exit return OR mark return
            "closed": bool,
        }

    Returns a stable envelope::

        {
            as_of, verdict, headline,
            n_samples,
            buckets: {NO_NEWS/SINGLE/SMALL_CHORUS/CHORUS/FLOOD:
                       {n, mean_pct, median_pct, win_rate}},
            thresholds, samples: [first 50 raw rows]
        }

    Pure — never raises. Malformed samples are silently dropped.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    norm: list[dict] = []
    for s in samples or ():
        if not isinstance(s, dict):
            continue
        realized = _num(s.get("realized_pct"))
        if realized is None:
            continue
        count = s.get("article_count")
        bucket = _bucket_for(count)
        # Normalise the surfaced count: anything non-numeric is reported
        # as 0 so the sample card never shows "None corroborating".
        if isinstance(count, bool) or not isinstance(count, (int, float)):
            count_out = 0
        elif count != count or count < 0:
            count_out = 0
        else:
            count_out = int(count)
        norm.append({
            "trade_id": s.get("trade_id"),
            "trade_ts": s.get("trade_ts"),
            "ticker": s.get("ticker"),
            "action": s.get("action"),
            "article_count": count_out,
            "realized_pct": realized,
            "closed": bool(s.get("closed", False)),
            "bucket": bucket,
        })

    n_total = len(norm)
    bucket_agg = {
        b: _summarise([x for x in norm if x["bucket"] == b])
        for b in _BUCKET_ORDER
    }

    # CHORUS+ is the operational "loud agreement" extreme.
    chorus_plus = [
        x for x in norm if x["bucket"] in (CHORUS, FLOOD)
    ]
    chorus_agg = _summarise(chorus_plus)
    single_agg = bucket_agg[SINGLE]

    if (
        n_total == 0
        or chorus_agg["n"] < min_per_bucket
        or single_agg["n"] < min_per_bucket
    ):
        verdict = INSUFFICIENT_DATA
        headline = (
            f"INSUFFICIENT_DATA — need ≥{min_per_bucket} samples in both "
            f"SINGLE ({single_agg['n']}) and CHORUS+ ({chorus_agg['n']}) "
            f"to call corroboration calibration."
        )
    else:
        gap = chorus_agg["mean_pct"] - single_agg["mean_pct"]
        if gap >= verdict_gap_pct:
            verdict = CORROBORATION_HELPS
            headline = (
                f"CORROBORATION_HELPS — CHORUS+ trades returned "
                f"{chorus_agg['mean_pct']:+.2f}% mean "
                f"({chorus_agg['n']}) vs SINGLE "
                f"{single_agg['mean_pct']:+.2f}% ({single_agg['n']}); "
                f"+{gap:.2f}pp — chorus is real edge."
            )
        elif gap <= -verdict_gap_pct:
            verdict = SINGLE_HELPS
            headline = (
                f"SINGLE_HELPS — CHORUS+ trades returned "
                f"{chorus_agg['mean_pct']:+.2f}% mean "
                f"({chorus_agg['n']}) vs SINGLE "
                f"{single_agg['mean_pct']:+.2f}% ({single_agg['n']}); "
                f"{gap:.2f}pp — chorus adds latency without edge."
            )
        else:
            verdict = NO_PATTERN
            headline = (
                f"NO_PATTERN — CHORUS+ mean "
                f"{chorus_agg['mean_pct']:+.2f}% ({chorus_agg['n']}) "
                f"vs SINGLE {single_agg['mean_pct']:+.2f}% "
                f"({single_agg['n']}); gap {gap:+.2f}pp within "
                f"±{verdict_gap_pct:.1f}pp tolerance — count "
                f"uninformative."
            )

    by_priority = sorted(
        norm,
        key=lambda x: (0 if x["closed"] else 1, x.get("trade_ts") or ""),
    )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "verdict": verdict,
        "headline": headline,
        "n_samples": n_total,
        "buckets": bucket_agg,
        "chorus_plus": chorus_agg,
        "thresholds": {
            "min_per_bucket": min_per_bucket,
            "verdict_gap_pct": verdict_gap_pct,
        },
        "samples": [
            {
                "trade_id": x["trade_id"],
                "trade_ts": x["trade_ts"],
                "ticker": x["ticker"],
                "action": x["action"],
                "article_count": x["article_count"],
                "realized_pct": round(x["realized_pct"], 2),
                "closed": x["closed"],
                "bucket": x["bucket"],
            }
            for x in by_priority[:50]
        ],
    }
