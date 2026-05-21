"""Conviction-deployment curve — does the bot's BUY size scale with the
news-signal conviction at the moment of entry?

Every other behavioural mirror watches the *quality* of the trade:

* ``add_discipline`` watches the moment of an ADD (chasing vs averaging
  down vs stacking).
* ``trade_asymmetry`` is the disposition gap (winners cut short, losers
  ridden long).
* ``loser_autopsy`` / ``winner_autopsy`` narrate closed round-trips.
* ``news_to_trade_lag`` measures reactivity — how fast the bot acts on
  fresh news.
* ``trade_attribution`` enumerates the highest-scored articles
  preceding each FILLED trade.

None of them ask the **conviction-to-size** question: when the news
ai_score on a ticker is screaming 9.5/10, does the bot deploy more
capital than when it's a tepid 6.5/10? Or is there a structural cap
(40% of book per name, an unwritten persona rule, an Opus risk anchor)
that *flattens* the curve so all entries look the same size regardless
of how loud the signal is? The live observation that motivated this
endpoint: NVDA at ai_score 10, news SURGING, z-score 105 — and the
position is sized at 44% of book with 55% cash idle. Is that an
explicit conviction-aware sizing? Or is the bot sizing on a habit
that ignores the score?

``build_conviction_deployment`` walks the BUY trade ledger, finds the
peak ai_score in a configurable window *before* each trade (default
6h — long enough for the catalyst article to land, short enough that
"the same news from yesterday" doesn't poison the attribution), then
classifies each BUY by ai_score bucket and reports the deployment
fraction (trade.value / equity_at_trade). The output has TWO
parallel surfaces — both ship at every sample size:

* ``evidence`` — per-trade chronological table the operator can eyeball
  even at N=1. This is the primary surface at low sample size.
* ``buckets`` — bucket aggregates with median / max / total — useful
  once each bucket has n≥3.

Verdict is a roll-up over buckets (not evidence): ``MONOTONIC`` (size
scales with conviction), ``FLAT`` (no relationship), ``INVERTED``
(smaller at higher conviction — the opposite of what's expected), or
``INSUFFICIENT`` (too few buckets meet the per-bucket sample floor).
The roll-up is the falsifiable claim — and it gates on density, not
total trade count, because two trades in the "9+" bucket tell you
nothing about the curve.

Pure builder. No DB, no network. Articles + trades + equity_curve in,
dict out, never raises on garbage rows. Articles missing for trades
older than the articles.db retention window degrade per-trade to
``score_unavailable=true`` (the memory note on shallow article
history) — they never throw and they never count toward bucket
density.

Observational only — never gates Opus, no caps, no claim about the
*outcome* of conviction-scaled sizing (that's ``add_discipline`` /
``loser_autopsy`` / ``winner_autopsy`` territory). This surface is
strictly about the bot's *deployment behaviour at entry*.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from datetime import datetime, timezone
from statistics import median
from typing import Any, Sequence

# Default lookback for the "peak ai_score before this BUY" join. Six
# hours is the catalyst-window the briefing-coverage-audit and
# news-to-trade-lag endpoints use; consistent with the rest of the
# behavioural-mirror surface.
DEFAULT_WINDOW_HOURS_PRE_TRADE = 6.0

# Per-bucket sample floor before a bucket contributes to the
# MONOTONIC / FLAT / INVERTED verdict. Below this the bucket still
# emits its row (n_buys + the evidence list) but the bucket's
# median is not load-bearing on the verdict.
STABLE_MIN_PER_BUCKET = 3

# Minimum number of populated buckets (i.e. with n≥STABLE_MIN_PER_BUCKET)
# before any non-INSUFFICIENT verdict can fire. With one populated bucket
# you cannot tell MONOTONIC from FLAT.
STABLE_MIN_POPULATED_BUCKETS = 2

# The bucket edges, inclusive lower / exclusive upper, except the top
# bucket which is open-ended (≥9.0). Six bins span the [0, 10] ai_score
# range with finer granularity at the high-conviction end (where the
# bot's behaviour matters more — a 9.5 vs 9.8 difference is more
# diagnostic than a 2.0 vs 3.0 difference).
_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("<6", 0.0, 6.0),
    ("6-7", 6.0, 7.0),
    ("7-8", 7.0, 8.0),
    ("8-9", 8.0, 9.0),
    ("9+", 9.0, None),
)

# Deployment-pct delta between adjacent populated buckets that flips
# the verdict from FLAT to MONOTONIC. Below this the slope is noise.
# Tuned conservatively — a 5pp difference is meaningful when 100% of
# book is in play; a 1pp difference is float jitter.
MONOTONIC_SLOPE_PP = 5.0


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if x != x:  # NaN
        return None
    return float(x)


def _parse_ts(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _ticker_in_title(title: Any, ticker: str) -> bool:
    """Word-boundary case-sensitive match — same convention as
    ``briefing_coverage_audit`` / ``portfolio_signals``. Avoids
    matching ``MU`` inside ``Museum`` or ``HK`` inside ``HK$``."""
    if not isinstance(title, str) or not isinstance(ticker, str):
        return False
    if not ticker:
        return False
    # \b in regex treats $ as a word boundary, so $NVDA matches NVDA cleanly.
    return re.search(rf"\b{re.escape(ticker)}\b", title) is not None


def _peak_score_for_ticker(
    ticker: str,
    articles: Sequence[dict],
    trade_ts: datetime,
    window_hours: float,
) -> tuple[float | None, dict | None]:
    """Return (peak_ai_score, top_article_row) for ``ticker`` in the
    ``(trade_ts - window, trade_ts]`` interval. If nothing matches,
    returns (None, None) — caller marks the trade ``score_unavailable``.

    Defensive: a malformed article row (missing first_seen, non-numeric
    ai_score, non-string title) is skipped, never crashes the scan."""
    if window_hours <= 0:
        return None, None
    window_start_seconds = trade_ts.timestamp() - window_hours * 3600.0
    trade_ts_seconds = trade_ts.timestamp()
    best_score: float | None = None
    best_row: dict | None = None
    for a in articles:
        if not isinstance(a, dict):
            continue
        ts = _parse_ts(a.get("first_seen"))
        if ts is None:
            continue
        t_s = ts.timestamp()
        if t_s <= window_start_seconds or t_s > trade_ts_seconds:
            continue
        score = _num(a.get("ai_score"))
        if score is None:
            continue
        if not _ticker_in_title(a.get("title"), ticker):
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_row = a
    return best_score, best_row


def _equity_at(
    trade_ts: datetime,
    equity_sorted_by_ts: Sequence[tuple[float, float]],
    fallback: float,
) -> float:
    """Find the equity-curve total_value nearest *before-or-equal* to
    ``trade_ts``. If the trade is older than every equity point, fall
    back to ``fallback`` (typically the starting capital). Returns
    ``fallback`` for an empty equity curve too."""
    if not equity_sorted_by_ts:
        return fallback
    target = trade_ts.timestamp()
    # Find the largest equity ts ≤ target.
    keys = [k for k, _ in equity_sorted_by_ts]
    idx = bisect_right(keys, target) - 1
    if idx < 0:
        return fallback
    return equity_sorted_by_ts[idx][1]


def _bucket_for(score: float | None) -> str | None:
    if score is None:
        return None
    for label, lo, hi in _BUCKETS:
        if hi is None:
            if score >= lo:
                return label
        else:
            if lo <= score < hi:
                return label
    return None


def _empty_buckets() -> list[dict]:
    return [{
        "label": label,
        "n_buys": 0,
        "median_size_pct": None,
        "max_size_pct": None,
        "total_value_usd": 0.0,
    } for label, _lo, _hi in _BUCKETS]


def _classify_verdict(buckets: list[dict]) -> str:
    """MONOTONIC / FLAT / INVERTED / INSUFFICIENT over the populated
    buckets (n ≥ STABLE_MIN_PER_BUCKET). Decision rule:

    * Walk populated buckets in ascending ai_score order.
    * If every consecutive median_size_pct delta is ≥ +MONOTONIC_SLOPE_PP
      and at least one delta is strictly positive ⇒ MONOTONIC.
    * If every consecutive delta is ≤ -MONOTONIC_SLOPE_PP and at least
      one delta is strictly negative ⇒ INVERTED.
    * Otherwise FLAT — the curve has no consistent direction at the
      MONOTONIC_SLOPE_PP grain.
    """
    populated = [b for b in buckets if b["n_buys"] >= STABLE_MIN_PER_BUCKET]
    if len(populated) < STABLE_MIN_POPULATED_BUCKETS:
        return "INSUFFICIENT"
    medians = [b["median_size_pct"] for b in populated]
    # All medians should be non-None by construction once n_buys>0.
    if any(m is None for m in medians):
        return "INSUFFICIENT"
    deltas = [medians[i + 1] - medians[i] for i in range(len(medians) - 1)]
    if all(d >= MONOTONIC_SLOPE_PP for d in deltas) and any(d > 0 for d in deltas):
        return "MONOTONIC"
    if all(d <= -MONOTONIC_SLOPE_PP for d in deltas) and any(d < 0 for d in deltas):
        return "INVERTED"
    return "FLAT"


def build_conviction_deployment(
    trades: Sequence[dict],
    articles: Sequence[dict],
    equity_curve: Sequence[dict],
    *,
    now: datetime | None = None,
    window_hours_pre_trade: float = DEFAULT_WINDOW_HOURS_PRE_TRADE,
    starting_capital: float = 1000.0,
) -> dict:
    """Build the conviction-deployment curve.

    Arguments:
        trades: trade rows in ``store.recent_trades`` shape — at minimum
            ``{timestamp, ticker, action, qty, price, value}``. Order
            irrelevant; the function sorts oldest-first internally.
            Only ``action="BUY"`` (and ``"ADD"``, treated identically)
            on stock-type trades counts; option BUY_CALL / BUY_PUT etc.
            are skipped because their notional carries different
            semantics from a stock-share dollar deployment.
        articles: ``[{title, ai_score, first_seen, ...}]`` from
            ``articles.db``. Must already be ``_LIVE_ONLY_CLAUSE``-
            filtered by the caller (the route applies that SQL clause;
            this builder doesn't know about live-vs-backtest).
        equity_curve: ``[{timestamp, total_value}]`` rows from
            ``store.recent_equity``. Used to compute deployment_pct =
            trade.value / equity_at_trade. Order irrelevant.
        now: optional override for ``as_of`` timestamp (test seam).
        window_hours_pre_trade: how far back the peak-score scan looks
            from each BUY's timestamp. Default 6h.
        starting_capital: fallback equity when the trade pre-dates the
            equity_curve (early-boot trades happen before the first
            equity point is recorded).

    Returns:
        Dict with stable keys regardless of state:

        * ``state`` — NO_DATA / EMERGING / STABLE
        * ``verdict`` — MONOTONIC / FLAT / INVERTED / INSUFFICIENT
        * ``headline`` — one-line operator summary
        * ``n_buys_scanned`` — every BUY considered
        * ``n_buys_with_score`` — count where the peak-score join hit
        * ``n_buys_score_unavailable`` — articles.db retention gap
        * ``window_hours_pre_trade`` — echoed back
        * ``buckets`` — list of 5 buckets, always present
        * ``evidence`` — per-trade chronological rows (oldest first)
        * ``as_of`` — UTC ISO timestamp
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # Normalise + sort equity_curve by timestamp ascending; build the
    # (ts, total_value) lookup once.
    equity_pairs: list[tuple[float, float]] = []
    for e in equity_curve or ():
        if not isinstance(e, dict):
            continue
        ts = _parse_ts(e.get("timestamp"))
        tv = _num(e.get("total_value"))
        if ts is None or tv is None or tv <= 0:
            continue
        equity_pairs.append((ts.timestamp(), tv))
    equity_pairs.sort()

    # Filter trades to BUY-on-stock, sort oldest-first.
    candidates: list[tuple[datetime, dict]] = []
    for t in trades or ():
        if not isinstance(t, dict):
            continue
        action = t.get("action")
        if action not in ("BUY", "ADD"):
            continue
        # Option types carry option-notional semantics — exclude.
        opt_type = t.get("option_type") or t.get("type")
        if isinstance(opt_type, str) and opt_type.upper() in ("CALL", "PUT"):
            continue
        # Some store shapes use type='stock' vs 'option'; respect that.
        ticker = t.get("ticker")
        if not isinstance(ticker, str) or not ticker:
            continue
        ts = _parse_ts(t.get("timestamp"))
        if ts is None:
            continue
        value = _num(t.get("value"))
        if value is None or value <= 0:
            # A non-positive deployment is meaningless for the curve.
            continue
        candidates.append((ts, t))
    candidates.sort(key=lambda r: r[0])

    evidence: list[dict] = []
    bucket_index = {label: i for i, (label, _l, _h) in enumerate(_BUCKETS)}
    bucket_size_pcts: list[list[float]] = [[] for _ in _BUCKETS]
    bucket_values: list[float] = [0.0 for _ in _BUCKETS]
    bucket_counts: list[int] = [0 for _ in _BUCKETS]

    n_with_score = 0
    n_unavailable = 0

    for ts, t in candidates:
        ticker = t["ticker"]
        value = float(t["value"])
        equity_at = _equity_at(ts, equity_pairs, starting_capital)
        size_pct = (value / equity_at) * 100.0 if equity_at > 0 else 0.0
        peak_score, top_row = _peak_score_for_ticker(
            ticker, articles, ts, window_hours_pre_trade,
        )
        score_unavailable = peak_score is None
        bucket_label = _bucket_for(peak_score)
        if bucket_label is not None:
            idx = bucket_index[bucket_label]
            bucket_size_pcts[idx].append(size_pct)
            bucket_values[idx] += value
            bucket_counts[idx] += 1
            n_with_score += 1
        else:
            n_unavailable += 1
        evidence.append({
            "ts": ts.isoformat(timespec="seconds"),
            "ticker": ticker,
            "action": t.get("action"),
            "qty": _num(t.get("qty")),
            "price": _num(t.get("price")),
            "value_usd": round(value, 2),
            "equity_at_trade_usd": round(equity_at, 2),
            "size_pct": round(size_pct, 2),
            "peak_ai_score_pre": peak_score if peak_score is None else round(peak_score, 2),
            "score_unavailable": score_unavailable,
            "bucket": bucket_label,
            "top_article_title": (top_row or {}).get("title") if top_row else None,
            "top_article_source": (top_row or {}).get("source") if top_row else None,
        })

    buckets = _empty_buckets()
    for i, (label, _lo, _hi) in enumerate(_BUCKETS):
        if bucket_counts[i] > 0:
            buckets[i] = {
                "label": label,
                "n_buys": bucket_counts[i],
                "median_size_pct": round(median(bucket_size_pcts[i]), 2),
                "max_size_pct": round(max(bucket_size_pcts[i]), 2),
                "total_value_usd": round(bucket_values[i], 2),
            }

    n_scanned = len(candidates)
    if n_scanned == 0:
        state = "NO_DATA"
        headline = "No BUY trades on the ledger."
        verdict = "INSUFFICIENT"
    elif n_with_score < STABLE_MIN_PER_BUCKET * STABLE_MIN_POPULATED_BUCKETS:
        state = "EMERGING"
        verdict = _classify_verdict(buckets)
        if n_unavailable > 0:
            headline = (
                f"EMERGING — {n_scanned} BUY(s); {n_unavailable} predate "
                f"articles.db retention; verdict {verdict}."
            )
        else:
            headline = (
                f"EMERGING — {n_scanned} BUY(s); verdict {verdict} "
                f"pending per-bucket density (≥{STABLE_MIN_PER_BUCKET})."
            )
    else:
        state = "STABLE"
        verdict = _classify_verdict(buckets)
        # Top-conviction bucket bias mention if MONOTONIC/INVERTED.
        top_bucket = next(
            (b for b in reversed(buckets) if b["n_buys"] > 0), None,
        )
        bottom_bucket = next(
            (b for b in buckets if b["n_buys"] > 0), None,
        )
        if verdict == "MONOTONIC" and top_bucket and bottom_bucket:
            headline = (
                f"MONOTONIC — top conviction sizes at "
                f"{top_bucket['median_size_pct']:.1f}% vs "
                f"{bottom_bucket['median_size_pct']:.1f}% at the floor; "
                f"the bot deploys on conviction."
            )
        elif verdict == "INVERTED" and top_bucket and bottom_bucket:
            headline = (
                f"INVERTED — top conviction sizes at "
                f"{top_bucket['median_size_pct']:.1f}% vs "
                f"{bottom_bucket['median_size_pct']:.1f}% at the floor; "
                f"the bot sizes DOWN at higher conviction."
            )
        elif verdict == "FLAT":
            headline = (
                f"FLAT — deployment ~same across {n_with_score} scored "
                f"BUY(s); the bot's sizing ignores conviction at the "
                f"{MONOTONIC_SLOPE_PP:.0f}pp grain."
            )
        else:
            headline = (
                f"INSUFFICIENT — {n_scanned} BUY(s) scored; need ≥"
                f"{STABLE_MIN_PER_BUCKET} per ≥{STABLE_MIN_POPULATED_BUCKETS} "
                f"buckets for the verdict to fire."
            )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "n_buys_scanned": n_scanned,
        "n_buys_with_score": n_with_score,
        "n_buys_score_unavailable": n_unavailable,
        "window_hours_pre_trade": window_hours_pre_trade,
        "stable_min_per_bucket": STABLE_MIN_PER_BUCKET,
        "stable_min_populated_buckets": STABLE_MIN_POPULATED_BUCKETS,
        "monotonic_slope_pp": MONOTONIC_SLOPE_PP,
        "buckets": buckets,
        "evidence": evidence,
    }
