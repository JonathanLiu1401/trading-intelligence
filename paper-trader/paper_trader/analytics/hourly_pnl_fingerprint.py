"""Hourly P&L fingerprint — bucket equity-curve cycle deltas by
NY-local hour-of-day, then surface where the book actually earns or
bleeds across the trading session.

Existing siblings cover clock-time *decision behaviour* but not
realised P&L:

* ``decision_clock`` — per-hour distribution of DECISIONS (with the
  ``HOURLY_CONCENTRATION`` verdict for NO_DECISION storms). Cadence,
  not P&L.
* ``decision_weekday`` — per-weekday distribution of DECISIONS.
  Cadence, not P&L.
* ``decision_drought`` — segments idle windows by alpha cost vs SPY.
  Window-relative, not clock-bucketed.
* ``session_delta`` — "what changed since T". Window-scoped, not a
  clock-time aggregate.

Nothing in the shelf answers the operator's *"in which clock hour does
my book earn alpha?"* question. A bot that prints +0.4% alpha during
10–11am ET and bleeds −0.6% during 14–15pm ET reads neutral on every
existing diagnostic — but the clock pattern is a real desk edge: the
trader can re-cadence its decision flow toward the productive window.

Method (pure read of ``equity_curve`` rows in ascending order):

  1. Walk consecutive (i-1, i) pairs.
  2. Skip pairs where ``total_value[i-1]`` is non-positive (can't %).
  3. Skip pairs where the prior timestamp is unparseable.
  4. Compute ``port_delta_pct = (val[i] - val[i-1]) / val[i-1] * 100``.
  5. Compute ``spy_delta_pct`` analogously when both ``sp500_price``
     legs are present and positive; else ``None``.
  6. ``alpha_pct = port_delta_pct - spy_delta_pct`` (only when both
     legs available).
  7. **Skip pairs where BOTH port and SPY deltas round to 0** — dead
     overnight / weekend ticks would dominate the bucket otherwise.
     This is the ``decision_drought.PARALYSIS_NO_DELTA`` precedent.
  8. Bucket each remaining pair by the **trailing timestamp's**
     NY-local hour (`America/New_York`), in the range 0..23.

Per-bucket aggregate:
  ``hour``, ``n``, ``mean_port_delta_pct``, ``mean_alpha_pct``,
  ``sum_port_delta_pct``, ``sum_alpha_pct``, ``n_alpha_samples``.

Aggregate verdict ladder:

  * ``INSUFFICIENT_DATA`` — total qualifying samples <
    ``MIN_TOTAL_SAMPLES`` (default 60).
  * ``NO_SPY_DATA`` — qualifying samples ≥ floor, but no pair has
    both legs of SPY price — alpha cannot be computed (a
    book-fresh-restart scenario where ``sp500_price`` is None).
    Per-hour port deltas still report.
  * ``MORNING_EDGE`` — best-bucket-by-mean-alpha hour ∈ [9, 12)
    AND best − worst ≥ ``ALPHA_SPREAD_PP`` (default 0.5pp).
  * ``MIDDAY_EDGE`` — best hour ∈ [12, 14) AND spread ≥ floor.
  * ``AFTERNOON_EDGE`` — best hour ∈ [14, 17) AND spread ≥ floor.
  * ``OFF_HOURS_EDGE`` — best hour outside [9, 17) AND spread ≥ floor
    (e.g. AH 17:00 ET on a futures-driven gap night). Surfaced as a
    distinct bucket so the operator can spot off-hours bleed
    asymmetry on the leveraged-ETF book.
  * ``FLAT_CLOCK`` — qualifying samples ≥ floor, ≥1 hour has alpha
    data, but the alpha spread is below ``ALPHA_SPREAD_PP``.

Order is load-bearing: ``INSUFFICIENT_DATA`` before any quant verdict
so a thin curve reads honestly; ``NO_SPY_DATA`` before the EDGE
ladder so a curve without SPY anchors never gets miscalled FLAT.

Pure builder. Inputs are list-of-dicts from ``store.equity_curve()``,
output is a stable envelope, never raises. Observational only — never
gates Opus, no caps (AGENTS.md #2/#12 — the ``cost_basis_ladder`` /
``catalyst_expiry_skill`` precedent).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence
from zoneinfo import ZoneInfo

# Verdict labels.
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
NO_SPY_DATA = "NO_SPY_DATA"
MORNING_EDGE = "MORNING_EDGE"
MIDDAY_EDGE = "MIDDAY_EDGE"
AFTERNOON_EDGE = "AFTERNOON_EDGE"
OFF_HOURS_EDGE = "OFF_HOURS_EDGE"
FLAT_CLOCK = "FLAT_CLOCK"

# Defaults — tuneable per call.
DEFAULT_MIN_TOTAL_SAMPLES = 60
DEFAULT_ALPHA_SPREAD_PP = 0.5
DEFAULT_TZ = "America/New_York"
# A single hour bucket must carry at least this many SPY-anchored
# samples before it is eligible to be the best/worst EDGE anchor. A
# bucket with n=2 at +5pp would otherwise trigger a spurious EDGE on
# pure noise. Mirrors signal_followthrough's `_MIN_ACTED = 8`.
DEFAULT_MIN_BUCKET_ALPHA_SAMPLES = 8

# Below this absolute value (in pp) a delta rounds to 0 for the
# "dead interval" filter. Mirrors typical equity-curve precision
# (~6 decimals) without churn from float noise.
_DEAD_DELTA_EPS = 1e-6

# Session bands (NY local hour ranges, half-open). 9..12 morning,
# 12..14 midday, 14..17 afternoon. Outside → OFF_HOURS.
_MORNING_RANGE = (9, 12)
_MIDDAY_RANGE = (12, 14)
_AFTERNOON_RANGE = (14, 17)


def _num(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        f = float(x)
        # Reject NaN / inf — they poison means.
        if f != f or f == float("inf") or f == float("-inf"):
            return None
        return f
    if isinstance(x, str):
        try:
            f = float(x.strip())
            if f != f or f == float("inf") or f == float("-inf"):
                return None
            return f
        except (TypeError, ValueError):
            return None
    return None


def _parse_iso(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def _ny_hour(dt: datetime, tz: ZoneInfo) -> int:
    return dt.astimezone(tz).hour


def _classify_hour(hour: int) -> str:
    """Return the EDGE label for the given NY-local hour."""
    if _MORNING_RANGE[0] <= hour < _MORNING_RANGE[1]:
        return MORNING_EDGE
    if _MIDDAY_RANGE[0] <= hour < _MIDDAY_RANGE[1]:
        return MIDDAY_EDGE
    if _AFTERNOON_RANGE[0] <= hour < _AFTERNOON_RANGE[1]:
        return AFTERNOON_EDGE
    return OFF_HOURS_EDGE


def build_hourly_pnl_fingerprint(
    equity_curve: Sequence[dict],
    *,
    min_total_samples: int = DEFAULT_MIN_TOTAL_SAMPLES,
    alpha_spread_pp: float = DEFAULT_ALPHA_SPREAD_PP,
    min_bucket_alpha_samples: int = DEFAULT_MIN_BUCKET_ALPHA_SAMPLES,
    tz_name: str = DEFAULT_TZ,
) -> dict:
    """Pure builder. ``equity_curve`` is ascending-by-time list of
    dicts with ``timestamp`` / ``total_value`` / ``sp500_price``.
    Returns a stable envelope; never raises.
    """
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("America/New_York")
        tz_name = "America/New_York"

    # Default envelope on any early-exit path.
    def _envelope(verdict: str, headline: str, **extra) -> dict:
        out = {
            "verdict": verdict,
            "headline": headline,
            "buckets": [],
            "best_hour": None,
            "worst_hour": None,
            "alpha_spread_pp": None,
            "n_total_samples": 0,
            "n_alpha_samples": 0,
            "thresholds": {
                "min_total_samples": int(min_total_samples),
                "alpha_spread_pp": float(alpha_spread_pp),
                "min_bucket_alpha_samples": int(min_bucket_alpha_samples),
                "tz": tz_name,
            },
        }
        out.update(extra)
        return out

    rows = equity_curve if isinstance(equity_curve, (list, tuple)) else []
    if len(rows) < 2:
        return _envelope(
            INSUFFICIENT_DATA,
            "no equity history — need ≥2 cycles",
        )

    # bucket key -> dict of running stats
    buckets: dict[int, dict] = {}
    n_total = 0
    n_alpha = 0

    prev = None
    for row in rows:
        if not isinstance(row, dict):
            prev = None
            continue
        ts = _parse_iso(row.get("timestamp"))
        val = _num(row.get("total_value"))
        spy = _num(row.get("sp500_price"))
        if ts is None or val is None or val <= 0:
            prev = None
            continue
        if prev is None:
            prev = (ts, val, spy)
            continue
        prev_ts, prev_val, prev_spy = prev
        # Compute the port delta against the PRIOR value.
        if prev_val <= 0:
            prev = (ts, val, spy)
            continue
        port_delta_pct = (val - prev_val) / prev_val * 100.0
        spy_delta_pct: float | None = None
        if spy is not None and prev_spy is not None and prev_spy > 0:
            spy_delta_pct = (spy - prev_spy) / prev_spy * 100.0
        # Dead-interval skip — both legs ~0 (overnight, weekend).
        if (
            abs(port_delta_pct) < _DEAD_DELTA_EPS
            and (spy_delta_pct is None or abs(spy_delta_pct) < _DEAD_DELTA_EPS)
        ):
            prev = (ts, val, spy)
            continue
        # Bucket on the trailing timestamp.
        hour = _ny_hour(ts, tz)
        b = buckets.setdefault(hour, {
            "hour": hour,
            "n": 0,
            "sum_port_delta_pct": 0.0,
            "n_alpha_samples": 0,
            "sum_spy_delta_pct": 0.0,
            "sum_alpha_pct": 0.0,
        })
        b["n"] += 1
        b["sum_port_delta_pct"] += port_delta_pct
        if spy_delta_pct is not None:
            b["n_alpha_samples"] += 1
            b["sum_spy_delta_pct"] += spy_delta_pct
            b["sum_alpha_pct"] += (port_delta_pct - spy_delta_pct)
            n_alpha += 1
        n_total += 1
        prev = (ts, val, spy)

    if n_total < int(min_total_samples):
        return _envelope(
            INSUFFICIENT_DATA,
            f"only {n_total} qualifying samples — need ≥{int(min_total_samples)}",
            n_total_samples=n_total,
            n_alpha_samples=n_alpha,
        )

    # Finalize buckets: emit sorted by hour ascending.
    bucket_list: list[dict] = []
    for hour in sorted(buckets.keys()):
        b = buckets[hour]
        mean_port = b["sum_port_delta_pct"] / b["n"]
        if b["n_alpha_samples"] > 0:
            mean_alpha = b["sum_alpha_pct"] / b["n_alpha_samples"]
            mean_spy = b["sum_spy_delta_pct"] / b["n_alpha_samples"]
        else:
            mean_alpha = None
            mean_spy = None
        bucket_list.append({
            "hour": hour,
            "label": _classify_hour(hour),
            "n": b["n"],
            "n_alpha_samples": b["n_alpha_samples"],
            "mean_port_delta_pct": round(mean_port, 4),
            "mean_spy_delta_pct": (
                round(mean_spy, 4) if mean_spy is not None else None
            ),
            "mean_alpha_pct": (
                round(mean_alpha, 4) if mean_alpha is not None else None
            ),
            "sum_port_delta_pct": round(b["sum_port_delta_pct"], 4),
            "sum_alpha_pct": (
                round(b["sum_alpha_pct"], 4) if b["n_alpha_samples"] > 0 else None
            ),
        })

    # Find best/worst by mean_alpha (alpha-anchored buckets only).
    alpha_buckets = [b for b in bucket_list if b["mean_alpha_pct"] is not None]
    if not alpha_buckets:
        # We have samples but no SPY-anchored pair. Surface buckets,
        # report port deltas, but verdict is NO_SPY_DATA.
        return _envelope(
            NO_SPY_DATA,
            f"{n_total} cycles bucketed, but no SPY-anchored pair",
            buckets=bucket_list,
            n_total_samples=n_total,
            n_alpha_samples=n_alpha,
        )

    # Only buckets carrying enough SPY-anchored samples are eligible
    # to anchor an EDGE verdict — a thin bucket (n=2 at +5pp) would
    # otherwise trigger a spurious clock edge on pure noise.
    eligible = [
        b for b in alpha_buckets
        if b["n_alpha_samples"] >= int(min_bucket_alpha_samples)
    ]
    if not eligible:
        return _envelope(
            INSUFFICIENT_DATA,
            f"{len(alpha_buckets)} hours have alpha data but none clears "
            f"the {int(min_bucket_alpha_samples)}-sample per-bucket floor",
            buckets=bucket_list,
            n_total_samples=n_total,
            n_alpha_samples=n_alpha,
        )

    best = max(eligible, key=lambda b: b["mean_alpha_pct"])
    worst = min(eligible, key=lambda b: b["mean_alpha_pct"])
    spread = best["mean_alpha_pct"] - worst["mean_alpha_pct"]

    if spread < float(alpha_spread_pp):
        verdict = FLAT_CLOCK
        headline = (
            f"FLAT_CLOCK — alpha spread {spread:.2f}pp across "
            f"{len(eligible)} hours < {float(alpha_spread_pp):.2f}pp floor"
        )
    else:
        verdict = _classify_hour(best["hour"])
        headline = (
            f"{verdict} — best hour {best['hour']:02d}:00 NY "
            f"mean_alpha {best['mean_alpha_pct']:+.2f}pp vs worst "
            f"{worst['hour']:02d}:00 NY {worst['mean_alpha_pct']:+.2f}pp "
            f"(spread {spread:.2f}pp)"
        )

    return _envelope(
        verdict,
        headline,
        buckets=bucket_list,
        best_hour={
            "hour": best["hour"],
            "label": best["label"],
            "mean_alpha_pct": best["mean_alpha_pct"],
            "n_alpha_samples": best["n_alpha_samples"],
        },
        worst_hour={
            "hour": worst["hour"],
            "label": worst["label"],
            "mean_alpha_pct": worst["mean_alpha_pct"],
            "n_alpha_samples": worst["n_alpha_samples"],
        },
        alpha_spread_pp=round(spread, 4),
        n_total_samples=n_total,
        n_alpha_samples=n_alpha,
    )
