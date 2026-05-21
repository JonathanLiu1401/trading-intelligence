"""Weekday P&L fingerprint — bucket equity-curve cycle deltas by
NY-local **weekday** and surface which days of the week the book
earns or bleeds.

Sister of :mod:`paper_trader.analytics.hourly_pnl_fingerprint`. Same
input contract (``equity_curve`` rows from ``store.equity_curve()``),
same dead-interval filter, same alpha decomposition vs SPY — the only
axis swap is the bucket key.

Existing siblings cover *cadence* by weekday but not P&L:

* ``decision_weekday`` — per-weekday distribution of DECISIONS.
  Cadence, not P&L.
* ``decision_drought`` — window-scoped idle-cost vs SPY, not
  weekday-bucketed.

A bot that prints +0.4% alpha on Tuesdays and bleeds −0.5% on
Fridays reads neutral on every existing diagnostic — but the weekday
pattern is a real desk edge.

Method — identical to ``hourly_pnl_fingerprint`` except the bucket
key is the NY-local **weekday** index (Monday=0 .. Sunday=6).

Per-bucket aggregate:
  ``weekday`` (0..6), ``weekday_name`` (Mon..Sun), ``n``,
  ``mean_port_delta_pct``, ``mean_alpha_pct``,
  ``sum_port_delta_pct``, ``sum_alpha_pct``, ``n_alpha_samples``.

Aggregate verdict ladder:

  * ``INSUFFICIENT_DATA`` — total qualifying samples <
    ``MIN_TOTAL_SAMPLES`` (default 60).
  * ``NO_SPY_DATA`` — qualifying samples ≥ floor, but no pair has
    both legs of SPY price.
  * ``WEEKDAY_EDGE`` — best-bucket-by-mean-alpha day is a weekday
    (Mon..Fri) AND best − worst ≥ ``ALPHA_SPREAD_PP`` (default 0.5pp).
  * ``WEEKEND_EDGE`` — best day is Sat/Sun AND spread ≥ floor.
    Rare (markets closed) but possible on gap-driven futures
    cycles. Surfaced separately so an operator can spot
    overnight/gap risk asymmetry on the leveraged-ETF book.
  * ``FLAT_WEEK`` — qualifying samples ≥ floor, ≥1 weekday has
    alpha data, but spread is below ``ALPHA_SPREAD_PP``.

Order is load-bearing: ``INSUFFICIENT_DATA`` before any quant
verdict; ``NO_SPY_DATA`` before the EDGE ladder.

Pure builder. Inputs are list-of-dicts from ``store.equity_curve()``,
output is a stable envelope, never raises. Observational only — never
gates Opus, no caps (AGENTS.md #2/#12).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence
from zoneinfo import ZoneInfo

# Verdict labels.
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
NO_SPY_DATA = "NO_SPY_DATA"
WEEKDAY_EDGE = "WEEKDAY_EDGE"
WEEKEND_EDGE = "WEEKEND_EDGE"
FLAT_WEEK = "FLAT_WEEK"

# Defaults.
DEFAULT_MIN_TOTAL_SAMPLES = 60
DEFAULT_ALPHA_SPREAD_PP = 0.5
DEFAULT_TZ = "America/New_York"
# A single weekday bucket must carry at least this many SPY-anchored
# samples before it is eligible to anchor an EDGE verdict — guards
# against a spurious edge on a thin bucket. Mirrors the hourly sister.
DEFAULT_MIN_BUCKET_ALPHA_SAMPLES = 8

_DEAD_DELTA_EPS = 1e-6

# 0=Mon .. 6=Sun (matches datetime.weekday()).
_WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _num(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        f = float(x)
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


def _ny_weekday(dt: datetime, tz: ZoneInfo) -> int:
    return dt.astimezone(tz).weekday()


def build_weekday_pnl_fingerprint(
    equity_curve: Sequence[dict],
    *,
    min_total_samples: int = DEFAULT_MIN_TOTAL_SAMPLES,
    alpha_spread_pp: float = DEFAULT_ALPHA_SPREAD_PP,
    min_bucket_alpha_samples: int = DEFAULT_MIN_BUCKET_ALPHA_SAMPLES,
    tz_name: str = DEFAULT_TZ,
) -> dict:
    """Pure builder. See module docstring."""
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("America/New_York")
        tz_name = "America/New_York"

    def _envelope(verdict: str, headline: str, **extra) -> dict:
        out = {
            "verdict": verdict,
            "headline": headline,
            "buckets": [],
            "best_weekday": None,
            "worst_weekday": None,
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
        if prev_val <= 0:
            prev = (ts, val, spy)
            continue
        port_delta_pct = (val - prev_val) / prev_val * 100.0
        spy_delta_pct: float | None = None
        if spy is not None and prev_spy is not None and prev_spy > 0:
            spy_delta_pct = (spy - prev_spy) / prev_spy * 100.0
        if (
            abs(port_delta_pct) < _DEAD_DELTA_EPS
            and (spy_delta_pct is None or abs(spy_delta_pct) < _DEAD_DELTA_EPS)
        ):
            prev = (ts, val, spy)
            continue
        wd = _ny_weekday(ts, tz)
        b = buckets.setdefault(wd, {
            "weekday": wd,
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

    bucket_list: list[dict] = []
    for wd in sorted(buckets.keys()):
        b = buckets[wd]
        mean_port = b["sum_port_delta_pct"] / b["n"]
        if b["n_alpha_samples"] > 0:
            mean_alpha = b["sum_alpha_pct"] / b["n_alpha_samples"]
            mean_spy = b["sum_spy_delta_pct"] / b["n_alpha_samples"]
        else:
            mean_alpha = None
            mean_spy = None
        bucket_list.append({
            "weekday": wd,
            "weekday_name": _WEEKDAY_NAMES[wd],
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

    alpha_buckets = [b for b in bucket_list if b["mean_alpha_pct"] is not None]
    if not alpha_buckets:
        return _envelope(
            NO_SPY_DATA,
            f"{n_total} cycles bucketed, but no SPY-anchored pair",
            buckets=bucket_list,
            n_total_samples=n_total,
            n_alpha_samples=n_alpha,
        )

    # Only buckets with enough SPY-anchored samples may anchor an
    # EDGE verdict — guards against a spurious edge on a thin bucket.
    eligible = [
        b for b in alpha_buckets
        if b["n_alpha_samples"] >= int(min_bucket_alpha_samples)
    ]
    if not eligible:
        return _envelope(
            INSUFFICIENT_DATA,
            f"{len(alpha_buckets)} weekdays have alpha data but none clears "
            f"the {int(min_bucket_alpha_samples)}-sample per-bucket floor",
            buckets=bucket_list,
            n_total_samples=n_total,
            n_alpha_samples=n_alpha,
        )

    best = max(eligible, key=lambda b: b["mean_alpha_pct"])
    worst = min(eligible, key=lambda b: b["mean_alpha_pct"])
    spread = best["mean_alpha_pct"] - worst["mean_alpha_pct"]

    if spread < float(alpha_spread_pp):
        verdict = FLAT_WEEK
        headline = (
            f"FLAT_WEEK — alpha spread {spread:.2f}pp across "
            f"{len(eligible)} weekdays < {float(alpha_spread_pp):.2f}pp floor"
        )
    else:
        verdict = WEEKEND_EDGE if best["weekday"] >= 5 else WEEKDAY_EDGE
        headline = (
            f"{verdict} — best {best['weekday_name']} "
            f"mean_alpha {best['mean_alpha_pct']:+.2f}pp vs worst "
            f"{worst['weekday_name']} {worst['mean_alpha_pct']:+.2f}pp "
            f"(spread {spread:.2f}pp)"
        )

    return _envelope(
        verdict,
        headline,
        buckets=bucket_list,
        best_weekday={
            "weekday": best["weekday"],
            "weekday_name": best["weekday_name"],
            "mean_alpha_pct": best["mean_alpha_pct"],
            "n_alpha_samples": best["n_alpha_samples"],
        },
        worst_weekday={
            "weekday": worst["weekday"],
            "weekday_name": worst["weekday_name"],
            "mean_alpha_pct": worst["mean_alpha_pct"],
            "n_alpha_samples": worst["n_alpha_samples"],
        },
        alpha_spread_pp=round(spread, 4),
        n_total_samples=n_total,
        n_alpha_samples=n_alpha,
    )
