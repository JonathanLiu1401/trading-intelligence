"""Closed-market fill skill — partitions every FILLED trade by whether
NYSE was open at the moment of execution.

Why the diagnostic matters. ``market.get_price`` returns the last-known
close when NYSE is closed (weekends, overnight, pre-market, half-day
afterhours, holidays). A BUY fired at 02:00 ET on a Wednesday fills
against Tuesday's 16:00 close — there is no real bid-ask, no liquidity
test, and no price discovery. The book records what *looks* like
alpha-attempt repositioning but every entry/exit during that closed
window happens at a stale snapshot. Pair this with the documented
frozen-mark pathology (``frozen_mark_execution_skill``) and the live
trader's 2026-05-20→05-21 NVDA cluster (5 fills at the SAME yfinance
close during a 13-hour overnight stretch) and the pattern is clear:
overnight/weekend fills are economically illusory.

Distinct from neighbours:

* ``frozen_mark_execution_skill`` — clusters of trades at the EXACT
  same price (effect). Doesn't catch single closed-session fills.
* ``decision_clock`` — distribution of DECISION cycles by hour-of-day,
  including NO_DECISIONs. Doesn't filter to FILLED trades or
  ``is_market_open``.
* ``decision_weekday`` — by weekday. Same surface as clock.

A trade is ``open_fill`` iff ``market.is_market_open(trade_ts)`` is True
at the moment of fill (NYSE regular or half-day session, NY tz,
post-9:30, pre-16:00 / pre-13:00, weekday, non-holiday). Everything
else is ``closed_fill`` — broken down further into ``weekend``,
``overnight`` (weekday outside the session window), and ``holiday``.

Verdict ladder (test-locked, on closed-fill share of in-window FILLS):

* ``SESSION_ALIGNED`` — closed_fill_pct ≤ ``aligned_pct`` (default 25).
* ``BALANCED`` — between aligned and ``after_hours_pct`` (default 50).
* ``AFTER_HOURS_HEAVY`` — between after_hours and ``dominated_pct``
  (default 75).
* ``OVERNIGHT_DOMINATED`` — closed_fill_pct ≥ ``dominated_pct``.
* ``INSUFFICIENT_DATA`` — fewer than ``MIN_FILLS_FOR_VERDICT`` (5)
  classifiable FILLED trades.

Pure builder. Trades in, dict out, never raises. Observational only —
never gates Opus, no caps (AGENTS.md #2/#12).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Sequence

from ..market import is_market_open as _real_is_market_open

# Verdict thresholds (percent of in-window FILLED trades that fired
# while NYSE was closed).
DEFAULT_ALIGNED_PCT = 25.0
DEFAULT_AFTER_HOURS_PCT = 50.0
DEFAULT_DOMINATED_PCT = 75.0

# Analysis window.
DEFAULT_WINDOW_DAYS = 30.0

# Below this floor the partition is too thin to read — emit envelope
# but withhold the verdict.
MIN_FILLS_FOR_VERDICT = 5

_FILL_ACTIONS = frozenset({
    "BUY", "SELL", "BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT",
})

try:
    from zoneinfo import ZoneInfo
    _NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover — zoneinfo is stdlib
    _NY = timezone.utc

# Lazy import of holiday set so tests injecting their own is_open_fn do
# not depend on market.py at all. The default closed-bucket classifier
# only consults the NY tz + the holiday-set on the live market module.
try:
    from ..market import NYSE_HOLIDAYS_2026 as _HOLIDAYS_2026
except Exception:  # pragma: no cover
    _HOLIDAYS_2026 = frozenset()


def _parse_iso(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _safe_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    if f in (float("inf"), float("-inf")):
        return None
    return f


def _closed_subbucket(ts_utc: datetime) -> str:
    """``weekend`` / ``overnight`` / ``holiday`` for a CLOSED fill ts.

    Mirrors ``market.is_market_open``'s holiday-set check so any holiday
    listed there is reported as ``holiday`` rather than ``weekend`` or
    ``overnight``. Pure — never raises.
    """
    try:
        ny = ts_utc.astimezone(_NY)
    except Exception:
        return "overnight"
    if ny.date() in _HOLIDAYS_2026:
        return "holiday"
    if ny.weekday() >= 5:
        return "weekend"
    return "overnight"


def _normalize_trade(tr: Any) -> dict | None:
    if not isinstance(tr, dict):
        return None
    action = tr.get("action")
    if not isinstance(action, str) or action.upper() not in _FILL_ACTIONS:
        return None
    ticker = tr.get("ticker")
    if not isinstance(ticker, str) or not ticker:
        return None
    price = _safe_float(tr.get("price"))
    if price is None or price <= 0:
        return None
    qty = _safe_float(tr.get("qty"))
    if qty is None:
        return None
    ts = _parse_iso(tr.get("timestamp"))
    if ts is None:
        return None
    return {
        "ticker": ticker.upper(),
        "action": action.upper(),
        "price": price,
        "qty": abs(qty),
        "ts": ts,
        "notional": abs(qty * price),
    }


def build_closed_market_fill_skill(
    trades: Sequence[Any] | None,
    *,
    now: datetime | None = None,
    window_days: float = DEFAULT_WINDOW_DAYS,
    aligned_pct: float = DEFAULT_ALIGNED_PCT,
    after_hours_pct: float = DEFAULT_AFTER_HOURS_PCT,
    dominated_pct: float = DEFAULT_DOMINATED_PCT,
    is_open_fn: Callable[[datetime], bool] | None = None,
) -> dict[str, Any]:
    """Pure FILL-side partition by NYSE-session state. Never raises.

    Inputs:
      ``trades`` — sequence of trade dicts (mixed actions). Caller does
        not need to pre-filter (HOLD / NO_DECISION rows are dropped).
      ``now`` — defaults to ``datetime.now(utc)``.
      ``window_days`` — analysis window (default 30d).
      ``aligned_pct`` / ``after_hours_pct`` / ``dominated_pct`` —
        verdict thresholds on the closed-fill percentage.
      ``is_open_fn`` — testability injection point. Default uses the
        live ``market.is_market_open`` (NYSE calendar, half-days,
        2026 holiday set). Pass an arbitrary ``Callable[[datetime],
        bool]`` for unit tests that don't want to depend on the live
        NYSE calendar.

    Envelope (keys always present):
      ``as_of``, ``verdict``, ``headline``, ``window_days``,
      ``thresholds`` (dict), ``stats`` (counts + percentages),
      ``per_ticker`` (closed-fill share per ticker, worst-first),
      ``closed_subbucket`` (overnight/weekend/holiday counts).
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    window_days = max(0.0, float(window_days))
    aligned_pct = max(0.0, min(100.0, float(aligned_pct)))
    after_hours_pct = max(0.0, min(100.0, float(after_hours_pct)))
    dominated_pct = max(0.0, min(100.0, float(dominated_pct)))
    # Enforce monotonicity — if a caller scrambles them, widen rather
    # than raise so the route never 500s on a bad query string.
    if after_hours_pct < aligned_pct:
        after_hours_pct = aligned_pct
    if dominated_pct < after_hours_pct:
        dominated_pct = after_hours_pct

    cutoff = now - timedelta(days=window_days)
    is_open_fn = is_open_fn or _real_is_market_open

    normalized: list[dict] = []
    if trades:
        for raw in trades:
            n = _normalize_trade(raw)
            if n is None:
                continue
            if n["ts"] < cutoff:
                continue
            normalized.append(n)

    thresholds = {
        "window_days": window_days,
        "aligned_pct": aligned_pct,
        "after_hours_pct": after_hours_pct,
        "dominated_pct": dominated_pct,
        "min_fills_for_verdict": MIN_FILLS_FOR_VERDICT,
    }

    if len(normalized) < MIN_FILLS_FOR_VERDICT:
        return {
            "as_of": now.isoformat(timespec="seconds"),
            "verdict": "INSUFFICIENT_DATA",
            "headline": (
                f"only {len(normalized)} classifiable FILLED trades in the last "
                f"{int(window_days)}d (need ≥ {MIN_FILLS_FOR_VERDICT})"
            ),
            "window_days": window_days,
            "thresholds": thresholds,
            "stats": _empty_stats(),
            "per_ticker": [],
            "closed_subbucket": {"overnight": 0, "weekend": 0, "holiday": 0},
        }

    # Partition.
    n_open = n_closed = 0
    notional_open = notional_closed = 0.0
    subbuckets = {"overnight": 0, "weekend": 0, "holiday": 0}
    per_ticker_tot: dict[str, dict[str, int]] = {}

    for tr in normalized:
        ticker = tr["ticker"]
        try:
            open_now = bool(is_open_fn(tr["ts"]))
        except Exception:
            # Defensive: treat unknown as closed (the conservative bucket
            # for the diagnostic — counts toward the "did we trade with
            # real price discovery?" warning).
            open_now = False
        bucket = "open" if open_now else "closed"
        if open_now:
            n_open += 1
            notional_open += tr["notional"]
        else:
            n_closed += 1
            notional_closed += tr["notional"]
            sub = _closed_subbucket(tr["ts"])
            subbuckets[sub] = subbuckets.get(sub, 0) + 1
        per_ticker_tot.setdefault(
            ticker, {"open": 0, "closed": 0},
        )[bucket] += 1

    n_total = n_open + n_closed
    closed_pct = round(100.0 * n_closed / n_total, 2) if n_total else 0.0
    open_pct = round(100.0 - closed_pct, 2)

    # Per-ticker breakdown, worst (highest closed%) first.
    per_ticker = []
    for tk, counts in per_ticker_tot.items():
        sub_total = counts["open"] + counts["closed"]
        sub_closed_pct = (
            round(100.0 * counts["closed"] / sub_total, 2) if sub_total else 0.0
        )
        per_ticker.append({
            "ticker": tk,
            "n_open": counts["open"],
            "n_closed": counts["closed"],
            "n_total": sub_total,
            "closed_pct": sub_closed_pct,
        })
    per_ticker.sort(key=lambda r: (-r["closed_pct"], -r["n_total"], r["ticker"]))

    if closed_pct >= dominated_pct:
        verdict = "OVERNIGHT_DOMINATED"
        headline = (
            f"{closed_pct:.1f}% of {n_total} FILLs landed while NYSE was "
            f"closed — fills are pricing off stale yfinance closes, not "
            f"a live bid-ask"
        )
    elif closed_pct >= after_hours_pct:
        verdict = "AFTER_HOURS_HEAVY"
        headline = (
            f"{closed_pct:.1f}% of {n_total} FILLs landed outside session "
            f"hours — meaningful share executed without live price discovery"
        )
    elif closed_pct >= aligned_pct:
        verdict = "BALANCED"
        headline = (
            f"{closed_pct:.1f}% of {n_total} FILLs landed outside session "
            f"hours — within tolerance"
        )
    else:
        verdict = "SESSION_ALIGNED"
        headline = (
            f"{closed_pct:.1f}% of {n_total} FILLs landed outside session "
            f"hours — fills track real price discovery"
        )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "verdict": verdict,
        "headline": headline,
        "window_days": window_days,
        "thresholds": thresholds,
        "stats": {
            "n_total": n_total,
            "n_open": n_open,
            "n_closed": n_closed,
            "open_pct": open_pct,
            "closed_pct": closed_pct,
            "notional_open": round(notional_open, 2),
            "notional_closed": round(notional_closed, 2),
            "notional_closed_pct": (
                round(100.0 * notional_closed / (notional_open + notional_closed), 2)
                if (notional_open + notional_closed) > 0 else 0.0
            ),
        },
        "per_ticker": per_ticker,
        "closed_subbucket": subbuckets,
    }


def _empty_stats() -> dict[str, Any]:
    return {
        "n_total": 0,
        "n_open": 0,
        "n_closed": 0,
        "open_pct": 0.0,
        "closed_pct": 0.0,
        "notional_open": 0.0,
        "notional_closed": 0.0,
        "notional_closed_pct": 0.0,
    }
