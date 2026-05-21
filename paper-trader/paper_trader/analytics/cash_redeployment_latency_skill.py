"""Cash-redeployment latency — when the desk closes a position (SELL fill),
how long until the freed cash is redeployed into ANY new BUY?

The existing surface answers adjacent questions but not this one:

* ``rebuy_regret`` — *$ regret* on SAME-ticker SELL→re-BUY (price delta
  over the close→re-entry window). Per-name cost, not per-cycle latency.
* ``reentry_velocity`` — *time* gap between SAME-ticker close and the
  next entry of the same key (ticker, type, strike, expiry). Per-name
  cadence, not cross-ticker redeployment.
* ``capital_paralysis`` — point-in-time idle-cash verdict (historical
  alpha-bleed window). Snapshot, not interval distribution.
* ``idle_opportunity`` — point-in-time signals × idle cash. Snapshot.
* ``cash_conviction_fit`` — book cash vs loudest live signal. Snapshot.

None ask: **"After I sell, how many hours until any buy at all?"**

That's the *cross-ticker cash-deployment latency*. A desk that closes a
position and then sits on cash for 5 days has the same headline P/L as
one that recycles within an hour — but the alpha bleed is wildly
different. The behavioural pathology this catches:

    Sold NVDA at 11:32 ET. Made a clean $42 on the swing. Then ...
    nothing. No BUY of any name for the next 6 trading days. Cash drag
    silently ate into the win.

That sell-then-sit pattern is what the live-trader's documented
``capital_paralysis FREE`` verdict misses — capital_paralysis fires on
the *standing* idle window, not on the *interval* following each
liberating SELL.

Verdict ladder (test-locked):

* ``FAST_REDEPLOY`` — median latency ≤ ``FAST_MEDIAN_H`` (6h)
  AND redeployment rate ≥ ``HEALTHY_REDEPLOY_PCT`` (80%). The desk
  recycles freed cash quickly into the next idea.
* ``STEADY`` — median ≤ ``STEADY_MEDIAN_H`` (24h) AND redeployment
  rate ≥ ``STEADY_REDEPLOY_PCT`` (70%). Within a session.
* ``SLOW`` — median ≤ ``SLOW_MEDIAN_H`` (72h) OR redeployment rate
  between 50-70%. A few SELLs left cash sitting through a session.
* ``STALLED`` — median > 72h OR redeployment rate < 50%. The
  documented sell-then-sit pathology — cash sits idle for days
  before redeployment, or many SELLs never redeploy at all.
* ``NO_DATA`` — fewer than ``MIN_SELLS_FOR_VERDICT`` (3) FILLED
  SELLs in the window. Always emits the envelope; never raises.

A SELL is counted as "stalled" if no subsequent FILLED BUY fires within
``stalled_cutoff_hours`` (default 168h = 1 week). That keeps the
window-edge SELLs from polluting the median: a SELL one hour before
the window-end has no chance of being "redeployed" yet by definition,
so we exclude SELLs whose window-end remaining time is below the
stalled-cutoff from the redeployed/stalled tallies (they go into
``n_window_edge``) — only SELLs old enough to have had a fair chance
contribute to the verdict.

Trade-action whitelist mirrors the live trader's executor: BUY /
BUY_CALL / BUY_PUT count as redeployment; SELL / SELL_CALL / SELL_PUT
count as freeing cash. Notional uses ``value`` (qty × price, the
trades-table column), falling back to ``notional`` for tests.

Pure builder. Trades in, dict out, never raises. Observational only —
never gates Opus, no caps (AGENTS.md #2/#12 — same precedent as the
``rebuy_regret`` / ``reentry_velocity`` builder split).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

# Verdict thresholds (hours). Adjust both sides together — tests pin them.
DEFAULT_FAST_MEDIAN_H = 6.0
DEFAULT_STEADY_MEDIAN_H = 24.0
DEFAULT_SLOW_MEDIAN_H = 72.0

# Redeployment-rate floors (percent of SELLs that were redeployed within
# the stalled-cutoff window).
DEFAULT_HEALTHY_REDEPLOY_PCT = 80.0
DEFAULT_STEADY_REDEPLOY_PCT = 70.0
DEFAULT_DEGRADED_REDEPLOY_PCT = 50.0

# How long a SELL has to wait before being declared STALLED. A SELL with no
# subsequent BUY within this window is excluded from the median (since its
# interval is unbounded) and counted toward n_stalled.
DEFAULT_STALLED_CUTOFF_H = 168.0  # 1 week

# Analysis window — how far back we scan SELLs.
DEFAULT_WINDOW_DAYS = 30.0

# Below this floor the per-pair distribution is too noisy to read; emit
# stats but withhold the verdict.
MIN_SELLS_FOR_VERDICT = 3

_BUY_ACTIONS = frozenset({"BUY", "BUY_CALL", "BUY_PUT"})
_SELL_ACTIONS = frozenset({"SELL", "SELL_CALL", "SELL_PUT"})


def _parse_iso(ts: Any) -> datetime | None:
    """Best-effort ISO → tz-aware datetime. Returns None on garbage."""
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


def _safe_notional(tr: dict) -> float:
    """Best-effort notional — uses `value` (trades-table column) first,
    falls back to `notional`, then to `qty * price`. Returns 0.0 on garbage."""
    for key in ("value", "notional"):
        v = tr.get(key)
        if isinstance(v, (int, float)):
            try:
                f = float(v)
                if f == f and f >= 0:
                    return abs(f)
            except (TypeError, ValueError):
                pass
    qty = tr.get("qty")
    price = tr.get("price") or tr.get("fill_price")
    if isinstance(qty, (int, float)) and isinstance(price, (int, float)):
        try:
            f = abs(float(qty) * float(price))
            if f == f:
                return f
        except (TypeError, ValueError):
            pass
    return 0.0


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _percentile(xs: list[float], p: float) -> float | None:
    """Linear-interp percentile (0..100). Returns None on empty."""
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    s = sorted(xs)
    n = len(s)
    pos = (p / 100.0) * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def build_cash_redeployment_latency_skill(
    trades: Sequence[Any] | None,
    *,
    now: datetime | None = None,
    window_days: float = DEFAULT_WINDOW_DAYS,
    stalled_cutoff_hours: float = DEFAULT_STALLED_CUTOFF_H,
    fast_median_h: float = DEFAULT_FAST_MEDIAN_H,
    steady_median_h: float = DEFAULT_STEADY_MEDIAN_H,
    slow_median_h: float = DEFAULT_SLOW_MEDIAN_H,
    healthy_redeploy_pct: float = DEFAULT_HEALTHY_REDEPLOY_PCT,
    steady_redeploy_pct: float = DEFAULT_STEADY_REDEPLOY_PCT,
    degraded_redeploy_pct: float = DEFAULT_DEGRADED_REDEPLOY_PCT,
) -> dict[str, Any]:
    """Pure SELL→next-BUY latency builder. Never raises.

    Inputs:
      ``trades`` — list of trade dicts ``{action, ticker, timestamp, value, ...}``.
        Mixed BUY/SELL; caller does not need to filter.
      ``now`` — defaults to ``datetime.now(utc)``.
      ``window_days`` — scan SELLs whose timestamp is within this many days
        from ``now``.
      ``stalled_cutoff_hours`` — a SELL with no subsequent BUY within this
        window is counted toward ``n_stalled``.

    Threshold overrides exposed for tests + caller knobs.
    """
    now = now or datetime.now(timezone.utc)
    window_cutoff = now - timedelta(days=max(0.0, window_days))
    stalled_cutoff = timedelta(hours=max(0.0, stalled_cutoff_hours))

    # Sort trades chronologically (oldest-first) so the "next BUY after each
    # SELL" lookup is a single forward walk. Defensive against garbage rows.
    parsed: list[tuple[datetime, str, dict]] = []
    for tr in (trades or []):
        if not isinstance(tr, dict):
            continue
        action = tr.get("action")
        if not isinstance(action, str):
            continue
        action_u = action.upper()
        if action_u not in (_BUY_ACTIONS | _SELL_ACTIONS):
            continue
        ts = tr.get("timestamp") or tr.get("ts")
        ts_dt = _parse_iso(ts)
        if ts_dt is None:
            continue
        parsed.append((ts_dt, action_u, tr))
    parsed.sort(key=lambda x: x[0])

    # Walk SELLs in the window; for each, find the earliest BUY whose ts is
    # strictly later. We don't require the BUY to be inside the analysis
    # window — a SELL on day 1 with a BUY on day 35 is a legitimate
    # late-redeployment data point so long as it's within stalled_cutoff.
    pairs: list[dict[str, Any]] = []
    n_sells_in_window = 0
    n_redeployed = 0
    n_stalled = 0
    n_window_edge = 0  # SELLs too close to `now` to fairly classify
    latencies_h: list[float] = []
    total_freed_usd = 0.0
    total_redeployed_usd = 0.0

    for i, (sell_ts, sell_action, sell_tr) in enumerate(parsed):
        if sell_action not in _SELL_ACTIONS:
            continue
        if sell_ts < window_cutoff:
            continue
        n_sells_in_window += 1

        # Find earliest BUY strictly after sell_ts
        next_buy_ts: datetime | None = None
        next_buy_tr: dict | None = None
        for j in range(i + 1, len(parsed)):
            cand_ts, cand_action, cand_tr = parsed[j]
            if cand_action in _BUY_ACTIONS:
                next_buy_ts = cand_ts
                next_buy_tr = cand_tr
                break

        sell_notional = _safe_notional(sell_tr)
        total_freed_usd += sell_notional

        # If the SELL is too close to `now` for stalled_cutoff to have elapsed,
        # we exclude it from redeployed/stalled tallies (otherwise we'd
        # systematically over-count STALLEDs at the trailing edge).
        time_remaining = now - sell_ts
        window_edge = time_remaining < stalled_cutoff and next_buy_ts is None
        if window_edge:
            n_window_edge += 1
            pairs.append({
                "sell_ts": sell_ts.isoformat(),
                "sell_ticker": sell_tr.get("ticker"),
                "sell_action": sell_action,
                "sell_notional_usd": round(sell_notional, 2),
                "next_buy_ts": None,
                "next_buy_ticker": None,
                "latency_h": None,
                "status": "WINDOW_EDGE",
            })
            continue

        if next_buy_ts is None:
            n_stalled += 1
            pairs.append({
                "sell_ts": sell_ts.isoformat(),
                "sell_ticker": sell_tr.get("ticker"),
                "sell_action": sell_action,
                "sell_notional_usd": round(sell_notional, 2),
                "next_buy_ts": None,
                "next_buy_ticker": None,
                "latency_h": None,
                "status": "STALLED",
            })
            continue

        delta = next_buy_ts - sell_ts
        delta_h = delta.total_seconds() / 3600.0
        if delta > stalled_cutoff:
            n_stalled += 1
            pairs.append({
                "sell_ts": sell_ts.isoformat(),
                "sell_ticker": sell_tr.get("ticker"),
                "sell_action": sell_action,
                "sell_notional_usd": round(sell_notional, 2),
                "next_buy_ts": next_buy_ts.isoformat(),
                "next_buy_ticker": next_buy_tr.get("ticker") if next_buy_tr else None,
                "latency_h": round(delta_h, 2),
                "status": "STALLED",
            })
            continue

        n_redeployed += 1
        latencies_h.append(delta_h)
        total_redeployed_usd += _safe_notional(next_buy_tr) if next_buy_tr else 0.0
        pairs.append({
            "sell_ts": sell_ts.isoformat(),
            "sell_ticker": sell_tr.get("ticker"),
            "sell_action": sell_action,
            "sell_notional_usd": round(sell_notional, 2),
            "next_buy_ts": next_buy_ts.isoformat(),
            "next_buy_ticker": next_buy_tr.get("ticker") if next_buy_tr else None,
            "latency_h": round(delta_h, 2),
            "status": "REDEPLOYED",
        })

    # Aggregates
    median_h = _median(latencies_h)
    p25_h = _percentile(latencies_h, 25.0)
    p75_h = _percentile(latencies_h, 75.0)

    # Redeploy rate is computed against the *classifiable* SELLs only —
    # window-edge SELLs are excluded from both numerator and denominator.
    n_classifiable = n_redeployed + n_stalled
    redeploy_pct: float | None
    if n_classifiable > 0:
        redeploy_pct = round((n_redeployed / n_classifiable) * 100.0, 2)
    else:
        redeploy_pct = None

    # Verdict ladder
    if n_classifiable < MIN_SELLS_FOR_VERDICT:
        verdict = "NO_DATA"
        headline = (
            f"insufficient: {n_classifiable} classifiable SELLs in last "
            f"{window_days:g}d (min {MIN_SELLS_FOR_VERDICT})"
        )
    else:
        m = median_h if median_h is not None else float("inf")
        r = redeploy_pct if redeploy_pct is not None else 0.0
        if m <= fast_median_h and r >= healthy_redeploy_pct:
            verdict = "FAST_REDEPLOY"
            headline = (
                f"median {m:.1f}h; {n_redeployed}/{n_classifiable} "
                f"SELLs redeployed ({r:.0f}%)"
            )
        elif m <= steady_median_h and r >= steady_redeploy_pct:
            verdict = "STEADY"
            headline = (
                f"median {m:.1f}h; {n_redeployed}/{n_classifiable} "
                f"SELLs redeployed ({r:.0f}%)"
            )
        elif m > slow_median_h or r < degraded_redeploy_pct:
            verdict = "STALLED"
            if median_h is None:
                headline = (
                    f"stalled: {n_stalled}/{n_classifiable} SELLs never "
                    f"redeployed within {stalled_cutoff_hours:g}h"
                )
            else:
                headline = (
                    f"stalled: median {m:.1f}h, only "
                    f"{n_redeployed}/{n_classifiable} redeployed ({r:.0f}%)"
                )
        else:
            verdict = "SLOW"
            headline = (
                f"slow: median {m:.1f}h; "
                f"{n_redeployed}/{n_classifiable} redeployed ({r:.0f}%)"
            )

    # Sort pairs newest-first so the dashboard shows the most recent
    # SELLs at the top.
    pairs.sort(key=lambda p: p["sell_ts"], reverse=True)

    return {
        "verdict": verdict,
        "headline": headline,
        "as_of": now.isoformat(),
        "window_days": window_days,
        "stats": {
            "n_sells_total": n_sells_in_window,
            "n_classifiable": n_classifiable,
            "n_redeployed": n_redeployed,
            "n_stalled": n_stalled,
            "n_window_edge": n_window_edge,
            "redeploy_pct": redeploy_pct,
            "median_latency_h": round(median_h, 2) if median_h is not None else None,
            "p25_latency_h": round(p25_h, 2) if p25_h is not None else None,
            "p75_latency_h": round(p75_h, 2) if p75_h is not None else None,
            "total_freed_usd": round(total_freed_usd, 2),
            "total_redeployed_usd": round(total_redeployed_usd, 2),
        },
        "thresholds": {
            "fast_median_h": fast_median_h,
            "steady_median_h": steady_median_h,
            "slow_median_h": slow_median_h,
            "healthy_redeploy_pct": healthy_redeploy_pct,
            "steady_redeploy_pct": steady_redeploy_pct,
            "degraded_redeploy_pct": degraded_redeploy_pct,
            "stalled_cutoff_h": stalled_cutoff_hours,
            "min_sells_for_verdict": MIN_SELLS_FOR_VERDICT,
        },
        "pairs": pairs,
    }
