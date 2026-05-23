"""Frozen-mark execution skill — detects FILLED trade clusters that all
executed at the EXACT same float price within a short window.

When ``market.get_price`` falls through to a cached yfinance close
(overnight, pre-market, weekend, or a brief upstream stall) every BUY
or SELL Opus fires routes to the same bit-identical float. The book
records what *looks* like alpha-attempt repositioning (BUY → SELL →
BUY round-trips) but the cluster nets zero P&L impact across same-
ticker fills because the price literally never moved.

Live example caught by this builder against ``paper_trader.db`` on
2026-05-23: 5 NVDA trades over a 13-hour window 2026-05-20T21:10 →
2026-05-21T10:00 — BUY 1, BUY 0.5, SELL 4.5, BUY 2, BUY 1 — all at
exactly ``$223.43499755859375``. The desk burned 5 decision cycles
re-shuffling against a frozen mark while the wall-clock advanced.

Distinct from the surrounding surface:

* ``mark_integrity`` — % of the DISPLAYED book held at a stale mark
  (advisory, snapshot, never reads ``trades``).
* ``rebuy_regret`` / ``reentry_velocity`` — same-ticker SELL→re-BUY $
  regret / timing. Doesn't filter for identical-price executions or
  detect 3-trade clusters within a single mark.
* ``churn`` — overall trade frequency without price-discovery context.

A cluster is defined as ≥ ``cluster_min`` FILLED trades on the same
ticker filling at the EXACT same float price within a
``cluster_span_hours`` window. The float equality is required to be
bit-identical (``==``) — a single penny of real price discovery
breaks the equality, so this catches only true frozen marks.

Verdict ladder (test-locked):

* ``CLEAN`` — frozen_trade_pct < ``occasional_pct`` (default 5%).
* ``OCCASIONAL`` — between CLEAN and HEAVY.
* ``FROZEN_MARK_HEAVY`` — frozen_trade_pct ≥ ``heavy_pct`` (25%).
* ``INSUFFICIENT_DATA`` — fewer than ``MIN_TRADES_FOR_VERDICT`` (5)
  trades in the analysis window.

Pure builder — list of trade dicts in, dict out, never raises.
Observational only — never gates Opus, no caps (AGENTS.md #2/#12 —
same precedent as the ``cash_redeployment_latency_skill`` /
``decision_vapor_skill`` builders).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

# Cluster definition.
DEFAULT_CLUSTER_MIN = 2          # 2+ trades at the same exact price ⇒ cluster
DEFAULT_CLUSTER_SPAN_HOURS = 24.0  # within a 24-hour window

# Verdict thresholds (percent of trades in any frozen-mark cluster).
DEFAULT_OCCASIONAL_PCT = 5.0
DEFAULT_HEAVY_PCT = 25.0

# Analysis window.
DEFAULT_WINDOW_DAYS = 30.0

# Below this floor the per-cluster distribution is too thin to read; the
# builder still emits a fully-populated envelope but withholds the verdict.
MIN_TRADES_FOR_VERDICT = 5

# Trade-action whitelist — mirrors the live executor's FILLED set, matching
# the cash_redeployment_latency_skill / churn builders. Anything outside
# this is dropped (HOLD/NO_DECISION/BLOCKED don't have a price to compare).
_FILL_ACTIONS = frozenset({
    "BUY", "SELL", "BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT",
})

# Per-cluster realized round-trip cap. A frozen-mark cluster with BUYs and
# SELLs in equal qty *at the same price* yields exactly zero realized
# P&L per share — that's the structural pathology this skill exists to
# catch. We expose the net qty delta + intra-cluster paper P&L (always 0
# when in-cluster, since price is constant) so the operator can see at a
# glance that a flurry of "trading" was a no-op against a constant mark.


def _parse_iso(ts: Any) -> datetime | None:
    """Best-effort ISO → tz-aware UTC datetime. None on garbage."""
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
    """Best-effort float that rejects NaN / inf. None on garbage."""
    if isinstance(v, bool):  # bool is an int subclass; reject explicitly
        return None
    if not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    if f in (float("inf"), float("-inf")):
        return None
    return f


def _normalize_trade(tr: Any) -> dict | None:
    """Pull out (ticker, action, price, qty, ts_utc) — drop anything not a
    FILLED equity-or-option trade with a comparable price+timestamp."""
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
    }


def _identical_price_clusters(
    trades: list[dict],
    cluster_min: int,
    cluster_span_hours: float,
) -> list[dict]:
    """Group trades by ``(ticker, exact-price)`` and return clusters whose
    member trades all fall inside a single ``cluster_span_hours`` window
    and contain at least ``cluster_min`` rows.

    A frozen-mark cluster is by definition a run of trades at the EXACT
    same float price; we further require the run to be temporally
    contiguous (no gap larger than ``cluster_span_hours`` between the
    first and last trade) — a single yfinance cache lasts hours, not
    weeks, so a year-apart pair at the same price is a coincidence, not
    a frozen mark.

    Each returned cluster carries the per-cluster summary the route
    surfaces directly: ticker, price, span, action mix, net qty delta,
    sample timestamps. Sorted DESC by trade count so the worst clusters
    surface first.

    Pure — never raises.
    """
    if not trades or cluster_min < 2:
        return []
    span_delta = timedelta(hours=max(0.0, float(cluster_span_hours)))

    by_key: dict[tuple[str, float], list[dict]] = {}
    for tr in trades:
        key = (tr["ticker"], tr["price"])
        by_key.setdefault(key, []).append(tr)

    clusters: list[dict] = []
    for (ticker, price), rows in by_key.items():
        if len(rows) < cluster_min:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["ts"])
        # Sliding-window grouping: walk chronologically, start a new run
        # whenever the gap from run-start exceeds the span. This keeps
        # frozen-mark clusters tight without merging unrelated re-visits
        # of the same price months apart.
        run: list[dict] = []
        run_start: datetime | None = None
        for r in rows_sorted:
            if not run:
                run = [r]
                run_start = r["ts"]
                continue
            if r["ts"] - run_start <= span_delta:
                run.append(r)
            else:
                if len(run) >= cluster_min:
                    clusters.append(_summarize_cluster(ticker, price, run))
                run = [r]
                run_start = r["ts"]
        if len(run) >= cluster_min:
            clusters.append(_summarize_cluster(ticker, price, run))

    clusters.sort(key=lambda c: (-c["n_trades"], c["ticker"]))
    return clusters


def _summarize_cluster(ticker: str, price: float, run: list[dict]) -> dict:
    """Compact per-cluster summary. Pure."""
    n = len(run)
    ts_first = run[0]["ts"]
    ts_last = run[-1]["ts"]
    span_s = (ts_last - ts_first).total_seconds()
    action_mix: dict[str, int] = {}
    buy_qty = 0.0
    sell_qty = 0.0
    for r in run:
        a = r["action"]
        action_mix[a] = action_mix.get(a, 0) + 1
        if a.startswith("BUY"):
            buy_qty += r["qty"]
        elif a.startswith("SELL"):
            sell_qty += r["qty"]
    qty_net = round(buy_qty - sell_qty, 8)
    realized_pnl_inside_cluster = 0.0  # constant-price invariant
    return {
        "ticker": ticker,
        "price": price,
        "n_trades": n,
        "span_seconds": span_s,
        "span_hours": round(span_s / 3600.0, 4),
        "ts_first": ts_first.isoformat(),
        "ts_last": ts_last.isoformat(),
        "action_mix": action_mix,
        "buy_qty": round(buy_qty, 8),
        "sell_qty": round(sell_qty, 8),
        "qty_net": qty_net,
        "realized_pnl_inside_cluster": realized_pnl_inside_cluster,
    }


def build_frozen_mark_execution_skill(
    trades: Sequence[Any] | None,
    *,
    now: datetime | None = None,
    window_days: float = DEFAULT_WINDOW_DAYS,
    cluster_min: int = DEFAULT_CLUSTER_MIN,
    cluster_span_hours: float = DEFAULT_CLUSTER_SPAN_HOURS,
    occasional_pct: float = DEFAULT_OCCASIONAL_PCT,
    heavy_pct: float = DEFAULT_HEAVY_PCT,
) -> dict[str, Any]:
    """Pure frozen-mark cluster detector. Never raises.

    Inputs:
      ``trades`` — sequence of trade dicts. Mixed actions; caller does
        not need to pre-filter (HOLD/NO_DECISION are dropped).
      ``now`` — defaults to ``datetime.now(utc)``.
      ``window_days`` — analysis window measured backward from ``now``;
        trades older than this are dropped before clustering.
      ``cluster_min`` — minimum trade count for a (ticker, price)
        sequence to count as a cluster. ``2`` is the natural floor.
      ``cluster_span_hours`` — max wall-clock span across a cluster.
      ``occasional_pct`` / ``heavy_pct`` — verdict thresholds on the
        ``frozen_trade_pct`` stat.

    Output envelope (keys always present even on the NO_DATA path):
      ``as_of``, ``verdict``, ``headline``,
      ``window_days``, ``thresholds`` (dict),
      ``stats`` — ``n_trades`` (in-window), ``n_clusters``,
        ``n_frozen_trades`` (total trades belonging to ANY cluster),
        ``frozen_trade_pct``, ``worst_cluster_n_trades``,
        ``unique_tickers_affected``.
      ``clusters`` — list of cluster summaries, worst-first.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    window_days = max(0.0, float(window_days))
    cluster_min = max(2, int(cluster_min))
    cluster_span_hours = max(0.0, float(cluster_span_hours))
    occasional_pct = max(0.0, min(100.0, float(occasional_pct)))
    heavy_pct = max(0.0, min(100.0, float(heavy_pct)))
    # Heavy must be ≥ occasional; if a caller flips them we still emit
    # a sane verdict by widening occasional rather than raising.
    if heavy_pct < occasional_pct:
        heavy_pct = occasional_pct

    cutoff = now - timedelta(days=window_days)

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
        "cluster_min": cluster_min,
        "cluster_span_hours": cluster_span_hours,
        "occasional_pct": occasional_pct,
        "heavy_pct": heavy_pct,
        "min_trades_for_verdict": MIN_TRADES_FOR_VERDICT,
    }

    if len(normalized) < MIN_TRADES_FOR_VERDICT:
        return {
            "as_of": now.isoformat(timespec="seconds"),
            "verdict": "INSUFFICIENT_DATA",
            "headline": (
                f"only {len(normalized)} classifiable trades in the last "
                f"{int(window_days)}d (need ≥ {MIN_TRADES_FOR_VERDICT})"
            ),
            "window_days": window_days,
            "thresholds": thresholds,
            "stats": {
                "n_trades": len(normalized),
                "n_clusters": 0,
                "n_frozen_trades": 0,
                "frozen_trade_pct": 0.0,
                "worst_cluster_n_trades": 0,
                "unique_tickers_affected": 0,
            },
            "clusters": [],
        }

    clusters = _identical_price_clusters(
        normalized, cluster_min, cluster_span_hours,
    )

    n_frozen_trades = sum(c["n_trades"] for c in clusters)
    frozen_pct = round(100.0 * n_frozen_trades / len(normalized), 2)
    worst_n = max((c["n_trades"] for c in clusters), default=0)
    unique_tickers = len({c["ticker"] for c in clusters})

    if frozen_pct >= heavy_pct:
        verdict = "FROZEN_MARK_HEAVY"
        headline = (
            f"{frozen_pct:.1f}% of {len(normalized)} trades filled inside "
            f"a frozen-mark cluster — at least {len(clusters)} clusters "
            f"(worst: {worst_n} trades at one price)"
        )
    elif frozen_pct >= occasional_pct:
        verdict = "OCCASIONAL"
        headline = (
            f"{frozen_pct:.1f}% frozen-mark fills across {len(clusters)} "
            f"clusters — within tolerance but worth tracking"
        )
    else:
        verdict = "CLEAN"
        headline = (
            f"frozen-mark fills only {frozen_pct:.1f}% of {len(normalized)} "
            f"trades — price discovery is functioning"
        )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "verdict": verdict,
        "headline": headline,
        "window_days": window_days,
        "thresholds": thresholds,
        "stats": {
            "n_trades": len(normalized),
            "n_clusters": len(clusters),
            "n_frozen_trades": n_frozen_trades,
            "frozen_trade_pct": frozen_pct,
            "worst_cluster_n_trades": worst_n,
            "unique_tickers_affected": unique_tickers,
        },
        "clusters": clusters,
    }
