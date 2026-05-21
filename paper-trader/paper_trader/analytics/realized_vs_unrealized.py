"""Realized vs unrealized P&L decomposition — the "banked vs paper" split.

Every existing equity / P&L surface answers a different question:

* ``/api/portfolio`` reports a single scalar ``pnl_vs_start_pct``.
* ``/api/drawdown`` reports the worst peak-to-trough on the equity curve.
* ``/api/equity-integrity`` is a sanity check (no negative cash, no
  unexplained jumps).
* ``/api/pnl-attribution`` decomposes the *open* book into β·SPY +
  idiosyncratic.
* ``/api/trade-asymmetry`` aggregates *closed* round-trips into win-rate
  / payoff ratio / expectancy.
* ``/api/open-attribution`` is the open-book alpha vs SPY scalar.

What no surface answers: **of today's net P&L, how much is locked-in
banked (proceeds minus running cost basis on every sell that has already
fired) vs paper (open positions vs running avg cost)?** A $11.95 net
gain that is 100% realized is a fundamentally different desk than the
same headline gain that is 100% open-book paper — the second is one
adverse mark-to-market away from zero. Live evidence (2026-05-21 NVDA
earnings night, $1011.95 book): realized = $11.94, unrealized ≈ $0 —
the desk is BANKED. After the next BUY-and-mark-up cycle it could flip
to PAPER_HEAVY and a chat warning is the right surface for that.

Single source of truth: the chronological trade walk reproduces the
exact running avg-cost basis the live engine maintains in
``positions.avg_cost``. Tests pin the algebraic invariant

  realized_pnl_at(t) + unrealized_pnl_at(t)
      == total_value_at(t) − starting_value          (within float ε)

against synthetic trade ladders so any sign / double-count drift
surfaces immediately. Read-only / observational — never gates Opus, no
caps (AGENTS.md #2/#12).

The verdict ladder, **most-specific first** — actionable verdicts the
chat block fires on, neutral verdicts collapse to silence:

  * ``DRAWING_DOWN``   — net P&L < ``-DD_PCT`` of starting (the book is
    bleeding overall; both halves may contribute).
  * ``LEAKING_PAPER``  — realized > 0 but unrealized has now turned
    negative beyond noise — the open book is undoing banked gains
    (classic "give back the gains" pathology).
  * ``PAPER_HEAVY``    — net positive AND unrealized share of total
    gain ≥ ``PAPER_HEAVY_SHARE`` AND net gain pct ≥ ``MIN_NET_PCT`` —
    most of the headline gain is paper, one bad mark from zero.
  * ``BANKED``         — net positive AND realized covers the lion's
    share (≥ 1 − ``PAPER_HEAVY_SHARE``) AND net gain pct ≥
    ``MIN_NET_PCT``. The "locked in" state.
  * ``BALANCED``       — neither tail of the gain skew, net within
    noise band. Healthy mid-state.
  * ``NO_DATA``        — no trades or no equity-curve points.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Thresholds. Module-owned so tests read constants — a retune cannot
# false-fail them. Mirrors the decision_paralysis / hold_discipline
# precedent.
MIN_NET_PCT = 0.5          # below this absolute |net%|, gains are noise
DD_PCT = 0.5               # net% below −DD_PCT ⇒ DRAWING_DOWN
PAPER_HEAVY_SHARE = 0.66   # unrealized ≥ 66% of total positive gain ⇒ PAPER_HEAVY
LEAK_PCT = 0.25            # unrealized below −LEAK_PCT (of starting) when realized>0 ⇒ LEAKING_PAPER

# Cap the time-series payload — equity_curve has hundreds of rows at the
# 60s-open / 3600s-closed cadence; UI consumers want a bounded length.
MAX_SERIES_POINTS = 500


def _parse_ts(ts):
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _walk_realized(trades):
    """Walk ``trades`` (oldest→newest), maintain per-position running
    ``(qty, total_cost)``, and emit a cumulative-realized timeline.

    Returns ``[(ts_dt, cumulative_realized_usd)]`` with one entry per
    trade processed (in trade order). Trades whose timestamp does not
    parse are still walked (so realized accounting stays correct) but
    are appended with the previous timestamp — never raises.

    Position key: ``(ticker, type, strike, expiry)``. ``type`` defaults
    to ``"stock"`` when ``option_type`` is missing.

    Algorithm uses ``trade.value`` (signed gross dollars, already
    factors the option ×100 multiplier) directly so the same code-path
    handles equities and options.
    """
    state: dict[tuple, dict] = {}
    out: list[tuple[datetime | None, float]] = []
    cum = 0.0
    last_ts = None
    def _safe_float(x):
        try:
            return float(x) if x is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
    for t in trades:
        action = (t.get("action") or "").upper()
        qty = _safe_float(t.get("qty"))
        value = _safe_float(t.get("value"))
        ts = _parse_ts(t.get("timestamp"))
        if ts is not None:
            last_ts = ts
        typ = t.get("option_type") or "stock"
        key = (t.get("ticker"), typ, t.get("strike"), t.get("expiry"))
        rec = state.setdefault(key, {"qty": 0.0, "total_cost": 0.0})
        if action.startswith("BUY") and qty > 0:
            rec["qty"] += qty
            rec["total_cost"] += value
        elif action.startswith("SELL") and qty > 0:
            if rec["qty"] > 1e-9:
                # Average cost of the shares being sold (running basis).
                avg_cost_per_unit = rec["total_cost"] / rec["qty"]
                cost_of_sold = avg_cost_per_unit * qty
                cum += value - cost_of_sold
                rec["qty"] -= qty
                rec["total_cost"] -= cost_of_sold
                # Numerical hygiene on full close.
                if abs(rec["qty"]) < 1e-9:
                    rec["qty"] = 0.0
                    rec["total_cost"] = 0.0
            else:
                # Naked short — outside the engine's contract; treat as
                # full realization of the proceeds (degrade, don't raise).
                cum += value
        out.append((ts or last_ts, round(cum, 6)))
    return out


def _attach_realized(curve, realized_timeline):
    """For each equity-curve point ``(ts, total)``, attach the cumulative
    realized P&L from ``realized_timeline`` whose trade timestamp is
    ≤ the curve point's timestamp. Both inputs are walked in order.
    """
    rt_idx = 0
    rt_len = len(realized_timeline)
    cum = 0.0
    out = []
    for cv in curve:
        ts = cv["ts_dt"]
        # Advance the realized cursor while next trade ts ≤ curve ts.
        while rt_idx < rt_len:
            rt_ts, rt_cum = realized_timeline[rt_idx]
            if rt_ts is None or (ts is not None and rt_ts <= ts):
                cum = rt_cum
                rt_idx += 1
            else:
                break
        out.append({**cv, "realized": round(cum, 4)})
    return out


def build_realized_vs_unrealized(trades, equity_curve,
                                 starting_value=1000.0,
                                 now=None):
    """Time-decomposed banked-vs-paper P&L split + verdict.

    Parameters
    ----------
    trades : list[dict]
        ``Store.recent_trades()``-shaped ledger ordered **oldest→newest**.
        (``recent_trades(N)`` returns newest→oldest; callers must pass
        ``list(reversed(store.recent_trades(N)))``.)
    equity_curve : list[dict]
        ``Store.equity_history()``-shaped rows, each with ``timestamp``
        and ``total_value`` (and optionally ``cash``). Walked in
        chronological order; out-of-order inputs are sorted by ts.
    starting_value : float
        Reference baseline. Defaults to ``INITIAL_CASH = 1000`` —
        callers must pass the live ``store.starting_value()`` to honour
        any future schema change.
    now : datetime, optional
        Test seam.
    """
    now = now or datetime.now(timezone.utc)
    out = {
        "as_of": now.isoformat(timespec="seconds"),
        "starting_value": round(float(starting_value), 4),
        "current_value": None,
        "realized_pnl_usd": 0.0,
        "unrealized_pnl_usd": 0.0,
        "net_pnl_usd": 0.0,
        "realized_pnl_pct": 0.0,
        "unrealized_pnl_pct": 0.0,
        "net_pnl_pct": 0.0,
        "n_trades_walked": 0,
        "n_curve_points": 0,
        "series": [],
        "thresholds": {
            "min_net_pct": MIN_NET_PCT,
            "drawdown_pct": DD_PCT,
            "paper_heavy_share": PAPER_HEAVY_SHARE,
            "leak_pct": LEAK_PCT,
        },
        "verdict": "NO_DATA",
        "headline": "No equity-curve points yet — cannot decompose P&L.",
    }
    # Normalise curve: keep only points with parseable ts + numeric total.
    norm_curve = []
    for cv in equity_curve or []:
        ts = _parse_ts(cv.get("timestamp"))
        try:
            tv = float(cv.get("total_value"))
        except (TypeError, ValueError):
            continue
        if ts is None:
            continue
        norm_curve.append({"ts_dt": ts, "ts": ts.isoformat(timespec="seconds"),
                           "total": round(tv, 4)})
    norm_curve.sort(key=lambda r: r["ts_dt"])
    out["n_curve_points"] = len(norm_curve)
    if not norm_curve:
        return out

    # Walk trades + build cumulative realized timeline.
    rt_timeline = _walk_realized(trades or [])
    out["n_trades_walked"] = len(rt_timeline)

    annotated = _attach_realized(norm_curve, rt_timeline)
    starting = float(starting_value)
    series = []
    for row in annotated:
        total = row["total"]
        realized = row["realized"]
        unrealized = round(total - starting - realized, 4)
        series.append({"ts": row["ts"], "total": total,
                       "realized": realized, "unrealized": unrealized})

    # Compress the time-series for the wire — keep first + downsample
    # the middle + always keep the last `MAX_SERIES_POINTS // 2` newest.
    out["series"] = _compress_series(series, MAX_SERIES_POINTS)

    last = series[-1]
    out["current_value"] = last["total"]
    out["realized_pnl_usd"] = last["realized"]
    out["unrealized_pnl_usd"] = last["unrealized"]
    out["net_pnl_usd"] = round(last["realized"] + last["unrealized"], 4)
    if starting > 0:
        out["realized_pnl_pct"] = round(last["realized"] / starting * 100, 4)
        out["unrealized_pnl_pct"] = round(last["unrealized"] / starting * 100, 4)
        out["net_pnl_pct"] = round(out["net_pnl_usd"] / starting * 100, 4)

    # Verdict ladder.
    r_pct = out["realized_pnl_pct"]
    u_pct = out["unrealized_pnl_pct"]
    net_pct = out["net_pnl_pct"]
    r_usd = out["realized_pnl_usd"]
    u_usd = out["unrealized_pnl_usd"]
    net_usd = out["net_pnl_usd"]

    # LEAKING_PAPER is intentionally checked BEFORE DRAWING_DOWN: when
    # realized > 0 and the open book has gone underwater enough to
    # either erase the banked gain or be material on its own, the
    # diagnostic "the desk gave back its banked gain" is strictly more
    # specific than "net is red". DRAWING_DOWN remains the catch-all
    # for losses without a positive realized leg.
    leak_active = (
        r_usd > 0.0 and u_usd < 0.0
        and u_pct < -LEAK_PCT
        and (abs(u_usd) >= 0.5 * r_usd or net_pct < -MIN_NET_PCT)
    )
    if leak_active:
        out["verdict"] = "LEAKING_PAPER"
        out["headline"] = (
            f"LEAKING_PAPER — realized ${r_usd:.2f} ({r_pct:+.2f}%) "
            f"banked but open book is now ${u_usd:.2f} ({u_pct:+.2f}%) "
            f"underwater — the open positions are undoing locked gains.")
    elif net_pct < -DD_PCT:
        out["verdict"] = "DRAWING_DOWN"
        out["headline"] = (
            f"DRAWING_DOWN — book down ${net_usd:.2f} ({net_pct:+.2f}%): "
            f"realized ${r_usd:.2f} + unrealized ${u_usd:.2f}. "
            f"Both halves negative or open book is dragging banked into "
            f"the red.")
    elif (net_pct >= MIN_NET_PCT and u_usd > 0 and
          (r_usd + u_usd) > 0 and
          u_usd / (r_usd + u_usd) >= PAPER_HEAVY_SHARE):
        share = u_usd / (r_usd + u_usd)
        out["verdict"] = "PAPER_HEAVY"
        out["headline"] = (
            f"PAPER_HEAVY — {share*100:.0f}% of today's ${net_usd:.2f} "
            f"({net_pct:+.2f}%) gain is unrealized paper (${u_usd:.2f}). "
            f"One bad mark-to-market and the headline evaporates.")
    elif (net_pct >= MIN_NET_PCT and r_usd > 0 and
          (r_usd + u_usd) > 0 and
          r_usd / (r_usd + u_usd) >= (1.0 - PAPER_HEAVY_SHARE)):
        share = r_usd / (r_usd + u_usd) if (r_usd + u_usd) > 0 else 1.0
        out["verdict"] = "BANKED"
        out["headline"] = (
            f"BANKED — {share*100:.0f}% of today's ${net_usd:.2f} "
            f"({net_pct:+.2f}%) gain is locked-in realized (${r_usd:.2f}). "
            f"Open book sits ${u_usd:+.2f}.")
    else:
        out["verdict"] = "BALANCED"
        out["headline"] = (
            f"BALANCED — net ${net_usd:+.2f} ({net_pct:+.2f}%): "
            f"realized ${r_usd:+.2f}, unrealized ${u_usd:+.2f}.")
    return out


def _compress_series(series, max_n):
    """Downsample a long series to ``max_n`` points while keeping the
    very first and the most recent ``max_n // 2`` rows untouched.
    """
    if len(series) <= max_n:
        return list(series)
    tail_n = max_n // 2
    head_budget = max_n - tail_n - 1
    head_pool = series[:-tail_n]
    if head_budget <= 0:
        return [series[0]] + series[-tail_n:]
    step = max(len(head_pool) // head_budget, 1)
    head = [series[0]] + head_pool[::step][1:head_budget]
    return head + series[-tail_n:]
