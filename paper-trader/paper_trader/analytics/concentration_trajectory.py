"""Concentration trajectory — is single-name concentration RISING, STEADY,
or FALLING over time?

Every existing concentration surface in this repo is **point-in-time**:

* ``risk_mirror`` reports current ``top_weight_pct`` / ``weight_hhi`` /
  ``effective_positions_naive``, fed into the live Opus prompt.
* ``analytics/correlation`` produces those numbers from a snapshot of the
  open book + an injected price-history dict.
* ``analytics/sector_exposure`` reports current per-sector %.
* ``/api/risk`` exposes ``concentration_top1_pct`` / ``top3_pct`` as a
  scalar.
* ``/api/analytics`` reports the same scalar.

What none of them answers is the first-derivative question: *over the
past N days, has the book's single-name concentration been rising,
falling, or steady?* A book sitting at 65% top-1 today reads identically
in every existing surface whether it ramped from 30% → 65% over a week
(concentration creep — the desk drifted in) or jumped 0% → 65% in the
last cycle (a single fill blew it up — different operator response).

This builder is the missing slope view. It walks the trade ledger
chronologically, snapshots the book at the close of each calendar day in
the window, marks each held position to that day's close (injected
``daily_closes`` dict — same prod-builder/test-seam split as
``correlation`` / ``thesis_drift``), and emits

* a daily ``series`` of ``(date, top1_pct, top1_ticker, top3_pct, hhi,
  effective_positions, n_positions, deployed_usd)``
* current snapshot fields restated for chat consumers
* the window's ``delta_top1_pct`` / ``max_top1_pct`` / ``min_top1_pct``
* a verdict ladder + headline

Verdict ladder (**most-specific first** so the chat surface fires on the
sharpest signal):

* ``CONCENTRATION_SPIKE`` — top-1 jumped from ``<SPIKE_FROM_PCT`` on the
  prior snapshot to ``>=SPIKE_TO_PCT`` on the latest snapshot. A single
  cycle blew up the book — the operator should know *now*, not after a
  ramp.
* ``RAMPING_UP`` — top-1 rose by at least ``RAMP_DELTA_PCT`` over the
  window AND the latest top-1 ≥ ``RAMP_TO_PCT``. Slow concentration
  creep into a single name.
* ``DECONCENTRATING`` — top-1 fell by at least ``RAMP_DELTA_PCT`` AND
  the first snapshot was ≥ ``DECONC_FROM_PCT``. The desk pared a
  concentrated bet.
* ``CONCENTRATED_STEADY`` — top-1 mean over the window ≥ ``STEADY_PCT``
  AND the spread (max − min) ≤ ``STEADY_BAND_PCT``. Parked in one name.
* ``DIVERSIFIED`` — max top-1 over the window < ``DIVERSIFIED_CEILING``.
  The book never piled into a single name in the window.
* ``BALANCED`` — neither extreme; mid-state.
* ``INSUFFICIENT_DATA`` — fewer than ``MIN_SNAPSHOTS`` daily snapshots.
* ``NO_DATA`` — no trades or no open positions ever in the window.

**Stocks-only** by deliberate carve-out (mirroring ``correlation`` and
``open_attribution``): mixing a non-linear option Greeks payoff into a
linear "% of book" concentration metric is not meaningful, and the live
book's documented pathologies are name-level on the equity leg.
Option-rows are walked for state but excluded from the per-snapshot
concentration math.

**Pure builder + test seam.** No DB, no network. The endpoint
(``/api/concentration-trajectory``) owns the daily-close fetch via the
existing ``_daily_history_cached`` helper. Read-only / observational —
never gates Opus, no caps (AGENTS.md invariants #2/#12). Never raises:
garbage rows, unparseable timestamps, missing closes for a snapshot date
all degrade to "drop that ticker for that snapshot", never an
exception.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

# Window cap — daily snapshots emitted in ``series``. 30 days is the
# operator-facing "month" view; further back is rarely the question.
MAX_SNAPSHOTS = 30
MIN_SNAPSHOTS = 3

# CONCENTRATION_SPIKE — single-cycle blow-up. Latest crosses SPIKE_TO_PCT
# while the prior snapshot was below SPIKE_FROM_PCT (i.e. it really IS a
# spike, not the tail of a long ramp).
SPIKE_FROM_PCT = 40.0
SPIKE_TO_PCT = 60.0

# RAMPING_UP / DECONCENTRATING — slope-over-window with magnitude floors
# so a 1% drift doesn't trip the verdict.
RAMP_DELTA_PCT = 15.0
RAMP_TO_PCT = 50.0
DECONC_FROM_PCT = 50.0

# CONCENTRATED_STEADY — high mean, low spread.
STEADY_PCT = 50.0
STEADY_BAND_PCT = 10.0

# DIVERSIFIED — book never piled in over the window.
DIVERSIFIED_CEILING = 35.0


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


def _safe_float(x, default=0.0):
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default


def _close_on_or_before(rows, target_date_str):
    """Return the close price from ``rows`` (``[(YYYY-MM-DD, close), ...]``,
    sorted ascending) on ``target_date_str`` or the latest available
    earlier date. ``None`` when no row is on-or-before. Linear scan — the
    inputs are small (≤ ~252 trading days even for a 1y window)."""
    if not rows:
        return None
    chosen = None
    for d_str, c in rows:
        if d_str <= target_date_str:
            chosen = c
        else:
            break
    return chosen


def _stock_positions_after_trade(state, trade):
    """Update ``state`` ({ticker: qty}) by applying one ``trade`` row.
    Options skipped (their state would only matter if we marked them at
    snapshot time, which we deliberately do not). Returns ``state``."""
    typ = trade.get("option_type") or "stock"
    if typ != "stock":
        return state
    action = (trade.get("action") or "").upper()
    qty = _safe_float(trade.get("qty"))
    ticker = (trade.get("ticker") or "").strip().upper()
    if not ticker or qty <= 0:
        return state
    if action.startswith("BUY"):
        state[ticker] = state.get(ticker, 0.0) + qty
    elif action.startswith("SELL"):
        cur = state.get(ticker, 0.0)
        new = cur - qty
        if new <= 1e-9:
            state.pop(ticker, None)
        else:
            state[ticker] = new
    return state


def _concentration_metrics(weights):
    """Given a list of position values (stocks only), compute the
    concentration scalars used by both the per-snapshot row and the
    headline. ``weights`` may be empty; in that case every numeric field
    is 0/None and ``n_positions`` is 0."""
    out = {
        "top1_pct": 0.0,
        "top1_ticker": None,
        "top3_pct": 0.0,
        "hhi": 0.0,
        "effective_positions": 0.0,
        "n_positions": 0,
        "deployed_usd": 0.0,
    }
    # weights: list of (ticker, value). Filter sub-cent dust.
    clean = [(t, v) for t, v in weights if v is not None and v > 1e-6]
    if not clean:
        return out
    clean.sort(key=lambda r: r[1], reverse=True)
    total = sum(v for _, v in clean)
    out["deployed_usd"] = round(total, 4)
    out["n_positions"] = len(clean)
    if total <= 0:
        return out
    top1_t, top1_v = clean[0]
    out["top1_ticker"] = top1_t
    out["top1_pct"] = round(top1_v / total * 100.0, 4)
    top3_v = sum(v for _, v in clean[:3])
    out["top3_pct"] = round(top3_v / total * 100.0, 4)
    fractions = [v / total for _, v in clean]
    hhi = sum(f * f for f in fractions)
    out["hhi"] = round(hhi, 6)
    # 1 / HHI is the effective N — collapses toward 1 as one name
    # dominates regardless of how many other tiny names sit on the book.
    out["effective_positions"] = round(1.0 / hhi, 4) if hhi > 0 else 0.0
    return out


def _verdict_for(series):
    """Compute the verdict + headline from the daily ``series``. Series
    is ordered oldest→newest; latest is the chat-actionable point."""
    if not series:
        return "NO_DATA", "No daily snapshots — concentration trajectory unavailable."
    if len(series) < MIN_SNAPSHOTS:
        return ("INSUFFICIENT_DATA",
                f"Only {len(series)} daily snapshot(s) — need ≥{MIN_SNAPSHOTS} to read trajectory.")
    latest = series[-1]
    first = series[0]
    top1_now = latest["top1_pct"]
    top1_first = first["top1_pct"]
    top1_max = max(s["top1_pct"] for s in series)
    top1_min = min(s["top1_pct"] for s in series)
    top1_mean = sum(s["top1_pct"] for s in series) / len(series)
    tk_now = latest["top1_ticker"] or "—"
    n_pos = latest["n_positions"]

    # CONCENTRATION_SPIKE — most specific. Fires when any adjacent pair
    # in the window jumps from <SPIKE_FROM to >=SPIKE_TO in ONE cycle
    # AND the latest snapshot is still concentrated (the spike hasn't
    # decayed). Scanning all pairs (not just the last) captures the
    # operator-relevant pathology where the fill happened mid-window
    # and the book has parked at the elevated level since — the desk
    # still needs to know the spike *happened* within the window.
    spike_pair = None
    for i in range(len(series) - 1):
        prior_pct = series[i]["top1_pct"]
        nxt_pct = series[i + 1]["top1_pct"]
        if prior_pct < SPIKE_FROM_PCT and nxt_pct >= SPIKE_TO_PCT:
            spike_pair = (i, prior_pct, nxt_pct,
                          series[i + 1]["top1_ticker"] or tk_now,
                          series[i]["date"], series[i + 1]["date"])
            break  # first (oldest) spike is reported
    if spike_pair is not None and top1_now >= SPIKE_TO_PCT:
        idx, p_pct, n_pct, tk, d_from, d_to = spike_pair
        delta = n_pct - p_pct
        return ("CONCENTRATION_SPIKE",
                f"CONCENTRATION_SPIKE — {tk} jumped from "
                f"{p_pct:.1f}% on {d_from} → {n_pct:.1f}% on {d_to} "
                f"({delta:+.1f}pp in one cycle); current top-1 "
                f"{top1_now:.1f}%. A single fill blew up single-name "
                f"exposure.")

    # RAMPING_UP — sustained climb, latest meaningfully concentrated.
    if top1_now - top1_first >= RAMP_DELTA_PCT and top1_now >= RAMP_TO_PCT:
        return ("RAMPING_UP",
                f"RAMPING_UP — {tk_now} climbed {top1_first:.1f}% → "
                f"{top1_now:.1f}% (top-1 of {n_pos} name(s)) over "
                f"{len(series)} day(s) — concentration creep into one "
                f"name.")

    # DECONCENTRATING — sustained fall from a previously concentrated state.
    if top1_first - top1_now >= RAMP_DELTA_PCT and top1_first >= DECONC_FROM_PCT:
        return ("DECONCENTRATING",
                f"DECONCENTRATING — top-1 weight fell {top1_first:.1f}% "
                f"→ {top1_now:.1f}% over {len(series)} day(s). The desk "
                f"pared the concentrated bet.")

    # CONCENTRATED_STEADY — parked in one name with a narrow band.
    if top1_mean >= STEADY_PCT and (top1_max - top1_min) <= STEADY_BAND_PCT:
        return ("CONCENTRATED_STEADY",
                f"CONCENTRATED_STEADY — top-1 averaged {top1_mean:.1f}% "
                f"(range {top1_min:.1f}%–{top1_max:.1f}%) over "
                f"{len(series)} day(s); book parked in {tk_now}.")

    # DIVERSIFIED — never piled into a single name in the window.
    if top1_max < DIVERSIFIED_CEILING:
        return ("DIVERSIFIED",
                f"DIVERSIFIED — top-1 never exceeded "
                f"{DIVERSIFIED_CEILING:.0f}% ({top1_max:.1f}% peak) over "
                f"{len(series)} day(s).")

    return ("BALANCED",
            f"BALANCED — top-1 {top1_now:.1f}% (range "
            f"{top1_min:.1f}%–{top1_max:.1f}%, mean {top1_mean:.1f}%). "
            f"No directional concentration signal over {len(series)} "
            f"day(s).")


def build_concentration_trajectory(trades, daily_closes, now=None,
                                   window_days=None):
    """Build the daily-snapshot concentration trajectory.

    Parameters
    ----------
    trades : list[dict]
        Store.recent_trades-shaped rows, ordered **oldest → newest**.
        (Callers using ``store.recent_trades(N)`` — which returns
        newest-first — must pass ``list(reversed(...))``.)
    daily_closes : dict[str, list[tuple[str, float]]]
        Per-ticker daily-close history, ``ticker → [(YYYY-MM-DD, close)]``
        sorted ascending by date. The endpoint fetches this via
        ``_daily_history_cached``; tests pass it directly. Missing or
        empty rows for a ticker mean that ticker's value is dropped from
        snapshots where no on-or-before close exists.
    now : datetime, optional
        Test seam. Defaults to ``datetime.now(tz=utc)``.
    window_days : int, optional
        Number of trailing daily snapshots to emit. Defaults to
        ``MAX_SNAPSHOTS``. Capped to ``[MIN_SNAPSHOTS, MAX_SNAPSHOTS]``.
    """
    now = now or datetime.now(timezone.utc)
    if window_days is None:
        window_days = MAX_SNAPSHOTS
    window_days = max(MIN_SNAPSHOTS, min(MAX_SNAPSHOTS, int(window_days)))

    out = {
        "as_of": now.isoformat(timespec="seconds"),
        "window_days": 0,
        "n_trades_walked": 0,
        "series": [],
        "current": {
            "top1_pct": 0.0,
            "top1_ticker": None,
            "top3_pct": 0.0,
            "hhi": 0.0,
            "effective_positions": 0.0,
            "n_positions": 0,
            "deployed_usd": 0.0,
        },
        "delta_top1_pct": None,
        "max_top1_pct": None,
        "min_top1_pct": None,
        "thresholds": {
            "spike_from_pct": SPIKE_FROM_PCT,
            "spike_to_pct": SPIKE_TO_PCT,
            "ramp_delta_pct": RAMP_DELTA_PCT,
            "ramp_to_pct": RAMP_TO_PCT,
            "deconc_from_pct": DECONC_FROM_PCT,
            "steady_pct": STEADY_PCT,
            "steady_band_pct": STEADY_BAND_PCT,
            "diversified_ceiling": DIVERSIFIED_CEILING,
            "min_snapshots": MIN_SNAPSHOTS,
            "max_snapshots": MAX_SNAPSHOTS,
        },
        "verdict": "NO_DATA",
        "headline": "No trades — concentration trajectory unavailable.",
    }

    # Normalise + sort trades chronologically (tolerate unparseable
    # timestamps by dropping them — same discipline as
    # realized_vs_unrealized's _walk_realized degrade-don't-raise path).
    norm_trades = []
    for t in trades or []:
        if not isinstance(t, dict):
            continue
        ts = _parse_ts(t.get("timestamp"))
        if ts is None:
            continue
        norm_trades.append((ts, t))
    norm_trades.sort(key=lambda r: r[0])
    out["n_trades_walked"] = len(norm_trades)
    if not norm_trades:
        return out

    # Build snapshot dates: last `window_days` calendar dates ending
    # at now.date() (inclusive). Dates older than the first trade are
    # filtered out — a book that didn't exist yet has no concentration
    # to track.
    today = now.astimezone(timezone.utc).date()
    first_trade_date = norm_trades[0][0].astimezone(timezone.utc).date()
    snapshot_dates = []
    for i in range(window_days - 1, -1, -1):
        d = today - timedelta(days=i)
        if d >= first_trade_date:
            snapshot_dates.append(d)
    if not snapshot_dates:
        return out

    # Walk trades, applying them up to and including each snapshot date
    # boundary, then snapshot. Single forward pass; trade-cursor never
    # re-rewinds.
    state: dict[str, float] = {}
    trade_idx = 0
    n_trades = len(norm_trades)
    series = []
    for d in snapshot_dates:
        # UTC end-of-day: any trade whose ts.date() <= d is included.
        while trade_idx < n_trades:
            ts, tr = norm_trades[trade_idx]
            if ts.astimezone(timezone.utc).date() <= d:
                _stock_positions_after_trade(state, tr)
                trade_idx += 1
            else:
                break
        # Mark to market using daily_closes on-or-before d.
        d_str = d.isoformat()
        weights = []
        for tk, qty in state.items():
            if qty <= 1e-9:
                continue
            rows = daily_closes.get(tk) if isinstance(daily_closes, dict) else None
            close = _close_on_or_before(rows or [], d_str)
            if close is None or close <= 0:
                continue
            weights.append((tk, qty * close))
        m = _concentration_metrics(weights)
        m["date"] = d_str
        series.append(m)

    # Trim leading rows with zero positions (book hadn't opened yet at
    # those snapshots even though the date is on-or-after first_trade —
    # e.g. closes weren't yet available).
    while series and series[0]["n_positions"] == 0:
        series.pop(0)

    out["series"] = series
    out["window_days"] = len(series)

    if not series:
        out["verdict"] = "NO_DATA"
        out["headline"] = "No deployed-book snapshots — concentration trajectory unavailable."
        return out

    latest = series[-1]
    out["current"] = {
        "top1_pct": latest["top1_pct"],
        "top1_ticker": latest["top1_ticker"],
        "top3_pct": latest["top3_pct"],
        "hhi": latest["hhi"],
        "effective_positions": latest["effective_positions"],
        "n_positions": latest["n_positions"],
        "deployed_usd": latest["deployed_usd"],
    }
    if series:
        top1s = [s["top1_pct"] for s in series]
        out["max_top1_pct"] = round(max(top1s), 4)
        out["min_top1_pct"] = round(min(top1s), 4)
        out["delta_top1_pct"] = round(top1s[-1] - top1s[0], 4)

    verdict, headline = _verdict_for(series)
    out["verdict"] = verdict
    out["headline"] = headline
    return out


if __name__ == "__main__":  # pragma: no cover
    import json
    # Tiny smoke: two-day ramp.
    today = datetime(2026, 5, 21, tzinfo=timezone.utc)
    yest = today - timedelta(days=1)
    trades = [
        {"timestamp": yest.isoformat(), "ticker": "AAPL", "action": "BUY",
         "qty": 1, "price": 100, "value": 100, "option_type": None},
        {"timestamp": today.isoformat(), "ticker": "NVDA", "action": "BUY",
         "qty": 4, "price": 200, "value": 800, "option_type": None},
    ]
    closes = {
        "AAPL": [(yest.date().isoformat(), 100.0),
                 (today.date().isoformat(), 100.0)],
        "NVDA": [(today.date().isoformat(), 200.0)],
    }
    print(json.dumps(build_concentration_trajectory(trades, closes, now=today),
                     indent=2, default=str))
