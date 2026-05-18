"""Equity-curve integrity — *can the trader trust the recorded P&L history?*

``mark_integrity`` answers "is my book stale **right now**" (a point-in-time
roll-up of ``stale_mark`` over the current open positions). Nothing answers
the orthogonal, equally trader-critical question: **is the recorded
``equity_curve`` itself internally consistent over time, or has a mismark /
price glitch / option-settlement artifact / execution overdraw silently
corrupted the headline P&L the dashboard, Discord and ``/api/analytics``
Sharpe all derive from?**

A live desk that cannot trust its equity series cannot trust any of:
``/api/drawdown``, ``/api/benchmark``, the hourly P/L line, or the Sharpe in
``/api/analytics`` — they are *all* downstream of ``equity_curve``. This
builder is the time-series sibling of ``mark_integrity``:

  * **NEGATIVE_CASH** — any recorded point with ``cash < 0``. The live
    trader has *no hard cap* by design (AGENTS.md #12); ``_execute`` only
    blocks a BUY when ``snapshot['cash'] - notional < 0`` against the
    *snapshot* cash, and ``update_portfolio`` is called with stale positions
    then corrected by a later snapshot — so a transient over-deployed /
    negative-cash equity point IS physically reachable and would mean the
    book was over-drawn. This must never silently pass.
  * **NONPOSITIVE_EQUITY** — ``total_value <= 0`` (ruin / corrupt row).
  * **SUSPECT_JUMP** — consecutive points whose ``|Δtotal_value|`` exceeds
    ``jump_pct_threshold`` of the prior value **with no trade in the window
    between them**. A no-trade window can only move by mark-to-market drift
    of *held* positions; a large single-cycle swing with zero trades is the
    signature of a mismark / stale-price unfreeze / option intrinsic
    settlement, not a real P&L move. A jump that *does* have a trade in its
    window is expected (cash↔position conversion at a price that differs
    from the mark) and is NOT flagged.

Pure, never raises (AGENTS.md #2/#12 — advisory only, gates nothing, adds
no caps; the ``mark_integrity`` / behavioural-builder ``_safe`` contract).
Garbage rows are skipped, never an exception. All timestamp comparisons are
lexical on the store's fixed-offset-UTC ISO strings — the documented,
codebase-wide ordering invariant (see ``signals._age_hours`` /
``reporter._activity_counts``).

Verdict ladder (severity-ordered, sample-size honest like the rest of the
desk):

  * ``NO_DATA``  — fewer than ``min_points`` usable points; nothing to say.
  * ``CLEAN``    — every point has cash >= 0 and positive equity, and no
    unexplained (no-trade) jump exceeds the threshold.
  * ``SUSPECT``  — >=1 no-trade jump over threshold, but no corrupt point.
  * ``CORRUPT``  — >=1 negative-cash or non-positive-equity point (the
    headline P&L history is unreliable; dominates SUSPECT).
"""
from __future__ import annotations

# A single-cycle move this large with NO trade in the window is anomalous for
# this $1000 book even with 3x leveraged ETFs (a 3x ETF would need its
# underlying to gap ~2.7% in one cycle while it is the *entire* book). Tuned
# to flag mismark/glitch artifacts, not ordinary leveraged drift.
DEFAULT_JUMP_PCT = 8.0
# Below this many usable points a "jump" is just the cold-start of the curve.
DEFAULT_MIN_POINTS = 3
# Float-noise guard: cash is dollars; -$0.01 is rounding, -$5 is an overdraw.
_CASH_EPS = -0.01
_MAX_OFFENDERS = 10  # cap the detail lists so the payload stays lean


def _f(x) -> float | None:
    """Best-effort float; None on garbage (never raises)."""
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _sorted_points(equity_points: list[dict] | None) -> list[dict]:
    """Usable points (parseable total_value) sorted ascending by timestamp.

    Defensive: callers pass ``store.equity_curve()`` which is already
    ascending, but a pure builder must not assume its input ordering."""
    rows = []
    for p in (equity_points or []):
        if not isinstance(p, dict):
            continue
        tv = _f(p.get("total_value"))
        ts = p.get("timestamp")
        if tv is None or not ts:
            continue
        rows.append((str(ts), tv, _f(p.get("cash"))))
    rows.sort(key=lambda r: r[0])
    return [{"timestamp": ts, "total_value": tv, "cash": cash}
            for ts, tv, cash in rows]


def _trade_stamps(trades: list[dict] | None) -> list[str]:
    """Sorted ISO timestamps of every trade (newest-first or oldest-first
    input both fine — we sort). Used only for window membership."""
    out = []
    for t in (trades or []):
        if not isinstance(t, dict):
            continue
        ts = t.get("timestamp")
        if ts:
            out.append(str(ts))
    out.sort()
    return out


def _trade_in_window(stamps: list[str], lo: str, hi: str) -> bool:
    """True if any trade timestamp falls in ``(lo, hi]``. Linear scan — the
    curve windows are few and trades modest for a $1000 paper book; a bisect
    is premature and would obscure the intent."""
    for s in stamps:
        if lo < s <= hi:
            return True
    return False


def build_equity_integrity(
    equity_points: list[dict] | None,
    trades: list[dict] | None,
    *,
    jump_pct_threshold: float = DEFAULT_JUMP_PCT,
    min_points: int = DEFAULT_MIN_POINTS,
) -> dict:
    """Time-series self-consistency audit of the recorded equity curve.

    Args:
        equity_points: ``store.equity_curve()`` rows
            (``{timestamp,total_value,cash,sp500_price}``); any order.
        trades: ``store.recent_trades()`` rows; only ``timestamp`` is used.
        jump_pct_threshold: a no-trade |Δtotal| over this % of the prior
            value is flagged SUSPECT.
        min_points: below this many usable points → NO_DATA.

    Returns a JSON-able dict; never raises.
    """
    pts = _sorted_points(equity_points)
    n = len(pts)
    base = {
        "jump_pct_threshold": jump_pct_threshold,
        "n_points": n,
        "window_start": pts[0]["timestamp"] if pts else None,
        "window_end": pts[-1]["timestamp"] if pts else None,
    }
    if n < min_points:
        return {
            **base,
            "verdict": "NO_DATA",
            "headline": (
                f"Only {n} usable equity point(s) (<{min_points}) — "
                f"too short to audit consistency."),
            "n_negative_cash": 0,
            "min_cash_usd": None,
            "n_nonpositive_equity": 0,
            "n_suspect_jumps": 0,
            "worst_jump": None,
            "negative_cash_points": [],
            "nonpositive_equity_points": [],
            "suspect_jumps": [],
        }

    neg_cash: list[dict] = []
    nonpos_eq: list[dict] = []
    min_cash: float | None = None
    for p in pts:
        c = p["cash"]
        if c is not None:
            if min_cash is None or c < min_cash:
                min_cash = c
            if c < _CASH_EPS:
                neg_cash.append({"timestamp": p["timestamp"],
                                 "cash": round(c, 2),
                                 "total_value": round(p["total_value"], 2)})
        if p["total_value"] <= 0.0:
            nonpos_eq.append({"timestamp": p["timestamp"],
                              "total_value": round(p["total_value"], 2)})

    stamps = _trade_stamps(trades)
    jumps: list[dict] = []
    for prev, cur in zip(pts, pts[1:]):
        pv = prev["total_value"]
        if pv <= 0.0:
            continue  # pct undefined; the non-positive point is flagged above
        delta = cur["total_value"] - pv
        pct = delta / pv * 100.0
        if abs(pct) < jump_pct_threshold:
            continue
        if _trade_in_window(stamps, prev["timestamp"], cur["timestamp"]):
            continue  # a trade explains the swing — expected, not suspect
        jumps.append({
            "from_ts": prev["timestamp"],
            "to_ts": cur["timestamp"],
            "from_value": round(pv, 2),
            "to_value": round(cur["total_value"], 2),
            "delta_usd": round(delta, 2),
            "delta_pct": round(pct, 2),
        })

    worst_jump = (max(jumps, key=lambda j: abs(j["delta_pct"]))
                  if jumps else None)
    corrupt = bool(neg_cash or nonpos_eq)

    if corrupt:
        verdict = "CORRUPT"
        bits = []
        if neg_cash:
            bits.append(
                f"{len(neg_cash)} negative-cash point(s) (min "
                f"${round(min_cash, 2) if min_cash is not None else 0.0})")
        if nonpos_eq:
            bits.append(f"{len(nonpos_eq)} non-positive-equity point(s)")
        headline = (
            f"Recorded equity is CORRUPT across {n} points: "
            + "; ".join(bits)
            + " — the book was over-drawn / mismarked; P&L history "
              "(drawdown, benchmark, Sharpe, hourly P/L) is unreliable.")
    elif jumps:
        verdict = "SUSPECT"
        wj = worst_jump
        headline = (
            f"{len(jumps)} unexplained equity jump(s) >="
            f"{jump_pct_threshold:g}% with no trade in the window across "
            f"{n} points; largest {wj['delta_pct']:+.2f}% "
            f"(${wj['delta_usd']:+.2f}) at {wj['to_ts'][:19]} — likely a "
            f"mismark / stale-price unfreeze / option-settlement artifact, "
            f"not a real P&L move.")
    else:
        verdict = "CLEAN"
        headline = (
            f"Equity curve consistent across {n} points — cash never "
            f"negative, no unexplained jump >={jump_pct_threshold:g}%.")

    return {
        **base,
        "verdict": verdict,
        "headline": headline,
        "n_negative_cash": len(neg_cash),
        "min_cash_usd": round(min_cash, 2) if min_cash is not None else None,
        "n_nonpositive_equity": len(nonpos_eq),
        "n_suspect_jumps": len(jumps),
        "worst_jump": worst_jump,
        "negative_cash_points": neg_cash[:_MAX_OFFENDERS],
        "nonpositive_equity_points": nonpos_eq[:_MAX_OFFENDERS],
        "suspect_jumps": sorted(
            jumps, key=lambda j: abs(j["delta_pct"]), reverse=True
        )[:_MAX_OFFENDERS],
    }
