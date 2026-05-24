"""Rolling cash opportunity cost vs the S&P 500.

The lifetime ``/api/benchmark`` panel says "lagging buy-and-hold S&P 500
by 2.22pp" — true, structural, but blunt. The operator's actionable
question is per-window: "of the dollars I am holding in CASH right now,
how much have they cost me in the last 24h / 7d while SPY moved Y%?".
Cash is the bot's largest position 100% of the time it is flat (it is
flat right now); a per-window rollup is the first surface that puts a
USD number on idle cash.

``build_cash_drag(equity_curve, windows_h=...)`` is the pure builder.
It walks the same ``equity_curve`` rows the existing benchmark analytics
read (single source of truth — no second store hop, no second SPY mark
re-fetch). For each requested window length in hours it computes:

  * ``avg_cash_usd`` — TIME-WEIGHTED average of the ``cash`` column
    inside the window (trapezoidal between consecutive points); falls
    back to the simple mean if only one point is in window.
  * ``sp500_return_pct`` — index-level percent change from the first
    benchmarkable point in window to the latest.
  * ``cash_drag_usd`` — ``avg_cash_usd * sp500_return_pct / 100``.
    Sign convention: POSITIVE drag means cash COST you money (SPY rose
    while you held cash); NEGATIVE drag means cash SAVED you money
    (SPY fell, so being out paid off). The headline carries the verdict
    word, so a UI consumer never has to invert the sign manually.

Per-window verdicts: ``COSTLY_CASH`` (drag > +$0.50) /
``HELPFUL_CASH`` (drag < -$0.50) / ``NEUTRAL`` (|drag| ≤ $0.50, or
SPY essentially flat) / ``INSUFFICIENT`` (window has < 2 benchmarkable
points or spans < 60% of the requested hours — honesty ladder mirrors
``benchmark.py``'s state ladder).

The top-level ``verdict`` is the worst (most costly) per-window
verdict so a single-line UI banner surfaces the highest-pain window.

Pure & network-free; never raises. Designed so the existing equity_curve
read in the dashboard's benchmark endpoint can be reused as the input
list (no re-query). Memory note: live ``equity_curve`` history is
shallow (~5 days), so the 7d arm typically emits OK once and the 30d
arm typically emits INSUFFICIENT — exactly the sample-size honesty the
``benchmark.py`` ladder pioneered.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Default rolling windows — chosen against the documented ~5-day depth
# of the live equity_curve. 168h (7d) is the longest that will hit OK on
# the typical live book; 720h (30d) is included for the day this stack
# accumulates more history without changing the contract.
_DEFAULT_WINDOWS_H = (24.0, 168.0, 720.0)

# A window is benchmarkable when its actual span (newest − oldest) is at
# least this fraction of the requested window length. Below this floor
# the SPY return is dominated by intra-day noise on a stub of data and
# the verdict is withheld.
_MIN_COVERAGE_FRACTION = 0.6

# Verdict band around zero drag — below this absolute USD threshold the
# window is neutral, not "costly" or "helpful". $0.50 is small enough
# that a $1000 book over 24h with SPY moving 0.05% (≈ noise on a
# weekend's overnight futures) does not flip the verdict.
_NEUTRAL_BAND_USD = 0.50


def _parse_ts(s) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _benchmarkable(row: dict) -> bool:
    """A row is benchmarkable only when it carries a parsable timestamp,
    a non-negative cash value, AND a positive S&P 500 mark. Mirrors the
    ``benchmark._usable`` discipline — yfinance hiccups on a cold cycle
    can null the index mark even when the bot recorded its own value."""
    try:
        if _parse_ts(row.get("timestamp")) is None:
            return False
        cash = row.get("cash")
        if cash is None or float(cash) < 0:
            return False
        sp = row.get("sp500_price")
        return sp is not None and float(sp) > 0
    except Exception:
        return False


def _time_weighted_mean(points: list[tuple[datetime, float]]) -> float:
    """Trapezoidal time-weighted mean across (ts, value) pairs ordered
    by ts. A single point degrades to its own value; two points → the
    trapezoid average; >2 → the trapezoid sum divided by the total
    span. Equal timestamps degrade to the simple mean."""
    if not points:
        return 0.0
    if len(points) == 1:
        return float(points[0][1])
    total_secs = (points[-1][0] - points[0][0]).total_seconds()
    if total_secs <= 0:
        return sum(p[1] for p in points) / len(points)
    weighted = 0.0
    for i in range(len(points) - 1):
        t0, v0 = points[i]
        t1, v1 = points[i + 1]
        dt = (t1 - t0).total_seconds()
        if dt <= 0:
            continue
        weighted += (v0 + v1) / 2.0 * dt
    return weighted / total_secs


def _window_block(window_h: float, rows: list[dict], now: datetime) -> dict:
    """Per-window block. ``rows`` are pre-filtered to ``_benchmarkable``
    and ordered ascending by timestamp; ``now`` is the snapshot anchor.
    Returns the dict the top-level builder embeds under ``windows[]``."""
    block = {
        "window_hours": round(window_h, 2),
        "state": "INSUFFICIENT",
        "verdict": None,
        "n_points": 0,
        "span_hours": 0.0,
        "avg_cash_usd": None,
        "sp500_return_pct": None,
        "cash_drag_usd": None,
        "headline": f"{int(window_h)}h: insufficient history.",
    }
    cutoff = now - timedelta(hours=window_h)
    in_window: list[dict] = [r for r in rows
                             if _parse_ts(r["timestamp"]) >= cutoff]
    block["n_points"] = len(in_window)
    if len(in_window) < 2:
        return block
    first_ts = _parse_ts(in_window[0]["timestamp"])
    last_ts = _parse_ts(in_window[-1]["timestamp"])
    span_h = (last_ts - first_ts).total_seconds() / 3600.0
    block["span_hours"] = round(span_h, 2)
    if span_h < window_h * _MIN_COVERAGE_FRACTION:
        block["headline"] = (
            f"{int(window_h)}h: insufficient history "
            f"({block['n_points']} pts spanning {span_h:.1f}h)."
        )
        return block

    # Numerics from here on.
    sp_open = float(in_window[0]["sp500_price"])
    sp_close = float(in_window[-1]["sp500_price"])
    sp_ret_pct = (sp_close / sp_open - 1.0) * 100.0 if sp_open > 0 else 0.0

    pts = [(_parse_ts(r["timestamp"]), float(r["cash"])) for r in in_window]
    avg_cash = _time_weighted_mean(pts)
    drag = avg_cash * sp_ret_pct / 100.0

    block["state"] = "OK"
    block["avg_cash_usd"] = round(avg_cash, 2)
    block["sp500_return_pct"] = round(sp_ret_pct, 4)
    block["cash_drag_usd"] = round(drag, 2)

    if abs(drag) <= _NEUTRAL_BAND_USD:
        verdict = "NEUTRAL"
        verb = "essentially flat"
    elif drag > 0:
        verdict = "COSTLY_CASH"
        verb = f"cost you ${drag:.2f}"
    else:
        verdict = "HELPFUL_CASH"
        verb = f"saved you ${-drag:.2f}"
    block["verdict"] = verdict
    block["headline"] = (
        f"{int(window_h)}h: cash {verb} "
        f"(SPY {sp_ret_pct:+.2f}%, avg cash ${avg_cash:.2f})."
    )
    return block


def _worst_verdict(blocks: list[dict]) -> tuple[str, dict | None]:
    """Returns the (verdict, block) for the single window with the most
    COSTLY cash. If no OK block exists, falls back to the first
    INSUFFICIENT block's verdict. NEUTRAL/HELPFUL never override a
    COSTLY block; HELPFUL is reported only when no window cost money."""
    ok = [b for b in blocks if b["state"] == "OK"]
    if not ok:
        # Sample-size-honest fallback — all windows still developing.
        return "INSUFFICIENT", None
    costly = [b for b in ok if b["verdict"] == "COSTLY_CASH"]
    if costly:
        worst = max(costly, key=lambda b: b["cash_drag_usd"])
        return "COSTLY_CASH", worst
    helpful = [b for b in ok if b["verdict"] == "HELPFUL_CASH"]
    if helpful:
        best = min(helpful, key=lambda b: b["cash_drag_usd"])
        return "HELPFUL_CASH", best
    return "NEUTRAL", ok[0]


def build_cash_drag(equity_curve: list[dict],
                    windows_h: tuple[float, ...] = _DEFAULT_WINDOWS_H,
                    now: datetime | None = None) -> dict:
    """Build the cash-drag snapshot.

    Args:
        equity_curve: chronological (ascending) list of
            ``{timestamp, total_value, cash, sp500_price}`` — exactly
            ``store.equity_curve(...)``'s shape and order.
        windows_h: rolling window lengths in hours.
        now: snapshot anchor (defaults to wall-clock UTC). Tests pass a
            fixed value to lock arithmetic.

    Top-level keys:
      ``as_of``: ISO timestamp.
      ``state``: ``NO_DATA`` (no benchmarkable rows) / ``OK``.
      ``verdict``: ``COSTLY_CASH`` / ``HELPFUL_CASH`` / ``NEUTRAL`` /
        ``INSUFFICIENT`` — worst across windows. ``None`` under
        ``NO_DATA``.
      ``headline``: one-line summary for the panel banner.
      ``windows``: per-window blocks (see ``_window_block``).
      ``n_total_points``: count of benchmarkable rows (all-time, before
        per-window filtering — a sanity number for the panel).
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    rows = [r for r in (equity_curve or []) if _benchmarkable(r)]
    rows.sort(key=lambda r: _parse_ts(r["timestamp"]))

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "state": "NO_DATA",
        "verdict": None,
        "headline": "No benchmarkable equity history yet — cash drag unmeasurable.",
        "windows": [],
        "n_total_points": 0,
    }
    if not rows:
        return base

    base["n_total_points"] = len(rows)
    blocks = [_window_block(w, rows, now) for w in windows_h]
    base["windows"] = blocks
    base["state"] = "OK"
    verdict, worst = _worst_verdict(blocks)
    base["verdict"] = verdict
    if verdict == "COSTLY_CASH" and worst is not None:
        base["headline"] = (
            f"COSTLY_CASH — worst window: {worst['headline']}"
        )
    elif verdict == "HELPFUL_CASH" and worst is not None:
        base["headline"] = (
            f"HELPFUL_CASH — best window: {worst['headline']}"
        )
    elif verdict == "NEUTRAL" and worst is not None:
        base["headline"] = (
            f"NEUTRAL — cash drag inside the ±${_NEUTRAL_BAND_USD:.2f} band "
            f"across all benchmarkable windows."
        )
    else:
        base["headline"] = (
            "INSUFFICIENT — no window has enough benchmarkable history yet."
        )
    return base


if __name__ == "__main__":
    import json
    import sys
    from .. import store as _store
    try:
        s = _store.get_store()
        rows = s.equity_curve(limit=10_000)
    except Exception as e:
        print(f"failed to read equity_curve: {e}", file=sys.stderr)
        rows = []
    print(json.dumps(build_cash_drag(rows), indent=2))
