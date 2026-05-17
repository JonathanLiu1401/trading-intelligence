"""Decision-drought drift — what the live trader's *inaction* actually cost.

``analytics/decision_health.py`` reports the NO_DECISION rate and cadence
(hours since the last fill). ``analytics/decision_forensics.py`` says *why* a
cycle produced no decision. Neither answers the trader's real question:

  **"While the bot wasn't trading, did the portfolio drift vs. the market?"**

A *drought* is a maximal run of consecutive non-FILLED cycles between two
FILLED trades (or from the first cycle / to the most recent cycle). For each
drought this module reports duration, the NO_DECISION vs HOLD vs BLOCKED mix,
the portfolio %-change over the window, the S&P %-change over the same window,
and the resulting alpha.

The honest distinction this draws — and the reason it isn't just a restatement
of the NO_DECISION rate — is **PARALYSIS vs DELIBERATE_HOLD**. A drought that is
mostly HOLD decisions is the bot *choosing* to stand pat; its drift is a
strategy outcome. A drought that is mostly NO_DECISION is the bot *unable* to
decide (parse failures / timeouts); its drift is involuntary. Summing the
negative alpha of the PARALYSIS droughts gives "involuntary alpha bleed" — the
cost of the parse-failure problem in the only unit that matters.

``build_decision_drought`` is pure: feed it ``store.recent_decisions(limit)``
(newest-first) and ``store.equity_curve(limit)`` (ascending) and it returns a
JSON-ready dict. No DB or network access. ``now`` is injectable for tests.
"""
from __future__ import annotations

from datetime import datetime, timezone

# A run is "PARALYSIS" when NO_DECISION dominates this fraction of its cycles.
_PARALYSIS_FRACTION = 0.5
# Droughts shorter than this (cycles) are noise — every fill is trivially
# bracketed by a 1-cycle "gap". We still count them in aggregates but don't
# surface them as individual rows.
_MIN_REPORTABLE_CYCLES = 2


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _classify(action_taken: str | None) -> str:
    """Bucket a decisions.action_taken string.

    strategy.py records: ``"BUY NVDA → FILLED"`` / ``"SELL MU → BLOCKED"`` /
    ``"HOLD NVDA → HOLD"`` / bare ``"NO_DECISION"`` (parse failure)."""
    raw = (action_taken or "").strip().upper()
    if not raw or raw == "NO_DECISION":
        return "NO_DECISION"
    if "FILLED" in raw:
        return "FILLED"
    if "BLOCKED" in raw:
        return "BLOCKED"
    if "HOLD" in raw:
        return "HOLD"
    return "OTHER"


def _equity_lookup(equity: list[dict]) -> list[tuple[datetime, float, float | None]]:
    """Sorted (dt, total_value, sp500_price) from the equity curve."""
    pts: list[tuple[datetime, float, float | None]] = []
    for e in equity or []:
        dt = _parse_ts(e.get("timestamp"))
        if dt is None:
            continue
        try:
            tv = float(e.get("total_value"))
        except (TypeError, ValueError):
            continue
        sp = e.get("sp500_price")
        try:
            sp = float(sp) if sp is not None else None
        except (TypeError, ValueError):
            sp = None
        pts.append((dt, tv, sp))
    pts.sort(key=lambda r: r[0])
    return pts


def _val_at(pts: list[tuple[datetime, float, float | None]],
            when: datetime) -> tuple[float | None, float | None]:
    """Equity (total_value, sp500) at the last point at/before ``when``.

    Falls back to the earliest point if ``when`` precedes the whole curve, so a
    drought that opens before the first equity sample still gets a baseline
    instead of silently dropping its alpha."""
    if not pts:
        return None, None
    chosen = None
    for dt, tv, sp in pts:
        if dt <= when:
            chosen = (tv, sp)
        else:
            break
    if chosen is None:
        chosen = (pts[0][1], pts[0][2])
    return chosen


def _pct(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a == 0:
        return None
    return round((b - a) / a * 100.0, 3)


def build_decision_drought(decisions: list[dict],
                           equity_curve: list[dict],
                           now: datetime | None = None) -> dict:
    """Segment cycles into droughts between FILLED trades; price the drift.

    Pure. ``decisions`` is newest-first (as ``store.recent_decisions``
    returns); ``equity_curve`` is ascending (as ``store.equity_curve``
    returns)."""
    now = now or datetime.now(timezone.utc)
    out: dict = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_cycles": 0,
        "n_fills": 0,
        "n_droughts": 0,
        "verdict": "NO_DATA",
        "verdict_reason": "no decisions recorded yet",
        "current_drought": None,
        "longest_drought": None,
        "worst_alpha_drought": None,
        "involuntary_alpha_bleed_pct": 0.0,
        "n_paralysis_droughts": 0,
        "droughts": [],
    }
    if not decisions:
        return out

    chrono = [d for d in reversed(decisions)]  # ascending by (ts, id)
    pts = _equity_lookup(equity_curve)

    out["n_cycles"] = len(chrono)
    out["n_fills"] = sum(1 for d in chrono if _classify(d.get("action_taken")) == "FILLED")

    # Walk cycles, accumulating a run of non-FILLED decisions. A FILLED cycle
    # closes the current run; the trailing run (no fill after it) is the
    # "current" / ongoing drought.
    droughts: list[dict] = []
    run: list[dict] = []

    def _close_run(run_cycles: list[dict], ongoing: bool) -> None:
        if not run_cycles:
            return
        start_ts = _parse_ts(run_cycles[0].get("timestamp"))
        end_ts = _parse_ts(run_cycles[-1].get("timestamp"))
        cats = [_classify(c.get("action_taken")) for c in run_cycles]
        n = len(run_cycles)
        n_nd = cats.count("NO_DECISION")
        n_hold = cats.count("HOLD")
        n_blocked = cats.count("BLOCKED")
        nd_frac = n_nd / n if n else 0.0
        if nd_frac >= _PARALYSIS_FRACTION:
            kind = "PARALYSIS"
        elif n_hold > n_nd:
            kind = "DELIBERATE_HOLD"
        else:
            kind = "MIXED"
        dur_h = None
        if start_ts and end_ts:
            dur_h = round((end_ts - start_ts).total_seconds() / 3600.0, 2)
        port_a = port_b = spy_a = spy_b = None
        if start_ts:
            port_a, spy_a = _val_at(pts, start_ts)
        if end_ts:
            port_b, spy_b = _val_at(pts, end_ts)
        port_pct = _pct(port_a, port_b)
        spy_pct = _pct(spy_a, spy_b)
        alpha = (round(port_pct - spy_pct, 3)
                 if port_pct is not None and spy_pct is not None else None)
        droughts.append({
            "start": start_ts.isoformat(timespec="seconds") if start_ts else None,
            "end": end_ts.isoformat(timespec="seconds") if end_ts else None,
            "duration_hours": dur_h,
            "n_cycles": n,
            "n_no_decision": n_nd,
            "n_hold": n_hold,
            "n_blocked": n_blocked,
            "no_decision_pct": round(nd_frac * 100.0, 1),
            "kind": kind,
            "ongoing": ongoing,
            "portfolio_pct": port_pct,
            "spy_pct": spy_pct,
            "alpha_pct": alpha,
        })

    for d in chrono:
        if _classify(d.get("action_taken")) == "FILLED":
            _close_run(run, ongoing=False)
            run = []
        else:
            run.append(d)
    # Trailing run is the live/ongoing drought (nothing filled after it).
    _close_run(run, ongoing=True)

    reportable = [d for d in droughts if d["n_cycles"] >= _MIN_REPORTABLE_CYCLES]
    out["n_droughts"] = len(reportable)
    out["droughts"] = sorted(
        reportable, key=lambda r: (r["start"] or ""), reverse=True
    )[:30]

    if droughts and droughts[-1]["ongoing"] and \
            droughts[-1]["n_cycles"] >= _MIN_REPORTABLE_CYCLES:
        out["current_drought"] = droughts[-1]

    if reportable:
        out["longest_drought"] = max(
            reportable, key=lambda r: (r["duration_hours"] or 0.0, r["n_cycles"])
        )
        with_alpha = [r for r in reportable if r["alpha_pct"] is not None]
        if with_alpha:
            out["worst_alpha_drought"] = min(with_alpha, key=lambda r: r["alpha_pct"])

    paralysis = [r for r in reportable if r["kind"] == "PARALYSIS"]
    out["n_paralysis_droughts"] = len(paralysis)
    bleed = sum(r["alpha_pct"] for r in paralysis
                if r["alpha_pct"] is not None and r["alpha_pct"] < 0)
    out["involuntary_alpha_bleed_pct"] = round(bleed, 3)

    # Verdict — judged on involuntary bleed first (the actionable signal),
    # then on whether the bot is currently stuck.
    cur = out["current_drought"]
    if out["n_fills"] == 0:
        out["verdict"] = "NEVER_TRADED"
        out["verdict_reason"] = (
            f"{out['n_cycles']} cycles, zero FILLED trades — the bot has "
            "never opened or closed a position")
    elif bleed <= -1.0:
        out["verdict"] = "BLEEDING"
        out["verdict_reason"] = (
            f"{abs(bleed):.2f}% of alpha lost across {len(paralysis)} "
            "involuntary (parse-failure) droughts — the NO_DECISION problem "
            "is costing real performance")
    elif cur and cur["kind"] == "PARALYSIS" and (cur["duration_hours"] or 0) >= 3:
        out["verdict"] = "STUCK"
        out["verdict_reason"] = (
            f"currently paralyzed for {cur['duration_hours']:.1f}h "
            f"({cur['n_no_decision']}/{cur['n_cycles']} cycles NO_DECISION)")
    else:
        out["verdict"] = "OK"
        out["verdict_reason"] = (
            f"{out['n_fills']} fills across {out['n_cycles']} cycles; "
            f"no material involuntary alpha bleed")
    return out


if __name__ == "__main__":  # smoke test against the live DB
    import json
    from paper_trader.store import get_store
    s = get_store()
    rep = build_decision_drought(s.recent_decisions(limit=2000),
                                 s.equity_curve(limit=3000))
    print(json.dumps(rep, indent=2, default=str))
