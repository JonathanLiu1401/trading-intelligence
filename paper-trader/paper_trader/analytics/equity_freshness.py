"""Equity-curve freshness — *is the recorded P&L point the operator's
headline KPIs are computed from actually current, or stale under load?*

``equity_integrity`` audits the recorded ``equity_curve`` for corruption
*within* its own points (negative cash, non-positive equity, no-trade jumps).
``mark_integrity`` audits whether the open *positions* are stale right now.
Neither answers the orthogonal, repeatedly-observed-live question this builder
exists for: **does the latest recorded ``equity_curve`` point still agree with
the live ``portfolio`` table, or has the curve frozen behind a fresher book?**

Why this is physically reachable (and not a bug to "fix" — AGENTS.md memory
``pt-portfolio-equity-divergence``):

* ``strategy._portfolio_snapshot`` writes the ``portfolio`` table at the
  **top** of every ``decide()`` cycle (it re-marks every open position to
  market first — ``strategy.py`` ~line 1216).
* ``store.record_equity_point`` is only ever called at the **end** of that
  same ``decide()`` (both the executed and the NO_DECISION branches —
  ``strategy.py`` ~lines 1597 / 1621). The dashboard never writes either
  table (verified: zero ``update_portfolio`` / ``record_equity_point``
  callers in ``dashboard.py``).

So between a cycle's top-of-cycle ``portfolio`` write and its end-of-cycle
``equity_curve`` write there is a window the full Claude budget wide
(``DECISION_TIMEOUT_S`` 180 + retry 45 + fallback 60 ≈ 5 min) **plus** the
inter-cycle sleep (``OPEN_INTERVAL_S`` 1800s / ``CLOSED_INTERVAL_S`` 3600s).
Under a host-saturation NO_DECISION storm (the recurring live pathology) the
``portfolio`` table re-marks every cycle while the latest ``equity_curve``
point lags one whole cycle behind.

The damage is silent and trader-critical: **every headline P&L surface the
operator reads** — ``/api/benchmark`` and the hourly ``_benchmark_line``,
``/api/drawdown``, the ``/api/analytics`` Sharpe, ``build_equity_integrity``
itself — is computed off ``equity_curve``, so a stale latest point makes the
benchmark/alpha headline understate (or overstate) the true account by the
divergence. Observed live 2026-05-18: ``/api/portfolio`` total ``$924.13``
while ``/api/benchmark`` reported ``$928.92`` ("lagging 6.86pp") off the
frozen curve — the operator's primary KPI lying by ~$4.79 with nothing in
Discord saying so. ``equity_integrity`` reads ``CLEAN`` (the divergence is
portfolio-vs-curve, not *within* recorded points, and the eventual
reconciliation step is under its 8% no-trade-jump gate) so it correctly does
NOT cover this — this builder is the missing dimension.

Advisory only — it reports, never gates Opus, adds no caps (AGENTS.md
invariants #2/#12; the ``equity_integrity`` / ``mark_integrity``
observational precedent). It deliberately does **not** recompute a
"corrected" benchmark/alpha — it reports the raw divergence the same way
``equity_integrity`` reports CORRUPT without repairing the row. Pure &
network-free: the caller does the two store reads and passes the dicts (the
``drawdown.py`` / ``equity_integrity.py`` "network in the endpoint, builder
takes the dicts" split), so the core is offline & deterministically testable.
Never raises — a malformed row degrades; the contract is "no freshness
verdict this cycle", never an exception.

Verdict ladder (severity-ordered; both staleness AND value-divergence are
required for the actionable verdict so normal mid-cycle drift never spams):

  * ``NO_DATA``     — no usable live total or no usable equity point.
  * ``FRESH``       — the curve is recent **or** the divergence is within the
    band: nothing actionable (the curve will reconcile next cycle; the
    headline KPIs are trustworthy).
  * ``STALE_CURVE`` — the latest equity point is older than the cadence-aware
    stale threshold but the live book has barely moved
    (``|Δ| <= divergence_pct``): the curve is lagging but the headline P&L is
    still ~right (a milder operability note).
  * ``DIVERGED``    — the latest equity point is BOTH stale AND materially
    off the live book (``|Δ| > divergence_pct``): every equity-curve-derived
    headline is misstating the true account by the divergence. Dominates
    ``STALE_CURVE`` (the actionable one).
"""
from __future__ import annotations

from datetime import datetime, timezone

# A live-vs-recorded gap this large means a headline P&L surface
# (benchmark/alpha/drawdown/Sharpe) materially misstates the account. 0.5% of
# a ~$1000 book ≈ $5 — the exact order of the observed-live $4.79 lie.
DEFAULT_DIVERGENCE_PCT = 0.5
# Cadence-aware staleness. A healthy cycle writes the equity point within the
# full Claude budget (~285s) of the top-of-cycle portfolio write, then sleeps
# OPEN_INTERVAL_S (1800s) / CLOSED_INTERVAL_S (3600s). 2× the cadence means at
# least one whole cycle's equity write was skipped/delayed — genuinely stale,
# not the by-construction ~1-cycle lag every healthy book always carries.
DEFAULT_OPEN_STALE_AGE_S = 3600.0      # 2 × OPEN_INTERVAL_S
DEFAULT_CLOSED_STALE_AGE_S = 7200.0    # 2 × CLOSED_INTERVAL_S


def _f(x) -> float | None:
    """Best-effort float; None on garbage (never raises)."""
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_ts(s) -> datetime | None:
    """Parse a store ISO timestamp to an aware UTC datetime; None on garbage.

    The store always writes ``datetime.now(timezone.utc).isoformat()`` so a
    parse failure means genuinely corrupt data — we degrade, never raise."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _latest_usable_point(equity_points) -> dict | None:
    """Newest point (lexical TS order — the codebase-wide fixed-offset-UTC
    ordering invariant) carrying a parseable, positive ``total_value``.

    Defensive about input ordering exactly like ``equity_integrity.
    _sorted_points``: ``store.equity_curve()`` is already ascending but a pure
    builder must not assume it. A non-positive curve value is corruption
    ``equity_integrity`` owns — skip it here so we anchor to the last *trust-
    able* recorded point, not a poisoned one."""
    best: dict | None = None
    best_ts = ""
    if not isinstance(equity_points, (list, tuple)):
        return None  # never raises — a non-iterable degrades to "no point"
    for p in equity_points:
        if not isinstance(p, dict):
            continue
        tv = _f(p.get("total_value"))
        ts = p.get("timestamp")
        if tv is None or tv <= 0.0 or not ts:
            continue
        sts = str(ts)
        if best is None or sts >= best_ts:
            best, best_ts = {"timestamp": sts, "total_value": tv}, sts
    return best


def build_equity_freshness(
    portfolio: dict | None,
    equity_points: list[dict] | None,
    market_open: bool,
    *,
    now: datetime | None = None,
    divergence_pct: float = DEFAULT_DIVERGENCE_PCT,
    open_stale_age_s: float = DEFAULT_OPEN_STALE_AGE_S,
    closed_stale_age_s: float = DEFAULT_CLOSED_STALE_AGE_S,
) -> dict:
    """Live ``portfolio`` total vs the latest recorded ``equity_curve`` point.

    Args:
        portfolio: ``store.get_portfolio()`` →
            ``{cash, total_value, positions, last_updated}``.
        equity_points: ``store.equity_curve(...)`` rows
            (``{timestamp,total_value,cash,sp500_price}``); any order.
        market_open: ``market.is_market_open()`` — selects the cadence-aware
            stale threshold (the ``build_runner_heartbeat`` market-open-param
            precedent: bool in, builder owns the thresholding).
        now: injected wall clock for deterministic tests (UTC).
        divergence_pct: ``|live-curve|/live`` over this % is "materially off".
        open_stale_age_s / closed_stale_age_s: latest equity point older than
            this (market-open-selected) is "stale".

    Returns a JSON-able dict; never raises.
    """
    now = now or datetime.now(timezone.utc)
    stale_age_s = (open_stale_age_s if market_open else closed_stale_age_s)

    pf = portfolio if isinstance(portfolio, dict) else {}
    live_value = _f(pf.get("total_value"))
    live_ts = pf.get("last_updated")
    latest = _latest_usable_point(equity_points)

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "market_open": bool(market_open),
        "divergence_pct": divergence_pct,
        "stale_age_s": stale_age_s,
        "live_value": round(live_value, 2) if live_value is not None else None,
        "live_ts": live_ts,
        "curve_value": (round(latest["total_value"], 2)
                        if latest else None),
        "curve_ts": latest["timestamp"] if latest else None,
        "curve_age_s": None,
        "delta_usd": None,
        "delta_pct": None,
    }

    if live_value is None or live_value <= 0.0 or latest is None:
        return {
            **base,
            "verdict": "NO_DATA",
            "headline": (
                "No usable live portfolio total and/or recorded equity "
                "point yet — nothing to reconcile."),
        }

    curve_value = latest["total_value"]
    delta_usd = live_value - curve_value
    delta_pct = (delta_usd / live_value * 100.0) if live_value else 0.0

    # Curve age. A future curve_ts means the wall clock stepped backward (the
    # documented NTP/VM clock-skew hazard — same clamp as strategy.
    # _hold_age_str / runner._restore_runner_state): clamp to 0 (not stale)
    # rather than render a negative age and falsely read "fresh forever".
    curve_dt = _parse_ts(latest["timestamp"])
    curve_age_s: float | None
    if curve_dt is None:
        curve_age_s = None  # age unknowable → cannot assert staleness
    else:
        curve_age_s = max(0.0, (now - curve_dt).total_seconds())

    base.update({
        "curve_age_s": (round(curve_age_s, 1)
                        if curve_age_s is not None else None),
        "delta_usd": round(delta_usd, 2),
        "delta_pct": round(delta_pct, 4),
    })

    stale = curve_age_s is not None and curve_age_s > stale_age_s
    diverged_val = abs(delta_pct) > divergence_pct

    age_str = (f"{curve_age_s / 60.0:.0f}m"
               if curve_age_s is not None else "age-unknown")
    money = (
        f"live ${live_value:.2f} (portfolio table) vs recorded equity point "
        f"${curve_value:.2f} ({age_str} old); "
        f"Δ ${delta_usd:+.2f} ({delta_pct:+.2f}%)")

    if stale and diverged_val:
        verdict = "DIVERGED"
        headline = (
            f"Recorded equity point is STALE *and* materially off the live "
            f"book — {money}. Every equity-curve-derived headline "
            f"(/api/benchmark + the hourly benchmark/alpha line, "
            f"/api/drawdown, /api/analytics Sharpe, the hourly P/L) is "
            f"computed off this stale point and misstates the true account "
            f"by ${abs(delta_usd):.2f}; trust /api/portfolio, not the "
            f"benchmark headline, until the next equity write reconciles "
            f"it (typically the NEXT completed decide() cycle).")
    elif stale:
        verdict = "STALE_CURVE"
        headline = (
            f"Recorded equity point is stale ({age_str} old, "
            f">{stale_age_s / 60.0:.0f}m) but the live book has barely moved "
            f"— {money}. Headline P&L is still ~right; the curve is lagging "
            f"the loop (a NO_DECISION storm / slow cycle) and will reconcile "
            f"on the next completed decide() cycle.")
    else:
        verdict = "FRESH"
        if curve_age_s is None:
            headline = (
                f"Recorded equity point timestamp is unparseable so its age "
                f"cannot be checked, but the live book agrees with it — "
                f"{money}. Headline P&L is trustworthy.")
        else:
            headline = (
                f"Recorded equity point is current and agrees with the live "
                f"book — {money}. Headline P&L (benchmark, drawdown, Sharpe) "
                f"is trustworthy.")

    return {**base, "verdict": verdict, "headline": headline}


if __name__ == "__main__":  # one-screen answer, usable when :8090 is wedged
    import json
    import sqlite3
    import sys
    from pathlib import Path

    db = Path(__file__).resolve().parents[2] / "data" / "paper_trader.db"
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        prow = c.execute(
            "SELECT cash,total_value,last_updated FROM portfolio WHERE id=1"
        ).fetchone()
        pf = ({"cash": prow[0], "total_value": prow[1],
               "last_updated": prow[2]} if prow else {})
        eq = [
            {"timestamp": r[0], "total_value": r[1], "cash": r[2]}
            for r in c.execute(
                "SELECT timestamp,total_value,cash FROM equity_curve "
                "ORDER BY timestamp ASC, id ASC").fetchall()
        ]
        c.close()
    except Exception as e:  # the benchmark / signals --check-freshness CLI precedent
        print(f"equity_freshness: cannot read {db}: {e}")
        sys.exit(2)

    # No market module dependency in the CLI path — assume open (the stricter,
    # shorter stale threshold) so the CLI never under-reports staleness.
    rep = build_equity_freshness(pf, eq, True)
    if "--json" in sys.argv:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(f"EQUITY FRESHNESS  [{rep['verdict']}]  {rep['headline']}")
    sys.exit(2 if rep["verdict"] == "DIVERGED" else 0)
