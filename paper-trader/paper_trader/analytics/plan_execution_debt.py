"""Plan-vs-action execution debt.

``/api/deployment-plan`` emits the scorer + half-Kelly + caps composition
of "what to BUY next with available cash" — the planner's recommended
allocation list. But there is no read-side endpoint that answers the
*follow-up* an operator asks every cycle:

  "The plan says deploy $589 across MUU + KLAC. Did the live trader
  actually execute that, or has the recommendation been sitting
  unfilled for hours? How much of the plan is unexecuted right now?"

Live evidence (2026-05-28): the deployment plan ranked MUU + KLAC for
$589 of available $660 cash, but the running trader has been HOLDing
AMD for the last 12+ market-open cycles — the plan exists on the
dashboard, the runner doesn't act on it. The cumulative *execution
debt* (recommended-but-unfilled $) is invisible to every adjacent
endpoint:

  * ``/api/deployment-plan``       — produces the plan, doesn't track action
  * ``/api/deployment-plan-conflicts`` — audits the plan for internal
    inverse-pair / directional-hedge pathology, not for execution
  * ``/api/funded-suggestions``    — funding feasibility of individual
    suggestions, not the plan
  * ``/api/intent-followthrough``  — measures whether *Opus's own*
    next-cycle declarations were honored, not whether the
    quant-driven deployment plan was honored
  * ``/api/cash-redeployment-latency-skill`` — time from SELL to next
    BUY, not from plan-recommended to filled

This module is the missing measurement. ``build_plan_execution_debt``
takes (a) the live deployment plan and (b) the recent BUY trades from
the live store, walks each plan ticker, and reports:

  * per-ticker ``executed_usd`` — sum of BUY trade ``value`` rows
    within the window that match the plan ticker
  * per-ticker ``status`` — UNEXECUTED / PARTIAL / EXECUTED
  * cumulative ``unexecuted_usd`` and ``executed_pct`` of plan
  * verdict ladder ALIGNED / TIGHTENING / DRIFTING / DISCONNECTED /
    NO_PLAN / ERROR

Pure & offline. No DB, no clock, no network — the caller hands in the
plan rows, the trade rows, and an optional ``now`` for the time window.
Garbage-safe: non-list inputs, missing fields, malformed numerics all
degrade to NO_PLAN / empty buckets, never raise.

Advisory only — never modifies anything, never gates Opus, no caps.
The operator decides whether the unexecuted recommendation is correct
restraint or a missed action (AGENTS.md invariants #2 / #12).

Verdict ladder:

| Verdict        | Trigger                                                                      |
|----------------|------------------------------------------------------------------------------|
| ``NO_PLAN``    | empty plan, all-zero plan, or plan_total_usd ≤ 0                             |
| ``ALIGNED``    | ≥ ALIGNED_FLOOR_PCT (80%) of plan deployed in the window                     |
| ``TIGHTENING`` | TIGHTENING_FLOOR_PCT (50%) ≤ executed < ALIGNED_FLOOR_PCT                    |
| ``DRIFTING``   | DRIFTING_FLOOR_PCT (20%) ≤ executed < TIGHTENING_FLOOR_PCT                   |
| ``DISCONNECTED`` | executed < DRIFTING_FLOOR_PCT                                              |
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


# Operator-decisive thresholds. Tuned so the live state above (0% of a
# $589 plan executed in 24h) lands in ``DISCONNECTED``, while a single
# fill of MUU at the recommended $295 lands at exactly 50% =
# ``TIGHTENING`` boundary. The floors are inclusive in the
# "above-or-equal" direction (≥ x → that bucket), so a plan that's
# exactly 80% filled is ``ALIGNED``, not ``TIGHTENING``.
ALIGNED_FLOOR_PCT = 80.0
TIGHTENING_FLOOR_PCT = 50.0
DRIFTING_FLOOR_PCT = 20.0

# Status thresholds for the per-ticker bucket (different scale —
# fraction of *this ticker's* plan that's been filled).
EXECUTED_FLOOR_PCT = 90.0   # ≥ 90% of plan_alloc on this ticker → EXECUTED
PARTIAL_FLOOR_PCT = 10.0    # ≥ 10% but < 90% → PARTIAL
                            # < 10% → UNEXECUTED

DEFAULT_WINDOW_HOURS = 24.0


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _z(v: Any, ndigits: int = 2):
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _parse_ts(s: Any) -> datetime | None:
    """Parse the store's ISO-with-tz timestamp into a UTC datetime.
    Returns None on garbage so the caller can skip the row."""
    if not isinstance(s, str) or not s:
        return None
    try:
        # store.recent_trades emits e.g. "2026-05-28T18:44:45.836442+00:00";
        # datetime.fromisoformat handles the offset correctly on 3.11+.
        dt = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ticker_status(rec_usd: float, executed_usd: float) -> str:
    """Per-ticker bucket — UNEXECUTED / PARTIAL / EXECUTED. Same
    rec_usd ≤ 0 guard as the rollup so a zero-allocation plan row
    that the planner emits (filtered below ``min_alloc_usd``) cannot
    surface as misleading EXECUTED."""
    if rec_usd <= 0:
        return "UNEXECUTED"
    pct = 100.0 * executed_usd / rec_usd
    if pct >= EXECUTED_FLOOR_PCT:
        return "EXECUTED"
    if pct >= PARTIAL_FLOOR_PCT:
        return "PARTIAL"
    return "UNEXECUTED"


def _rollup_verdict(execution_pct: float) -> str:
    if execution_pct >= ALIGNED_FLOOR_PCT:
        return "ALIGNED"
    if execution_pct >= TIGHTENING_FLOOR_PCT:
        return "TIGHTENING"
    if execution_pct >= DRIFTING_FLOOR_PCT:
        return "DRIFTING"
    return "DISCONNECTED"


def build_plan_execution_debt(
    plan: Any,
    trades: Any,
    window_hours: float | None = None,
    now: datetime | None = None,
) -> dict:
    """Pure builder. Inputs:

      * ``plan`` — the ``plan`` row list as emitted by
        ``build_deployment_plan`` (or the ``plan`` field on
        ``/api/deployment-plan``). Each row carries ``ticker`` +
        ``alloc_usd``; optional ``scorer_verdict`` / ``sector`` /
        ``is_leveraged`` are echoed back for the operator.
      * ``trades`` — the trade rows from ``store.recent_trades(N)``
        (or the ``all_trades`` field on ``/api/state``). Each row
        carries ``ticker`` / ``action`` / ``value`` / ``timestamp``.
        Only ``action == 'BUY'`` rows within the window contribute.
      * ``window_hours`` — how far back to count BUY trades; default
        24h, clamped to [0.5, 720].
      * ``now`` — reference UTC clock; default ``datetime.now(UTC)``.
        Tests inject a fixed clock; production never sets this.

    Returns the verdict ladder above. Never raises."""

    wh = _f(window_hours, DEFAULT_WINDOW_HOURS) if window_hours is not None else DEFAULT_WINDOW_HOURS
    # Clamp — a sub-30min window doesn't see typical cycle cadence, a
    # >30d window starts to bleed previous plan epochs into this one.
    wh = max(0.5, min(720.0, wh))

    ref = now if isinstance(now, datetime) else datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    cutoff = ref - timedelta(hours=wh)

    plan_rows = plan if isinstance(plan, list) else []
    trade_rows = trades if isinstance(trades, list) else []

    # Index plan by ticker. A plan with the same ticker twice is
    # malformed (the planner emits unique tickers by construction),
    # so we coalesce by summing alloc_usd — same defensive shape as
    # ``backtest_trade_delta._norm`` for duplicates.
    plan_index: dict[str, dict] = {}
    plan_total_usd = 0.0
    for row in plan_rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        rec_usd = _f(row.get("alloc_usd"), 0.0)
        if rec_usd <= 0:
            continue
        prev = plan_index.get(ticker)
        if prev is None:
            plan_index[ticker] = {
                "ticker": ticker,
                "rec_alloc_usd": rec_usd,
                "rec_alloc_pct_of_book": _f(row.get("alloc_pct_of_book"), 0.0),
                "scorer_verdict": row.get("scorer_verdict") or None,
                "sector": row.get("sector") or None,
                "is_leveraged": bool(row.get("is_leveraged")),
                "pred_5d_return_pct": _f(row.get("pred_5d_return_pct"), 0.0),
            }
        else:
            prev["rec_alloc_usd"] += rec_usd
            prev["rec_alloc_pct_of_book"] += _f(row.get("alloc_pct_of_book"), 0.0)
        plan_total_usd += rec_usd

    if plan_total_usd <= 0 or not plan_index:
        return {
            "verdict": "NO_PLAN",
            "headline": "no plan rows — nothing to track for execution",
            "as_of": ref.isoformat(),
            "window_hours": _z(wh),
            "n_plan_rows": 0,
            "plan_total_usd": _z(0.0),
            "executed_total_usd": _z(0.0),
            "execution_pct": _z(0.0),
            "unexecuted_usd": _z(0.0),
            "partial_gap_usd": _z(0.0),
            "n_executed": 0,
            "n_partial": 0,
            "n_unexecuted": 0,
            "by_ticker": [],
        }

    # Walk BUY trades in window, summing executed $ per plan ticker.
    # Trades for tickers NOT in the plan are ignored — they're a real
    # action the trader took, but they don't reduce *this* plan's
    # execution debt. Operators reading the report can cross-check
    # against /api/today-action-tape for the full action picture.
    executed_by_ticker: dict[str, float] = {}
    for t in trade_rows:
        if not isinstance(t, dict):
            continue
        action = str(t.get("action") or "").upper().strip()
        if action != "BUY":
            continue
        ticker = str(t.get("ticker") or "").upper().strip()
        if not ticker or ticker not in plan_index:
            continue
        ts = _parse_ts(t.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        value = _f(t.get("value"), 0.0)
        if value <= 0:
            continue
        executed_by_ticker[ticker] = executed_by_ticker.get(ticker, 0.0) + value

    # Per-ticker rows, sorted by largest-unexecuted-first so the
    # operator's eye lands on the biggest miss.
    by_ticker: list[dict] = []
    executed_total_usd = 0.0
    n_executed = n_partial = n_unexecuted = 0
    partial_gap_usd = 0.0

    for ticker, p in plan_index.items():
        executed = executed_by_ticker.get(ticker, 0.0)
        rec = p["rec_alloc_usd"]
        status = _ticker_status(rec, executed)
        executed_total_usd += executed
        if status == "EXECUTED":
            n_executed += 1
        elif status == "PARTIAL":
            n_partial += 1
            partial_gap_usd += max(0.0, rec - executed)
        else:
            n_unexecuted += 1

        by_ticker.append({
            "ticker": ticker,
            "rec_alloc_usd": _z(rec),
            "rec_alloc_pct_of_book": _z(p["rec_alloc_pct_of_book"]),
            "executed_usd": _z(executed),
            "executed_pct": _z(100.0 * executed / rec if rec > 0 else 0.0),
            "gap_usd": _z(max(0.0, rec - executed)),
            "status": status,
            "scorer_verdict": p["scorer_verdict"],
            "sector": p["sector"],
            "is_leveraged": p["is_leveraged"],
            "pred_5d_return_pct": _z(p["pred_5d_return_pct"]),
        })

    by_ticker.sort(key=lambda r: (-(r["gap_usd"] or 0.0), r["ticker"]))

    # Executed % is bounded at 100 even if the trader OVER-bought a
    # plan ticker (BUYs > recommended) — the rollup measures fill of
    # the *plan*, not over-execution.
    execution_pct = min(100.0, 100.0 * executed_total_usd / plan_total_usd) if plan_total_usd > 0 else 0.0
    unexecuted_usd = max(0.0, plan_total_usd - executed_total_usd)
    verdict = _rollup_verdict(execution_pct)

    # Headline names the worst-gap ticker so the operator sees the
    # specific recommendation that's been ignored, not just the rollup.
    worst = by_ticker[0] if by_ticker else None
    worst_clause = ""
    if worst and (worst.get("gap_usd") or 0.0) > 0:
        worst_clause = (
            f" Largest miss: {worst['ticker']} (${worst['gap_usd']:.0f} "
            f"of ${worst['rec_alloc_usd']:.0f} {worst['status'].lower()})."
        )
    headline = (
        f"{verdict} — {execution_pct:.0f}% of recommended ${plan_total_usd:.0f} "
        f"deployed in last {wh:.0f}h ({n_executed}/{len(plan_index)} fully filled).{worst_clause}"
    )

    return {
        "verdict": verdict,
        "headline": headline,
        "as_of": ref.isoformat(),
        "window_hours": _z(wh),
        "n_plan_rows": len(plan_index),
        "plan_total_usd": _z(plan_total_usd),
        "executed_total_usd": _z(executed_total_usd),
        "execution_pct": _z(execution_pct),
        "unexecuted_usd": _z(unexecuted_usd),
        "partial_gap_usd": _z(partial_gap_usd),
        "n_executed": n_executed,
        "n_partial": n_partial,
        "n_unexecuted": n_unexecuted,
        "by_ticker": by_ticker,
        "thresholds": {
            "aligned_floor_pct": ALIGNED_FLOOR_PCT,
            "tightening_floor_pct": TIGHTENING_FLOOR_PCT,
            "drifting_floor_pct": DRIFTING_FLOOR_PCT,
            "executed_floor_pct": EXECUTED_FLOOR_PCT,
            "partial_floor_pct": PARTIAL_FLOOR_PCT,
        },
    }
