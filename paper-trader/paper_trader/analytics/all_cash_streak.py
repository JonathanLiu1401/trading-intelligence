"""All-cash streak — how long has the book been entirely on the sidelines?

The existing cash analytics each describe a different shape:

  * ``/api/cash-drag``                  — time-weighted AVERAGE cash inside
    fixed rolling windows (24h / 168h / 720h). A partial-cash period dilutes
    a fully-flat period — they answer "what did cash cost on average".
  * ``/api/cash-conviction-fit``        — per-cycle sizing-vs-signal fit;
    flags over- or under-deployed against the current signal slate.
  * ``/api/cash-redeployment-latency-skill`` — per-SELL: hours until the
    next BUY. A trade-level metric.

None of them surface the **book-level contiguous-streak** the operator
actually wants when the live ``portfolio`` panel shows
``cash=$987.39 / positions=0``:

  **"How long has the book held NOTHING, and what has SPY done while we
  sat in cash?"**

This module walks ``equity_curve`` newest-first to find the streak of
contiguous points where ``cash`` ≈ ``total_value`` (i.e. no position
value), reports the start timestamp, hours-elapsed, and the SPY benchmark
return over that exact streak window. Past completed streaks are also
returned (newest-first) so the operator can see "we have sat flat 3
times in the last 5 days for an aggregate of 22h".

Verdict ladder:

  * ``NO_DATA``               — empty equity_curve.
  * ``INSUFFICIENT_HISTORY``  — < ``_MIN_POINTS`` curve points overall.
  * ``NOT_ALL_CASH``          — newest point still holds positions; surfaces
                                the most recent COMPLETED streak (if any).
  * ``BRIEF_HOLDOUT``         — currently all-cash, < ``_BRIEF_HOURS``.
  * ``EXTENDED_HOLDOUT``      — currently all-cash, < ``_PROLONGED_HOURS``.
  * ``PROLONGED_HOLDOUT``     — currently all-cash, ≥ ``_PROLONGED_HOURS``.

Pure builder, never raises. Observational only — never gates Opus, no
caps, no path to ``_execute()`` (AGENTS.md invariants #2/#12).

Memory note: the live ``equity_curve`` is per-cycle ~5 days deep
(``paper_trader equity_curve shallow/lumpy``), so the ``streaks``
history will be short by construction. The ``INSUFFICIENT_HISTORY``
verdict matches the honesty pattern pioneered by ``benchmark.py`` and
``cash_drag.py``.

Run as a CLI for a one-shot ops view::

    python3 -m paper_trader.analytics.all_cash_streak
"""
from __future__ import annotations

from datetime import datetime, timezone

# A point is "all-cash" when |total_value - cash| ≤ this many dollars.
# Open positions mark-to-market into ``total_value`` but not ``cash``, so
# the gap is the live position value. A few cents of float noise (rounding
# from yfinance fills, option ×100) shouldn't unseat the verdict.
_FLAT_EPSILON_USD = 0.50

# Honesty floor — below this many total points we can't say anything.
_MIN_POINTS = 3

# Verdict thresholds.
_BRIEF_HOURS = 6.0
_PROLONGED_HOURS = 48.0

# Honesty floor for SPY-return computation inside a streak — need at least
# two valid SPY marks within the streak window or the return is meaningless.
_MIN_SPY_POINTS_PER_STREAK = 2

# Cap on completed-streak history surfaced (newest-first) so the response
# stays bounded even on a very long equity_curve.
_MAX_HISTORY = 20


def _parse_ts(ts):
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        s = ts.replace("Z", "+00:00") if isinstance(ts, str) else ""
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _is_all_cash(point: dict) -> bool:
    """A point is all-cash when its position value (``total_value - cash``)
    is below the flat epsilon. ``cash`` and ``total_value`` are floats;
    missing/None values default to 0.0 (a missing total counts as flat
    relative to cash=0, the NO_DATA path catches this case anyway)."""
    total = point.get("total_value")
    cash = point.get("cash")
    if total is None or cash is None:
        return False
    try:
        return abs(float(total) - float(cash)) <= _FLAT_EPSILON_USD
    except (TypeError, ValueError):
        return False


def _spy_return_pct(curve_slice: list[dict]) -> float | None:
    """SPY % return from the first benchmarkable point to the last.

    ``curve_slice`` is a contiguous slice of equity_curve points (oldest
    first). A "benchmarkable" point has a non-null, non-zero
    ``sp500_price``. Returns ``None`` if fewer than
    ``_MIN_SPY_POINTS_PER_STREAK`` benchmarkable points exist or if the
    anchor price is non-positive.
    """
    marks = []
    for p in curve_slice:
        sp = p.get("sp500_price")
        if sp is None:
            continue
        try:
            sp_f = float(sp)
        except (TypeError, ValueError):
            continue
        if sp_f > 0:
            marks.append(sp_f)
    if len(marks) < _MIN_SPY_POINTS_PER_STREAK:
        return None
    first, last = marks[0], marks[-1]
    if first <= 0:
        return None
    return round((last - first) / first * 100.0, 4)


def _streak_record(curve_slice: list[dict]) -> dict:
    """Build a streak record from an oldest→newest contiguous all-cash
    slice. Caller guarantees every row in the slice is all-cash."""
    first = curve_slice[0]
    last = curve_slice[-1]
    start_ts = first.get("timestamp")
    end_ts = last.get("timestamp")
    start_dt = _parse_ts(start_ts)
    end_dt = _parse_ts(end_ts)
    if start_dt is not None and end_dt is not None and end_dt >= start_dt:
        hours = round((end_dt - start_dt).total_seconds() / 3600.0, 2)
    else:
        hours = None
    spy_ret = _spy_return_pct(curve_slice)
    # Use the CASH at the start of the streak as the "$ that have been
    # sitting idle" — alpha cost is computed against that. Both start and
    # end cash should be ~equal anyway (the book is flat), but the start
    # is the canonical anchor for "this is what we sidelined".
    try:
        cash_anchor = float(first.get("cash") or 0.0)
    except (TypeError, ValueError):
        cash_anchor = 0.0
    if spy_ret is None:
        alpha_cost = None
    else:
        alpha_cost = round(cash_anchor * spy_ret / 100.0, 4)
    return {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "hours": hours,
        "n_points": len(curve_slice),
        "cash_usd": round(cash_anchor, 4),
        "spy_return_pct": spy_ret,
        "alpha_cost_usd": alpha_cost,
    }


def build_all_cash_streak(
    equity_curve: list[dict] | None,
    now: datetime | None = None,
) -> dict:
    """Walk ``equity_curve`` to find the current all-cash streak (if any)
    and a history of recently-completed streaks.

    ``equity_curve`` shape: ``store.equity_curve()`` returns OLDEST→NEWEST
    points with ``timestamp``, ``total_value``, ``cash``, ``sp500_price``.

    Result shape (always present, never raises):

      * ``state``                       — ``"NO_DATA"`` if empty; else ``"OK"``.
      * ``n_points``                    — total curve points considered.
      * ``newest_is_all_cash``          — bool — is the very last point flat?
      * ``current_streak``              — None or a streak record (see below)
                                          extended to ``now`` for
                                          ``hours_elapsed_to_now``.
      * ``most_recent_completed_streak``— None or the freshest CLOSED streak
                                          (a streak followed by a non-flat
                                          point). Useful when
                                          ``newest_is_all_cash=False`` —
                                          the operator still wants to know
                                          when they were last flat.
      * ``streaks``                     — newest-first list of past streaks
                                          (capped at ``_MAX_HISTORY``);
                                          excludes the open ``current_streak``.
      * ``total_streaks``               — count of all closed streaks found.
      * ``aggregate_flat_hours``        — sum(hours of all closed streaks).
      * ``aggregate_alpha_cost_usd``    — sum of closed-streak alpha costs
                                          (None values excluded).
      * ``verdict``                     — see ladder above.
      * ``verdict_detail``              — one-line explanation.
      * ``headline``                    — dashboard one-liner.
      * ``thresholds``                  — module constants for UI.

    Streak record shape:

      * ``start_ts`` / ``end_ts``       — ISO strings.
      * ``hours``                       — float, end - start.
      * ``hours_elapsed_to_now``        — float, now - start (only on
                                          ``current_streak``).
      * ``n_points``                    — count of contiguous all-cash points.
      * ``cash_usd``                    — anchor cash at streak start.
      * ``spy_return_pct``              — % SPY change within streak window.
      * ``alpha_cost_usd``              — cash * spy_return_pct / 100 — what
                                          the sidelined $ would have made if
                                          they had tracked SPY over the
                                          streak. Sign convention: POSITIVE
                                          means sidelined cash COST you
                                          money; NEGATIVE means it SAVED
                                          you money.
    """
    now = now or datetime.now(timezone.utc)
    out = {
        "as_of": now.isoformat(timespec="seconds"),
        "state": "NO_DATA",
        "n_points": 0,
        "newest_is_all_cash": False,
        "current_streak": None,
        "most_recent_completed_streak": None,
        "streaks": [],
        "total_streaks": 0,
        "aggregate_flat_hours": 0.0,
        "aggregate_alpha_cost_usd": None,
        "verdict": "NO_DATA",
        "verdict_detail": "no equity curve points on record",
        "headline": "no equity curve points — cannot grade all-cash tenure",
        "thresholds": {
            "flat_epsilon_usd": _FLAT_EPSILON_USD,
            "min_points": _MIN_POINTS,
            "brief_hours": _BRIEF_HOURS,
            "prolonged_hours": _PROLONGED_HOURS,
            "max_history": _MAX_HISTORY,
        },
    }
    rows = list(equity_curve or [])
    if not rows:
        return out

    out["n_points"] = len(rows)
    if len(rows) < _MIN_POINTS:
        out["state"] = "OK"
        out["verdict"] = "INSUFFICIENT_HISTORY"
        out["verdict_detail"] = (
            f"only {len(rows)} equity_curve point(s) on record "
            f"(need ≥ {_MIN_POINTS}) — cannot grade contiguous-streak tenure"
        )
        out["headline"] = (
            f"INSUFFICIENT_HISTORY — only {len(rows)} curve point(s)")
        return out

    # Walk OLDEST → NEWEST, building contiguous all-cash runs. The store
    # returns oldest→newest already (verified in store.py::equity_curve);
    # we don't assume — we trust the input order.
    closed: list[dict] = []
    open_streak: list[dict] = []
    for p in rows:
        if _is_all_cash(p):
            open_streak.append(p)
        else:
            if open_streak:
                closed.append(_streak_record(open_streak))
                open_streak = []

    newest_is_all_cash = _is_all_cash(rows[-1])
    out["newest_is_all_cash"] = newest_is_all_cash

    current_streak = None
    if newest_is_all_cash and open_streak:
        rec = _streak_record(open_streak)
        start_dt = _parse_ts(rec["start_ts"])
        if start_dt is not None and now >= start_dt:
            rec["hours_elapsed_to_now"] = round(
                (now - start_dt).total_seconds() / 3600.0, 2)
        else:
            rec["hours_elapsed_to_now"] = rec.get("hours")
        current_streak = rec
    # If the newest point is non-flat, any trailing ``open_streak`` would
    # be empty already (the else branch closed it). If the trailing point
    # is non-flat the bot has just re-entered a position; the most recent
    # CLOSED streak is the freshest interesting record.

    out["current_streak"] = current_streak
    out["total_streaks"] = len(closed)
    # Most recent CLOSED streak is the LAST element of ``closed`` (we
    # appended oldest→newest as we walked). Newest-first history list is
    # ``reversed(closed)`` truncated.
    if closed:
        out["most_recent_completed_streak"] = closed[-1]
    history = list(reversed(closed))[:_MAX_HISTORY]
    out["streaks"] = history

    agg_hours = sum(
        (s.get("hours") or 0.0) for s in closed if s.get("hours") is not None)
    out["aggregate_flat_hours"] = round(agg_hours, 2)
    alpha_pieces = [
        s["alpha_cost_usd"] for s in closed
        if s.get("alpha_cost_usd") is not None
    ]
    out["aggregate_alpha_cost_usd"] = (
        round(sum(alpha_pieces), 4) if alpha_pieces else None)

    # ── verdict ──────────────────────────────────────────────────
    if current_streak is not None:
        elapsed = current_streak.get("hours_elapsed_to_now") or 0.0
        cash = current_streak.get("cash_usd") or 0.0
        spy = current_streak.get("spy_return_pct")
        alpha = current_streak.get("alpha_cost_usd")
        if elapsed < _BRIEF_HOURS:
            verdict = "BRIEF_HOLDOUT"
        elif elapsed < _PROLONGED_HOURS:
            verdict = "EXTENDED_HOLDOUT"
        else:
            verdict = "PROLONGED_HOLDOUT"
        if spy is None:
            cost_clause = "SPY change unavailable for this window"
        elif alpha is None:
            cost_clause = f"SPY {spy:+.2f}%"
        elif alpha > 0.0:
            cost_clause = f"SPY {spy:+.2f}% → cash cost ${alpha:.2f}"
        elif alpha < 0.0:
            cost_clause = f"SPY {spy:+.2f}% → cash saved ${-alpha:.2f}"
        else:
            cost_clause = f"SPY {spy:+.2f}% → no alpha cost"
        verdict_detail = (
            f"book has held nothing for {elapsed:.1f}h on ${cash:.2f} cash; "
            f"{cost_clause}"
        )
        headline = (
            f"all-cash {elapsed:.1f}h on ${cash:.2f}; {cost_clause} — {verdict}"
        )
    else:
        verdict = "NOT_ALL_CASH"
        last = out["most_recent_completed_streak"]
        if last is not None:
            last_hours = last.get("hours") or 0.0
            last_end = last.get("end_ts") or "?"
            verdict_detail = (
                f"book currently holds positions; most recent flat streak "
                f"lasted {last_hours:.1f}h, ended {last_end}"
            )
            headline = (
                f"NOT_ALL_CASH — last flat streak {last_hours:.1f}h "
                f"(ended {last_end})"
            )
        else:
            verdict_detail = (
                "book currently holds positions; no all-cash streak in "
                "visible history"
            )
            headline = (
                "NOT_ALL_CASH — no all-cash streak in visible history"
            )

    out["state"] = "OK"
    out["verdict"] = verdict
    out["verdict_detail"] = verdict_detail
    out["headline"] = headline
    return out


def _main() -> int:
    """One-shot CLI — read live ``equity_curve`` and print JSON."""
    import json
    from ..store import get_store
    out = build_all_cash_streak(get_store().equity_curve(limit=5000))
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
