"""Per-open-position P/L runrate: dollars-per-day-held and the verdict that
falls out of it.

A live trader's #1 question while holding into a thesis is "is this thing
bleeding faster than I'd tolerate, or is it actually working?". The dashboard
shows unrealized P/L and the prompt shows hold age, but the *pace* — P/L
per day held — is left to the trader to compute in their head. This builder
makes the pace explicit: $-5.62 on a 1.2-day hold is $-4.68/day, projecting
to roughly $-140/month if the slope stays flat. That's the headline number
a discretionary desk reads next to absolute P/L.

Pure arithmetic over ``store.open_positions()`` (already carries qty,
avg_cost, current_price, unrealized_pl, opened_at, stale_mark) plus
``portfolio.total_value`` (the equity base for relative framing). NO
network, NO extra store reads — same hot-path discipline as
``risk_mirror`` / ``buying_power``. Single source of truth (AGENTS.md
invariant #10) for the per-position pace math so the endpoint and any
future reporter line never disagree.

Verdict thresholds key off **annualized return %** rather than absolute
dollars, so a $1000 book and a $100000 book read the same scale. The
band edges (BLEEDING < -100%/yr ≈ -0.4%/day; WORKING > 0; FLAT in
between) are deliberately wide — the runrate of a fresh fill is noisy
(<1 day held), so a position has to be visibly bad before we yell.
Stale marks (price unavailable; mark fell back to avg_cost) are tagged
``stale_mark=True`` on the row and contribute ``runrate_per_day_usd=0``
with verdict ``UNKNOWN`` — never a falsely calm "FLAT" verdict next to
a price the trader cannot trust (the ``_pos_pct_weight`` precedent in
``reporter.py``).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Annualized-return thresholds for the verdict (decimal fractions per year).
# Trading days per year ≈ 252; we calendarize on 365 since opened_at to now
# is a wall-clock interval, not a trading-day interval.
_BLEEDING_ANNUAL_PCT = -100.0   # < -100%/yr ⇒ BLEEDING
_WORKING_ANNUAL_PCT = 25.0      # > +25%/yr ⇒ WORKING
# Minimum hold to compute a meaningful runrate. A 5-minute-old fill has
# essentially infinite annualized slope on any price wiggle — surface raw
# P/L but suppress the verdict / projections rather than yelling on noise.
_MIN_HOLD_HOURS_FOR_VERDICT = 1.0
_SECS_PER_DAY = 86400.0


def _parse_iso(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if x != x:  # NaN guard
        return None
    return float(x)


def _verdict(annual_pct: float | None, stale: bool, hours_held: float) -> str:
    if stale:
        return "UNKNOWN"
    if hours_held < _MIN_HOLD_HOURS_FOR_VERDICT:
        return "FRESH"
    if annual_pct is None:
        return "UNKNOWN"
    if annual_pct <= _BLEEDING_ANNUAL_PCT:
        return "BLEEDING"
    if annual_pct >= _WORKING_ANNUAL_PCT:
        return "WORKING"
    return "FLAT"


def build_position_runrate(positions: list[dict],
                            total_value: float | None,
                            now: datetime | None = None) -> dict:
    """Returns a dict with ``state``, ``rows``, and aggregate fields.

    ``state`` ∈ {NO_DATA, OK} — NO_DATA when the book is empty.
    Each row in ``rows`` carries:
      * ticker, type, qty, opened_at
      * hold_seconds, hold_days (>= 0)
      * unrealized_pl (USD)
      * pl_pct (return % from cost, suppressed when stale or avg_cost ≤ 0)
      * runrate_per_day_usd (None when hold too short or stale_mark)
      * annualized_pct (pl_pct * 365/hold_days; None when not derivable)
      * projected_pl_30d_usd (= runrate_per_day_usd * 30; None when not derivable)
      * book_weight_pct (None when total_value missing/0)
      * stale_mark (passthrough from the snapshot row)
      * verdict ∈ {BLEEDING, FLAT, WORKING, UNKNOWN, FRESH}

    Aggregate fields surface the book-wide pace at a glance:
      * total_runrate_per_day_usd — Σ runrate (None when no row has a runrate)
      * total_projected_pl_30d_usd — Σ projected 30d
      * worst_runrate — row with the most-negative runrate (or None)
      * any_bleeding — True iff at least one row reads BLEEDING
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    rows: list[dict] = []
    total_runrate = 0.0
    have_any_runrate = False
    worst_runrate_row: dict | None = None
    any_bleeding = False

    tv = _num(total_value)

    for p in positions or []:
        if not isinstance(p, dict):
            continue
        ticker = (p.get("ticker") or "").upper()
        ptype = (p.get("type") or "").lower()
        qty = _num(p.get("qty"))
        avg = _num(p.get("avg_cost"))
        cur = _num(p.get("current_price"))
        pl = _num(p.get("unrealized_pl"))
        stale = bool(p.get("stale_mark"))
        opened_dt = _parse_iso(p.get("opened_at"))

        hold_seconds = 0
        hold_days = 0.0
        if opened_dt is not None:
            secs = (now - opened_dt).total_seconds()
            if secs < 0:
                secs = 0.0
            hold_seconds = int(secs)
            hold_days = round(secs / _SECS_PER_DAY, 4)
        hours_held = hold_seconds / 3600.0

        is_opt = ptype in ("call", "put")
        mult = 100.0 if is_opt else 1.0

        # P/L % from cost (per-share return). Suppressed when stale or when
        # avg_cost / current_price is unusable. Mirrors _pos_pct_weight.
        pl_pct: float | None = None
        if (not stale and avg is not None and avg > 0
                and cur is not None and cur > 0):
            pl_pct = round((cur - avg) / avg * 100.0, 2)

        # runrate_per_day_usd: dollars of unrealized P/L per day held. Only
        # meaningful once the position has been held long enough that the
        # slope isn't dominated by intraday noise (>= _MIN_HOLD_HOURS_FOR_VERDICT)
        # and the mark is real (not stale).
        runrate_per_day_usd: float | None = None
        annualized_pct: float | None = None
        projected_30d: float | None = None
        if (pl is not None and hold_seconds > 0 and not stale
                and hours_held >= _MIN_HOLD_HOURS_FOR_VERDICT):
            runrate_per_day_usd = round(pl / max(hold_days, 1e-6), 2)
            if pl_pct is not None and hold_days > 0:
                annualized_pct = round(pl_pct * 365.0 / hold_days, 2)
            projected_30d = round(runrate_per_day_usd * 30.0, 2)

        # Book weight % — same arithmetic as _pos_pct_weight (the SSOT
        # comment is load-bearing: any change here must match reporter).
        book_weight_pct: float | None = None
        if (tv is not None and tv > 0 and cur is not None and cur > 0
                and qty is not None):
            mv = cur * qty * mult
            book_weight_pct = round(mv / tv * 100.0, 2)

        verdict = _verdict(annualized_pct, stale, hours_held)

        row = {
            "ticker": ticker,
            "type": ptype,
            "qty": qty,
            "opened_at": p.get("opened_at"),
            "hold_seconds": hold_seconds,
            "hold_days": hold_days,
            "unrealized_pl": round(pl, 2) if pl is not None else None,
            "pl_pct": pl_pct,
            "runrate_per_day_usd": runrate_per_day_usd,
            "annualized_pct": annualized_pct,
            "projected_pl_30d_usd": projected_30d,
            "book_weight_pct": book_weight_pct,
            "stale_mark": stale,
            "verdict": verdict,
        }
        if is_opt:
            row["strike"] = p.get("strike")
            row["expiry"] = p.get("expiry")
        rows.append(row)

        if runrate_per_day_usd is not None:
            total_runrate += runrate_per_day_usd
            have_any_runrate = True
            if (worst_runrate_row is None
                    or runrate_per_day_usd
                    < (worst_runrate_row.get("runrate_per_day_usd") or 0.0)):
                worst_runrate_row = row
        if verdict == "BLEEDING":
            any_bleeding = True

    if not rows:
        return {
            "state": "NO_DATA",
            "rows": [],
            "total_runrate_per_day_usd": None,
            "total_projected_pl_30d_usd": None,
            "worst_runrate": None,
            "any_bleeding": False,
            "headline": "no open positions",
        }

    total_runrate_val = round(total_runrate, 2) if have_any_runrate else None
    total_proj_30d = (round(total_runrate * 30.0, 2)
                      if have_any_runrate else None)

    # Headline — one-liner readable from a Discord summary or a CLI tail.
    if any_bleeding and worst_runrate_row is not None:
        w = worst_runrate_row
        headline = (
            f"BLEEDING — {w['ticker']} at ${w['runrate_per_day_usd']:+.2f}/day "
            f"({w['annualized_pct']:+.0f}%/yr); book pace "
            f"${total_runrate_val:+.2f}/day"
            if total_runrate_val is not None
            else f"BLEEDING — {w['ticker']} at "
                 f"${w['runrate_per_day_usd']:+.2f}/day"
        )
    elif total_runrate_val is not None:
        sign = "earning" if total_runrate_val >= 0 else "losing"
        headline = (
            f"book {sign} ${abs(total_runrate_val):.2f}/day across "
            f"{len(rows)} position(s) (30d projection "
            f"${total_proj_30d:+.2f})"
        )
    else:
        headline = (
            f"{len(rows)} position(s); runrate not yet derivable "
            "(positions too fresh or marks stale)")

    return {
        "state": "OK",
        "rows": rows,
        "total_runrate_per_day_usd": total_runrate_val,
        "total_projected_pl_30d_usd": total_proj_30d,
        "worst_runrate": worst_runrate_row,
        "any_bleeding": any_bleeding,
        "headline": headline,
    }
