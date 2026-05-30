"""Recent-vs-prior win-rate trend — "is the desk getting better or worse?".

The desk-level metrics already report a lifetime win rate via
``loser_autopsy`` / ``trader_scorecard``. Neither answers the
forward-looking question a trader asks every session:

    *"Of my last N closed round-trips, was my win rate BETTER or WORSE
    than my track record before them?"*

A trader who reads "lifetime win rate 16.7%" cannot tell whether that
average is being PULLED UP by a recent cluster of wins (the engine is
improving) or PROPPED UP by ancient wins despite a recent collapse
(the engine is degrading). The same number can describe a recovering
desk and a bleeding one; only the trend within it is actionable.

``build_win_rate_trend`` splits the closed round-trip ledger into a
RECENT window (the last ``recent_n`` trips, default 20) and a PRIOR
window (everything before), then compares win rates. The trader sees
the *direction of travel* — the orthogonal signal to the aggregate.

Pure & offline: the caller hands in the round-trip list (oldest→newest)
from ``round_trips.build_round_trips`` and an optional ``recent_n``.
No store reads, no clock — the verdict is a function of the ledger
shape alone. Garbage-safe: non-list inputs, missing ``pnl_usd`` fields,
zero-width windows all degrade to ``NO_DATA`` / ``INSUFFICIENT`` without
raising.

Advisory only — never modifies anything, never gates Opus, no caps
(AGENTS.md invariants #2 / #12). Observational only.

Verdict ladder:

| Verdict          | Trigger                                                                     |
|------------------|-----------------------------------------------------------------------------|
| ``NO_DATA``      | < ``MIN_TOTAL`` (10) closed round-trips total                               |
| ``INSUFFICIENT`` | ≥ MIN_TOTAL total but either window has < ``MIN_WINDOW`` (5) round-trips    |
| ``TRENDING_UP``  | ``recent_win_rate − prior_win_rate ≥ +DELTA_THRESHOLD_PP`` (+10pp)          |
| ``TRENDING_DOWN``| ``recent_win_rate − prior_win_rate ≤ -DELTA_THRESHOLD_PP`` (-10pp)          |
| ``STABLE``       | ``|recent − prior| < DELTA_THRESHOLD_PP`` (within 10pp — noise)             |

Headline names both win rates and the delta in plain English; callers
suppress STABLE / INSUFFICIENT / NO_DATA to keep the hourly summary
silent on non-actionable states (the silence-when-nothing-actionable
precedent established by ``_hold_discipline_line`` /
``_drawdown_line``).
"""
from __future__ import annotations

from typing import Any


DEFAULT_RECENT_N = 20

# Need at least this many round-trips total before a recent-vs-prior
# comparison is meaningful. Five-on-five would be statistically silly;
# ten total gives enough room for a 5+5 split AND prevents the verdict
# from flapping on a 4-trip desk (a single trade flips lifetime by 25pp).
MIN_TOTAL = 10

# Both windows need at least this count; a one-trade window has 0% or
# 100% win rate with no information content. Five in each window is the
# minimum where a one-trade flip changes the rate by ≤ 20pp.
MIN_WINDOW = 5

# Verdict threshold: window-on-window swing in winning %. 10pp is large
# enough to clear typical sampling variance on small windows
# (~σ √(p(1-p)/n) ≈ √(0.25/10) ≈ 16% at p=0.5, n=10 → so ±10pp is a
# conservative shift, not noise) yet small enough to catch real drift.
DELTA_THRESHOLD_PP = 10.0


def _is_win(rt: Any) -> bool | None:
    """True if the round-trip pnl > 0, False if ≤ 0, None on garbage.
    Mirrors ``build_round_trips``'s field names exactly."""
    if not isinstance(rt, dict):
        return None
    pnl = rt.get("pnl_usd")
    try:
        v = float(pnl) if pnl is not None else None
    except (TypeError, ValueError):
        return None
    if v is None:
        return None
    return v > 0


def _win_rate_pct(rts: list[dict]) -> float | None:
    """Win % over the round-trip slice, or None when the slice is empty
    or has zero parseable rows."""
    parsed = [_is_win(rt) for rt in rts]
    parsed = [b for b in parsed if b is not None]
    if not parsed:
        return None
    wins = sum(1 for b in parsed if b)
    return round(100.0 * wins / len(parsed), 2)


def build_win_rate_trend(
    round_trips_oldest_first: Any,
    recent_n: int = DEFAULT_RECENT_N,
) -> dict:
    """Pure builder. ``round_trips_oldest_first`` is the
    ``build_round_trips`` output (oldest first). ``recent_n`` is the
    size of the RECENT window (default 20); everything before that is
    the PRIOR window. Returns a stable shape:

      * ``state``                — verdict ladder member
      * ``headline``             — operator one-sentence summary
                                    (empty on NO_DATA)
      * ``recent_n``             — count of round-trips in the recent
                                    window (≤ recent_n)
      * ``prior_n``              — count in the prior window
      * ``total_n``              — total parseable round-trips
      * ``recent_win_rate_pct``  — win % over the recent window (None
                                    when INSUFFICIENT_RECENT)
      * ``prior_win_rate_pct``   — win % over the prior window (None
                                    when INSUFFICIENT_PRIOR)
      * ``lifetime_win_rate_pct``— win % over total_n (None on NO_DATA)
      * ``delta_pp``             — recent − prior, in percentage points
                                    (None on INSUFFICIENT)
      * ``threshold_pp``         — the verdict threshold in pp
      * ``min_total``            — the NO_DATA floor
      * ``min_window``           — the INSUFFICIENT-window floor

    Never raises."""
    base: dict = {
        "state": "NO_DATA",
        "headline": "",
        "recent_n": 0,
        "prior_n": 0,
        "total_n": 0,
        "recent_win_rate_pct": None,
        "prior_win_rate_pct": None,
        "lifetime_win_rate_pct": None,
        "delta_pp": None,
        "threshold_pp": DELTA_THRESHOLD_PP,
        "min_total": MIN_TOTAL,
        "min_window": MIN_WINDOW,
    }

    if not isinstance(round_trips_oldest_first, list):
        return base
    # Filter to parseable rows. A garbage row is excluded entirely from
    # both counting and rate math — same defensive shape as the rest
    # of the analytics layer.
    rts = [rt for rt in round_trips_oldest_first if _is_win(rt) is not None]
    total = len(rts)
    base["total_n"] = total

    if total < MIN_TOTAL:
        base["headline"] = (
            f"only {total} closed round-trip"
            + ("s" if total != 1 else "")
            + " yet — need ≥"
            + f"{MIN_TOTAL} for a trend read"
        )
        # Compute lifetime win % even on NO_DATA so callers can show it
        # if they want — costs nothing and matches the "report what we
        # can" discipline.
        base["lifetime_win_rate_pct"] = _win_rate_pct(rts)
        return base

    try:
        n = int(recent_n)
    except (TypeError, ValueError):
        n = DEFAULT_RECENT_N
    # Clamp: ≥ MIN_WINDOW so any recent window has minimum signal; ≤
    # total - MIN_WINDOW so the prior window also has signal. With
    # total ≥ MIN_TOTAL = 10 and MIN_WINDOW = 5, the upper clamp is at
    # least 5 — so a clamp always succeeds.
    n = max(MIN_WINDOW, min(total - MIN_WINDOW, n))

    recent = rts[-n:] if n > 0 else []
    prior = rts[:-n] if n > 0 else rts

    base["recent_n"] = len(recent)
    base["prior_n"] = len(prior)
    base["lifetime_win_rate_pct"] = _win_rate_pct(rts)

    if len(recent) < MIN_WINDOW or len(prior) < MIN_WINDOW:
        base["state"] = "INSUFFICIENT"
        base["headline"] = (
            f"need ≥{MIN_WINDOW} round-trips in each window "
            f"(have {len(recent)} recent, {len(prior)} prior)"
        )
        return base

    rwr = _win_rate_pct(recent)
    pwr = _win_rate_pct(prior)
    base["recent_win_rate_pct"] = rwr
    base["prior_win_rate_pct"] = pwr

    if rwr is None or pwr is None:
        # Shouldn't happen — we filtered to parseable rows above — but
        # defensive: a recent slice with zero parseable rows degrades
        # honestly.
        base["state"] = "INSUFFICIENT"
        base["headline"] = (
            "win-rate computation failed on at least one window")
        return base

    delta = round(rwr - pwr, 2)
    base["delta_pp"] = delta

    if delta >= DELTA_THRESHOLD_PP:
        base["state"] = "TRENDING_UP"
        base["headline"] = (
            f"recent win rate {rwr:.1f}% vs prior {pwr:.1f}% "
            f"(+{delta:.1f}pp over last {len(recent)} closed trips) — "
            f"the desk is improving"
        )
    elif delta <= -DELTA_THRESHOLD_PP:
        base["state"] = "TRENDING_DOWN"
        base["headline"] = (
            f"recent win rate {rwr:.1f}% vs prior {pwr:.1f}% "
            f"({delta:.1f}pp over last {len(recent)} closed trips) — "
            f"the desk is regressing"
        )
    else:
        base["state"] = "STABLE"
        base["headline"] = (
            f"recent win rate {rwr:.1f}% vs prior {pwr:.1f}% "
            f"({delta:+.1f}pp, within ±{DELTA_THRESHOLD_PP:.0f}pp noise)"
        )
    return base
