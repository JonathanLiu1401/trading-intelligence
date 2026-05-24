"""Cycle-gap summary — historical view of inter-decision gaps vs the dynamic
cadence floor.

A trader-grade operational gap no existing surface fills. The existing
panels each answer ONE thing:

  * ``/api/decision-cadence`` — is the CURRENT cycle overdue against the
    dynamic-interval tier the runner would pick RIGHT NOW?
  * ``/api/runner-heartbeat`` — single-bucket health on the most-recent
    decision (HEALTHY / LAGGING / STALLED).
  * ``/api/decision-paralysis`` — STREAK length of HOLD / NO_DECISION.
  * ``/api/decision-clock`` — per-NY-hour decision *count* concentration.
  * ``/api/decision-efficacy`` — share of recent cycles that produced
    *any* decision (not NO_DECISION).

None of them answer the literal historical question a portfolio manager
asks when the live ``/api/runner-heartbeat`` headline looks fine:

  **"Of the LAST N cycles, how many actually fired on the schedule the
  runner asked for? Was the loop SMOOTH or did it lurch through long
  gaps?"**

The dynamic-interval tier varies per cycle (60s..5400s), so the right
baseline is the per-cycle median over a windowed history — not a single
hardcoded number. ``build_cycle_gap_summary`` walks the most-recent N
decision rows newest→oldest, computes the pairwise wall-clock gaps
between consecutive timestamps, summarises them (median / p95 / max /
worst-row), and emits a 4-state verdict:

  * ``NO_DATA``           — < 2 decisions in the window
  * ``INSUFFICIENT``      — fewer than ``MIN_GAPS`` (8) usable gaps
  * ``STALLED``           — max gap exceeds ``STALL_GAP_S``
                            (3 hours of nothing — beyond any healthy tier)
                            OR p95 exceeds ``STALL_P95_S``
                            (sustained tail above the longest healthy tier
                            — multiple wedges in the window)
  * ``JITTERY``           — coefficient of variation
                            (stdev / median) > ``JITTERY_CV`` (1.0)
  * ``SMOOTH``            — neither stalled nor jittery

The thresholds are conservative on purpose: dynamic_interval already
shrinks the gap to 60-300s during the SESSION_OPEN / EARNINGS_WINDOW
tiers and stretches it to 3600-5400s during quiet-closed, so a window
spanning both will have natural variance that is healthy. STALL_GAP_S
is set above the longest healthy tier (QUIET_CLOSED = 5400s) so only
loop wedges trip it.

Same operational discipline as every sibling builder:

  * Pure read — no DB writes, no network, no subprocess.
  * Never raises (the ``_safe_secs`` / ``except Exception`` discipline).
  * Observational only — never gates Opus, never caps trades
    (AGENTS.md invariants #2/#12).
  * Single-source verdict thresholds — module-level constants so the
    endpoint and any future Discord wiring read the same numbers.

Run as a CLI for a one-shot ops view::

    python3 -m paper_trader.analytics.cycle_gap_summary
"""
from __future__ import annotations

from datetime import datetime, timezone

# Verdict thresholds. STALL_GAP_S sits above QUIET_CLOSED (5400s) so the
# only way a single inter-cycle gap trips STALLED is a genuine wedge —
# the dynamic-interval tier never schedules a gap that long.
#
# Live-data calibration (2026-05-24): a 50-decision window spanning the
# 16:00 ET close had median 350s (OPEN tier) and p95 3870s (QUIET_CLOSED
# tier), which an early STALL_P95_MULT=6.0 rule wrongly flagged STALLED.
# Real tier transitions are NOT stalls. The current rule uses ABSOLUTE
# p95 instead: a wedge has p95 above the longest healthy tier (5400s);
# a legitimate tier mix has p95 below it. JITTERY (high CoV) still
# catches the lurchy-cadence case without false-positiving on dynamic
# interval transitions.
STALL_GAP_S = 3 * 3600          # 10800s — any single gap >= this is a wedge
STALL_P95_S = 2 * 3600          # 7200s — sustained p95 above any healthy tier
JITTERY_CV = 1.0                # coefficient-of-variation cutoff

# Minimum number of usable gaps before the detector commits to a
# verdict. A window of 1-2 cycles is statistically meaningless and the
# silence-by-default reporter line (if any) stays silent — matches the
# ``passive_signal_density.MIN_PASSIVE_RUN=5`` floor discipline.
MIN_GAPS = 8

# Default scan window. Most callers (the dashboard endpoint) pass their
# own ``store.recent_decisions(limit=...)`` so this is just the CLI's
# default. 50 cycles ≈ a day of QUIET_CLOSED or ≈ 30min of SESSION_OPEN
# — wide enough to span a regime change without diluting recent state.
DEFAULT_LIMIT = 50


def _parse_iso(ts) -> datetime | None:
    """Permissive ISO-8601 parser → tz-aware UTC datetime, or None on
    any fault. Matches the convention every sibling builder uses
    (``decision_cadence._parse_iso``, ``signals._age_hours``)."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _median(values: list[float]) -> float:
    """Sample median of a non-empty float sequence. Caller guarantees
    non-empty; the public builder short-circuits to INSUFFICIENT before
    ever calling this on an empty list."""
    n = len(values)
    s = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0,1]. Empty list → 0.0.
    Used for p95; the same convention numpy / statistics module use so
    the value matches the figure an operator would compute by hand."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n == 1:
        return float(s[0])
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def _stdev(values: list[float], mean: float) -> float:
    """Population standard deviation. Returns 0.0 for n<2 (a single
    sample has no spread)."""
    n = len(values)
    if n < 2:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / n
    return var ** 0.5


def build_cycle_gap_summary(
    decisions: list[dict] | None,
    *,
    now: datetime | None = None,
) -> dict:
    """Build the cycle-gap summary payload.

    ``decisions`` is newest-first (the convention
    ``store.recent_decisions`` returns by default — same orientation
    every other analytics builder consumes). Each row needs only a
    ``timestamp`` key; everything else is ignored. Rows with
    unparseable timestamps are silently skipped (never coerced, mirror
    of ``decision_clock._parse_iso`` / ``passive_signal_density``'s
    UNKNOWN-skip discipline).

    Returned dict shape — every field always present, every numeric
    field None when not applicable (mirrors the
    ``decision_cadence`` / ``all_cash_streak`` contract):

      ``as_of``             — ISO of the call instant
      ``state``             — ``"NO_DATA"``, ``"INSUFFICIENT"``, or
                              ``"OK"`` (verdict is meaningful)
      ``verdict``           — ``NO_DATA`` / ``INSUFFICIENT`` / ``SMOOTH``
                              / ``JITTERY`` / ``STALLED``
      ``headline``          — human one-liner
      ``n_decisions``       — usable decisions after timestamp parse
      ``n_gaps``            — pairwise gaps (= n_decisions - 1)
      ``median_gap_s``
      ``mean_gap_s``
      ``p95_gap_s``
      ``max_gap_s``
      ``min_gap_s``
      ``stdev_gap_s``
      ``coefficient_of_variation`` — stdev / median (None on
                                     median=0 — degenerate identical-ts)
      ``n_stalled_gaps``    — gaps >= ``STALL_GAP_S``
      ``pct_stalled``       — n_stalled_gaps / n_gaps × 100
      ``worst_gap``         — ``{"start_ts": …, "end_ts": …, "gap_s": …}``
                              for the single largest gap, or None
      ``stall_gap_threshold_s``
      ``stall_p95_threshold_s``
      ``jittery_cv_threshold``
      ``min_gaps``

    Pure read; never raises; degrades the row, never the whole verdict.
    """
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    # Normalize to (ts_iso, dt) tuples, drop unparseable rows. Preserves
    # the newest-first orientation the caller passed in.
    parsed: list[tuple[str, datetime]] = []
    for d in decisions or []:
        ts = d.get("timestamp")
        dt = _parse_iso(ts)
        if dt is None:
            continue
        parsed.append((ts, dt))

    n_decisions = len(parsed)
    base = {
        "as_of": now_utc.isoformat(timespec="seconds"),
        "n_decisions": n_decisions,
        "n_gaps": 0,
        "median_gap_s": None,
        "mean_gap_s": None,
        "p95_gap_s": None,
        "max_gap_s": None,
        "min_gap_s": None,
        "stdev_gap_s": None,
        "coefficient_of_variation": None,
        "n_stalled_gaps": 0,
        "pct_stalled": None,
        "worst_gap": None,
        "stall_gap_threshold_s": STALL_GAP_S,
        "stall_p95_threshold_s": STALL_P95_S,
        "jittery_cv_threshold": JITTERY_CV,
        "min_gaps": MIN_GAPS,
    }

    if n_decisions < 2:
        return {
            **base,
            "state": "NO_DATA",
            "verdict": "NO_DATA",
            "headline": (
                "NO_DATA — fewer than 2 decisions in the window; "
                "cycle-gap summary not available."
            ),
        }

    # Compute pairwise gaps. parsed is newest-first so to get
    # chronologically-ordered gaps (older→newer) we walk pairs (i+1, i)
    # — the time from the OLDER row to the NEWER row. Negative or zero
    # gaps (two rows at the same microsecond, or a wall-clock step-back)
    # clamp to 0 so a wall-clock skew can never render as a negative
    # cycle gap (matches ``signals._age_hours`` / ``decision_cadence``
    # hardening).
    gap_rows: list[tuple[str, str, float]] = []  # (start_ts, end_ts, gap_s)
    for i in range(len(parsed) - 1):
        older_ts, older_dt = parsed[i + 1]
        newer_ts, newer_dt = parsed[i]
        raw = (newer_dt - older_dt).total_seconds()
        if raw < 0:
            raw = 0.0
        gap_rows.append((older_ts, newer_ts, raw))

    gaps = [g[2] for g in gap_rows]
    n_gaps = len(gaps)

    if n_gaps < MIN_GAPS:
        return {
            **base,
            "n_gaps": n_gaps,
            "state": "INSUFFICIENT",
            "verdict": "INSUFFICIENT",
            "headline": (
                f"INSUFFICIENT — only {n_gaps} usable cycle gap"
                f"{'s' if n_gaps != 1 else ''} in the window "
                f"(need {MIN_GAPS}+ for a verdict)."
            ),
        }

    median = _median(gaps)
    mean = sum(gaps) / n_gaps
    p95 = _percentile(gaps, 0.95)
    mx = max(gaps)
    mn = min(gaps)
    stdev = _stdev(gaps, mean)
    cov = (stdev / median) if median > 0 else None
    n_stalled = sum(1 for g in gaps if g >= STALL_GAP_S)
    pct_stalled = n_stalled / n_gaps * 100.0 if n_gaps > 0 else 0.0

    # Locate the worst-gap row for surface (start_ts → end_ts → gap_s).
    worst = max(gap_rows, key=lambda r: r[2])
    worst_dict = {
        "start_ts": worst[0],
        "end_ts": worst[1],
        "gap_s": round(worst[2], 1),
    }

    # Verdict ladder. STALLED takes precedence over JITTERY: a long-tail
    # wedge with a single 4h gap can also have a high CV, but the
    # operator action is "investigate the wedge", not "smooth the tail".
    stalled_by_max = mx >= STALL_GAP_S
    stalled_by_tail = p95 >= STALL_P95_S
    if stalled_by_max or stalled_by_tail:
        verdict = "STALLED"
        reasons: list[str] = []
        if stalled_by_max:
            reasons.append(
                f"max gap {int(mx)}s >= {STALL_GAP_S}s "
                f"(threshold for a wedge)"
            )
        if stalled_by_tail:
            reasons.append(
                f"p95 {int(p95)}s >= {STALL_P95_S}s "
                f"(longest healthy tier)"
            )
        headline = (
            f"STALLED — {n_gaps} cycle gaps, "
            f"median {int(median)}s, max {int(mx)}s. "
            + " · ".join(reasons)
            + ". Loop appears wedged for at least one stretch in the window."
        )
    elif cov is not None and cov > JITTERY_CV:
        verdict = "JITTERY"
        headline = (
            f"JITTERY — {n_gaps} cycle gaps, "
            f"median {int(median)}s, stdev {int(stdev)}s "
            f"(CoV {cov:.2f} > {JITTERY_CV:.1f}). "
            f"Cycle cadence is varying widely — check the dynamic-"
            f"interval tier mix or transient host saturation."
        )
    else:
        verdict = "SMOOTH"
        cov_token = f"{cov:.2f}" if cov is not None else "n/a"
        headline = (
            f"SMOOTH — {n_gaps} cycle gaps, "
            f"median {int(median)}s, max {int(mx)}s "
            f"(CoV {cov_token}). "
            f"Loop is running within the expected cadence window."
        )

    return {
        **base,
        "n_gaps": n_gaps,
        "state": "OK",
        "verdict": verdict,
        "headline": headline,
        "median_gap_s": round(median, 1),
        "mean_gap_s": round(mean, 1),
        "p95_gap_s": round(p95, 1),
        "max_gap_s": round(mx, 1),
        "min_gap_s": round(mn, 1),
        "stdev_gap_s": round(stdev, 1),
        "coefficient_of_variation": (
            round(cov, 4) if cov is not None else None
        ),
        "n_stalled_gaps": n_stalled,
        "pct_stalled": round(pct_stalled, 2),
        "worst_gap": worst_dict,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json

    from paper_trader.store import get_store
    s = get_store()
    rep = build_cycle_gap_summary(
        s.recent_decisions(limit=DEFAULT_LIMIT)
    )
    print(json.dumps(rep, indent=2, default=str))
