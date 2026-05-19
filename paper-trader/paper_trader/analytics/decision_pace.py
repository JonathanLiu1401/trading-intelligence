"""Inter-decision latency distribution — is the runner cycling on cadence?

Closes a gap between three existing surfaces:

* ``decision_health`` reports a single ``decisions/day`` aggregate. One scalar
  cannot say whether the box runs smoothly at 48 cycles/day with rare hiccups,
  or thrashes between 20-minute and 4-hour gaps that average out to the same
  rate.
* ``runner_heartbeat`` answers "is the loop alive *right now*" by comparing
  ``now - last_decision_ts`` to the expected cadence. It is a single-sample
  test on the most-recent gap; it does not characterise the distribution.
* ``decision_drought`` segments runs between FILLED trades and prices their
  alpha drift; it ignores the inter-cycle latency of cycles that did *not*
  fill (i.e. almost all of them).

This module computes the **rolling p50 / p95 / max** of the gap (in seconds)
between consecutive ``decisions`` rows over standard windows, separately for
market-open and market-closed cycles. The split matters: runner cadence is
bimodal — ``OPEN_INTERVAL_S = 1800`` vs ``CLOSED_INTERVAL_S = 3600`` — so a
merged distribution is the wrong unit. Each gap is classified by the
``market_open`` value of its **trailing** (later) decision row, which is the
cycle that "should have fired ``CADENCE_S`` ago".

Pure: feed it ``store.recent_decisions(limit)`` (newest-first) and a window
list; it never touches the DB.

Sample-size honesty (the ``streak`` / ``churn`` idiom):

* ``NO_DATA`` — fewer than ``MIN_GAPS_FOR_DISTRIBUTION`` gaps over the whole
  series (one decision yields zero gaps, so a single-row trader is honest
  about having nothing to measure).
* ``EMERGING`` — gaps exist but the largest window has fewer than
  ``STABLE_MIN_GAPS`` samples; numeric percentiles emit, the **verdict** is
  withheld.
* ``STABLE`` — verdict is gated.

Verdict ladder (gated to STABLE):

* ``HEALTHY`` — every window's p95 is within
  ``CADENCE_TOLERANCE_FACTOR`` × expected cadence.
* ``LAGGING`` — the freshest window's (the one named in ``WINDOWS[0]``) p95
  exceeds ``CADENCE_TOLERANCE_FACTOR`` × expected cadence but stays under
  ``CADENCE_STALL_FACTOR``.
* ``STALLED`` — that p95 exceeds ``CADENCE_STALL_FACTOR`` × expected cadence
  (a runner that misses a cycle every ~20 cycles is stalling, not jittering).

Advisory only — never gates Opus, never injected into the decision prompt,
adds no caps (AGENTS.md #2/#12). Never raises.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Mirrors runner.OPEN_INTERVAL_S / CLOSED_INTERVAL_S — deliberately not
# imported (the runner_heartbeat precedent: keep this leaf pure and
# import-cycle-free; the test reads these constants so a retune cannot
# false-fail the suite).
OPEN_INTERVAL_S = 1800.0
CLOSED_INTERVAL_S = 3600.0

# A gap inside this multiple of its cadence is "on schedule".
CADENCE_TOLERANCE_FACTOR = 1.5
# Past this, it has skipped a full cycle and then some.
CADENCE_STALL_FACTOR = 2.5

# Windows for the rolling p50/p95 (hours). Order matters: WINDOWS[0] is the
# freshest, which feeds the verdict — operator question is "is the loop
# healthy *right now*", not "was it healthy last week".
WINDOWS = (1, 6, 24, 168)

# Sample-size gates — same idiom as streak.py / churn.py.
MIN_GAPS_FOR_DISTRIBUTION = 2
STABLE_MIN_GAPS = 10


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile (NIST type-7 / numpy default).

    ``values`` need not be sorted; we sort a copy. Returns None on empty —
    every caller already branches on emptiness, but the explicit None keeps
    the JSON honest. p0/p100 collapse to min/max for any length.
    """
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def _distribution(gaps: list[float]) -> dict:
    """``{n, p50, p95, max, mean}`` — None percentiles on empty input."""
    if not gaps:
        return {"n": 0, "p50": None, "p95": None, "max": None, "mean": None}
    return {
        "n": len(gaps),
        "p50": round(_percentile(gaps, 50.0), 2),
        "p95": round(_percentile(gaps, 95.0), 2),
        "max": round(max(gaps), 2),
        "mean": round(sum(gaps) / len(gaps), 2),
    }


def build_decision_pace(decisions: list[dict],
                        now: datetime | None = None) -> dict:
    """Rolling inter-decision latency distribution + verdict.

    Args:
        decisions: ``store.recent_decisions(limit=...)`` rows (newest-first).
        now: injectable for deterministic tests; defaults to UTC now.

    Returns a JSON-ready dict; never raises.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # ---- parse & sort chronologically (ascending) ----------------------
    parsed: list[tuple[datetime, int]] = []
    for d in decisions or []:
        if not isinstance(d, dict):
            continue
        dt = _parse_ts(d.get("timestamp"))
        if dt is None:
            continue
        try:
            mo = int(d.get("market_open") or 0)
        except (TypeError, ValueError):
            mo = 0
        parsed.append((dt, 1 if mo else 0))
    parsed.sort(key=lambda r: r[0])

    # ---- build gaps. Each gap is classified by the trailing cycle's
    # market_open — "this cycle should have fired one cadence_s ago". -----
    all_gaps: list[tuple[datetime, float, int]] = []  # (trailing_dt, gap_s, mo)
    for prev, nxt in zip(parsed, parsed[1:]):
        gap_s = (nxt[0] - prev[0]).total_seconds()
        if gap_s < 0:  # clock step-back; chronological by construction
            continue
        all_gaps.append((nxt[0], gap_s, nxt[1]))

    n_total_gaps = len(all_gaps)
    last_ts = parsed[-1][0] if parsed else None
    last_mo = parsed[-1][1] if parsed else 0
    age_now_s = round((now - last_ts).total_seconds(), 2) if last_ts else None

    # ---- per-window distributions, split by market_open ----------------
    windows: dict[str, dict] = {}
    for h in WINDOWS:
        cutoff = now - timedelta(hours=h)
        bucket = [(g, mo) for (t, g, mo) in all_gaps if t >= cutoff]
        open_g = [g for (g, mo) in bucket if mo]
        closed_g = [g for (g, mo) in bucket if not mo]
        windows[f"{h}h"] = {
            "open": _distribution(open_g),
            "closed": _distribution(closed_g),
            "all": _distribution([g for (g, _) in bucket]),
        }

    # ---- state ladder ---------------------------------------------------
    if n_total_gaps < MIN_GAPS_FOR_DISTRIBUTION:
        state = "NO_DATA"
    elif max((w["all"]["n"] or 0) for w in windows.values()) >= STABLE_MIN_GAPS:
        state = "STABLE"
    else:
        state = "EMERGING"

    # ---- verdict (gated to STABLE, judged on the freshest window) ------
    verdict: str | None = None
    verdict_reason: str | None = None
    if state == "STABLE":
        fresh = windows[f"{WINDOWS[0]}h"]
        # Pick the cadence the freshest window actually covers. If both
        # open & closed samples exist, judge each against its own cadence
        # and take the worse verdict (a stall is a stall whichever cadence
        # it broke).
        checks: list[tuple[str, float, float]] = []  # (regime, p95, cadence)
        if fresh["open"]["n"] and fresh["open"]["p95"] is not None:
            checks.append(("open", fresh["open"]["p95"], OPEN_INTERVAL_S))
        if fresh["closed"]["n"] and fresh["closed"]["p95"] is not None:
            checks.append(("closed", fresh["closed"]["p95"], CLOSED_INTERVAL_S))
        worst_kind = "HEALTHY"
        worst_detail = ""
        for regime, p95, cad in checks:
            ratio = p95 / cad
            if ratio >= CADENCE_STALL_FACTOR:
                kind = "STALLED"
            elif ratio >= CADENCE_TOLERANCE_FACTOR:
                kind = "LAGGING"
            else:
                kind = "HEALTHY"
            rank = {"HEALTHY": 0, "LAGGING": 1, "STALLED": 2}
            if rank[kind] > rank[worst_kind]:
                worst_kind = kind
                worst_detail = (
                    f"{regime}-cycle p95={p95:.0f}s "
                    f"vs cadence {cad:.0f}s (×{ratio:.2f})")
        verdict = worst_kind
        if worst_kind == "HEALTHY":
            verdict_reason = (
                f"p95 within ×{CADENCE_TOLERANCE_FACTOR:g} cadence over the "
                f"last {WINDOWS[0]}h")
        else:
            verdict_reason = worst_detail

    # ---- headline ------------------------------------------------------
    if state == "NO_DATA":
        headline = (
            f"No inter-decision gaps yet — need at least "
            f"{MIN_GAPS_FOR_DISTRIBUTION} cycles to measure pace.")
    elif state == "EMERGING":
        headline = (
            f"Emerging — {n_total_gaps} gap"
            f"{'s' if n_total_gaps != 1 else ''} measured so far; verdict "
            f"withheld until any window reaches {STABLE_MIN_GAPS} samples.")
    else:
        fresh = windows[f"{WINDOWS[0]}h"]
        a = fresh["all"]
        p50 = "n/a" if a["p50"] is None else f"{a['p50']:.0f}s"
        p95 = "n/a" if a["p95"] is None else f"{a['p95']:.0f}s"
        headline = (
            f"{verdict} — last {WINDOWS[0]}h: p50 {p50} / p95 {p95} over "
            f"{a['n']} cycles ({verdict_reason}).")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "headline": headline,
        "n_decisions": len(parsed),
        "n_gaps": n_total_gaps,
        "last_decision_ts": (last_ts.isoformat(timespec="seconds")
                             if last_ts else None),
        "last_decision_market_open": bool(last_mo) if parsed else None,
        "age_since_last_decision_s": age_now_s,
        "windows": windows,
        "open_cadence_s": OPEN_INTERVAL_S,
        "closed_cadence_s": CLOSED_INTERVAL_S,
        "cadence_tolerance_factor": CADENCE_TOLERANCE_FACTOR,
        "cadence_stall_factor": CADENCE_STALL_FACTOR,
        "stable_min_gaps": STABLE_MIN_GAPS,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json

    from paper_trader.store import get_store
    s = get_store()
    rep = build_decision_pace(s.recent_decisions(limit=2000))
    print(json.dumps(rep, indent=2, default=str))
