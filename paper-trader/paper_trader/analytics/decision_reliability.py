"""Decision reliability — the *true current-regime* parse-failure rate.

The mature analytics layer already measures the NO_DECISION pathology three
ways: ``decision_health`` (the rate), ``decision_forensics`` (the *why* /
failure taxonomy), ``decision_drought`` (the *cost* / involuntary alpha bleed).
None of them answer the question an operator actually has when they see a scary
headline number like "61% NO_DECISION":

    *Is that still happening, or is it dominated by dead legacy rows?*

``strategy.py`` tags pre-diagnostics parse failures as the legacy string
``"claude returned no parseable JSON"``. Once the runner restarts onto
diagnostic code those rows **stop accruing** — they are a fixed historical
mass. A lifetime rate computed over them is misleading: it neither reflects the
current code path nor decays. This module partitions the decision log at the
**newest legacy failure** and reports the failure rate of the *current* regime
only, with explicit sample-size honesty (verdict withheld until enough
post-restart cycles exist) and a ``restart_recommended`` signal when the log is
still legacy-dominated.

It is a **pure composition** of the single-source-of-truth builders — the
failure taxonomy comes from ``decision_forensics.classify_failure``, the
realized bleed from ``decision_drought.build_decision_drought``, the
legacy-inclusive headline rate from ``build_decision_forensics``. Nothing here
re-derives a metric those modules own (the ``capital_paralysis`` precedent).
Advisory only — it never gates the trader and adds no caps (AGENTS.md
invariants #2 / #12).
"""
from __future__ import annotations

from datetime import datetime, timezone

from .decision_drought import build_decision_drought
from .decision_forensics import (
    MODES,
    _is_no_decision,
    _parse_ts,
    build_decision_forensics,
    classify_failure,
)

# Post-restart cycles required before the current-regime verdict is trusted.
# Mirrors the news_edge / trade_asymmetry "numbers from n=1, label from N"
# convention; tests read this constant so a retune can't false-fail them.
MIN_CURRENT = 12

# Current-regime rate thresholds — deliberately identical to
# decision_forensics' verdict bands so the two never disagree on the same data.
CRITICAL_PCT = 50.0
DEGRADED_PCT = 25.0


def build_decision_reliability(decisions: list[dict],
                               equity_curve: list[dict],
                               now: datetime | None = None) -> dict:
    """Regime-partitioned NO_DECISION reliability (newest-first row list).

    ``decisions`` is newest-first (as ``store.recent_decisions`` returns);
    ``equity_curve`` is ascending (as ``store.equity_curve`` returns). Pure:
    never touches the DB. ``now`` is injectable for deterministic tests.
    """
    now = now or datetime.now(timezone.utc)
    n = len(decisions)
    out: dict = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_decisions": n,
        "regime_boundary": None,
        "legacy_failures": 0,
        "legacy_share_pct": 0.0,
        "current_total": 0,
        "current_failures": 0,
        "current_failure_rate_pct": 0.0,
        "current_mode_mix": [],
        "headline_failure_rate_pct": 0.0,
        "involuntary_alpha_bleed_pct": 0.0,
        "decisions_per_day": None,
        "dead_cycles_per_day": None,
        "min_current": MIN_CURRENT,
        "state": "NO_DATA",
        "restart_recommended": False,
        "headline": "No decisions recorded yet.",
    }
    if not decisions:
        return out

    # Single source of truth: legacy-inclusive headline rate + realized bleed.
    fz = build_decision_forensics(decisions, now=now)
    dr = build_decision_drought(decisions, equity_curve or [], now=now)
    headline_rate = fz.get("failure_rate_pct") or 0.0
    bleed = dr.get("involuntary_alpha_bleed_pct") or 0.0

    # Single pass: classify each row via the canonical taxonomy helpers.
    rows: list[tuple[datetime | None, bool, str | None]] = []
    legacy_fail_ts: list[datetime] = []
    legacy_failures = 0
    parsed_ts_all: list[datetime] = []
    for d in decisions:
        ts = _parse_ts(d.get("timestamp"))
        if ts is not None:
            parsed_ts_all.append(ts)
        nd = _is_no_decision(d.get("action_taken"))
        mode = None
        if nd:
            cls = classify_failure(d.get("reasoning"))
            mode = cls["mode"]
            if cls["tag"] == "legacy":
                legacy_failures += 1
                if ts is not None:
                    legacy_fail_ts.append(ts)
        rows.append((ts, nd, mode))

    # Regime boundary = newest legacy failure timestamp. No legacy failure (or
    # all legacy rows have unparseable timestamps) ⇒ no boundary ⇒ every row is
    # "current" (there is no dead pre-restart mass to exclude).
    boundary = max(legacy_fail_ts) if legacy_fail_ts else None
    out["regime_boundary"] = boundary.isoformat() if boundary else None

    if boundary is None:
        current = list(rows)
    else:
        current = [r for r in rows if r[0] is not None and r[0] > boundary]

    current_total = len(current)
    current_failures = sum(1 for (_, nd, _) in current if nd)
    current_rate = (round(current_failures / current_total * 100, 1)
                    if current_total else 0.0)

    mode_n: dict[str, int] = {}
    for (_, nd, mode) in current:
        if nd and mode:
            mode_n[mode] = mode_n.get(mode, 0) + 1
    current_mode_mix = sorted(
        ({"mode": m, "n": c, "pct": round(c / current_failures * 100, 1)}
         for m, c in mode_n.items()),
        key=lambda r: (-r["n"],
                       MODES.index(r["mode"]) if r["mode"] in MODES else 99),
    )

    legacy_share = round(legacy_failures / n * 100, 1) if n else 0.0

    # Cadence from the full timestamp span (most samples ⇒ stablest estimate of
    # the trader's cycle rate). Guarded against a zero / single-point span so it
    # never divides by zero — degrades to None, never raises.
    decisions_per_day = None
    dead_cycles_per_day = None
    if len(parsed_ts_all) >= 2:
        span_s = (max(parsed_ts_all) - min(parsed_ts_all)).total_seconds()
        if span_s > 0:
            decisions_per_day = round(n / (span_s / 86400.0), 3)
            if current_total:
                dead_cycles_per_day = round(
                    current_failures / current_total * decisions_per_day, 2)

    # State machine. STALE_LEGACY_DOMINATED is the actionable case: the log is
    # still mostly dead legacy rows AND the post-restart sample is too small to
    # judge — the fix is to restart the runner so failures get diagnostic tags
    # and the current sample grows. Once enough current cycles accumulate the
    # verdict is judged on the *current* rate regardless of the legacy mass.
    if (boundary is not None and legacy_failures > current_total
            and current_total < MIN_CURRENT):
        state, restart = "STALE_LEGACY_DOMINATED", True
        headline = (
            f"STALE — {legacy_failures} legacy pre-diagnostics NO_DECISION "
            f"row(s) dominate ({legacy_share}% of {n}); the post-restart "
            f"sample is only {current_total} (<{MIN_CURRENT}). Restart the "
            f"paper-trader runner so new failures carry diagnostic tags and "
            f"the true current-regime rate becomes measurable.")
    elif current_total < MIN_CURRENT:
        state, restart = "INSUFFICIENT", False
        headline = (
            f"INSUFFICIENT — current-regime sample {current_total} "
            f"(<{MIN_CURRENT}); the true parse-fail rate is not yet "
            f"judgeable. Headline {headline_rate}% still includes "
            f"{legacy_failures} dead legacy row(s).")
    else:
        if current_rate >= CRITICAL_PCT:
            state = "CRITICAL"
        elif current_rate >= DEGRADED_PCT:
            state = "DEGRADED"
        else:
            state = "HEALTHY"
        restart = False
        bleed_clause = (
            f" Inaction during these has bled {bleed:.2f}% alpha."
            if bleed < 0 else "")
        gap = ""
        if legacy_failures:
            gap = (f" (the {headline_rate}% headline is inflated by "
                   f"{legacy_failures} dead legacy row(s))")
        headline = (
            f"{state} — current-regime parse-fail {current_rate}% over "
            f"{current_total} cycle(s){gap}.{bleed_clause}")

    out.update({
        "legacy_failures": legacy_failures,
        "legacy_share_pct": legacy_share,
        "current_total": current_total,
        "current_failures": current_failures,
        "current_failure_rate_pct": current_rate,
        "current_mode_mix": current_mode_mix,
        "headline_failure_rate_pct": headline_rate,
        "involuntary_alpha_bleed_pct": bleed,
        "decisions_per_day": decisions_per_day,
        "dead_cycles_per_day": dead_cycles_per_day,
        "state": state,
        "restart_recommended": restart,
        "headline": headline,
    })
    return out
