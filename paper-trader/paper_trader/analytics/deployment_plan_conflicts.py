"""Deployment-plan internal-conflict audit.

The desk's ``inverse_pair_conflict`` flags when the *held book* has
simultaneous leveraged-long and leveraged-inverse exposure on the same
underlying family (a TQQQ + SQQQ pathology that pays leverage decay on
both sides). That audit only looks at what's already on. But the
``/api/deployment-plan`` planner is happy to *propose* a plan that
embeds the same pathology: it ranks purely by ``pred_5d_return_pct`` and
will pick TECS (-3x USTECH inverse) alongside TECL (+3x USTECH long) if
both clear the pred floor and survive the per-name / sector / leverage
caps. Live evidence (2026-05-27 production trader): the planner emitted
a 4-name plan with TECS -3x ($292), NVDU +2x ($58), AMAT 1x ($292), NVDA
1x ($175) — TECS is leveraged-inverse tech while NVDU+AMAT+NVDA are
long-tech-direction, a textbook "directional hedge" the operator wasn't
told about anywhere on the dashboard.

This module is the missing mirror for the *plan*. Given the plan rows
emitted by ``build_deployment_plan``, it walks the proposed allocations,
projects each ticker onto the same ``inverse_pair_conflict._PAIR_FAMILIES``
taxonomy (single source of truth), and flags two distinct pathologies:

1. **Family conflict** — within a single family the plan has both
   leveraged-long and leveraged-inverse (``CARRY_WASTE``), or 1x-core
   + leveraged-inverse (``OPPOSING_UNLEVERED``). Same severity ladder
   as ``inverse_pair_conflict``.
2. **Directional hedge** — across families, the plan has meaningful
   ``$_long_lev`` AND ``$_inverse_lev`` (both ≥ ``DIRECTIONAL_MIN_PCT``
   of total deployed $). This catches the live case where TECS is the
   only leveraged-inverse name but NVDU is the only leveraged-long
   name and they're in different families — still a hedge the operator
   should know they're paying decay on.

Pure & offline. No DB, no network. Walks the plan rows + the existing
family taxonomy and returns a deterministic JSON-ready dict. Same
``_safe`` discipline as adjacent analytics: garbage rows contribute
nothing, empty plan degrades to ``NO_PLAN``. Never raises.

Advisory only — never modifies the plan, never gates Opus, no caps.
The operator decides whether the hedge is intentional or a model
artifact to drop (AGENTS.md invariants #2 / #12).

Verdict ladder (most-severe-first; first-match wins):

| Verdict                 | Meaning                                                                  |
|-------------------------|--------------------------------------------------------------------------|
| ``NO_PLAN``             | empty plan — nothing to evaluate                                         |
| ``CARRY_WASTE``         | at least one family has BOTH leveraged-long AND leveraged-inverse        |
|                         | populated in the plan — operator pays decay on both sides                |
| ``OPPOSING_UNLEVERED``  | at least one family has 1x-core + leveraged-inverse (or leveraged-long  |
|                         | + 1x-core) — delta offsets, decay on the leveraged sleeve only           |
| ``DIRECTIONAL_HEDGE``   | aggregate long-lev $ ≥ DIRECTIONAL_MIN_PCT AND inverse-lev $ ≥           |
|                         | DIRECTIONAL_MIN_PCT — cross-family hedge with double decay carry         |
| ``CLEAN``               | none of the above                                                        |
"""
from __future__ import annotations

from typing import Any

# Single-source taxonomy. Importing the index ensures any drift between
# inverse_pair_conflict's family map and this module is impossible by
# construction — same precedent as
# ``deployment_plan.py`` reusing ``sector_exposure.SECTOR_MAP``.
from .inverse_pair_conflict import (
    _PAIR_FAMILIES,
    _TICKER_INDEX,
    SEVERITY_HIGH_PCT,
    SEVERITY_MEDIUM_PCT,
)

# Aggregate cross-family hedge threshold. Below this, the "hedge" is too
# small to be operationally meaningful; above it the operator should
# know they're paying carry on both directions. 5% is a tight band by
# design — even a token leveraged-inverse exposure inside an otherwise
# long-leveraged plan deserves a banner.
DIRECTIONAL_MIN_PCT = 5.0


def _f(x, default: float = 0.0) -> float:
    """Float coercion; garbage → default, never raises."""
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _z(v, ndigits: int = 2):
    """Round; fold ``-0.0 → 0.0``; None / non-numeric → None."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _severity(cancelled_delta: float, gross_delta: float) -> str:
    """Mirror ``inverse_pair_conflict._severity`` so per-family report
    thresholds are stable across the held-book and proposed-plan audits."""
    if gross_delta <= 0:
        return "LOW"
    pct = 100.0 * cancelled_delta / gross_delta
    if pct >= SEVERITY_HIGH_PCT:
        return "HIGH"
    if pct >= SEVERITY_MEDIUM_PCT:
        return "MEDIUM"
    return "LOW"


def build_deployment_plan_conflicts(plan: Any) -> dict:
    """Pure builder. ``plan`` is the row list from ``build_deployment_plan``
    (same shape as ``/api/deployment-plan``'s ``plan`` field). Each row
    contributes ``ticker`` + ``alloc_usd`` (and an optional explicit
    ``leverage_factor`` / ``is_leveraged`` we mirror back into the
    report). Returns the verdict ladder above. Never raises.

    The plan-row format already carries ``leverage_factor`` (1, ±2, ±3)
    set by ``build_deployment_plan``, but we re-derive the leverage role
    from the family taxonomy so a stale or wrong ``leverage_factor`` on
    the input can't poison the audit — taxonomy is SSOT."""

    rows = plan if isinstance(plan, list) else []

    family_buckets: dict[str, dict[str, list[tuple[str, float, float]]]] = {}
    n_plan_rows = 0
    total_plan_usd = 0.0
    total_long_lev_usd = 0.0
    total_inverse_lev_usd = 0.0
    total_unlev_usd = 0.0

    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        usd = _f(row.get("alloc_usd"), 0.0)
        if usd <= 0:
            continue
        n_plan_rows += 1
        total_plan_usd += usd

        # Cross-family directional totals — driven by the taxonomy when
        # available, falling back to the input ``leverage_factor`` for
        # tickers the taxonomy doesn't cover (e.g. NVDU/SOXL-only single-
        # name leveraged ETFs that aren't in _PAIR_FAMILIES). The fallback
        # preserves directional intent without inventing a family slot.
        idx = _TICKER_INDEX.get(ticker)
        if idx is not None:
            fam_key, role, lev = idx
            bucket = family_buckets.setdefault(
                fam_key, {"long": [], "inverse": [], "core": []}
            )
            bucket[role].append((ticker, usd, lev))
            if role == "long":
                total_long_lev_usd += usd
            elif role == "inverse":
                total_inverse_lev_usd += usd
            else:
                total_unlev_usd += usd
        else:
            # Off-taxonomy ticker — use the plan's declared leverage_factor.
            lev_factor = _f(row.get("leverage_factor"), 1.0)
            if lev_factor > 1.0:
                total_long_lev_usd += usd
            elif lev_factor < 0.0:
                total_inverse_lev_usd += usd
            else:
                total_unlev_usd += usd

    if n_plan_rows == 0:
        return {
            "verdict": "NO_PLAN",
            "headline": "no plan rows — nothing to audit",
            "n_plan_rows": 0,
            "total_plan_usd": _z(0.0),
            "totals": {
                "long_leveraged_usd": _z(0.0),
                "inverse_leveraged_usd": _z(0.0),
                "unleveraged_usd": _z(0.0),
                "long_leveraged_pct": _z(0.0),
                "inverse_leveraged_pct": _z(0.0),
            },
            "family_conflicts": [],
            "n_family_conflicts": 0,
            "directional_hedge": False,
        }

    # ── Per-family conflict eval ─────────────────────────────────────
    family_conflicts: list[dict] = []
    any_carry_waste = False
    any_opposing_unlevered = False

    for fam_key, bucket in family_buckets.items():
        longs = bucket["long"]
        inverses = bucket["inverse"]
        cores = bucket["core"]
        has_long = bool(longs)
        has_inverse = bool(inverses)
        has_core = bool(cores)

        # A family contributes a conflict only if the plan has *opposing*
        # direction within it — leveraged-inverse alongside positive-side
        # (leveraged-long OR 1x-core). A family with only longs (e.g.
        # TQQQ alone) is not a conflict.
        if not (has_inverse and (has_long or has_core)):
            continue

        if has_long:
            any_carry_waste = True
            classification = "CARRY_WASTE"
        else:
            any_opposing_unlevered = True
            classification = "OPPOSING_UNLEVERED"

        long_notional = sum(usd for _t, usd, _l in longs)
        inverse_notional = sum(usd for _t, usd, _l in inverses)
        core_notional = sum(usd for _t, usd, _l in cores)
        # Signed-leverage deltas mirror ``inverse_pair_conflict``.
        long_delta = sum(usd * lev for _t, usd, lev in longs)
        core_delta = sum(usd * lev for _t, usd, lev in cores)
        positive_delta = long_delta + core_delta
        inverse_delta = sum(usd * lev for _t, usd, lev in inverses)
        cancelled_delta = min(abs(positive_delta), abs(inverse_delta))
        net_delta = positive_delta + inverse_delta
        gross_delta = abs(positive_delta) + abs(inverse_delta)
        severity = _severity(cancelled_delta, gross_delta)

        def _fmt(holdings: list[tuple[str, float, float]]) -> list[dict]:
            return [
                {"ticker": t, "alloc_usd": _z(u), "leverage": _z(l, 1)}
                for t, u, l in sorted(holdings, key=lambda r: -r[1])
            ]

        family_conflicts.append({
            "family": fam_key,
            "family_label": _PAIR_FAMILIES[fam_key]["label"],
            "classification": classification,
            "severity": severity,
            "long_holdings": _fmt(longs),
            "inverse_holdings": _fmt(inverses),
            "core_holdings": _fmt(cores),
            "long_notional_usd": _z(long_notional),
            "inverse_notional_usd": _z(inverse_notional),
            "core_notional_usd": _z(core_notional),
            "long_delta_usd": _z(positive_delta),
            "inverse_delta_usd": _z(inverse_delta),
            "cancelled_delta_usd": _z(cancelled_delta),
            "net_delta_usd": _z(net_delta),
        })

    # Worst-cancelled first so the operator reads the most pathological
    # family at the top.
    family_conflicts.sort(
        key=lambda c: _f(c.get("cancelled_delta_usd"), 0.0), reverse=True
    )

    # ── Aggregate directional hedge (cross-family) ───────────────────
    long_lev_pct = (
        100.0 * total_long_lev_usd / total_plan_usd if total_plan_usd > 0 else 0.0
    )
    inverse_lev_pct = (
        100.0 * total_inverse_lev_usd / total_plan_usd if total_plan_usd > 0 else 0.0
    )
    directional_hedge = (
        long_lev_pct >= DIRECTIONAL_MIN_PCT
        and inverse_lev_pct >= DIRECTIONAL_MIN_PCT
    )

    # ── Verdict ladder ───────────────────────────────────────────────
    if any_carry_waste:
        verdict = "CARRY_WASTE"
    elif any_opposing_unlevered:
        verdict = "OPPOSING_UNLEVERED"
    elif directional_hedge:
        verdict = "DIRECTIONAL_HEDGE"
    else:
        verdict = "CLEAN"

    # Headline: most-severe issue first, with the live numbers.
    if verdict == "CARRY_WASTE":
        worst = family_conflicts[0]
        headline = (
            "plan contains %s + %s in %s — both sides paying leverage decay"
            % (
                ", ".join(h["ticker"] for h in worst["long_holdings"]),
                ", ".join(h["ticker"] for h in worst["inverse_holdings"]),
                worst["family_label"],
            )
        )
    elif verdict == "OPPOSING_UNLEVERED":
        worst = family_conflicts[0]
        headline = (
            "plan contains %s + %s in %s — delta offset, leverage decay on inverse sleeve"
            % (
                ", ".join(
                    h["ticker"]
                    for h in (worst["long_holdings"] + worst["core_holdings"])
                ),
                ", ".join(h["ticker"] for h in worst["inverse_holdings"]),
                worst["family_label"],
            )
        )
    elif verdict == "DIRECTIONAL_HEDGE":
        headline = (
            "plan has %.1f%% long-leveraged + %.1f%% inverse-leveraged — "
            "cross-family hedge, double-decay carry"
            % (long_lev_pct, inverse_lev_pct)
        )
    else:
        headline = "no inverse-pair conflicts in plan"

    return {
        "verdict": verdict,
        "headline": headline,
        "n_plan_rows": n_plan_rows,
        "total_plan_usd": _z(total_plan_usd),
        "totals": {
            "long_leveraged_usd": _z(total_long_lev_usd),
            "inverse_leveraged_usd": _z(total_inverse_lev_usd),
            "unleveraged_usd": _z(total_unlev_usd),
            "long_leveraged_pct": _z(long_lev_pct),
            "inverse_leveraged_pct": _z(inverse_lev_pct),
        },
        "family_conflicts": family_conflicts,
        "n_family_conflicts": len(family_conflicts),
        "directional_hedge": directional_hedge,
        "constraints": {
            "directional_min_pct": _z(DIRECTIONAL_MIN_PCT),
        },
    }
