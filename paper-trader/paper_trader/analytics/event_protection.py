"""Actionable per-event trim and put-hedge sizing for held imminent-
earnings names.

``/api/implied-move`` reports the **market-priced 1σ move** on each held
name that reports inside the horizon. ``/api/event-readiness`` reports
the **bot's ability to react** before the print. ``/api/position-action-
brief`` reports a **qualitative** recommendation (``TRIM_BEFORE_EVENT``
/ ``HOLD_THROUGH_EVENT``). All three describe the *situation*; none of
them answers the **quantitative** question every discretionary trader
asks the morning of an earnings print:

    *"How many shares do I trim to cap my 1σ downside at X% of book —
    and what would the equivalent put hedge cost?"*

This builder fills that gap. It composes ``build_implied_move``'s rows
**verbatim** (single source of truth, AGENTS.md #10) so this endpoint,
``/api/implied-move`` and ``/api/earnings-shock`` can never disagree on
the per-event 1σ pct or dollar move. For each row whose ``current 1σ
book pct`` exceeds the configurable target it computes:

* the minimum **whole-share trim qty** such that the *remaining* 1σ
  book pct ≤ target, plus the exact (fractional) theoretical minimum
  for math reconciliation;
* the proceeds and post-trim 1σ numbers (a "before / after" the
  operator can act on without a calculator);
* the **ATM put hedge alternative** — contracts to cover the held qty
  at the chain's existing ATM strike (already in the implied-move row),
  the approximate premium cost from the ATM put mid, and the cost as
  % of book so the operator can compare "trim qty × ~7% downside" to
  "hedge cost × 100% downside cap".

State ladder (mirrors ``implied_move`` / ``earnings_shock`` precedent):

* ``NO_DATA``   — empty book / unpriceable / no total_value (the
  ``earnings_shock`` "no priced book" branch).
* ``NO_EVENTS`` — book is fine, calendar is fine, but no held name has
  an imminent print inside the horizon (calendar quiet).
* ``NO_BREACH`` — events exist and are priced but no row breaches the
  target — *nothing to do* (intentionally distinct from NO_EVENTS so
  the operator can tell "no event" from "event but already safe").
* ``OK``        — at least one breaching event with a sized plan.

Per-row state ladder (independent of the top-level):

* ``OK``         — implied σ available, breach computed, plan emitted.
* ``NO_IMPLIED`` — the implied-move row is ``NO_CHAIN`` / ``NO_QUOTES``
  (chain unavailable or thin); we can't size the plan honestly, so the
  row surfaces the event + current exposure but withholds trim/hedge
  numbers rather than fabricating them from a zero σ.
* ``WITHIN_TARGET`` — implied σ available, current 1σ book pct already
  ≤ target; no trim recommended.

Trim math (per row, with ``q``=current qty, ``p``=spot, ``σ``=implied
1σ as decimal, ``tv``=total_value, ``T``=target_max_1sigma_loss_pct):

* current 1σ book pct  = (q · p · σ) / tv · 100
* theoretical min trim = max(0, q − T · tv / (100 · p · σ))
* whole-share trim     = ceil(theoretical) clamped to [0, q]
* post-trim 1σ book pct = ((q − trim) · p · σ) / tv · 100

If theoretical > q the target is unreachable without a **full exit**;
the row carries ``trim.full_exit_required: true`` and the trim qty is
clamped to q (the operator gets the closest-attainable answer rather
than a NaN). Whole-share rounding can occasionally still leave the
post-trim figure marginally above target (e.g. target=5%, breach=5.1%,
ceil rounds up by ½ share for a tiny over-shoot); the ``trim.achieves_
target`` flag is the literal verdict against the rounded plan, not the
theoretical one — operator honesty over implied precision.

Hedge math (ATM put alternative, sized to the held qty):

* contracts_needed = ceil(q / 100)   — each listed put covers 100 shares
* approx_cost_usd  = contracts_needed · atm_put_mid · 100
* cost_book_pct    = approx_cost_usd / tv · 100
* max_protection_strike_usd = contracts_needed · 100 · atm_put_strike

The hedge premium is the desk-standard ATM-put-mid from the same chain
``implied_move`` already pulled; we never re-query yfinance here (the
implied-move endpoint owns the I/O — this builder is pure). The
``max_protection_strike_usd`` is the dollarized "floor" the put gives
the operator (exercise value if the underlying goes to zero) — not the
realized P&L of the hedge, which depends on the actual move; it is
surfaced because the trader's mental frame is "what's the worst
realized loss I can cap at?" Note: when the position is a fraction of
100 shares the ATM-put contract over-hedges (a 2-share NVDA position
with one 100-share put has 98 shares of *negative* gamma against the
underlying rallying). The ``hedge.note`` field flags this honestly.

Observational / advisory only — never gates Opus, never injected into
the decision prompt, no caps (invariants #2/#12 — the ``stress_
scenarios`` / ``implied_move`` precedent).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

DEFAULT_TARGET_MAX_1SIGMA_PCT = 5.0   # % of book — the desk-standard cap
DEFAULT_HORIZON_DAYS = 7.0
PUT_CONTRACT_MULTIPLIER = 100         # standard listed equity option multiplier


def _z(v, ndigits: int = 2):
    """Round; fold -0.0 → 0.0 (the ``implied_move._z`` precedent — same
    contract). A non-numeric / None input degrades to ``None``, never
    raises."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _safe_float(x):
    """Best-effort coercion to ``float`` with ``NaN`` rejected (the
    ``implied_move._safe_float`` precedent)."""
    try:
        if x is None:
            return None
        f = float(x)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _qty_by_ticker(positions: list[dict]) -> dict[str, float]:
    """Sum the *stock* qty per ticker (skip option positions — an
    earnings hedge sized against a stock position should not have its
    qty inflated by a coincidentally-held call/put on the same name)."""
    out: dict[str, float] = {}
    for p in positions or []:
        if not isinstance(p, dict):
            continue
        ptype = (p.get("type") or "stock").lower()
        if ptype in ("call", "put"):
            continue
        tk = (p.get("ticker") or "").upper() if isinstance(p.get("ticker"), str) else ""
        if not tk:
            continue
        q = _safe_float(p.get("qty"))
        if q is None or q <= 0:
            continue
        out[tk] = out.get(tk, 0.0) + q
    return out


def _compute_trim(
    qty: float,
    spot: float,
    sigma_pct: float,
    total_value: float,
    target_pct: float,
) -> dict:
    """Pure trim math. Returns the breach + sized plan against a target
    1σ book pct cap. ``sigma_pct`` is in percent (13.5 == 13.5%);
    ``target_pct`` is the cap in percent of total_value.

    Theoretical minimum trim (a continuous fractional answer) and the
    rounded whole-share trim (the executable answer) are both surfaced
    so the operator can reconcile the math without a calculator and a
    test can pin the rounding rule rather than the floating-point
    result."""
    sigma_dec = sigma_pct / 100.0
    current_1sigma_usd = qty * spot * sigma_dec
    current_1sigma_book_pct = current_1sigma_usd / total_value * 100.0
    breaches = current_1sigma_book_pct > target_pct

    plan = {
        "current_1sigma_loss_usd": _z(current_1sigma_usd),
        "current_1sigma_loss_book_pct": _z(current_1sigma_book_pct),
        "target_max_1sigma_loss_pct": _z(target_pct),
        "exceeds_target": bool(breaches),
    }

    if not breaches:
        plan["row_verdict"] = "WITHIN_TARGET"
        plan["trim"] = None
        return plan

    # Theoretical fractional minimum: ((q − t) · p · σ) / tv · 100 ≤ T
    # → t ≥ q − T · tv / (100 · p · σ). Clamp at 0.
    denom = 100.0 * spot * sigma_dec
    if denom <= 0:
        # Defensive: σ=0 (priced as no move) would divide by zero — we
        # already guarded above (breach requires current > target which
        # requires σ > 0 for non-trivial qty), but be explicit.
        plan["row_verdict"] = "NO_IMPLIED"
        plan["trim"] = None
        return plan

    theoretical_trim = qty - (target_pct * total_value) / denom
    theoretical_trim = max(0.0, theoretical_trim)
    full_exit_required = theoretical_trim > qty - 1e-9 and theoretical_trim >= qty
    # Whole-share executable trim: round UP so we strictly meet target
    # in the absence of full-exit. clamp to [0, q] so a target unreachable
    # without exit doesn't synthesize a phantom share.
    whole_trim = int(math.ceil(theoretical_trim))
    whole_trim = max(0, min(whole_trim, int(math.ceil(qty))))
    remaining_qty = max(0.0, qty - whole_trim)
    post_value = remaining_qty * spot
    post_1sigma_usd = post_value * sigma_dec
    post_1sigma_book_pct = post_1sigma_usd / total_value * 100.0
    achieves = post_1sigma_book_pct <= target_pct + 1e-9

    headline = (
        f"Trim {whole_trim} share{'s' if whole_trim != 1 else ''} "
        f"→ 1σ downside drops "
        f"from {current_1sigma_book_pct:.1f}% "
        f"to {post_1sigma_book_pct:.1f}% of book"
    )
    if full_exit_required:
        headline = (
            f"Full exit ({int(math.ceil(qty))} shares) required — "
            f"1σ downside cannot be capped at {target_pct:.1f}% "
            f"of book while holding any shares"
        )

    plan["trim"] = {
        "qty_to_trim_exact": _z(theoretical_trim, 4),
        "qty_to_trim": whole_trim,
        "proceeds_usd": _z(whole_trim * spot),
        "remaining_qty": _z(remaining_qty, 4),
        "post_trim_value_usd": _z(post_value),
        "post_trim_1sigma_loss_usd": _z(post_1sigma_usd),
        "post_trim_1sigma_loss_book_pct": _z(post_1sigma_book_pct),
        "achieves_target": bool(achieves),
        "full_exit_required": bool(full_exit_required),
        "headline": headline,
    }
    plan["row_verdict"] = "BREACHES_TARGET"
    return plan


def _compute_hedge(
    qty: float,
    total_value: float,
    atm_put_strike,
    atm_put_mid,
) -> dict | None:
    """Pure put-hedge math from the implied-move row's ATM put hint.

    Returns ``None`` (the calling row will carry ``hedge: None``) when
    either input is missing — we never fabricate a hedge cost from a
    half-priced chain. ``atm_put_mid == 0`` is also rejected since a
    zero-cost hedge would mis-frame the trim/hedge tradeoff."""
    strike = _safe_float(atm_put_strike)
    mid = _safe_float(atm_put_mid)
    if strike is None or mid is None or mid <= 0:
        return None
    contracts = max(1, int(math.ceil(qty / PUT_CONTRACT_MULTIPLIER)))
    approx_cost = contracts * mid * PUT_CONTRACT_MULTIPLIER
    cost_book_pct = (approx_cost / total_value * 100.0) if total_value > 0 else None
    max_protection = contracts * PUT_CONTRACT_MULTIPLIER * strike
    covered_shares = contracts * PUT_CONTRACT_MULTIPLIER
    over_hedge = covered_shares > qty + 1e-9

    note = (
        "ATM put hint from /api/implied-move; verify chain liquidity "
        "before executing"
    )
    if over_hedge:
        note += (
            f" — note: {contracts} contract(s) cover {covered_shares} shares "
            f"but position is only {qty:g} shares (net short underlying "
            "above strike, reduces upside)"
        )

    return {
        "atm_put_strike": _z(strike),
        "atm_put_mid": _z(mid),
        "contracts_needed": contracts,
        "approx_cost_usd": _z(approx_cost),
        "cost_book_pct": _z(cost_book_pct),
        "max_protection_strike_usd": _z(max_protection),
        "covered_shares": covered_shares,
        "over_hedges_position": bool(over_hedge),
        "note": note,
    }


def build_event_protection_plan(
    positions: list[dict],
    total_value: float,
    implied_move_result: dict | None,
    target_max_1sigma_loss_pct: float = DEFAULT_TARGET_MAX_1SIGMA_PCT,
    now: datetime | None = None,
) -> dict:
    """Pure: no I/O, never raises. ``positions`` is the
    ``store.open_positions()`` shape. ``implied_move_result`` is the
    dict from ``build_implied_move`` — its ``events`` list is the SSOT
    for held imminent earnings + the implied 1σ per row. ``target_max_
    1sigma_loss_pct`` is the operator-chosen cap (default 5% of book).

    The builder never re-derives implied σ — it relies on the upstream
    builder so any change in implied-move's math propagates here
    without a second source of truth."""
    now = now or datetime.now(timezone.utc)
    try:
        tv = float(total_value or 0.0)
    except (TypeError, ValueError):
        tv = 0.0

    try:
        target_pct = float(target_max_1sigma_loss_pct)
    except (TypeError, ValueError):
        target_pct = DEFAULT_TARGET_MAX_1SIGMA_PCT
    if target_pct <= 0:
        # An impossible cap — degrade honestly so the caller knows the
        # config is bad rather than silently treating it as "trim everything".
        target_pct = DEFAULT_TARGET_MAX_1SIGMA_PCT

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "target_max_1sigma_loss_pct": _z(target_pct),
        "horizon_days": None,
        "n_events": 0,
        "n_breaching": 0,
        "n_within_target": 0,
        "n_no_implied": 0,
        "events": [],
        "summary": None,
        "state": None,
    }

    qty_by = _qty_by_ticker(positions)
    if not qty_by or tv <= 0:
        base["state"] = "NO_DATA"
        base["summary"] = (
            "No priced stock positions — event-protection plan not applicable."
        )
        return base

    im_events: list[dict] = []
    if isinstance(implied_move_result, dict):
        try:
            im_events = list(implied_move_result.get("events") or [])
        except (TypeError, AttributeError):
            im_events = []
        try:
            base["horizon_days"] = _z(implied_move_result.get("horizon_days"))
        except (TypeError, AttributeError):
            pass

    if not im_events:
        base["state"] = "NO_EVENTS"
        base["summary"] = (
            "No held name reports inside the implied-move horizon "
            "— nothing to size."
        )
        return base

    rows: list[dict] = []
    for ev in im_events:
        if not isinstance(ev, dict):
            continue
        tk = (ev.get("ticker") or "").upper() if isinstance(ev.get("ticker"), str) else ""
        if not tk:
            continue
        qty = qty_by.get(tk)
        if qty is None or qty <= 0:
            # implied-move shouldn't surface unheld names, but defensive.
            continue
        spot = _safe_float(ev.get("spot"))
        sigma_pct = _safe_float(ev.get("implied_one_sigma_pct"))
        value = _safe_float(ev.get("current_value_usd"))
        weight_pct = _safe_float(ev.get("weight_pct"))

        row = {
            "ticker": tk,
            "days_away": _z(ev.get("days_to_earnings")),
            "earnings_date": ev.get("earnings_date"),
            "tier": ev.get("tier"),
            "current_qty": _z(qty, 4),
            "current_value_usd": _z(value if value is not None else (qty * spot if spot else 0.0)),
            "current_weight_pct": _z(weight_pct),
            "spot": _z(spot),
            "implied_1sigma_pct": _z(sigma_pct),
            "implied_move_pct": _z(ev.get("implied_move_pct")),
            "trim": None,
            "hedge": None,
            "row_verdict": None,
        }

        # If implied σ is missing, surface the awareness row but withhold
        # the trim/hedge plan rather than zeroing the math.
        if spot is None or sigma_pct is None or sigma_pct <= 0:
            row["row_verdict"] = "NO_IMPLIED"
            row["state"] = ev.get("state", "NO_IMPLIED")
            row["headline"] = (
                f"{tk} reports in {ev.get('days_to_earnings')}d — implied σ "
                "unavailable (chain unavailable or thin), trim/hedge "
                "sizing withheld."
            )
            rows.append(row)
            continue

        plan = _compute_trim(qty, spot, sigma_pct, tv, target_pct)
        row["current_1sigma_loss_usd"] = plan.get("current_1sigma_loss_usd")
        row["current_1sigma_loss_book_pct"] = plan.get("current_1sigma_loss_book_pct")
        row["exceeds_target"] = plan.get("exceeds_target")
        row["row_verdict"] = plan.get("row_verdict")
        row["state"] = "OK"
        if plan.get("trim") is not None:
            row["trim"] = plan["trim"]

        row["hedge"] = _compute_hedge(
            qty=qty,
            total_value=tv,
            atm_put_strike=ev.get("atm_put_strike"),
            atm_put_mid=ev.get("put_mid"),
        )
        rows.append(row)

    rows.sort(key=lambda r: (r.get("days_away") if r.get("days_away") is not None else 1e9))
    base["events"] = rows
    base["n_events"] = len(rows)
    base["n_breaching"] = sum(
        1 for r in rows if r.get("row_verdict") == "BREACHES_TARGET"
    )
    base["n_within_target"] = sum(
        1 for r in rows if r.get("row_verdict") == "WITHIN_TARGET"
    )
    base["n_no_implied"] = sum(
        1 for r in rows if r.get("row_verdict") == "NO_IMPLIED"
    )

    if base["n_breaching"] == 0:
        if base["n_no_implied"] == base["n_events"]:
            base["state"] = "NO_BREACH"
            base["summary"] = (
                f"{base['n_events']} held event(s) inside horizon, "
                "implied σ unavailable on all — sizing withheld."
            )
        else:
            base["state"] = "NO_BREACH"
            base["summary"] = (
                f"{base['n_events']} held event(s) inside horizon; "
                f"all within {target_pct:.1f}% 1σ target — no trim required."
            )
        return base

    base["state"] = "OK"
    worst = max(
        (r for r in rows if r.get("row_verdict") == "BREACHES_TARGET"),
        key=lambda r: (r.get("current_1sigma_loss_book_pct") or 0.0),
    )
    parts = [
        f"{base['n_breaching']} of {base['n_events']} held event(s) breach "
        f"{target_pct:.1f}% 1σ target."
    ]
    trim_plan = worst.get("trim") or {}
    if trim_plan.get("headline"):
        parts.append(trim_plan["headline"] + ".")
    hedge_plan = worst.get("hedge") or {}
    if hedge_plan.get("approx_cost_usd") is not None:
        parts.append(
            f"Hedge alternative: {hedge_plan.get('contracts_needed')}× "
            f"{worst['ticker']} ${hedge_plan['atm_put_strike']} put @ "
            f"${hedge_plan['atm_put_mid']} ≈ ${hedge_plan['approx_cost_usd']} "
            f"({hedge_plan.get('cost_book_pct')}% of book)."
        )
    base["summary"] = " ".join(parts)
    return base
