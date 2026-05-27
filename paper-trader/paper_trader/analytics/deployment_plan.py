"""Multi-trade capital-deployment planner.

The desk already has:

  * ``/api/scorer-opportunities`` — ML-ranked watchlist names with a
    ``pred_5d_return_pct`` and a coarse ``verdict`` (STRONG_HOLD / HOLD /
    NEUTRAL / TRIM / EXIT — see ``dashboard._scorer_verdict``);
  * ``/api/kelly-sizing`` — a single-position half-Kelly target derived
    from realised win-rate + payoff;
  * ``/api/concentration-cap`` — per-name cap policy (default 25%);
  * ``/api/leverage-exposure`` (new) — factor breakdown of book + slate.

What's been missing is the bridge between them: a concrete *multi-trade*
allocation plan that says "given $X cash, deploy $A1 into T1, $A2 into
T2, … under these caps, and here is what the book looks like after."
``/api/game-plan`` only consumes intern-driven suggestions (the
hand-curated WATCH list) — it ignores the scorer's own ranked opportunity
set entirely. That's the gap this fills.

The planner is *advisory*, *pure*, and *never raises*. It does not
execute trades, does not touch ``store``, and surfaces every constraint
in the response so the operator can re-tune by hand.

Algorithm (deliberately simple — sized to a $1k-$10k book):

  1. Filter the slate: drop already-held names, drop names whose verdict
     isn't STRONG_HOLD / HOLD, drop names below ``min_pred_pct``.
  2. Rank survivors by ``pred_5d_return_pct`` descending.
  3. Compute the deployable budget = ``cash_usd * (1 - reserve_pct/100)``.
  4. Use the half-Kelly target as the *initial* per-name allocation
     (clamped to ``per_name_cap_pct`` of total_value and capped to
     remaining budget). For each candidate, greedy-allocate while
     respecting per-sector and per-leverage caps.
  5. Stop when budget is exhausted, candidates exhausted, or the next
     candidate's allocation would fall below ``min_alloc_usd``.

The verdict ladder:

  * NO_OPPORTUNITIES — slate yielded zero survivors after filtering;
  * INSUFFICIENT_CASH — deployable budget < ``min_alloc_usd``;
  * READY — at least one trade allocated, no gating constraint hit;
  * GATED — survivors exist but per-sector / per-leverage caps blocked
    every remaining candidate before budget was exhausted.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .leverage_exposure import LEVERAGE_FACTOR, classify as classify_leverage
from .sector_exposure import SECTOR_MAP

# Verdict labels we treat as *buy-eligible*. ``dashboard._scorer_verdict``
# emits STRONG_HOLD at pred>=3pp and HOLD at pred>=1pp; both are upward
# directional predictions despite the "HOLD" wording. NEUTRAL / TRIM /
# EXIT are never buy-eligible from this planner.
BUY_VERDICTS = frozenset({"STRONG_HOLD", "HOLD"})

#: Default policy. Each parameter is overridable per-request via the
#: ``/api/deployment-plan`` query string.
DEFAULT_RESERVE_CASH_PCT = 10.0    # always hold this % of cash back
DEFAULT_PER_NAME_CAP_PCT = 25.0    # mirrors concentration_cap default
DEFAULT_PER_SECTOR_CAP_PCT = 40.0  # soft second-order cap
DEFAULT_LEVERAGED_CAP_PCT = 30.0   # combined +/- leveraged-ETF cap
DEFAULT_KELLY_PCT = 26.0           # half-Kelly anchor (overridden by live)
DEFAULT_MIN_PRED_PCT = 1.0         # only deploy on pred_5d >= 1%
DEFAULT_MIN_ALLOC_USD = 20.0       # don't open positions tinier than $20
DEFAULT_MAX_TRADES = 8             # don't fan the plan into noise

# Wide guard-rails to keep the route from melting with absurd inputs.
# 100% reserve = "don't deploy"; 0% per-name cap = "block every trade".
MIN_RESERVE_CASH_PCT = 0.0
MAX_RESERVE_CASH_PCT = 100.0
MIN_PER_NAME_CAP_PCT = 1.0
MAX_PER_NAME_CAP_PCT = 100.0
MIN_PRED_PCT_FLOOR = 0.0
MAX_PRED_PCT_FLOOR = 50.0


def _utcnow_iso(now: datetime | None) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.isoformat(timespec="seconds")


def _z(v, ndigits: int = 2):
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _f(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker.upper(), "other")


def _held_set(positions: list[dict] | None) -> set[str]:
    out: set[str] = set()
    for p in positions or []:
        sym = p.get("ticker")
        if sym:
            out.add(str(sym).upper())
    return out


def build_deployment_plan(
    opportunities: list[dict] | None,
    positions: list[dict] | None,
    cash_usd: float | None,
    total_value: float | None,
    kelly_pct: float | None = None,
    *,
    reserve_cash_pct: float = DEFAULT_RESERVE_CASH_PCT,
    per_name_cap_pct: float = DEFAULT_PER_NAME_CAP_PCT,
    per_sector_cap_pct: float = DEFAULT_PER_SECTOR_CAP_PCT,
    leveraged_cap_pct: float = DEFAULT_LEVERAGED_CAP_PCT,
    min_pred_pct: float = DEFAULT_MIN_PRED_PCT,
    min_alloc_usd: float = DEFAULT_MIN_ALLOC_USD,
    max_trades: int = DEFAULT_MAX_TRADES,
    now: datetime | None = None,
) -> dict:
    """Pure: compose a deployable trade list. See module docstring for
    the algorithm and verdict ladder.

    All percent inputs are clamped to sane bands; the response echoes
    the *effective* policy under ``constraints`` so the caller can verify
    what was applied.
    """
    # ── Sanitise inputs ────────────────────────────────────────────────
    cash = max(_f(cash_usd, 0.0), 0.0)
    tv = max(_f(total_value, 0.0), 0.0)
    reserve_pct = _clamp(_f(reserve_cash_pct, DEFAULT_RESERVE_CASH_PCT),
                         MIN_RESERVE_CASH_PCT, MAX_RESERVE_CASH_PCT)
    name_cap = _clamp(_f(per_name_cap_pct, DEFAULT_PER_NAME_CAP_PCT),
                      MIN_PER_NAME_CAP_PCT, MAX_PER_NAME_CAP_PCT)
    sector_cap = _clamp(_f(per_sector_cap_pct, DEFAULT_PER_SECTOR_CAP_PCT),
                        MIN_PER_NAME_CAP_PCT, MAX_PER_NAME_CAP_PCT)
    lev_cap = _clamp(_f(leveraged_cap_pct, DEFAULT_LEVERAGED_CAP_PCT),
                     MIN_PER_NAME_CAP_PCT, MAX_PER_NAME_CAP_PCT)
    pred_floor = _clamp(_f(min_pred_pct, DEFAULT_MIN_PRED_PCT),
                        MIN_PRED_PCT_FLOOR, MAX_PRED_PCT_FLOOR)
    min_alloc = max(_f(min_alloc_usd, DEFAULT_MIN_ALLOC_USD), 0.0)
    try:
        max_trades_i = max(int(max_trades), 1)
    except (TypeError, ValueError):
        max_trades_i = DEFAULT_MAX_TRADES
    kelly = _f(kelly_pct, DEFAULT_KELLY_PCT)
    if kelly <= 0.0:
        kelly = DEFAULT_KELLY_PCT

    effective = {
        "reserve_cash_pct": _z(reserve_pct),
        "per_name_cap_pct": _z(name_cap),
        "per_sector_cap_pct": _z(sector_cap),
        "leveraged_cap_pct": _z(lev_cap),
        "min_pred_pct": _z(pred_floor),
        "min_alloc_usd": _z(min_alloc),
        "max_trades": max_trades_i,
        "kelly_anchor_pct": _z(kelly),
    }

    held = _held_set(positions)
    # ── Existing book exposure (the starting point for the caps) ──────
    held_sector_usd: dict[str, float] = {}
    held_leveraged_usd = 0.0
    for p in positions or []:
        sym = str(p.get("ticker") or "").upper()
        if not sym:
            continue
        mv = p.get("market_value")
        if mv is not None:
            try:
                usd = float(mv)
            except (TypeError, ValueError):
                usd = 0.0
        else:
            try:
                qty = float(p.get("qty") or 0)
                price = _f(p.get("current_price") or p.get("avg_cost"), 0.0)
                mult = 100 if (p.get("type") in ("call", "put")) else 1
                usd = qty * price * mult
            except (TypeError, ValueError):
                usd = 0.0
        sec = _sector(sym)
        held_sector_usd[sec] = held_sector_usd.get(sec, 0.0) + usd
        if abs(LEVERAGE_FACTOR.get(sym, 1)) > 1:
            held_leveraged_usd += usd

    # ── Filter the slate ──────────────────────────────────────────────
    survivors: list[dict] = []
    skipped: list[dict] = []
    for o in opportunities or []:
        sym = str(o.get("ticker") or "").upper()
        if not sym:
            continue
        verdict = str(o.get("verdict") or "").upper()
        pred = _f(o.get("pred_5d_return_pct"), 0.0)
        if sym in held:
            skipped.append({"ticker": sym, "reason": "already held"})
            continue
        if verdict not in BUY_VERDICTS:
            skipped.append({"ticker": sym,
                            "reason": f"verdict {verdict or 'UNKNOWN'} not buy-eligible"})
            continue
        if pred < pred_floor:
            skipped.append({"ticker": sym,
                            "reason": f"pred {pred:+.2f}% below floor {pred_floor:.2f}%"})
            continue
        survivors.append({**o, "ticker": sym,
                          "verdict": verdict,
                          "pred_5d_return_pct": pred,
                          "sector": _sector(sym),
                          "leverage_factor": LEVERAGE_FACTOR.get(sym, 1)})
    survivors.sort(key=lambda r: r["pred_5d_return_pct"], reverse=True)

    # ── Budget ────────────────────────────────────────────────────────
    deployable = max(cash * (1.0 - reserve_pct / 100.0), 0.0)
    # The per-name dollar cap derives from total_value (book size),
    # falling back to deployable if total_value is unset (e.g. a brand
    # new account where the dashboard's running balance hasn't refreshed
    # yet — pure-fn callers in tests don't set total_value).
    book_size_for_cap = tv if tv > 0 else (cash if cash > 0 else 0.0)
    per_name_cap_usd = book_size_for_cap * name_cap / 100.0
    per_sector_cap_usd = book_size_for_cap * sector_cap / 100.0
    lev_cap_usd = book_size_for_cap * lev_cap / 100.0
    kelly_target_usd = book_size_for_cap * kelly / 100.0

    plan: list[dict] = []
    rejected_by_constraint: list[dict] = []
    sector_committed = dict(held_sector_usd)
    leveraged_committed = held_leveraged_usd
    remaining = deployable
    gated = False

    if not survivors:
        verdict = "NO_OPPORTUNITIES"
        headline = "Slate is empty — no buy-eligible opportunities after filtering."
    elif deployable < min_alloc:
        verdict = "INSUFFICIENT_CASH"
        headline = (f"Deployable ${deployable:.2f} below min alloc "
                    f"${min_alloc:.2f} (reserve {reserve_pct:.0f}% held).")
    else:
        verdict = "PENDING"
        headline = ""
        for cand in survivors:
            if len(plan) >= max_trades_i:
                break
            if remaining < min_alloc:
                break
            sym = cand["ticker"]
            sec = cand["sector"]
            lev = cand["leverage_factor"]
            is_lev = abs(lev) > 1
            # Initial target = Kelly anchor, scaled lightly by pred edge
            # over the floor: a +12% pred gets a slightly bigger slice
            # than a +2% pred. Cap to per-name $.
            edge_mult = 1.0
            try:
                edge_mult = max(0.5, min(2.0,
                    1.0 + 0.05 * (cand["pred_5d_return_pct"] - DEFAULT_MIN_PRED_PCT)))
            except Exception:
                edge_mult = 1.0
            target = min(kelly_target_usd * edge_mult, per_name_cap_usd, remaining)
            if target < min_alloc:
                rejected_by_constraint.append({
                    "ticker": sym,
                    "reason": f"target ${target:.2f} < min alloc ${min_alloc:.2f}",
                })
                continue
            sec_avail = max(per_sector_cap_usd - sector_committed.get(sec, 0.0), 0.0)
            if sec_avail < min_alloc:
                rejected_by_constraint.append({
                    "ticker": sym,
                    "reason": f"sector '{sec}' cap reached ({per_sector_cap_usd:.0f} cap)",
                })
                gated = True
                continue
            target = min(target, sec_avail)
            if is_lev:
                lev_avail = max(lev_cap_usd - leveraged_committed, 0.0)
                if lev_avail < min_alloc:
                    rejected_by_constraint.append({
                        "ticker": sym,
                        "reason": (f"leveraged cap reached ({lev_cap:.0f}% / "
                                   f"${lev_cap_usd:.0f})"),
                    })
                    gated = True
                    continue
                target = min(target, lev_avail)
            if target < min_alloc:
                rejected_by_constraint.append({
                    "ticker": sym,
                    "reason": "post-cap target below min alloc",
                })
                continue
            alloc = round(target, 2)
            plan.append({
                "ticker": sym,
                "alloc_usd": alloc,
                "alloc_pct_of_book": _z(alloc / book_size_for_cap * 100)
                                       if book_size_for_cap > 0 else None,
                "pred_5d_return_pct": _z(cand["pred_5d_return_pct"]),
                "scorer_verdict": cand["verdict"],
                "sector": sec,
                "leverage_factor": lev,
                "is_leveraged": is_lev,
                "rationale": _rationale(cand, alloc, kelly_target_usd, per_name_cap_usd),
            })
            remaining -= alloc
            sector_committed[sec] = sector_committed.get(sec, 0.0) + alloc
            if is_lev:
                leveraged_committed += alloc
        if not plan:
            verdict = "GATED"
            headline = (f"{len(survivors)} candidate(s) passed the filter but "
                        f"every one was blocked by caps "
                        f"(per-name / sector / leverage).")
        else:
            verdict = "GATED" if gated and len(plan) < len(survivors) else "READY"
            blended = (sum(t["pred_5d_return_pct"] * t["alloc_usd"] for t in plan)
                       / max(sum(t["alloc_usd"] for t in plan), 1e-9))
            deployed = sum(t["alloc_usd"] for t in plan)
            headline = (f"Deploy ${deployed:.0f} across {len(plan)} name(s) "
                        f"({deployed / max(cash, 1e-9) * 100:.0f}% of cash); "
                        f"blended pred 5d {blended:+.2f}%.")

    # ── Implied post-deployment book ──────────────────────────────────
    deployed = sum(t["alloc_usd"] for t in plan)
    post_cash = max(cash - deployed, 0.0)
    post_total = tv if tv > 0 else (cash if cash > 0 else 0.0)
    # If positions already exist, post_total stays the same (we're
    # redeploying cash into named positions — total_value is conserved
    # because the dollars stay in the book just in a different form).
    if post_total <= 0:
        post_total = deployed + post_cash
    top1_pct = None
    top3_pct = None
    if plan:
        sorted_alloc = sorted(plan, key=lambda t: t["alloc_usd"], reverse=True)
        if post_total > 0:
            top1_pct = _z(sorted_alloc[0]["alloc_usd"] / post_total * 100)
            top3_pct = _z(sum(t["alloc_usd"] for t in sorted_alloc[:3])
                          / post_total * 100)
    blended_pred = None
    if plan:
        blended_pred = _z(sum(t["pred_5d_return_pct"] * t["alloc_usd"] for t in plan)
                          / max(sum(t["alloc_usd"] for t in plan), 1e-9))
    implied_lev_pct = None
    if post_total > 0:
        new_lev_usd = sum(t["alloc_usd"] for t in plan if t["is_leveraged"])
        implied_lev_pct = _z((held_leveraged_usd + new_lev_usd) / post_total * 100)
    implied_book = {
        "post_cash_usd": _z(post_cash),
        "post_cash_pct_of_book": _z(post_cash / post_total * 100) if post_total > 0 else None,
        "post_n_positions": len(held) + len(plan),
        "post_top1_pct": top1_pct,
        "post_top3_pct": top3_pct,
        "post_leveraged_pct": implied_lev_pct,
        "blended_pred_5d_return_pct": blended_pred,
    }

    return {
        "as_of": _utcnow_iso(now),
        "verdict": verdict,
        "headline": headline,
        "cash_available_usd": _z(cash),
        "deployable_usd": _z(deployable),
        "deployed_usd": _z(deployed),
        "n_plan": len(plan),
        "plan": plan,
        "skipped": skipped,
        "rejected_by_constraint": rejected_by_constraint,
        "constraints": effective,
        "implied_book": implied_book,
    }


def _rationale(cand: dict, alloc_usd: float,
               kelly_target_usd: float, per_name_cap_usd: float) -> str:
    sym = cand["ticker"]
    pred = cand["pred_5d_return_pct"]
    verdict = cand["verdict"]
    parts = [f"scorer {verdict} pred {pred:+.2f}%"]
    if alloc_usd >= per_name_cap_usd - 0.01:
        parts.append(f"capped at per-name cap ${per_name_cap_usd:.0f}")
    elif alloc_usd >= kelly_target_usd * 0.99:
        parts.append(f"sized at half-Kelly target ${kelly_target_usd:.0f}")
    if abs(cand.get("leverage_factor", 1)) > 1:
        parts.append(f"{cand['leverage_factor']:+d}x leveraged ETF")
    return f"{sym}: " + "; ".join(parts)
