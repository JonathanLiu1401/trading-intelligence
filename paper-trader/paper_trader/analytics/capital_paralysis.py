"""Capital paralysis & unlock — the trap *and* the way out, in one view.

Observed live (2026-05-16): $6.23 cash of a $972 book (99.4% deployed) across
two underwater names, the loop HOLD/NO_DECISION every cycle, and
``/api/decision-drought`` reporting ``BLEEDING`` (−2.21% alpha lost across the
involuntary parse-failure droughts). Three endpoints already see a *piece* of
this — ``/api/liquidity`` (the trap: no dry powder), ``/api/decision-drought``
(the cost: alpha bled while stuck), ``/api/suggestions`` (ideas it can't fund)
— but nothing connects **trap → cost → unlock**. A desk doesn't just want to
hear "you're pinned"; it wants "sell *this* and you can act again, and here is
what staying pinned has already cost."

``build_capital_paralysis`` composes the two existing pure-core builders
(``build_liquidity`` + ``build_decision_drought`` — single source of truth, no
re-derived metrics) and adds the genuinely new synthesis: the **unlock
ladder**. Positions are ranked in desk cut-priority (biggest loser first — a
sale that both frees cash *and* stops the bleed), and for each rung we compute
the cash it frees, the deployed-% after, and whether that sale alone restores
the ability to act on a fresh signal.

This is a *diagnostic / advisory* panel only. It never gates Opus and adds no
position caps — it respects the "no hard risk limits, Opus has full autonomy"
invariant (paper-trader AGENTS.md #2) exactly as ``/api/liquidity`` and
``/api/risk`` do. Pure: feed it the store reads; ``now`` is injectable for
deterministic tests.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .liquidity import build_liquidity
from .decision_drought import build_decision_drought


def _compact_drought(d: dict | None) -> dict | None:
    """Trim a drought record to the fields the paralysis panel surfaces."""
    if not d:
        return None
    return {
        "kind": d.get("kind"),
        "start": d.get("start"),
        "end": d.get("end"),
        "duration_hours": d.get("duration_hours"),
        "n_cycles": d.get("n_cycles"),
        "no_decision_pct": d.get("no_decision_pct"),
        "alpha_pct": d.get("alpha_pct"),
    }


def build_capital_paralysis(portfolio: dict,
                            positions: list[dict],
                            trades: list[dict],
                            decisions: list[dict],
                            equity_curve: list[dict],
                            now: datetime | None = None) -> dict:
    """Trap + cost + unlock ladder. Pure — composes liquidity & drought."""
    now = now or datetime.now(timezone.utc)

    liq = build_liquidity(portfolio, positions, trades, now=now)
    dr = build_decision_drought(decisions or [], equity_curve or [], now=now)

    total_value = liq["total_value"]
    cash = liq["cash"]
    can_act = liq["can_act_on_signal"]
    detail = liq["positions"]  # already sorted by market_value desc
    n_positions = liq["n_positions"]

    # Minimum cash to act on a fresh signal — mirrors liquidity's can_act rule
    # (cash ≥ $1 AND ≥ 1% of book). Expressed as a single USD threshold so the
    # ladder can answer "does selling X put us back above the line?".
    min_actionable_usd = round(max(1.0, total_value * 0.01), 2) if total_value > 0 else 1.0

    # Cut priority: losers before winners (a loser sale frees cash AND stops
    # the bleed), then larger market value first (frees more, de-concentrates
    # more). Stable within ties via the original value-desc order.
    ordered = sorted(
        detail,
        key=lambda p: (p["unrealized_pl"] >= 0, -p["market_value"]),
    )

    ladder: list[dict] = []
    cumulative = 0.0
    recommended = None
    for p in ordered:
        cumulative += p["market_value"]
        cash_after = round(cash + cumulative, 2)
        deployed_after = (round(max(0.0, 100.0 - cash_after / total_value * 100.0), 2)
                          if total_value > 0 else 0.0)
        # "Sell *this one alone*" — the single-position unlock the desk cares
        # about (not the cumulative running total, which is for laddering out).
        solo_cash = round(cash + p["market_value"], 2)
        restores_alone = (solo_cash >= min_actionable_usd and solo_cash >= 1.0
                          and not can_act)
        rung = {
            "ticker": p["ticker"],
            "type": p["type"],
            "weight_pct": p["weight_pct"],
            "unrealized_pl": p["unrealized_pl"],
            "pl_pct": p["pl_pct"],
            "frees_usd": p["market_value"],
            "cash_if_sold_alone": solo_cash,
            "cumulative_freed_usd": round(cumulative, 2),
            "cash_after_cumulative": cash_after,
            "deployed_pct_after_cumulative": deployed_after,
            "restores_action_alone": restores_alone,
        }
        ladder.append(rung)
        if recommended is None and restores_alone:
            recommended = {
                "ticker": p["ticker"],
                "frees_usd": p["market_value"],
                "pl_pct": p["pl_pct"],
                "reason": (
                    f"largest underwater name ({p['pl_pct']:+.1f}%) — selling "
                    f"it frees ${p['market_value']:.2f} and restores the "
                    f"ability to act on a fresh signal"
                    if p["unrealized_pl"] < 0 else
                    f"selling it frees ${p['market_value']:.2f} and restores "
                    f"the ability to act on a fresh signal"),
            }

    bleed = dr.get("involuntary_alpha_bleed_pct") or 0.0
    paralysis = {
        "verdict": dr.get("verdict"),
        "verdict_reason": dr.get("verdict_reason"),
        "involuntary_alpha_bleed_pct": bleed,
        "n_paralysis_droughts": dr.get("n_paralysis_droughts", 0),
        "current_drought": _compact_drought(dr.get("current_drought")),
        "worst_alpha_drought": _compact_drought(dr.get("worst_alpha_drought")),
    }

    # Cycles the bot has gone without a FILL while pinned — the live "how long
    # have we been stuck" number, taken from the ongoing drought.
    cur = dr.get("current_drought") or {}
    cycles_since_fill = cur.get("n_cycles", 0) if cur else 0

    if total_value <= 0 and n_positions == 0:
        state = "NO_DATA"
    elif can_act:
        state = "FREE"
    elif n_positions == 0:
        # No cash AND nothing to sell — a genuinely stuck, empty book.
        state = "EMPTY"
    else:
        state = "PINNED"

    if state == "PINNED" and recommended:
        bleed_clause = (
            f" Staying pinned has bled {bleed:.2f}% alpha across "
            f"{paralysis['n_paralysis_droughts']} paralysis drought(s)."
            if bleed < 0 else "")
        headline = (
            f"PINNED — ${cash:.2f} cash ({liq['cash_pct']:.1f}%) across "
            f"{n_positions} position(s); selling {recommended['ticker']} "
            f"frees ${recommended['frees_usd']:.2f} → can act again."
            + bleed_clause)
    elif state == "FREE":
        headline = (
            f"FREE — ${cash:.2f} cash ({liq['cash_pct']:.1f}%) available; "
            f"the book can act on a new signal without selling.")
    elif state == "EMPTY":
        headline = "EMPTY — no cash and no open positions to free capital from."
    elif state == "PINNED":
        headline = (
            f"PINNED — ${cash:.2f} cash across {n_positions} position(s); "
            f"no single sale modeled (no positions detail).")
    else:
        headline = "No portfolio data."

    flags = list(liq.get("flags") or [])
    if bleed < 0 and paralysis["n_paralysis_droughts"]:
        flags.append(
            f"inaction has cost {bleed:.2f}% alpha "
            f"({paralysis['n_paralysis_droughts']} paralysis drought(s))")
    if state == "PINNED" and recommended:
        flags.append(
            f"unlock: sell {recommended['ticker']} → free "
            f"${recommended['frees_usd']:.2f}")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "headline": headline,
        "cash": cash,
        "total_value": total_value,
        "cash_pct": liq["cash_pct"],
        "deployed_pct": liq["deployed_pct"],
        "can_act_on_signal": can_act,
        "min_actionable_usd": min_actionable_usd,
        "n_positions": n_positions,
        "cycles_since_last_fill": cycles_since_fill,
        "liquidity_status": liq["status"],
        "recommended_unlock": recommended,
        "unlock_ladder": ladder,
        "paralysis": paralysis,
        "flags": flags,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json
    from paper_trader.store import get_store
    s = get_store()
    rep = build_capital_paralysis(
        s.get_portfolio(),
        s.open_positions(),
        s.recent_trades(200),
        s.recent_decisions(limit=3000),
        s.equity_curve(limit=5000),
    )
    print(json.dumps(rep, indent=2, default=str))
