"""Deployable-cash awareness, fed into the live Opus decision prompt.

The behavioural mirrors (``self_review`` / ``track_record`` / ``risk_mirror``)
and the forward ``event_calendar`` block all reach the decision prompt. The
single biggest *operational* fact the prompt still omitted is the one a
discretionary desk checks before every order: **how much can I actually
deploy, and if I'm pinned, what unlocks me?**

This is the #2 documented live pathology (paper-trader AGENTS.md review
pass #14, finding #4; ``analytics/capital_paralysis`` docstring): a $972 book
with ~$18 free cash across two underwater names. The prompt showed Opus only
a raw ``cash: $18.49`` line and a scattered WATCHLIST-PRICES list — it had to
mentally derive that *no* whole-share entry is fundable and that freeing cash
means selling MU/LITE. ``/api/capital-paralysis`` already synthesises this on
the **dashboard**, but the *decision engine itself never saw it* — exactly the
gap ``event_calendar`` was built to close, one dimension over.

This block is the **lean, prompt-facing complement** to the heavy dashboard
``capital_paralysis`` (which composes ``build_liquidity`` +
``build_decision_drought`` + the unlock ladder). It deliberately does **no**
extra store reads and **no** network: it is pure arithmetic over the
already-marked ``snapshot`` and the already-fetched ``watch_prices``
``decide()`` holds — the ``risk_mirror`` hot-path discipline (a per-position
yfinance call on the live cycle is a documented latency/flake hazard).

**Observational, never prescriptive.** Same contract as
``risk_mirror`` / ``event_calendar`` (AGENTS.md #2/#12): it states facts
(free cash, deployed %, affordable whole-share counts, which single position
frees the most cash) and reaffirms full autonomy in its preamble. It issues
no directive, imposes no cap, and never gates a trade — ``_execute`` still
runs the only real cash check.

Pure and deterministic (no clock, no IO). Never raises — the ``_safe``
contract (the caller in ``decide()`` also wraps it).
"""
from __future__ import annotations

_PREAMBLE = (
    "BUYING POWER (what your free cash can actually fund right now — facts "
    "for sizing awareness only, NOT a directive or limit; you retain "
    "complete autonomy over the next decision):"
)

# Cap the affordable-names list so the block stays one short prompt section.
_MAX_AFFORDABLE_NAMES = 6


def _f(x, default: float = 0.0) -> float:
    """Best-effort float coercion — a garbage cell degrades to ``default``,
    never raises (the _safe contract)."""
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _position_mark_value(p: dict) -> float:
    """Current mark value of an open position in dollars.

    Prefers the enriched ``market_value`` ``_mark_to_market`` already computed
    (it bakes in the option ×100 multiplier — never re-derive it). Falls back
    to ``current_price``×``qty``×(100 for options) only when the enriched key
    is absent, so a plain ``open_positions()`` row still values sanely."""
    mv = p.get("market_value")
    if isinstance(mv, (int, float)):
        return float(mv)
    mult = 100 if p.get("type") in ("call", "put") else 1
    px = _f(p.get("current_price")) or _f(p.get("avg_cost"))
    return px * _f(p.get("qty")) * mult


def build_buying_power(snapshot: dict, watch_prices: dict,
                       names_in_play) -> dict:
    """Compose the deployable-cash awareness block.

    ``snapshot`` — the ``strategy._portfolio_snapshot`` dict (``cash``,
    ``total_value``, enriched ``positions``). ``watch_prices`` — the
    ``market.get_prices(WATCHLIST)`` dict ``decide()`` already fetched.
    ``names_in_play`` — the ``strategy._names_in_play`` set so the
    affordability sizing is against the same "what matters this cycle"
    universe the quant / track-record blocks use (a lean, relevant list,
    not the full 50-name watchlist).

    Returns ``{state, summary, prompt_block, cash, deployed_pct,
    affordable, cheapest_name, cheapest_price, unlock}``. Pure; never raises.
    """
    try:
        cash = _f((snapshot or {}).get("cash"))
        total = _f((snapshot or {}).get("total_value"))
        positions = list((snapshot or {}).get("positions") or [])
        in_play = {str(t).upper() for t in (names_in_play or set())}

        # Priced, in-play names to size against (positive live price only).
        priced: list[tuple[str, float]] = []
        for tk, px in (watch_prices or {}).items():
            t = str(tk).upper()
            if t not in in_play:
                continue
            p = _f(px)
            if p > 0:
                priced.append((t, p))
        priced.sort(key=lambda kv: kv[0])

        deployed_pct = ((total - cash) / total * 100.0) if total > 0 else None

        # Unlock fact: the single position whose exit frees the most cash.
        # Desk cut-priority is "biggest loser first" (the capital_paralysis
        # single-source-of-truth spirit): a sale that frees cash AND stops
        # the largest bleed. Prefer the most-underwater position; if nothing
        # is underwater, the largest position by mark value (most cash freed).
        unlock = None
        if positions:
            losers = [p for p in positions if _f(p.get("unrealized_pl")) < 0]
            if losers:
                pick = min(losers, key=lambda p: _f(p.get("unrealized_pl")))
            else:
                pick = max(positions, key=_position_mark_value)
            unlock = {
                "ticker": (pick.get("ticker") or "").upper(),
                "frees_usd": round(_position_mark_value(pick), 2),
                "unrealized_pl": round(_f(pick.get("unrealized_pl")), 2),
            }

        if total <= 0 or not snapshot:
            return {
                "state": "NO_DATA",
                "summary": "buying power unavailable",
                "prompt_block": (f"{_PREAMBLE}\n  (portfolio value "
                                 f"unavailable — no buying-power awareness "
                                 f"this cycle)"),
                "cash": round(cash, 2), "deployed_pct": deployed_pct,
                "affordable": [], "cheapest_name": None,
                "cheapest_price": None, "unlock": unlock,
            }

        dep = f"{deployed_pct:.1f}%" if deployed_pct is not None else "n/a"

        if not priced:
            # Nothing priced to size against (rare — yfinance fully down for
            # the in-play set). State the cash + deployed fact honestly.
            return {
                "state": "NO_PRICED_NAMES",
                "summary": f"${cash:.2f} free, {dep} deployed; no priced "
                           f"in-play names to size against",
                "prompt_block": (f"{_PREAMBLE}\n  ${cash:.2f} free cash "
                                 f"({dep} of the book deployed). No live "
                                 f"price for any in-play name this cycle — "
                                 f"size with care."),
                "cash": round(cash, 2), "deployed_pct": deployed_pct,
                "affordable": [], "cheapest_name": None,
                "cheapest_price": None, "unlock": unlock,
            }

        affordable = [
            {"ticker": t, "price": round(px, 2),
             "whole_shares": int(cash // px)}
            for t, px in priced
        ]
        cheapest_name, cheapest_price = min(priced, key=lambda kv: kv[1])
        constrained = cash < cheapest_price

        unlock_line = ""
        if unlock and unlock["ticker"]:
            if unlock["unrealized_pl"] < 0:
                unlock_line = (
                    f"\n  Most-underwater position: {unlock['ticker']} "
                    f"(${unlock['unrealized_pl']:+.2f} unrealized) — its exit "
                    f"would free ≈${unlock['frees_usd']:.2f} and remove the "
                    f"largest drag.")
            else:
                unlock_line = (
                    f"\n  Largest position by mark value: "
                    f"{unlock['ticker']} (${unlock['frees_usd']:.2f}) — its "
                    f"exit would free the most cash.")

        if constrained:
            state = "CASH_CONSTRAINED"
            summary = (f"${cash:.2f} free, {dep} deployed — below the price "
                       f"of every in-play name (cheapest {cheapest_name} "
                       f"@ ${cheapest_price:.2f})")
            prompt_block = (
                f"{_PREAMBLE}\n  ${cash:.2f} free cash ({dep} of the book "
                f"deployed). This is below the price of every in-play name "
                f"(cheapest: {cheapest_name} @ ${cheapest_price:.2f}), so no "
                f"new whole-share entry is fundable — only fractional buys, "
                f"SELL and HOLD are actionable.{unlock_line}")
        else:
            state = "DEPLOYABLE"
            top = affordable[:_MAX_AFFORDABLE_NAMES]
            shares_str = " · ".join(
                f"{a['ticker']} {a['whole_shares']}" for a in top)
            summary = f"${cash:.2f} free, {dep} deployed"
            prompt_block = (
                f"{_PREAMBLE}\n  ${cash:.2f} free cash ({dep} of the book "
                f"deployed). Whole shares affordable now at live prices: "
                f"{shares_str}.{unlock_line}")

        return {
            "state": state,
            "summary": summary,
            "prompt_block": prompt_block,
            "cash": round(cash, 2),
            "deployed_pct": deployed_pct,
            "affordable": affordable,
            "cheapest_name": cheapest_name,
            "cheapest_price": round(cheapest_price, 2),
            "unlock": unlock,
        }
    except Exception:
        # The _safe contract: a diagnostics fault must never sink a live
        # decision cycle. One honest line, no exception.
        return {
            "state": "ERROR",
            "summary": "buying power unavailable",
            "prompt_block": (f"{_PREAMBLE}\n  (buying-power computation "
                             f"unavailable this cycle)"),
            "cash": None, "deployed_pct": None, "affordable": [],
            "cheapest_name": None, "cheapest_price": None, "unlock": None,
        }


if __name__ == "__main__":  # smoke test against the live book
    import json as _json

    from paper_trader.store import get_store
    from paper_trader.strategy import (WATCHLIST, _names_in_play,
                                       _portfolio_snapshot)
    from paper_trader import market as _mkt

    s = get_store()
    snap = _portfolio_snapshot(s)
    wp = _mkt.get_prices(WATCHLIST)
    rep = build_buying_power(
        snap, wp,
        _names_in_play(snap.get("positions") or [], [], WATCHLIST),
    )
    print(rep["prompt_block"])
    print("\n---\n")
    print(_json.dumps({k: v for k, v in rep.items() if k != "prompt_block"},
                       indent=2, default=str))
