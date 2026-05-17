"""Funded suggestions — pair every unfundable idea with the sale that funds it.

``/api/liquidity`` sees the trap (no dry powder). ``/api/decision-drought``
sees the cost (alpha bled while pinned). ``/api/capital-paralysis`` connects
those and ranks the unlock ladder. ``/api/suggestions`` lists trade ideas — but
when the book is PINNED every BUY/ADD it surfaces is unfundable and the panel
says so without saying *what to sell to fund it*.

This composes the two structures the dashboard already computes — the
suggestions list and the ``build_capital_paralysis`` dict (its unlock ladder is
already in desk cut-priority: biggest loser first) — and attaches, to each
actionable BUY/ADD idea, the **minimum prefix of sales** whose cumulative freed
cash covers an advisory suggested notional. No metric is re-derived here (the
``capital_paralysis`` single-source-of-truth precedent); it is pure
recombination.

Advisory only. It never gates the trader, sizes nothing automatically, and adds
no caps — the suggested notional is explicitly labelled advisory (AGENTS.md
invariants #2 / #12). Only BUY/ADD ideas are funding-checked: HOLD/WATCH are
no-ops and TRIM/EXIT *raise* cash, so they never consume buying power.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Actions that consume buying power and therefore need funding.
_ACTIONABLE = {"BUY", "ADD"}


def _headline(state, can_act, actionable, top, pairing,
              n_funded, n_unlockable, n_unfundable) -> str:
    if not actionable:
        return "No actionable BUY/ADD ideas in the current suggestion set."
    t = (f"{top['action']} {top['ticker']} (conv {top['conviction']:.2f})"
         if top else "—")
    if can_act:
        return (f"FREE — {n_funded} BUY/ADD idea(s) fundable from cash now; "
                f"best: {t}.")
    if state == "PINNED" and pairing:
        return (f"PINNED — best idea {t}; sell {pairing['sell']} → free "
                f"${pairing['frees_usd']:.2f} → can act.")
    if state == "PINNED":
        return (f"PINNED — best idea {t}; {n_unlockable} unlockable via "
                f"sale(s), {n_unfundable} unfundable even after selling.")
    return (f"{state} — {len(actionable)} BUY/ADD idea(s) but no capital to "
            f"fund them (best: {t}).")


def build_funded_suggestions(suggestions: list[dict],
                             paralysis: dict,
                             now: datetime | None = None) -> dict:
    """Attach an unlock chain to each actionable idea. Pure; ``now`` injectable.

    ``suggestions`` is the ``/api/suggestions`` list; ``paralysis`` is the
    ``build_capital_paralysis`` dict (its ``unlock_ladder`` is already in
    desk cut-priority order — losers before winners, larger first).
    """
    now = now or datetime.now(timezone.utc)
    p = paralysis or {}
    state = p.get("state") or "NO_DATA"
    can_act = bool(p.get("can_act_on_signal"))
    total_value = float(p.get("total_value") or 0.0)
    ladder = list(p.get("unlock_ladder") or [])
    recommended_unlock = p.get("recommended_unlock")

    ladder_tickers = [str(r.get("ticker")) for r in ladder]
    ladder_total = round(
        (ladder[-1].get("cumulative_freed_usd") if ladder else 0.0) or 0.0, 2)

    actionable = [s for s in (suggestions or [])
                  if str(s.get("action") or "").upper() in _ACTIONABLE]
    # conviction desc, then ticker asc — deterministic tie-break.
    actionable.sort(key=lambda s: (-float(s.get("conviction") or 0.0),
                                   str(s.get("ticker") or "")))

    ideas: list[dict] = []
    for s in actionable:
        tk = str(s.get("ticker") or "")
        conv = float(s.get("conviction") or 0.0)
        notional = round(conv * total_value, 2) if total_value > 0 else 0.0

        if can_act:
            fund, by, frees, enough = "FUNDED", [], 0.0, True
            note = "cash available now — no sale required"
        elif state == "PINNED" and ladder:
            by, frees, enough = [], ladder_total, False
            if notional > 0:
                for r in ladder:
                    by.append(str(r.get("ticker")))
                    cum = round(r.get("cumulative_freed_usd") or 0.0, 2)
                    if cum >= notional:
                        frees, enough = cum, True
                        break
                else:  # ladder exhausted without covering the notional
                    by = list(ladder_tickers)
                    frees, enough = ladder_total, False
            if enough:
                fund = "UNLOCKABLE"
                note = (f"PINNED — sell {', '.join(by)} → free "
                        f"${frees:.2f} ≥ ${notional:.2f} advisory notional")
            else:
                fund = "UNFUNDABLE"
                note = (f"PINNED — even selling the whole ladder frees only "
                        f"${frees:.2f} (< ${notional:.2f} advisory notional)")
        else:
            fund, by, frees, enough = "UNFUNDABLE", [], 0.0, False
            note = {
                "EMPTY": "no cash and no positions to free capital from",
                "NO_DATA": "no portfolio data",
            }.get(state, "PINNED — no positions to unlock")

        ideas.append({
            "ticker": tk,
            "action": str(s.get("action") or "").upper(),
            "conviction": conv,
            "suggested_notional_usd": notional,
            "fundability": fund,
            "funded_by": by,
            "frees_usd": frees,
            "enough": enough,
            "note": note,
        })

    n_funded = sum(1 for i in ideas if i["fundability"] == "FUNDED")
    n_unlockable = sum(1 for i in ideas if i["fundability"] == "UNLOCKABLE")
    n_unfundable = sum(1 for i in ideas if i["fundability"] == "UNFUNDABLE")

    top = None
    if actionable:
        b = actionable[0]
        top = {"ticker": str(b.get("ticker") or ""),
               "action": str(b.get("action") or "").upper(),
               "conviction": float(b.get("conviction") or 0.0)}

    pairing = None
    if (state == "PINNED" and top and isinstance(recommended_unlock, dict)
            and recommended_unlock.get("ticker")):
        top_idea = next((i for i in ideas if i["ticker"] == top["ticker"]),
                        None)
        pairing = {
            "sell": recommended_unlock["ticker"],
            "buy": top["ticker"],
            "frees_usd": top_idea["frees_usd"] if top_idea else 0.0,
            "buy_conviction": top["conviction"],
        }

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "can_act": can_act,
        "n_actionable": len(actionable),
        "n_funded": n_funded,
        "n_unlockable": n_unlockable,
        "n_unfundable": n_unfundable,
        "ideas": ideas,
        "top_actionable": top,
        "recommended_pairing": pairing,
        "headline": _headline(state, can_act, actionable, top, pairing,
                              n_funded, n_unlockable, n_unfundable),
    }
