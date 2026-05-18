"""Game Plan — the single prioritised, trader-facing action view.

The dashboard already exposes the *ingredients* of a trading decision as
separate endpoints: the co-pilot's per-ticker verb (``/api/suggestions`` via
``_classify_action``), the disposition-trap detector (``/api/hold-discipline``),
concentration / single-name risk (``/api/risk``) and the earnings calendar
(``/api/event-calendar``). A trader sitting down before the open had to open
four panels and fuse them in their head.

``build_game_plan`` is the fusion layer. It is **pure** (no I/O, never raises —
the ``hold_discipline`` / ``event_calendar`` precedent) and **observational**:
it reorders and annotates existing signals; it never sizes a trade, never gates
the live Opus loop, and adds no hard limit (invariant #12). The route at
``/api/game-plan`` does the data-gathering and reuses ``_classify_action`` so
the per-ticker verb logic stays single-sourced.

Fusion rules (deterministic):
  * an overstayed *losing* position escalates a co-pilot HOLD to ``REVIEW EXIT``
    (a disposition trap the co-pilot alone can't see) — but a stronger verb the
    co-pilot already produced (``EXIT``) is never weakened;
  * the single largest position under HIGH concentration is pushed to ``TRIM``;
  * imminent earnings on a *held* name is *awareness* — it raises priority and
    annotates, it does not invent a sell verb on its own;
  * priority is an additive urgency score; ties break deterministically.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Sell-side verbs ranked by strength. Escalation only ever moves *up* this
# ladder — it never downgrades a verb the co-pilot already made stronger.
_SELL_RANK = {"EXIT": 4, "REVIEW EXIT": 3, "TRIM": 2}
_RANK_VERB = {4: "EXIT", 3: "REVIEW EXIT", 2: "TRIM"}

# Concentration severities (from dashboard._concentration_severity) that are
# strong enough to force a trim and raise a HIGH portfolio directive.
_STRONG_CONC = {"HIGH"}

# An opportunity must clear this conviction floor to make the list — below it
# the co-pilot is essentially shrugging and it would only add noise.
_MIN_OPP_CONVICTION = 0.30

_SECTOR_HEAVY_PCT = 60.0   # one sector past this is flagged
_LOW_CASH_PCT = 5.0        # dry-powder warning threshold


def _f(x, default: float = 0.0) -> float:
    """Coerce to float, never raise — garbage marks must not sink the panel."""
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def build_game_plan(
    positions: list[dict],
    total_value: float,
    cash: float,
    hold_discipline: dict,
    concentration: dict,
    earnings_events: list[dict],
    classified: dict,
    now: datetime | None = None,
) -> dict:
    """Fuse the four diagnostics into one prioritised plan. Pure; never raises.

    ``positions`` — ``Store.open_positions()`` rows.
    ``hold_discipline`` — ``build_hold_discipline(...)`` output (``positions``
        cards carry ``overstayed`` / ``overstay_mult`` / ``age_days``).
    ``concentration`` — ``{severity, top1_ticker, top1_pct, top3_pct,
        cash_pct, sector_pct}`` (route builds this from the /api/risk pieces).
    ``earnings_events`` — ``build_event_calendar(...)['events']``.
    ``classified`` — ``{ticker: {action, conviction, reasons, held_qty,
        news_max_score, price}}`` from ``_classify_action`` (reused, no fork).
    """
    now = now or datetime.now(timezone.utc)
    tv = _f(total_value)
    cash_f = _f(cash)
    conc = concentration or {}
    hd = hold_discipline or {}

    held_tks = {
        str(p.get("ticker")).upper()
        for p in (positions or []) if p.get("ticker")
    }

    # ── disposition cards keyed by ticker ──────────────────────────────
    hd_cards: dict[str, dict] = {}
    for c in (hd.get("positions") or []):
        tk = c.get("ticker")
        if tk:
            hd_cards[str(tk).upper()] = c

    # ── soonest earnings per held ticker ───────────────────────────────
    earn: dict[str, dict] = {}
    for ev in (earnings_events or []):
        tk = str(ev.get("ticker") or "").upper()
        da = ev.get("days_away")
        if not tk or da is None:
            continue
        if tk not in earn or _f(da) < _f(earn[tk].get("days_away")):
            earn[tk] = ev

    conc_sev = conc.get("severity")
    conc_top1 = str(conc.get("top1_ticker") or "").upper()
    conc_strong = conc_sev in _STRONG_CONC

    # ── per-held-position fusion ───────────────────────────────────────
    position_actions: list[dict] = []
    for p in (positions or []):
        try:
            tk = str(p.get("ticker") or "").upper()
            if not tk:
                continue
            c = classified.get(tk) or classified.get(p.get("ticker")) or {}
            base = str(c.get("action") or "HOLD").upper()
            reasons: list[str] = list(c.get("reasons") or [])
            conviction = round(_f(c.get("conviction")), 2)

            qty = _f(p.get("qty"))
            cur = _f(p.get("current_price")) or _f(p.get("avg_cost"))
            mult = 100 if p.get("type") in ("call", "put") else 1
            mv = cur * qty * mult
            pct_port = round((mv / tv * 100) if tv else 0.0, 2)
            upl = _f(p.get("unrealized_pl"))

            priority = 0
            sell_floor = 0

            card = hd_cards.get(tk)
            if card and bool(card.get("overstayed")):
                priority += 3
                sell_floor = max(sell_floor, 3)  # at least REVIEW EXIT
                age = card.get("age_days")
                mlt = card.get("overstay_mult")
                age_s = f"{age:.1f}d" if isinstance(age, (int, float)) else "?"
                mlt_s = (f"{mlt:.1f}×" if isinstance(mlt, (int, float))
                         else "well past")
                reasons.insert(
                    0, f"⚠ disposition trap — held {age_s}, {mlt_s} "
                    f"the desk's own losing-cut time")

            is_top1 = bool(conc_top1) and tk == conc_top1
            if conc_strong and is_top1:
                priority += 2
                sell_floor = max(sell_floor, 2)  # at least TRIM
                reasons.append(
                    f"largest position — {pct_port:.1f}% of book "
                    f"(concentration {conc_sev})")

            ev = earn.get(tk)
            if ev is not None:
                da = _f(ev.get("days_away"))
                ed = str(ev.get("earnings_date") or "")[:10]
                if da <= 3:
                    priority += 2
                    reasons.append(
                        f"⏰ earnings in {da:.1f}d ({ed}) — event "
                        f"risk on a held name")
                elif da <= 7:
                    priority += 1
                    reasons.append(
                        f"earnings in {da:.1f}d ({ed}) — within a week")

            if upl < 0:
                priority += 1

            # Final verb: the strongest of {co-pilot's own sell verb, the
            # floor the risk flags imply}. Awareness signals (earnings,
            # losing) raise priority but never the sell floor.
            final_rank = max(_SELL_RANK.get(base, 0), sell_floor)
            if final_rank > 0:
                action = _RANK_VERB[final_rank]
                if base in ("ADD", "BUY"):
                    reasons.append(
                        f"co-pilot said {base} but risk flags override")
            elif base in ("ADD", "BUY"):
                action = base
            else:
                action = "HOLD"

            position_actions.append({
                "ticker": tk,
                "action": action,
                "priority": priority,
                "conviction": conviction,
                "unrealized_pl": round(upl, 2),
                "pct_port": pct_port,
                "reasons": reasons,
            })
        except Exception:  # noqa: BLE001 — a diagnostic must never raise
            continue

    # Most urgent first; deterministic tie-break (worst loss, then ticker).
    position_actions.sort(
        key=lambda c: (-c["priority"], c["unrealized_pl"], c["ticker"]))

    # ── portfolio-level directives ─────────────────────────────────────
    directives: list[dict] = []
    if conc_sev in _STRONG_CONC:
        directives.append({
            "kind": "CONCENTRATION", "severity": "HIGH",
            "text": (
                f"Top position {conc.get('top1_ticker') or '?'} is "
                f"{_f(conc.get('top1_pct')):.1f}% of book (top-3 "
                f"{_f(conc.get('top3_pct')):.1f}%) — single-name risk; "
                f"consider trimming into strength."),
        })
    if hd.get("state") == "DISPOSITION_DRAG":
        directives.append({
            "kind": "DISPOSITION", "severity": "HIGH",
            "text": (
                f"Disposition drag ${_f(hd.get('disposition_drag_usd')):+.2f} "
                f"across {int(_f(hd.get('n_overstayed')))} overstayed losing "
                f"position(s) — review the sell-side cards above."),
        })
    for sec, pct in (conc.get("sector_pct") or {}).items():
        if _f(pct) > _SECTOR_HEAVY_PCT:
            directives.append({
                "kind": "SECTOR", "severity": "MEDIUM",
                "text": (f"{sec} is {_f(pct):.1f}% of book — "
                         f"sector-concentrated, correlated drawdown risk."),
            })
    cash_pct = _f(conc.get("cash_pct"),
                  (cash_f / tv * 100) if tv else 0.0)
    if cash_pct < _LOW_CASH_PCT:
        directives.append({
            "kind": "DRY_POWDER", "severity": "MEDIUM",
            "text": (f"Only {cash_pct:.1f}% cash — limited room to act on "
                     f"new setups or average down."),
        })
    directives.sort(key=lambda d: 0 if d["severity"] == "HIGH" else 1)

    # ── opportunities: non-held BUY/WATCH the co-pilot likes ───────────
    opportunities: list[dict] = []
    for tk, c in (classified or {}).items():
        try:
            tku = str(tk).upper()
            if tku in held_tks:
                continue
            act = str((c or {}).get("action") or "").upper()
            conv = _f((c or {}).get("conviction"))
            if act not in ("BUY", "WATCH") or conv < _MIN_OPP_CONVICTION:
                continue
            opportunities.append({
                "ticker": tku,
                "action": act,
                "conviction": round(conv, 2),
                "news_max_score": round(_f((c or {}).get("news_max_score")), 1),
                "price": (c or {}).get("price"),
                "reasons": list((c or {}).get("reasons") or []),
            })
        except Exception:  # noqa: BLE001
            continue
    opportunities.sort(key=lambda o: (-o["conviction"], o["ticker"]))
    opportunities = opportunities[:5]

    # ── state / headline ───────────────────────────────────────────────
    actionable = [c for c in position_actions if c["action"] != "HOLD"]
    high_dirs = [d for d in directives if d["severity"] == "HIGH"]
    n_actions = len(actionable) + len(high_dirs)
    n_open = len(position_actions)

    if not position_actions and not opportunities:
        state = "NO_DATA"
        headline = "No open positions and no actionable setups."
    elif n_actions == 0:
        state = "STEADY"
        headline = (
            f"Book steady — {n_open} position(s) within discipline; "
            f"nothing high-priority for the next session.")
    else:
        state = "ACTIONS_PRESENT"
        top_bits = [f"{c['action']} {c['ticker']}" for c in actionable[:2]]
        if not top_bits and high_dirs:
            top_bits = [high_dirs[0]["kind"]]
        headline = (
            f"{n_actions} action(s) for the next session"
            + (": " + " · ".join(top_bits) if top_bits else ""))

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "headline": headline,
        "n_actions": n_actions,
        "n_open": n_open,
        "position_actions": position_actions,
        "portfolio_directives": directives,
        "opportunities": opportunities,
    }
