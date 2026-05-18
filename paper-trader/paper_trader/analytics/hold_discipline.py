"""Loser hold-time discipline — the disposition trap, caught *while it is
still happening* on the **open** book.

The desk's documented live pathology is the disposition effect: a
16.7%-win-rate book with a ~0.52-day median hold that cuts winners fast
and rides losers down. Every neighbour sees this *after the fact* or from
a *different* slice:

* ``/api/loser-autopsy`` / ``/api/trade-asymmetry`` — post-mortems on
  trades **already closed**. They tell you the median *losing* hold but
  never check the position you are sitting on **right now** against it.
* ``/api/thesis-drift`` — re-tests an open position against the *reason*
  it was opened. Says nothing about *time held vs how you actually cut*.
* ``/api/capital-paralysis`` — cash drag / the unlock ladder. About not
  having dry powder, not about overstaying a loser.
* ``/api/position-thesis`` — a per-holding roll-up (days held + P/L +
  news + scorer). It *shows* days held but has **no empirical reference**
  to judge it against.

``build_hold_discipline`` is the missing forward question a disciplined
trader asks every day: *"Which open positions am I, right now, holding
at a loss past my own historical losing-cut time — i.e. actively
repeating the disposition trap?"*

It anchors on the desk's **own** behaviour, not an arbitrary stop: the
empirical median *losing* hold is consumed **verbatim** from
``loser_autopsy.build_loser_autopsy`` (``median_loser_hold_days`` /
``n_losers``), which itself consumes ``round_trips.build_round_trips`` —
so the reference can never drift from ``/api/loser-autopsy`` and there is
no second hand-rolled P&L or hold computation (AGENTS.md invariant #10).
Per-position dollars are read **directly** from the positions table's
``unrealized_pl`` (the option ×100 multiplier is already baked into that
column — re-deriving from ``avg_cost × qty`` would silently halve/×100 an
option's risk).

Sample-size honesty mirrors ``loser_autopsy``: below
``MIN_REFERENCE_LOSERS`` closed losers the empirical median is noise, so
the state is ``INSUFFICIENT`` and the per-position cards are still emitted
(age + loss) but **no position is flagged overstayed and the verdict is
withheld**. It is a *diagnostic / advisory* surface only — it never gates
Opus, adds no caps, and is **not** injected into the decision prompt
(AGENTS.md invariants #2/#12; the ``loser_autopsy`` / ``winner_autopsy``
endpoint precedent).
"""
from __future__ import annotations

from datetime import datetime, timezone

from .loser_autopsy import build_loser_autopsy
from .round_trips import _parse_ts

# Below this many closed *losing* round-trips the empirical median losing
# hold is too noisy to call a live position "overstayed" — the verdict is
# withheld (the build_correlation / loser_autopsy honesty precedent). A
# documented module constant, not a tunable Opus ever sees.
MIN_REFERENCE_LOSERS = 3


def _age_days(opened_at: str | None, now: datetime) -> float | None:
    """Calendar days a position has been open. ``None`` (never an error)
    on an unparseable / missing / future ``opened_at`` — the same defensive
    contract as ``round_trips._hold_days``."""
    o = _parse_ts(opened_at)
    if o is None:
        return None
    if o.tzinfo is None:
        o = o.replace(tzinfo=timezone.utc)
    dd = (now - o).total_seconds() / 86400.0
    return round(dd, 4) if dd >= 0 else None


def build_hold_discipline(open_positions: list[dict],
                          trades: list[dict],
                          now: datetime | None = None) -> dict:
    """Open-book disposition-trap detector. Pure, never raises.

    ``open_positions`` is ``Store.open_positions()``-shaped (each row has
    ``ticker/type/qty/avg_cost/current_price/unrealized_pl/opened_at``).
    ``trades`` is the **oldest→newest** ledger the round-trip builders
    expect — pass exactly what ``/api/loser-autopsy`` passes:
    ``list(reversed(store.recent_trades(2000)))``.

    A position is *overstayed* iff it is **losing**
    (``unrealized_pl < 0`` — the strict ``round_trips``/#10 convention)
    **and** its age strictly exceeds the empirical median losing hold
    (``age_days > median``; ``==`` is *within* discipline — the
    ``loser_autopsy`` strict-boundary idiom).
    """
    now = now or datetime.now(timezone.utc)

    # ── Single source of truth for the reference (invariant #10) ───────
    # loser_autopsy → round_trips. We never re-derive the median or P&L.
    # _safe-wrapped (the event_calendar / risk_mirror precedent): a fault
    # in the composed builder degrades to "no reference this run" — the
    # verdict is withheld — never an exception that 500s the endpoint or
    # kills the daily-close report.
    ref_error = False
    try:
        la = build_loser_autopsy(trades)
        median = la.get("median_loser_hold_days")
        n_closed_losers = la.get("n_losers") or 0
        reference_state = la.get("state")
    except Exception as e:  # noqa: BLE001 — diagnostics must not raise
        ref_error = True
        median, n_closed_losers, reference_state = None, 0, f"ERROR: {e}"
    reference_ok = (median is not None
                    and n_closed_losers >= MIN_REFERENCE_LOSERS)

    # ── Per-open-position cards ────────────────────────────────────────
    cards: list[dict] = []
    for p in (open_positions or []):
        try:
            upl = float(p.get("unrealized_pl") or 0.0)
        except (TypeError, ValueError):
            # A garbage non-numeric mark must not sink the whole panel
            # (the loser_autopsy "never raises on garbage" purity).
            upl = 0.0
        age = _age_days(p.get("opened_at"), now)
        is_losing = upl < 0.0
        overstayed = bool(
            reference_ok and is_losing
            and age is not None and age > median)
        cards.append({
            "ticker": p.get("ticker"),
            "type": p.get("type"),
            "qty": p.get("qty"),
            "avg_cost": p.get("avg_cost"),
            "current_price": p.get("current_price"),
            "unrealized_pl": round(upl, 2),
            "age_days": age,
            "is_losing": is_losing,
            "overstayed": overstayed,
            # How many multiples of the desk's own losing-cut time this
            # position has run, only when meaningfully overstayed.
            "overstay_mult": (round(age / median, 2)
                              if (overstayed and median) else None),
        })

    n_open = len(cards)
    losing_cards = [c for c in cards if c["is_losing"]]
    n_losing_open = len(losing_cards)
    overstayed_cards = [c for c in cards if c["overstayed"]]
    n_overstayed = len(overstayed_cards)

    # Overstayed first, then most-negative unrealized P/L, then ticker
    # (deterministic tie-break — two identical losses never reorder).
    cards.sort(key=lambda c: (not c["overstayed"],
                              c["unrealized_pl"],
                              c["ticker"] or ""))
    overstayed_cards.sort(key=lambda c: (c["unrealized_pl"],
                                         c["ticker"] or ""))

    disposition_drag_usd = (
        round(sum(c["unrealized_pl"] for c in overstayed_cards), 2)
        if overstayed_cards else 0.0)
    worst = overstayed_cards[0] if overstayed_cards else None

    # ── state / verdict (verdict gated to a stable reference) ──────────
    if n_open == 0:
        state = "NO_DATA"
    elif not reference_ok:
        state = "INSUFFICIENT"
    elif n_overstayed >= 1:
        state = "DISPOSITION_DRAG"
    else:
        state = "DISCIPLINED"

    verdict = state if state in ("DISCIPLINED", "DISPOSITION_DRAG") else None

    # ── headline (observational, no directive verb) ────────────────────
    if state == "NO_DATA":
        headline = "No open positions — nothing to check for hold discipline."
    elif state == "INSUFFICIENT":
        if ref_error:
            headline = (
                f"{n_open} open position(s); hold-discipline reference "
                f"unavailable (loser-autopsy fault) — verdict withheld.")
        else:
            headline = (
                f"{n_open} open position(s); only {n_closed_losers} closed "
                f"losing round-trip(s) (< {MIN_REFERENCE_LOSERS}) — no "
                f"empirical losing-hold reference yet (verdict withheld).")
    elif state == "DISCIPLINED":
        if n_losing_open == 0:
            headline = (
                f"No losing open positions; empirical median losing hold is "
                f"{median:.2f}d — no disposition drag.")
        else:
            headline = (
                f"All {n_losing_open} losing open position(s) within the "
                f"empirical median losing hold of {median:.2f}d — no "
                f"disposition drag.")
    else:  # DISPOSITION_DRAG
        wp = worst or {}
        wa = wp.get("age_days")
        wa_s = f"{wa:.1f}d" if wa is not None else "?d"
        headline = (
            f"{n_overstayed} open position(s) held past the empirical median "
            f"losing hold ({median:.2f}d); largest: {wp.get('ticker')} "
            f"({wa_s}, ${wp.get('unrealized_pl', 0.0):+.2f}). Disposition "
            f"drag ${disposition_drag_usd:+.2f} unrealized.")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "reference_median_losing_hold_days": median,
        "reference_n_closed_losers": n_closed_losers,
        "reference_state": reference_state,
        "min_reference_losers": MIN_REFERENCE_LOSERS,
        "n_open": n_open,
        "n_losing_open": n_losing_open,
        "n_overstayed": n_overstayed,
        "disposition_drag_usd": disposition_drag_usd,
        "worst_overstayed": worst,
        "positions": cards,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json

    from ..store import get_store

    st = get_store()
    out = build_hold_discipline(
        st.open_positions(), list(reversed(st.recent_trades(2000))))
    print(json.dumps(out, indent=2, default=str))
