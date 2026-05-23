"""Forced-hold attribution — for every currently OPEN position, partition
the decision cycles since it was opened into ``blind`` (NO_DECISION —
Opus could not act this cycle: quota / host saturation / model timeout /
parse miss) vs ``sighted`` (any real decision row: HOLD / BUY / SELL /
BLOCKED). Then judge whether the position is on the book by *choice*
(sighted-dominant) or by *force* (blind-dominant — Opus was unable to
sell even if it wanted to).

Why this exists
---------------

Every neighbour answers a different question and leaves this gap open:

* ``/api/hold-discipline`` — "is this losing position past the desk's
  empirical median losing hold?" (disposition trap on the OPEN book).
  Reads age in days. Does not look at whether the trader *could* have
  acted on those days.
* ``/api/thesis-drift`` — "does the verbatim entry-reason still hold up
  against current quant + news?". Re-tests the *thesis*, not the
  *agency*.
* ``/api/no-decision-recovery`` — "is the current wedge anomalous vs
  history?". Aggregates wedge run-lengths across the WHOLE decision
  tape. Does not attribute wedges to OPEN positions.
* ``/api/decision-vapor-skill`` — grades the *reasoning specificity*
  of FILLED decisions. Says nothing about cycles that produced no
  decision at all.
* ``/api/today-action-tape`` — flat aggregate of today's cycles. Does
  not partition by which position was on the book.

The trader's question this answers
-----------------------------------

"My NVDA position has been open for 48 hours and is -2.49%. Is that
because Opus is *choosing* to ride it through a HOLD-able regime, or
because the box has been host-saturated all day and Opus literally
couldn't issue a SELL even if it wanted to?"

These are very different. A `CHOSEN_HOLD` is a real trading decision
backed by reasoning the operator can audit; a `FORCED_HOLD` is the
position the trader is silently *stuck with*. The latter is
operationally actionable (lift the saturation, restart the runner,
escalate) even if the trade itself looks identical from the outside.

Per-position attribution
-------------------------

For each open position ``p``:

* ``cycles_total`` — decisions with ``timestamp >= p.opened_at``
  (ISO-string lex compare — works for the entire ISO-8601 UTC range
  this DB writes; the ``signals._age_hours`` lexical-compare precedent).
* ``cycles_blind`` — of those, decisions whose ``action_taken``
  begins with the literal token ``NO_DECISION`` (the exact string
  ``store.record_decision`` writes for every failed cycle —
  ``strategy.decide()`` at strategy.py:1881). Includes every blind
  sub-bucket (host_skip / model_empty / quota / parse_failed /
  retry_failed) — they are all "Opus did not act" for the trader's
  agency question.
* ``cycles_sighted`` — every other decision row (HOLD, FILLED, BLOCKED
  — anything that came back as a real verb).
* ``blind_pct`` — ``cycles_blind / cycles_total``; ``0.0`` when
  ``cycles_total == 0`` (degrade-safe — never raises).
* ``forced_hold`` — bool, set when the verdict is FORCED_HOLD.

Per-position verdict (only when ``cycles_total >= MIN_CYCLES = 10``):

* ``FORCED_HOLD``     — ``blind_pct >= 0.50``
* ``PARTIALLY_FORCED``— ``0.25 <= blind_pct < 0.50``
* ``MIXED``           — ``0.10 <= blind_pct < 0.25``
* ``CHOSEN_HOLD``     — ``blind_pct < 0.10``

Below ``MIN_CYCLES`` the per-position verdict is ``None`` (state is
``EMERGING`` at the aggregate level — the ``loser_autopsy`` /
``hold_discipline`` honesty precedent: a 3-cycle sample is noise).

Aggregate state / verdict
-------------------------

* ``NO_DATA``     — no open positions; verdict ``None``.
* ``EMERGING``    — at least one open position has
  ``cycles_total < MIN_CYCLES``; verdict withheld so a 2-cycle reading
  can never call a fresh position "FORCED_HOLD".
* ``STABLE``      — every open position has ``>= MIN_CYCLES``; an
  aggregate verdict is emitted:

  * ``FORCED_HOLD_DOMINANT`` — at least one position's verdict is
    ``FORCED_HOLD``.
  * ``PARTIALLY_FORCED``     — at least one position is
    ``PARTIALLY_FORCED`` and none are ``FORCED_HOLD``.
  * ``MOSTLY_CHOSEN``        — every position is ``CHOSEN_HOLD`` or
    ``MIXED``.

Headline is a short observational sentence — never directive
(invariants #2 / #12 — the ``loser_autopsy`` / ``hold_discipline``
precedent). Verdict precedence: ``FORCED_HOLD_DOMINANT`` beats
``PARTIALLY_FORCED`` beats ``MOSTLY_CHOSEN``.

Pure builder. Open positions in, decisions in, dict out, never raises
on garbage (the ``hold_discipline`` "never raises" purity contract).
"""
from __future__ import annotations

from datetime import datetime, timezone


# Below this many decision cycles a per-position attribution is too
# noisy to call. The state is ``EMERGING`` at the aggregate level and
# the per-position verdict is withheld. Documented module constant
# (the ``hold_discipline.MIN_REFERENCE_LOSERS`` precedent), not a
# tunable Opus ever sees.
MIN_CYCLES = 10

# Verdict thresholds on ``blind_pct``. Boundaries are inclusive on the
# upper-discipline side (``< 0.50`` is PARTIALLY_FORCED, ``>= 0.50``
# is FORCED_HOLD) — same strict-boundary idiom as
# ``hold_discipline.overstayed`` (``age > median`` is overstayed,
# ``age == median`` is within discipline). Tests pin every boundary.
FORCED_HOLD_BLIND_PCT = 0.50
PARTIALLY_FORCED_BLIND_PCT = 0.25
MIXED_BLIND_PCT = 0.10


def _classify(blind_pct: float, cycles_total: int) -> str | None:
    """Per-position verdict from ``blind_pct``. ``None`` when sample is
    below ``MIN_CYCLES`` — the honesty gate."""
    if cycles_total < MIN_CYCLES:
        return None
    if blind_pct >= FORCED_HOLD_BLIND_PCT:
        return "FORCED_HOLD"
    if blind_pct >= PARTIALLY_FORCED_BLIND_PCT:
        return "PARTIALLY_FORCED"
    if blind_pct >= MIXED_BLIND_PCT:
        return "MIXED"
    return "CHOSEN_HOLD"


def _age_hours(opened_at: str | None, now: datetime) -> float | None:
    """Hours a position has been open. ``None`` on missing / unparseable /
    future ``opened_at`` (defensive — mirrors ``signals._age_hours`` and
    ``hold_discipline._age_days``)."""
    if not opened_at:
        return None
    try:
        dt = datetime.fromisoformat(str(opened_at).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    secs = (now - dt).total_seconds()
    if secs < 0:
        return None
    return round(secs / 3600.0, 2)


def _is_blind(action_taken: str | None) -> bool:
    """True iff ``action_taken`` indicates the cycle produced no real
    decision. Matches the exact literal ``store.record_decision`` writes
    on the NO_DECISION path (``strategy.decide()`` at strategy.py:1881)."""
    if not action_taken:
        return False
    return str(action_taken).strip().upper().startswith("NO_DECISION")


def build_forced_hold_attribution(
    open_positions: list[dict],
    decisions: list[dict],
    now: datetime | None = None,
) -> dict:
    """Forced-vs-chosen hold attribution. Pure, never raises.

    ``open_positions`` is ``Store.open_positions()``-shaped (each row
    has at minimum ``ticker`` and ``opened_at``). ``decisions`` is
    ``Store.recent_decisions(N)``-shaped (newest-first; each row has
    ``timestamp`` and ``action_taken``). Order does not matter — we
    only filter on ``timestamp >= opened_at`` per position.

    Returns a dict with ``state`` / ``verdict`` / ``headline`` plus
    per-position cards and a flat ``n_forced`` / ``n_partially_forced``
    count. Never raises — a garbage row in either input is skipped.
    """
    now = now or datetime.now(timezone.utc)

    decision_rows: list[tuple[str, bool]] = []
    for d in (decisions or []):
        if not isinstance(d, dict):
            continue
        ts = d.get("timestamp")
        if not isinstance(ts, str) or not ts:
            continue
        decision_rows.append((ts, _is_blind(d.get("action_taken"))))

    cards: list[dict] = []
    for p in (open_positions or []):
        if not isinstance(p, dict):
            continue
        ticker = p.get("ticker")
        opened_at = p.get("opened_at")
        cycles_total = 0
        cycles_blind = 0
        if isinstance(opened_at, str) and opened_at:
            for ts, blind in decision_rows:
                # ISO-8601 UTC strings compare lexically — the
                # ``signals._age_hours`` lex-compare precedent. Inclusive
                # on the open instant: a decision row written in the
                # same microsecond as ``opened_at`` (rare but possible
                # under load) still counts toward this position.
                if ts >= opened_at:
                    cycles_total += 1
                    if blind:
                        cycles_blind += 1
        cycles_sighted = cycles_total - cycles_blind
        blind_pct = (cycles_blind / cycles_total) if cycles_total > 0 else 0.0
        verdict = _classify(blind_pct, cycles_total)
        cards.append({
            "ticker": ticker,
            "opened_at": opened_at,
            "age_hours": _age_hours(opened_at, now),
            "cycles_total": cycles_total,
            "cycles_blind": cycles_blind,
            "cycles_sighted": cycles_sighted,
            "blind_pct": round(blind_pct, 4),
            "verdict": verdict,
            "forced_hold": verdict == "FORCED_HOLD",
        })

    # Sort: forced first (worst blind_pct first), then partially-forced
    # by blind_pct desc, then everything else by ticker.
    def _sort_key(c):
        v = c.get("verdict")
        rank = {"FORCED_HOLD": 0, "PARTIALLY_FORCED": 1,
                "MIXED": 2, "CHOSEN_HOLD": 3}.get(v, 4)
        return (rank, -float(c.get("blind_pct") or 0.0),
                c.get("ticker") or "")
    cards.sort(key=_sort_key)

    n_open = len(cards)
    n_forced = sum(1 for c in cards if c["verdict"] == "FORCED_HOLD")
    n_partially_forced = sum(
        1 for c in cards if c["verdict"] == "PARTIALLY_FORCED")
    n_chosen = sum(1 for c in cards if c["verdict"] == "CHOSEN_HOLD")
    n_mixed = sum(1 for c in cards if c["verdict"] == "MIXED")

    # Aggregate state. EMERGING fires when *any* per-position card has
    # cycles_total < MIN_CYCLES — even one fresh position contaminates a
    # would-be STABLE aggregate (the loser_autopsy/hold_discipline honesty
    # precedent). NO_DATA covers the empty-book case.
    if n_open == 0:
        state = "NO_DATA"
    elif any(c["cycles_total"] < MIN_CYCLES for c in cards):
        state = "EMERGING"
    else:
        state = "STABLE"

    verdict: str | None
    if state != "STABLE":
        verdict = None
    elif n_forced >= 1:
        verdict = "FORCED_HOLD_DOMINANT"
    elif n_partially_forced >= 1:
        verdict = "PARTIALLY_FORCED"
    else:
        verdict = "MOSTLY_CHOSEN"

    # Headline — observational only, no directive verb (#2/#12).
    if state == "NO_DATA":
        headline = "No open positions — nothing to attribute."
    elif state == "EMERGING":
        # Identify the freshest under-sampled position so the operator
        # knows *which* one is keeping the verdict withheld.
        emerging = [c for c in cards if c["cycles_total"] < MIN_CYCLES]
        if len(emerging) == len(cards):
            tk = emerging[0]["ticker"] or "?"
            ct = emerging[0]["cycles_total"]
            if n_open == 1:
                headline = (
                    f"{tk} has only seen {ct}/{MIN_CYCLES} decision cycles "
                    f"since it was opened — too fresh to attribute "
                    f"forced-vs-chosen (verdict withheld).")
            else:
                headline = (
                    f"{n_open} open position(s); all are below the "
                    f"{MIN_CYCLES}-cycle minimum for attribution "
                    f"(verdict withheld).")
        else:
            tk = emerging[0]["ticker"] or "?"
            headline = (
                f"{n_open} open position(s); {tk} (and {len(emerging) - 1} "
                f"other) below the {MIN_CYCLES}-cycle attribution floor — "
                f"verdict withheld."
                if len(emerging) > 1
                else
                f"{n_open} open position(s); {tk} is fresh "
                f"({emerging[0]['cycles_total']}/{MIN_CYCLES} cycles) — "
                f"verdict withheld.")
    elif verdict == "FORCED_HOLD_DOMINANT":
        worst = cards[0]
        wtk = worst.get("ticker") or "?"
        wpct = (worst.get("blind_pct") or 0.0) * 100.0
        headline = (
            f"{n_forced} of {n_open} open position(s) FORCED-HELD — "
            f"largest: {wtk} ({wpct:.0f}% of {worst['cycles_total']} "
            f"cycles since open were NO_DECISION). Operationally stuck — "
            f"sighted decision cadence is the bottleneck, not the thesis.")
    elif verdict == "PARTIALLY_FORCED":
        worst = cards[0]  # PARTIALLY_FORCED sorts first when no FORCED
        wtk = worst.get("ticker") or "?"
        wpct = (worst.get("blind_pct") or 0.0) * 100.0
        headline = (
            f"{n_partially_forced} of {n_open} open position(s) "
            f"PARTIALLY-FORCED — largest: {wtk} ({wpct:.0f}% blind cycles "
            f"since open).")
    else:  # MOSTLY_CHOSEN
        headline = (
            f"All {n_open} open position(s) are choices — every position "
            f"sees Opus reach a sighted decision in >= "
            f"{int((1 - MIXED_BLIND_PCT) * 100)}% of cycles since open.")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "min_cycles": MIN_CYCLES,
        "n_open": n_open,
        "n_forced": n_forced,
        "n_partially_forced": n_partially_forced,
        "n_mixed": n_mixed,
        "n_chosen": n_chosen,
        "positions": cards,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json

    from ..store import get_store

    st = get_store()
    out = build_forced_hold_attribution(
        st.open_positions(), st.recent_decisions(500))
    print(json.dumps(out, indent=2, default=str))
