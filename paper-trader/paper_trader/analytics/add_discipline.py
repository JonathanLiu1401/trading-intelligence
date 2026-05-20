"""Add-discipline audit — when the book ADDs to an existing position, is it
**chasing** (paying up vs running cost basis) or **averaging down** (paying
below)?

Every other behavioural mirror on this desk catches a different pathology:

* ``trade_asymmetry`` is the disposition gap (winners cut short, losers
  ridden long).
* ``churn`` counts overtrading.
* ``loser_autopsy`` / ``winner_autopsy`` narrate closed round-trips.
* ``thesis_drift`` flags rationale incoherence across cycles.
* ``round_trip_postmortem`` scores exit timing.
* ``blocked_repeats`` flags the engine refusing actions.

None of them watch **the moment of an ADD**: a BUY into a name the book
already holds. That action carries a sign — paying *above* the running
avg_cost is the textbook *chasing* behaviour (anchoring on entry, bidding
into strength after the easy money has been made); paying *below* is the
textbook *averaging down* behaviour (the rationality of which depends on
whether the original thesis is intact, and which can quietly double down
on a broken thesis). A discretionary PM watches every ADD with that lens;
nothing on the bot's dashboard had eyes on it until this surface.

``build_add_discipline`` walks the trade ledger chronologically per
position-key, maintains a running avg_cost (the same VWAP arithmetic
``store.upsert_position`` uses), and classifies each non-opening BUY:

* **CHASING** — add price ≥ ``CHASE_THRESHOLD_PCT`` above running cost.
  The bot bid up its average basis; the easy entry is behind it.
* **AVERAGING_DOWN** — add price ≤ ``-CHASE_THRESHOLD_PCT`` below running
  cost. The bot lowered its average basis; this is the textbook
  "doubling down" — rational if the thesis is intact, dangerous if it's
  not (the ``loser_autopsy`` SLOW_BLEED setup).
* **STACKING** — add price inside the band. Neutral — same conviction,
  same regime.

Then the closed-round-trip view: every closed trip is tagged by its
*dominant ADD style* (most frequent of CHASING / STACKING /
AVERAGING_DOWN among its ADDs; ties broken in that order — chasing
dominates the verdict because it's the riskiest behaviour). Per-style
aggregate P/L lets the operator answer the falsifiable question:

    *Did chasing-ADDs produce worse round-trip P/L than averaging-down
    ADDs? Or is averaging into broken theses bleeding more?*

Pure builder over already-stored trade rows + the closed round-trips from
``build_round_trips``. No DB read, no network. Never raises on garbage
rows. Observational only — never gates Opus, no caps (AGENTS.md #2/#12).

Sample-size honesty: ``NO_DATA`` (no ADDs in the ledger — every BUY was a
fresh open) → ``EMERGING`` (some ADDs but below ``STABLE_MIN_ADDS``,
counts emit but pattern verdict withheld) → ``STABLE`` (≥ STABLE_MIN_ADDS).
The closed-round-trip outcome rollup is its own gate: it emits when
``STABLE_MIN_OUTCOMES`` round-trips have been *tagged* with a dominant
style — a single CHASING trip's pnl_pct is one data point, not a pattern.
"""
from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any

# Above ±CHASE_THRESHOLD_PCT of the running avg_cost an ADD flips category.
# Wide enough that ordinary intraday price drift doesn't read as chasing
# (S&P daily 1-sigma ≈ 1%); narrow enough that a leveraged-ETF entry +5%
# above basis is unambiguously chasing.
CHASE_THRESHOLD_PCT = 1.5

# Below this many ADDs the per-ticker / overall mix is too thin for a
# headline-worthy pattern verdict; counts still emit (the EMERGING state).
STABLE_MIN_ADDS = 3

# Below this many tagged round-trips the per-style P/L comparison is
# noise. The round-trip rollup gates separately because not every ADD has
# yet closed into a round-trip.
STABLE_MIN_OUTCOMES = 3

# Category labels — kept as module-level constants so tests and callers
# can reference them without string drift.
CHASING = "CHASING"
STACKING = "STACKING"
AVERAGING_DOWN = "AVERAGING_DOWN"
_CATEGORIES = (CHASING, STACKING, AVERAGING_DOWN)

# Verdict precedence when a round-trip has multiple ADD styles: CHASING
# dominates (riskiest behaviour) → AVERAGING_DOWN → STACKING.
_DOMINANT_PRECEDENCE = (CHASING, AVERAGING_DOWN, STACKING)


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if x != x:  # NaN
        return None
    return float(x)


# A small inclusive-boundary epsilon. ``100 * 1.015 - 100 == 1.499…9`` in
# IEEE-754 float, so an honest "exactly at the threshold" caller would
# otherwise slip into STACKING. Loosen by ~1e-6 percentage points so a
# textbook chase at the band edge classifies as CHASING.
_BOUNDARY_EPS_PCT = 1e-6


def _classify(add_price: float, running_avg_cost: float) -> str:
    """Return the category for an ADD trade given the running avg_cost
    *immediately before* the ADD was applied. Inclusive at the band edges
    (modulo ``_BOUNDARY_EPS_PCT`` for float jitter). A non-positive basis
    falls through to STACKING — it cannot meaningfully be "above" or
    "below" a zero/negative anchor."""
    if running_avg_cost <= 0:
        return STACKING  # Defensive — cannot compute pct vs a non-positive
                         # basis. Categorise neutrally rather than emit
                         # bogus percentages.
    pct = (add_price - running_avg_cost) / running_avg_cost * 100.0
    if pct >= CHASE_THRESHOLD_PCT - _BOUNDARY_EPS_PCT:
        return CHASING
    if pct <= -CHASE_THRESHOLD_PCT + _BOUNDARY_EPS_PCT:
        return AVERAGING_DOWN
    return STACKING


def _pos_key(t: dict) -> tuple:
    """Same position-key tuple ``build_round_trips`` uses — kept identical
    so a future caller that joins ADDs back to a round-trip can do it on
    matching keys (single source of truth, AGENTS.md #10)."""
    typ = t.get("option_type") or "stock"
    return (t.get("ticker"), typ, t.get("strike"), t.get("expiry"))


def _safe_dominant(counts: Counter) -> str | None:
    """Pick the dominant category from a per-trip counter using the
    CHASING > AVERAGING_DOWN > STACKING precedence on ties. Returns None
    when no ADD was recorded (the round-trip had a single opening BUY)."""
    if not counts:
        return None
    top = max(counts.values())
    for cat in _DOMINANT_PRECEDENCE:
        if counts.get(cat, 0) == top:
            return cat
    return None


def build_add_discipline(trades: list[dict],
                         round_trips: list[dict] | None = None) -> dict:
    """Classify every ADD (non-opening BUY) and roll up per-style stats.

    ``trades`` is the ledger shape ``store.recent_trades()`` returns,
    chronologically oldest-first (the same input ``build_round_trips``
    expects). ``round_trips`` is the output of
    ``analytics.round_trips.build_round_trips`` on the same input;
    omitting it just suppresses the closed-trip P/L rollup but doesn't
    raise.

    Returns a dict carrying:
      * ``state`` ∈ {NO_DATA, EMERGING, STABLE}
      * ``n_buys_total`` — every BUY (opening + ADD)
      * ``n_opens`` — opening BUYs (qty-before == 0)
      * ``n_adds`` — ADDs (qty-before > 0)
      * ``counts`` — dict of {CATEGORY → count} across all ADDs
      * ``pct`` — dict of {CATEGORY → % of n_adds}
      * ``adds`` — chronological per-ADD rows: ticker, ts, price,
        running_avg_cost_before, pct_above_cost, category
      * ``by_ticker`` — per-ticker mini-counter + n_total_adds
      * ``closed_outcomes`` — list of per-closed-round-trip dicts:
        ticker, entry_ts, exit_ts, n_adds_in_trip, dominant_style,
        pnl_usd, pnl_pct
      * ``outcomes_by_style`` — {style → {n, total_pnl_usd, mean_pnl_pct,
        median_pnl_pct}} for trips whose ``dominant_style`` is set
      * ``dominant_style_overall`` — most-frequent category across all
        ADDs (the headline ADD personality of the book; only emitted in
        the STABLE state, withheld below STABLE_MIN_ADDS)
      * ``chase_threshold_pct`` — the band the classification used (so
        the response is self-describing)
      * ``stable_min_adds`` / ``stable_min_outcomes`` — echoed gates
      * ``headline`` — one-line operator headline

    Never raises. A malformed row degrades the row (it doesn't count
    toward the categorisation) — the contract is "no verdict on bad
    data", never an exception.
    """
    rows = trades if isinstance(trades, list) else []

    # Per-position cumulative state — same VWAP as ``store.upsert_position``.
    cost_basis: dict[tuple, float] = {}   # avg_cost
    qty_held: dict[tuple, float] = {}     # shares (or contracts) currently open

    adds: list[dict] = []
    counts = Counter()
    by_ticker: dict[str, Counter] = {}
    by_ticker_total: dict[str, int] = {}
    n_buys_total = 0
    n_opens = 0
    n_adds = 0

    # Build a trade_id → category map so the closed-trip rollup can join
    # back via ``entry_trade_ids`` (the SSOT for per-trip BUY membership).
    trade_id_to_category: dict[Any, str] = {}

    for t in rows:
        if not isinstance(t, dict):
            continue
        action = (t.get("action") or "").upper()
        if not action.startswith("BUY") and not action.startswith("SELL"):
            continue
        ticker = (t.get("ticker") or "")
        key = _pos_key(t)
        price = _num(t.get("price"))
        qty = _num(t.get("qty"))
        if price is None or qty is None or qty <= 0:
            # Can't classify or update basis without price + qty. Skip;
            # this trade simply doesn't contribute (degrade-not-raise).
            continue

        if action.startswith("SELL"):
            # SELL reduces held qty. If held returns to zero we reset basis
            # so the next BUY is an "open", not an ADD. This mirrors
            # ``build_round_trips``'s round-trip cycling so the two views
            # never disagree on what counts as "open vs add".
            held_before = qty_held.get(key, 0.0)
            held_after = held_before - qty
            if abs(held_after) < 1e-4 or held_after <= 0:
                qty_held[key] = 0.0
                cost_basis[key] = 0.0
            else:
                qty_held[key] = held_after
            continue

        # action starts with BUY (covers BUY, BUY_CALL, BUY_PUT etc.)
        n_buys_total += 1
        held_before = qty_held.get(key, 0.0)
        avg_before = cost_basis.get(key, 0.0)
        if held_before <= 1e-4:
            # Opening BUY — no ADD classification.
            n_opens += 1
            qty_held[key] = qty
            cost_basis[key] = price
            continue
        # ADD path — classify using basis BEFORE this trade is applied.
        category = _classify(price, avg_before)
        n_adds += 1
        counts[category] += 1
        if ticker:
            by_ticker.setdefault(ticker, Counter())[category] += 1
            by_ticker_total[ticker] = by_ticker_total.get(ticker, 0) + 1
        pct_above = round(
            (price - avg_before) / avg_before * 100.0, 3
        ) if avg_before > 0 else None
        adds.append({
            "ticker": ticker,
            "type": t.get("option_type") or "stock",
            "strike": t.get("strike"),
            "expiry": t.get("expiry"),
            "ts": t.get("timestamp"),
            "price": round(price, 4),
            "qty": round(qty, 6),
            "running_avg_cost_before": round(avg_before, 4),
            "pct_above_cost": pct_above,
            "category": category,
        })
        tid = t.get("id")
        if tid is not None:
            trade_id_to_category[tid] = category
        # Update VWAP after the ADD (Σ qty·price / Σ qty).
        new_qty = held_before + qty
        new_avg = ((avg_before * held_before) + (price * qty)) / new_qty
        qty_held[key] = new_qty
        cost_basis[key] = new_avg

    # Per-ticker rollup as dicts (Counter→dict so the JSON is plain).
    by_ticker_out: list[dict] = []
    for tk, ctr in by_ticker.items():
        by_ticker_out.append({
            "ticker": tk,
            "n_adds": by_ticker_total[tk],
            "counts": {c: ctr.get(c, 0) for c in _CATEGORIES},
            "dominant": _safe_dominant(ctr),
        })
    # Sort: most ADDs first, ties by ticker for determinism.
    by_ticker_out.sort(key=lambda r: (-r["n_adds"], r["ticker"]))

    # Closed-round-trip outcome rollup. A trip's dominant style is the
    # mode of its ADDs' categories under the CHASING > AVG_DOWN > STACK
    # precedence (so chasing isn't masked by a single counter-balancing
    # average-down). Trips with no ADDs at all (single-BUY open→close)
    # are emitted with dominant_style=None.
    closed_outcomes: list[dict] = []
    for rt in (round_trips or []):
        if not isinstance(rt, dict):
            continue
        entry_ids = rt.get("entry_trade_ids") or []
        trip_counts: Counter = Counter()
        n_adds_in_trip = 0
        for tid in entry_ids:
            cat = trade_id_to_category.get(tid)
            if cat is not None:
                trip_counts[cat] += 1
                n_adds_in_trip += 1
        dom = _safe_dominant(trip_counts) if n_adds_in_trip else None
        closed_outcomes.append({
            "ticker": rt.get("ticker"),
            "entry_ts": rt.get("entry_ts"),
            "exit_ts": rt.get("exit_ts"),
            "n_adds_in_trip": n_adds_in_trip,
            "dominant_style": dom,
            "pnl_usd": _num(rt.get("pnl_usd")),
            "pnl_pct": _num(rt.get("pnl_pct")),
        })

    # outcomes_by_style: aggregate over trips with a non-None
    # dominant_style — those are the trips whose P/L is *attributable* to
    # one of the three ADD personalities.
    outcomes_by_style: dict[str, dict] = {}
    for cat in _CATEGORIES:
        trips_in_cat = [
            o for o in closed_outcomes
            if o["dominant_style"] == cat
            and o["pnl_usd"] is not None
            and o["pnl_pct"] is not None
        ]
        if not trips_in_cat:
            outcomes_by_style[cat] = {
                "n": 0,
                "total_pnl_usd": 0.0,
                "mean_pnl_pct": None,
                "median_pnl_pct": None,
            }
            continue
        pcts = [o["pnl_pct"] for o in trips_in_cat]
        outcomes_by_style[cat] = {
            "n": len(trips_in_cat),
            "total_pnl_usd": round(
                sum(o["pnl_usd"] for o in trips_in_cat), 4),
            "mean_pnl_pct": round(mean(pcts), 4),
            "median_pnl_pct": round(median(pcts), 4),
        }

    # State + headline.
    if n_adds == 0:
        state = "NO_DATA"
        dominant_style_overall = None
        if n_buys_total == 0:
            headline = (
                "no BUY trades on file yet — add-discipline analysis "
                "not yet available."
            )
        else:
            headline = (
                f"{n_buys_total} BUY(s) on file but no ADDs — every BUY "
                "opened a fresh position, no chasing/averaging-down to "
                "evaluate."
            )
    elif n_adds < STABLE_MIN_ADDS:
        state = "EMERGING"
        dominant_style_overall = None  # withheld below the stable gate
        # Compose a short emerging headline so the dashboard / chat have
        # something specific to render (the counts are real, only the
        # pattern verdict is withheld).
        counts_str = ", ".join(
            f"{c}={counts.get(c, 0)}" for c in _CATEGORIES if counts.get(c, 0)
        )
        headline = (
            f"{n_adds}/{STABLE_MIN_ADDS} ADD(s) for a stable mix verdict "
            f"({counts_str or 'mixed'})."
        )
    else:
        state = "STABLE"
        # Dominant style overall — use the precedence to break ties so
        # chasing isn't masked when chasing-count == averaging-down-count.
        dominant_style_overall = _safe_dominant(counts)
        share = round(
            counts.get(dominant_style_overall, 0) / n_adds * 100.0, 1
        ) if dominant_style_overall else None
        headline = (
            f"{n_adds} ADD(s) — dominant style {dominant_style_overall} "
            f"({share:.1f}%); chasing={counts.get(CHASING, 0)}, "
            f"averaging_down={counts.get(AVERAGING_DOWN, 0)}, "
            f"stacking={counts.get(STACKING, 0)}."
        )

    pct = {
        c: (round(counts.get(c, 0) / n_adds * 100.0, 2) if n_adds > 0 else 0.0)
        for c in _CATEGORIES
    }

    return {
        "state": state,
        "n_buys_total": n_buys_total,
        "n_opens": n_opens,
        "n_adds": n_adds,
        "counts": {c: counts.get(c, 0) for c in _CATEGORIES},
        "pct": pct,
        "adds": adds,
        "by_ticker": by_ticker_out,
        "closed_outcomes": closed_outcomes,
        "outcomes_by_style": outcomes_by_style,
        "dominant_style_overall": dominant_style_overall,
        "chase_threshold_pct": CHASE_THRESHOLD_PCT,
        "stable_min_adds": STABLE_MIN_ADDS,
        "stable_min_outcomes": STABLE_MIN_OUTCOMES,
        "headline": headline,
    }
