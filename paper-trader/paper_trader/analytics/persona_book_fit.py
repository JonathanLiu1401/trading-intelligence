"""Persona-Book Fit — does the live portfolio look like a persona that
actually carries alpha, or like one that's a documented drag?

The operator runs a 10-persona backtest committee. ``/api/persona-leaderboard``
already grades each persona EDGE / FLAT / DRAG / INSUFFICIENT from historical
``backtest_runs``. But there is no surface that closes the loop on the *live*
book: "this open portfolio looks most like persona X — which is rated Y." A
book whose weight distribution mirrors a DRAG persona is structurally adding
variance not alpha, even if every individual trade looked reasonable at the
time. That mismatch is invisible from every other panel on the desk.

Pure & offline. The builder takes the open positions, the persona archetype
weight maps (``_PERSONA_BOOSTS``), the persona name map (``PERSONAS``), and
the persona-leaderboard report verbatim. It overlaps the book's
market-value-weight distribution onto each persona's boost-weighted ticker
set, picks the dominant persona, looks that persona's verdict up in the
leaderboard, and emits a single ALIGNED_EDGE / ALIGNED_FLAT / ALIGNED_DRAG /
INSUFFICIENT_PERSONA / WEAK_OVERLAP / NO_BOOK verdict. No yfinance, no
articles.db, no scorer call — same module-pure contract as ``desk_pulse``.

Advisory only. It mints no directive, imposes no cap, and has no path to
``_execute()`` (AGENTS.md invariants #2 / #12 — Opus has full autonomy; this
is a mirror, not a cage). Never raises — a malformed leaderboard, an empty
book, or a totally idiosyncratic book all degrade to a deterministic
"insufficient" / "no_book" / "weak_overlap" verdict, never an exception.

The dominant-persona overlap rule. For each persona p with boost map
``boosts[p] = {ticker: boost}``, the score is

    score(p) = Σ_{t ∈ book ∩ boosts[p]}  book_weight[t] × boosts[p][t]

where ``book_weight[t]`` is the position's market value as a fraction of
total book equity (cash excluded — cash is not a persona signature). The
dominant persona is the argmax; ties break alphabetically by persona name
(deterministic, cycle-to-cycle stable). The associated **overlap fraction**
(plain Σ book_weight of boost-tickers — no multiplication by boost
magnitude) gates the WEAK_OVERLAP verdict: a book that has < ``MIN_DOMINANT_OVERLAP_PCT``
of its equity in *any* persona's boost set is idiosyncratic — not
mis-aligned, just outside the persona taxonomy — and the verdict is
``WEAK_OVERLAP``, not ``ALIGNED_*``.

The verdict ladder (threshold-driven, exactly testable):

| Verdict             | Meaning                                                                                       |
|---------------------|-----------------------------------------------------------------------------------------------|
| ``NO_BOOK``         | No open stock/ETF positions — no signature to fit                                             |
| ``WEAK_OVERLAP``    | Best-fit overlap < ``MIN_DOMINANT_OVERLAP_PCT`` — book matches no persona archetype           |
| ``INSUFFICIENT_PERSONA`` | Dominant persona has ``< MIN_RUNS_PER_PERSONA`` runs in leaderboard — verdict unstable    |
| ``ALIGNED_DRAG``    | Dominant persona is rated DRAG (median vs_spy ≤ 0) — book mirrors a known underperformer     |
| ``ALIGNED_FLAT``    | Dominant persona is rated FLAT — positive but weak edge                                       |
| ``ALIGNED_EDGE``    | Dominant persona is rated EDGE — book mirrors a known alpha-positive persona                  |

On ``ALIGNED_DRAG`` only, the report additionally surfaces up to
``TOP_EDGE_ALTERNATIVES`` EDGE-rated personas the book does NOT resemble —
the natural rotation hint. This is data for an operator decision, not a
trade recommendation — the engine still picks tickers.
"""
from __future__ import annotations

# Minimum book-equity-weight overlap (sum of book_weight over tickers in
# the persona's boost set) to consider any persona match credible. Below
# this the book is idiosyncratic, not mis-aligned.
MIN_DOMINANT_OVERLAP_PCT = 30.0
# Mirror persona_leaderboard.MIN_RUNS_PER_PERSONA — fewer than this and the
# dominant persona's leaderboard verdict is too small-sample to act on.
# Intentionally hardcoded (not imported) so the test reads this constant
# and a leaderboard retune cannot false-fail this fit verdict.
MIN_RUNS_PER_PERSONA = 5
# When ALIGNED_DRAG fires, surface this many EDGE personas as rotation
# hints — keeps the chat block tight (the briefing's _coverage_gap_lines
# precedent).
TOP_EDGE_ALTERNATIVES = 3


def _market_value(position) -> float:
    """Stocks only — options have an inherent leverage signature that does
    not align with the boost-weighted archetype model. Returns 0.0 for a
    malformed row or an option position; never raises.
    """
    try:
        ptype = (position.get("type") or "stock").lower()
        if ptype in ("call", "put"):
            return 0.0  # option positions don't carry a persona signature
        price = position.get("current_price") or position.get("avg_cost") or 0.0
        qty = position.get("qty") or 0.0
        v = float(price) * float(qty)
        return v if v > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _book_weights(positions) -> tuple[dict[str, float], float]:
    """Return (ticker → market_value_weight_pct, total_stock_value).

    Weights sum to 100.0 over the stock leg. Cash is intentionally NOT in
    the weight basis: cash has no persona signature. An options-only book
    returns ({}, 0.0) and the caller emits NO_BOOK.
    """
    raw: dict[str, float] = {}
    for p in positions or []:
        if not isinstance(p, dict):
            continue
        ticker = (p.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        mv = _market_value(p)
        if mv <= 0:
            continue
        raw[ticker] = raw.get(ticker, 0.0) + mv
    total = sum(raw.values())
    if total <= 0:
        return {}, 0.0
    return ({t: round(mv / total * 100.0, 4) for t, mv in raw.items()}, total)


def _persona_scores(book_weights, persona_archetypes):
    """For each persona, return (raw_score, plain_overlap_pct).

    raw_score = Σ_{t ∈ book ∩ boosts[p]} book_weight[t] × boost
    plain_overlap_pct = Σ_{t ∈ book ∩ boosts[p]} book_weight[t]

    Deterministic — both are pure sums over an iteration order that is
    not exposed.
    """
    out: dict[int, dict[str, float]] = {}
    for p_idx, boosts in (persona_archetypes or {}).items():
        score = 0.0
        overlap = 0.0
        if not isinstance(boosts, dict):
            out[p_idx] = {"raw_score": 0.0, "overlap_pct": 0.0,
                          "matched_tickers": []}
            continue
        matched = []
        for ticker, weight in book_weights.items():
            boost = boosts.get(ticker)
            if boost is None:
                continue
            try:
                b = float(boost)
            except (TypeError, ValueError):
                continue
            score += weight * b
            overlap += weight
            matched.append({"ticker": ticker, "book_weight_pct": round(weight, 4),
                            "boost": round(b, 4)})
        matched.sort(key=lambda d: (-d["book_weight_pct"], d["ticker"]))
        out[p_idx] = {
            "raw_score": round(score, 4),
            "overlap_pct": round(overlap, 4),
            "matched_tickers": matched,
        }
    return out


def _leaderboard_lookup(leaderboard_report) -> dict[str, dict]:
    """Persona name → leaderboard row. Tolerates missing report / malformed
    rows — every accessor downstream uses ``.get`` with safe defaults.
    """
    out: dict[str, dict] = {}
    if not isinstance(leaderboard_report, dict):
        return out
    rows = leaderboard_report.get("leaderboard") or []
    if not isinstance(rows, list):
        return out
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = r.get("persona")
        if isinstance(name, str) and name:
            out[name] = r
    return out


def build_persona_book_fit(positions, persona_archetypes, persona_names,
                            leaderboard_report):
    """Compute the book→persona fit verdict.

    Args:
        positions: list of dicts shaped like ``store.open_positions()`` rows
            ({ticker, qty, type, current_price, avg_cost, ...}).
        persona_archetypes: ``{persona_idx: {ticker: boost_float}}`` — the
            ``_PERSONA_BOOSTS`` constant from ``paper_trader.backtest``.
        persona_names: ``{persona_idx: name}`` — derived from
            ``PERSONAS[i]["name"]``.
        leaderboard_report: output dict of
            ``paper_trader.ml.persona_leaderboard.persona_leaderboard``
            (the report itself; verdict + leaderboard rows).

    Returns:
        A plain dict — see the verdict ladder in the module docstring.
        Never raises.
    """
    book_weights, total_book = _book_weights(positions)
    if not book_weights:
        return {
            "status": "no_book",
            "verdict": "NO_BOOK",
            "headline": ("NO_BOOK — no open stock positions to fit against "
                         "the persona archetypes (cash-only or options-only "
                         "book has no persona signature)."),
            "dominant": None,
            "alternatives": [],
            "n_personas_scored": 0,
            "book_total_value": round(total_book, 4),
        }

    scores = _persona_scores(book_weights, persona_archetypes or {})
    if not scores:
        return {
            "status": "no_archetypes",
            "verdict": "WEAK_OVERLAP",
            "headline": ("WEAK_OVERLAP — no persona archetype data; the "
                         "book has no comparable model."),
            "dominant": None,
            "alternatives": [],
            "n_personas_scored": 0,
            "book_total_value": round(total_book, 4),
        }

    # Pick the persona with the highest raw_score; tiebreak alphabetically
    # on persona name (deterministic; the persona_leaderboard sort
    # precedent of "stable order under ties").
    def _sort_key(item):
        p_idx, d = item
        name = (persona_names or {}).get(p_idx) or f"persona_{p_idx}"
        return (-d["raw_score"], name)

    ordered = sorted(scores.items(), key=_sort_key)
    dom_idx, dom = ordered[0]
    dom_name = (persona_names or {}).get(dom_idx) or f"persona_{dom_idx}"
    overlap_pct = dom["overlap_pct"]

    # Always report the runner-up too — it's the natural counterfactual for
    # the operator ("you look most like X, but you're 30% Y too").
    if len(ordered) >= 2:
        ru_idx, ru = ordered[1]
        runner_up = {
            "persona": (persona_names or {}).get(ru_idx) or f"persona_{ru_idx}",
            "raw_score": ru["raw_score"],
            "overlap_pct": ru["overlap_pct"],
        }
    else:
        runner_up = None

    lb = _leaderboard_lookup(leaderboard_report)
    dom_row = lb.get(dom_name)

    base_payload = {
        "n_personas_scored": len(scores),
        "book_total_value": round(total_book, 4),
        "book_weights": [
            {"ticker": t, "weight_pct": w}
            for t, w in sorted(book_weights.items(), key=lambda kv: -kv[1])
        ],
        "dominant": {
            "persona": dom_name,
            "raw_score": dom["raw_score"],
            "overlap_pct": overlap_pct,
            "matched_tickers": dom["matched_tickers"],
            "leaderboard_row": dom_row,
        },
        "runner_up": runner_up,
        "alternatives": [],
    }

    if overlap_pct < MIN_DOMINANT_OVERLAP_PCT:
        return {
            **base_payload,
            "status": "weak_overlap",
            "verdict": "WEAK_OVERLAP",
            "headline": (
                f"WEAK_OVERLAP — best-fit persona ‘{dom_name}’ overlaps only "
                f"{overlap_pct:.1f}% of book equity (< {MIN_DOMINANT_OVERLAP_PCT:.0f}% "
                f"threshold). Book is idiosyncratic — no persona archetype is "
                f"a credible comparator."),
        }

    if dom_row is None:
        return {
            **base_payload,
            "status": "insufficient_persona",
            "verdict": "INSUFFICIENT_PERSONA",
            "headline": (
                f"INSUFFICIENT_PERSONA — book most resembles ‘{dom_name}’ "
                f"({overlap_pct:.1f}% overlap) but the leaderboard has no "
                f"row for that persona. Need backtest runs of this persona "
                f"to validate fit."),
        }

    dom_verdict = (dom_row.get("verdict") or "").upper()
    n_runs = dom_row.get("n") or 0
    med_vs_spy = dom_row.get("median_vs_spy")

    if dom_verdict == "INSUFFICIENT" or n_runs < MIN_RUNS_PER_PERSONA:
        return {
            **base_payload,
            "status": "insufficient_persona",
            "verdict": "INSUFFICIENT_PERSONA",
            "headline": (
                f"INSUFFICIENT_PERSONA — book mirrors ‘{dom_name}’ "
                f"({overlap_pct:.1f}% overlap) but only {n_runs} run(s) "
                f"in the leaderboard for that persona (need ≥{MIN_RUNS_PER_PERSONA})."),
        }

    if dom_verdict == "DRAG":
        # Surface EDGE alternatives the book does NOT mirror — the natural
        # rotation hint. Sort by median_vs_spy desc; drop tickers that
        # already overlap heavily (don't recommend what's already in book).
        alts = []
        for row in (leaderboard_report or {}).get("leaderboard") or []:
            if not isinstance(row, dict):
                continue
            if (row.get("verdict") or "").upper() != "EDGE":
                continue
            name = row.get("persona")
            if not isinstance(name, str) or name == dom_name:
                continue
            alts.append({
                "persona": name,
                "median_vs_spy": row.get("median_vs_spy"),
                "win_rate": row.get("win_rate"),
                "n_runs": row.get("n"),
            })
            if len(alts) >= TOP_EDGE_ALTERNATIVES:
                break

        med_str = f"{med_vs_spy:+.1f}pp" if isinstance(med_vs_spy, (int, float)) else "—"
        alt_str = (", ".join(a["persona"] for a in alts)) if alts else "—"
        return {
            **base_payload,
            "status": "aligned_drag",
            "verdict": "ALIGNED_DRAG",
            "headline": (
                f"ALIGNED_DRAG — book most resembles ‘{dom_name}’ "
                f"({overlap_pct:.1f}% overlap, n={n_runs}, median vs_spy "
                f"{med_str}); leaderboard rates this persona DRAG. "
                f"EDGE personas the book does NOT resemble: {alt_str}."),
            "alternatives": alts,
        }

    if dom_verdict == "EDGE":
        med_str = f"{med_vs_spy:+.1f}pp" if isinstance(med_vs_spy, (int, float)) else "—"
        return {
            **base_payload,
            "status": "aligned_edge",
            "verdict": "ALIGNED_EDGE",
            "headline": (
                f"ALIGNED_EDGE — book most resembles ‘{dom_name}’ "
                f"({overlap_pct:.1f}% overlap, n={n_runs}, median vs_spy "
                f"{med_str}); leaderboard rates this persona EDGE."),
        }

    if dom_verdict == "FLAT":
        med_str = f"{med_vs_spy:+.1f}pp" if isinstance(med_vs_spy, (int, float)) else "—"
        return {
            **base_payload,
            "status": "aligned_flat",
            "verdict": "ALIGNED_FLAT",
            "headline": (
                f"ALIGNED_FLAT — book most resembles ‘{dom_name}’ "
                f"({overlap_pct:.1f}% overlap, n={n_runs}, median vs_spy "
                f"{med_str}); leaderboard rates this persona FLAT — positive "
                f"but weak edge."),
        }

    # Unknown verdict string — degrade to INSUFFICIENT_PERSONA rather than
    # raise (the leaderboard verdict enum may grow without breaking this).
    return {
        **base_payload,
        "status": "insufficient_persona",
        "verdict": "INSUFFICIENT_PERSONA",
        "headline": (
            f"INSUFFICIENT_PERSONA — book mirrors ‘{dom_name}’ but the "
            f"leaderboard row has an unknown verdict "
            f"({dom_row.get('verdict')!r})."),
    }
