"""Unit tests for paper_trader.analytics.persona_book_fit.

Locks the verdict ladder (NO_BOOK / WEAK_OVERLAP / INSUFFICIENT_PERSONA /
ALIGNED_DRAG / ALIGNED_FLAT / ALIGNED_EDGE), the dominant-persona overlap
math, the alternatives-on-DRAG behaviour, and the never-raises contract.
The builder is pure — no DB / yfinance / scorer reach — so these tests are
fast and assert exact outputs.
"""
from __future__ import annotations

from paper_trader.analytics.persona_book_fit import (
    MIN_DOMINANT_OVERLAP_PCT,
    MIN_RUNS_PER_PERSONA,
    TOP_EDGE_ALTERNATIVES,
    build_persona_book_fit,
)


# Two synthetic personas. ``MOMENTUM`` boosts the 3x ETFs (SOXL / TQQQ);
# ``VALUE`` boosts the boring compounders (MSFT / JPM).
_ARCHETYPES = {
    1: {"MSFT": 1.5, "JPM": 1.5, "GOOGL": 1.5},
    2: {"SOXL": 4.0, "TQQQ": 3.5, "QQQ": 1.5},
}
_NAMES = {1: "Value Investor", 2: "Momentum Trader"}


def _leaderboard(verdicts_by_persona):
    """Synthesise a persona_leaderboard report with the given per-persona
    verdicts. Returns the wrapping report dict shape.
    """
    rows = []
    for name, verdict in verdicts_by_persona.items():
        rows.append({
            "persona": name,
            "n": 50,
            "median_vs_spy": {"DRAG": -10.0, "FLAT": 5.0, "EDGE": 35.0,
                              "INSUFFICIENT": 0.0}[verdict],
            "win_rate": {"DRAG": 0.3, "FLAT": 0.45, "EDGE": 0.65,
                          "INSUFFICIENT": 0.5}[verdict],
            "verdict": verdict,
        })
    return {"status": "ok", "verdict": "HEALTHY", "leaderboard": rows}


# ─── verdict ladder ──────────────────────────────────────────────────────


def test_no_book_when_positions_empty():
    out = build_persona_book_fit(
        positions=[],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report=_leaderboard({"Momentum Trader": "EDGE"}),
    )
    assert out["verdict"] == "NO_BOOK"
    assert out["dominant"] is None
    assert out["book_total_value"] == 0.0


def test_no_book_when_only_options():
    # Options have ``type`` ∈ {call, put} — they don't carry a persona signature.
    out = build_persona_book_fit(
        positions=[{"ticker": "NVDA", "qty": 1, "type": "call",
                    "current_price": 5.0, "avg_cost": 4.0}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report=_leaderboard({"Momentum Trader": "EDGE"}),
    )
    assert out["verdict"] == "NO_BOOK"


def test_weak_overlap_when_book_has_no_archetype_tickers():
    # NFLX is in neither persona's boost set ⇒ 0% overlap.
    out = build_persona_book_fit(
        positions=[{"ticker": "NFLX", "qty": 10, "type": "stock",
                    "current_price": 100.0}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report=_leaderboard({"Momentum Trader": "EDGE"}),
    )
    assert out["verdict"] == "WEAK_OVERLAP"
    assert out["dominant"]["overlap_pct"] == 0.0


def test_aligned_edge_when_dominant_persona_is_edge():
    # Book is 100% SOXL → matches Momentum Trader at full overlap.
    out = build_persona_book_fit(
        positions=[{"ticker": "SOXL", "qty": 10, "type": "stock",
                    "current_price": 50.0}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report=_leaderboard({"Momentum Trader": "EDGE"}),
    )
    assert out["verdict"] == "ALIGNED_EDGE"
    assert out["dominant"]["persona"] == "Momentum Trader"
    assert out["dominant"]["overlap_pct"] == 100.0


def test_aligned_drag_surfaces_top_edge_alternatives():
    # Momentum is DRAG; Value is EDGE — the rotation hint.
    out = build_persona_book_fit(
        positions=[{"ticker": "SOXL", "qty": 10, "type": "stock",
                    "current_price": 50.0}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report=_leaderboard({"Momentum Trader": "DRAG",
                                          "Value Investor": "EDGE"}),
    )
    assert out["verdict"] == "ALIGNED_DRAG"
    assert out["dominant"]["persona"] == "Momentum Trader"
    alt_names = [a["persona"] for a in out["alternatives"]]
    assert "Value Investor" in alt_names
    # Never recommend the dominant persona itself as an alternative.
    assert "Momentum Trader" not in alt_names
    assert len(out["alternatives"]) <= TOP_EDGE_ALTERNATIVES


def test_aligned_flat_when_dominant_persona_is_flat():
    out = build_persona_book_fit(
        positions=[{"ticker": "SOXL", "qty": 10, "type": "stock",
                    "current_price": 50.0}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report=_leaderboard({"Momentum Trader": "FLAT"}),
    )
    assert out["verdict"] == "ALIGNED_FLAT"


def test_insufficient_persona_when_leaderboard_says_insufficient():
    out = build_persona_book_fit(
        positions=[{"ticker": "SOXL", "qty": 10, "type": "stock",
                    "current_price": 50.0}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report=_leaderboard({"Momentum Trader": "INSUFFICIENT"}),
    )
    assert out["verdict"] == "INSUFFICIENT_PERSONA"


def test_insufficient_persona_when_n_runs_below_threshold():
    rows = [{"persona": "Momentum Trader", "n": MIN_RUNS_PER_PERSONA - 1,
             "median_vs_spy": 5.0, "win_rate": 0.5, "verdict": "FLAT"}]
    out = build_persona_book_fit(
        positions=[{"ticker": "SOXL", "qty": 10, "type": "stock",
                    "current_price": 50.0}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report={"status": "ok", "leaderboard": rows},
    )
    assert out["verdict"] == "INSUFFICIENT_PERSONA"


def test_insufficient_persona_when_leaderboard_missing_dominant():
    # Leaderboard exists but has no row for the dominant persona.
    out = build_persona_book_fit(
        positions=[{"ticker": "SOXL", "qty": 10, "type": "stock",
                    "current_price": 50.0}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report={"status": "ok", "leaderboard": [
            {"persona": "Value Investor", "n": 50,
             "median_vs_spy": 30.0, "win_rate": 0.6, "verdict": "EDGE"},
        ]},
    )
    assert out["verdict"] == "INSUFFICIENT_PERSONA"


# ─── dominant-persona math ───────────────────────────────────────────────


def test_dominant_persona_is_argmax_of_weighted_overlap_score():
    # Book: 50% MSFT (Value boost 1.5) + 50% SOXL (Momentum boost 4.0).
    # Raw scores: Value = 50*1.5 = 75. Momentum = 50*4.0 = 200.
    # Momentum should win.
    out = build_persona_book_fit(
        positions=[{"ticker": "MSFT", "qty": 1, "type": "stock", "current_price": 100.0},
                   {"ticker": "SOXL", "qty": 1, "type": "stock", "current_price": 100.0}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report=_leaderboard({"Momentum Trader": "EDGE",
                                          "Value Investor": "EDGE"}),
    )
    assert out["dominant"]["persona"] == "Momentum Trader"


def test_runner_up_is_reported_for_counterfactual():
    out = build_persona_book_fit(
        positions=[{"ticker": "MSFT", "qty": 1, "type": "stock", "current_price": 100.0},
                   {"ticker": "SOXL", "qty": 1, "type": "stock", "current_price": 100.0}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report=_leaderboard({"Momentum Trader": "EDGE",
                                          "Value Investor": "EDGE"}),
    )
    assert out["runner_up"]["persona"] == "Value Investor"


def test_weak_overlap_threshold_is_exact():
    # Force overlap just below MIN_DOMINANT_OVERLAP_PCT — should be WEAK_OVERLAP.
    # Book: SOXL is 29% (below threshold) + NFLX 71% (no archetype match).
    out = build_persona_book_fit(
        positions=[{"ticker": "SOXL", "qty": 29, "type": "stock", "current_price": 1.0},
                   {"ticker": "NFLX", "qty": 71, "type": "stock", "current_price": 1.0}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report=_leaderboard({"Momentum Trader": "EDGE"}),
    )
    assert out["verdict"] == "WEAK_OVERLAP"
    assert out["dominant"]["overlap_pct"] < MIN_DOMINANT_OVERLAP_PCT

    # Just above ⇒ ALIGNED_EDGE.
    out2 = build_persona_book_fit(
        positions=[{"ticker": "SOXL", "qty": 31, "type": "stock", "current_price": 1.0},
                   {"ticker": "NFLX", "qty": 69, "type": "stock", "current_price": 1.0}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report=_leaderboard({"Momentum Trader": "EDGE"}),
    )
    assert out2["verdict"] == "ALIGNED_EDGE"


def test_book_weights_sum_to_100_pct():
    out = build_persona_book_fit(
        positions=[{"ticker": "MSFT", "qty": 1, "type": "stock", "current_price": 300.0},
                   {"ticker": "SOXL", "qty": 2, "type": "stock", "current_price": 100.0}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report=_leaderboard({"Momentum Trader": "EDGE"}),
    )
    total = sum(w["weight_pct"] for w in out["book_weights"])
    assert abs(total - 100.0) < 0.01


def test_drag_alternatives_does_not_include_dominant_persona():
    # Both personas DRAG except the dominant ⇒ no alternatives.
    out = build_persona_book_fit(
        positions=[{"ticker": "SOXL", "qty": 10, "type": "stock",
                    "current_price": 50.0}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report=_leaderboard({"Momentum Trader": "DRAG",
                                          "Value Investor": "DRAG"}),
    )
    assert out["verdict"] == "ALIGNED_DRAG"
    assert out["alternatives"] == []


# ─── robustness ──────────────────────────────────────────────────────────


def test_never_raises_on_malformed_inputs():
    # None archetypes, None leaderboard, garbage positions.
    out = build_persona_book_fit(
        positions=[{"junk": "row"}, None, {"ticker": ""}],
        persona_archetypes=None,
        persona_names=None,
        leaderboard_report=None,
    )
    assert out["verdict"] == "NO_BOOK"


def test_never_raises_on_non_numeric_prices():
    out = build_persona_book_fit(
        positions=[{"ticker": "SOXL", "qty": "ten", "type": "stock",
                    "current_price": "lots"}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report=_leaderboard({"Momentum Trader": "EDGE"}),
    )
    assert out["verdict"] == "NO_BOOK"


def test_aligned_edge_headline_includes_persona_and_overlap():
    out = build_persona_book_fit(
        positions=[{"ticker": "SOXL", "qty": 10, "type": "stock",
                    "current_price": 50.0}],
        persona_archetypes=_ARCHETYPES,
        persona_names=_NAMES,
        leaderboard_report=_leaderboard({"Momentum Trader": "EDGE"}),
    )
    h = out["headline"]
    assert "ALIGNED_EDGE" in h
    assert "Momentum Trader" in h
    assert "EDGE" in h


# ─── Flask route ─────────────────────────────────────────────────────────


class TestPersonaBookFitRoute:
    def test_route_returns_well_formed_envelope(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get("/api/persona-book-fit")
        assert resp.status_code in (200, 500), resp.status_code
        body = resp.get_json()
        assert isinstance(body, dict)
        # Must always carry a verdict + headline, regardless of inner state.
        assert "verdict" in body
        assert "headline" in body
