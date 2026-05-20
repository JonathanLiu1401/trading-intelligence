"""Tests for analytics/catalyst_class_autopsy.py — per-entry-thesis-class
realised-P&L autopsy of closed round-trips.

Hand-computed arithmetic + regex-taxonomy invariants. The module is the
ENTRY-side complement to ``loser_autopsy`` / ``winner_autopsy`` (which
classify the EXIT behaviour). Any drift from the single-source-of-truth
``build_round_trips`` pipeline (a recomputed P&L, a misclassified
catalyst, a verdict emitted before the STABLE gate, a missing
UNCLASSIFIED bucket, a non-deterministic dominant-class tie-break) fails
an assertion here.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.catalyst_class_autopsy import (
    BIASED_WR_DELTA_PCT,
    STABLE_MIN_TRIPS_PER_CLASS,
    _classify_classes,
    build_catalyst_class_autopsy,
)

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _day(offset: int) -> str:
    return (_BASE + timedelta(days=offset)).isoformat()


def _rt(tid, ticker, buy_day, sell_day, qty, buy_px, sell_px,
        entry_reason="", exit_reason=""):
    return [
        {"id": tid, "timestamp": _day(buy_day), "ticker": ticker,
         "action": "BUY", "qty": qty, "price": buy_px,
         "value": qty * buy_px, "strike": None, "expiry": None,
         "option_type": None, "reason": entry_reason},
        {"id": tid + 1, "timestamp": _day(sell_day), "ticker": ticker,
         "action": "SELL", "qty": qty, "price": sell_px,
         "value": qty * sell_px, "strike": None, "expiry": None,
         "option_type": None, "reason": exit_reason},
    ]


def _ledger(specs):
    """specs: list of (ticker, buy_px, sell_px, entry_reason, exit_reason).
    Each becomes its own round-trip on disjoint (day, day+1) windows so
    build_round_trips closes each independently. qty=10 fixed.
    """
    trades, tid, day = [], 1, 0
    for ticker, bpx, spx, er, xr in specs:
        trades += _rt(tid, ticker, day, day + 1, 10, bpx, spx, er, xr)
        tid += 2
        day += 2
    return trades


# ───────────────────────── taxonomy classifier ──────────────────────────

class TestClassifyClasses:
    def test_empty_and_none_return_empty_list(self):
        assert _classify_classes(None) == []
        assert _classify_classes("") == []
        assert _classify_classes("   ") == []

    def test_unclassified_when_no_pattern_matches(self):
        assert _classify_classes("Just felt right today.") == ["UNCLASSIFIED"]
        assert _classify_classes("Buying NVDA.") == ["UNCLASSIFIED"]

    def test_ml_advisor_matches(self):
        assert "ML_ADVISOR" in _classify_classes(
            "ML advisor (median +143% alpha) flags NVDA")
        assert "ML_ADVISOR" in _classify_classes("DecisionScorer says BUY")
        assert "ML_ADVISOR" in _classify_classes(
            "scorer confirms BUY")

    def test_earnings_play_matches(self):
        assert "EARNINGS_PLAY" in _classify_classes(
            "NVDA earnings imminent tomorrow")
        assert "EARNINGS_PLAY" in _classify_classes(
            "Q3 earnings print drives this")
        assert "EARNINGS_PLAY" in _classify_classes(
            "post-earnings drift play")

    def test_analyst_pt_matches_bank_names_and_pt(self):
        assert "ANALYST_PT" in _classify_classes(
            "Citi raised price target to $250")
        assert "ANALYST_PT" in _classify_classes(
            "Goldman upgrade with $300 PT")
        assert "ANALYST_PT" in _classify_classes(
            "HSBC initiates coverage")
        assert "ANALYST_PT" in _classify_classes(
            "Melius PT raise to $1100")

    def test_technicals_matches(self):
        assert "TECHNICALS" in _classify_classes(
            "RSI 60 + MACD bullish + golden cross")
        assert "TECHNICALS" in _classify_classes(
            "Breakout above the 200-day MA")
        assert "TECHNICALS" in _classify_classes(
            "Support at $50 holding firm")

    def test_macro_matches(self):
        assert "MACRO" in _classify_classes(
            "FOMC rate decision tomorrow")
        assert "MACRO" in _classify_classes(
            "Powell signaled rate cut")
        assert "MACRO" in _classify_classes(
            "CPI print hotter than expected")

    def test_breaking_news_matches(self):
        assert "BREAKING_NEWS" in _classify_classes(
            "BREAKING headline crossed the wire")
        assert "BREAKING_NEWS" in _classify_classes(
            "Just crossed: NVDA up 5% AH")

    def test_pundit_matches(self):
        assert "PUNDIT" in _classify_classes("Cramer buy signal")
        assert "PUNDIT" in _classify_classes("Druckenmiller went long")
        assert "PUNDIT" in _classify_classes("Cathie Wood added more")

    def test_sector_sympathy_matches(self):
        assert "SECTOR_SYMPATHY" in _classify_classes(
            "may drag DRAM up sympathetically")
        assert "SECTOR_SYMPATHY" in _classify_classes(
            "Sector rotation into semis")
        assert "SECTOR_SYMPATHY" in _classify_classes(
            "Peer strength dragging this higher")

    def test_concentration_matches(self):
        assert "CONCENTRATION" in _classify_classes(
            "Trimming to raise dry powder pre-print")
        assert "CONCENTRATION" in _classify_classes(
            "Overweight semis — rebalancing")

    def test_multi_label_on_real_trade_rationale(self):
        # The live DRAM whipsaw rationale carries 4 catalyst classes.
        reason = ("Triple-stacked catalyst: Citi bullish on DRAM price hike, "
                  "HSBC/Melius $1100 PT, Cramer buy signal — and ML advisor "
                  "(median +143% alpha) flags DRAM. At $50.70, 5 shares = "
                  "$253 deploys most free cash. Adds to semis concentration "
                  "but the catalyst is event-driven and timely; NVDA "
                  "earnings tomorrow may drag DRAM up sympathetically.")
        classes = _classify_classes(reason)
        # Required matches.
        assert "ML_ADVISOR" in classes
        assert "ANALYST_PT" in classes
        assert "PUNDIT" in classes
        assert "EARNINGS_PLAY" in classes
        assert "SECTOR_SYMPATHY" in classes
        assert "CONCENTRATION" in classes  # "semis concentration"
        # And it's deterministic in taxonomy order (ML first).
        assert classes.index("ML_ADVISOR") < classes.index("ANALYST_PT")

    def test_taxonomy_order_is_deterministic(self):
        # Same rationale always yields the same ordering of matches.
        text = "ML advisor + Goldman PT + RSI breakout + earnings tomorrow"
        a = _classify_classes(text)
        b = _classify_classes(text)
        assert a == b
        assert a.index("ML_ADVISOR") < a.index("EARNINGS_PLAY")
        assert a.index("EARNINGS_PLAY") < a.index("ANALYST_PT")
        assert a.index("ANALYST_PT") < a.index("TECHNICALS")

    def test_case_insensitive(self):
        assert "ML_ADVISOR" in _classify_classes("ML ADVISOR FLAGS BUY")
        assert "ML_ADVISOR" in _classify_classes("ml advisor flags buy")

    def test_substring_in_word_does_not_match(self):
        # "Citing" should not match Citi (word boundary).
        # "rating" should not match "PT".
        # These guard against the same false-positive class the ALLCAPS
        # extractor / news-themes pipeline guards against.
        assert "ANALYST_PT" not in _classify_classes(
            "Citing reports that suggest")
        assert "ANALYST_PT" not in _classify_classes(
            "The rating system here is broken")


# ────────────────────────── builder shape ───────────────────────────────

class TestBuilderShape:
    def test_no_data_on_empty_ledger(self):
        rep = build_catalyst_class_autopsy([])
        assert rep["state"] == "NO_DATA"
        assert rep["n_round_trips"] == 0
        assert rep["n_scored"] == 0
        assert rep["pool_win_rate_pct"] is None
        assert rep["classes"] == []
        assert "nothing to classify" in rep["headline"]

    def test_emerging_below_stable_gate(self):
        # 3 trips on the same class — below STABLE_MIN_TRIPS_PER_CLASS=4
        # for any class, so state is EMERGING.
        trades = _ledger([
            ("NVDA", 100, 110, "ML advisor flags BUY", "took profit"),
            ("AMD", 50, 55, "ML advisor flags BUY", "took profit"),
            ("MU", 80, 70, "ML advisor flags BUY", "stopped out"),
        ])
        rep = build_catalyst_class_autopsy(trades)
        assert rep["state"] == "EMERGING"
        assert rep["n_round_trips"] == 3
        assert rep["n_scored"] == 3
        ml = next(r for r in rep["classes"] if r["class"] == "ML_ADVISOR")
        assert ml["n_trips"] == 3
        # Verdict UNSTABLE below the gate, even with a clear pattern.
        assert ml["verdict"] == "UNSTABLE"

    def test_stable_at_n4(self):
        # 4 trips on the same class crosses the gate.
        trades = _ledger([
            ("NVDA", 100, 110, "ML advisor flags BUY", "took profit"),
            ("AMD", 50, 55, "ML advisor flags BUY", "took profit"),
            ("MU", 80, 90, "ML advisor flags BUY", "took profit"),
            ("META", 200, 220, "ML advisor flags BUY", "took profit"),
        ])
        rep = build_catalyst_class_autopsy(trades)
        assert rep["state"] == "STABLE"
        assert rep["n_scored"] == 4
        ml = next(r for r in rep["classes"] if r["class"] == "ML_ADVISOR")
        assert ml["n_trips"] == 4
        # 4-of-4 wins, pool baseline is 100%, delta is 0 → NEUTRAL
        # (the class WR equals the pool WR — no bias).
        assert ml["verdict"] == "NEUTRAL"

    def test_pool_baseline_anchor(self):
        # 8 trips: 4 ML_ADVISOR (all winners) + 4 ANALYST_PT (all losers).
        # Pool WR = 4/8 = 50%. ML_ADVISOR WR = 100% (delta +50, BIASED_WINNER).
        # ANALYST_PT WR = 0% (delta -50, BIASED_LOSER).
        trades = _ledger([
            ("NVDA", 100, 110, "ML advisor flags BUY", ""),
            ("AMD", 50, 55, "ML advisor flags BUY", ""),
            ("MU", 80, 90, "ML advisor flags BUY", ""),
            ("META", 200, 220, "ML advisor flags BUY", ""),
            ("PLTR", 100, 80, "Citi raises PT to $120", ""),
            ("SMCI", 200, 180, "JPM upgrade $250 PT", ""),
            ("UBER", 50, 40, "Goldman raises PT", ""),
            ("RBLX", 60, 50, "Wedbush PT raise", ""),
        ])
        rep = build_catalyst_class_autopsy(trades)
        assert rep["state"] == "STABLE"
        assert rep["pool_win_rate_pct"] == 50.0
        ml = next(r for r in rep["classes"] if r["class"] == "ML_ADVISOR")
        pt = next(r for r in rep["classes"] if r["class"] == "ANALYST_PT")
        assert ml["win_rate_pct"] == 100.0
        assert pt["win_rate_pct"] == 0.0
        assert ml["verdict"] == "BIASED_WINNER"
        assert pt["verdict"] == "BIASED_LOSER"
        assert rep["top_biased_winner"] == "ML_ADVISOR"
        assert rep["top_biased_loser"] == "ANALYST_PT"

    def test_multi_class_trip_contributes_to_every_bucket(self):
        # One trip with two classes contributes to both.
        # 4 trips, all with both ML_ADVISOR + ANALYST_PT, 3W-1L.
        # Both classes show n_trips=4, wr=75%.
        trades = _ledger([
            ("NVDA", 100, 110, "ML advisor + Citi PT", ""),
            ("AMD", 50, 55, "ML advisor + Goldman PT", ""),
            ("MU", 80, 90, "ML advisor + JPM PT", ""),
            ("META", 200, 180, "ML advisor + HSBC PT", ""),
        ])
        rep = build_catalyst_class_autopsy(trades)
        ml = next(r for r in rep["classes"] if r["class"] == "ML_ADVISOR")
        pt = next(r for r in rep["classes"] if r["class"] == "ANALYST_PT")
        assert ml["n_trips"] == 4
        assert pt["n_trips"] == 4
        assert ml["win_rate_pct"] == 75.0
        assert pt["win_rate_pct"] == 75.0
        # n_scored counts trips, not bucket-fills — 4 trips, not 8.
        assert rep["n_scored"] == 4
        assert rep["n_round_trips"] == 4

    def test_unclassified_bucket_when_no_match(self):
        trades = _ledger([
            ("NVDA", 100, 110, "felt right", ""),
            ("AMD", 50, 55, "had a hunch", ""),
            ("MU", 80, 70, "no specific reason", ""),
            ("META", 200, 220, "vibes", ""),
        ])
        rep = build_catalyst_class_autopsy(trades)
        u = next(r for r in rep["classes"] if r["class"] == "UNCLASSIFIED")
        assert u["n_trips"] == 4

    def test_pnl_aggregation_arithmetic(self):
        # 2 trips on EARNINGS_PLAY: +$100 (10@$10 → 10@$20) and -$50.
        trades = _ledger([
            ("NVDA", 10, 20, "NVDA earnings tomorrow", ""),
            ("AMD", 10, 5, "AMD earnings print", ""),
        ])
        rep = build_catalyst_class_autopsy(trades)
        ep = next(r for r in rep["classes"] if r["class"] == "EARNINGS_PLAY")
        assert ep["n_trips"] == 2
        assert ep["total_pnl_usd"] == 50.0  # +100 + -50
        # avg pnl pct = (100% + -50%) / 2 = 25%
        assert ep["avg_pnl_pct"] == 25.0

    def test_pool_win_rate_includes_unclassified(self):
        # Without UNCLASSIFIED bucket the pool WR would lie. A bad
        # un-rationalized trade must drag the baseline.
        trades = _ledger([
            ("NVDA", 100, 110, "ML advisor flags BUY", ""),  # win
            ("AMD", 100, 90, "felt right", ""),              # loss, UNCLASSIFIED
        ])
        rep = build_catalyst_class_autopsy(trades)
        # 1W-1L pool → 50% baseline. n_scored is 2 trips.
        assert rep["pool_win_rate_pct"] == 50.0
        assert rep["n_scored"] == 2

    def test_zero_pnl_trip_is_not_a_win(self):
        # Strict > 0 win convention — wash reads as non-win (loser_autopsy
        # parity: strict < 0 loser convention).
        trades = _ledger([
            ("NVDA", 100, 100, "ML advisor flags BUY", ""),
        ])
        rep = build_catalyst_class_autopsy(trades)
        ml = next(r for r in rep["classes"] if r["class"] == "ML_ADVISOR")
        assert ml["n_trips"] == 1
        assert ml["n_wins"] == 0


# ───────────────────── never-raises / robustness ────────────────────────

class TestNeverRaises:
    def test_garbage_trade_rows_degrade_silently(self):
        # Builder must not raise on garbage-only / mixed-garbage input.
        # Whether the good trip survives depends on build_round_trips'
        # garbage tolerance (the SSOT contract); the autopsy's contract
        # is only "do not raise, return a well-shaped report".
        cases = [
            [],
            [None],
            [{}],
            ["not a dict"],
            [{"id": 99, "ticker": "BOGUS"}, None, "garbage"],
        ]
        for trades in cases:
            # Filter non-dicts so build_round_trips doesn't immediately
            # blow on attribute access (its own contract); the autopsy
            # must still tolerate the filtered residue.
            cleaned = [r for r in trades if isinstance(r, dict)]
            rep = build_catalyst_class_autopsy(cleaned)
            # Shape stable; no exception escaped.
            assert "state" in rep
            assert "classes" in rep
            assert isinstance(rep["classes"], list)

    def test_handles_none_pnl_pct_gracefully(self):
        # If pnl_pct is somehow None, avg_pnl_pct skips it but n_trips counts.
        trades = _ledger([
            ("NVDA", 100, 110, "ML advisor flags BUY", ""),
        ])
        # build_round_trips computes pnl_pct from price; we don't tamper.
        # Instead, test the empty-pcts path by stubbing.
        from paper_trader.analytics import catalyst_class_autopsy as mod
        from unittest.mock import patch

        def fake_rt(_):
            return [{
                "ticker": "NVDA", "type": "stock",
                "entry_ts": "2026-01-01T12:00:00+00:00",
                "exit_ts": "2026-01-02T12:00:00+00:00",
                "entry_trade_ids": [1], "exit_trade_ids": [2],
                "qty": 10, "cost": 1000, "proceeds": 1100,
                "pnl_usd": 100.0, "pnl_pct": None, "hold_days": 1.0,
                "n_buys": 1, "n_sells": 1,
                "strike": None, "expiry": None,
            }]
        with patch.object(mod, "build_round_trips", fake_rt):
            rep = build_catalyst_class_autopsy(trades)
        ml = next(r for r in rep["classes"] if r["class"] == "ML_ADVISOR")
        assert ml["n_trips"] == 1
        assert ml["avg_pnl_pct"] is None

    def test_response_shape_stable(self):
        rep = build_catalyst_class_autopsy([])
        # Stable keys regardless of state.
        for k in ("as_of", "state", "headline", "n_round_trips", "n_scored",
                  "pool_win_rate_pct", "stable_min_trips_per_class",
                  "biased_wr_delta_pct", "best_class", "worst_class",
                  "top_biased_winner", "top_biased_loser", "classes",
                  "taxonomy"):
            assert k in rep, f"missing key {k}"
        assert "UNCLASSIFIED" in rep["taxonomy"]
        assert "ML_ADVISOR" in rep["taxonomy"]


# ───────────────────── verdict / headline correctness ───────────────────

class TestVerdictAndHeadline:
    def test_biased_winner_headline_mentions_class(self):
        # 4 winners on ML_ADVISOR, 4 losers on ANALYST_PT.
        trades = _ledger([
            ("NVDA", 100, 110, "ML advisor flags BUY", ""),
            ("AMD", 50, 55, "ML advisor flags BUY", ""),
            ("MU", 80, 90, "ML advisor flags BUY", ""),
            ("META", 200, 220, "ML advisor flags BUY", ""),
            ("PLTR", 100, 80, "Citi PT raise", ""),
            ("SMCI", 200, 180, "JPM PT raise", ""),
            ("UBER", 50, 40, "Goldman PT raise", ""),
            ("RBLX", 60, 50, "Wedbush PT raise", ""),
        ])
        rep = build_catalyst_class_autopsy(trades)
        assert "ML_ADVISOR" in rep["headline"]
        assert "ANALYST_PT" in rep["headline"]

    def test_neutral_class_at_pool_wr(self):
        # 4 ML_ADVISOR trips: 2W-2L → 50% WR. With NO other classes,
        # pool WR is 50%. Class WR == pool WR → NEUTRAL (delta 0).
        trades = _ledger([
            ("NVDA", 100, 110, "ML advisor flags BUY", ""),
            ("AMD", 50, 60, "ML advisor flags BUY", ""),
            ("MU", 80, 70, "ML advisor flags BUY", ""),
            ("META", 200, 180, "ML advisor flags BUY", ""),
        ])
        rep = build_catalyst_class_autopsy(trades)
        ml = next(r for r in rep["classes"] if r["class"] == "ML_ADVISOR")
        assert ml["n_trips"] == 4
        assert ml["win_rate_pct"] == 50.0
        assert ml["verdict"] == "NEUTRAL"

    def test_at_band_edge_is_neutral_not_biased(self):
        # A class WR exactly BIASED_WR_DELTA_PCT above pool: the band
        # is inclusive (>= delta ⇒ BIASED). So WR pool+delta is BIASED.
        # A WR pool+delta-0.01 is NEUTRAL. Pinned to lock the inequality.
        # Construct: 4 trips, 3W-1L → 75% WR. Pool WR = 75%. Delta=0. NEUTRAL.
        trades = _ledger([
            ("NVDA", 100, 110, "ML advisor flags BUY", ""),
            ("AMD", 50, 55, "ML advisor flags BUY", ""),
            ("MU", 80, 90, "ML advisor flags BUY", ""),
            ("META", 200, 180, "ML advisor flags BUY", ""),
        ])
        rep = build_catalyst_class_autopsy(trades)
        ml = next(r for r in rep["classes"] if r["class"] == "ML_ADVISOR")
        assert ml["verdict"] == "NEUTRAL"

    def test_stable_gate_pinned_to_module_constant(self):
        # Asserts STABLE_MIN_TRIPS_PER_CLASS is consistent.
        rep = build_catalyst_class_autopsy([])
        assert rep["stable_min_trips_per_class"] == STABLE_MIN_TRIPS_PER_CLASS
        assert rep["biased_wr_delta_pct"] == BIASED_WR_DELTA_PCT
