"""Tests for analytics/exit_priority_ranking.py + /api/exit-priority-ranking.

Contract:
* Pure ``build_exit_priority_ranking`` takes open_positions + total_value
  + news_counts_by_ticker. Never raises on garbage inputs.
* State ladder: NO_DATA → ALL_CASH → OK.
* Score 0..100 composed from (concentration, pnl, age, silence) weighted
  factors. -10 penalty when current_price mark is stale.
* Option positions are excluded (option P/L is not comparable to %-of-NAV).
* current_price == 0 (the upsert reset state before the next mark)
  collapses the position to unmarked — pnl_factor is treated as neutral,
  weight_pct as 0 (the position still ranks on age/silence).
* Losers heavier than winners on pnl_factor (cut-loss tier maxes the
  factor; let-run tier collapses it).
* Rankings sorted by score DESC then ticker ASC.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics.exit_priority_ranking import (
    DEFAULT_HIGH_CONC_PCT,
    DEFAULT_STALE_NEWS_HOURS,
    DEFAULT_LONG_HOLD_DAYS,
    _concentration_factor,
    _pnl_factor,
    _age_factor,
    _silence_factor,
    build_exit_priority_ranking,
)


NOW = datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc)


def _pos(ticker: str, qty: float, avg_cost: float, current_price: float,
         opened_days_ago: float, type_: str = "stock") -> dict:
    return {
        "ticker": ticker, "type": type_, "qty": qty, "avg_cost": avg_cost,
        "current_price": current_price,
        "unrealized_pl": qty * (current_price - avg_cost),
        "opened_at": (NOW - timedelta(days=opened_days_ago)).isoformat(),
    }


# ─── State ladder ──────────────────────────────────────────────────


class TestStateLadder:
    def test_no_total_value_is_no_data(self):
        r = build_exit_priority_ranking([], None, now=NOW)
        assert r["state"] == "NO_DATA"
        assert r["headline"].startswith("Exit priority: insufficient")

    def test_negative_total_value_is_no_data(self):
        r = build_exit_priority_ranking([], -100.0, now=NOW)
        assert r["state"] == "NO_DATA"

    def test_empty_positions_is_all_cash(self):
        r = build_exit_priority_ranking([], 1000.0, now=NOW)
        assert r["state"] == "ALL_CASH"
        assert r["n_ranked"] == 0
        assert "100% cash" in r["headline"]

    def test_only_options_is_all_cash(self):
        # Options are excluded from the ranker by design.
        opts = [
            {"ticker": "NVDA", "type": "call", "qty": 1, "avg_cost": 2.0,
             "current_price": 1.5, "opened_at": NOW.isoformat()},
        ]
        r = build_exit_priority_ranking(opts, 1000.0, now=NOW)
        assert r["state"] == "ALL_CASH"

    def test_one_stock_position_state_ok(self):
        r = build_exit_priority_ranking(
            [_pos("NVDA", 5, 100.0, 110.0, opened_days_ago=10)],
            1000.0, now=NOW,
        )
        assert r["state"] == "OK"
        assert r["n_ranked"] == 1
        assert r["top_exit"] == "NVDA"


# ─── Factor mapping (sanity) ───────────────────────────────────────


class TestFactorMapping:
    def test_concentration_saturates_at_high_conc_pct(self):
        assert _concentration_factor(0, 25) == 0.0
        assert _concentration_factor(12.5, 25) == 0.5
        assert _concentration_factor(25, 25) == 1.0
        assert _concentration_factor(60, 25) == 1.0  # clamp

    def test_pnl_factor_losers_max(self):
        # Cut-loss tier: ≤ -5% pnl → factor 1.0
        assert _pnl_factor(-5.0) == 1.0
        assert _pnl_factor(-20.0) == 1.0

    def test_pnl_factor_winners_min(self):
        # Let-run tier: ≥ +10% pnl → factor 0.0
        assert _pnl_factor(10.0) == 0.0
        assert _pnl_factor(50.0) == 0.0

    def test_pnl_factor_neutral_breakeven(self):
        assert _pnl_factor(0.0) == 0.5
        assert _pnl_factor(None) == 0.5  # unknown → neutral

    def test_age_factor_saturates(self):
        assert _age_factor(0, 30) == 0.0
        assert _age_factor(15, 30) == 0.5
        assert _age_factor(30, 30) == 1.0
        assert _age_factor(90, 30) == 1.0  # clamp

    def test_silence_factor_clamped(self):
        # 168h = 1 week stale_news_h
        assert _silence_factor(0, 168) == 0.0
        assert _silence_factor(168, 168) == 1.0
        assert _silence_factor(500, 168) == 1.0


# ─── Ranking logic ─────────────────────────────────────────────────


class TestRankingLogic:
    def test_loser_outranks_winner_holding_other_factors_equal(self):
        positions = [
            _pos("LOSE", 5, 100.0, 90.0, opened_days_ago=10),   # -10% loser
            _pos("WIN",  5, 100.0, 115.0, opened_days_ago=10),   # +15% winner
        ]
        # Same concentration roughly, same age, same news silence.
        r = build_exit_priority_ranking(
            positions, total_value=2000.0,
            news_counts_by_ticker={
                "LOSE": {"n_articles": 5, "hours_since_last": 1.0},
                "WIN":  {"n_articles": 5, "hours_since_last": 1.0},
            },
            now=NOW,
        )
        assert r["rankings"][0]["ticker"] == "LOSE"
        assert r["rankings"][1]["ticker"] == "WIN"

    def test_high_concentration_dominates_low_concentration(self):
        # Same pnl, same age, same news. The 80%-weight position should
        # outrank the 5%-weight one.
        positions = [
            _pos("BIG",   80, 10.0, 10.0, opened_days_ago=10),  # mv=800
            _pos("SMALL", 5,  10.0, 10.0, opened_days_ago=10),  # mv=50
        ]
        r = build_exit_priority_ranking(
            positions, total_value=1000.0,
            news_counts_by_ticker={
                "BIG":   {"n_articles": 5, "hours_since_last": 1.0},
                "SMALL": {"n_articles": 5, "hours_since_last": 1.0},
            },
            now=NOW,
        )
        assert r["rankings"][0]["ticker"] == "BIG"

    def test_news_silence_increases_exit_priority(self):
        # Two identical losers, one silent, one chatty. Silent ranks higher.
        positions = [
            _pos("SILENT", 5, 100.0, 95.0, opened_days_ago=10),
            _pos("CHATTY", 5, 100.0, 95.0, opened_days_ago=10),
        ]
        r = build_exit_priority_ranking(
            positions, total_value=1000.0,
            news_counts_by_ticker={
                "SILENT": {"n_articles": 0, "hours_since_last": 240.0},
                "CHATTY": {"n_articles": 10, "hours_since_last": 1.0},
            },
            now=NOW,
        )
        # Silence factor is the only difference; SILENT must rank first.
        assert r["rankings"][0]["ticker"] == "SILENT"

    def test_old_position_outranks_fresh_when_others_equal(self):
        positions = [
            _pos("OLD",   5, 100.0, 100.0, opened_days_ago=40),
            _pos("FRESH", 5, 100.0, 100.0, opened_days_ago=1),
        ]
        r = build_exit_priority_ranking(
            positions, total_value=1000.0,
            news_counts_by_ticker={
                "OLD":   {"n_articles": 5, "hours_since_last": 1.0},
                "FRESH": {"n_articles": 5, "hours_since_last": 1.0},
            },
            now=NOW,
        )
        assert r["rankings"][0]["ticker"] == "OLD"

    def test_score_clamped_0_100(self):
        # Max-everything position should not exceed 100.
        positions = [
            _pos("MAX", 50, 100.0, 80.0, opened_days_ago=200),  # -20% loser
        ]
        r = build_exit_priority_ranking(
            positions, total_value=1000.0,
            news_counts_by_ticker={"MAX": {"n_articles": 0, "hours_since_last": 999.0}},
            now=NOW,
        )
        assert 0.0 <= r["rankings"][0]["score"] <= 100.0

    def test_rankings_sorted_score_desc_then_ticker_asc(self):
        # Two identical positions → deterministic tie-break on ticker.
        positions = [
            _pos("ZULU",  5, 100.0, 100.0, opened_days_ago=10),
            _pos("ALPHA", 5, 100.0, 100.0, opened_days_ago=10),
        ]
        r = build_exit_priority_ranking(
            positions, total_value=1000.0, now=NOW,
            news_counts_by_ticker={
                "ZULU":  {"n_articles": 5, "hours_since_last": 1.0},
                "ALPHA": {"n_articles": 5, "hours_since_last": 1.0},
            },
        )
        s_zulu = next(r2["score"] for r2 in r["rankings"] if r2["ticker"] == "ZULU")
        s_alpha = next(r2["score"] for r2 in r["rankings"] if r2["ticker"] == "ALPHA")
        assert s_zulu == s_alpha
        # Identical scores → ticker ASC.
        assert r["rankings"][0]["ticker"] == "ALPHA"


# ─── Mark-staleness penalty ────────────────────────────────────────


class TestMarkStalenessPenalty:
    def test_stale_mark_penalty_subtracted(self):
        # Two identical positions; one has a stale mark_updated. The stale
        # one's score must be exactly penalty lower (modulo float rounding).
        fresh_mark = NOW.isoformat()
        stale_mark = (NOW - timedelta(hours=72)).isoformat()
        positions = [
            {**_pos("STALE", 5, 100.0, 100.0, opened_days_ago=10),
             "mark_updated": stale_mark},
            {**_pos("FRESH", 5, 100.0, 100.0, opened_days_ago=10),
             "mark_updated": fresh_mark},
        ]
        r = build_exit_priority_ranking(
            positions, total_value=1000.0, now=NOW,
            news_counts_by_ticker={
                "STALE": {"n_articles": 5, "hours_since_last": 1.0},
                "FRESH": {"n_articles": 5, "hours_since_last": 1.0},
            },
            stale_mark_h=24.0, stale_mark_penalty=10.0,
        )
        s = {x["ticker"]: x["score"] for x in r["rankings"]}
        assert s["FRESH"] - s["STALE"] == pytest.approx(10.0, abs=0.01)
        stale = next(x for x in r["rankings"] if x["ticker"] == "STALE")
        assert stale["stale_mark"] is True
        assert "stale mark (penalty applied)" in stale["reasons"]


# ─── current_price == 0 edge case (upsert reset) ───────────────────


class TestUnmarkedPosition:
    def test_zero_current_price_does_not_imply_minus_100_pct_loss(self):
        # The positions.current_price column is reset to 0 on every upsert
        # and re-marked by strategy._portfolio_snapshot before the next
        # cycle. A leftover 0 must NOT be read as a -100% loss.
        positions = [
            {"ticker": "UM", "type": "stock", "qty": 5, "avg_cost": 100.0,
             "current_price": 0.0, "unrealized_pl": 0.0,
             "opened_at": (NOW - timedelta(days=10)).isoformat()},
        ]
        r = build_exit_priority_ranking(positions, 1000.0, now=NOW)
        row = r["rankings"][0]
        assert row["unmarked"] is True
        assert row["unrealized_pl_pct"] is None
        assert "unmarked (current_price=0)" in row["reasons"]
        assert r["n_unmarked"] == 1


# ─── Robustness ────────────────────────────────────────────────────


class TestRobustness:
    def test_garbage_position_rows_dropped(self):
        bad = [
            None,
            {},  # no ticker
            {"ticker": "OK", "type": "stock", "qty": -1, "avg_cost": 10,
             "current_price": 10, "opened_at": NOW.isoformat()},  # qty<=0
            {"ticker": "GOOD", "type": "stock", "qty": 5, "avg_cost": 100,
             "current_price": 100, "opened_at": NOW.isoformat()},
        ]
        r = build_exit_priority_ranking(bad, 1000.0, now=NOW)
        assert [x["ticker"] for x in r["rankings"]] == ["GOOD"]

    def test_garbage_news_counts_default_to_mid(self):
        positions = [_pos("NVDA", 5, 100.0, 100.0, opened_days_ago=10)]
        # ticker not in news_map at all
        r = build_exit_priority_ranking(positions, 1000.0,
                                        news_counts_by_ticker={}, now=NOW)
        row = r["rankings"][0]
        assert row["news_silence_hours"] is None
        # silence_factor falls back to 0.5 (mid)
        assert row["factors"]["silence"] == 0.5

    def test_does_not_raise_on_unparseable_opened_at(self):
        positions = [{
            "ticker": "X", "type": "stock", "qty": 1, "avg_cost": 10,
            "current_price": 10, "opened_at": "garbage",
        }]
        r = build_exit_priority_ranking(positions, 1000.0, now=NOW)
        assert r["state"] == "OK"
        assert r["rankings"][0]["days_held"] is None
