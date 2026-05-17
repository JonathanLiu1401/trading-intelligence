"""Tests for analytics/capital_paralysis.py — trap + cost + unlock ladder.

Every number is hand-computed from the inputs, so a wrong ladder rung, a
broken cut-priority sort, a mis-thresholded ``restores_action_alone``, or a
dropped drought pass-through fails the assertion (not just "no crash").
``build_capital_paralysis`` composes the already-tested ``build_liquidity``
and ``build_decision_drought``; these tests pin the *synthesis* it adds.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from paper_trader.analytics.capital_paralysis import build_capital_paralysis

NOW = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)


def _pos(ticker, qty, avg, cur, type_="stock"):
    return {"ticker": ticker, "type": type_, "qty": qty,
            "avg_cost": avg, "current_price": cur}


def _dec(action_taken, mins_ago):
    return {"timestamp": (NOW - timedelta(minutes=mins_ago)).isoformat(),
            "action_taken": action_taken}


def _eq(total, sp500, mins_ago):
    return {"timestamp": (NOW - timedelta(minutes=mins_ago)).isoformat(),
            "total_value": total, "cash": 6.0, "sp500_price": sp500}


class TestPinnedBookUnlockLadder:
    """The observed live trap: ~0% cash, two underwater names."""

    def setup_method(self):
        # LITE: mv 790, cost 800, pl -10, weight 79%. NVDA: mv 180, cost 200,
        # pl -20, weight 18%. cash 6 of 1000 → 0.6% → can't act.
        self.r = build_capital_paralysis(
            {"cash": 6.0, "total_value": 1000.0},
            [_pos("LITE", 1.0, 800.0, 790.0),
             _pos("NVDA", 1.0, 200.0, 180.0)],
            [{"timestamp": (NOW - timedelta(days=1)).isoformat(),
              "action": "BUY", "ticker": "LITE"}],
            decisions=[],
            equity_curve=[],
            now=NOW,
        )

    def test_state_is_pinned(self):
        assert self.r["state"] == "PINNED"
        assert self.r["can_act_on_signal"] is False
        assert self.r["liquidity_status"] == "NO_DRY_POWDER"

    def test_min_actionable_is_one_percent_of_book(self):
        # max(1.0, 1000 * 0.01) = 10.0
        assert self.r["min_actionable_usd"] == 10.0

    def test_ladder_cut_priority_biggest_loser_value_first(self):
        tickers = [r["ticker"] for r in self.r["unlock_ladder"]]
        assert tickers == ["LITE", "NVDA"]

    def test_ladder_lite_rung_math(self):
        lite = self.r["unlock_ladder"][0]
        assert lite["frees_usd"] == 790.0
        assert lite["cash_if_sold_alone"] == 796.0          # 6 + 790
        assert lite["cumulative_freed_usd"] == 790.0
        assert lite["cash_after_cumulative"] == 796.0
        # deployed after = 100 - 796/1000*100 = 20.4
        assert lite["deployed_pct_after_cumulative"] == 20.4
        assert lite["restores_action_alone"] is True

    def test_ladder_nvda_rung_is_cumulative(self):
        nvda = self.r["unlock_ladder"][1]
        assert nvda["frees_usd"] == 180.0
        assert nvda["cash_if_sold_alone"] == 186.0          # 6 + 180
        assert nvda["cumulative_freed_usd"] == 970.0        # 790 + 180
        assert nvda["cash_after_cumulative"] == 976.0       # 6 + 970
        # deployed after = 100 - 976/1000*100 = 2.4
        assert nvda["deployed_pct_after_cumulative"] == 2.4

    def test_recommended_unlock_is_first_restoring_sale(self):
        rec = self.r["recommended_unlock"]
        assert rec is not None
        assert rec["ticker"] == "LITE"
        assert rec["frees_usd"] == 790.0
        assert rec["pl_pct"] == -1.25                       # -10 / 800 * 100

    def test_headline_names_the_unlock(self):
        assert "PINNED" in self.r["headline"]
        assert "LITE" in self.r["headline"]

    def test_flags_include_unlock_hint(self):
        assert any("unlock: sell LITE" in f for f in self.r["flags"])


class TestLoserCutBeforeWinnerRegardlessOfSize:
    """Cut-priority must put a *loser* ahead of a bigger *winner* — a sale
    that frees cash AND stops a bleed beats one that books a gain."""

    def test_small_loser_ranked_above_big_winner(self):
        r = build_capital_paralysis(
            {"cash": 1.0, "total_value": 1000.0},
            [_pos("WIN", 1.0, 100.0, 700.0),    # mv 700, +600 winner
             _pos("LOSE", 1.0, 300.0, 280.0)],  # mv 280, -20 loser
            [], decisions=[], equity_curve=[], now=NOW,
        )
        assert [x["ticker"] for x in r["unlock_ladder"]] == ["LOSE", "WIN"]
        assert r["recommended_unlock"]["ticker"] == "LOSE"


class TestParalysisPassThrough:
    """Drought metrics must survive the composition, drive the bleed clause,
    and add a flag — the 'cost' half of trap+cost+unlock."""

    def setup_method(self):
        # chrono: FILLED@-300, ND@-240, ND@-180, FILLED@-120
        # → one closed PARALYSIS drought (2 ND cycles, nd_frac 1.0).
        decisions = [
            _dec("BUY NVDA → FILLED", 120),
            _dec("NO_DECISION", 180),
            _dec("NO_DECISION", 240),
            _dec("BUY LITE → FILLED", 300),
        ]
        # equity at the drought endpoints: portfolio -1%, SPY +1% → alpha -2.0
        equity = [_eq(1000.0, 100.0, 240), _eq(990.0, 101.0, 180)]
        self.r = build_capital_paralysis(
            {"cash": 6.0, "total_value": 1000.0},
            [_pos("LITE", 1.0, 800.0, 790.0)],
            [], decisions=decisions, equity_curve=equity, now=NOW,
        )

    def test_bleed_and_verdict_passed_through(self):
        p = self.r["paralysis"]
        assert p["involuntary_alpha_bleed_pct"] == -2.0
        assert p["n_paralysis_droughts"] == 1
        assert p["verdict"] == "BLEEDING"

    def test_worst_drought_is_compacted(self):
        w = self.r["paralysis"]["worst_alpha_drought"]
        assert w["kind"] == "PARALYSIS"
        assert w["alpha_pct"] == -2.0
        # compaction drops the heavy fields
        assert "portfolio_pct" not in w and "spy_pct" not in w

    def test_headline_carries_bleed_clause(self):
        assert "bled -2.00% alpha" in self.r["headline"]

    def test_flag_quantifies_inaction_cost(self):
        assert any("inaction has cost -2.00% alpha" in f
                   for f in self.r["flags"])


class TestStateBranches:
    def test_free_when_dry_powder_available(self):
        r = build_capital_paralysis(
            {"cash": 500.0, "total_value": 1000.0},
            [_pos("NVDA", 1.0, 200.0, 180.0)],
            [], decisions=[], equity_curve=[], now=NOW,
        )
        assert r["state"] == "FREE"
        assert r["can_act_on_signal"] is True
        assert r["recommended_unlock"] is None
        assert "FREE" in r["headline"]

    def test_no_data_when_empty(self):
        r = build_capital_paralysis(
            {"cash": 0.0, "total_value": 0.0}, [],
            [], decisions=[], equity_curve=[], now=NOW,
        )
        assert r["state"] == "NO_DATA"
        assert r["unlock_ladder"] == []

    def test_empty_when_no_cash_no_positions_but_value(self):
        # Degenerate guard: value but neither cash nor sellable positions.
        r = build_capital_paralysis(
            {"cash": 0.0, "total_value": 100.0}, [],
            [], decisions=[], equity_curve=[], now=NOW,
        )
        assert r["state"] == "EMPTY"
        assert r["recommended_unlock"] is None

    def test_cycles_since_last_fill_from_ongoing_drought(self):
        # Three trailing non-fill cycles, no FILL after → ongoing drought of 3.
        decisions = [
            _dec("HOLD NVDA → HOLD", 60),
            _dec("HOLD NVDA → HOLD", 120),
            _dec("NO_DECISION", 180),
        ]
        r = build_capital_paralysis(
            {"cash": 6.0, "total_value": 1000.0},
            [_pos("LITE", 1.0, 800.0, 790.0)],
            [], decisions=decisions, equity_curve=[], now=NOW,
        )
        assert r["cycles_since_last_fill"] == 3
