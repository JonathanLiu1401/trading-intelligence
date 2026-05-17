"""Tests for analytics/trade_asymmetry.py — exit/sizing behaviour pathology.

Hand-computed arithmetic. The module is a *diagnostic* layered on the
single-source-of-truth ``build_round_trips`` (AGENTS.md #10): a wrong payoff
ratio, a wrong breakeven win-rate, a verdict emitted before the STABLE
sample-size gate, an ∞ payoff when there are no losers, or a disposition
gap that double-counts wash trades all fail an assertion here.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.round_trips import build_round_trips
from paper_trader.analytics.trade_asymmetry import (
    DISPOSITION_EPS_DAYS,
    FLAT_EPS_USD,
    build_trade_asymmetry,
)


def _rt(tid, ticker, buy_iso, sell_iso, qty, buy_px, sell_px):
    """A buy+sell pair that build_round_trips folds into one round-trip."""
    return [
        {"id": tid, "timestamp": buy_iso, "ticker": ticker, "action": "BUY",
         "qty": qty, "price": buy_px, "value": qty * buy_px,
         "strike": None, "expiry": None, "option_type": None},
        {"id": tid + 1, "timestamp": sell_iso, "ticker": ticker,
         "action": "SELL", "qty": qty, "price": sell_px,
         "value": qty * sell_px, "strike": None, "expiry": None,
         "option_type": None},
    ]


_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _day(offset):
    """ISO timestamp `offset` days after the base — always a valid date."""
    return (_BASE + timedelta(days=offset)).isoformat()


def _ledger(specs):
    """specs: list of (ticker, qty, buy_px, sell_px, hold_days). Each becomes
    its own round-trip with a unique, strictly increasing window so
    build_round_trips closes it (re-BUY after full close → new round-trip)."""
    trades = []
    tid = 1
    day = 0
    for (ticker, qty, bpx, spx, hold) in specs:
        # buy on `day`, sell `hold` days later (integer days → exact hold_days)
        trades += _rt(tid, ticker, _day(day), _day(day + int(hold)),
                       qty, bpx, spx)
        tid += 2
        day += int(hold) + 1
    return trades


class TestSampleSizeGate:
    def test_no_trades_is_no_data(self):
        r = build_trade_asymmetry([])
        assert r["state"] == "NO_DATA"
        assert r["verdict"] is None
        assert r["n_round_trips"] == 0
        assert r["payoff_ratio"] is None

    def test_few_trips_emerging_has_metrics_no_verdict(self):
        # 3 round-trips: metrics present, verdict withheld until STABLE.
        trades = _ledger([
            ("A", 10, 10.0, 12.0, 1),   # +$20 winner
            ("B", 10, 10.0, 9.0, 1),    # -$10 loser
            ("C", 10, 10.0, 11.0, 1),   # +$10 winner
        ])
        r = build_trade_asymmetry(trades)
        assert r["state"] == "EMERGING"
        assert r["verdict"] is None          # gated to STABLE
        assert r["n_round_trips"] == 3
        assert r["payoff_ratio"] is not None  # metrics still emitted
        assert r["expectancy_usd"] is not None
        assert "emerging" in r["headline"].lower()

    def test_twenty_trips_is_stable_with_verdict(self):
        trades = _ledger([("W", 10, 10.0, 11.0, 1)] * 20)
        r = build_trade_asymmetry(trades)
        assert r["state"] == "STABLE"
        assert r["verdict"] is not None


class TestPayoffArithmetic:
    def test_payoff_breakeven_expectancy_exact(self):
        # 5 winners @ +$1, 15 losers @ -$5  (qty 1, so px delta == pnl)
        specs = [("W", 1, 100.0, 101.0, 1)] * 5 + [("L", 1, 100.0, 95.0, 1)] * 15
        trades = _ledger(specs)
        r = build_trade_asymmetry(trades)

        assert r["n_round_trips"] == 20
        assert r["n_wins"] == 5
        assert r["n_losses"] == 15
        assert r["avg_winner_usd"] == 1.0
        assert r["avg_loser_usd"] == -5.0
        # payoff = mean(win) / mean(|loss|) = 1 / 5 = 0.2
        assert r["payoff_ratio"] == 0.2
        # breakeven = 1/(1+0.2) = 0.8333.. → 83.33%
        assert r["breakeven_win_rate_pct"] == 83.33
        # actual = 5/20 = 25%
        assert r["actual_win_rate_pct"] == 25.0
        # expectancy = (5*1 + 15*-5)/20 = -70/20 = -3.5
        assert r["expectancy_usd"] == -3.5
        assert r["state"] == "STABLE"
        assert r["verdict"] == "PAYOFF_TRAP"

    def test_all_losers_is_payoff_trap_not_flat(self):
        # 20 straight losers, no winners → no payoff ratio can be formed, but
        # a unanimously losing book must NOT read as FLAT. It's the trap.
        trades = _ledger([("L", 1, 100.0, 95.0, 1)] * 20)
        r = build_trade_asymmetry(trades)
        assert r["n_wins"] == 0
        assert r["n_losses"] == 20
        assert r["payoff_ratio"] is None        # no winner mean to form it
        assert r["expectancy_usd"] == -5.0
        assert r["state"] == "STABLE"
        assert r["verdict"] == "PAYOFF_TRAP"
        assert r["verdict_reason"]               # non-empty, no None formatting

    def test_no_losers_payoff_is_none_not_infinity(self):
        trades = _ledger([("W", 1, 100.0, 110.0, 1)] * 20)  # all winners
        r = build_trade_asymmetry(trades)
        assert r["n_losses"] == 0
        assert r["payoff_ratio"] is None          # not inf, not a huge number
        assert r["breakeven_win_rate_pct"] is None
        assert r["expectancy_usd"] == 10.0
        # net positive, no disposition skew (all equal holds) → EDGE_POSITIVE
        assert r["verdict"] == "EDGE_POSITIVE"

    def test_metrics_match_build_round_trips_no_reimplementation(self):
        # Feed an asymmetric ledger; the realized total must equal the sum of
        # build_round_trips' own pnl_usd (i.e. we consume it, not recompute).
        specs = [("W", 2, 50.0, 53.0, 1)] * 6 + [("L", 3, 40.0, 38.0, 1)] * 14
        trades = _ledger(specs)
        rts = build_round_trips(trades)
        expected_total = round(sum(rt["pnl_usd"] for rt in rts), 4)
        expected_expectancy = round(expected_total / len(rts), 4)
        r = build_trade_asymmetry(trades)
        assert r["n_round_trips"] == len(rts)
        assert r["expectancy_usd"] == expected_expectancy
        assert r["realized_pl_usd"] == expected_total


class TestVerdictReachability:
    def test_disposition_bleed_positive_but_cuts_winners_fast(self):
        # 15 winners +$2 held 1d, 5 losers -$1 held 9d.
        #   expectancy = (15*2 + 5*-1)/20 = 25/20 = +1.25  (> FLAT_EPS)
        #   payoff = 2/1 = 2 → breakeven = 33.33%, actual = 75% → NOT trap
        #   disposition_gap = mean(win hold 1) - mean(loss hold 9) = -8 days
        specs = [("W", 1, 100.0, 102.0, 1)] * 15 + [("L", 1, 100.0, 99.0, 9)] * 5
        r = build_trade_asymmetry(_ledger(specs))
        assert r["expectancy_usd"] > FLAT_EPS_USD
        assert r["actual_win_rate_pct"] > r["breakeven_win_rate_pct"]
        assert r["disposition_gap_days"] < -DISPOSITION_EPS_DAYS
        assert r["verdict"] == "DISPOSITION_BLEED"

    def test_edge_positive_winners_held_longer(self):
        # Mirror image: winners held 9d, losers cut at 1d → good discipline.
        specs = [("W", 1, 100.0, 102.0, 9)] * 15 + [("L", 1, 100.0, 99.0, 1)] * 5
        r = build_trade_asymmetry(_ledger(specs))
        assert r["expectancy_usd"] > FLAT_EPS_USD
        assert r["disposition_gap_days"] > DISPOSITION_EPS_DAYS
        assert r["verdict"] == "EDGE_POSITIVE"

    def test_flat_zero_expectancy(self):
        # 10 winners +$1 held 1d, 10 losers -$1 held 1d → expectancy 0.
        specs = [("W", 1, 100.0, 101.0, 1)] * 10 + [("L", 1, 100.0, 99.0, 1)] * 10
        r = build_trade_asymmetry(_ledger(specs))
        assert abs(r["expectancy_usd"]) <= FLAT_EPS_USD
        assert r["verdict"] == "FLAT"

    def test_payoff_trap_headline_carries_disposition_clause(self):
        # Live-shaped: tiny winners cut fast, big losers ridden long.
        specs = [("W", 1, 100.0, 101.0, 1)] * 5 + [("L", 1, 100.0, 95.0, 9)] * 15
        r = build_trade_asymmetry(_ledger(specs))
        assert r["verdict"] == "PAYOFF_TRAP"
        assert r["disposition_gap_days"] < -DISPOSITION_EPS_DAYS
        # the "why" is surfaced in the headline even though label stays TRAP
        assert "winner" in r["headline"].lower()


class TestDispositionWashHandling:
    def test_wash_excluded_from_winloss_and_hold_means(self):
        # 19 decided trips + 1 exact wash ($0 pnl). The wash must not count
        # as a win or loss, and its hold-days must not pollute either mean.
        specs = (
            [("W", 1, 100.0, 102.0, 2)] * 12
            + [("L", 1, 100.0, 99.0, 4)] * 7
            + [("WASH", 1, 100.0, 100.0, 30)]   # 30d hold, pnl exactly 0
        )
        r = build_trade_asymmetry(_ledger(specs))
        assert r["n_round_trips"] == 20
        assert r["n_wins"] == 12
        assert r["n_losses"] == 7
        # disposition gap = mean(win hold 2) - mean(loss hold 4) = -2 exactly.
        # If the 30d wash leaked into either mean this would not be -2.0.
        assert r["disposition_gap_days"] == -2.0
