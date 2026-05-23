"""Tests for analytics/kelly_sizing.py — Kelly-criterion sizing diagnostic.

Hand-computed arithmetic. The module sits on top of
``build_trade_asymmetry`` (single source of truth for ``payoff_ratio`` and
``actual_win_rate_pct`` — invariant #10). A wrong full-Kelly value, a
verdict emitted before the STABLE sample-size gate, a divide-by-zero on
all-winners or all-losers, a band edge that drifts, or a None
``top_position_pct`` that explodes all fail an assertion here.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.kelly_sizing import (
    ALIGNED_MAX_RATIO,
    OVERSIZED_MAX_RATIO,
    STABLE_MIN_RTS,
    UNDERSIZED_MAX_RATIO,
    _kelly_fraction,
    build_kelly_sizing,
)


def _rt(tid, ticker, buy_iso, sell_iso, qty, buy_px, sell_px):
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
    return (_BASE + timedelta(days=offset)).isoformat()


def _ledger(specs):
    """specs: (ticker, qty, buy_px, sell_px, hold_days). Each spec becomes
    one closed round-trip — re-BUY after full close starts a new one."""
    trades = []
    tid = 1
    day = 0
    for (ticker, qty, bpx, spx, hold) in specs:
        trades += _rt(tid, ticker, _day(day), _day(day + int(hold)),
                      qty, bpx, spx)
        tid += 2
        day += int(hold) + 1
    return trades


# ── Kelly arithmetic ────────────────────────────────────────────────────
class TestKellyFractionMath:
    def test_kelly_classic_60pct_payoff_2(self):
        # Coin with 60% wins paying 2:1: f* = 0.6 - 0.4/2 = 0.4 = 40%.
        assert _kelly_fraction(60.0, 2.0) == 40.0

    def test_kelly_breakeven_no_edge_returns_zero(self):
        # 50% win-rate at 1:1 payoff: f* = 0.5 - 0.5/1 = 0.
        assert _kelly_fraction(50.0, 1.0) == 0.0

    def test_kelly_negative_edge_negative_fraction(self):
        # 30% win-rate at 1:1 payoff: f* = 0.3 - 0.7/1 = -0.4 = -40%.
        # Kelly says do not bet at all when negative.
        assert _kelly_fraction(30.0, 1.0) == -40.0

    def test_kelly_high_payoff_low_winrate(self):
        # Live trader 2026-05-23 shape: 66.67% wr, 13.7585 payoff.
        # f* = 0.6667 - 0.3333/13.7585 = 0.6667 - 0.02422 ≈ 0.6425 → 64.25%.
        f = _kelly_fraction(66.67, 13.7585)
        assert f is not None
        assert abs(f - 64.245) < 0.01

    def test_kelly_returns_none_on_missing_inputs(self):
        assert _kelly_fraction(None, 2.0) is None
        assert _kelly_fraction(60.0, None) is None
        assert _kelly_fraction(None, None) is None

    def test_kelly_rejects_zero_or_negative_payoff(self):
        # A non-positive payoff_ratio cannot come from build_trade_asymmetry
        # (it filters those upstream), but the guard prevents div-by-zero
        # if any future caller passes garbage.
        assert _kelly_fraction(50.0, 0.0) is None
        assert _kelly_fraction(50.0, -1.0) is None

    def test_kelly_clamps_extreme_negative(self):
        # 1% win-rate at 0.001 payoff: f* = 0.01 - 0.99/0.001 = -989.99 → -100% floor.
        assert _kelly_fraction(1.0, 0.001) == -100.0


# ── sample-size gate ───────────────────────────────────────────────────
class TestSampleSizeGate:
    def test_no_trades_is_no_data(self):
        r = build_kelly_sizing([])
        assert r["state"] == "NO_DATA"
        assert r["verdict"] is None
        assert r["full_kelly_pct"] is None
        assert r["n_round_trips"] == 0

    def test_few_trips_emerging_metrics_only(self):
        # 3 round-trips with 1 winner, 1 loser, 1 winner. Payoff defined,
        # numerics emit, but verdict withheld below STABLE_MIN_RTS.
        trades = _ledger([
            ("A", 10, 10.0, 12.0, 1),   # +$20 winner
            ("B", 10, 10.0, 9.0, 1),    # -$10 loser
            ("C", 10, 10.0, 11.0, 1),   # +$10 winner
        ])
        r = build_kelly_sizing(trades, top_position_pct=50.0,
                               top_position_ticker="X")
        assert r["state"] == "EMERGING"
        assert r["verdict"] is None
        assert r["payoff_ratio"] is not None
        assert r["full_kelly_pct"] is not None
        assert r["half_kelly_pct"] == round(r["full_kelly_pct"] / 2.0, 4)
        assert r["quarter_kelly_pct"] == round(r["full_kelly_pct"] / 4.0, 4)

    def test_stable_threshold_emits_verdict(self):
        # Exactly STABLE_MIN_RTS round-trips with a clear positive edge and
        # top-position deep inside the KELLY_ALIGNED band.
        # Pattern: 14 winners +$20 each, 6 losers -$5 each.
        # payoff = 20 / 5 = 4. p = 14/20 = 70%. q = 30%.
        # f* = 0.7 - 0.3/4 = 0.625 = 62.5%. HK = 31.25%.
        specs = [("W{}".format(i), 10, 10.0, 12.0, 1) for i in range(14)]
        specs += [("L{}".format(i), 10, 10.0, 9.5, 1) for i in range(6)]
        assert len(specs) == STABLE_MIN_RTS
        trades = _ledger(specs)
        r = build_kelly_sizing(trades, top_position_pct=31.25,
                               top_position_ticker="W0")
        assert r["state"] == "STABLE"
        assert r["full_kelly_pct"] == 62.5
        assert r["half_kelly_pct"] == 31.25
        assert r["verdict"] == "KELLY_ALIGNED"


# ── verdict bands ──────────────────────────────────────────────────────
class TestVerdictBands:
    """Half-Kelly target = 31.25% (from the STABLE fixture above).
    UNDERSIZED if top < 0.5 × HK = 15.625%.
    KELLY_ALIGNED if 15.625% ≤ top ≤ 1.25 × HK = 39.0625%.
    OVERSIZED if 39.0625% < top ≤ 2.0 × HK = 62.5%.
    EXTREMELY_OVERSIZED if top > 62.5%.
    """
    @staticmethod
    def _stable_ledger():
        specs = [("W{}".format(i), 10, 10.0, 12.0, 1) for i in range(14)]
        specs += [("L{}".format(i), 10, 10.0, 9.5, 1) for i in range(6)]
        return _ledger(specs)

    def test_undersized_when_top_below_half_of_half_kelly(self):
        r = build_kelly_sizing(self._stable_ledger(), top_position_pct=10.0,
                               top_position_ticker="X")
        assert r["verdict"] == "UNDERSIZED"

    def test_aligned_at_exact_half_kelly(self):
        r = build_kelly_sizing(self._stable_ledger(), top_position_pct=31.25,
                               top_position_ticker="X")
        assert r["verdict"] == "KELLY_ALIGNED"

    def test_aligned_at_upper_band_edge(self):
        # Exactly 1.25 × half-Kelly = 39.0625% — still ALIGNED (inclusive).
        r = build_kelly_sizing(self._stable_ledger(), top_position_pct=39.0625,
                               top_position_ticker="X")
        assert r["verdict"] == "KELLY_ALIGNED"

    def test_oversized_above_aligned(self):
        # 50% sits between 39.0625% and 62.5% → OVERSIZED.
        r = build_kelly_sizing(self._stable_ledger(), top_position_pct=50.0,
                               top_position_ticker="X")
        assert r["verdict"] == "OVERSIZED"

    def test_extremely_oversized_above_full_kelly(self):
        # 65% NVDA — the actual live state on 2026-05-23. Above full Kelly
        # (62.5%) — EXTREMELY_OVERSIZED.
        r = build_kelly_sizing(self._stable_ledger(), top_position_pct=65.0,
                               top_position_ticker="NVDA")
        assert r["verdict"] == "EXTREMELY_OVERSIZED"
        # Headline must surface the ticker.
        assert "NVDA" in r["headline"] or "EXTREMELY_OVERSIZED" in r["headline"]


# ── edge cases ─────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_undefined_payoff_all_winners(self):
        # 20 winners, no losers → payoff undefined → state UNDEFINED_PAYOFF,
        # no verdict, no Kelly fraction.
        specs = [("W{}".format(i), 10, 10.0, 12.0, 1) for i in range(20)]
        r = build_kelly_sizing(_ledger(specs), top_position_pct=30.0,
                               top_position_ticker="X")
        assert r["state"] == "UNDEFINED_PAYOFF"
        assert r["verdict"] is None
        assert r["full_kelly_pct"] is None
        assert r["half_kelly_pct"] is None
        assert "no losing round-trips yet" in r["headline"]

    def test_undefined_payoff_all_losers(self):
        specs = [("L{}".format(i), 10, 10.0, 9.0, 1) for i in range(20)]
        r = build_kelly_sizing(_ledger(specs), top_position_pct=30.0,
                               top_position_ticker="X")
        assert r["state"] == "UNDEFINED_PAYOFF"
        assert r["verdict"] is None
        assert "no winning round-trips yet" in r["headline"]

    def test_negative_edge_verdict_at_stable(self):
        # 20 trades, 6 winners @ +$10, 14 losers @ -$10. p=30%, b=1, f*=-40%.
        specs = [("W{}".format(i), 10, 10.0, 11.0, 1) for i in range(6)]
        specs += [("L{}".format(i), 10, 10.0, 9.0, 1) for i in range(14)]
        r = build_kelly_sizing(_ledger(specs), top_position_pct=10.0,
                               top_position_ticker="X")
        assert r["state"] == "STABLE"
        assert r["full_kelly_pct"] == -40.0
        assert r["verdict"] == "NEGATIVE_EDGE"
        assert "negative" in r["headline"].lower()

    def test_top_position_pct_none_book_all_cash(self):
        # Stable book + no current top position (all cash). Numerics emit,
        # verdict is None (nothing to benchmark against), no crash.
        specs = [("W{}".format(i), 10, 10.0, 12.0, 1) for i in range(14)]
        specs += [("L{}".format(i), 10, 10.0, 9.5, 1) for i in range(6)]
        r = build_kelly_sizing(_ledger(specs), top_position_pct=None,
                               top_position_ticker=None)
        assert r["state"] == "STABLE"
        assert r["verdict"] is None
        assert r["delta_vs_half_kelly_pct"] is None
        assert r["top_position_pct"] is None

    def test_band_constants_are_self_consistent(self):
        # The bands must be strictly ordered; a refactor that swaps these
        # would silently mis-classify the verdict bands.
        assert (UNDERSIZED_MAX_RATIO
                < ALIGNED_MAX_RATIO
                < OVERSIZED_MAX_RATIO)

    def test_payload_shape_pinned(self):
        # All documented keys must be present so the dashboard wire-up
        # doesn't KeyError on a missing field after a refactor.
        r = build_kelly_sizing([])
        for k in ("as_of", "state", "verdict", "verdict_reason", "headline",
                  "n_round_trips", "n_wins", "n_losses",
                  "actual_win_rate_pct", "payoff_ratio", "full_kelly_pct",
                  "half_kelly_pct", "quarter_kelly_pct",
                  "top_position_pct", "top_position_ticker",
                  "delta_vs_half_kelly_pct", "stable_min_round_trips",
                  "thresholds"):
            assert k in r, f"missing key: {k}"
