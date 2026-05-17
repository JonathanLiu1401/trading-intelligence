"""Tests for analytics/loser_autopsy.py — per-closed-losing-trade post-mortem.

Hand-computed arithmetic. The module is a *diagnostic* layered on the
single-source-of-truth ``build_round_trips`` (AGENTS.md invariant #10): a
recomputed P&L, a misclassified failure mode, a verdict emitted before the
STABLE sample-size gate, a wash counted as a loss, an entry/exit reason that
is not surfaced verbatim, or a non-deterministic dominant-mode tie-break all
fail an assertion here.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.loser_autopsy import (
    BIG_LOSS_PCT,
    FAST_HOLD_DAYS,
    SLOW_HOLD_DAYS,
    SMALL_LOSS_PCT,
    STABLE_MIN_LOSERS,
    _classify,
    build_loser_autopsy,
)

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _day(offset: int) -> str:
    return (_BASE + timedelta(days=offset)).isoformat()


def _rt(tid, ticker, buy_day, sell_day, qty, buy_px, sell_px,
        entry_reason="", exit_reason=""):
    """A buy+sell pair build_round_trips folds into one closed round-trip."""
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
    """specs: (ticker, buy_px, sell_px, hold_days, entry_reason, exit_reason).
    Each becomes its own round-trip on a strictly increasing, disjoint
    window (qty fixed at 10) so build_round_trips closes each independently.
    """
    trades, tid, day = [], 1, 0
    for ticker, bpx, spx, hold, er, xr in specs:
        trades += _rt(tid, ticker, day, day + hold, 10, bpx, spx, er, xr)
        tid += 2
        day += hold + 1
    return trades


# ───────────────────────── _classify boundaries ─────────────────────────

class TestClassify:
    def test_knife_catch_wins_even_when_fast(self):
        # Big loss takes precedence over the fast/shallow WHIPSAW arm.
        assert _classify(0.1, BIG_LOSS_PCT) == "KNIFE_CATCH"
        assert _classify(0.1, BIG_LOSS_PCT - 5) == "KNIFE_CATCH"

    def test_just_above_big_loss_is_not_knife(self):
        # -15.0 is KNIFE (<=); -14.999 is not.
        assert _classify(10.0, BIG_LOSS_PCT + 0.001) != "KNIFE_CATCH"

    def test_whipsaw_fast_and_shallow(self):
        assert _classify(FAST_HOLD_DAYS - 0.01, SMALL_LOSS_PCT + 0.1) == \
            "WHIPSAW"

    def test_fast_but_deep_is_not_whipsaw(self):
        # Inside a day but a -8% loss → not shallow → STOPPED_OUT (not big
        # enough for KNIFE, not slow enough for BLEED).
        assert _classify(0.5, -8.0) == "STOPPED_OUT"

    def test_exactly_fast_boundary_is_not_fast(self):
        # `< FAST_HOLD_DAYS` is strict — exactly 1.0 day is not "fast".
        assert _classify(FAST_HOLD_DAYS, SMALL_LOSS_PCT + 1) != "WHIPSAW"

    def test_slow_bleed_inclusive_boundary(self):
        assert _classify(SLOW_HOLD_DAYS, -6.0) == "SLOW_BLEED"
        assert _classify(SLOW_HOLD_DAYS - 0.01, -6.0) == "STOPPED_OUT"

    def test_none_inputs_never_raise_and_default(self):
        assert _classify(None, None) == "STOPPED_OUT"
        assert _classify(None, BIG_LOSS_PCT) == "KNIFE_CATCH"
        assert _classify(0.2, None) == "STOPPED_OUT"  # can't prove shallow


# ───────────────────────── state / sample-size gate ─────────────────────

class TestStateGate:
    def test_no_data(self):
        r = build_loser_autopsy([])
        assert r["state"] == "NO_DATA"
        assert r["verdict"] is None
        assert r["n_round_trips"] == 0
        assert r["worst_losers"] == []
        assert "autopsy" in r["headline"].lower()

    def test_no_losses_when_only_winners(self):
        # Two profitable round-trips, zero losers.
        trades = _ledger([
            ("AAA", 10.0, 12.0, 1, "buy a", "take profit"),
            ("BBB", 20.0, 25.0, 2, "buy b", "take profit"),
        ])
        r = build_loser_autopsy(trades)
        assert r["state"] == "NO_LOSSES"
        assert r["n_round_trips"] == 2
        assert r["n_losers"] == 0
        assert r["verdict"] is None
        assert r["total_loss_usd"] == 0.0

    def test_wash_is_not_a_loss(self):
        # sell_px == buy_px → pnl exactly 0 → excluded from losers (#10).
        trades = _ledger([("WSH", 10.0, 10.0, 1, "flat", "flat")])
        r = build_loser_autopsy(trades)
        assert r["n_round_trips"] == 1
        assert r["n_losers"] == 0
        assert r["state"] == "NO_LOSSES"

    def test_emerging_has_cards_and_numerics_but_no_verdict(self):
        # 3 losers (< STABLE_MIN_LOSERS) → metrics + cards, verdict withheld.
        assert STABLE_MIN_LOSERS > 3
        trades = _ledger([
            ("AAA", 10.0, 9.0, 1, "thesis a", "stopped"),
            ("BBB", 10.0, 8.0, 1, "thesis b", "stopped"),
            ("CCC", 10.0, 7.0, 1, "thesis c", "stopped"),
        ])
        r = build_loser_autopsy(trades)
        assert r["state"] == "EMERGING"
        assert r["verdict"] is None
        assert r["n_losers"] == 3
        assert len(r["worst_losers"]) == 3
        # -$10 + -$20 + -$30 (qty 10) = -$60.
        assert r["total_loss_usd"] == -60.0
        assert r["avg_loss_usd"] == -20.0
        assert "Emerging" in r["headline"]

    def test_stable_emits_verdict(self):
        # STABLE_MIN_LOSERS identical STOPPED_OUT losers → verdict set.
        specs = [(f"T{i}", 10.0, 9.5, 2, f"e{i}", f"x{i}")
                 for i in range(STABLE_MIN_LOSERS)]
        r = build_loser_autopsy(_ledger(specs))
        assert r["n_losers"] == STABLE_MIN_LOSERS
        assert r["state"] == "STABLE"
        assert r["verdict"] == "STOPPED_OUT"
        assert r["dominant_failure_mode"] == "STOPPED_OUT"


# ───────────────────────── verbatim reason join ─────────────────────────

class TestVerbatimReasons:
    def test_entry_and_exit_reason_surfaced_verbatim(self):
        er = "NVDA: 20d momentum +11.7%, MACD bullish, earnings catalyst"
        xr = "thesis broke — guidance miss, cutting the loss"
        r = build_loser_autopsy(_ledger([("NVDA", 100.0, 90.0, 2, er, xr)]))
        card = r["worst_losers"][0]
        assert card["entry_reason"] == er   # exact, not parsed
        assert card["exit_reason"] == xr
        assert card["ticker"] == "NVDA"
        assert card["pnl_usd"] == -100.0    # (90-100)*10

    def test_blank_reason_degrades_to_none(self):
        r = build_loser_autopsy(_ledger([("AAA", 10.0, 8.0, 1, "", "   ")]))
        card = r["worst_losers"][0]
        assert card["entry_reason"] is None
        assert card["exit_reason"] is None

    def test_missing_id_and_reason_keys_never_raise(self):
        # A buy row with no id / no reason key (defensive) still autopsies.
        trades = [
            {"timestamp": _day(0), "ticker": "ZZZ", "action": "BUY",
             "qty": 5, "price": 10.0, "value": 50.0, "strike": None,
             "expiry": None, "option_type": None},  # no id, no reason
            {"id": 2, "timestamp": _day(1), "ticker": "ZZZ",
             "action": "SELL", "qty": 5, "price": 8.0, "value": 40.0,
             "strike": None, "expiry": None, "option_type": None,
             "reason": "cut"},
        ]
        r = build_loser_autopsy(trades)
        assert r["n_losers"] == 1
        card = r["worst_losers"][0]
        assert card["entry_reason"] is None       # buy had no id
        assert card["exit_reason"] == "cut"
        assert card["pnl_usd"] == -10.0


# ───────────────────────── aggregates & ordering ────────────────────────

class TestAggregates:
    def test_worst_first_ordering_and_worst_n_cap(self):
        trades = _ledger([
            ("AAA", 10.0, 9.0, 1, "a", "x"),    # -$10
            ("BBB", 10.0, 6.0, 1, "b", "x"),    # -$40  (worst)
            ("CCC", 10.0, 8.0, 1, "c", "x"),    # -$20
        ])
        r = build_loser_autopsy(trades, worst_n=2)
        assert [c["ticker"] for c in r["worst_losers"]] == ["BBB", "CCC"]
        assert r["worst_losers"][0]["pnl_usd"] == -40.0
        # aggregates still span ALL losers even though cards capped at 2.
        assert r["n_losers"] == 3
        assert r["total_loss_usd"] == -70.0

    def test_median_loser_hold_even_count(self):
        # holds 1,2,3,4 → even → mean of middle two = 2.5.
        trades = _ledger([
            ("AAA", 10.0, 9.0, 1, "a", "x"),
            ("BBB", 10.0, 9.0, 2, "b", "x"),
            ("CCC", 10.0, 9.0, 3, "c", "x"),
            ("DDD", 10.0, 9.0, 4, "d", "x"),
        ])
        r = build_loser_autopsy(trades)
        assert r["median_loser_hold_days"] == 2.5

    def test_median_loser_hold_odd_count(self):
        trades = _ledger([
            ("AAA", 10.0, 9.0, 1, "a", "x"),
            ("BBB", 10.0, 9.0, 3, "b", "x"),
            ("CCC", 10.0, 9.0, 9, "c", "x"),
        ])
        r = build_loser_autopsy(trades)
        assert r["median_loser_hold_days"] == 3

    def test_ticker_breakdown_and_repeat_offenders(self):
        # NVDA loses twice (-$10, -$30 = -$40), AMD once (-$20).
        trades = _ledger([
            ("NVDA", 10.0, 9.0, 1, "a", "x"),
            ("AMD", 10.0, 8.0, 1, "b", "x"),
            ("NVDA", 10.0, 7.0, 1, "c", "x"),
        ])
        r = build_loser_autopsy(trades)
        tb = r["ticker_breakdown"]
        # Sorted most-negative $ first → NVDA (-$40) before AMD (-$20).
        assert tb[0]["ticker"] == "NVDA"
        assert tb[0]["loss_usd"] == -40.0
        assert tb[0]["n"] == 2
        assert tb[1]["ticker"] == "AMD"
        assert tb[1]["loss_usd"] == -20.0
        assert r["repeat_offenders"] == ["NVDA"]

    def test_dominant_mode_tiebreak_is_deterministic_by_severity(self):
        # 1 KNIFE_CATCH (-20%, qty10 buy100 → -$200) vs 1 WHIPSAW
        # (fast, -1%). Counts tie at 1 each → severity order puts
        # KNIFE_CATCH ahead of WHIPSAW deterministically.
        trades = _ledger([
            ("AAA", 100.0, 80.0, 3, "deep", "knife"),     # -20% KNIFE
            ("BBB", 100.0, 99.0, 0, "quick", "whipsaw"),  # -1% same-day
        ])
        r = build_loser_autopsy(trades)
        assert r["failure_mode_counts"] == {"KNIFE_CATCH": 1, "WHIPSAW": 1}
        assert r["dominant_failure_mode"] == "KNIFE_CATCH"

    def test_pnl_is_consumed_from_round_trips_not_recomputed(self):
        # Partial-then-full close: build_round_trips is the only P&L author.
        # Buy 10@10 (=100), sell 4@9 (=36), sell 6@8 (=48). One round-trip:
        # cost 100, proceeds 84, pnl -16.
        trades = [
            {"id": 1, "timestamp": _day(0), "ticker": "PRT", "action": "BUY",
             "qty": 10, "price": 10.0, "value": 100.0, "strike": None,
             "expiry": None, "option_type": None, "reason": "open"},
            {"id": 2, "timestamp": _day(1), "ticker": "PRT", "action": "SELL",
             "qty": 4, "price": 9.0, "value": 36.0, "strike": None,
             "expiry": None, "option_type": None, "reason": "trim"},
            {"id": 3, "timestamp": _day(2), "ticker": "PRT", "action": "SELL",
             "qty": 6, "price": 8.0, "value": 48.0, "strike": None,
             "expiry": None, "option_type": None, "reason": "exit rest"},
        ]
        r = build_loser_autopsy(trades)
        assert r["n_losers"] == 1
        card = r["worst_losers"][0]
        assert card["pnl_usd"] == -16.0
        assert card["cost"] == 100.0
        assert card["proceeds"] == 84.0
        # entry reason = first BUY; exit reason = last SELL (verbatim).
        assert card["entry_reason"] == "open"
        assert card["exit_reason"] == "exit rest"


class TestPurity:
    def test_never_raises_on_garbage(self):
        # Malformed rows must not blow up a daemon-adjacent read.
        r = build_loser_autopsy([{"ticker": "X", "action": "SELL"}])
        assert isinstance(r, dict)
        assert r["state"] in ("NO_DATA", "NO_LOSSES")
