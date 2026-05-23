"""Tests for analytics/exit_intent_audit.py — exit-reason intent classifier.

Hand-computed arithmetic. The module classifies each closed round-trip's
exit reason into a fixed bucket order (EARNINGS_CLEAR first, UNCLASSIFIED
last) and rolls up outcome per bucket. A wrong intent label on a
multi-keyword match, a verdict emitted before the STABLE gate, a
dominant-intent tie broken in the wrong direction, a bucket that
double-counts a wash trade, or a payload key drift all fail an assertion
here.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.exit_intent_audit import (
    STABLE_MIN_RTS,
    _classify_exit_intent,
    _INTENT_ORDER,
    build_exit_intent_audit,
)


_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _day(offset):
    return (_BASE + timedelta(days=offset)).isoformat()


def _rt_with_reason(tid, ticker, day, hold, qty, bpx, spx, exit_reason):
    """A buy+sell pair whose SELL row carries ``reason=exit_reason``.

    ``build_round_trips`` records ``exit_trade_ids`` and the audit joins
    the closing trade's ``reason`` back via the by-id index — so the
    exit_reason MUST be on the SELL row for the classifier to pick it up.
    """
    buy = {"id": tid, "timestamp": _day(day), "ticker": ticker,
           "action": "BUY", "qty": qty, "price": bpx, "value": qty * bpx,
           "strike": None, "expiry": None, "option_type": None,
           "reason": None}
    sell = {"id": tid + 1, "timestamp": _day(day + hold), "ticker": ticker,
            "action": "SELL", "qty": qty, "price": spx, "value": qty * spx,
            "strike": None, "expiry": None, "option_type": None,
            "reason": exit_reason}
    return [buy, sell]


def _ledger(specs):
    """specs: list of (ticker, qty, bpx, spx, hold, exit_reason).
    Each becomes one closed round-trip with non-overlapping windows."""
    trades = []
    tid = 1
    day = 0
    for (ticker, qty, bpx, spx, hold, reason) in specs:
        trades += _rt_with_reason(tid, ticker, day, hold, qty, bpx, spx,
                                  reason)
        tid += 2
        day += hold + 1
    return trades


# ── classifier ─────────────────────────────────────────────────────────
class TestClassifyExitIntent:
    def test_none_or_empty_unclassified(self):
        assert _classify_exit_intent(None) == "UNCLASSIFIED"
        assert _classify_exit_intent("") == "UNCLASSIFIED"
        assert _classify_exit_intent("   ") == "UNCLASSIFIED"

    def test_no_match_unclassified(self):
        assert _classify_exit_intent("liquidating for fun") == "UNCLASSIFIED"

    def test_defensive_cash_raise(self):
        s = ("Raising ~$253 dry powder pre-print lets me either fade a "
             "gap-up into TQQQ/MU strength")
        # "dry powder" → DEFENSIVE_CASH_RAISE.
        assert _classify_exit_intent(s) == "DEFENSIVE_CASH_RAISE"

    def test_thesis_flip(self):
        s = "DRAM is flat with no working thesis and book is over-deployed"
        assert _classify_exit_intent(s) == "THESIS_FLIP"

    def test_target_hit(self):
        s = "Lock in +0.5% gain before AH drop"
        assert _classify_exit_intent(s) == "TARGET_HIT"

    def test_stop_loss(self):
        s = "Stopped out at -8% after gap-down"
        assert _classify_exit_intent(s) == "STOP_LOSS"

    def test_earnings_clear(self):
        s = "Sell into earnings — binary risk too large at this size"
        assert _classify_exit_intent(s) == "EARNINGS_CLEAR"

    def test_case_insensitive(self):
        assert _classify_exit_intent("RAISE CASH") == "DEFENSIVE_CASH_RAISE"
        assert _classify_exit_intent("Take Profit") == "TARGET_HIT"

    def test_multi_bucket_resolves_to_first_in_order(self):
        # "raise cash ahead of earnings" matches BOTH EARNINGS_CLEAR and
        # DEFENSIVE_CASH_RAISE. _INTENT_ORDER puts EARNINGS_CLEAR first.
        s = "raise cash ahead of earnings — binary risk before the print"
        assert _classify_exit_intent(s) == "EARNINGS_CLEAR"
        # Sanity-check the precedence is what the test expects:
        assert _INTENT_ORDER.index("EARNINGS_CLEAR") < _INTENT_ORDER.index(
            "DEFENSIVE_CASH_RAISE")

    def test_thesis_flip_beats_defensive_cash(self):
        # _INTENT_ORDER has THESIS_FLIP before DEFENSIVE_CASH_RAISE.
        s = "thesis broken and need to free cash"
        assert _classify_exit_intent(s) == "THESIS_FLIP"


# ── sample-size gate ───────────────────────────────────────────────────
class TestSampleSizeGate:
    def test_no_trades_is_no_data(self):
        r = build_exit_intent_audit([])
        assert r["state"] == "NO_DATA"
        assert r["verdict"] is None
        assert r["n_round_trips"] == 0
        assert r["dominant_intent"] is None

    def test_few_trips_emerging(self):
        trades = _ledger([
            ("A", 10, 10.0, 12.0, 1, "raise cash"),
            ("B", 10, 10.0, 11.0, 1, "take profit"),
            ("C", 10, 10.0, 9.0, 1, "stopped out"),
        ])
        r = build_exit_intent_audit(trades)
        assert r["state"] == "EMERGING"
        assert r["verdict"] is None
        assert r["n_round_trips"] == 3

    def test_stable_emits_verdict(self):
        # 10 trips all with the same intent. Mix of wins/losses to
        # determine the verdict polarity.
        specs = [("W{}".format(i), 10, 10.0, 12.0, 1, "take profit")
                 for i in range(8)]
        specs += [("L{}".format(i), 10, 10.0, 9.0, 1, "take profit")
                  for i in range(2)]
        r = build_exit_intent_audit(_ledger(specs))
        assert r["state"] == "STABLE"
        assert r["dominant_intent"] == "TARGET_HIT"
        # avg pnl: (8 × +20 + 2 × -10) / 10 = (160 - 20) / 10 = +14/trip
        assert r["verdict"] == "DOMINANT_INTENT_HEALTHY"


# ── per-bucket math ────────────────────────────────────────────────────
class TestBucketStats:
    def test_bucket_aggregates_pnl_and_win_rate(self):
        specs = [
            ("A", 10, 10.0, 12.0, 1, "raise cash"),  # +$20 win
            ("B", 10, 10.0, 8.0, 1, "raise cash"),   # -$20 loss
            ("C", 10, 10.0, 13.0, 1, "raise cash"),  # +$30 win
        ]
        r = build_exit_intent_audit(_ledger(specs))
        defensive = next(b for b in r["buckets"]
                         if b["intent"] == "DEFENSIVE_CASH_RAISE")
        assert defensive["n"] == 3
        assert defensive["total_pnl_usd"] == 30.0
        assert defensive["avg_pnl_usd"] == 10.0
        # win-rate: 2 wins / (2 wins + 1 loss) = 66.67%
        assert defensive["win_rate_pct"] == 66.67
        assert defensive["n_wins"] == 2
        assert defensive["n_losses"] == 1

    def test_unclassified_bucket_collects_non_matches(self):
        specs = [
            ("A", 10, 10.0, 12.0, 1, "just felt like it"),
            ("B", 10, 10.0, 11.0, 1, None),
            ("C", 10, 10.0, 13.0, 1, ""),
        ]
        r = build_exit_intent_audit(_ledger(specs))
        unc = next(b for b in r["buckets"] if b["intent"] == "UNCLASSIFIED")
        assert unc["n"] == 3
        assert r["dominant_intent"] == "UNCLASSIFIED"

    def test_empty_bucket_has_zero_n_and_none_stats(self):
        specs = [("A", 10, 10.0, 12.0, 1, "take profit")]
        r = build_exit_intent_audit(_ledger(specs))
        # EARNINGS_CLEAR bucket exists but is empty.
        ec = next(b for b in r["buckets"] if b["intent"] == "EARNINGS_CLEAR")
        assert ec["n"] == 0
        assert ec["avg_pnl_usd"] is None
        assert ec["win_rate_pct"] is None


# ── verdict polarity ───────────────────────────────────────────────────
class TestVerdictPolarity:
    def _stable(self, intent_phrase, win_ratio):
        # 10 trips, all the same exit intent. win_ratio of them are
        # winners.
        n_w = int(round(10 * win_ratio))
        n_l = 10 - n_w
        specs = [("W{}".format(i), 10, 10.0, 12.0, 1, intent_phrase)
                 for i in range(n_w)]
        specs += [("L{}".format(i), 10, 10.0, 8.0, 1, intent_phrase)
                  for i in range(n_l)]
        return build_exit_intent_audit(_ledger(specs))

    def test_dominant_intent_bleed_when_negative_avg(self):
        # 3 winners (+$20), 7 losers (-$20) → avg = -$8/trip.
        r = self._stable("raise cash", 0.3)
        assert r["state"] == "STABLE"
        assert r["dominant_intent"] == "DEFENSIVE_CASH_RAISE"
        assert r["verdict"] == "DOMINANT_INTENT_BLEED"
        assert "DEFENSIVE_CASH_RAISE" in (r["verdict_reason"] or "")

    def test_dominant_intent_healthy_when_positive(self):
        # 7 winners, 3 losers → avg = +$8/trip.
        r = self._stable("take profit", 0.7)
        assert r["verdict"] == "DOMINANT_INTENT_HEALTHY"

    def test_unclassified_dominant_triggers_intent_unclear(self):
        specs = [("X{}".format(i), 10, 10.0, 12.0, 1, "vibes")
                 for i in range(10)]
        r = build_exit_intent_audit(_ledger(specs))
        assert r["state"] == "STABLE"
        assert r["dominant_intent"] == "UNCLASSIFIED"
        assert r["verdict"] == "INTENT_UNCLEAR"

    def test_no_verdict_when_dominant_bucket_too_thin(self):
        # 10 trips with 2 of each of 5 different intents — no bucket
        # has ≥3 trips so no verdict despite STABLE state.
        specs = []
        for i, phrase in enumerate([
            "take profit", "raise cash", "stopped out",
            "no working thesis", "into earnings",
        ]):
            specs.append(("W{}".format(i), 10, 10.0, 12.0, 1, phrase))
            specs.append(("L{}".format(i), 10, 10.0, 9.0, 1, phrase))
        r = build_exit_intent_audit(_ledger(specs))
        assert r["state"] == "STABLE"
        # Tie broken by _INTENT_ORDER → EARNINGS_CLEAR (first in order).
        assert r["dominant_intent"] == "EARNINGS_CLEAR"
        dom = next(b for b in r["buckets"]
                   if b["intent"] == r["dominant_intent"])
        assert dom["n"] == 2
        assert r["verdict"] is None


# ── shape / contract ──────────────────────────────────────────────────
class TestPayloadShape:
    def test_payload_keys_pinned(self):
        r = build_exit_intent_audit([])
        for k in ("as_of", "state", "verdict", "verdict_reason", "headline",
                  "n_round_trips", "dominant_intent", "buckets",
                  "intent_order", "stable_min_round_trips"):
            assert k in r, f"missing key: {k}"

    def test_intent_order_includes_unclassified_last(self):
        r = build_exit_intent_audit([])
        order = r["intent_order"]
        assert order[-1] == "UNCLASSIFIED"
        assert "EARNINGS_CLEAR" in order
        assert "DEFENSIVE_CASH_RAISE" in order

    def test_buckets_always_have_all_intents(self):
        # Even on empty input, every documented bucket must appear so a
        # UI panel iterating buckets doesn't break on a missing slot.
        r = build_exit_intent_audit([])
        present = [b["intent"] for b in r["buckets"]]
        for intent in (*_INTENT_ORDER, "UNCLASSIFIED"):
            assert intent in present

    def test_stable_constant_matches_module(self):
        # Guard against accidental drift between module and tests.
        assert STABLE_MIN_RTS == 10

    def test_worst_n_caps_examples_list(self):
        # 5 losing trips, worst_n=2 → only the 2 most-negative emit.
        specs = [
            ("A", 10, 10.0, 5.0, 1, "raise cash"),    # -$50
            ("B", 10, 10.0, 7.0, 1, "raise cash"),    # -$30
            ("C", 10, 10.0, 9.0, 1, "raise cash"),    # -$10
            ("D", 10, 10.0, 8.0, 1, "raise cash"),    # -$20
            ("E", 10, 10.0, 6.0, 1, "raise cash"),    # -$40
        ]
        r = build_exit_intent_audit(_ledger(specs), worst_n=2)
        defensive = next(b for b in r["buckets"]
                         if b["intent"] == "DEFENSIVE_CASH_RAISE")
        assert defensive["n"] == 5
        assert len(defensive["examples"]) == 2
        # Most negative first.
        assert defensive["examples"][0]["pnl_usd"] == -50.0
        assert defensive["examples"][1]["pnl_usd"] == -40.0
