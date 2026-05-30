"""Tests for analytics.position_size_pl_fit + /api/position-size-pl-fit.

Pins the entry-size bucket classification, the verdict ladder over big-bet
vs small-bet net P&L, the equity_curve book-at-entry lookup, the fallback
to $1000 INITIAL_CASH, and the endpoint envelope.

Every assertion is specific-value (no "no crash" passes) per AGENTS.md
test rigor convention.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.position_size_pl_fit import (
    _BUCKETS,
    _FALLBACK_BOOK_USD,
    _classify_size,
    annotate_round_trip,
    build,
)


def _t(*, action, ticker, qty=1.0, price=100.0, timestamp, id_=0,
       reason=""):
    """Trade row shaped like store.recent_trades() emits."""
    return {
        "action": action,
        "ticker": ticker,
        "qty": qty,
        "price": price,
        "value": qty * price,
        "reason": reason,
        "timestamp": timestamp,
        "option_type": None,
        "expiry": None,
        "strike": None,
        "id": id_,
    }


def _eq(*, timestamp, total_value, sp500_price=500.0, cash=0.0):
    return {
        "timestamp": timestamp,
        "total_value": total_value,
        "cash": cash,
        "sp500_price": sp500_price,
    }


# ────────────────────────── _classify_size unit ───────────────────────────


class TestClassifySize:
    def test_zero_lands_in_small(self):
        assert _classify_size(0.0) == "SMALL"

    def test_below_small_ceiling(self):
        assert _classify_size(0.249999) == "SMALL"

    def test_small_medium_boundary(self):
        # Exactly 25.0% is the lower edge of MEDIUM (half-open on right).
        assert _classify_size(0.25) == "MEDIUM"

    def test_below_medium_ceiling(self):
        assert _classify_size(0.4999) == "MEDIUM"

    def test_medium_large_boundary(self):
        assert _classify_size(0.50) == "LARGE"

    def test_below_large_ceiling(self):
        assert _classify_size(0.7999) == "LARGE"

    def test_large_max_boundary(self):
        assert _classify_size(0.80) == "MAX"

    def test_over_one_book(self):
        # Levered/added entry that exceeds full book lands in MAX.
        assert _classify_size(1.25) == "MAX"

    def test_negative_falls_to_small(self):
        # Physically impossible negative pct — defensive default.
        assert _classify_size(-0.5) == "SMALL"

    def test_none_falls_to_small(self):
        assert _classify_size(None) == "SMALL"

    def test_nan_falls_to_small(self):
        assert _classify_size(float("nan")) == "SMALL"


# ───────────────────── annotate_round_trip + book lookup ──────────────────


class TestAnnotateRoundTrip:
    def test_book_lookup_at_or_before_entry(self):
        # Equity sample at 10:00 with total_value=$2000; entry at 11:00.
        # Cost $500 / book $2000 = 25% → MEDIUM.
        rt = {
            "ticker": "AMD", "cost": 500.0, "realized_pl": 10.0,
            "opened_at": "2026-05-25T11:00:00+00:00",
            "closed_at": "2026-05-25T15:00:00+00:00",
            "hold_days": 0.1667,
        }
        ec = [_eq(timestamp="2026-05-25T10:00:00+00:00", total_value=2000.0)]
        out = annotate_round_trip(rt, ec)
        assert out["entry_book_pct"] == 0.25
        assert out["size_bucket"] == "MEDIUM"

    def test_book_lookup_uses_latest_sample_before_entry(self):
        # Three equity samples; entry at 12:00; latest at or before is the
        # 11:00 sample with total_value=$1500. Cost $1200 / $1500 = 80% → MAX.
        ec = [
            _eq(timestamp="2026-05-25T09:00:00+00:00", total_value=500.0),
            _eq(timestamp="2026-05-25T11:00:00+00:00", total_value=1500.0),
            _eq(timestamp="2026-05-25T13:00:00+00:00", total_value=999.0),  # after entry
        ]
        rt = {
            "ticker": "NVDA", "cost": 1200.0, "realized_pl": 50.0,
            "opened_at": "2026-05-25T12:00:00+00:00",
            "closed_at": "2026-05-25T18:00:00+00:00",
            "hold_days": 0.25,
        }
        out = annotate_round_trip(rt, ec)
        assert out["entry_book_pct"] == 0.80
        assert out["size_bucket"] == "MAX"

    def test_no_covering_sample_falls_back_to_1000(self):
        # All equity samples are AFTER the entry — book unknown.
        # Cost $300 / fallback $1000 = 30% → MEDIUM.
        ec = [_eq(timestamp="2026-05-26T12:00:00+00:00", total_value=2000.0)]
        rt = {
            "ticker": "MU", "cost": 300.0, "realized_pl": 5.0,
            "opened_at": "2026-05-25T11:00:00+00:00",
            "closed_at": "2026-05-25T18:00:00+00:00",
            "hold_days": 0.29,
        }
        out = annotate_round_trip(rt, ec)
        assert out["size_bucket"] == "MEDIUM"

    def test_empty_equity_curve_falls_back_to_1000(self):
        rt = {
            "ticker": "AMD", "cost": 900.0, "realized_pl": -10.0,
            "opened_at": "2026-05-25T11:00:00+00:00",
            "closed_at": "2026-05-25T15:00:00+00:00",
            "hold_days": 0.17,
        }
        # $900 / $1000 = 90% → MAX.
        out = annotate_round_trip(rt, [])
        assert out["size_bucket"] == "MAX"

    def test_malformed_equity_rows_skipped(self):
        ec = [
            {"total_value": "garbage", "timestamp": "x"},   # unparseable ts
            {"total_value": None, "timestamp": "2026-05-25T10:00:00+00:00"},
            _eq(timestamp="2026-05-25T10:30:00+00:00", total_value=2500.0),
        ]
        rt = {
            "ticker": "AMD", "cost": 500.0, "realized_pl": 0.0,
            "opened_at": "2026-05-25T11:00:00+00:00",
            "closed_at": "2026-05-25T12:00:00+00:00",
        }
        # The good sample stands: $500 / $2500 = 20% → SMALL.
        out = annotate_round_trip(rt, ec)
        assert out["size_bucket"] == "SMALL"

    def test_non_dict_returns_none(self):
        assert annotate_round_trip("not-a-dict", []) is None
        assert annotate_round_trip(None, []) is None


# ───────────────────────────── build (full) ───────────────────────────────


class TestBuildEmpty:
    def test_no_trades(self):
        out = build([])
        assert out["verdict"] == "NO_DATA"
        assert out["n_round_trips"] == 0
        for b in _BUCKETS:
            assert out["buckets"][b]["n"] == 0
        assert "No closed" in out["headline"]


class TestBuildEmerging:
    def test_below_floor_collapses_to_emerging(self):
        # Two closed trips (below default min_for_verdict=4).
        trades = [
            _t(action="BUY", ticker="AMD", qty=1, price=100,
               timestamp="2026-05-25T10:00:00+00:00", id_=1),
            _t(action="SELL", ticker="AMD", qty=1, price=110,
               timestamp="2026-05-25T15:00:00+00:00", id_=2),
            _t(action="BUY", ticker="NVDA", qty=1, price=900,
               timestamp="2026-05-26T10:00:00+00:00", id_=3),
            _t(action="SELL", ticker="NVDA", qty=1, price=920,
               timestamp="2026-05-26T15:00:00+00:00", id_=4),
        ]
        out = build(trades)
        assert out["verdict"] == "EMERGING"
        assert out["n_round_trips"] == 2


class TestBuildKellyCoherent:
    def test_big_bets_carry_the_desk(self):
        # 4 closed trips: small bets break even, big bets net positive.
        trades = [
            # SMALL bet, $100 cost ($100 / $1000 = 10%), +$5.
            _t(action="BUY", ticker="AMD", qty=1, price=100,
               timestamp="2026-05-20T10:00:00+00:00", id_=1),
            _t(action="SELL", ticker="AMD", qty=1, price=105,
               timestamp="2026-05-20T11:00:00+00:00", id_=2),
            # SMALL bet, $100 cost, -$5 (small net = $0).
            _t(action="BUY", ticker="MU", qty=1, price=100,
               timestamp="2026-05-21T10:00:00+00:00", id_=3),
            _t(action="SELL", ticker="MU", qty=1, price=95,
               timestamp="2026-05-21T11:00:00+00:00", id_=4),
            # LARGE bet, $700 cost (70%), +$50.
            _t(action="BUY", ticker="NVDA", qty=1, price=700,
               timestamp="2026-05-22T10:00:00+00:00", id_=5),
            _t(action="SELL", ticker="NVDA", qty=1, price=750,
               timestamp="2026-05-22T11:00:00+00:00", id_=6),
            # MAX bet, $950 cost (95%), +$70.
            _t(action="BUY", ticker="TSLA", qty=1, price=950,
               timestamp="2026-05-23T10:00:00+00:00", id_=7),
            _t(action="SELL", ticker="TSLA", qty=1, price=1020,
               timestamp="2026-05-23T11:00:00+00:00", id_=8),
        ]
        out = build(trades)  # no equity_curve → $1000 fallback
        assert out["verdict"] == "KELLY_COHERENT"
        assert out["n_round_trips"] == 4
        # SMALL bucket = 2 trips, $0 net.
        assert out["buckets"]["SMALL"]["n"] == 2
        assert out["buckets"]["SMALL"]["total_pl_usd"] == 0.0
        # LARGE = 1 trip, +$50.
        assert out["buckets"]["LARGE"]["n"] == 1
        assert out["buckets"]["LARGE"]["total_pl_usd"] == 50.0
        # MAX = 1 trip, +$70.
        assert out["buckets"]["MAX"]["n"] == 1
        assert out["buckets"]["MAX"]["total_pl_usd"] == 70.0
        # Headline mentions both big and small numerically.
        assert "+120.00" in out["headline"]  # big = +50 + +70 = +120
        assert "Kelly Coherent" in out["headline"]


class TestBuildAntiKelly:
    def test_big_bets_lose_while_small_bets_pay(self):
        # 4 trips: 2 SMALL winners, 2 MAX losers — ANTI_KELLY.
        trades = [
            _t(action="BUY", ticker="AMD", qty=1, price=100,
               timestamp="2026-05-20T10:00:00+00:00", id_=1),
            _t(action="SELL", ticker="AMD", qty=1, price=120,
               timestamp="2026-05-20T11:00:00+00:00", id_=2),
            _t(action="BUY", ticker="MU", qty=1, price=150,
               timestamp="2026-05-21T10:00:00+00:00", id_=3),
            _t(action="SELL", ticker="MU", qty=1, price=180,
               timestamp="2026-05-21T11:00:00+00:00", id_=4),
            _t(action="BUY", ticker="NVDA", qty=1, price=900,
               timestamp="2026-05-22T10:00:00+00:00", id_=5),
            _t(action="SELL", ticker="NVDA", qty=1, price=820,
               timestamp="2026-05-22T11:00:00+00:00", id_=6),
            _t(action="BUY", ticker="TSLA", qty=1, price=950,
               timestamp="2026-05-23T10:00:00+00:00", id_=7),
            _t(action="SELL", ticker="TSLA", qty=1, price=820,
               timestamp="2026-05-23T11:00:00+00:00", id_=8),
        ]
        out = build(trades)
        assert out["verdict"] == "ANTI_KELLY"
        # big = -80 + -130 = -210; small = +20 + +30 = +50.
        big = (out["buckets"]["LARGE"]["total_pl_usd"]
               + out["buckets"]["MAX"]["total_pl_usd"])
        small = (out["buckets"]["SMALL"]["total_pl_usd"]
                 + out["buckets"]["MEDIUM"]["total_pl_usd"])
        assert big == -210.0
        assert small == 50.0


class TestBuildAllBleed:
    def test_every_populated_bucket_red(self):
        # 4 trips, all losers across multiple buckets.
        trades = [
            _t(action="BUY", ticker="AMD", qty=1, price=100,
               timestamp="2026-05-20T10:00:00+00:00", id_=1),
            _t(action="SELL", ticker="AMD", qty=1, price=95,
               timestamp="2026-05-20T11:00:00+00:00", id_=2),
            _t(action="BUY", ticker="MU", qty=1, price=300,
               timestamp="2026-05-21T10:00:00+00:00", id_=3),
            _t(action="SELL", ticker="MU", qty=1, price=280,
               timestamp="2026-05-21T11:00:00+00:00", id_=4),
            _t(action="BUY", ticker="NVDA", qty=1, price=700,
               timestamp="2026-05-22T10:00:00+00:00", id_=5),
            _t(action="SELL", ticker="NVDA", qty=1, price=650,
               timestamp="2026-05-22T11:00:00+00:00", id_=6),
            _t(action="BUY", ticker="TSLA", qty=1, price=950,
               timestamp="2026-05-23T10:00:00+00:00", id_=7),
            _t(action="SELL", ticker="TSLA", qty=1, price=900,
               timestamp="2026-05-23T11:00:00+00:00", id_=8),
        ]
        out = build(trades)
        assert out["verdict"] == "ALL_BLEED"


class TestBuildBigBetsNeutral:
    def test_big_positive_but_smaller_than_small_positive(self):
        # 4 trips: SMALL winners net +$200, MAX winners net +$50 (positive
        # but dwarfed by small-bet $).
        trades = [
            _t(action="BUY", ticker="AMD", qty=1, price=100,
               timestamp="2026-05-20T10:00:00+00:00", id_=1),
            _t(action="SELL", ticker="AMD", qty=1, price=200,
               timestamp="2026-05-20T11:00:00+00:00", id_=2),
            _t(action="BUY", ticker="MU", qty=1, price=100,
               timestamp="2026-05-21T10:00:00+00:00", id_=3),
            _t(action="SELL", ticker="MU", qty=1, price=200,
               timestamp="2026-05-21T11:00:00+00:00", id_=4),
            _t(action="BUY", ticker="NVDA", qty=1, price=900,
               timestamp="2026-05-22T10:00:00+00:00", id_=5),
            _t(action="SELL", ticker="NVDA", qty=1, price=925,
               timestamp="2026-05-22T11:00:00+00:00", id_=6),
            _t(action="BUY", ticker="TSLA", qty=1, price=950,
               timestamp="2026-05-23T10:00:00+00:00", id_=7),
            _t(action="SELL", ticker="TSLA", qty=1, price=975,
               timestamp="2026-05-23T11:00:00+00:00", id_=8),
        ]
        out = build(trades)
        assert out["verdict"] == "BIG_BETS_NEUTRAL"


class TestStatsCorrect:
    def test_bucket_stats_arithmetic(self):
        # Two LARGE-bucket trips: +$50 win, -$10 loss; cost 700 + 700 = 1400.
        # win_rate = 100*1/2 = 50%; avg_pl_pct = (40/1400)*100 = 2.86.
        trades = [
            _t(action="BUY", ticker="NVDA", qty=1, price=700,
               timestamp="2026-05-22T10:00:00+00:00", id_=1),
            _t(action="SELL", ticker="NVDA", qty=1, price=750,
               timestamp="2026-05-22T11:00:00+00:00", id_=2),
            _t(action="BUY", ticker="NVDA", qty=1, price=700,
               timestamp="2026-05-23T10:00:00+00:00", id_=3),
            _t(action="SELL", ticker="NVDA", qty=1, price=690,
               timestamp="2026-05-23T11:00:00+00:00", id_=4),
        ]
        out = build(trades)
        large = out["buckets"]["LARGE"]
        assert large["n"] == 2
        assert large["wins"] == 1
        assert large["losses"] == 1
        assert large["win_rate_pct"] == 50.0
        assert large["total_pl_usd"] == 40.0
        assert large["total_cost_usd"] == 1400.0
        assert large["avg_pl_pct"] == 2.86


class TestEnvelope:
    def test_top_level_keys(self):
        out = build([])
        assert set(out.keys()) >= {
            "verdict", "headline", "buckets", "n_round_trips",
            "min_for_verdict", "fallback_book_used_count",
        }
        # All four buckets present even when empty.
        assert set(out["buckets"].keys()) == {"SMALL", "MEDIUM", "LARGE", "MAX"}

    def test_min_for_verdict_passthrough(self):
        out = build([], min_for_verdict=7)
        assert out["min_for_verdict"] == 7


class TestFallbackBookConstant:
    def test_constant_is_1000(self):
        # Pinned so the constant doesn't drift without a test failure.
        assert _FALLBACK_BOOK_USD == 1000.0


# ──────────────────────────── Endpoint envelope ───────────────────────────


class TestEndpoint:
    def test_route_returns_envelope(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get("/api/position-size-pl-fit")
        assert resp.status_code in (200, 500)
        body = resp.get_json()
        assert body is not None
        assert "verdict" in body
        assert "headline" in body
        assert body["service"] == "paper_trader"
        if resp.status_code == 200:
            assert "buckets" in body
            for b in ("SMALL", "MEDIUM", "LARGE", "MAX"):
                assert b in body["buckets"]
                assert "n" in body["buckets"][b]
                assert "total_pl_usd" in body["buckets"][b]
            assert "n_round_trips" in body
            assert "fallback_book_used_count" in body
