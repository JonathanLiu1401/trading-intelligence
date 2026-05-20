"""Tests for paper_trader.analytics.holding_period_distribution.

Asserts EXACT bucket assignment around the SCALP/INTRADAY/OVERNIGHT/SWING/
TREND/POSITION edges, exact per-bucket arithmetic (total_pnl_usd, win_rate,
share_of_trips_pct, share_of_abs_pnl_pct), the alpha-engine / dominant /
worst pick logic, the STABLE_MIN_TRIPS gate, and degrade-not-raise on
garbage round-trip rows. The boundary edges in particular are pinned —
silently widening SCALP_MAX_HOURS would shift a whole bucket of trips.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.holding_period_distribution import (
    BUCKETS,
    INTRADAY_MAX_HOURS,
    OVERNIGHT_MAX_HOURS,
    SCALP_MAX_HOURS,
    STABLE_MIN_TRIPS,
    SWING_MAX_HOURS,
    TREND_MAX_HOURS,
    build_holding_period_distribution,
)


def _rt(ticker: str, *, hold_hours: float, pnl_usd: float,
        ticker_qty: float = 1.0) -> dict:
    """Construct a round-trip row matching build_round_trips' output shape.
    Only the fields the bucketer reads are populated; the rest are
    placeholders so the dict round-trips through a JSON serialiser."""
    cost = 100.0
    proceeds = cost + pnl_usd
    return {
        "ticker": ticker,
        "type": "stock",
        "strike": None,
        "expiry": None,
        "entry_ts": "2026-05-01T00:00:00+00:00",
        "exit_ts": "2026-05-01T00:00:00+00:00",
        "qty": ticker_qty,
        "cost": cost,
        "proceeds": proceeds,
        "pnl_usd": pnl_usd,
        "pnl_pct": round(pnl_usd / cost * 100, 4),
        "hold_days": hold_hours / 24.0,
        "n_buys": 1,
        "n_sells": 1,
        "entry_trade_ids": [1],
        "exit_trade_ids": [2],
    }


class TestBucketAssignment:
    def test_scalp_under_one_hour(self):
        out = build_holding_period_distribution(
            [_rt("AAA", hold_hours=0.4, pnl_usd=1.0)],
        )
        scalp = next(b for b in out["buckets"] if b["bucket"] == "SCALP")
        assert scalp["n_trips"] == 1

    def test_intraday_boundary_inclusive_at_one_hour(self):
        # Edge: exactly 1.0h ⇒ INTRADAY (lower edge inclusive, upper
        # exclusive — the `< INTRADAY_MAX_HOURS` test in _bucket_for_hours).
        # A drift to `<=` on SCALP_MAX_HOURS would land this in SCALP and
        # break the discrimination.
        out = build_holding_period_distribution(
            [_rt("AAA", hold_hours=SCALP_MAX_HOURS, pnl_usd=1.0)],
        )
        intra = next(b for b in out["buckets"] if b["bucket"] == "INTRADAY")
        scalp = next(b for b in out["buckets"] if b["bucket"] == "SCALP")
        assert intra["n_trips"] == 1
        assert scalp["n_trips"] == 0

    def test_overnight_lower_edge_at_six_hours(self):
        out = build_holding_period_distribution(
            [_rt("AAA", hold_hours=INTRADAY_MAX_HOURS, pnl_usd=1.0)],
        )
        on = next(b for b in out["buckets"] if b["bucket"] == "OVERNIGHT")
        assert on["n_trips"] == 1

    def test_swing_lower_edge_at_one_day(self):
        out = build_holding_period_distribution(
            [_rt("AAA", hold_hours=OVERNIGHT_MAX_HOURS, pnl_usd=1.0)],
        )
        sw = next(b for b in out["buckets"] if b["bucket"] == "SWING")
        assert sw["n_trips"] == 1

    def test_trend_lower_edge_at_three_days(self):
        out = build_holding_period_distribution(
            [_rt("AAA", hold_hours=SWING_MAX_HOURS, pnl_usd=1.0)],
        )
        tr = next(b for b in out["buckets"] if b["bucket"] == "TREND")
        assert tr["n_trips"] == 1

    def test_position_lower_edge_at_seven_days(self):
        out = build_holding_period_distribution(
            [_rt("AAA", hold_hours=TREND_MAX_HOURS, pnl_usd=1.0)],
        )
        pos = next(b for b in out["buckets"] if b["bucket"] == "POSITION")
        assert pos["n_trips"] == 1

    def test_canonical_bucket_order(self):
        # The order in `buckets` must be SCALP → POSITION for the UI to
        # render left-to-right by duration. A drift could reorder
        # surreptitiously after a refactor.
        out = build_holding_period_distribution([])
        assert [b["bucket"] for b in out["buckets"]] == list(BUCKETS)


class TestStateLadder:
    def test_no_data_empty_input(self):
        out = build_holding_period_distribution([])
        assert out["state"] == "NO_DATA"
        assert out["n_trips"] == 0
        assert out["total_pnl_usd"] == 0.0
        assert out["alpha_engine"] is None
        assert out["dominant_bucket"] is None
        assert out["worst_bucket"] is None
        # NO_DATA headline calls out the absence explicitly.
        assert "not yet available" in out["headline"]

    def test_no_data_when_only_garbage_rows(self):
        out = build_holding_period_distribution([
            None, "garbage", {}, {"hold_days": None, "pnl_usd": 1.0},
            {"hold_days": 0.1, "pnl_usd": None},
        ])
        assert out["state"] == "NO_DATA"
        # n_unbucketed accounts for the rows that couldn't classify.
        assert out["n_unbucketed"] >= 4

    def test_insufficient_below_stable_min(self):
        rts = [_rt(f"AAA{i}", hold_hours=0.5, pnl_usd=1.0)
               for i in range(STABLE_MIN_TRIPS - 1)]
        out = build_holding_period_distribution(rts)
        assert out["state"] == "INSUFFICIENT"
        # Bucket rows still real, only the verdict is withheld.
        scalp = next(b for b in out["buckets"] if b["bucket"] == "SCALP")
        assert scalp["n_trips"] == STABLE_MIN_TRIPS - 1
        assert out["alpha_engine"] is None
        assert out["dominant_bucket"] is None
        assert "withheld" in out["headline"]

    def test_ok_at_stable_min(self):
        # Exactly STABLE_MIN_TRIPS in one bucket — must promote to OK.
        rts = [_rt(f"AAA{i}", hold_hours=0.5, pnl_usd=1.0)
               for i in range(STABLE_MIN_TRIPS)]
        out = build_holding_period_distribution(rts)
        assert out["state"] == "OK"
        assert out["alpha_engine"] == "SCALP"
        assert out["dominant_bucket"] == "SCALP"


class TestPerBucketArithmetic:
    def test_total_pnl_and_win_rate(self):
        # 3 winners +1, 2 losers -1 → total +1, win_rate = 3/5 = 60%.
        rts = [
            _rt("A", hold_hours=0.5, pnl_usd=1.0),
            _rt("B", hold_hours=0.5, pnl_usd=1.0),
            _rt("C", hold_hours=0.5, pnl_usd=1.0),
            _rt("D", hold_hours=0.5, pnl_usd=-1.0),
            _rt("E", hold_hours=0.5, pnl_usd=-1.0),
        ]
        out = build_holding_period_distribution(rts)
        scalp = next(b for b in out["buckets"] if b["bucket"] == "SCALP")
        assert scalp["n_trips"] == 5
        assert scalp["n_winners"] == 3
        assert scalp["n_losers"] == 2
        assert scalp["total_pnl_usd"] == 1.0
        assert scalp["win_rate_pct"] == 60.0
        assert scalp["share_of_trips_pct"] == 100.0
        assert out["win_rate_pct"] == 60.0

    def test_share_of_abs_pnl(self):
        # SCALP: 4 winners @ +1 = +4 → |P/L|=4
        # SWING: 1 winner @ +6 = +6 → |P/L|=6
        # Total |P/L| = 10. Shares: SCALP=40, SWING=60.
        rts = [
            _rt("A", hold_hours=0.5, pnl_usd=1.0),
            _rt("B", hold_hours=0.5, pnl_usd=1.0),
            _rt("C", hold_hours=0.5, pnl_usd=1.0),
            _rt("D", hold_hours=0.5, pnl_usd=1.0),
            _rt("E", hold_hours=36.0, pnl_usd=6.0),  # SWING
        ]
        out = build_holding_period_distribution(rts)
        scalp = next(b for b in out["buckets"] if b["bucket"] == "SCALP")
        swing = next(b for b in out["buckets"] if b["bucket"] == "SWING")
        assert scalp["share_of_abs_pnl_pct"] == 40.0
        assert swing["share_of_abs_pnl_pct"] == 60.0
        # SWING is the engine even though SCALP dominates by trip count.
        assert out["alpha_engine"] == "SWING"
        assert out["dominant_bucket"] == "SCALP"

    def test_median_pnl(self):
        # SCALP trips: -2, -1, 0, +1, +2 → median 0; +3 winners = 5.
        rts = [
            _rt("A", hold_hours=0.5, pnl_usd=-2.0),
            _rt("B", hold_hours=0.5, pnl_usd=-1.0),
            _rt("C", hold_hours=0.5, pnl_usd=0.0),
            _rt("D", hold_hours=0.5, pnl_usd=1.0),
            _rt("E", hold_hours=0.5, pnl_usd=2.0),
        ]
        out = build_holding_period_distribution(rts)
        scalp = next(b for b in out["buckets"] if b["bucket"] == "SCALP")
        assert scalp["median_pnl_usd"] == 0.0
        assert scalp["avg_pnl_usd"] == 0.0

    def test_zero_pnl_excluded_from_winrate_decided(self):
        # A pnl=0 trip should count in n_trips but NOT in n_winners or
        # n_losers; win_rate uses decided = winners + losers as the
        # denominator (the track_record convention).
        rts = [
            _rt("A", hold_hours=0.5, pnl_usd=1.0),
            _rt("B", hold_hours=0.5, pnl_usd=0.0),
            _rt("C", hold_hours=0.5, pnl_usd=-1.0),
            _rt("D", hold_hours=0.5, pnl_usd=1.0),
            _rt("E", hold_hours=0.5, pnl_usd=1.0),
        ]
        out = build_holding_period_distribution(rts)
        scalp = next(b for b in out["buckets"] if b["bucket"] == "SCALP")
        assert scalp["n_winners"] == 3
        assert scalp["n_losers"] == 1
        # decided = 4; 3/4 = 75%.
        assert scalp["win_rate_pct"] == 75.0


class TestVerdictPicks:
    def test_alpha_engine_must_be_net_positive(self):
        # All buckets net negative ⇒ alpha_engine is None (least-loss
        # is not "alpha"); worst_bucket points to most-negative.
        rts = [
            _rt("A", hold_hours=0.5, pnl_usd=-1.0),
            _rt("B", hold_hours=0.5, pnl_usd=-1.0),
            _rt("C", hold_hours=0.5, pnl_usd=-1.0),
            _rt("D", hold_hours=36.0, pnl_usd=-5.0),  # SWING bleeds
            _rt("E", hold_hours=36.0, pnl_usd=-5.0),
        ]
        out = build_holding_period_distribution(rts)
        assert out["state"] == "OK"
        assert out["alpha_engine"] is None
        assert out["worst_bucket"] == "SWING"
        # Headline mentions the bleed.
        assert "bleed" in out["headline"]

    def test_dominant_ties_resolve_in_canonical_order(self):
        # Two buckets at 3 trips each; canonical order SCALP→...→POSITION.
        # SCALP comes before SWING — must win the tie.
        rts = [
            _rt("A", hold_hours=0.5, pnl_usd=1.0),
            _rt("B", hold_hours=0.5, pnl_usd=1.0),
            _rt("C", hold_hours=0.5, pnl_usd=1.0),
            _rt("D", hold_hours=36.0, pnl_usd=10.0),
            _rt("E", hold_hours=36.0, pnl_usd=10.0),
            _rt("F", hold_hours=36.0, pnl_usd=10.0),
        ]
        out = build_holding_period_distribution(rts)
        # SWING has the most $ → engine. Tied n_trips → SCALP first by
        # canonical order ⇒ dominant.
        assert out["alpha_engine"] == "SWING"
        assert out["dominant_bucket"] == "SCALP"

    def test_engine_and_dominant_match_when_same_bucket(self):
        rts = [_rt(f"A{i}", hold_hours=0.5, pnl_usd=1.0)
               for i in range(STABLE_MIN_TRIPS)]
        out = build_holding_period_distribution(rts)
        assert out["alpha_engine"] == "SCALP"
        assert out["dominant_bucket"] == "SCALP"
        # Headline picks the "engine" framing when they match.
        assert "alpha engine" in out["headline"]


class TestDegradeNotRaise:
    def test_garbage_round_trips_dont_raise(self):
        # Should categorise these as unbucketed and still emit a valid
        # NO_DATA / INSUFFICIENT response.
        rts = [
            None,
            "not a dict",
            {"pnl_usd": "bad", "hold_days": 0.5},     # pnl coerce fails
            {"pnl_usd": 1.0, "hold_days": "bad"},     # hold coerce fails
            {"pnl_usd": 1.0, "hold_days": -1.0},      # negative hold
            {"pnl_usd": 1.0, "hold_days": float("nan")},  # NaN hold
            {"pnl_usd": float("nan"), "hold_days": 0.5},  # NaN pnl
            {"pnl_usd": True, "hold_days": 0.5},      # bool ≠ number
        ]
        out = build_holding_period_distribution(rts)
        # None of these are valid → all unbucketed.
        assert out["n_unbucketed"] == len(rts)
        assert out["n_trips"] == 0
        assert out["state"] == "NO_DATA"

    def test_non_list_input_returns_no_data(self):
        out = build_holding_period_distribution(None)  # type: ignore[arg-type]
        assert out["state"] == "NO_DATA"
        out = build_holding_period_distribution("not a list")  # type: ignore[arg-type]
        assert out["state"] == "NO_DATA"
