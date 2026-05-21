"""Tests for paper_trader.analytics.repeat_loser.

The new per-ticker losing-streak detector. The aggregate ``build_streak``
already covers cross-ticker streak behaviour (locked by
tests/test_streak.py / test_streak_reporter.py); these tests pin the
per-ticker logic specifically.

Notation in the fixtures: a "trip" is a BUY/SELL pair on a single ticker
inserted into a flat trade ledger. The helper ``_pair`` emits a winning or
losing pair at a given timestamp prefix so the ordering by ``exit_ts``
matches the test's intent.
"""
from __future__ import annotations

import pytest

from paper_trader.analytics.repeat_loser import (
    REPEAT_LOSER_MIN,
    build_repeat_loser,
)


def _pair(ticker: str, day: str, *, win: bool,
          cost: float = 100.0, pnl: float | None = None) -> list[dict]:
    """Yield a [BUY, SELL] pair on ``ticker`` whose round-trip PnL is
    ``+5`` (win), ``-5`` (loss), or the explicit ``pnl`` override.

    ``day`` is a date string ('2026-05-01'); the BUY lands at 09:00 UTC
    and the SELL at 15:00 UTC the same day so build_round_trips closes
    a complete trip.
    """
    if pnl is None:
        pnl = 5.0 if win else -5.0
    proceeds = cost + pnl
    return [
        {
            "ticker": ticker, "action": "BUY", "qty": 1.0,
            "price": cost, "value": cost,
            "timestamp": f"{day}T09:00:00+00:00",
            "option_type": None, "strike": None, "expiry": None, "id": None,
        },
        {
            "ticker": ticker, "action": "SELL", "qty": 1.0,
            "price": proceeds, "value": proceeds,
            "timestamp": f"{day}T15:00:00+00:00",
            "option_type": None, "strike": None, "expiry": None, "id": None,
        },
    ]


class TestEmptyAndNoData:
    def test_empty_trades_returns_no_data(self):
        rep = build_repeat_loser([])
        assert rep["state"] == "NO_DATA"
        assert rep["verdict"] is None
        assert rep["offenders"] == []
        assert rep["n_round_trips"] == 0
        assert rep["threshold"] == REPEAT_LOSER_MIN

    def test_no_data_headline_is_set(self):
        rep = build_repeat_loser([])
        assert "No closed round-trips" in rep["headline"]

    def test_open_position_only_is_no_data(self):
        # A BUY with no matching SELL is an open lot; round_trips emits
        # nothing → no_data, not OK.
        trades = [{
            "ticker": "NVDA", "action": "BUY", "qty": 1.0,
            "price": 100.0, "value": 100.0,
            "timestamp": "2026-05-01T09:00:00+00:00",
            "option_type": None, "strike": None, "expiry": None, "id": None,
        }]
        rep = build_repeat_loser(trades)
        assert rep["state"] == "NO_DATA"


class TestSingleTicker:
    def test_single_winning_trip_is_ok(self):
        rep = build_repeat_loser(_pair("NVDA", "2026-05-01", win=True))
        assert rep["state"] == "OK"
        assert rep["verdict"] is None
        assert rep["n_offenders"] == 0
        assert rep["per_ticker"]["NVDA"]["current_loss_streak"] == 0

    def test_single_losing_trip_below_threshold(self):
        rep = build_repeat_loser(_pair("NVDA", "2026-05-01", win=False))
        # threshold is 2 by default, 1 loss is below it
        assert rep["state"] == "OK"
        assert rep["verdict"] is None
        # but it IS tracked in per_ticker (just not surfaced as offender)
        assert rep["per_ticker"]["NVDA"]["current_loss_streak"] == 1
        assert rep["per_ticker"]["NVDA"]["current_loss_usd"] == -5.0

    def test_two_consecutive_losses_triggers_repeat_loser(self):
        trades = (
            _pair("LITE", "2026-05-01", win=False, pnl=-4.0)
            + _pair("LITE", "2026-05-05", win=False, pnl=-6.0)
        )
        rep = build_repeat_loser(trades)
        assert rep["state"] == "REPEAT_LOSER"
        assert rep["verdict"] == "REPEAT_LOSER"
        assert rep["n_offenders"] == 1
        off = rep["offenders"][0]
        assert off["ticker"] == "LITE"
        assert off["streak"] == 2
        assert off["loss_usd"] == pytest.approx(-10.0)
        assert off["last_exit_ts"] == "2026-05-05T15:00:00+00:00"

    def test_loss_streak_resets_on_win(self):
        # LITE: L, L, W, L → current streak is 1 (not 3), since the W resets
        trades = (
            _pair("LITE", "2026-05-01", win=False)
            + _pair("LITE", "2026-05-02", win=False)
            + _pair("LITE", "2026-05-03", win=True)
            + _pair("LITE", "2026-05-04", win=False)
        )
        rep = build_repeat_loser(trades)
        assert rep["state"] == "OK"  # current streak is 1, below threshold
        assert rep["per_ticker"]["LITE"]["current_loss_streak"] == 1
        assert rep["per_ticker"]["LITE"]["n_round_trips"] == 4

    def test_flat_trip_does_not_reset_streak(self):
        # mirror streak.py: a flat trip (pnl=0) skips, doesn't break a streak
        trades = (
            _pair("MU", "2026-05-01", win=False, pnl=-3.0)
            + _pair("MU", "2026-05-02", win=False, pnl=0.0)   # flat
            + _pair("MU", "2026-05-03", win=False, pnl=-2.0)
        )
        rep = build_repeat_loser(trades)
        # 2 losses + 1 flat = streak 2 (flat skipped), sum -5
        assert rep["state"] == "REPEAT_LOSER"
        off = rep["offenders"][0]
        assert off["streak"] == 2
        assert off["loss_usd"] == pytest.approx(-5.0)


class TestMultipleTickers:
    def test_per_ticker_isolation(self):
        # NVDA wins, LITE loses 3× → only LITE flags
        trades = (
            _pair("NVDA", "2026-05-01", win=True)
            + _pair("LITE", "2026-05-02", win=False)
            + _pair("NVDA", "2026-05-03", win=True)
            + _pair("LITE", "2026-05-04", win=False)
            + _pair("LITE", "2026-05-05", win=False)
        )
        rep = build_repeat_loser(trades)
        assert rep["state"] == "REPEAT_LOSER"
        assert rep["n_offenders"] == 1
        assert rep["offenders"][0]["ticker"] == "LITE"
        assert rep["offenders"][0]["streak"] == 3

    def test_multiple_offenders_sorted_by_streak(self):
        # LITE × 2, MU × 4 → MU should sort first (longer streak)
        trades = (
            _pair("LITE", "2026-05-01", win=False)
            + _pair("LITE", "2026-05-02", win=False)
            + _pair("MU", "2026-05-03", win=False)
            + _pair("MU", "2026-05-04", win=False)
            + _pair("MU", "2026-05-05", win=False)
            + _pair("MU", "2026-05-06", win=False)
        )
        rep = build_repeat_loser(trades)
        assert rep["n_offenders"] == 2
        assert rep["offenders"][0]["ticker"] == "MU"
        assert rep["offenders"][0]["streak"] == 4
        assert rep["offenders"][1]["ticker"] == "LITE"

    def test_tie_break_by_deepest_loss(self):
        # both 2-loss streaks; LITE -20 vs MU -5; LITE should sort first
        trades = (
            _pair("LITE", "2026-05-01", win=False, pnl=-10.0)
            + _pair("LITE", "2026-05-02", win=False, pnl=-10.0)
            + _pair("MU", "2026-05-03", win=False, pnl=-3.0)
            + _pair("MU", "2026-05-04", win=False, pnl=-2.0)
        )
        rep = build_repeat_loser(trades)
        assert rep["offenders"][0]["ticker"] == "LITE"
        assert rep["offenders"][0]["loss_usd"] == pytest.approx(-20.0)
        assert rep["offenders"][1]["ticker"] == "MU"


class TestHeadlineWording:
    def test_single_offender_headline_names_the_ticker(self):
        trades = (
            _pair("LITE", "2026-05-01", win=False, pnl=-4.0)
            + _pair("LITE", "2026-05-02", win=False, pnl=-6.0)
        )
        rep = build_repeat_loser(trades)
        h = rep["headline"]
        assert "LITE" in h
        assert "2-loss run" in h
        # Loss usd appears in headline
        assert "-10.00" in h or "-10.0" in h

    def test_multi_offender_headline_mentions_others(self):
        trades = (
            _pair("LITE", "2026-05-01", win=False)
            + _pair("LITE", "2026-05-02", win=False)
            + _pair("LITE", "2026-05-03", win=False)
            + _pair("MU", "2026-05-04", win=False)
            + _pair("MU", "2026-05-05", win=False)
        )
        rep = build_repeat_loser(trades)
        h = rep["headline"]
        assert "LITE" in h  # top offender
        assert "MU×2" in h  # secondary offender named in others clause

    def test_ok_headline_when_no_offenders(self):
        rep = build_repeat_loser(_pair("NVDA", "2026-05-01", win=True))
        h = rep["headline"]
        assert "OK" in h
        assert "no ticker" in h


class TestRobustness:
    def test_ticker_normalized_to_uppercase(self):
        # The store rows are typically uppercase but defensively normalize
        trades = (
            _pair("nvda", "2026-05-01", win=False)
            + _pair("nvda", "2026-05-02", win=False)
        )
        rep = build_repeat_loser(trades)
        assert "NVDA" in rep["per_ticker"]
        assert rep["offenders"][0]["ticker"] == "NVDA"

    def test_missing_pnl_does_not_crash(self):
        # A round-trip with NaN-ish or missing fields → degrade silently;
        # build_round_trips computes pnl from value, so we need to drop
        # the value field on a trade to simulate corruption. The builder
        # falls back to 0.0 cost → 0.0 pnl_pct=None; treat as a flat trip.
        bad = [
            {
                "ticker": "X", "action": "BUY", "qty": 1.0,
                "price": 100.0, "value": None,  # corrupt
                "timestamp": "2026-05-01T09:00:00+00:00",
                "option_type": None, "strike": None, "expiry": None, "id": None,
            },
            {
                "ticker": "X", "action": "SELL", "qty": 1.0,
                "price": 100.0, "value": None,
                "timestamp": "2026-05-01T15:00:00+00:00",
                "option_type": None, "strike": None, "expiry": None, "id": None,
            },
        ]
        rep = build_repeat_loser(bad)
        # Did not raise; state may be OK or NO_DATA depending on round_trip
        # emit on corrupt input — both are acceptable degraded outputs.
        assert rep["state"] in ("OK", "NO_DATA")

    def test_threshold_is_inclusive(self):
        # exactly REPEAT_LOSER_MIN losses must trigger
        trades = []
        for i in range(REPEAT_LOSER_MIN):
            day = f"2026-05-{i+1:02d}"
            trades += _pair("X", day, win=False)
        rep = build_repeat_loser(trades)
        assert rep["state"] == "REPEAT_LOSER"
        assert rep["offenders"][0]["streak"] == REPEAT_LOSER_MIN

    def test_n_round_trips_counts_flats_too(self):
        # n_round_trips tracks total closed trips per ticker, flats included
        trades = (
            _pair("Z", "2026-05-01", win=False, pnl=0.0)
            + _pair("Z", "2026-05-02", win=False, pnl=-1.0)
        )
        rep = build_repeat_loser(trades)
        assert rep["per_ticker"]["Z"]["n_round_trips"] == 2
        # streak is 1 (one loss; the flat doesn't extend or break)
        assert rep["per_ticker"]["Z"]["current_loss_streak"] == 1


def _streak(ticker: str, n: int, start_day: int = 1) -> list[dict]:
    """N consecutive losing pairs on ``ticker`` (May days start_day…start_day+n-1)."""
    trades: list[dict] = []
    for i in range(n):
        day = f"2026-05-{start_day + i:02d}"
        trades += _pair(ticker, day, win=False)
    return trades


class TestPromptBlock:
    """The prompt_block surface is silent on a clean book and lean on an
    offending one — the track_record advisory-only contract."""

    def test_empty_trades_block_is_none(self):
        rep = build_repeat_loser([])
        assert rep["prompt_block"] is None

    def test_ok_state_block_is_none(self):
        # A single losing trip is below REPEAT_LOSER_MIN — must be silent.
        rep = build_repeat_loser(_pair("X", "2026-05-01", win=False))
        assert rep["state"] == "OK"
        assert rep["prompt_block"] is None

    def test_at_threshold_block_surfaces_ticker_and_streak(self):
        rep = build_repeat_loser(_streak("LOSE", REPEAT_LOSER_MIN))
        block = rep["prompt_block"]
        assert block is not None
        assert "LOSE" in block
        assert f"{REPEAT_LOSER_MIN} consecutive losses" in block
        # Cumulative loss must surface in dollars so Opus does not have to
        # re-derive it from the streak length.
        assert "$" in block
        assert "complete autonomy" in block.lower()  # advisory contract

    def test_multiple_offenders_ordered_deepest_first(self):
        # AAA on a 2-loss run @ $-5 each (-$10), BBB on a 3-loss run
        # @ $-5 each (-$15). Builder orders by streak desc, then deepest
        # loss; block must mirror that order verbatim.
        trades = _streak("AAA", 2, start_day=1) + _streak("BBB", 3, start_day=10)
        rep = build_repeat_loser(trades)
        assert rep["n_offenders"] == 2
        block = rep["prompt_block"]
        assert block is not None
        i_bbb = block.find("BBB")
        i_aaa = block.find("AAA")
        assert 0 <= i_bbb < i_aaa, "deeper streak (BBB×3) must precede AAA×2"

    def test_names_filter_scopes_block_only(self):
        # AAA and ZZZ both at threshold; only AAA is in play this cycle.
        trades = (_streak("AAA", REPEAT_LOSER_MIN, start_day=1)
                  + _streak("ZZZ", REPEAT_LOSER_MIN, start_day=10))
        rep = build_repeat_loser(trades, names={"AAA"})
        assert rep["n_offenders"] == 2  # full payload untouched
        block = rep["prompt_block"]
        assert block is not None
        assert "AAA" in block
        # ZZZ must not appear in the lean block — the lean discipline.
        assert "ZZZ" not in block

    def test_names_filter_all_out_of_scope_collapses_to_silence(self):
        # Offender exists but is not in play this cycle → block is None.
        trades = _streak("OFFEND", REPEAT_LOSER_MIN)
        rep = build_repeat_loser(trades, names={"OTHER_NAME"})
        assert rep["n_offenders"] == 1
        assert rep["prompt_block"] is None

    def test_block_is_factual_not_directive(self):
        rep = build_repeat_loser(_streak("X", REPEAT_LOSER_MIN))
        block = rep["prompt_block"]
        assert block is not None
        low = block.lower()
        assert "you must" not in low
        assert "you should" not in low
        assert " avoid " not in low
        assert " do not " not in low
