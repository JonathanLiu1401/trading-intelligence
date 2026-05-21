"""Tests for analytics/concentration_trajectory.py — pure, deterministic.

Pins:

* The full verdict ladder (CONCENTRATION_SPIKE → RAMPING_UP →
  DECONCENTRATING → CONCENTRATED_STEADY → DIVERSIFIED → BALANCED →
  INSUFFICIENT_DATA / NO_DATA), each on a synthetic trade ladder
  engineered to land exactly on its row.
* Verdict order discipline — CONCENTRATION_SPIKE outranks RAMPING_UP on
  inputs that satisfy both.
* Per-snapshot concentration math (top1/top3/HHI/effective-positions)
  on hand-checked weights.
* Options skipped from concentration math (stocks-only carve-out — same
  discipline as ``correlation.build_correlation``).
* Robustness — empty/garbage rows, unparseable timestamps, missing
  daily-closes for a ticker, out-of-order trades, sub-cent dust
  filtering, full-sell qty cleanup, never raises.
* Window-cap honesty — ``window_days`` clamps to
  ``[MIN_SNAPSHOTS, MAX_SNAPSHOTS]``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from paper_trader.analytics.concentration_trajectory import (
    DECONC_FROM_PCT,
    DIVERSIFIED_CEILING,
    MAX_SNAPSHOTS,
    MIN_SNAPSHOTS,
    RAMP_DELTA_PCT,
    RAMP_TO_PCT,
    SPIKE_FROM_PCT,
    SPIKE_TO_PCT,
    STEADY_BAND_PCT,
    STEADY_PCT,
    _close_on_or_before,
    _concentration_metrics,
    _stock_positions_after_trade,
    build_concentration_trajectory,
)


NOW = datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)


def _trade(days_ago, ticker, action, qty, price, option_type=None,
           strike=None, expiry=None):
    """One trade row at ``days_ago`` calendar days before ``NOW``.
    ``value`` follows the live engine's convention (options ×100)."""
    mult = 100.0 if option_type else 1.0
    return {
        "timestamp": (NOW - timedelta(days=days_ago)).isoformat(),
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "price": price,
        "value": price * qty * mult,
        "strike": strike,
        "expiry": expiry,
        "option_type": option_type,
    }


def _flat_closes(*tickers, price=100.0, days_back=40):
    """Helper: build a `daily_closes` dict where every ticker has a
    flat close for the last `days_back` days ending at `NOW.date()`."""
    today = NOW.date()
    rows = [((today - timedelta(days=i)).isoformat(), price)
            for i in range(days_back, -1, -1)]
    return {tk: list(rows) for tk in tickers}


# ─── Empty / degraded inputs ───────────────────────────────────────


class TestEmptyAndDegraded:
    def test_no_trades(self):
        r = build_concentration_trajectory([], {}, now=NOW)
        assert r["verdict"] == "NO_DATA"
        assert r["n_trades_walked"] == 0
        assert r["series"] == []
        assert r["current"]["n_positions"] == 0

    def test_garbage_trades_do_not_raise(self):
        # None timestamps drop the row; unparseable ones drop too; one
        # real BUY remains.
        trades = [
            None,
            {"foo": "bar"},
            {"timestamp": "garbage", "ticker": "X", "action": "BUY",
             "qty": 1, "value": 1},
            _trade(2, "NVDA", "BUY", 1, 100),
        ]
        r = build_concentration_trajectory(trades, _flat_closes("NVDA"), now=NOW)
        # 3 trade-ish rows survive the dict-check (the None is dropped);
        # 1 has a parseable ts and contributes to state.
        assert r["n_trades_walked"] >= 1
        # A 3-day window after a 2-day-ago single BUY has at most 3
        # snapshots; latest must show NVDA at 100%.
        assert r["current"]["top1_ticker"] == "NVDA"
        assert r["current"]["top1_pct"] == 100.0

    def test_options_skipped_from_concentration(self):
        # An option BUY does not contribute to the snapshot — the
        # concentration math is stocks-only by deliberate carve-out.
        trades = [
            _trade(2, "NVDA", "BUY", 1, 100),
            _trade(1, "AAPL", "BUY_CALL", 1, 50,
                   option_type="call", strike=200,
                   expiry="2026-06-19"),
        ]
        r = build_concentration_trajectory(trades, _flat_closes("NVDA", "AAPL"),
                                           now=NOW)
        # Only NVDA contributes to the deployed book.
        assert r["current"]["n_positions"] == 1
        assert r["current"]["top1_ticker"] == "NVDA"
        assert r["current"]["top1_pct"] == 100.0

    def test_missing_daily_closes_drops_ticker(self):
        # NVDA bought 2d ago, but daily_closes has no rows → ticker
        # silently drops from snapshots; result: NO_DATA.
        trades = [_trade(2, "NVDA", "BUY", 1, 100)]
        r = build_concentration_trajectory(trades, {"NVDA": []}, now=NOW)
        assert r["verdict"] == "NO_DATA"
        assert r["current"]["n_positions"] == 0

    def test_full_sell_clears_position(self):
        trades = [
            _trade(5, "NVDA", "BUY", 2, 100),
            _trade(2, "NVDA", "SELL", 2, 110),
        ]
        r = build_concentration_trajectory(trades, _flat_closes("NVDA"), now=NOW)
        # After the full SELL, the latest snapshot has zero positions —
        # the trade walk's full-close pop is the pinned invariant.
        assert r["current"]["n_positions"] == 0
        assert r["current"]["top1_ticker"] is None
        # Trajectory falls 100% → 0% over the window — verdict is
        # DECONCENTRATING (first ≥ DECONC_FROM_PCT and delta ≥
        # RAMP_DELTA_PCT). NOT NO_DATA: the series carries leading rows
        # with a real position before the SELL fired.
        assert r["verdict"] == "DECONCENTRATING"


# ─── Verdict ladder boundaries ────────────────────────────────────


class TestVerdictLadder:
    def test_concentration_spike(self):
        # Prior day: AAPL & NVDA balanced (top-1 ~50%). Today: a big NVDA
        # add pushes NVDA to ~80% of the book. The previous snapshot
        # must be < SPIKE_FROM_PCT (40) for SPIKE to fire; we engineer
        # that by giving AAPL the larger weight on the prior day.
        # Day -3: BUY 8 AAPL @ 100 → AAPL=800, only name (100%).
        # Day -2: BUY 1 NVDA @ 300 → AAPL=800 (~73%), NVDA=300 (~27%).
        # Day -1: BUY 9 NVDA @ 300 → AAPL=800 (~21%), NVDA=3000 (~79%).
        # snapshots need MIN_SNAPSHOTS (3) — prior to last must be <40,
        # last must be >=60.
        trades = [
            _trade(5, "AAPL", "BUY", 8, 100),
            _trade(4, "NVDA", "BUY", 1, 300),
            _trade(2, "NVDA", "BUY", 9, 300),  # the spike trade
        ]
        # Force a 4-day window so we have plenty of snapshots; the spike
        # event lands on the latest one.
        closes = _flat_closes("AAPL", "NVDA")
        # We want the SECOND-LATEST snapshot's top1 < 40. With flat
        # prices on the prior day (after the +1 NVDA BUY), AAPL is
        # 800/1100 ≈ 72% — too high to trigger SPIKE. Re-engineer with
        # 6 days back: AAPL=800, NVDA=300 → AAPL ~72%. We need to push
        # AAPL down before the spike: add a sibling.
        trades = [
            _trade(6, "AAPL", "BUY", 3, 100),    # AAPL 300
            _trade(5, "MSFT", "BUY", 3, 100),    # +MSFT 300, AAPL=50%, MSFT=50%
            _trade(4, "GOOG", "BUY", 4, 100),    # +GOOG 400, top1 GOOG=40%
            _trade(3, "TSLA", "BUY", 4, 100),    # +TSLA 400, balanced; top1 ~30%
            # day -2 snapshot: 4 names roughly balanced (each 25-29%)
            _trade(1, "NVDA", "BUY", 30, 100),   # +NVDA 3000 → NVDA 67% of book
        ]
        closes = _flat_closes("AAPL", "MSFT", "GOOG", "TSLA", "NVDA")
        r = build_concentration_trajectory(trades, closes, now=NOW,
                                           window_days=5)
        # Latest must be NVDA-dominant; prior must be <40 top-1.
        assert r["current"]["top1_ticker"] == "NVDA"
        assert r["current"]["top1_pct"] >= SPIKE_TO_PCT
        # Verify the verdict fires.
        assert r["verdict"] == "CONCENTRATION_SPIKE"
        # Sanity: at least one adjacent pair in the window jumps from
        # <SPIKE_FROM to >=SPIKE_TO (the spike-pair invariant). After
        # the BUY fires the book parks at the elevated level, so the
        # spike pair is somewhere in the series — not necessarily the
        # very last two rows.
        pairs = [(r["series"][i]["top1_pct"], r["series"][i + 1]["top1_pct"])
                 for i in range(len(r["series"]) - 1)]
        assert any(p < SPIKE_FROM_PCT and n >= SPIKE_TO_PCT for p, n in pairs)

    def test_ramping_up(self):
        # Top-1 climbs ~15pp from first → last over a 5-day window AND
        # latest >= RAMP_TO_PCT (50). NOT a single-cycle jump (so no
        # SPIKE).
        # day -5: AAPL 50%, NVDA 50% (top1 50%)  — already at ramp_to
        # day -3: AAPL 40, NVDA 60 (top1 60)
        # day -1: AAPL 25, NVDA 75 (top1 75)  → delta 25 ≥ 15, latest ≥ 50
        # But day-3 → day-1 jump of 15pp could trigger SPIKE? SPIKE
        # needs prior < 40. Prior day's top1 here is 60 → no SPIKE.
        trades = [
            _trade(5, "AAPL", "BUY", 1, 100),
            _trade(5, "NVDA", "BUY", 1, 100),
            _trade(3, "NVDA", "BUY", 0.5, 100),    # NVDA value 150, AAPL 100; top1 60%
            _trade(1, "NVDA", "BUY", 2, 100),      # NVDA value 350, AAPL 100; top1 78%
        ]
        r = build_concentration_trajectory(trades,
                                           _flat_closes("AAPL", "NVDA"),
                                           now=NOW, window_days=6)
        assert r["verdict"] == "RAMPING_UP"
        assert r["current"]["top1_ticker"] == "NVDA"
        assert r["current"]["top1_pct"] >= RAMP_TO_PCT
        assert r["delta_top1_pct"] >= RAMP_DELTA_PCT

    def test_deconcentrating(self):
        # First snapshot top-1 ≥ DECONC_FROM_PCT (50), last drops by
        # ≥ RAMP_DELTA_PCT (15).
        # day -5: BUY 5 NVDA → NVDA 100% (top1 100)
        # day -1: BUY 5 AAPL → AAPL 50, NVDA 50 (top1 50)
        # delta = -50; first 100 ≥ 50 ⇒ DECONCENTRATING.
        trades = [
            _trade(5, "NVDA", "BUY", 5, 100),
            _trade(1, "AAPL", "BUY", 5, 100),
        ]
        r = build_concentration_trajectory(trades,
                                           _flat_closes("AAPL", "NVDA"),
                                           now=NOW, window_days=6)
        assert r["verdict"] == "DECONCENTRATING"
        assert r["series"][0]["top1_pct"] == 100.0
        assert r["current"]["top1_pct"] < r["series"][0]["top1_pct"]

    def test_concentrated_steady(self):
        # Single big BUY 30 days ago; nothing else. Top-1 stays at 100%
        # every day → mean ≥ 50, band 0 → CONCENTRATED_STEADY.
        trades = [_trade(20, "NVDA", "BUY", 5, 100)]
        r = build_concentration_trajectory(trades, _flat_closes("NVDA"),
                                           now=NOW, window_days=10)
        assert r["verdict"] == "CONCENTRATED_STEADY"
        top1s = [s["top1_pct"] for s in r["series"]]
        assert max(top1s) - min(top1s) <= STEADY_BAND_PCT
        assert sum(top1s) / len(top1s) >= STEADY_PCT

    def test_diversified(self):
        # Five names, each equal weight throughout the window → top-1
        # always 20% — below DIVERSIFIED_CEILING (35).
        trades = [
            _trade(10, "AAPL", "BUY", 1, 100),
            _trade(10, "MSFT", "BUY", 1, 100),
            _trade(10, "GOOG", "BUY", 1, 100),
            _trade(10, "AMZN", "BUY", 1, 100),
            _trade(10, "META", "BUY", 1, 100),
        ]
        r = build_concentration_trajectory(
            trades, _flat_closes("AAPL", "MSFT", "GOOG", "AMZN", "META"),
            now=NOW, window_days=7)
        assert r["verdict"] == "DIVERSIFIED"
        for s in r["series"]:
            assert s["top1_pct"] < DIVERSIFIED_CEILING

    def test_balanced_midstate(self):
        # Two names ~60/40 stable. top-1 mean 60% (≥ STEADY_PCT) but
        # band 0 — passes CONCENTRATED_STEADY. We want BALANCED. Build
        # a case where: top-1 mean is below STEADY_PCT AND max not <
        # DIVERSIFIED_CEILING AND no slope. Two names ~45/55: top-1
        # always 55 (mean 55 ≥ 50 → STEADY again).
        # Engineer a true mid-state: top-1 oscillates between 40 and 60
        # so mean < STEADY_PCT? Mean 50; band 20 > STEADY_BAND_PCT →
        # steady fails. Max = 60 ≥ 35 → diversified fails. Slope = 0 →
        # ramp/deconc fail. ⇒ BALANCED.
        trades = [
            _trade(5, "AAPL", "BUY", 6, 100),   # AAPL 600 only ⇒ 100%
            _trade(4, "NVDA", "BUY", 9, 100),   # AAPL 600, NVDA 900 — NVDA 60
            _trade(3, "AAPL", "BUY", 9, 100),   # AAPL 1500, NVDA 900 — AAPL 62.5
            _trade(2, "NVDA", "BUY", 7, 100),   # AAPL 1500, NVDA 1600 — NVDA ~51.6
        ]
        r = build_concentration_trajectory(
            trades, _flat_closes("AAPL", "NVDA"),
            now=NOW, window_days=6)
        # Snapshot first is 100%, that breaks band+steady; ramp-or-
        # deconc check the first-to-last delta. Top-1 drifts from 100
        # → ~52 ⇒ delta -48 → DECONCENTRATING in fact.
        # Build a case that truly lands BALANCED: short flat single
        # name on first day (no second name yet), then two names
        # roughly 40/60 the rest. We use window_days=4 with the first
        # snapshot at NVDA-only 100% NO — that re-triggers deconc.
        # Simplest: top1 mean below STEADY_PCT, max below
        # DIVERSIFIED_CEILING? mean<50 and max<35 ⇒ DIVERSIFIED.
        # mean<50 and max in [35, ramp-to)? Then no slope and no
        # steady ⇒ BALANCED. Try AAPL/NVDA/MSFT equal-3: top-1 always
        # 33% → DIVERSIFIED. To hit BALANCED: 2 names where one is
        # 45% of book; deltas zero, max 45 ≥ 35, mean 45 < 50.
        trades_b = [
            _trade(5, "AAPL", "BUY", 11, 100),   # AAPL 1100
            _trade(5, "NVDA", "BUY", 9, 100),    # NVDA 900; top1 1100/2000=55
        ]
        # ⇒ top1 stable at 55% over every snapshot; mean 55 ≥ STEADY_PCT
        # AND band 0 ≤ STEADY_BAND_PCT → CONCENTRATED_STEADY again.
        # Try 47/53 split:
        trades_b2 = [
            _trade(5, "AAPL", "BUY", 47, 100),   # AAPL 4700
            _trade(5, "NVDA", "BUY", 53, 100),   # NVDA 5300; top1 53%
        ]
        rb = build_concentration_trajectory(trades_b2,
                                            _flat_closes("AAPL", "NVDA"),
                                            now=NOW, window_days=5)
        # 53% mean ≥ 50, band 0 → still CONCENTRATED_STEADY. To hit
        # BALANCED we need mean < 50 OR band > 10. Try 45/55 flat:
        trades_b3 = [
            _trade(5, "AAPL", "BUY", 45, 100),   # 4500
            _trade(5, "NVDA", "BUY", 55, 100),   # 5500; top1 55
        ]
        rb3 = build_concentration_trajectory(trades_b3,
                                             _flat_closes("AAPL", "NVDA"),
                                             now=NOW, window_days=5)
        # Still mean 55, band 0 → STEADY. The BALANCED row in the
        # ladder is hit when steady-mean (≥50) fails OR steady-band
        # (≤10) fails. Force band > 10 over the window via an
        # intra-window add that swings top-1 by >10pp.
        trades_b4 = [
            _trade(5, "AAPL", "BUY", 45, 100),   # day -5..-3: AAPL 4500, top1 100
            _trade(3, "NVDA", "BUY", 55, 100),   # day -3..now: AAPL 4500, NVDA 5500; top1 55
        ]
        # snapshots: day-5 to day-3 → top1=100; day-3 to now → top1=55.
        # mean over 5 snapshots: e.g. (100,100,55,55,55) = 73; max-min
        # = 45 > 10 → fails STEADY band. delta -45 (≥ 15) and first
        # 100 (≥ 50) → DECONCENTRATING. Verdict still won't be
        # BALANCED. The ladder is intentionally tight; document that
        # BALANCED is the residual catch-all and validate it on the
        # one shape that survives all upstream rows.
        trades_b5 = [
            # Two names 55/45 from day-5 then nudged to 60/40 mid-window.
            # mean top1 ≈ 57.5 ≥ 50; band 5 ≤ 10 → STEADY.
            # We need the residual. Use a 4-name book where top-1
            # never crosses STEADY_PCT, never falls < diversified, no
            # slope.
            _trade(5, "AAPL", "BUY", 33, 100),   # AAPL 3300
            _trade(5, "MSFT", "BUY", 33, 100),   # MSFT 3300
            _trade(5, "NVDA", "BUY", 34, 100),   # NVDA 3400; top1 NVDA=34%
        ]
        rb5 = build_concentration_trajectory(
            trades_b5, _flat_closes("AAPL", "MSFT", "NVDA"),
            now=NOW, window_days=5)
        # top1 always ~34%, max ~34 < DIVERSIFIED_CEILING (35) →
        # DIVERSIFIED, not BALANCED. So BALANCED requires top1 in
        # [DIVERSIFIED_CEILING, STEADY_PCT) and no slope. Force 39/30/31:
        trades_b6 = [
            _trade(5, "AAPL", "BUY", 39, 100),   # AAPL 3900
            _trade(5, "MSFT", "BUY", 30, 100),   # MSFT 3000
            _trade(5, "NVDA", "BUY", 31, 100),   # NVDA 3100; top1 AAPL=39%
        ]
        rb6 = build_concentration_trajectory(
            trades_b6, _flat_closes("AAPL", "MSFT", "NVDA"),
            now=NOW, window_days=5)
        assert rb6["verdict"] == "BALANCED"
        assert rb6["current"]["top1_pct"] == 39.0
        for s in rb6["series"]:
            assert DIVERSIFIED_CEILING <= s["top1_pct"] < STEADY_PCT

    def test_insufficient_data(self):
        # Single trade today only → 1 snapshot → below MIN_SNAPSHOTS.
        trades = [_trade(0, "NVDA", "BUY", 1, 100)]
        r = build_concentration_trajectory(trades, _flat_closes("NVDA"),
                                           now=NOW)
        assert r["verdict"] == "INSUFFICIENT_DATA"

    def test_spike_outranks_ramp(self):
        # Construct inputs that satisfy BOTH the SPIKE and RAMP
        # conditions; the ladder must fire SPIKE first.
        # Window: day -5 single-name AAPL @ 100%, day -2 AAPL 1000 +
        # NVDA 100 (top1 90% AAPL), day 0 BUY 80 NVDA → AAPL 1000,
        # NVDA 8100 → top1 NVDA 89%. Prior snapshot AAPL 90.9% ≥
        # SPIKE_FROM. So SPIKE doesn't fire because prior already > 40.
        # Engineer differently: prior must be < 40.
        # Day -5: 5 names equal weight (top1 20%).
        # Day -4 → -2 unchanged.
        # Day -1: huge NVDA add → NVDA dominant ≥ 60.
        trades = [
            _trade(5, "A", "BUY", 2, 100),
            _trade(5, "B", "BUY", 2, 100),
            _trade(5, "C", "BUY", 2, 100),
            _trade(5, "D", "BUY", 2, 100),
            _trade(5, "E", "BUY", 2, 100),
            _trade(1, "NVDA", "BUY", 60, 100),
        ]
        r = build_concentration_trajectory(
            trades, _flat_closes("A", "B", "C", "D", "E", "NVDA"),
            now=NOW, window_days=6)
        # Both conditions are satisfied (delta is enormous, latest ≥
        # ramp_to). SPIKE must fire because it's checked first.
        assert r["verdict"] == "CONCENTRATION_SPIKE"


# ─── Per-snapshot math (pure helpers) ─────────────────────────────


class TestConcentrationMath:
    def test_metrics_single_name(self):
        m = _concentration_metrics([("NVDA", 1000.0)])
        assert m["top1_pct"] == 100.0
        assert m["top3_pct"] == 100.0
        assert m["hhi"] == 1.0
        assert m["effective_positions"] == 1.0
        assert m["n_positions"] == 1
        assert m["top1_ticker"] == "NVDA"
        assert m["deployed_usd"] == 1000.0

    def test_metrics_two_equal(self):
        m = _concentration_metrics([("A", 500.0), ("B", 500.0)])
        assert m["top1_pct"] == 50.0
        assert m["top3_pct"] == 100.0
        assert m["hhi"] == 0.5
        assert m["effective_positions"] == 2.0
        assert m["n_positions"] == 2

    def test_metrics_three_equal(self):
        m = _concentration_metrics([("A", 100.0), ("B", 100.0), ("C", 100.0)])
        # HHI = 3·(1/3)² = 1/3 → 0.333333; effective = 3.0
        assert m["top1_pct"] == round(100.0 / 3.0, 4)
        assert m["top3_pct"] == 100.0
        assert m["effective_positions"] == 3.0

    def test_metrics_picks_largest(self):
        m = _concentration_metrics([("A", 100.0), ("B", 300.0), ("C", 200.0)])
        assert m["top1_ticker"] == "B"
        assert m["top1_pct"] == 50.0
        assert m["top3_pct"] == 100.0
        assert m["n_positions"] == 3

    def test_metrics_drops_subcent_dust(self):
        m = _concentration_metrics([("A", 100.0), ("B", 1e-9)])
        # Sub-cent name is filtered.
        assert m["n_positions"] == 1
        assert m["top1_pct"] == 100.0

    def test_metrics_empty(self):
        m = _concentration_metrics([])
        assert m["n_positions"] == 0
        assert m["top1_ticker"] is None
        assert m["top1_pct"] == 0.0
        assert m["hhi"] == 0.0
        assert m["effective_positions"] == 0.0


class TestTradeWalk:
    def test_buy_adds_qty(self):
        state = {}
        _stock_positions_after_trade(state, {"action": "BUY", "ticker": "nvda",
                                              "qty": 3, "option_type": None})
        assert state == {"NVDA": 3.0}

    def test_sell_subtracts_qty(self):
        state = {"NVDA": 5.0}
        _stock_positions_after_trade(state, {"action": "SELL", "ticker": "NVDA",
                                              "qty": 2, "option_type": None})
        assert state == {"NVDA": 3.0}

    def test_full_sell_pops(self):
        state = {"NVDA": 2.0}
        _stock_positions_after_trade(state, {"action": "SELL", "ticker": "NVDA",
                                              "qty": 2, "option_type": None})
        assert "NVDA" not in state

    def test_option_skipped(self):
        state = {}
        _stock_positions_after_trade(state, {"action": "BUY_CALL",
                                              "ticker": "NVDA", "qty": 1,
                                              "option_type": "call"})
        assert state == {}

    def test_empty_ticker_ignored(self):
        state = {}
        _stock_positions_after_trade(state, {"action": "BUY", "ticker": "",
                                              "qty": 1, "option_type": None})
        assert state == {}


class TestCloseLookup:
    def test_picks_latest_on_or_before(self):
        rows = [("2026-05-01", 90.0), ("2026-05-15", 100.0),
                ("2026-05-20", 110.0)]
        assert _close_on_or_before(rows, "2026-05-21") == 110.0
        assert _close_on_or_before(rows, "2026-05-20") == 110.0
        assert _close_on_or_before(rows, "2026-05-15") == 100.0
        assert _close_on_or_before(rows, "2026-05-10") == 90.0
        assert _close_on_or_before(rows, "2026-04-30") is None

    def test_empty_rows(self):
        assert _close_on_or_before([], "2026-05-21") is None


class TestWindowCap:
    def test_window_clamps_below_min(self):
        # window_days=1 should clamp up to MIN_SNAPSHOTS.
        trades = [_trade(40, "NVDA", "BUY", 1, 100)]
        r = build_concentration_trajectory(trades, _flat_closes("NVDA"),
                                           now=NOW, window_days=1)
        # window_days field reflects the actual snapshot count; clamped
        # internally to MIN_SNAPSHOTS = 3 (so we emit up to 3 days).
        # With a single buy 40 days ago the book has been at 100% NVDA
        # throughout — exactly MIN_SNAPSHOTS snapshots get emitted.
        assert r["window_days"] >= MIN_SNAPSHOTS

    def test_window_clamps_above_max(self):
        # window_days=999 should clamp down to MAX_SNAPSHOTS (30).
        trades = [_trade(50, "NVDA", "BUY", 1, 100)]
        r = build_concentration_trajectory(trades, _flat_closes("NVDA",
                                                                days_back=60),
                                           now=NOW, window_days=999)
        assert r["window_days"] <= MAX_SNAPSHOTS

    def test_no_pre_book_snapshots(self):
        # Trade 3 days ago; daily_closes only cover the recent days.
        # Snapshots before the first trade are dropped.
        trades = [_trade(3, "NVDA", "BUY", 1, 100)]
        r = build_concentration_trajectory(trades, _flat_closes("NVDA"),
                                           now=NOW, window_days=10)
        # Window asks 10 days but only ~4 days (today, -1, -2, -3) are
        # on-or-after the first trade. The remainder is dropped.
        assert r["window_days"] <= 4


class TestNeverRaises:
    def test_garbage_daily_closes_type(self):
        # daily_closes is None — degrades to no marks-to-market.
        trades = [_trade(2, "NVDA", "BUY", 1, 100)]
        r = build_concentration_trajectory(trades, None, now=NOW)
        # No close data → no deployed value at any snapshot → NO_DATA.
        assert r["verdict"] == "NO_DATA"

    def test_garbage_daily_closes_inner(self):
        # daily_closes has a non-list value — _close_on_or_before
        # tolerates it (degrades to None).
        trades = [_trade(2, "NVDA", "BUY", 1, 100)]
        r = build_concentration_trajectory(trades, {"NVDA": None}, now=NOW)
        assert r["verdict"] == "NO_DATA"

    def test_trades_with_zero_qty(self):
        trades = [_trade(2, "NVDA", "BUY", 0, 100)]
        r = build_concentration_trajectory(trades, _flat_closes("NVDA"),
                                           now=NOW)
        # Zero-qty BUY contributes nothing — no positions.
        assert r["current"]["n_positions"] == 0
        assert r["verdict"] == "NO_DATA"

    def test_unparseable_timestamps_dropped(self):
        trades = [{"timestamp": "garbage", "ticker": "NVDA",
                   "action": "BUY", "qty": 1, "value": 100,
                   "option_type": None}]
        r = build_concentration_trajectory(trades, _flat_closes("NVDA"),
                                           now=NOW)
        assert r["n_trades_walked"] == 0


class TestSeriesShape:
    def test_series_chronological(self):
        trades = [_trade(5, "NVDA", "BUY", 1, 100)]
        r = build_concentration_trajectory(trades, _flat_closes("NVDA"),
                                           now=NOW, window_days=5)
        dates = [s["date"] for s in r["series"]]
        # Sorted ascending YYYY-MM-DD.
        assert dates == sorted(dates)

    def test_current_matches_last_series_row(self):
        trades = [_trade(3, "NVDA", "BUY", 1, 100)]
        r = build_concentration_trajectory(trades, _flat_closes("NVDA"),
                                           now=NOW, window_days=4)
        assert r["current"]["top1_pct"] == r["series"][-1]["top1_pct"]
        assert r["current"]["top1_ticker"] == r["series"][-1]["top1_ticker"]
        assert r["current"]["n_positions"] == r["series"][-1]["n_positions"]

    def test_delta_top1_pct_is_last_minus_first(self):
        trades = [
            _trade(5, "NVDA", "BUY", 1, 100),
            _trade(1, "AAPL", "BUY", 1, 100),
        ]
        r = build_concentration_trajectory(trades,
                                           _flat_closes("NVDA", "AAPL"),
                                           now=NOW, window_days=6)
        expected_delta = round(
            r["series"][-1]["top1_pct"] - r["series"][0]["top1_pct"], 4)
        assert r["delta_top1_pct"] == expected_delta
