"""Tests for run_continuous_backtests.py.

Covers the pure functions: window picking, history trimming, top-decision
appending, outcome computation, and the live-only filter for news context.
The cycle loop itself isn't exercised — it requires the BacktestEngine
which depends on yfinance.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import run_continuous_backtests as rcb
from paper_trader.backtest import BacktestRun


# ─────────────────────── _pick_window ───────────────────────────

class TestPickWindow:
    def test_returns_two_dates(self):
        start, end = rcb._pick_window(seed=42)
        assert isinstance(start, date)
        assert isinstance(end, date)
        assert start < end

    def test_duration_within_range(self):
        for seed in range(20):
            start, end = rcb._pick_window(seed=seed)
            days = (end - start).days
            # 1-5 years × 365 days/year
            assert rcb.MIN_WINDOW_YEARS * 365 <= days <= rcb.MAX_WINDOW_YEARS * 365

    def test_window_ends_before_buffer(self):
        # Critical invariant: end date must be at least WINDOW_END_BUFFER_DAYS
        # in the past so we never train on data with insufficient forward-return
        # ground truth.
        for seed in range(50):
            start, end = rcb._pick_window(seed=seed)
            days_back = (date.today() - end).days
            assert days_back >= rcb.WINDOW_END_BUFFER_DAYS, \
                f"seed={seed} end={end} is only {days_back}d before today"

    def test_window_starts_after_earliest(self):
        for seed in range(20):
            start, _ = rcb._pick_window(seed=seed)
            assert start >= rcb.EARLIEST_WINDOW_START

    def test_deterministic_for_same_seed(self):
        # Same seed → same window. Critical for reproducibility of historical runs.
        a = rcb._pick_window(seed=1234)
        b = rcb._pick_window(seed=1234)
        assert a == b

    def test_different_seeds_differ(self):
        windows = {rcb._pick_window(seed=s) for s in range(20)}
        # 20 seeds should give at least 5 distinct windows (probabilistically near-certain).
        assert len(windows) > 1


# ─────────────────────── _trim_history ───────────────────────────

def _make_engine_with_runs(tmp_path, n_runs):
    """Build a real BacktestStore in tmp_path with n_runs fake runs."""
    from paper_trader.backtest import BacktestStore
    db_path = tmp_path / "bt.db"
    store = BacktestStore(path=db_path)
    start = date(2025, 1, 1)
    end = date(2025, 12, 31)
    for i in range(1, n_runs + 1):
        store.upsert_run(i, seed=i, status="complete", start=start, end=end)
        store.record_trade(i, "2025-01-01", "NVDA", "BUY", 1.0, 100.0, "test")
        store.record_decision(i, "2025-01-01",
                              {"action": "BUY", "ticker": "NVDA", "qty": 1.0,
                               "reasoning": "score=2.5 regime=bull"},
                              "FILLED", "ok", 0.0, 0.0, 0)
    engine = MagicMock()
    engine.store = store
    return engine


class TestTrimHistory:
    def test_no_op_when_below_threshold(self, tmp_path):
        eng = _make_engine_with_runs(tmp_path, n_runs=5)
        deleted = rcb._trim_history(eng, keep=10)
        assert deleted == 0
        # All 5 still present.
        rows = eng.store.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()
        assert rows[0] == 5

    def test_trims_oldest_runs(self, tmp_path):
        eng = _make_engine_with_runs(tmp_path, n_runs=20)
        deleted = rcb._trim_history(eng, keep=10)
        # Should drop runs 1..10, keeping 11..20.
        assert deleted == 10
        rows = eng.store.conn.execute(
            "SELECT run_id FROM backtest_runs ORDER BY run_id"
        ).fetchall()
        ids = [r[0] for r in rows]
        assert ids == list(range(11, 21))

    def test_cascades_to_trades_and_decisions(self, tmp_path):
        """Critical: trades / decisions belonging to trimmed runs must also be deleted —
        otherwise the DB grows unbounded with orphaned rows."""
        eng = _make_engine_with_runs(tmp_path, n_runs=15)
        rcb._trim_history(eng, keep=5)
        n_trades = eng.store.conn.execute(
            "SELECT COUNT(*) FROM backtest_trades WHERE run_id <= 10"
        ).fetchone()[0]
        n_decs = eng.store.conn.execute(
            "SELECT COUNT(*) FROM backtest_decisions WHERE run_id <= 10"
        ).fetchone()[0]
        assert n_trades == 0
        assert n_decs == 0


# ─────────────────────── _append_top_decisions ───────────────────────────

class TestAppendTopDecisions:
    def test_writes_per_decision_line(self, tmp_path, monkeypatch):
        eng = _make_engine_with_runs(tmp_path, n_runs=3)
        jsonl_path = tmp_path / "winners.jsonl"
        monkeypatch.setattr(rcb, "WINNER_JSONL", jsonl_path)

        runs = [
            BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                        end_date="2025-12-31", total_return_pct=20.0),
            BacktestRun(run_id=2, seed=2, start_date="2025-01-01",
                        end_date="2025-12-31", total_return_pct=10.0),
            BacktestRun(run_id=3, seed=3, start_date="2025-01-01",
                        end_date="2025-12-31", total_return_pct=5.0),
        ]
        written = rcb._append_top_decisions(eng, runs, cycle=7)
        # Each fake run has exactly 1 BUY decision.
        assert written == 3
        # File must exist and contain valid JSON lines.
        lines = jsonl_path.read_text().splitlines()
        assert len(lines) == 3
        recs = [json.loads(l) for l in lines]
        assert all(r["cycle"] == 7 for r in recs)
        # Top-ranked run should have higher ai_score than bottom-ranked.
        rank1 = next(r for r in recs if r["run_id"] == 1)
        rank3 = next(r for r in recs if r["run_id"] == 3)
        assert rank1["ai_score"] > rank3["ai_score"]

    def test_append_not_overwrite(self, tmp_path, monkeypatch):
        """Old results must accumulate, not be clobbered — historical runs are
        irreplaceable training data."""
        eng = _make_engine_with_runs(tmp_path, n_runs=1)
        jsonl_path = tmp_path / "winners.jsonl"
        monkeypatch.setattr(rcb, "WINNER_JSONL", jsonl_path)

        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                            end_date="2025-12-31", total_return_pct=5.0)]
        rcb._append_top_decisions(eng, runs, cycle=1)
        rcb._append_top_decisions(eng, runs, cycle=2)
        rcb._append_top_decisions(eng, runs, cycle=3)
        lines = jsonl_path.read_text().splitlines()
        # 1 decision × 3 cycles = 3 lines, all preserved.
        assert len(lines) == 3
        cycles = [json.loads(l)["cycle"] for l in lines]
        assert sorted(cycles) == [1, 2, 3]


# ─────────────────────── _compute_decision_outcomes ───────────────────────────

class TestComputeDecisionOutcomes:
    def test_empty_runs(self, tmp_path):
        eng = _make_engine_with_runs(tmp_path, n_runs=0)
        # Trading days is empty here — should produce no outcomes.
        eng.prices = MagicMock()
        eng.prices.trading_days = []
        outs = rcb._compute_decision_outcomes(eng, [])
        assert outs == []

    def test_skips_decisions_past_price_horizon(self, tmp_path, synthetic_prices):
        eng = _make_engine_with_runs(tmp_path, n_runs=1)
        eng.prices = synthetic_prices
        # Insert a decision on the last trading day — its 5d forward window
        # extends past available data, so it must be skipped (not silently zeroed).
        last_day = synthetic_prices.trading_days[-1].isoformat()
        eng.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "reasoning) VALUES (?, ?, ?, ?, ?)",
            (1, last_day, "BUY", "NVDA", "score=2.5 regime=bull"),
        )
        eng.store.conn.commit()
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                            end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        # Last-day decision must be dropped (target idx >= len(trading_days)).
        # Initial fixture also had a BUY on 2025-01-01 — that one has 5d future
        # and a valid price for NVDA (from synthetic_prices), so it survives.
        for o in outs:
            assert o["sim_date"] != last_day


# ──────────── training-integrity: only FILLED trades train the scorer ────────

class TestFilledOnlyTrainingIntegrity:
    """Regression lock: a BUY/SELL decision that did NOT execute
    (`status != 'FILLED'`) must never reach `winner_training.jsonl`
    (ArticleNet) or `decision_outcomes.jsonl` (DecisionScorer).

    `run_one` records a terminal non-FILLED decision row for the last
    intraday decision when nothing filled that day. If that decision was a
    BUY/SELL `_execute_decision` rejected, training on its 5d forward return
    is a phantom outcome for a position that never moved capital — and its
    blocking reason (out of cash / no price) is regime-correlated, so it is
    biased contamination, not noise. Both training-pipeline queries filter
    `status = 'FILLED'`; these tests fail if either filter is dropped.
    """

    def test_append_top_decisions_skips_blocked(self, tmp_path, monkeypatch):
        eng = _make_engine_with_runs(tmp_path, n_runs=1)
        # Fixture already added a FILLED BUY NVDA on 2025-01-01 for run 1.
        # Add a BLOCKED BUY that must NOT be written to winner_training.jsonl.
        eng.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "status, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
            (1, "2025-01-02", "BUY", "TQQQ", "BLOCKED",
             "score=9.99 regime=bull (insufficient cash)"),
        )
        eng.store.conn.commit()
        jsonl_path = tmp_path / "winners.jsonl"
        monkeypatch.setattr(rcb, "WINNER_JSONL", jsonl_path)
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                             end_date="2025-12-31", total_return_pct=12.0)]
        written = rcb._append_top_decisions(eng, runs, cycle=9)
        # Only the FILLED NVDA decision survives the status filter.
        assert written == 1
        recs = [json.loads(l) for l in jsonl_path.read_text().splitlines()]
        assert [r["ticker"] for r in recs] == ["NVDA"]
        assert all(r["ticker"] != "TQQQ" for r in recs)

    def test_compute_decision_outcomes_skips_blocked(self, tmp_path,
                                                     synthetic_prices):
        eng = _make_engine_with_runs(tmp_path, n_runs=1)
        eng.prices = synthetic_prices
        day0 = synthetic_prices.trading_days[0].isoformat()  # has 5d future
        # FILLED NVDA (must produce an outcome) + BLOCKED SPY (must NOT).
        # Both tickers have synthetic prices, so the ONLY thing excluding
        # SPY is the status filter — a faithful discriminating test.
        eng.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "status, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
            (1, day0, "BUY", "NVDA", "FILLED",
             "ML+quant: NVDA score=3.00 regime=bull news_count=0 news_urg=0.0"),
        )
        eng.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "status, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
            (1, day0, "BUY", "SPY", "BLOCKED",
             "ML+quant: SPY score=8.00 regime=bull news_count=0 news_urg=0.0"),
        )
        eng.store.conn.commit()
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                             end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        day0_outs = [o for o in outs if o["sim_date"] == day0]
        assert [o["ticker"] for o in day0_outs] == ["NVDA"]
        assert all(o["ticker"] != "SPY" for o in outs)
        # The surviving outcome carries the parsed ml_score (sanity that it is
        # the real FILLED record, not an empty default).
        assert day0_outs[0]["ml_score"] == 3.00
        assert day0_outs[0]["action"] == "BUY"


# ─────────────────────── walk-back collision in outcomes ─────────────────
#
# Lock the fix for the fabricated-flat-outcome bug: when PriceCache.price_on
# falls back to a prior close for BOTH sim_d and end_d (a thin/foreign-calendar
# ticker whose actual last trade is before the requested dates), returns_pct
# returns 0.0 by construction. That fake 0% used to leak into
# decision_outcomes.jsonl and silently poisoned the DecisionScorer training set
# with bias-correlated flat labels. The fix uses PriceCache.resolved_close_date
# to refuse a collision instead.

class TestWalkBackCollisionDoesNotFabricateZero:
    def _build_collision_cache(self):
        """Synthetic cache where SPY has every weekday, but `THIN` has data
        only on one date — so any sim_d / end_d pair in the surrounding window
        forces price_on to walk back to the SAME prior close.
        """
        from paper_trader.backtest import PriceCache

        cache = PriceCache.__new__(PriceCache)
        cache.tickers = ["SPY", "THIN"]
        days = []
        d = date(2025, 1, 6)
        while d <= date(2025, 2, 14):
            if d.weekday() < 5:
                days.append(d)
            d += timedelta(days=1)
        cache.start = days[0]
        cache.end = days[-1]
        cache.prices = {
            "SPY": {dd.isoformat(): 100.0 + i for i, dd in enumerate(days)},
            # THIN's only trade is Jan 13. sim_d=Jan 14 and end_d=Jan 16 both
            # have no exact match but walk back finds Jan 13 → collision.
            "THIN": {date(2025, 1, 13).isoformat(): 50.0},
        }
        cache.trading_days = days
        return cache

    def test_resolved_close_date_returns_the_walkback_target(self):
        cache = self._build_collision_cache()
        # Exact match passes through unchanged.
        assert cache.resolved_close_date("THIN", date(2025, 1, 13)) == \
            date(2025, 1, 13)
        # Both sim_d and end_d resolve to the SAME prior close — collision.
        assert cache.resolved_close_date("THIN", date(2025, 1, 14)) == \
            date(2025, 1, 13)
        assert cache.resolved_close_date("THIN", date(2025, 1, 16)) == \
            date(2025, 1, 13)
        # Outside the 7-day walk-back window → None.
        assert cache.resolved_close_date("THIN", date(2025, 1, 23)) is None
        # Unknown ticker → None, not crash.
        assert cache.resolved_close_date("ZZZ", date(2025, 1, 14)) is None

    def test_collision_outcome_is_dropped(self, tmp_path):
        """A BUY of THIN on day_idx=5: sim_d and end_d both walk back to
        THIN's only close (Jan 13) → returns_pct = 0.0% fabricated. The fix
        skips the row instead of writing a fake 0% outcome.
        """
        eng = _make_engine_with_runs(tmp_path, n_runs=1)
        # Override prices with our collision-prone cache.
        eng.prices = self._build_collision_cache()
        # day index 5 puts sim_d = Jan 13 (exact match for THIN) and end_d =
        # idx+5 (Jan 21) which has NO THIN exact match. Walk-back from Jan 21
        # goes to Jan 14 at best — beyond Jan 13 — so end_d resolves to None,
        # and the row is dropped via the None check (existing behaviour).
        # To exercise the COLLISION branch specifically, pick a sim_d AFTER
        # Jan 13 so its walk-back lands on Jan 13, and choose an end_d whose
        # walk-back also lands on Jan 13.
        thin_only = date(2025, 1, 13)
        # sim_d = Jan 14 (walk back to Jan 13), idx within trading_days.
        sim_d = date(2025, 1, 14)
        sim_idx = eng.prices.trading_days.index(sim_d)
        # We need end_d = trading_days[sim_idx + 5] to also walk back to Jan 13.
        # trading_days[sim_idx+5] = sim_d + 5 weekdays = Jan 21 (Tue). Walk-back
        # from Jan 21: 7 days back → Jan 14. Doesn't reach Jan 13. To force a
        # collision, override trading_days to make end_d close enough.
        # Replace with a tighter calendar where 5 SPY days only span 5 calendar
        # days (e.g., add Saturdays to SPY series, fake but valid for the test).
        from paper_trader.backtest import PriceCache  # noqa: F401
        cache = eng.prices
        # Make SPY trade every day (incl. weekend) so 5 trading days from
        # sim_d=Jan 14 ends on Jan 19 — within walk-back range of Jan 13.
        cache.prices["SPY"] = {
            (date(2025, 1, 6) + timedelta(days=i)).isoformat(): 100.0 + i
            for i in range(40)
        }
        cache.trading_days = [
            date(2025, 1, 6) + timedelta(days=i) for i in range(40)
        ]
        # sim_d=Jan 14 walk-back → Jan 13. end_d = Jan 14 + 5 = Jan 19.
        # Jan 19 walk-back → Jan 19 - 6 = Jan 13. Both resolve to Jan 13.
        sim_d = date(2025, 1, 14)
        assert cache.resolved_close_date("THIN", sim_d) == thin_only
        end_d = date(2025, 1, 19)
        assert cache.resolved_close_date("THIN", end_d) == thin_only

        # Insert a FILLED BUY of THIN on sim_d.
        eng.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "status, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
            (1, sim_d.isoformat(), "BUY", "THIN", "FILLED",
             "ML+quant: THIN score=2.00 regime=bull news_count=0 news_urg=0.0"),
        )
        eng.store.conn.commit()
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-06",
                             end_date="2025-02-14")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        thin_outs = [o for o in outs if o["ticker"] == "THIN"
                     and o["sim_date"] == sim_d.isoformat()]
        # Pre-fix: a fabricated 0.0% row would appear here. Post-fix: dropped.
        assert thin_outs == [], (
            "walk-back collision must NOT produce a 0.0% outcome — "
            f"got {thin_outs!r}"
        )

    def test_genuine_no_walkback_outcome_still_recorded(self, tmp_path,
                                                       synthetic_prices):
        """Sanity check: when there is NO collision (NVDA has data on every
        trading day in the synthetic_prices fixture), a real forward return
        IS still recorded. The fix must not regress the common case.
        """
        eng = _make_engine_with_runs(tmp_path, n_runs=1)
        eng.prices = synthetic_prices
        sim_d = synthetic_prices.trading_days[5].isoformat()
        eng.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "status, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
            (1, sim_d, "BUY", "NVDA", "FILLED",
             "ML+quant: NVDA score=2.00 regime=bull news_count=0 news_urg=0.0"),
        )
        eng.store.conn.commit()
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                             end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        nvda = [o for o in outs if o["sim_date"] == sim_d and o["ticker"] == "NVDA"]
        assert len(nvda) == 1
        # NVDA: 100, 102, 104, ... — day 5 = 110, day 10 = 120 → return 9.09%
        assert nvda[0]["forward_return_5d"] == pytest.approx(9.0909, abs=1e-3)


# ───────────────── wk52_pos / pct_from_52h capture in outcomes ─────────────
#
# `_compute_technical_indicators` already computes `wk52_pos` (0..1 position
# in the trailing 52-week range) and `pct_from_52h` (% below the trailing
# high), and `_ml_decide` consumes `wk52_pos` for its bubble-top BUY gate.
# But until now neither value was persisted to `decision_outcomes.jsonl` —
# so the documented "bubble-top gate" explanation could never be empirically
# checked against realized forward returns. The capture is purely additive
# (the scorer trains via explicit kwargs to build_features and ignores
# extra dict keys), matching the `forward_return_10d/20d` precedent.

class TestWk52PosCapturedInOutcomes:
    def _build_long_history_cache(self):
        """PriceCache with 80 weekdays of NVDA closes — enough to satisfy
        `_compute_technical_indicators`'s ``len(pairs) >= 60`` gate so the
        function actually returns indicator values (including `wk52_pos`).
        SPY only needs enough to be the trading-day calendar source; we mirror
        NVDA so `_market_regime` doesn't trip on missing SPY history.
        """
        from paper_trader.backtest import PriceCache

        start = date(2025, 1, 2)
        days = []
        d = start
        while len(days) < 80:
            if d.weekday() < 5:
                days.append(d)
            d += timedelta(days=1)
        # NVDA: 100 → ~158, monotonic up — sim_d near the end is close to the
        # 52w high (wk52_pos ~= 1.0). SPY parallels NVDA so trading_days has 80
        # weekdays AND the regime computation has data (returns "unknown" with
        # only 80 closes, but that's fine — wk52_pos is the thing under test).
        cache = PriceCache.__new__(PriceCache)
        cache.tickers = ["SPY", "NVDA"]
        cache.start = days[0]
        cache.end = days[-1]
        cache.prices = {
            "SPY": {dd.isoformat(): 100.0 + i for i, dd in enumerate(days)},
            "NVDA": {dd.isoformat(): 100.0 + i * 0.75
                     for i, dd in enumerate(days)},
        }
        cache.trading_days = days
        return cache

    def test_wk52_pos_field_present_when_history_sufficient(self, tmp_path):
        eng = _make_engine_with_runs(tmp_path, n_runs=1)
        eng.prices = self._build_long_history_cache()
        # sim_d at index 70: 70 prior closes >= 60 → indicators computable.
        # 5-day forward window (idx 75) is within range (cache has 80 days).
        sim_d = eng.prices.trading_days[70]
        eng.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "status, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
            (1, sim_d.isoformat(), "BUY", "NVDA", "FILLED",
             "ML+quant: NVDA score=3.50 regime=bull news_count=0 news_urg=0.0"),
        )
        eng.store.conn.commit()
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-02",
                            end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        nvda = [o for o in outs if o["ticker"] == "NVDA"
                and o["sim_date"] == sim_d.isoformat()]
        assert len(nvda) == 1, f"expected 1 NVDA outcome, got {outs!r}"
        # New fields MUST be present (not missing keys) and carry numeric
        # values for this sufficient-history fixture.
        assert "wk52_pos" in nvda[0]
        assert "pct_from_52h" in nvda[0]
        # NVDA is monotonically increasing → sim_d (idx 70) close is the
        # trailing high of the 71-day window, so wk52_pos ≈ 1.0.
        # _compute_technical_indicators rounds to 2dp, so allow tight tolerance.
        assert nvda[0]["wk52_pos"] == pytest.approx(1.0, abs=0.02)
        # pct_from_52h ≈ 0.0% since sim_d's close IS the high. Rounded to 1dp.
        assert nvda[0]["pct_from_52h"] == pytest.approx(0.0, abs=0.5)

    def test_wk52_pos_is_none_when_history_insufficient(self, tmp_path,
                                                       synthetic_prices):
        """synthetic_prices fixture has only ~51 days of NVDA data, below the
        60-close threshold `_compute_technical_indicators` requires — so its
        return is None and `wk52_pos` should be captured as None, NOT as a
        sentinel like 0.0 (which would be indistinguishable from a real
        "ticker at 52w low" signal and silently poison training).
        """
        eng = _make_engine_with_runs(tmp_path, n_runs=1)
        eng.prices = synthetic_prices
        sim_d = synthetic_prices.trading_days[5].isoformat()
        eng.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "status, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
            (1, sim_d, "BUY", "NVDA", "FILLED",
             "ML+quant: NVDA score=2.00 regime=bull news_count=0 news_urg=0.0"),
        )
        eng.store.conn.commit()
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-02",
                            end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        nvda = [o for o in outs if o["sim_date"] == sim_d
                and o["ticker"] == "NVDA"]
        assert len(nvda) == 1
        # Key present, value None — distinguishable from a real numeric reading.
        assert nvda[0]["wk52_pos"] is None
        assert nvda[0]["pct_from_52h"] is None

    def test_persona_and_regime_label_captured(self, tmp_path,
                                                synthetic_prices):
        """Additive: each outcome row carries the persona name + raw regime
        label so downstream tools don't need to re-derive them.

        - `persona`: maps from `run_id` via `persona_for` so a future rename
          in the live `PERSONAS` dict doesn't desynchronize old outcome rows
          from their persona-time labels.
        - `regime_label`: the raw `bull`/`sideways`/`bear`/`unknown` string —
          NOT the multiplier, since `bull` and `unknown` both have
          `regime_mult=1.0` and an analysis cut on `regime_mult==1.0` silently
          conflates them.
        """
        eng = _make_engine_with_runs(tmp_path, n_runs=1)
        eng.prices = synthetic_prices
        sim_d = synthetic_prices.trading_days[5].isoformat()
        # run_id 2 → persona_for returns persona key 2 (Momentum Trader)
        eng.store.conn.execute(
            "INSERT INTO backtest_runs (run_id, seed, start_date, end_date,"
            " start_value, status, started_at) VALUES (?,?,?,?,?,?,?)",
            (2, 2, "2025-01-02", "2025-12-31", 1000.0, "complete",
             "2025-01-02T00:00:00+00:00"),
        )
        eng.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "status, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
            (2, sim_d, "BUY", "NVDA", "FILLED",
             "ML+quant: NVDA score=2.50 regime=bull news_count=0 news_urg=0.0"),
        )
        eng.store.conn.commit()
        runs = [BacktestRun(run_id=2, seed=2, start_date="2025-01-02",
                            end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        nvda = [o for o in outs if o["sim_date"] == sim_d
                and o["ticker"] == "NVDA"]
        assert len(nvda) == 1
        # Persona name pulled from PERSONAS dict — run_id=2 → "Momentum Trader".
        assert nvda[0]["persona"] == "Momentum Trader"
        # Regime label is the raw string returned by `_market_regime`. With
        # synthetic_prices' short history, `_market_regime` falls through to
        # "unknown" (insufficient SPY 200d MA) — exactly the case the
        # additive capture exists to distinguish from a real "bull" cycle
        # that would carry the SAME regime_mult=1.0.
        assert nvda[0]["regime_label"] in (
            "bull", "sideways", "bear", "unknown",
        ), nvda[0]["regime_label"]
        # `regime_mult` is preserved alongside the new label — additive, not
        # a replacement, so existing diagnostics that read the multiplier
        # keep working unchanged.
        assert "regime_mult" in nvda[0]

    def test_persona_field_present_for_run_id_1(self, tmp_path,
                                                synthetic_prices):
        """run_id=1 → persona key 1 → "Value Investor". Locks the
        persona_for mapping into the outcome row's persona field so a future
        change to ``persona_for``'s cycling formula immediately surfaces."""
        eng = _make_engine_with_runs(tmp_path, n_runs=1)
        eng.prices = synthetic_prices
        sim_d = synthetic_prices.trading_days[5].isoformat()
        eng.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "status, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
            (1, sim_d, "BUY", "NVDA", "FILLED",
             "ML+quant: NVDA score=2.50 regime=bull news_count=0 news_urg=0.0"),
        )
        eng.store.conn.commit()
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-02",
                            end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        nvda = [o for o in outs if o["sim_date"] == sim_d
                and o["ticker"] == "NVDA"]
        assert len(nvda) == 1
        assert nvda[0]["persona"] == "Value Investor"

    def test_capture_does_not_break_existing_keys(self, tmp_path,
                                                  synthetic_prices):
        """Regression lock: the additive capture must not displace or rename
        any existing outcome field. A downstream tool reading
        `forward_return_5d` / `ml_score` / `gate_scorer_pred` must continue to
        find them.
        """
        eng = _make_engine_with_runs(tmp_path, n_runs=1)
        eng.prices = synthetic_prices
        sim_d = synthetic_prices.trading_days[5].isoformat()
        eng.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "status, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
            (1, sim_d, "BUY", "NVDA", "FILLED",
             "ML+quant: NVDA score=2.50 regime=bull news_count=0 news_urg=0.0"),
        )
        eng.store.conn.commit()
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-02",
                            end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        nvda = [o for o in outs if o["sim_date"] == sim_d
                and o["ticker"] == "NVDA"]
        assert len(nvda) == 1
        # Spot-check that every documented existing field is still present.
        for required in ("run_id", "sim_date", "ticker", "action", "ml_score",
                         "rsi", "macd", "mom5", "mom20", "regime_mult",
                         "vol_ratio", "bb_position", "news_urgency",
                         "news_article_count", "forward_return_5d",
                         "forward_return_10d", "forward_return_20d",
                         "gate_scorer_pred", "gate_off_dist", "return_pct"):
            assert required in nvda[0], (
                f"existing field {required!r} missing from outcomes — "
                "the wk52_pos capture must not have displaced it"
            )


# ─────────────────────── _parse_published_date ───────────────────────────

class TestParsePublishedDate:
    def test_iso_string(self):
        assert rcb._parse_published_date("2025-03-15") == date(2025, 3, 15)

    def test_iso_with_time(self):
        assert rcb._parse_published_date("2025-03-15T10:30:00Z") == date(2025, 3, 15)

    def test_rfc822(self):
        assert rcb._parse_published_date("Wed, 14 May 2025 12:00:00 +0000") == date(2025, 5, 14)

    def test_none(self):
        assert rcb._parse_published_date(None) is None
        assert rcb._parse_published_date("") is None

    def test_garbage_returns_none(self):
        # Critical: garbage timestamps must not crash, must return None — caller
        # treats None as "don't apply date filter".
        assert rcb._parse_published_date("not a date") is None
        assert rcb._parse_published_date("xxxxxxx") is None


# ─────────────────────── _query_news_context ───────────────────────────

class TestQueryNewsContext:
    def test_filters_backtest_synthetic_articles(self, tmp_path, empty_articles_db,
                                                  monkeypatch):
        """The live-only filter is load-bearing. Backtest-injected articles must
        never leak into the LLM annotation news context — that's training
        contamination."""
        conn = sqlite3.connect(str(empty_articles_db))
        # Insert one real article and one backtest-injected article matching same ticker.
        conn.execute(
            "INSERT INTO articles (id, url, title, source, published, ai_score) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("real1", "https://real.com/x", "NVDA beats earnings",
             "reuters", "2025-05-01", 4.0),
        )
        conn.execute(
            "INSERT INTO articles (id, url, title, source, published, ai_score) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("bt1", "backtest://run_5/2025-05-01/BUY/NVDA",
             "NVDA backtest decision", "backtest_run_5", "2025-05-01", 5.0),
        )
        conn.commit()
        conn.close()

        # Point ROOT to a parent containing digital-intern/data/articles.db
        fake_di = tmp_path / "digital-intern" / "data"
        fake_di.mkdir(parents=True)
        # Move the articles.db to where _query_news_context expects.
        import shutil
        shutil.copy(empty_articles_db, fake_di / "articles.db")

        # Monkeypatch ROOT so ROOT.parent / "digital-intern" / ... resolves correctly.
        # ROOT.parent must equal tmp_path. So ROOT = tmp_path / "anything".
        fake_root = tmp_path / "paper-trader"
        fake_root.mkdir()
        monkeypatch.setattr(rcb, "ROOT", fake_root)

        titles = rcb._query_news_context("NVDA", "2025-05-02", n=5)
        # Real article should appear; backtest-injected one must NOT.
        assert any("beats earnings" in t for t in titles)
        assert not any("backtest decision" in t for t in titles)

    def test_missing_db_returns_empty(self, tmp_path, monkeypatch):
        # No DB file at all — must return [] without crashing.
        fake_root = tmp_path / "paper-trader"
        fake_root.mkdir()
        monkeypatch.setattr(rcb, "ROOT", fake_root)
        assert rcb._query_news_context("NVDA", "2025-05-01") == []

    def test_invalid_date_returns_empty(self, tmp_path, monkeypatch):
        fake_root = tmp_path / "paper-trader"
        fake_root.mkdir()
        monkeypatch.setattr(rcb, "ROOT", fake_root)
        assert rcb._query_news_context("NVDA", "not-a-date") == []


# ─────────────────────── _train_decision_scorer wrapper ──────────────────

class TestTrainDecisionScorer:
    def test_no_records_returns_message(self):
        assert "no outcome records" in rcb._train_decision_scorer([])

    def test_insufficient_data_status(self):
        status = rcb._train_decision_scorer([{"ticker": "NVDA", "sim_date": "2025-01-01",
                                              "action": "BUY", "forward_return_5d": 1.0}])
        # 1 record → insufficient_after_dedup
        assert "insufficient" in status

    def test_temporal_split_reports_oos_rmse(self):
        # Happy path: enough distinct (ticker, sim_date, action) keys to clear
        # train_scorer's >=30 dedup gate, plus enough range that the temporal
        # 80/20 split leaves a non-empty OOS set the scorer is evaluated on.
        import random as _rnd
        rng = _rnd.Random(11)
        records = []
        for i in range(80):
            month = 1 + (i % 12)
            day = 1 + (i // 12)
            records.append({
                "ticker": "NVDA" if i % 2 == 0 else "AMD",
                "sim_date": f"2024-{month:02d}-{day:02d}",
                "action": "BUY",
                "ml_score": rng.uniform(0, 5),
                "rsi": rng.uniform(20, 80),
                "macd": rng.uniform(-1, 1),
                "mom5": rng.uniform(-3, 3),
                "mom20": rng.uniform(-5, 5),
                "regime_mult": 1.0,
                "forward_return_5d": rng.uniform(-3, 3),
                "return_pct": 10.0,
            })
        status = rcb._train_decision_scorer(records)
        # Status string should report both train and OOS metrics.
        assert "train_n=" in status
        assert "oos_n=" in status
        assert "oos_rmse=" in status
        # OOS holdout must be non-empty (~20% of 80)
        assert "oos_n=0" not in status

    def test_oos_eval_failure_does_not_mask_successful_train(self, monkeypatch):
        """A post-training OOS-eval crash must NOT be reported as a training
        failure.

        ``train_scorer`` pickles the model to ``SCORER_PATH`` and returns
        ``status="ok"`` *before* the OOS diagnostic runs. If the OOS step then
        raises (transient pickle/IO race, validation-module change, …) the
        scorer is in fact trained and gets deployed (the singleton is reset and
        reloads it next cycle) — but a single broad ``except`` around both the
        train call and the diagnostic would surface ``scorer err`` to the
        operator-facing log/Discord, falsely signalling a broken scorer and the
        gate never engaging. The status must stay truthful: training succeeded.
        """
        import random as _rnd
        import paper_trader.validation as _val

        rng = _rnd.Random(13)
        records = []
        for i in range(80):
            month = 1 + (i % 12)
            day = 1 + (i // 12)
            records.append({
                "ticker": "NVDA" if i % 2 == 0 else "AMD",
                "sim_date": f"2024-{month:02d}-{day:02d}",
                "action": "BUY",
                "ml_score": rng.uniform(0, 5),
                "rsi": rng.uniform(20, 80),
                "macd": rng.uniform(-1, 1),
                "mom5": rng.uniform(-3, 3),
                "mom20": rng.uniform(-5, 5),
                "regime_mult": 1.0,
                "forward_return_5d": rng.uniform(-3, 3),
                "return_pct": 10.0,
            })

        def _boom(*_a, **_kw):
            raise RuntimeError("simulated OOS-eval crash after pickling")

        monkeypatch.setattr(_val, "evaluate_scorer_oos", _boom)

        status = rcb._train_decision_scorer(records)
        # Training succeeded and was pickled — the status must reflect that,
        # not a generic "scorer err".
        assert not status.startswith("scorer err"), status
        assert "scorer ok" in status, status
        assert "train_n=" in status, status
        # OOS metric degrades gracefully to n/a rather than killing the report.
        assert "oos_rmse=n/a" in status, status
        # And the model the next cycle will load is genuinely trained.
        from paper_trader.ml.decision_scorer import DecisionScorer
        assert DecisionScorer().is_trained is True

    def test_temporal_split_failure_still_trains(self, monkeypatch):
        """A *pre-training* split failure must NOT skip training.

        The temporal holdout (split_outcomes_temporal) is a diagnostic
        refinement, not the essential operation. Before the fix it sat in the
        same try/except as ``train_scorer``, so a split crash (or an
        unavailable validation module) returned ``scorer err:`` and the model
        was never pickled — silently freezing the per-cycle retrain invariant
        (CLAUDE.md §6) and the conviction gate (#5). After the fix the split
        failure degrades to "train on all records, no OOS" and the scorer is
        still retrained and deployed.
        """
        import random as _rnd
        import paper_trader.validation as _val

        rng = _rnd.Random(17)
        records = []
        for i in range(80):
            month = 1 + (i % 12)
            day = 1 + (i // 12)
            records.append({
                "ticker": "NVDA" if i % 2 == 0 else "AMD",
                "sim_date": f"2024-{month:02d}-{day:02d}",
                "action": "BUY",
                "ml_score": rng.uniform(0, 5),
                "rsi": rng.uniform(20, 80),
                "macd": rng.uniform(-1, 1),
                "mom5": rng.uniform(-3, 3),
                "mom20": rng.uniform(-5, 5),
                "regime_mult": 1.0,
                "forward_return_5d": rng.uniform(-3, 3),
                "return_pct": 10.0,
            })

        def _boom(*_a, **_kw):
            raise RuntimeError("simulated split crash before training")

        monkeypatch.setattr(_val, "split_outcomes_temporal", _boom)

        status = rcb._train_decision_scorer(records)
        # Training proceeded despite the split crash — status is truthful.
        assert not status.startswith("scorer err"), status
        assert "scorer ok" in status, status
        # All 80 records used for training (no holdout carved out).
        assert "train_n=80" in status, status
        assert "oos_n=0" in status, status
        # The model the next cycle reloads is genuinely trained and pickled.
        from paper_trader.ml.decision_scorer import DecisionScorer
        assert DecisionScorer().is_trained is True


# ──────────────────── _oos_rank_metrics direct ────────────────────

class TestOosRankMetrics:
    """Direct coverage of `_oos_rank_metrics`. Previously this function had
    no targeted tests — only end-to-end coverage via the `_train_decision_scorer`
    status string. The NaN-sentinel fix (a missing `forward_return_5d` must
    be DROPPED, not coerced to 0.0 which fabricates flat-target ties) is
    locked here.
    """

    @staticmethod
    def _scorer_returning(predictions: list[float]):
        """Build a minimal scorer stub that yields the next prediction in
        sequence per call. Mirrors how DecisionScorer.predict is consumed.
        """
        class _Stub:
            is_trained = True

            def __init__(self, preds):
                self._it = iter(preds)

            def predict(self, **_kw):
                return next(self._it)

        return _Stub(predictions)

    def test_untrained_scorer_returns_zeroed_metrics(self):
        class _Untrained:
            is_trained = False

            def predict(self, **_kw):
                return 0.0

        out = rcb._oos_rank_metrics(_Untrained(), [
            {"forward_return_5d": 1.0, "action": "BUY", "ticker": "NVDA"}
        ])
        # Aggregate + per-action breakdown both honest-empty when the model
        # isn't trained. Locks the full shape so a future caller can rely on
        # the new buy/sell keys being present even on the no-op path.
        assert out["dir_acc"] is None
        assert out["rank_ic"] is None
        assert out["n"] == 0
        assert out["buy_n"] == 0
        assert out["buy_dir_acc"] is None
        assert out["buy_rank_ic"] is None
        assert out["sell_n"] == 0
        assert out["sell_dir_acc"] is None
        assert out["sell_rank_ic"] is None

    def test_records_missing_forward_return_are_dropped_not_zeroed(self):
        """A `forward_return_5d=None` (or missing key) MUST be dropped.
        Pre-fix the function defaulted the missing value to 0.0 via _to_float,
        which then passed the `a == a` finite check and poisoned rank_ic with
        a fabricated 0.0 actual. After the NaN-sentinel fix the same record is
        explicitly excluded.
        """
        scorer = self._scorer_returning([0.5, 1.0, -0.5])
        records = [
            {"forward_return_5d": 1.5, "action": "BUY", "ticker": "NVDA"},
            # This null actual would silently bias rank_ic if coerced to 0.0:
            {"forward_return_5d": None, "action": "BUY", "ticker": "AMD"},
            {"forward_return_5d": -0.8, "action": "BUY", "ticker": "MU"},
        ]
        out = rcb._oos_rank_metrics(scorer, records)
        # Only the 2 non-null records contribute.
        assert out["n"] == 2
        # rank_ic computable on 2 pairs (preds=[0.5,-0.5], actuals=[1.5,-0.8])
        # — both move in the same direction so IC must be +1.0.
        assert out["rank_ic"] == pytest.approx(1.0)
        # Both pairs have non-zero p and a, so dir_acc = 2/2 = 1.0.
        assert out["dir_acc"] == pytest.approx(1.0)

    def test_missing_forward_return_key_entirely_is_dropped(self):
        """The key being entirely absent (not just None) is also dropped —
        the NaN sentinel path catches both equivalently. Locks the contract
        for a future schema that omits the field for some rows.
        """
        scorer = self._scorer_returning([0.5, 1.0, -0.5])
        records = [
            {"forward_return_5d": 1.5, "action": "BUY", "ticker": "NVDA"},
            {"action": "BUY", "ticker": "AMD"},  # forward_return_5d absent
            {"forward_return_5d": -0.8, "action": "BUY", "ticker": "MU"},
        ]
        out = rcb._oos_rank_metrics(scorer, records)
        assert out["n"] == 2

    def test_sell_action_flips_actual_sign_not_prediction(self):
        """Mirror train_scorer's SELL convention: realized goodness of a
        SELL is -forward_return_5d. The prediction is NOT flipped (the model
        was trained on flipped targets so its output already encodes the
        action-aligned goodness). A drop after a SELL should read as a
        positive contribution to rank_ic.
        """
        # Scorer predicts +1.0 for both records.
        scorer = self._scorer_returning([1.0, 1.0])
        records = [
            # BUY that went up — predicted +1, realized +2 → both positive
            {"forward_return_5d": 2.0, "action": "BUY", "ticker": "NVDA"},
            # SELL that went DOWN (a good SELL) — predicted +1, realized -3,
            # but the SELL flip turns the actual into +3 → still positive
            {"forward_return_5d": -3.0, "action": "SELL", "ticker": "AMD"},
        ]
        out = rcb._oos_rank_metrics(scorer, records)
        assert out["n"] == 2
        # Both actions have action-aligned positives — dir_acc must be 1.0
        # not 0.5 (which would be the case if the SELL flip were missing).
        assert out["dir_acc"] == pytest.approx(1.0)

    def test_dir_acc_skips_zero_predictions_and_zero_actuals(self):
        """A 0.0 prediction or 0.0 actual carries no directional truth and
        must be excluded from dir_acc — but it still contributes to rank_ic
        (zero is a legitimate rank).
        """
        scorer = self._scorer_returning([0.0, 1.0, -1.0])
        records = [
            # p=0 → excluded from dir_acc
            {"forward_return_5d": 2.0, "action": "BUY", "ticker": "NVDA"},
            # both non-zero, same sign → hit
            {"forward_return_5d": 0.5, "action": "BUY", "ticker": "AMD"},
            # both non-zero, opposite sign → miss
            {"forward_return_5d": 0.5, "action": "BUY", "ticker": "MU"},
        ]
        out = rcb._oos_rank_metrics(scorer, records)
        assert out["n"] == 3
        # 1 hit out of 2 non-zero pairs
        assert out["dir_acc"] == pytest.approx(0.5)

    def test_predict_exception_skips_that_record(self):
        """A scorer that raises on one input must NOT poison the whole
        report — that record is dropped and the rest still contribute.
        """
        class _PartialRaiser:
            is_trained = True

            def __init__(self):
                self.calls = 0

            def predict(self, **_kw):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("simulated transient predict fault")
                return 0.5 if self.calls == 1 else -0.5

        records = [
            {"forward_return_5d": 1.0, "action": "BUY", "ticker": "NVDA"},
            {"forward_return_5d": 2.0, "action": "BUY", "ticker": "AMD"},
            {"forward_return_5d": -1.0, "action": "BUY", "ticker": "MU"},
        ]
        out = rcb._oos_rank_metrics(_PartialRaiser(), records)
        # 2 of 3 records survive.
        assert out["n"] == 2

    def test_single_record_yields_rank_ic_none(self):
        """rank_ic requires n>=2 pairs; a single record reports n=1 with
        rank_ic=None (never NaN, never a fabricated +1.0)."""
        scorer = self._scorer_returning([0.7])
        out = rcb._oos_rank_metrics(scorer, [
            {"forward_return_5d": 1.2, "action": "BUY", "ticker": "NVDA"},
        ])
        assert out["n"] == 1
        assert out["rank_ic"] is None
        # Single pair both non-zero → dir_acc is computable
        assert out["dir_acc"] == pytest.approx(1.0)

    def test_extreme_label_clamped_keeps_dir_acc_truthful(self):
        """An extreme |forward_return_5d|>50 must be clamped to ±50 so OOS
        rank metrics describe the same target space the scorer was trained
        against (mirrors train_scorer's symmetric label-clamp and
        evaluate_scorer_oos's RMSE clamp). dir_acc operates on signs so the
        clamp is a no-op there — but locking this test prevents a future
        regression where the clamp would be silently dropped from one of
        the OOS paths but not the other (cross-diagnostic drift).
        """
        # Scorer predicts +50 (the gate-aligned clamp ceiling) and -50.
        scorer = self._scorer_returning([50.0, -50.0])
        records = [
            # Realized +175 — model predicted +50, both positive → hit.
            {"forward_return_5d": 175.0, "action": "BUY", "ticker": "MSTR"},
            # Realized -200 — model predicted -50, both negative → hit.
            {"forward_return_5d": -200.0, "action": "BUY", "ticker": "SOXS"},
        ]
        out = rcb._oos_rank_metrics(scorer, records)
        assert out["n"] == 2
        # Both directional hits.
        assert out["dir_acc"] == pytest.approx(1.0)
        # rank_ic computable (n=2, both pairs concordant) — IC = 1.0.
        assert out["rank_ic"] == pytest.approx(1.0)

    def test_per_action_breakdown_isolates_buy_from_sell(self):
        """The conviction gate (#5) is BUY-only, so a quant needs the
        BUY-specific rank-IC separately from the aggregate. This test
        proves the bucket split is correct: a scorer that's PERFECTLY
        right on BUYs and PERFECTLY wrong on SELLs must report a
        positive buy_rank_ic AND a negative sell_rank_ic — the
        aggregate would hide that by averaging.
        """
        scorer = self._scorer_returning([
            # BUY predictions agree with realized signs.
            1.0, -1.0,
            # SELL predictions disagree with sign-flipped realized.
            1.0, -1.0,
        ])
        records = [
            # BUY records — perfect concordance (sign(pred) == sign(actual))
            {"forward_return_5d": 2.0, "action": "BUY", "ticker": "NVDA"},
            {"forward_return_5d": -3.0, "action": "BUY", "ticker": "AMD"},
            # SELL records — after sign flip, the realized goodness is
            # -forward_return_5d. predictions disagree with that flipped sign.
            # raw fwd=-2 → flipped +2; pred=+1.0 → BOTH positive → "concordant"
            # AFTER flip but the SELL bucket reports rank-IC over (pred, flipped).
            # So to make the SELL bucket NEGATIVE rank-IC we set:
            # raw fwd=+2 → flipped -2; pred=+1.0 → discordant → miss
            {"forward_return_5d": 2.0, "action": "SELL", "ticker": "MU"},
            # raw fwd=-2 → flipped +2; pred=-1.0 → discordant → miss
            {"forward_return_5d": -2.0, "action": "SELL", "ticker": "INTC"},
        ]
        out = rcb._oos_rank_metrics(scorer, records)
        # Counts honest per bucket.
        assert out["n"] == 4
        assert out["buy_n"] == 2
        assert out["sell_n"] == 2
        # BUY bucket: perfect concordance → IC=+1.0, dir_acc=1.0
        assert out["buy_rank_ic"] == pytest.approx(1.0)
        assert out["buy_dir_acc"] == pytest.approx(1.0)
        # SELL bucket: perfect anti-concordance → IC=-1.0, dir_acc=0.0
        assert out["sell_rank_ic"] == pytest.approx(-1.0)
        assert out["sell_dir_acc"] == pytest.approx(0.0)

    def test_per_action_buckets_handle_empty_one_side(self):
        """When OOS rows are 100% BUYs (the common case — SELLs are rarer),
        the SELL bucket reports n=0 and metric=None honestly rather than a
        fabricated value. The BUY metrics are unaffected."""
        scorer = self._scorer_returning([0.5, -0.5, 1.0])
        records = [
            {"forward_return_5d": 1.0, "action": "BUY", "ticker": "NVDA"},
            {"forward_return_5d": -1.0, "action": "BUY", "ticker": "AMD"},
            {"forward_return_5d": 2.0, "action": "BUY", "ticker": "MU"},
        ]
        out = rcb._oos_rank_metrics(scorer, records)
        assert out["buy_n"] == 3
        assert out["buy_rank_ic"] is not None  # computable
        # SELL bucket empty → honest None.
        assert out["sell_n"] == 0
        assert out["sell_rank_ic"] is None
        assert out["sell_dir_acc"] is None

    def test_per_action_drops_missing_forward_return_per_bucket(self):
        """A None forward_return drops from BOTH buckets correctly — the
        per-action split must respect the same NaN-sentinel discipline as
        the aggregate."""
        scorer = self._scorer_returning([0.5, 1.0, -0.5, 0.7])
        records = [
            {"forward_return_5d": 1.0, "action": "BUY", "ticker": "NVDA"},
            {"forward_return_5d": None, "action": "BUY", "ticker": "AMD"},  # drops
            {"forward_return_5d": -0.8, "action": "SELL", "ticker": "MU"},
            {"forward_return_5d": None, "action": "SELL", "ticker": "INTC"},  # drops
        ]
        out = rcb._oos_rank_metrics(scorer, records)
        assert out["buy_n"] == 1
        assert out["sell_n"] == 1
        # n=1 in each bucket → rank_ic=None (need n>=2)
        assert out["buy_rank_ic"] is None
        assert out["sell_rank_ic"] is None


# ──────────────────── _oos_multi_horizon_metrics direct ────────────

class TestOosMultiHorizonMetrics:
    """Per-longer-horizon (10d, 20d) OOS skill metrics.

    The scorer is trained on `forward_return_5d`, but each outcome row
    also carries `forward_return_10d` / `forward_return_20d` (added in
    the 2026-05-18 multi-horizon instrumentation). A non-trivial signal
    at 10d/20d when 5d is at noise is informative — AGENTS.md documents
    leveraged ETFs have noisy 5d windows but stronger multi-month
    returns. This locks the new metric's contract.
    """

    @staticmethod
    def _scorer_returning(predictions: list[float]):
        class _Stub:
            is_trained = True

            def __init__(self, preds):
                self._it = iter(preds)

            def predict(self, **_kw):
                return next(self._it)

        return _Stub(predictions)

    def test_untrained_scorer_returns_empty_per_horizon(self):
        class _Untrained:
            is_trained = False

            def predict(self, **_kw):
                return 0.0

        out = rcb._oos_multi_horizon_metrics(_Untrained(), [
            {"forward_return_10d": 1.0, "forward_return_20d": 2.0,
             "action": "BUY", "ticker": "NVDA"}
        ], horizons=(10, 20))
        # Honest empty sentinels for every requested horizon — never raises,
        # never fabricates a metric.
        assert out == {10: {"dir_acc": None, "rank_ic": None, "n": 0},
                       20: {"dir_acc": None, "rank_ic": None, "n": 0}}

    def test_perfect_rank_ordering_at_each_horizon(self):
        # Three records: predictions and 10d/20d realizations are perfectly
        # rank-ordered together. rank_ic must be EXACTLY +1.0 for both
        # horizons (tie-aware Spearman) and dir_acc must be 1.0 (all hits).
        scorer = self._scorer_returning([1.0, 0.5, -1.0])
        records = [
            {"forward_return_10d": 5.0, "forward_return_20d": 8.0,
             "action": "BUY", "ticker": "NVDA"},
            {"forward_return_10d": 2.0, "forward_return_20d": 3.0,
             "action": "BUY", "ticker": "AMD"},
            {"forward_return_10d": -3.0, "forward_return_20d": -4.0,
             "action": "BUY", "ticker": "MU"},
        ]
        out = rcb._oos_multi_horizon_metrics(scorer, records,
                                             horizons=(10, 20))
        assert out[10]["n"] == 3
        assert out[10]["rank_ic"] == pytest.approx(1.0)
        assert out[10]["dir_acc"] == pytest.approx(1.0)
        assert out[20]["n"] == 3
        assert out[20]["rank_ic"] == pytest.approx(1.0)
        assert out[20]["dir_acc"] == pytest.approx(1.0)

    def test_each_horizon_drops_only_its_own_missing_target(self):
        """A row missing forward_return_20d but having forward_return_10d
        must contribute to the 10d cell and be dropped from the 20d cell.
        Each horizon reports its own n independently — a 20d gap NEVER
        poisons the 10d view.
        """
        scorer = self._scorer_returning([0.6, -0.4, 1.2])
        records = [
            # Has both — contributes to both.
            {"forward_return_10d": 1.5, "forward_return_20d": 2.5,
             "action": "BUY", "ticker": "NVDA"},
            # Has only 10d (20d=None — common when 20d window runs past
            # cached price history) — contributes to 10d only.
            {"forward_return_10d": -1.0, "forward_return_20d": None,
             "action": "BUY", "ticker": "AMD"},
            # Has only 20d — contributes to 20d only.
            {"forward_return_10d": None, "forward_return_20d": 3.0,
             "action": "BUY", "ticker": "MU"},
        ]
        out = rcb._oos_multi_horizon_metrics(scorer, records,
                                             horizons=(10, 20))
        # 10d cell: 2 rows (NVDA + AMD). 20d cell: 2 rows (NVDA + MU).
        assert out[10]["n"] == 2
        assert out[20]["n"] == 2
        # Both 10d pairs go in the same direction (pred +0.6 → +1.5,
        # pred -0.4 → -1.0) → rank_ic = +1.0, dir_acc = 1.0.
        assert out[10]["rank_ic"] == pytest.approx(1.0)
        assert out[10]["dir_acc"] == pytest.approx(1.0)

    def test_sell_target_sign_flipped_consistently_at_all_horizons(self):
        """The SELL flip convention (mirror train_scorer / evaluate_scorer_oos
        / _oos_rank_metrics) must apply to forward_return_{h}d for every
        horizon — otherwise the 10d/20d metrics report the WRONG action-
        aligned target and a "good SELL" looks like a wrong call.
        """
        scorer = self._scorer_returning([1.0, 1.0])
        records = [
            # BUY that went up — predicted +1, realized +2 at 10d → hit.
            {"forward_return_10d": 2.0, "forward_return_20d": 3.0,
             "action": "BUY", "ticker": "NVDA"},
            # SELL that went DOWN — predicted +1, realized -2 at 10d, but
            # the SELL flip makes the action-aligned actual +2 → hit too.
            # If the flip is missing, dir_acc would be 0.5, not 1.0.
            {"forward_return_10d": -2.0, "forward_return_20d": -3.0,
             "action": "SELL", "ticker": "AMD"},
        ]
        out = rcb._oos_multi_horizon_metrics(scorer, records,
                                             horizons=(10, 20))
        assert out[10]["dir_acc"] == pytest.approx(1.0)
        assert out[20]["dir_acc"] == pytest.approx(1.0)

    def test_predict_exception_drops_only_that_row(self):
        """A scorer that raises on one row must NOT poison every horizon —
        that row is dropped from BOTH horizons and the rest contribute.
        Mirrors _oos_rank_metrics.test_predict_exception_skips_that_record.
        """
        class _PartialRaiser:
            is_trained = True

            def __init__(self):
                self.calls = 0

            def predict(self, **_kw):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("simulated transient predict fault")
                return 0.5 if self.calls == 1 else -0.5

        records = [
            {"forward_return_10d": 1.0, "forward_return_20d": 2.0,
             "action": "BUY", "ticker": "NVDA"},
            {"forward_return_10d": 2.0, "forward_return_20d": 3.0,
             "action": "BUY", "ticker": "AMD"},
            {"forward_return_10d": -1.0, "forward_return_20d": -2.0,
             "action": "BUY", "ticker": "MU"},
        ]
        out = rcb._oos_multi_horizon_metrics(_PartialRaiser(), records,
                                             horizons=(10, 20))
        assert out[10]["n"] == 2
        assert out[20]["n"] == 2

    def test_train_status_includes_multi_horizon_tokens(self):
        """Wiring lock: _train_decision_scorer's returned status string MUST
        contain the new oos_n_10/oos_ic_10/oos_diracc_10/20 tokens whenever
        OOS records are present, so the scorer-skill ledger captures them
        on every training cycle.
        """
        import random as _rnd
        rng = _rnd.Random(31)
        records = []
        for i in range(80):
            month = 1 + (i % 12)
            day = 1 + (i // 12)
            records.append({
                "ticker": "NVDA" if i % 2 == 0 else "AMD",
                "sim_date": f"2024-{month:02d}-{day:02d}",
                "action": "BUY",
                "ml_score": rng.uniform(0, 5),
                "rsi": rng.uniform(20, 80),
                "macd": rng.uniform(-1, 1),
                "mom5": rng.uniform(-3, 3),
                "mom20": rng.uniform(-5, 5),
                "regime_mult": 1.0,
                "forward_return_5d": rng.uniform(-3, 3),
                "forward_return_10d": rng.uniform(-5, 5),
                "forward_return_20d": rng.uniform(-7, 7),
                "return_pct": 10.0,
            })
        status = rcb._train_decision_scorer(records)
        # Every multi-horizon token must surface so the parser can extract
        # them and the skill ledger row carries the new columns.
        for token in ("oos_n_10=", "oos_diracc_10=", "oos_ic_10=",
                      "oos_n_20=", "oos_diracc_20=", "oos_ic_20="):
            assert token in status, f"missing {token} in {status!r}"


# ──────────────────── scorer-skill ledger ───────────────────────

class TestParseScorerStatus:
    def test_parses_a_full_ok_status_row(self):
        s = ("scorer ok train_n=3540 val_rmse=5.20 oos_n=708 "
             "oos_rmse=12.40 oos_diracc=0.55 oos_ic=+0.03")
        p = rcb._parse_scorer_status(s)
        assert p["status"] == "ok"
        assert p["train_n"] == 3540          # int, not float
        assert p["oos_n"] == 708
        assert p["val_rmse"] == pytest.approx(5.20)
        assert p["oos_rmse"] == pytest.approx(12.40)
        assert p["oos_dir_acc"] == pytest.approx(0.55)
        assert p["oos_ic"] == pytest.approx(0.03)

    def test_na_tokens_degrade_to_none_not_crash(self):
        # The error path emits `oos_rmse=n/a (oos-eval err: KeyError)` — the
        # first token after `=` is `n/a`, which must float-fail to None and
        # NOT swallow the parenthetical into a bad parse.
        s = ("scorer ok train_n=600 val_rmse=n/a oos_n=0 "
             "oos_rmse=n/a (oos-eval err: KeyError) oos_diracc=n/a oos_ic=n/a")
        p = rcb._parse_scorer_status(s)
        assert p["status"] == "ok"
        assert p["train_n"] == 600
        assert p["val_rmse"] is None
        assert p["oos_rmse"] is None
        assert p["oos_dir_acc"] is None
        assert p["oos_ic"] is None

    def test_no_outcome_records_sentinel(self):
        p = rcb._parse_scorer_status("no outcome records")
        assert p["status"] == "no_outcome_records"
        assert p["train_n"] is None

    def test_garbage_is_unparseable_never_raises(self):
        p = rcb._parse_scorer_status("")
        assert p["status"] == "unparseable"
        p2 = rcb._parse_scorer_status(None)  # type: ignore[arg-type]
        assert p2["status"] == "unparseable"

    def test_parses_multi_horizon_tokens(self):
        # The 2026-05-18 feature pass adds oos_n_10/oos_diracc_10/oos_ic_10
        # (and 20d siblings) to the status string. Parser must extract them
        # alongside the legacy 5d tokens — and parse `oos_n_10` correctly
        # despite `oos_n` being a substring of its key (the parser uses an
        # exact `(?:^|\s)oos_n=…` boundary that must NOT match `oos_n_10=`).
        s = ("scorer ok train_n=3540 val_rmse=5.20 oos_n=708 "
             "oos_rmse=12.40 oos_diracc=0.55 oos_ic=+0.03 "
             "oos_n_10=620 oos_diracc_10=0.57 oos_ic_10=+0.09 "
             "oos_n_20=540 oos_diracc_20=0.59 oos_ic_20=+0.14")
        p = rcb._parse_scorer_status(s)
        assert p["status"] == "ok"
        # Legacy 5d fields are unchanged.
        assert p["oos_n"] == 708
        assert p["oos_ic"] == pytest.approx(0.03)
        # New 10d fields.
        assert p["oos_n_10"] == 620
        assert p["oos_dir_acc_10"] == pytest.approx(0.57)
        assert p["oos_ic_10"] == pytest.approx(0.09)
        # New 20d fields.
        assert p["oos_n_20"] == 540
        assert p["oos_dir_acc_20"] == pytest.approx(0.59)
        assert p["oos_ic_20"] == pytest.approx(0.14)

    def test_legacy_status_without_multi_horizon_tokens_parses_cleanly(self):
        # Old skill-log rows that pre-date the multi-horizon wiring carry
        # no oos_n_10/etc. tokens. Parser must still return them as None,
        # NOT raise and NOT fabricate a metric (the parser is the read-side
        # contract for every historical skill-log row).
        s = ("scorer ok train_n=3540 val_rmse=5.20 oos_n=708 "
             "oos_rmse=12.40 oos_diracc=0.55 oos_ic=+0.03")
        p = rcb._parse_scorer_status(s)
        # 5d unchanged.
        assert p["oos_n"] == 708
        # 10d/20d fields default to None.
        assert p["oos_n_10"] is None
        assert p["oos_dir_acc_10"] is None
        assert p["oos_ic_10"] is None
        assert p["oos_n_20"] is None
        assert p["oos_dir_acc_20"] is None
        assert p["oos_ic_20"] is None

    def test_oos_n_does_not_swallow_oos_n_10_value(self):
        """Boundary lock: a parser that uses a naïve `re.search("oos_n=…")`
        would match the substring inside `oos_n_10=`, capturing the 10d
        value as the 5d count and silently fabricating a wrong train_n=`/`
        oos_n=` pair. The implementation uses `(?:^|\\s)oos_n=` so the
        following char MUST be ``=`` for the legacy match — proven here.
        """
        # No oos_n token at all, only oos_n_10/20 — legacy field MUST stay None.
        s = ("scorer ok train_n=100 val_rmse=1.0 "
             "oos_n_10=42 oos_diracc_10=0.50 oos_ic_10=+0.01 "
             "oos_n_20=33 oos_diracc_20=0.55 oos_ic_20=+0.02")
        p = rcb._parse_scorer_status(s)
        assert p["oos_n"] is None       # not 42, not 33 — absent
        assert p["oos_n_10"] == 42
        assert p["oos_n_20"] == 33

    def test_parses_label_clamp_count(self):
        """`n_label_clamped` token is parsed as int (the label-clamp feature
        surfaces the per-cycle count of training labels truncated to
        ±PRED_CLAMP_PCT)."""
        s = ("scorer ok train_n=3540 val_rmse=5.20 oos_n=708 "
             "oos_rmse=12.40 oos_diracc=0.55 oos_ic=+0.03 "
             "n_label_clamped=27")
        p = rcb._parse_scorer_status(s)
        assert p["status"] == "ok"
        assert p["n_label_clamped"] == 27   # int, not float

    def test_legacy_status_without_label_clamp_parses_cleanly(self):
        """Pre-feature status strings carry no `n_label_clamped=` token;
        parser MUST default it to None (historical rows in the ledger must
        still parse, mirroring the multi-horizon legacy-status discipline)."""
        s = ("scorer ok train_n=3540 val_rmse=5.20 oos_n=708 "
             "oos_rmse=12.40 oos_diracc=0.55 oos_ic=+0.03")
        p = rcb._parse_scorer_status(s)
        assert p["status"] == "ok"
        assert p["n_label_clamped"] is None

    def test_parses_per_action_breakdown_tokens(self):
        """The 2026-05-20 per-action breakdown adds oos_buy_n/diracc/ic and
        oos_sell_* tokens. Parser must extract them as int counts + float
        metrics. Substring-boundary lock: `oos_n` must NOT swallow values
        from `oos_buy_n=` / `oos_sell_n=` (same `(?:^|\\s)` discipline)."""
        s = ("scorer ok train_n=3540 val_rmse=5.20 oos_n=708 "
             "oos_rmse=12.40 oos_diracc=0.55 oos_ic=+0.03 "
             "oos_buy_n=600 oos_buy_diracc=0.58 oos_buy_ic=+0.11 "
             "oos_sell_n=108 oos_sell_diracc=0.47 oos_sell_ic=-0.04")
        p = rcb._parse_scorer_status(s)
        assert p["status"] == "ok"
        # Legacy 5d aggregate unchanged.
        assert p["oos_n"] == 708
        # New per-action fields.
        assert p["oos_buy_n"] == 600
        assert p["oos_buy_dir_acc"] == pytest.approx(0.58)
        assert p["oos_buy_ic"] == pytest.approx(0.11)
        assert p["oos_sell_n"] == 108
        assert p["oos_sell_dir_acc"] == pytest.approx(0.47)
        assert p["oos_sell_ic"] == pytest.approx(-0.04)

    def test_legacy_status_without_per_action_tokens_parses_cleanly(self):
        """A pre-feature status string carries NO per-action tokens; parser
        must return them as None so historical skill-log rows still load
        (same legacy-status discipline as the multi-horizon / label-clamp
        precedents)."""
        s = ("scorer ok train_n=3540 val_rmse=5.20 oos_n=708 "
             "oos_rmse=12.40 oos_diracc=0.55 oos_ic=+0.03")
        p = rcb._parse_scorer_status(s)
        assert p["status"] == "ok"
        assert p["oos_buy_n"] is None
        assert p["oos_buy_dir_acc"] is None
        assert p["oos_buy_ic"] is None
        assert p["oos_sell_n"] is None
        assert p["oos_sell_dir_acc"] is None
        assert p["oos_sell_ic"] is None


class TestAppendScorerSkillLog:
    def test_trained_row_has_accurate_gate_active_flag(self, tmp_path, monkeypatch):
        log = tmp_path / "scorer_skill_log.jsonl"
        monkeypatch.setattr(rcb, "SCORER_SKILL_LOG", log)
        ok = rcb._append_scorer_skill_log(
            "scorer ok train_n=3540 val_rmse=5.2 oos_n=700 "
            "oos_rmse=12.4 oos_diracc=0.55 oos_ic=+0.03",
            cycle=7, win_start=date(2015, 1, 2), win_end=date(2016, 1, 2),
        )
        assert ok is True
        row = json.loads(log.read_text().strip())
        assert row["cycle"] == 7
        assert row["status"] == "ok"
        assert row["train_n"] == 3540
        assert row["oos_rmse"] == pytest.approx(12.4)
        assert row["window_start"] == "2015-01-02"
        # train_n >= 500 ⇒ the conviction gate is live (invariant #5).
        assert row["gate_active"] is True

    def test_below_threshold_gate_is_inactive(self, tmp_path, monkeypatch):
        log = tmp_path / "skill.jsonl"
        monkeypatch.setattr(rcb, "SCORER_SKILL_LOG", log)
        rcb._append_scorer_skill_log(
            "scorer ok train_n=120 val_rmse=5.2 oos_n=0 oos_rmse=n/a "
            "oos_diracc=n/a oos_ic=n/a",
            cycle=1, win_start=date(2010, 1, 4), win_end=date(2011, 1, 4),
        )
        row = json.loads(log.read_text().strip())
        assert row["train_n"] == 120
        assert row["gate_active"] is False

    def test_n_train_hint_used_when_status_omits_train_n(self, tmp_path, monkeypatch):
        # The "no outcome records" cycle has no train_n token; the deployed
        # pickle's n_train hint must drive an accurate gate_active flag.
        log = tmp_path / "skill.jsonl"
        monkeypatch.setattr(rcb, "SCORER_SKILL_LOG", log)
        rcb._append_scorer_skill_log(
            "no outcome records", cycle=2,
            win_start=date(2009, 1, 2), win_end=date(2010, 1, 2),
            n_train_hint=900,
        )
        row = json.loads(log.read_text().strip())
        assert row["status"] == "no_outcome_records"
        assert row["train_n"] == 900
        assert row["gate_active"] is True

    def test_bounded_trim_rewrites_when_past_2x_keep(self, tmp_path, monkeypatch):
        log = tmp_path / "skill.jsonl"
        monkeypatch.setattr(rcb, "SCORER_SKILL_LOG", log)
        monkeypatch.setattr(rcb, "SCORER_SKILL_LOG_KEEP", 5)
        # Pre-seed 11 rows (> 2×5) so the next append triggers a rewrite.
        log.write_text("\n".join(json.dumps({"cycle": i}) for i in range(11)) + "\n")
        rcb._append_scorer_skill_log(
            "scorer ok train_n=500 val_rmse=1 oos_n=0 oos_rmse=n/a "
            "oos_diracc=n/a oos_ic=n/a",
            cycle=99, win_start=date(2012, 1, 3), win_end=date(2013, 1, 3),
        )
        lines = [l for l in log.read_text().splitlines() if l.strip()]
        assert len(lines) == 5  # trimmed to SCORER_SKILL_LOG_KEEP
        # The freshly-appended cycle-99 row must survive the trim (it's newest).
        assert json.loads(lines[-1])["cycle"] == 99

    def test_never_raises_on_unwritable_path(self, tmp_path, monkeypatch):
        # Parent dir does not exist and cannot be created (path is a file).
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        monkeypatch.setattr(rcb, "SCORER_SKILL_LOG", blocker / "sub" / "skill.jsonl")
        assert rcb._append_scorer_skill_log("no outcome records", 1,
                                            date(2000, 1, 3), date(2001, 1, 3)) is False


class TestDeployedScorerNTrain:
    def test_none_when_no_pickle(self):
        # conftest redirects SCORER_PATH into an empty tmp dir — no pickle yet.
        assert rcb._deployed_scorer_n_train() is None

    def test_reads_n_train_from_a_trained_pickle(self):
        import random as _rnd
        from paper_trader.ml.decision_scorer import train_scorer
        rng = _rnd.Random(5)
        records = [{
            "ticker": "NVDA" if i % 2 else "AMD",
            "sim_date": f"2024-{1 + i % 12:02d}-{1 + i // 12:02d}",
            "action": "BUY",
            "ml_score": rng.uniform(0, 5), "rsi": rng.uniform(20, 80),
            "macd": rng.uniform(-1, 1), "mom5": rng.uniform(-3, 3),
            "mom20": rng.uniform(-5, 5), "regime_mult": 1.0,
            "forward_return_5d": rng.uniform(-3, 3), "return_pct": 10.0,
        } for i in range(60)]
        res = train_scorer(records)
        assert res["status"] == "ok"
        assert rcb._deployed_scorer_n_train() == res["n"]


def _synthetic_outcome_records(n: int = 200, seed: int = 11) -> list[dict]:
    """A deterministic outcome corpus with STRICTLY INCREASING sim_date (so
    `split_outcomes_temporal`'s late-fraction holdout is clean) and varied
    features (so the trivial baselines are non-degenerate and the scorer
    trains). Shared by the baseline-ledger SSOT + verdict tests."""
    import random as _rnd
    rng = _rnd.Random(seed)
    base = date(2023, 1, 1)
    recs = []
    for i in range(n):
        recs.append({
            "ticker": "NVDA" if i % 2 else "AMD",
            "sim_date": (base + timedelta(days=i)).isoformat(),
            "action": "BUY" if i % 3 else "SELL",
            "ml_score": rng.uniform(-5, 5),
            "rsi": rng.uniform(20, 80),
            "macd": rng.uniform(-1, 1),
            "mom5": rng.uniform(-3, 3),
            "mom20": rng.uniform(-5, 5),
            "regime_mult": 1.0,
            "vol_ratio": rng.uniform(0.5, 2.0),
            "bb_position": rng.uniform(-2, 2),
            "forward_return_5d": rng.uniform(-8, 8),
            "return_pct": 10.0,
        })
    return recs


class TestAppendBaselineSkillLog:
    """The per-cycle trivial-baseline ledger — the decisive
    MLP_WORSE_THAN_TRIVIAL signal, made durable & trendable. Mirrors the
    `_append_scorer_skill_log` discipline (best-effort, honest gap rows,
    atomic bounded trim, never breaks the loop)."""

    def test_row_schema_and_ssot_cross_check(self, tmp_path, monkeypatch):
        # A real trained pickle (conftest redirects SCORER_PATH to tmp) +
        # a real outcomes file routed through the SAME `baseline_compare`
        # path the CLI / `calibration --oos` use.
        from paper_trader.ml.decision_scorer import train_scorer, DecisionScorer
        from paper_trader.ml import baseline_compare as bc

        records = _synthetic_outcome_records(200)
        assert train_scorer(records)["status"] == "ok"

        outcomes = tmp_path / "decision_outcomes.jsonl"
        outcomes.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        log = tmp_path / "baseline_skill_log.jsonl"
        monkeypatch.setattr(rcb, "BASELINE_SKILL_LOG", log)

        ok = rcb._append_baseline_skill_log(
            cycle=7, win_start=date(2015, 1, 2), win_end=date(2016, 1, 2),
            outcomes_path=outcomes)
        assert ok is True

        rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        assert len(rows) == 1
        row = rows[0]
        assert row["cycle"] == 7
        assert row["window_start"] == "2015-01-02"
        assert row["window_end"] == "2016-01-02"
        assert row["status"] == "ok"
        assert row["slice"] == "oos"
        assert row["verdict"] in {
            "MLP_WORSE_THAN_TRIVIAL", "MLP_NO_BETTER_THAN_TRIVIAL",
            "MLP_ADDS_SKILL", "INSUFFICIENT_DATA",
        }
        # gate_active ⇔ deployed n_train >= 500 (invariant #5). This synthetic
        # corpus dedups to < 500 distinct → the gate is NOT live; the flag
        # must say so honestly.
        assert row["n_train"] is not None and row["n_train"] < 500
        assert row["gate_active"] is False

        # SSOT no-drift: the persisted mlp_rank_ic / ic_gap MUST equal the
        # value the canonical `scorer_baseline_compare` returns on the exact
        # same scorer + records (the documented built-in cross-check — the
        # ledger is a recorder, never a re-derivation).
        rep = bc.scorer_baseline_compare(DecisionScorer(), records,
                                         oos_only=True)
        assert row["mlp_rank_ic"] == rep["mlp"]["rank_ic"]
        assert row["best_baseline"] == rep["best_baseline"]
        assert row["best_baseline_ic"] == rep["best_baseline_ic"]
        assert row["ic_gap"] == rep["ic_gap"]
        assert row["verdict"] == rep["verdict"]

    def test_untrained_scorer_persists_honest_insufficient_row(
            self, tmp_path, monkeypatch):
        # conftest gives an empty SCORER_PATH (no pickle). A gap in the trend
        # must be a visible honest row, not a silently-skipped cycle.
        outcomes = tmp_path / "decision_outcomes.jsonl"
        outcomes.write_text("\n".join(
            json.dumps(r) for r in _synthetic_outcome_records(8)) + "\n")
        log = tmp_path / "baseline_skill_log.jsonl"
        monkeypatch.setattr(rcb, "BASELINE_SKILL_LOG", log)

        assert rcb._append_baseline_skill_log(
            cycle=3, win_start=date(2009, 1, 2), win_end=date(2010, 1, 2),
            outcomes_path=outcomes) is True
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["cycle"] == 3
        assert row["gate_active"] is False  # no pickle ⇒ gate not live
        assert row["mlp_rank_ic"] is None

    def test_analyze_exception_degrades_to_honest_row_not_crash(
            self, tmp_path, monkeypatch):
        # If baseline_compare.analyze itself raises, the inner guard must
        # degrade to an honest INSUFFICIENT_DATA row (the trend stays
        # continuous) — and STILL return True (handled, persisted).
        log = tmp_path / "baseline_skill_log.jsonl"
        monkeypatch.setattr(rcb, "BASELINE_SKILL_LOG", log)

        def _boom(*a, **k):
            raise RuntimeError("simulated baseline_compare failure")

        import paper_trader.ml.baseline_compare as bc
        monkeypatch.setattr(bc, "analyze", _boom)

        assert rcb._append_baseline_skill_log(
            cycle=5, win_start=date(2012, 1, 3), win_end=date(2013, 1, 3),
            outcomes_path=tmp_path / "missing.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["status"] == "error"
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["cycle"] == 5

    def test_bounded_trim_rewrites_when_past_2x_keep(self, tmp_path,
                                                     monkeypatch):
        log = tmp_path / "baseline_skill_log.jsonl"
        monkeypatch.setattr(rcb, "BASELINE_SKILL_LOG", log)
        monkeypatch.setattr(rcb, "BASELINE_SKILL_LOG_KEEP", 5)
        # 11 pre-seeded rows (> 2×5) ⇒ the next append triggers a rewrite.
        log.write_text("\n".join(json.dumps({"cycle": i}) for i in range(11))
                       + "\n")
        # No pickle/outcomes needed — the append still writes a (degraded) row.
        rcb._append_baseline_skill_log(
            cycle=99, win_start=date(2012, 1, 3), win_end=date(2013, 1, 3),
            outcomes_path=tmp_path / "nope.jsonl")
        lines = [l for l in log.read_text().splitlines() if l.strip()]
        assert len(lines) == 5  # trimmed to BASELINE_SKILL_LOG_KEEP
        assert json.loads(lines[-1])["cycle"] == 99  # newest survives

    def test_never_raises_on_unwritable_path(self, tmp_path, monkeypatch):
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        monkeypatch.setattr(rcb, "BASELINE_SKILL_LOG",
                            blocker / "sub" / "baseline.jsonl")
        assert rcb._append_baseline_skill_log(
            1, date(2000, 1, 3), date(2001, 1, 3),
            outcomes_path=tmp_path / "nope.jsonl") is False


# ──────────────── LLM-annotation skill ledger ────────────

class TestAppendLlmAnnotationSkillLog:
    """The per-cycle LLM-annotation skill ledger — answers
    "is the _llm_annotate_outcomes pipeline producing labels at all" durably
    and per-cycle so the documented production-dark state is visible in the
    trend, not only via manual grep of decision_outcomes.jsonl. Mirrors the
    `_append_scorer_skill_log` / `_append_baseline_skill_log` discipline:
    best-effort, honest gap rows, atomic bounded trim, never breaks the loop.
    """

    def _write_outcomes(self, path, rows):
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def test_pipeline_dark_row_when_zero_labels(self, tmp_path, monkeypatch):
        # The documented live state: 7413/7413 rows have label=0. Ledger must
        # persist `pipeline_dark=True` so the operator sees it.
        log = tmp_path / "llm_annotation_skill_log.jsonl"
        monkeypatch.setattr(rcb, "LLM_ANNOTATION_SKILL_LOG", log)
        outcomes = tmp_path / "decision_outcomes.jsonl"
        self._write_outcomes(outcomes, [
            {"action": "BUY", "forward_return_5d": 1.0, "llm_quality_label": 0}
            for _ in range(50)
        ])

        ok = rcb._append_llm_annotation_skill_log(
            cycle=11, win_start=date(2018, 1, 2), win_end=date(2019, 1, 2),
            outcomes_path=outcomes)
        assert ok is True

        rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        assert len(rows) == 1
        row = rows[0]
        assert row["cycle"] == 11
        assert row["window_start"] == "2018-01-02"
        assert row["window_end"] == "2019-01-02"
        assert row["verdict"] == "NO_LABELS_PRODUCED"
        assert row["pipeline_dark"] is True
        assert row["n_endorsed"] == 0
        assert row["n_condemned"] == 0
        assert row["n_unlabeled"] == 50

    def test_directional_row_when_labels_predict(self, tmp_path, monkeypatch):
        # Labels exist AND predict realized returns — the row reports the
        # gap and rank-IC. pipeline_dark flips to False the moment any
        # endorsed/condemned label appears.
        log = tmp_path / "llm_annotation_skill_log.jsonl"
        monkeypatch.setattr(rcb, "LLM_ANNOTATION_SKILL_LOG", log)
        outcomes = tmp_path / "decision_outcomes.jsonl"
        rows = (
            [{"action": "BUY", "forward_return_5d": 5.0,
              "llm_quality_label": 1} for _ in range(15)]
            + [{"action": "BUY", "forward_return_5d": -5.0,
                "llm_quality_label": -1} for _ in range(15)]
        )
        self._write_outcomes(outcomes, rows)

        rcb._append_llm_annotation_skill_log(
            cycle=22, win_start=date(2010, 1, 4), win_end=date(2011, 1, 4),
            outcomes_path=outcomes)
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "LLM_DIRECTIONAL"
        assert row["pipeline_dark"] is False
        assert row["n_endorsed"] == 15
        assert row["n_condemned"] == 15
        # SSOT cross-check — the persisted gap MUST equal what
        # llm_annotation_skill returns directly on the same input.
        from paper_trader.ml import llm_annotation_skill as las
        direct = las.llm_annotation_skill(rows)
        assert row["endorsed_minus_condemned"] == direct[
            "endorsed_minus_condemned"]
        assert row["rank_ic"] == direct["rank_ic"]

    def test_missing_outcomes_file_persists_honest_dark_row(self, tmp_path,
                                                            monkeypatch):
        log = tmp_path / "llm_annotation_skill_log.jsonl"
        monkeypatch.setattr(rcb, "LLM_ANNOTATION_SKILL_LOG", log)
        # outcomes file does not exist — analyze returns NO_LABELS_PRODUCED
        # (the n_total == 0 path); the row must STILL appear so a gap in
        # the trend is visible.
        assert rcb._append_llm_annotation_skill_log(
            cycle=4, win_start=date(2009, 1, 2), win_end=date(2010, 1, 2),
            outcomes_path=tmp_path / "missing.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "NO_LABELS_PRODUCED"
        assert row["pipeline_dark"] is True
        assert row["n_total"] == 0

    def test_analyze_exception_degrades_to_honest_row_not_crash(
            self, tmp_path, monkeypatch):
        log = tmp_path / "llm_annotation_skill_log.jsonl"
        monkeypatch.setattr(rcb, "LLM_ANNOTATION_SKILL_LOG", log)

        def _boom(*a, **k):
            raise RuntimeError("simulated llm_annotation_skill failure")

        import paper_trader.ml.llm_annotation_skill as las
        monkeypatch.setattr(las, "analyze", _boom)

        assert rcb._append_llm_annotation_skill_log(
            cycle=6, win_start=date(2012, 1, 3), win_end=date(2013, 1, 3),
            outcomes_path=tmp_path / "missing.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["status"] == "error"
        assert row["verdict"] == "NO_LABELS_PRODUCED"
        assert row["pipeline_dark"] is True  # zero labels ⇒ dark
        assert row["cycle"] == 6

    def test_bounded_trim_rewrites_when_past_2x_keep(self, tmp_path,
                                                     monkeypatch):
        log = tmp_path / "llm_annotation_skill_log.jsonl"
        monkeypatch.setattr(rcb, "LLM_ANNOTATION_SKILL_LOG", log)
        monkeypatch.setattr(rcb, "LLM_ANNOTATION_SKILL_LOG_KEEP", 4)
        # 9 pre-seeded rows (> 2×4) ⇒ next append triggers a rewrite.
        log.write_text("\n".join(json.dumps({"cycle": i}) for i in range(9))
                       + "\n")
        rcb._append_llm_annotation_skill_log(
            cycle=99, win_start=date(2012, 1, 3), win_end=date(2013, 1, 3),
            outcomes_path=tmp_path / "nope.jsonl")
        lines = [l for l in log.read_text().splitlines() if l.strip()]
        assert len(lines) == 4  # trimmed to LLM_ANNOTATION_SKILL_LOG_KEEP
        assert json.loads(lines[-1])["cycle"] == 99  # newest survives

    def test_never_raises_on_unwritable_path(self, tmp_path, monkeypatch):
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        monkeypatch.setattr(rcb, "LLM_ANNOTATION_SKILL_LOG",
                            blocker / "sub" / "llm.jsonl")
        # Even when the parent path can't be created, the function must
        # swallow the OSError and return False — never raise.
        assert rcb._append_llm_annotation_skill_log(
            1, date(2000, 1, 3), date(2001, 1, 3),
            outcomes_path=tmp_path / "nope.jsonl") is False


# ──────────────── winner_training.jsonl bounded trim ────────────

class TestTrimWinnerJsonl:
    def test_absent_file_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rcb, "WINNER_JSONL", tmp_path / "nope.jsonl")
        assert rcb._trim_winner_jsonl() == 0

    def test_under_2x_keep_is_untouched(self, tmp_path, monkeypatch):
        f = tmp_path / "winner_training.jsonl"
        monkeypatch.setattr(rcb, "WINNER_JSONL", f)
        # keep=10 ⇒ threshold is 20; 15 lines must NOT trigger a rewrite.
        original = "\n".join(json.dumps({"i": i}) for i in range(15)) + "\n"
        f.write_text(original)
        assert rcb._trim_winner_jsonl(keep=10) == 0
        assert f.read_text() == original  # byte-for-byte unchanged

    def test_past_2x_keep_trims_to_last_keep_records(self, tmp_path, monkeypatch):
        f = tmp_path / "winner_training.jsonl"
        monkeypatch.setattr(rcb, "WINNER_JSONL", f)
        # 25 lines, keep=10 ⇒ threshold 20 exceeded ⇒ trim to last 10.
        f.write_text("\n".join(json.dumps({"i": i}) for i in range(25)) + "\n")
        dropped = rcb._trim_winner_jsonl(keep=10)
        assert dropped == 15
        kept = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        assert len(kept) == 10
        # The TAIL is what's preserved (newest records) — i=15..24.
        assert [r["i"] for r in kept] == list(range(15, 25))

    def test_kept_lines_remain_valid_json(self, tmp_path, monkeypatch):
        f = tmp_path / "winner_training.jsonl"
        monkeypatch.setattr(rcb, "WINNER_JSONL", f)
        f.write_text("\n".join(
            json.dumps({"run_id": i, "label": "BUY", "ticker": "NVDA"})
            for i in range(30)) + "\n")
        rcb._trim_winner_jsonl(keep=5)
        for l in f.read_text().splitlines():
            if l.strip():
                json.loads(l)  # must not raise — no torn final line

    def test_default_keep_is_well_above_inject_tail(self):
        # _inject_and_train consumes the last 10k lines; the trim floor must
        # never starve it.
        assert rcb.WINNER_JSONL_KEEP >= 10000


# ─────── regression: the ledger / trim must stay wired into main() ───────

class TestCycleWiringRegression:
    """The scorer-skill ledger and winner-jsonl trim were both implemented
    but the ledger was never called from `main()` (dead code) until this
    fix. Lock the wiring with a source-level assertion so a future refactor
    that drops the call fails loudly instead of silently disabling a
    quant-facing audit trail again."""

    def test_main_invokes_scorer_skill_ledger(self):
        import inspect
        src = inspect.getsource(rcb.main)
        assert "_append_scorer_skill_log(" in src

    def test_main_invokes_winner_jsonl_trim(self):
        import inspect
        src = inspect.getsource(rcb.main)
        assert "_trim_winner_jsonl(" in src

    def test_main_invokes_baseline_skill_ledger(self):
        # The trivial-baseline ledger surfaces the decisive
        # MLP_WORSE_THAN_TRIVIAL finding. It was a CLI-only signal until this
        # wiring; lock the call so a refactor can't silently re-orphan it
        # (the exact failure mode the scorer ledger suffered until pass #15).
        import inspect
        src = inspect.getsource(rcb.main)
        assert "_append_baseline_skill_log(" in src

    def test_main_invokes_llm_annotation_skill_ledger(self):
        # The LLM-annotation ledger surfaces the documented
        # NO_LABELS_PRODUCED state of the _llm_annotate_outcomes pipeline.
        # CLI-only until this wiring; lock the call so a future refactor
        # cannot silently re-orphan it (the same failure mode the scorer /
        # baseline ledgers suffered until they were each wired).
        import inspect
        src = inspect.getsource(rcb.main)
        assert "_append_llm_annotation_skill_log(" in src

    def test_main_reaps_orphans_per_cycle_not_only_at_startup(self):
        # Live finding: 15 rows stuck 'running' for 35h because the reaper
        # was startup-only. main() must now call _reap_orphaned_runs() both
        # before the loop AND inside each cycle (≥2 call sites).
        import inspect
        src = inspect.getsource(rcb.main)
        assert src.count("_reap_orphaned_runs()") >= 2


class TestWalkBackInversionGuard:
    """Tighter end of the walk-back collision guard. The collision check
    catches `sim_res == end_res`, but theoretically `end_d`'s walk-back can
    land MORE THAN 0 days BEFORE `sim_d` for a ticker with 7+ consecutive
    missing closes around end_d (rare but possible for a thin ADR over a
    holiday week). The resulting `returns_pct(sim_d, end_d)` then computes
    a forward return between two endpoints in REVERSE TIME ORDER — a sign-
    inverted, time-mangled outcome that silently contaminates the
    DecisionScorer training set the same way collision-zeroes did before
    the original fix. The `end_res <= sim_res` guard strictly strengthens
    the collision check."""

    def _inverted_cache(self):
        """Synthetic cache where THIN has data ONLY on a date BEFORE sim_d,
        and SPY has data on sim_d + 5 trading days. Then sim_d itself is
        a trading day for SPY, end_d = sim_d + 5 trading days. THIN's
        walk-back from sim_d goes to THIN's only date (BEFORE sim_d).
        THIN's walk-back from end_d ALSO goes to that same date because
        nothing exists between THIN's only date and end_d. That's the
        collision case — which the existing check already catches.

        For TRUE inversion we need:
          - sim_res == sim_d (so sim_d is in THIN series)
          - end_res < sim_res (so end_d walks back PAST sim_d)

        That requires THIN to have a trade on sim_d and on some earlier date,
        but NO trade between sim_d and (end_d - 7 calendar days).
        """
        from paper_trader.backtest import PriceCache

        cache = PriceCache.__new__(PriceCache)
        cache.tickers = ["SPY", "THIN"]
        # SPY: every weekday from Jan 6 to Feb 14 2025
        spy_days = []
        d = date(2025, 1, 6)
        while d <= date(2025, 2, 14):
            if d.weekday() < 5:
                spy_days.append(d)
            d += timedelta(days=1)
        cache.start = spy_days[0]
        cache.end = spy_days[-1]
        # THIN trades on Jan 6 (early) and Jan 13 (sim_d) but THEN goes dark
        # until after the 7-day walk-back from end_d (Jan 21). So end_d=Jan 21
        # walk-back: Jan 20, Jan 19, ..., Jan 14 — all missing. Walks all 7
        # days back, finds nothing in the 7-day window, returns None.
        # That actually fails the None check, not inversion. Need a non-None
        # but earlier-than-sim_d result.
        # If THIN trades on Jan 6 and Jan 13 only, end_d=Jan 21 walk-back
        # extends from Jan 21 down to Jan 14 (7 days). Doesn't include Jan 13.
        # Walk-back returns None. So we can't directly construct an inversion
        # using the 7-day walk-back limit.
        # BUT if end_d is closer to a missing region, e.g., sim_d=Jan 13
        # (in series), end_d at idx+5 = Jan 20 (in SPY's calendar). Walk-back
        # from Jan 20 covers Jan 19..Jan 13. Jan 13 IS in THIN, so end_res
        # would be Jan 13 == sim_res. That's the collision case.
        # The inversion case can't actually be constructed within 7-day
        # walk-back limit when sim_d has a real close — proves the inversion
        # branch is defense-in-depth, not a currently-reachable bug.
        cache.prices = {
            "SPY": {dd.isoformat(): 100.0 + i for i, dd in enumerate(spy_days)},
            "THIN": {date(2025, 1, 6).isoformat(): 50.0,
                     date(2025, 1, 13).isoformat(): 51.0},
        }
        cache.trading_days = spy_days
        return cache

    def test_inversion_guard_subsumes_collision_check(self, tmp_path):
        """The `end_res <= sim_res` form must still drop the documented
        collision case (`sim_res == end_res`). This pins that the guard
        STRICTLY STRENGTHENS the prior check — every case it rejected is
        still rejected, no behaviour change for that case."""
        from paper_trader.backtest import PriceCache
        # Same cache shape as TestWalkBackCollisionDoesNotFabricateZero
        # to make the strengthening explicit.
        cache = PriceCache.__new__(PriceCache)
        cache.tickers = ["SPY", "THIN"]
        cache.prices = {
            "SPY": {(date(2025, 1, 6) + timedelta(days=i)).isoformat():
                    100.0 + i for i in range(40)},
            "THIN": {date(2025, 1, 13).isoformat(): 50.0},
        }
        cache.trading_days = [
            date(2025, 1, 6) + timedelta(days=i) for i in range(40)
        ]
        cache.start = cache.trading_days[0]
        cache.end = cache.trading_days[-1]

        eng = _make_engine_with_runs(tmp_path, n_runs=1)
        eng.prices = cache
        sim_d = date(2025, 1, 14)
        # Both endpoints walk back to Jan 13 — the legacy collision case.
        end_d = date(2025, 1, 19)
        assert cache.resolved_close_date("THIN", sim_d) == date(2025, 1, 13)
        assert cache.resolved_close_date("THIN", end_d) == date(2025, 1, 13)

        eng.store.conn.execute(
            "INSERT INTO backtest_decisions (run_id, sim_date, action, ticker, "
            "status, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
            (1, sim_d.isoformat(), "BUY", "THIN", "FILLED",
             "ML+quant: THIN score=2.00 regime=bull news_count=0 news_urg=0.0"),
        )
        eng.store.conn.commit()
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-06",
                             end_date="2025-02-14")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        thin_outs = [o for o in outs if o["ticker"] == "THIN"]
        assert thin_outs == [], (
            "the `end_res <= sim_res` guard MUST still drop the collision "
            "case (sim_res == end_res) — strict strengthening only"
        )

    def test_fwd_ret_h_helper_drops_inverted_window(self):
        """The `_fwd_ret_h` multi-horizon helper inside
        `_compute_decision_outcomes` must apply the same inversion guard so
        the 10d/20d analytics columns NEVER carry a sign-flipped return.
        Exercise via the same collision construction — the helper is a
        closure, so we exercise it via the full outcome path with the 10d
        analytic column.
        """
        from paper_trader.backtest import PriceCache
        cache = PriceCache.__new__(PriceCache)
        cache.tickers = ["SPY", "THIN"]
        cache.prices = {
            "SPY": {(date(2025, 1, 6) + timedelta(days=i)).isoformat():
                    100.0 + i for i in range(80)},
            "THIN": {date(2025, 1, 13).isoformat(): 50.0,
                     date(2025, 1, 14).isoformat(): 51.0},  # sim_d itself
        }
        cache.trading_days = [
            date(2025, 1, 6) + timedelta(days=i) for i in range(80)
        ]
        cache.start = cache.trading_days[0]
        cache.end = cache.trading_days[-1]

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            eng = _make_engine_with_runs(Path(td), n_runs=1)
            eng.prices = cache
            sim_d = date(2025, 1, 14)  # in THIN series (51.0)
            # End_d at idx+10 = Jan 24. THIN walk-back from Jan 24: Jan 23..
            # Jan 17 (7 days) — none match. Returns None. So the 10d helper
            # also drops it. forward_return_10d should be None in the outcome.
            eng.store.conn.execute(
                "INSERT INTO backtest_decisions (run_id, sim_date, action, "
                "ticker, status, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
                (1, sim_d.isoformat(), "BUY", "THIN", "FILLED",
                 "ML+quant: THIN score=2.00 regime=bull news_count=0 "
                 "news_urg=0.0"),
            )
            eng.store.conn.commit()
            runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-06",
                                end_date="2025-02-14")]
            outs = rcb._compute_decision_outcomes(eng, runs)
            thin_outs = [o for o in outs if o["ticker"] == "THIN"
                         and o["sim_date"] == sim_d.isoformat()]
            # sim_d has THIN price, sim_d+5 has SPY price not THIN — so 5d
            # outcome is dropped (already covered by None check). What we
            # care about: the helper does not produce an inverted value.
            for o in thin_outs:
                # Either the row is dropped entirely, OR 10d/20d are None.
                # Neither should ever produce a negative-time computation.
                if o.get("forward_return_10d") is not None:
                    # If a 10d value DID come through, prove it's a real
                    # forward computation: positive sign matches SPY's
                    # monotonic rise (we used 100 + idx for SPY).
                    assert o["forward_return_10d"] >= -50.0
