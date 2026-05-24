"""Tests for ``paper_trader.ml.backfill_intraperiod``.

The module backfills ``forward_intraperiod_min_5d`` /
``forward_intraperiod_max_5d`` into ``decision_outcomes.jsonl`` from
existing per-window price caches. These tests pin:

  * Synthetic monotone-up + monotone-down + V-shape series produce the
    EXACT expected min/max %.
  * Walk-back collision (forward-day resolves to or before sim_d's
    resolved close) returns (None, None) — never fabricates flat 0%.
  * Rows that already have finite intraperiod fields are NEVER
    overwritten.
  * Rows with non-eligible action (HOLD) are passed through verbatim.
  * Atomic rewrite: tmp file pattern + concurrent-write abort.
  * Honest counts in the report dict.

No yfinance, no DB, no network — all in-memory / tmp-path fixtures.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from paper_trader.ml import backfill_intraperiod as bi


# ─────────────────────────── fixtures ───────────────────────────


def _write_price_cache(cache_dir: Path, start: date, end: date,
                       series_by_ticker: dict[str, dict[str, float]]) -> Path:
    """Write a per-window price cache file with the exact schema PriceCache
    writes (``{_meta, TICKER: {iso_date: close, …}}``)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"prices_{start.isoformat()}_{end.isoformat()}.json"
    payload = {"_meta": {"start": start.isoformat(),
                         "end": end.isoformat(),
                         "tickers": list(series_by_ticker.keys()),
                         "saved_at": "2026-01-01T00:00:00+00:00"}}
    payload.update(series_by_ticker)
    path.write_text(json.dumps(payload))
    return path


def _make_outcomes(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a decision_outcomes.jsonl with the given rows."""
    out = tmp_path / "decision_outcomes.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return out


# ─────────────────── unit: walk-back + extremes ────────────────────


def test_walk_back_close_exact_hit():
    series = {"2026-05-01": 100.0, "2026-05-02": 101.0}
    got = bi._walk_back_close(series, date(2026, 5, 1))
    assert got == (date(2026, 5, 1), 100.0)


def test_walk_back_close_falls_back_within_7_days():
    series = {"2026-05-01": 100.0}
    # 2026-05-05 not in series — walks back to 05-01 (4 days back).
    got = bi._walk_back_close(series, date(2026, 5, 5))
    assert got == (date(2026, 5, 1), 100.0)


def test_walk_back_close_returns_none_beyond_7_days():
    series = {"2026-05-01": 100.0}
    # 2026-05-10 is 9 days after — beyond the 7-day cap.
    got = bi._walk_back_close(series, date(2026, 5, 10))
    assert got is None


def test_walk_back_close_empty_series():
    assert bi._walk_back_close({}, date(2026, 5, 1)) is None


def _ten_trading_day_calendar(start: date) -> list[date]:
    """Generate 10 mock trading days (skip weekends)."""
    days: list[date] = []
    d = start
    while len(days) < 10:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def test_compute_extremes_monotone_up_series_max_is_h_close():
    # SPY: 10 trading days, used as the trading-day anchor.
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))  # Monday
    spy_series = {d.isoformat(): 100.0 for d in spy_days}
    # FOO: monotonically increasing — 100, 101, 102, 103, 104, 105, ...
    foo_series = {d.isoformat(): 100.0 + i for i, d in enumerate(spy_days)}
    prices = {"SPY": spy_series, "FOO": foo_series}
    trading_days = bi.build_trading_days(prices)
    # sim_d = day 0 (close=100), forward 1..5 days: closes 101..105
    sim_d = spy_days[0]
    intra_min, intra_max = bi.compute_intraperiod_extremes(
        "FOO", sim_d, foo_series, trading_days, horizon=5)
    # min is at +1d (101 vs 100 → +1.0%), max at +5d (105 vs 100 → +5.0%).
    assert intra_min == pytest.approx(1.0)
    assert intra_max == pytest.approx(5.0)


def test_compute_extremes_v_shape_captures_both_extremes():
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))
    spy_series = {d.isoformat(): 100.0 for d in spy_days}
    # FOO: V-shape — sim_d=100, day+1=90, day+2=80, day+3=110, day+4=130, day+5=120
    closes = [100.0, 90.0, 80.0, 110.0, 130.0, 120.0]
    foo_series = {d.isoformat(): closes[i] for i, d in enumerate(spy_days[:6])}
    prices = {"SPY": spy_series, "FOO": foo_series}
    trading_days = bi.build_trading_days(prices)
    intra_min, intra_max = bi.compute_intraperiod_extremes(
        "FOO", spy_days[0], foo_series, trading_days, horizon=5)
    # min is day+2 (80 vs 100 → -20%), max is day+4 (130 vs 100 → +30%).
    assert intra_min == pytest.approx(-20.0)
    assert intra_max == pytest.approx(30.0)


def test_compute_extremes_collision_returns_none():
    # FOO has only ONE close (sim_d). Forward days all walk back to sim_d
    # and are rejected by the collision guard.
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))
    spy_series = {d.isoformat(): 100.0 for d in spy_days}
    foo_series = {spy_days[0].isoformat(): 100.0}
    prices = {"SPY": spy_series, "FOO": foo_series}
    trading_days = bi.build_trading_days(prices)
    intra_min, intra_max = bi.compute_intraperiod_extremes(
        "FOO", spy_days[0], foo_series, trading_days, horizon=5)
    assert intra_min is None and intra_max is None


def test_compute_extremes_no_sim_day_resolution_returns_none():
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))
    spy_series = {d.isoformat(): 100.0 for d in spy_days}
    # FOO has data far in the future — nothing within walk-back of sim_d.
    foo_series = {(spy_days[0] + timedelta(days=30)).isoformat(): 100.0}
    prices = {"SPY": spy_series, "FOO": foo_series}
    trading_days = bi.build_trading_days(prices)
    intra_min, intra_max = bi.compute_intraperiod_extremes(
        "FOO", spy_days[0], foo_series, trading_days, horizon=5)
    assert intra_min is None and intra_max is None


def test_compute_extremes_partial_coverage_still_works():
    # FOO: sim_d + only days 1, 3, 5 present (no 2 or 4).
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))
    spy_series = {d.isoformat(): 100.0 for d in spy_days}
    foo_series = {
        spy_days[0].isoformat(): 100.0,
        spy_days[1].isoformat(): 105.0,
        spy_days[3].isoformat(): 95.0,
        spy_days[5].isoformat(): 110.0,
    }
    prices = {"SPY": spy_series, "FOO": foo_series}
    trading_days = bi.build_trading_days(prices)
    intra_min, intra_max = bi.compute_intraperiod_extremes(
        "FOO", spy_days[0], foo_series, trading_days, horizon=5)
    # min from day 3 (95 → -5%), max from day 5 (110 → +10%).
    # Note: day 5 IS at the horizon edge — `_fwd_intraperiod_extremes`
    # walks k in 1..h, so k=5 IS included.
    assert intra_min == pytest.approx(-5.0)
    assert intra_max == pytest.approx(10.0)


# ─────────────────── unit: load + build helpers ────────────────────


def test_load_price_caches_unions_multiple_files(tmp_path: Path):
    cache_dir = tmp_path / "backtest_cache"
    spy_days_a = _ten_trading_day_calendar(date(2026, 1, 5))
    spy_days_b = _ten_trading_day_calendar(date(2026, 2, 2))
    _write_price_cache(cache_dir, spy_days_a[0], spy_days_a[-1], {
        "SPY": {d.isoformat(): 100.0 for d in spy_days_a},
        "FOO": {d.isoformat(): 50.0 for d in spy_days_a},
    })
    _write_price_cache(cache_dir, spy_days_b[0], spy_days_b[-1], {
        "SPY": {d.isoformat(): 200.0 for d in spy_days_b},
        "BAR": {d.isoformat(): 75.0 for d in spy_days_b},
    })
    prices = bi.load_price_caches(cache_dir)
    assert set(prices.keys()) == {"SPY", "FOO", "BAR"}
    # SPY series unioned across both windows.
    assert len(prices["SPY"]) == 20
    # FOO only in window A.
    assert len(prices["FOO"]) == 10
    # BAR only in window B.
    assert len(prices["BAR"]) == 10


def test_load_price_caches_ignores_meta_key(tmp_path: Path):
    cache_dir = tmp_path / "backtest_cache"
    _write_price_cache(cache_dir, date(2026, 1, 1), date(2026, 1, 5),
                       {"SPY": {"2026-01-01": 100.0}})
    prices = bi.load_price_caches(cache_dir)
    assert "_meta" not in prices
    assert "SPY" in prices


def test_load_price_caches_missing_dir_returns_empty(tmp_path: Path):
    assert bi.load_price_caches(tmp_path / "nonexistent") == {}


def test_load_price_caches_skips_corrupt_file(tmp_path: Path):
    cache_dir = tmp_path / "backtest_cache"
    cache_dir.mkdir()
    (cache_dir / "prices_2026-01-01_2026-01-05.json").write_text("not json{{")
    # Add a valid one alongside.
    _write_price_cache(cache_dir, date(2026, 2, 1), date(2026, 2, 5),
                       {"SPY": {"2026-02-01": 100.0}})
    prices = bi.load_price_caches(cache_dir)
    assert "SPY" in prices  # corrupt one skipped, valid one loaded


def test_build_trading_days_falls_back_to_densest_series():
    # No SPY — densest series (FOO) becomes the calendar.
    prices = {
        "FOO": {f"2026-05-0{i}": 100.0 for i in range(1, 8)},
        "BAR": {"2026-05-01": 100.0},  # sparse
    }
    days = bi.build_trading_days(prices)
    assert len(days) == 7


def test_build_trading_days_empty_when_no_prices():
    assert bi.build_trading_days({}) == []


# ─────────────────── integration: analyze + apply ───────────────────


def test_analyze_reports_nothing_to_backfill_when_already_has(tmp_path: Path):
    cache_dir = tmp_path / "backtest_cache"
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))
    _write_price_cache(cache_dir, spy_days[0], spy_days[-1], {
        "SPY": {d.isoformat(): 100.0 for d in spy_days},
        "FOO": {d.isoformat(): 100.0 + i for i, d in enumerate(spy_days)},
    })
    outcomes = _make_outcomes(tmp_path, [{
        "run_id": 1, "sim_date": spy_days[0].isoformat(),
        "ticker": "FOO", "action": "BUY",
        "forward_return_5d": 5.0,
        "forward_intraperiod_min_5d": 1.0,
        "forward_intraperiod_max_5d": 5.0,
    }])
    rep = bi.analyze(outcomes, cache_dir)
    assert rep["status"] == "ok"
    assert rep["verdict"] == "NOTHING_TO_BACKFILL"
    assert rep["rows_eligible"] == 1
    assert rep["rows_already_has"] == 1
    assert rep["rows_backfillable"] == 0


def test_analyze_reports_ready_to_backfill_when_eligible(tmp_path: Path):
    cache_dir = tmp_path / "backtest_cache"
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))
    _write_price_cache(cache_dir, spy_days[0], spy_days[-1], {
        "SPY": {d.isoformat(): 100.0 for d in spy_days},
        "FOO": {d.isoformat(): 100.0 + i for i, d in enumerate(spy_days)},
    })
    outcomes = _make_outcomes(tmp_path, [{
        "run_id": 1, "sim_date": spy_days[0].isoformat(),
        "ticker": "FOO", "action": "BUY",
        "forward_return_5d": 5.0,
        # No intraperiod fields.
    }])
    rep = bi.analyze(outcomes, cache_dir)
    assert rep["verdict"] == "READY_TO_BACKFILL"
    assert rep["rows_backfillable"] == 1


def test_analyze_reports_no_price_cache_when_dir_empty(tmp_path: Path):
    outcomes = _make_outcomes(tmp_path, [{
        "run_id": 1, "sim_date": "2026-05-04",
        "ticker": "FOO", "action": "BUY",
        "forward_return_5d": 5.0,
    }])
    rep = bi.analyze(outcomes, tmp_path / "empty_cache_dir")
    assert rep["verdict"] == "NO_PRICE_CACHE"
    assert rep["rows_total"] == 1


def test_apply_backfill_writes_correct_intraperiod_values(tmp_path: Path):
    cache_dir = tmp_path / "backtest_cache"
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))
    # FOO: V-shape with known extremes.
    closes = [100.0, 90.0, 80.0, 110.0, 130.0, 120.0]
    foo_series = {d.isoformat(): closes[i] for i, d in enumerate(spy_days[:6])}
    _write_price_cache(cache_dir, spy_days[0], spy_days[-1], {
        "SPY": {d.isoformat(): 100.0 for d in spy_days},
        "FOO": foo_series,
    })
    outcomes = _make_outcomes(tmp_path, [{
        "run_id": 1, "sim_date": spy_days[0].isoformat(),
        "ticker": "FOO", "action": "BUY", "forward_return_5d": 20.0,
    }])
    rep = bi.apply_backfill(outcomes, cache_dir)
    assert rep["status"] == "ok"
    assert rep["applied"] is True
    assert rep["rows_backfilled"] == 1
    # Re-read the file and verify the row was updated with the EXACT V-shape extremes.
    rebuilt = json.loads(outcomes.read_text().strip())
    assert rebuilt["forward_intraperiod_min_5d"] == pytest.approx(-20.0)
    assert rebuilt["forward_intraperiod_max_5d"] == pytest.approx(30.0)
    # forward_return_5d preserved exactly.
    assert rebuilt["forward_return_5d"] == 20.0


def test_apply_backfill_never_overwrites_existing_values(tmp_path: Path):
    cache_dir = tmp_path / "backtest_cache"
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))
    foo_series = {d.isoformat(): 100.0 + i for i, d in enumerate(spy_days)}
    _write_price_cache(cache_dir, spy_days[0], spy_days[-1], {
        "SPY": {d.isoformat(): 100.0 for d in spy_days},
        "FOO": foo_series,
    })
    # Pre-populated SENTINEL values that DIFFER from what backfill would compute.
    outcomes = _make_outcomes(tmp_path, [{
        "run_id": 1, "sim_date": spy_days[0].isoformat(),
        "ticker": "FOO", "action": "BUY", "forward_return_5d": 5.0,
        "forward_intraperiod_min_5d": 999.999,
        "forward_intraperiod_max_5d": -999.999,
    }])
    rep = bi.apply_backfill(outcomes, cache_dir)
    assert rep["rows_backfilled"] == 0
    assert rep["rows_already_has"] == 1
    # Existing values still present and untouched.
    rebuilt = json.loads(outcomes.read_text().strip())
    assert rebuilt["forward_intraperiod_min_5d"] == 999.999
    assert rebuilt["forward_intraperiod_max_5d"] == -999.999


def test_apply_backfill_passes_through_hold_rows(tmp_path: Path):
    cache_dir = tmp_path / "backtest_cache"
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))
    _write_price_cache(cache_dir, spy_days[0], spy_days[-1], {
        "SPY": {d.isoformat(): 100.0 for d in spy_days},
        "FOO": {d.isoformat(): 100.0 + i for i, d in enumerate(spy_days)},
    })
    outcomes = _make_outcomes(tmp_path, [{
        "run_id": 1, "sim_date": spy_days[0].isoformat(),
        "ticker": "FOO", "action": "HOLD", "forward_return_5d": 5.0,
    }])
    rep = bi.apply_backfill(outcomes, cache_dir)
    assert rep["rows_backfilled"] == 0
    rebuilt = json.loads(outcomes.read_text().strip())
    assert "forward_intraperiod_min_5d" not in rebuilt
    assert "forward_intraperiod_max_5d" not in rebuilt


def test_apply_backfill_no_writes_when_nothing_eligible(tmp_path: Path):
    """If the rewrite produces no backfilled rows, no rewrite happens —
    keeps the file's mtime stable so consumers polling on mtime don't
    re-process a no-op."""
    cache_dir = tmp_path / "backtest_cache"
    outcomes = _make_outcomes(tmp_path, [{
        "run_id": 1, "sim_date": "2026-05-04",
        "ticker": "FOO", "action": "BUY", "forward_return_5d": 5.0,
    }])
    pre_mtime = outcomes.stat().st_mtime_ns
    rep = bi.apply_backfill(outcomes, cache_dir)
    assert rep["applied"] is False
    assert outcomes.stat().st_mtime_ns == pre_mtime


def test_apply_backfill_aborts_on_concurrent_write(tmp_path: Path,
                                                    monkeypatch):
    cache_dir = tmp_path / "backtest_cache"
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))
    _write_price_cache(cache_dir, spy_days[0], spy_days[-1], {
        "SPY": {d.isoformat(): 100.0 for d in spy_days},
        "FOO": {d.isoformat(): 100.0 + i for i, d in enumerate(spy_days)},
    })
    outcomes = _make_outcomes(tmp_path, [{
        "run_id": 1, "sim_date": spy_days[0].isoformat(),
        "ticker": "FOO", "action": "BUY", "forward_return_5d": 5.0,
    }])

    # Monkeypatch `compute_intraperiod_extremes` so that DURING the read
    # → write window, an external writer touches the file. This simulates
    # a re-awoken continuous loop appending a new row.
    original_compute = bi.compute_intraperiod_extremes

    def race_then_compute(*a, **kw):
        # Append a fake row + retouch mtime.
        with outcomes.open("a") as fh:
            fh.write(json.dumps({"action": "BUY", "ticker": "BAR",
                                 "sim_date": "2026-05-04",
                                 "forward_return_5d": 1.0}) + "\n")
        return original_compute(*a, **kw)

    monkeypatch.setattr(bi, "compute_intraperiod_extremes", race_then_compute)
    rep = bi.apply_backfill(outcomes, cache_dir)
    assert rep["status"] == "aborted_concurrent_write"
    assert rep["applied"] is False


def test_apply_backfill_atomic_no_tmp_left_behind(tmp_path: Path):
    cache_dir = tmp_path / "backtest_cache"
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))
    _write_price_cache(cache_dir, spy_days[0], spy_days[-1], {
        "SPY": {d.isoformat(): 100.0 for d in spy_days},
        "FOO": {d.isoformat(): 100.0 + i for i, d in enumerate(spy_days)},
    })
    outcomes = _make_outcomes(tmp_path, [{
        "run_id": 1, "sim_date": spy_days[0].isoformat(),
        "ticker": "FOO", "action": "BUY", "forward_return_5d": 5.0,
    }])
    rep = bi.apply_backfill(outcomes, cache_dir)
    assert rep["applied"] is True
    # tmp file MUST NOT remain after a successful rewrite (Path.replace
    # is atomic move, not copy).
    tmp_path_check = outcomes.with_suffix(".jsonl.backfill_tmp")
    assert not tmp_path_check.exists()


def test_apply_backfill_preserves_existing_fields_unchanged(tmp_path: Path):
    """Every existing field in a backfilled row must survive verbatim —
    backfill is additive only, never destructive."""
    cache_dir = tmp_path / "backtest_cache"
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))
    _write_price_cache(cache_dir, spy_days[0], spy_days[-1], {
        "SPY": {d.isoformat(): 100.0 for d in spy_days},
        "FOO": {d.isoformat(): 100.0 + i for i, d in enumerate(spy_days)},
    })
    outcomes = _make_outcomes(tmp_path, [{
        "run_id": 99001, "sim_date": spy_days[0].isoformat(),
        "ticker": "FOO", "action": "BUY", "forward_return_5d": 5.0,
        "rsi": 65.3, "macd": 0.12, "regime_mult": 1.0, "persona": "Momentum",
        "ml_score": 3.14, "wk52_pos": 0.42, "conviction_pct": 0.25,
        "llm_quality_label": 0, "return_pct": 7.5,
    }])
    bi.apply_backfill(outcomes, cache_dir)
    rebuilt = json.loads(outcomes.read_text().strip())
    # Pre-existing fields preserved with exact values.
    for field, expected in {
        "run_id": 99001, "ticker": "FOO", "action": "BUY",
        "forward_return_5d": 5.0, "rsi": 65.3, "macd": 0.12,
        "regime_mult": 1.0, "persona": "Momentum", "ml_score": 3.14,
        "wk52_pos": 0.42, "conviction_pct": 0.25, "llm_quality_label": 0,
        "return_pct": 7.5,
    }.items():
        assert rebuilt[field] == expected, f"{field} mutated by backfill"
    # New fields added.
    assert "forward_intraperiod_min_5d" in rebuilt
    assert "forward_intraperiod_max_5d" in rebuilt


def test_apply_backfill_passes_through_unparseable_lines(tmp_path: Path):
    cache_dir = tmp_path / "backtest_cache"
    outcomes = tmp_path / "decision_outcomes.jsonl"
    outcomes.write_text("not json{\n{}garbage\n")
    rep = bi.apply_backfill(outcomes, cache_dir)
    assert rep["rows_unmodified_passthrough"] >= 1
    # File must still contain the unparseable lines after the (no-op) read.
    raw = outcomes.read_text()
    assert "not json" in raw


def test_apply_backfill_outcomes_file_missing(tmp_path: Path):
    rep = bi.apply_backfill(tmp_path / "nonexistent.jsonl",
                            tmp_path / "cache")
    assert rep["status"] == "error"
    assert rep["applied"] is False


def test_analyze_outcomes_file_missing(tmp_path: Path):
    rep = bi.analyze(tmp_path / "nonexistent.jsonl", tmp_path / "cache")
    assert rep["status"] == "error"
    assert rep["verdict"] == "OUTCOMES_FILE_MISSING"


def test_apply_backfill_eligible_but_no_cache_for_ticker(tmp_path: Path):
    """Cache exists but doesn't carry FOO — row counts as
    no_price_cache, not as walk_back_collision."""
    cache_dir = tmp_path / "backtest_cache"
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))
    _write_price_cache(cache_dir, spy_days[0], spy_days[-1], {
        "SPY": {d.isoformat(): 100.0 for d in spy_days},
        # Note: no FOO series.
    })
    outcomes = _make_outcomes(tmp_path, [{
        "run_id": 1, "sim_date": spy_days[0].isoformat(),
        "ticker": "FOO", "action": "BUY", "forward_return_5d": 5.0,
    }])
    rep = bi.apply_backfill(outcomes, cache_dir)
    assert rep["rows_no_price_cache"] == 1
    assert rep["rows_walk_back_collision"] == 0
    assert rep["rows_backfilled"] == 0


def test_apply_backfill_collision_counts_separately(tmp_path: Path):
    """A ticker with ONLY sim_d's close (no forward days) is a
    walk_back_collision, NOT a no_price_cache miss."""
    cache_dir = tmp_path / "backtest_cache"
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))
    _write_price_cache(cache_dir, spy_days[0], spy_days[-1], {
        "SPY": {d.isoformat(): 100.0 for d in spy_days},
        # FOO has nearby data but every forward day walks back to sim_d.
        # Put a close very close to sim_d so the cheap window pre-check
        # passes, but then the actual compute hits collisions.
        "FOO": {spy_days[0].isoformat(): 100.0,
                (spy_days[0] - timedelta(days=1)).isoformat(): 99.0},
    })
    outcomes = _make_outcomes(tmp_path, [{
        "run_id": 1, "sim_date": spy_days[0].isoformat(),
        "ticker": "FOO", "action": "BUY", "forward_return_5d": 5.0,
    }])
    rep = bi.apply_backfill(outcomes, cache_dir)
    assert rep["rows_walk_back_collision"] == 1
    assert rep["rows_no_price_cache"] == 0


def test_apply_backfill_sell_action_also_eligible(tmp_path: Path):
    """SELL rows are also eligible for backfill — the intraperiod
    fields are action-agnostic % moves."""
    cache_dir = tmp_path / "backtest_cache"
    spy_days = _ten_trading_day_calendar(date(2026, 5, 4))
    _write_price_cache(cache_dir, spy_days[0], spy_days[-1], {
        "SPY": {d.isoformat(): 100.0 for d in spy_days},
        "FOO": {d.isoformat(): 100.0 + i for i, d in enumerate(spy_days)},
    })
    outcomes = _make_outcomes(tmp_path, [{
        "run_id": 1, "sim_date": spy_days[0].isoformat(),
        "ticker": "FOO", "action": "SELL", "forward_return_5d": -3.0,
    }])
    rep = bi.apply_backfill(outcomes, cache_dir)
    assert rep["rows_backfilled"] == 1
    rebuilt = json.loads(outcomes.read_text().strip())
    assert "forward_intraperiod_min_5d" in rebuilt
    assert "forward_intraperiod_max_5d" in rebuilt


def test_is_finite_rejects_bool_none_nan_inf():
    assert bi._is_finite(1.0) is True
    assert bi._is_finite(-50.0) is True
    assert bi._is_finite(0) is True
    assert bi._is_finite(None) is False
    assert bi._is_finite(True) is False  # bool excluded — `isinstance(True, int)` would slip through
    assert bi._is_finite(False) is False
    assert bi._is_finite(float("nan")) is False
    assert bi._is_finite(float("inf")) is False
    assert bi._is_finite(float("-inf")) is False
    assert bi._is_finite("3.14") is True  # parseable string accepted (mirrors `_to_float`)
    assert bi._is_finite("abc") is False
