"""Tests for paper_trader.analytics.hourly_pnl_fingerprint.

Exact-value verdict assertions over synthesised equity-curve rows.

Helper design:
  * All rows for a single hour-bucket chunk live on the SAME day at
    distinct minutes within the same UTC hour. NY EDT = UTC-4, so
    NY-hour H stamps as UTC hour H+4 (or +1 day rollover when H≥20).
  * Each chunk emits 2 + 2*(n_pairs-1) = 2*n_pairs rows. Alternating
    pairs are alive (port/spy moves) and dead (no change) so the
    builder's dead-interval filter passes exactly n_pairs samples.
  * Multi-bucket tests insert ``None`` between chunks — the builder
    treats non-dict rows as a chain break (``prev = None``), so the
    cross-chunk pair is suppressed and each chunk's stats remain
    clean.
"""
from __future__ import annotations

import pytest

from paper_trader.analytics.hourly_pnl_fingerprint import (
    AFTERNOON_EDGE,
    FLAT_CLOCK,
    INSUFFICIENT_DATA,
    MIDDAY_EDGE,
    MORNING_EDGE,
    NO_SPY_DATA,
    OFF_HOURS_EDGE,
    DEFAULT_ALPHA_SPREAD_PP,
    DEFAULT_MIN_BUCKET_ALPHA_SAMPLES,
    DEFAULT_MIN_TOTAL_SAMPLES,
    build_hourly_pnl_fingerprint,
)


def _utc_ts(day: int, hour: int, minute: int = 0, second: int = 0) -> str:
    return f"2026-07-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}+00:00"


def _row(day: int, utc_hour: int, val: float, spy: float | None = 100.0,
         minute: int = 0, second: int = 0) -> dict:
    return {
        "timestamp": _utc_ts(day, utc_hour, minute, second),
        "total_value": float(val),
        "cash": 0.0,
        "sp500_price": (float(spy) if spy is not None else None),
    }


def _curve_for_hour(
    ny_hour: int,
    n_pairs: int,
    *,
    port_delta_pct: float,
    spy_delta_pct: float = 0.0,
    day: int = 15,
) -> list[dict]:
    """All rows live on ``day`` in NY hour ``ny_hour`` (EDT, UTC-4)
    at consecutive minute offsets. Produces ``2 * n_pairs`` rows so
    that exactly ``n_pairs`` alive (prev, curr) pairs land in the
    target bucket (alternating dead pairs are filtered out by the
    builder's dead-interval guard).
    """
    utc_hour = ny_hour + 4  # EDT
    spill_day = day
    if utc_hour >= 24:
        utc_hour -= 24
        spill_day = day + 1
    rows: list[dict] = []
    val = 1000.0
    spy = 100.0
    assert n_pairs * 2 <= 60, "single-hour chunk capped at 60 rows"
    for i in range(n_pairs):
        m = i * 2
        rows.append(_row(spill_day, utc_hour, val, spy=spy, minute=m))
        val = val * (1.0 + port_delta_pct / 100.0)
        spy = spy * (1.0 + spy_delta_pct / 100.0)
        rows.append(_row(spill_day, utc_hour, val, spy=spy, minute=m + 1))
    return rows


# Sentinel that breaks the builder's prev chain — used to concat
# chunks without contaminating either bucket.
BREAK: dict = None  # type: ignore[assignment]


class TestInsufficientData:
    def test_empty_curve(self):
        out = build_hourly_pnl_fingerprint([])
        assert out["verdict"] == INSUFFICIENT_DATA
        assert out["buckets"] == []
        assert out["best_hour"] is None
        assert out["worst_hour"] is None

    def test_single_row(self):
        out = build_hourly_pnl_fingerprint([_row(15, 14, 1000.0)])
        assert out["verdict"] == INSUFFICIENT_DATA

    def test_two_rows_below_floor(self):
        out = build_hourly_pnl_fingerprint([
            _row(15, 13, 1000.0),
            _row(15, 14, 1010.0),
        ])
        assert out["verdict"] == INSUFFICIENT_DATA
        assert out["n_total_samples"] == 1

    def test_non_list_input_safe(self):
        out = build_hourly_pnl_fingerprint("not a list")  # type: ignore
        assert out["verdict"] == INSUFFICIENT_DATA


class TestEdgeVerdicts:
    def test_morning_edge(self):
        rows = (
            _curve_for_hour(10, 30, port_delta_pct=0.6, spy_delta_pct=0.1, day=15)
            + [BREAK]
            + _curve_for_hour(15, 30, port_delta_pct=-0.4, spy_delta_pct=0.1, day=16)
        )
        out = build_hourly_pnl_fingerprint(rows, min_total_samples=50)
        assert out["verdict"] == MORNING_EDGE
        assert out["best_hour"]["hour"] == 10
        assert out["worst_hour"]["hour"] == 15
        assert out["best_hour"]["mean_alpha_pct"] == pytest.approx(0.5, abs=0.001)
        assert out["worst_hour"]["mean_alpha_pct"] == pytest.approx(-0.5, abs=0.001)
        assert out["alpha_spread_pp"] == pytest.approx(1.0, abs=0.001)
        assert "MORNING_EDGE" in out["headline"]
        assert out["n_alpha_samples"] == 60

    def test_midday_edge(self):
        rows = (
            _curve_for_hour(13, 30, port_delta_pct=0.7, spy_delta_pct=0.0, day=15)
            + [BREAK]
            + _curve_for_hour(10, 30, port_delta_pct=-0.3, spy_delta_pct=0.0, day=16)
        )
        out = build_hourly_pnl_fingerprint(rows, min_total_samples=50)
        assert out["verdict"] == MIDDAY_EDGE
        assert out["best_hour"]["hour"] == 13
        assert out["worst_hour"]["hour"] == 10

    def test_afternoon_edge(self):
        rows = (
            _curve_for_hour(15, 30, port_delta_pct=0.8, spy_delta_pct=0.0, day=15)
            + [BREAK]
            + _curve_for_hour(10, 30, port_delta_pct=-0.2, spy_delta_pct=0.0, day=16)
        )
        out = build_hourly_pnl_fingerprint(rows, min_total_samples=50)
        assert out["verdict"] == AFTERNOON_EDGE
        assert out["best_hour"]["hour"] == 15

    def test_off_hours_edge(self):
        rows = (
            _curve_for_hour(20, 30, port_delta_pct=0.7, spy_delta_pct=0.0, day=15)
            + [BREAK]
            + _curve_for_hour(10, 30, port_delta_pct=-0.3, spy_delta_pct=0.0, day=16)
        )
        out = build_hourly_pnl_fingerprint(rows, min_total_samples=50)
        assert out["verdict"] == OFF_HOURS_EDGE
        assert out["best_hour"]["hour"] == 20

    def test_flat_clock(self):
        rows = (
            _curve_for_hour(10, 30, port_delta_pct=0.05, spy_delta_pct=0.05, day=15)
            + [BREAK]
            + _curve_for_hour(15, 30, port_delta_pct=0.05, spy_delta_pct=0.05, day=16)
        )
        out = build_hourly_pnl_fingerprint(rows, min_total_samples=50)
        assert out["verdict"] == FLAT_CLOCK
        assert out["alpha_spread_pp"] is not None
        assert out["alpha_spread_pp"] < DEFAULT_ALPHA_SPREAD_PP


class TestSpyHandling:
    def test_no_spy_anywhere(self):
        rows = []
        val = 1000.0
        for i in range(61):
            rows.append({
                "timestamp": _utc_ts(15, 14, i // 2, (i % 2) * 30),
                "total_value": val,
                "cash": 0.0,
                "sp500_price": None,
            })
            val += 1.0
        out = build_hourly_pnl_fingerprint(rows, min_total_samples=50)
        assert out["verdict"] == NO_SPY_DATA
        assert out["n_total_samples"] >= 50
        assert out["n_alpha_samples"] == 0
        assert len(out["buckets"]) >= 1
        for b in out["buckets"]:
            assert b["mean_alpha_pct"] is None
            assert b["mean_port_delta_pct"] is not None

    def test_partial_spy_flat_when_only_one_alpha_bucket(self):
        # One chunk has SPY (alpha computable), another has no SPY.
        # Only the SPY-anchored bucket contributes to alpha_buckets;
        # spread across a single alpha bucket is 0 → FLAT_CLOCK.
        no_spy_rows = []
        val = 1000.0
        for i in range(30):
            m = i * 2
            no_spy_rows.append({
                "timestamp": _utc_ts(16, 19, m),  # NY 15 EDT
                "total_value": val,
                "cash": 0.0,
                "sp500_price": None,
            })
            val *= 0.999
            no_spy_rows.append({
                "timestamp": _utc_ts(16, 19, m + 1),
                "total_value": val,
                "cash": 0.0,
                "sp500_price": None,
            })
        rows = (
            _curve_for_hour(10, 30, port_delta_pct=0.6, spy_delta_pct=0.1, day=15)
            + [BREAK]
            + no_spy_rows
        )
        out = build_hourly_pnl_fingerprint(rows, min_total_samples=40)
        assert out["verdict"] == FLAT_CLOCK
        h15 = next(b for b in out["buckets"] if b["hour"] == 15)
        assert h15["mean_alpha_pct"] is None
        assert h15["n_alpha_samples"] == 0


class TestRobustness:
    def test_dead_intervals_skipped(self):
        rows = [
            _row(15, 14, 1000.0, spy=100.0, minute=0),
            _row(15, 14, 1000.0, spy=100.0, minute=1),
            _row(15, 14, 1000.0, spy=100.0, minute=2),
        ]
        out = build_hourly_pnl_fingerprint(rows, min_total_samples=1)
        assert out["verdict"] == INSUFFICIENT_DATA
        assert out["n_total_samples"] == 0

    def test_non_dict_rows_dropped(self):
        rows = [
            _row(15, 13, 1000.0),
            "garbage",  # type: ignore
            _row(15, 14, 1010.0),
        ]
        out = build_hourly_pnl_fingerprint(rows, min_total_samples=1)
        assert out["verdict"] == INSUFFICIENT_DATA
        assert out["n_total_samples"] == 0

    def test_bad_timestamp_drops_row(self):
        rows = [
            _row(15, 13, 1000.0, minute=0),
            {"timestamp": "not-a-date", "total_value": 1010.0, "sp500_price": 100.0},
            _row(15, 13, 1020.0, minute=2),
        ]
        out = build_hourly_pnl_fingerprint(rows, min_total_samples=1)
        # Bad row resets prev; only the trailing pair from rows[2]
        # may survive (but it has no prev), so total samples = 0.
        assert out["n_total_samples"] == 0

    def test_non_positive_total_value_skipped(self):
        rows = [
            _row(15, 13, 1000.0, minute=0),
            _row(15, 13, 0.0, minute=1),
            _row(15, 13, 1010.0, minute=2),
        ]
        out = build_hourly_pnl_fingerprint(rows, min_total_samples=1)
        assert out["verdict"] == INSUFFICIENT_DATA

    def test_nan_inf_total_value_rejected(self):
        rows = [
            _row(15, 13, 1000.0, minute=0),
            {"timestamp": _utc_ts(15, 13, 1), "total_value": float("nan"),
             "sp500_price": 100.0},
            _row(15, 13, 1010.0, minute=2),
        ]
        out = build_hourly_pnl_fingerprint(rows, min_total_samples=1)
        assert out["verdict"] == INSUFFICIENT_DATA

    def test_buckets_sorted_ascending(self):
        rows = (
            _curve_for_hour(15, 10, port_delta_pct=0.3, spy_delta_pct=0.0, day=15)
            + [BREAK]
            + _curve_for_hour(10, 10, port_delta_pct=0.1, spy_delta_pct=0.0, day=16)
            + [BREAK]
            + _curve_for_hour(13, 10, port_delta_pct=0.2, spy_delta_pct=0.0, day=17)
        )
        out = build_hourly_pnl_fingerprint(rows, min_total_samples=20)
        hours = [b["hour"] for b in out["buckets"]]
        assert hours == sorted(hours)
        assert set(hours) == {10, 13, 15}


class TestThresholds:
    def test_min_total_samples_override(self):
        # 2 alpha samples per bucket — also lower the per-bucket floor
        # so the verdict computes instead of failing the thin guard.
        rows = (
            _curve_for_hour(10, 2, port_delta_pct=0.5, spy_delta_pct=0.0, day=15)
            + [BREAK]
            + _curve_for_hour(15, 2, port_delta_pct=-0.5, spy_delta_pct=0.0, day=16)
        )
        out = build_hourly_pnl_fingerprint(
            rows, min_total_samples=4, min_bucket_alpha_samples=2,
        )
        assert out["verdict"] in {MORNING_EDGE, AFTERNOON_EDGE,
                                  FLAT_CLOCK, MIDDAY_EDGE, OFF_HOURS_EDGE}
        assert out["n_total_samples"] >= 4

    def test_alpha_spread_pp_override_tightens_to_flat(self):
        rows = (
            _curve_for_hour(10, 30, port_delta_pct=0.6, spy_delta_pct=0.1, day=15)
            + [BREAK]
            + _curve_for_hour(15, 30, port_delta_pct=-0.4, spy_delta_pct=0.1, day=16)
        )
        out = build_hourly_pnl_fingerprint(
            rows, min_total_samples=50, alpha_spread_pp=5.0,
        )
        assert out["verdict"] == FLAT_CLOCK

    def test_thresholds_echoed(self):
        out = build_hourly_pnl_fingerprint(
            [], min_total_samples=99, alpha_spread_pp=2.5,
            min_bucket_alpha_samples=12,
        )
        assert out["thresholds"]["min_total_samples"] == 99
        assert out["thresholds"]["alpha_spread_pp"] == 2.5
        assert out["thresholds"]["min_bucket_alpha_samples"] == 12
        assert out["thresholds"]["tz"] == "America/New_York"

    def test_tz_override_changes_bucket(self):
        rows = _curve_for_hour(10, 30, port_delta_pct=0.6, spy_delta_pct=0.1, day=15)
        out_ny = build_hourly_pnl_fingerprint(rows, min_total_samples=20)
        out_utc = build_hourly_pnl_fingerprint(
            rows, min_total_samples=20, tz_name="UTC",
        )
        ny_hours = {b["hour"] for b in out_ny["buckets"]}
        utc_hours = {b["hour"] for b in out_utc["buckets"]}
        # NY hour 10 (EDT) = UTC 14.
        assert 10 in ny_hours
        assert 14 in utc_hours

    def test_invalid_tz_falls_back_to_ny(self):
        rows = _curve_for_hour(10, 30, port_delta_pct=0.6, spy_delta_pct=0.1, day=15)
        out = build_hourly_pnl_fingerprint(
            rows, min_total_samples=20, tz_name="Not/A_TZ",
        )
        ny_hours = {b["hour"] for b in out["buckets"]}
        assert 10 in ny_hours
        assert out["thresholds"]["tz"] == "America/New_York"


class TestPerBucketFloor:
    def test_thin_bucket_cannot_anchor_edge(self):
        # Hour 10: 30 samples at a modest +0.5pp alpha.
        # Hour 15: only 3 samples at an extreme -10pp alpha (noise).
        # Without the per-bucket floor, hour 10 would win MORNING_EDGE
        # on a 10.5pp spread driven by the 3-sample noise bucket.
        # With the floor (default 8), hour 15 is ineligible — only
        # hour 10 anchors, spread collapses to 0 → FLAT_CLOCK.
        rows = (
            _curve_for_hour(10, 30, port_delta_pct=0.6, spy_delta_pct=0.1, day=15)
            + [BREAK]
            + _curve_for_hour(15, 3, port_delta_pct=-10.0, spy_delta_pct=0.0, day=16)
        )
        out = build_hourly_pnl_fingerprint(rows, min_total_samples=30)
        assert out["verdict"] == FLAT_CLOCK
        # The thin bucket still surfaces in the buckets list — it is
        # only barred from anchoring the verdict.
        h15 = next(b for b in out["buckets"] if b["hour"] == 15)
        assert h15["n_alpha_samples"] == 3
        assert out["best_hour"]["hour"] == 10
        assert out["worst_hour"]["hour"] == 10

    def test_all_buckets_thin_reads_insufficient(self):
        # Two buckets, 3 alpha samples each — total 6 ≥ floor of 4,
        # but neither bucket clears the 8-sample per-bucket floor.
        rows = (
            _curve_for_hour(10, 3, port_delta_pct=0.5, spy_delta_pct=0.0, day=15)
            + [BREAK]
            + _curve_for_hour(15, 3, port_delta_pct=-0.5, spy_delta_pct=0.0, day=16)
        )
        out = build_hourly_pnl_fingerprint(rows, min_total_samples=4)
        assert out["verdict"] == INSUFFICIENT_DATA
        assert "per-bucket floor" in out["headline"]
        # Buckets still surface for inspection.
        assert len(out["buckets"]) == 2

    def test_floor_override_lets_thin_bucket_anchor(self):
        # Same data as test_thin_bucket_cannot_anchor_edge, but lower
        # the floor to 3 — now hour 15 IS eligible and the extreme
        # spread drives a verdict.
        rows = (
            _curve_for_hour(10, 30, port_delta_pct=0.6, spy_delta_pct=0.1, day=15)
            + [BREAK]
            + _curve_for_hour(15, 3, port_delta_pct=-10.0, spy_delta_pct=0.0, day=16)
        )
        out = build_hourly_pnl_fingerprint(
            rows, min_total_samples=30, min_bucket_alpha_samples=3,
        )
        assert out["verdict"] == MORNING_EDGE
        assert out["worst_hour"]["hour"] == 15


class TestEnvelopeStability:
    def test_envelope_keys_always_present(self):
        out = build_hourly_pnl_fingerprint([])
        for key in (
            "verdict", "headline", "buckets", "best_hour", "worst_hour",
            "alpha_spread_pp", "n_total_samples", "n_alpha_samples",
            "thresholds",
        ):
            assert key in out

    def test_defaults_constants(self):
        assert DEFAULT_MIN_TOTAL_SAMPLES == 60
        assert DEFAULT_ALPHA_SPREAD_PP == 0.5
