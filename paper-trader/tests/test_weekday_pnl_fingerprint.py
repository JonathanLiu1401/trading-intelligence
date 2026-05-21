"""Tests for paper_trader.analytics.weekday_pnl_fingerprint.

Date anchors (2026, EDT — UTC-4):

  * 2026-07-13 Mon — weekday 0
  * 2026-07-14 Tue — weekday 1
  * 2026-07-15 Wed — weekday 2
  * 2026-07-16 Thu — weekday 3
  * 2026-07-17 Fri — weekday 4
  * 2026-07-18 Sat — weekday 5
  * 2026-07-19 Sun — weekday 6

Helper packs all rows for a single weekday-bucket chunk into the same
UTC hour at distinct minute offsets. Multi-bucket tests insert a
``BREAK`` (``None``) between chunks to reset the builder's ``prev``
pointer so cross-chunk pairs cannot contaminate either bucket's stats.
"""
from __future__ import annotations

import pytest

from paper_trader.analytics.weekday_pnl_fingerprint import (
    DEFAULT_ALPHA_SPREAD_PP,
    DEFAULT_MIN_BUCKET_ALPHA_SAMPLES,
    DEFAULT_MIN_TOTAL_SAMPLES,
    FLAT_WEEK,
    INSUFFICIENT_DATA,
    NO_SPY_DATA,
    WEEKDAY_EDGE,
    WEEKEND_EDGE,
    build_weekday_pnl_fingerprint,
)


def _ts(day: int, utc_hour: int = 18, minute: int = 0, second: int = 0) -> str:
    return f"2026-07-{day:02d}T{utc_hour:02d}:{minute:02d}:{second:02d}+00:00"


def _row(day: int, val: float, spy: float | None = 100.0,
         utc_hour: int = 18, minute: int = 0, second: int = 0) -> dict:
    return {
        "timestamp": _ts(day, utc_hour, minute, second),
        "total_value": float(val),
        "cash": 0.0,
        "sp500_price": (float(spy) if spy is not None else None),
    }


def _curve_for_weekday(
    day: int,
    n_pairs: int,
    *,
    port_delta_pct: float,
    spy_delta_pct: float = 0.0,
) -> list[dict]:
    """Pack ``2 * n_pairs`` rows on a single day, all at UTC 18:00
    (NY 14:00 EDT) with consecutive minute offsets. Yields ``n_pairs``
    alive pairs (alternating dead pairs filtered by the builder's
    dead-interval guard).
    """
    rows: list[dict] = []
    val = 1000.0
    spy = 100.0
    assert n_pairs * 2 <= 60, "single-day chunk capped at 60 rows"
    for i in range(n_pairs):
        m = i * 2
        rows.append(_row(day, val, spy=spy, minute=m))
        val = val * (1.0 + port_delta_pct / 100.0)
        spy = spy * (1.0 + spy_delta_pct / 100.0)
        rows.append(_row(day, val, spy=spy, minute=m + 1))
    return rows


BREAK: dict = None  # type: ignore[assignment]


class TestInsufficientData:
    def test_empty(self):
        out = build_weekday_pnl_fingerprint([])
        assert out["verdict"] == INSUFFICIENT_DATA
        assert out["buckets"] == []
        assert out["best_weekday"] is None

    def test_single_row(self):
        out = build_weekday_pnl_fingerprint([_row(15, 1000.0)])
        assert out["verdict"] == INSUFFICIENT_DATA

    def test_below_floor(self):
        out = build_weekday_pnl_fingerprint([
            _row(15, 1000.0, minute=0),
            _row(15, 1010.0, minute=1),
        ])
        assert out["verdict"] == INSUFFICIENT_DATA
        assert out["n_total_samples"] == 1


class TestEdgeVerdicts:
    def test_weekday_edge_best_tuesday(self):
        rows = (
            _curve_for_weekday(14, 30, port_delta_pct=0.6, spy_delta_pct=0.1)
            + [BREAK]
            + _curve_for_weekday(17, 30, port_delta_pct=-0.4, spy_delta_pct=0.1)
        )
        out = build_weekday_pnl_fingerprint(rows, min_total_samples=50)
        assert out["verdict"] == WEEKDAY_EDGE
        assert out["best_weekday"]["weekday"] == 1
        assert out["best_weekday"]["weekday_name"] == "Tue"
        assert out["worst_weekday"]["weekday"] == 4
        assert out["worst_weekday"]["weekday_name"] == "Fri"
        assert out["best_weekday"]["mean_alpha_pct"] == pytest.approx(0.5, abs=0.001)
        assert out["worst_weekday"]["mean_alpha_pct"] == pytest.approx(-0.5, abs=0.001)
        assert out["alpha_spread_pp"] == pytest.approx(1.0, abs=0.001)
        assert out["n_alpha_samples"] == 60

    def test_weekend_edge_best_saturday(self):
        rows = (
            _curve_for_weekday(18, 30, port_delta_pct=0.7, spy_delta_pct=0.0)  # Sat
            + [BREAK]
            + _curve_for_weekday(15, 30, port_delta_pct=-0.3, spy_delta_pct=0.0)  # Wed
        )
        out = build_weekday_pnl_fingerprint(rows, min_total_samples=50)
        assert out["verdict"] == WEEKEND_EDGE
        assert out["best_weekday"]["weekday"] == 5
        assert out["best_weekday"]["weekday_name"] == "Sat"

    def test_flat_week(self):
        rows = (
            _curve_for_weekday(14, 30, port_delta_pct=0.05, spy_delta_pct=0.05)
            + [BREAK]
            + _curve_for_weekday(15, 30, port_delta_pct=0.05, spy_delta_pct=0.05)
        )
        out = build_weekday_pnl_fingerprint(rows, min_total_samples=50)
        assert out["verdict"] == FLAT_WEEK
        assert out["alpha_spread_pp"] < DEFAULT_ALPHA_SPREAD_PP


class TestSpyHandling:
    def test_no_spy_anywhere(self):
        rows = []
        val = 1000.0
        for i in range(61):
            rows.append({
                "timestamp": _ts(15, 18, i // 2, (i % 2) * 30),
                "total_value": val,
                "cash": 0.0,
                "sp500_price": None,
            })
            val += 1.0
        out = build_weekday_pnl_fingerprint(rows, min_total_samples=50)
        assert out["verdict"] == NO_SPY_DATA
        assert out["n_total_samples"] >= 50
        assert out["n_alpha_samples"] == 0
        for b in out["buckets"]:
            assert b["mean_alpha_pct"] is None


class TestRobustness:
    def test_dead_intervals_skipped(self):
        rows = [
            _row(15, 1000.0, spy=100.0, minute=0),
            _row(15, 1000.0, spy=100.0, minute=1),
            _row(15, 1000.0, spy=100.0, minute=2),
        ]
        out = build_weekday_pnl_fingerprint(rows, min_total_samples=1)
        assert out["verdict"] == INSUFFICIENT_DATA
        assert out["n_total_samples"] == 0

    def test_non_dict_row_breaks_chain(self):
        rows = [
            _row(15, 1000.0, minute=0),
            "garbage",  # type: ignore
            _row(15, 1010.0, minute=2),
        ]
        out = build_weekday_pnl_fingerprint(rows, min_total_samples=1)
        assert out["verdict"] == INSUFFICIENT_DATA

    def test_buckets_sorted_by_weekday_ascending(self):
        rows = (
            _curve_for_weekday(17, 10, port_delta_pct=0.3)  # Fri (4)
            + [BREAK]
            + _curve_for_weekday(13, 10, port_delta_pct=0.2)  # Mon (0)
            + [BREAK]
            + _curve_for_weekday(15, 10, port_delta_pct=0.4)  # Wed (2)
        )
        out = build_weekday_pnl_fingerprint(rows, min_total_samples=20)
        wds = [b["weekday"] for b in out["buckets"]]
        assert wds == sorted(wds)
        assert set(wds) == {0, 2, 4}


class TestThresholds:
    def test_min_total_samples_override(self):
        # 2 alpha samples per bucket — lower the per-bucket floor too.
        rows = (
            _curve_for_weekday(14, 2, port_delta_pct=0.5)
            + [BREAK]
            + _curve_for_weekday(17, 2, port_delta_pct=-0.5)
        )
        out = build_weekday_pnl_fingerprint(
            rows, min_total_samples=4, min_bucket_alpha_samples=2,
        )
        assert out["verdict"] != INSUFFICIENT_DATA
        assert out["n_total_samples"] >= 4

    def test_alpha_spread_pp_override_tightens(self):
        rows = (
            _curve_for_weekday(14, 30, port_delta_pct=0.6, spy_delta_pct=0.1)
            + [BREAK]
            + _curve_for_weekday(17, 30, port_delta_pct=-0.4, spy_delta_pct=0.1)
        )
        out = build_weekday_pnl_fingerprint(
            rows, min_total_samples=50, alpha_spread_pp=5.0,
        )
        assert out["verdict"] == FLAT_WEEK

    def test_thresholds_echoed(self):
        out = build_weekday_pnl_fingerprint(
            [], min_total_samples=99, alpha_spread_pp=2.5,
            min_bucket_alpha_samples=12,
        )
        assert out["thresholds"]["min_total_samples"] == 99
        assert out["thresholds"]["alpha_spread_pp"] == 2.5
        assert out["thresholds"]["min_bucket_alpha_samples"] == 12
        assert out["thresholds"]["tz"] == "America/New_York"


class TestPerBucketFloor:
    def test_thin_bucket_cannot_anchor_edge(self):
        # Tue: 30 samples at +0.5pp alpha. Fri: 3 samples at -10pp.
        # The default 8-sample per-bucket floor bars Fri from
        # anchoring — only Tue is eligible, spread collapses → FLAT.
        rows = (
            _curve_for_weekday(14, 30, port_delta_pct=0.6, spy_delta_pct=0.1)
            + [BREAK]
            + _curve_for_weekday(17, 3, port_delta_pct=-10.0, spy_delta_pct=0.0)
        )
        out = build_weekday_pnl_fingerprint(rows, min_total_samples=30)
        assert out["verdict"] == FLAT_WEEK
        fri = next(b for b in out["buckets"] if b["weekday"] == 4)
        assert fri["n_alpha_samples"] == 3
        assert out["best_weekday"]["weekday"] == 1

    def test_all_buckets_thin_reads_insufficient(self):
        rows = (
            _curve_for_weekday(14, 3, port_delta_pct=0.5)
            + [BREAK]
            + _curve_for_weekday(17, 3, port_delta_pct=-0.5)
        )
        out = build_weekday_pnl_fingerprint(rows, min_total_samples=4)
        assert out["verdict"] == INSUFFICIENT_DATA
        assert "per-bucket floor" in out["headline"]
        assert len(out["buckets"]) == 2

    def test_floor_override_lets_thin_bucket_anchor(self):
        rows = (
            _curve_for_weekday(14, 30, port_delta_pct=0.6, spy_delta_pct=0.1)
            + [BREAK]
            + _curve_for_weekday(17, 3, port_delta_pct=-10.0, spy_delta_pct=0.0)
        )
        out = build_weekday_pnl_fingerprint(
            rows, min_total_samples=30, min_bucket_alpha_samples=3,
        )
        assert out["verdict"] == WEEKDAY_EDGE
        assert out["worst_weekday"]["weekday"] == 4


class TestEnvelopeStability:
    def test_envelope_keys_always_present(self):
        out = build_weekday_pnl_fingerprint([])
        for key in (
            "verdict", "headline", "buckets", "best_weekday",
            "worst_weekday", "alpha_spread_pp", "n_total_samples",
            "n_alpha_samples", "thresholds",
        ):
            assert key in out

    def test_defaults_constants(self):
        assert DEFAULT_MIN_TOTAL_SAMPLES == 60
        assert DEFAULT_ALPHA_SPREAD_PP == 0.5
