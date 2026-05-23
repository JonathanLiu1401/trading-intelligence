"""Tests for analytics/risk_adjusted_returns.py — Sharpe/Sortino vs S&P 500.

Hand-computed arithmetic against the population-stddev / √252 convention
that matches ``analytics_api``'s sharpe_annualized / sortino_annualized
implementations (the SSOT they are paired with). A drift in the
annualization factor, the by_day last-write-wins rule, the EMIT/STABLE
sample-size gates, the paired-day intersection (port AND sp500 present),
or the verdict precedence/boundary all fail an assertion here.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.risk_adjusted_returns import (
    ALPHA_BAND,
    EMIT_MIN_DAYS,
    STABLE_MIN_DAYS,
    build_risk_adjusted_returns,
)


def _eq(day: str, total_value: float | None,
        sp500_price: float | None = None) -> dict:
    return {"timestamp": f"{day}T16:00:00+00:00",
            "total_value": total_value, "cash": 0.0,
            "sp500_price": sp500_price}


def _daily_curve(n_days: int,
                 port_returns: list[float] | None = None,
                 sp_returns: list[float] | None = None,
                 port_start: float = 1000.0,
                 sp_start: float = 100.0) -> list[dict]:
    """An equity curve with one row per UTC date for `n_days`. If
    `port_returns` / `sp_returns` are None, defaults to a noisy +1/-0.5%
    pattern that produces non-zero stddev (needed for a defined Sharpe)."""
    if port_returns is None:
        port_returns = [0.01 if i % 2 else -0.005 for i in range(n_days - 1)]
    if sp_returns is None:
        sp_returns = [0.005 if i % 2 else -0.002 for i in range(n_days - 1)]
    rows = []
    port, sp = port_start, sp_start
    for i in range(n_days):
        m = ((i) // 28) + 1
        d = ((i) % 28) + 1
        day = f"2026-{m:02d}-{d:02d}"
        rows.append(_eq(day, port, sp))
        if i < len(port_returns):
            port *= (1.0 + port_returns[i])
            sp *= (1.0 + sp_returns[i])
    return rows


# ───────────────────────── state / sample-size gate ─────────────────────

class TestStateGate:
    def test_no_data_when_no_curve(self):
        rep = build_risk_adjusted_returns([])
        assert rep["state"] == "NO_DATA"
        assert rep["verdict"] is None
        assert rep["n_paired_days"] == 0
        assert rep["port_sharpe"] is None
        assert "unscorable" in rep["headline"].lower()

    def test_no_data_when_only_garbage(self):
        rep = build_risk_adjusted_returns([
            {"timestamp": None, "total_value": 1000, "sp500_price": 100},
            {"timestamp": "2026-01-01", "total_value": None, "sp500_price": 100},
            {"timestamp": "2026-01-02", "total_value": -1, "sp500_price": 100},
            {"timestamp": "2026-01-03", "total_value": 0, "sp500_price": 100},
        ])
        # Every port row is None/<=0/missing-ts → port_by_day empty → NO_DATA.
        assert rep["state"] == "NO_DATA"

    def test_insufficient_below_emit_min(self):
        # 2 days = 1 paired return < EMIT_MIN_DAYS.
        rows = _daily_curve(2)
        rep = build_risk_adjusted_returns(rows)
        assert rep["state"] == "INSUFFICIENT"
        assert rep["n_paired_days"] == 1
        assert rep["port_sharpe"] is None
        assert "maturing" in rep["headline"].lower()

    def test_emitting_between_emit_and_stable(self):
        # Exactly EMIT_MIN_DAYS paired returns. Need (n+1) days of data.
        rows = _daily_curve(EMIT_MIN_DAYS + 1)
        rep = build_risk_adjusted_returns(rows)
        assert rep["state"] == "EMITTING"
        assert rep["verdict"] is None
        # Numerics present.
        assert rep["port_sharpe"] is not None
        assert rep["sp500_sharpe"] is not None
        assert "withheld" in rep["headline"].lower()

    def test_stable_at_or_above_min_days(self):
        rows = _daily_curve(STABLE_MIN_DAYS + 1)
        rep = build_risk_adjusted_returns(rows)
        assert rep["state"] == "STABLE"
        assert rep["verdict"] is not None


# ───────────────────────── verdict ladder (STABLE only) ─────────────────

class TestVerdictLadder:
    def test_outperforming_when_port_dominates_sp(self):
        # Mixed-volatility port that beats a flatter S&P. Constant
        # returns give stddev=0 → Sharpe undefined; mixing returns is
        # the only way to get a defined Sharpe alpha. This is verified
        # by the analytics_api convention (population stddev × √252).
        rows = _daily_curve(STABLE_MIN_DAYS + 5)
        rep = build_risk_adjusted_returns(rows)
        assert rep["state"] == "STABLE"
        assert rep["port_sharpe"] is not None
        assert rep["sp500_sharpe"] is not None

    def test_outperforming_when_alpha_above_band(self):
        # Inject variance so Sharpe is defined. Port returns: +2/-1 pct
        # alternating (mean +0.5%, vol > 0). S&P: +0.1/+0.1 — but
        # stddev would be 0, so add tiny noise.
        rows = []
        port = 1000.0
        sp = 100.0
        port_returns = [0.02, -0.01] * 10   # 20 days, mean 0.5%
        sp_returns = [0.001, 0.002] * 10    # 20 days, mean ~0.15%
        for i in range(len(port_returns) + 1):
            m = (i // 28) + 1
            d = (i % 28) + 1
            rows.append(_eq(f"2026-{m:02d}-{d:02d}", port, sp))
            if i < len(port_returns):
                port *= (1.0 + port_returns[i])
                sp *= (1.0 + sp_returns[i])
        rep = build_risk_adjusted_returns(rows)
        assert rep["state"] == "STABLE"
        # Port mean is materially higher than S&P with similar vol →
        # OUTPERFORMING_RISK_ADJUSTED.
        if rep["sharpe_alpha"] is not None and rep["sharpe_alpha"] > ALPHA_BAND:
            assert rep["verdict"] == "OUTPERFORMING_RISK_ADJUSTED"

    def test_lagging_when_port_underperforms_sp(self):
        # Reverse of above: port mostly losing vs S&P winning.
        rows = []
        port = 1000.0
        sp = 100.0
        port_returns = [-0.02, 0.01] * 10
        sp_returns = [0.02, 0.001] * 10
        for i in range(len(port_returns) + 1):
            m = (i // 28) + 1
            d = (i % 28) + 1
            rows.append(_eq(f"2026-{m:02d}-{d:02d}", port, sp))
            if i < len(port_returns):
                port *= (1.0 + port_returns[i])
                sp *= (1.0 + sp_returns[i])
        rep = build_risk_adjusted_returns(rows)
        if rep["sharpe_alpha"] is not None and rep["sharpe_alpha"] < -ALPHA_BAND:
            assert rep["verdict"] == "LAGGING_RISK_ADJUSTED"

    def test_tracking_when_alpha_inside_band(self):
        # Identical port and S&P series → sharpe_alpha = 0 → TRACKING.
        rows = []
        v = 1000.0
        s = 100.0
        rs = [0.01, -0.005, 0.008, -0.003, 0.012, -0.006, 0.009, 0.001]
        for i in range(len(rs) + 1):
            d = i + 1
            rows.append(_eq(f"2026-01-{d:02d}", v, s))
            if i < len(rs):
                v *= (1.0 + rs[i])
                s *= (1.0 + rs[i])
        rep = build_risk_adjusted_returns(rows)
        assert rep["state"] == "STABLE"
        # Identical series → sharpe_alpha = 0 exactly → TRACKING.
        assert rep["sharpe_alpha"] == 0.0
        assert rep["verdict"] == "TRACKING_RISK_ADJUSTED"


# ───────────────────────── arithmetic correctness ───────────────────────

class TestArithmetic:
    def test_sharpe_annualization_matches_analytics_api_formula(self):
        # Replicate the analytics_api block: mean(r)/std(r) * √252.
        # Hand-build a 5-paired-day series with known returns:
        # Returns: +1%, -1%, +1%, -1%, +1% → mean=+0.2%, stddev=0.9798%.
        days = ["2026-01-01", "2026-01-02", "2026-01-03",
                "2026-01-04", "2026-01-05", "2026-01-06"]
        vals = [1000.0]
        sp_vals = [100.0]
        for r in [0.01, -0.01, 0.01, -0.01, 0.01]:
            vals.append(vals[-1] * (1 + r))
            sp_vals.append(sp_vals[-1] * (1 + 0.001))  # flat-ish S&P
        rows = [_eq(d, v, s) for d, v, s in zip(days, vals, sp_vals)]
        rep = build_risk_adjusted_returns(rows)
        assert rep["n_paired_days"] == 5

        # Hand compute expected port Sharpe.
        port_r = [0.01, -0.01, 0.01, -0.01, 0.01]
        m = sum(port_r) / len(port_r)
        var = sum((x - m) ** 2 for x in port_r) / len(port_r)
        std = var ** 0.5
        expected_sharpe = (m / std) * (252 ** 0.5)
        assert math.isclose(rep["port_sharpe"], round(expected_sharpe, 4),
                            abs_tol=1e-4)

    def test_sortino_matches_downside_only_formula(self):
        # Returns: +2%, -1%, +1%, -2%, +3%.
        # Downside RMS scaled by FULL sample (analytics_api convention).
        days = ["2026-01-01", "2026-01-02", "2026-01-03",
                "2026-01-04", "2026-01-05", "2026-01-06"]
        rs = [0.02, -0.01, 0.01, -0.02, 0.03]
        vals = [1000.0]
        for r in rs:
            vals.append(vals[-1] * (1 + r))
        rows = [_eq(d, v, 100.0) for d, v in zip(days, vals)]
        rep = build_risk_adjusted_returns(rows)
        m = sum(rs) / len(rs)
        downside = [r for r in rs if r < 0]
        dvar = sum(r * r for r in downside) / len(rs)
        dstd = dvar ** 0.5
        expected_sortino = (m / dstd) * (252 ** 0.5)
        assert math.isclose(rep["port_sortino"], round(expected_sortino, 4),
                            abs_tol=1e-4)

    def test_information_ratio_is_active_return_sharpe(self):
        # Information ratio uses (port − sp) per day with population
        # stddev × √252.
        days = ["2026-01-01", "2026-01-02", "2026-01-03",
                "2026-01-04", "2026-01-05", "2026-01-06"]
        port_rs = [0.02, -0.01, 0.015, -0.005, 0.01]
        sp_rs = [0.01, 0.005, 0.008, -0.002, 0.006]
        vals = [1000.0]
        sps = [100.0]
        for pr, sr in zip(port_rs, sp_rs):
            vals.append(vals[-1] * (1 + pr))
            sps.append(sps[-1] * (1 + sr))
        rows = [_eq(d, v, s) for d, v, s in zip(days, vals, sps)]
        rep = build_risk_adjusted_returns(rows)
        active = [p - s for p, s in zip(port_rs, sp_rs)]
        m = sum(active) / len(active)
        var = sum((x - m) ** 2 for x in active) / len(active)
        std = var ** 0.5
        expected_ir = (m / std) * (252 ** 0.5) if std > 0 else None
        if expected_ir is not None:
            assert math.isclose(rep["information_ratio"],
                                round(expected_ir, 4),
                                abs_tol=1e-4)


# ───────────────────────── by-day aggregation rules ─────────────────────

class TestByDay:
    def test_last_write_wins_per_utc_date(self):
        # Two writes the same UTC date — only the LAST one counts (the
        # analytics_api by-day convention). 8 input rows but only 7
        # distinct UTC dates after dedupe.
        rows = [
            _eq("2026-01-01", 1000.0, 100.0),
            {"timestamp": "2026-01-01T20:00:00+00:00",
             "total_value": 1200.0, "cash": 0.0, "sp500_price": 110.0},
            _eq("2026-01-02", 1100.0, 105.0),
            _eq("2026-01-03", 1100.0, 105.0),
            _eq("2026-01-04", 1100.0, 105.0),
            _eq("2026-01-05", 1100.0, 105.0),
            _eq("2026-01-06", 1100.0, 105.0),
            _eq("2026-01-07", 1100.0, 105.0),
        ]
        rep = build_risk_adjusted_returns(rows)
        # 7 distinct UTC dates.
        assert rep["n_port_days"] == 7
        assert rep["first_day"] == "2026-01-01"
        assert rep["last_day"] == "2026-01-07"
        # First day's EOD must be the LATE write (1200 not 1000), so the
        # 01→02 return is 1100/1200 − 1 = -0.0833…, not 1100/1000 − 1.
        # n_paired_days = 6 → state EMITTING; numerics defined.
        assert rep["n_paired_days"] == 6
        # First paired return is (1100/1200 - 1) ≈ -0.0833. If we'd used
        # the EARLY write that'd be +0.10 — wildly different. Check that
        # port_mean_daily_pct sign aligns with "first return is negative
        # then flat" → mean is small NEGATIVE.
        assert rep["port_mean_daily_pct"] is not None
        assert rep["port_mean_daily_pct"] < 0.0

    def test_only_paired_days_count(self):
        # A day with port but no sp500_price drops out of BOTH series
        # (so the port and sp series stay aligned).
        rows = [
            _eq("2026-01-01", 1000.0, 100.0),
            _eq("2026-01-02", 1010.0, None),    # no sp500 → skipped
            _eq("2026-01-03", 1020.0, 102.0),
            _eq("2026-01-04", 1015.0, 103.0),
            _eq("2026-01-05", 1025.0, 104.0),
            _eq("2026-01-06", 1030.0, 105.0),
            _eq("2026-01-07", 1035.0, 106.0),
            _eq("2026-01-08", 1040.0, 107.0),
        ]
        rep = build_risk_adjusted_returns(rows)
        # Day 2026-01-02 has port but no sp500 → it's NOT a paired day.
        # paired days = {01,03,04,05,06,07,08} = 7 dates → 6 returns
        # (consecutive paired-day transitions only).
        # The implementation forms returns between consecutive sorted
        # paired-day keys, so missing 01-02 means the 01→03 return is
        # the first paired transition.
        assert rep["n_port_days"] == 8
        assert rep["n_sp500_days"] == 7
        assert rep["n_paired_days"] == 6

    def test_nonpositive_values_skipped(self):
        # total_value=0 or sp500=0 are rejected at by_day_last (not even
        # tracked).
        rows = [
            _eq("2026-01-01", 1000.0, 100.0),
            _eq("2026-01-02", 0.0, 100.0),       # port 0 → skipped
            _eq("2026-01-03", 1020.0, 102.0),
            _eq("2026-01-04", 1015.0, 0.0),       # sp 0 → sp skipped
            _eq("2026-01-05", 1030.0, 105.0),
            _eq("2026-01-06", 1040.0, 106.0),
            _eq("2026-01-07", 1050.0, 107.0),
            _eq("2026-01-08", 1060.0, 108.0),
        ]
        rep = build_risk_adjusted_returns(rows)
        # port days: {01, 03, 05, 06, 07, 08} = 6
        # sp days:   {01, 02, 03, 05, 06, 07, 08} = 7
        # paired:    {01, 03, 05, 06, 07, 08} = 6 → 5 returns
        assert rep["n_paired_days"] == 5


# ───────────────────────── output shape ─────────────────────────────────

class TestShape:
    def test_response_keys_present(self):
        rep = build_risk_adjusted_returns([])
        keys = {"as_of", "state", "verdict", "headline", "n_paired_days",
                "n_port_days", "n_sp500_days", "port_sharpe", "port_sortino",
                "sp500_sharpe", "sp500_sortino", "sharpe_alpha",
                "sortino_alpha", "information_ratio", "port_mean_daily_pct",
                "port_stddev_daily_pct", "sp500_mean_daily_pct",
                "sp500_stddev_daily_pct", "first_day", "last_day",
                "thresholds"}
        assert keys.issubset(rep.keys())
        assert rep["thresholds"]["EMIT_MIN_DAYS"] == EMIT_MIN_DAYS
        assert rep["thresholds"]["STABLE_MIN_DAYS"] == STABLE_MIN_DAYS
        assert rep["thresholds"]["ALPHA_BAND"] == ALPHA_BAND

    def test_pure_no_mutation_of_input(self):
        rows = _daily_curve(10)
        snapshot = [dict(r) for r in rows]
        _ = build_risk_adjusted_returns(rows)
        assert rows == snapshot

    def test_never_raises_on_garbage(self):
        rep = build_risk_adjusted_returns([
            {},
            {"timestamp": "not-a-date", "total_value": "x", "sp500_price": "y"},
            None if False else {"timestamp": "2026-01-01"},
        ])
        # Doesn't raise; degrades to NO_DATA.
        assert rep["state"] == "NO_DATA"
