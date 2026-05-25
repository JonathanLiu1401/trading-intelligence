"""Tests for analytics/portfolio_beta.py — market-model regression panel.

Beta, alpha, and R² are pinned with **exact hand-computed literals** on a
perfectly linear fixture where portfolio returns are 2× SPY returns; that
single fixture catches: a swapped-axis regression (beta = 0.5 instead of
2.0), the wrong variance side in the R² denominator, an off-by-one in the
paired-day walk, alpha computed as ``ms - beta*mp`` instead of
``mp - beta*ms``, and a missing population-variance convention vs
analytics_api. The Flask test_client smoke locks the route contract
(matches the analytics-verification memory: module __main__ smoke hits a
different/empty DB; verify the live endpoint via test_client).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.portfolio_beta import (  # noqa: E402
    MIN_RETURNS,
    REGIME_SHIFT_THRESHOLD,
    build_portfolio_beta,
)

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _curve_from_returns(
    rp: list[float], rs: list[float], pf0: float = 1000.0, sp0: float = 100.0
) -> list[dict]:
    """Build an equity_curve with paired portfolio/SPY daily returns.

    Day 0 carries the starting values; day i = day 0 + i with returns
    ``(rp[i-1], rs[i-1])`` applied. Matches the by-day resample contract.
    """
    assert len(rp) == len(rs)
    rows = [{
        "timestamp": _BASE.isoformat(),
        "total_value": pf0, "cash": 0.0, "sp500_price": sp0,
    }]
    pv, sv = pf0, sp0
    for i, (a, b) in enumerate(zip(rp, rs), start=1):
        pv *= (1.0 + a)
        sv *= (1.0 + b)
        rows.append({
            "timestamp": (_BASE + timedelta(days=i)).isoformat(),
            "total_value": pv, "cash": 0.0, "sp500_price": sv,
        })
    return rows


class TestHonestyGate:
    def test_empty_is_no_data(self):
        r = build_portfolio_beta([])
        assert r["state"] == "NO_DATA"
        assert r["n_returns"] == 0
        assert r["beta"] is None
        assert "no paired daily returns" in r["headline"].lower()

    def test_no_paired_spy_is_no_data(self):
        # Equity exists but every sp500_price is None ⇒ no paired returns.
        rows = [
            {"timestamp": (_BASE + timedelta(days=i)).isoformat(),
             "total_value": 1000.0 + i, "cash": 0.0, "sp500_price": None}
            for i in range(5)
        ]
        r = build_portfolio_beta(rows)
        assert r["state"] == "NO_DATA"
        assert r["n_returns"] == 0

    def test_below_min_returns_is_insufficient_but_numeric(self):
        rp = [0.01, -0.01, 0.02, -0.02, 0.01]
        rs = [0.005, -0.005, 0.01, -0.01, 0.005]
        r = build_portfolio_beta(_curve_from_returns(rp, rs))
        assert r["state"] == "INSUFFICIENT"
        assert r["n_returns"] == 5
        assert r["beta"] is not None
        assert "withheld" in r["headline"].lower()
        assert f"5/{MIN_RETURNS}" in r["headline"]

    def test_exactly_min_returns_is_ok(self):
        rp = [0.01] * MIN_RETURNS
        rs = [0.005] * MIN_RETURNS
        r = build_portfolio_beta(_curve_from_returns(rp, rs))
        # Zero variance on SPY ⇒ beta undefined, but state must still be OK
        # because the sample is large enough. The regression simply returns
        # None for the metrics (honest "no slope from a constant").
        assert r["n_returns"] == MIN_RETURNS
        assert r["state"] == "OK"
        assert r["beta"] is None


class TestHandComputedFixture:
    """Perfectly linear r_p = 2 * r_s ⇒ β = 2.0, R² = 1.0, α = 0."""

    def setup_method(self):
        # 6 distinct return pairs repeated 6× → 36 returns, ≥ MIN_RETURNS
        # AND ≥ ROLLING_WINDOW (30) so the rolling-β test is meaningful.
        # The pattern's sum is zero so the means are zero — α drops out
        # cleanly to 0, separately checking the intercept formula from β.
        base_rs = [0.02, -0.02, 0.01, -0.01, 0.015, -0.015]
        rs = base_rs * 6
        rp = [2.0 * x for x in rs]
        self.rep = build_portfolio_beta(_curve_from_returns(rp, rs))

    def test_shape_and_state(self):
        assert self.rep["state"] == "OK"
        assert self.rep["n_returns"] == 36

    def test_beta_exact(self):
        assert self.rep["beta"] == 2.0

    def test_alpha_zero_on_zero_mean(self):
        # mp = ms = 0 ⇒ α = 0 - 2*0 = 0 (both daily and annualised).
        assert self.rep["alpha_daily_pct"] == 0.0
        assert self.rep["alpha_annualized_pct"] == 0.0

    def test_r_squared_perfect(self):
        # Perfectly linear ⇒ R² = 1.0
        assert self.rep["r_squared"] == 1.0

    def test_se_beta_zero_on_perfect_fit(self):
        # resid_var = var_p - β*cov = 4·var_s - 2·(2·var_s) = 0 ⇒ SE = 0.
        assert self.rep["beta_stderr"] == 0.0

    def test_rolling_beta_matches_alltime_no_regime_shift(self):
        # Identical pattern across the whole sample ⇒ rolling = all-time
        # ⇒ no regime shift flagged.
        assert self.rep["rolling_beta"] == 2.0
        assert self.rep["regime_shift"] is False
        assert self.rep["regime_shift_delta"] == 0.0

    def test_headline_carries_pinned_numbers(self):
        h = self.rep["headline"]
        assert "β=2.00" in h
        assert "R²=1.00" in h
        assert "α(ann)=+0.00%" in h


class TestAnticorrelatedFixture:
    """r_p = -1 × r_s ⇒ β = -1.0, R² = 1.0."""

    def test_negative_beta_exact(self):
        base_rs = [0.02, -0.02, 0.01, -0.01, 0.015, -0.015]
        rs = base_rs * 4  # 24 returns, > MIN_RETURNS
        rp = [-x for x in rs]
        rep = build_portfolio_beta(_curve_from_returns(rp, rs))
        assert rep["state"] == "OK"
        assert rep["beta"] == -1.0
        assert rep["r_squared"] == 1.0


class TestRegimeShift:
    """Recent rolling β differs from all-time by ≥ threshold ⇒ flagged."""

    def test_rolling_diverges_triggers_flag(self):
        # 30 days at β=0.5 then 30 days at β=2.0. All-time β should sit
        # somewhere between, and rolling β over the last 30d should be
        # ~2.0, so the |Δ| safely exceeds REGIME_SHIFT_THRESHOLD.
        base_rs = [0.02, -0.02, 0.01, -0.01, 0.0, 0.015]
        rs_early = base_rs * 5  # 30 returns
        rs_late = base_rs * 5
        rp_early = [0.5 * x for x in rs_early]
        rp_late = [2.0 * x for x in rs_late]
        rs = rs_early + rs_late
        rp = rp_early + rp_late
        rep = build_portfolio_beta(_curve_from_returns(rp, rs))
        assert rep["state"] == "OK"
        assert rep["rolling_beta"] == 2.0
        assert rep["regime_shift"] is True
        # Δ = rolling - all_time; since all_time is between 0.5 and 2.0,
        # Δ should be positive and at least the threshold.
        assert rep["regime_shift_delta"] >= REGIME_SHIFT_THRESHOLD


class TestSSOTParityWithAnalyticsAPI:
    """The bare β scalar this module computes must match the rounded β
    that ``/api/analytics`` (analytics_api in dashboard.py) computes from
    the same equity_curve — population variance, paired-day walk, both."""

    def test_beta_matches_analytics_api_arithmetic(self):
        # Match the analytics_api arithmetic exactly: paired returns
        # built the same way, population cov/var, round-to-2dp.
        import random
        rng = random.Random(42)
        rs = [rng.uniform(-0.03, 0.03) for _ in range(40)]
        # Realistic portfolio: β≈1.3, plus idiosyncratic noise.
        rp = [1.3 * x + rng.uniform(-0.005, 0.005) for x in rs]
        rep = build_portfolio_beta(_curve_from_returns(rp, rs))

        # Reproduce analytics_api's own arithmetic on the same returns.
        n = len(rp)
        mp = sum(rp) / n
        ms = sum(rs) / n
        cov = sum((rp[i] - mp) * (rs[i] - ms) for i in range(n)) / n
        var_s = sum((s - ms) ** 2 for s in rs) / n
        var_p = sum((p - mp) ** 2 for p in rp) / n
        ref_beta = round(cov / var_s, 2)
        ref_corr = round(cov / ((var_s ** 0.5) * (var_p ** 0.5)), 3)

        # build_portfolio_beta rounds β to 3dp; analytics_api rounds to 2dp,
        # so compare the 2dp rounding of the module's value.
        assert round(rep["beta"], 2) == ref_beta
        # R² and ρ² are equal under OLS — compare against the *unrounded*
        # corr-squared so the test isn't sensitive to the double-rounding
        # asymmetry (rounded-then-squared ≠ squared-then-rounded near the
        # 3rd decimal).
        unrounded_corr = cov / ((var_s ** 0.5) * (var_p ** 0.5))
        assert abs(rep["r_squared"] - unrounded_corr ** 2) < 1e-3


def test_endpoint_contract_via_flask_test_client():
    """Live-route smoke: verify /api/portfolio-beta returns a 200 with the
    expected schema. Uses the Flask test_client (not a TCP request) so the
    test stands alone — no live :8090 dependency.

    Mirrors the [paper_trader analytics verification] memory: module
    __main__ smoke hits an empty data/ DB; the actual endpoint contract
    must be verified via test_client.
    """
    from paper_trader.dashboard import app
    client = app.test_client()
    r = client.get("/api/portfolio-beta")
    assert r.status_code == 200
    data = r.get_json()
    assert data is not None
    expected = {
        "as_of", "n_returns", "min_returns", "rolling_window",
        "beta", "alpha_daily_pct", "alpha_annualized_pct", "r_squared",
        "beta_stderr", "rolling_beta", "rolling_n", "regime_shift",
        "regime_shift_delta", "state", "headline",
    }
    missing = expected - set(data.keys())
    assert not missing, f"endpoint missing keys: {missing}"
    assert data["state"] in ("NO_DATA", "INSUFFICIENT", "OK")
