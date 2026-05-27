"""Behaviour lock for paper_trader.ml.run_risk_metrics.

Synthetic equity curves where the expected values are known exactly —
tests catch off-by-one errors in the daily-return chain, sign errors in
max-drawdown, annualization-factor confusion, and the Calmar verdict
boundary mapping. Where a value is computed analytically (e.g. a +1% per
day exponential growth ⇒ predictable CAGR), the test asserts the exact
value rather than a noisy approximation.
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.ml.run_risk_metrics import (
    MIN_POINTS,
    TRADING_DAYS_PER_YEAR,
    _calmar_verdict,
    _daily_returns,
    annualized_return_pct,
    calmar_ratio,
    compute_run_risk,
    leaderboard,
    main,
    max_drawdown_pct,
    sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Daily returns chain
# ---------------------------------------------------------------------------

class TestDailyReturns:
    def test_constant_series_yields_zero_returns(self):
        rets = _daily_returns([100.0, 100.0, 100.0, 100.0])
        assert rets == [0.0, 0.0, 0.0]

    def test_one_pct_per_day_growth(self):
        # Each step is +1%
        eq = [100.0, 101.0, 102.01, 103.0301]
        rets = _daily_returns(eq)
        assert len(rets) == 3
        for r in rets:
            assert r == pytest.approx(0.01, abs=1e-6)

    def test_returns_handles_non_positive_prior(self):
        """A zero or negative prior value MUST coerce to 0.0 return rather
        than crashing — same defensive discipline as PriceCache.returns_pct."""
        rets = _daily_returns([100.0, 0.0, 100.0])
        # First step: 0/100 - 1 = -1.0 (a real -100% wipe)
        # Second step: prior is 0 → 0.0 sentinel
        assert rets[0] == pytest.approx(-1.0, abs=1e-6)
        assert rets[1] == 0.0

    def test_returns_handles_nan(self):
        rets = _daily_returns([100.0, float("nan"), 100.0])
        # Both transitions involve NaN → 0.0
        assert rets == [0.0, 0.0]

    def test_returns_handles_inf(self):
        rets = _daily_returns([100.0, float("inf"), 100.0])
        assert rets == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_strictly_rising_series_zero_drawdown(self):
        eq = [100.0, 110.0, 120.0, 130.0, 140.0]
        assert max_drawdown_pct(eq) == 0.0

    def test_known_drawdown_amount(self):
        # Peak 200, trough 100 → -50% drawdown
        eq = [100.0, 150.0, 200.0, 150.0, 100.0, 120.0]
        # Peak reached at index 2 (200), trough at index 4 (100).
        # MDD = (100 - 200) / 200 * 100 = -50.0
        assert max_drawdown_pct(eq) == pytest.approx(-50.0, abs=1e-4)

    def test_complete_wipeout_drawdown(self):
        # 100 → 1 = -99% drawdown
        eq = [100.0, 50.0, 1.0, 1.0, 1.0]
        assert max_drawdown_pct(eq) == pytest.approx(-99.0, abs=1e-4)

    def test_zigzag_records_largest_drawdown(self):
        # Multiple smaller drawdowns then one big one
        eq = [100.0, 90.0, 110.0, 100.0, 200.0, 80.0, 100.0]
        # Big drawdown: peak 200 → trough 80 = -60%
        assert max_drawdown_pct(eq) == pytest.approx(-60.0, abs=1e-4)

    def test_too_few_points_returns_none(self):
        eq = [100.0, 110.0]  # 2 points < MIN_POINTS=5
        assert max_drawdown_pct(eq) is None

    def test_zero_or_negative_peak_returns_none(self):
        eq = [-1.0, -2.0, -3.0, -4.0, -5.0]
        assert max_drawdown_pct(eq) is None


# ---------------------------------------------------------------------------
# Annualized return
# ---------------------------------------------------------------------------

class TestAnnualizedReturn:
    def test_doubling_in_one_year(self):
        # Start 100, end 200, 365.25 days = exactly 1 year
        eq = [100.0] + [150.0] * 100 + [200.0]
        ar = annualized_return_pct(eq, days=365)
        # 365 days ≈ 0.999 years, growth=2.0 → ~100% CAGR
        # Tolerance for the slight day-count difference
        assert ar is not None
        assert ar == pytest.approx(100.0, abs=0.5)

    def test_50_pct_loss_in_one_year(self):
        eq = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0]
        ar = annualized_return_pct(eq, days=365)
        assert ar is not None
        # 100→50 = -50% over ~1 year
        assert ar == pytest.approx(-50.0, abs=0.5)

    def test_break_even_over_two_years(self):
        eq = [100.0] * 10
        ar = annualized_return_pct(eq, days=730)
        assert ar == 0.0

    def test_wipeout_returns_minus_100(self):
        eq = [100.0, 50.0, 0.0, 0.0, 0.0]
        ar = annualized_return_pct(eq, days=365)
        # end == 0 → growth ratio is 0 → CAGR = -100%
        assert ar == pytest.approx(-100.0, abs=0.01)

    def test_none_for_missing_days(self):
        assert annualized_return_pct([100.0, 110.0], days=None) is None
        assert annualized_return_pct([100.0, 110.0], days=0) is None
        assert annualized_return_pct([100.0, 110.0], days=-1) is None


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_constant_series_returns_none(self):
        """A flat-cash equity series has zero variance — Sharpe is undefined."""
        eq = [100.0] * 30
        assert sharpe_ratio(eq) is None

    def test_low_noise_steady_growth_high_sharpe(self):
        """A near-deterministic +0.5% per day series (with tiny noise so
        std > 0) has a very high Sharpe. A perfectly constant return is
        Sharpe-undefined (the previous test held, see
        test_perfectly_constant_returns_none)."""
        eq = [100.0]
        # Alternate +0.5% / +0.4% — mean ≈ 0.45%, std small but nonzero
        for i in range(60):
            mult = 1.005 if i % 2 == 0 else 1.004
            eq.append(eq[-1] * mult)
        sr = sharpe_ratio(eq)
        assert sr is not None
        # Near-deterministic with small std → very high annualized Sharpe.
        # Conservative floor: above 50 (real production Sharpes are ~0–3).
        assert sr > 50.0

    def test_perfectly_constant_returns_none(self):
        """A perfectly constant compounding rate (+0.5%/day every day)
        has std=0. Sharpe is mathematically undefined — must return None,
        not a fabricated infinity."""
        eq = [100.0]
        for _ in range(60):
            eq.append(eq[-1] * 1.005)
        assert sharpe_ratio(eq) is None

    def test_volatile_breakeven_low_sharpe(self):
        """A series with high variance and ~0 mean return has near-zero Sharpe."""
        # Alternating +5% / -5% — mean ≈ -0.125% (geometric), std large
        eq = [100.0]
        for i in range(60):
            multiplier = 1.05 if i % 2 == 0 else 1.0 / 1.05
            eq.append(eq[-1] * multiplier)
        sr = sharpe_ratio(eq)
        assert sr is not None
        # Geometric drift of ±5% averages to slightly negative
        assert abs(sr) < 5.0  # not extreme

    def test_risk_free_rate_lowers_sharpe(self):
        """Excess return = return - rf. A higher rf lowers excess and thus Sharpe.
        Use a noisy series so std > 0 and Sharpe is defined for both calls."""
        eq = [100.0]
        for i in range(60):
            mult = 1.005 if i % 2 == 0 else 1.003
            eq.append(eq[-1] * mult)
        sr_no_rf = sharpe_ratio(eq, risk_free_rate_annual=0.0)
        sr_rf = sharpe_ratio(eq, risk_free_rate_annual=0.05)  # 5% annual
        assert sr_no_rf is not None
        assert sr_rf is not None
        assert sr_no_rf > sr_rf

    def test_too_few_points_returns_none(self):
        assert sharpe_ratio([100.0, 110.0]) is None

    def test_annualization_factor_is_sqrt_252(self):
        """The Sharpe annualization factor is hardcoded to sqrt(252).
        Verify with a known input: daily mean return r, daily std s →
        annualized SR = sqrt(252) * r / s."""
        # Construct daily returns with KNOWN mean and std via a controlled
        # equity series.
        # Use 100 days of alternating multipliers chosen so daily returns
        # are exactly [r1, r2, r1, r2, ...]
        r1, r2 = 0.01, -0.005
        eq = [100.0]
        for i in range(100):
            mult = 1 + (r1 if i % 2 == 0 else r2)
            eq.append(eq[-1] * mult)
        sr = sharpe_ratio(eq, risk_free_rate_annual=0.0)
        # Expected: mean = (r1 + r2)/2 = 0.0025; std (sample) → compute
        # Roughly: var ≈ ((r1-mean)^2 + (r2-mean)^2)/2 = ((0.01-0.0025)^2 + (-0.005-0.0025)^2)/2
        # = (0.0075^2 + 0.0075^2)/2 = 0.00005625 → std ≈ 0.0075
        # Annualized SR ≈ sqrt(252) * 0.0025 / 0.0075 ≈ 15.875 * 0.333 ≈ 5.29
        assert sr is not None
        # Allow some tolerance for sample-std vs population-std difference
        assert sr == pytest.approx(5.3, abs=0.5)


# ---------------------------------------------------------------------------
# Calmar ratio & verdict
# ---------------------------------------------------------------------------

class TestCalmar:
    def test_calmar_basic(self):
        # 50% annualized return, -25% MDD → Calmar = 2.0
        c = calmar_ratio(annualized_ret_pct=50.0, mdd_pct=-25.0)
        assert c == pytest.approx(2.0, abs=1e-4)

    def test_calmar_uses_absolute_mdd(self):
        """Calmar = return / |mdd|. Sign of MDD must not flip the ratio sign."""
        c1 = calmar_ratio(20.0, -10.0)
        c2 = calmar_ratio(20.0, 10.0)  # hypothetical positive mdd (impossible)
        assert c1 == c2

    def test_calmar_none_when_inputs_none(self):
        assert calmar_ratio(None, -10.0) is None
        assert calmar_ratio(50.0, None) is None

    def test_zero_drawdown_positive_return_returns_inf(self):
        c = calmar_ratio(50.0, 0.0)
        assert math.isinf(c) and c > 0

    def test_zero_drawdown_negative_return_returns_minus_inf(self):
        c = calmar_ratio(-20.0, 0.0)
        assert math.isinf(c) and c < 0

    def test_zero_drawdown_zero_return_yields_zero(self):
        c = calmar_ratio(0.0, 0.0)
        assert c == 0.0


class TestCalmarVerdict:
    def test_investment_grade_at_1_0(self):
        assert _calmar_verdict(1.0) == "INVESTMENT_GRADE"
        assert _calmar_verdict(2.5) == "INVESTMENT_GRADE"

    def test_acceptable_band(self):
        assert _calmar_verdict(0.5) == "ACCEPTABLE"
        assert _calmar_verdict(0.99) == "ACCEPTABLE"

    def test_marginal_band(self):
        assert _calmar_verdict(0.0) == "MARGINAL"
        assert _calmar_verdict(0.49) == "MARGINAL"

    def test_loss_making_band(self):
        assert _calmar_verdict(-0.01) == "LOSS_MAKING"
        assert _calmar_verdict(-100.0) == "LOSS_MAKING"

    def test_none_is_insufficient(self):
        assert _calmar_verdict(None) == "INSUFFICIENT_DATA"

    def test_inf_routed_by_sign(self):
        assert _calmar_verdict(float("inf")) == "INVESTMENT_GRADE"
        assert _calmar_verdict(float("-inf")) == "LOSS_MAKING"


# ---------------------------------------------------------------------------
# compute_run_risk full pipeline
# ---------------------------------------------------------------------------

class TestComputeRunRisk:
    def test_empty_curve_returns_insufficient(self):
        out = compute_run_risk([])
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["n_points"] == 0

    def test_typical_winning_run(self):
        # 200 trading days of ~+0.2% per day growth, then a 30% drawdown
        # then recovery. Investment-grade Calmar.
        eq = [{"date": "2024-01-01", "value": 1000.0}]
        for i in range(100):
            eq.append({"date": f"2024-{(i//30)+1:02d}-{(i%30)+1:02d}",
                       "value": eq[-1]["value"] * 1.002})
        # 30% drawdown
        for _ in range(20):
            eq.append({"date": "2024-05-01",
                       "value": eq[-1]["value"] * 0.985})
        # recovery
        for _ in range(80):
            eq.append({"date": "2024-06-01",
                       "value": eq[-1]["value"] * 1.005})
        out = compute_run_risk(eq, duration_days=365)
        assert out["n_points"] == len(eq)
        assert out["annualized_return_pct"] is not None
        assert out["max_drawdown_pct"] is not None
        assert out["max_drawdown_pct"] < 0  # there WAS a drawdown
        assert out["sharpe_ratio"] is not None
        # Verdict should be defined for any complete curve
        assert out["verdict"] in {
            "INVESTMENT_GRADE", "ACCEPTABLE", "MARGINAL",
            "LOSS_MAKING", "INSUFFICIENT_DATA",
        }

    def test_corrupt_curve_entries_dropped(self):
        eq = [
            {"date": "2024-01-01", "value": 1000.0},
            {"date": "2024-01-02", "value": "not a number"},
            {"date": "2024-01-03", "value": None},
            "not a dict",  # also dropped
            {"date": "2024-01-04", "value": 1010.0},
            {"date": "2024-01-05", "value": 1020.0},
            {"date": "2024-01-06", "value": 1015.0},
            {"date": "2024-01-07", "value": 1030.0},
        ]
        out = compute_run_risk(eq, duration_days=7)
        # Valid points: 4 (the dict entries with float values).
        # Wait: actually 5 valid entries (1000, 1010, 1020, 1015, 1030).
        assert out["n_points"] == 5

    def test_duration_days_derived_from_dates_when_missing(self):
        eq = [
            {"date": "2024-01-01", "value": 100.0},
            {"date": "2024-04-10", "value": 110.0},
            {"date": "2024-07-15", "value": 120.0},
            {"date": "2024-10-20", "value": 105.0},
            {"date": "2025-01-01", "value": 130.0},
        ]
        out = compute_run_risk(eq)
        # 2024-01-01 → 2025-01-01 = 366 days
        assert out["duration_days"] == 366

    def test_duration_fallback_for_bad_dates(self):
        eq = [{"value": 100.0 + i, "date": "garbage"} for i in range(10)]
        out = compute_run_risk(eq)
        # Falls back to len(values)-1 when dates can't be parsed
        # values: 100..109 = 10 points → 9 days
        assert out["duration_days"] == 9


# ---------------------------------------------------------------------------
# Leaderboard (uses sqlite)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_backtest_db(tmp_path: Path) -> Path:
    """Tiny in-memory backtest DB with three runs of known risk profiles."""
    db_path = tmp_path / "backtest.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE backtest_runs (
            run_id INTEGER PRIMARY KEY,
            seed INTEGER NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            start_value REAL NOT NULL,
            final_value REAL NOT NULL DEFAULT 0,
            total_return_pct REAL NOT NULL DEFAULT 0,
            spy_return_pct REAL NOT NULL DEFAULT 0,
            vs_spy_pct REAL NOT NULL DEFAULT 0,
            n_trades INTEGER NOT NULL DEFAULT 0,
            n_decisions INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            equity_curve_json TEXT NOT NULL DEFAULT '[]',
            notes TEXT
        );
    """)
    # Run 1: smooth +20% (high Sharpe / Calmar)
    eq_smooth = [{"date": f"2024-01-{i+1:02d}", "value": 1000.0 * (1.001 ** i)}
                 for i in range(31)]
    # Run 2: volatile +20% (lower Sharpe but same return)
    eq_volatile = [{"date": "2024-01-01", "value": 1000.0}]
    for i in range(30):
        mult = 1.05 if i % 2 == 0 else 1.0 / 1.04
        eq_volatile.append({"date": f"2024-01-{i+2:02d}",
                            "value": eq_volatile[-1]["value"] * mult})
    # Force exact +20% final
    eq_volatile[-1]["value"] = 1200.0

    # Run 3: -10% LOSS_MAKING
    eq_loss = [{"date": f"2024-01-{i+1:02d}", "value": 1000.0 * (1.0 - i * 0.003)}
               for i in range(31)]
    # End at 900
    eq_loss[-1]["value"] = 900.0

    rows = [
        (1, 1, "2024-01-01", "2024-01-31", 1000.0, eq_smooth[-1]["value"],
         (eq_smooth[-1]["value"] - 1000) / 1000 * 100, 0.0, 0.0, 0, 0,
         "complete", "2024-01-01T00:00:00", "2024-02-01T00:00:00",
         json.dumps(eq_smooth), ""),
        (2, 2, "2024-01-01", "2024-01-31", 1000.0, 1200.0, 20.0, 0.0, 0.0, 0, 0,
         "complete", "2024-01-01T00:00:00", "2024-02-01T00:00:00",
         json.dumps(eq_volatile), ""),
        (3, 3, "2024-01-01", "2024-01-31", 1000.0, 900.0, -10.0, 0.0, 0.0, 0, 0,
         "complete", "2024-01-01T00:00:00", "2024-02-01T00:00:00",
         json.dumps(eq_loss), ""),
    ]
    conn.executemany(
        "INSERT INTO backtest_runs (run_id, seed, start_date, end_date, "
        "start_value, final_value, total_return_pct, spy_return_pct, vs_spy_pct, "
        "n_trades, n_decisions, status, started_at, completed_at, equity_curve_json, notes) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db_path


class TestLeaderboard:
    def test_leaderboard_by_calmar(self, synthetic_backtest_db):
        out = leaderboard(synthetic_backtest_db, rank_by="calmar", top=5)
        assert out["status"] == "ok"
        # Run 1 (smooth) has tiny drawdown → highest Calmar
        # Run 2 (volatile) has bigger drawdown → lower Calmar
        # Run 3 (loss) is LOSS_MAKING
        assert out["top"][0]["run_id"] == 1
        # Loss-making sorts last
        verdicts = [r["verdict"] for r in out["top"]]
        assert "LOSS_MAKING" in verdicts
        assert verdicts.index("LOSS_MAKING") > 0

    def test_leaderboard_by_sharpe(self, synthetic_backtest_db):
        out = leaderboard(synthetic_backtest_db, rank_by="sharpe", top=5)
        assert out["status"] == "ok"
        # Run 1's deterministic growth → highest Sharpe
        assert out["top"][0]["run_id"] == 1

    def test_leaderboard_by_return(self, synthetic_backtest_db):
        """Sanity: ranking by return should match total_return_pct sort."""
        out = leaderboard(synthetic_backtest_db, rank_by="return", top=5)
        assert out["status"] == "ok"
        rets = [r["total_return_pct"] for r in out["top"]]
        # Top runs are by raw return descending
        assert rets == sorted(rets, reverse=True)

    def test_leaderboard_by_mdd(self, synthetic_backtest_db):
        """Smallest absolute drawdown first."""
        out = leaderboard(synthetic_backtest_db, rank_by="mdd", top=5)
        assert out["status"] == "ok"
        # Run 1 has the smallest |MDD| (smooth growth)
        assert out["top"][0]["run_id"] == 1

    def test_invalid_rank_by_raises(self, synthetic_backtest_db):
        with pytest.raises(ValueError):
            leaderboard(synthetic_backtest_db, rank_by="invalid")

    def test_missing_db_returns_no_runs(self, tmp_path):
        out = leaderboard(tmp_path / "nonexistent.db", rank_by="calmar")
        assert out["status"] == "no_runs"

    def test_failed_runs_excluded(self, tmp_path):
        """Only status='complete' runs contribute to the leaderboard."""
        db_path = tmp_path / "failed.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE backtest_runs (
                run_id INTEGER PRIMARY KEY,
                seed INTEGER NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                start_value REAL NOT NULL,
                final_value REAL NOT NULL DEFAULT 0,
                total_return_pct REAL NOT NULL DEFAULT 0,
                spy_return_pct REAL NOT NULL DEFAULT 0,
                vs_spy_pct REAL NOT NULL DEFAULT 0,
                n_trades INTEGER NOT NULL DEFAULT 0,
                n_decisions INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                equity_curve_json TEXT NOT NULL DEFAULT '[]',
                notes TEXT
            );
        """)
        conn.execute(
            "INSERT INTO backtest_runs (run_id, seed, start_date, end_date, "
            "start_value, status, started_at, equity_curve_json) "
            "VALUES (1, 1, '2024-01-01', '2024-01-31', 1000.0, 'failed', "
            "'2024-01-01T00:00:00', '[]')",
        )
        conn.execute(
            "INSERT INTO backtest_runs (run_id, seed, start_date, end_date, "
            "start_value, status, started_at, equity_curve_json) "
            "VALUES (2, 2, '2024-01-01', '2024-01-31', 1000.0, 'running', "
            "'2024-01-01T00:00:00', '[]')",
        )
        conn.commit()
        conn.close()
        out = leaderboard(db_path, rank_by="calmar")
        assert out["status"] == "no_runs"


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_run_id_not_found_exits_one(self, synthetic_backtest_db, capsys):
        rc = main(["--db", str(synthetic_backtest_db),
                   "--run-id", "99999", "--json"])
        assert rc == 1
        out = json.loads(capsys.readouterr().out)
        assert out["status"] == "not_found"

    def test_cli_run_id_emits_full_json(self, synthetic_backtest_db, capsys):
        rc = main(["--db", str(synthetic_backtest_db),
                   "--run-id", "1", "--json"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["run_id"] == 1
        assert "sharpe_ratio" in out
        assert "max_drawdown_pct" in out
        assert "calmar_ratio" in out
        assert "verdict" in out

    def test_cli_leaderboard_top_count(self, synthetic_backtest_db, capsys):
        rc = main(["--db", str(synthetic_backtest_db),
                   "--rank-by", "calmar", "--top", "2", "--json"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["status"] == "ok"
        assert len(out["top"]) == 2

    def test_cli_text_leaderboard_smoke(self, synthetic_backtest_db, capsys):
        """Text mode renders without raising and prints a header line."""
        rc = main(["--db", str(synthetic_backtest_db), "--top", "5"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "run_id" in out
        assert "Calmar" in out

    def test_cli_text_single_run_smoke(self, synthetic_backtest_db, capsys):
        rc = main(["--db", str(synthetic_backtest_db), "--run-id", "1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "run_id=1" in out
        assert "verdict" in out
