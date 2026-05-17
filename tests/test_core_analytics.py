"""Regression test for the /api/analytics round-trip metrics.

Written before the inline round-trip block in dashboard.analytics_api was
replaced with analytics.round_trips.build_round_trips, and kept afterwards:
the hand-computed expectations below must hold for *both* implementations,
which is exactly the regression guarantee the extraction needed.

It seeds a real Store (temp DB) with a fixed trade ledger, hits the Flask
endpoint via the test client, and asserts exact win_rate / profit_factor /
avg_holding_days / realized P&L — not "the call returned 200".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import store as store_mod
from paper_trader.store import Store


@pytest.fixture
def seeded_client(tmp_path, monkeypatch):
    """Fresh Store at a temp DB, seeded with a known ledger, + Flask client."""
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()

    # Four closed round-trips with known P&L:
    #   AAPL  BUY 10@10 -> SELL 10@12          = +20  (win)
    #   MSFT  BUY  5@20 -> SELL  5@16          = -20  (loss)
    #   NVDA  BUY  2@50 -> SELL 1@60, SELL 1@70 = +30  (win, partial sells)
    #   TSLA  BUY_CALL 1@2 -> SELL_CALL 1@3    = +100 (win, option x100)
    s.record_trade("AAPL", "BUY", 10, 10.0)
    s.record_trade("AAPL", "SELL", 10, 12.0)
    s.record_trade("MSFT", "BUY", 5, 20.0)
    s.record_trade("MSFT", "SELL", 5, 16.0)
    s.record_trade("NVDA", "BUY", 2, 50.0)
    s.record_trade("NVDA", "SELL", 1, 60.0)
    s.record_trade("NVDA", "SELL", 1, 70.0)
    s.record_trade("TSLA", "BUY_CALL", 1, 2.0, expiry="2026-06-19",
                   strike=100.0, option_type="call")
    s.record_trade("TSLA", "SELL_CALL", 1, 3.0, expiry="2026-06-19",
                   strike=100.0, option_type="call")
    # No open positions; give the portfolio a non-zero total so the sector
    # branch doesn't divide by zero (irrelevant to the round-trip asserts).
    s.update_portfolio(cash=1130.0, total_value=1130.0, positions=[])

    from paper_trader import dashboard
    dashboard.app.config["TESTING"] = True
    try:
        with dashboard.app.test_client() as client:
            yield client
    finally:
        s.close()


class TestAnalyticsRoundTripMetrics:
    def test_round_trip_aggregates_are_exact(self, seeded_client):
        resp = seeded_client.get("/api/analytics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "error" not in data, data

        # round_trips pnl = [+20, -20, +30, +100]
        assert data["n_round_trips"] == 4
        assert data["win_rate_pct"] == 75.0          # 3 of 4
        assert data["avg_winner_usd"] == 50.0        # (20+30+100)/3
        assert data["avg_loser_usd"] == -20.0
        assert data["realized_pl_usd"] == 130.0      # 20-20+30+100
        assert data["profit_factor"] == 7.5          # 150 / 20
        # All trades recorded back-to-back -> sub-second holds. The list is
        # non-empty so it must not be None; value is ~0 (allow slack so a
        # slow unattended CI run can't flake on the 4dp hold_days rounding).
        assert data["avg_holding_days"] is not None
        assert 0.0 <= data["avg_holding_days"] < 0.01

    def test_open_position_is_excluded_from_round_trips(self, tmp_path, monkeypatch):
        db = tmp_path / "pt.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        # One closed winner + one still-open lot (BUY with no closing SELL).
        s.record_trade("AAPL", "BUY", 1, 10.0)
        s.record_trade("AAPL", "SELL", 1, 15.0)
        s.record_trade("MU", "BUY", 100, 5.0)  # open, must not count
        s.update_portfolio(cash=505.0, total_value=505.0, positions=[])

        from paper_trader import dashboard
        try:
            with dashboard.app.test_client() as client:
                data = client.get("/api/analytics").get_json()
            assert data["n_round_trips"] == 1
            assert data["win_rate_pct"] == 100.0
            assert data["realized_pl_usd"] == 5.0
            assert data["profit_factor"] is None  # no losses -> gross_loss 0
        finally:
            s.close()

    def test_no_trades_yields_null_metrics(self, tmp_path, monkeypatch):
        db = tmp_path / "pt2.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        from paper_trader import dashboard
        try:
            with dashboard.app.test_client() as client:
                data = client.get("/api/analytics").get_json()
            assert data["n_round_trips"] == 0
            assert data["win_rate_pct"] is None
            assert data["profit_factor"] is None
            assert data["avg_holding_days"] is None
            assert data["realized_pl_usd"] == 0.0
        finally:
            s.close()


class TestCalmarBaseline:
    """Calmar's annualized-return numerator must derive its baseline from the
    store INITIAL_CASH constant, never a hardcoded 1000.0. A literal there
    silently desyncs Calmar the moment INITIAL_CASH moves (the same
    hardcoded-baseline desync class fixed in reporter.py, commit 2a154df).

    This pins the *number*, not just the import: the test patches the baseline
    to a non-1000 value, so a future re-hardcode of `1000.0` makes the endpoint
    disagree with the expected formula and this test fails.
    """

    def test_calmar_uses_store_initial_cash_baseline(self, tmp_path, monkeypatch):
        db = tmp_path / "calmar.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()

        # 22 distinct calendar days of equity → 21 daily returns (Calmar needs
        # >= 20). First point is the global peak (1200); the post-peak trough
        # is exactly 900 → max drawdown = (1200-900)/1200 = 25.00%. No later
        # point exceeds the peak, so the running-peak max-DD scan is exact.
        values = [1200, 1150, 1100, 1050, 1000, 950, 900, 950, 1000, 1020,
                  1030, 1040, 1050, 1060, 1070, 1080, 1085, 1090, 1093, 1096,
                  1098, 1100]
        assert len(values) == 22
        rows = [
            (f"2026-01-{day:02d}T20:00:00+00:00", float(v), float(v), None)
            for day, v in enumerate(values, start=1)
        ]
        s.conn.executemany(
            "INSERT INTO equity_curve (timestamp, total_value, cash, sp500_price) "
            "VALUES (?,?,?,?)",
            rows,
        )
        s.conn.commit()
        # Final mark-to-market equity = last curve point.
        s.update_portfolio(cash=1100.0, total_value=1100.0, positions=[])

        from paper_trader import dashboard

        # Patch the baseline to a value that is NOT 1000 so a re-hardcoded
        # literal would produce a different total_return_pct → different Calmar.
        BASELINE = 500.0
        monkeypatch.setattr(dashboard, "INITIAL_CASH", BASELINE)

        try:
            with dashboard.app.test_client() as client:
                data = client.get("/api/analytics").get_json()
        finally:
            s.close()

        assert "error" not in data, data
        assert data["max_drawdown_pct"] == 25.0
        assert data["n_trading_days"] == 21  # 22 days → 21 returns

        # Recompute Calmar with the same formula analytics_api uses, but with
        # the patched baseline — exactly what a correct implementation yields.
        total_return_pct = (1100.0 / BASELINE - 1.0) * 100.0   # 120.0
        years = 22 / 252.0                                     # day_keys = 22
        expected_calmar = round((total_return_pct / years) / 25.0, 2)
        assert expected_calmar == 54.98  # sanity-pin the literal too
        assert data["calmar_ratio"] == expected_calmar
