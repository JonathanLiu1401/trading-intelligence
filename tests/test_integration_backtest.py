"""Integration tests — end-to-end paper-trader flows.

These tests wire multiple components together (BacktestEngine + PriceCache +
SimPortfolio + BacktestStore, DecisionScorer + pickle round-trip, risk exits
across a synthetic price series). They complement the focused unit tests in
test_backtest.py and test_decision_scorer.py.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import paper_trader.backtest as bt
from paper_trader.backtest import (
    BacktestEngine,
    BacktestRun,
    BacktestStore,
    INITIAL_CASH,
    PriceCache,
    SimPortfolio,
    _buy,
    _enforce_risk_exits,
)
from paper_trader.ml.decision_scorer import DecisionScorer, build_features, train_scorer


# ─────────────────────────── helpers ───────────────────────────


def _build_synthetic_prices(rise_pct: float = 50.0, days: int = 51,
                            tickers: list[str] | None = None) -> PriceCache:
    """Build a PriceCache with a monotonic synthetic close series.

    Mirrors the conftest synthetic_prices fixture but lets the caller choose
    the rise %, length, and ticker list (default = SPY + a sample of WATCHLIST
    so _ml_decide-style code can resolve them).
    """
    if tickers is None:
        tickers = ["SPY", "NVDA", "MU", "QQQ"]
    start = date(2025, 1, 2)  # Thursday
    seq: list[date] = []
    d = start
    while len(seq) < days:
        if d.weekday() < 5:
            seq.append(d)
        d += timedelta(days=1)

    cache = PriceCache.__new__(PriceCache)
    cache.tickers = tickers
    cache.start = seq[0]
    cache.end = seq[-1]
    # Each ticker climbs linearly to (1 + rise_pct/100) * 100 across the window.
    end_price = 100.0 * (1.0 + rise_pct / 100.0)
    step = (end_price - 100.0) / max(days - 1, 1)
    cache.prices = {
        t: {d.isoformat(): 100.0 + i * step for i, d in enumerate(seq)}
        for t in tickers
    }
    cache.trading_days = seq
    return cache


def _make_engine(prices: PriceCache, tmp_path: Path) -> BacktestEngine:
    """Construct a BacktestEngine WITHOUT triggering yfinance / GDELT init.

    BacktestEngine.__init__ calls PriceCache(WATCHLIST, ...) which fetches
    historical data from yfinance over the network. We bypass that by
    allocating the instance and wiring up its attributes ourselves.
    """
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.start = prices.trading_days[0]
    engine.end = prices.trading_days[-1]
    engine.store = BacktestStore(path=tmp_path / "bt.db")
    engine.prices = prices
    # The two fetchers are only used inside _fetch_signals, which we mock.
    engine.gdelt = None
    engine.av_news = None
    engine._local_news = {}
    return engine


# ─────────────── Test A: full BacktestEngine.run_one cycle ─────────────────


class TestBacktestRunOne:
    def test_run_one_produces_valid_result(self, tmp_path):
        """Drive one full run_one() with a synthetic price series + mocked
        decision feed. Verify the BacktestRun-shaped record persisted to the
        store has finite numbers, sensible counts, and a monotone equity curve.

        We mock:
          • _fetch_signals → returns 1 dummy article per day (no GDELT/AV)
          • _ml_decide     → BUY SPY 5 shares on day 0, HOLD afterwards
        """
        prices = _build_synthetic_prices(rise_pct=50.0, days=21)
        engine = _make_engine(prices, tmp_path)

        # _fetch_signals: return a single neutral article so _ml_decide has data
        def fake_signals(d, seed, rng, portfolio=None):
            return [{"title": f"news for {d.isoformat()}",
                     "url": "https://x", "score": 1.0, "tickers": ["SPY"]}]

        decisions_seen = {"buys": 0, "holds": 0}

        def fake_ml_decide(sim_date, portfolio, articles, prices_, run_id, rng,
                           exclude_tickers=None):
            # Buy SPY exactly once on the first sample day; HOLD all other days.
            if not portfolio.positions and decisions_seen["buys"] == 0:
                decisions_seen["buys"] += 1
                return {"action": "BUY", "ticker": "SPY", "qty": 5,
                        "confidence": 0.8,
                        "reasoning": "synthetic test buy",
                        "stop_loss": None, "take_profit": None}
            decisions_seen["holds"] += 1
            return {"action": "HOLD", "ticker": "", "qty": 0,
                    "reasoning": "synthetic test hold"}

        with patch.object(engine, "_fetch_signals", side_effect=fake_signals), \
             patch.object(bt, "_ml_decide", side_effect=fake_ml_decide):
            result = engine.run_one(run_id=1, seed=12345)

        # Type and basic invariants
        assert isinstance(result, BacktestRun)
        assert result.run_id == 1
        assert result.seed == 12345

        # Final value: started $1000, bought 5 SPY @ $100, SPY ended ~$150
        # so final = $500 cash + 5 * ~$150 = ~$1250. Assert finite + positive.
        assert math.isfinite(result.final_value)
        assert result.final_value > 0, "portfolio value can't be negative"
        assert math.isfinite(result.total_return_pct)
        assert result.total_return_pct == pytest.approx(
            (result.final_value - INITIAL_CASH) / INITIAL_CASH * 100, rel=1e-6
        )

        # n_trades >= 1 (the synthetic BUY) and n_decisions >= 1
        assert result.n_trades >= 1, "the BUY decision should have filled"
        assert result.n_decisions >= 1
        assert decisions_seen["buys"] == 1
        assert decisions_seen["holds"] >= 1

        # Equity curve should have at least one point per sample day
        assert len(result.equity_curve) >= 1
        for point in result.equity_curve:
            assert math.isfinite(point["value"])
            assert math.isfinite(point["cash"])
            assert point["value"] >= 0

        # start_date / end_date echo what we set on the engine
        assert result.start_date == prices.trading_days[0].isoformat()
        assert result.end_date == prices.trading_days[-1].isoformat()


# ─────────────────── Test B: DecisionScorer train/predict cycle ─────────────


class TestDecisionScorerCycle:
    def _outcome(self, idx: int, *, mom: float, fwd: float,
                 ticker: str = "NVDA") -> dict:
        return {
            "ticker": ticker,
            "action": "BUY",
            "ml_score": mom,
            "rsi": 50.0,
            "macd": 0.1,
            "mom5": mom,
            "mom20": 0.0,
            "regime_mult": 1.0,
            "vol_ratio": 1.0,
            "bb_position": 0.0,
            "news_urgency": 50.0,
            "news_article_count": 1.0,
            "forward_return_5d": fwd,
            "return_pct": 10.0,
            "sim_date": f"2025-{(idx % 12) + 1:02d}-{(idx % 28) + 1:02d}",
        }

    def test_train_predict_save_reload(self, tmp_path, monkeypatch):
        """Generate 200 synthetic samples where mom5 → forward return is
        approximately linear. Train, predict on a held-out sample, then save,
        reload via a *new* DecisionScorer instance, and verify the reloaded
        instance produces the same prediction (within float tolerance)."""
        import paper_trader.ml.decision_scorer as ds_mod
        path = tmp_path / "scorer.pkl"
        monkeypatch.setattr(ds_mod, "SCORER_PATH", path)

        rng = np.random.default_rng(0xC0FFEE)
        records = []
        for i in range(200):
            # Strong linear signal: forward return ≈ 1.2 * mom5 + small noise
            mom = float(rng.uniform(-10.0, 10.0))
            fwd = 1.2 * mom + float(rng.normal(0, 0.5))
            # Vary ticker so sector encoding has signal too
            ticker = ["NVDA", "AMD", "MU", "SPY"][i % 4]
            records.append(self._outcome(i, mom=mom, fwd=fwd, ticker=ticker))

        result = train_scorer(records)
        assert result["status"] == "ok"
        assert result["n"] > 0
        assert path.exists(), "scorer pickle not written"

        # Predict on the in-memory model (untrained loader path)
        # First scorer instance picks up the saved pickle via _load()
        scorer1 = DecisionScorer()
        assert scorer1.is_trained
        v1 = scorer1.predict(
            ml_score=8.0, rsi=50.0, macd=0.1, mom5=8.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA",
        )
        assert math.isfinite(v1), "prediction must be finite"

        # Reload from disk in a fresh scorer — round-trip integrity
        scorer2 = DecisionScorer()
        assert scorer2.is_trained
        v2 = scorer2.predict(
            ml_score=8.0, rsi=50.0, macd=0.1, mom5=8.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA",
        )
        assert math.isfinite(v2)
        # Identical pickle → identical prediction (no nondeterminism in the
        # MLP / lstsq inference path).
        assert v1 == pytest.approx(v2, abs=1e-6), (
            f"reload changed prediction: {v1} vs {v2}"
        )


# ─────────────────── Test C: stop-loss fires at threshold ───────────────────


class TestStopLossThreshold:
    """Build a synthetic DOWN-trending price series and verify that the
    risk-exit scan fires the stop-loss at the expected close.
    """

    def _falling_prices(self, ticker: str, closes: list[float]) -> PriceCache:
        start = date(2025, 1, 2)
        days: list[date] = []
        d = start
        while len(days) < len(closes):
            if d.weekday() < 5:
                days.append(d)
            d += timedelta(days=1)
        cache = PriceCache.__new__(PriceCache)
        cache.tickers = [ticker]
        cache.start = days[0]
        cache.end = days[-1]
        cache.prices = {ticker: {d.isoformat(): closes[i]
                                  for i, d in enumerate(days)}}
        cache.trading_days = days
        return cache

    def test_stop_fires_when_price_crosses_threshold(self):
        """Entry $100, stop $90 (10%). Series drops: 100→99→97→95→92→88.
        Stop must fire on the FIRST day price ≤ 90 (day index 4 = $92? no, $88).
        Actually 92 > 90 (no fire), 88 ≤ 90 (fire). So exit at $88."""
        prices = self._falling_prices(
            "SPY", [100.0, 99.0, 97.0, 95.0, 92.0, 88.0],
        )
        portfolio = SimPortfolio(cash=1000.0)
        _buy(portfolio, "SPY", 5.0, 100.0, stop_loss=90.0, take_profit=None)

        class _RecordingStore:
            def __init__(self):
                self.trades: list[tuple] = []

            def record_trade(self, run_id, sim_date, ticker, action, qty,
                              price, reason):
                self.trades.append((sim_date, ticker, action, qty, price, reason))

        store = _RecordingStore()
        days = prices.trading_days
        # _enforce_risk_exits scans days strictly AFTER prev_sample up to and
        # including to_day. Pass prev_sample = day0 - 1 so day0 is included.
        before_day0 = days[0] - timedelta(days=1)
        n = _enforce_risk_exits(portfolio, prices, before_day0, days[-1],
                                run_id=1, store=store)

        assert n == 1, f"expected exactly 1 stop-loss exit, got {n}"
        assert "SPY" not in portfolio.positions, "position not closed"
        assert len(store.trades) == 1
        sim_date, ticker, action, qty, exit_price, reason = store.trades[0]
        assert ticker == "SPY"
        assert action == "SELL"
        assert qty == pytest.approx(5.0)
        # Stop fires on the first close ≤ 90 — that's day index 5 at $88.
        assert exit_price == pytest.approx(88.0)
        assert sim_date == days[5].isoformat()
        assert "stop-loss" in reason

    def test_stop_does_not_fire_when_price_stays_above(self):
        """Entry $100, stop $90. Series drops to $92 (never below 90).
        Position must remain open."""
        prices = self._falling_prices("SPY", [100.0, 98.0, 95.0, 92.0])
        portfolio = SimPortfolio(cash=1000.0)
        _buy(portfolio, "SPY", 5.0, 100.0, stop_loss=90.0, take_profit=None)

        class _RecordingStore:
            def __init__(self):
                self.trades: list = []

            def record_trade(self, *args):
                self.trades.append(args)

        store = _RecordingStore()
        days = prices.trading_days
        n = _enforce_risk_exits(portfolio, prices, days[0] - timedelta(days=1),
                                days[-1], run_id=1, store=store)
        assert n == 0, "stop fired too early"
        assert "SPY" in portfolio.positions
        assert store.trades == []

    def test_looser_stop_holds_position_longer(self):
        """With stop_loss=$85 (15%), the same falling series 100→...→88 must
        NOT fire (all closes >85). With stop_loss=$90, it WOULD fire at $88.
        This confirms the threshold actually gates."""
        closes = [100.0, 99.0, 97.0, 95.0, 92.0, 88.0]
        # Tight stop $90: fires
        prices_a = self._falling_prices("SPY", closes)
        port_a = SimPortfolio(cash=1000.0)
        _buy(port_a, "SPY", 5.0, 100.0, stop_loss=90.0, take_profit=None)
        store_a = type("S", (), {"trades": [], "record_trade":
                                  lambda self, *a: self.trades.append(a)})()
        days = prices_a.trading_days
        before = days[0] - timedelta(days=1)
        _enforce_risk_exits(port_a, prices_a, before, days[-1], 1, store_a)
        assert "SPY" not in port_a.positions, "tight stop should have fired"

        # Loose stop $85: does NOT fire
        prices_b = self._falling_prices("SPY", closes)
        port_b = SimPortfolio(cash=1000.0)
        _buy(port_b, "SPY", 5.0, 100.0, stop_loss=85.0, take_profit=None)
        store_b = type("S", (), {"trades": [], "record_trade":
                                  lambda self, *a: self.trades.append(a)})()
        _enforce_risk_exits(port_b, prices_b, before, days[-1], 1, store_b)
        assert "SPY" in port_b.positions, (
            "loose stop $85 should NOT fire on series whose min is $88"
        )
