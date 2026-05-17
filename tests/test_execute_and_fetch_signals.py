"""Regression locks for two backtest seams that had NO direct unit coverage.

Found during the ML+backtest review pass (2026-05-16): `_execute_decision`
and `_fetch_signals` were only exercised *indirectly* through
`test_integration_backtest.py`, where `_fetch_signals` is fully mocked out —
so its real ranking + URL-dedup logic was never verified, and the
`_execute_decision` BLOCKED/clamp branches were never asserted.

These tests assert exact expected values so a real logic regression fails
loudly:

  * `_fetch_signals` must NOT collapse multiple URL-less articles into one
    (documented invariant: "Skip dedup for empty URLs"), MUST dedup repeated
    real URLs, and MUST keep only the top-10-by-score before sampling 5.
  * `_execute_decision` must allow an exact-cash BUY (`cash - notional == 0`),
    block a BUY that overspends by a cent, and clamp a SELL to the held qty
    (position size can never be exceeded — the task's explicit invariant).

All offline: a far-past `sim_date` makes `_fetch_signals` skip the yfinance
(today-30d) and Alpha-Vantage (today-14d) tiers, and a non-empty
`_local_news` makes it skip the GDELT tier — so no network seat is reached.
"""
from __future__ import annotations

import random
from datetime import date

import pytest

from paper_trader.backtest import (
    BacktestEngine,
    BacktestStore,
    SimPortfolio,
)


# ───────────────────────── _fetch_signals ─────────────────────────


def _engine_with_local_news(tmp_path, local_news: dict) -> BacktestEngine:
    """Bare engine wired only with what `_fetch_signals` tier-1 touches.

    `gdelt`/`av_news` are left as None on purpose: a far-past date plus a
    non-empty article list means neither is ever dereferenced. If a
    regression made `_fetch_signals` reach them, the test would crash with
    AttributeError — itself a useful signal.
    """
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.start = date(2020, 1, 1)
    engine.end = date(2020, 12, 31)
    engine.store = BacktestStore(path=tmp_path / "bt.db")
    engine.prices = None
    engine.gdelt = None
    engine.av_news = None
    engine._local_news = local_news
    return engine


# A date far enough in the past that the today-30d (yfinance) and today-14d
# (Alpha Vantage) guards in _fetch_signals are both False.
_PAST = date(2020, 6, 1)


class TestFetchSignalsUrlDedup:
    def test_empty_url_articles_are_not_collapsed(self, tmp_path):
        """The documented invariant: URL-less articles must each survive —
        an empty string must never enter `seen_urls` and dedup them away."""
        day = _PAST.isoformat()
        news = [
            {"title": "alpha headline", "url": "", "score": 4.0, "urgency": 0.0},
            {"title": "bravo headline", "url": "", "score": 3.0, "urgency": 0.0},
            {"title": "charlie headline", "url": "", "score": 2.0, "urgency": 0.0},
            {"title": "delta headline", "url": "", "score": 1.5, "urgency": 0.0},
        ]
        engine = _engine_with_local_news(tmp_path, {day: news})

        out = engine._fetch_signals(_PAST, seed=0, rng=random.Random(1))

        # 4 distinct URL-less articles in, 4 out (<=5 → no sampling), sorted desc.
        assert [a["title"] for a in out] == [
            "alpha headline", "bravo headline",
            "charlie headline", "delta headline",
        ]
        assert [a["score"] for a in out] == [4.0, 3.0, 2.0, 1.5]

    def test_repeated_real_url_is_deduped(self, tmp_path):
        """A repeated non-empty URL must collapse to a single article."""
        day = _PAST.isoformat()
        news = [
            {"title": "first", "url": "https://x/a", "score": 5.0, "urgency": 0.0},
            {"title": "dup of first", "url": "https://x/a", "score": 4.0, "urgency": 0.0},
            {"title": "other", "url": "https://x/b", "score": 3.0, "urgency": 0.0},
        ]
        engine = _engine_with_local_news(tmp_path, {day: news})

        out = engine._fetch_signals(_PAST, seed=0, rng=random.Random(1))

        assert len(out) == 2
        urls = {a["url"] for a in out}
        assert urls == {"https://x/a", "https://x/b"}
        # The first occurrence of the duplicated URL is the one kept.
        kept = next(a for a in out if a["url"] == "https://x/a")
        assert kept["title"] == "first"


class TestFetchSignalsRanking:
    def test_exactly_five_returns_all_sorted_desc(self, tmp_path):
        day = _PAST.isoformat()
        news = [
            {"title": f"h{i}", "url": f"https://x/{i}", "score": float(i),
             "urgency": 0.0}
            for i in (3, 1, 5, 2, 4)
        ]
        engine = _engine_with_local_news(tmp_path, {day: news})

        out = engine._fetch_signals(_PAST, seed=0, rng=random.Random(1))

        assert [a["score"] for a in out] == [5.0, 4.0, 3.0, 2.0, 1.0]

    def test_only_top_ten_by_score_survive_sampling(self, tmp_path):
        """12 articles → the 2 lowest-scored are sliced off by `articles[:10]`
        BEFORE the rng.sample(5), so they can never appear in the result,
        regardless of the RNG seed. This locks the top-10 cut."""
        day = _PAST.isoformat()
        news = [
            {"title": f"h{i}", "url": f"https://x/{i}", "score": float(i),
             "urgency": 0.0}
            for i in range(1, 13)  # scores 1.0 .. 12.0
        ]
        engine = _engine_with_local_news(tmp_path, {day: news})

        for seed in range(25):  # seed-independent invariant
            out = engine._fetch_signals(_PAST, seed=0, rng=random.Random(seed))
            scores = [a["score"] for a in out]
            assert len(out) == 5
            assert len(set(a["url"] for a in out)) == 5  # no dup
            assert scores == sorted(scores, reverse=True)  # always sorted desc
            # scores 1.0 and 2.0 are outside the top-10 → must never appear.
            assert 1.0 not in scores
            assert 2.0 not in scores
            assert min(scores) >= 3.0


# ───────────────────────── _execute_decision ─────────────────────────


def _engine_with_prices(synthetic_prices, tmp_path) -> BacktestEngine:
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.start = synthetic_prices.trading_days[0]
    engine.end = synthetic_prices.trading_days[-1]
    engine.store = BacktestStore(path=tmp_path / "bt.db")
    engine.prices = synthetic_prices
    engine.gdelt = None
    engine.av_news = None
    engine._local_news = {}
    return engine


class TestExecuteDecisionBoundaries:
    def test_exact_cash_buy_is_filled_not_blocked(self, synthetic_prices,
                                                  tmp_path):
        """`cash - notional == 0` must be allowed (full deployment). A
        regression flipping the guard to `<= 0` would block this."""
        engine = _engine_with_prices(synthetic_prices, tmp_path)
        d0 = synthetic_prices.trading_days[0]
        price = synthetic_prices.price_on("SPY", d0)
        assert price == 100.0  # synthetic_prices fixture invariant

        pf = SimPortfolio(cash=500.0)  # exactly 5 * 100
        status, detail = engine._execute_decision(
            run_id=1, sim_date=d0,
            decision={"action": "BUY", "ticker": "SPY", "qty": 5,
                      "reasoning": "exact cash"},
            portfolio=pf,
        )

        assert status == "FILLED", detail
        assert pf.cash == pytest.approx(0.0)
        assert pf.positions["SPY"]["qty"] == 5

    def test_buy_one_cent_over_cash_is_blocked(self, synthetic_prices,
                                               tmp_path):
        engine = _engine_with_prices(synthetic_prices, tmp_path)
        d0 = synthetic_prices.trading_days[0]

        pf = SimPortfolio(cash=499.99)
        status, detail = engine._execute_decision(
            run_id=1, sim_date=d0,
            decision={"action": "BUY", "ticker": "SPY", "qty": 5,
                      "reasoning": "overspend"},
            portfolio=pf,
        )

        assert status == "BLOCKED"
        assert detail.startswith("insufficient cash")
        assert pf.positions == {}
        assert pf.cash == pytest.approx(499.99)  # untouched

    def test_sell_qty_is_clamped_to_held_position(self, synthetic_prices,
                                                  tmp_path):
        """Position size can never be exceeded: SELL 10 of a 3-share holding
        liquidates exactly 3 (the task's explicit invariant)."""
        engine = _engine_with_prices(synthetic_prices, tmp_path)
        d0 = synthetic_prices.trading_days[0]
        price = synthetic_prices.price_on("SPY", d0)  # 100.0

        pf = SimPortfolio(cash=0.0)
        pf.positions["SPY"] = {"qty": 3, "avg_cost": 90.0,
                               "stop_loss": None, "take_profit": None}

        status, detail = engine._execute_decision(
            run_id=1, sim_date=d0,
            decision={"action": "SELL", "ticker": "SPY", "qty": 10,
                      "reasoning": "oversell"},
            portfolio=pf,
        )

        assert status == "FILLED", detail
        assert "SPY" not in pf.positions          # fully closed, not negative
        assert pf.cash == pytest.approx(3 * price)  # only 3 shares' proceeds
        assert "3" in detail                      # clamped qty echoed in detail

    def test_sell_with_no_position_is_blocked(self, synthetic_prices,
                                              tmp_path):
        """NVDA has a price in the fixture but is not held → the SELL must
        be rejected by the no-position guard, not the no-price guard."""
        engine = _engine_with_prices(synthetic_prices, tmp_path)
        d0 = synthetic_prices.trading_days[0]
        assert synthetic_prices.price_on("NVDA", d0) is not None

        pf = SimPortfolio(cash=1000.0)
        status, detail = engine._execute_decision(
            run_id=1, sim_date=d0,
            decision={"action": "SELL", "ticker": "NVDA", "qty": 1,
                      "reasoning": "phantom"},
            portfolio=pf,
        )

        assert status == "BLOCKED"
        assert detail == "no open position in NVDA"
        assert pf.cash == pytest.approx(1000.0)
