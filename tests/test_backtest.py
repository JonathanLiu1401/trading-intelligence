"""Tests for paper_trader.backtest.

Targets the deterministic, network-free parts of the engine: price cache
lookups, portfolio bookkeeping, risk exits, technical indicator math,
heuristic scorer, and the _ml_decide branching logic.
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from paper_trader.backtest import (
    INITIAL_CASH,
    PERSONAS,
    PriceCache,
    SimPortfolio,
    _article_sentiment,
    _buy,
    _compute_technical_indicators,
    _ema,
    _enforce_risk_exits,
    _ml_decide,
    _parse_decision,
    _rsi,
    _sell,
    persona_for,
    score_article,
)


# ─────────────────────── PriceCache ───────────────────────────

class TestPriceCache:
    def test_price_on_exact_match(self, synthetic_prices):
        # First synthetic price is 100.0 at days[0].
        d0 = synthetic_prices.trading_days[0]
        assert synthetic_prices.price_on("SPY", d0) == 100.0

    def test_price_on_missing_walks_back(self, synthetic_prices):
        # Saturday/Sunday between weekdays: should walk back to Friday's close.
        d0 = synthetic_prices.trading_days[0]
        weekend = d0 + timedelta(days=1)
        # weekend is a Friday (since d0 is Thu) — exists. Skip 2 more.
        sat = d0 + timedelta(days=2)
        # If sat happens to be a trading day in our synthetic series, this still
        # returns a real close. If not, walks back.
        v = synthetic_prices.price_on("SPY", sat)
        assert v is not None
        assert isinstance(v, float)

    def test_price_on_unknown_ticker(self, synthetic_prices):
        assert synthetic_prices.price_on("ZZZNOTREAL", date(2025, 1, 2)) is None

    def test_returns_pct_correctness(self, synthetic_prices):
        # SPY rises 100 → 100 + 50 = 150 across 51 days. returns_pct must be +50%.
        days = synthetic_prices.trading_days
        ret = synthetic_prices.returns_pct("SPY", days[0], days[-1])
        # Synthetic series: SPY[i] = 100 + i for i in 0..50, so days[-1] = 150.
        assert ret == pytest.approx(50.0, rel=1e-6)

    def test_returns_pct_missing_ticker(self, synthetic_prices):
        ret = synthetic_prices.returns_pct("ZZZ", date(2025, 1, 1), date(2025, 1, 2))
        assert ret == 0.0


# ─────────────────────── SimPortfolio ───────────────────────────

class TestSimPortfolio:
    def test_initial_state(self):
        p = SimPortfolio()
        assert p.cash == INITIAL_CASH
        assert p.positions == {}

    def test_buy_records_position(self):
        p = SimPortfolio(cash=1000.0)
        _buy(p, "NVDA", 5.0, 100.0, stop_loss=90.0, take_profit=120.0)
        assert p.cash == 500.0  # 1000 - 5*100
        assert "NVDA" in p.positions
        assert p.positions["NVDA"]["qty"] == 5.0
        assert p.positions["NVDA"]["avg_cost"] == 100.0
        assert p.positions["NVDA"]["stop_loss"] == 90.0
        assert p.positions["NVDA"]["take_profit"] == 120.0

    def test_buy_blends_avg_cost(self):
        p = SimPortfolio(cash=1000.0)
        _buy(p, "NVDA", 5.0, 100.0, stop_loss=None, take_profit=None)
        _buy(p, "NVDA", 5.0, 200.0, stop_loss=None, take_profit=None)
        # avg = (5*100 + 5*200) / 10 = 150
        assert p.positions["NVDA"]["qty"] == 10.0
        assert p.positions["NVDA"]["avg_cost"] == pytest.approx(150.0)

    def test_sell_full_position_removes_it(self):
        p = SimPortfolio(cash=500.0)
        _buy(p, "NVDA", 5.0, 100.0, stop_loss=None, take_profit=None)
        proceeds = _sell(p, "NVDA", 5.0, 120.0)
        assert proceeds == 600.0  # 5 * 120
        # Position should be removed when qty hits 0.
        assert "NVDA" not in p.positions
        assert p.cash == pytest.approx(500.0 - 500.0 + 600.0)

    def test_sell_caps_at_held_qty(self):
        """Critical invariant: oversell must be clamped, not blow through to negative qty."""
        p = SimPortfolio(cash=500.0)
        _buy(p, "NVDA", 3.0, 100.0, stop_loss=None, take_profit=None)
        proceeds = _sell(p, "NVDA", 999.0, 110.0)
        # Only 3 shares could be sold.
        assert proceeds == 330.0
        assert "NVDA" not in p.positions

    def test_sell_unknown_ticker_returns_zero(self):
        p = SimPortfolio(cash=500.0)
        assert _sell(p, "ZZZ", 1.0, 50.0) == 0.0
        assert p.cash == 500.0  # unchanged

    def test_total_value_marks_to_market(self, synthetic_prices):
        p = SimPortfolio(cash=1000.0)
        _buy(p, "SPY", 5.0, 100.0, stop_loss=None, take_profit=None)
        # After buy: cash = 1000 - 5*100 = 500.
        # SPY[days[-1]] = 150 (the synthetic curve's last close).
        d = synthetic_prices.trading_days[-1]
        val = p.total_value(synthetic_prices, d)
        # Total = cash 500 + 5*150 = 1250.
        assert val == pytest.approx(1250.0)


# ─────────────────────── _enforce_risk_exits ───────────────────────────

class TestRiskExits:
    def test_stop_loss_below_all_prices_never_fires(self, synthetic_prices):
        """Stop set below the entire price series must never fire — the position survives."""
        p = SimPortfolio(cash=1000.0)
        # Synthetic SPY ranges 100..150. Stop at 50 is below all closes.
        _buy(p, "SPY", 5.0, 100.0, stop_loss=50.0, take_profit=None)

        store = MagicMock()
        days = synthetic_prices.trading_days
        n_exits = _enforce_risk_exits(p, synthetic_prices, days[0], days[-1],
                                       run_id=1, store=store)
        assert n_exits == 0
        assert "SPY" in p.positions

    def test_take_profit_triggers(self, synthetic_prices):
        p = SimPortfolio(cash=500.0)
        # Synthetic SPY: 100 → 150 over 51 days. TP at 120 should fire ~day 20.
        _buy(p, "SPY", 5.0, 100.0, stop_loss=None, take_profit=120.0)

        store = MagicMock()
        days = synthetic_prices.trading_days
        n_exits = _enforce_risk_exits(p, synthetic_prices, days[0], days[-1],
                                       run_id=1, store=store)
        assert n_exits == 1
        assert "SPY" not in p.positions
        # The trade must have been recorded.
        store.record_trade.assert_called_once()
        # Trade price recorded should match the TP trigger close (≥120).
        called_args = store.record_trade.call_args[0]
        triggered_price = called_args[5]  # positional: run_id, sim_date, ticker, action, qty, price
        assert triggered_price >= 120.0

    def test_stop_loss_actually_fires(self, synthetic_prices):
        """Force a stop-loss by setting it above the synthetic series' max close.
        That guarantees ALL subsequent closes are below the stop trigger."""
        # We need a downward stop — set entry where stop > current price.
        # The synthetic SPY series rises 100→150. Build a scenario where we
        # entered at 200 (above series), stop at 180 (still above series max).
        # When the daily scan sees first close (100) ≤ 180, stop fires.
        p = SimPortfolio(cash=1000.0)
        _buy(p, "SPY", 1.0, 200.0, stop_loss=180.0, take_profit=None)
        store = MagicMock()
        days = synthetic_prices.trading_days
        n_exits = _enforce_risk_exits(p, synthetic_prices, days[0], days[-1],
                                       run_id=1, store=store)
        assert n_exits == 1
        assert "SPY" not in p.positions

    def test_no_exits_when_no_positions(self, synthetic_prices):
        p = SimPortfolio()
        store = MagicMock()
        days = synthetic_prices.trading_days
        assert _enforce_risk_exits(p, synthetic_prices, days[0], days[-1], 1, store) == 0
        store.record_trade.assert_not_called()


# ─────────────────────── Buy-and-hold sanity ───────────────────────────

class TestBuyAndHold:
    def test_buy_and_hold_exact_return(self, synthetic_prices):
        """Buy at first close, mark to last close, return must equal series return."""
        p = SimPortfolio(cash=1000.0)
        days = synthetic_prices.trading_days
        d0_price = synthetic_prices.price_on("SPY", days[0])
        # All-in on SPY: qty = 1000 / 100 = 10
        qty = 10.0
        _buy(p, "SPY", qty, d0_price, stop_loss=None, take_profit=None)
        # Now mark to market at days[-1] (price = 150).
        end_val = p.total_value(synthetic_prices, days[-1])
        # Final = cash + qty * 150 = 0 + 1500 = 1500
        assert end_val == pytest.approx(1500.0)
        # Return = 50% — same as the SPY series return.
        ret_pct = (end_val - 1000.0) / 1000.0 * 100
        assert ret_pct == pytest.approx(50.0)


# ─────────────────────── technical indicators ───────────────────────────

class TestTechnicalIndicators:
    def test_rsi_constant_series_returns_max(self):
        # All gains, no losses → avg_l = 0 → RSI = 100.
        closes = list(range(1, 30))
        rsi = _rsi(closes, period=14)
        assert rsi == 100.0

    def test_rsi_insufficient_history(self):
        assert _rsi([1, 2, 3], period=14) is None

    def test_rsi_falling_series(self):
        # Monotonically falling — RSI must be < 30 (oversold).
        closes = [100 - i for i in range(30)]
        rsi = _rsi(closes, period=14)
        assert rsi is not None
        assert rsi < 30

    def test_ema_too_short(self):
        assert _ema([1, 2, 3], 10) == []

    def test_compute_indicators_insufficient_history(self, synthetic_prices):
        # Synthetic SPY has only 51 closes — below the 60-day minimum.
        d = synthetic_prices.trading_days[-1]
        ind = _compute_technical_indicators("SPY", d, synthetic_prices)
        assert ind is None

    def test_compute_indicators_with_enough_history(self):
        """Build a 100-day price series, verify lowercase numeric keys come back."""
        cache = PriceCache.__new__(PriceCache)
        cache.tickers = ["TEST"]
        days = []
        d = date(2024, 1, 1)
        while len(days) < 100:
            if d.weekday() < 5:
                days.append(d)
            d += timedelta(days=1)
        cache.prices = {"TEST": {d.isoformat(): 100.0 + i for i, d in enumerate(days)}}
        cache.trading_days = days
        cache.start = days[0]
        cache.end = days[-1]

        # Patch _ensure_volume_for so it doesn't try to fetch volumes for "TEST".
        import paper_trader.backtest as bt
        with patch.object(bt, "_ensure_volume_for", return_value={}):
            ind = _compute_technical_indicators("TEST", days[-1], cache)
        assert ind is not None
        # Both legacy uppercase + new lowercase numeric keys must be present —
        # missing lowercase silently no-op'd quant adjustments in _ml_decide.
        assert "RSI" in ind
        assert "rsi" in ind
        assert "macd_signal" in ind  # numeric — used by scorer
        assert "mom_5d" in ind
        # Monotonic rising series → momentum positive.
        assert ind["mom_5d"] > 0


# ─────────────────────── heuristic scorer ───────────────────────────

class TestScoreArticle:
    def test_bullish_phrase_raises_score(self):
        s_bull, _ = score_article({"title": "NVDA beat earnings, guidance raised"})
        s_neutral, _ = score_article({"title": "Some random headline"})
        # The bull phrase title must score higher than the neutral one —
        # otherwise the kw_score signal is dead.
        assert s_bull > s_neutral

    def test_bearish_phrase_lowers_score(self):
        s_bear, _ = score_article({"title": "NVDA earnings miss; guidance cut"})
        s_neutral, _ = score_article({"title": "Some random headline"})
        assert s_bear < s_neutral

    def test_score_clamped_to_0_5(self):
        s, _ = score_article({"title": "rally surge beat earnings record breakthrough"})
        assert 0.0 <= s <= 5.0
        s, _ = score_article({
            "title": "bankruptcy fraud crash plunge miss earnings cut"
        })
        assert 0.0 <= s <= 5.0

    def test_extracted_tickers(self):
        _, tickers = score_article({"title": "NVDA beat earnings, AMD upgrade"})
        assert "NVDA" in tickers
        assert "AMD" in tickers
        # Common false-positives must NOT show as tickers.
        _, tickers = score_article({"title": "THE CEO of WHO said NEW data"})
        assert "THE" not in tickers
        assert "CEO" not in tickers
        assert "WHO" not in tickers


class TestArticleSentiment:
    def test_bullish_words_positive(self):
        v = _article_sentiment("NVDA beats earnings, surges higher")
        assert v > 0

    def test_bearish_words_negative(self):
        v = _article_sentiment("NVDA misses earnings, plunges lower")
        assert v < 0

    def test_neutral_zero(self):
        v = _article_sentiment("some random headline")
        assert v == 0.0


# ─────────────────────── persona_for ───────────────────────────

class TestPersonaFor:
    def test_run_id_1_maps_to_persona_1(self):
        assert persona_for(1)["name"] == PERSONAS[1]["name"]

    def test_run_id_cycles_after_10(self):
        # run_id 11 → persona 1 (since (11-1) % 10 + 1 = 1)
        assert persona_for(11)["name"] == PERSONAS[1]["name"]
        assert persona_for(20)["name"] == PERSONAS[10]["name"]
        assert persona_for(21)["name"] == PERSONAS[1]["name"]

    def test_zero_or_negative_run_id_handled(self):
        # Negative ids shouldn't crash — they cycle backwards through the persona map.
        assert persona_for(0)["name"] in {p["name"] for p in PERSONAS.values()}


# ─────────────────────── _parse_decision ───────────────────────────

class TestParseDecision:
    def test_plain_json(self):
        d = _parse_decision('{"action":"BUY","ticker":"NVDA","qty":1}')
        assert d == {"action": "BUY", "ticker": "NVDA", "qty": 1}

    def test_strips_json_fence(self):
        d = _parse_decision('```json\n{"action":"HOLD","ticker":"","qty":0}\n```')
        assert d == {"action": "HOLD", "ticker": "", "qty": 0}

    def test_returns_none_on_garbage(self):
        assert _parse_decision("definitely not JSON") is None

    def test_returns_none_on_empty(self):
        assert _parse_decision("") is None
        assert _parse_decision(None) is None

    def test_extracts_first_json_from_prose(self):
        raw = 'Here is the trade: {"action":"BUY","ticker":"AMD","qty":2} good luck'
        d = _parse_decision(raw)
        assert d["action"] == "BUY"
        assert d["ticker"] == "AMD"


# ─────────────────────── volume cache concurrency ─────────────────────

class TestVolumeCacheConcurrency:
    """Regression test for the `_persist_volume_cache_for_window` race.

    The persist helper used to iterate the shared `_VOLUME_CACHE` dict
    WITHOUT holding `_VOLUME_CACHE_LOCK`, while parallel run threads insert
    into it under that lock. The iterator is not protected by another
    thread's lock, so it raised `RuntimeError: dictionary changed size
    during iteration` — which was swallowed by the function's own
    try/except, silently breaking volume-cache persistence under the
    parallel continuous loop.
    """

    def test_persist_holds_lock_while_iterating_cache(self, monkeypatch):
        """Deterministic lock-discipline check (no thread-scheduling races).

        The fix wraps the `_VOLUME_CACHE.items()` snapshot in
        `with _VOLUME_CACHE_LOCK:`. We swap in a dict whose `.items()`
        records whether the cache lock is currently held at call time. A
        non-reentrant Lock cannot be re-acquired by the holder, so
        `acquire(blocking=False)` returning True proves NO thread holds it —
        i.e. the pre-fix unlocked iteration. The fixed code must produce
        zero such violations.
        """
        import json

        import paper_trader.backtest as bt

        start = date(2024, 1, 1)
        end = date(2024, 12, 31)
        s_iso, e_iso = start.isoformat(), end.isoformat()
        lock = bt._VOLUME_CACHE_LOCK
        violations: list[str] = []

        class _LockCheckingDict(dict):
            def items(self):
                # If we can grab the lock, nobody holds it → the caller
                # iterated the shared cache WITHOUT serialising. That is the
                # exact race that raised "dictionary changed size during
                # iteration" under the parallel continuous loop.
                if lock.acquire(blocking=False):
                    lock.release()
                    violations.append("items() called without _VOLUME_CACHE_LOCK")
                return super().items()

        cache = _LockCheckingDict()
        for i in range(50):
            cache[(f"T{i}", s_iso, e_iso)] = {"2024-01-02": float(i)}
        monkeypatch.setattr(bt, "_VOLUME_CACHE", cache)

        bt._persist_volume_cache_for_window(start, end)

        assert violations == [], violations
        # And the snapshot must still have been written correctly.
        path = bt._volume_cache_path(start, end)
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 50
        assert data["T0"] == {"2024-01-02": 0.0}


# ─────────────────────── _ml_decide (smoke tests) ──────────────────────

class TestMlDecide:
    def test_no_signals_returns_hold(self, synthetic_prices):
        # No articles, no positions — must yield HOLD without crashing.
        p = SimPortfolio()
        rng = random.Random(42)
        d = synthetic_prices.trading_days[-1]
        decision = _ml_decide(d, p, [], synthetic_prices, run_id=1, rng=rng)
        assert decision["action"] == "HOLD"

    def test_bullish_news_can_trigger_buy(self, synthetic_prices):
        # Build an article that should map to NVDA via _WORD_TO_TICKER and
        # produce a positive sentiment score.
        p = SimPortfolio(cash=1000.0)
        rng = random.Random(42)
        articles = [{
            "title": "Nvidia beats earnings, guidance raised, semiconductor surge",
            "score": 3.0,
            "tickers": ["NVDA"],
        }]
        d = synthetic_prices.trading_days[-1]
        decision = _ml_decide(d, p, articles, synthetic_prices, run_id=1, rng=rng)
        # Should be either BUY or HOLD depending on regime; must not crash.
        assert decision["action"] in {"BUY", "HOLD", "SELL"}

    def test_low_score_articles_filtered(self, synthetic_prices):
        p = SimPortfolio()
        rng = random.Random(42)
        # raw_score < 1.0 — _ml_decide skips these entirely.
        articles = [{"title": "x", "score": 0.5, "tickers": ["NVDA"]}]
        d = synthetic_prices.trading_days[-1]
        decision = _ml_decide(d, p, articles, synthetic_prices, run_id=1, rng=rng)
        # No effective signal — should be HOLD.
        assert decision["action"] == "HOLD"

    def test_none_score_article_does_not_crash(self, synthetic_prices):
        """Regression: a present-but-None `score` (malformed article dict) used
        to reach `float(None)` in _ml_decide and raise TypeError — uncaught,
        killing the whole run thread mid-cycle. It must now be coerced to 0.0
        and skipped as no-signal (the `< 1.0` guard), yielding a clean HOLD.

        `.get("score", 0.0)` does NOT default on a None VALUE, only a missing
        key — so this is distinct from `test_no_signals_returns_hold` (empty
        list) and from a missing-key article.
        """
        p = SimPortfolio(cash=1000.0)
        rng = random.Random(42)
        d = synthetic_prices.trading_days[-1]
        articles = [
            {"title": "Nvidia beats earnings, guidance raised",
             "score": None, "tickers": ["NVDA"]},          # the bug case
            {"title": "AMD upgrade", "tickers": ["AMD"]},   # missing key entirely
        ]
        # Must not raise; both articles carry no usable score → HOLD.
        decision = _ml_decide(d, p, articles, synthetic_prices,
                              run_id=1, rng=rng)
        assert decision["action"] == "HOLD"

    def test_oversize_buy_clipped_by_cash(self, synthetic_prices):
        """A BUY must never recommend a notional > 95% of available cash."""
        p = SimPortfolio(cash=100.0)
        rng = random.Random(42)
        # Make the article super-strong so conviction is at its max.
        articles = [{"title": "Nvidia beats earnings, guidance raised, semiconductor surge",
                     "score": 5.0, "tickers": ["NVDA"]}]
        d = synthetic_prices.trading_days[-1]
        decision = _ml_decide(d, p, articles, synthetic_prices, run_id=1, rng=rng)
        if decision["action"] == "BUY":
            price = synthetic_prices.price_on(decision["ticker"], d)
            notional = decision["qty"] * price
            # Must not exceed 95% of cash.
            assert notional <= 100.0 * 0.95 + 0.01  # tiny rounding allowance

    def test_conviction_caps_position_size_when_cash_is_abundant(
            self, synthetic_prices, monkeypatch):
        """Regression guard for the OTHER arm of position sizing.

        ``buy_notional = min(total_val * conviction, cash * 0.95)``.
        ``test_oversize_buy_clipped_by_cash`` only exercises the cash arm. With
        abundant cash the *conviction* cap is what bounds the trade — and a
        regression that dropped ``min(0.25, …)`` would let position size balloon
        with no other test catching it.

        Deterministic trace with ``synthetic_prices`` (51 trading days):
          - 51 SPY days < 200 → ``_market_regime`` == "unknown" → regime_mult 1.0
          - 51 days < 60 → ``_compute_technical_indicators`` is None → no quant adj
          - sentiment of the headline = +1.0 (3 bullish words, 0 bearish)
          - ticker_scores[NVDA] = score(10.0) × sentiment(1.0) = 10.0
          - NVDA ∉ _LEVERAGED_ETFS → conviction = min(0.25, 10.0/20) = 0.25
            (the cap *binds*: uncapped would be 0.50, so a regression that
            dropped ``min(0.25, …)`` doubles the notional and fails here)
          - cash 100_000 ≫ → conviction arm binds, NOT the cash arm
        ⇒ notional == 0.25 × 100_000 == 25_000.0 ; qty == 25_000 / 200 == 125.0
        """
        import paper_trader.backtest as bt
        # The module-level scorer singleton is not reset by conftest; pin it
        # untrained so the gate cannot perturb conviction in this assertion.
        monkeypatch.setattr(bt, "_DECISION_SCORER", None, raising=False)

        p = SimPortfolio(cash=100_000.0)
        rng = random.Random(42)
        articles = [{"title": "Nvidia beats earnings, guidance raised, "
                              "semiconductor surge",
                     "score": 10.0, "tickers": ["NVDA"]}]
        d = synthetic_prices.trading_days[-1]
        decision = _ml_decide(d, p, articles, synthetic_prices,
                              run_id=1, rng=rng)

        assert decision["action"] == "BUY"
        assert decision["ticker"] == "NVDA"
        price = synthetic_prices.price_on("NVDA", d)
        assert price == 200.0  # 100 + 50*2, last synthetic NVDA close
        total_value = p.total_value(synthetic_prices, d)  # == cash, no positions
        notional = decision["qty"] * price
        # Conviction cap is exactly 25% of total value here — pin both the
        # derived qty and the notional so any change to the conviction
        # formula (cap removal, regime double-count, scorer leakage) fails.
        assert decision["qty"] == pytest.approx(125.0, abs=1e-6)
        assert notional == pytest.approx(25_000.0, abs=0.01)
        assert notional <= total_value * 0.25 + 0.01
        # And it must stay under the absolute leveraged-ETF ceiling (0.40)
        # regardless of which arm bound it.
        assert notional <= total_value * 0.40 + 0.01


# ─────────────────────── BacktestStore isolation ───────────────────────────

class TestBacktestStoreIsolation:
    """Regression guard: BacktestStore() must resolve BACKTEST_DB at call time,
    not capture it as an import-time default parameter.

    A `def __init__(self, path=BACKTEST_DB)` default binds the module global's
    value when backtest.py is imported, so conftest's
    `monkeypatch.setattr(bt, "BACKTEST_DB", tmp)` was silently ineffective —
    every BacktestStore()/BacktestEngine() connected to the real persistent
    backtest.db, polluting it across tests and causing order-dependent flaky
    failures. This test fails on the pre-fix code and passes after.
    """

    def test_no_arg_store_respects_monkeypatched_backtest_db(self, tmp_path,
                                                              monkeypatch):
        import paper_trader.backtest as bt

        target = tmp_path / "isolated" / "backtest.db"
        monkeypatch.setattr(bt, "BACKTEST_DB", target, raising=False)

        store = bt.BacktestStore()  # no arg — must honor the monkeypatch
        try:
            # The connection must point at the monkeypatched path, NOT the
            # real repo-root backtest.db.
            db_files = store.conn.execute("PRAGMA database_list").fetchall()
            main_file = [r for r in db_files if r[1] == "main"][0][2]
            assert main_file == str(target), (
                f"BacktestStore() connected to {main_file!r}, "
                f"expected the monkeypatched {target!r}"
            )
            assert target.exists()
            # And a write/read round-trips on that isolated DB.
            store.upsert_run(run_id=7, seed=1, status="running",
                             start=date(2021, 1, 4), end=date(2021, 6, 30))
            row = store.conn.execute(
                "SELECT start_date FROM backtest_runs WHERE run_id=7"
            ).fetchone()
            assert row is not None
            assert row["start_date"] == "2021-01-04"
        finally:
            store.conn.close()

    def test_explicit_path_still_honored(self, tmp_path):
        import paper_trader.backtest as bt

        explicit = tmp_path / "explicit.db"
        store = bt.BacktestStore(explicit)
        try:
            assert explicit.exists()
        finally:
            store.conn.close()
