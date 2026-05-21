"""ML + backtest review pass — regression locks for real logic bugs.

Added during a systematic ML/backtest review. Every test here asserts a
specific expected value (not "did not crash") and would fail if the logic
it guards regressed. Grouped by the bug class each test pins:

* TestEnforceRiskExitsSemantics — stop-loss priority, full-qty exit, and
  the no-double-fire invariant (also exercises the O(1) trading-day
  set-membership fix in `_enforce_risk_exits`).
* TestMlDecideSellAndExclude — the SELL branch and `exclude_tickers`
  (previously only the HOLD/BUY paths of `_ml_decide` were covered).
* TestMlDecideScorerGate — the DecisionScorer conviction gate arms with
  EXACT expected position sizes (locks invariants #5/#6: gate only
  engages at n_train ≥ 500; ×0.6 / ×0.85 / unchanged / ×1.15 / ×1.3).
* TestComputeDecisionOutcomesLogic — SELL passthrough (raw fwd return,
  sign flip is train_scorer's job), regime_mult mapping, and the
  reasoning-string feature parser.
* TestInjectAndTrain — the SQL column/value alignment and the JSON-null
  hardening fix in `_inject_and_train` (a null `ai_score`/`weight` must
  NOT abort the whole injection batch).
"""
from __future__ import annotations

import json
import random
import sqlite3
import types
from datetime import date

import pytest

import paper_trader.backtest as bt
import run_continuous_backtests as rcb
from paper_trader.backtest import (
    BacktestRun,
    BacktestStore,
    SimPortfolio,
    _buy,
    _enforce_risk_exits,
    _ml_decide,
)


# ───────────────────── _enforce_risk_exits semantics ─────────────────────

class TestEnforceRiskExitsSemantics:
    def test_stop_loss_takes_priority_over_take_profit(self, synthetic_prices):
        """The exit logic is `if sl…: elif tp…`. With a stop that is always
        satisfied (sl above every close) AND a take-profit that the rising
        synthetic series also eventually reaches, the SL branch must win on
        the first scanned day — the position never lives long enough for TP.
        A regression that reordered the branches (TP first) would record a
        'take-profit' trade instead.
        """
        p = SimPortfolio(cash=10_000.0)
        # SPY synthetic series is 100→150. sl=10_000 ⇒ px<=sl always true on
        # day 1; tp=120 ⇒ would only trigger ~day 20.
        _buy(p, "SPY", 5.0, 100.0, stop_loss=10_000.0, take_profit=120.0)

        store = bt.BacktestStore.__new__(bt.BacktestStore)  # unused-attr safe
        recorded: list = []

        class _Rec:
            def record_trade(self, *a):
                recorded.append(a)

        days = synthetic_prices.trading_days
        n = _enforce_risk_exits(p, synthetic_prices, days[0], days[-1],
                                run_id=1, store=_Rec())
        assert n == 1
        assert "SPY" not in p.positions
        assert len(recorded) == 1
        # positional record_trade(run_id, sim_date, ticker, action, qty, price, reason)
        reason = recorded[0][6]
        assert "stop-loss" in reason
        assert "take-profit" not in reason

    def test_exit_sells_full_quantity(self, synthetic_prices):
        """SL/TP must liquidate the ENTIRE position (`pos['qty']`), not a
        fraction. A regression to a partial exit would leave a dangling lot
        and silently understate realized risk."""
        p = SimPortfolio(cash=10_000.0)
        _buy(p, "SPY", 7.0, 100.0, stop_loss=10_000.0, take_profit=None)
        captured: list = []

        class _Rec:
            def record_trade(self, *a):
                captured.append(a)

        days = synthetic_prices.trading_days
        _enforce_risk_exits(p, synthetic_prices, days[0], days[-1], 1, _Rec())
        assert "SPY" not in p.positions          # fully closed
        assert captured[0][4] == 7.0             # qty arg == full held qty

    def test_no_double_fire_after_exit(self, synthetic_prices):
        """Once a position is stopped out it is removed; a second scan over
        an overlapping window must NOT resurrect/re-sell it (returns 0, no
        further record_trade). Locks idempotency across the per-sample
        `_enforce_risk_exits` calls in `run_one`."""
        p = SimPortfolio(cash=10_000.0)
        _buy(p, "SPY", 5.0, 100.0, stop_loss=10_000.0, take_profit=None)
        calls: list = []

        class _Rec:
            def record_trade(self, *a):
                calls.append(a)

        days = synthetic_prices.trading_days
        first = _enforce_risk_exits(p, synthetic_prices, days[0], days[-1],
                                    1, _Rec())
        second = _enforce_risk_exits(p, synthetic_prices, days[0], days[-1],
                                     1, _Rec())
        assert first == 1
        assert second == 0
        assert len(calls) == 1  # only the first scan recorded a trade

    def test_skips_non_trading_days_only(self, synthetic_prices):
        """The set-membership fix must preserve behavior: a SL set below the
        entire series never fires regardless of weekend/holiday gaps in the
        calendar walk."""
        p = SimPortfolio(cash=10_000.0)
        _buy(p, "SPY", 5.0, 100.0, stop_loss=1.0, take_profit=None)
        store = type("S", (), {"record_trade": lambda *a, **k: None})()
        days = synthetic_prices.trading_days
        assert _enforce_risk_exits(p, synthetic_prices, days[0], days[-1],
                                   1, store) == 0
        assert "SPY" in p.positions


# ───────────────────── _ml_decide SELL + exclude ─────────────────────

def _bearish_article(score: float = 3.0) -> dict:
    # 4 bearish stems (miss/plunge/lower/downgrade), 0 bullish → sentiment -1.0.
    return {"title": "Nvidia misses earnings, plunges lower, downgrade",
            "score": score, "tickers": ["NVDA"]}


class TestMlDecideSellAndExclude:
    def test_strong_negative_signal_sells_half_of_held(self, synthetic_prices):
        """A held position with a strongly negative ML/quant score must be
        SOLD, and `_ml_decide` trims exactly 50% of the held quantity."""
        p = SimPortfolio(cash=500.0)
        _buy(p, "NVDA", 10.0, 100.0, stop_loss=None, take_profit=None)
        d = synthetic_prices.trading_days[-1]
        decision = _ml_decide(d, p, [_bearish_article()], synthetic_prices,
                              run_id=1, rng=random.Random(42))
        assert decision["action"] == "SELL"
        assert decision["ticker"] == "NVDA"
        assert decision["qty"] == pytest.approx(5.0)  # round(10 * 0.5, 4)

    def test_exclude_tickers_blocks_both_sell_and_buy(self, synthetic_prices):
        """`exclude_tickers` (the per-day `traded_today` set) must remove a
        ticker from BOTH the sell scan and the buy scan, so the same name
        cannot be acted on twice in one trading day. With NVDA excluded and
        no other priced candidate, the only outcome is HOLD."""
        p = SimPortfolio(cash=500.0)
        _buy(p, "NVDA", 10.0, 100.0, stop_loss=None, take_profit=None)
        d = synthetic_prices.trading_days[-1]
        decision = _ml_decide(d, p, [_bearish_article()], synthetic_prices,
                              run_id=1, rng=random.Random(42),
                              exclude_tickers={"NVDA"})
        assert decision["action"] == "HOLD"


class TestMlDecideMalformedArticles:
    """A present-but-None `tickers` (a malformed article dict) must not crash
    `_ml_decide`. Line 1339 hardens `score` against exactly this — a None
    there reaches `float(None)` and the uncaught TypeError kills the whole
    run thread (run recorded 'failed', zero decisions). The very next line
    (`list(a.get("tickers", []))`) had the IDENTICAL failure mode: a
    `"tickers": null` makes `list(None)` raise the same uncaught TypeError.
    These tests lock the parity: a None `tickers` (and a None `score`) must
    be treated as absent/empty, producing the SAME decision as the
    well-formed article, never an exception.
    """

    def test_none_tickers_does_not_crash_and_matches_wellformed(
        self, synthetic_prices
    ):
        p = SimPortfolio(cash=100_000.0)
        d = synthetic_prices.trading_days[-1]
        title = "Nvidia beats earnings, guidance raised, semiconductor surge"
        # Reference: well-formed article with an explicit tickers list.
        good = _ml_decide(
            d, SimPortfolio(cash=100_000.0),
            [{"title": title, "score": 10.0, "tickers": ["NVDA"], "url": ""}],
            synthetic_prices, run_id=1, rng=random.Random(42),
        )
        # Malformed: tickers is an explicit JSON null. Must not raise and must
        # land on the same decision (None ⇒ treated as the empty list; the
        # word→ticker map re-derives NVDA from the headline either way).
        malformed = _ml_decide(
            d, p,
            [{"title": title, "score": 10.0, "tickers": None, "url": ""}],
            synthetic_prices, run_id=1, rng=random.Random(42),
        )
        assert good["action"] == "BUY" and good["ticker"] == "NVDA"
        assert malformed["action"] == good["action"]
        assert malformed["ticker"] == good["ticker"]
        assert malformed["qty"] == pytest.approx(good["qty"])

    def test_none_score_and_none_tickers_together(self, synthetic_prices):
        """Both hardened fields null on the same article: the `score or 0.0`
        guard skips it (no signal), so the only outcome is a clean HOLD —
        never a TypeError from either `float(None)` or `list(None)`."""
        p = SimPortfolio(cash=100_000.0)
        d = synthetic_prices.trading_days[-1]
        decision = _ml_decide(
            d, p,
            [{"title": "Some headline", "score": None,
              "tickers": None, "url": ""}],
            synthetic_prices, run_id=1, rng=random.Random(42),
        )
        assert decision["action"] == "HOLD"


# ───────────────────── DecisionScorer conviction gate ─────────────────────

class _FakeScorer:
    """Deterministic stand-in for the module DecisionScorer singleton."""

    def __init__(self, value: float, n_train: int = 1000,
                 trained: bool = True) -> None:
        self._value = value
        self._n_train = n_train
        self.is_trained = trained

    def predict(self, **_kw) -> float:
        return self._value


def _buy_qty_with_scorer(synthetic_prices, monkeypatch, scorer) -> float:
    """Drive `_ml_decide` to a BUY NVDA with a known conviction trace.

    With `synthetic_prices` (51 days): regime is "unknown" (→ mult 1.0),
    quant indicators are None (<60 closes), the headline sentiment is +1.0
    and score 10.0 ⇒ best_score=10, NVDA ∉ _LEVERAGED_ETFS ⇒ base
    conviction = min(0.25, 10/20) = 0.25. Abundant cash ⇒ the conviction
    arm binds. price(NVDA)=200 ⇒ qty = round(total*conv/200, 4).
    """
    monkeypatch.setattr(bt, "_DECISION_SCORER", scorer, raising=False)
    p = SimPortfolio(cash=100_000.0)
    d = synthetic_prices.trading_days[-1]
    articles = [{"title": "Nvidia beats earnings, guidance raised, "
                          "semiconductor surge",
                 "score": 10.0, "tickers": ["NVDA"]}]
    decision = _ml_decide(d, p, articles, synthetic_prices,
                          run_id=1, rng=random.Random(42))
    assert decision["action"] == "BUY"
    assert decision["ticker"] == "NVDA"
    return decision["qty"]


class TestMlDecideScorerGate:
    """Locks invariants #5/#6: the gate only engages at n_train ≥ 500 and
    each arm scales the EXACT base conviction (0.25) as documented in
    AGENTS.md's conviction table. base notional = 100_000 * conv;
    qty = notional / 200."""

    def test_gate_inactive_below_500_samples(self, synthetic_prices, monkeypatch):
        # Trained but only 100 samples → predictions too noisy, gate must
        # NOT modulate. Even a catastrophic -50 prediction leaves conv=0.25.
        qty = _buy_qty_with_scorer(
            synthetic_prices, monkeypatch, _FakeScorer(-50.0, n_train=100))
        assert qty == pytest.approx(125.0)          # 0.25 unchanged

    def test_neutral_band_unchanged(self, synthetic_prices, monkeypatch):
        qty = _buy_qty_with_scorer(
            synthetic_prices, monkeypatch, _FakeScorer(2.0))
        assert qty == pytest.approx(125.0)          # 0 ≤ p ≤ 5 → ×1.0

    def test_strong_headwind_scales_0_6(self, synthetic_prices, monkeypatch):
        qty = _buy_qty_with_scorer(
            synthetic_prices, monkeypatch, _FakeScorer(-20.0))
        assert qty == pytest.approx(75.0)           # 0.25×0.6=0.15

    def test_mild_headwind_scales_0_85(self, synthetic_prices, monkeypatch):
        qty = _buy_qty_with_scorer(
            synthetic_prices, monkeypatch, _FakeScorer(-5.0))
        assert qty == pytest.approx(106.25)         # 0.25×0.85=0.2125

    def test_mild_tailwind_scales_1_15(self, synthetic_prices, monkeypatch):
        qty = _buy_qty_with_scorer(
            synthetic_prices, monkeypatch, _FakeScorer(8.0))
        assert qty == pytest.approx(143.75)         # 0.25×1.15=0.2875

    def test_strong_tailwind_scales_1_3(self, synthetic_prices, monkeypatch):
        qty = _buy_qty_with_scorer(
            synthetic_prices, monkeypatch, _FakeScorer(20.0))
        assert qty == pytest.approx(162.5)          # 0.25×1.3=0.325


class _FakeMetaScorer:
    """Stand-in that implements the predict_with_meta trust contract so the
    off-distribution gate-abstention can be exercised deterministically."""

    def __init__(self, pred: float, off_distribution: bool,
                 n_train: int = 1000, trained: bool = True) -> None:
        self._pred = pred
        self._off = off_distribution
        self._n_train = n_train
        self.is_trained = trained

    def predict_with_meta(self, **_kw) -> dict:
        return {"pred": self._pred, "raw": self._pred,
                "clamped": self._off, "off_distribution": self._off}

    def predict(self, **_kw) -> float:  # parity with the real scorer's API
        return self._pred


class TestMlDecideOffDistributionGate:
    """Phase-2 behaviour: when the scorer flags the prediction
    off_distribution (the unbounded MLP head extrapolated past the empirical
    label support), the conviction gate must ABSTAIN — leave the
    quant-derived conviction untouched — instead of acting on clamped noise.
    Baseline (no modulation) qty for this fixture is 125.0 (conv 0.25)."""

    def test_off_distribution_skips_modulation_despite_catastrophic_pred(
            self, synthetic_prices, monkeypatch):
        # pred = -50 (a clamped extrapolation) would normally hit the
        # `p < -10 → ×0.6` arm and slash qty to 75.0. off_distribution=True
        # ⇒ the gate abstains ⇒ conviction stays 0.25 ⇒ qty stays 125.0.
        qty = _buy_qty_with_scorer(
            synthetic_prices, monkeypatch,
            _FakeMetaScorer(-50.0, off_distribution=True))
        assert qty == pytest.approx(125.0)

    def test_in_distribution_meta_path_still_modulates(
            self, synthetic_prices, monkeypatch):
        # Same meta scorer but off_distribution=False ⇒ the -20 prediction
        # must still fire the strong-headwind arm (×0.6 → 75.0). Proves the
        # meta path has identical in-distribution behaviour to predict().
        qty = _buy_qty_with_scorer(
            synthetic_prices, monkeypatch,
            _FakeMetaScorer(-20.0, off_distribution=False))
        assert qty == pytest.approx(75.0)

    def test_off_distribution_is_surfaced_in_reasoning(
            self, synthetic_prices, monkeypatch):
        monkeypatch.setattr(
            bt, "_DECISION_SCORER",
            _FakeMetaScorer(-50.0, off_distribution=True), raising=False)
        p = SimPortfolio(cash=100_000.0)
        d = synthetic_prices.trading_days[-1]
        articles = [{"title": "Nvidia beats earnings, guidance raised, "
                              "semiconductor surge",
                     "score": 10.0, "tickers": ["NVDA"]}]
        decision = _ml_decide(d, p, articles, synthetic_prices,
                              run_id=1, rng=random.Random(42))
        assert decision["action"] == "BUY"
        assert "off-dist,gate-skipped" in decision["reasoning"]

    def test_off_distribution_below_500_still_inactive(
            self, synthetic_prices, monkeypatch):
        # Defensive: the n_train<500 gate and the off_dist abstention are
        # independent guards; with both tripped conviction is still 0.25.
        qty = _buy_qty_with_scorer(
            synthetic_prices, monkeypatch,
            _FakeMetaScorer(-50.0, off_distribution=True, n_train=100))
        assert qty == pytest.approx(125.0)


# ───────────────────── buy_ticker quant-feature parity ─────────────────────


class _CapturingScorer:
    """Stand-in DecisionScorer that captures the LAST `predict_with_meta`
    feature kwargs so we can assert the scorer sees the same indicators that
    `_compute_decision_outcomes` writes to the training row.
    """

    def __init__(self, n_train: int = 1000) -> None:
        self._n_train = n_train
        self.is_trained = True
        self.last_kwargs: dict = {}

    def predict_with_meta(self, **kw) -> dict:
        self.last_kwargs = dict(kw)
        return {"pred": 0.0, "raw": 0.0, "clamped": False,
                "off_distribution": False}

    def predict(self, **kw) -> float:
        self.last_kwargs = dict(kw)
        return 0.0


def _price_cache_with_history(tickers: list[str], n_days: int = 70):
    """A PriceCache with >=60 closes for every ticker so
    `_compute_technical_indicators` produces real RSI/MACD/etc.
    """
    from datetime import date, timedelta
    from paper_trader.backtest import PriceCache

    start = date(2025, 1, 2)
    days: list = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)

    cache = PriceCache.__new__(PriceCache)
    cache.tickers = list(tickers)
    cache.start = days[0]
    cache.end = days[-1]
    # Rising series — RSI ends up high (close-to-overbought) but non-None.
    cache.prices = {
        t: {d.isoformat(): 100.0 + i for i, d in enumerate(days)}
        for t in tickers
    }
    cache.trading_days = days
    return cache


class TestBuyTickerQuantFeatureParity:
    """Locks the training/inference feature parity fix in `_ml_decide`.

    Bug: `buy_ticker` outside `QUANT_SIGNAL_TICKERS ∪ portfolio.positions`
    (a sentiment-only pick — empirically ~21% of all BUYs across the live
    `decision_outcomes.jsonl` tail) used to feed the scorer `build_features`
    neutral defaults (rsi=50/macd=0/...) at inference while
    `_compute_decision_outcomes` writes the REAL indicators to the training
    row via `_get_quant_signals(sim_date, [ticker], ...)`. The scorer then
    trained on a feature manifold the gate never visited, and predicted at
    a manifold the model never saw.

    The fix lazily computes `_get_quant_signals` for the picked ticker if it
    falls outside the cycle's pre-fetched set, so the scorer feature vector
    matches the training-time feature vector for the same (sim_date, ticker).
    """

    def test_sentiment_only_buy_feeds_scorer_real_rsi(self, monkeypatch):
        """A BUY of XLF (∉ QUANT_SIGNAL_TICKERS, sentiment-only pick) must
        feed the scorer a NUMERIC `rsi`, not the build_features default.
        """
        from paper_trader.backtest import QUANT_SIGNAL_TICKERS
        # Sanity guard: this whole test is meaningful only if XLF is outside
        # the pre-fetched set — pin it so a future watchlist refactor that
        # silently moves XLF in would not falsely "pass" this test.
        assert "XLF" not in QUANT_SIGNAL_TICKERS, (
            "Test premise broken: XLF moved into QUANT_SIGNAL_TICKERS; "
            "pick a different watchlist ticker outside the pre-fetched set "
            "to keep this regression honest.")

        # Need SPY for regime (>=200 closes for a real bull/bear/sideways
        # verdict; <200 yields "unknown" with regime_mult 1.0 — fine for
        # this test). XLF needs >=60 for indicators.
        prices = _price_cache_with_history(["SPY", "XLF"], n_days=70)
        scorer = _CapturingScorer()
        monkeypatch.setattr(bt, "_DECISION_SCORER", scorer, raising=False)

        p = SimPortfolio(cash=100_000.0)
        d = prices.trading_days[-1]
        # `financials` → XLF via _WORD_TO_TICKER; bullish phrasing → +1 sent.
        articles = [{"title": "Financials beat earnings, guidance raised "
                              "across the sector",
                     "score": 10.0, "tickers": []}]
        decision = _ml_decide(d, p, articles, prices,
                              run_id=1, rng=random.Random(42))
        assert decision["action"] == "BUY"
        assert decision["ticker"] == "XLF"

        # Pre-fix: scorer.last_kwargs["rsi"] would be None (build_features
        # then substitutes 50.0 internally). Post-fix: lazy fetch populates
        # the real RSI computed from the (rising) synthetic series.
        assert scorer.last_kwargs["rsi"] is not None, (
            "scorer received rsi=None — the lazy quant lookup for a "
            "non-pre-fetched buy_ticker regressed")
        assert isinstance(scorer.last_kwargs["rsi"], (int, float))
        # The REASONING string is the operator-visible artefact of this fix.
        # Pre-fix: "RSI=N/A". Post-fix: "RSI=<number>". (The reasoning uses
        # the same q_buy.get('rsi') call as the scorer feed.)
        assert "RSI=N/A" not in decision["reasoning"]

    def test_pre_fetched_ticker_path_unchanged(self, monkeypatch):
        """A BUY of NVDA (IN QUANT_SIGNAL_TICKERS) must keep using the
        pre-fetched quant signal — the fix must NOT re-fetch when the
        ticker is already in `quant`. Belt-and-braces for the no-regression
        guard on the covered-ticker path.
        """
        prices = _price_cache_with_history(["SPY", "NVDA"], n_days=70)
        scorer = _CapturingScorer()
        monkeypatch.setattr(bt, "_DECISION_SCORER", scorer, raising=False)

        p = SimPortfolio(cash=100_000.0)
        d = prices.trading_days[-1]
        articles = [{"title": "Nvidia beats earnings, guidance raised, "
                              "semiconductor surge",
                     "score": 10.0, "tickers": ["NVDA"]}]
        decision = _ml_decide(d, p, articles, prices,
                              run_id=1, rng=random.Random(42))
        assert decision["action"] == "BUY"
        assert decision["ticker"] == "NVDA"
        # NVDA is in the pre-fetched set ⇒ its indicators are present.
        assert scorer.last_kwargs["rsi"] is not None

    def test_held_position_buy_path_unchanged(self, monkeypatch):
        """If the buy_ticker happens to already be held, it lives in
        `portfolio.positions` ⇒ the pre-fetched `quant_tickers` already
        includes it. The lazy fetch is a no-op — guard against an
        accidental refetch.
        """
        prices = _price_cache_with_history(["SPY", "BABA"], n_days=70)
        scorer = _CapturingScorer()
        monkeypatch.setattr(bt, "_DECISION_SCORER", scorer, raising=False)

        p = SimPortfolio(cash=100_000.0)
        # Open a tiny BABA lot — now BABA ∈ portfolio.positions and
        # therefore ∈ quant_tickers via the union.
        _buy(p, "BABA", 0.5, 100.0, stop_loss=None, take_profit=None)

        d = prices.trading_days[-1]
        articles = [{"title": "China Alibaba surge, raised guidance",
                     "score": 10.0, "tickers": []}]
        decision = _ml_decide(d, p, articles, prices,
                              run_id=1, rng=random.Random(42))
        # Either BUY adds to BABA or HOLD; in both cases no crash.
        assert decision["action"] in {"BUY", "HOLD", "SELL"}
        if decision["action"] == "BUY":
            assert scorer.last_kwargs["rsi"] is not None


# ───────────────────── _compute_decision_outcomes logic ─────────────────────

def _engine_with_decision(tmp_path, synthetic_prices, *, action, ticker,
                           day_index, reasoning):
    """Real BacktestStore (tmp) + synthetic prices + one decision row."""
    store = BacktestStore(path=tmp_path / "bt.db")
    sim_date = synthetic_prices.trading_days[day_index].isoformat()
    store.upsert_run(1, seed=1, status="complete",
                     start=date(2025, 1, 1), end=date(2025, 12, 31))
    store.record_decision(
        1, sim_date,
        {"action": action, "ticker": ticker, "qty": 5.0,
         "confidence": 0.5, "reasoning": reasoning},
        "FILLED", "ok", 0.0, 0.0, 1,
    )
    engine = types.SimpleNamespace(store=store, prices=synthetic_prices)
    return engine, sim_date


class TestComputeDecisionOutcomesLogic:
    def test_sell_passthrough_and_regime_and_feature_parse(
            self, tmp_path, synthetic_prices):
        """A SELL outcome must carry the RAW forward return (the SELL sign
        flip is train_scorer's job, NOT this stage). regime_mult for the
        short synthetic SPY history is "unknown" → 1.0 (must NOT collapse to
        the bear 0.3 bucket). ml_score / news_urgency / news_count are parsed
        out of the reasoning string."""
        # NVDA synthetic: price = 100 + 2*i. day 10 → 120, day 15 → 130.
        # forward 5-trading-day return = (130-120)/120*100 = 8.3333%.
        eng, sim_date = _engine_with_decision(
            tmp_path, synthetic_prices, action="SELL", ticker="NVDA",
            day_index=10,
            reasoning="ML+quant: NVDA score=-1.50 regime=bull RSI=72 "
                      "news_count=3 news_urg=4.0 — reducing",
        )
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                            end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        assert len(outs) == 1
        o = outs[0]
        assert o["action"] == "SELL"
        # RAW return, not sign-flipped here.
        assert o["forward_return_5d"] == pytest.approx(8.3333, abs=1e-4)
        # Synthetic SPY has only ~11 closes by day 10 (<200) → "unknown" → 1.0.
        assert o["regime_mult"] == 1.0
        assert o["ml_score"] == pytest.approx(-1.5)
        assert o["news_urgency"] == pytest.approx(4.0)
        assert o["news_article_count"] == pytest.approx(3.0)
        # <60 closes ⇒ no quant indicators ⇒ these stay None (not 0).
        assert o["rsi"] is None
        assert o["macd"] is None

    def test_zero_news_count_nulls_news_features(self, tmp_path,
                                                 synthetic_prices):
        """When the reasoning reports `news_count=0`, both news features must
        be set to None so training and inference share ONE encoding of the
        no-news condition (build_features then applies its neutral defaults).
        """
        eng, _ = _engine_with_decision(
            tmp_path, synthetic_prices, action="BUY", ticker="NVDA",
            day_index=5,
            reasoning="ML+quant: NVDA score=2.00 regime=bull RSI=40 "
                      "news_count=0 news_urg=0.0 conviction=10%",
        )
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                            end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        assert len(outs) == 1
        o = outs[0]
        assert o["news_urgency"] is None
        assert o["news_article_count"] is None
        assert o["ml_score"] == pytest.approx(2.0)


# ───────────────────── _inject_and_train SQL + null hardening ─────────────

class TestInjectAndTrain:
    def _fake_trainer_ok(self):
        return types.SimpleNamespace(
            returncode=0, stdout="trainer n=1 loss=0.0 val=0.0", stderr="")

    def test_null_ai_score_and_weight_do_not_abort_batch(
            self, tmp_path, empty_articles_db, monkeypatch):
        """Regression: `float(rec.get('ai_score', 0))` only defaults on a
        MISSING key — an explicit JSON `null` reaches `float(None)` and
        raises, aborting the WHOLE injection batch (outer except →
        'inject err'), so ArticleNet never retrains that cycle. The fix
        (`or` fallback) must coerce null → safe defaults (ai 0.0, weight
        1.0 → eff 0.0) and still insert the row.
        """
        jsonl = tmp_path / "winners.jsonl"
        good = {"title": "BUY NVDA on 2025-01-01", "ai_score": 4.0,
                "weight": 0.8, "ticker": "NVDA", "reasoning": "r",
                "sim_date": "2025-01-01", "label": "BUY", "run_id": 1,
                "cycle": 3}
        nulled = {"title": "BUY AMD", "ai_score": None, "weight": None,
                  "ticker": "AMD", "reasoning": "r2",
                  "sim_date": "2025-01-02", "label": "BUY", "run_id": 1,
                  "cycle": 3}
        # `good` written twice → exercises INSERT OR IGNORE dedup.
        jsonl.write_text("\n".join(json.dumps(r)
                                   for r in (good, nulled, good)) + "\n")

        monkeypatch.setattr(rcb, "WINNER_JSONL", jsonl)
        monkeypatch.setattr(rcb, "DIGITAL_INTERN_ARTICLES_DB",
                            str(empty_articles_db))
        monkeypatch.setattr(rcb.subprocess, "run",
                            lambda *a, **k: self._fake_trainer_ok())

        status = rcb._inject_and_train()

        # Did NOT raise / abort: status reports a successful injection.
        assert not status.startswith("inject err"), status
        assert status.startswith("injected 2 new"), status  # dedup → 2 distinct

        conn = sqlite3.connect(str(empty_articles_db))
        try:
            rows = conn.execute(
                "SELECT url, source, ai_score, kw_score, urgency, cycle "
                "FROM articles ORDER BY url"
            ).fetchall()
        finally:
            conn.close()
        assert len(rows) == 2  # the duplicate `good` was IGNOREd

        by_url = {r[0]: r for r in rows}
        # good: eff = min(10, 4.0 * 0.8) = 3.2; source/urgency/cycle exact.
        g = by_url["backtest://run_1/2025-01-01/BUY/NVDA"]
        assert g[1] == "backtest_run_1"
        assert g[2] == pytest.approx(3.2)   # ai_score == eff
        assert g[3] == pytest.approx(3.2)   # kw_score == eff (same value)
        assert g[4] == 0                    # urgency hard-coded 0
        assert g[5] == 3                    # cycle passthrough
        # nulled: the bug case — eff = min(10, 0.0 * 1.0) = 0.0, row present.
        n = by_url["backtest://run_1/2025-01-02/BUY/AMD"]
        assert n[2] == pytest.approx(0.0)
        assert n[3] == pytest.approx(0.0)

    def test_missing_jsonl_returns_sentinel(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rcb, "WINNER_JSONL", tmp_path / "nope.jsonl")
        assert rcb._inject_and_train() == "no jsonl"

    # ── transient-lock retry (2026-05-18 fix) ────────────────────────────
    # digital-intern's daemon is a heavy concurrent writer to the SAME
    # ~1.4 GB articles.db. Observed live (continuous.log: 7 cycles) the
    # daemon held the write lock past sqlite3's `timeout=`, so
    # `OperationalError: database is locked` aborted the WHOLE ArticleNet
    # feedback batch (CLAUDE.md §5 step 5) with no retry. `_inject_and_train`
    # now retries connect→write→commit on a *transient* lock with backoff
    # (INSERT OR IGNORE makes the replay idempotent).

    def _winners_one(self, tmp_path, monkeypatch, empty_articles_db):
        jsonl = tmp_path / "winners.jsonl"
        rec = {"title": "BUY NVDA on 2025-01-01", "ai_score": 4.0,
               "weight": 1.0, "ticker": "NVDA", "reasoning": "r",
               "sim_date": "2025-01-01", "label": "BUY", "run_id": 1,
               "cycle": 3}
        jsonl.write_text(json.dumps(rec) + "\n")
        monkeypatch.setattr(rcb, "WINNER_JSONL", jsonl)
        monkeypatch.setattr(rcb, "DIGITAL_INTERN_ARTICLES_DB",
                            str(empty_articles_db))
        monkeypatch.setattr(rcb.subprocess, "run",
                            lambda *a, **k: self._fake_trainer_ok())
        sleeps: list[float] = []
        monkeypatch.setattr(rcb.time, "sleep", lambda s: sleeps.append(s))
        return sleeps

    def test_transient_lock_is_retried_then_succeeds(
            self, tmp_path, empty_articles_db, monkeypatch):
        sleeps = self._winners_one(tmp_path, monkeypatch, empty_articles_db)
        real_connect = sqlite3.connect
        state = {"calls": 0}

        class _LockedConn:
            def execute(self, *a, **k):
                raise sqlite3.OperationalError("database is locked")
            def rollback(self): pass
            def commit(self): pass
            def close(self): pass

        def fake_connect(*a, **k):
            state["calls"] += 1
            # First TWO attempts hit the daemon's write lock; the 3rd
            # finds the lock released and writes for real.
            if state["calls"] <= 2:
                return _LockedConn()
            return real_connect(*a, **k)

        monkeypatch.setattr(rcb.sqlite3, "connect", fake_connect)
        status = rcb._inject_and_train()

        assert not status.startswith("inject err"), status
        assert status.startswith("injected 1 new"), status
        # Two failed attempts → exactly the first two backoff sleeps, then
        # the 3rd attempt succeeds (no 3rd sleep).
        assert sleeps == [3.0, 8.0], sleeps
        conn = real_connect(str(empty_articles_db))
        try:
            n = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        finally:
            conn.close()
        assert n == 1  # row landed exactly once (idempotent replay)

    def test_persistent_lock_exhausts_retries_with_honest_status(
            self, tmp_path, empty_articles_db, monkeypatch):
        sleeps = self._winners_one(tmp_path, monkeypatch, empty_articles_db)

        class _LockedConn:
            def execute(self, *a, **k):
                raise sqlite3.OperationalError("database is locked")
            def rollback(self): pass
            def close(self): pass

        monkeypatch.setattr(rcb.sqlite3, "connect",
                            lambda *a, **k: _LockedConn())
        status = rcb._inject_and_train()

        # 1 initial + 3 retries = 4 attempts, then an honest lock status —
        # NOT a generic "inject err: <exc>" and NOT a false success.
        assert status.startswith("inject err: database locked after 4 "
                                 "attempts"), status
        assert sleeps == [3.0, 8.0, 15.0], sleeps

    def test_non_lock_operational_error_fails_fast_no_backoff(
            self, tmp_path, empty_articles_db, monkeypatch):
        sleeps = self._winners_one(tmp_path, monkeypatch, empty_articles_db)
        state = {"calls": 0}

        class _BrokenConn:
            def execute(self, *a, **k):
                raise sqlite3.OperationalError("disk I/O error")
            def rollback(self): pass
            def close(self): pass

        def fake_connect(*a, **k):
            state["calls"] += 1
            return _BrokenConn()

        monkeypatch.setattr(rcb.sqlite3, "connect", fake_connect)
        status = rcb._inject_and_train()

        # A non-lock OperationalError is a real bug, not contention — return
        # immediately, exactly one attempt, ZERO pointless backoff.
        assert status == "inject err: disk I/O error", status
        assert sleeps == [], sleeps
        assert state["calls"] == 1, state["calls"]
