"""Coverage-gap regression locks for the ML / backtest domain.

Added by the ML+backtest review pass. The existing suite is comprehensive,
but two pieces of load-bearing logic were unlocked:

1. ``backtest._market_regime`` — its return value drives ``regime_mult``
   (bull=1.0, sideways=0.6, bear=0.3, unknown=1.0). That multiplier scales
   *every* signal in ``_ml_decide`` and every training feature in
   ``_compute_decision_outcomes``. A flipped bull/bear classification, or a
   regression of the deliberate "unknown → neutral 1.0" mapping (early
   backtest days previously fell into the 0.3 bear bucket and silently
   dampened the first ~200 trading days of every run), would be invisible
   to the rest of the suite — ``synthetic_prices`` only has 51 SPY days so
   it always returns "unknown". This builds a ≥200-day SPY series and
   asserts each branch.

2. ``decision_scorer.train_scorer``'s numpy weighted-least-squares fallback
   (the ``except ImportError`` arm). Every existing ``TestTrainScorer`` test
   runs with sklearn installed, so the ``_LstsqModel`` / ``_LstsqScaler``
   path — the entire scorer on a sklearn-less deploy — is never exercised.
   A broken pickle round-trip or a shape bug in ``_LstsqModel.predict``
   would silently zero out the conviction gate in production. This forces
   the ImportError path and asserts: status ok, pickle round-trips,
   predictions are finite, batch predict returns shape (n,), and a cleanly
   linearly-separable training set is ranked in the correct order.

All offline, deterministic. ``conftest._isolate_data_dir`` redirects
``SCORER_PATH`` into ``tmp_path`` so the real pickle is never touched.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

import pytest

from paper_trader.backtest import PriceCache, _market_regime


def _weekdays(n: int, end: date) -> list[date]:
    """Return ``n`` consecutive weekday dates ending on/just before ``end``."""
    days: list[date] = []
    d = end
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return list(reversed(days))


def _spy_cache(closes: list[float], days: list[date]) -> PriceCache:
    """Build a PriceCache (via __new__, like the synthetic_prices fixture)
    holding only a SPY series of the given closes on the given days."""
    assert len(closes) == len(days)
    cache = PriceCache.__new__(PriceCache)
    cache.tickers = ["SPY"]
    cache.start = days[0]
    cache.end = days[-1]
    cache.prices = {"SPY": {d.isoformat(): c for d, c in zip(days, closes)}}
    cache.trading_days = list(days)
    return cache


# ───────────────────────── _market_regime ─────────────────────────

class TestMarketRegimeClassification:
    """``_market_regime`` is SPY-only: needs ≥200 closes ≤ sim_date, then
    classifies on (last vs MA50 vs MA200)."""

    SIM_DATE = date(2025, 6, 2)

    def test_unknown_when_insufficient_history(self):
        # 51 days < 200 → "unknown" (NOT bear — the neutral mapping is
        # deliberate; see the comment in _ml_decide step 3).
        days = _weekdays(51, self.SIM_DATE)
        cache = _spy_cache([100.0 + i for i in range(51)], days)
        assert _market_regime(self.SIM_DATE, cache) == "unknown"

    def test_strictly_rising_series_is_bull(self):
        # last > MA50 > MA200 holds for any strictly increasing series.
        days = _weekdays(260, self.SIM_DATE)
        cache = _spy_cache([100.0 + i for i in range(260)], days)
        assert _market_regime(self.SIM_DATE, cache) == "bull"

    def test_strictly_falling_series_is_bear(self):
        # last < MA50 < MA200 holds for any strictly decreasing series.
        days = _weekdays(260, self.SIM_DATE)
        cache = _spy_cache([400.0 - i for i in range(260)], days)
        assert _market_regime(self.SIM_DATE, cache) == "bear"

    def test_flat_series_is_sideways(self):
        # last == MA50 == MA200 → neither strict-bull nor strict-bear chain
        # holds → "sideways" (the else branch, regime_mult 0.6).
        days = _weekdays(260, self.SIM_DATE)
        cache = _spy_cache([200.0] * 260, days)
        assert _market_regime(self.SIM_DATE, cache) == "sideways"

    def test_recent_dip_below_ma50_in_uptrend_is_sideways(self):
        # 220 rising days then a sharp 10-day drop: last < MA50 but the long
        # uptrend keeps MA50 > MA200, so neither the bull chain
        # (last > MA50) nor the bear chain (MA50 < MA200) holds → sideways.
        rising = [100.0 + i for i in range(250)]
        rising[-10:] = [rising[-11] - 30.0 * (j + 1) for j in range(10)]
        days = _weekdays(250, self.SIM_DATE)
        cache = _spy_cache(rising, days)
        regime = _market_regime(self.SIM_DATE, cache)
        assert regime == "sideways", regime

    def test_only_history_on_or_before_sim_date_is_used(self):
        # Future closes (after sim_date) must be ignored — _series_up_to
        # filters on d <= sim_date. Put a wild future spike after the sim
        # date; a strictly-rising past must still classify bull regardless.
        days = _weekdays(260, date(2025, 12, 31))
        closes = [100.0 + i for i in range(260)]
        cache = _spy_cache(closes, days)
        sim = days[210]  # 211 trading days of history ≤ sim → enough
        # Inject an absurd post-sim spike that would break MA50 if leaked.
        cache.prices["SPY"][days[259].isoformat()] = 1e6
        assert _market_regime(sim, cache) == "bull"


# ─────────────── train_scorer numpy lstsq fallback ────────────────

def _force_no_sklearn(monkeypatch):
    """Make every ``from sklearn... import`` inside train_scorer raise
    ImportError. A ``None`` entry in sys.modules makes the import machinery
    raise ImportError for that dotted name."""
    for mod in ("sklearn", "sklearn.neural_network",
                "sklearn.preprocessing", "sklearn.model_selection"):
        monkeypatch.setitem(sys.modules, mod, None)


def _linear_records(n: int) -> list[dict]:
    """n outcome records where forward_return_5d == ml_score exactly, so a
    linear least-squares fit must learn a clean monotone mapping. Distinct
    sim_dates so train_scorer's (ticker, sim_date, action) dedup keeps all."""
    base = date(2025, 1, 6)
    recs = []
    for i in range(n):
        score = float(i % 20)  # 0..19, repeating
        recs.append({
            "ticker": "NVDA",
            "sim_date": (base + timedelta(days=i)).isoformat(),
            "action": "BUY",
            "ml_score": score,
            "rsi": 50.0,
            "macd": 0.0,
            "mom5": 0.0,
            "mom20": 0.0,
            "regime_mult": 1.0,
            "forward_return_5d": score,  # target == ml_score
        })
    return recs


class TestLstsqFallbackScorer:
    """The numpy weighted-least-squares fallback used when sklearn is
    unavailable. Exercises the whole ImportError arm of train_scorer."""

    def test_fallback_trains_and_status_ok(self, monkeypatch):
        from paper_trader.ml.decision_scorer import train_scorer

        _force_no_sklearn(monkeypatch)
        recs = _linear_records(60)
        result = train_scorer(recs)
        assert result["status"] == "ok"
        assert result["n"] == 60
        # val_rmse is float('nan') on the fallback path (no holdout split).
        assert result["val_rmse"] != result["val_rmse"]  # NaN

    def test_fallback_pickle_round_trips_and_predicts_finite(self, monkeypatch):
        import math

        from paper_trader.ml.decision_scorer import (
            DecisionScorer, _LstsqModel, _LstsqScaler, train_scorer,
        )

        _force_no_sklearn(monkeypatch)
        train_scorer(_linear_records(60))

        # Fresh instance reloads SCORER_PATH (conftest-redirected to tmp).
        scorer = DecisionScorer()
        assert scorer.is_trained is True
        assert scorer.n_train == 60
        # The pickled objects must be the module-level pickle-safe stand-ins,
        # not sklearn types (closures / sklearn objects would not round-trip
        # on a sklearn-less host).
        assert isinstance(scorer._model, _LstsqModel)
        assert isinstance(scorer._scaler, _LstsqScaler)

        pred = scorer.predict(
            ml_score=10.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA",
        )
        assert isinstance(pred, float)
        assert math.isfinite(pred)
        # Clamp invariant still applies on the fallback path.
        assert -50.0 <= pred <= 50.0

    def test_fallback_model_predict_returns_one_d_batch(self, monkeypatch):
        import numpy as np

        from paper_trader.ml.decision_scorer import (
            DecisionScorer, build_features, train_scorer,
        )

        _force_no_sklearn(monkeypatch)
        train_scorer(_linear_records(60))
        scorer = DecisionScorer()

        batch = np.array([
            build_features(5.0, 50.0, 0.0, 0.0, 0.0, 1.0, "NVDA"),
            build_features(15.0, 50.0, 0.0, 0.0, 0.0, 1.0, "NVDA"),
            build_features(0.0, 50.0, 0.0, 0.0, 0.0, 1.0, "AMD"),
        ], dtype=np.float32)
        scaled = scorer._scaler.transform(batch)
        out = scorer._model.predict(scaled)
        assert out.shape == (3,)
        assert np.all(np.isfinite(out))

    def test_fallback_ranks_high_ml_score_above_low(self, monkeypatch):
        # A linear fit on y == ml_score must keep the monotone ordering: a
        # high ml_score vector predicts a strictly higher 5d return than a
        # low one. Catches a sign-flip or a broken weighted-lstsq solve.
        from paper_trader.ml.decision_scorer import DecisionScorer, train_scorer

        _force_no_sklearn(monkeypatch)
        train_scorer(_linear_records(80))
        scorer = DecisionScorer()

        common = dict(rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
                      regime_mult=1.0, ticker="NVDA")
        low = scorer.predict(ml_score=1.0, **common)
        high = scorer.predict(ml_score=18.0, **common)
        assert high > low, f"expected high({high}) > low({low})"

    def test_fallback_handles_non_finite_forward_return(self, monkeypatch):
        # The _to_float NaN/inf guard must hold on the fallback path too — a
        # single poisoned forward_return_5d row would otherwise feed inf into
        # the lstsq solve and wedge every subsequent retrain.
        from paper_trader.ml.decision_scorer import DecisionScorer, train_scorer

        _force_no_sklearn(monkeypatch)
        recs = _linear_records(60)
        recs[7]["forward_return_5d"] = float("inf")
        recs[19]["forward_return_5d"] = float("nan")
        result = train_scorer(recs)
        assert result["status"] == "ok"
        scorer = DecisionScorer()
        pred = scorer.predict(
            ml_score=10.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA",
        )
        assert pred == pred  # not NaN
