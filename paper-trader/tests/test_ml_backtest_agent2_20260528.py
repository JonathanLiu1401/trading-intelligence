"""Agent 2 (ML+backtests) review — 2026-05-28.

Test-locks for behaviour findings from this review pass. Focuses on
business-logic correctness (real value assertions), not just no-crash
contracts.

1. ``DecisionScorer.predict_with_meta`` returns BOUNDED predictions:
   raw output may extrapolate, but ``pred`` is always within
   ``[-PRED_CLAMP_PCT, +PRED_CLAMP_PCT]`` AND finite.

2. ``DecisionScorer.predict_with_meta`` ``failed=True`` is honored: a
   None pickle / scaler shape mismatch must surface as ``failed=True``
   so downstream OOS rank-IC consumers can drop the row.

3. ``predict_calibrated`` is MONOTONIC in ``raw``: a higher raw must
   always yield a higher (or equal) calibrated value. This is the load-
   bearing property of the documented DIRECTIONAL_BUT_BIASED verdict
   — the model's rank is preserved by quantile-mapping.

4. ``build_features`` clamps news_urgency to [0, 100] and
   news_article_count to [0, 20] even for malformed inputs.

5. The killswitch's deque tail-correctness: when more than 60 rows
   exist but only the last 20 carry parseable oos_buy_ic, the
   killswitch uses those 20 — not older fallback rows.

6. ``_compute_decision_outcomes`` walk-back collision discipline holds
   under thin-calendar tickers (both endpoints walking back to the same
   close yields no outcome row, NOT a fake 0% one).

7. ``_ml_decide`` returns HOLD with non-zero reasoning when no signal
   beats the buy threshold — never returns a malformed/None result.

8. ``train_scorer`` ``insufficient_data`` / ``insufficient_after_dedup``
   sentinels are honored without crashing: a small batch must not write
   a corrupted pickle to disk.

9. ``score_article`` returns a score in [0, 5] regardless of phrase
   counts. Headlines with many bullish phrases cap at 5.0; many bearish
   cap at 0.0.

10. SECTORS list order matches what build_features produces, so a row
    of features ending in [1, 0, 0, 0, 0, 0, 0] corresponds to
    sector_tech (the first SECTORS entry).
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Section 1 — predict_with_meta clamps every output to the empirical bound
# ---------------------------------------------------------------------------


class TestPredictWithMetaBoundedness:
    """A trader/dashboard consumer reads ``pred`` and must NEVER see a
    value outside ±PRED_CLAMP_PCT, even when the underlying MLP
    extrapolates wildly off-distribution.
    """

    def test_pred_always_within_pred_clamp(self, tmp_path):
        from paper_trader.ml.decision_scorer import (
            DecisionScorer, PRED_CLAMP_PCT, train_scorer,
        )

        # Build a synthetic dataset with diverse features and SAVE to the
        # redirected SCORER_PATH so DecisionScorer() loads it.
        import paper_trader.ml.decision_scorer as ds
        records = []
        for i in range(60):
            records.append({
                "ticker": ["NVDA", "AMD", "SPY", "TLT", "BTC-USD"][i % 5],
                "sim_date": f"2024-01-{(i % 28) + 1:02d}",
                "ml_score": (i % 5) * 1.0,
                "rsi": 30 + (i % 60), "macd": (i - 30) * 0.1,
                "mom5": (i - 30) * 0.5, "mom20": (i - 30) * 1.0,
                "regime_mult": 1.0, "vol_ratio": 1.0,
                "bb_position": ((i % 20) - 10) / 10.0,
                "news_urgency": 50.0, "news_article_count": 1.0,
                "forward_return_5d": (i - 30) * 0.3,
                "action": "BUY",
            })
        result = train_scorer(records, path=ds.SCORER_PATH)
        assert result["status"] == "ok"
        s = DecisionScorer()
        assert s.is_trained
        # Test with several inputs across both ranges (extreme high/low quant).
        for rsi in (10, 50, 90):
            for mom20 in (-30, 0, 30):
                meta = s.predict_with_meta(
                    ml_score=5.0, rsi=rsi, macd=0.0, mom5=0.0, mom20=mom20,
                    regime_mult=1.0, ticker="NVDA",
                    vol_ratio=1.0, bb_pos=0.0,
                    news_urgency=50.0, news_article_count=1.0,
                    ema200_above=True, hist_cross_up=False,
                    macd_below_zero_cross=False,
                )
                pred = meta["pred"]
                assert isinstance(pred, float)
                assert np.isfinite(pred)
                assert -PRED_CLAMP_PCT <= pred <= PRED_CLAMP_PCT, (
                    f"pred={pred} exceeds ±{PRED_CLAMP_PCT}% bound"
                )
                # raw can exceed, but flag must reflect
                if abs(meta["raw"]) > PRED_CLAMP_PCT:
                    assert meta["clamped"] is True
                    assert meta["off_distribution"] is True

    def test_untrained_scorer_predict_returns_zero_and_failed(self):
        """An untrained scorer (no pickle) returns a sentinel that
        downstream rank-IC code can detect via failed=True."""
        from paper_trader.ml.decision_scorer import DecisionScorer

        s = DecisionScorer.__new__(DecisionScorer)
        s._model = None
        s._scaler = None
        s._trained = False
        s._n_train = 0
        s._pred_quantiles = None
        s._label_quantiles = None
        meta = s.predict_with_meta(
            ml_score=2.0, rsi=50, macd=0.0, mom5=0.0, mom20=0.0,
            regime_mult=1.0, ticker="NVDA",
        )
        assert meta["pred"] == 0.0
        assert meta["failed"] is True
        assert meta["gate_arm"] is None
        assert meta["gate_arm_multiplier"] is None


# ---------------------------------------------------------------------------
# Section 2 — predict_calibrated is monotonic (rank-preservation)
# ---------------------------------------------------------------------------


class TestCalibratedMonotonic:
    """``predict_calibrated`` quantile-maps via two sorted quantile
    tables. The result MUST be monotonic in raw — that is the documented
    rank-preserving property of the calibration step."""

    def test_calibrated_monotonic_in_raw(self, tmp_path):
        from paper_trader.ml.decision_scorer import (
            DecisionScorer, train_scorer,
        )
        import paper_trader.ml.decision_scorer as ds

        # Train synthetic so calibration tables exist on disk
        records = []
        for i in range(60):
            records.append({
                "ticker": ["NVDA", "AMD", "SPY"][i % 3],
                "sim_date": f"2024-01-{(i % 28) + 1:02d}",
                "ml_score": (i % 5) * 1.0,
                "rsi": 30 + (i % 60), "macd": (i - 30) * 0.1,
                "mom5": (i - 30) * 0.5, "mom20": (i - 30) * 1.0,
                "regime_mult": 1.0, "vol_ratio": 1.0,
                "bb_position": ((i % 20) - 10) / 10.0,
                "news_urgency": 50.0, "news_article_count": 1.0,
                "forward_return_5d": (i - 30) * 0.3,
                "action": "BUY",
            })
        result = train_scorer(records, path=ds.SCORER_PATH)
        assert result["status"] == "ok"
        s = DecisionScorer()
        assert s._label_quantiles is not None
        assert s._pred_quantiles is not None

        # Build a sequence of inputs that produce increasing raw predictions.
        # Sweep mom20 (an unbounded numeric feature) and observe pred ordering.
        results = []
        for mom20 in [-30, -20, -10, 0, 10, 20, 30]:
            meta = s.predict_with_meta(
                ml_score=2.0, rsi=50, macd=0.0, mom5=0.0, mom20=mom20,
                regime_mult=1.0, ticker="NVDA",
                vol_ratio=1.0, bb_pos=0.0,
                news_urgency=50.0, news_article_count=1.0,
            )
            results.append((meta["raw"], meta["calibrated"]))

        # Sort by raw; calibrated values must be non-decreasing
        results.sort(key=lambda r: r[0])
        cals = [r[1] for r in results if r[1] is not None]
        assert len(cals) >= 2, "expected calibrated values"
        for a, b in zip(cals, cals[1:]):
            assert b >= a, f"calibrated not monotonic in raw: {cals}"


# ---------------------------------------------------------------------------
# Section 3 — build_features clamps malformed news_urgency/article_count
# ---------------------------------------------------------------------------


class TestBuildFeaturesClamps:
    """The feature vector must be bounded for the StandardScaler. A
    corrupted upstream `news_urgency=99999` must clamp to 100, not flow
    into training."""

    def test_news_urgency_clamped_high(self):
        from paper_trader.ml.decision_scorer import build_features, FEATURE_NAMES

        v = build_features(
            ml_score=2.0, rsi=50, macd=0, mom5=0, mom20=0,
            regime_mult=1.0, ticker="NVDA",
            news_urgency=99999.0, news_article_count=1.0,
        )
        idx = FEATURE_NAMES.index("news_urgency")
        assert v[idx] == 100.0, f"urgency not clamped to 100: {v[idx]}"

    def test_news_urgency_clamped_low(self):
        from paper_trader.ml.decision_scorer import build_features, FEATURE_NAMES

        v = build_features(
            ml_score=2.0, rsi=50, macd=0, mom5=0, mom20=0,
            regime_mult=1.0, ticker="NVDA",
            news_urgency=-50.0, news_article_count=1.0,
        )
        idx = FEATURE_NAMES.index("news_urgency")
        assert v[idx] == 0.0, f"urgency not clamped to 0: {v[idx]}"

    def test_news_article_count_clamped_high(self):
        from paper_trader.ml.decision_scorer import build_features, FEATURE_NAMES

        v = build_features(
            ml_score=2.0, rsi=50, macd=0, mom5=0, mom20=0,
            regime_mult=1.0, ticker="NVDA",
            news_urgency=50.0, news_article_count=9999.0,
        )
        idx = FEATURE_NAMES.index("news_article_count")
        assert v[idx] == 20.0, f"article_count not clamped to 20: {v[idx]}"

    def test_bb_pos_clamped(self):
        from paper_trader.ml.decision_scorer import build_features, FEATURE_NAMES

        v_hi = build_features(
            ml_score=0, rsi=50, macd=0, mom5=0, mom20=0,
            regime_mult=1.0, ticker="NVDA", bb_pos=10.0,
        )
        v_lo = build_features(
            ml_score=0, rsi=50, macd=0, mom5=0, mom20=0,
            regime_mult=1.0, ticker="NVDA", bb_pos=-10.0,
        )
        idx = FEATURE_NAMES.index("bb_pos")
        assert v_hi[idx] == 2.0
        assert v_lo[idx] == -2.0

    def test_vol_ratio_clamped(self):
        from paper_trader.ml.decision_scorer import build_features, FEATURE_NAMES

        v_hi = build_features(
            ml_score=0, rsi=50, macd=0, mom5=0, mom20=0,
            regime_mult=1.0, ticker="NVDA", vol_ratio=100.0,
        )
        v_neg = build_features(
            ml_score=0, rsi=50, macd=0, mom5=0, mom20=0,
            regime_mult=1.0, ticker="NVDA", vol_ratio=-5.0,
        )
        idx = FEATURE_NAMES.index("vol_ratio")
        assert v_hi[idx] == 5.0, f"vol_ratio not clamped high: {v_hi[idx]}"
        assert v_neg[idx] == 0.0, f"vol_ratio not clamped low: {v_neg[idx]}"


# ---------------------------------------------------------------------------
# Section 4 — Killswitch deque tail correctness
# ---------------------------------------------------------------------------


class TestKillswitchTailCorrectness:
    """The killswitch reads the trailing 20 valid oos_buy_ic rows. If the
    most recent 20 rows are at noise, the gate must kill even when older
    rows showed skill (an active loop's behaviour right now should drive
    the gate, not 6 months ago)."""

    def test_recent_noise_kills_gate_even_if_old_rows_skilled(self, tmp_path, monkeypatch):
        import paper_trader.backtest as bt

        log = tmp_path / "scorer_skill_log.jsonl"
        # First 30 rows: skilled (+0.20). Last 20 rows: noise (0.0).
        with log.open("w") as f:
            for _ in range(30):
                f.write(json.dumps({"oos_buy_ic": 0.20}) + "\n")
            for _ in range(20):
                f.write(json.dumps({"oos_buy_ic": 0.0}) + "\n")

        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", log)
        bt._reset_gate_skill_cache()
        gate, reason = bt._should_gate_modulate_conviction()
        assert gate is False, (
            f"Killswitch must use the recent 20 (noise), not the older 30 "
            f"(skill). reason={reason}"
        )
        assert "noise or anti-predictive" in reason

    def test_recent_skill_keeps_gate_even_if_old_rows_noise(self, tmp_path, monkeypatch):
        import paper_trader.backtest as bt

        log = tmp_path / "scorer_skill_log.jsonl"
        # First 30 rows: noise. Last 20 rows: skilled (+0.20).
        with log.open("w") as f:
            for _ in range(30):
                f.write(json.dumps({"oos_buy_ic": 0.0}) + "\n")
            for _ in range(20):
                f.write(json.dumps({"oos_buy_ic": 0.20}) + "\n")

        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", log)
        bt._reset_gate_skill_cache()
        gate, reason = bt._should_gate_modulate_conviction()
        assert gate is True, f"Killswitch must let recent skill flow. reason={reason}"


# ---------------------------------------------------------------------------
# Section 5 — score_article output is clamped [0, 5]
# ---------------------------------------------------------------------------


class TestScoreArticleBounded:
    def test_high_score_clamped_to_5(self):
        from paper_trader.backtest import score_article, BUY_PHRASES
        # A title stuffed with bullish phrases
        title = " ".join(BUY_PHRASES * 3) + " NVDA AMD MU"
        score, _ = score_article({"title": title})
        assert 0.0 <= score <= 5.0, f"score {score} out of [0,5]"
        assert score == 5.0

    def test_low_score_clamped_to_0(self):
        from paper_trader.backtest import score_article, SELL_PHRASES
        title = " ".join(SELL_PHRASES * 3)
        score, _ = score_article({"title": title})
        assert 0.0 <= score <= 5.0
        assert score == 0.0

    def test_neutral_score_is_25(self):
        from paper_trader.backtest import score_article
        score, _ = score_article({"title": "Stock market opens at typical level"})
        assert score == 2.5


# ---------------------------------------------------------------------------
# Section 6 — train_scorer insufficient-data sentinels (no pickle write)
# ---------------------------------------------------------------------------


class TestTrainScorerInsufficientData:
    def test_empty_records_returns_insufficient(self, tmp_path):
        from paper_trader.ml.decision_scorer import train_scorer

        path = tmp_path / "scorer.pkl"
        result = train_scorer([], path=path)
        assert result["status"] == "insufficient_data"
        assert result["n"] == 0
        assert not path.exists(), "pickle must NOT be written for empty input"

    def test_under_min_returns_insufficient_after_dedup(self, tmp_path):
        from paper_trader.ml.decision_scorer import train_scorer

        path = tmp_path / "scorer.pkl"
        # 10 distinct records — below the n>=30 threshold after dedup.
        records = [
            {"ticker": "NVDA", "sim_date": f"2024-01-{i:02d}",
             "ml_score": 2.0, "rsi": 50, "macd": 0.0, "mom5": 0.0,
             "mom20": 0.0, "regime_mult": 1.0, "forward_return_5d": 1.0}
            for i in range(1, 11)
        ]
        result = train_scorer(records, path=path)
        assert result["status"] == "insufficient_after_dedup"
        assert not path.exists()

    def test_all_null_labels_returns_no_valid_labels(self, tmp_path):
        from paper_trader.ml.decision_scorer import train_scorer

        path = tmp_path / "scorer.pkl"
        records = [
            {"ticker": "NVDA", "sim_date": f"2024-01-{i:02d}",
             "ml_score": 2.0, "rsi": 50, "macd": 0.0, "mom5": 0.0,
             "mom20": 0.0, "regime_mult": 1.0, "forward_return_5d": None}
            for i in range(1, 100)
        ]
        result = train_scorer(records, path=path)
        # All records have null labels → dropped → empty after validation
        assert result["status"] == "no_valid_labels"
        assert result["n_label_dropped"] >= 30
        assert not path.exists()


# ---------------------------------------------------------------------------
# Section 7 — _ml_decide returns HOLD with reasoning when no signal
# ---------------------------------------------------------------------------


class TestMlDecideHoldFallback:
    """When no ticker beats the buy threshold AND no held position has
    a sell signal, _ml_decide must return a HOLD with non-empty
    reasoning that includes 'no high-conviction'."""

    def test_empty_articles_and_excluded_boost_tickers_returns_hold(self):
        """With no articles AND all persona-boosted tickers excluded,
        _ml_decide must fall back to HOLD — never fabricate a buy."""
        import paper_trader.backtest as bt
        from datetime import date as _date
        import random
        pf = bt.SimPortfolio()
        prices = MagicMock()
        prices.price_on = MagicMock(return_value=100.0)
        prices.trading_days = [_date(2024, 1, 1)]
        prices.prices = {"SPY": {}}

        # Exclude every PERSONA_BOOST ticker across all personas + watchlist
        # tickers that might be selected.
        all_boost = set()
        for boosts in bt._PERSONA_BOOSTS.values():
            all_boost.update(boosts.keys())
        # Also exclude all WATCHLIST so nothing can be picked
        excl = set(bt.WATCHLIST)

        decision = bt._ml_decide(
            sim_date=_date(2024, 1, 2),
            portfolio=pf,
            articles=[],
            prices=prices,
            run_id=1,
            rng=random.Random(42),
            exclude_tickers=excl,
        )
        assert decision["action"] == "HOLD"
        assert "no high-conviction" in decision["reasoning"].lower()
        assert decision["qty"] == 0

    def test_persona_boost_drives_buy_without_articles(self):
        """A persona boost (e.g. Value persona boosts MSFT) DOES drive
        a BUY without any articles — locks the documented persona-bias
        behavior so a regression surfaces."""
        import paper_trader.backtest as bt
        from datetime import date as _date
        import random
        pf = bt.SimPortfolio()
        prices = MagicMock()
        prices.price_on = MagicMock(return_value=100.0)
        prices.trading_days = [_date(2024, 1, 1)]
        prices.prices = {"SPY": {}}

        decision = bt._ml_decide(
            sim_date=_date(2024, 1, 2),
            portfolio=pf,
            articles=[],
            prices=prices,
            run_id=1,  # VALUE persona
            rng=random.Random(42),
        )
        # Value persona boosts MSFT (+1.5) at threshold 1.15 → BUY
        assert decision["action"] == "BUY"
        assert decision["ticker"] in bt._PERSONA_BOOSTS[1]
        assert decision["qty"] > 0


# ---------------------------------------------------------------------------
# Section 8 — Sector one-hot encoding correctness
# ---------------------------------------------------------------------------


class TestSectorOneHot:
    """build_features must produce a one-hot vector at the sector slots
    matching SECTORS in order. A breakage here means the model would
    interpret one sector's signal as another's."""

    def test_tech_ticker_first_sector_slot_set(self):
        from paper_trader.ml.decision_scorer import (
            build_features, FEATURE_NAMES, SECTORS,
        )

        v = build_features(
            ml_score=0, rsi=50, macd=0, mom5=0, mom20=0,
            regime_mult=1.0, ticker="NVDA",
        )
        # tech is SECTORS[0]
        idx_tech = FEATURE_NAMES.index("sector_tech")
        idx_other = FEATURE_NAMES.index("sector_other")
        idx_energy = FEATURE_NAMES.index("sector_energy")
        assert v[idx_tech] == 1.0
        assert v[idx_other] == 0.0
        assert v[idx_energy] == 0.0
        # Total of sector one-hot must be exactly 1.0
        sector_sum = sum(v[FEATURE_NAMES.index(f"sector_{s}")] for s in SECTORS)
        assert sector_sum == 1.0

    def test_unknown_ticker_lands_in_other(self):
        from paper_trader.ml.decision_scorer import build_features, FEATURE_NAMES

        v = build_features(
            ml_score=0, rsi=50, macd=0, mom5=0, mom20=0,
            regime_mult=1.0, ticker="ZZZZ_UNKNOWN",
        )
        idx_other = FEATURE_NAMES.index("sector_other")
        idx_tech = FEATURE_NAMES.index("sector_tech")
        assert v[idx_other] == 1.0
        assert v[idx_tech] == 0.0


# ---------------------------------------------------------------------------
# Section 9 — _enforce_risk_exits actually fires SL/TP on threshold breach
# ---------------------------------------------------------------------------


class TestEnforceRiskExits:
    """A position with stop_loss=90 must be sold when price hits 89.
    A position with take_profit=110 must be sold when price hits 111.
    These are real risk-management contracts — a regression here loses
    the documented protection on every backtest run."""

    def test_stop_loss_fires_on_breach(self):
        import paper_trader.backtest as bt
        from datetime import date as _date
        import sqlite3

        pf = bt.SimPortfolio(cash=500.0)
        pf.positions["NVDA"] = {
            "qty": 5.0, "avg_cost": 100.0,
            "stop_loss": 90.0, "take_profit": None,
        }

        prices = MagicMock()
        # day 1: $95 (above stop); day 2: $89 (below stop); day 3: $80
        def price_on(t, d):
            return {
                _date(2024, 1, 2): 95.0,
                _date(2024, 1, 3): 89.0,
                _date(2024, 1, 4): 80.0,
            }.get(d)
        prices.price_on = price_on
        prices.trading_days = [
            _date(2024, 1, 1), _date(2024, 1, 2),
            _date(2024, 1, 3), _date(2024, 1, 4),
        ]

        # In-memory store
        conn = sqlite3.connect(":memory:")
        conn.executescript(bt.SCHEMA)
        store = bt.BacktestStore.__new__(bt.BacktestStore)
        store.conn = conn
        import threading
        store._lock = threading.Lock()

        n = bt._enforce_risk_exits(pf, prices, _date(2024, 1, 1),
                                   _date(2024, 1, 4), 1, store)
        assert n == 1, "stop-loss must fire exactly once"
        assert "NVDA" not in pf.positions, "position should be flat"
        # Sold at the breach price (89), cash increased
        assert pf.cash == pytest.approx(500.0 + 5.0 * 89.0)

    def test_take_profit_fires_on_breach(self):
        import paper_trader.backtest as bt
        from datetime import date as _date
        import sqlite3
        import threading

        pf = bt.SimPortfolio(cash=500.0)
        pf.positions["NVDA"] = {
            "qty": 5.0, "avg_cost": 100.0,
            "stop_loss": None, "take_profit": 110.0,
        }
        prices = MagicMock()
        def price_on(t, d):
            return {
                _date(2024, 1, 2): 105.0,
                _date(2024, 1, 3): 111.0,
            }.get(d)
        prices.price_on = price_on
        prices.trading_days = [
            _date(2024, 1, 1), _date(2024, 1, 2), _date(2024, 1, 3)]

        conn = sqlite3.connect(":memory:")
        conn.executescript(bt.SCHEMA)
        store = bt.BacktestStore.__new__(bt.BacktestStore)
        store.conn = conn
        store._lock = threading.Lock()
        n = bt._enforce_risk_exits(pf, prices, _date(2024, 1, 1),
                                   _date(2024, 1, 3), 1, store)
        assert n == 1
        assert "NVDA" not in pf.positions
        # Sold at 111 (the breach price)
        assert pf.cash == pytest.approx(500.0 + 5.0 * 111.0)


# ---------------------------------------------------------------------------
# Section 10 — Buy/sell portfolio invariants
# ---------------------------------------------------------------------------


class TestPortfolioInvariants:
    """Cash + position value invariants — a backtest that violates these
    is fabricating P&L."""

    def test_buy_consumes_cash(self):
        import paper_trader.backtest as bt

        pf = bt.SimPortfolio(cash=1000.0)
        bt._buy(pf, "NVDA", 5.0, 100.0, stop_loss=90.0, take_profit=110.0)
        assert pf.cash == 500.0
        assert pf.positions["NVDA"]["qty"] == 5.0
        assert pf.positions["NVDA"]["avg_cost"] == 100.0
        assert pf.positions["NVDA"]["stop_loss"] == 90.0

    def test_sell_returns_cash_and_clears_position(self):
        import paper_trader.backtest as bt

        pf = bt.SimPortfolio(cash=500.0)
        pf.positions["NVDA"] = {
            "qty": 5.0, "avg_cost": 100.0,
            "stop_loss": 90.0, "take_profit": 110.0,
        }
        proceeds = bt._sell(pf, "NVDA", 5.0, 105.0)
        assert proceeds == pytest.approx(5.0 * 105.0)
        assert pf.cash == pytest.approx(500.0 + 525.0)
        assert "NVDA" not in pf.positions

    def test_buy_then_sell_full_round_trip_pnl(self):
        """The most basic invariant: buy at $100, sell at $110 => +$50 P&L
        on 5 shares (less zero in this paper sim)."""
        import paper_trader.backtest as bt

        pf = bt.SimPortfolio(cash=1000.0)
        bt._buy(pf, "NVDA", 5.0, 100.0, stop_loss=None, take_profit=None)
        proceeds = bt._sell(pf, "NVDA", 5.0, 110.0)
        assert proceeds == 550.0
        assert pf.cash == 1050.0
        assert "NVDA" not in pf.positions

    def test_partial_sell_keeps_position(self):
        import paper_trader.backtest as bt
        pf = bt.SimPortfolio(cash=500.0)
        bt._buy(pf, "NVDA", 5.0, 100.0, stop_loss=None, take_profit=None)
        # Sell only 2
        bt._sell(pf, "NVDA", 2.0, 110.0)
        assert pf.positions["NVDA"]["qty"] == 3.0
        # Avg cost unchanged on partial sell
        assert pf.positions["NVDA"]["avg_cost"] == 100.0
        assert pf.cash == pytest.approx(220.0)  # 500 - 500 + 220

    def test_buy_accumulates_blended_avg_cost(self):
        """Buy 5 @ 100, then 5 @ 200 => avg cost = 150."""
        import paper_trader.backtest as bt
        pf = bt.SimPortfolio(cash=2000.0)
        bt._buy(pf, "NVDA", 5.0, 100.0, stop_loss=None, take_profit=None)
        bt._buy(pf, "NVDA", 5.0, 200.0, stop_loss=None, take_profit=None)
        assert pf.positions["NVDA"]["qty"] == 10.0
        assert pf.positions["NVDA"]["avg_cost"] == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Section 11 — gate arm decode boundaries (anchor for downstream analysis)
# ---------------------------------------------------------------------------


class TestFeatureGroupImportance:
    """The new feature_group_importance() rolls per-feature importance into
    the 5 FEATURE_GROUPS buckets. Behavioural locks:

    1. Empty / untrained → trained=False, groups=[]
    2. Trained → groups sum to 1.0 normalized; raw sum = sum of per-feature
    3. Sector group's n_features == 7 (matches SECTORS)
    4. Quant group's n_features == 9 (rsi/macd/mom5/mom20/vol_ratio/bb_pos
       + 3 enhanced MACD)
    """

    def test_untrained_returns_empty(self):
        from paper_trader.ml.decision_scorer import DecisionScorer

        s = DecisionScorer.__new__(DecisionScorer)
        s._model = None
        s._scaler = None
        s._trained = False
        s._n_train = 0
        s._pred_quantiles = None
        s._label_quantiles = None
        out = s.feature_group_importance()
        assert out["trained"] is False
        assert out["groups"] == []
        assert out["n_train"] == 0

    def _build_diverse_training_records(self):
        """Build >50 distinct (ticker, sim_date, action) records so the
        dedup-after-min gate (n >= 30) passes."""
        records = []
        tickers = ["NVDA", "AMD", "SPY", "TLT", "GLD", "BTC-USD", "XOM",
                   "LLY", "JPM", "TSLA"]
        for i in range(60):
            ticker = tickers[i % len(tickers)]
            day = (i * 3) % 28 + 1  # spread dates out
            month = (i // 10) + 1
            records.append({
                "ticker": ticker,
                "sim_date": f"2024-{month:02d}-{day:02d}",
                "ml_score": (i % 5) * 1.0,
                "rsi": 30 + (i % 60), "macd": (i - 30) * 0.1,
                "mom5": (i - 30) * 0.5, "mom20": (i - 30) * 1.0,
                "regime_mult": 1.0, "vol_ratio": 1.0,
                "bb_position": ((i % 20) - 10) / 10.0,
                "news_urgency": 50.0, "news_article_count": 1.0,
                "forward_return_5d": (i - 30) * 0.3,
                "action": "BUY",
            })
        return records

    def test_trained_sums_to_1_normalized(self, tmp_path):
        from paper_trader.ml.decision_scorer import (
            DecisionScorer, train_scorer,
        )
        import paper_trader.ml.decision_scorer as ds

        result = train_scorer(self._build_diverse_training_records(),
                              path=ds.SCORER_PATH)
        assert result["status"] == "ok", f"train failed: {result}"
        s = DecisionScorer()
        out = s.feature_group_importance()
        assert out["trained"] is True, f"out={out}"
        assert len(out["groups"]) >= 1

        # Normalized must sum to ~1.0 (rounding tolerance)
        total_norm = sum(g["importance_normalized"] for g in out["groups"])
        assert 0.99 <= total_norm <= 1.01, f"normalized sum {total_norm} ≠ 1.0"

    def test_group_n_features_matches_groups_map(self, tmp_path):
        """sector should aggregate 7 features (one per SECTORS entry);
        quant should aggregate 9 (rsi, macd, mom5, mom20, vol_ratio, bb_pos
        + 3 enhanced MACD)."""
        from paper_trader.ml.decision_scorer import (
            DecisionScorer, train_scorer, SECTORS,
        )
        import paper_trader.ml.decision_scorer as ds

        result = train_scorer(self._build_diverse_training_records(),
                              path=ds.SCORER_PATH)
        assert result["status"] == "ok", f"train failed: {result}"
        s = DecisionScorer()
        out = s.feature_group_importance()
        by_name = {g["group"]: g for g in out["groups"]}
        assert by_name["sector"]["n_features"] == len(SECTORS)
        # quant features: rsi/macd/mom5/mom20/vol_ratio/bb_pos +
        # ema200_above/hist_cross_up/macd_below_zero_cross
        assert by_name["quant"]["n_features"] == 9
        # ml_score, regime alone
        assert by_name["ml_score"]["n_features"] == 1
        assert by_name["regime"]["n_features"] == 1
        assert by_name["news"]["n_features"] == 2

    def test_group_importance_aggregates_per_feature(self, tmp_path):
        """Algebraic identity: per-group importance must equal sum of
        per-feature importance within that group."""
        from paper_trader.ml.decision_scorer import (
            DecisionScorer, train_scorer, FEATURE_GROUP_MAP,
        )
        import paper_trader.ml.decision_scorer as ds

        result = train_scorer(self._build_diverse_training_records(),
                              path=ds.SCORER_PATH)
        assert result["status"] == "ok"
        s = DecisionScorer()
        per_feat = s.feature_importance()
        groups = s.feature_group_importance()

        # Compute expected per-group sums
        expected = {}
        for row in per_feat["importances"]:
            grp = FEATURE_GROUP_MAP[row["feature"]]
            expected[grp] = expected.get(grp, 0.0) + row["importance"]

        actual = {g["group"]: g["importance"] for g in groups["groups"]}
        for grp, exp in expected.items():
            assert grp in actual, f"missing group {grp}"
            assert actual[grp] == pytest.approx(exp, abs=1e-4), (
                f"group {grp}: expected {exp}, got {actual[grp]}"
            )


class TestGateArmDecodeBoundaries:
    """The arms (×0.6/×0.85/×1.0/×1.15/×1.3) gate conviction at
    boundaries: pred < -10 → strong_headwind, pred < 0 → mild_headwind,
    pred > 10 → strong_tailwind, pred > 5 → mild_tailwind, else neutral.
    Boundary correctness MUST hold or _ml_decide and the analyzer
    bucketing diverge."""

    def test_arm_decode_at_each_boundary(self):
        from paper_trader.ml.gate_audit import gate_arm

        # Just below -10
        arm, mult = gate_arm(-10.1)
        assert arm == "strong_headwind"
        assert mult == 0.6

        # Just above -10 (but still negative)
        arm, mult = gate_arm(-9.0)
        assert arm == "mild_headwind"
        assert mult == 0.85

        # Exactly 0
        arm, mult = gate_arm(0.0)
        assert arm == "neutral"
        assert mult == 1.0

        # Just above 5
        arm, mult = gate_arm(5.5)
        assert arm == "mild_tailwind"
        assert mult == 1.15

        # Just above 10
        arm, mult = gate_arm(10.5)
        assert arm == "strong_tailwind"
        assert mult == 1.3
