"""Scorer honesty across dashboard panels.

The DecisionScorer is an unbounded MLP that extrapolates to nonsense
(observed: +709% / -244% 5d "predictions" for an optical-networking stock)
on off-distribution feature vectors. ``predict_with_meta()`` clamps the
point estimate to ±50% and flags ``off_distribution`` so a clamped
floor/ceiling is never mistaken for a confident signal.

``_live_scorer_predictions`` already surfaces the flag. The primary
``/api/scorer-predictions`` and ``/api/position-thesis`` endpoints, plus
``/api/disagreement`` rows, did NOT — they called the scalar ``predict()``
and dropped the metadata, so a trader (and the unified conviction board
that reads position-thesis) saw a clamped ±50 as gospel. These tests lock
the honest contract end-to-end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import store as store_mod
from paper_trader.store import Store
# Bind the real class at import so _scorer_returning() still works after a
# test monkeypatches paper_trader.ml.decision_scorer.DecisionScorer.
from paper_trader.ml.decision_scorer import DecisionScorer as _RealScorer


class _FixedModel:
    """predict() returns a value we control, so clamp/metadata can be
    tested without MLP training noise."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X) -> np.ndarray:
        return np.array([self.value], dtype=np.float64)


def _scorer_returning(value: float):
    s = _RealScorer()
    s._model = _FixedModel(value)
    s._scaler = None
    s._trained = True
    s._n_train = 1000
    return s


_QUANT = {
    "NVDA": {"rsi": 64.7, "RSI": 64.7, "macd_signal": 0.4, "MACD": "bullish",
             "mom_5d": 4.7, "mom_20d": 11.7, "vol_ratio": 1.1, "bb_position": 0.5},
    "SPY": {"mom_5d": 0.2},
}


@pytest.fixture
def held_client(tmp_path, monkeypatch):
    """Flask client with one open NVDA stock position and all network /
    DB-reaching helpers stubbed deterministic + offline."""
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    s.record_trade("NVDA", "BUY", 0.8, 225.0)
    s.upsert_position("NVDA", "stock", 0.8, 225.0)
    s.update_portfolio(cash=820.0, total_value=1000.0, positions=[
        {"ticker": "NVDA", "type": "stock", "qty": 0.8, "avg_cost": 225.0,
         "current_price": 225.0, "unrealized_pl": 0.0},
    ])

    import paper_trader.strategy as strat
    import paper_trader.signals as sigs
    import paper_trader.analytics.position_thesis as pt
    monkeypatch.setattr(strat, "get_quant_signals_live",
                        lambda tks: {t: _QUANT.get(t, {}) for t in tks})
    monkeypatch.setattr(sigs, "ticker_sentiments",
                        lambda tks, hours=4: [])
    monkeypatch.setattr(pt, "_ticker_news", lambda tk, hours=24, limit=3: {
        "headlines": [], "bull": 0, "bear": 0, "n": 0,
        "avg_score": 0.0, "max_score": 0.0})

    from paper_trader import dashboard
    dashboard.app.config["TESTING"] = True
    try:
        with dashboard.app.test_client() as client:
            yield client, monkeypatch
    finally:
        s.close()


# ── pure function: build_thesis_cards propagates the honesty flag ─────────

def test_build_thesis_cards_propagates_off_distribution():
    from paper_trader.analytics.position_thesis import build_thesis_cards
    import paper_trader.analytics.position_thesis as pt
    pt._ticker_news = lambda tk, hours=24, limit=3: {  # type: ignore
        "headlines": [], "bull": 0, "bear": 0, "n": 0,
        "avg_score": 0.0, "max_score": 0.0}

    positions = [{"ticker": "LITE", "type": "stock", "qty": 1.0,
                  "avg_cost": 100.0, "current_price": 99.0,
                  "unrealized_pl": -1.0, "opened_at": None}]
    scorer_preds = [{"ticker": "LITE", "pred_5d_return_pct": 50.0,
                     "verdict": "STRONG_HOLD", "off_distribution": True,
                     "raw_pred_5d_return_pct": 244.248}]
    out = build_thesis_cards(positions, [], scorer_preds, {"LITE": {}})
    card = out["cards"][0]
    assert card["scorer_pred_5d"] == 50.0
    assert card["off_distribution"] is True
    assert card["raw_pred_5d_return_pct"] == pytest.approx(244.248)


def test_build_thesis_cards_in_distribution_flag_false():
    from paper_trader.analytics.position_thesis import build_thesis_cards
    import paper_trader.analytics.position_thesis as pt
    pt._ticker_news = lambda tk, hours=24, limit=3: {  # type: ignore
        "headlines": [], "bull": 0, "bear": 0, "n": 0,
        "avg_score": 0.0, "max_score": 0.0}
    positions = [{"ticker": "NVDA", "type": "stock", "qty": 1.0,
                  "avg_cost": 100.0, "current_price": 101.0,
                  "unrealized_pl": 1.0, "opened_at": None}]
    scorer_preds = [{"ticker": "NVDA", "pred_5d_return_pct": -8.3,
                     "verdict": "EXIT", "off_distribution": False}]
    card = build_thesis_cards(positions, [], scorer_preds, {"NVDA": {}})["cards"][0]
    assert card["scorer_pred_5d"] == pytest.approx(-8.3)
    assert card["off_distribution"] is False
    assert card.get("raw_pred_5d_return_pct") is None


# ── /api/scorer-predictions emits the honesty flag ───────────────────────

def test_scorer_predictions_clamps_and_flags_extrapolation(held_client):
    client, mp = held_client
    import paper_trader.ml.decision_scorer as ds
    mp.setattr(ds, "DecisionScorer", lambda: _scorer_returning(244.248))

    data = client.get("/api/scorer-predictions").get_json()
    assert data["is_trained"] is True
    row = next(r for r in data["predictions"] if r["ticker"] == "NVDA")
    assert row["pred_5d_return_pct"] == pytest.approx(50.0)  # clamped
    assert row["off_distribution"] is True
    assert row["raw_pred_5d_return_pct"] == pytest.approx(244.248)


def test_scorer_predictions_in_distribution_has_no_raw(held_client):
    client, mp = held_client
    import paper_trader.ml.decision_scorer as ds
    mp.setattr(ds, "DecisionScorer", lambda: _scorer_returning(-8.3))

    row = next(r for r in client.get("/api/scorer-predictions").get_json()
               ["predictions"] if r["ticker"] == "NVDA")
    assert row["pred_5d_return_pct"] == pytest.approx(-8.3)
    assert row["off_distribution"] is False
    assert "raw_pred_5d_return_pct" not in row


# ── /api/position-thesis card carries the clamped value + flag ───────────

def test_position_thesis_card_is_clamped_and_flagged(held_client):
    client, mp = held_client
    import paper_trader.ml.decision_scorer as ds
    mp.setattr(ds, "DecisionScorer", lambda: _scorer_returning(-244.248))

    cards = client.get("/api/position-thesis").get_json()["cards"]
    card = next(c for c in cards if c["ticker"] == "NVDA")
    assert card["scorer_pred_5d"] == pytest.approx(-50.0)  # clamp floor
    assert card["off_distribution"] is True
    assert card["raw_pred_5d_return_pct"] == pytest.approx(-244.248)


# ── /api/disagreement rows expose the flag ───────────────────────────────

def test_disagreement_rows_propagate_off_distribution(held_client):
    client, mp = held_client
    from paper_trader import dashboard
    import paper_trader.analytics.scorer_confidence as sc

    mp.setattr(dashboard, "_live_scorer_predictions", lambda scorer: [
        {"ticker": "NVDA", "pred_5d_return_pct": 50.0,
         "verdict": "STRONG_HOLD", "off_distribution": True,
         "raw_pred_5d_return_pct": 709.874}])
    mp.setattr(sc, "build_scorer_confidence",
               lambda outcomes, scorer: {"overall": {"n": 10}})
    mp.setattr(sc, "interval_for", lambda pred, conf: {"low": 0.0, "high": 0.0})

    data = client.get("/api/disagreement").get_json()
    row = next(r for r in data["rows"] if r["ticker"] == "NVDA")
    assert row["off_distribution"] is True
    assert row["raw_pred_5d_return_pct"] == pytest.approx(709.874)
