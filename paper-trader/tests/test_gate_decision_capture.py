"""Tests for the additive gate-decision capture in
`run_continuous_backtests._compute_decision_outcomes` (2026-05-18 feature).

`_parse_gate_decision` extracts the conviction gate's TRUE then-deployed
decision (`scorer=±X%` + off-distribution abstention) from a backtest
decision's reasoning string, and `_compute_decision_outcomes` now records it
as additive `gate_scorer_pred` / `gate_off_dist` keys. This lets a gate
diagnostic measure the gate's realized effect instead of re-predicting with
today's pickle (the documented `gate_pnl` reconstruction residual).

All offline / deterministic. Exact-value locks (mirroring
test_horizon_audit's additive-capture style): a regex/format change must
update the literals deliberately. The 5d-path byte-identity assertions are
the regression anchor — the additive keys must NOT perturb scorer training.
"""
from __future__ import annotations

from datetime import date

import pytest

import run_continuous_backtests as rcb
from paper_trader.backtest import BacktestRun, BacktestStore


# ───────────────────────── _parse_gate_decision (pure) ──────────────────────

class TestParseGateDecision:
    def test_buy_in_distribution_real_prediction(self):
        # `_ml_decide` BUY note (gate acted): trailing ` scorer=+5.2%`.
        pred, off = rcb._parse_gate_decision(
            "ML+quant: SOXL score=2.45 regime=bull RSI=40 "
            "news_count=0 news_urg=0.0 conviction=40% scorer=+5.2%"
        )
        assert pred == 5.2
        assert off is False

    def test_buy_off_distribution_abstention(self):
        # Off-dist guard fired → gate ABSTAINED; clamped ±50 surfaced.
        pred, off = rcb._parse_gate_decision(
            "ML+quant: LITE score=3.10 regime=bull conviction=25% "
            "scorer=-50.0%(off-dist,gate-skipped)"
        )
        assert pred == -50.0
        assert off is True

    def test_zero_prediction_boundary(self):
        # `:+.1f` always emits a sign; +0.0 must parse to 0.0 (not None).
        pred, off = rcb._parse_gate_decision(
            "ML+quant: NVDA score=1.50 conviction=10% scorer=+0.0%"
        )
        assert pred == 0.0
        assert off is False

    def test_untrained_cycle_no_scorer_token(self):
        # Scorer untrained / n_train<500 → `_ml_decide` emits NO scorer note.
        pred, off = rcb._parse_gate_decision(
            "ML+quant: TQQQ score=2.00 regime=bull news_count=0 "
            "news_urg=0.0 conviction=10%"
        )
        assert pred is None
        assert off is None

    def test_sell_reasoning_has_no_gate(self):
        # SELL reasoning never carries `scorer=` (gate is BUY-only).
        pred, off = rcb._parse_gate_decision(
            "ML+quant: SOXL score=-1.20 regime=bear RSI=72 "
            "news_count=0 news_urg=0.0 — reducing"
        )
        assert pred is None
        assert off is None

    def test_score_vs_scorer_disambiguation(self):
        # The captured value is the GATE's `scorer=`, never the `ml_score`
        # `score=` (the dual side of the documented first-match rule).
        pred, off = rcb._parse_gate_decision(
            "ML+quant: AMD score=9.99 regime=bull conviction=25% scorer=+3.1%"
        )
        assert pred == 3.1
        assert off is False

    def test_garbage_and_none_inputs_never_raise(self):
        assert rcb._parse_gate_decision(None) == (None, None)
        assert rcb._parse_gate_decision("") == (None, None)
        # A malformed numeric (regex requires [+-]?[0-9.]+) → no match.
        assert rcb._parse_gate_decision("conviction=10% scorer=abc%") == (
            None, None)
        # Non-string garbage must not raise.
        assert rcb._parse_gate_decision(12345) == (None, None)  # type: ignore[arg-type]


# ──────────── end-to-end through _compute_decision_outcomes ──────────────────

def _engine_with_decision(tmp_path, synthetic_prices, *, action, ticker,
                          day_index, reasoning):
    """Mirror test_horizon_audit's harness: a FILLED decision in a fresh
    temp BacktestStore + a SimpleNamespace engine over synthetic_prices."""
    import types
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
    return types.SimpleNamespace(store=store, prices=synthetic_prices)


class TestComputeDecisionOutcomesGateCapture:
    """synthetic_prices: NVDA close = 100 + 2*i over 51 trading days
    (day 10 → 120, day 15 → 130)."""

    def test_gate_fields_land_and_5d_path_unchanged(self, tmp_path,
                                                    synthetic_prices):
        eng = _engine_with_decision(
            tmp_path, synthetic_prices, action="BUY", ticker="NVDA",
            day_index=10,
            reasoning="ML+quant: NVDA score=2.00 regime=bull RSI=40 "
                      "news_count=0 news_urg=0.0 conviction=10% scorer=+7.5%",
        )
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                            end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        assert len(outs) == 1
        o = outs[0]
        # Additive gate capture lands with exact parsed values.
        assert o["gate_scorer_pred"] == 7.5
        assert o["gate_off_dist"] is False
        # Regression anchor: the scorer-training 5d path is byte-identical
        # ((130-120)/120*100) — the additive keys must not perturb it.
        assert o["forward_return_5d"] == pytest.approx(8.3333, abs=1e-4)

    def test_untrained_cycle_records_none_none(self, tmp_path,
                                               synthetic_prices):
        eng = _engine_with_decision(
            tmp_path, synthetic_prices, action="BUY", ticker="NVDA",
            day_index=10,
            reasoning="ML+quant: NVDA score=2.00 regime=bull "
                      "news_count=0 news_urg=0.0 conviction=10%",
        )
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                            end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        assert len(outs) == 1
        o = outs[0]
        assert o["gate_scorer_pred"] is None
        assert o["gate_off_dist"] is None
        # 5d outcome still correct & present (training unaffected).
        assert o["forward_return_5d"] == pytest.approx(8.3333, abs=1e-4)
