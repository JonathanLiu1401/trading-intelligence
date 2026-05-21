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


# ───────────────────────── _parse_conviction_pct (pure) ──────────────────────

class TestParseConvictionPct:
    """Pins the conviction-sizing capture (2026-05-21 feature). The token
    ``conviction=X%`` is already emitted by every BUY reasoning in
    ``_ml_decide``; capturing it as a fraction in [0,1] unlocks
    sizing-weighted realized analysis. Mirrors ``TestParseGateDecision``'s
    exact-literal style — a regex change must update the locks deliberately."""

    def test_typical_buy_conviction(self):
        # `_ml_decide`'s standard BUY reasoning carries `conviction=25%`.
        # Parser must return the FRACTION (0.25), not the integer 25 —
        # matching the inference-side variable's unit so a downstream
        # consumer can multiply realized_return * conviction_pct directly.
        assert rcb._parse_conviction_pct(
            "ML+quant: NVDA score=2.0 regime=bull conviction=25% scorer=+5.2%"
        ) == 0.25

    def test_leveraged_etf_high_conviction(self):
        # The leveraged-ETF arm allows convictions up to 40%; 40% must
        # parse as 0.40 (defensively below the 1.0 clamp).
        assert rcb._parse_conviction_pct(
            "ML+quant: SOXL score=8.0 regime=bull conviction=40% scorer=+12%"
        ) == 0.40

    def test_zero_conviction_boundary(self):
        # `_ml_decide` can emit `conviction=0%` for tiny notional trades
        # (rare but possible); parser must return 0.0, NOT None — None
        # means "no conviction token at all", which is semantically distinct
        # from "the gate sized 0%". Mirrors the `+0.0%` gate-decision
        # boundary test exactly.
        assert rcb._parse_conviction_pct(
            "ML+quant: PLTR score=1.5 conviction=0% scorer=+0.0%"
        ) == 0.0

    def test_hold_reasoning_no_conviction(self):
        # `_ml_decide` HOLD reasoning never includes `conviction=` — the
        # token is BUY-only. Must read None so downstream analyses can
        # filter HOLD rows from sizing studies.
        assert rcb._parse_conviction_pct(
            "ML+quant: no high-conviction signal 2025-06-15 regime=sideways"
        ) is None

    def test_sell_reasoning_no_conviction(self):
        # `_ml_decide` SELL reasoning never includes `conviction=` either
        # (the SELL path emits `— reducing` instead). Parser returns None,
        # matching the `gate_scorer_pred` SELL convention.
        assert rcb._parse_conviction_pct(
            "ML+quant: SOXL score=-1.20 regime=bear RSI=72 "
            "news_count=0 news_urg=0.0 — reducing"
        ) is None

    def test_garbage_and_none_inputs_never_raise(self):
        assert rcb._parse_conviction_pct(None) is None
        assert rcb._parse_conviction_pct("") is None
        # Non-digit value → no regex match → None (not a parse exception).
        assert rcb._parse_conviction_pct("conviction=abc%") is None
        # Non-string garbage must not raise.
        assert rcb._parse_conviction_pct(12345) is None  # type: ignore[arg-type]

    def test_out_of_range_value_clamped(self):
        # A malformed reasoning emitting an impossible percentage (>100%)
        # must clamp to 1.0 — never propagating an impossible sizing into
        # the outcomes corpus. Defense-in-depth: `_ml_decide` caps at 95%
        # in source, so this branch only triggers on a hand-crafted or
        # corrupted reasoning string.
        assert rcb._parse_conviction_pct(
            "ML+quant: X score=99 conviction=900%"
        ) == 1.0

    def test_first_match_disambiguation(self):
        # If multiple `conviction=` tokens somehow appear in one reasoning
        # (impossible today but defensive), the FIRST match wins — same
        # discipline as the existing `ml_score` `score=` first-match rule
        # documented in `_parse_gate_decision`.
        assert rcb._parse_conviction_pct(
            "ML+quant: X conviction=25% later conviction=99%"
        ) == 0.25


# ──────────── conviction_pct lands in decision_outcomes row ──────────────────

class TestComputeDecisionOutcomesConvictionCapture:
    """End-to-end: the conviction parsed from reasoning lands in the
    additive ``conviction_pct`` outcome key, exactly like the gate-decision
    capture lands ``gate_scorer_pred`` / ``gate_off_dist``. Synthetic-prices
    fixture is shared with the gate-capture suite above."""

    def test_buy_conviction_pct_lands_and_5d_unchanged(self, tmp_path,
                                                       synthetic_prices):
        eng = _engine_with_decision(
            tmp_path, synthetic_prices, action="BUY", ticker="NVDA",
            day_index=10,
            reasoning=("ML+quant: NVDA score=2.00 regime=bull RSI=40 "
                       "news_count=0 news_urg=0.0 conviction=25% "
                       "scorer=+7.5%"),
        )
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                            end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        assert len(outs) == 1
        o = outs[0]
        # Additive sizing capture lands as a fraction in [0,1].
        assert o["conviction_pct"] == 0.25
        # Regression anchor: the 5d training path is byte-identical
        # (additive keys must not perturb the scorer's only target).
        assert o["forward_return_5d"] == pytest.approx(8.3333, abs=1e-4)
        # The companion gate-capture fields are still populated; the
        # conviction addition does not shadow them.
        assert o["gate_scorer_pred"] == 7.5
        assert o["gate_off_dist"] is False

    def test_hold_or_missing_conviction_records_none(self, tmp_path,
                                                     synthetic_prices):
        # A BUY decision whose reasoning was hand-crafted without the
        # conviction token — captured as None, not 0.0 (the two are
        # semantically distinct: 0.0 means "gate sized 0%", None means
        # "no conviction token was emitted").
        eng = _engine_with_decision(
            tmp_path, synthetic_prices, action="BUY", ticker="NVDA",
            day_index=10,
            reasoning="ML+quant: NVDA score=2.00 regime=bull "
                      "news_count=0 news_urg=0.0",
        )
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                            end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        assert len(outs) == 1
        assert outs[0]["conviction_pct"] is None
        # 5d outcome still correct (training unaffected by missing token).
        assert outs[0]["forward_return_5d"] == pytest.approx(8.3333, abs=1e-4)
