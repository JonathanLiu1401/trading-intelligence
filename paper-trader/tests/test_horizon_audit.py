"""Tests for the additive multi-horizon outcome capture
(`run_continuous_backtests._compute_decision_outcomes`) and the
`paper_trader.ml.horizon_audit` read-only diagnostic.

All offline / deterministic. Exact-value verdict + IC locks (mirroring
test_gate_audit / test_baseline_compare): a threshold or formula change
must update the literals deliberately.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

import run_continuous_backtests as rcb
from paper_trader.backtest import BacktestRun, BacktestStore
from paper_trader.ml import horizon_audit as ha
from paper_trader.ml.horizon_audit import (
    EDGE_FLOOR,
    IC_MARGIN,
    MIN_PAIRS,
    horizon_audit_report,
)


# ───────────── additive multi-horizon capture in _compute_decision_outcomes ──

def _engine_with_decision(tmp_path, synthetic_prices, *, action, ticker,
                           day_index, reasoning):
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
    engine = types.SimpleNamespace(store=store, prices=synthetic_prices)
    return engine, sim_date


class TestComputeDecisionOutcomesMultiHorizon:
    """synthetic_prices: NVDA close = 100 + 2*i over 51 trading days.
    day 10 → 120, day 15 → 130, day 20 → 140, day 30 → 160."""

    def test_exact_5_10_20d_returns(self, tmp_path, synthetic_prices):
        eng, _ = _engine_with_decision(
            tmp_path, synthetic_prices, action="BUY", ticker="NVDA",
            day_index=10,
            reasoning="ML+quant: NVDA score=2.00 regime=bull RSI=40 "
                      "news_count=0 news_urg=0.0 conviction=10%",
        )
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                            end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        assert len(outs) == 1
        o = outs[0]
        # 5d unchanged (regression anchor): (130-120)/120*100
        assert o["forward_return_5d"] == pytest.approx(8.3333, abs=1e-4)
        # additive horizons — (140-120)/120 and (160-120)/120
        assert o["forward_return_10d"] == pytest.approx(16.6667, abs=1e-4)
        assert o["forward_return_20d"] == pytest.approx(33.3333, abs=1e-4)

    def test_long_horizon_past_history_is_none_5d_still_emitted(
            self, tmp_path, synthetic_prices):
        """A decision whose 10d/20d window runs past cached price history must
        still emit the row with a correct 5d outcome and None for the longer
        horizons — the scorer-training 5d path must be byte-identical."""
        # day 45 → price 190; 5d→idx 50 (price 200, valid); 10d→55, 20d→65 (past)
        eng, _ = _engine_with_decision(
            tmp_path, synthetic_prices, action="BUY", ticker="NVDA",
            day_index=45,
            reasoning="ML+quant: NVDA score=2.00 regime=bull news_count=0 "
                      "news_urg=0.0",
        )
        runs = [BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                            end_date="2025-12-31")]
        outs = rcb._compute_decision_outcomes(eng, runs)
        assert len(outs) == 1
        o = outs[0]
        assert o["forward_return_5d"] == pytest.approx(5.2632, abs=1e-4)
        assert o["forward_return_10d"] is None
        assert o["forward_return_20d"] is None


# ───────────────────────── horizon_audit_report ──────────────────────────────

def _rows(n, *, fwd5, fwd10, fwd20, ml=lambda i: float(i),
          mom20=None, action="BUY"):
    """n synthetic outcome rows. fwd* are callables i->value or None to omit
    the key entirely (legacy-row shape)."""
    out = []
    for i in range(n):
        r = {"action": action, "ticker": "NVDA", "sim_date": "2025-01-01",
             "ml_score": ml(i),
             "mom20": (mom20(i) if mom20 is not None else float(i))}
        if fwd5 is not None:
            r["forward_return_5d"] = fwd5(i)
        if fwd10 is not None:
            r["forward_return_10d"] = fwd10(i)
        if fwd20 is not None:
            r["forward_return_20d"] = fwd20(i)
        out.append(r)
    return out


def _palindrome(n):
    """Tent sequence symmetric about the midpoint: Pearson(ramp, tent) is
    EXACTLY 0 by the antisymmetric/symmetric pairing, so a strictly
    increasing probe scores Spearman exactly 0.0 against it."""
    m = n // 2
    return lambda i: float(i if i < m else (n - 1 - i))


class TestHorizonAuditReport:
    def test_insufficient_data_below_min_pairs(self):
        rep = horizon_audit_report(_rows(MIN_PAIRS - 1,
                                         fwd5=lambda i: float(i),
                                         fwd10=lambda i: float(i),
                                         fwd20=lambda i: float(i)))
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["status"] == "ok"

    def test_insufficient_long_horizon_legacy_rows(self):
        """Legacy file shape: only forward_return_5d present. 5d is
        well-sampled and perfectly predictable but no longer horizon exists
        yet — the honest pre-accumulation state."""
        n = 40
        rep = horizon_audit_report(_rows(n, fwd5=lambda i: float(i),
                                         fwd10=None, fwd20=None))
        assert rep["verdict"] == "INSUFFICIENT_LONG_HORIZON"
        assert rep["best_5d_ic"] == 1.0  # strictly monotone → exact Spearman 1
        # every long-horizon cell is unsampled
        long_cells = [c for c in rep["cells"] if c["horizon"] in (10, 20)]
        assert all(c["n"] == 0 and c["rank_ic"] is None for c in long_cells)

    def test_no_horizon_has_edge_exact_zero_ic(self):
        """Strictly increasing probe vs a symmetric palindrome target →
        Spearman EXACTLY 0.0 at every horizon → NO_HORIZON_HAS_EDGE."""
        n = 40
        pal = _palindrome(n)
        rep = horizon_audit_report(_rows(n, fwd5=pal, fwd10=pal, fwd20=pal,
                                         mom20=lambda i: float(i)))
        assert rep["verdict"] == "NO_HORIZON_HAS_EDGE"
        for c in rep["cells"]:
            assert c["rank_ic"] == 0.0          # exact, not approx
            assert abs(c["rank_ic"]) < EDGE_FLOOR
            assert c["n"] == n

    def test_longer_horizon_more_predictable(self):
        """5d/10d targets are rank-noise (palindrome → IC 0.0) but the 20d
        target is strictly monotone in ml_score (IC 1.0)."""
        n = 40
        pal = _palindrome(n)
        rep = horizon_audit_report(_rows(
            n, fwd5=pal, fwd10=pal, fwd20=lambda i: float(i),
            mom20=lambda i: 7.0))  # mom20 constant → degenerate, ignored
        assert rep["verdict"] == "LONGER_HORIZON_MORE_PREDICTABLE"
        assert rep["best_5d_ic"] == 0.0
        assert rep["best_long_ic"] == 1.0
        assert rep["best_long_horizon"] == 20

    def test_5d_adequate_when_all_horizons_equal(self):
        """ml_score perfectly predicts every horizon — the 5d target is NOT
        the bottleneck, so betting on a longer one buys nothing."""
        n = 40
        rep = horizon_audit_report(_rows(
            n, fwd5=lambda i: float(i), fwd10=lambda i: float(i),
            fwd20=lambda i: float(i)))
        assert rep["verdict"] == "5D_ADEQUATE"
        assert rep["best_5d_ic"] == 1.0
        assert rep["best_long_ic"] == 1.0

    def test_sell_sign_flip_makes_correct_bearish_call_skilled(self):
        """A SELL with a more-negative ml_score preceding a bigger drop is a
        GOOD call. Without the codebase-universal flip applied to BOTH probe
        and target it would read as perfectly anti-correlated; with it, the
        cell IC is +1.0 (skilled), not -1.0."""
        n = 40
        rep = horizon_audit_report(_rows(
            n, fwd5=lambda i: -float(i + 1), fwd10=None, fwd20=None,
            ml=lambda i: -float(i + 1), mom20=lambda i: 5.0, action="SELL"))
        ml5 = next(c for c in rep["cells"]
                   if c["probe"] == "ml_score" and c["horizon"] == 5)
        assert ml5["rank_ic"] == 1.0
        assert ml5["n"] == n

    def test_never_raises_on_garbage(self):
        assert horizon_audit_report([])["verdict"] == "INSUFFICIENT_DATA"
        # missing keys / wrong types must not crash, just under-sample
        rep = horizon_audit_report([{"action": "BUY"},
                                    {"forward_return_5d": "nan"},
                                    {"ml_score": None,
                                     "forward_return_5d": float("inf")}])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["status"] == "ok"


class TestHorizonAuditAnalyze:
    def test_missing_file(self, tmp_path):
        rep = ha.analyze(tmp_path / "nope.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "no outcomes file" in rep["hint"]

    def test_analyze_reads_oos_slice(self, tmp_path):
        """analyze() must take the temporal-OOS tail (the same split the
        loop logs oos_rmse on) — slice must read 'oos' and the verdict must
        be computable from the held-out rows alone."""
        p = tmp_path / "decision_outcomes.jsonl"
        n = 200  # oos_fraction 0.2 → 40 held-out rows ≥ MIN_PAIRS
        # _rows sets a constant sim_date; split_outcomes_temporal's sort is
        # stable, so the OOS slice is the last 20% in row order — still a
        # strictly-monotone ml_score↔return relationship (IC 1.0).
        recs = _rows(n, fwd5=lambda i: float(i), fwd10=lambda i: float(i),
                     fwd20=lambda i: float(i))
        p.write_text("\n".join(json.dumps(r) for r in recs))
        rep = ha.analyze(p, oos_only=True)
        assert rep["slice"] == "oos"
        assert rep["n_records_total"] == n
        assert rep["n_records"] == max(1, int(n * 0.2))
        assert rep["verdict"] == "5D_ADEQUATE"


def test_module_constants_are_sane():
    # mirrors the calibration/gate_audit constant-echo guard
    assert MIN_PAIRS == 30
    assert IC_MARGIN == 0.05
    assert EDGE_FLOOR == 0.10
    assert ha.HORIZONS == (5, 10, 20)
    assert set(ha.PROBES) == {"ml_score", "mom20"}
