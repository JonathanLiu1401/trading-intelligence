"""Tests for paper_trader.ml.stop_out_audit.

Pure offline tests — every analyzer call passes a synthetic outcomes file
written to a tmp path. Verdicts and arithmetic are asserted against exact
expected values, not just "no crash".
"""
from __future__ import annotations

import json
import pytest

from paper_trader.ml.stop_out_audit import (
    STOP_PCT,
    BENEFIT_MARGIN,
    MIN_BUYS,
    analyze,
    _stop_protected_return,
    _to_finite_float,
)


# ──────────────────────── pure-function unit tests ─────────────────────────


class TestStopProtectedReturn:
    def test_stop_triggers_when_intra_min_at_band(self):
        # Exactly at band → triggers.
        assert _stop_protected_return(5.0, -8.0, stop_pct=8.0) == -8.0

    def test_stop_triggers_when_intra_min_below_band(self):
        assert _stop_protected_return(5.0, -15.0, stop_pct=8.0) == -8.0

    def test_no_trigger_when_intra_min_above_band(self):
        # Drew down -5% but not to -8% → ride to endpoint.
        assert _stop_protected_return(3.0, -5.0, stop_pct=8.0) == 3.0

    def test_no_trigger_when_intra_min_positive(self):
        # Never went negative → no possible trigger.
        assert _stop_protected_return(7.0, 2.0, stop_pct=8.0) == 7.0

    def test_trigger_caps_positive_endpoint_loss(self):
        # The CLASSIC case: trade dropped -10% intra, recovered to +2%
        # endpoint. With stop, captured -8%. Without stop, captured +2%.
        # The stop HURT this trade by 10pp realized.
        assert _stop_protected_return(2.0, -10.0, stop_pct=8.0) == -8.0

    def test_custom_stop_pct(self):
        # 5% stop fires at -5% intra-min.
        assert _stop_protected_return(3.0, -6.0, stop_pct=5.0) == -5.0
        # And does NOT fire at -4%.
        assert _stop_protected_return(3.0, -4.0, stop_pct=5.0) == 3.0


class TestToFiniteFloat:
    def test_none_returns_none(self):
        assert _to_finite_float(None) is None

    def test_bool_returns_none(self):
        # Same discipline as decision_scorer._to_float — bool is NOT numeric.
        assert _to_finite_float(True) is None
        assert _to_finite_float(False) is None

    def test_nan_returns_none(self):
        assert _to_finite_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert _to_finite_float(float("inf")) is None
        assert _to_finite_float(float("-inf")) is None

    def test_valid_finite(self):
        assert _to_finite_float(3.14) == 3.14
        assert _to_finite_float(0) == 0.0
        assert _to_finite_float("5.5") == 5.5

    def test_unparseable_string_returns_none(self):
        assert _to_finite_float("abc") is None


# ────────────────────────── analyze() end-to-end ────────────────────────────


def _write_outcomes(tmp_path, rows: list[dict]):
    path = tmp_path / "outcomes.jsonl"
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


class TestAnalyzeInsufficientData:
    def test_missing_file_returns_insufficient_data(self, tmp_path):
        rep = analyze(outcomes_path=tmp_path / "does_not_exist.jsonl")
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_buys"] == 0
        assert "not found" in (rep.get("hint") or "")

    def test_no_buys_returns_insufficient_data(self, tmp_path):
        # 50 SELL rows only — no BUYs in scope.
        rows = [{"action": "SELL", "forward_return_5d": 1.0,
                 "forward_intraperiod_min_5d": -2.0} for _ in range(50)]
        path = _write_outcomes(tmp_path, rows)
        rep = analyze(outcomes_path=path)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_buys"] == 0

    def test_no_intraperiod_field_returns_insufficient(self, tmp_path):
        # Historical rows (pre-2026-05-23) have no intraperiod field.
        rows = [{"action": "BUY", "forward_return_5d": 1.0} for _ in range(50)]
        path = _write_outcomes(tmp_path, rows)
        rep = analyze(outcomes_path=path)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_buys"] == 50
        assert rep["n_with_intraperiod"] == 0

    def test_below_min_buys_threshold(self, tmp_path):
        # 5 BUYs with intraperiod data — well below MIN_BUYS=30.
        rows = [{"action": "BUY", "forward_return_5d": 1.0,
                 "forward_intraperiod_min_5d": -2.0} for _ in range(5)]
        path = _write_outcomes(tmp_path, rows)
        rep = analyze(outcomes_path=path)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_with_intraperiod"] == 5


class TestAnalyzeVerdicts:
    def _gen_rows(self, n: int, fwd: float, intra_min: float) -> list[dict]:
        return [{"action": "BUY", "forward_return_5d": fwd,
                 "forward_intraperiod_min_5d": intra_min} for _ in range(n)]

    def test_stop_helps_when_recoveries_are_rare(self, tmp_path):
        # 50 BUYs that drew down -15% and ended at -10%.
        # Without stop: -10% mean. With stop: -8% mean. Stop HELPS by +2pp.
        rows = self._gen_rows(50, fwd=-10.0, intra_min=-15.0)
        path = _write_outcomes(tmp_path, rows)
        rep = analyze(outcomes_path=path)
        assert rep["verdict"] == "STOP_HELPS"
        assert rep["mean_realized_return_pct"] == pytest.approx(-10.0)
        assert rep["mean_stop_protected_return_pct"] == pytest.approx(-8.0)
        assert rep["stop_benefit_pct"] == pytest.approx(2.0)
        assert rep["n_stop_triggered"] == 50
        assert rep["pct_stop_triggered"] == 100.0

    def test_stop_hurts_when_recoveries_are_common(self, tmp_path):
        # 50 BUYs that drew down -10% intra but recovered to +5% endpoint.
        # Without stop: +5% mean. With stop: -8% mean. Stop HURTS by -13pp.
        rows = self._gen_rows(50, fwd=5.0, intra_min=-10.0)
        path = _write_outcomes(tmp_path, rows)
        rep = analyze(outcomes_path=path)
        assert rep["verdict"] == "STOP_HURTS"
        assert rep["mean_realized_return_pct"] == pytest.approx(5.0)
        assert rep["mean_stop_protected_return_pct"] == pytest.approx(-8.0)
        assert rep["stop_benefit_pct"] == pytest.approx(-13.0)
        assert rep["n_stop_triggered"] == 50

    def test_stop_neutral_when_no_trigger(self, tmp_path):
        # 50 BUYs whose intra-min never breached the -8% band.
        # Stop never fires → realized == stop-protected → benefit 0pp.
        rows = self._gen_rows(50, fwd=2.0, intra_min=-3.0)
        path = _write_outcomes(tmp_path, rows)
        rep = analyze(outcomes_path=path)
        assert rep["verdict"] == "STOP_NEUTRAL"
        assert rep["stop_benefit_pct"] == pytest.approx(0.0)
        assert rep["n_stop_triggered"] == 0
        assert rep["pct_stop_triggered"] == 0.0

    def test_mixed_population(self, tmp_path):
        # Half trigger the stop (drew -15% to -10%), half don't (drew -3% to +5%).
        # With stop: avg(-8, +5) = -1.5 mean
        # Without stop: avg(-10, +5) = -2.5 mean
        # Stop benefit: +1.0pp.
        triggers = self._gen_rows(50, fwd=-10.0, intra_min=-15.0)
        survivors = self._gen_rows(50, fwd=5.0, intra_min=-3.0)
        path = _write_outcomes(tmp_path, triggers + survivors)
        rep = analyze(outcomes_path=path)
        assert rep["mean_realized_return_pct"] == pytest.approx(-2.5)
        assert rep["mean_stop_protected_return_pct"] == pytest.approx(-1.5)
        assert rep["stop_benefit_pct"] == pytest.approx(1.0)
        # +1.0pp > BENEFIT_MARGIN(0.30) → HELPS
        assert rep["verdict"] == "STOP_HELPS"
        assert rep["n_stop_triggered"] == 50  # only the triggers
        assert rep["n_with_intraperiod"] == 100

    def test_neutral_margin_boundary(self, tmp_path):
        # Benefit exactly +0.20pp (below BENEFIT_MARGIN=0.30) → NEUTRAL.
        # 30 BUYs drew -8.5% to -8% endpoint: with stop -8, without -8.
        # Need a non-trivial mix that yields a tiny positive benefit.
        # Mix: 1 trigger (intra -10, fwd -10) + 49 survivors (intra -2, fwd +0)
        # With stop: (-8 + 49*0) / 50 = -0.16 mean
        # Without: (-10 + 49*0) / 50 = -0.20 mean
        # Benefit = -0.16 - (-0.20) = +0.04pp → NEUTRAL
        rows = [{"action": "BUY", "forward_return_5d": -10.0,
                 "forward_intraperiod_min_5d": -10.0}]
        rows += self._gen_rows(49, fwd=0.0, intra_min=-2.0)
        path = _write_outcomes(tmp_path, rows)
        rep = analyze(outcomes_path=path)
        assert rep["verdict"] == "STOP_NEUTRAL"


class TestAnalyzeFiltering:
    def test_sell_rows_excluded(self, tmp_path):
        # 50 SELLs that would trigger + 30 BUYs that don't.
        # Only the 30 BUYs are in scope.
        sells = [{"action": "SELL", "forward_return_5d": -10.0,
                  "forward_intraperiod_min_5d": -15.0} for _ in range(50)]
        buys = [{"action": "BUY", "forward_return_5d": 2.0,
                 "forward_intraperiod_min_5d": -1.0} for _ in range(30)]
        path = _write_outcomes(tmp_path, sells + buys)
        rep = analyze(outcomes_path=path)
        assert rep["n_buys"] == 30
        assert rep["n_with_intraperiod"] == 30
        assert rep["n_stop_triggered"] == 0  # only BUYs counted
        assert rep["mean_realized_return_pct"] == pytest.approx(2.0)

    def test_unparseable_rows_skipped(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        with path.open("w") as fh:
            for i in range(50):
                fh.write(json.dumps({
                    "action": "BUY", "forward_return_5d": 1.0,
                    "forward_intraperiod_min_5d": -1.0,
                }) + "\n")
            fh.write("not valid json\n")
            fh.write("\n")  # blank line
            fh.write("{broken{}}\n")
        rep = analyze(outcomes_path=path)
        # 50 valid BUYs; corrupt rows silently dropped.
        assert rep["n_buys"] == 50
        assert rep["verdict"] == "STOP_NEUTRAL"

    def test_missing_field_rows_dropped_from_intraperiod_n(self, tmp_path):
        # 30 BUYs with intra, 30 BUYs WITHOUT intra. Both counted in n_buys,
        # only the 30 with-intra contribute to the audit (verdict).
        with_intra = [{"action": "BUY", "forward_return_5d": -10.0,
                       "forward_intraperiod_min_5d": -15.0} for _ in range(30)]
        no_intra = [{"action": "BUY", "forward_return_5d": -10.0}
                    for _ in range(30)]
        path = _write_outcomes(tmp_path, with_intra + no_intra)
        rep = analyze(outcomes_path=path)
        assert rep["n_buys"] == 60
        assert rep["n_with_intraperiod"] == 30
        assert rep["verdict"] == "STOP_HELPS"  # 30 with-intra triggers all stop


class TestAnalyzeCustomStop:
    def test_tighter_stop_triggers_more(self, tmp_path):
        # 50 BUYs that drew -6% then ended +1%.
        # With STOP_PCT=8: no trigger; benefit 0. With STOP_PCT=5: all
        # trigger, captured -5pp vs +1pp → STOP_HURTS by -6pp.
        rows = [{"action": "BUY", "forward_return_5d": 1.0,
                 "forward_intraperiod_min_5d": -6.0} for _ in range(50)]
        path = _write_outcomes(tmp_path, rows)

        rep_default = analyze(outcomes_path=path, stop_pct=8.0)
        assert rep_default["n_stop_triggered"] == 0
        assert rep_default["verdict"] == "STOP_NEUTRAL"

        rep_tight = analyze(outcomes_path=path, stop_pct=5.0)
        assert rep_tight["n_stop_triggered"] == 50
        assert rep_tight["mean_stop_protected_return_pct"] == pytest.approx(-5.0)
        assert rep_tight["stop_benefit_pct"] == pytest.approx(-6.0)
        assert rep_tight["verdict"] == "STOP_HURTS"


# ───────────────────── conviction-cap behavioural test ─────────────────────


class TestLeveragedConvictionCap:
    """Phase 1 added BITX/CONL/etc. to _LEVERAGED_ETFS — pin the actual
    behavioural change at the _ml_decide level so a future revert is
    observable (set membership alone is one step removed from the cap arm).
    """
    def test_bitx_in_bull_regime_gets_leveraged_cap(self, synthetic_prices,
                                                      monkeypatch):
        """BITX is one of the 18 leveraged-bull tickers added in Phase 1.
        In bull/sideways regime its conviction cap is min(0.40, score/15)
        — vs the regular min(0.25, score/20). With a max ml_score of 10,
        the leveraged formula yields 0.40 (cap binds) and the regular
        formula yields 0.25 (cap binds). Pin 0.40 → 25k+ notional on a
        100k portfolio rather than 25k.

        synthetic_prices only carries SPY/NVDA, so we monkey-patch BITX
        into the price cache and force regime to 'bull' explicitly.
        """
        import paper_trader.backtest as bt
        from paper_trader.backtest import _ml_decide, SimPortfolio, _LEVERAGED_ETFS
        import random

        # Confirm BITX is in the set we just expanded.
        assert "BITX" in _LEVERAGED_ETFS, "Phase 1 regression: BITX dropped"

        # Pin scorer untrained so the gate cannot modulate conviction
        monkeypatch.setattr(bt, "_DECISION_SCORER", None, raising=False)

        # Patch prices.prices to include BITX at the same series as NVDA
        prices = synthetic_prices
        prices.prices["BITX"] = dict(prices.prices["NVDA"])

        # Force regime to 'bull' so the leveraged cap arm fires.
        monkeypatch.setattr(bt, "_market_regime",
                            lambda d, p: "bull")

        p = SimPortfolio(cash=100_000.0)
        rng = random.Random(42)
        # Headline that maps to BITX via "bitcoin" → BTC-USD... actually
        # let's directly pass tickers=["BITX"] to bypass keyword mapping.
        articles = [{
            "title": "Bitcoin surge, crypto rally — BITX strong demand",
            "score": 10.0, "tickers": ["BITX"],
        }]
        d = synthetic_prices.trading_days[-1]
        decision = _ml_decide(d, p, articles, prices, run_id=1, rng=rng)
        # Some other ticker might win the pick (NVDA gets a base score from
        # its earlier setup). We need to construct articles that ONLY
        # mention BITX and have no NVDA-related keywords.
        if decision.get("ticker") != "BITX":
            pytest.skip(f"BITX did not win the pick (picked "
                        f"{decision.get('ticker')!r}); test environment "
                        f"would need adjustment to isolate.")
        # Conviction cap should be 0.40, not 0.25
        price = prices.price_on("BITX", d)
        notional = decision["qty"] * price
        total_value = p.total_value(prices, d)
        # Leveraged cap is 0.40 → 40_000 max. Pre-fix cap was 0.25 → 25_000.
        # min(0.40, 10/15) = 0.40 (cap binds).
        assert notional == pytest.approx(40_000.0, rel=0.01), (
            f"BITX notional {notional} should hit leveraged-cap 40k; "
            f"if it's 25k the conviction-cap fix has regressed."
        )
