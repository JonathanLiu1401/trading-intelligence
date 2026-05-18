"""Exact-value locks for the conviction-gate economic counterfactual
(`paper_trader/ml/gate_pnl.py`, 2026-05-18 ML+backtest hybrid feature).

Mirrors test_gate_audit.py / test_calibration.py: deterministic synthetic
data, exact metrics and exact verdicts (not ranges) so a logic change must
update the literals deliberately. All offline, no network, no trained MLP.

Load-bearing assertions:
  * the equal-weight (assumption-free) contribution `Σmᵢrᵢ/Σmᵢ − mean(rᵢ)`
    is the verdict driver, with exact pp values hand-computed below.
  * the SELL `-forward_return_5d` sign-flip matches train_scorer (without
    it the GATE_ADDS fixture would read GATE_RETURN_NEUTRAL — the
    regression lock).
  * the sized (informational) contribution `Σwᵢmᵢrᵢ/Σwᵢmᵢ − Σwᵢrᵢ/Σwᵢ`
    is exact and is NOT the verdict driver.
  * `_reconstruct_base_conviction` mirrors `_ml_decide`'s
    cap/divisor/leveraged-ETF/regime branches exactly.
  * `gate_arm` is the SAME object imported from gate_audit (no arm drift
    between the two gate diagnostics).
  * `oos_only` restricts to the temporal-OOS slice.
"""
from __future__ import annotations

import math

import pytest

from paper_trader.ml import gate_pnl as gp
from paper_trader.ml import gate_audit as ga


class _FakeScorer:
    """predict() echoes ml_score so the SELL flip / kwarg names / OOS split
    are testable without a trained MLP."""

    is_trained = True
    n_train = 999

    def predict(self, **kw) -> float:
        return float(kw["ml_score"])


class _RaisingScorer:
    is_trained = True

    def predict(self, **kw):
        raise ValueError("model/feature shape mismatch")


# ─────────────────────── SSOT: gate_arm is shared ───────────────────────


class TestArmSingleSourceOfTruth:
    def test_gate_arm_is_the_same_object_as_gate_audit(self):
        # Importing (not redeclaring) is the invariant — the five arms must
        # never drift between the two gate diagnostics.
        assert gp.gate_arm is ga.gate_arm

    def test_tolerances_match_gate_audit(self):
        assert gp.MIN_TOTAL == ga.MIN_TOTAL
        assert gp.EDGE_TOL_PP == ga.EDGE_TOL_PP


# ─────────────────────── equal-weight verdict math ──────────────────────


class TestEqualWeightContribution:
    """Hand-computed: 15 rows pred=+20 (mult 1.30) and 15 rows pred=-20
    (mult 0.60). gate_off = mean(r); gate_on = Σmr/Σm."""

    def test_gate_adds_return_exact(self):
        triples = ([(20.0, 10.0, None)] * 15
                   + [(-20.0, -10.0, None)] * 15)
        rep = gp.gate_pnl_report(triples)
        # mean(r) = 0 ; Σmr = 195 − 90 = 105 ; Σm = 19.5 + 9.0 = 28.5
        # gate_on = 105 / 28.5 = 3.684210…  contribution = +3.6842
        assert rep["verdict"] == "GATE_ADDS_RETURN"
        assert rep["n"] == 30
        assert rep["gate_off_mean_pct"] == 0.0
        assert rep["gate_on_mean_pct"] == 3.6842
        assert rep["equal_weight_gate_contribution_pp"] == 3.6842
        assert rep["avg_gate_multiplier"] == 0.95  # 28.5 / 30

    def test_gate_subtracts_return_exact(self):
        # Inverted: winners get headwind (0.60), losers get tailwind (1.30).
        triples = ([(-20.0, 10.0, None)] * 15
                   + [(20.0, -10.0, None)] * 15)
        rep = gp.gate_pnl_report(triples)
        # Σmr = 90 − 195 = −105 ; Σm = 28.5 ; gate_on = −3.6842
        assert rep["verdict"] == "GATE_SUBTRACTS_RETURN"
        assert rep["gate_on_mean_pct"] == -3.6842
        assert rep["equal_weight_gate_contribution_pp"] == -3.6842

    def test_gate_return_neutral_when_realized_constant(self):
        # Realized constant ⇒ any reweighting is a no-op ⇒ contribution 0.
        triples = ([(-20.0, 5.0, None)] * 10      # mult 0.60
                   + [(3.0, 5.0, None)] * 10      # mult 1.00
                   + [(20.0, 5.0, None)] * 10)    # mult 1.30
        rep = gp.gate_pnl_report(triples)
        assert rep["verdict"] == "GATE_RETURN_NEUTRAL"
        assert rep["gate_off_mean_pct"] == 5.0
        assert rep["gate_on_mean_pct"] == 5.0
        assert rep["equal_weight_gate_contribution_pp"] == 0.0
        # avg multiplier = (6 + 10 + 13) / 30 = 29/30 = 0.9667
        assert rep["avg_gate_multiplier"] == 0.9667

    def test_insufficient_data_below_min_total(self):
        triples = [(20.0, 10.0, None)] * (gp.MIN_TOTAL - 1)
        rep = gp.gate_pnl_report(triples)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == gp.MIN_TOTAL - 1
        assert rep["equal_weight_gate_contribution_pp"] is None
        assert rep["sized_gate_contribution_pp"] is None

    def test_nonfinite_rows_dropped_not_poisoning(self):
        triples = ([(20.0, 10.0, None)] * 15
                   + [(-20.0, -10.0, None)] * 15
                   + [(float("nan"), 1.0, None), (1.0, float("inf"), None)])
        rep = gp.gate_pnl_report(triples)
        # The two non-finite rows are dropped → identical to the clean 30.
        assert rep["n"] == 30
        assert rep["equal_weight_gate_contribution_pp"] == 3.6842


# ─────────────────────── sized (informational) metric ───────────────────


class TestSizedContribution:
    def test_sized_contribution_exact_and_not_verdict_driver(self):
        # Group P: 15 rows pred=+20 (m=1.30) r=+10 base w=0.40
        # Group Q: 15 rows pred=-20 (m=0.60) r=-10 base w=0.10
        triples = ([(20.0, 10.0, 0.40)] * 15
                   + [(-20.0, -10.0, 0.10)] * 15)
        rep = gp.gate_pnl_report(triples)
        # Equal-weight (verdict driver): +3.6842 ⇒ GATE_ADDS_RETURN
        assert rep["verdict"] == "GATE_ADDS_RETURN"
        assert rep["equal_weight_gate_contribution_pp"] == 3.6842
        # off_sized = Σwr/Σw = (60 − 15)/(6.0 + 1.5) = 45/7.5 = 6.0
        # on_w = w·m → P:0.52  Q:0.06
        # on_sized = (78 − 9)/(7.8 + 0.9) = 69/8.7 = 7.931034…
        # sized contribution = 7.931034 − 6.0 = 1.9310
        assert rep["sized_n"] == 30
        assert rep["sized_gate_contribution_pp"] == 1.9310

    def test_sized_none_when_no_base_supplied(self):
        triples = [(20.0, 10.0, None)] * 30
        rep = gp.gate_pnl_report(triples)
        assert rep["sized_n"] == 0
        assert rep["sized_gate_contribution_pp"] is None
        # Verdict still produced from the assumption-free path.
        assert rep["verdict"] in ("GATE_ADDS_RETURN", "GATE_RETURN_NEUTRAL",
                                   "GATE_SUBTRACTS_RETURN")

    def test_nonpositive_base_excluded_from_sized(self):
        # base ≤ 0 / non-numeric is not a usable size → row skipped in sized.
        triples = ([(20.0, 10.0, 0.40)] * 15
                   + [(-20.0, -10.0, 0.0)] * 8
                   + [(-20.0, -10.0, "x")] * 7)
        rep = gp.gate_pnl_report(triples)
        assert rep["n"] == 30
        assert rep["sized_n"] == 15  # only the positive-base group


# ─────────────────────── scorer path + SELL sign-flip ───────────────────


def _rec(ml_score, fwd, action="BUY", sim_date="2020-01-01", ticker="AMZN"):
    return {"ml_score": ml_score, "forward_return_5d": fwd, "action": action,
            "sim_date": sim_date, "ticker": ticker, "rsi": None, "macd": None,
            "mom5": None, "mom20": None, "regime_mult": 1.0}


class TestScorerGatePnlSellFlip:
    def test_sell_sign_flip_changes_verdict(self):
        # 15 SELL rows ml_score=+20 (pred +20, mult 1.30), fwd=-10 → a
        # correct bearish call → flipped realized = +10.
        # 15 BUY rows ml_score=-20 (pred -20, mult 0.60), fwd=-10 → -10.
        # WITH flip: tailwind on the +10s, headwind on the -10s ⇒
        #            equal-weight +3.6842 ⇒ GATE_ADDS_RETURN.
        # WITHOUT flip (the bug): all realized −10 ⇒ contribution 0 ⇒
        #            GATE_RETURN_NEUTRAL.
        recs = ([_rec(20.0, -10.0, action="SELL") for _ in range(15)]
                + [_rec(-20.0, -10.0, action="BUY") for _ in range(15)])
        rep = gp.scorer_gate_pnl(_FakeScorer(), recs, oos_only=False)
        assert rep["slice"] == "all"
        assert rep["verdict"] == "GATE_ADDS_RETURN"
        assert rep["equal_weight_gate_contribution_pp"] == 3.6842

    def test_none_forward_return_row_skipped(self):
        recs = ([_rec(20.0, 10.0) for _ in range(30)]
                + [_rec(20.0, None) for _ in range(5)])
        rep = gp.scorer_gate_pnl(_FakeScorer(), recs, oos_only=False)
        assert rep["n"] == 30  # the 5 None-fwd rows dropped

    def test_raising_scorer_degrades_not_raises(self):
        recs = [_rec(20.0, 10.0) for _ in range(30)]
        rep = gp.scorer_gate_pnl(_RaisingScorer(), recs, oos_only=False)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0


# ─────────────────────── temporal-OOS restriction ───────────────────────


class TestOosRestriction:
    def _balanced_dated_recs(self, n: int):
        # Alternating +20/+10 and -20/-10 by index parity so ANY even-length
        # contiguous tail is +/- balanced (⇒ stable +3.6842 contribution).
        out = []
        base = __import__("datetime").date(2020, 1, 1)
        for i in range(n):
            d = (base + __import__("datetime").timedelta(days=i)).isoformat()
            if i % 2 == 0:
                out.append(_rec(20.0, 10.0, sim_date=d))
            else:
                out.append(_rec(-20.0, -10.0, sim_date=d))
        return out

    def test_oos_only_restricts_to_temporal_slice(self):
        recs = self._balanced_dated_recs(200)
        rep = gp.scorer_gate_pnl(_FakeScorer(), recs, oos_only=True)
        # split_outcomes_temporal holds out the most-recent int(200*0.2)=40.
        assert rep["slice"] == "oos"
        assert rep["n_records_considered"] == 40
        assert rep["n"] == 40
        assert rep["verdict"] == "GATE_ADDS_RETURN"
        assert rep["equal_weight_gate_contribution_pp"] == 3.6842

    def test_all_slice_uses_every_record(self):
        recs = self._balanced_dated_recs(200)
        rep = gp.scorer_gate_pnl(_FakeScorer(), recs, oos_only=False)
        assert rep["slice"] == "all"
        assert rep["n_records_considered"] == 200
        assert rep["n"] == 200
        assert rep["equal_weight_gate_contribution_pp"] == 3.6842


# ─────────────────────── base-conviction reconstruction ─────────────────


class TestReconstructBaseConviction:
    def test_normal_ticker_bull_uses_25_cap_div20(self):
        # min(0.25, 3.0/20) = 0.15
        assert gp._reconstruct_base_conviction(3.0, 1.0, "AMZN") == 0.15

    def test_normal_ticker_cap_binds(self):
        # min(0.25, 10.0/20=0.5) = 0.25
        assert gp._reconstruct_base_conviction(10.0, 1.0, "AMZN") == 0.25

    def test_leveraged_etf_bull_uses_40_cap_div15(self):
        # SOXL ∈ _LEVERAGED_ETFS, regime bull → min(0.40, 3.0/15) = 0.20
        assert gp._reconstruct_base_conviction(3.0, 1.0, "SOXL") == 0.20

    def test_leveraged_etf_cap_binds(self):
        # min(0.40, 9.0/15=0.6) = 0.40
        assert gp._reconstruct_base_conviction(9.0, 1.0, "SOXL") == 0.40

    def test_leveraged_etf_sideways_still_leveraged_branch(self):
        # regime_mult 0.6 → sideways → leveraged branch: min(0.40, 3/15)=0.20
        assert gp._reconstruct_base_conviction(3.0, 0.6, "SOXL") == 0.20

    def test_leveraged_etf_bear_falls_to_normal_branch(self):
        # regime_mult 0.3 → bear → NOT leveraged → min(0.25, 3/20) = 0.15
        assert gp._reconstruct_base_conviction(3.0, 0.3, "SOXL") == 0.15

    def test_unusable_ml_score_returns_none(self):
        assert gp._reconstruct_base_conviction(None, 1.0, "AMZN") is None
        assert gp._reconstruct_base_conviction(float("nan"), 1.0, "AMZN") is None
        assert gp._reconstruct_base_conviction("x", 1.0, "AMZN") is None


# ─────────────────────── never-raises / analyze ─────────────────────────


class TestRobustness:
    def test_garbage_iterable_never_raises(self):
        rep = gp.gate_pnl_report([("x", None, None), (1, 2), 7, None,
                                  (float("nan"), float("nan"), 1.0)])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["status"] == "ok"

    def test_analyze_missing_file(self):
        out = gp.analyze("/nonexistent/decision_outcomes.jsonl")
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert "no outcomes file" in out["hint"]

    def test_analyze_untrained_scorer(self, tmp_path, monkeypatch):
        f = tmp_path / "o.jsonl"
        f.write_text(__import__("json").dumps(_rec(20.0, 10.0)) + "\n")

        class _Untrained:
            is_trained = False

        monkeypatch.setattr(
            "paper_trader.ml.decision_scorer.DecisionScorer",
            lambda: _Untrained(),
        )
        out = gp.analyze(f)
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert "not trained" in out["hint"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
