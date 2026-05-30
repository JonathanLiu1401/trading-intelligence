"""Tests for paper_trader.ml.sizing_rule_counterfactual.

The analyzer compares the gate's actual sizing rule against alternative
rules on the same outcome corpus. Tests assert the EXACT per-rule
totals against hand-calculated expected values so a regression in any
rule's math fails sharply (not just "rebalanced the verdict").
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_trader.ml import sizing_rule_counterfactual as src


# ─────────────────────────── helpers ──────────────────────────────


def _buy(conviction, fwd_ret, ml_score=0.0, news_urgency=None):
    """Build a minimal BUY outcome row."""
    return {
        "action": "BUY",
        "conviction_pct": conviction,
        "forward_return_5d": fwd_ret,
        "ml_score": ml_score,
        "news_urgency": news_urgency,
    }


# ───────────────── individual sizing-rule unit tests ──────────────


class TestRuleFunctions:
    """Each rule is a pure function; tests pin the exact output."""

    def test_actual_uses_conviction_pct(self):
        assert src._rule_actual({"conviction_pct": 0.25}) == 0.25
        assert src._rule_actual({"conviction_pct": 0.10}) == 0.10

    def test_actual_returns_none_on_missing(self):
        assert src._rule_actual({}) is None
        assert src._rule_actual({"conviction_pct": None}) is None

    def test_actual_returns_none_out_of_range(self):
        # The upstream parser clamps to [0,1]; out-of-range = corrupt.
        assert src._rule_actual({"conviction_pct": -0.5}) is None
        assert src._rule_actual({"conviction_pct": 1.5}) is None

    def test_actual_rejects_bool_input(self):
        # bool is int subclass; _to_finite_float must reject it. Without
        # this guard, `conviction_pct=True` would compute as 1.0.
        assert src._rule_actual({"conviction_pct": True}) is None
        assert src._rule_actual({"conviction_pct": False}) is None

    def test_uniform_rules_are_constant(self):
        # UNIFORM_10 and UNIFORM_25 ignore the record entirely.
        assert src._rule_uniform_10({}) == 0.10
        assert src._rule_uniform_25({}) == 0.25
        assert src._rule_uniform_10({"ml_score": 999.0}) == 0.10
        assert src._rule_uniform_25({"ml_score": 999.0}) == 0.25

    def test_score_based_matches_live_formula(self):
        # min(0.25, max(0, ml_score / 20))
        assert src._rule_score_based({"ml_score": 0.0}) == 0.0
        assert src._rule_score_based({"ml_score": 2.0}) == pytest.approx(0.10)
        assert src._rule_score_based({"ml_score": 5.0}) == 0.25  # capped
        assert src._rule_score_based({"ml_score": -1.0}) == 0.0  # floor at 0
        assert src._rule_score_based({"ml_score": 999.0}) == 0.25  # cap

    def test_score_based_handles_missing(self):
        # Missing ml_score defaults to 0 → 0.0 conviction.
        assert src._rule_score_based({}) == 0.0

    def test_inverse_score_inverts_conviction(self):
        # min(0.25, max(0.05, 0.25 - ml_score/100))
        # ml_score=0  → 0.25 (cap)
        # ml_score=10 → 0.15
        # ml_score=20 → 0.05 (floor)
        # ml_score=25 → 0.05 (floor — would be 0.0 without floor)
        # ml_score=50 → 0.05
        assert src._rule_inverse_score({"ml_score": 0.0}) == 0.25
        assert src._rule_inverse_score({"ml_score": 10.0}) == pytest.approx(0.15)
        assert src._rule_inverse_score({"ml_score": 20.0}) == 0.05
        assert src._rule_inverse_score({"ml_score": 50.0}) == 0.05
        # Negative score → 0.25 - (-1/100) = 0.26 → cap 0.25
        assert src._rule_inverse_score({"ml_score": -1.0}) == 0.25

    def test_news_driven_uses_urgency_when_positive(self):
        # urgency=50 → 0.50/1.0 = 0.50 capped at 0.25
        assert src._rule_news_driven({"news_urgency": 50.0}) == 0.25
        assert src._rule_news_driven({"news_urgency": 10.0}) == pytest.approx(0.10)
        assert src._rule_news_driven({"news_urgency": 25.0}) == 0.25

    def test_news_driven_fallback_when_missing(self):
        assert src._rule_news_driven({}) == 0.10
        assert src._rule_news_driven({"news_urgency": None}) == 0.10
        assert src._rule_news_driven({"news_urgency": 0}) == 0.10
        assert src._rule_news_driven({"news_urgency": -5}) == 0.10


# ─────────────── analyzer integration tests ──────────────────────


class TestBuildSizingCounterfactual:
    def test_insufficient_data_below_min_rows(self):
        # 10 rows < MIN_ROWS (60)
        recs = [_buy(0.10, 1.0) for _ in range(10)]
        rep = src.build_sizing_counterfactual(recs)
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 10

    def test_empty_input(self):
        rep = src.build_sizing_counterfactual([])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_sells_are_filtered(self):
        # Mix BUYs (60 needed) and SELLs (should be excluded).
        recs = [_buy(0.10, 1.0) for _ in range(70)]
        for _ in range(50):
            recs.append({
                "action": "SELL", "conviction_pct": None,
                "forward_return_5d": 5.0,
            })
        rep = src.build_sizing_counterfactual(recs)
        assert rep["n"] == 70  # only BUYs
        assert rep["n_dropped_action"] == 50

    def test_actual_total_pp_arithmetic(self):
        # 100 BUYs, each conviction=0.10, fwd_ret=2.0
        # Σ actual = 100 × 0.10 × 2.0 = 20.0 pp
        recs = [_buy(0.10, 2.0) for _ in range(100)]
        rep = src.build_sizing_counterfactual(recs)
        actual = next(r for r in rep["rules"] if r["name"] == "ACTUAL")
        assert actual["total_pp"] == pytest.approx(20.0, abs=1e-3)
        # per-trade = 20.0 / 100 = 0.20
        assert actual["per_trade_pp"] == pytest.approx(0.20, abs=1e-4)
        # mean_conviction = 0.10
        assert actual["mean_conviction"] == pytest.approx(0.10, abs=1e-4)

    def test_uniform_25_total_pp_arithmetic(self):
        # 100 BUYs, conviction=0.10, fwd_ret=2.0
        # UNIFORM_25 = 100 × 0.25 × 2.0 = 50.0 pp
        recs = [_buy(0.10, 2.0) for _ in range(100)]
        rep = src.build_sizing_counterfactual(recs)
        uniform_25 = next(r for r in rep["rules"] if r["name"] == "UNIFORM_25")
        assert uniform_25["total_pp"] == pytest.approx(50.0, abs=1e-3)
        assert uniform_25["mean_conviction"] == pytest.approx(0.25, abs=1e-4)

    def test_uniform_25_wins_when_average_return_positive(self):
        # When mean return is positive and ACTUAL sizes smaller than
        # UNIFORM_25, UNIFORM_25 must win the counterfactual.
        recs = [_buy(0.10, 2.0) for _ in range(100)]
        rep = src.build_sizing_counterfactual(recs)
        assert rep["verdict"] == "ALT_BEATS_ACTUAL"
        assert rep["best_alt_rule"] == "UNIFORM_25"
        # Relative improvement: (50 - 20) / 20 = 1.5 = +150%
        assert rep["rel_improvement"] == pytest.approx(1.5, abs=1e-3)

    def test_actual_best_when_others_lose(self):
        # Construct a corpus where ACTUAL's variable conviction
        # correctly identifies winners (high conv on winners, low conv
        # on losers) while every score-based / score-free alternative
        # gets it wrong. ml_score is held at 0 across all rows so
        # SCORE_BASED collapses to zero conviction (≡ no signal) and
        # INVERSE_SCORE collapses to the cap (≡ UNIFORM_25), making
        # the comparison clean: only ACTUAL has the conviction-return
        # alignment.
        recs = []
        # 35 high-conv winners (conviction=0.25, ret=+5%)
        for _ in range(35):
            recs.append(_buy(0.25, 5.0, ml_score=0.0))
        # 30 low-conv losers (conviction=0.05, ret=-5%)
        for _ in range(30):
            recs.append(_buy(0.05, -5.0, ml_score=0.0))
        rep = src.build_sizing_counterfactual(recs)
        # ACTUAL: 35×0.25×5 + 30×0.05×(-5) = 43.75 - 7.5 = +36.25
        # UNIFORM_25: 0.25 × (35×5 + 30×(-5)) = 0.25 × 25 = +6.25
        # SCORE_BASED (ml_score=0 → 0.0 conv): total = 0.0
        # INVERSE_SCORE (ml_score=0 → 0.25): same as UNIFORM_25 = +6.25
        # ACTUAL wins by >>20%.
        actual = next(r for r in rep["rules"] if r["name"] == "ACTUAL")
        uniform_25 = next(r for r in rep["rules"] if r["name"] == "UNIFORM_25")
        score_based = next(r for r in rep["rules"] if r["name"] == "SCORE_BASED")
        assert actual["total_pp"] == pytest.approx(36.25, abs=1e-2)
        assert uniform_25["total_pp"] == pytest.approx(6.25, abs=1e-2)
        assert score_based["total_pp"] == pytest.approx(0.0, abs=1e-2)
        # Verdict should be ACTUAL_BEST (actual >> best alt by >20%)
        assert rep["verdict"] == "ACTUAL_BEST"

    def test_tie_when_within_threshold(self):
        # 100 BUYs, conviction=0.20, fwd_ret=1.0 → ACTUAL = 20.0
        # UNIFORM_25 = 100 × 0.25 × 1.0 = 25.0 → +25% over actual.
        # That's just at the 20% threshold — should be ALT_BEATS.
        # Try conviction=0.22: ACTUAL = 22.0, UNIFORM_25 = 25.0,
        # rel improvement = (25-22)/22 = 0.1364 = 13.6% → TIE.
        recs = [_buy(0.22, 1.0) for _ in range(100)]
        rep = src.build_sizing_counterfactual(recs)
        assert rep["verdict"] == "TIE"

    def test_alt_beats_actual_at_threshold_exact(self):
        # ACTUAL: 100 × 0.20 × 1.0 = 20.0
        # UNIFORM_25: 100 × 0.25 × 1.0 = 25.0
        # rel = (25 - 20) / 20 = 0.25 = 25% > 20% → ALT_BEATS_ACTUAL
        recs = [_buy(0.20, 1.0) for _ in range(100)]
        rep = src.build_sizing_counterfactual(recs)
        assert rep["verdict"] == "ALT_BEATS_ACTUAL"
        assert rep["best_alt_rule"] == "UNIFORM_25"
        assert rep["rel_improvement"] == pytest.approx(0.25, abs=1e-3)

    def test_no_raise_on_corrupt_record(self):
        # A record with NaN/inf return must be dropped, not blow up.
        recs = [_buy(0.10, 1.0) for _ in range(60)]
        recs.append({"action": "BUY", "conviction_pct": 0.10,
                     "forward_return_5d": float("nan")})
        recs.append({"action": "BUY", "conviction_pct": 0.10,
                     "forward_return_5d": float("inf")})
        rep = src.build_sizing_counterfactual(recs)
        assert rep["status"] == "ok"
        assert rep["n"] == 60  # Two nan/inf records dropped
        assert rep["n_dropped_return"] == 2

    def test_news_driven_with_real_urgency(self):
        # 60 rows: half have news_urgency=20 (→ 0.20), half have None
        # (→ 0.10 fallback). All have fwd_ret=2.0, conviction=0.10.
        recs = []
        for _ in range(30):
            recs.append(_buy(0.10, 2.0, news_urgency=20.0))
        for _ in range(30):
            recs.append(_buy(0.10, 2.0, news_urgency=None))
        rep = src.build_sizing_counterfactual(recs)
        news = next(r for r in rep["rules"] if r["name"] == "NEWS_DRIVEN")
        # 30×0.20×2 + 30×0.10×2 = 12 + 6 = 18.0
        assert news["total_pp"] == pytest.approx(18.0, abs=1e-3)
        # mean_conviction = (30×0.20 + 30×0.10) / 60 = 9/60 = 0.15
        assert news["mean_conviction"] == pytest.approx(0.15, abs=1e-4)

    def test_inverse_score_with_varying_score(self):
        # 60 rows, ml_score=10 each, conviction=0.10, fwd_ret=1.0
        # ACTUAL = 60 × 0.10 × 1.0 = 6.0
        # INVERSE_SCORE for ml_score=10: min(0.25, max(0.05, 0.25-0.1)) = 0.15
        # INVERSE_SCORE total = 60 × 0.15 × 1.0 = 9.0
        recs = [_buy(0.10, 1.0, ml_score=10.0) for _ in range(60)]
        rep = src.build_sizing_counterfactual(recs)
        inv = next(r for r in rep["rules"] if r["name"] == "INVERSE_SCORE")
        assert inv["total_pp"] == pytest.approx(9.0, abs=1e-3)
        assert inv["mean_conviction"] == pytest.approx(0.15, abs=1e-4)

    def test_corrupt_input_returns_error_envelope(self):
        # build_sizing_counterfactual must never raise. A pathological
        # iterable still degrades to a status envelope.
        rep = src.build_sizing_counterfactual([
            "not a dict",
            123,
            None,
            [1, 2, 3],
        ])
        # Treated as invalid records, all dropped.
        assert rep["status"] == "insufficient_data"
        # All 4 non-dict objects counted as dropped_action.
        assert rep["n_dropped_action"] == 4
        assert rep["n"] == 0


# ─────────────── CLI tests ────────────────────────────────────────


class TestCLI:
    def test_cli_exit_0_on_actual_best(self, tmp_path, capsys):
        # Construct a corpus where ACTUAL is decisively the winner.
        # ml_score=0 across all rows so SCORE_BASED/INVERSE_SCORE
        # don't accidentally beat ACTUAL.
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as f:
            for _ in range(35):
                f.write(json.dumps(_buy(0.25, 5.0, ml_score=0.0)) + "\n")
            for _ in range(30):
                f.write(json.dumps(_buy(0.05, -5.0, ml_score=0.0)) + "\n")
        rc = src._cli(["--outcomes", str(p)])
        assert rc == 0

    def test_cli_exit_2_on_alt_beats_actual(self, tmp_path, capsys):
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as f:
            for _ in range(70):
                f.write(json.dumps(_buy(0.10, 2.0)) + "\n")
        rc = src._cli(["--outcomes", str(p)])
        # ACTUAL=14, UNIFORM_25=35 → ALT_BEATS → exit 2.
        assert rc == 2

    def test_cli_json_output_shape(self, tmp_path, capsys):
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as f:
            for _ in range(70):
                f.write(json.dumps(_buy(0.10, 1.0)) + "\n")
        src._cli(["--outcomes", str(p), "--json"])
        captured = capsys.readouterr().out
        rep = json.loads(captured)
        assert "verdict" in rep
        assert "rules" in rep
        assert "actual_total_pp" in rep
        rule_names = {r["name"] for r in rep["rules"]}
        assert {"ACTUAL", "UNIFORM_10", "UNIFORM_25",
                "SCORE_BASED", "INVERSE_SCORE", "NEWS_DRIVEN"} == rule_names

    def test_cli_missing_outcomes_file(self, tmp_path):
        # A missing file degrades to INSUFFICIENT_DATA (load_outcomes
        # returns []), not a crash.
        rc = src._cli(["--outcomes", str(tmp_path / "does_not_exist.jsonl")])
        # INSUFFICIENT_DATA → exit 0 (informational)
        assert rc == 0
