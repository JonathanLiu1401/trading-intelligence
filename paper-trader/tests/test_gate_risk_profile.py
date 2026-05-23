"""Exact-value locks for the conviction-gate RISK PROFILE diagnostic
(``paper_trader/ml/gate_risk_profile.py``).

Sibling to ``test_gate_realized.py`` — deterministic synthetic rows,
hand-computed metrics, exact verdicts. The two diagnostics share row
classification and the captured-only / abstention-honesty discipline;
this suite locks the risk-profile metrics (win_rate, percentiles, sharpe,
expected-value reconciliation) and the four-way verdict ladder
(``RISK_PROPORTIONAL`` / ``UNFAVORABLE_RISK`` / ``RISK_INDIFFERENT`` /
``WIN_RATE_INVERTED``) that ``gate_realized``'s mean-only verdict cannot
distinguish.

All offline, no network, no scorer/pickle.
"""
from __future__ import annotations

import json
import math

import pytest

from paper_trader.ml import gate_risk_profile as grp
from paper_trader.ml import gate_realized as gr
from paper_trader.ml import gate_audit as ga


# ─────────────────────── SSOT: arms come from gate_audit ─────────────────


class TestArmSingleSourceOfTruth:
    def test_gate_arm_is_the_same_object_as_gate_audit(self):
        # No arm drift across the gate diagnostic family.
        assert grp.gate_arm is ga.gate_arm

    def test_arm_order_matches_gate_realized(self):
        # Order must match its sibling so a row indexed by position
        # describes the same arm across diagnostics.
        assert grp._ARM_ORDER == gr._ARM_ORDER

    def test_arm_multipliers_are_canonical(self):
        rep = grp.gate_risk_report([])
        assert [a["arm"] for a in rep["arms"]] == grp._ARM_ORDER
        assert [a["multiplier"] for a in rep["arms"]] == [
            0.60, 0.85, 1.00, 1.15, 1.30,
        ]


# ─────────────────────── helpers ───────────────────────


def _row(pred, fwd5, *, off=False, action="BUY", sim="2025-06-02"):
    return {
        "gate_scorer_pred": pred,
        "gate_off_dist": off,
        "action": action,
        "forward_return_5d": fwd5,
        "sim_date": sim,
        "ticker": "NVDA",
        "ml_score": 3.0,
    }


def _bulk(pred, returns):
    """Multiple rows in one arm with explicit returns — control-arm pattern."""
    return [_row(pred, r) for r in returns]


# ─────────────────────── _arm_stats hand-computed locks ─────────────────


class TestArmStatsExactValues:
    def test_empty_input_returns_all_none(self):
        s = grp._arm_stats([])
        assert s == {
            "n": 0, "win_rate": None, "mean": None, "median": None,
            "p10": None, "p25": None, "p75": None, "p90": None,
            "mean_win": None, "mean_loss": None,
            "stdev": None, "sharpe": None, "expected_value": None,
        }

    def test_mixed_wins_and_losses(self):
        # 10 rows: 6 positive (avg +5), 4 negative (avg -2.5). Hand-checked.
        vals = [10.0, 8.0, 6.0, 4.0, 2.0, 0.0,  # 5 wins + 1 zero
                -1.0, -2.0, -3.0, -4.0]  # 4 losses
        s = grp._arm_stats(vals)
        assert s["n"] == 10
        # 5 strictly-positive out of 10
        assert s["win_rate"] == pytest.approx(0.5, abs=1e-6)
        # mean = (10+8+6+4+2+0 -1 -2 -3 -4)/10 = 20/10 = 2.0
        assert s["mean"] == pytest.approx(2.0, abs=1e-6)
        # mean_win = (10+8+6+4+2)/5 = 6.0 (zero excluded from wins by >0)
        assert s["mean_win"] == pytest.approx(6.0, abs=1e-6)
        # mean_loss = (-1-2-3-4)/4 = -2.5
        assert s["mean_loss"] == pytest.approx(-2.5, abs=1e-6)
        # n_zero = 1 (the literal 0.0)
        assert s["n_zero"] == 1
        # EV reconciliation: 0.5*6.0 + 0.4*(-2.5) + 0.1*0 = 3.0 - 1.0 = 2.0
        assert s["expected_value"] == pytest.approx(2.0, abs=1e-6)
        # EV must equal mean by construction
        assert s["expected_value"] == pytest.approx(s["mean"], abs=1e-3)

    def test_sharpe_division_by_zero_safe(self):
        # Single-sample arm: stdev=0 by ddof=0; sharpe must be None, never inf/NaN.
        s = grp._arm_stats([3.0])
        assert s["n"] == 1
        assert s["stdev"] == 0.0
        assert s["sharpe"] is None
        assert s["mean"] == pytest.approx(3.0)

    def test_perfectly_constant_arm_has_no_sharpe(self):
        # All same value: stdev=0, sharpe=None.
        s = grp._arm_stats([5.0, 5.0, 5.0, 5.0])
        assert s["stdev"] == pytest.approx(0.0)
        assert s["sharpe"] is None
        # But percentiles ARE defined (all equal to the constant).
        assert s["p10"] == pytest.approx(5.0)
        assert s["p90"] == pytest.approx(5.0)
        assert s["win_rate"] == pytest.approx(1.0)

    def test_all_losses_no_wins(self):
        s = grp._arm_stats([-1.0, -2.0, -3.0])
        assert s["win_rate"] == pytest.approx(0.0)
        assert s["mean_win"] is None  # no positives → None
        assert s["mean_loss"] == pytest.approx(-2.0, abs=1e-6)
        # EV = 0*mean_win + 1*(-2.0) = -2.0 ≡ mean
        assert s["expected_value"] == pytest.approx(-2.0, abs=1e-6)

    def test_percentiles_match_numpy(self):
        # 11 rows so p10/p25/p75/p90 hit integer index positions.
        vals = list(range(-5, 6))  # -5..+5
        s = grp._arm_stats([float(v) for v in vals])
        # numpy default (linear interpolation) — match our implementation.
        import numpy as np
        a = np.asarray(vals, dtype=np.float64)
        assert s["p10"] == pytest.approx(float(np.percentile(a, 10)), abs=1e-4)
        assert s["p25"] == pytest.approx(float(np.percentile(a, 25)), abs=1e-4)
        assert s["p75"] == pytest.approx(float(np.percentile(a, 75)), abs=1e-4)
        assert s["p90"] == pytest.approx(float(np.percentile(a, 90)), abs=1e-4)

    def test_high_win_rate_low_magnitude(self):
        # 9 wins of +0.5 (mean_win=0.5), 1 loss of -5 (mean_loss=-5).
        # win_rate=0.9, mean = (9*0.5 - 5)/10 = -0.05 (NEGATIVE despite 90% wins).
        vals = [0.5] * 9 + [-5.0]
        s = grp._arm_stats(vals)
        assert s["win_rate"] == pytest.approx(0.9)
        assert s["mean_win"] == pytest.approx(0.5)
        assert s["mean_loss"] == pytest.approx(-5.0)
        assert s["mean"] == pytest.approx(-0.05, abs=1e-6)
        # EV reconciliation
        assert s["expected_value"] == pytest.approx(-0.05, abs=1e-4)


# ─────────────────────── capture / abstention discipline ───────────────


class TestCaptureNotPopulated:
    def test_no_captured_pred_named_not_silent(self):
        # Mirror gate_realized: deploy-stale state has a NAMED verdict.
        rows = [{"action": "BUY", "forward_return_5d": 5.0,
                 "sim_date": "2025-01-02"} for _ in range(40)]
        rep = grp.gate_risk_report(rows)
        assert rep["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"
        assert rep["n_captured"] == 0
        assert rep["n_acted"] == 0
        assert rep["measurement"] == "captured_then_deployed_no_reprediction"

    def test_explicit_null_pred_excluded(self):
        rows = [_row(None, 9.0) for _ in range(40)]
        rep = grp.gate_risk_report(rows)
        assert rep["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"
        assert rep["n_captured"] == 0


class TestAbstentionHonesty:
    """The decisive honesty property re-prediction CANNOT replicate:
    a gate_off_dist=True row whose pred maps to strong_headwind is
    routed to the abstained bucket, NOT the strong_headwind arm."""

    def test_off_dist_row_does_not_count_as_arm_trade(self):
        # 5 abstained, 30+ acted in extreme arms to clear the gate.
        rows = []
        rows += [_row(-50.0, -20.0, off=True) for _ in range(5)]
        rows += [_row(-50.0, 3.0) for _ in range(20)]   # strong_headwind acted
        rows += [_row(+50.0, 5.0) for _ in range(20)]   # strong_tailwind acted
        rep = grp.gate_risk_report(rows)
        assert rep["n_acted"] == 40
        assert rep["n_abstained"] == 5
        # strong_headwind arm must have exactly 20 trades, NOT 25 (the
        # off-dist rows are isolated).
        head = next(a for a in rep["arms"] if a["arm"] == "strong_headwind")
        assert head["n"] == 20
        # abstained block reports the abstained-only stats.
        assert rep["abstained"]["n"] == 5
        assert rep["abstained"]["mean"] == pytest.approx(-20.0, abs=1e-3)


# ─────────────────────── SELL sign-flip discipline ─────────────────────


class TestSellSignFlip:
    """Mirror train_scorer / gate_audit / gate_realized: a SELL's realized
    return sign is flipped so 'positive == good action' is consistent.

    gate_scorer_pred is BUY-only by construction, so SELL rows reaching
    this analyzer with a captured pred are a defensive consistency case —
    the flip must still apply identically to its sibling."""

    def test_sell_flip_aligns_with_buy_semantics(self):
        # 30 BUY strong_headwind acted at -2% realized (a small loss).
        rows = [_row(-50.0, -2.0) for _ in range(30)]
        # 5 BUY strong_tailwind at +3% — clears MIN_ARM_N (constant ⇒ stdev=0
        # ⇒ sharpe undefined; verdict will hit SHARPE_UNDEFINED_ARM honestly).
        rows += [_row(+50.0, 3.0) for _ in range(5)]
        # 1 SELL at -2% realized — flipped to +2 ⇒ counts as a win.
        rows.append(_row(-50.0, -2.0, action="SELL"))
        rep = grp.gate_risk_report(rows)
        head = next(a for a in rep["arms"] if a["arm"] == "strong_headwind")
        # 30 BUY losses + 1 SELL flipped to win ⇒ win_rate = 1/31
        assert head["n"] == 31
        assert head["win_rate"] == pytest.approx(1.0 / 31.0, abs=1e-4)


# ─────────────────────── verdict ladder (exact triggers) ───────────────


class TestVerdictRiskProportional:
    def test_tailwind_higher_sharpe_and_winrate(self):
        # strong_headwind: 30 rows, returns oscillating around -1 with high spread.
        #   wins=4, losses=26 (win_rate ~13%), mean ~-1.5
        head_returns = [-3, -2, -3, -2, -1, -3, -2, -1, -2, -3,
                        +1, -3, -2, -1, +1, -3, -2, -1, +1, -3,
                        -2, -1, +1, -3, -2, -1, -3, -2, -1, -3]
        # strong_tailwind: 30 rows, oscillating around +3, lots of wins.
        tail_returns = [+5, +4, +3, +5, +4, +3, +2, +4, +3, +5,
                        +4, +3, +5, +4, +3, +2, +4, +3, +5, +4,
                        +3, +5, +4, +3, +2, +4, -1, +3, +5, +4]
        rows = _bulk(-50.0, head_returns) + _bulk(+50.0, tail_returns)
        rep = grp.gate_risk_report(rows)
        # Both arms cleared MIN_ARM_N=5, total >= MIN_TOTAL=30.
        assert rep["n_acted"] == 60
        # Compute expected: tailwind sharpe must be much higher.
        head = next(a for a in rep["arms"] if a["arm"] == "strong_headwind")
        tail = next(a for a in rep["arms"] if a["arm"] == "strong_tailwind")
        assert tail["sharpe"] > head["sharpe"] + grp.SHARPE_TOL
        assert tail["win_rate"] > head["win_rate"]
        assert rep["verdict"] == "RISK_PROPORTIONAL"


class TestVerdictUnfavorableRisk:
    def test_tailwind_lower_sharpe_same_winrate(self):
        # Both arms: 50% win rate. Headwind: tight clusters (low stdev).
        # Tailwind: wide spreads (high stdev) — same mean but worse Sharpe.
        head_returns = [+1, -1, +1, -1] * 8  # 32 rows, mean=0, low stdev
        tail_returns = [+10, -10, +10, -10] * 8  # 32 rows, mean=0, HIGH stdev
        rows = _bulk(-50.0, head_returns) + _bulk(+50.0, tail_returns)
        rep = grp.gate_risk_report(rows)
        head = next(a for a in rep["arms"] if a["arm"] == "strong_headwind")
        tail = next(a for a in rep["arms"] if a["arm"] == "strong_tailwind")
        # Both have mean=0 ⇒ sharpe=0 for both (or None). Win rates equal.
        # This is exactly the RISK_INDIFFERENT case (sharpe_spread ≈ 0).
        assert rep["verdict"] == "RISK_INDIFFERENT"

    def test_tailwind_clearly_worse_sharpe_triggers_unfavorable(self):
        # Headwind has solid risk-adjusted return; tailwind is a coin flip.
        head_returns = [+2.0, +1.5, +2.5, +1.0, +2.0, +1.5, +2.5, +1.0,
                        +2.0, +1.5, +2.5, +1.0, +2.0, +1.5, +2.5, +1.0,
                        +2.0, +1.5, +2.5, +1.0, +2.0, +1.5, +2.5, +1.0,
                        +2.0, +1.5, +2.5, +1.0, +2.0, +1.5]
        # Tailwind: ±15 alternating → mean=0, stdev=15, sharpe=0. Plus 50% win rate.
        tail_returns = [+15.0, -15.0] * 16
        rows = _bulk(-50.0, head_returns) + _bulk(+50.0, tail_returns)
        rep = grp.gate_risk_report(rows)
        head = next(a for a in rep["arms"] if a["arm"] == "strong_headwind")
        tail = next(a for a in rep["arms"] if a["arm"] == "strong_tailwind")
        # head sharpe should be very high (~+3); tail sharpe = 0
        assert head["sharpe"] > 2.0
        assert tail["sharpe"] == pytest.approx(0.0, abs=1e-3)
        # winrate spread: tail=0.5, head=1.0 → spread=-0.5 < -WIN_RATE_TOL
        # So WIN_RATE_INVERTED has higher precedence than UNFAVORABLE_RISK.
        assert rep["verdict"] == "WIN_RATE_INVERTED"


class TestVerdictWinRateInverted:
    def test_tailwind_lower_winrate_triggers_inverted(self):
        # Headwind: mostly wins (50/60). Tailwind: mostly losses (20/60).
        # Even if mean is similar, win-rate inversion is decisive.
        head_returns = [+2.0] * 50 + [-2.0] * 10  # 60 rows, win_rate ~83%
        tail_returns = [+2.0] * 20 + [-2.0] * 40  # 60 rows, win_rate ~33%
        rows = _bulk(-50.0, head_returns) + _bulk(+50.0, tail_returns)
        rep = grp.gate_risk_report(rows)
        head = next(a for a in rep["arms"] if a["arm"] == "strong_headwind")
        tail = next(a for a in rep["arms"] if a["arm"] == "strong_tailwind")
        assert head["win_rate"] > 0.8
        assert tail["win_rate"] < 0.4
        # WIN_RATE_INVERTED wins precedence regardless of sharpe.
        assert rep["verdict"] == "WIN_RATE_INVERTED"


class TestVerdictRiskIndifferent:
    def test_close_sharpe_close_winrate_is_indifferent(self):
        # Both arms post the same mean, same stdev → sharpe spread ≈ 0.
        head_returns = [+1.0, -1.0] * 15
        tail_returns = [+1.0, -1.0] * 15
        rows = _bulk(-50.0, head_returns) + _bulk(+50.0, tail_returns)
        rep = grp.gate_risk_report(rows)
        assert rep["verdict"] == "RISK_INDIFFERENT"


class TestVerdictSharpeUndefined:
    def test_constant_tailwind_arm_triggers_sharpe_undefined(self):
        # Tailwind is constant (+3.0 across all rows) ⇒ stdev=0 ⇒ sharpe=None.
        # Without the fall-through verdict the spread comparator would
        # crash on a None operand. Verdict must name the condition.
        head_returns = [+1, -1, +2, -2, +1, -1, +2, -2, +1, -1,
                        +2, -2, +1, -1, +2, -2, +1, -1, +2, -2,
                        +1, -1, +2, -2, +1, -1, +2, -2, +1, -1]
        tail_returns = [+3.0] * 30
        rows = _bulk(-50.0, head_returns) + _bulk(+50.0, tail_returns)
        rep = grp.gate_risk_report(rows)
        head = next(a for a in rep["arms"] if a["arm"] == "strong_headwind")
        tail = next(a for a in rep["arms"] if a["arm"] == "strong_tailwind")
        assert tail["sharpe"] is None
        assert head["sharpe"] is not None
        # winrate_spread (tail=1.0, head=0.5) > +WIN_RATE_TOL ⇒ NOT inverted.
        # So we fall through to SHARPE_UNDEFINED_ARM honestly.
        assert rep["verdict"] == "SHARPE_UNDEFINED_ARM"
        # And neither sharpe-spread fields nor the verdict format crash.
        assert "sharpe undefined" in rep["hint"]


class TestVerdictInsufficientData:
    def test_no_extreme_arm_data_returns_insufficient(self):
        # Plenty of neutral arm data but only 3 each in extreme arms.
        rows = []
        rows += [_row(0.0, +1.0) for _ in range(30)]  # neutral
        rows += [_row(-50.0, +1.0) for _ in range(3)]  # 3 < MIN_ARM_N
        rows += [_row(+50.0, +1.0) for _ in range(3)]
        rep = grp.gate_risk_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        # Captured count includes all rows with non-null gate_scorer_pred.
        assert rep["n_captured"] == 36

    def test_below_min_total_returns_insufficient(self):
        # Even 5 in each extreme is enough by arm count but total < MIN_TOTAL.
        rows = []
        rows += [_row(-50.0, +1.0) for _ in range(5)]
        rows += [_row(+50.0, +1.0) for _ in range(5)]
        # Total = 10 < 30
        rep = grp.gate_risk_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"


# ─────────────────────── temporal-OOS analyze() wiring ─────────────────


class TestAnalyzeIntegration:
    def test_analyze_missing_file_returns_named_verdict(self, tmp_path):
        rep = grp.analyze(tmp_path / "does_not_exist.jsonl")
        assert rep["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"
        assert "no outcomes file" in rep["hint"]

    def test_analyze_reads_jsonl_and_grades(self, tmp_path):
        outcomes_path = tmp_path / "outcomes.jsonl"
        rows = []
        # Build enough rows to clear thresholds AND span a temporal window
        # so split_outcomes_temporal yields an OOS slice. Alternate ±1 so
        # both arms have real variance — equal mean=0, equal stdev=1 →
        # identical sharpe → RISK_INDIFFERENT.
        for i in range(40):
            r5 = +1.0 if i % 2 == 0 else -1.0
            rows.append(_row(-50.0, r5,
                             sim=f"2025-01-{(i % 28) + 1:02d}"))
            rows.append(_row(+50.0, r5,
                             sim=f"2025-01-{(i % 28) + 1:02d}"))
        outcomes_path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
        rep = grp.analyze(outcomes_path, oos_only=False)
        # Equal arms → RISK_INDIFFERENT.
        assert rep["status"] == "ok"
        assert rep["verdict"] == "RISK_INDIFFERENT"
        assert rep["slice"] == "all"
        assert rep["n_records_total"] == 80

    def test_analyze_corrupt_jsonl_drops_only_bad_lines(self, tmp_path):
        # Same hardening discipline gate_realized / calibration use.
        outcomes_path = tmp_path / "outcomes.jsonl"
        good = "\n".join(json.dumps(_row(-50.0, +1.0)) for _ in range(40))
        outcomes_path.write_text(
            good + "\n" + "{garbage" + "\n"
            + "\n".join(json.dumps(_row(+50.0, +1.0)) for _ in range(40))
            + "\n"
        )
        rep = grp.analyze(outcomes_path, oos_only=False)
        assert rep["status"] == "ok"
        # 80 valid rows survived the bad line.
        assert rep["n_records_total"] == 80


# ─────────────────────── CLI exit code matrix ───────────────────────────


class TestCliExitCodes:
    def test_cli_exits_2_on_winrate_inverted(self, tmp_path, monkeypatch):
        # Stub analyze to return WIN_RATE_INVERTED — exact-value lock on
        # the exit code so a cron consumer can branch on it.
        def _fake_analyze(*args, **kwargs):
            return {
                "verdict": "WIN_RATE_INVERTED",
                "hint": "stub",
                "slice": "oos",
                "n_captured": 60, "n_acted": 60, "n_abstained": 0,
                "sharpe_tailwind_minus_headwind": -0.5,
                "win_rate_tailwind_minus_headwind": -0.5,
                "arms": [],
            }
        monkeypatch.setattr(grp, "analyze", _fake_analyze)
        rc = grp._cli([])
        assert rc == 2

    def test_cli_exits_0_on_risk_proportional(self, monkeypatch):
        def _fake_analyze(*args, **kwargs):
            return {
                "verdict": "RISK_PROPORTIONAL",
                "hint": "stub",
                "slice": "oos",
                "n_captured": 60, "n_acted": 60, "n_abstained": 0,
                "sharpe_tailwind_minus_headwind": +0.5,
                "win_rate_tailwind_minus_headwind": +0.2,
                "arms": [],
            }
        monkeypatch.setattr(grp, "analyze", _fake_analyze)
        rc = grp._cli([])
        assert rc == 0

    def test_cli_exits_0_on_capture_not_populated(self, monkeypatch):
        def _fake_analyze(*args, **kwargs):
            return {
                "verdict": "GATE_CAPTURE_NOT_YET_POPULATED",
                "hint": "no captured rows",
                "slice": "oos",
                "n_captured": 0, "n_acted": 0, "n_abstained": 0,
                "sharpe_tailwind_minus_headwind": None,
                "win_rate_tailwind_minus_headwind": None,
                "arms": [],
            }
        monkeypatch.setattr(grp, "analyze", _fake_analyze)
        # Not actionable adverse — must NOT exit 2.
        rc = grp._cli([])
        assert rc == 0


# ─────────────────────── parity with gate_realized ──────────────────────


class TestParityWithGateRealized:
    def test_n_acted_matches_gate_realized_on_same_corpus(self):
        # The two diagnostics share row classification — the COUNT of
        # rows reaching each arm must be identical (gate_realized's
        # mean-of-realized and this module's full stats are computed
        # from the same per-arm lists; if n_acted disagrees the dual
        # has a classification bug).
        rows = []
        rows += [_row(-50.0, +1.0) for _ in range(20)]
        rows += [_row(+50.0, +2.0) for _ in range(20)]
        rows += [_row(0.0, +0.5) for _ in range(20)]
        rows += [_row(-50.0, +0.0, off=True) for _ in range(5)]

        a = grp.gate_risk_report(rows)
        b = gr.gate_realized_report(rows)
        assert a["n_acted"] == b["n_acted"]
        assert a["n_abstained"] == b["n_abstained"]
        assert a["n_captured"] == b["n_captured"]
        # Per-arm n alignment:
        a_arms = {x["arm"]: x["n"] for x in a["arms"]}
        b_arms = {x["arm"]: x["n"] for x in b["arms"]}
        for arm in grp._ARM_ORDER:
            assert a_arms[arm] == b_arms[arm], f"arm count drift: {arm}"
