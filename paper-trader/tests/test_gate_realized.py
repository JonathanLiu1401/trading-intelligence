"""Exact-value locks for the REALIZED conviction-gate diagnostic
(`paper_trader/ml/gate_realized.py`, 2026-05-18 ML+backtest hybrid feature).

Mirrors test_gate_audit.py / test_gate_pnl.py: deterministic synthetic
rows, exact metrics and exact verdicts (not ranges) so a logic change must
update the literals deliberately. All offline, no network, no trained MLP,
**no scorer/pickle at all** — this tool reads only the captured
`gate_scorer_pred` / `gate_off_dist` fields, never re-predicts.

Load-bearing assertions:
  * `gate_arm` is the SAME object imported from gate_audit (no arm drift
    between the three gate diagnostics).
  * the verdict is driven by the realized 5d spread between the two extreme
    *acted* arms, with exact hand-computed pp values.
  * the decisive honesty property re-prediction CANNOT replicate: a
    `gate_off_dist=True` row whose pred maps to strong_headwind is routed
    to the `abstained` bucket, NOT the strong_headwind arm, and is
    excluded from the verdict.
  * `gate_scorer_pred is None` rows (SELL / untrained / pre-60b20d9
    deploy-stale) are excluded entirely.
  * the SELL `-forward_return` sign-flip matches train_scorer/gate_audit.
  * the `GATE_CAPTURE_NOT_YET_POPULATED` deploy-stale state is named, not
    a silent INSUFFICIENT_DATA.
  * multi-horizon 10d/20d per-arm means are reported but not in the verdict.
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml import gate_realized as gr
from paper_trader.ml import gate_audit as ga


# ─────────────────────── SSOT: gate_arm is shared ───────────────────────


class TestArmSingleSourceOfTruth:
    def test_gate_arm_is_the_same_object_as_gate_audit(self):
        # Importing (not redeclaring) is the invariant — the five arms must
        # never drift between gate_audit / gate_pnl / gate_realized.
        assert gr.gate_arm is ga.gate_arm

    def test_thresholds_match_gate_audit(self):
        assert gr.MIN_TOTAL == ga.MIN_TOTAL == 30
        assert gr.MIN_ARM_N == ga.MIN_ARM_N == 5
        assert gr.EDGE_TOL_PP == ga.EDGE_TOL_PP == 1.0

    def test_arm_order_and_multipliers_resolve_via_gate_arm(self):
        rep = gr.gate_realized_report([])
        # Even with no data the five arms are emitted in multiplier order
        # with the SSOT-derived multipliers.
        arms = rep["arms"]
        assert [a["arm"] for a in arms] == [
            "strong_headwind", "mild_headwind", "neutral",
            "mild_tailwind", "strong_tailwind",
        ]
        assert [a["multiplier"] for a in arms] == [0.60, 0.85, 1.00, 1.15, 1.30]


# ─────────────────────── helpers ───────────────────────


def _row(pred, fwd5, *, off=False, action="BUY", fwd10=None, fwd20=None,
         sim="2025-06-02"):
    r = {
        "gate_scorer_pred": pred,
        "gate_off_dist": off,
        "action": action,
        "forward_return_5d": fwd5,
        "sim_date": sim,
        "ticker": "NVDA",
        "ml_score": 3.0,
    }
    if fwd10 is not None:
        r["forward_return_10d"] = fwd10
    if fwd20 is not None:
        r["forward_return_20d"] = fwd20
    return r


def _bulk(pred, fwd5, n, **kw):
    return [_row(pred, fwd5, **kw) for _ in range(n)]


# ─────────────────────── deploy-stale state is NAMED ───────────────────────


class TestCaptureNotPopulated:
    def test_no_captured_pred_is_named_not_silent_insufficient(self):
        # Every row mirrors the live deploy-stale corpus: gate_scorer_pred
        # absent. Must be the explicit GATE_CAPTURE_NOT_YET_POPULATED state.
        rows = [{"action": "BUY", "forward_return_5d": 5.0,
                 "sim_date": "2025-01-02"} for _ in range(40)]
        rep = gr.gate_realized_report(rows)
        assert rep["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"
        assert rep["n_captured"] == 0
        assert rep["n_acted"] == 0
        assert rep["measurement"] == "captured_then_deployed_no_reprediction"

    def test_explicit_null_pred_excluded(self):
        rows = [_row(None, 9.0) for _ in range(40)]
        rep = gr.gate_realized_report(rows)
        assert rep["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"
        assert rep["n_captured"] == 0


# ─────────────────────── the decisive honesty property ───────────────────────


class TestOffDistAbstentionHonesty:
    """A gate_off_dist=True row whose pred → strong_headwind must NOT be
    counted as a strong_headwind arm trade. This is the exact effect
    re-prediction (gate_audit/gate_pnl) structurally cannot reproduce —
    the load-bearing reason this tool exists."""

    def test_off_dist_row_goes_to_abstained_not_an_arm(self):
        # 5 abstained rows whose pred (-50) would map to strong_headwind.
        rows = _bulk(-50.0, -20.0, 5, off=True)
        # plus a populated acted set so the report progresses past capture.
        rows += _bulk(-50.0, -8.0, 6)        # real strong_headwind, mean -8
        rows += _bulk(+50.0, +12.0, 6)       # real strong_tailwind, mean +12
        rows += _bulk(0.0, 1.0, 20)          # neutral filler → n_acted=32
        rep = gr.gate_realized_report(rows)

        assert rep["n_abstained"] == 5
        assert rep["n_acted"] == 32
        sh = next(a for a in rep["arms"] if a["arm"] == "strong_headwind")
        # The 5 off-dist −20% rows must NOT have polluted strong_headwind.
        assert sh["n"] == 6
        assert sh["mean_realized_5d"] == -8.0
        # abstained mean is reported (informational), not in any arm.
        assert rep["abstained_mean_realized_5d"] == -20.0
        # Verdict graded only on acted arms: +12 − (−8) = +20 → EFFECTIVE.
        assert rep["verdict"] == "GATE_EFFECTIVE"
        assert rep["strong_tailwind_minus_headwind_pp"] == 20.0


# ─────────────────────── exact verdicts ───────────────────────


class TestVerdicts:
    def _scaffold(self, head_fwd, tail_fwd):
        # 6 strong_headwind + 6 strong_tailwind + 20 neutral = 32 acted.
        rows = _bulk(-50.0, head_fwd, 6)
        rows += _bulk(+50.0, tail_fwd, 6)
        rows += _bulk(0.0, 0.0, 20)
        return rows

    def test_gate_effective_exact_spread(self):
        rep = gr.gate_realized_report(self._scaffold(-5.0, +10.0))
        assert rep["verdict"] == "GATE_EFFECTIVE"
        # tail mean +10 − head mean −5 = +15 pp.
        assert rep["strong_tailwind_minus_headwind_pp"] == 15.0
        assert rep["n_acted"] == 32

    def test_gate_harmful_exact_spread(self):
        rep = gr.gate_realized_report(self._scaffold(+12.0, -9.0))
        assert rep["verdict"] == "GATE_HARMFUL"
        # tail −9 − head +12 = −21 pp < −EDGE_TOL_PP.
        assert rep["strong_tailwind_minus_headwind_pp"] == -21.0

    def test_cli_exits_2_only_on_gate_harmful(self, monkeypatch, capsys):
        # The cron-branchable contract: exit 2 ⇔ GATE_HARMFUL, else 0.
        monkeypatch.setattr(gr, "analyze",
                            lambda *a, **k: {"verdict": "GATE_HARMFUL"})
        assert gr._cli([]) == 2
        monkeypatch.setattr(gr, "analyze",
                            lambda *a, **k: {"verdict": "GATE_EFFECTIVE"})
        assert gr._cli([]) == 0
        monkeypatch.setattr(
            gr, "analyze",
            lambda *a, **k: {"verdict": "GATE_CAPTURE_NOT_YET_POPULATED"})
        assert gr._cli([]) == 0

    def test_gate_ineffective_within_tolerance(self):
        # tail +2.0 − head +2.5 = −0.5 pp, |−0.5| ≤ 1.0 → INEFFECTIVE.
        rep = gr.gate_realized_report(self._scaffold(2.5, 2.0))
        assert rep["verdict"] == "GATE_INEFFECTIVE"
        assert rep["strong_tailwind_minus_headwind_pp"] == -0.5

    def test_insufficient_when_extreme_arm_underpopulated(self):
        # Captured rows exist, but only 4 in strong_headwind (< MIN_ARM_N=5).
        rows = _bulk(-50.0, -3.0, 4)
        rows += _bulk(+50.0, 6.0, 6)
        rows += _bulk(0.0, 0.0, 25)
        rep = gr.gate_realized_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_captured"] == 35
        assert "strong_headwind=4" in rep["hint"]

    def test_insufficient_when_total_acted_below_min(self):
        # Both extreme arms ok (5 each) but total acted = 10 < MIN_TOTAL=30.
        rows = _bulk(-50.0, -3.0, 5) + _bulk(+50.0, 6.0, 5)
        rep = gr.gate_realized_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_acted"] == 10


# ─────────────────────── arm bucketing == gate_arm boundaries ────────────


class TestArmBoundaryParity:
    def test_boundary_predictions_bucket_exactly_like_gate_arm(self):
        # pred == -10 → mild_headwind (NOT strong_headwind: _ml_decide uses
        # `p < -10.0`); pred == 10 → mild_tailwind; pred == 5 / 0 → neutral.
        rows = []
        rows += _bulk(-10.0, 1.0, 6)   # mild_headwind
        rows += _bulk(-10.001, 1.0, 6) # strong_headwind
        rows += _bulk(10.0, 1.0, 6)    # mild_tailwind
        rows += _bulk(10.001, 1.0, 6)  # strong_tailwind
        rows += _bulk(5.0, 1.0, 4)     # neutral
        rows += _bulk(0.0, 1.0, 4)     # neutral
        rep = gr.gate_realized_report(rows)
        by = {a["arm"]: a["n"] for a in rep["arms"]}
        assert by["strong_headwind"] == 6
        assert by["mild_headwind"] == 6
        assert by["mild_tailwind"] == 6
        assert by["strong_tailwind"] == 6
        assert by["neutral"] == 8  # 4 + 4


# ─────────────────────── SELL sign-flip + bad inputs ───────────────────────


class TestSignFlipAndHardening:
    def test_sell_realized_is_flipped(self):
        # gate_scorer_pred is BUY-only by construction, but the flip is a
        # defensive consistency guard — a SELL row's −8% realized is the
        # *right* outcome, so it must score as +8 in the arm bucket.
        rows = _bulk(+50.0, -8.0, 6, action="SELL")
        rows += _bulk(-50.0, -3.0, 6)
        rows += _bulk(0.0, 0.0, 20)
        rep = gr.gate_realized_report(rows)
        st = next(a for a in rep["arms"] if a["arm"] == "strong_tailwind")
        assert st["mean_realized_5d"] == 8.0  # −(−8)

    def test_non_finite_pred_dropped(self):
        rows = [_row(float("nan"), 5.0) for _ in range(10)]
        rows += [_row(float("inf"), 5.0) for _ in range(10)]
        rep = gr.gate_realized_report(rows)
        assert rep["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"
        assert rep["n_captured"] == 0

    def test_row_missing_5d_excluded_even_if_captured(self):
        rows = [{"gate_scorer_pred": 50.0, "action": "BUY",
                 "sim_date": "2025-02-02"} for _ in range(40)]
        rep = gr.gate_realized_report(rows)
        # The gate DID act on these (captured pred present) so the capture
        # is populated — but none carry a usable 5d anchor, so the verdict
        # is the honest INSUFFICIENT_DATA, NOT the deploy-stale
        # GATE_CAPTURE_NOT_YET_POPULATED (which means "nothing captured").
        assert rep["n_captured"] == 40
        assert rep["n_acted"] == 0
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_bool_pred_is_not_treated_as_numeric(self):
        # bool is a float subclass — True must NOT become a 1.0 prediction.
        rows = [_row(True, 5.0) for _ in range(40)]
        rep = gr.gate_realized_report(rows)
        assert rep["n_captured"] == 0

    def test_never_raises_on_garbage(self):
        rep = gr.gate_realized_report(
            [None, 42, "x", {"gate_scorer_pred": "bad"}, {}]
        )
        assert rep["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"


# ─────────────────────── multi-horizon (informational) ───────────────────────


class TestMultiHorizon:
    def test_10d_20d_means_reported_per_arm_not_in_verdict(self):
        rows = _bulk(+50.0, 4.0, 6, fwd10=9.0, fwd20=18.0)
        rows += _bulk(-50.0, -2.0, 6, fwd10=-5.0, fwd20=-11.0)
        rows += _bulk(0.0, 0.0, 20)
        rep = gr.gate_realized_report(rows)
        st = next(a for a in rep["arms"] if a["arm"] == "strong_tailwind")
        sh = next(a for a in rep["arms"] if a["arm"] == "strong_headwind")
        assert st["mean_realized_10d"] == 9.0
        assert st["mean_realized_20d"] == 18.0
        assert st["n_10d"] == 6
        assert sh["mean_realized_10d"] == -5.0
        # Verdict still anchored on 5d: +4 − (−2) = +6 → EFFECTIVE.
        assert rep["verdict"] == "GATE_EFFECTIVE"
        assert rep["strong_tailwind_minus_headwind_pp"] == 6.0


# ─────────────────────── arm_monotone_fraction (informational) ────────────


class TestArmMonotoneInformational:
    def test_monotone_fraction_computed_not_in_verdict(self):
        # Realized means perfectly increasing with multiplier → 1.0.
        rows = _bulk(-50.0, -10.0, 6)   # strong_headwind  mean -10
        rows += _bulk(-5.0, -5.0, 6)    # mild_headwind    mean -5
        rows += _bulk(0.0, 0.0, 6)      # neutral          mean  0
        rows += _bulk(7.5, 5.0, 6)      # mild_tailwind    mean +5
        rows += _bulk(50.0, 10.0, 6)    # strong_tailwind  mean +10
        rep = gr.gate_realized_report(rows)
        assert rep["arm_monotone_fraction"] == 1.0
        # Independent of the verdict (driven by the 2-arm spread = 20).
        assert rep["verdict"] == "GATE_EFFECTIVE"


# ─────────────────────── analyze() — file + slice + safety ───────────────


class TestAnalyze:
    def test_missing_file_degrades(self, tmp_path):
        rep = gr.analyze(tmp_path / "nope.jsonl")
        assert rep["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"
        assert "no outcomes file" in rep["hint"]

    def test_analyze_reads_jsonl_and_reports_oos_slice(self, tmp_path):
        p = tmp_path / "decision_outcomes.jsonl"
        rows = []
        # 200 old neutral rows then a 32-row recent block. With oos_fraction
        # 0.2 the holdout is the latest int(232*0.2)=46 rows by sim_date:
        # the whole recent block (6 SH + 6 ST + 20 neutral) + 14 leaked old
        # neutral rows → n_acted=46, extremes 6/6, spread anchored on the
        # recent block (the old leak is all neutral, untouched).
        for _ in range(200):
            rows.append(_row(0.0, 0.0, sim="2024-01-02"))
        rows += _bulk(-50.0, -4.0, 6, sim="2025-12-30")
        rows += _bulk(+50.0, +9.0, 6, sim="2025-12-31")
        rows += _bulk(0.0, 0.0, 20, sim="2025-12-31")
        with p.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        rep = gr.analyze(p, oos_only=True)
        assert rep["slice"] == "oos"
        # spread 9 − (−4) = 13 pp; old leak is neutral so does not move it.
        assert rep["strong_tailwind_minus_headwind_pp"] == 13.0
        assert rep["verdict"] == "GATE_EFFECTIVE"
        assert rep["n_records_total"] == 232

    def test_analyze_all_slice_uses_everything(self, tmp_path):
        p = tmp_path / "decision_outcomes.jsonl"
        rows = _bulk(-50.0, -4.0, 6) + _bulk(+50.0, 9.0, 6) + _bulk(0.0, 0.0, 20)
        with p.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        rep = gr.analyze(p, oos_only=False)
        assert rep["slice"] == "all"
        assert rep["verdict"] == "GATE_EFFECTIVE"

    def test_corrupt_lines_skipped(self, tmp_path):
        p = tmp_path / "decision_outcomes.jsonl"
        good = _bulk(-50.0, -4.0, 6) + _bulk(+50.0, 9.0, 6) + _bulk(0.0, 0.0, 20)
        with p.open("w") as fh:
            fh.write("{not json\n")
            for r in good:
                fh.write(json.dumps(r) + "\n")
            fh.write("[]\n")  # non-dict JSON — skipped
        rep = gr.analyze(p, oos_only=False)
        assert rep["verdict"] == "GATE_EFFECTIVE"
        assert rep["n_records_total"] == 32


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
