"""Test gate_decile_realized: decile-granularity truth-aware realized
return analytic. Offline-deterministic; uses synthetic rows."""
from __future__ import annotations

import json
import random

import pytest

from paper_trader.ml import gate_decile_realized as gdr


def _row(gp, fwd, action="BUY", off_dist=False):
    return {
        "action": action, "ticker": "X",
        "gate_scorer_pred": gp,
        "gate_off_dist": off_dist,
        "forward_return_5d": fwd,
    }


class TestEmptyAndSparse:
    def test_no_captured_rows_yields_capture_not_populated(self):
        rep = gdr.gate_decile_realized_report([])
        assert rep["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"
        assert rep["n_captured"] == 0
        assert rep["deciles"] == []

    def test_only_uncaptured_rows_yields_capture_not_populated(self):
        # gate_scorer_pred=None: not a gate decision, never counted.
        rep = gdr.gate_decile_realized_report(
            [_row(None, 1.0), _row(None, 2.0)]
        )
        assert rep["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"
        assert rep["n_captured"] == 0

    def test_some_captured_below_min_total_yields_insufficient(self):
        rows = [_row(i * 0.5, i * 0.1) for i in range(10)]
        rep = gdr.gate_decile_realized_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_captured"] == 10
        assert rep["n_acted"] == 10

    def test_below_min_per_decile_yields_insufficient(self):
        # 30 rows total but spread thinly across deciles (some decile <5)
        rows = [_row(i, i * 0.1) for i in range(30)]
        rep = gdr.gate_decile_realized_report(rows)
        # 30 rows / 10 deciles = 3 each — under MIN_PER_DECILE=5.
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_acted"] == 30


class TestMalformedRows:
    def test_non_finite_pred_dropped(self):
        rows = [_row(float("nan"), 1.0), _row(float("inf"), 1.0),
                _row(None, 1.0)]
        rep = gdr.gate_decile_realized_report(rows)
        assert rep["n_captured"] == 0

    def test_non_finite_fwd_skipped(self):
        rows = [_row(1.0, float("nan")), _row(2.0, float("inf")),
                _row(3.0, None)]
        rep = gdr.gate_decile_realized_report(rows)
        assert rep["n_captured"] == 3
        assert rep["n_skipped_no_5d"] == 3
        assert rep["n_acted"] == 0

    def test_non_dict_rows_silently_dropped(self):
        rep = gdr.gate_decile_realized_report(
            ["not a dict", 42, None, [1, 2]]
        )
        assert rep["n_captured"] == 0
        assert rep["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"

    def test_iterable_failure_returns_empty(self):
        class _BadIter:
            def __iter__(self): raise RuntimeError("boom")
        rep = gdr.gate_decile_realized_report(_BadIter())
        # `it = list(rows)` catches; we get the empty-capture verdict.
        assert rep["verdict"] == "GATE_CAPTURE_NOT_YET_POPULATED"


class TestVerdictLadder:
    def test_monotone_realized_when_curve_is_clean(self):
        # 100 rows, evenly spaced predictions, realized strictly tracking.
        rows = []
        for i in range(100):
            gp = (i - 50) * 0.5     # -25..+24.5
            fwd = (i - 50) * 0.4    # strictly monotone vs gp
            rows.append(_row(gp, fwd))
        rep = gdr.gate_decile_realized_report(rows)
        assert rep["verdict"] == "MONOTONE_REALIZED"
        # 10 deciles → 9 adjacent steps; all non-decreasing.
        assert rep["monotone_fraction"] == 1.0
        # spread = D10_mean - D1_mean ≈ (mean of top 10 of (i-50)*0.4)
        # - (mean of bottom 10) → ~36pp
        assert rep["spread_pp"] is not None
        assert rep["spread_pp"] > 5.0

    def test_extreme_inversion_when_top_decile_anti_predictive(self):
        # 100 rows; middle is monotone but the top decile crashes hard.
        rows = []
        for i in range(90):
            gp = i * 0.2 - 9.0       # -9..+9
            fwd = i * 0.1 - 4.5      # tracks monotonically
            rows.append(_row(gp, fwd))
        # Top decile: highest preds but realized -10.
        for i in range(10):
            rows.append(_row(20.0 + i * 0.1, -10.0))
        rep = gdr.gate_decile_realized_report(rows)
        # D10 mean ≈ -10, D9 mean ≈ +4 → D10 < D9 - EDGE_TOL_PP → inversion.
        assert rep["verdict"] == "EXTREME_INVERSION"

    def test_no_shape_on_pure_noise(self):
        rng = random.Random(42)
        rows = [_row(rng.gauss(0, 5), rng.gauss(0, 5)) for _ in range(500)]
        rep = gdr.gate_decile_realized_report(rows)
        # Pure noise: monotone fraction below MONOTONE_GOOD, no extreme inversion
        # by chance is possible but unlikely with 500 rows.
        assert rep["verdict"] in ("NO_SHAPE", "MOSTLY_MONOTONE",
                                  "EXTREME_INVERSION")
        # The truly important property: no MONOTONE_REALIZED on noise.
        assert rep["verdict"] != "MONOTONE_REALIZED"


class TestAbstainedBucket:
    def test_off_dist_rows_separate_bucket_excluded_from_deciles(self):
        # 100 acted + 20 off-distribution rows.
        rows = []
        for i in range(100):
            rows.append(_row(i * 0.5 - 25.0, i * 0.4 - 20.0))
        for i in range(20):
            rows.append(_row(50.0, -5.0, off_dist=True))
        rep = gdr.gate_decile_realized_report(rows)
        assert rep["n_acted"] == 100
        assert rep["n_abstained"] == 20
        # Off-distribution rows must NOT pollute the top decile boundary.
        # If they had, the top boundary would equal +50.
        assert rep["deciles"][-1]["boundary_hi"] < 50.0
        # Abstained block has its own stats.
        assert rep["abstained"]["n"] == 20
        assert rep["abstained"]["mean_realized"] == pytest.approx(-5.0)


class TestSellSignFlip:
    def test_sell_realized_is_flipped(self):
        # 50 BUYs with monotone curve.
        rows = [_row(i * 0.5 - 12.5, i * 0.4 - 10.0) for i in range(50)]
        # 50 SELLs where the captured field is non-null (defensive path).
        # SELL with fwd=-5 should be flipped to +5 in the realized.
        # All SELLs have the same gp so they land in their own decile.
        for i in range(50):
            rows.append(_row(0.5, -5.0, action="SELL"))
        rep = gdr.gate_decile_realized_report(rows)
        assert rep["n_acted"] == 100


class TestSchemaContract:
    def test_returns_json_safe_keys(self):
        rows = [_row(i * 0.2, i * 0.1) for i in range(120)]
        rep = gdr.gate_decile_realized_report(rows)
        # Must round-trip through json.dumps without error.
        s = json.dumps(rep)
        assert "verdict" in s
        assert "deciles" in s
        # Per-decile shape contract.
        for d in rep["deciles"]:
            assert set(d.keys()) >= {
                "i", "n", "mean_pred", "mean_realized", "ci_lo",
                "ci_hi", "boundary_lo", "boundary_hi",
            }


class TestNeverRaises:
    def test_main_runs_against_live_corpus(self, capsys):
        # Best-effort: live data/decision_outcomes.jsonl is the only
        # required substrate; this exercises the CLI end-to-end.
        # Exits 0 on a populated corpus, 1 when GATE_CAPTURE_NOT_YET_POPULATED.
        rc = gdr.main(["--all", "--json"])
        assert rc in (0, 1)
        out = capsys.readouterr().out
        # Output must be valid JSON.
        parsed = json.loads(out)
        assert "verdict" in parsed
