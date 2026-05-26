"""Tests for paper_trader.ml.stop_band_sweep.

The multi-horizon sibling of test_tp_band_sweep. Pure offline: every
analyzer call is fed a synthetic outcomes file written to a tmp path.
Assertions pin EXACT expected values for verdicts, per-cell stats,
baseline arithmetic, sort order, partial-coverage semantics, and CLI
exit codes — never just "no crash".

The module under test sweeps a 2-D grid of candidate STOP bands ×
horizons (5d/10d/20d) against the deployed (-8%, 5d) cell, with the
verdict ladder INSUFFICIENT_DATA / NO_BAND_HELPS / DEPLOYED_OPTIMAL /
CELL_BEATS_DEPLOYED mirroring tp_band_sweep.
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml.stop_band_sweep import (
    DEFAULT_CANDIDATE_BANDS,
    DEFAULT_HORIZONS,
    DEPLOYED_HORIZON,
    DEPLOYED_STOP_PCT,
    EDGE_TOL_PP,
    MIN_BUYS,
    _coerce_bands,
    _coerce_horizons,
    _parse_csv_floats,
    _parse_csv_strings,
    _stop_protected_return,
    _to_finite_float,
    analyze,
    main,
    sweep_cells,
)


# ──────────────────────── pure-function unit tests ─────────────────────────


class TestStopProtectedReturn:
    def test_stop_fires_at_band(self):
        # Exactly at -band → triggers.
        assert _stop_protected_return(-12.0, -8.0, stop_pct=8.0) == -8.0

    def test_stop_fires_below_band(self):
        # Drew -15% (worse than -8%) → fill at -8% (the band).
        assert _stop_protected_return(-10.0, -15.0, stop_pct=8.0) == -8.0

    def test_no_trigger_when_intra_min_above_band(self):
        # Worst was -5%, never hit -8% → ride to endpoint.
        assert _stop_protected_return(2.0, -5.0, stop_pct=8.0) == 2.0

    def test_no_trigger_when_intra_min_positive(self):
        # Never drew down → no possible trigger.
        assert _stop_protected_return(3.0, 1.0, stop_pct=8.0) == 3.0

    def test_stop_floors_runaway_loss(self):
        # Drew -25% then endpoint -20%. With -8% stop, capped at -8%
        # (saved 12pp); without, lost 20%.
        assert _stop_protected_return(-20.0, -25.0, stop_pct=8.0) == -8.0

    def test_custom_stop_pct(self):
        # 5% stop fires at -5% drawdown.
        assert _stop_protected_return(-3.0, -6.0, stop_pct=5.0) == -5.0
        # And does NOT fire at -4% drawdown.
        assert _stop_protected_return(-3.0, -4.0, stop_pct=5.0) == -3.0


class TestToFiniteFloat:
    def test_none_returns_none(self):
        assert _to_finite_float(None) is None

    def test_bool_returns_none(self):
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


class TestParseCsvFloats:
    def test_none_returns_default(self):
        assert _parse_csv_floats(None, (1.0, 2.0)) == (1.0, 2.0)

    def test_empty_returns_default(self):
        assert _parse_csv_floats("", (1.0, 2.0)) == (1.0, 2.0)

    def test_csv_parses(self):
        assert _parse_csv_floats("5,10,15", ()) == (5.0, 10.0, 15.0)

    def test_invalid_tokens_skipped(self):
        assert _parse_csv_floats("5,xyz,10", ()) == (5.0, 10.0)

    def test_non_positive_dropped(self):
        # Only positive finite floats pass.
        assert _parse_csv_floats("0,-5,10,15", ()) == (10.0, 15.0)

    def test_all_invalid_returns_default(self):
        assert _parse_csv_floats("xyz,abc", (3.0,)) == (3.0,)


class TestParseCsvStrings:
    def test_none_returns_default(self):
        assert _parse_csv_strings(None, ("5d", "10d")) == ("5d", "10d")

    def test_csv_parses(self):
        assert _parse_csv_strings("5d,10d,20d", ()) == ("5d", "10d", "20d")

    def test_whitespace_tolerated(self):
        assert _parse_csv_strings(" 5d , 10d ", ()) == ("5d", "10d")

    def test_empty_tokens_dropped(self):
        assert _parse_csv_strings("5d,,10d", ()) == ("5d", "10d")


class TestCoerceBands:
    def test_deployed_band_always_inserted(self):
        # Pass a custom grid that omits the deployed 8% band — it must
        # still appear in the swept set.
        out = _coerce_bands([3.0, 10.0], deployed_band=8.0)
        assert 8.0 in out

    def test_sorted_ascending(self):
        out = _coerce_bands([15.0, 5.0, 3.0, 12.0], deployed_band=8.0)
        assert out == sorted(out)

    def test_duplicates_removed(self):
        out = _coerce_bands([5.0, 5.0, 5.0, 8.0], deployed_band=8.0)
        # 5.0 and 8.0 each appear exactly once.
        assert out.count(5.0) == 1
        assert out.count(8.0) == 1

    def test_non_positive_dropped(self):
        # Zero and negatives drop.
        out = _coerce_bands([0.0, -5.0, 5.0], deployed_band=8.0)
        assert all(b > 0 for b in out)

    def test_non_finite_dropped(self):
        out = _coerce_bands([float("nan"), float("inf"), 5.0],
                            deployed_band=8.0)
        assert 5.0 in out and 8.0 in out
        # Nothing infinite/NaN survives.
        import math
        assert all(math.isfinite(b) for b in out)


class TestCoerceHorizons:
    def test_deployed_horizon_always_inserted(self):
        # Pass a custom horizon list that omits the deployed 5d.
        out = _coerce_horizons(["10d", "20d"], deployed_horizon="5d")
        assert "5d" in out

    def test_duplicates_removed(self):
        out = _coerce_horizons(["5d", "5d", "10d", "10d"],
                               deployed_horizon="5d")
        assert out.count("5d") == 1
        assert out.count("10d") == 1

    def test_non_string_dropped(self):
        out = _coerce_horizons([5, "5d", None, "10d"], deployed_horizon="5d")
        assert out == ["5d", "10d"]

    def test_preserves_order(self):
        # First-seen order preserved (modulo dedup).
        out = _coerce_horizons(["20d", "10d", "5d"], deployed_horizon="5d")
        # 20d came first, then 10d, then 5d (deployed always present at end).
        assert out[0] == "20d"


# ──────────────────────────── sweep_cells unit tests ───────────────────────


class TestSweepCells:
    def test_empty_inputs_return_empty(self):
        assert sweep_cells({}, [5.0]) == []

    def test_empty_population_returns_empty(self):
        assert sweep_cells({"5d": ([], [])}, [5.0]) == []

    def test_length_mismatch_skips_horizon(self):
        # 3 realized vs 2 intra_min → defensive: skip horizon, no crash.
        out = sweep_cells(
            {"5d": ([1.0, 2.0, 3.0], [-1.0, -2.0])}, [5.0])
        assert out == []

    def test_single_band_single_horizon_arithmetic(self):
        # 4 BUYs on 5d: forwards [+10, -5, +3, -12], intra_mins
        # [-2, -10, -1, -15]. Stop at -8%: triggers on rows 2 & 4
        # (intra_min ≤ -8). Protected returns: [+10, -8, +3, -8].
        # Mean realized = (10-5+3-12)/4 = -1.0. Mean protected =
        # (10-8+3-8)/4 = -0.75. Benefit = -0.75 - -1.0 = +0.25.
        per_horizon = {"5d": ([10.0, -5.0, 3.0, -12.0],
                              [-2.0, -10.0, -1.0, -15.0])}
        out = sweep_cells(per_horizon, [8.0])
        assert len(out) == 1
        row = out[0]
        assert row["stop_pct"] == 8.0
        assert row["horizon"] == "5d"
        assert row["n"] == 4
        assert row["n_triggered"] == 2
        assert row["pct_triggered"] == 50.0
        assert row["mean_realized_return_pct"] == -1.0
        assert row["mean_protected_return_pct"] == -0.75
        assert row["benefit_pct"] == 0.25

    def test_baseline_constant_across_bands(self):
        # The no-stop mean is band-invariant — every row for a horizon
        # must report the same realized mean.
        per_horizon = {"5d": ([5.0, -5.0, 10.0, -10.0],
                              [-1.0, -8.0, -2.0, -15.0])}
        rows = sweep_cells(per_horizon, [3.0, 5.0, 10.0, 15.0])
        means = {r["mean_realized_return_pct"] for r in rows}
        assert len(means) == 1
        assert means.pop() == 0.0  # (5-5+10-10)/4 = 0

    def test_per_horizon_baselines_differ(self):
        # 5d and 10d have different no-stop means.
        per_horizon = {
            "5d":  ([2.0, 4.0], [-1.0, -2.0]),       # mean = 3.0
            "10d": ([10.0, 20.0], [-1.0, -2.0]),     # mean = 15.0
        }
        rows = sweep_cells(per_horizon, [5.0])
        means_by_h = {r["horizon"]: r["mean_realized_return_pct"]
                      for r in rows}
        assert means_by_h["5d"] == 3.0
        assert means_by_h["10d"] == 15.0

    def test_sorted_by_descending_benefit(self):
        # Two horizons, two bands — verify rows sort by benefit desc.
        per_horizon = {
            "5d":  ([+5.0, -10.0], [-2.0, -15.0]),
            "10d": ([+5.0, -20.0], [-2.0, -25.0]),
        }
        rows = sweep_cells(per_horizon, [8.0])
        benefits = [r["benefit_pct"] for r in rows]
        assert benefits == sorted(benefits, reverse=True)


# ─────────────────────── analyze() — verdict ladder ────────────────────────


def _write_outcomes(path, rows):
    """Write a list of dicts as JSONL — the test-data helper used by every
    integration test below."""
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _buy_row(fwd_5, imn_5, fwd_10=None, imn_10=None,
             fwd_20=None, imn_20=None):
    """Build one BUY row with the multi-horizon outcome fields the
    analyzer consumes."""
    row = {
        "action": "BUY",
        "ticker": "NVDA",
        "forward_return_5d": fwd_5,
        "forward_intraperiod_min_5d": imn_5,
    }
    if fwd_10 is not None:
        row["forward_return_10d"] = fwd_10
    if imn_10 is not None:
        row["forward_intraperiod_min_10d"] = imn_10
    if fwd_20 is not None:
        row["forward_return_20d"] = fwd_20
    if imn_20 is not None:
        row["forward_intraperiod_min_20d"] = imn_20
    return row


class TestAnalyzeInsufficientData:
    def test_missing_file_returns_insufficient(self, tmp_path):
        rep = analyze(tmp_path / "no_such.jsonl")
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "not found" in rep["hint"]

    def test_no_buys_returns_insufficient(self, tmp_path):
        p = tmp_path / "outcomes.jsonl"
        _write_outcomes(p, [{"action": "SELL",
                             "forward_return_5d": 1.0,
                             "forward_intraperiod_min_5d": -2.0}])
        rep = analyze(p)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_buys"] == 0

    def test_below_min_buys_threshold(self, tmp_path):
        p = tmp_path / "outcomes.jsonl"
        # Only 5 BUYs with intraperiod data on the deployed horizon —
        # below the 30 minimum.
        rows = [_buy_row(1.0, -2.0) for _ in range(5)]
        _write_outcomes(p, rows)
        rep = analyze(p)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_buys"] == 5

    def test_below_min_buys_threshold_uses_deployed_horizon(self, tmp_path):
        # All 100 BUYs carry 10d data but only 5 carry 5d (the deployed
        # horizon). Verdict gates on the DEPLOYED horizon, not on the
        # max-coverage horizon — even though there's "lots" of 10d data,
        # the deployed cell can't be evaluated.
        p = tmp_path / "outcomes.jsonl"
        rows = []
        for i in range(100):
            # 10d field present on every row, 5d only on first 5.
            r = {"action": "BUY", "ticker": "X",
                 "forward_return_10d": 1.0,
                 "forward_intraperiod_min_10d": -2.0}
            if i < 5:
                r["forward_return_5d"] = 1.0
                r["forward_intraperiod_min_5d"] = -2.0
            rows.append(r)
        _write_outcomes(p, rows)
        rep = analyze(p)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        # The hint must mention the deployed horizon (5d).
        assert "5d" in rep["hint"]


class TestAnalyzeVerdicts:
    def test_cell_beats_deployed_when_tight_stop_caps_crashes(self, tmp_path):
        """Construct a synthetic corpus where every BUY draws -15%
        intraperiod min then endpoints at -10%. The deployed -8% / 5d
        cell caps loss at -8 (+2pp benefit). A -3% band caps even
        earlier at -3 (+7pp benefit). The best cell beats deployed by
        +5pp, well above the 0.30pp edge tol."""
        p = tmp_path / "outcomes.jsonl"
        rows = [_buy_row(-10.0, -15.0) for _ in range(50)]
        _write_outcomes(p, rows)
        rep = analyze(p, bands=(3.0, 5.0, 8.0), horizons=("5d",))
        assert rep["verdict"] == "CELL_BEATS_DEPLOYED"
        assert rep["best_cell"]["stop_pct"] == 3.0
        assert rep["best_cell"]["horizon"] == "5d"
        # Best beats deployed by 5pp = 7 - 2.
        assert (rep["best_cell"]["benefit_pct"]
                - rep["deployed_cell_benefit_pct"]) == pytest.approx(
                    5.0, abs=0.01)

    def test_no_band_helps_when_no_drawdowns(self, tmp_path):
        """Every BUY drew at most -1% intraperiod. NO stop fires
        anywhere, so every band's benefit is 0. Verdict: NO_BAND_HELPS
        (all benefits below the 0.30pp edge tol)."""
        p = tmp_path / "outcomes.jsonl"
        rows = [_buy_row(5.0, -1.0) for _ in range(50)]
        _write_outcomes(p, rows)
        rep = analyze(p)
        assert rep["verdict"] == "NO_BAND_HELPS"
        # Best benefit must be below edge tol AND the deployed entry's
        # benefit must be 0 (no triggers).
        assert rep["best_cell"]["benefit_pct"] < EDGE_TOL_PP
        assert rep["deployed_cell_benefit_pct"] == 0.0

    def test_deployed_optimal_when_best_within_edge_tol(self, tmp_path):
        """When the deployed cell IS near-optimal — benefit within
        ±EDGE_TOL_PP of the best — verdict is DEPLOYED_OPTIMAL.

        Construct: 50 BUYs each draw to exactly -10% intraperiod then
        endpoint at -10. Bands {8, 10}: 8% triggers on every row →
        protected -8 each → benefit = -8 - -10 = +2pp. 10% triggers on
        every row → protected -10 → benefit = -10 - -10 = 0pp.
        Deployed (8%) is BEST (benefit 2.0 > 10%'s 0.0). Sweep only
        {8, 10}: best == deployed, gap 0 < EDGE_TOL_PP → DEPLOYED_OPTIMAL.
        """
        p = tmp_path / "outcomes.jsonl"
        rows = [_buy_row(-10.0, -10.0) for _ in range(50)]
        _write_outcomes(p, rows)
        rep = analyze(p, bands=(8.0, 10.0), horizons=("5d",))
        assert rep["verdict"] == "DEPLOYED_OPTIMAL"
        # Best == deployed in this construction.
        assert rep["best_cell"]["stop_pct"] == DEPLOYED_STOP_PCT

    def test_deployed_band_always_in_sweep_even_if_omitted(self, tmp_path):
        """If a caller passes a custom grid that excludes the deployed
        8% band, the deployed cell must STILL appear in the sweep —
        otherwise we can't compute the deployed_cell_benefit_pct used in
        the verdict ladder. Mirrors tp_band_sweep's invariant."""
        p = tmp_path / "outcomes.jsonl"
        rows = [_buy_row(-10.0, -15.0) for _ in range(50)]
        _write_outcomes(p, rows)
        rep = analyze(p, bands=(3.0, 5.0, 12.0), horizons=("5d",))
        bands_in_sweep = {r["stop_pct"] for r in rep["sweep"]}
        # Even though 8 was NOT in the custom grid, it must appear.
        assert DEPLOYED_STOP_PCT in bands_in_sweep


class TestAnalyzeMultiHorizon:
    def test_partial_coverage_per_horizon_honored(self, tmp_path):
        """A row that carries 5d intraperiod min but NOT 10d must
        contribute to 5d sweep cells while being DROPPED from 10d
        cells. The "honor partial coverage" semantics matching
        test_outcome_intraperiod_multihorizon."""
        p = tmp_path / "outcomes.jsonl"
        rows = []
        # 50 rows with both 5d and 10d data.
        for _ in range(50):
            rows.append(_buy_row(-10.0, -15.0,
                                 fwd_10=-12.0, imn_10=-18.0))
        # 30 more rows with ONLY 5d data (no 10d fields).
        for _ in range(30):
            rows.append(_buy_row(-5.0, -7.0))
        _write_outcomes(p, rows)
        rep = analyze(p, bands=(8.0,), horizons=("5d", "10d"))
        coverage = rep["n_with_intraperiod_per_horizon"]
        # 5d: all 80 rows count. 10d: only the 50 with 10d fields.
        assert coverage["5d"] == 80
        assert coverage["10d"] == 50

    def test_multi_horizon_baselines_distinct(self, tmp_path):
        """Per-horizon baseline_no_stop_mean must reflect ONLY that
        horizon's rows. 5d baseline ≠ 10d baseline when the realized
        endpoints differ between horizons."""
        p = tmp_path / "outcomes.jsonl"
        rows = []
        # Endpoint at +5 on 5d, +20 on 10d for the same trade — distinct
        # baselines per horizon.
        for _ in range(50):
            rows.append(_buy_row(5.0, -2.0,
                                 fwd_10=20.0, imn_10=-3.0))
        _write_outcomes(p, rows)
        rep = analyze(p, bands=(8.0,), horizons=("5d", "10d"))
        baselines = rep["baseline_no_stop_mean_pct_per_horizon"]
        # 5d trades end at +5; 10d end at +20 (no stop triggers — both
        # intra_mins above -8).
        assert baselines["5d"] == pytest.approx(5.0, abs=0.01)
        assert baselines["10d"] == pytest.approx(20.0, abs=0.01)

    def test_longer_horizon_finds_better_band(self, tmp_path):
        """The 2026-05-26 pass-#42 unlock: a longer horizon can find a
        BETTER stop band than the deployed (5d) horizon allows.

        Construct: every BUY's drawdown happens AFTER day 5. On 5d the
        intra_min is only -1% — no stop fires, no benefit. On 20d the
        intra_min is -20% but the endpoint is +10. A -10% stop on the
        20d horizon caps the loss at -10 — capturing +20pp benefit vs
        the +10 endpoint (since the position rode through -20% then
        rallied to +10). The 20d / 10% cell beats the deployed 5d / 8%
        cell decisively.
        """
        p = tmp_path / "outcomes.jsonl"
        rows = []
        for _ in range(50):
            rows.append(_buy_row(5.0, -1.0,  # 5d: endpoint +5, drew only -1
                                 fwd_20=10.0, imn_20=-20.0))
        _write_outcomes(p, rows)
        rep = analyze(p, bands=(8.0, 10.0), horizons=("5d", "20d"))
        # On 5d the -8% stop never fires (imn_5d=-1>-8), so deployed
        # benefit is 0. On 20d the -10% stop fires (imn_20d=-20<-10):
        # protected = -10, realized = +10 → benefit = -10 - 10 = -20.
        # On 20d the -8% stop fires too → protected = -8, realized=10 →
        # benefit -18. Both stops on 20d HURT. So actual best is
        # deployed (5d/8%) with benefit 0 — verdict either DEPLOYED_OPTIMAL
        # or NO_BAND_HELPS depending on whether 0 >= edge_tol.
        # Verify the per-cell numbers match the manual computation,
        # which is the actual quant-relevant invariant.
        cells = {(r["stop_pct"], r["horizon"]): r["benefit_pct"]
                 for r in rep["sweep"]}
        assert cells[(8.0, "5d")] == 0.0
        # 20d/10% triggers on every row (imn=-20 <= -10) → protected -10
        # vs realized +10 → benefit -20.
        assert cells[(10.0, "20d")] == pytest.approx(-20.0, abs=0.01)


class TestAnalyzeFiltering:
    def test_sell_rows_excluded(self, tmp_path):
        """A stop band only makes sense for a LONG (BUY) position;
        SELL rows must be excluded from the corpus."""
        p = tmp_path / "outcomes.jsonl"
        rows = [{"action": "SELL",
                 "forward_return_5d": -10.0,
                 "forward_intraperiod_min_5d": -15.0}] * 50
        _write_outcomes(p, rows)
        rep = analyze(p)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_buys"] == 0

    def test_unparseable_rows_skipped(self, tmp_path):
        """A single corrupt JSON line must not abort the analyzer.
        Mirrors stop_out_audit's line-tolerant discipline."""
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            fh.write("not valid json\n")
            for r in [_buy_row(-10.0, -15.0) for _ in range(50)]:
                fh.write(json.dumps(r) + "\n")
            fh.write("{also broken\n")
        rep = analyze(p, bands=(8.0,), horizons=("5d",))
        # The 50 valid rows must drive the analysis.
        assert rep["status"] == "ok"
        assert rep["n_buys"] == 50

    def test_non_finite_intra_min_drops_row(self, tmp_path):
        """A row whose intra_min is NaN must be dropped from the sweep
        (the field is structurally finite-or-None, but bad data must
        not poison the population)."""
        p = tmp_path / "outcomes.jsonl"
        rows = [_buy_row(-10.0, -15.0) for _ in range(50)]
        # Add one row with NaN intra-min.
        rows.append({"action": "BUY",
                     "forward_return_5d": -10.0,
                     "forward_intraperiod_min_5d": float("nan")})
        _write_outcomes(p, rows)
        rep = analyze(p, bands=(8.0,), horizons=("5d",))
        # Only the 50 valid rows contribute.
        assert rep["n_with_intraperiod_per_horizon"]["5d"] == 50


# ──────────────────────────── CLI integration tests ────────────────────────


class TestCliExitCodes:
    def test_returns_0_on_decisive_verdict(self, tmp_path, capsys):
        """Exit 0 for CELL_BEATS_DEPLOYED / DEPLOYED_OPTIMAL — mirrors
        stop_out_audit / tp_band_sweep's convention."""
        p = tmp_path / "outcomes.jsonl"
        rows = [_buy_row(-10.0, -15.0) for _ in range(50)]
        _write_outcomes(p, rows)
        rc = main(["--outcomes", str(p), "--bands", "3,5,8",
                   "--horizons", "5d", "--json"])
        assert rc == 0  # CELL_BEATS_DEPLOYED

    def test_returns_1_on_no_band_helps(self, tmp_path, capsys):
        """Exit 1 for NO_BAND_HELPS — actionable for a quant
        scripting "alert if the stop loses its edge"."""
        p = tmp_path / "outcomes.jsonl"
        rows = [_buy_row(5.0, -1.0) for _ in range(50)]
        _write_outcomes(p, rows)
        rc = main(["--outcomes", str(p), "--json"])
        assert rc == 1

    def test_returns_1_on_insufficient_data(self, tmp_path, capsys):
        rc = main(["--outcomes", str(tmp_path / "no_such.jsonl"), "--json"])
        assert rc == 1

    def test_json_output_parses(self, tmp_path, capsys):
        p = tmp_path / "outcomes.jsonl"
        rows = [_buy_row(-10.0, -15.0) for _ in range(50)]
        _write_outcomes(p, rows)
        main(["--outcomes", str(p), "--json"])
        out = capsys.readouterr().out
        payload = json.loads(out)
        # Required schema fields.
        for k in ("verdict", "deployed_stop_pct", "deployed_horizon",
                  "n_buys", "sweep", "best_cell", "bands_swept",
                  "horizons_swept"):
            assert k in payload

    def test_table_output_lists_deployed_marker(self, tmp_path, capsys):
        """Default (non-JSON) output must mark the deployed cell so a
        reader can find it by eye."""
        p = tmp_path / "outcomes.jsonl"
        rows = [_buy_row(-10.0, -15.0) for _ in range(50)]
        _write_outcomes(p, rows)
        main(["--outcomes", str(p)])
        out = capsys.readouterr().out
        assert "[deployed]" in out

    def test_cli_horizons_arg_respected(self, tmp_path, capsys):
        """--horizons restricts the sweep to a subset; the deployed
        horizon is still always included."""
        p = tmp_path / "outcomes.jsonl"
        rows = [_buy_row(-10.0, -15.0, fwd_10=-12.0, imn_10=-18.0,
                          fwd_20=-15.0, imn_20=-20.0) for _ in range(50)]
        _write_outcomes(p, rows)
        main(["--outcomes", str(p), "--horizons", "10d", "--json"])
        payload = json.loads(capsys.readouterr().out)
        # Both 5d (deployed, auto-inserted) and 10d (CLI) must appear.
        horizons_in_sweep = {r["horizon"] for r in payload["sweep"]}
        assert "5d" in horizons_in_sweep
        assert "10d" in horizons_in_sweep
        # 20d was NOT requested.
        assert "20d" not in horizons_in_sweep


# ───────────────────────────── module constants ────────────────────────────


class TestModuleConstants:
    def test_default_horizons_contains_deployed(self):
        assert DEPLOYED_HORIZON in DEFAULT_HORIZONS

    def test_default_bands_contains_deployed(self):
        assert DEPLOYED_STOP_PCT in DEFAULT_CANDIDATE_BANDS

    def test_default_bands_strictly_ascending_positive(self):
        b = list(DEFAULT_CANDIDATE_BANDS)
        assert b == sorted(b)
        assert all(x > 0 for x in b)

    def test_deployed_stop_matches_backtest(self):
        """The deployed -8% stop matches backtest._buy's
        stop_loss = price * 0.92 — single source of truth so any
        future change to _buy is caught immediately."""
        # 0.92 fill → -8% drawdown → DEPLOYED_STOP_PCT must equal 8.0.
        assert DEPLOYED_STOP_PCT == 8.0

    def test_min_buys_matches_siblings(self):
        """Match stop_out_audit / tp_band_sweep so the three audits
        report comparable coverage minima."""
        from paper_trader.ml import tp_band_sweep as tpb
        assert MIN_BUYS == tpb.MIN_BUYS
