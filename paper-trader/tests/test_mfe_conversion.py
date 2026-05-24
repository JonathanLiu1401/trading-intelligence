"""Tests for `paper_trader.ml.mfe_conversion` — the read-only diagnostic
that audits the deployed `_buy` ``take_profit = price * 1.15`` band's
realized economic effect on the captured ``forward_intraperiod_max_5d``
field. Sibling of test_stop_out_audit.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_trader.ml.mfe_conversion import (
    BENEFIT_MARGIN,
    MIN_BUYS,
    TP_PCT,
    _tp_protected_return,
    analyze,
    main,
)


# ─────────────────────────── _tp_protected_return ───────────────────────────


class TestTPProtectedReturn:
    def test_tp_fires_exactly_at_band(self):
        # intra_max == tp_pct triggers (>=), realized exactly the band
        assert _tp_protected_return(2.0, 15.0, tp_pct=15.0) == 15.0

    def test_tp_fires_above_band(self):
        # intra_max above tp_pct — fill modeled at band (conservative)
        assert _tp_protected_return(-5.0, 22.0, tp_pct=15.0) == 15.0

    def test_tp_does_not_fire_below_band(self):
        # intra_max just below band — endpoint return is captured
        assert _tp_protected_return(8.5, 14.9, tp_pct=15.0) == 8.5

    def test_no_positive_excursion_passes_endpoint(self):
        # MFE at or below 0 means TP can never fire
        assert _tp_protected_return(-3.0, -0.5, tp_pct=15.0) == -3.0
        assert _tp_protected_return(-3.0, 0.0, tp_pct=15.0) == -3.0

    def test_custom_tp_pct(self):
        assert _tp_protected_return(2.0, 10.0, tp_pct=10.0) == 10.0
        assert _tp_protected_return(2.0, 9.9, tp_pct=10.0) == 2.0


# ─────────────────────────── analyze() — fixture-driven ─────────────────────


def _write_outcomes(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _buy_row(fwd: float, intra_max: float, **kw) -> dict:
    base = {
        "action": "BUY",
        "ticker": "NVDA",
        "sim_date": "2026-01-01",
        "forward_return_5d": fwd,
        "forward_intraperiod_max_5d": intra_max,
    }
    base.update(kw)
    return base


class TestAnalyzeReturnsInsufficientWhenFileMissing:
    def test_missing_outcomes_file(self, tmp_path: Path):
        rep = analyze(outcomes_path=tmp_path / "absent.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["status"] == "insufficient_data"
        assert "not found" in rep["hint"]


class TestAnalyzeFiltersAndCounts:
    def test_sell_rows_excluded_from_n_buys(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        rows = [_buy_row(5.0, 12.0)]
        # Add SELL row that should be ignored entirely
        rows.append({"action": "SELL", "ticker": "AAPL",
                     "forward_return_5d": -3.0,
                     "forward_intraperiod_max_5d": 5.0})
        _write_outcomes(p, rows)
        rep = analyze(outcomes_path=p, min_buys=1)
        assert rep["n_buys"] == 1
        assert rep["n_with_intraperiod"] == 1

    def test_rows_missing_intra_dropped_from_with_count(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        rows = [
            _buy_row(5.0, 12.0),
            # missing forward_intraperiod_max_5d → not counted
            {"action": "BUY", "ticker": "X", "forward_return_5d": 2.0},
            # explicit null intra → dropped
            _buy_row(1.0, None),
        ]
        _write_outcomes(p, rows)
        rep = analyze(outcomes_path=p, min_buys=1)
        assert rep["n_buys"] == 3
        assert rep["n_with_intraperiod"] == 1

    def test_nonfinite_intra_dropped(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        rows = [
            _buy_row(2.0, float("nan")),
            _buy_row(2.0, float("inf")),
            _buy_row(2.0, 10.0),
        ]
        # nan/inf serialize as floats — but JSON doesn't allow them with
        # strict mode. Manually write as the json module's default
        # (allow_nan=True writes "NaN"/"Infinity"). _iter_rows uses
        # json.loads which accepts these by default.
        _write_outcomes(p, rows)
        rep = analyze(outcomes_path=p, min_buys=1)
        # Only the third row (finite 10.0) counts toward intraperiod
        assert rep["n_with_intraperiod"] == 1


class TestAnalyzeBelowMinBuys:
    def test_insufficient_data_when_under_min_buys(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        # 5 BUYs all with intraperiod data; min_buys=30 default
        rows = [_buy_row(2.0, 10.0) for _ in range(5)]
        _write_outcomes(p, rows)
        rep = analyze(outcomes_path=p)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_buys"] == 5
        assert rep["n_with_intraperiod"] == 5
        assert "5 BUYs with intraperiod" in rep["hint"]


class TestAnalyzeTPHelpsWhenPositionsPeakThenRevert:
    """Synthetic corpus: every BUY peaks above the +15% TP band then
    reverts to a much lower endpoint. With-TP captures 15%; without
    captures the lower endpoint. TP must help."""

    def test_tp_helps_on_peak_then_revert_corpus(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        # 50 BUYs: each peaks at +20% then reverts to +1% (poor conversion)
        rows = [_buy_row(1.0, 20.0) for _ in range(50)]
        _write_outcomes(p, rows)
        rep = analyze(outcomes_path=p)

        # Every BUY triggers the +15% TP
        assert rep["n_tp_triggered"] == 50
        assert rep["pct_tp_triggered"] == 100.0
        # Mean realized: 1.0pp. Mean with TP: 15.0pp. Benefit: +14.0pp
        assert rep["mean_realized_return_pct"] == pytest.approx(1.0, abs=1e-6)
        assert rep["mean_tp_protected_return_pct"] == pytest.approx(15.0, abs=1e-6)
        assert rep["tp_benefit_pct"] == pytest.approx(14.0, abs=1e-6)
        assert rep["verdict"] == "TP_HELPS"
        # MFE = +20% on every row
        assert rep["mean_mfe_pct"] == pytest.approx(20.0, abs=1e-6)
        # Conversion ratio = 1.0 / 20.0 = 0.05 on every row
        assert rep["mean_conversion_ratio"] == pytest.approx(0.05, abs=1e-6)
        assert rep["n_positive_mfe"] == 50
        assert rep["n_reverted"] == 0  # endpoint +1 is still positive


class TestAnalyzeTPHurtsWhenPositionsAlsoBlowPast:
    """Synthetic: every BUY peaks at exactly +15% (just triggers TP)
    then continues higher to +30% endpoint. With-TP exits at +15%
    forfeiting the further +15%. TP must hurt by ~15pp."""

    def test_tp_hurts_on_through_peak_corpus(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        # 50 BUYs: peak at +30%, endpoint at +30% (peak == endpoint = no revert)
        rows = [_buy_row(30.0, 30.0) for _ in range(50)]
        _write_outcomes(p, rows)
        rep = analyze(outcomes_path=p)

        assert rep["n_tp_triggered"] == 50
        # Without TP: +30 mean. With TP: +15 mean. Benefit: -15pp.
        assert rep["mean_realized_return_pct"] == pytest.approx(30.0, abs=1e-6)
        assert rep["mean_tp_protected_return_pct"] == pytest.approx(15.0, abs=1e-6)
        assert rep["tp_benefit_pct"] == pytest.approx(-15.0, abs=1e-6)
        assert rep["verdict"] == "TP_HURTS"
        # Conversion is perfect (endpoint == peak)
        assert rep["mean_conversion_ratio"] == pytest.approx(1.0, abs=1e-6)


class TestAnalyzeTPNeutralWhenBandRarelyFires:
    """Synthetic: every BUY peaks at +5% (below the +15% band) — TP
    never fires, so with-TP equals without-TP exactly. Benefit must be
    exactly zero → NEUTRAL."""

    def test_tp_neutral_when_no_triggers(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        rows = [_buy_row(2.0, 5.0) for _ in range(50)]
        _write_outcomes(p, rows)
        rep = analyze(outcomes_path=p)
        assert rep["n_tp_triggered"] == 0
        assert rep["pct_tp_triggered"] == 0.0
        assert rep["tp_benefit_pct"] == pytest.approx(0.0, abs=1e-9)
        assert rep["verdict"] == "TP_NEUTRAL"


class TestRevertedCounter:
    """A BUY whose intra_max was positive but endpoint is non-positive
    is "reverted" — the count is the central peak-then-crater signal."""

    def test_reverted_counted_only_for_positive_mfe_with_nonpositive_endpoint(
        self, tmp_path: Path
    ):
        p = tmp_path / "outcomes.jsonl"
        rows = [
            # Positive MFE + non-positive endpoint → reverted
            _buy_row(-3.0, 5.0),  # peaked at +5%, ended at -3%
            _buy_row(0.0, 4.0),   # peaked at +4%, ended at exactly 0
            # Positive MFE + positive endpoint → NOT reverted
            _buy_row(2.0, 10.0),
            # Non-positive MFE → conversion not computed, can't revert
            _buy_row(-5.0, -1.0),
            _buy_row(-2.0, 0.0),
        ]
        # Pad to satisfy MIN_BUYS=30
        rows += [_buy_row(2.0, 5.0) for _ in range(30)]
        _write_outcomes(p, rows)
        rep = analyze(outcomes_path=p, min_buys=30)
        # 2 reverted (rows with positive MFE + non-positive endpoint)
        assert rep["n_reverted"] == 2
        # positive-MFE count: rows 1, 2, 3 + 30 padding = 33
        assert rep["n_positive_mfe"] == 33
        # pct_reverted = 2/33 * 100 = 6.06%
        assert rep["pct_reverted"] == pytest.approx(6.06, abs=0.01)


class TestConversionRatioClamping:
    """A thin positive peak (+0.1%) with a massive negative endpoint
    (-9%) would naively give ratio = -90; clamping to -10 keeps the
    statistic representative of "peak then crater" without letting one
    pathological row dominate the mean."""

    def test_ratio_floored_at_minus_10(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        # ratio = -9 / 0.1 = -90, clamped to -10
        rows = [_buy_row(-9.0, 0.1)]
        # Pad above MIN_BUYS
        rows += [_buy_row(0.5, 1.0) for _ in range(30)]
        _write_outcomes(p, rows)
        rep = analyze(outcomes_path=p, min_buys=30)
        # Expected mean: (1 * -10 + 30 * 0.5) / 31 = (-10 + 15) / 31 = 5/31 = 0.161
        assert rep["mean_conversion_ratio"] == pytest.approx(5.0 / 31.0, abs=1e-4)

    def test_ratio_capped_at_1(self, tmp_path: Path):
        # An impossible case where endpoint > MFE (shouldn't happen with
        # honest data) is capped at 1.0 — defensive.
        p = tmp_path / "outcomes.jsonl"
        rows = [_buy_row(20.0, 10.0)]
        rows += [_buy_row(0.5, 1.0) for _ in range(30)]
        _write_outcomes(p, rows)
        rep = analyze(outcomes_path=p, min_buys=30)
        # First row contributes capped 1.0 (not 2.0)
        # mean = (1.0 + 30 * 0.5) / 31 = 16/31
        assert rep["mean_conversion_ratio"] == pytest.approx(16.0 / 31.0, abs=1e-4)


class TestEnvelopeShapeMatchesStopOutAudit:
    """The on-disk shape of our report must mirror stop_out_audit's
    enough that a ledger / dashboard can table both audits side-by-side.
    Verify the key set on a typical OK report."""

    def test_ok_envelope_keys(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        rows = [_buy_row(2.0, 10.0) for _ in range(50)]
        _write_outcomes(p, rows)
        rep = analyze(outcomes_path=p)
        # status / verdict / n_buys / n_with_intraperiod / band-related keys
        for k in ("status", "verdict", "tp_pct", "benefit_margin_pp",
                  "n_buys", "n_with_intraperiod", "n_tp_triggered",
                  "pct_tp_triggered", "mean_realized_return_pct",
                  "mean_tp_protected_return_pct", "tp_benefit_pct",
                  "median_realized_return_pct",
                  "median_tp_protected_return_pct",
                  "mean_mfe_pct", "median_mfe_pct",
                  "mean_conversion_ratio", "median_conversion_ratio",
                  "n_positive_mfe", "n_reverted", "pct_reverted",
                  "hint"):
            assert k in rep, f"missing key in OK envelope: {k}"
        # Verdict alone is one of the four documented
        assert rep["verdict"] in {"TP_HELPS", "TP_HURTS", "TP_NEUTRAL",
                                  "INSUFFICIENT_DATA"}


class TestAnalyzeHandlesMalformedJSON:
    """A single corrupt line must not abort the whole audit — same
    discipline as _compute_decision_outcomes / _inject_and_train."""

    def test_corrupt_line_dropped(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            fh.write(json.dumps(_buy_row(2.0, 10.0)) + "\n")
            fh.write("this is not json\n")  # corrupt line
            fh.write(json.dumps(_buy_row(3.0, 11.0)) + "\n")
        rep = analyze(outcomes_path=p, min_buys=1)
        # 2 BUYs honored; corrupt line silently dropped
        assert rep["n_buys"] == 2
        assert rep["n_with_intraperiod"] == 2

    def test_non_dict_row_dropped(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            fh.write(json.dumps([1, 2, 3]) + "\n")  # list, not dict
            fh.write(json.dumps(_buy_row(2.0, 10.0)) + "\n")
        rep = analyze(outcomes_path=p, min_buys=1)
        assert rep["n_buys"] == 1


# ─────────────────────────── CLI ─────────────────────────────────────────────


class TestCLIExitCode:
    def test_exit_0_on_decisive_verdict(self, tmp_path: Path, capsys):
        p = tmp_path / "outcomes.jsonl"
        # TP_HELPS corpus
        rows = [_buy_row(1.0, 20.0) for _ in range(50)]
        _write_outcomes(p, rows)
        rc = main(["--outcomes", str(p), "--json"])
        captured = capsys.readouterr()
        out = json.loads(captured.out)
        assert out["verdict"] == "TP_HELPS"
        assert rc == 0

    def test_exit_1_on_neutral(self, tmp_path: Path, capsys):
        p = tmp_path / "outcomes.jsonl"
        # TP never fires — neutral
        rows = [_buy_row(2.0, 5.0) for _ in range(50)]
        _write_outcomes(p, rows)
        rc = main(["--outcomes", str(p), "--json"])
        captured = capsys.readouterr()
        out = json.loads(captured.out)
        assert out["verdict"] == "TP_NEUTRAL"
        assert rc == 1

    def test_exit_1_on_insufficient_data(self, tmp_path: Path, capsys):
        p = tmp_path / "outcomes.jsonl"
        _write_outcomes(p, [_buy_row(2.0, 10.0) for _ in range(3)])
        rc = main(["--outcomes", str(p), "--json"])
        captured = capsys.readouterr()
        out = json.loads(captured.out)
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert rc == 1

    def test_custom_tp_flag(self, tmp_path: Path, capsys):
        p = tmp_path / "outcomes.jsonl"
        # Peak at +12 → triggers a +10% TP, but not the +15% default
        rows = [_buy_row(2.0, 12.0) for _ in range(50)]
        _write_outcomes(p, rows)
        rc = main(["--outcomes", str(p), "--tp", "10.0", "--json"])
        captured = capsys.readouterr()
        out = json.loads(captured.out)
        # With --tp 10: every row triggers; with-TP = +10, raw = +2 → benefit +8
        assert out["n_tp_triggered"] == 50
        assert out["mean_tp_protected_return_pct"] == pytest.approx(10.0, abs=1e-6)
        assert out["tp_benefit_pct"] == pytest.approx(8.0, abs=1e-6)
        assert out["verdict"] == "TP_HELPS"
