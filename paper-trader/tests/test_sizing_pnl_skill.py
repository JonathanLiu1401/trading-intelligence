"""Tests for paper_trader.ml.sizing_pnl_skill.

The diagnostic buckets BUY outcome rows by ``conviction_pct`` quantile,
computes per-bucket mean of ``conviction_pct × forward_return_5d`` (the
sized PnL contribution per trade in pp of book), and verdicts on whether
the top conviction bucket out-realizes the bottom in dollar terms.

Verdict ladder (test-locked, exact-value):

| Verdict | Trigger |
|---|---|
| INSUFFICIENT_DATA | < MIN_ROWS BUY rows with conviction_pct + forward_return_5d OR < BUCKET_MIN_ROWS in top/bottom |
| TOP_BUCKET_BLEEDS | top bucket mean realized PnL ≤ BLEED_PCT (largest bets lose money) |
| SIZING_INVERTED   | bottom > top by ≥ INVERSION_PCT (sizing rule is anti-skillful) |
| SIZING_PAYS       | top > bottom by ≥ EDGE_PCT AND top > 0 |
| BALANCED          | |spread| ≤ FLAT_BAND_PCT |
| WEAK_EDGE         | 0 < spread < EDGE_PCT (positive but below threshold) |
| WEAK_INVERSION    | -INVERSION_PCT < spread < 0 (negative but above threshold) |
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_trader.ml import sizing_pnl_skill as sps


def _row(conv: float | None, fr5: float | None, action: str = "BUY",
         **extra) -> dict:
    base = {
        "action": action,
        "conviction_pct": conv,
        "forward_return_5d": fr5,
    }
    base.update(extra)
    return base


def _uniform_buy_rows(n: int, conv: float, fr: float) -> list[dict]:
    return [_row(conv, fr) for _ in range(n)]


# ──────────────────────────── Verdict ladder ────────────────────────────


class TestVerdictLadder:

    def test_insufficient_data_below_min_rows(self):
        rows = _uniform_buy_rows(sps.MIN_ROWS - 1, 0.20, 5.0)
        rep = sps.sizing_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == sps.MIN_ROWS - 1
        assert rep["by_bucket"] == []

    def test_insufficient_data_empty(self):
        rep = sps.sizing_report([])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_top_bucket_bleeds(self):
        # Bottom bucket positive (small bets + small profit), top bucket
        # negative (big bets + big losses). Should fire TOP_BUCKET_BLEEDS
        # because top bucket mean PnL ≤ 0. Uses 3 buckets so the conv
        # values 0.05/0.15/0.40 each land in a distinct bucket cleanly.
        rows = []
        # 20 low-conviction wins: conv=0.05, fr=+2 → PnL = 0.10pp
        for _ in range(20):
            rows.append(_row(0.05, 2.0))
        # 20 mid: conv=0.15, fr=+1 → PnL = +0.15pp
        for _ in range(20):
            rows.append(_row(0.15, 1.0))
        # 20 high-conviction losses: conv=0.40, fr=-3 → PnL = -1.20pp
        for _ in range(20):
            rows.append(_row(0.40, -3.0))
        rep = sps.sizing_report(rows, n_buckets=3)
        assert rep["verdict"] == "TOP_BUCKET_BLEEDS"
        top = rep["by_bucket"][-1]
        assert top["mean_realized_pnl"] < 0
        assert top["mean_conv"] == pytest.approx(0.40, abs=1e-6)

    def test_sizing_pays(self):
        # Higher conviction → higher realized return → higher per-trade PnL.
        # bottom conv=0.05 fr=+1 → PnL=+0.05pp
        # top conv=0.40 fr=+5 → PnL=+2.0pp → spread ≈ 1.95pp, way above EDGE_PCT=0.10
        rows = []
        for _ in range(20):
            rows.append(_row(0.05, 1.0))
        for _ in range(20):
            rows.append(_row(0.15, 2.0))
        for _ in range(20):
            rows.append(_row(0.40, 5.0))
        rep = sps.sizing_report(rows, n_buckets=3)
        assert rep["verdict"] == "SIZING_PAYS"
        # Verify the bucket means are as expected
        top = rep["by_bucket"][-1]
        bot = rep["by_bucket"][0]
        assert top["mean_realized_pnl"] == pytest.approx(0.40 * 5.0, abs=1e-4)
        assert bot["mean_realized_pnl"] == pytest.approx(0.05 * 1.0, abs=1e-4)
        assert rep["spread_pct"] == pytest.approx(top["mean_realized_pnl"]
                                                  - bot["mean_realized_pnl"],
                                                  abs=1e-4)

    def test_sizing_inverted(self):
        # Low conviction realizes more than high → top - bottom ≤ -INVERSION_PCT.
        # bottom conv=0.05 fr=+8 → PnL=+0.40pp
        # top conv=0.40 fr=+0.5 → PnL=+0.20pp (still positive — not BLEEDS)
        # spread = 0.20 - 0.40 = -0.20pp ≤ -INVERSION_PCT=0.10 → SIZING_INVERTED
        rows = []
        for _ in range(20):
            rows.append(_row(0.05, 8.0))
        for _ in range(20):
            rows.append(_row(0.15, 2.0))
        for _ in range(20):
            rows.append(_row(0.40, 0.5))
        rep = sps.sizing_report(rows, n_buckets=3)
        assert rep["verdict"] == "SIZING_INVERTED"
        top = rep["by_bucket"][-1]
        assert top["mean_realized_pnl"] > 0  # not BLEEDS
        assert rep["spread_pct"] < -sps.INVERSION_PCT

    def test_balanced(self):
        # Set up so spread is tiny: all buckets contribute similar mean PnL.
        # bottom conv=0.10 fr=+2 → +0.20pp
        # top conv=0.20 fr=+1 → +0.20pp
        # spread = 0.0 → BALANCED
        rows = []
        for _ in range(20):
            rows.append(_row(0.10, 2.0))
        for _ in range(20):
            rows.append(_row(0.15, 1.333333))   # → 0.20
        for _ in range(20):
            rows.append(_row(0.20, 1.0))
        rep = sps.sizing_report(rows, n_buckets=3)
        assert rep["verdict"] == "BALANCED"
        assert abs(rep["spread_pct"]) <= sps.FLAT_BAND_PCT

    def test_weak_edge(self):
        # spread positive but below EDGE_PCT=0.10.
        # bottom conv=0.10 fr=+1 → +0.10pp
        # top conv=0.20 fr=+0.8 → +0.16pp
        # spread = 0.06 → between FLAT_BAND (0.05) and EDGE (0.10) → WEAK_EDGE
        rows = []
        for _ in range(20):
            rows.append(_row(0.10, 1.0))
        for _ in range(20):
            rows.append(_row(0.15, 0.9))
        for _ in range(20):
            rows.append(_row(0.20, 0.8))
        rep = sps.sizing_report(rows, n_buckets=3)
        # spread = 0.20*0.8 - 0.10*1.0 = 0.16 - 0.10 = 0.06
        assert rep["spread_pct"] == pytest.approx(0.06, abs=1e-4)
        assert rep["verdict"] == "WEAK_EDGE"

    def test_weak_inversion(self):
        # spread between -INVERSION_PCT (-0.10) and -FLAT_BAND (-0.05).
        # bottom conv=0.10 fr=+2 → +0.20pp
        # top conv=0.20 fr=+0.65 → +0.13pp
        # spread = -0.07 → WEAK_INVERSION
        rows = []
        for _ in range(20):
            rows.append(_row(0.10, 2.0))
        for _ in range(20):
            rows.append(_row(0.15, 1.3))
        for _ in range(20):
            rows.append(_row(0.20, 0.65))
        rep = sps.sizing_report(rows, n_buckets=3)
        assert rep["spread_pct"] == pytest.approx(0.20 * 0.65 - 0.10 * 2.0,
                                                  abs=1e-4)
        assert rep["verdict"] == "WEAK_INVERSION"


# ──────────────────────────── Bucket geometry ────────────────────────────


class TestBucketGeometry:

    def test_quantile_edges_quartile(self):
        # Equal-spacing input — quartile cuts at 0.25, 0.5, 0.75
        # quantiles of [0..1] in 5-step intervals.
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        edges = sps._quantile_edges(values, 4)
        assert len(edges) == 3
        # numpy.percentile([0,.25,.5,.75,1], [25,50,75], method='linear')
        # ⇒ [0.25, 0.5, 0.75]
        assert edges[0] == pytest.approx(0.25, abs=1e-6)
        assert edges[1] == pytest.approx(0.50, abs=1e-6)
        assert edges[2] == pytest.approx(0.75, abs=1e-6)

    def test_assign_bucket_top_inclusive(self):
        edges = [0.10, 0.20, 0.30]
        assert sps._assign_bucket(0.05, edges) == 0
        assert sps._assign_bucket(0.10, edges) == 0   # ≤ edge → bucket 0
        assert sps._assign_bucket(0.15, edges) == 1
        assert sps._assign_bucket(0.20, edges) == 1
        assert sps._assign_bucket(0.25, edges) == 2
        assert sps._assign_bucket(0.30, edges) == 2
        assert sps._assign_bucket(0.40, edges) == 3   # > last edge → top

    def test_bucket_count_sums_to_total(self):
        rows = _uniform_buy_rows(60, 0.20, 5.0)
        # All identical conviction → all rows in bucket 0; the verdict will
        # be INSUFFICIENT_DATA because top bucket is empty. But the n total
        # must still match the input.
        rep = sps.sizing_report(rows, n_buckets=4)
        total_n = sum(b["n"] for b in rep["by_bucket"])
        assert total_n == 60

    def test_bucket_n_distribution(self):
        # 30 rows at conv 0.05, 30 at conv 0.40 — should land roughly half
        # in low half-bucket and half in top half-bucket with 2 buckets.
        rows = ([_row(0.05, 1.0) for _ in range(30)]
                + [_row(0.40, 1.0) for _ in range(30)])
        rep = sps.sizing_report(rows, n_buckets=2)
        # Bucket 0 should contain only the 0.05 rows, bucket 1 only the 0.40.
        assert rep["by_bucket"][0]["n"] == 30
        assert rep["by_bucket"][1]["n"] == 30

    def test_total_realized_pnl_matches_sum(self):
        # Total PnL across all BUY rows = sum(conv × fr).
        rows = []
        for _ in range(20):
            rows.append(_row(0.10, 1.0))     # 0.10 each
        for _ in range(20):
            rows.append(_row(0.20, 2.0))     # 0.40 each
        for _ in range(20):
            rows.append(_row(0.30, 3.0))     # 0.90 each
        # Expected: 20*0.10 + 20*0.40 + 20*0.90 = 2 + 8 + 18 = 28
        rep = sps.sizing_report(rows, n_buckets=3)
        assert rep["total_realized_pnl_pct"] == pytest.approx(28.0, abs=1e-3)

    def test_top_bucket_share(self):
        # Top bucket should have a positive share since both buckets
        # contribute positive PnL.
        rows = []
        for _ in range(30):
            rows.append(_row(0.10, 1.0))     # PnL=0.10 each, total=3.0
        for _ in range(30):
            rows.append(_row(0.30, 3.0))     # PnL=0.90 each, total=27.0
        # Grand total = 30.0; top share = 27/30 = 0.9
        rep = sps.sizing_report(rows, n_buckets=2)
        assert rep["top_bucket_share"] == pytest.approx(0.9, abs=1e-3)


# ──────────────────────────── Row filtering ────────────────────────────


class TestRowFilter:

    def test_sell_rows_dropped(self):
        # SELL rows must be excluded (the conviction emission is BUY-only).
        rows = (_uniform_buy_rows(30, 0.20, 5.0)
                + [_row(0.40, 100.0, action="SELL") for _ in range(30)])
        rep = sps.sizing_report(rows, n_buckets=4)
        assert rep["n"] == 30  # only BUY survives

    def test_hold_rows_dropped(self):
        rows = (_uniform_buy_rows(30, 0.20, 5.0)
                + [_row(0.40, 100.0, action="HOLD") for _ in range(30)])
        rep = sps.sizing_report(rows, n_buckets=4)
        assert rep["n"] == 30

    def test_missing_conviction_dropped(self):
        rows = (_uniform_buy_rows(20, 0.20, 5.0)
                + [_row(None, 5.0) for _ in range(20)])
        rep = sps.sizing_report(rows, n_buckets=4)
        assert rep["n"] == 20

    def test_missing_fwd_ret_dropped(self):
        rows = (_uniform_buy_rows(20, 0.20, 5.0)
                + [_row(0.20, None) for _ in range(20)])
        rep = sps.sizing_report(rows, n_buckets=4)
        assert rep["n"] == 20

    def test_non_finite_dropped(self):
        rows = _uniform_buy_rows(20, 0.20, 5.0).copy()
        rows.append(_row(float("nan"), 5.0))
        rows.append(_row(0.20, float("inf")))
        rows.append(_row(float("inf"), 5.0))
        rep = sps.sizing_report(rows, n_buckets=4)
        assert rep["n"] == 20

    def test_out_of_range_conviction_dropped(self):
        # conv must be in [0, 1] — malformed rows like 5.0 (500%) are dropped.
        rows = _uniform_buy_rows(20, 0.20, 5.0).copy()
        rows.append(_row(5.0, 5.0))     # > 1
        rows.append(_row(-0.1, 5.0))    # < 0
        rep = sps.sizing_report(rows, n_buckets=4)
        assert rep["n"] == 20

    def test_bool_conviction_dropped(self):
        # bool subclasses int; _is_finite_number must reject it.
        rows = _uniform_buy_rows(20, 0.20, 5.0).copy()
        rows.append(_row(True, 5.0))   # type: ignore[arg-type]
        rep = sps.sizing_report(rows, n_buckets=4)
        assert rep["n"] == 20

    def test_non_dict_row_dropped(self):
        rows = _uniform_buy_rows(20, 0.20, 5.0).copy()
        rows.append("garbage")           # type: ignore[arg-type]
        rows.append(42)                  # type: ignore[arg-type]
        rows.append(None)                # type: ignore[arg-type]
        rep = sps.sizing_report(rows, n_buckets=4)
        assert rep["n"] == 20

    def test_action_case_normalised(self):
        # "buy" (lowercase) should match BUY case-insensitively.
        rows = [_row(0.10, 1.0, action="buy") for _ in range(20)]
        rep = sps.sizing_report(rows, n_buckets=4)
        assert rep["n"] == 20


# ──────────────────────────── never raises ────────────────────────────


class TestNeverRaises:

    def test_analyze_missing_file(self, tmp_path):
        path = tmp_path / "does-not-exist.jsonl"
        rep = sps.analyze(path)
        assert rep["status"] == "error"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0

    def test_analyze_malformed_lines(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        path.write_text(
            'not json\n'
            '{"action": "BUY", "conviction_pct": 0.2, "forward_return_5d": 5}\n'
            '\n'  # empty
            'garbage{"\n'
        )
        rep = sps.analyze(path)
        # 1 valid row but below MIN_ROWS → INSUFFICIENT_DATA
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 1

    def test_analyze_with_real_corpus(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        records = []
        # Build a corpus that should trigger SIZING_PAYS
        for _ in range(20):
            records.append({"action": "BUY", "conviction_pct": 0.05,
                            "forward_return_5d": 1.0})
        for _ in range(20):
            records.append({"action": "BUY", "conviction_pct": 0.15,
                            "forward_return_5d": 2.0})
        for _ in range(20):
            records.append({"action": "BUY", "conviction_pct": 0.40,
                            "forward_return_5d": 5.0})
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        rep = sps.analyze(path, n_buckets=3)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "SIZING_PAYS"


# ──────────────────────────── CLI exit code ────────────────────────────


class TestCli:

    def _build_outcomes(self, tmp_path, rows):
        path = tmp_path / "outcomes.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return path

    def test_cli_exit_0_on_sizing_pays(self, tmp_path, capsys):
        rows = ([{"action": "BUY", "conviction_pct": 0.05,
                  "forward_return_5d": 1.0} for _ in range(30)]
                + [{"action": "BUY", "conviction_pct": 0.40,
                    "forward_return_5d": 5.0} for _ in range(30)])
        path = self._build_outcomes(tmp_path, rows)
        rc = sps._cli(["--outcomes", str(path), "--buckets", "2", "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        rep = json.loads(out)
        assert rep["verdict"] == "SIZING_PAYS"

    def test_cli_exit_2_on_top_bucket_bleeds(self, tmp_path, capsys):
        rows = []
        # bottom: small wins
        for _ in range(20):
            rows.append({"action": "BUY", "conviction_pct": 0.05,
                         "forward_return_5d": 2.0})
        # mid
        for _ in range(20):
            rows.append({"action": "BUY", "conviction_pct": 0.15,
                         "forward_return_5d": 1.0})
        # top: big losses
        for _ in range(20):
            rows.append({"action": "BUY", "conviction_pct": 0.40,
                         "forward_return_5d": -3.0})
        path = self._build_outcomes(tmp_path, rows)
        rc = sps._cli(["--outcomes", str(path), "--buckets", "3", "--json"])
        assert rc == 2
        out = capsys.readouterr().out
        rep = json.loads(out)
        assert rep["verdict"] == "TOP_BUCKET_BLEEDS"

    def test_cli_exit_2_on_sizing_inverted(self, tmp_path, capsys):
        rows = []
        # bottom: small bets, large wins
        for _ in range(20):
            rows.append({"action": "BUY", "conviction_pct": 0.05,
                         "forward_return_5d": 8.0})    # PnL = 0.40
        # mid
        for _ in range(20):
            rows.append({"action": "BUY", "conviction_pct": 0.15,
                         "forward_return_5d": 2.0})
        # top: big bets, tiny wins
        for _ in range(20):
            rows.append({"action": "BUY", "conviction_pct": 0.40,
                         "forward_return_5d": 0.5})    # PnL = 0.20
        path = self._build_outcomes(tmp_path, rows)
        rc = sps._cli(["--outcomes", str(path), "--buckets", "3", "--json"])
        assert rc == 2
        out = capsys.readouterr().out
        rep = json.loads(out)
        assert rep["verdict"] == "SIZING_INVERTED"

    def test_cli_table_output_runs(self, tmp_path, capsys):
        rows = ([{"action": "BUY", "conviction_pct": 0.20,
                  "forward_return_5d": 1.0} for _ in range(80)])
        path = self._build_outcomes(tmp_path, rows)
        rc = sps._cli(["--outcomes", str(path), "--buckets", "4"])
        # Will be INSUFFICIENT_DATA due to uniform conv → only one bucket
        # has rows; exit 0.
        assert rc == 0
        out = capsys.readouterr().out
        assert "VERDICT:" in out


# ──────────────────────────── analyze return shape ────────────────────────────


class TestAnalyzeShape:

    def test_keys_present(self, tmp_path):
        rep = sps.analyze(tmp_path / "missing.jsonl")
        for k in ("status", "verdict", "n", "n_buckets", "by_bucket",
                  "spread_pct", "total_realized_pnl_pct", "top_bucket_share",
                  "hint"):
            assert k in rep

    def test_default_outcomes_path(self):
        # Don't pass a path; the function should resolve OUTCOMES_DEFAULT.
        # Whether that file exists or not, the call must not raise and must
        # return a dict with the expected shape.
        rep = sps.analyze()
        assert isinstance(rep, dict)
        assert "verdict" in rep
