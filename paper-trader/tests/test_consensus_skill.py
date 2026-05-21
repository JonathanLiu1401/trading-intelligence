"""Tests for paper_trader.ml.consensus_skill.

The diagnostic groups outcome rows by (sim_date, ticker, action),
buckets each group by its distinct-run-count (1, 2, 3+), and reports
whether higher consensus predicts a better realized 5d forward return.

Verdict ladder (test-locked, exact-value):

| Verdict | Trigger |
|---|---|
| INSUFFICIENT_DATA | < MIN_ROWS lone-bucket rows OR < MIN_GROUPS multi groups OR no top bucket with BUCKET_MIN_ROWS rows |
| INVERTED          | lone - top > REVERSAL_MIN_PCT  (consensus is anti-predictive) |
| CONSENSUS_EDGE    | top - lone ≥ EDGE_MIN_PCT |
| WEAK_EDGE         | spread positive but below EDGE_MIN_PCT |
| NO_EDGE           | |spread| ≤ FLAT_BAND_PCT |
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_trader.ml import consensus_skill as cs


def _row(run_id: int, sd: str, ticker: str, action: str,
         fr5: float | None) -> dict:
    return {
        "run_id": run_id,
        "sim_date": sd,
        "ticker": ticker,
        "action": action,
        "forward_return_5d": fr5,
    }


def _many_lone_rows(n: int, fr_mean: float) -> list[dict]:
    """Build n DISTINCT-key lone-bucket rows whose mean fr_5d ≈ fr_mean."""
    return [_row(1000 + i, f"2020-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                 f"TKR{i:03d}", "BUY", fr_mean)
            for i in range(n)]


def _agreeing_rows(rid_count: int, sd: str, ticker: str, action: str,
                   fr5: float) -> list[dict]:
    """``rid_count`` distinct-run rows that agree on the SAME key."""
    return [_row(2000 + i, sd, ticker, action, fr5)
            for i in range(rid_count)]


class TestBucketLabel:
    """Label assignment is exact (the verdict reads it)."""

    def test_one_distinct_run_is_bucket_1(self):
        assert cs._bucket_label(1) == "1"

    def test_two_distinct_runs_is_bucket_2(self):
        assert cs._bucket_label(2) == "2"

    def test_three_runs_is_bucket_3plus(self):
        assert cs._bucket_label(3) == "3+"

    def test_ten_runs_is_bucket_3plus(self):
        assert cs._bucket_label(10) == "3+"

    def test_zero_or_negative_falls_into_lone(self):
        # Defensive — shouldn't happen but the bucketer must not crash.
        assert cs._bucket_label(0) == "1"


class TestConsensusReportVerdicts:
    def test_insufficient_when_no_rows(self):
        rep = cs.consensus_report([])
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["spread_pct"] is None
        assert rep["top_bucket"] is None

    def test_insufficient_when_too_few_lone_rows(self):
        # Only 50 lone rows (< MIN_ROWS=100) — verdict must degrade.
        rows = _many_lone_rows(50, 1.0)
        # Add plenty of multi-run groups so the OTHER threshold passes.
        for i in range(40):
            rows.extend(_agreeing_rows(2, f"2021-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"X{i}", "BUY", 3.0))
        rep = cs.consensus_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_insufficient_when_too_few_multi_groups(self):
        # Plenty of lone rows but only 10 multi-run groups (< MIN_GROUPS=20).
        rows = _many_lone_rows(200, 1.0)
        for i in range(10):
            rows.extend(_agreeing_rows(2, f"2021-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"X{i}", "BUY", 3.0))
        rep = cs.consensus_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_consensus_edge_when_top_beats_lone(self):
        # Lone bucket mean = +1.0%; 2-bucket mean = +4.0% (spread = +3pp > 0.5).
        rows = _many_lone_rows(200, 1.0)
        # 25 multi-run groups, each 2 agreeing rows → 50 bucket-2 rows, > BUCKET_MIN_ROWS.
        for i in range(25):
            rows.extend(_agreeing_rows(2, f"2021-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"X{i}", "BUY", 4.0))
        rep = cs.consensus_report(rows)
        assert rep["verdict"] == "CONSENSUS_EDGE"
        assert rep["top_bucket"] == "2"
        assert rep["spread_pct"] is not None and rep["spread_pct"] > 0.5

    def test_inverted_when_lone_beats_consensus_by_margin(self):
        # Lone mean = +5%, multi-run mean = +1% → spread = -4pp → INVERTED.
        rows = _many_lone_rows(200, 5.0)
        for i in range(25):
            rows.extend(_agreeing_rows(2, f"2022-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"Y{i}", "BUY", 1.0))
        rep = cs.consensus_report(rows)
        assert rep["verdict"] == "INVERTED"
        assert rep["spread_pct"] is not None and rep["spread_pct"] < -0.5

    def test_no_edge_in_flat_band(self):
        # Lone = +2.0%, multi = +2.1% → spread = +0.1pp, inside ±0.25pp band.
        rows = _many_lone_rows(200, 2.0)
        for i in range(25):
            rows.extend(_agreeing_rows(2, f"2023-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"Z{i}", "BUY", 2.1))
        rep = cs.consensus_report(rows)
        assert rep["verdict"] == "NO_EDGE"
        assert rep["spread_pct"] is not None and abs(rep["spread_pct"]) <= 0.25

    def test_weak_edge_between_flat_and_strong(self):
        # spread = +0.4pp → above flat band (0.25) but below edge (0.5) → WEAK_EDGE.
        rows = _many_lone_rows(200, 1.0)
        for i in range(25):
            rows.extend(_agreeing_rows(2, f"2024-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"W{i}", "BUY", 1.4))
        rep = cs.consensus_report(rows)
        assert rep["verdict"] == "WEAK_EDGE"


class TestBucketStatistics:
    def test_lone_bucket_n_equals_lone_row_count(self):
        rows = _many_lone_rows(150, 1.0)
        # Add some multi-run groups so MIN_GROUPS passes for the verdict's
        # sake; the lone-bucket n remains 150.
        for i in range(25):
            rows.extend(_agreeing_rows(2, f"2025-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"V{i}", "BUY", 2.0))
        rep = cs.consensus_report(rows)
        assert rep["by_bucket"]["1"]["n"] == 150
        assert rep["by_bucket"]["1"]["mean"] == pytest.approx(1.0)

    def test_bucket_2_counts_each_agreeing_row_separately(self):
        # 25 groups of 2 agreeing runs each → 50 rows in bucket "2".
        rows = _many_lone_rows(150, 1.0)
        for i in range(25):
            rows.extend(_agreeing_rows(2, f"2025-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"V{i}", "BUY", 2.0))
        rep = cs.consensus_report(rows)
        assert rep["by_bucket"]["2"]["n"] == 50
        assert rep["by_bucket"]["2"]["n_groups"] == 25

    def test_bucket_3plus_aggregates_3_and_more_run_groups(self):
        rows = _many_lone_rows(150, 1.0)
        # Mix: 25 two-run groups + 7 three-run groups + 3 four-run groups
        for i in range(25):
            rows.extend(_agreeing_rows(2, f"2026-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"M{i}", "BUY", 2.0))
        for i in range(7):
            rows.extend(_agreeing_rows(3, f"2026-04-{i+1:02d}",
                                       f"T{i}", "BUY", 4.0))
        for i in range(3):
            rows.extend(_agreeing_rows(4, f"2026-05-{i+1:02d}",
                                       f"F{i}", "BUY", 5.0))
        rep = cs.consensus_report(rows)
        # 7*3 + 3*4 = 33 rows in bucket "3+"
        assert rep["by_bucket"]["3+"]["n"] == 33
        # 7 + 3 = 10 groups
        assert rep["by_bucket"]["3+"]["n_groups"] == 10
        # Verdict prefers "3+" because it has BUCKET_MIN_ROWS (20) rows AND
        # is the higher-consensus bucket.
        assert rep["top_bucket"] == "3+"


class TestRowFiltering:
    def test_drops_non_dict_rows(self):
        rows = _many_lone_rows(150, 1.0)
        # Add 25 multi-run groups for verdict viability.
        for i in range(25):
            rows.extend(_agreeing_rows(2, f"2020-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"X{i}", "BUY", 2.0))
        # Inject garbage that must NOT be counted.
        rows.extend([None, "string-row", 42, ["list-row"]])
        rep = cs.consensus_report(rows)
        # Lone n unchanged — the 4 junk rows were dropped.
        assert rep["by_bucket"]["1"]["n"] == 150

    def test_drops_rows_missing_required_fields(self):
        rows = _many_lone_rows(150, 1.0)
        for i in range(25):
            rows.extend(_agreeing_rows(2, f"2020-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"X{i}", "BUY", 2.0))
        # Various malformed rows — must all be silently dropped.
        rows.extend([
            {"run_id": 1, "ticker": "X", "action": "BUY",
             "forward_return_5d": 1.0},                       # missing sim_date
            {"sim_date": "2020-01-01", "ticker": "X",
             "action": "BUY", "forward_return_5d": 1.0},      # missing run_id
            {"run_id": 1, "sim_date": "2020-01-01",
             "ticker": "", "action": "BUY",
             "forward_return_5d": 1.0},                       # empty ticker
            {"run_id": 1, "sim_date": "2020-01-01",
             "ticker": "X", "action": "BUY",
             "forward_return_5d": None},                      # missing target
        ])
        rep = cs.consensus_report(rows)
        assert rep["by_bucket"]["1"]["n"] == 150

    def test_drops_rows_with_non_finite_target(self):
        rows = _many_lone_rows(150, 1.0)
        for i in range(25):
            rows.extend(_agreeing_rows(2, f"2020-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"X{i}", "BUY", 2.0))
        rows.extend([
            {"run_id": 999, "sim_date": "2020-06-15", "ticker": "NAN",
             "action": "BUY", "forward_return_5d": float("nan")},
            {"run_id": 999, "sim_date": "2020-06-15", "ticker": "INF",
             "action": "BUY", "forward_return_5d": float("inf")},
        ])
        rep = cs.consensus_report(rows)
        # Lone n still 150 — neither NaN nor inf was counted.
        assert rep["by_bucket"]["1"]["n"] == 150

    def test_action_normalised_to_upper_so_BUY_and_buy_collide(self):
        # Same key in mixed case should still register as agreeing (action
        # is upper-cased before keying). Two rows on the same date/ticker
        # with mixed-case "BUY" vs "buy" must end up in bucket 2, not 1.
        rows = _many_lone_rows(150, 1.0)
        for i in range(25):
            rows.extend(_agreeing_rows(2, f"2020-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"X{i}", "BUY", 2.0))
        rows.append(_row(50, "2030-01-15", "MIXED", "BUY", 7.0))
        rows.append(_row(51, "2030-01-15", "MIXED", "buy", 7.0))
        rep = cs.consensus_report(rows)
        # That extra agreeing pair adds 2 to bucket "2", 1 to n_groups.
        assert rep["by_bucket"]["2"]["n"] == 25 * 2 + 2
        assert rep["by_bucket"]["2"]["n_groups"] == 25 + 1


class TestLoadOutcomes:
    def test_missing_file_returns_empty(self, tmp_path):
        path = tmp_path / "nope.jsonl"
        assert cs._load_outcomes(path) == []

    def test_malformed_lines_are_skipped(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        path.write_text(
            "\n".join([
                json.dumps({"run_id": 1, "sim_date": "2020-01-01",
                            "ticker": "X", "action": "BUY",
                            "forward_return_5d": 1.0}),
                "not-json",
                json.dumps([1, 2, 3]),  # list, not dict
                json.dumps({"run_id": 2, "sim_date": "2020-01-02",
                            "ticker": "Y", "action": "BUY",
                            "forward_return_5d": 2.0}),
            ]) + "\n"
        )
        rows = cs._load_outcomes(path)
        # Only the two valid dict rows survive.
        assert len(rows) == 2
        assert {r["ticker"] for r in rows} == {"X", "Y"}


class TestAnalyzeEntryNeverRaises:
    def test_missing_file_path_returns_safe_dict(self, tmp_path):
        rep = cs.analyze(tmp_path / "does-not-exist.jsonl")
        assert rep["status"] == "error"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        # JSON-safe structure preserved
        assert rep["by_bucket"]["1"]["n"] == 0
        assert rep["spread_pct"] is None

    def test_corrupt_file_does_not_raise(self, tmp_path):
        path = tmp_path / "outcomes.jsonl"
        path.write_text("\x00\x01garbage")  # binary garbage
        rep = cs.analyze(path)
        # File parses as 0 valid rows → INSUFFICIENT_DATA, no exception.
        assert rep["verdict"] == "INSUFFICIENT_DATA"


class TestCli:
    def test_cli_zero_exit_on_consensus_edge(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "outcomes.jsonl"
        rows = _many_lone_rows(200, 1.0)
        for i in range(25):
            rows.extend(_agreeing_rows(2, f"2021-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"X{i}", "BUY", 5.0))
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        rc = cs._cli(["--outcomes", str(path), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        rep = json.loads(out)
        assert rep["verdict"] == "CONSENSUS_EDGE"

    def test_cli_exit_2_on_inverted(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "outcomes.jsonl"
        rows = _many_lone_rows(200, 5.0)
        for i in range(25):
            rows.extend(_agreeing_rows(2, f"2021-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"Y{i}", "BUY", 0.5))
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        rc = cs._cli(["--outcomes", str(path), "--json"])
        # Exit code 2 only on INVERTED (the quant-decisive harmful state).
        assert rc == 2

    def test_cli_zero_exit_on_no_edge(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "outcomes.jsonl"
        rows = _many_lone_rows(200, 2.0)
        for i in range(25):
            rows.extend(_agreeing_rows(2, f"2024-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                                       f"Z{i}", "BUY", 2.1))
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        rc = cs._cli(["--outcomes", str(path), "--json"])
        assert rc == 0
