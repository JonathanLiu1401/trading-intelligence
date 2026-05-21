"""Exact-value verdict tests for gate_capture_coverage.

The verdict ladder is the diagnostic's whole point; every threshold below
must trigger the documented verdict on hand-engineered synthetic data so a
future refactor cannot silently change the bucket boundaries.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_trader.ml import gate_capture_coverage as gcc


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _buy(run_id: int, captured: bool, off_dist: bool = False,
         sim_date: str = "2024-01-01", ticker: str = "NVDA") -> dict:
    pred: float | None = -3.0 if captured else None
    if captured and off_dist:
        pred = -50.0
    return {
        "run_id": run_id, "sim_date": sim_date, "ticker": ticker,
        "action": "BUY",
        "gate_scorer_pred": pred,
        "gate_off_dist": off_dist if captured else None,
    }


def _sell(run_id: int, gate_scorer_pred=None,
          sim_date: str = "2024-01-01", ticker: str = "NVDA") -> dict:
    return {
        "run_id": run_id, "sim_date": sim_date, "ticker": ticker,
        "action": "SELL", "gate_scorer_pred": gate_scorer_pred,
        "gate_off_dist": None,
    }


class TestInsufficientData:
    def test_zero_buys_gives_insufficient(self, tmp_path):
        p = _write_jsonl(tmp_path / "o.jsonl", [
            _sell(i) for i in range(1, 100)
        ])
        rep = gcc.analyze(p)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["total_buys"] == 0

    def test_fifty_nine_buys_is_insufficient(self, tmp_path):
        # MIN_BUY_ROWS is 60 — exactly 59 must still read INSUFFICIENT.
        rows = [_buy(i, captured=True) for i in range(1, 60)]
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["total_buys"] == 59

    def test_sixty_buys_crosses_into_active_verdict(self, tmp_path):
        rows = [_buy(i, captured=True) for i in range(1, 61)]
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        # 60 BUYs all captured → 100% → HEALTHY (no trend split is possible
        # with all equal, oldest==newest==100%, trend_pp=0 → not DEGRADING)
        assert rep["verdict"] == "GATE_CAPTURE_HEALTHY"
        assert rep["total_buys"] == 60


class TestSchemaViolation:
    def test_one_sell_with_capture_is_schema_violation(self, tmp_path):
        rows = [_buy(i, captured=True) for i in range(1, 100)]
        rows.append(_sell(999, gate_scorer_pred=3.2))
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        assert rep["verdict"] == "SCHEMA_VIOLATION"
        assert rep["schema_violation_count"] == 1
        assert rep["schema_violations"][0]["gate_scorer_pred"] == 3.2

    def test_schema_violation_always_wins_over_quant_verdict(self, tmp_path):
        # Even with healthy capture, a single SELL+capture row trumps it.
        rows = [_buy(i, captured=True) for i in range(1, 200)]
        rows.append(_sell(999, gate_scorer_pred=-50.0))
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        assert rep["verdict"] == "SCHEMA_VIOLATION"
        # The healthy-capture metrics are still reported so the operator
        # can see what the quant verdict WOULD have been.
        assert rep["overall_buy_capture_pct"] == 100.0

    def test_violations_capped_at_ten(self, tmp_path):
        rows = [_buy(i, captured=True) for i in range(1, 100)]
        # 15 schema violators
        for i in range(15):
            rows.append(_sell(900 + i, gate_scorer_pred=float(i)))
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        assert rep["schema_violation_count"] == 15
        assert len(rep["schema_violations"]) == 10  # cap


class TestDark:
    def test_all_uncaptured_is_dark(self, tmp_path):
        rows = [_buy(i, captured=False) for i in range(1, 100)]
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        assert rep["verdict"] == "GATE_CAPTURE_DARK"
        assert rep["overall_buy_capture_pct"] == 0.0

    def test_just_under_dark_threshold(self, tmp_path):
        # DARK_PCT is 5.0%. 100 rows, 4 captured → 4% < 5% → DARK
        rows = [_buy(i, captured=(i <= 4)) for i in range(1, 101)]
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        assert rep["overall_buy_capture_pct"] == 4.0
        assert rep["verdict"] == "GATE_CAPTURE_DARK"

    def test_just_at_dark_boundary_is_not_dark(self, tmp_path):
        # 100 rows, 5 captured → 5.0% — not < 5.0, so not DARK.
        # 5/100 captured all at low run_ids → newest q quartile (n=25) is 0%,
        # oldest q (n=25) carries all 5 → 20% → trend = 0-20 = -20pp →
        # DEGRADING (>= -TREND_PP).
        rows = [_buy(i, captured=(i <= 5)) for i in range(1, 101)]
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        assert rep["overall_buy_capture_pct"] == 5.0
        assert rep["verdict"] == "GATE_CAPTURE_DEGRADING"


class TestTrend:
    def test_degrading_when_recent_drops(self, tmp_path):
        # 100 rows. Oldest 25 all captured (100%), newest 25 all uncaptured (0%)
        # → trend = -100pp → DEGRADING.
        rows = []
        for i in range(1, 101):
            captured = i <= 75  # first 75 captured, last 25 not
            rows.append(_buy(i, captured=captured))
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        assert rep["oldest_quartile_buy_capture_pct"] == 100.0
        assert rep["newest_quartile_buy_capture_pct"] == 0.0
        assert rep["trend_pp"] == -100.0
        assert rep["verdict"] == "GATE_CAPTURE_DEGRADING"

    def test_improving_when_recent_rises(self, tmp_path):
        # First 25 uncaptured (oldest), last 75 captured. Oldest q=0%, newest q=100%
        rows = []
        for i in range(1, 101):
            captured = i > 25
            rows.append(_buy(i, captured=captured))
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        assert rep["oldest_quartile_buy_capture_pct"] == 0.0
        assert rep["newest_quartile_buy_capture_pct"] == 100.0
        assert rep["trend_pp"] == 100.0
        assert rep["verdict"] == "GATE_CAPTURE_IMPROVING"

    def test_trend_below_threshold_falls_through_to_partial_or_healthy(
            self, tmp_path):
        # 100 rows. Oldest q (25) has 20/25 captured = 80%, newest q (25)
        # has 24/25 = 96% → trend +16pp < TREND_PP=20 → falls through.
        # Overall: (20 + 50 (middle) + 24) / 100 = need to set middle too.
        # Easier: 100 rows, 90 captured, captured indices chosen so each
        # quartile holds exactly the same captured count.
        rows = []
        for i in range(1, 101):
            # Captured every row except 1, 26, 51, 76 (one per quartile)
            captured = i not in (1, 26, 51, 76)
            rows.append(_buy(i, captured=captured))
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        assert rep["overall_buy_capture_pct"] == 96.0  # 96/100
        assert rep["trend_pp"] == 0.0
        # Overall >= 90% AND no degrading trend → HEALTHY.
        assert rep["verdict"] == "GATE_CAPTURE_HEALTHY"

    def test_partial_when_mid_capture_no_trend(self, tmp_path):
        # 100 rows. Every 2nd captured → 50% capture, no trend.
        rows = []
        for i in range(1, 101):
            captured = (i % 2 == 0)
            rows.append(_buy(i, captured=captured))
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        assert rep["overall_buy_capture_pct"] == 50.0
        # Each quartile (25 rows) holds either 12 or 13 captured. Trend
        # is at most ±4pp — well under TREND_PP=20.
        assert abs(rep["trend_pp"]) <= 4.0
        assert rep["verdict"] == "GATE_CAPTURE_PARTIAL"


class TestHealthy:
    def test_all_captured_is_healthy(self, tmp_path):
        rows = [_buy(i, captured=True) for i in range(1, 200)]
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        assert rep["overall_buy_capture_pct"] == 100.0
        assert rep["verdict"] == "GATE_CAPTURE_HEALTHY"


class TestOffDistribution:
    def test_off_dist_count_and_pct(self, tmp_path):
        # 100 BUYs all captured; 5 of them off-dist.
        rows = []
        for i in range(1, 101):
            rows.append(_buy(i, captured=True, off_dist=(i <= 5)))
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        assert rep["buy_off_dist_count"] == 5
        assert rep["buy_off_dist_pct_of_captured"] == 5.0

    def test_off_dist_excludes_uncaptured(self, tmp_path):
        # Uncaptured rows have off_dist=None and must not be counted.
        rows = [_buy(i, captured=False) for i in range(1, 100)]
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        assert rep["buy_off_dist_count"] == 0
        assert rep["buy_off_dist_pct_of_captured"] == 0.0


class TestRobustness:
    def test_missing_file_returns_insufficient(self, tmp_path):
        rep = gcc.analyze(tmp_path / "does_not_exist.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["total_rows"] == 0
        assert rep["total_buys"] == 0

    def test_malformed_lines_dropped(self, tmp_path):
        p = tmp_path / "o.jsonl"
        p.write_text(
            json.dumps(_buy(1, captured=True)) + "\n"
            + "not valid json\n"
            + json.dumps(_buy(2, captured=True)) + "\n"
            + "{\"truncated\": " + "\n"
        )
        rep = gcc.analyze(p)
        assert rep["total_rows"] == 2  # malformed dropped silently
        assert rep["total_buys"] == 2

    def test_non_dict_rows_dropped(self, tmp_path):
        p = tmp_path / "o.jsonl"
        p.write_text(
            json.dumps(_buy(1, captured=True)) + "\n"
            + json.dumps([1, 2, 3]) + "\n"  # list, not dict
            + json.dumps("string row") + "\n"
            + json.dumps(42) + "\n"
            + json.dumps(_buy(2, captured=True)) + "\n"
        )
        rep = gcc.analyze(p)
        assert rep["total_rows"] == 2  # only the 2 dict rows
        assert rep["total_buys"] == 2

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "o.jsonl"
        p.write_text(
            "\n\n"
            + json.dumps(_buy(1, captured=True)) + "\n"
            + "\n"
            + json.dumps(_buy(2, captured=True)) + "\n"
        )
        rep = gcc.analyze(p)
        assert rep["total_rows"] == 2

    def test_unknown_action_ignored(self, tmp_path):
        rows = [_buy(i, captured=True) for i in range(1, 100)]
        rows.append({"action": "HOLD", "gate_scorer_pred": 5.0})  # ignored
        rows.append({"action": "MAYBE_BUY", "gate_scorer_pred": -5.0})  # ignored
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        assert rep["total_buys"] == 99
        assert rep["total_sells"] == 0

    def test_row_without_run_id_excluded_from_trend_split(self, tmp_path):
        # 80 rows have run_id, 20 don't. The 80 are split into quartiles.
        rows = [_buy(i, captured=True) for i in range(1, 81)]
        for i in range(20):
            r = _buy(i, captured=False)
            del r["run_id"]
            rows.append(r)
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rep = gcc.analyze(p)
        # 80 captured / 100 total = 80%
        assert rep["overall_buy_capture_pct"] == 80.0
        # Trend split is over the 80 with run_id (all captured) → 100% in both
        assert rep["quartile_n"] == 20  # 80//4
        assert rep["oldest_quartile_buy_capture_pct"] == 100.0
        assert rep["newest_quartile_buy_capture_pct"] == 100.0


class TestCli:
    def test_cli_exit_0_on_healthy(self, tmp_path, capsys):
        rows = [_buy(i, captured=True) for i in range(1, 200)]
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rc = gcc.main(["--outcomes", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "GATE_CAPTURE_HEALTHY" in out

    def test_cli_exit_0_on_insufficient(self, tmp_path):
        p = _write_jsonl(tmp_path / "o.jsonl", [_buy(1, captured=True)])
        rc = gcc.main(["--outcomes", str(p)])
        assert rc == 0  # gap, not harm

    def test_cli_exit_2_on_dark(self, tmp_path):
        rows = [_buy(i, captured=False) for i in range(1, 100)]
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rc = gcc.main(["--outcomes", str(p)])
        assert rc == 2

    def test_cli_exit_2_on_degrading(self, tmp_path):
        rows = []
        for i in range(1, 101):
            rows.append(_buy(i, captured=(i <= 75)))
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rc = gcc.main(["--outcomes", str(p)])
        assert rc == 2

    def test_cli_exit_2_on_schema_violation(self, tmp_path):
        rows = [_buy(i, captured=True) for i in range(1, 100)]
        rows.append(_sell(999, gate_scorer_pred=3.2))
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rc = gcc.main(["--outcomes", str(p)])
        assert rc == 2

    def test_cli_exit_0_on_improving(self, tmp_path):
        rows = []
        for i in range(1, 101):
            rows.append(_buy(i, captured=(i > 25)))
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rc = gcc.main(["--outcomes", str(p)])
        assert rc == 0

    def test_cli_json_mode(self, tmp_path, capsys):
        rows = [_buy(i, captured=True) for i in range(1, 100)]
        p = _write_jsonl(tmp_path / "o.jsonl", rows)
        rc = gcc.main(["--outcomes", str(p), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        # Stable schema — verdict key and the load-bearing counters present
        assert parsed["verdict"] == "GATE_CAPTURE_HEALTHY"
        assert "buy_captured" in parsed
        assert "overall_buy_capture_pct" in parsed
        assert "schema_violations" in parsed
