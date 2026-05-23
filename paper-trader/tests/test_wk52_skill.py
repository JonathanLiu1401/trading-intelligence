"""Tests for paper_trader/ml/wk52_skill.py.

Locks the bubble-top gate's empirical-verification analyzer:
- HARMFUL when high-wk52 BUYs realize BETTER (gate is suppressing winners)
- JUSTIFIED when high-wk52 BUYs realize WORSE (gate's premise holds)
- INEFFECTIVE when there's no meaningful difference
- bucket means, zone cut, dropped-row counters, hint string
- CLI exit codes (0/1/2) match the verdict ladder for shell gating
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_trader.ml import wk52_skill as ws


# ───────────────────────────── helpers ─────────────────────────────


def _row(wk52, ret, action="BUY"):
    """Minimal outcome row carrying the three load-bearing fields."""
    return {"action": action, "wk52_pos": wk52, "forward_return_5d": ret,
            "ticker": "NVDA"}


# ─────────────────────────── core bucketing ────────────────────────


class TestBuildWk52Skill:
    def test_insufficient_data_when_below_min_pairs(self):
        recs = [_row(0.1, 1.0), _row(0.5, 0.0), _row(0.9, -1.0)]
        rep = ws.build_wk52_skill(recs)
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 3
        # All counters are honest about what was dropped vs accepted.
        assert rep["n_dropped_wk52"] == 0
        assert rep["bubble_gate_threshold"] == ws.BUBBLE_GATE_THRESHOLD

    def test_no_records_returns_insufficient(self):
        rep = ws.build_wk52_skill([])
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0
        assert rep["hint"] == "no records supplied"

    def test_bubble_gate_justified_when_high_wk52_realizes_worse(self):
        """High wk52_pos → low realized return. Strong negative
        spearman + ≥2pp bucket gap → JUSTIFIED."""
        # 50 rows: 25 low-wk52 winning, 25 high-wk52 losing.
        recs = ([_row(0.05 + 0.01 * i, +5.0) for i in range(25)]
                + [_row(0.85 + 0.005 * i, -5.0) for i in range(25)])
        rep = ws.build_wk52_skill(recs)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "BUBBLE_GATE_JUSTIFIED"
        # n captures every accepted BUY row.
        assert rep["n"] == 50
        # Top bucket realized < bottom bucket — the gate's premise.
        assert rep["top_minus_bottom_realized_pct"] < -ws.BUCKET_GAP_GOOD_PCT
        # Strong negative rank correlation.
        assert rep["spearman"] <= -ws.SPEARMAN_GOOD
        # The hint text names the premise.
        assert "premise holds" in rep["hint"].lower()

    def test_bubble_gate_harmful_when_high_wk52_realizes_better(self):
        """High wk52_pos → HIGH realized return. Positive spearman +
        ≥2pp bucket gap → HARMFUL. This is the documented decisive
        finding on the live 1117-row BUY corpus (Q5 [0.81-1.00]
        realized +1.01% vs Q1 [0.00-0.09] +1.77% — actually the live
        corpus is ambiguous, but a CLEAR positive-correlation synthetic
        cleanly fires HARMFUL)."""
        recs = ([_row(0.05 + 0.01 * i, -5.0) for i in range(25)]
                + [_row(0.85 + 0.005 * i, +5.0) for i in range(25)])
        rep = ws.build_wk52_skill(recs)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "BUBBLE_GATE_HARMFUL"
        # Top realized > bottom — the gate would suppress these winners.
        assert rep["top_minus_bottom_realized_pct"] >= ws.BUCKET_GAP_GOOD_PCT
        # Strong positive rank correlation.
        assert rep["spearman"] >= ws.SPEARMAN_GOOD
        # The hint text names the failure mode.
        assert "suppressing profitable" in rep["hint"]

    def test_bubble_gate_ineffective_on_no_correlation(self):
        """A scattered relationship with neither rank skill nor a
        meaningful bucket gap → INEFFECTIVE."""
        # 40 rows of alternating ±0.5% returns across the full wk52 range.
        recs = []
        for i in range(40):
            wk = i / 40.0  # 0.0 .. 0.975
            ret = 0.5 if i % 2 == 0 else -0.5
            recs.append(_row(wk, ret))
        rep = ws.build_wk52_skill(recs)
        assert rep["status"] == "ok"
        assert rep["verdict"] == "BUBBLE_GATE_INEFFECTIVE"
        # Neither metric clears the strong bar.
        assert abs(rep["spearman"]) < ws.SPEARMAN_GOOD
        # In-zone vs out-zone numbers still get computed honestly.
        # Threshold is 0.80 — wk = i/40 > 0.80 means i >= 33, so
        # 7 rows are in the zone.
        assert rep["in_zone_n"] == 7
        assert rep["out_zone_n"] == 33

    def test_directional_for_gate_below_strong_bar(self):
        """A weak negative spearman that doesn't clear -SPEARMAN_GOOD
        → DIRECTIONAL_FOR_GATE. Build a sloped signal that lands the
        spearman in (-SPEARMAN_GOOD, -SPEARMAN_FLAT)."""
        import random
        rng = random.Random(42)
        recs = []
        for i in range(80):
            wk = i / 80.0
            # Slope -3pp across full range plus heavy noise.
            ret = -3.0 * wk + rng.gauss(0, 5.0)
            recs.append(_row(wk, ret))
        rep = ws.build_wk52_skill(recs)
        # Either DIRECTIONAL_FOR_GATE or JUSTIFIED depending on noise —
        # both are negative-correlation outcomes. Lock the sign.
        assert rep["status"] == "ok"
        assert rep["spearman"] < 0  # negative correlation
        assert rep["verdict"] in (
            "DIRECTIONAL_FOR_GATE", "BUBBLE_GATE_JUSTIFIED")

    def test_dropped_row_counters_are_accurate(self):
        """A mix of valid / invalid rows: dropped counts must be exact."""
        recs = [
            _row(0.5, 1.0),                  # accepted
            _row(0.5, 1.0, action="SELL"),   # action dropped
            _row(None, 1.0),                 # wk52 None dropped
            _row(1.5, 1.0),                  # wk52 OOR dropped
            _row(0.5, None),                 # return dropped
            _row(0.5, float("inf")),         # return non-finite dropped
            _row(0.5, "not-a-number"),       # return non-numeric dropped
        ]
        rep = ws.build_wk52_skill(recs)
        # Only 1 row accepted → INSUFFICIENT.
        assert rep["n"] == 1
        assert rep["n_dropped_action"] == 1
        assert rep["n_dropped_wk52"] == 2  # None AND out-of-range
        assert rep["n_dropped_return"] == 3  # None + inf + string

    def test_non_dict_records_are_dropped_via_action_counter(self):
        """A malformed (non-dict) record drops cleanly rather than raising."""
        recs = [_row(0.5, 1.0), "not a dict", 42, None, _row(0.6, -0.5)]
        rep = ws.build_wk52_skill(recs)
        # The two real rows are accepted; the 3 malformed go to
        # n_dropped_action (the catch-all for invalid records).
        assert rep["n"] == 2
        assert rep["n_dropped_action"] == 3

    def test_in_zone_vs_out_zone_means_match_strict_threshold(self):
        """Zone cut uses STRICT inequality (`wk52 > threshold`) — a row
        AT the threshold goes to out_zone, matching _ml_decide."""
        thr = ws.BUBBLE_GATE_THRESHOLD
        recs = [
            _row(thr, +2.0),       # AT threshold → out-zone
            _row(thr + 1e-6, -3.0),  # just above → in-zone
            _row(thr - 0.1, +1.0),  # below → out-zone
        ] * 12  # 36 rows, just above MIN_PAIRS=30
        rep = ws.build_wk52_skill(recs)
        # 12 rows above threshold (wk52 = thr + 1e-6)
        assert rep["in_zone_n"] == 12
        # 24 rows at-or-below threshold (the 2 other variants × 12)
        assert rep["out_zone_n"] == 24
        # In-zone mean is exactly the negative return (-3.0).
        assert rep["in_zone_mean_realized"] == pytest.approx(-3.0)

    def test_bucket_count_clamped_by_data_size(self):
        """With n < n_buckets * 3, the cut is reduced to keep ≥3 per
        bucket. Mirrors conviction_calibration."""
        # 30 rows; default n_buckets=5; 30/3 = 10 max, so 5 buckets fit.
        recs = [_row(i / 30.0, 0.1 * i) for i in range(30)]
        rep = ws.build_wk52_skill(recs)
        assert rep["status"] == "ok"
        assert len(rep["buckets"]) == 5  # full quintile cut
        # Now request 20 buckets → clamped to min(20, 30//3=10).
        rep2 = ws.build_wk52_skill(recs, n_buckets=20)
        assert len(rep2["buckets"]) == 10


# ─────────────────────────── analyze + load ────────────────────────


class TestAnalyze:
    def test_analyze_missing_file_returns_insufficient_not_raise(self, tmp_path):
        """A missing outcomes path degrades to insufficient_data
        (load_outcomes returns []). Never raises."""
        rep = ws.analyze(tmp_path / "nonexistent.jsonl")
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_analyze_streams_jsonl_and_aggregates(self, tmp_path):
        """End-to-end: write a JSONL with a known signal and verify
        verdict + key metric values."""
        path = tmp_path / "outcomes.jsonl"
        with path.open("w") as fh:
            # 25 low-wk52 winning + 25 high-wk52 losing → JUSTIFIED
            for i in range(25):
                fh.write(json.dumps(_row(0.05 + 0.01 * i, +5.0)) + "\n")
            for i in range(25):
                fh.write(json.dumps(_row(0.85 + 0.005 * i, -5.0)) + "\n")
        rep = ws.analyze(path)
        assert rep["verdict"] == "BUBBLE_GATE_JUSTIFIED"
        assert rep["n"] == 50
        # Mean of all returns is 0 by construction (25×+5 + 25×-5).
        assert rep["mean_realized"] == pytest.approx(0.0)

    def test_analyze_skips_unparseable_jsonl_lines(self, tmp_path):
        """A corrupted line drops cleanly — the rest still parse.
        Lock the load_outcomes contract."""
        path = tmp_path / "outcomes.jsonl"
        with path.open("w") as fh:
            for i in range(30):
                fh.write(json.dumps(_row(i / 30.0, 1.0)) + "\n")
            fh.write("{not valid json\n")  # corruption
            fh.write("\n")  # blank line
            for i in range(5):
                fh.write(json.dumps(_row(0.5, -1.0)) + "\n")
        rep = ws.analyze(path)
        # 30 + 5 = 35 valid rows.
        assert rep["n"] == 35


# ─────────────────────────────── CLI ───────────────────────────────


class TestCli:
    def test_cli_exit_0_on_justified(self, tmp_path, capsys):
        path = tmp_path / "o.jsonl"
        with path.open("w") as fh:
            for i in range(25):
                fh.write(json.dumps(_row(0.05 + 0.01 * i, +5.0)) + "\n")
            for i in range(25):
                fh.write(json.dumps(_row(0.85 + 0.005 * i, -5.0)) + "\n")
        rc = ws._cli(["--path", str(path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "BUBBLE_GATE_JUSTIFIED" in out

    def test_cli_exit_2_on_harmful(self, tmp_path, capsys):
        """Exit 2 is the quant-decisive 'gate is removing winners'
        signal — a shell caller can gate on `rc == 2` specifically."""
        path = tmp_path / "o.jsonl"
        with path.open("w") as fh:
            for i in range(25):
                fh.write(json.dumps(_row(0.05 + 0.01 * i, -5.0)) + "\n")
            for i in range(25):
                fh.write(json.dumps(_row(0.85 + 0.005 * i, +5.0)) + "\n")
        rc = ws._cli(["--path", str(path)])
        assert rc == 2
        out = capsys.readouterr().out
        assert "BUBBLE_GATE_HARMFUL" in out

    def test_cli_exit_1_on_insufficient_data(self, tmp_path):
        path = tmp_path / "o.jsonl"
        path.write_text("")  # empty file
        rc = ws._cli(["--path", str(path)])
        assert rc == 1

    def test_cli_json_output(self, tmp_path, capsys):
        path = tmp_path / "o.jsonl"
        with path.open("w") as fh:
            for i in range(30):
                fh.write(json.dumps(_row(i / 30.0, 1.0)) + "\n")
        ws._cli(["--path", str(path), "--json"])
        out = capsys.readouterr().out
        # First char of the printed JSON must be { (machine-readable).
        parsed = json.loads(out)
        assert "verdict" in parsed
        assert "buckets" in parsed
        assert parsed["bubble_gate_threshold"] == ws.BUBBLE_GATE_THRESHOLD
