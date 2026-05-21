"""Tests for paper_trader/ml/conviction_calibration.py — the
sizing-calibration consumer of the 2026-05-21 ``conviction_pct`` capture.

All offline, synthetic-only — the live ``decision_outcomes.jsonl`` carries
zero rows with ``conviction_pct`` (the loop has not completed a cycle
since the field landed), so production data cannot exercise these paths.
Exact-value verdict locks mirror the sibling ``test_calibration.py`` /
``test_gate_decision_capture.py`` style: a threshold tweak must update
the literal asserts deliberately.
"""
from __future__ import annotations

import json
import random

import pytest

from paper_trader.ml import conviction_calibration as cc


# ───────────────────────── fixtures ──────────────────────────


def _row(action="BUY", conviction=0.25, ret_5d=2.0):
    """One outcome-shaped dict. Mirrors the keys ``_compute_decision_outcomes``
    emits — extra keys are ignored by `build_conviction_calibration`."""
    return {
        "action": action,
        "conviction_pct": conviction,
        "forward_return_5d": ret_5d,
        "ticker": "NVDA",
        "sim_date": "2025-06-15",
    }


def _well_calibrated_corpus(n_per_bucket=20, seed=42):
    """Synthetic: 5 conviction buckets in [0.05..0.45], realized return
    rises linearly with conviction. Locked to give WELL_CALIBRATED."""
    rng = random.Random(seed)
    rows: list[dict] = []
    buckets = [(0.05, -2.0), (0.15, 0.0), (0.25, 2.0), (0.35, 4.0), (0.45, 6.0)]
    for conv, mean_ret in buckets:
        for _ in range(n_per_bucket):
            # small noise so std_realized is non-zero but doesn't muddy the rank
            rows.append(_row(conviction=conv,
                             ret_5d=mean_ret + rng.gauss(0, 0.2)))
    return rows


def _inverted_corpus(n_per_bucket=20, seed=42):
    """Realized return DROPS with conviction — the GATE_HARMFUL shape."""
    rng = random.Random(seed)
    rows: list[dict] = []
    buckets = [(0.05, 6.0), (0.15, 4.0), (0.25, 2.0), (0.35, 0.0), (0.45, -2.0)]
    for conv, mean_ret in buckets:
        for _ in range(n_per_bucket):
            rows.append(_row(conviction=conv,
                             ret_5d=mean_ret + rng.gauss(0, 0.2)))
    return rows


def _flat_corpus(n_per_bucket=20):
    """Realized return is INDEPENDENT of conviction — GATE_INEFFECTIVE.

    Deterministic: every conviction bucket sees the SAME multiset of
    realized values (cycled through [-2, -1, 0, 1, 2, ...]), so the
    spearman correlation is ~0 by construction (no rank ordering of
    realized given conviction). This avoids the rng-driven false-positive
    rank correlation a gaussian-noise corpus can carry over only n=100.
    """
    rows: list[dict] = []
    # 20 distinct realized values, replicated identically across the 5
    # conviction buckets. With identical bucket distributions the mean
    # realized is identical, and the global spearman of conviction → ret
    # is 0 (every conviction value is tied with multiple realized values
    # spanning the full range).
    values = [(-2.0 + 0.2 * i) for i in range(n_per_bucket)]
    for conv in (0.05, 0.15, 0.25, 0.35, 0.45):
        for v in values:
            rows.append(_row(conviction=conv, ret_5d=v))
    return rows


def _directional_corpus(n_per_bucket=20, seed=42):
    """Some rank skill (spearman in (FLAT, GOOD)) but spread too small for
    WELL_CALIBRATED. Tunable bridge between MISCALIBRATED and WELL."""
    rng = random.Random(seed)
    rows: list[dict] = []
    # Tiny linear lift: realized rises from 0 → 1pp across the conviction
    # quintiles — spearman is positive but the spread is below BUCKET_GAP_GOOD_PCT.
    buckets = [(0.05, 0.0), (0.15, 0.25), (0.25, 0.5), (0.35, 0.75), (0.45, 1.0)]
    for conv, mean_ret in buckets:
        for _ in range(n_per_bucket):
            rows.append(_row(conviction=conv,
                             ret_5d=mean_ret + rng.gauss(0, 0.3)))
    return rows


# ───────────────────────── verdict locks ──────────────────────────


class TestVerdictLadder:
    def test_well_calibrated(self):
        rep = cc.build_conviction_calibration(_well_calibrated_corpus())
        assert rep["status"] == "ok"
        assert rep["verdict"] == "WELL_CALIBRATED"
        assert rep["n"] == 100
        # Spearman should be very high — clean monotone with low noise.
        assert rep["spearman"] >= cc.SPEARMAN_GOOD
        # Top-vs-bottom spread is ~8pp (6 − -2), well above BUCKET_GAP_GOOD_PCT.
        assert rep["top_minus_bottom_realized_pct"] >= cc.BUCKET_GAP_GOOD_PCT
        # Strictly non-decreasing across 5 buckets ⇒ all 4 steps non-dec.
        assert rep["monotone_fraction"] == 1.0
        assert len(rep["buckets"]) == 5

    def test_inverted(self):
        rep = cc.build_conviction_calibration(_inverted_corpus())
        assert rep["verdict"] == "INVERTED"
        # Top bucket realized < bottom bucket realized (negative spread).
        assert rep["top_minus_bottom_realized_pct"] <= -cc.BUCKET_GAP_TOL_PCT
        # Spearman is strongly negative.
        assert rep["spearman"] <= -cc.SPEARMAN_FLAT

    def test_miscalibrated_flat(self):
        rep = cc.build_conviction_calibration(_flat_corpus())
        assert rep["verdict"] == "MISCALIBRATED"
        # |spearman| < FLAT threshold.
        assert abs(rep["spearman"]) < cc.SPEARMAN_FLAT
        # The buckets all hover around the same mean — top-vs-bottom is small.
        assert abs(rep["top_minus_bottom_realized_pct"]) < cc.BUCKET_GAP_TOL_PCT

    def test_directional_below_strong_bar(self):
        rep = cc.build_conviction_calibration(_directional_corpus())
        assert rep["verdict"] == "DIRECTIONAL"
        assert rep["spearman"] > cc.SPEARMAN_FLAT
        # Spread is below the WELL_CALIBRATED bar but above zero.
        assert rep["top_minus_bottom_realized_pct"] < cc.BUCKET_GAP_GOOD_PCT
        assert rep["top_minus_bottom_realized_pct"] > 0


# ─────────────────── INSUFFICIENT_DATA cases ──────────────────


class TestInsufficientData:
    def test_empty_records(self):
        rep = cc.build_conviction_calibration([])
        assert rep["status"] == "insufficient_data"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == 0
        assert rep["buckets"] == []

    def test_below_min_pairs(self):
        rows = [_row(conviction=0.1 + i * 0.01, ret_5d=float(i))
                for i in range(cc.MIN_PAIRS - 1)]
        rep = cc.build_conviction_calibration(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n"] == cc.MIN_PAIRS - 1

    def test_exactly_min_pairs_ok(self):
        # Exactly MIN_PAIRS BUY rows ⇒ NOT INSUFFICIENT_DATA. Locks the
        # boundary (the contract is `n < MIN_PAIRS` ⇒ insufficient, not `<=`).
        rng = random.Random(7)
        rows = []
        for i in range(cc.MIN_PAIRS):
            conv = 0.05 + (i % 5) * 0.10
            rows.append(_row(conviction=conv, ret_5d=conv * 10 + rng.gauss(0, 0.1)))
        rep = cc.build_conviction_calibration(rows)
        assert rep["status"] == "ok"
        assert rep["n"] == cc.MIN_PAIRS


# ─────────────────── drop / filter contract ──────────────────


class TestRowFilter:
    def test_sell_rows_dropped(self):
        """SELL rows have conviction_pct=None by design (gate is BUY-only)
        — they must not pollute the count."""
        rows = (_well_calibrated_corpus()
                + [_row(action="SELL", conviction=None, ret_5d=5.0)
                   for _ in range(10)])
        rep = cc.build_conviction_calibration(rows)
        assert rep["n"] == 100  # only the BUY rows
        assert rep["n_dropped_action"] == 10
        # Verdict unchanged by the SELL pollution.
        assert rep["verdict"] == "WELL_CALIBRATED"

    def test_missing_conviction_dropped(self):
        rows = (_well_calibrated_corpus()
                + [_row(conviction=None, ret_5d=5.0) for _ in range(15)])
        rep = cc.build_conviction_calibration(rows)
        assert rep["n"] == 100
        assert rep["n_dropped_conviction"] == 15

    def test_out_of_range_conviction_dropped(self):
        # Upstream parser clamps to [0,1]; an out-of-range row here is a
        # corrupted record. Must drop, not raise, not silently include.
        rows = _well_calibrated_corpus() + [
            _row(conviction=1.5, ret_5d=10.0),
            _row(conviction=-0.1, ret_5d=10.0),
        ]
        rep = cc.build_conviction_calibration(rows)
        assert rep["n"] == 100
        assert rep["n_dropped_conviction"] == 2

    def test_non_finite_return_dropped(self):
        rows = _well_calibrated_corpus() + [
            _row(conviction=0.25, ret_5d=float("nan")),
            _row(conviction=0.25, ret_5d=float("inf")),
            _row(conviction=0.25, ret_5d=None),
        ]
        rep = cc.build_conviction_calibration(rows)
        assert rep["n"] == 100
        assert rep["n_dropped_return"] == 3

    def test_non_dict_rows_drop_in_action(self):
        # Garbage rows (a corrupted JSONL line that parses to a non-dict
        # via some upstream tool) must drop silently, not raise.
        rows = list(_well_calibrated_corpus()) + [None, 42, "row", [1, 2]]
        rep = cc.build_conviction_calibration(rows)  # type: ignore[arg-type]
        assert rep["n"] == 100
        assert rep["n_dropped_action"] == 4

    def test_bool_conviction_rejected(self):
        # `True` is a Python int (1), but `_to_finite_float` rejects bool
        # explicitly so it can't masquerade as a 1.0 conviction.
        rows = _well_calibrated_corpus() + [
            _row(conviction=True, ret_5d=5.0),
            _row(conviction=False, ret_5d=5.0),
        ]
        rep = cc.build_conviction_calibration(rows)
        assert rep["n"] == 100
        assert rep["n_dropped_conviction"] == 2


# ─────────────────── bucket geometry ──────────────────


class TestBucketGeometry:
    def test_quintile_default(self):
        rep = cc.build_conviction_calibration(_well_calibrated_corpus())
        assert len(rep["buckets"]) == 5
        # Buckets are 1-indexed and ordered by conviction ascending.
        idxs = [b["idx"] for b in rep["buckets"]]
        assert idxs == [1, 2, 3, 4, 5]
        # mean_conviction must rise monotonically across buckets (the cut
        # is on the conviction quantile so this is mechanical).
        means = [b["mean_conviction"] for b in rep["buckets"]]
        assert means == sorted(means)

    def test_custom_bucket_count(self):
        rep = cc.build_conviction_calibration(
            _well_calibrated_corpus(), n_buckets=3
        )
        assert len(rep["buckets"]) == 3

    def test_bucket_count_clamped_by_n(self):
        # 30 rows / 3 = 10 ⇒ even with n_buckets=100 the clamp k = n // 3 = 10
        # caps the bucket count at 10.
        rng = random.Random(11)
        rows = [_row(conviction=0.05 + (i % 5) * 0.10,
                     ret_5d=float(i) * 0.1 + rng.gauss(0, 0.01))
                for i in range(cc.MIN_PAIRS)]
        rep = cc.build_conviction_calibration(rows, n_buckets=100)
        assert len(rep["buckets"]) <= cc.MIN_PAIRS // 3

    def test_std_realized_zero_when_bucket_size_1(self):
        # n=30 with n_buckets=30 → each bucket has 1 sample → std=0.0
        # by `len(seg_y) >= 2` guard.
        rows = [_row(conviction=0.01 + i * 0.01, ret_5d=float(i))
                for i in range(cc.MIN_PAIRS)]
        rep = cc.build_conviction_calibration(rows, n_buckets=30)
        for b in rep["buckets"]:
            if b["n"] < 2:
                assert b["std_realized"] == 0.0


# ─────────────────── analyze() — loader contract ──────────────────


class TestAnalyzeLoader:
    def test_missing_file_returns_insufficient(self, tmp_path):
        rep = cc.analyze(tmp_path / "no_such_file.jsonl")
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["status"] == "insufficient_data"

    def test_empty_file_returns_insufficient(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        rep = cc.analyze(p)
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_malformed_lines_skipped_not_raised(self, tmp_path):
        p = tmp_path / "mixed.jsonl"
        good = _well_calibrated_corpus()
        lines = []
        lines.append("not json")
        lines.append("{broken")
        for r in good:
            lines.append(json.dumps(r))
        lines.append("")  # blank
        lines.append("42")  # parses to int — load_outcomes keeps it; analyzer drops it
        p.write_text("\n".join(lines))
        rep = cc.analyze(p)
        assert rep["verdict"] == "WELL_CALIBRATED"
        assert rep["n"] == 100  # ints + garbage lines dropped

    def test_load_outcomes_streaming(self, tmp_path):
        p = tmp_path / "outcomes.jsonl"
        rows = _well_calibrated_corpus(n_per_bucket=5)
        p.write_text("\n".join(json.dumps(r) for r in rows))
        loaded = cc.load_outcomes(p)
        assert len(loaded) == 25


# ─────────────────── CLI ──────────────────


class TestCli:
    def test_exit_code_well_calibrated(self, tmp_path, capsys):
        p = tmp_path / "outcomes.jsonl"
        p.write_text("\n".join(json.dumps(r)
                               for r in _well_calibrated_corpus()))
        rc = cc._cli(["--path", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "WELL_CALIBRATED" in out

    def test_exit_code_inverted(self, tmp_path, capsys):
        p = tmp_path / "outcomes.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in _inverted_corpus()))
        rc = cc._cli(["--path", str(p)])
        assert rc == 2  # the quant-decisive exit code
        assert "INVERTED" in capsys.readouterr().out

    def test_exit_code_other_verdicts_are_one(self, tmp_path, capsys):
        p = tmp_path / "outcomes.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in _flat_corpus()))
        rc = cc._cli(["--path", str(p)])
        assert rc == 1
        assert "MISCALIBRATED" in capsys.readouterr().out

    def test_json_mode_emits_valid_json(self, tmp_path, capsys):
        p = tmp_path / "outcomes.jsonl"
        p.write_text("\n".join(json.dumps(r)
                               for r in _well_calibrated_corpus()))
        rc = cc._cli(["--path", str(p), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        obj = json.loads(out)
        assert obj["verdict"] == "WELL_CALIBRATED"
        assert obj["status"] == "ok"
        assert obj["n"] == 100
        # All expected keys present (lock the JSON schema deliberately).
        for k in ("status", "verdict", "n", "spearman", "mean_conviction",
                  "mean_realized", "top_minus_bottom_realized_pct",
                  "buckets", "monotone_fraction", "hint"):
            assert k in obj

    def test_missing_file_exit_one(self, tmp_path):
        rc = cc._cli(["--path", str(tmp_path / "no.jsonl")])
        assert rc == 1


# ─────────────────── never-raises discipline ──────────────────


class TestNeverRaises:
    def test_garbage_in_build_never_raises(self):
        # Mix of valid + garbage rows. The function MUST return a dict,
        # never propagate a ValueError/TypeError from `int("abc")` etc.
        for v in [None, 0, "str", [1], {}]:
            try:
                rep = cc.build_conviction_calibration([v] * 50)  # type: ignore[list-item]
            except Exception:
                pytest.fail("build_conviction_calibration raised")
            assert isinstance(rep, dict)
            assert "verdict" in rep

    def test_analyze_catches_loader_corruption(self, tmp_path):
        # A binary file. load_outcomes catches the decode error and returns
        # []; analyze degrades to INSUFFICIENT_DATA.
        p = tmp_path / "binary.jsonl"
        p.write_bytes(b"\x00\x01\x02\xff\xfe")
        rep = cc.analyze(p)
        assert isinstance(rep, dict)
        assert rep["verdict"] in ("INSUFFICIENT_DATA", "ERROR")
