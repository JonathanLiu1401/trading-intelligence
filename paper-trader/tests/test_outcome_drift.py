"""Tests for paper_trader.ml.outcome_drift — concept-drift report on
decision_outcomes.jsonl. The builder is pure / no I/O / never raises; the
tests pin exact-arithmetic boundaries, the state ladder, the
``_safe_float`` discriminator, sort order, and the load CLI's degrade
contract."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from paper_trader.ml import outcome_drift as od


# ─────────────────────────── _safe_float ───────────────────────────

class TestSafeFloat:
    def test_finite_float_unchanged(self):
        assert od._safe_float(3.14) == 3.14

    def test_int_coerced(self):
        assert od._safe_float(7) == 7.0

    def test_string_numeric_coerced(self):
        # JSON serializers sometimes emit numbers as strings; we accept them.
        assert od._safe_float("2.5") == 2.5

    def test_string_garbage_rejected(self):
        assert od._safe_float("abc") is None

    def test_none_rejected(self):
        assert od._safe_float(None) is None

    def test_nan_rejected(self):
        assert od._safe_float(float("nan")) is None

    def test_inf_rejected(self):
        assert od._safe_float(float("inf")) is None
        assert od._safe_float(float("-inf")) is None

    def test_bool_rejected(self):
        # bool is a Python int subclass — must NOT silently become 1.0/0.0,
        # which would corrupt feature statistics. Mirrors
        # decision_scorer._to_float's identical guard.
        assert od._safe_float(True) is None
        assert od._safe_float(False) is None


# ─────────────────────────── _population_stats ───────────────────

class TestPopulationStats:
    def test_empty_returns_zero(self):
        assert od._population_stats([]) == (0.0, 0.0, 0)

    def test_single_value(self):
        # n=1: std is 0 by population convention (we're not inferring).
        mean, std, n = od._population_stats([3.5])
        assert mean == 3.5
        assert std == 0.0
        assert n == 1

    def test_known_population_variance(self):
        # Population variance of [1,2,3,4,5] = ((-2)²+(-1)²+0²+1²+2²)/5 = 10/5 = 2.
        mean, std, n = od._population_stats([1, 2, 3, 4, 5])
        assert mean == 3.0
        assert std == pytest.approx(math.sqrt(2.0), rel=1e-9)
        assert n == 5

    def test_all_same_value_zero_std(self):
        mean, std, n = od._population_stats([7, 7, 7, 7])
        assert mean == 7.0
        assert std == 0.0
        assert n == 4


# ─────────────────────────── _drift_score ────────────────────────

class TestDriftScore:
    def _make_buckets(self, recent, older):
        return od._drift_score(recent, older)

    def test_insufficient_sample(self):
        # < MIN_PER_BUCKET in either bucket → INSUFFICIENT, no drift.
        r = self._make_buckets([1.0] * 5, [1.0] * 50)
        assert r["status"] == "INSUFFICIENT"
        assert r["drift_score"] is None
        r = self._make_buckets([1.0] * 50, [1.0] * 5)
        assert r["status"] == "INSUFFICIENT"

    def test_zero_drift_when_buckets_identical(self):
        # 50/50 identical values: mean shift = 0, score = 0.
        n = od.MIN_PER_BUCKET
        r = self._make_buckets([2.0] * n, [2.0] * n)
        assert r["status"] == "OK"
        assert r["drift_score"] == 0.0
        assert r["std_older"] == 0.0
        assert r["mean_recent"] == 2.0
        assert r["mean_older"] == 2.0

    def test_exact_one_sigma_drift(self):
        # σ_older = sqrt(2). Mean shift = sqrt(2) → drift_score = +1.0.
        n = od.MIN_PER_BUCKET
        older = [1.0, 2.0, 3.0, 4.0, 5.0] * (n // 5)
        # μ_o = 3.0, σ_o = sqrt(2). Add sqrt(2) ≈ 1.4142 to each.
        delta = math.sqrt(2.0)
        recent = [v + delta for v in older]
        r = self._make_buckets(recent, older)
        assert r["status"] == "OK"
        assert r["drift_score"] == pytest.approx(1.0, abs=1e-4)

    def test_negative_drift(self):
        n = od.MIN_PER_BUCKET
        older = [1.0, 2.0, 3.0, 4.0, 5.0] * (n // 5)
        # σ_o = sqrt(2). Shift recent down by half σ.
        recent = [v - (math.sqrt(2.0) / 2.0) for v in older]
        r = self._make_buckets(recent, older)
        assert r["drift_score"] == pytest.approx(-0.5, abs=1e-4)

    def test_constant_older_with_drift_yields_inf(self):
        # σ_older = 0 + non-zero mean shift → ±inf (a constant feature that
        # suddenly varies IS a regime change, by convention).
        n = od.MIN_PER_BUCKET
        r = self._make_buckets([5.0] * n, [3.0] * n)
        assert math.isinf(r["drift_score"])
        assert r["drift_score"] > 0
        r = self._make_buckets([1.0] * n, [3.0] * n)
        assert math.isinf(r["drift_score"])
        assert r["drift_score"] < 0

    def test_constant_older_without_drift_yields_zero(self):
        n = od.MIN_PER_BUCKET
        r = self._make_buckets([3.0] * n, [3.0] * n)
        assert r["drift_score"] == 0.0


# ─────────────────────────── _classify_feature ───────────────────

class TestClassifyFeature:
    def test_stable_inside_mild_threshold(self):
        # |drift| < DRIFT_MILD = 0.5 → STABLE.
        assert od._classify_feature(0.0) == "STABLE"
        assert od._classify_feature(0.49) == "STABLE"
        assert od._classify_feature(-0.49) == "STABLE"

    def test_mild_drift_boundary(self):
        # 0.5 ≤ |drift| < 1.0 → MILD_DRIFT.
        assert od._classify_feature(0.5) == "MILD_DRIFT"
        assert od._classify_feature(0.99) == "MILD_DRIFT"
        assert od._classify_feature(-0.7) == "MILD_DRIFT"

    def test_severe_drift_boundary(self):
        # |drift| ≥ 1.0 → SEVERE_DRIFT.
        assert od._classify_feature(1.0) == "SEVERE_DRIFT"
        assert od._classify_feature(-5.0) == "SEVERE_DRIFT"

    def test_inf_severe(self):
        assert od._classify_feature(math.inf) == "SEVERE_DRIFT"
        assert od._classify_feature(-math.inf) == "SEVERE_DRIFT"

    def test_none_unknown(self):
        assert od._classify_feature(None) == "UNKNOWN"


# ─────────────────────────── build_outcome_drift ─────────────────

class TestBuildOutcomeDrift:
    def test_empty_returns_no_data(self):
        rep = od.build_outcome_drift([])
        assert rep["state"] == "NO_DATA"
        assert rep["verdict"] == "UNKNOWN"
        assert rep["features"] == []

    def test_none_returns_no_data(self):
        rep = od.build_outcome_drift(None)
        assert rep["state"] == "NO_DATA"

    def test_insufficient_returns_insufficient(self):
        # 10 rows total — bucket sizes (8 older / 2 recent) both below MIN_PER_BUCKET=20.
        records = [
            {"sim_date": f"2024-01-{i:02d}", "ml_score": float(i)}
            for i in range(1, 11)
        ]
        rep = od.build_outcome_drift(records)
        assert rep["state"] == "INSUFFICIENT"
        assert rep["verdict"] == "UNKNOWN"

    def test_stable_when_buckets_identical(self):
        # 100 rows with identical features → STABLE verdict.
        records = []
        for i in range(100):
            records.append({
                "sim_date": f"2024-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}",
                "ml_score": 2.5, "rsi": 50.0, "macd": 0.0,
                "mom5": 1.0, "mom20": 2.0, "regime_mult": 1.0,
                "vol_ratio": 1.0, "bb_position": 0.0,
                "news_urgency": 50.0, "news_article_count": 1.0,
                "forward_return_5d": 0.5,
            })
        rep = od.build_outcome_drift(records)
        assert rep["state"] == "OK"
        assert rep["verdict"] == "STABLE"
        for feat in rep["features"]:
            assert feat["classification"] == "STABLE"
            assert feat["drift_score"] == 0.0

    def test_severe_drift_on_ml_score(self):
        # 100 records: first 75 have ml_score~0, last 25 have ml_score~10.
        # All other features stable. Verdict: SEVERE_DRIFT, worst=ml_score.
        records = []
        for i in range(100):
            ml = 10.0 if i >= 75 else 0.0
            records.append({
                "sim_date": f"2024-01-{(i % 28) + 1:02d}",
                # Sort order: padded by year so the temporal slice is unambiguous.
                "_idx": i,  # unused, just an idempotent column
                "ml_score": ml, "rsi": 50.0, "macd": 0.0,
                "mom5": 1.0, "mom20": 2.0, "regime_mult": 1.0,
                "vol_ratio": 1.0, "bb_position": 0.0,
                "news_urgency": 50.0, "news_article_count": 1.0,
                "forward_return_5d": 0.5,
            })
        # Re-create with deterministic date ordering so the bucket split is
        # i<75 in older, i>=75 in recent.
        for j, r in enumerate(records):
            r["sim_date"] = f"2024-{(j // 28) + 1:02d}-{(j % 28) + 1:02d}"
        rep = od.build_outcome_drift(records)
        assert rep["state"] == "OK"
        assert rep["verdict"] == "SEVERE_DRIFT"
        assert rep["worst_feature"] == "ml_score"
        # Drift score is +inf (older σ = 0 for constant 0, recent mean = 10).
        ml_feat = next(f for f in rep["features"] if f["feature"] == "ml_score")
        assert math.isinf(ml_feat["drift_score"])
        assert ml_feat["drift_score"] > 0

    def test_realized_return_direction_flip_captured(self):
        # Older bucket has fwd_5d ~ +2%, recent bucket ~ -3% — the
        # market literally flipped direction. label_shift_pct must be
        # exactly mean_recent - mean_older.
        records = []
        for i in range(100):
            fr = -3.0 if i >= 75 else 2.0
            records.append({
                "sim_date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                "ml_score": 2.0, "rsi": 50.0, "macd": 0.0,
                "mom5": 0.0, "mom20": 0.0, "regime_mult": 1.0,
                "vol_ratio": 1.0, "bb_position": 0.0,
                "news_urgency": 50.0, "news_article_count": 1.0,
                "forward_return_5d": fr,
            })
        rep = od.build_outcome_drift(records)
        assert rep["state"] == "OK"
        # Recent (last 25) − older (first 75): -3 − 2 = -5.0.
        assert rep["label_shift_pct"] == -5.0

    def test_features_sorted_by_abs_drift_desc(self):
        # Construct two features with different drift magnitudes:
        # ml_score shifts 1σ (SEVERE), mom5 shifts 0.6σ (MILD).
        # Sort must place ml_score (|1.0|) ahead of mom5 (|0.6|).
        records = []
        for i in range(120):
            if i >= 90:  # recent (last 25%)
                ml = 1.0
                m5 = 0.6
            else:
                ml = 0.0
                m5 = 0.0
            records.append({
                "sim_date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                # Older std for ml_score: spread = constant 0 → σ=0 → +inf drift.
                # That's not what we want — we need a non-zero σ_older. Add
                # small jitter so σ_older ≈ 1.
                "ml_score": ml + (i % 3 - 1),  # values -1, 0, 1 cycled
                "rsi": 50.0, "macd": 0.0,
                "mom5": m5 + (i % 3 - 1),  # same scale
                "mom20": 0.0, "regime_mult": 1.0,
                "vol_ratio": 1.0, "bb_position": 0.0,
                "news_urgency": 50.0, "news_article_count": 1.0,
                "forward_return_5d": 0.0,
            })
        rep = od.build_outcome_drift(records)
        assert rep["state"] == "OK"
        # First feature by sort must have the largest |drift_score| of those OK.
        first_ok = next(
            f for f in rep["features"] if f.get("drift_score") is not None
        )
        # ml_score must outrank mom5 because its mean shift is larger.
        ml_idx = next(i for i, f in enumerate(rep["features"])
                      if f["feature"] == "ml_score")
        m5_idx = next(i for i, f in enumerate(rep["features"])
                      if f["feature"] == "mom5")
        assert ml_idx < m5_idx, (
            f"ml_score (|drift|={abs(rep['features'][ml_idx]['drift_score']):.3f}) "
            f"must precede mom5 (|drift|={abs(rep['features'][m5_idx]['drift_score']):.3f})"
        )

    def test_garbage_records_degrade_safely(self):
        # Mix in dict-shaped garbage: None, malformed values, missing keys.
        # 100 well-formed rows is comfortably above the 20-per-bucket gate
        # (25 recent / 75 older) so per-feature INSUFFICIENT statuses below
        # come from missing column coverage, not from sample-size guard.
        records = [None, "not a dict", 42]
        good = []
        for i in range(100):
            good.append({
                "sim_date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                "ml_score": float(i % 5),
                "rsi": 50.0, "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
                "regime_mult": 1.0, "vol_ratio": 1.0, "bb_position": 0.0,
                "news_urgency": None,  # None values must be dropped, not 0-imputed.
                "news_article_count": None,
                "forward_return_5d": 0.0,
            })
        records.extend(good)
        # Must not raise.
        rep = od.build_outcome_drift(records)
        # Garbage rows are filtered out at the dict-shape gate; only the 100
        # well-formed rows reach population stats.
        assert rep["n_total"] == 100
        assert rep["state"] == "OK"
        # news_urgency / news_article_count have ALL None values → both
        # buckets empty (n=0) → INSUFFICIENT status, not crash. This is the
        # honest "column missing from the trainer's view" branch.
        nu = next(f for f in rep["features"]
                  if f["feature"] == "news_urgency")
        assert nu["status"] == "INSUFFICIENT"
        assert nu["drift_score"] is None
        nac = next(f for f in rep["features"]
                   if f["feature"] == "news_article_count")
        assert nac["status"] == "INSUFFICIENT"

    def test_recent_fraction_clamped(self):
        # 0.95 (above the 0.5 cap) must clamp to 0.5; -1.0 to 0.05.
        records = []
        for i in range(60):
            records.append({
                "sim_date": f"2024-01-{(i % 28) + 1:02d}",
                "ml_score": float(i),
                "rsi": 50.0, "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
                "regime_mult": 1.0, "vol_ratio": 1.0, "bb_position": 0.0,
                "news_urgency": 50.0, "news_article_count": 1.0,
                "forward_return_5d": 0.0,
            })
        rep = od.build_outcome_drift(records, recent_fraction=0.95)
        assert rep["recent_fraction"] == 0.5
        # 30/30 split with recent=last 30.
        assert rep["n_recent"] == 30
        assert rep["n_older"] == 30

        rep = od.build_outcome_drift(records, recent_fraction=-1.0)
        assert rep["recent_fraction"] == 0.05

    def test_recent_fraction_invalid_type_falls_back(self):
        # A non-numeric recent_fraction must not crash; default sticks.
        records = []
        for i in range(80):
            records.append({
                "sim_date": f"2024-01-{(i % 28) + 1:02d}",
                "ml_score": float(i),
                "rsi": 50.0, "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
                "regime_mult": 1.0, "vol_ratio": 1.0, "bb_position": 0.0,
                "news_urgency": 50.0, "news_article_count": 1.0,
                "forward_return_5d": 0.0,
            })
        rep = od.build_outcome_drift(records, recent_fraction="not a number")
        # Default RECENT_FRACTION = 0.25 → 20 recent / 60 older.
        assert rep["recent_fraction"] == od.RECENT_FRACTION


# ─────────────────────────── load_outcomes + analyze ─────────────

class TestLoadOutcomes:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        path = tmp_path / "nope.jsonl"
        assert od.load_outcomes(path) == []

    def test_malformed_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "x.jsonl"
        path.write_text(
            '{"a": 1}\n'
            'not json\n'
            '\n'
            '{"b": 2}\n'
        )
        assert od.load_outcomes(path) == [{"a": 1}, {"b": 2}]

    def test_analyze_roundtrips(self, tmp_path: Path):
        path = tmp_path / "out.jsonl"
        recs = []
        for i in range(60):
            recs.append({
                "sim_date": f"2024-01-{(i % 28) + 1:02d}",
                "ml_score": float(i % 3),
                "rsi": 50.0, "macd": 0.0, "mom5": 0.0, "mom20": 0.0,
                "regime_mult": 1.0, "vol_ratio": 1.0, "bb_position": 0.0,
                "news_urgency": 50.0, "news_article_count": 1.0,
                "forward_return_5d": 0.5,
            })
        with path.open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        rep = od.analyze(path)
        # 60 records → recent=15, older=45. Bucket size 15 is BELOW
        # MIN_PER_BUCKET=20 → INSUFFICIENT. This is the honest read.
        assert rep["state"] == "INSUFFICIENT"


# ─────────────────────────── CLI ──────────────────────────────────

class TestCLI:
    def test_cli_emits_json_on_request(self, tmp_path, capsys):
        path = tmp_path / "x.jsonl"
        path.write_text("")
        rc = od._cli(["--path", str(path), "--json"])
        out = capsys.readouterr().out
        # Empty file → NO_DATA → exit 1, valid JSON in stdout.
        assert rc == 1
        parsed = json.loads(out)
        assert parsed["state"] == "NO_DATA"

    def test_cli_table_on_missing_file(self, tmp_path, capsys):
        path = tmp_path / "nope.jsonl"
        rc = od._cli(["--path", str(path)])
        out = capsys.readouterr().out
        assert rc == 1
        assert "state=NO_DATA" in out

    def test_cli_table_severe_drift(self, tmp_path, capsys):
        path = tmp_path / "drift.jsonl"
        recs = []
        for i in range(100):
            ml = 10.0 if i >= 75 else 0.0
            recs.append({
                "sim_date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                "ml_score": ml, "rsi": 50.0, "macd": 0.0, "mom5": 0.0,
                "mom20": 0.0, "regime_mult": 1.0, "vol_ratio": 1.0,
                "bb_position": 0.0, "news_urgency": 50.0,
                "news_article_count": 1.0, "forward_return_5d": 0.0,
            })
        with path.open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        rc = od._cli(["--path", str(path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "verdict=SEVERE_DRIFT" in out
        assert "ml_score" in out
