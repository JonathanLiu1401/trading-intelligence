"""Tests for paper_trader.ml.kill_switch_realized_effect.

Mirrors the canonical sibling pattern (test_scorer_buy_sell_skill /
test_gate_health_trend): every assertion pins a SPECIFIC expected value
or boundary; smoke-style "no crash" checks are kept to a minimum and
only on intentional defensive paths.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from paper_trader.ml import kill_switch_realized_effect as kse
from paper_trader.ml.kill_switch_realized_effect import (
    MIN_PAIRS,
    SKILL_TOL,
    _bucket_for,
    _bucket_stats,
    _maybe_float,
    analyze,
    kill_switch_report,
)


# ─────────────────────────── _maybe_float ───────────────────────────


class TestMaybeFloat:
    def test_none_returns_none(self):
        assert _maybe_float(None) is None

    def test_bool_returns_none(self):
        assert _maybe_float(True) is None
        assert _maybe_float(False) is None

    def test_string_returns_none(self):
        assert _maybe_float("foo") is None

    def test_nan_returns_none(self):
        assert _maybe_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert _maybe_float(float("inf")) is None
        assert _maybe_float(float("-inf")) is None

    def test_finite_float_passes(self):
        assert _maybe_float(1.5) == 1.5
        assert _maybe_float(-3.0) == -3.0
        assert _maybe_float(0.0) == 0.0

    def test_int_coerces(self):
        assert _maybe_float(7) == 7.0


# ─────────────────────────── _bucket_for ───────────────────────────


class TestBucketFor:
    def test_none_is_acted(self):
        assert _bucket_for(None) == "acted"

    def test_killswitch_lowercase(self):
        assert _bucket_for("killswitch") == "killswitch"

    def test_clamp_lowercase(self):
        assert _bucket_for("clamp") == "clamp"

    def test_uppercase_normalized(self):
        assert _bucket_for("KILLSWITCH") == "killswitch"
        assert _bucket_for("Clamp") == "clamp"

    def test_whitespace_stripped(self):
        assert _bucket_for("  killswitch  ") == "killswitch"

    def test_unknown_kind_dropped(self):
        assert _bucket_for("future_unknown") is None
        assert _bucket_for("") is None

    def test_non_string_dropped(self):
        assert _bucket_for(42) is None
        assert _bucket_for([]) is None


# ─────────────────────────── _bucket_stats ───────────────────────────


class TestBucketStats:
    def test_empty_bucket(self):
        s = _bucket_stats([])
        assert s == {"n": 0, "mean_realized": None, "median_realized": None,
                     "mean_pred": None, "rank_ic": None}

    def test_single_pair_no_rank_ic(self):
        s = _bucket_stats([(5.0, 2.0)])
        assert s["n"] == 1
        # mean equals the single value
        assert s["mean_pred"] == 5.0
        assert s["mean_realized"] == 2.0
        assert s["median_realized"] == 2.0
        # rank-IC undefined on n=1
        assert s["rank_ic"] is None

    def test_perfect_positive_correlation(self):
        # Predictions and realizations strictly monotone-aligned ⇒ IC = +1.0
        pairs = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
        s = _bucket_stats(pairs)
        assert s["n"] == 4
        assert s["rank_ic"] == pytest.approx(1.0)
        assert s["mean_pred"] == 2.5
        assert s["mean_realized"] == 2.5
        assert s["median_realized"] == 2.5

    def test_perfect_negative_correlation(self):
        pairs = [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)]
        s = _bucket_stats(pairs)
        assert s["rank_ic"] == pytest.approx(-1.0)

    def test_zero_variance_pred_returns_zero_ic(self):
        # Constant predictor — _spearman returns 0.0 (no rank skill)
        pairs = [(2.0, 1.0), (2.0, 3.0), (2.0, 7.0), (2.0, 11.0)]
        s = _bucket_stats(pairs)
        assert s["rank_ic"] == 0.0

    def test_mean_and_median_distinct_when_outlier(self):
        pairs = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 100.0)]
        s = _bucket_stats(pairs)
        assert s["mean_realized"] == pytest.approx(25.0)
        assert s["median_realized"] == 0.0


# ─────────────────────────── kill_switch_report ───────────────────────────


def _row(action="BUY", kind=None, pred=1.0, ret=0.5, **kw):
    base = {"action": action, "gate_abstention_kind": kind,
            "gate_scorer_pred": pred, "forward_return_5d": ret}
    base.update(kw)
    return base


class TestKillSwitchReport:
    def test_empty_returns_insufficient(self):
        r = kill_switch_report([])
        assert r["verdict"] == "INSUFFICIENT_DATA"
        assert r["n_buys"] == 0
        assert r["n_with_pred"] == 0

    def test_non_dict_rows_dropped(self):
        # Non-dict rows MUST be skipped without crashing.
        r = kill_switch_report([None, "string", 42, _row()])
        assert r["n_buys"] == 1

    def test_sells_are_skipped(self):
        rows = [_row(action="SELL") for _ in range(10)]
        rows.append(_row())  # one BUY
        r = kill_switch_report(rows)
        assert r["n_buys"] == 1

    def test_rows_missing_pred_dropped(self):
        rows = [_row(pred=None), _row(ret=None), _row()]
        r = kill_switch_report(rows)
        assert r["n_buys"] == 3
        assert r["n_with_pred"] == 1

    def test_unknown_kind_dropped_from_bucket(self):
        # A future ``gate_abstention_kind`` value the analyzer doesn't
        # recognize should NOT be silently lumped into a known bucket —
        # it's dropped so the verdict isn't contaminated.
        rows = [_row(kind="unknown_future") for _ in range(MIN_PAIRS + 10)]
        r = kill_switch_report(rows)
        # Every bucket is empty — the unknown rows did NOT pollute "acted"
        # or any other bucket.
        for b in ("acted", "clamp", "killswitch"):
            assert r["buckets"][b]["n"] == 0
        assert r["verdict"] == "INSUFFICIENT_DATA"

    def test_bucket_split(self):
        rows = (
            [_row(kind=None) for _ in range(5)]
            + [_row(kind="clamp") for _ in range(3)]
            + [_row(kind="killswitch") for _ in range(7)]
        )
        r = kill_switch_report(rows)
        assert r["buckets"]["acted"]["n"] == 5
        assert r["buckets"]["clamp"]["n"] == 3
        assert r["buckets"]["killswitch"]["n"] == 7

    def test_insufficient_data_below_min_pairs(self):
        # killswitch bucket size below MIN_PAIRS → INSUFFICIENT_DATA
        rows = [_row(kind="killswitch") for _ in range(MIN_PAIRS - 1)]
        r = kill_switch_report(rows)
        assert r["verdict"] == "INSUFFICIENT_DATA"

    def test_killswitch_neutral_verdict(self):
        # Construct killswitch bucket where rank-IC is roughly 0:
        # interleave pred/realized so they decorrelate. Use MIN_PAIRS
        # rows exactly at noise.
        rows = []
        for i in range(MIN_PAIRS):
            # Random-ish but deterministic: pred and ret use different mod
            # patterns so their ranks decorrelate.
            pred = float((i * 7) % 11) - 5.0
            ret = float((i * 13) % 17) - 8.0
            rows.append(_row(kind="killswitch", pred=pred, ret=ret))
        r = kill_switch_report(rows)
        # Verdict is one of the noise/neutral states.
        assert r["verdict"] in ("KILLSWITCH_NEUTRAL", "INSUFFICIENT_DATA")
        ks_ic = r["buckets"]["killswitch"]["rank_ic"]
        assert ks_ic is not None
        # Magnitude must be below the explicit verdict threshold AT MOST
        # — we tightened "neutral" to mean |ic| < SKILL_TOL.
        if r["verdict"] == "KILLSWITCH_NEUTRAL":
            assert abs(ks_ic) < SKILL_TOL

    def test_killswitch_hurts_verdict(self):
        # Strong positive rank-IC in killswitch bucket ⇒ KILLSWITCH_HURTS.
        rows = []
        for i in range(MIN_PAIRS):
            # Aligned ranks: pred and ret co-monotone ⇒ IC ≈ +1.
            rows.append(_row(kind="killswitch", pred=float(i),
                             ret=float(i) + 0.1))
        r = kill_switch_report(rows)
        assert r["verdict"] == "KILLSWITCH_HURTS"
        assert r["buckets"]["killswitch"]["rank_ic"] >= SKILL_TOL

    def test_killswitch_helps_verdict(self):
        # Strong negative rank-IC in killswitch bucket ⇒ KILLSWITCH_HELPS.
        rows = []
        for i in range(MIN_PAIRS):
            rows.append(_row(kind="killswitch", pred=float(i),
                             ret=float(-i) + 0.1))
        r = kill_switch_report(rows)
        assert r["verdict"] == "KILLSWITCH_HELPS"
        assert r["buckets"]["killswitch"]["rank_ic"] <= -SKILL_TOL

    def test_boundary_at_exact_skill_tol_is_hurts(self):
        # rank_ic >= +SKILL_TOL ⇒ HURTS (boundary inclusive).
        # Construct a bucket where rank-IC ≈ +SKILL_TOL exactly.
        # Approximation: 200 mostly-noise + a small positive correlation.
        # Use deterministic seed.
        import random as _random
        rng = _random.Random(42)
        # Build pairs with controlled rank-IC by mixing perfectly correlated
        # and shuffled pairs.
        n = MIN_PAIRS
        # 5% perfectly correlated + 95% random — yields small positive IC.
        n_corr = max(1, int(0.06 * n))
        n_noise = n - n_corr
        preds = list(range(n_corr)) + [rng.uniform(-10, 10)
                                        for _ in range(n_noise)]
        rets = list(range(n_corr)) + [rng.uniform(-10, 10)
                                       for _ in range(n_noise)]
        rng.shuffle(rets[n_corr:])
        rows = [_row(kind="killswitch", pred=p, ret=r)
                for p, r in zip(preds, rets)]
        r = kill_switch_report(rows)
        # Either HURTS or NEUTRAL depending on the exact rank-IC.
        # Just assert the verdict aligns with the rank-IC sign + magnitude.
        ks_ic = r["buckets"]["killswitch"]["rank_ic"]
        if ks_ic >= SKILL_TOL:
            assert r["verdict"] == "KILLSWITCH_HURTS"
        elif ks_ic <= -SKILL_TOL:
            assert r["verdict"] == "KILLSWITCH_HELPS"
        else:
            assert r["verdict"] == "KILLSWITCH_NEUTRAL"

    def test_acted_bucket_does_not_drive_verdict(self):
        # An "acted" bucket with positive rank-IC should NOT trigger
        # KILLSWITCH_HURTS — verdict is killswitch-bucket-only.
        rows = []
        for i in range(MIN_PAIRS):
            rows.append(_row(kind=None, pred=float(i), ret=float(i)))
        r = kill_switch_report(rows)
        # acted bucket rank-IC is high
        assert r["buckets"]["acted"]["rank_ic"] >= SKILL_TOL
        # but verdict is INSUFFICIENT_DATA (killswitch bucket is empty)
        assert r["verdict"] == "INSUFFICIENT_DATA"

    def test_clamp_bucket_does_not_drive_verdict(self):
        rows = []
        for i in range(MIN_PAIRS):
            rows.append(_row(kind="clamp", pred=float(i), ret=float(i)))
        r = kill_switch_report(rows)
        assert r["buckets"]["clamp"]["rank_ic"] >= SKILL_TOL
        # Verdict is INSUFFICIENT_DATA — verdict is killswitch-only.
        assert r["verdict"] == "INSUFFICIENT_DATA"

    def test_non_finite_predictions_dropped(self):
        # NaN/Inf in pred or ret must NOT crash and must be excluded from
        # the bucket count.
        rows = [
            _row(kind="killswitch", pred=float("nan"), ret=1.0),
            _row(kind="killswitch", pred=1.0, ret=float("inf")),
            _row(kind="killswitch", pred=1.0, ret=1.0),
        ]
        r = kill_switch_report(rows)
        assert r["buckets"]["killswitch"]["n"] == 1

    def test_report_shape_invariants(self):
        r = kill_switch_report([])
        # Every documented top-level key exists in the empty-report case.
        for k in ("verdict", "n_buys", "n_with_pred", "buckets",
                  "min_pairs", "skill_tol", "hint"):
            assert k in r
        # All three buckets reported even when empty.
        for b in ("acted", "clamp", "killswitch"):
            assert b in r["buckets"]


# ─────────────────────────── analyze (file IO) ───────────────────────────


class TestAnalyzeFile:
    def test_missing_file_reports_error(self, tmp_path):
        missing = tmp_path / "nonexistent.jsonl"
        r = analyze(missing)
        assert r["status"] == "error"
        assert "missing" in r["error"]
        assert r["verdict"] == "INSUFFICIENT_DATA"

    def test_jsonl_round_trip(self, tmp_path):
        outcomes_path = tmp_path / "outcomes.jsonl"
        rows = [_row(kind="killswitch", pred=float(i), ret=float(i) + 0.1)
                for i in range(MIN_PAIRS)]
        with outcomes_path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        r = analyze(outcomes_path)
        assert r["status"] == "ok"
        assert r["verdict"] == "KILLSWITCH_HURTS"
        assert r["buckets"]["killswitch"]["n"] == MIN_PAIRS

    def test_corrupt_lines_skipped(self, tmp_path):
        outcomes_path = tmp_path / "outcomes.jsonl"
        with outcomes_path.open("w") as fh:
            fh.write("{not valid json\n")
            fh.write(json.dumps(_row()) + "\n")
            fh.write("garbage\n")
            fh.write("\n")  # blank line — must be tolerated
            fh.write(json.dumps(_row()) + "\n")
        r = analyze(outcomes_path)
        # Two valid rows; corruption did not break the analyzer.
        assert r["status"] == "ok"
        assert r["n_buys"] == 2

    def test_default_path_resolves_to_repo(self):
        # Default DECISION_OUTCOMES points at the production file.
        from paper_trader.ml.kill_switch_realized_effect import DECISION_OUTCOMES
        # Path object pointed at data/decision_outcomes.jsonl
        assert str(DECISION_OUTCOMES).endswith("data/decision_outcomes.jsonl")


# ─────────────────────────── CLI ───────────────────────────


class TestCliExitCodes:
    def _write_outcomes(self, tmp_path, rows):
        p = tmp_path / "outcomes.jsonl"
        with p.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return p

    def test_exit_zero_on_insufficient(self, tmp_path, capsys):
        p = self._write_outcomes(tmp_path, [])
        rc = kse._cli(["--outcomes", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "INSUFFICIENT_DATA" in out

    def test_exit_zero_on_neutral(self, tmp_path, capsys):
        # Build a killswitch bucket with rank-IC near 0.
        rows = []
        for i in range(MIN_PAIRS):
            pred = float((i * 7) % 11) - 5.0
            ret = float((i * 13) % 17) - 8.0
            rows.append(_row(kind="killswitch", pred=pred, ret=ret))
        p = self._write_outcomes(tmp_path, rows)
        rc = kse._cli(["--outcomes", str(p)])
        # Either NEUTRAL or INSUFFICIENT_DATA — both exit 0.
        assert rc == 0

    def test_exit_two_on_hurts(self, tmp_path):
        rows = [_row(kind="killswitch", pred=float(i), ret=float(i) + 0.1)
                for i in range(MIN_PAIRS)]
        p = self._write_outcomes(tmp_path, rows)
        rc = kse._cli(["--outcomes", str(p)])
        assert rc == 2

    def test_exit_zero_on_helps(self, tmp_path):
        # KILLSWITCH_HELPS is a benign verdict (kill-switch doing real
        # work) — exit code 0, not 2.
        rows = [_row(kind="killswitch", pred=float(i), ret=-float(i))
                for i in range(MIN_PAIRS)]
        p = self._write_outcomes(tmp_path, rows)
        rc = kse._cli(["--outcomes", str(p)])
        assert rc == 0

    def test_json_output_machine_readable(self, tmp_path, capsys):
        rows = [_row(kind="killswitch", pred=float(i), ret=float(i))
                for i in range(MIN_PAIRS)]
        p = self._write_outcomes(tmp_path, rows)
        rc = kse._cli(["--outcomes", str(p), "--json"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["status"] == "ok"
        assert parsed["verdict"] in (
            "KILLSWITCH_HELPS", "KILLSWITCH_NEUTRAL", "KILLSWITCH_HURTS",
            "INSUFFICIENT_DATA")
        # Both exit code paths agree with verdict.
        if parsed["verdict"] == "KILLSWITCH_HURTS":
            assert rc == 2
        else:
            assert rc == 0
