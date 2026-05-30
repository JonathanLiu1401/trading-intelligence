"""Tests for paper_trader.ml.scorer_offdist_rate.

Exercises the analyzer's real correctness, not just "no crash":
* known input → known counts (every rate ratio asserted on integer
  numerators).
* verdict ladder boundaries (HEALTHY / MILD_OOD / SEVERE_OOD /
  GATE_DARK) pinned at threshold +/- one row.
* untrained / missing-predict-with-meta / missing-file paths surface
  the documented INSUFFICIENT_DATA envelope (NOT a crash).
* raw-distribution stats (min/max/mean/p5/p95) are computed from
  finite values only, with NaN/Inf raw values silently filtered.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_trader.ml import scorer_offdist_rate as mod


class _FakeScorer:
    """Lightweight stand-in for ``DecisionScorer`` — returns a programmed
    ``predict_with_meta`` envelope per call, in order, regardless of
    input. Mirrors just enough of the contract that the analyzer can
    exercise its full code paths offline."""

    def __init__(self, envelopes, is_trained=True):
        self._envelopes = list(envelopes)
        self.is_trained = is_trained
        self._i = 0

    def predict_with_meta(self, **kw):
        if self._i >= len(self._envelopes):
            raise IndexError("FakeScorer ran out of programmed envelopes")
        env = self._envelopes[self._i]
        self._i += 1
        return env


def _envelope(*, raw=0.0, pred=0.0, clamped=False, off_distribution=False,
              failed=False):
    """Build a predict_with_meta dict identical to what DecisionScorer emits."""
    return {
        "pred": pred, "raw": raw,
        "clamped": clamped, "off_distribution": off_distribution,
        "failed": failed,
        # The analyzer reads pred/raw/clamped/off_distribution/failed only,
        # but emit the full envelope for fidelity with the live scorer:
        "percentile": None, "calibrated": None,
        "gate_arm": None, "gate_arm_multiplier": None,
    }


def _row(ml_score=1.0, ticker="NVDA"):
    """Minimal outcome-style dict — every field None/0.0 is fine because
    _FakeScorer ignores its kwargs and returns the next programmed envelope."""
    return {"ml_score": ml_score, "ticker": ticker, "regime_mult": 1.0}


# ---------------------------- core counting -----------------------------

def test_healthy_all_in_distribution():
    envelopes = [_envelope(raw=v, pred=v) for v in [1.0, 2.0, -1.5, 0.5] * 10]
    rows = [_row() for _ in envelopes]
    report = mod.offdist_rate_report(_FakeScorer(envelopes), rows)
    # MIN_TOTAL=30 so 40 rows is plenty
    assert report["n"] == 40
    assert report["n_failed"] == 0
    assert report["n_off_distribution"] == 0
    assert report["n_in_distribution"] == 40
    assert report["off_dist_rate"] == 0.0
    assert report["verdict"] == "HEALTHY"
    assert report["has_failures"] is False


def test_mild_ood_rate_just_over_5pct():
    # 40 rows, 3 off-distribution = 7.5% > 5% (MILD_THRESHOLD)
    envelopes = (
        [_envelope(raw=50.0, pred=50.0, clamped=True, off_distribution=True)] * 3
        + [_envelope(raw=2.0, pred=2.0)] * 37
    )
    rows = [_row() for _ in envelopes]
    report = mod.offdist_rate_report(_FakeScorer(envelopes), rows)
    assert report["n"] == 40
    assert report["n_off_distribution"] == 3
    assert report["off_dist_rate"] == round(3 / 40, 4)
    assert report["verdict"] == "MILD_OOD"


def test_severe_ood_rate_over_25pct():
    # 40 rows, 12 off-distribution = 30%
    envelopes = (
        [_envelope(raw=80.0, pred=50.0, clamped=True, off_distribution=True)] * 12
        + [_envelope(raw=2.0, pred=2.0)] * 28
    )
    rows = [_row() for _ in envelopes]
    report = mod.offdist_rate_report(_FakeScorer(envelopes), rows)
    assert report["n"] == 40
    assert report["n_off_distribution"] == 12
    assert report["verdict"] == "SEVERE_OOD"


def test_gate_dark_rate_over_50pct():
    # 40 rows, 22 off-distribution = 55% > GATE_DARK_THRESHOLD (50%)
    envelopes = (
        [_envelope(raw=100.0, pred=50.0, clamped=True, off_distribution=True)] * 22
        + [_envelope(raw=2.0, pred=2.0)] * 18
    )
    rows = [_row() for _ in envelopes]
    report = mod.offdist_rate_report(_FakeScorer(envelopes), rows)
    assert report["n"] == 40
    assert report["n_off_distribution"] == 22
    assert report["off_dist_rate"] == round(22 / 40, 4)
    assert report["verdict"] == "GATE_DARK"


# ---------------------------- boundary checks ---------------------------

def test_healthy_boundary_exact_5pct_is_healthy():
    """off_dist_rate == 5% → HEALTHY (the verdict uses `>` not `>=`)."""
    # 40 rows, 2 off-distribution = exactly 5%
    envelopes = (
        [_envelope(raw=80.0, pred=50.0, clamped=True, off_distribution=True)] * 2
        + [_envelope(raw=2.0, pred=2.0)] * 38
    )
    rows = [_row() for _ in envelopes]
    report = mod.offdist_rate_report(_FakeScorer(envelopes), rows)
    assert report["off_dist_rate"] == 0.05
    assert report["verdict"] == "HEALTHY"


def test_insufficient_data_below_min_total():
    # MIN_TOTAL=30: 25 rows is below the gate
    envelopes = [_envelope(raw=2.0, pred=2.0)] * 25
    rows = [_row() for _ in envelopes]
    report = mod.offdist_rate_report(_FakeScorer(envelopes), rows)
    assert report["n"] == 25
    assert report["verdict"] == "INSUFFICIENT_DATA"


def test_min_total_constant_is_30():
    """Pin the documented threshold so a future tune is observable."""
    assert mod.MIN_TOTAL == 30
    assert mod.MILD_THRESHOLD == 0.05
    assert mod.SEVERE_THRESHOLD == 0.25
    assert mod.GATE_DARK_THRESHOLD == 0.50


# ---------------------------- failed path -------------------------------

def test_failed_rate_separated_from_off_dist():
    # 4 failed (predict couldn't be produced — failed=True) + 36 in-dist.
    # failed rows: predict_with_meta returns {failed:True, raw:0.0,
    # off_distribution:True, clamped:True}. In the failed=True envelope
    # the off_distribution flag is True too (per predict_with_meta's
    # contract: a failed prediction is the maximally untrustworthy
    # result), so it COUNTS in n_off_distribution. The has_failures
    # flag is separately True so the operator can drill into it.
    failed_env = _envelope(raw=0.0, pred=0.0, clamped=True,
                           off_distribution=True, failed=True)
    envelopes = [failed_env] * 4 + [_envelope(raw=2.0, pred=2.0)] * 36
    rows = [_row() for _ in envelopes]
    report = mod.offdist_rate_report(_FakeScorer(envelopes), rows)
    assert report["n"] == 40
    assert report["n_failed"] == 4
    assert report["failed_rate"] == round(4 / 40, 4)
    assert report["has_failures"] is True
    # off_distribution flag fires on a failed-row's envelope too:
    assert report["n_off_distribution"] == 4
    # in_distribution = total - off_dist - failed; the analyzer guards
    # the underflow so failed-AND-off-dist rows aren't double-counted.
    assert report["n_in_distribution"] == 36


def test_predict_with_meta_raises_counts_as_failed():
    """A scorer that itself raises on predict — counted as failed, raw
    distribution unaffected by that row."""
    class _RaisingScorer:
        is_trained = True

        def __init__(self, n_raise, n_ok):
            self.n_raise = n_raise
            self.n_ok = n_ok
            self.i = 0

        def predict_with_meta(self, **kw):
            self.i += 1
            if self.i <= self.n_raise:
                raise RuntimeError("synthetic predict failure")
            return _envelope(raw=3.0, pred=3.0)

    rows = [_row() for _ in range(40)]
    report = mod.offdist_rate_report(_RaisingScorer(5, 35), rows)
    assert report["n"] == 40
    assert report["n_failed"] == 5
    # off_distribution did NOT increment for the raising rows — those
    # rows have no envelope to read flags from.
    assert report["n_off_distribution"] == 0
    assert report["has_failures"] is True
    # raw_min / raw_max are computed from the 35 successful rows only.
    assert report["raw_min"] == 3.0
    assert report["raw_max"] == 3.0


# ---------------------------- raw distribution --------------------------

def test_raw_distribution_min_max_p5_p95():
    # 40 envelopes with raws spanning -80 .. +80 (some clamped); verify
    # the distribution stats are computed on the RAW (pre-clamp) values.
    raws = list(range(-20, 20)) + [80.0]  # 41 values
    envelopes = [_envelope(raw=float(r), pred=max(-50, min(50, float(r))),
                           clamped=abs(r) > 50,
                           off_distribution=abs(r) > 50)
                 for r in raws]
    rows = [_row() for _ in envelopes]
    report = mod.offdist_rate_report(_FakeScorer(envelopes), rows)
    assert report["raw_min"] == -20.0
    assert report["raw_max"] == 80.0
    # Only ONE row has |raw|>50, so n_off_distribution=1.
    assert report["n_off_distribution"] == 1


def test_raw_skips_non_finite():
    """A scorer that returns NaN/Inf raw — the analyzer filters those
    out of the distribution stats (mirrors decision_scorer._to_float)."""
    import math
    envelopes = (
        [_envelope(raw=float('nan'), pred=0.0, failed=True,
                   clamped=True, off_distribution=True)] * 5
        + [_envelope(raw=2.0, pred=2.0)] * 35
    )
    rows = [_row() for _ in envelopes]
    report = mod.offdist_rate_report(_FakeScorer(envelopes), rows)
    assert report["n"] == 40
    # NaN raws were filtered from the distribution but the rows still
    # count toward failure and total counts.
    assert report["n_failed"] == 5
    assert report["raw_min"] == 2.0
    assert report["raw_max"] == 2.0
    assert math.isfinite(report["raw_mean"])


# ---------------------------- empty / untrained / no-meta ---------------

def test_untrained_scorer_returns_insufficient_data():
    scorer = _FakeScorer([], is_trained=False)
    report = mod.offdist_rate_report(scorer, [_row() for _ in range(50)])
    assert report["verdict"] == "INSUFFICIENT_DATA"
    assert report["n"] == 0
    assert "not trained" in report["hint"]


def test_scorer_without_predict_with_meta_returns_insufficient():
    class _LegacyScalarOnly:
        is_trained = True

        def predict(self, **kw):
            return 0.0

    report = mod.offdist_rate_report(_LegacyScalarOnly(), [_row()] * 50)
    assert report["verdict"] == "INSUFFICIENT_DATA"
    assert "predict_with_meta" in report["hint"]


def test_empty_rows_returns_insufficient_data():
    report = mod.offdist_rate_report(_FakeScorer([]), [])
    assert report["verdict"] == "INSUFFICIENT_DATA"
    assert report["n"] == 0


def test_non_dict_rows_silently_skipped():
    envelopes = [_envelope(raw=1.0, pred=1.0)] * 30
    rows = [_row()] * 30 + ["not a dict", 12345, None]  # 33 inputs, 30 valid
    report = mod.offdist_rate_report(_FakeScorer(envelopes), rows)
    # Only 30 dict rows are processed.
    assert report["n"] == 30


# ---------------------------- analyze() integration ---------------------

def test_analyze_missing_outcomes_returns_error(tmp_path):
    p = tmp_path / "no_such_outcomes.jsonl"
    report = mod.analyze(p)
    assert report["status"] == "error"
    assert "missing" in report["error"]


def test_analyze_full_corpus_runs(tmp_path, monkeypatch):
    """Smoke-test the end-to-end analyze() path on a tiny synthetic
    outcomes file. Uses the real DecisionScorer load path — when the
    pickle is absent (conftest redirects SCORER_PATH to tmp), the
    scorer is unloaded and analyze() reports INSUFFICIENT_DATA cleanly."""
    p = tmp_path / "outcomes.jsonl"
    rows = []
    for i in range(40):
        rows.append({
            "ticker": "NVDA", "sim_date": f"2024-{(i % 12) + 1:02d}-01",
            "ml_score": 1.0 + i * 0.01, "rsi": 50.0, "macd": 0.0,
            "mom5": 0.0, "mom20": 0.0, "regime_mult": 1.0,
            "forward_return_5d": 0.0, "action": "BUY",
        })
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # No scorer pickle in the conftest tmp dir → DecisionScorer is
    # untrained → INSUFFICIENT_DATA verdict (the clean degrade path,
    # not a crash).
    report = mod.analyze(p, oos_only=False)
    assert report["status"] == "ok"
    assert report["verdict"] == "INSUFFICIENT_DATA"


def test_analyze_temporal_split_uses_20pct_tail(tmp_path):
    """OOS-only honors split_outcomes_temporal — tail 20% of dates is
    held out for the OOS slice."""
    p = tmp_path / "outcomes.jsonl"
    # 100 rows with monotone sim_date; OOS slice = last 20.
    rows = [{
        "ticker": "NVDA", "sim_date": f"2024-01-{(i % 28) + 1:02d}",
        "ml_score": 1.0, "rsi": 50.0, "macd": 0.0,
        "mom5": 0.0, "mom20": 0.0, "regime_mult": 1.0,
        "forward_return_5d": 0.0, "action": "BUY",
    } for i in range(100)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    report = mod.analyze(p, oos_only=True)
    # Scorer pickle absent → INSUFFICIENT_DATA but oos_only flag persists.
    assert report["status"] == "ok"
    assert report["oos_only"] is True


# ---------------------------- safety helpers ----------------------------

def test_maybe_float_handles_edge_cases():
    assert mod._maybe_float(None) is None
    assert mod._maybe_float(True) is None    # bool excluded
    assert mod._maybe_float(False) is None
    assert mod._maybe_float(float('nan')) is None
    assert mod._maybe_float(float('inf')) is None
    assert mod._maybe_float(float('-inf')) is None
    assert mod._maybe_float("not a number") is None
    assert mod._maybe_float(3) == 3.0
    assert mod._maybe_float(3.14) == 3.14
    assert mod._maybe_float("3.14") == 3.14


# ---------------------------- CLI ---------------------------------------

def test_cli_json_output_runs(tmp_path, capsys):
    p = tmp_path / "outcomes.jsonl"
    p.write_text("")  # empty file → analyze still runs cleanly
    rc = mod._cli(["--all", "--json", "--outcomes", str(p)])
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert out["status"] == "ok"
    assert out["verdict"] == "INSUFFICIENT_DATA"
    assert rc == 0


def test_cli_human_readable_runs(tmp_path, capsys):
    p = tmp_path / "outcomes.jsonl"
    p.write_text("")
    rc = mod._cli(["--all", "--outcomes", str(p)])
    captured = capsys.readouterr()
    assert "scorer_offdist_rate" in captured.out
    assert "verdict=INSUFFICIENT_DATA" in captured.out
    assert rc == 0


def test_cli_exits_one_on_missing_outcomes(tmp_path, capsys):
    """Missing file → status=error → exit code 1 (shell-gateable)."""
    rc = mod._cli(["--all", "--json", "--outcomes",
                   str(tmp_path / "nope.jsonl")])
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert out["status"] == "error"
    assert rc == 1
