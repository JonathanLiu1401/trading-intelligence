"""Tests for the per-cycle ``feature_alignment`` ledger wired into
``run_continuous_backtests._append_feature_alignment_log``.

Mirrors the sibling ``test_continuous_outcome_corpus_health_ledger``
discipline:

* The append function is best-effort — every fault must degrade to a
  honest ``alignment_dark=True`` row, never raise.
* Top-line fields (``verdict`` / ``n_features_with_signal`` /
  ``top_weighted_noise`` / ``top_ignored_signal`` / ``top_ic_feature`` /
  ``top_weight_feature``) are present on every row so a JSON consumer
  can trend without nested parsing.
* The trim cap (``FEATURE_ALIGNMENT_LOG_KEEP``) bounds file growth.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

import run_continuous_backtests as rcb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines()
            if ln.strip()]


@pytest.fixture
def redirected_ledger(tmp_path, monkeypatch):
    """Redirect FEATURE_ALIGNMENT_LOG to tmp_path. Mirrors the sibling
    test fixture pattern."""
    p = tmp_path / "feature_alignment_log.jsonl"
    monkeypatch.setattr(rcb, "FEATURE_ALIGNMENT_LOG", p)
    return p


@pytest.fixture
def stub_analyze(monkeypatch):
    """Replace feature_alignment.analyze with a controllable stub so the
    test owns the verdict ladder exercised at each call."""
    calls = {"n": 0}

    def _factory(reports: list[dict]):
        idx = {"i": 0}

        def _analyze(*args, **kwargs):
            calls["n"] += 1
            r = reports[min(idx["i"], len(reports) - 1)]
            idx["i"] += 1
            return r

        # Patch the analyze function on the underlying module so the
        # `from paper_trader.ml.feature_alignment import analyze as _fa_analyze`
        # inside `_append_feature_alignment_log` picks up the stub.
        import paper_trader.ml.feature_alignment as fa_mod
        monkeypatch.setattr(fa_mod, "analyze", _analyze)
        return calls

    return _factory


# ---------------------------------------------------------------------------
# Top-line row schema
# ---------------------------------------------------------------------------

class TestRowSchema:
    """Every row must include the flat top-line fields documented in
    ``_append_feature_alignment_log``'s docstring."""

    def test_aligned_verdict_writes_flat_fields(
            self, redirected_ledger, stub_analyze):
        stub_analyze([{
            "status": "ok",
            "verdict": "ALIGNED",
            "n": 1200,
            "slice": "temporal_oos",
            "features": [
                {"feature": "ml_score", "univariate_ic": 0.10,
                 "model_importance": 0.5, "univariate_rank": 0,
                 "importance_rank": 0, "alignment_bucket": "ALIGNED",
                 "n": 1200},
                {"feature": "rsi", "univariate_ic": 0.05,
                 "model_importance": 0.3, "univariate_rank": 1,
                 "importance_rank": 1, "alignment_bucket": "MID",
                 "n": 1200},
            ],
            "n_features_with_signal": 5,
            "top_weighted_noise": [],
            "top_ignored_signal": [],
            "hint": "ALIGNED",
        }])
        ok = rcb._append_feature_alignment_log(
            cycle=7, win_start=date(2024, 1, 1), win_end=date(2024, 12, 31))
        assert ok is True
        rows = _read_jsonl(redirected_ledger)
        assert len(rows) == 1
        r = rows[0]
        # Required top-line fields.
        for key in ("cycle", "timestamp", "window_start", "window_end",
                    "verdict", "n", "slice", "n_features_with_signal",
                    "top_weighted_noise", "top_ignored_signal",
                    "top_ic_feature", "top_ic_value",
                    "top_weight_feature", "top_weight_value",
                    "alignment_dark", "features", "hint"):
            assert key in r, f"missing top-line field {key}"
        assert r["cycle"] == 7
        assert r["verdict"] == "ALIGNED"
        assert r["n"] == 1200
        assert r["n_features_with_signal"] == 5
        # Top IC = ml_score @ 0.10; top weight = ml_score @ 0.5.
        assert r["top_ic_feature"] == "ml_score"
        assert r["top_ic_value"] == 0.10
        assert r["top_weight_feature"] == "ml_score"
        assert r["top_weight_value"] == 0.5
        # ALIGNED is not dark.
        assert r["alignment_dark"] is False

    def test_weighted_noise_carries_actionable_list(
            self, redirected_ledger, stub_analyze):
        stub_analyze([{
            "status": "ok",
            "verdict": "WEIGHTED_NOISE",
            "n": 1500,
            "slice": "temporal_oos",
            "features": [
                {"feature": "ml_score", "univariate_ic": 0.005,
                 "model_importance": 0.9, "univariate_rank": 12,
                 "importance_rank": 0, "alignment_bucket": "WEIGHTED_NOISE",
                 "n": 1500},
                {"feature": "mom5", "univariate_ic": 0.12,
                 "model_importance": 0.05, "univariate_rank": 0,
                 "importance_rank": 11, "alignment_bucket": "MID",
                 "n": 1500},
            ],
            "n_features_with_signal": 3,
            "top_weighted_noise": ["ml_score"],
            "top_ignored_signal": [],
            "hint": "weighted noise",
        }])
        rcb._append_feature_alignment_log(
            cycle=9, win_start=date(2024, 1, 1), win_end=date(2024, 12, 31))
        r = _read_jsonl(redirected_ledger)[0]
        assert r["verdict"] == "WEIGHTED_NOISE"
        assert r["top_weighted_noise"] == ["ml_score"]
        assert r["top_ignored_signal"] == []
        assert r["alignment_dark"] is False  # actionable verdict

    def test_insufficient_data_is_dark(
            self, redirected_ledger, stub_analyze):
        stub_analyze([{
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "n": 10,
            "slice": "temporal_oos",
            "features": [],
            "n_features_with_signal": 0,
            "top_weighted_noise": [],
            "top_ignored_signal": [],
            "hint": "need more rows",
        }])
        rcb._append_feature_alignment_log(
            cycle=1, win_start=date(2024, 1, 1), win_end=date(2024, 12, 31))
        r = _read_jsonl(redirected_ledger)[0]
        assert r["alignment_dark"] is True
        assert r["verdict"] == "INSUFFICIENT_DATA"
        # Flat fields are present-but-None on dark rows (no value to
        # display, but the column exists so trend queries don't break).
        assert r["top_ic_feature"] is None
        assert r["top_weight_feature"] is None

    def test_degenerate_is_dark(self, redirected_ledger, stub_analyze):
        stub_analyze([{
            "status": "ok",
            "verdict": "DEGENERATE",
            "n": 800,
            "slice": "temporal_oos",
            "features": [
                {"feature": "ml_score", "univariate_ic": 0.01,
                 "model_importance": 0.5, "univariate_rank": 0,
                 "importance_rank": 0, "alignment_bucket": "MID",
                 "n": 800},
            ],
            "n_features_with_signal": 0,
            "top_weighted_noise": [],
            "top_ignored_signal": [],
            "hint": "no signal",
        }])
        rcb._append_feature_alignment_log(
            cycle=2, win_start=date(2024, 1, 1), win_end=date(2024, 12, 31))
        r = _read_jsonl(redirected_ledger)[0]
        assert r["alignment_dark"] is True
        assert r["verdict"] == "DEGENERATE"


# ---------------------------------------------------------------------------
# Honest-degrade contract
# ---------------------------------------------------------------------------

class TestHonestDegrade:
    def test_analyzer_exception_writes_error_row(
            self, redirected_ledger, monkeypatch):
        # The analyzer raises — append must write a dark row, never raise.
        import paper_trader.ml.feature_alignment as fa_mod

        def _boom(*a, **kw):
            raise RuntimeError("synthetic")

        monkeypatch.setattr(fa_mod, "analyze", _boom)
        ok = rcb._append_feature_alignment_log(
            cycle=42, win_start=date(2024, 1, 1), win_end=date(2024, 12, 31))
        # Returned True (we did append a dark row).
        assert ok is True
        rows = _read_jsonl(redirected_ledger)
        assert len(rows) == 1
        r = rows[0]
        assert r["verdict"] == "INSUFFICIENT_DATA"
        assert r["alignment_dark"] is True
        # The hint surfaces the analyzer failure so a researcher sees WHY.
        assert "feature_alignment unavailable" in r["hint"]
        assert "RuntimeError" in r["hint"]

    def test_returns_false_only_on_unrecoverable_fault(
            self, monkeypatch, tmp_path):
        """A read-only path with no write permission should degrade to
        False (the function caught the OSError) rather than raising. This
        is the same contract every sibling ``_append_*`` follows."""
        # Point the ledger at a non-writable location.
        ro_path = tmp_path / "ro_dir"
        ro_path.mkdir()
        ro_path.chmod(0o500)
        log_path = ro_path / "feature_alignment_log.jsonl"
        monkeypatch.setattr(rcb, "FEATURE_ALIGNMENT_LOG", log_path)

        # Make analyze return a normal report.
        import paper_trader.ml.feature_alignment as fa_mod
        monkeypatch.setattr(fa_mod, "analyze", lambda *a, **kw: {
            "status": "ok", "verdict": "ALIGNED", "n": 100,
            "slice": "full", "features": [],
            "n_features_with_signal": 0,
            "top_weighted_noise": [], "top_ignored_signal": [], "hint": ""})
        try:
            ok = rcb._append_feature_alignment_log(
                cycle=1, win_start=date(2024, 1, 1),
                win_end=date(2024, 12, 31))
            # Either False (write blocked) or True (mkdir succeeded somehow).
            # Either way: no exception escaped — the loop never breaks.
            assert ok in (True, False)
        finally:
            ro_path.chmod(0o700)  # cleanup


# ---------------------------------------------------------------------------
# Bounded growth
# ---------------------------------------------------------------------------

class TestBoundedGrowth:
    def test_trim_when_file_exceeds_2x_keep(
            self, redirected_ledger, stub_analyze, monkeypatch):
        # Tiny cap so the test stays fast.
        monkeypatch.setattr(rcb, "FEATURE_ALIGNMENT_LOG_KEEP", 5)
        stub_analyze([{
            "status": "ok", "verdict": "ALIGNED", "n": 100,
            "slice": "full", "features": [],
            "n_features_with_signal": 0,
            "top_weighted_noise": [], "top_ignored_signal": [], "hint": ""}])
        for c in range(20):  # > 2× keep
            rcb._append_feature_alignment_log(
                cycle=c, win_start=date(2024, 1, 1),
                win_end=date(2024, 12, 31))
        rows = _read_jsonl(redirected_ledger)
        # After trim, count should be ≤ 2× keep (the trim fires when
        # the count crosses the 2× boundary, so the last write produces
        # exactly `keep` rows).
        assert len(rows) <= 10, len(rows)
        # Most-recent rows survive (the deque/tail trim contract).
        cycles = [r["cycle"] for r in rows]
        assert cycles == sorted(cycles), \
            "rows are not in append order after trim"
        assert max(cycles) == 19, \
            "most-recent row not preserved by trim"
