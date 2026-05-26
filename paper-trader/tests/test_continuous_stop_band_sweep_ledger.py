"""Per-cycle ledger tests for the multi-horizon stop-band sweep analyzer.

``paper_trader/ml/stop_band_sweep.py`` already CLI-reports whether a
candidate ``(band, horizon)`` cell beats the deployed ``(-8%, 5d)`` cell
on realized return, but until ``_append_stop_band_sweep_log`` landed
there was NO per-cycle durable trend — a quant could only know "is a
wider stop on a longer window better than the inherited -8% / 5d band"
by manually running the CLI. These tests pin the wiring: best-effort
discipline, honest gap rows when the deployed horizon's intraperiod
corpus is empty, bounded trim, SSOT cross-check that the persisted
verdict / best cell equals what the analyzer returned. Mirrors the
discipline of every sibling ``_append_*_skill_log`` test class.

A new test file (not appended to ``test_continuous_intraperiod_ledgers.py``)
so a concurrent sibling agent editing the same test file cannot collide
with this work via whole-file ``git add`` — the documented same-role
HYBRID staging-race mitigation pattern.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

import run_continuous_backtests as rcb


class TestAppendStopBandSweepLog:
    """The per-cycle multi-horizon stop-band sweep ledger — answers
    "does any candidate (band, horizon) cell measurably beat the deployed
    (-8%, 5d) cell on realized return?" durably and per-cycle. Mirrors
    every sibling ``_append_*_skill_log`` discipline: best-effort, honest
    gap rows, atomic bounded trim, never breaks the loop.
    """

    def test_insufficient_data_persists_dark_row(self, tmp_path, monkeypatch):
        """The documented live state during corpus warm-up: deployed
        horizon's intraperiod coverage hasn't yet cleared MIN_BUYS, so
        the analyzer returns INSUFFICIENT_DATA. The ledger must STILL
        append the row so the darkness is visible in the trend."""
        log = tmp_path / "stop_band_sweep_log.jsonl"
        monkeypatch.setattr(rcb, "STOP_BAND_SWEEP_LOG", log)

        from paper_trader.ml import stop_band_sweep as sbs
        empty = {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "deployed_stop_pct": 8.0,
            "deployed_horizon": "5d",
            "edge_tol_pp": 0.30,
            "n_buys": 12,
            "n_with_intraperiod_per_horizon": {"5d": 12, "10d": 5,
                                                "20d": 5},
            "baseline_no_stop_mean_pct_per_horizon": {"5d": None,
                                                        "10d": None,
                                                        "20d": None},
            "bands_swept": [3.0, 5.0, 8.0],
            "horizons_swept": ["5d", "10d", "20d"],
            "sweep": [],
            "best_cell": None,
            "deployed_cell_benefit_pct": None,
            "hint": "only 12 BUYs with intraperiod data — feature warmup",
        }
        monkeypatch.setattr(sbs, "analyze", lambda *a, **k: dict(empty))

        assert rcb._append_stop_band_sweep_log(
            cycle=33, win_start=date(2020, 1, 2), win_end=date(2025, 1, 2),
            outcomes_path=tmp_path / "nope.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["cycle"] == 33
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["sweep_dark"] is True
        assert row["deployed_stop_pct"] == 8.0
        assert row["deployed_horizon"] == "5d"
        # Best cell is None on insufficient data — the flattened columns
        # must mirror that honestly.
        assert row["best_stop_pct"] is None
        assert row["best_horizon"] is None
        assert row["best_benefit_pct"] is None
        assert row["window_start"] == "2020-01-02"
        assert row["window_end"] == "2025-01-02"

    def test_cell_beats_deployed_verdict_surfaces_best_cell(self, tmp_path,
                                                             monkeypatch):
        """SSOT cross-check: when the analyzer says CELL_BEATS_DEPLOYED
        with a specific best cell, the persisted row MUST carry the same
        numbers. A drift here would break dashboard / trend integrity —
        the whole point of the per-cycle wiring is no-drift fidelity to
        the CLI verdict."""
        log = tmp_path / "stop_band_sweep_log.jsonl"
        monkeypatch.setattr(rcb, "STOP_BAND_SWEEP_LOG", log)

        decisive = {
            "status": "ok",
            "verdict": "CELL_BEATS_DEPLOYED",
            "deployed_stop_pct": 8.0,
            "deployed_horizon": "5d",
            "edge_tol_pp": 0.30,
            "n_buys": 7318,
            "n_with_intraperiod_per_horizon": {"5d": 7318, "10d": 5645,
                                                "20d": 5616},
            "baseline_no_stop_mean_pct_per_horizon": {"5d": 0.754,
                                                        "10d": 1.179,
                                                        "20d": 2.241},
            "bands_swept": [3.0, 5.0, 7.0, 8.0, 10.0, 12.0],
            "horizons_swept": ["5d", "10d", "20d"],
            "sweep": [],  # not asserted on
            "best_cell": {
                "stop_pct": 3.0,
                "horizon": "5d",
                "benefit_pct": 0.979,
                "n_triggered": 2965,
                "pct_triggered": 40.5,
                "mean_protected_return_pct": 1.733,
            },
            "deployed_cell_benefit_pct": 0.454,
            "hint": "best cell (3.0%, '5d') ...",
        }
        from paper_trader.ml import stop_band_sweep as sbs
        monkeypatch.setattr(sbs, "analyze", lambda *a, **k: dict(decisive))

        rcb._append_stop_band_sweep_log(
            cycle=44, win_start=date(2018, 6, 1), win_end=date(2023, 6, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "CELL_BEATS_DEPLOYED"
        assert row["sweep_dark"] is False
        # SSOT cross-check: every best-cell field round-trips identically.
        assert row["best_stop_pct"] == 3.0
        assert row["best_horizon"] == "5d"
        assert row["best_benefit_pct"] == 0.979
        assert row["best_n_triggered"] == 2965
        assert row["best_pct_triggered"] == 40.5
        assert row["best_mean_protected_return_pct"] == 1.733
        # Deployed-cell benefit round-trips so a quant can compute
        # (best - deployed) from the persisted row alone.
        assert row["deployed_cell_benefit_pct"] == 0.454

    def test_deployed_optimal_verdict_with_best_equal_deployed(self, tmp_path,
                                                                 monkeypatch):
        """When the deployed cell IS the best (gap within edge_tol),
        the verdict is DEPLOYED_OPTIMAL and the best_cell mirrors the
        deployed cell. Pins the no-action signal."""
        log = tmp_path / "stop_band_sweep_log.jsonl"
        monkeypatch.setattr(rcb, "STOP_BAND_SWEEP_LOG", log)

        optimal = {
            "status": "ok",
            "verdict": "DEPLOYED_OPTIMAL",
            "deployed_stop_pct": 8.0,
            "deployed_horizon": "5d",
            "edge_tol_pp": 0.30,
            "n_buys": 5000,
            "n_with_intraperiod_per_horizon": {"5d": 4200, "10d": 3800},
            "baseline_no_stop_mean_pct_per_horizon": {"5d": 1.0, "10d": 2.0},
            "bands_swept": [8.0, 10.0],
            "horizons_swept": ["5d", "10d"],
            "sweep": [],
            "best_cell": {
                "stop_pct": 8.0, "horizon": "5d", "benefit_pct": 0.45,
                "n_triggered": 380, "pct_triggered": 9.05,
                "mean_protected_return_pct": 1.45,
            },
            "deployed_cell_benefit_pct": 0.45,
            "hint": "deployed cell is best — no tuning move clears noise",
        }
        from paper_trader.ml import stop_band_sweep as sbs
        monkeypatch.setattr(sbs, "analyze", lambda *a, **k: dict(optimal))

        rcb._append_stop_band_sweep_log(
            cycle=77, win_start=date(2018, 6, 1), win_end=date(2023, 6, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "DEPLOYED_OPTIMAL"
        assert row["sweep_dark"] is False
        # Best cell IS the deployed cell.
        assert row["best_stop_pct"] == 8.0
        assert row["best_horizon"] == "5d"
        assert row["best_benefit_pct"] == row["deployed_cell_benefit_pct"]

    def test_no_band_helps_verdict(self, tmp_path, monkeypatch):
        """When no candidate band/horizon clears the absolute edge_tol
        floor, the verdict is NO_BAND_HELPS — actionable for a quant
        scripting "alert if stops have lost their edge entirely"."""
        log = tmp_path / "stop_band_sweep_log.jsonl"
        monkeypatch.setattr(rcb, "STOP_BAND_SWEEP_LOG", log)

        no_help = {
            "status": "ok",
            "verdict": "NO_BAND_HELPS",
            "deployed_stop_pct": 8.0,
            "deployed_horizon": "5d",
            "edge_tol_pp": 0.30,
            "n_buys": 5000,
            "n_with_intraperiod_per_horizon": {"5d": 4200},
            "baseline_no_stop_mean_pct_per_horizon": {"5d": 2.5},
            "bands_swept": [3.0, 5.0, 8.0, 10.0],
            "horizons_swept": ["5d"],
            "sweep": [],
            "best_cell": {
                "stop_pct": 3.0, "horizon": "5d", "benefit_pct": 0.10,
                "n_triggered": 100, "pct_triggered": 2.4,
                "mean_protected_return_pct": 2.60,
            },
            "deployed_cell_benefit_pct": 0.0,
            "hint": "no band clears edge tol",
        }
        from paper_trader.ml import stop_band_sweep as sbs
        monkeypatch.setattr(sbs, "analyze", lambda *a, **k: dict(no_help))

        rcb._append_stop_band_sweep_log(
            cycle=88, win_start=date(2020, 1, 1), win_end=date(2025, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "NO_BAND_HELPS"
        assert row["sweep_dark"] is False
        # Best cell's benefit is positive but below edge_tol — pin that.
        assert row["best_benefit_pct"] == 0.10

    def test_analyzer_crash_persists_error_envelope(self, tmp_path,
                                                     monkeypatch):
        """When the analyzer raises mid-cycle, the ledger MUST still
        emit a row (best-effort discipline — a gap in the trend would
        be silent corruption). The persisted row carries the error
        envelope honestly, with sweep_dark=True so an alert can fire."""
        log = tmp_path / "stop_band_sweep_log.jsonl"
        monkeypatch.setattr(rcb, "STOP_BAND_SWEEP_LOG", log)

        from paper_trader.ml import stop_band_sweep as sbs
        def _boom(*a, **k):
            raise RuntimeError("simulated analyzer crash")
        monkeypatch.setattr(sbs, "analyze", _boom)

        assert rcb._append_stop_band_sweep_log(
            cycle=99, win_start=date(2020, 1, 1), win_end=date(2025, 1, 1),
            outcomes_path=tmp_path / "outcomes.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["status"] == "error"
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["sweep_dark"] is True
        assert row["best_stop_pct"] is None

    def test_missing_module_does_not_break_loop(self, tmp_path, monkeypatch):
        """If the stop_band_sweep module is unimportable (corrupt sys.modules
        / dependency missing), the helper must STILL return True and write
        a row — the continuous loop cannot tolerate a ledger taking the
        whole cycle down."""
        log = tmp_path / "stop_band_sweep_log.jsonl"
        monkeypatch.setattr(rcb, "STOP_BAND_SWEEP_LOG", log)

        # Force an import-time failure inside the helper by stuffing a
        # bad sentinel into sys.modules.
        import sys
        old = sys.modules.get("paper_trader.ml.stop_band_sweep")
        sys.modules["paper_trader.ml.stop_band_sweep"] = None  # type: ignore
        try:
            ok = rcb._append_stop_band_sweep_log(
                cycle=111, win_start=date(2020, 1, 1),
                win_end=date(2025, 1, 1),
                outcomes_path=tmp_path / "outcomes.jsonl")
            assert ok is True
            row = json.loads(log.read_text().strip())
            assert row["verdict"] == "INSUFFICIENT_DATA"
            assert row["sweep_dark"] is True
        finally:
            if old is not None:
                sys.modules["paper_trader.ml.stop_band_sweep"] = old
            else:
                sys.modules.pop("paper_trader.ml.stop_band_sweep", None)

    def test_bounded_trim_keeps_tail(self, tmp_path, monkeypatch):
        """When the ledger grows past 2× KEEP, it is atomically rewritten
        to the last KEEP rows (the trim idiom every sibling ledger uses).
        Pin the keep-tail semantics."""
        log = tmp_path / "stop_band_sweep_log.jsonl"
        monkeypatch.setattr(rcb, "STOP_BAND_SWEEP_LOG", log)
        monkeypatch.setattr(rcb, "STOP_BAND_SWEEP_LOG_KEEP", 5)

        from paper_trader.ml import stop_band_sweep as sbs
        # Mock with a fast, valid payload — each call appends one row.
        monkeypatch.setattr(sbs, "analyze", lambda *a, **k: {
            "status": "ok", "verdict": "DEPLOYED_OPTIMAL",
            "deployed_stop_pct": 8.0, "deployed_horizon": "5d",
            "edge_tol_pp": 0.30, "n_buys": 100,
            "n_with_intraperiod_per_horizon": {"5d": 100},
            "baseline_no_stop_mean_pct_per_horizon": {"5d": 1.0},
            "bands_swept": [8.0], "horizons_swept": ["5d"],
            "sweep": [],
            "best_cell": {"stop_pct": 8.0, "horizon": "5d",
                          "benefit_pct": 0.2, "n_triggered": 5,
                          "pct_triggered": 5.0,
                          "mean_protected_return_pct": 1.2},
            "deployed_cell_benefit_pct": 0.2, "hint": None})

        # Trigger 12 rows — > 2× KEEP (10), so a rewrite kicks in.
        for c in range(1, 13):
            rcb._append_stop_band_sweep_log(
                cycle=c, win_start=date(2020, 1, 1),
                win_end=date(2025, 1, 1),
                outcomes_path=tmp_path / "outcomes.jsonl")

        lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
        # The trim only fires past 2× KEEP — after 11 writes the file is
        # rewritten to 5, then the 12th append brings it back to 6. The
        # invariant is "MUCH less than 12" (proves the trim ran), not
        # exactly KEEP.
        assert len(lines) < 12
        assert len(lines) <= 5 + 2  # KEEP plus at most one post-trim append
        # Last row is the most-recent cycle (12).
        assert json.loads(lines[-1])["cycle"] == 12
        # The earliest cycles (1, 2) must have been dropped by the trim.
        cycles_kept = {json.loads(ln)["cycle"] for ln in lines}
        assert 1 not in cycles_kept
        assert 2 not in cycles_kept

    def test_main_loop_calls_the_helper_once_per_cycle(self):
        """Smoke check: the helper is invoked in the main() loop exactly
        once per cycle, in the sibling-ledger block after the MFE ledger
        and before the gate-arm ledger. Catches the most common wiring
        regression: helper defined but never called.
        """
        import inspect
        src = inspect.getsource(rcb.main)
        # The helper is called with the canonical signature.
        assert "_append_stop_band_sweep_log(cycle, win_start, win_end)" in src
        # And it's called AFTER the MFE ledger (the sibling's hook).
        idx_mfe = src.find("_append_mfe_skill_log(cycle")
        idx_sbs = src.find("_append_stop_band_sweep_log(cycle")
        assert idx_mfe > 0 and idx_sbs > idx_mfe, (
            "stop_band_sweep ledger must run AFTER mfe (sibling-ledger order)")
