"""Per-cycle ledger tests for the intraperiod stop-out / take-profit
analyzers.

``paper_trader/ml/stop_out_audit.py`` and ``paper_trader/ml/mfe_conversion.py``
already CLI-report the realized economic effect of the inherited
``backtest._buy`` ``stop_loss = price * 0.92`` / ``take_profit = price * 1.15``
bands, on top of the ``forward_intraperiod_min_5d`` /
``forward_intraperiod_max_5d`` columns ``_compute_decision_outcomes`` started
persisting on 2026-05-23. Until ``_append_stop_out_skill_log`` /
``_append_mfe_skill_log`` landed there was NO per-cycle durable trend for
either CLI verdict — a quant could only know "is the inherited band a real
defensive arm or variance-only chop" by manually running the CLI. These
tests pin the wiring: best-effort discipline, honest gap rows when the
intraperiod corpus is empty (the documented current state), bounded trim,
SSOT cross-check that the persisted verdict equals what the analyzer
returned. Mirrors the discipline of every sibling ``_append_*_skill_log`` /
``_append_*_calibration_log`` test class in ``test_continuous.py``.

A new test file (not appended to ``test_continuous.py``) so a concurrent
sibling agent editing the same test file cannot collide with this work via
whole-file ``git add`` — the documented same-role HYBRID staging-race
mitigation pattern.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

import run_continuous_backtests as rcb


# ─────────────────────── _append_stop_out_skill_log ───────────────────────


class TestAppendStopOutSkillLog:
    """The per-cycle stop-out skill ledger — answers "is the inherited
    -8% stop_loss band a real defensive arm, or variance-only chop the
    gate would do better without?" durably and per-cycle so a quant can
    trend the realized economic benefit. Mirrors the sibling
    ``_append_calibrated_reliability_log`` / ``_append_conviction_calibration_log``
    discipline: best-effort, honest gap rows, atomic bounded trim, never
    breaks the loop.
    """

    def test_insufficient_data_persists_dark_row(self, tmp_path, monkeypatch):
        """The documented live state: historical corpus pre-dates the
        2026-05-23 ``forward_intraperiod_min_5d`` feature, so
        ``stop_out_audit.analyze`` returns INSUFFICIENT_DATA. The ledger
        must STILL append the row so the darkness is visible in the trend
        — exactly how the calibrated_reliability ledger surfaces its own
        legacy-pickle ``calibrated_dark`` state.
        """
        log = tmp_path / "stop_out_skill_log.jsonl"
        monkeypatch.setattr(rcb, "STOP_OUT_SKILL_LOG", log)

        from paper_trader.ml import stop_out_audit as soa
        empty = {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "stop_pct": 8.0,
            "n_buys": 8753,
            "n_with_intraperiod": 0,
            "n_stop_triggered": 0,
            "pct_stop_triggered": None,
            "mean_realized_return_pct": None,
            "mean_stop_protected_return_pct": None,
            "stop_benefit_pct": None,
            "median_realized_return_pct": None,
            "median_stop_protected_return_pct": None,
            "hint": "older outcome rows predate forward_intraperiod_* feature",
        }
        monkeypatch.setattr(soa, "analyze", lambda *a, **k: dict(empty))

        assert rcb._append_stop_out_skill_log(
            cycle=33, win_start=date(2020, 1, 2), win_end=date(2025, 1, 2),
            outcomes_path=tmp_path / "nope.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["cycle"] == 33
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["stop_dark"] is True  # the mirror of calibrated_dark
        assert row["n_with_intraperiod"] == 0
        assert row["n_buys"] == 8753
        assert row["stop_pct"] == 8.0
        assert row["window_start"] == "2020-01-02"
        assert row["window_end"] == "2025-01-02"

    def test_stop_helps_verdict_surfaces_benefit(self, tmp_path, monkeypatch):
        """SSOT cross-check: when the analyzer says STOP_HELPS with a +0.45pp
        benefit, the persisted row MUST carry the same number. A drift here
        would silently break dashboard / trend integrity — the whole point
        of the per-cycle wiring is no-drift fidelity to the CLI verdict.
        """
        log = tmp_path / "stop_out_skill_log.jsonl"
        monkeypatch.setattr(rcb, "STOP_OUT_SKILL_LOG", log)

        helpful = {
            "status": "ok",
            "verdict": "STOP_HELPS",
            "stop_pct": 8.0,
            "benefit_margin_pp": 0.30,
            "n_buys": 5000,
            "n_with_intraperiod": 4200,
            "n_stop_triggered": 380,
            "pct_stop_triggered": 9.05,
            "mean_realized_return_pct": 1.234,
            "mean_stop_protected_return_pct": 1.684,
            "stop_benefit_pct": 0.45,
            "median_realized_return_pct": 0.55,
            "median_stop_protected_return_pct": 0.55,
            "hint": None,
        }
        from paper_trader.ml import stop_out_audit as soa
        monkeypatch.setattr(soa, "analyze", lambda *a, **k: dict(helpful))

        rcb._append_stop_out_skill_log(
            cycle=44, win_start=date(2018, 6, 1), win_end=date(2023, 6, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "STOP_HELPS"
        assert row["stop_dark"] is False
        assert row["stop_benefit_pct"] == 0.45  # SSOT cross-check
        assert row["n_with_intraperiod"] == 4200
        assert row["n_stop_triggered"] == 380
        assert row["pct_stop_triggered"] == 9.05
        assert row["mean_realized_return_pct"] == 1.234
        assert row["mean_stop_protected_return_pct"] == 1.684

    def test_stop_hurts_verdict_with_negative_benefit(self, tmp_path,
                                                       monkeypatch):
        """When STOP_HURTS, the benefit is negative — a quant cares about
        the SIGN as the actionable signal. Sign-preserving round-trip
        through the ledger is the invariant the test pins.
        """
        log = tmp_path / "stop_out_skill_log.jsonl"
        monkeypatch.setattr(rcb, "STOP_OUT_SKILL_LOG", log)

        hurtful = {
            "status": "ok",
            "verdict": "STOP_HURTS",
            "stop_pct": 8.0,
            "benefit_margin_pp": 0.30,
            "n_buys": 5000,
            "n_with_intraperiod": 4200,
            "n_stop_triggered": 1200,
            "pct_stop_triggered": 28.57,
            "mean_realized_return_pct": 2.5,
            "mean_stop_protected_return_pct": 1.7,
            "stop_benefit_pct": -0.80,
            "median_realized_return_pct": 1.0,
            "median_stop_protected_return_pct": 1.0,
            "hint": None,
        }
        from paper_trader.ml import stop_out_audit as soa
        monkeypatch.setattr(soa, "analyze", lambda *a, **k: dict(hurtful))

        rcb._append_stop_out_skill_log(
            cycle=55, win_start=date(2018, 6, 1), win_end=date(2023, 6, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "STOP_HURTS"
        assert row["stop_benefit_pct"] == -0.80  # negative sign preserved
        assert row["stop_dark"] is False

    def test_analyze_exception_degrades_to_honest_row_not_crash(
            self, tmp_path, monkeypatch):
        """A ledger write must NEVER break the continuous loop — the
        documented best-effort discipline of every sibling
        ``_append_*_skill_log`` function. If ``stop_out_audit.analyze``
        raises (e.g. corrupted outcomes file the analyzer can't iterate),
        the ledger must still persist an honest INSUFFICIENT_DATA row
        instead of bubbling the exception.
        """
        log = tmp_path / "stop_out_skill_log.jsonl"
        monkeypatch.setattr(rcb, "STOP_OUT_SKILL_LOG", log)

        def _boom(*a, **k):
            raise RuntimeError("simulated stop_out_audit failure")

        from paper_trader.ml import stop_out_audit as soa
        monkeypatch.setattr(soa, "analyze", _boom)

        assert rcb._append_stop_out_skill_log(
            cycle=55, win_start=date(2012, 1, 3), win_end=date(2013, 1, 3),
            outcomes_path=tmp_path / "missing.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["status"] == "error"
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["stop_dark"] is True
        assert row["cycle"] == 55

    def test_bounded_trim_rewrites_when_past_2x_keep(self, tmp_path,
                                                      monkeypatch):
        """Bounded growth — when the file exceeds 2× the KEEP cap, the next
        append atomically rewrites the file to the last KEEP rows. Mirrors
        the sibling ``CALIBRATED_RELIABILITY_LOG`` / ``SCORER_SKILL_LOG``
        trim idiom exactly: only pay the rewrite when well past the cap,
        write tmp + ``.replace`` so a torn truncate cannot lose history.
        """
        log = tmp_path / "stop_out_skill_log.jsonl"
        monkeypatch.setattr(rcb, "STOP_OUT_SKILL_LOG", log)
        monkeypatch.setattr(rcb, "STOP_OUT_SKILL_LOG_KEEP", 4)
        # 9 pre-seeded rows (> 2×4) ⇒ next append triggers the rewrite.
        log.write_text("\n".join(json.dumps({"cycle": i}) for i in range(9))
                       + "\n")

        from paper_trader.ml import stop_out_audit as soa
        monkeypatch.setattr(soa, "analyze",
                            lambda *a, **k: {"status": "insufficient_data",
                                             "verdict": "INSUFFICIENT_DATA",
                                             "n_buys": 0,
                                             "n_with_intraperiod": 0})

        rcb._append_stop_out_skill_log(
            cycle=77, win_start=date(2012, 1, 3), win_end=date(2013, 1, 3),
            outcomes_path=tmp_path / "nope.jsonl")
        lines = [l for l in log.read_text().splitlines() if l.strip()]
        assert len(lines) == 4  # trimmed to STOP_OUT_SKILL_LOG_KEEP
        assert json.loads(lines[-1])["cycle"] == 77  # newest survives

    def test_directory_created_when_missing(self, tmp_path, monkeypatch):
        """The ledger's parent dir is created on first write — mirrors
        every sibling ``_append_*`` function. A fresh repo with no
        ``data/`` dir must not break the first cycle.
        """
        log = tmp_path / "nested" / "deep" / "stop_out_skill_log.jsonl"
        monkeypatch.setattr(rcb, "STOP_OUT_SKILL_LOG", log)

        from paper_trader.ml import stop_out_audit as soa
        monkeypatch.setattr(soa, "analyze",
                            lambda *a, **k: {"status": "insufficient_data",
                                             "verdict": "INSUFFICIENT_DATA",
                                             "n_buys": 0,
                                             "n_with_intraperiod": 0})

        assert rcb._append_stop_out_skill_log(
            cycle=1, win_start=date(2020, 1, 1), win_end=date(2020, 6, 1),
            outcomes_path=tmp_path / "nope.jsonl") is True
        assert log.exists()

    def test_never_raises_on_unwritable_path(self, tmp_path, monkeypatch):
        """Even when the parent path can't be created, the function MUST
        swallow the OSError and return False — never raise. Mirrors the
        discipline of every sibling ``_append_*_skill_log`` function.
        """
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        monkeypatch.setattr(rcb, "STOP_OUT_SKILL_LOG",
                            blocker / "sub" / "stop.jsonl")

        from paper_trader.ml import stop_out_audit as soa
        monkeypatch.setattr(soa, "analyze",
                            lambda *a, **k: {"status": "insufficient_data",
                                             "verdict": "INSUFFICIENT_DATA",
                                             "n_buys": 0,
                                             "n_with_intraperiod": 0})
        assert rcb._append_stop_out_skill_log(
            1, date(2000, 1, 3), date(2001, 1, 3),
            outcomes_path=tmp_path / "nope.jsonl") is False


# ─────────────────────── _append_mfe_skill_log ───────────────────────


class TestAppendMfeSkillLog:
    """The per-cycle MFE-conversion / take-profit skill ledger — answers
    "does the inherited +15% take_profit band capture more upside than it
    forfeits?" durably and per-cycle. Sibling to
    ``_append_stop_out_skill_log`` on the matching upside arm — same
    structure, same defensive discipline, same SSOT cross-check.
    """

    def test_insufficient_data_persists_dark_row(self, tmp_path, monkeypatch):
        """Same documented live state as the stop-out ledger: corpus
        pre-dates the intraperiod feature, analyzer returns
        INSUFFICIENT_DATA, ledger persists an honest gap row so the
        darkness is visible in the trend.
        """
        log = tmp_path / "mfe_skill_log.jsonl"
        monkeypatch.setattr(rcb, "MFE_SKILL_LOG", log)

        from paper_trader.ml import mfe_conversion as mfe
        empty = {
            "status": "insufficient_data",
            "verdict": "INSUFFICIENT_DATA",
            "tp_pct": 15.0,
            "n_buys": 8753,
            "n_with_intraperiod": 0,
            "n_tp_triggered": 0,
            "pct_tp_triggered": None,
            "n_positive_mfe": 0,
            "n_reverted": 0,
            "pct_reverted": None,
            "mean_realized_return_pct": None,
            "mean_tp_protected_return_pct": None,
            "tp_benefit_pct": None,
            "median_realized_return_pct": None,
            "median_tp_protected_return_pct": None,
            "mean_mfe_pct": None,
            "median_mfe_pct": None,
            "mean_conversion_ratio": None,
            "median_conversion_ratio": None,
            "hint": "older outcome rows predate forward_intraperiod_* feature",
        }
        monkeypatch.setattr(mfe, "analyze", lambda *a, **k: dict(empty))

        assert rcb._append_mfe_skill_log(
            cycle=33, win_start=date(2020, 1, 2), win_end=date(2025, 1, 2),
            outcomes_path=tmp_path / "nope.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["cycle"] == 33
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["tp_dark"] is True
        assert row["n_with_intraperiod"] == 0
        assert row["n_buys"] == 8753
        assert row["tp_pct"] == 15.0
        assert row["window_start"] == "2020-01-02"
        assert row["window_end"] == "2025-01-02"

    def test_tp_helps_verdict_surfaces_benefit_and_conversion(
            self, tmp_path, monkeypatch):
        """SSOT cross-check: when the analyzer says TP_HELPS, the
        persisted row carries the SAME benefit + the SAME MFE-conversion
        ratio. Conversion ratio is the textbook "fraction of peak captured"
        a TP-economic decision rides on — drifting it would silently
        mislead a quant evaluating the band.
        """
        log = tmp_path / "mfe_skill_log.jsonl"
        monkeypatch.setattr(rcb, "MFE_SKILL_LOG", log)

        helpful = {
            "status": "ok",
            "verdict": "TP_HELPS",
            "tp_pct": 15.0,
            "benefit_margin_pp": 0.30,
            "n_buys": 5000,
            "n_with_intraperiod": 4200,
            "n_tp_triggered": 220,
            "pct_tp_triggered": 5.24,
            "n_positive_mfe": 2800,
            "n_reverted": 1400,
            "pct_reverted": 50.0,
            "mean_realized_return_pct": 1.1,
            "mean_tp_protected_return_pct": 1.65,
            "tp_benefit_pct": 0.55,
            "median_realized_return_pct": 0.4,
            "median_tp_protected_return_pct": 0.4,
            "mean_mfe_pct": 3.7,
            "median_mfe_pct": 2.1,
            "mean_conversion_ratio": 0.42,
            "median_conversion_ratio": 0.5,
            "hint": None,
        }
        from paper_trader.ml import mfe_conversion as mfe
        monkeypatch.setattr(mfe, "analyze", lambda *a, **k: dict(helpful))

        rcb._append_mfe_skill_log(
            cycle=44, win_start=date(2018, 6, 1), win_end=date(2023, 6, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "TP_HELPS"
        assert row["tp_dark"] is False
        assert row["tp_benefit_pct"] == 0.55  # SSOT cross-check
        assert row["mean_conversion_ratio"] == 0.42  # SSOT cross-check
        assert row["median_conversion_ratio"] == 0.5
        assert row["mean_mfe_pct"] == 3.7
        assert row["pct_reverted"] == 50.0
        assert row["n_positive_mfe"] == 2800

    def test_tp_hurts_verdict_with_negative_benefit(self, tmp_path,
                                                     monkeypatch):
        """Sign-preserving round-trip: TP_HURTS ⇒ negative benefit ⇒ the
        persisted row carries a negative number with the SAME magnitude.
        """
        log = tmp_path / "mfe_skill_log.jsonl"
        monkeypatch.setattr(rcb, "MFE_SKILL_LOG", log)

        hurtful = {
            "status": "ok",
            "verdict": "TP_HURTS",
            "tp_pct": 15.0,
            "benefit_margin_pp": 0.30,
            "n_buys": 5000,
            "n_with_intraperiod": 4200,
            "n_tp_triggered": 800,
            "pct_tp_triggered": 19.05,
            "n_positive_mfe": 2800,
            "n_reverted": 900,
            "pct_reverted": 32.14,
            "mean_realized_return_pct": 2.5,
            "mean_tp_protected_return_pct": 1.7,
            "tp_benefit_pct": -0.80,
            "median_realized_return_pct": 1.0,
            "median_tp_protected_return_pct": 1.0,
            "mean_mfe_pct": 6.5,
            "median_mfe_pct": 4.0,
            "mean_conversion_ratio": 0.65,
            "median_conversion_ratio": 0.7,
            "hint": None,
        }
        from paper_trader.ml import mfe_conversion as mfe
        monkeypatch.setattr(mfe, "analyze", lambda *a, **k: dict(hurtful))

        rcb._append_mfe_skill_log(
            cycle=55, win_start=date(2018, 6, 1), win_end=date(2023, 6, 1),
            outcomes_path=tmp_path / "outcomes.jsonl")
        row = json.loads(log.read_text().strip())
        assert row["verdict"] == "TP_HURTS"
        assert row["tp_benefit_pct"] == -0.80  # negative sign preserved
        assert row["tp_dark"] is False

    def test_analyze_exception_degrades_to_honest_row_not_crash(
            self, tmp_path, monkeypatch):
        log = tmp_path / "mfe_skill_log.jsonl"
        monkeypatch.setattr(rcb, "MFE_SKILL_LOG", log)

        def _boom(*a, **k):
            raise RuntimeError("simulated mfe_conversion failure")

        from paper_trader.ml import mfe_conversion as mfe
        monkeypatch.setattr(mfe, "analyze", _boom)

        assert rcb._append_mfe_skill_log(
            cycle=55, win_start=date(2012, 1, 3), win_end=date(2013, 1, 3),
            outcomes_path=tmp_path / "missing.jsonl") is True
        row = json.loads(log.read_text().strip())
        assert row["status"] == "error"
        assert row["verdict"] == "INSUFFICIENT_DATA"
        assert row["tp_dark"] is True
        assert row["cycle"] == 55

    def test_bounded_trim_rewrites_when_past_2x_keep(self, tmp_path,
                                                      monkeypatch):
        log = tmp_path / "mfe_skill_log.jsonl"
        monkeypatch.setattr(rcb, "MFE_SKILL_LOG", log)
        monkeypatch.setattr(rcb, "MFE_SKILL_LOG_KEEP", 4)
        log.write_text("\n".join(json.dumps({"cycle": i}) for i in range(9))
                       + "\n")

        from paper_trader.ml import mfe_conversion as mfe
        monkeypatch.setattr(mfe, "analyze",
                            lambda *a, **k: {"status": "insufficient_data",
                                             "verdict": "INSUFFICIENT_DATA",
                                             "n_buys": 0,
                                             "n_with_intraperiod": 0})

        rcb._append_mfe_skill_log(
            cycle=77, win_start=date(2012, 1, 3), win_end=date(2013, 1, 3),
            outcomes_path=tmp_path / "nope.jsonl")
        lines = [l for l in log.read_text().splitlines() if l.strip()]
        assert len(lines) == 4
        assert json.loads(lines[-1])["cycle"] == 77

    def test_directory_created_when_missing(self, tmp_path, monkeypatch):
        log = tmp_path / "nested" / "deep" / "mfe_skill_log.jsonl"
        monkeypatch.setattr(rcb, "MFE_SKILL_LOG", log)

        from paper_trader.ml import mfe_conversion as mfe
        monkeypatch.setattr(mfe, "analyze",
                            lambda *a, **k: {"status": "insufficient_data",
                                             "verdict": "INSUFFICIENT_DATA",
                                             "n_buys": 0,
                                             "n_with_intraperiod": 0})

        assert rcb._append_mfe_skill_log(
            cycle=1, win_start=date(2020, 1, 1), win_end=date(2020, 6, 1),
            outcomes_path=tmp_path / "nope.jsonl") is True
        assert log.exists()

    def test_never_raises_on_unwritable_path(self, tmp_path, monkeypatch):
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        monkeypatch.setattr(rcb, "MFE_SKILL_LOG",
                            blocker / "sub" / "mfe.jsonl")

        from paper_trader.ml import mfe_conversion as mfe
        monkeypatch.setattr(mfe, "analyze",
                            lambda *a, **k: {"status": "insufficient_data",
                                             "verdict": "INSUFFICIENT_DATA",
                                             "n_buys": 0,
                                             "n_with_intraperiod": 0})
        assert rcb._append_mfe_skill_log(
            1, date(2000, 1, 3), date(2001, 1, 3),
            outcomes_path=tmp_path / "nope.jsonl") is False


# ─────────────────────── main() wiring regression ───────────────────────


class TestIntraperiodLedgerWiring:
    """Source-level regression: both new ledgers MUST be invoked from
    ``main()`` every cycle. Mirrors the existing
    ``TestCycleWiringRegression`` discipline in ``test_continuous.py``:
    every prior ledger (scorer / baseline / llm_annotation /
    calibrated_reliability / conviction_calibration) was implemented but
    NOT called from ``main()`` for a window of time, silently disabling a
    quant-facing audit trail. A source-level assertion catches the same
    regression class for these two new ledgers loudly.
    """

    def test_main_invokes_stop_out_skill_ledger(self):
        import inspect
        src = inspect.getsource(rcb.main)
        assert "_append_stop_out_skill_log(" in src

    def test_main_invokes_mfe_skill_ledger(self):
        import inspect
        src = inspect.getsource(rcb.main)
        assert "_append_mfe_skill_log(" in src
