"""Tests for the off-distribution gate-abstention diagnostic
``paper_trader/ml/gate_abstention.py``.

The diagnostic reports how often the live gate's off-distribution guard
fired (``gate_off_dist=True`` rows), exact-value-locks the verdict
boundaries, and surfaces a temporal trend axis. All tests are pure /
offline — synthetic dict rows, no DB, no network, no pickle.

Mirrors the assert-exact style of ``test_gate_realized.py``, ``test_
gate_audit.py``, and ``test_baseline_trend.py``: a threshold tune or
output-shape change must update these literals deliberately.
"""
from __future__ import annotations

import json

import pytest

from paper_trader.ml import gate_abstention as ga


# ───────────────────────── synthetic row helpers ────────────────────────────

def _row(*, sim_date="2025-01-01", ticker="NVDA",
         gate_scorer_pred=5.0, gate_off_dist=False, action="BUY"):
    return {
        "sim_date": sim_date, "ticker": ticker, "action": action,
        "gate_scorer_pred": gate_scorer_pred,
        "gate_off_dist": gate_off_dist,
        "forward_return_5d": 1.0,  # field is required elsewhere; inert here
    }


# ───────────────────────── pure abstention_report ───────────────────────────

class TestAbstentionReport:

    def test_insufficient_data_below_min_total(self):
        # 29 captured rows (1 below MIN_TOTAL=30) ⇒ INSUFFICIENT_DATA
        rows = [_row(sim_date=f"2025-01-{i+1:02d}") for i in range(29)]
        rep = ga.abstention_report(rows)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_captured"] == 29
        assert rep["n_abstained"] == 0
        assert rep["n_acted"] == 29
        assert rep["rate"] == 0.0

    def test_none_gate_scorer_pred_rows_are_excluded(self):
        # SELL / pre-60b20d9 / untrained rows have gate_scorer_pred=None and
        # must be excluded from n_captured entirely — they have no gate
        # decision to characterize.
        rows = [_row(gate_scorer_pred=None) for _ in range(50)]
        rep = ga.abstention_report(rows)
        assert rep["n_captured"] == 0
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_guard_inactive_below_threshold(self):
        # 200 captured, 0 abstained ⇒ rate=0.0 < INACTIVE_MAX=0.005
        rows = [_row(sim_date=f"2025-{m:02d}-{d:02d}",
                     gate_scorer_pred=2.0, gate_off_dist=False)
                for m in range(1, 11) for d in range(1, 21)]
        rep = ga.abstention_report(rows)
        assert len(rows) == 200
        assert rep["n_captured"] == 200
        assert rep["n_abstained"] == 0
        assert rep["rate"] == 0.0
        assert rep["verdict"] == "GUARD_INACTIVE"

    def test_guard_healthy_in_band(self):
        # 100 captured, 5 abstained (5%) — within [0.5%, 15%) ⇒ HEALTHY
        rows = []
        for i in range(100):
            off = i < 5  # first 5 rows abstain
            rows.append(_row(sim_date=f"2025-01-{(i % 30) + 1:02d}",
                             gate_scorer_pred=50.0 if off else 3.0,
                             gate_off_dist=off))
        rep = ga.abstention_report(rows)
        assert rep["n_captured"] == 100
        assert rep["n_abstained"] == 5
        assert rep["rate"] == 0.05
        assert rep["verdict"] == "GUARD_HEALTHY"

    def test_guard_rampant_at_or_above_threshold(self):
        # 100 captured, 15 abstained = exactly 15% (== RAMPANT_MIN, inclusive)
        rows = []
        for i in range(100):
            off = i < 15
            rows.append(_row(sim_date=f"2025-02-{(i % 28) + 1:02d}",
                             gate_scorer_pred=50.0 if off else 1.0,
                             gate_off_dist=off))
        rep = ga.abstention_report(rows)
        assert rep["n_captured"] == 100
        assert rep["n_abstained"] == 15
        assert rep["rate"] == 0.15
        assert rep["verdict"] == "GUARD_RAMPANT"

    def test_arm_distribution_for_abstained_rows(self):
        # Off-dist abstentions on extreme predictions (the expected pattern)
        rows = []
        # 30 in-distribution acted rows
        for i in range(30):
            rows.append(_row(sim_date=f"2025-01-{(i % 28) + 1:02d}",
                             gate_scorer_pred=3.0, gate_off_dist=False))
        # 10 abstained at +50 (would-have-been strong_tailwind)
        for i in range(10):
            rows.append(_row(sim_date=f"2025-02-{(i % 28) + 1:02d}",
                             gate_scorer_pred=50.0, gate_off_dist=True))
        # 5 abstained at -50 (would-have-been strong_headwind)
        for i in range(5):
            rows.append(_row(sim_date=f"2025-03-{(i % 28) + 1:02d}",
                             gate_scorer_pred=-50.0, gate_off_dist=True))
        rep = ga.abstention_report(rows)
        arm = {a["arm"]: a["n_abstained"] for a in rep["arm_dist"]}
        assert arm["strong_tailwind"] == 10
        assert arm["strong_headwind"] == 5
        # Other arms must be 0 — exact zeros, not just absent.
        assert arm["neutral"] == 0
        assert arm["mild_headwind"] == 0
        assert arm["mild_tailwind"] == 0

    def test_top_tickers_ordered_by_abstention_count(self):
        # AAA abstained 7×, BBB 3×, CCC 1×; some non-abstained AAA rows mixed
        # in to verify the count is over ABSTAINED rows only.
        rows = []
        for i in range(7):
            rows.append(_row(sim_date=f"2025-01-{i+1:02d}", ticker="AAA",
                             gate_scorer_pred=50.0, gate_off_dist=True))
        for i in range(3):
            rows.append(_row(sim_date=f"2025-02-{i+1:02d}", ticker="BBB",
                             gate_scorer_pred=50.0, gate_off_dist=True))
        rows.append(_row(sim_date="2025-03-01", ticker="CCC",
                         gate_scorer_pred=50.0, gate_off_dist=True))
        # Non-abstained AAA rows — must NOT count toward AAA's abstention tally
        for i in range(20):
            rows.append(_row(sim_date=f"2025-04-{i+1:02d}", ticker="AAA",
                             gate_scorer_pred=2.0, gate_off_dist=False))
        rep = ga.abstention_report(rows)
        tops = rep["top_tickers"]
        assert tops[0] == {"ticker": "AAA", "n_abstained": 7}
        assert tops[1] == {"ticker": "BBB", "n_abstained": 3}
        assert tops[2] == {"ticker": "CCC", "n_abstained": 1}

    def test_trend_degrading_when_recent_rate_higher(self):
        # Older half: 30 rows, 0 abstain. Recent half: 30 rows, 9 abstain (30%).
        # diff = +0.30, > TREND_TOL_PP=0.02 ⇒ DEGRADING.
        rows = []
        for i in range(30):
            rows.append(_row(sim_date=f"2024-01-{(i % 28) + 1:02d}",
                             gate_scorer_pred=2.0, gate_off_dist=False))
        for i in range(30):
            off = i < 9
            rows.append(_row(sim_date=f"2025-12-{(i % 28) + 1:02d}",
                             gate_scorer_pred=50.0 if off else 2.0,
                             gate_off_dist=off))
        rep = ga.abstention_report(rows)
        assert rep["trend"] == "DEGRADING"
        # 0.0 older, 0.3 recent, diff 0.3
        assert rep["rate_older"] == 0.0
        assert rep["rate_recent"] == 0.3
        assert rep["trend_rate_diff_pp"] == 0.3

    def test_trend_improving_when_recent_rate_lower(self):
        # Older: 30 rows, 12 abstain (40%). Recent: 30 rows, 0 abstain.
        # diff = -0.40 ⇒ IMPROVING.
        rows = []
        for i in range(30):
            off = i < 12
            rows.append(_row(sim_date=f"2024-01-{(i % 28) + 1:02d}",
                             gate_scorer_pred=50.0 if off else 2.0,
                             gate_off_dist=off))
        for i in range(30):
            rows.append(_row(sim_date=f"2025-12-{(i % 28) + 1:02d}",
                             gate_scorer_pred=2.0, gate_off_dist=False))
        rep = ga.abstention_report(rows)
        assert rep["trend"] == "IMPROVING"
        assert rep["rate_older"] == 0.4
        assert rep["rate_recent"] == 0.0

    def test_trend_stable_when_rates_close(self):
        # Equal abstention rates older/recent ⇒ diff=0 < TREND_TOL_PP ⇒ STABLE
        rows = []
        for half in (0, 1):
            year = "2024" if half == 0 else "2025"
            for i in range(30):
                off = i < 6  # 20% each side
                rows.append(_row(sim_date=f"{year}-01-{(i % 28) + 1:02d}",
                                 gate_scorer_pred=50.0 if off else 2.0,
                                 gate_off_dist=off))
        rep = ga.abstention_report(rows)
        assert rep["trend"] == "STABLE"
        assert rep["rate_older"] == rep["rate_recent"]

    def test_trend_unknown_when_insufficient_per_side(self):
        # Need 2*MIN_PER_SIDE = 30 total for trend computation; with <30 ⇒ UNKNOWN
        rows = [_row(sim_date=f"2025-01-{i+1:02d}") for i in range(29)]
        rep = ga.abstention_report(rows)
        assert rep["trend"] == "UNKNOWN"
        assert rep["trend_rate_diff_pp"] is None

    def test_never_raises_on_garbage_input(self):
        # Mixed garbage: None, non-dict, missing keys, bad types — must NOT raise
        rows = [None, "string", 12345, {}, {"gate_scorer_pred": "not-a-number"},
                {"gate_scorer_pred": float("nan")},
                {"gate_scorer_pred": float("inf")},
                {"gate_scorer_pred": True},  # bool excluded by _f
                _row(gate_scorer_pred=2.0, gate_off_dist=False)]
        rep = ga.abstention_report(rows)
        # Only the last (real) row should count as captured.
        assert rep["n_captured"] == 1
        assert rep["status"] == "ok"


# ──────────────────────── _f finite-coercion helper ──────────────────────────

class TestFCoercion:

    def test_bool_returns_none(self):
        # Python bool subclasses int but is meaningless as a prediction.
        assert ga._f(True) is None
        assert ga._f(False) is None

    def test_nan_and_inf_return_none(self):
        assert ga._f(float("nan")) is None
        assert ga._f(float("inf")) is None
        assert ga._f(float("-inf")) is None

    def test_numeric_string_passes_through(self):
        # `_f` mirrors the `gate_realized._f` precedent: a numeric-string
        # round-trips through `float()` and is accepted; the JSON parser
        # already returned proper numeric types for real records, so this
        # is a defensive accept rather than a strict reject. Non-numeric
        # strings still fail the float cast and return None.
        assert ga._f("5.0") == 5.0
        assert ga._f("nope") is None
        # Stringified non-finite is rejected by the isfinite guard.
        assert ga._f("nan") is None

    def test_finite_float_passes_through(self):
        assert ga._f(0.0) == 0.0
        assert ga._f(-50.0) == -50.0
        assert ga._f(99.99) == 99.99


# ────────────────────────── analyze() file path ──────────────────────────────

class TestAnalyze:

    def test_missing_file_returns_safe_shape(self, tmp_path):
        rep = ga.analyze(tmp_path / "nonexistent.jsonl")
        assert rep["status"] == "error"
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_captured"] == 0
        assert "no outcomes file at" in rep["hint"]

    def test_empty_file_returns_insufficient_data(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        rep = ga.analyze(p)
        # No records ⇒ falls through abstention_report's early branch
        assert rep["n_captured"] == 0
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_corrupt_lines_skipped_not_fatal(self, tmp_path):
        # One valid row + one corrupt + one non-dict JSON
        p = tmp_path / "outcomes.jsonl"
        valid = {"sim_date": "2025-01-01", "gate_scorer_pred": 5.0,
                 "gate_off_dist": False, "action": "BUY",
                 "forward_return_5d": 1.0, "ticker": "NVDA"}
        p.write_text(
            "this is not json\n"
            "[1, 2, 3]\n"          # JSON list, not dict
            f"{json.dumps(valid)}\n"
        )
        # oos_only=False so we keep the single row (the temporal split would
        # otherwise need ≥1 row to land in the OOS slice).
        rep = ga.analyze(p, oos_only=False)
        assert rep["n_captured"] == 1
        assert rep["n_records_total"] == 1

    def test_slice_label_is_oos_by_default(self, tmp_path):
        p = tmp_path / "outcomes.jsonl"
        rows = []
        for i in range(60):
            rows.append({
                "sim_date": f"2025-01-{(i % 28) + 1:02d}",
                "gate_scorer_pred": 2.0,
                "gate_off_dist": False,
                "action": "BUY",
                "forward_return_5d": 0.5,
                "ticker": "NVDA",
            })
        p.write_text("\n".join(json.dumps(r) for r in rows))
        rep = ga.analyze(p)  # oos_only=True default
        assert rep["slice"] == "oos"
        assert rep["n_records_total"] == 60

    def test_all_slice_label_when_oos_disabled(self, tmp_path):
        p = tmp_path / "outcomes.jsonl"
        p.write_text(json.dumps({
            "sim_date": "2025-01-01", "gate_scorer_pred": 1.0,
            "gate_off_dist": False, "action": "BUY",
            "forward_return_5d": 0.5, "ticker": "NVDA",
        }) + "\n")
        rep = ga.analyze(p, oos_only=False)
        assert rep["slice"] == "all"


# ─────────────────────────── CLI exit codes ──────────────────────────────────

class TestCliExitCodes:

    def test_exit_0_on_inactive(self, tmp_path, monkeypatch):
        # Build a 200-row all-acted outcomes file ⇒ INACTIVE
        p = tmp_path / "decision_outcomes.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = [{
            "sim_date": f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            "gate_scorer_pred": 3.0, "gate_off_dist": False,
            "action": "BUY", "forward_return_5d": 1.0, "ticker": "NVDA",
        } for i in range(200)]
        p.write_text("\n".join(json.dumps(r) for r in rows))
        monkeypatch.setattr(
            ga, "analyze",
            lambda *_a, **_kw: ga.abstention_report(rows))
        rc = ga._cli([])
        assert rc == 0

    def test_exit_2_on_rampant(self, tmp_path, monkeypatch):
        # 100 captured, 20 abstained (20% ≥ RAMPANT_MIN=15%) ⇒ RAMPANT ⇒ exit 2
        rows = []
        for i in range(100):
            off = i < 20
            rows.append({
                "sim_date": f"2025-01-{(i % 28) + 1:02d}",
                "gate_scorer_pred": 50.0 if off else 2.0,
                "gate_off_dist": off, "action": "BUY",
                "forward_return_5d": 1.0, "ticker": "NVDA",
            })
        monkeypatch.setattr(
            ga, "analyze",
            lambda *_a, **_kw: ga.abstention_report(rows))
        rc = ga._cli([])
        assert rc == 2


# ───────────── single-source-of-truth identity with gate_audit ───────────────

class TestSingleSourceOfTruthArmBoundaries:
    """The arm boundaries used for the would-have-been distribution must come
    from `gate_audit.gate_arm` verbatim — the documented gate_pnl /
    gate_realized precedent. A drift here would silently mis-bucket
    abstentions."""

    def test_arm_function_is_imported_from_gate_audit(self):
        from paper_trader.ml import gate_audit
        assert ga.gate_arm is gate_audit.gate_arm

    def test_arm_dist_uses_live_gate_boundaries(self):
        # gate_audit.gate_arm: < -10 ⇒ strong_headwind, < 0 ⇒ mild_headwind,
        # 0..5 ⇒ neutral, > 5 ⇒ mild_tailwind, > 10 ⇒ strong_tailwind.
        # Pin -11 → strong_headwind, +11 → strong_tailwind.
        rows = [
            _row(sim_date="2025-01-01", gate_scorer_pred=-11.0,
                 gate_off_dist=True),
            _row(sim_date="2025-01-02", gate_scorer_pred=11.0,
                 gate_off_dist=True),
        ]
        rep = ga.abstention_report(rows)
        arm = {a["arm"]: a["n_abstained"] for a in rep["arm_dist"]}
        assert arm["strong_headwind"] == 1
        assert arm["strong_tailwind"] == 1
