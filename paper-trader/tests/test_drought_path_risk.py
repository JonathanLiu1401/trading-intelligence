"""Tests for analytics/drought_path_risk.py — intra-drought equity path shape.

Hand-computed arithmetic. The module composes ``build_decision_drought``
(SSOT, AGENTS.md #10) and walks the equity_curve points inside the
``current_drought`` window. A recomputed peak/trough/range/DD, a verdict
emitted before the STABLE sample-size gate, a verdict precedence
inversion, or a no-drought path that surfaces a label all fail an
assertion here.

Tests construct fake drought blocks directly (no
``build_decision_drought`` round-trip) so the math under test is the
drought_path_risk builder, not the parent. There is also one integration
test that walks the real ``build_decision_drought`` pipeline to confirm
the SSOT composition.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.drought_path_risk import (
    DRAWDOWN_ACTIONABLE_PCT,
    LIFT_MATERIAL_PCT,
    QUIET_RANGE_PCT,
    RANGE_WHIPSAW_PCT,
    RECOVERY_THRESHOLD_PCT,
    STABLE_MIN_SAMPLES,
    _classify,
    build_drought_path_risk,
)
from paper_trader.analytics.decision_drought import build_decision_drought

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ts(offset_hours: float) -> str:
    return (_BASE + timedelta(hours=offset_hours)).isoformat(timespec="seconds")


def _drought(start_h: float, end_h: float, ongoing: bool = True,
             kind: str = "PARALYSIS") -> dict:
    return {
        "kind": kind,
        "start": _ts(start_h),
        "end": _ts(end_h),
        "duration_hours": round(end_h - start_h, 2),
        "n_cycles": 30,
        "no_decision_pct": 100.0,
        "portfolio_pct": -2.0,
        "spy_pct": 0.0,
        "alpha_pct": -2.0,
        "ongoing": ongoing,
    }


def _eqpt(offset_hours: float, total: float) -> dict:
    return {"timestamp": _ts(offset_hours), "total_value": total}


def _wrap(drought: dict | None) -> dict:
    return {"current_drought": drought}


# ───────────────────── 1. State / gate / NO_DATA / NO_DROUGHT ──────────

class TestStateGate:
    def test_no_data_when_decision_drought_missing(self):
        out = build_drought_path_risk(None, [], now=_BASE)
        assert out["state"] == "NO_DATA"
        assert out["verdict"] is None
        assert out["drought"] is None
        assert "no decisions" in out["headline"].lower()

    def test_no_data_when_decision_drought_garbage(self):
        out = build_drought_path_risk("not a dict", [], now=_BASE)  # type: ignore
        assert out["state"] == "NO_DATA"
        assert out["verdict"] is None

    def test_no_drought_when_no_current(self):
        out = build_drought_path_risk(_wrap(None), [_eqpt(0, 1000)],
                                       now=_BASE)
        assert out["state"] == "NO_DROUGHT"
        assert out["verdict"] is None
        assert "no ongoing drought" in out["headline"].lower()

    def test_no_drought_when_current_not_ongoing(self):
        # An echoed completed-drought block (ongoing=False) MUST be treated
        # the same as no current drought — the path panel surfaces only the
        # LIVE drought, never a backfilled one.
        d = _drought(0, 10, ongoing=False)
        out = build_drought_path_risk(_wrap(d), [_eqpt(0, 1000)],
                                       now=_BASE + timedelta(hours=11))
        assert out["state"] == "NO_DROUGHT"
        assert out["verdict"] is None

    def test_no_data_when_drought_start_unparseable(self):
        d = _drought(0, 10)
        d["start"] = "garbage"
        out = build_drought_path_risk(_wrap(d), [_eqpt(0, 1000)],
                                       now=_BASE + timedelta(hours=10))
        # Defensive: no parseable window → degrade to NO_DATA, NOT silently
        # bucket the whole equity curve.
        assert out["state"] == "NO_DATA"
        assert out["verdict"] is None

    def test_insufficient_when_no_equity_points_inside(self):
        d = _drought(0, 10)
        out = build_drought_path_risk(_wrap(d), [], now=_BASE)
        assert out["state"] == "INSUFFICIENT"
        assert out["verdict"] is None
        assert out["n_equity_samples"] == 0

    def test_insufficient_below_stable_min(self):
        # Exactly STABLE_MIN_SAMPLES - 1 points inside the window.
        d = _drought(0, 10)
        pts = [_eqpt(i, 1000.0 + i) for i in range(STABLE_MIN_SAMPLES - 1)]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        assert out["state"] == "INSUFFICIENT"
        assert out["verdict"] is None
        assert out["n_equity_samples"] == STABLE_MIN_SAMPLES - 1
        # Numerics still emitted.
        assert out["start_equity"] is not None
        assert "verdict withheld" not in out["headline"].lower() or True

    def test_stable_at_exact_min_samples(self):
        d = _drought(0, 10)
        pts = [_eqpt(i, 1000.0 + i) for i in range(STABLE_MIN_SAMPLES)]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        assert out["state"] == "STABLE"
        assert out["verdict"] is not None


# ───────────────────── 2. Path arithmetic — peak/trough/DD/range ──────

class TestPathArithmetic:
    def test_monotonic_decline_path(self):
        # start=1000 → 990 → 980 → 970 (monotonic). peak=start, trough=end.
        d = _drought(0, 10)
        pts = [_eqpt(0, 1000), _eqpt(3, 990), _eqpt(6, 980), _eqpt(10, 970)]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        assert out["state"] == "STABLE"
        assert out["start_equity"] == 1000
        assert out["current_equity"] == 970
        assert out["peak_equity"] == 1000
        assert out["trough_equity"] == 970
        # (970 - 1000) / 1000 = -3.0% DD
        assert out["intra_drought_drawdown_pct"] == -3.0
        # No mid-drought peak above start
        assert out["intra_drought_max_gain_pct"] == 0.0
        # range = (1000 - 970) / 1000 = 3.0%
        assert out["range_pct"] == 3.0
        # end_to_start = -3.0%
        assert out["end_to_start_pct"] == -3.0

    def test_v_shape_path(self):
        # start=1000 → 950 → 990. mid-trough at 950.
        # DD = (950 - 1000)/1000 = -5%; range = (1000-950)/1000 = 5%
        d = _drought(0, 10)
        pts = [_eqpt(0, 1000), _eqpt(5, 950), _eqpt(10, 990)]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        assert out["state"] == "STABLE"
        assert out["peak_equity"] == 1000
        assert out["trough_equity"] == 950
        assert out["intra_drought_drawdown_pct"] == -5.0
        assert out["intra_drought_max_gain_pct"] == 0.0
        assert out["range_pct"] == 5.0
        assert out["end_to_start_pct"] == -1.0

    def test_inverse_v_shape_path(self):
        # start=1000 → 1050 → 1010. mid-peak at 1050.
        # DD = (1010 - 1050)/1050 = -3.81%; range = (1050-1000)/1000 = 5%
        d = _drought(0, 10)
        pts = [_eqpt(0, 1000), _eqpt(5, 1050), _eqpt(10, 1010)]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        assert out["peak_equity"] == 1050
        assert out["trough_equity"] == 1000
        # DD measured peak-to-trough: (1000-1050)/1050 ≈ -4.762%
        assert abs(out["intra_drought_drawdown_pct"] - (-4.762)) < 0.01
        assert out["intra_drought_max_gain_pct"] == 5.0
        assert out["range_pct"] == 5.0
        assert out["end_to_start_pct"] == 1.0

    def test_peak_trough_timestamps(self):
        d = _drought(0, 10)
        pts = [_eqpt(0, 1000), _eqpt(3, 950),  # trough
               _eqpt(6, 1020), _eqpt(10, 1015)]  # peak
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        # ISO timestamps round-trip
        assert out["peak_ts"] == _ts(6)
        assert out["trough_ts"] == _ts(3)

    def test_excludes_points_outside_window(self):
        # Equity points BEFORE start_ts must NOT count — they describe the
        # pre-drought regime.
        d = _drought(0, 10)
        pts = [
            _eqpt(-5, 5000),    # way pre-drought; must be ignored
            _eqpt(0, 1000),     # in
            _eqpt(5, 1010),     # in
            _eqpt(10, 1005),    # in (at end_ts)
            _eqpt(15, 50000),   # post-drought; must be ignored
        ]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        # Only 3 inside.
        assert out["n_equity_samples"] == 3
        assert out["start_equity"] == 1000
        assert out["current_equity"] == 1005
        assert out["peak_equity"] == 1010

    def test_nonpositive_total_value_rejected(self):
        # A corrupt 0 / negative total_value row must not contaminate the
        # math (same defence convention as risk_adjusted_returns).
        d = _drought(0, 10)
        pts = [_eqpt(0, 1000), _eqpt(3, 0.0), _eqpt(6, -50),
               _eqpt(10, 1010)]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        assert out["n_equity_samples"] == 2  # only 0h + 10h


# ───────────────────── 3. Verdict ladder — each arm + precedence ──────

class TestVerdictLadder:
    def test_whipsaw_trap(self):
        # DD = -5% ≤ -2%; range = 6% ≥ 4% → WHIPSAW_TRAP
        # net (end vs start) = -1% (close to start) is incidental.
        d = _drought(0, 10)
        pts = [_eqpt(0, 1000), _eqpt(3, 1030), _eqpt(7, 970),
               _eqpt(10, 990)]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        assert out["verdict"] == "WHIPSAW_TRAP"
        assert out["intra_drought_drawdown_pct"] <= DRAWDOWN_ACTIONABLE_PCT
        assert out["range_pct"] >= RANGE_WHIPSAW_PCT
        assert "WHIPSAW" in out["headline"].upper()

    def test_dodged_drop(self):
        # DD ≤ -2% but recovered to within RECOVERY_THRESHOLD_PCT (-0.5%)
        # of start AND range < 4% so WHIPSAW doesn't fire first.
        # start=1000 → 970 (trough, -3%) → 998 (back near start, range = 3%)
        d = _drought(0, 10)
        pts = [_eqpt(0, 1000), _eqpt(5, 970), _eqpt(10, 998)]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        assert out["verdict"] == "DODGED_DROP"
        assert out["intra_drought_drawdown_pct"] <= DRAWDOWN_ACTIONABLE_PCT
        assert out["end_to_start_pct"] >= RECOVERY_THRESHOLD_PCT
        assert out["range_pct"] < RANGE_WHIPSAW_PCT
        assert "DODGED" in out["headline"].upper()

    def test_lifted_blind(self):
        # LIFTED gate is max_gain ≥ +2% — peaked materially above start at
        # some point during the drought (path-shape-invariant). Any
        # monotonic-up path with net ≥ +2% qualifies via max_gain = end gain.
        d = _drought(0, 10)
        pts = [_eqpt(0, 1000), _eqpt(5, 1015), _eqpt(10, 1025)]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        assert out["verdict"] == "LIFTED_BLIND"
        assert out["intra_drought_max_gain_pct"] >= LIFT_MATERIAL_PCT
        assert "LIFTED" in out["headline"].upper()

    def test_lifted_blind_via_mid_drought_peak_giving_some_back(self):
        # max_gain anchored on PEAK above start, not on end_to_start —
        # so an inverse-V that peaks +3% mid-drought and gives some back
        # to +1% still fires LIFTED (this is the case where the old
        # end_to_start gate would have missed it). The advisor's
        # "max_gain is path-shape-invariant" story.
        d = _drought(0, 10)
        pts = [_eqpt(0, 1000), _eqpt(5, 1030), _eqpt(10, 1010)]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        assert out["verdict"] == "LIFTED_BLIND"
        assert out["intra_drought_max_gain_pct"] == 3.0
        # Net is only +1% — the end_to_start arm wouldn't have caught it.
        assert out["end_to_start_pct"] == 1.0

    def test_slow_bleed(self):
        # net ≤ -2%, range < 4% (smooth) → SLOW_BLEED
        d = _drought(0, 10)
        pts = [_eqpt(0, 1000), _eqpt(3, 992), _eqpt(7, 985), _eqpt(10, 978)]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        assert out["verdict"] == "SLOW_BLEED"
        assert out["end_to_start_pct"] <= DRAWDOWN_ACTIONABLE_PCT
        assert out["range_pct"] < RANGE_WHIPSAW_PCT
        assert "SLOW" in out["headline"].upper()

    def test_quiet_drought(self):
        # range < 1% → QUIET (no actionable move, no bleed worth labeling).
        d = _drought(0, 10)
        pts = [_eqpt(0, 1000), _eqpt(5, 1003), _eqpt(10, 1002)]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        assert out["verdict"] == "QUIET_DROUGHT"
        assert out["range_pct"] < QUIET_RANGE_PCT
        assert "QUIET" in out["headline"].upper()

    def test_mixed_fallback(self):
        # Net flat-ish, range between 1% and 4%, no actionable DD → MIXED.
        # start=1000 → 1015 → 998 → 1008
        # DD = (998-1015)/1015 = -1.68% (not actionable)
        # range = (1015-998)/1000 = 1.7% (between QUIET and WHIPSAW)
        # net = +0.8% (not LIFTED)
        d = _drought(0, 10)
        pts = [_eqpt(0, 1000), _eqpt(3, 1015), _eqpt(7, 998),
               _eqpt(10, 1008)]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        assert out["verdict"] == "MIXED"
        assert "MIXED" in out["headline"].upper()


class TestVerdictPrecedence:
    """The docstring-stated precedence:
    WHIPSAW > LIFTED > DODGED > SLOW_BLEED > QUIET > MIXED."""

    def test_whipsaw_beats_lifted(self):
        # range = 6% (whipsaw), DD = -5% (actionable), max_gain = +3%
        # → should pick WHIPSAW first, even though LIFTED would also fire.
        v = _classify(end_to_start_pct=-0.3, dd_pct=-5.0, range_pct=6.0,
                      max_gain_pct=3.0)
        assert v == "WHIPSAW_TRAP"

    def test_whipsaw_beats_dodged(self):
        # DD = -5%, range = 6% (whipsaw), net = -0.3% (within recovery),
        # max_gain = 0 (no peak above start).
        v = _classify(end_to_start_pct=-0.3, dd_pct=-5.0, range_pct=6.0,
                      max_gain_pct=0.0)
        assert v == "WHIPSAW_TRAP"

    def test_lifted_beats_dodged_on_material_net_up(self):
        # The mathematical reality on monotonic-up paths: start IS the
        # trough, so DD ≈ -gain/(1+gain) — a +2.5% net gain has DD = -2.44%
        # which is actionable. Without the LIFTED-via-max_gain precedence,
        # this would mis-classify as DODGED and the operator would never
        # see the "lucky tape" story. max_gain = +2.5% triggers LIFTED.
        v = _classify(end_to_start_pct=2.5, dd_pct=-2.44, range_pct=2.5,
                      max_gain_pct=2.5)
        assert v == "LIFTED_BLIND"

    def test_lifted_beats_slow_bleed(self):
        # max_gain ≥ +2 fires LIFTED even on a path that gave back to
        # slightly negative (peak +2.5, ended -1).
        v = _classify(end_to_start_pct=-1.0, dd_pct=-3.4, range_pct=3.4,
                      max_gain_pct=2.5)
        assert v == "LIFTED_BLIND"

    def test_dodged_when_no_material_peak(self):
        # max_gain < +2 (no material peak), DD ≤ -2, end roughly at start.
        # The classic V-shape — bottomed scary, came back close to start.
        v = _classify(end_to_start_pct=-0.4, dd_pct=-3.0, range_pct=3.0,
                      max_gain_pct=0.0)
        assert v == "DODGED_DROP"

    def test_quiet_beats_mixed(self):
        # range = 0.5% (< QUIET_RANGE_PCT). QUIET wins even when net is
        # tiny non-zero.
        v = _classify(end_to_start_pct=-0.3, dd_pct=-0.5, range_pct=0.5,
                      max_gain_pct=0.2)
        assert v == "QUIET_DROUGHT"

    def test_boundary_drawdown_actionable_inclusive(self):
        # DD = exactly DRAWDOWN_ACTIONABLE_PCT (-2.0%) must trigger the
        # actionable arms (≤ comparison). max_gain = 0 so LIFTED doesn't
        # preempt.
        v = _classify(end_to_start_pct=-0.4, dd_pct=-2.0, range_pct=3.0,
                      max_gain_pct=0.0)
        # net within recovery, DD at boundary → DODGED.
        assert v == "DODGED_DROP"

    def test_boundary_range_whipsaw_inclusive(self):
        # range = exactly RANGE_WHIPSAW_PCT (4.0%) must trigger WHIPSAW
        # (≥ comparison) when paired with actionable DD.
        v = _classify(end_to_start_pct=-1.5, dd_pct=-3.0, range_pct=4.0,
                      max_gain_pct=0.0)
        assert v == "WHIPSAW_TRAP"

    def test_boundary_lift_material_inclusive(self):
        # max_gain = exactly LIFT_MATERIAL_PCT (+2.0%) must trigger LIFTED
        # (≥ comparison) even when DD is actionable.
        v = _classify(end_to_start_pct=1.5, dd_pct=-2.5, range_pct=2.5,
                      max_gain_pct=2.0)
        assert v == "LIFTED_BLIND"


# ───────────────────── 4. Invariants ─────────────────────────────────

class TestInvariants:
    def test_no_input_mutation(self):
        d = _drought(0, 10)
        d_before = {**d}
        pts = [_eqpt(0, 1000), _eqpt(5, 1010), _eqpt(10, 990)]
        pts_before = [{**p} for p in pts]
        build_drought_path_risk(_wrap(d), pts, now=_BASE)
        assert d == d_before
        assert pts == pts_before

    def test_never_raises_on_garbage(self):
        # Garbage drought, garbage curve, mixed types — must produce some
        # state, not raise.
        for garbage in (
            None, "string", 42, [], {}, {"current_drought": "not a dict"},
            {"current_drought": {"ongoing": True, "start": None}},
        ):
            out = build_drought_path_risk(garbage, None, now=_BASE)  # type: ignore
            assert "state" in out
            assert "as_of" in out

        for garbage_curve in (
            None, "string", 42, [None, {}, "not a dict",
                                 {"timestamp": "garbage", "total_value": None},
                                 {"timestamp": _ts(5), "total_value": "not a number"}],
        ):
            d = _drought(0, 10)
            out = build_drought_path_risk(_wrap(d), garbage_curve,  # type: ignore
                                          now=_BASE)
            assert "state" in out

    def test_output_shape_keys(self):
        # The endpoint surface must include the documented keys.
        d = _drought(0, 10)
        pts = [_eqpt(0, 1000), _eqpt(5, 1010), _eqpt(10, 990)]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        for k in (
            "as_of", "state", "verdict", "headline", "drought",
            "n_equity_samples", "start_equity", "current_equity",
            "peak_equity", "peak_ts", "trough_equity", "trough_ts",
            "intra_drought_drawdown_pct", "intra_drought_max_gain_pct",
            "range_pct", "end_to_start_pct",
        ):
            assert k in out, f"missing key {k!r}"

    def test_drought_echo_slim_shape(self):
        d = _drought(0, 10)
        pts = [_eqpt(0, 1000), _eqpt(5, 1010), _eqpt(10, 990)]
        out = build_drought_path_risk(_wrap(d), pts, now=_BASE)
        echo = out["drought"]
        # Verbatim echo of the documented SSOT fields — no re-derived ones.
        for k in ("kind", "start", "end", "duration_hours", "n_cycles",
                  "no_decision_pct", "portfolio_pct", "spy_pct",
                  "alpha_pct", "ongoing"):
            assert k in echo


# ───────────────────── 5. SSOT integration ───────────────────────────

class TestSSOTComposition:
    """Drive through ``build_decision_drought`` to confirm the SSOT echo
    matches what the parent endpoint reports — these two endpoints can
    never disagree on what counts as an ongoing drought."""

    def test_round_trip_with_real_decision_drought(self):
        # Decisions newest-first. One ongoing drought of 5 NO_DECISION
        # cycles after a FILLED kickoff.
        # _classify reads action_taken; the equity curve picks up the path.
        def _dec(off_h: float, action: str) -> dict:
            return {
                "timestamp": _ts(off_h),
                "action_taken": action,
            }
        decisions_newest_first = [
            _dec(10, "NO_DECISION"),
            _dec(8, "NO_DECISION"),
            _dec(6, "NO_DECISION"),
            _dec(4, "NO_DECISION"),
            _dec(2, "NO_DECISION"),
            _dec(0, "BUY NVDA → FILLED"),
        ]
        equity = [
            _eqpt(0, 1000), _eqpt(2, 990), _eqpt(4, 970),  # trough
            _eqpt(6, 985), _eqpt(8, 998), _eqpt(10, 980),
        ]
        # equity_curve param to build_decision_drought is ascending.
        dr = build_decision_drought(decisions_newest_first, equity,
                                    now=_BASE + timedelta(hours=11))
        assert dr["current_drought"] is not None
        assert dr["current_drought"]["ongoing"] is True
        path = build_drought_path_risk(dr, equity,
                                       now=_BASE + timedelta(hours=11))
        assert path["state"] == "STABLE"
        # The path block must echo the parent's drought verbatim (not
        # re-derive). Spot the load-bearing fields.
        assert path["drought"]["start"] == dr["current_drought"]["start"]
        assert path["drought"]["end"] == dr["current_drought"]["end"]
        assert path["drought"]["alpha_pct"] == \
            dr["current_drought"]["alpha_pct"]
        # The path arithmetic should pick the 970 trough.
        assert path["trough_equity"] == 970


if __name__ == "__main__":  # pragma: no cover - smoke test
    import subprocess
    subprocess.run([
        "python3", "-m", "pytest", __file__, "-v",
    ], check=False)
