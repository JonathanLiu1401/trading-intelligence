"""Tests for paper_trader.analytics.all_cash_streak.

The verdict drives the operator's read on whether the bot is actively
deployed or sidelined. A silently-broken streak walk, an off-by-one in
the contiguous-run grouping, or a wrong sign on alpha_cost_usd would
misdirect, so every assertion below is on a *specific* expected value
or verdict, not just "no crash".
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics import all_cash_streak as acs  # noqa: E402

_NOW = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


def _pt(hours_ago: float, total: float, cash: float, spy: float | None = 5500.0):
    """One equity_curve point. Oldest-first input convention; pass freshest
    point last."""
    ts = (_NOW - timedelta(hours=hours_ago)).isoformat()
    return {
        "timestamp": ts,
        "total_value": total,
        "cash": cash,
        "sp500_price": spy,
    }


# ─── NO_DATA / INSUFFICIENT_HISTORY ───────────────────────────────────

def test_empty_curve_returns_no_data():
    out = acs.build_all_cash_streak([], now=_NOW)
    assert out["state"] == "NO_DATA"
    assert out["verdict"] == "NO_DATA"
    assert out["n_points"] == 0
    assert out["current_streak"] is None
    assert out["streaks"] == []
    assert out["headline"]


def test_none_curve_returns_no_data():
    out = acs.build_all_cash_streak(None, now=_NOW)
    assert out["state"] == "NO_DATA"


def test_two_points_below_min_returns_insufficient_history():
    curve = [_pt(48.0, 1000.0, 1000.0), _pt(24.0, 1000.0, 1000.0)]
    out = acs.build_all_cash_streak(curve, now=_NOW)
    assert out["state"] == "OK"
    assert out["verdict"] == "INSUFFICIENT_HISTORY"
    assert out["n_points"] == 2


# ─── all-cash detection epsilon ───────────────────────────────────────

def test_is_all_cash_within_epsilon():
    # Position value of $0.25 < epsilon $0.50 ⇒ still considered all-cash.
    p = _pt(0.0, 1000.25, 1000.0)
    assert acs._is_all_cash(p) is True


def test_not_all_cash_above_epsilon():
    # Position value of $5 > epsilon ⇒ NOT all-cash.
    p = _pt(0.0, 1005.0, 1000.0)
    assert acs._is_all_cash(p) is False


def test_missing_fields_not_all_cash():
    assert acs._is_all_cash({"timestamp": "x"}) is False
    assert acs._is_all_cash({"cash": 100, "total_value": None}) is False
    # Non-numeric — defensive against malformed rows.
    assert acs._is_all_cash({"cash": "abc", "total_value": "abc"}) is False


# ─── current-streak path ──────────────────────────────────────────────

def test_currently_all_cash_brief_holdout():
    curve = [
        _pt(10.0, 1000.0, 500.0, spy=5500.0),   # held a position
        _pt(8.0, 1000.0, 500.0, spy=5510.0),    # still held
        _pt(4.0, 1000.0, 1000.0, spy=5520.0),   # closed out — all-cash starts
        _pt(2.0, 1000.0, 1000.0, spy=5530.0),
        _pt(0.5, 1000.0, 1000.0, spy=5550.0),
    ]
    out = acs.build_all_cash_streak(curve, now=_NOW)
    assert out["state"] == "OK"
    assert out["newest_is_all_cash"] is True
    assert out["current_streak"] is not None
    # Streak started 4h ago; current cycle is BRIEF (< 6h).
    assert out["verdict"] == "BRIEF_HOLDOUT"
    cs = out["current_streak"]
    assert cs["n_points"] == 3
    assert abs(cs["hours_elapsed_to_now"] - 4.0) < 0.01
    assert cs["cash_usd"] == 1000.0
    # SPY rose 5520 → 5550 = +0.543% over the streak.
    expected_spy = round((5550.0 - 5520.0) / 5520.0 * 100.0, 4)
    assert cs["spy_return_pct"] == expected_spy
    # alpha_cost = cash * spy_ret / 100, positive (cash sat through a rally).
    assert cs["alpha_cost_usd"] is not None
    assert cs["alpha_cost_usd"] > 0


def test_currently_all_cash_extended_holdout():
    # 24h all-cash streak ⇒ EXTENDED (between BRIEF and PROLONGED).
    pts = []
    # 30h ago: held a position, then closes 25h ago.
    pts.append(_pt(30.0, 1000.0, 200.0, spy=5500.0))
    pts.append(_pt(28.0, 1000.0, 200.0, spy=5510.0))
    for i in range(25, -1, -1):
        pts.append(_pt(float(i), 1000.0, 1000.0, spy=5500.0 + i * 0.5))
    out = acs.build_all_cash_streak(pts, now=_NOW)
    assert out["newest_is_all_cash"] is True
    assert out["verdict"] == "EXTENDED_HOLDOUT"
    cs = out["current_streak"]
    assert cs is not None
    assert 23.0 < cs["hours_elapsed_to_now"] < 27.0


def test_currently_all_cash_prolonged_holdout():
    # 50h all-cash streak ⇒ PROLONGED (≥ 48h).
    pts = [_pt(60.0, 1000.0, 200.0, spy=5500.0),
           _pt(55.0, 1000.0, 200.0, spy=5500.0)]
    for i in range(50, -1, -1):
        pts.append(_pt(float(i), 1000.0, 1000.0, spy=5500.0))
    out = acs.build_all_cash_streak(pts, now=_NOW)
    assert out["verdict"] == "PROLONGED_HOLDOUT"
    cs = out["current_streak"]
    assert cs is not None
    assert cs["hours_elapsed_to_now"] >= 48.0


# ─── NOT_ALL_CASH path ────────────────────────────────────────────────

def test_not_all_cash_reports_last_completed_streak():
    curve = [
        _pt(20.0, 1000.0, 1000.0, spy=5500.0),  # flat
        _pt(18.0, 1000.0, 1000.0, spy=5510.0),  # flat
        _pt(15.0, 1000.0, 1000.0, spy=5520.0),  # flat (end of streak A)
        _pt(10.0, 1000.0, 200.0, spy=5530.0),   # held — closes streak A
        _pt(5.0, 1010.0, 210.0, spy=5540.0),    # still held
        _pt(0.0, 1015.0, 215.0, spy=5550.0),    # still held — newest
    ]
    out = acs.build_all_cash_streak(curve, now=_NOW)
    assert out["state"] == "OK"
    assert out["newest_is_all_cash"] is False
    assert out["verdict"] == "NOT_ALL_CASH"
    assert out["current_streak"] is None
    last = out["most_recent_completed_streak"]
    assert last is not None
    assert last["n_points"] == 3
    # Streak ran 20h ago → 15h ago = 5h.
    assert abs(last["hours"] - 5.0) < 0.01
    assert "20.0h" not in out["headline"]  # the headline cites the COMPLETED streak length


def test_not_all_cash_no_prior_streak():
    curve = [_pt(10.0, 1000.0, 200.0, spy=5500.0),
             _pt(5.0, 1010.0, 210.0, spy=5510.0),
             _pt(0.0, 1015.0, 215.0, spy=5520.0)]
    out = acs.build_all_cash_streak(curve, now=_NOW)
    assert out["state"] == "OK"
    assert out["verdict"] == "NOT_ALL_CASH"
    assert out["most_recent_completed_streak"] is None


# ─── history / aggregate ──────────────────────────────────────────────

def test_multiple_closed_streaks_aggregated():
    # Two flat streaks broken up by a held position.
    curve = [
        _pt(50.0, 1000.0, 1000.0, spy=5500.0),  # streak A start
        _pt(48.0, 1000.0, 1000.0, spy=5510.0),  # streak A end
        _pt(45.0, 1000.0, 500.0, spy=5520.0),   # held — closes A
        _pt(40.0, 1000.0, 500.0, spy=5530.0),   # held
        _pt(35.0, 1000.0, 1000.0, spy=5540.0),  # streak B start
        _pt(33.0, 1000.0, 1000.0, spy=5550.0),
        _pt(30.0, 1000.0, 1000.0, spy=5560.0),  # streak B end
        _pt(25.0, 1000.0, 200.0, spy=5570.0),   # held — closes B
        _pt(20.0, 1000.0, 200.0, spy=5580.0),
        _pt(0.0, 1015.0, 215.0, spy=5600.0),    # still held — newest
    ]
    out = acs.build_all_cash_streak(curve, now=_NOW)
    assert out["verdict"] == "NOT_ALL_CASH"
    assert out["total_streaks"] == 2
    # Newest-first → streak B first.
    assert len(out["streaks"]) == 2
    sb = out["streaks"][0]
    sa = out["streaks"][1]
    assert abs(sb["hours"] - 5.0) < 0.01  # 35h → 30h
    assert abs(sa["hours"] - 2.0) < 0.01  # 50h → 48h
    # Aggregate flat hours = 5 + 2 = 7.
    assert abs(out["aggregate_flat_hours"] - 7.0) < 0.01


def test_history_cap_max_history():
    # Synthesize 25 alternating closed streaks; result must be capped at
    # _MAX_HISTORY=20 in the surfaced list, but total_streaks remains 25.
    pts = []
    for i in range(25):
        # Streak point i (flat).
        pts.append(_pt(200.0 - i * 4.0, 1000.0, 1000.0, spy=5500.0))
        # Then a held point to close the streak.
        pts.append(_pt(200.0 - i * 4.0 - 1.0, 1000.0, 100.0, spy=5510.0))
    out = acs.build_all_cash_streak(pts, now=_NOW)
    assert out["total_streaks"] == 25
    assert len(out["streaks"]) == acs._MAX_HISTORY


# ─── SPY benchmark inside streak ──────────────────────────────────────

def test_spy_return_none_when_too_few_marks():
    # Streak of 3 flat points but only one has a benchmarkable spy mark.
    curve = [
        _pt(10.0, 1000.0, 200.0, spy=5500.0),   # held
        _pt(8.0, 1000.0, 1000.0, spy=None),     # flat, no SPY
        _pt(6.0, 1000.0, 1000.0, spy=None),     # flat, no SPY
        _pt(4.0, 1000.0, 1000.0, spy=5520.0),   # flat, one SPY mark
    ]
    out = acs.build_all_cash_streak(curve, now=_NOW)
    cs = out["current_streak"]
    assert cs is not None
    assert cs["spy_return_pct"] is None
    assert cs["alpha_cost_usd"] is None


def test_negative_spy_means_cash_saved():
    # SPY drops during the all-cash streak ⇒ alpha_cost_usd is negative
    # (the operator-facing meaning is "cash SAVED you money"). The streak
    # is brief (< 6h) so the verdict is BRIEF_HOLDOUT.
    curve = [
        _pt(10.0, 1000.0, 200.0, spy=5500.0),   # held
        _pt(8.0, 1000.0, 200.0, spy=5500.0),    # held
        _pt(4.0, 1000.0, 1000.0, spy=5500.0),   # flat start
        _pt(2.0, 1000.0, 1000.0, spy=5450.0),
        _pt(0.0, 1000.0, 1000.0, spy=5400.0),   # flat end
    ]
    out = acs.build_all_cash_streak(curve, now=_NOW)
    cs = out["current_streak"]
    assert cs is not None
    assert cs["spy_return_pct"] < 0
    assert cs["alpha_cost_usd"] < 0
    # Headline should mention "saved" not "cost".
    assert "saved" in out["headline"]


def test_zero_spy_change_neutral_alpha():
    curve = [
        _pt(10.0, 1000.0, 200.0, spy=5500.0),   # held
        _pt(8.0, 1000.0, 200.0, spy=5500.0),    # held
        _pt(4.0, 1000.0, 1000.0, spy=5500.0),   # flat
        _pt(2.0, 1000.0, 1000.0, spy=5500.0),
        _pt(0.0, 1000.0, 1000.0, spy=5500.0),
    ]
    out = acs.build_all_cash_streak(curve, now=_NOW)
    cs = out["current_streak"]
    assert cs is not None
    assert cs["spy_return_pct"] == 0.0
    assert cs["alpha_cost_usd"] == 0.0


# ─── output stability / never-raises discipline ───────────────────────

def test_malformed_timestamp_does_not_raise():
    curve = [
        _pt(10.0, 1000.0, 200.0, spy=5500.0),
        {"timestamp": "garbage", "total_value": 1000.0,
         "cash": 1000.0, "sp500_price": 5510.0},
        _pt(0.0, 1000.0, 1000.0, spy=5520.0),
    ]
    out = acs.build_all_cash_streak(curve, now=_NOW)
    # Doesn't raise; surfaces something coherent.
    assert "verdict" in out


def test_thresholds_surfaced_in_response():
    out = acs.build_all_cash_streak([_pt(0.0, 1000.0, 1000.0)], now=_NOW)
    thr = out["thresholds"]
    assert thr["flat_epsilon_usd"] == acs._FLAT_EPSILON_USD
    assert thr["brief_hours"] == acs._BRIEF_HOURS
    assert thr["prolonged_hours"] == acs._PROLONGED_HOURS
    assert thr["max_history"] == acs._MAX_HISTORY


def test_aggregate_alpha_cost_excludes_none_values():
    # First streak has full SPY data; second doesn't.
    curve = [
        _pt(50.0, 1000.0, 1000.0, spy=5500.0),
        _pt(48.0, 1000.0, 1000.0, spy=5510.0),  # streak A end
        _pt(45.0, 1000.0, 500.0, spy=5520.0),   # held
        _pt(40.0, 1000.0, 500.0, spy=5530.0),
        _pt(35.0, 1000.0, 1000.0, spy=None),    # streak B start, no SPY
        _pt(33.0, 1000.0, 1000.0, spy=None),
        _pt(30.0, 1000.0, 1000.0, spy=None),    # streak B end, all None
        _pt(25.0, 1000.0, 200.0, spy=5570.0),   # held
        _pt(0.0, 1015.0, 215.0, spy=5600.0),    # held — newest
    ]
    out = acs.build_all_cash_streak(curve, now=_NOW)
    assert out["total_streaks"] == 2
    # Aggregate is non-None because streak A had SPY data; streak B contributes nothing.
    assert out["aggregate_alpha_cost_usd"] is not None
