"""Tests for analytics/session_delta.py — "what changed since you last looked".

Exact-value, hand-computed. Pins the six correctness items the design review
flagged: strict-exclusive window boundary applied identically to every event
class; EQUITY_MOVE anchor (at-or-before / older-than-all fallback / single
point); SPY-NULL → drop alpha but keep Δ%; INACTION free-text classification
of all four observed ``action_taken`` shapes; never-raises (a faulting
build_round_trips drops only POSITION_CLOSED); and an end-to-end Flask
test-client check that seeds equity points spanning the window (module
``__main__`` smoke hits a different/empty DB — see AGENTS.md).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics import session_delta as sd
from paper_trader.analytics.round_trips import build_round_trips
from paper_trader.analytics.session_delta import (
    DD_PCT,
    INACTION_MIN_CYCLES,
    build_session_delta,
)

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ts(min_offset: float, sec: int = 0) -> str:
    return (_BASE + timedelta(minutes=min_offset, seconds=sec)).isoformat()


def _dt(min_offset: float, sec: int = 0) -> datetime:
    return _BASE + timedelta(minutes=min_offset, seconds=sec)


def _trade(tid, ticker, action, min_off, qty=1.0, px=100.0):
    val = qty * px
    return {"id": tid, "timestamp": _ts(min_off), "ticker": ticker,
            "action": action, "qty": qty, "price": px, "value": val,
            "reason": "", "strike": None, "expiry": None, "option_type": None}


def _eq(min_off, total, cash=100.0, spy=5000.0):
    return {"timestamp": _ts(min_off), "total_value": total, "cash": cash,
            "sp500_price": spy}


def _dec(min_off, action_taken):
    return {"timestamp": _ts(min_off), "market_open": 1, "signal_count": 0,
            "action_taken": action_taken, "reasoning": "", "id": int(min_off)}


# ── 1. window boundary: strict-exclusive lower, identical for all classes ──
class TestWindowBoundary:
    def test_trade_exactly_at_since_excluded_one_sec_after_included(self):
        since = _dt(60)
        now = _dt(120)
        at = build_session_delta([_trade(1, "AMD", "BUY", 60)], [], [],
                                  since, now)
        # exactly at since_ts → "before you looked" → excluded
        assert at["n_fills"] == 0
        assert at["state"] == "QUIET"
        after = build_session_delta(
            [{**_trade(1, "AMD", "BUY", 60), "timestamp": _ts(60, sec=1)}],
            [], [], since, now)
        assert after["n_fills"] == 1
        assert after["state"] == "ACTIVE"
        assert after["events"][0]["kind"] == "TRADE"

    def test_position_closed_keys_on_exit_ts_in_window(self):
        # opened before the window, closed inside it → in scope.
        trades = [_trade(1, "NVDA", "BUY", 10, qty=1, px=100.0),
                  _trade(2, "NVDA", "SELL", 90, qty=1, px=130.0)]
        r = build_session_delta(trades, [], [], _dt(60), _dt(120))
        closes = [e for e in r["events"] if e["kind"] == "POSITION_CLOSED"]
        assert len(closes) == 1
        # P&L consumed verbatim from build_round_trips (SSOT #10)
        rt = build_round_trips(trades)[0]
        assert closes[0]["pnl_usd"] == rt["pnl_usd"] == 30.0
        assert r["n_closed"] == 1
        assert r["net_realized_usd"] == 30.0

    def test_position_closed_before_window_excluded(self):
        trades = [_trade(1, "NVDA", "BUY", 10), _trade(2, "NVDA", "SELL", 40)]
        r = build_session_delta(trades, [], [], _dt(60), _dt(120))
        assert [e for e in r["events"] if e["kind"] == "POSITION_CLOSED"] == []


# ── 2. EQUITY_MOVE anchor sub-cases ──
class TestEquityAnchor:
    def test_anchor_is_at_or_before_since(self):
        # points at t=0,55,115; since=60 → anchor must be the t=55 point
        # ("value when you last looked"), not the first in-window point.
        eq = [_eq(0, 1000.0), _eq(55, 1005.0), _eq(115, 1002.0)]
        r = build_session_delta([], [], eq, _dt(60), _dt(120))
        ev = [e for e in r["events"] if e["kind"] == "EQUITY_MOVE"][0]
        assert ev["anchor_value"] == 1005.0          # the t=55 point
        assert ev["latest_value"] == 1002.0
        assert ev["delta_usd"] == round(1002.0 - 1005.0, 2)
        assert r["anchor_fallback"] is False

    def test_since_older_than_all_history_flags_fallback(self):
        eq = [_eq(70, 1000.0), _eq(110, 1010.0)]
        r = build_session_delta([], [], eq, _dt(10), _dt(120))
        ev = [e for e in r["events"] if e["kind"] == "EQUITY_MOVE"][0]
        assert ev["anchor_value"] == 1000.0          # earliest, no point ≤ since
        assert ev["anchor_fallback"] is True
        assert r["anchor_fallback"] is True

    def test_single_equity_point_emits_no_move(self):
        r = build_session_delta([], [], [_eq(90, 1000.0)], _dt(60), _dt(120))
        assert [e for e in r["events"] if e["kind"] == "EQUITY_MOVE"] == []
        assert r["equity_delta_usd"] is None

    def test_all_points_before_since_no_move(self):
        eq = [_eq(10, 1000.0), _eq(20, 1010.0)]
        r = build_session_delta([], [], eq, _dt(60), _dt(120))
        assert [e for e in r["events"] if e["kind"] == "EQUITY_MOVE"] == []


# ── 3. SPY NULL → drop alpha, keep Δ% ──
class TestSpyNullHandling:
    def test_null_spy_on_anchor_drops_alpha_keeps_delta(self):
        eq = [{**_eq(55, 1000.0), "sp500_price": None}, _eq(115, 1010.0)]
        r = build_session_delta([], [], eq, _dt(60), _dt(120))
        ev = [e for e in r["events"] if e["kind"] == "EQUITY_MOVE"][0]
        assert ev["delta_pct"] == pytest.approx(1.0)
        assert ev["alpha_pp"] is None
        assert ev["spy_delta_pct"] is None
        assert r["alpha_pp"] is None

    def test_both_spy_present_computes_alpha(self):
        eq = [_eq(55, 1000.0, spy=5000.0), _eq(115, 1010.0, spy=5025.0)]
        r = build_session_delta([], [], eq, _dt(60), _dt(120))
        ev = [e for e in r["events"] if e["kind"] == "EQUITY_MOVE"][0]
        assert ev["delta_pct"] == pytest.approx(1.0)
        assert ev["spy_delta_pct"] == pytest.approx(0.5)
        assert ev["alpha_pp"] == pytest.approx(0.5)


# ── 4. INACTION free-text classification ──
class TestInaction:
    def test_four_action_shapes_classified(self):
        decs = [
            _dec(70, "HOLD NONE → HOLD"),
            _dec(75, "NO_DECISION"),
            _dec(80, "BLOCKED SELL exceeds held qty"),
            _dec(85, "HOLD NVDA → HOLD"),
        ]
        r = build_session_delta([], decs, [], _dt(60), _dt(120))
        ic = r["inaction"]
        assert ic is not None
        assert ic["n_cycles"] == 4
        assert ic["n_hold"] == 2
        assert ic["n_no_decision"] == 1
        assert ic["n_blocked"] == 1
        ev = [e for e in r["events"] if e["kind"] == "INACTION"][0]
        assert ev["severity"] == "MED"

    def test_a_filled_cycle_suppresses_inaction(self):
        decs = [_dec(70, "HOLD NONE → HOLD"),
                _dec(75, "BUY NVDA → FILLED"),
                _dec(80, "NO_DECISION")]
        r = build_session_delta([], decs, [], _dt(60), _dt(120))
        assert r["inaction"] is None
        assert [e for e in r["events"] if e["kind"] == "INACTION"] == []

    def test_below_min_cycles_no_inaction(self):
        decs = [_dec(70, "HOLD NONE → HOLD")] * 1
        assert INACTION_MIN_CYCLES > 1
        r = build_session_delta([], decs, [], _dt(60), _dt(120))
        assert r["inaction"] is None


# ── 5. never-raises: a faulting build_round_trips drops only one class ──
class TestNeverRaises:
    def test_round_trips_blowing_up_does_not_sink_report(self, monkeypatch):
        def _boom(_):
            raise RuntimeError("synthetic round_trips fault")

        monkeypatch.setattr(sd, "build_round_trips", _boom)
        trades = [{**_trade(1, "AMD", "BUY", 70), "timestamp": _ts(70)}]
        eq = [_eq(55, 1000.0), _eq(115, 1010.0)]
        r = build_session_delta(trades, [], eq, _dt(60), _dt(120))
        assert r["state"] in ("QUIET", "ACTIVE")
        kinds = {e["kind"] for e in r["events"]}
        assert "POSITION_CLOSED" not in kinds      # the faulting class is gone
        assert "TRADE" in kinds                     # siblings survive
        assert "EQUITY_MOVE" in kinds


# ── DRAWDOWN_LOW window-scoped peak→trough ──
class TestDrawdown:
    def test_intra_window_drawdown_detected_above_threshold(self):
        # anchor 1000 (t=55), peak 1010 (t=70), trough 980 (t=90) → DD ≈2.97%.
        eq = [_eq(55, 1000.0), _eq(70, 1010.0), _eq(90, 980.0),
              _eq(115, 1005.0)]
        r = build_session_delta([], [], eq, _dt(60), _dt(120))
        dd = [e for e in r["events"] if e["kind"] == "DRAWDOWN_LOW"]
        assert len(dd) == 1
        assert dd[0]["peak_value"] == 1010.0
        assert dd[0]["trough_value"] == 980.0
        # builder rounds to 4dp (codebase convention) — assert the rounded value
        assert dd[0]["drawdown_pct"] == round((1010 - 980) / 1010 * 100, 4)
        assert dd[0]["drawdown_pct"] >= DD_PCT

    def test_shallow_dip_below_threshold_no_event(self):
        eq = [_eq(55, 1000.0), _eq(90, 999.0), _eq(115, 1000.0)]
        r = build_session_delta([], [], eq, _dt(60), _dt(120))
        assert [e for e in r["events"] if e["kind"] == "DRAWDOWN_LOW"] == []


# ── ranking + state machine ──
class TestRankingAndState:
    def test_high_before_med_recent_first_within_severity(self):
        trades = [_trade(1, "AMD", "BUY", 70), _trade(2, "MU", "BUY", 110)]
        eq = [_eq(55, 1000.0), _eq(115, 1000.4)]   # +0.04% → MED equity move
        r = build_session_delta(trades, [], eq, _dt(60), _dt(120))
        sevs = [e["severity"] for e in r["events"]]
        # all HIGH (the two trades) precede the MED equity move
        assert sevs == sorted(sevs, key=lambda s: sd._SEV_RANK[s])
        highs = [e for e in r["events"] if e["severity"] == "HIGH"]
        assert highs[0]["ticker"] == "MU"          # t=110, more recent first
        assert highs[1]["ticker"] == "AMD"         # t=70

    def test_empty_store_is_no_data(self):
        r = build_session_delta([], [], [], _dt(60), _dt(120))
        assert r["state"] == "NO_DATA"
        assert r["n_events"] == 0
        assert "No trader activity" in r["headline"]

    def test_data_but_outside_window_is_quiet(self):
        r = build_session_delta([_trade(1, "AMD", "BUY", 10)], [], [],
                                _dt(60), _dt(120))
        assert r["state"] == "QUIET"
        assert "Quiet since" in r["headline"]

    def test_future_since_clamped_no_negative_window(self):
        r = build_session_delta([], [], [], _dt(999), _dt(120))
        assert r["window_seconds"] >= 0
        assert r["state"] == "NO_DATA"


# ── 6. end-to-end via the Flask test client (NOT module __main__) ──
@pytest.fixture
def seeded_client(tmp_path, monkeypatch):
    from paper_trader import store as store_mod
    from paper_trader.store import Store

    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    now = datetime.now(timezone.utc)
    # Anchor point ~2h ago (before a 60-min since), a fill ~30m ago, and a
    # fresher equity point — the discriminating EQUITY_MOVE needs points that
    # span the window with non-NULL sp500 (design-review item #6).
    s.conn.execute(
        "INSERT INTO equity_curve (timestamp,total_value,cash,sp500_price) "
        "VALUES (?,?,?,?)",
        ((now - timedelta(hours=2)).isoformat(), 1000.0, 500.0, 5000.0))
    s.conn.execute(
        "INSERT INTO equity_curve (timestamp,total_value,cash,sp500_price) "
        "VALUES (?,?,?,?)",
        ((now - timedelta(minutes=5)).isoformat(), 1012.0, 500.0, 5025.0))
    s.conn.execute(
        "INSERT INTO trades (timestamp,ticker,action,qty,price,value,reason) "
        "VALUES (?,?,?,?,?,?,?)",
        ((now - timedelta(minutes=30)).isoformat(), "NVDA", "BUY",
         0.4, 225.79, 90.32, "momentum"))
    s.conn.commit()
    s.update_portfolio(cash=500.0, total_value=1012.0, positions=[])

    from paper_trader import dashboard
    dashboard.app.config["TESTING"] = True
    try:
        with dashboard.app.test_client() as client:
            yield client
    finally:
        s.close()


class TestSessionDeltaEndpoint:
    def test_endpoint_returns_active_with_equity_move(self, seeded_client):
        resp = seeded_client.get("/api/session-delta?minutes=60")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "error" not in data, data
        assert data["state"] == "ACTIVE"
        assert {"as_of", "since", "window_seconds", "window_label",
                "headline", "events", "n_fills"} <= set(data)
        kinds = {e["kind"] for e in data["events"]}
        assert "TRADE" in kinds
        assert "EQUITY_MOVE" in kinds
        em = [e for e in data["events"] if e["kind"] == "EQUITY_MOVE"][0]
        assert em["delta_usd"] == pytest.approx(12.0, abs=0.01)
        assert em["alpha_pp"] is not None        # both sp500 points non-NULL

    def test_endpoint_quiet_when_window_too_short(self, seeded_client):
        # 5-minute window: the fill is ~30m old → nothing material in window.
        resp = seeded_client.get("/api/session-delta?minutes=5")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["state"] in ("QUIET", "ACTIVE")
        assert data["n_fills"] == 0
