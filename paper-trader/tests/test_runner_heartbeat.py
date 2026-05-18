"""Exact-value tests for analytics/runner_heartbeat.build_runner_heartbeat
plus the /api/runner-heartbeat endpoint end-to-end.

The feature's whole point: every other diagnostic panel reasons over rows
that *exist* in `decisions` (drought/reliability/feed-health) or over code
SHA / article age (build-info). None close a verdict on
``now - max(decisions.timestamp)`` vs the runner's expected cadence, so a
dead/wedged `paper_trader.runner` is invisible — the panels show
frozen-but-plausible state. These pin the exact cadence arithmetic, the
market-open (1800s) vs market-closed (3600s) threshold selection, the
NO_DATA / STALLED / LAGGING / HEALTHY verdict precedence + boundaries, the
future-skew clamp, the constant echo (retune-proof), and the endpoint
behaviour through the Flask test client on a real temp Store (per the
paper-trader-analytics-verification discipline — never a __main__ smoke).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.runner_heartbeat import (
    build_runner_heartbeat,
    OPEN_INTERVAL_S,
    CLOSED_INTERVAL_S,
    LAGGING_MULT,
    STALLED_MULT,
)

NOW = datetime(2026, 5, 17, 14, 0, 0, tzinfo=timezone.utc)


def _ago(seconds: float) -> str:
    return (NOW - timedelta(seconds=seconds)).isoformat()


# ─────────────────────── constant echo (retune-proof) ───────────────────────

def test_module_constants_are_the_spec():
    assert OPEN_INTERVAL_S == 1800.0
    assert CLOSED_INTERVAL_S == 3600.0
    assert LAGGING_MULT == 1.25
    assert STALLED_MULT == 2.0


def test_output_echoes_constants_and_inputs():
    out = build_runner_heartbeat(_ago(600), market_open=True, now=NOW)
    assert out["expected_interval_s"] == OPEN_INTERVAL_S
    assert out["lagging_mult"] == LAGGING_MULT
    assert out["stalled_mult"] == STALLED_MULT
    assert out["market_open"] is True
    assert out["as_of"] == NOW.isoformat(timespec="seconds")


# ─────────────────────────── HEALTHY ───────────────────────────

def test_recent_decision_market_open_is_healthy():
    out = build_runner_heartbeat(_ago(600), market_open=True, now=NOW)
    assert out["verdict"] == "HEALTHY"
    assert out["restart_recommended"] is False
    assert out["secs_since_last_decision"] == pytest.approx(600.0)
    assert out["intervals_elapsed"] == round(600.0 / OPEN_INTERVAL_S, 3)
    assert "HEALTHY" in out["headline"]


def test_just_under_lagging_is_still_healthy():
    # build the boundary FROM the module constant so a retune can't false-fail
    secs = LAGGING_MULT * OPEN_INTERVAL_S - 60
    out = build_runner_heartbeat(_ago(secs), market_open=True, now=NOW)
    assert out["verdict"] == "HEALTHY"


# ─────────────────────────── LAGGING ───────────────────────────

def test_just_over_lagging_is_lagging_not_stalled():
    secs = LAGGING_MULT * OPEN_INTERVAL_S + 60
    out = build_runner_heartbeat(_ago(secs), market_open=True, now=NOW)
    assert out["verdict"] == "LAGGING"
    assert out["restart_recommended"] is False          # LAGGING does not recommend restart
    assert "LAGGING" in out["headline"]


def test_just_under_stalled_is_lagging():
    secs = STALLED_MULT * OPEN_INTERVAL_S - 60
    out = build_runner_heartbeat(_ago(secs), market_open=True, now=NOW)
    assert out["verdict"] == "LAGGING"


# ─────────────────────────── STALLED ───────────────────────────

def test_just_over_stalled_is_stalled_and_recommends_restart():
    secs = STALLED_MULT * OPEN_INTERVAL_S + 60
    out = build_runner_heartbeat(_ago(secs), market_open=True, now=NOW)
    assert out["verdict"] == "STALLED"
    assert out["restart_recommended"] is True
    assert "STALLED" in out["headline"]
    assert "estart" in out["headline"]                  # "Restart paper-trader"


# ─────────────────── market-open vs market-closed cadence ───────────────────

def test_same_gap_different_verdict_by_market_state():
    """A 70-min (4200s) silence: STALLED while open (4200/1800=2.33), but
    HEALTHY while closed (4200/3600=1.17 < 1.25). This is the whole reason
    the cadence selector exists — pin it with one elapsed gap."""
    gap = 70 * 60
    open_out = build_runner_heartbeat(_ago(gap), market_open=True, now=NOW)
    closed_out = build_runner_heartbeat(_ago(gap), market_open=False, now=NOW)
    assert open_out["verdict"] == "STALLED"
    assert open_out["expected_interval_s"] == OPEN_INTERVAL_S
    assert closed_out["verdict"] == "HEALTHY"
    assert closed_out["expected_interval_s"] == CLOSED_INTERVAL_S


# ─────────────────────────── NO_DATA ───────────────────────────

def test_none_timestamp_is_no_data():
    out = build_runner_heartbeat(None, market_open=True, now=NOW)
    assert out["verdict"] == "NO_DATA"
    assert out["restart_recommended"] is False
    assert out["secs_since_last_decision"] is None
    assert out["intervals_elapsed"] is None
    assert out["last_decision_ts"] is None


def test_unparseable_timestamp_is_no_data_never_raises():
    out = build_runner_heartbeat("not-a-timestamp", market_open=False, now=NOW)
    assert out["verdict"] == "NO_DATA"
    assert out["restart_recommended"] is False


# ─────────────────────── future-skew clamp ───────────────────────

def test_future_timestamp_is_healthy_clamped_not_stalled():
    """A clock-skewed future ts is a *just-written* decision, not a stall."""
    out = build_runner_heartbeat(_ago(-300), market_open=True, now=NOW)
    assert out["verdict"] == "HEALTHY"
    assert out["intervals_elapsed"] == 0.0
    assert out["restart_recommended"] is False


# ═══════════════════════ endpoint (Flask test client) ═══════════════════════

@pytest.fixture
def client(tmp_path, monkeypatch):
    from paper_trader import store as store_mod
    from paper_trader.store import Store

    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()

    from paper_trader import dashboard
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c, s
    s.close()
    store_mod._singleton = None


def test_endpoint_no_data_on_empty_decisions(client):
    c, _s = client
    r = c.get("/api/runner-heartbeat")
    assert r.status_code == 200
    d = r.get_json()
    assert d["verdict"] == "NO_DATA"
    assert d["restart_recommended"] is False


def test_endpoint_stalled_on_old_decision(client):
    """A 5h-old decision is STALLED whether the market is open (18000/1800=10)
    or closed (18000/3600=5) — both > STALLED_MULT — so the assertion is
    deterministic regardless of when the suite runs."""
    c, s = client
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    with s._lock:
        s.conn.execute(
            "INSERT INTO decisions (timestamp, market_open, signal_count, "
            "action_taken, reasoning, portfolio_value, cash) "
            "VALUES (?,?,?,?,?,?,?)",
            (stale_ts, 0, 0, "HOLD NONE → HOLD", "{}", 972.0, 6.0),
        )
        s.conn.commit()
    r = c.get("/api/runner-heartbeat")
    assert r.status_code == 200
    d = r.get_json()
    assert d["verdict"] == "STALLED"
    assert d["restart_recommended"] is True
    assert d["secs_since_last_decision"] >= 5 * 3600 - 120


def test_endpoint_healthy_on_fresh_decision(client):
    c, s = client
    s.record_decision(True, 3, "BUY NVDA → FILLED", "{}", 1000.0, 100.0)
    r = c.get("/api/runner-heartbeat")
    assert r.status_code == 200
    d = r.get_json()
    assert d["verdict"] == "HEALTHY"
    assert d["restart_recommended"] is False


def test_endpoint_surfaces_degraded_singleton_lock(client, monkeypatch):
    """Additive (2026-05-18): /api/runner-heartbeat exposes the lock state of
    the runner serving the dashboard so a guard-less (degraded) runner —
    double-trade risk — is no longer invisible. The pre-existing liveness
    verdict is unchanged (a different, test-locked concern)."""
    c, s = client
    s.record_decision(True, 3, "BUY NVDA → FILLED", "{}", 1000.0, 100.0)
    from paper_trader import runner as _runner
    monkeypatch.setattr(_runner, "singleton_lock_state", lambda: {
        "status": "degraded", "holder_pid": None,
        "have_lock": False, "degraded": True})
    r = c.get("/api/runner-heartbeat")
    assert r.status_code == 200
    d = r.get_json()
    assert d["verdict"] == "HEALTHY"  # unchanged
    assert d["singleton_lock"]["degraded"] is True
    assert d["singleton_lock"]["headline"].startswith("DEGRADED")


def test_endpoint_singleton_lock_ok_when_acquired(client, monkeypatch):
    c, s = client
    s.record_decision(True, 3, "BUY NVDA → FILLED", "{}", 1000.0, 100.0)
    from paper_trader import runner as _runner
    monkeypatch.setattr(_runner, "singleton_lock_state", lambda: {
        "status": "acquired", "holder_pid": 4242,
        "have_lock": True, "degraded": False})
    r = c.get("/api/runner-heartbeat")
    d = r.get_json()
    assert d["singleton_lock"]["degraded"] is False
    assert d["singleton_lock"]["headline"].startswith("OK")
    assert "4242" in d["singleton_lock"]["headline"]
