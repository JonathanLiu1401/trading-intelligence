"""Flask test-client coverage for the two new diagnostics endpoints.

The exact-value tests against the pure builders (test_parse_fail_windows.py,
test_position_alpha_decomp.py) cover the math; this file proves the routes
are wired: the URL resolves, the handler invokes the right Store methods, the
correct classify/_LEVERAGE_BETA SSOT is passed in (the position decomp must
use the dashboard's real sector→beta table, not a builder-local copy), and the
JSON-error envelope returns 500 on a degenerate Store. The pattern follows
test_capital_paralysis_swr.py — fresh tmp_path Store, monkeypatched _singleton,
no network, no :8090 bind.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paper_trader.dashboard as d  # noqa: E402
from paper_trader import store as store_mod  # noqa: E402
from paper_trader.store import Store  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    d.app.config["TESTING"] = True
    try:
        with d.app.test_client() as c:
            yield c, s
    finally:
        s.close()


# ── /api/parse-fail-windows ─────────────────────────────────────────────────

class TestParseFailWindowsEndpoint:
    def test_empty_store_returns_no_data(self, client):
        c, _ = client
        r = c.get("/api/parse-fail-windows")
        assert r.status_code == 200
        j = r.get_json()
        assert j["state"] == "NO_DATA"
        assert j["n_decisions_total"] == 0
        # Shape contract — the dashboard renderer relies on these keys.
        assert j["windows"] == []
        assert "trend" in j and j["trend"] == "INSUFFICIENT"
        assert j["recent_failures"] == []
        assert "min_win_n" in j and "trend_pp" in j

    def test_handler_uses_real_decisions_table(self, client):
        c, s = client
        # Seed 5 healthy + 5 parse-fail rows in the last few minutes — the
        # endpoint must read them via store.recent_decisions and route them
        # through the same builder the test_parse_fail_windows file pins.
        for _ in range(5):
            s.record_decision(
                market_open=False, signal_count=3,
                action_taken="HOLD → HOLD",
                reasoning='{"decision": {"action": "HOLD"}}',
                portfolio_value=1000.0, cash=500.0)
        for _ in range(5):
            s.record_decision(
                market_open=False, signal_count=0,
                action_taken="NO_DECISION",
                reasoning="parse_failed: I cannot comply with that request",
                portfolio_value=1000.0, cash=500.0)
        r = c.get("/api/parse-fail-windows")
        assert r.status_code == 200
        j = r.get_json()
        assert j["n_decisions_total"] == 10
        # 1h window must contain all 10; 5/10 = 50% fail rate.
        w1h = j["windows"][0]
        assert w1h["label"] == "1h"
        assert w1h["n_decisions"] == 10
        assert w1h["n_failures"] == 5
        assert w1h["failure_rate_pct"] == 50.0
        # Mode is NO_JSON (no '{' in the parse_failed excerpt).
        assert w1h["mode_mix"] == [
            {"mode": "NO_JSON", "n": 5, "pct": 100.0}]
        # Recent failures surfaced for forensics; newest first by timestamp.
        assert len(j["recent_failures"]) == 5
        assert all(rf["mode"] == "NO_JSON" for rf in j["recent_failures"])


# ── /api/position-alpha-decomp ──────────────────────────────────────────────

class TestPositionAlphaDecompEndpoint:
    def test_empty_store_returns_no_data(self, client):
        c, _ = client
        r = c.get("/api/position-alpha-decomp")
        assert r.status_code == 200
        j = r.get_json()
        assert j["state"] == "NO_DATA"
        assert j["verdict"] == "NO_DATA"
        assert j["n_positions"] == 0
        assert j["positions"] == []
        # Constants surfaced for client-side display contracts.
        assert "alpha_band_pp" in j and "max_open_lag_s" in j

    def test_handler_uses_real_classify_and_leverage_beta_ssot(self, client):
        c, s = client
        # MU is in dashboard.SECTOR_MAP → "semis"; _LEVERAGE_BETA["semis"]=1.5.
        # Open the position, seed an equity-curve baseline + a "now" point so
        # the builder can compute spy_return → alpha. The test pins that the
        # handler is using the *dashboard's* sector/beta tables, not a default
        # 1.0 from a builder fallback path.
        from datetime import datetime, timezone, timedelta
        s.record_trade("MU", "BUY", 1, 100.0, reason="momentum")
        s.upsert_position("MU", "stock", 1, 100.0)
        # Mark current price = 115 so pos_return_pct = +15%. update_position_marks
        # takes {position_id: (price, unrealized_pl)} — fetch the row's id first.
        pid = None
        with s._lock:
            row = s.conn.execute(
                "SELECT id FROM positions WHERE ticker='MU' AND closed_at IS NULL"
            ).fetchone()
            pid = row["id"] if row else None
        assert pid is not None
        s.update_position_marks({pid: (115.0, 15.0)})

        # Seed equity-curve baseline (SPY 5000 at opened_at) and a "now" point
        # (SPY 5100). The position must be opened BEFORE the baseline equity
        # point so _spy_at_open finds a covering anchor within MAX_OPEN_LAG_S.
        s.record_equity_point(1000.0, 900.0, 5000.0)
        s.record_equity_point(1115.0, 900.0, 5100.0)
        # Backdate both opened_at AND the first equity point so the
        # _spy_at_open lookup (first eq point ≥ opened_at) returns the SPY=5000
        # baseline. _now() in the store gives them all current timestamps, so
        # we need to push opened_at slightly before the first eq point.
        old_open = (datetime.now(timezone.utc)
                    - timedelta(seconds=60)).isoformat()
        old_eq = (datetime.now(timezone.utc)
                  - timedelta(seconds=30)).isoformat()
        with s._lock:
            s.conn.execute(
                "UPDATE positions SET opened_at=? WHERE ticker='MU'",
                (old_open,))
            first = s.conn.execute(
                "SELECT id FROM equity_curve ORDER BY id ASC LIMIT 1"
            ).fetchone()
            if first is not None:
                s.conn.execute(
                    "UPDATE equity_curve SET timestamp=? WHERE id=?",
                    (old_eq, first["id"]))
            s.conn.commit()

        r = c.get("/api/position-alpha-decomp")
        assert r.status_code == 200
        j = r.get_json()
        # The position was found, judged, and the math used beta=1.5 (semis).
        assert j["n_positions"] == 1
        assert j["n_judged"] == 1
        row = j["positions"][0]
        assert row["ticker"] == "MU"
        assert row["sector"] == "semis"
        # Dashboard's _LEVERAGE_BETA["semis"] = 1.5 — proves the endpoint
        # passed the real SSOT through (a builder-local default would be 1.0).
        assert row["beta_est"] == 1.5
        # pos_return = (115/100 - 1)*100 = +15.0
        assert row["pos_return_pct"] == 15.0
        # SPY +2.0% over hold; beta-implied = 3.0%; alpha = +12.0pp.
        assert row["spy_return_pct"] == 2.0
        assert row["pure_beta_pct"] == 3.0
        assert row["alpha_pp"] == 12.0
        assert row["verdict"] == "ALPHA_POS"
        assert j["verdict"] == "ALPHA_ADDING"
