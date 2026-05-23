"""Tests for /api/last-real-decision — the dedicated "when did the engine
last actually decide?" surface that wraps ``store.last_real_decision`` with
a verdict ladder (NEVER / FRESH / DELAYED / STALE).

These pin the exact verdict precedence + boundary arithmetic, the NEVER
empty-store contract, the parse of free-text ``action_taken`` into
ticker/verb/status (AGENTS.md invariant #11), and the end-to-end Flask
endpoint behaviour on a real temp Store (the paper-trader-analytics-
verification discipline — never a __main__ smoke).
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
    OPEN_INTERVAL_S,
    CLOSED_INTERVAL_S,
    LAGGING_MULT,
    STALLED_MULT,
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fresh-temp-Store dashboard client. Mirrors the
    test_runner_heartbeat.py client fixture so the endpoint runs against a
    real Store, not a mock — the verdict bands depend on actual row writes
    + the wall clock."""
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


# ───────────────────────── NEVER ─────────────────────────


def test_empty_store_returns_never(client):
    c, _s = client
    d = c.get("/api/last-real-decision").get_json()
    assert d["verdict"] == "NEVER"
    assert d["row"] is None
    assert d["secs_since"] is None
    assert d["age"] is None
    assert d["ticker"] is None
    assert "never produced" in d["headline"].lower()


def test_only_no_decision_rows_returns_never(client):
    """A book that has cycled 10 times but each cycle is a NO_DECISION row
    must still read NEVER — the whole point of this endpoint is to ignore
    NO_DECISION noise the bare ``recent_decisions`` would surface."""
    c, s = client
    for _ in range(10):
        s.record_decision(False, 0, "NO_DECISION",
                          "claude returned no response (timeout)",
                          972.0, 6.0)
    d = c.get("/api/last-real-decision").get_json()
    assert d["verdict"] == "NEVER"
    assert d["row"] is None


# ───────────────────────── FRESH ─────────────────────────


def test_fresh_filled_decision(client):
    c, s = client
    s.record_decision(True, 5, "BUY NVDA → FILLED",
                      '{"decision": {"reasoning": "test"}}',
                      1500.0, 200.0)
    d = c.get("/api/last-real-decision").get_json()
    assert d["verdict"] == "FRESH"
    assert d["ticker"] == "NVDA"
    assert d["action_verb"] == "BUY"
    assert d["status"] == "FILLED"
    assert d["secs_since"] is not None and d["secs_since"] < 60.0
    assert d["row"]["action_taken"] == "BUY NVDA → FILLED"


def test_hold_counts_as_a_real_decision(client):
    """A HOLD is a deliberate agency choice (not a NO_DECISION noise row);
    the store-side primitive includes it and the endpoint must mirror that."""
    c, s = client
    s.record_decision(True, 3, "HOLD MU → HOLD", "{}", 1000.0, 100.0)
    d = c.get("/api/last-real-decision").get_json()
    assert d["verdict"] == "FRESH"
    assert d["ticker"] == "MU"
    assert d["action_verb"] == "HOLD"


def test_blocked_counts_as_a_real_decision(client):
    """A BLOCKED is a real decision the risk-check rejected — meaningfully
    different from a NO_DECISION cycle (the engine DID decide; the action
    was just rejected)."""
    c, s = client
    s.record_decision(True, 3, "SELL X → BLOCKED",
                      "no open stock position in X to close",
                      1000.0, 100.0)
    d = c.get("/api/last-real-decision").get_json()
    assert d["verdict"] == "FRESH"
    assert d["status"] == "BLOCKED"
    assert d["action_verb"] == "SELL"


# ────────────────────── DELAYED / STALE ──────────────────────


def _backdate_last(store, hours: float, action_taken: str = "BUY NVDA → FILLED"):
    """Insert a decision row directly with a past timestamp (the public API
    always writes ``_now()``). Returns the inserted row id."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with store._lock:
        cur = store.conn.execute(
            "INSERT INTO decisions (timestamp, market_open, signal_count, "
            "action_taken, reasoning, portfolio_value, cash) "
            "VALUES (?,?,?,?,?,?,?)",
            (ts, 0, 0, action_taken, "{}", 1000.0, 100.0),
        )
        store.conn.commit()
        return cur.lastrowid


def test_stale_when_real_decision_older_than_stalled_mult(client):
    """5h old > STALLED_MULT × CLOSED_INTERVAL_S (= 2 × 3600 = 7200s) ✓ and
    also > STALLED_MULT × OPEN_INTERVAL_S (= 3600s), so this is STALE under
    both market states — deterministic regardless of when the suite runs."""
    c, s = client
    _backdate_last(s, hours=5)
    d = c.get("/api/last-real-decision").get_json()
    assert d["verdict"] == "STALE"
    assert d["secs_since"] >= 5 * 3600 - 120
    assert "STALE" in d["headline"]


def test_endpoint_skips_intervening_no_decisions(client):
    """The whole point: a parade of recent NO_DECISIONs do NOT mask the
    stale-but-real decision underneath. The store-side primitive is the
    SSOT; this confirms the endpoint passes it through cleanly."""
    c, s = client
    _backdate_last(s, hours=6, action_taken="HOLD MU → HOLD")
    for _ in range(20):
        s.record_decision(False, 0, "NO_DECISION",
                          "host saturated (skipped claude)",
                          1000.0, 50.0)
    d = c.get("/api/last-real-decision").get_json()
    assert d["verdict"] == "STALE"
    assert d["row"]["action_taken"] == "HOLD MU → HOLD"
    assert d["action_verb"] == "HOLD"


# ─────────────────── envelope shape (always present) ───────────────────


def test_envelope_carries_constants_and_market_state(client):
    """Every response carries the cadence constants + market_open so a
    consumer can locally re-derive the verdict bucket without scraping the
    headline (the analytics/runner_heartbeat constant-echo precedent)."""
    c, s = client
    s.record_decision(True, 3, "BUY NVDA → FILLED", "{}", 1500.0, 200.0)
    d = c.get("/api/last-real-decision").get_json()
    assert d["lagging_mult"] == LAGGING_MULT
    assert d["stalled_mult"] == STALLED_MULT
    assert d["expected_interval_s"] in (OPEN_INTERVAL_S, CLOSED_INTERVAL_S)
    assert isinstance(d["market_open"], bool)


def test_action_with_no_arrow_still_parses_safely(client):
    """A malformed action_taken (no '→') must not crash the endpoint — the
    parser falls back to ``status=None, ticker=None``, verdict still
    classified by age."""
    c, s = client
    s.record_decision(True, 0, "HOLD", "", 1000.0, 100.0)
    d = c.get("/api/last-real-decision").get_json()
    assert d["verdict"] == "FRESH"
    assert d["ticker"] is None       # no arrow → no ticker extracted
    assert d["status"] is None       # no arrow → no status
    assert d["action_verb"] == "HOLD"
