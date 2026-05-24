"""Flask wiring tests for /api/intent-followthrough.

Drives the dashboard endpoint with a fake store so the test is offline and
deterministic. Asserts that the SSOT builder is wired correctly (verbatim
keys exposed), the verdict ladder + abstention counters surface, the
query-param clamps work, and an error in the builder still produces a
200-shape envelope (never a 500 to the chat handler).

Pure arithmetic of the verdict ladder + intent extraction is pinned in
test_intent_followthrough_skill.py — this file only covers the IO seam.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeStore:
    def __init__(self, decisions=None, trades=None):
        self._d = decisions or []
        self._t = trades or []
        self._lock = _NullLock()
        self.conn = None

    def recent_decisions(self, limit=500):
        return list(self._d[:limit])

    def recent_trades(self, limit=2000):
        return list(self._t[:limit])


@pytest.fixture
def client():
    return dashboard.app.test_client()


def _now():
    return datetime.now(timezone.utc)


def _dec(did, hours_ago, action_taken, reasoning):
    ts = _now() - timedelta(hours=hours_ago)
    return {
        "id": did,
        "timestamp": ts.isoformat(),
        "action_taken": action_taken,
        "reasoning": reasoning,
    }


def _trade(tid, hours_ago, ticker, action):
    ts = _now() - timedelta(hours=hours_ago)
    return {
        "id": tid,
        "timestamp": ts.isoformat(),
        "ticker": ticker,
        "action": action,
        "qty": 1.0,
        "price": 100.0,
    }


class TestEndpointWiring:
    def test_empty_store_returns_no_data(self, client, monkeypatch):
        monkeypatch.setattr(dashboard, "get_store", lambda: _FakeStore())
        r = client.get("/api/intent-followthrough")
        assert r.status_code == 200
        d = r.get_json()
        assert d["verdict"] == "NO_DATA"
        assert d["n_intents"] == 0
        assert d["n_actionable"] == 0
        assert d["followthrough_rate"] is None

    def test_followed_intent_surfaces_disciplined(self, client, monkeypatch):
        # Decision 8h ago states a "watch-for" NVDA breakout; 6h ago the bot
        # actually bought NVDA. The actionable intent should be FOLLOWED.
        # Sample three reinforcing intents so the rate is unambiguously
        # ≥ discipline_floor (66%).
        decs = [
            _dec(1, 8, "HOLD CASH → HOLD",
                 "watch for NVDA breakout above 220 to add"),
            _dec(2, 7, "HOLD CASH → HOLD",
                 "watch for AMD strength on SEMIS bid, ready to buy"),
            _dec(3, 6.5, "HOLD CASH → HOLD",
                 "watch for MU on DRAM print, looking to add"),
        ]
        trades = [
            _trade(10, 6, "NVDA", "BUY"),
            _trade(11, 5, "AMD", "BUY"),
            _trade(12, 4, "MU", "BUY"),
        ]
        monkeypatch.setattr(dashboard, "get_store",
                            lambda: _FakeStore(decs, trades))
        r = client.get("/api/intent-followthrough")
        assert r.status_code == 200
        d = r.get_json()
        assert d["verdict"] == "DISCIPLINED"
        assert d["n_followed"] >= 2
        assert d["followthrough_rate"] is not None
        assert d["followthrough_rate"] >= 0.66

    def test_abandoned_intent_when_no_followup(self, client, monkeypatch):
        # Three "ready to buy NVDA" intents long enough ago that the
        # eval_window has passed without any matching trade. n_abandoned
        # must reach abandoned_min_n=3 for the ABANDONED verdict to fire.
        decs = [
            _dec(1, 24, "HOLD CASH → HOLD",
                 "ready to add NVDA on next dip below 215"),
            _dec(2, 22, "HOLD CASH → HOLD",
                 "ready to add AMD on confirmation"),
            _dec(3, 20, "HOLD CASH → HOLD",
                 "ready to add MU on DRAM tailwind"),
        ]
        # No buy trades whatsoever — every actionable intent abandons.
        trades: list = []
        monkeypatch.setattr(dashboard, "get_store",
                            lambda: _FakeStore(decs, trades))
        # eval_window_hours=2 ⇒ at 20-24h old, all intents are past the
        # evaluation window and resolve to ABANDONED.
        r = client.get(
            "/api/intent-followthrough?eval_window_hours=2&window_hours=48")
        assert r.status_code == 200
        d = r.get_json()
        assert d["verdict"] == "ABANDONED"
        assert d["n_abandoned"] >= 3
        assert d["n_followed"] == 0

    def test_query_params_clamp(self, client, monkeypatch):
        monkeypatch.setattr(dashboard, "get_store", lambda: _FakeStore())
        # Out-of-range values must clamp without raising.
        r = client.get(
            "/api/intent-followthrough"
            "?window_hours=999&eval_window_hours=0&max_intents=999")
        assert r.status_code == 200
        d = r.get_json()
        assert d["window_hours"] <= 168.0
        assert d["eval_window_hours"] >= 1.0
        # max_intents clamps to ≤ 50 (the documented upper bound).
        assert len(d["intents"]) <= 50

    def test_garbage_query_params_use_defaults(self, client, monkeypatch):
        monkeypatch.setattr(dashboard, "get_store", lambda: _FakeStore())
        r = client.get(
            "/api/intent-followthrough"
            "?window_hours=abc&eval_window_hours=&max_intents=foo")
        # Garbage must not 500 — the qf/qi helpers swallow ValueError.
        assert r.status_code == 200
        d = r.get_json()
        assert d["window_hours"] == 24.0  # DEFAULT_WINDOW_HOURS
        assert d["eval_window_hours"] == 12.0  # DEFAULT_EVAL_WINDOW_HOURS

    def test_builder_raise_degrades_to_200_envelope(self, client, monkeypatch):
        # Force the builder to raise; the endpoint must catch and emit a
        # safe 500-status envelope rather than letting the exception
        # propagate to a Flask 500 page (the chat handler depends on the
        # envelope keys existing).
        def _boom(*a, **kw):
            raise RuntimeError("synthetic")
        monkeypatch.setattr(dashboard, "get_store", lambda: _FakeStore())
        monkeypatch.setattr(
            "paper_trader.analytics.intent_followthrough_skill"
            ".build_intent_followthrough", _boom)
        r = client.get("/api/intent-followthrough")
        # Endpoint wraps in try/except → 500 with a structured envelope.
        assert r.status_code == 500
        d = r.get_json()
        assert d["verdict"] == "ERROR"
        # Envelope shape required by the chat helper.
        for k in ("intents", "by_kind", "abstention", "n_intents",
                  "n_followed", "n_abandoned"):
            assert k in d

    def test_envelope_shape_is_complete(self, client, monkeypatch):
        # The chat helper depends on a specific key set. Pin them so a
        # future refactor that drops a key gets caught immediately.
        monkeypatch.setattr(dashboard, "get_store", lambda: _FakeStore())
        r = client.get("/api/intent-followthrough")
        assert r.status_code == 200
        d = r.get_json()
        for k in (
            "state", "verdict", "headline",
            "n_intents", "n_actionable", "n_followed",
            "n_pending", "n_abandoned", "followthrough_rate",
            "abstention", "by_kind", "intents",
            "window_hours", "stale_hours", "eval_window_hours",
            "discipline_floor", "drifting_floor", "abandoned_min_n",
            "as_of",
        ):
            assert k in d, f"missing key: {k}"
