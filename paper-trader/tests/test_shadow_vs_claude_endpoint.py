"""End-to-end Flask-client tests for /api/shadow-vs-claude.

Convention mirrors tests/test_baseline_compare_endpoint.py — real Flask app,
real builder, deterministic offline data; no :8090 bind, no live DB.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paper_trader.dashboard as d


NOW = datetime(2026, 5, 19, 2, 30, tzinfo=timezone.utc)


def _fake_suggestions_payload(suggestions):
    """Build a /api/suggestions-shaped JSON payload."""
    return {
        "as_of": NOW.isoformat(timespec="seconds"),
        "n_candidates": len(suggestions),
        "n_signals_used": 30,
        "action_counts": {},
        "suggestions": suggestions,
    }


def _patch_suggestions(monkeypatch, suggestions):
    """Replace suggestions_api with a stub that returns a controllable JSON."""
    from flask import jsonify

    def _stub():
        # Build the jsonify response *inside* an app context (Flask quirk:
        # jsonify needs to know which app to attach to). The dashboard's app
        # is the one being tested.
        with d.app.app_context():
            return jsonify(_fake_suggestions_payload(suggestions))

    monkeypatch.setattr(d, "suggestions_api", _stub)


def _patch_recent_decisions(monkeypatch, decisions):
    fake_store = MagicMock()
    fake_store.recent_decisions = MagicMock(return_value=decisions)
    monkeypatch.setattr(d, "get_store", lambda: fake_store)


@pytest.fixture
def client(monkeypatch):
    d.app.config["TESTING"] = True
    with d.app.test_client() as c:
        yield c, monkeypatch


def test_route_returns_verdict_shape(client):
    c, mp = client
    _patch_suggestions(mp, [
        {"action": "BUY", "ticker": "MU", "conviction": 0.84,
         "reasons": ["URGENT news", "MACD bullish"],
         "rsi": 59.6, "macd": "bullish",
         "news_urgent": True, "news_max_score": 8.0,
         "top_headline": "BofA buy on MU"},
    ])
    _patch_recent_decisions(mp, [{
        "action_taken": "NO_DECISION",
        "timestamp": (NOW - timedelta(minutes=5)).isoformat(),
        "confidence": None,
        "reasoning": "claude returned no response (timeout/empty)",
    }])
    r = c.get("/api/shadow-vs-claude")
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    # Contract: the dashboard card reads each of these keys.
    for k in ("as_of", "shadow", "claude", "aligned", "verdict", "headline"):
        assert k in j, f"missing key {k!r}"


def test_missed_opportunity_verdict_on_no_decision_with_strong_shadow(client):
    """The headline operational case this endpoint exists for: live shape on
    2026-05-19 — NO_DECISION storm while shadow has MU BUY at 0.84."""
    c, mp = client
    _patch_suggestions(mp, [{
        "action": "BUY", "ticker": "MU", "conviction": 0.84,
        "reasons": ["URGENT news"],
        "rsi": 59.6, "macd": "bullish",
        "news_urgent": True, "news_max_score": 8.0,
        "top_headline": "headline",
    }])
    _patch_recent_decisions(mp, [{
        "action_taken": "NO_DECISION",
        "timestamp": (NOW - timedelta(minutes=5)).isoformat(),
        "confidence": None,
        "reasoning": "claude returned no response (timeout/empty)",
    }])
    j = c.get("/api/shadow-vs-claude").get_json()
    assert j["verdict"] == "MISSED_OPPORTUNITY"
    assert j["shadow"]["ticker"] == "MU"
    assert j["shadow"]["strong"] is True
    assert j["claude"]["action"] == "NO_DECISION"


def test_aligned_verdict_when_claude_buy_matches_shadow_add(client):
    c, mp = client
    _patch_suggestions(mp, [{
        "action": "ADD", "ticker": "NVDA", "conviction": 0.8,
        "reasons": ["MACD bullish"],
        "rsi": 60.0, "macd": "bullish",
        "news_urgent": False, "news_max_score": 6.0,
        "top_headline": "h",
    }])
    _patch_recent_decisions(mp, [{
        "action_taken": "BUY NVDA → FILLED",
        "timestamp": (NOW - timedelta(minutes=10)).isoformat(),
        "confidence": 0.7,
    }])
    j = c.get("/api/shadow-vs-claude").get_json()
    assert j["verdict"] == "ALIGNED"
    assert j["aligned"] is True


def test_no_claude_data_when_decisions_empty(client):
    c, mp = client
    _patch_suggestions(mp, [{
        "action": "BUY", "ticker": "MU", "conviction": 0.8,
        "reasons": [], "rsi": 60, "macd": "bullish",
        "news_urgent": True, "news_max_score": 8.0,
        "top_headline": "h",
    }])
    _patch_recent_decisions(mp, [])
    j = c.get("/api/shadow-vs-claude").get_json()
    assert j["verdict"] == "NO_CLAUDE_DATA"


def test_suggestions_error_is_surfaced_not_swallowed(client):
    """When /api/suggestions returns an error, the endpoint passes it
    through under suggestions_error so the operator can diagnose, instead of
    silently degrading to NO_SHADOW_DATA."""
    from flask import jsonify
    c, mp = client

    def _err_stub():
        with d.app.app_context():
            return jsonify({"error": "signals unavailable: db locked",
                            "suggestions": []})
    mp.setattr(d, "suggestions_api", _err_stub)
    _patch_recent_decisions(mp, [{
        "action_taken": "NO_DECISION",
        "timestamp": (NOW - timedelta(minutes=2)).isoformat(),
        "confidence": None,
    }])
    j = c.get("/api/shadow-vs-claude").get_json()
    assert "suggestions_error" in j
    assert "db locked" in j["suggestions_error"]


def test_cors_header_present_for_cross_fetch(client):
    c, mp = client
    _patch_suggestions(mp, [])
    _patch_recent_decisions(mp, [])
    r = c.get("/api/shadow-vs-claude")
    assert r.headers.get("Access-Control-Allow-Origin") == "*"


def test_endpoint_degrades_to_error_body_on_internal_raise(monkeypatch):
    """Any builder-level raise must surface as an error JSON, not a 500
    HTML stack (the panel reads a JSON body)."""
    d.app.config["TESTING"] = True

    def _boom():
        raise RuntimeError("boom")
    monkeypatch.setattr(d, "suggestions_api", _boom)
    with d.app.test_client() as c:
        r = c.get("/api/shadow-vs-claude")
        assert r.status_code == 500
        j = r.get_json()
        assert "error" in j
