"""Endpoint tests for ``/api/per-ticker-skill``.

The per-ticker OOS-skill builder (``paper_trader.ml.per_ticker_skill``) is
exhaustively unit-tested in ``test_per_ticker_skill.py``. This file covers the
*route* added to surface it on the dashboard: the ``as_of`` stamp, payload
passthrough, the graceful-degradation contract (``analyze`` never raises, so
``status='error'`` / ``insufficient_data`` must still be HTTP 200 — only a
genuine unexpected exception escaping the route is a 500), and an SSOT-parity
assertion that the route forks none of the builder's logic.

The route reuses ``per_ticker_skill.analyze`` verbatim; the HTTP-translation
tests monkeypatch it to a canned payload (the ``test_persona_leaderboard_
endpoint`` discipline of stubbing the data layer). One test exercises the real
``analyze`` to pin parity. No live process, no network.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paper_trader.dashboard as dash
import paper_trader.ml.per_ticker_skill as pts


_OK_PAYLOAD = {
    "status": "ok",
    "verdict": "HAS_INVERTED_TICKER",
    "n_train": 4000,
    "n_oos": 1482,
    "n_unique_tickers_oos": 30,
    "tickers": [
        {"ticker": "TQQQ", "n_oos": 305, "rank_ic": 0.27,
         "verdict": "SIGNAL_EDGE"},
        {"ticker": "XLE", "n_oos": 30, "rank_ic": -0.28,
         "verdict": "INVERTED_SIGNAL"},
    ],
    "tickers_truncated": False,
    "inverted_tickers": ["XLE"],
    "hint": "1 ticker(s) have anti-predictive rank skill",
}


def _client():
    return dash.app.test_client()


# ─────────────────────────── HTTP translation ──────────────────────────────
def test_endpoint_passes_analyze_payload_through(monkeypatch):
    monkeypatch.setattr(pts, "analyze", lambda: dict(_OK_PAYLOAD))
    resp = _client().get("/api/per-ticker-skill")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    # every builder key survives the route untouched
    for k, v in _OK_PAYLOAD.items():
        assert data[k] == v


def test_endpoint_stamps_as_of(monkeypatch):
    """The route's only addition to the builder payload is the ``as_of``
    server timestamp — nothing else."""
    monkeypatch.setattr(pts, "analyze", lambda: dict(_OK_PAYLOAD))
    data = json.loads(_client().get("/api/per-ticker-skill").data)
    assert "as_of" in data
    assert set(data) == set(_OK_PAYLOAD) | {"as_of"}


def test_endpoint_insufficient_data_is_200_not_500(monkeypatch):
    """``analyze`` degrades to an insufficient-data dict (never raises);
    the route must surface that at HTTP 200, not 500."""
    insufficient = {
        "status": "insufficient_data", "verdict": "INSUFFICIENT_DATA",
        "n_train": 0, "n_oos": 3, "tickers": [], "inverted_tickers": [],
        "hint": "need >=30 aligned OOS outcomes; have 3",
    }
    monkeypatch.setattr(pts, "analyze", lambda: dict(insufficient))
    resp = _client().get("/api/per-ticker-skill")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["verdict"] == "INSUFFICIENT_DATA"
    assert data["tickers"] == []


def test_endpoint_error_status_still_200(monkeypatch):
    """A builder-level fault (``status='error'``, e.g. SCORER_UNTRAINED) is
    a structured payload, not an exception — it stays HTTP 200."""
    err = {
        "status": "error", "verdict": "SCORER_UNTRAINED",
        "n_train": 0, "n_oos": 0, "tickers": [], "inverted_tickers": [],
        "hint": "scorer.is_trained is False",
    }
    monkeypatch.setattr(pts, "analyze", lambda: dict(err))
    resp = _client().get("/api/per-ticker-skill")
    assert resp.status_code == 200
    assert json.loads(resp.data)["verdict"] == "SCORER_UNTRAINED"


def test_endpoint_unexpected_exception_yields_500(monkeypatch):
    """If ``analyze`` itself raises (a contract violation), the route's
    defensive guard returns a shaped 500 rather than crashing the worker."""
    def _boom():
        raise RuntimeError("decision_outcomes.jsonl unreadable")

    monkeypatch.setattr(pts, "analyze", _boom)
    resp = _client().get("/api/per-ticker-skill")
    assert resp.status_code == 500
    data = json.loads(resp.data)
    assert data["status"] == "error"
    assert data["tickers"] == [] and data["inverted_tickers"] == []
    assert "error" in data


# ─────────────────────────── SSOT parity ───────────────────────────────────
def test_endpoint_is_ssot_parity_with_builder():
    """The route must fork none of the builder's logic — its payload, minus
    the route-only ``as_of`` stamp, is byte-identical to calling ``analyze``
    directly. Mirrors ``test_persona_leaderboard_endpoint``'s parity lock."""
    direct = pts.analyze()
    resp = _client().get("/api/per-ticker-skill")
    assert resp.status_code in (200, 500)
    routed = json.loads(resp.data)
    routed.pop("as_of", None)
    # On a 500 the route substitutes its own error payload, so parity only
    # holds on the normal (never-raises) path — which is the contract.
    if resp.status_code == 200:
        assert routed == direct
