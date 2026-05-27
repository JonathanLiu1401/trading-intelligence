"""Flask wiring smokes for /api/actionable-opportunities.

The pure builder is exhaustively tested in test_actionable_opportunities.py.
This file pins ONLY the IO seam: route registration, SWR envelope reaches a
populated body, the source-availability block reflects what the intern fetch
returned (and degrades when intern is unreachable), and the builder's
verdict + shape pass through verbatim.

The endpoint composes three sub-fetches in-process via ``app.test_client()``
(``/api/scorer-opportunities`` + ``/api/persistent-watchlist-opportunity``)
plus an HTTP cross-fetch to digital-intern's ``/api/ticker-news-burst``. In
the test env we patch the in-process sub-fetches at the route boundary
and the intern fetch via ``urllib.request.urlopen`` — so the endpoint
behaviour is fully deterministic offline.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard  # noqa: E402


@pytest.fixture
def client():
    return dashboard.app.test_client()


def _wait_swr(client, url, max_wait_s: float = 20.0):
    """Hammer the SWR endpoint until the warming envelope resolves to a
    populated body — same shape as the project's other SWR endpoint tests.

    A populated body has a ``verdict`` key. A warming envelope carries
    ``warming: True`` until the async worker fills the cache. Each test
    uses a distinct ``?n=`` query so its SWR cache slot is isolated.
    """
    deadline = time.time() + max_wait_s
    last = None
    while time.time() < deadline:
        r = client.get(url)
        d = r.get_json()
        last = d
        if d and "verdict" in d and "by_ticker" in d:
            return d
        time.sleep(0.4)
    raise AssertionError(f"SWR never populated for {url}; last={last}")


def _fake_urlopen_factory(payload: dict | None):
    """Build a context-manager-shaped object urlopen returns. ``None``
    payload simulates connection failure (raises)."""
    if payload is None:
        def _raise(*a, **k):
            raise OSError("intern unreachable")
        return _raise

    class _FakeResp:
        def __init__(self, body: bytes):
            self._body = body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return self._body

    body = json.dumps(payload).encode("utf-8")
    return lambda *a, **k: _FakeResp(body)


def _patch_sub_endpoints(monkeypatch, scorer_body, persistent_body):
    """Replace the in-process sub-endpoint responses by patching
    ``app.test_client`` at the level the route uses it. The simplest
    approach: monkey-patch the cached scorer-opps + persistent route view
    functions to return canned bodies. Done at the dashboard.app routing
    table level via the @app.test_client() call — but those go through
    the SWR decorator. Easier: patch the *underlying* view functions
    that the SWR decorators wrap, which are registered as endpoints on
    ``dashboard.app``."""
    # The route accesses /api/scorer-opportunities via test_client; the
    # SWR decorator's warming envelope would fight us. Replace the swr-
    # cached entry directly using the decorator's get_cached() / set()
    # APIs if present, else stub the cache.
    #
    # Concrete approach: patch ``dashboard.app.test_client`` to return a
    # tiny shim that resolves the two sub-paths to canned bodies and
    # raises on anything else (forcing us to declare every dependency
    # explicitly).
    real_test_client = dashboard.app.test_client

    class _ShimClient:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def get(self, path):
            class _R:
                def __init__(self, status, body):
                    self.status_code = status
                    self._body = body
                def get_json(self):
                    return self._body
            if path == "/api/scorer-opportunities":
                return _R(200, scorer_body)
            if path == "/api/persistent-watchlist-opportunity":
                return _R(200, persistent_body)
            # Anything else: defer to the real test_client (so /api/actionable
            # itself still works when WE call it via the real client).
            with real_test_client() as c:
                resp = c.get(path)
                return _R(resp.status_code, resp.get_json())

    def _factory(*a, **k):
        # If called from outside (test driver) return the real client; if
        # from inside the route, return the shim. We can distinguish by a
        # call-stack peek but that's fragile — instead, the route uses
        # test_client() as a context manager, so we just always return
        # the shim for in-process consumers. The test driver uses the
        # ``client`` pytest fixture which captured the real test_client
        # at module-load time, before we patch it here.
        return _ShimClient()

    monkeypatch.setattr(dashboard.app, "test_client", _factory)


class TestEndpointWiring:
    def test_route_registered(self, client):
        r = client.get("/api/actionable-opportunities?n=11")
        # 200 even on the cold first warming hit (the SWR decorator
        # returns a warming envelope, not a 500).
        assert r.status_code == 200

    def test_warming_envelope_then_populated(self, client, monkeypatch):
        # No mocks — the route will sub-fetch real scorer/persistent
        # endpoints (probably untrained/empty in tests), then attempt
        # intern fetch (which will fail in test env). The builder
        # tolerates all of this; final verdict is INSUFFICIENT_DATA
        # because the test scorer is not trained.
        d = _wait_swr(client, "/api/actionable-opportunities?n=12")
        assert d["verdict"] in (
            "INSUFFICIENT_DATA", "ALL_QUIET", "SCORER_BUT_NO_NEWS",
            "NEWS_BUT_NO_SCORER", "NEWS_CONFIRMED",
            "PERSISTENT_FOLLOWUP", "HIGH_CONVICTION_FOUND",
        )
        assert "by_ticker" in d
        assert "sources" in d
        for src_key in ("scorer_ok", "news_burst_ok", "persistent_ok"):
            assert src_key in d["sources"]

    def test_high_conviction_via_patched_sources(self, client, monkeypatch):
        """When the scorer is trained and the news burst confirms a strong
        pick, the verdict must be HIGH_CONVICTION_FOUND and the headline
        must name the top ticker."""
        scorer_body = {
            "is_trained": True, "n_train": 3914, "gate_threshold": 500,
            "n_candidates": 1, "opportunities": [
                {"ticker": "AMD", "pred_5d_return_pct": 26.1,
                 "verdict": "STRONG_HOLD"},
            ],
        }
        persistent_body = {"state": "NO_PERSISTENT", "opportunities": []}
        intern_body = {
            "verdict": "HOT",
            "by_ticker": [{
                "ticker": "AMD", "verdict": "HOT", "spike": 6.0,
                "count_window": 5, "count_baseline": 1,
            }],
        }
        _patch_sub_endpoints(monkeypatch, scorer_body, persistent_body)
        monkeypatch.setattr(
            "urllib.request.urlopen", _fake_urlopen_factory(intern_body),
        )
        d = _wait_swr(client, "/api/actionable-opportunities?n=13")
        assert d["verdict"] == "HIGH_CONVICTION_FOUND"
        assert "AMD" in d["headline"]
        assert d["sources"]["scorer_ok"] is True
        assert d["sources"]["news_burst_ok"] is True

    def test_intern_unreachable_collapses_to_scorer_only(
            self, client, monkeypatch):
        """When the intern fetch fails (timeout / connection refused), the
        news axis collapses to COLD and a STRONG_HOLD pick falls into
        SCORER_BUT_NO_NEWS. The ``sources.news_burst_ok`` flag must
        correctly report False."""
        scorer_body = {
            "is_trained": True, "n_train": 3914, "gate_threshold": 500,
            "n_candidates": 1, "opportunities": [
                {"ticker": "AMD", "pred_5d_return_pct": 26.1,
                 "verdict": "STRONG_HOLD"},
            ],
        }
        persistent_body = {"state": "NO_PERSISTENT", "opportunities": []}
        _patch_sub_endpoints(monkeypatch, scorer_body, persistent_body)
        # Simulate intern unreachable.
        monkeypatch.setattr(
            "urllib.request.urlopen", _fake_urlopen_factory(None),
        )
        d = _wait_swr(client, "/api/actionable-opportunities?n=14")
        assert d["verdict"] == "SCORER_BUT_NO_NEWS"
        amd = next(r for r in d["by_ticker"] if r["ticker"] == "AMD")
        assert amd["news_burst_verdict"] == "COLD"
        assert d["sources"]["news_burst_ok"] is False

    def test_scorer_unqualified_yields_insufficient(
            self, client, monkeypatch):
        scorer_body = {
            "is_trained": False, "n_train": 0, "gate_threshold": 500,
            "opportunities": [],
        }
        _patch_sub_endpoints(monkeypatch, scorer_body, {"opportunities": []})
        monkeypatch.setattr(
            "urllib.request.urlopen", _fake_urlopen_factory(None),
        )
        d = _wait_swr(client, "/api/actionable-opportunities?n=15")
        assert d["verdict"] == "INSUFFICIENT_DATA"
        assert d["by_ticker"] == []

    def test_top_n_query_param_clamped(self, client):
        """?n=999 must clamp to 50; ?n=-1 must clamp to 1. Route-level
        guard — the builder's own top_n cap is the SSOT."""
        d1 = _wait_swr(client, "/api/actionable-opportunities?n=999")
        assert len(d1["by_ticker"]) <= 50
        d2 = _wait_swr(client, "/api/actionable-opportunities?n=-1")
        # n=-1 clamps to max(1, ...) = 1; with no scorer data the list is
        # empty anyway, so just confirm we don't crash and never exceed 1.
        assert len(d2["by_ticker"]) <= 1
