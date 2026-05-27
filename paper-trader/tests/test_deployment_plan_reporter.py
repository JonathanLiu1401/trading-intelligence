"""Reporter wiring for the deployment-plan verdict.

The ``_deployment_plan_line`` helper closes a real dashboard→Discord
gap: when the book holds meaningful idle cash AND the scorer slate has
a buy-eligible plan, the operator-facing summary historically said
nothing about WHAT to do with that cash. This suite locks:

  * Suppression contract (silence-when-not-actionable) at every level
    of the cash-pct / verdict / n_plan ladder.
  * SSOT contract — the endpoint's headline ships verbatim
    (AGENTS.md invariant #10 so this Discord line and
    /api/deployment-plan can never disagree).
  * Wiring — the helper actually fires from BOTH ``send_hourly_summary``
    and ``send_daily_close``.
"""
from __future__ import annotations

import io
import json
from unittest.mock import patch, MagicMock

from paper_trader import reporter


class _FakeStore:
    """Minimal store stub — only ``get_portfolio`` is consulted by the
    helper, but the surrounding summary path calls many other reads.
    Return safe defaults so test_wired_into_* can run end-to-end."""

    def __init__(self, cash=1000.0, total_value=1000.0):
        self._pf = {
            "cash": cash,
            "total_value": total_value,
            "positions": [],
            "last_updated": "2026-05-27T12:00:00+00:00",
        }

    def get_portfolio(self):
        return dict(self._pf)


def _mock_urlopen(payload, status: int = 200):
    """Build a context-manager mock matching urllib.request.urlopen."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    return cm


# ── Cash-pct floor suppression ───────────────────────────────────


def test_silent_when_cash_pct_below_floor():
    """Book is mostly deployed (cash 30% of book) — the deployment plan
    is not the right Discord nudge; trim/rotate is. Helper must NOT
    fire even if the endpoint would return a READY plan."""
    store = _FakeStore(cash=300.0, total_value=1000.0)
    # Even if urlopen WOULD have returned a READY plan, we must not
    # reach it — patch it so a wrong fire would surface in the test
    # as a non-empty headline string.
    with patch("urllib.request.urlopen",
               return_value=_mock_urlopen({"verdict": "READY", "n_plan": 4,
                                            "headline": "should not appear"})):
        out = reporter._deployment_plan_line(store)
    assert out == "", \
        "cash 30% of book must suppress — deployment plan is not the right nudge"


def test_fires_at_cash_pct_floor_boundary():
    """Cash exactly at the 50% floor — the helper fires. The floor is a
    minimum, not an exclusive bound."""
    store = _FakeStore(cash=500.0, total_value=1000.0)
    payload = {
        "verdict": "READY",
        "n_plan": 2,
        "headline": "Deploy $400 across 2 name(s) (40% of cash); blended pred 5d +5.00%.",
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        out = reporter._deployment_plan_line(store)
    assert "DEPLOY PLAN" in out, "cash exactly at floor must fire"


def test_silent_when_total_value_zero():
    """A boot-edge / corrupt-read book with total_value=0 must not
    crash; helper silently degrades."""
    store = _FakeStore(cash=1000.0, total_value=0.0)
    out = reporter._deployment_plan_line(store)
    assert out == ""


def test_silent_when_total_value_negative():
    """Defensive — non-physical total_value (numeric corruption) must
    not cross the floor predicate."""
    store = _FakeStore(cash=1000.0, total_value=-500.0)
    out = reporter._deployment_plan_line(store)
    assert out == ""


# ── Verdict-ladder suppression ───────────────────────────────────


def test_silent_on_no_opportunities_verdict():
    """The endpoint surfaces NO_OPPORTUNITIES when the scorer slate
    has no buy-eligible candidates — silent in Discord."""
    store = _FakeStore(cash=1000.0, total_value=1000.0)
    payload = {"verdict": "NO_OPPORTUNITIES", "n_plan": 0, "headline": "x"}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert reporter._deployment_plan_line(store) == ""


def test_silent_on_insufficient_cash_verdict():
    """The endpoint surfaces INSUFFICIENT_CASH when the deployable
    cash falls below min_alloc — silent (the line can't say anything
    actionable when the gate already declined to size a trade)."""
    store = _FakeStore(cash=1000.0, total_value=1000.0)
    payload = {"verdict": "INSUFFICIENT_CASH", "n_plan": 0, "headline": "x"}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert reporter._deployment_plan_line(store) == ""


def test_silent_on_pending_verdict():
    """PENDING is the transient pre-pipeline state — silent. The plan
    has not finished computing; surfacing it would be misleading."""
    store = _FakeStore(cash=1000.0, total_value=1000.0)
    payload = {"verdict": "PENDING", "n_plan": 0, "headline": "still computing"}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert reporter._deployment_plan_line(store) == ""


def test_silent_on_error_verdict():
    """The endpoint surfaces ERROR when the builder raised — silent.
    A summary line saying 'ERROR — something broke' adds noise without
    a path to action."""
    store = _FakeStore(cash=1000.0, total_value=1000.0)
    payload = {"verdict": "ERROR", "n_plan": 0, "headline": "ERROR — x"}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert reporter._deployment_plan_line(store) == ""


# ── Fires on READY / GATED ───────────────────────────────────────


def test_fires_on_ready_verdict():
    """READY with a real plan must fire. The block header carries the
    READY label and the builder's headline verbatim."""
    store = _FakeStore(cash=1167.71, total_value=1167.71)
    payload = {
        "verdict": "READY",
        "n_plan": 4,
        "headline": "Deploy $817 across 4 name(s) (70% of cash); blended pred 5d +7.13%.",
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        out = reporter._deployment_plan_line(store)
    assert "DEPLOY PLAN" in out
    assert "READY" in out
    assert "Deploy $817" in out
    assert "+7.13%" in out


def test_fires_on_gated_verdict():
    """GATED (some candidates blocked by caps but plan still has rows)
    must fire — the operator still wants to see what survived."""
    store = _FakeStore(cash=1167.71, total_value=1167.71)
    payload = {
        "verdict": "GATED",
        "n_plan": 2,
        "headline": "Deploy $300 across 2 name(s) (26% of cash); blended pred 5d +4.20%.",
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        out = reporter._deployment_plan_line(store)
    assert "DEPLOY PLAN" in out
    assert "GATED" in out


def test_silent_on_empty_plan():
    """Defensive: even if verdict is READY/GATED, an empty plan list
    means there's nothing to deploy. Helper suppresses rather than
    shipping a fake-deploy line."""
    store = _FakeStore(cash=1000.0, total_value=1000.0)
    payload = {"verdict": "GATED", "n_plan": 0, "headline": "Every candidate blocked."}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert reporter._deployment_plan_line(store) == ""


def test_silent_on_empty_headline():
    """If the endpoint returns a fireable verdict with an empty headline,
    the helper must NOT emit a bare DEPLOY PLAN block with no body."""
    store = _FakeStore(cash=1000.0, total_value=1000.0)
    payload = {"verdict": "READY", "n_plan": 3, "headline": ""}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert reporter._deployment_plan_line(store) == ""


def test_headline_shipped_verbatim_ssot():
    """Invariant #10 — the headline is the endpoint's own, never
    re-derived in the reporter. Patch the endpoint to return a
    sentinel headline and verify it appears unchanged."""
    store = _FakeStore(cash=1000.0, total_value=1000.0)
    payload = {
        "verdict": "READY",
        "n_plan": 1,
        "headline": "SENTINEL_DEPLOY_HEADLINE_DO_NOT_CHANGE",
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        out = reporter._deployment_plan_line(store)
    assert "SENTINEL_DEPLOY_HEADLINE_DO_NOT_CHANGE" in out, \
        "builder/endpoint headline must ship verbatim (invariant #10)"


# ── Failure modes ────────────────────────────────────────────────


def test_silent_on_urlopen_exception():
    """Dashboard down / port not bound / network blip — silent.
    Discord-path discipline: a transient endpoint failure must never
    take down the summary. A diagnostic line is printed to stdout."""
    store = _FakeStore(cash=1000.0, total_value=1000.0)
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError):
        out = reporter._deployment_plan_line(store)
    assert out == ""


def test_silent_on_non_200_status():
    """The endpoint returns 500 on builder fault — silent. Helper does
    not parse a 500 body even if shaped like a result dict."""
    store = _FakeStore(cash=1000.0, total_value=1000.0)
    payload = {"verdict": "READY", "n_plan": 1, "headline": "x"}
    with patch("urllib.request.urlopen",
               return_value=_mock_urlopen(payload, status=500)):
        assert reporter._deployment_plan_line(store) == ""


def test_silent_on_malformed_json():
    """A response body that is not valid JSON — silent."""
    store = _FakeStore(cash=1000.0, total_value=1000.0)
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = b"not valid json{{"
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    with patch("urllib.request.urlopen", return_value=cm):
        assert reporter._deployment_plan_line(store) == ""


def test_silent_on_non_dict_response():
    """The endpoint returns a JSON array instead of a dict (regression
    safety) — silent."""
    store = _FakeStore(cash=1000.0, total_value=1000.0)
    with patch("urllib.request.urlopen",
               return_value=_mock_urlopen(["unexpected", "shape"])):
        assert reporter._deployment_plan_line(store) == ""


def test_helper_never_raises_on_store_failure():
    """A store fault must not propagate as an exception — Discord
    summary protection."""
    store = MagicMock()
    store.get_portfolio.side_effect = RuntimeError("simulated store crash")
    out = reporter._deployment_plan_line(store)
    assert out == ""


# ── Wiring into hourly + daily close ─────────────────────────────


def test_hourly_summary_invokes_deployment_plan_helper():
    """Regression guard: the helper must actually be called by
    ``send_hourly_summary``. If a future refactor accidentally drops
    the call, this test fails."""
    called = {"n": 0}

    def fake_deploy(_store):
        called["n"] += 1
        return ""

    with patch.object(reporter, "_deployment_plan_line",
                      side_effect=fake_deploy), \
         patch.object(reporter, "_send", return_value=True):
        try:
            reporter.send_hourly_summary()
        except Exception:
            pass
    assert called["n"] >= 1, \
        "send_hourly_summary must call _deployment_plan_line"


def test_daily_close_invokes_deployment_plan_helper():
    """Same regression guard for daily close — both surfaces must
    surface the deploy plan."""
    called = {"n": 0}

    def fake_deploy(_store):
        called["n"] += 1
        return ""

    with patch.object(reporter, "_deployment_plan_line",
                      side_effect=fake_deploy), \
         patch.object(reporter, "_send", return_value=True):
        try:
            reporter.send_daily_close()
        except Exception:
            pass
    assert called["n"] >= 1, \
        "send_daily_close must call _deployment_plan_line"


def test_silent_helper_leaves_no_header_in_summary():
    """When the helper returns "", the hourly body must NOT contain
    a stray "DEPLOY PLAN" header. End-to-end suppression contract."""
    captured = {"body": None}

    def capture_send(message):
        captured["body"] = message
        return True

    with patch.object(reporter, "_deployment_plan_line", return_value=""), \
         patch.object(reporter, "_send", side_effect=capture_send):
        try:
            reporter.send_hourly_summary()
        except Exception:
            pass
    body = captured["body"] or ""
    assert "DEPLOY PLAN" not in body, \
        "silent helper must not leave a DEPLOY PLAN header in the summary"
