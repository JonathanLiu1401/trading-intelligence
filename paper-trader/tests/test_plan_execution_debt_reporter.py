"""Reporter wiring for the plan-execution-debt verdict.

The ``_plan_execution_debt_line`` helper closes a real dashboard→Discord
gap: ``/api/plan-execution-debt`` shows on the dashboard that the gate's
buy list is being IGNORED (DISCONNECTED / DRIFTING / TIGHTENING), but
the operator on Discord historically only saw the IDLE_STORM cause-code
and never the alpha-actionable "your gate's plan is being ignored"
signal.

Live evidence (the gap that motivated this surface): 2026-05-30 session
— planner wants $188 into MSFU at STRONG_HOLD (+7.35% pred), engine
wedged since boot, scorer-recommended capital sitting unfilled. The
``/api/plan-execution-debt`` verdict is DISCONNECTED, but the operator
on Discord only sees IDLE_STORM.

Suite locks:
  * Suppression contract (silence-when-not-actionable) at every level
    of the verdict / unexec_$ ladder.
  * SSOT contract — the endpoint's headline ships verbatim
    (AGENTS.md invariant #10 so this Discord line and
    ``/api/plan-execution-debt`` can never disagree).
  * Wiring — the helper actually fires from BOTH
    ``send_hourly_summary`` and ``send_daily_close``.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from paper_trader import reporter


def _mock_urlopen(payload, status: int = 200):
    """Context-manager mock matching urllib.request.urlopen."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    return cm


# ── Verdict-ladder suppression — silent for non-actionable verdicts ───


def test_silent_on_aligned_verdict():
    """ALIGNED — plan is being honored ≥80%. The summary must never
    become its own lying green light; an "ALIGNED — plan is honored"
    line would train the operator to ignore the surface entirely."""
    payload = {
        "verdict": "ALIGNED",
        "unexecuted_usd": 10.0,
        "headline": "ALIGNED — 85% of recommended $400 deployed.",
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert reporter._plan_execution_debt_line() == ""


def test_silent_on_no_plan_verdict():
    """NO_PLAN — there is nothing to track. Silent."""
    payload = {"verdict": "NO_PLAN", "unexecuted_usd": 0.0,
               "headline": "no plan rows — nothing to track for execution"}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert reporter._plan_execution_debt_line() == ""


def test_silent_on_error_verdict():
    """ERROR — dashboard endpoint already surfaces the diagnostic;
    don't pollute the summary with a recurring 'ERROR' alarm."""
    payload = {"verdict": "ERROR", "unexecuted_usd": 0.0,
               "headline": "ERROR — builder raised"}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert reporter._plan_execution_debt_line() == ""


def test_silent_on_unknown_verdict():
    """Defensive — an unrecognised verdict (e.g. a future verdict the
    builder adds) defaults to silent, never an opaque line shipping
    a verdict word the operator has no context for."""
    payload = {"verdict": "FUTURE_VERDICT", "unexecuted_usd": 500.0,
               "headline": "x"}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert reporter._plan_execution_debt_line() == ""


# ── Dollar-floor suppression ──────────────────────────────────────────


def test_silent_on_disconnected_below_dollar_floor():
    """A $5 unexecuted plan is technically DISCONNECTED but not worth
    the Discord chatter. The Discord-side actionability gate kicks in
    below ``_PLAN_EXECUTION_DEBT_MIN_USD`` ($50)."""
    payload = {
        "verdict": "DISCONNECTED",
        "unexecuted_usd": 5.0,
        "headline": "DISCONNECTED — 0% of recommended $5 deployed.",
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert reporter._plan_execution_debt_line() == ""


def test_fires_at_dollar_floor_boundary():
    """A $50 gap (== floor) must fire — the floor is inclusive."""
    payload = {
        "verdict": "DRIFTING",
        "unexecuted_usd": 50.0,
        "headline": "DRIFTING — 30% of recommended $71 deployed.",
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        out = reporter._plan_execution_debt_line()
    assert "PLAN DEBT" in out
    assert "DRIFTING" in out


# ── Fires on actionable verdicts with sufficient $-gap ────────────────


def test_fires_on_disconnected():
    """DISCONNECTED with a meaningful $-gap is the loudest alarm.
    The block carries the verdict + builder headline verbatim."""
    payload = {
        "verdict": "DISCONNECTED",
        "unexecuted_usd": 188.46,
        "headline": (
            "DISCONNECTED — 0% of recommended $188 deployed in last 24h "
            "(0/1 fully filled). Largest miss: MSFU "
            "($188 of $188 unexecuted)."
        ),
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        out = reporter._plan_execution_debt_line()
    assert "PLAN DEBT" in out
    assert "DISCONNECTED" in out
    assert "MSFU" in out                       # worst-gap ticker propagates
    assert "0% of recommended" in out


def test_fires_on_drifting():
    """DRIFTING (20-50% executed) fires — middle alarm tier."""
    payload = {
        "verdict": "DRIFTING",
        "unexecuted_usd": 200.0,
        "headline": "DRIFTING — 30% of recommended $286 deployed.",
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        out = reporter._plan_execution_debt_line()
    assert "PLAN DEBT" in out
    assert "DRIFTING" in out


def test_fires_on_tightening():
    """TIGHTENING (50-80% executed) still fires — partial is still
    actionable: the trader can see what's left."""
    payload = {
        "verdict": "TIGHTENING",
        "unexecuted_usd": 100.0,
        "headline": "TIGHTENING — 60% of recommended $250 deployed.",
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        out = reporter._plan_execution_debt_line()
    assert "PLAN DEBT" in out
    assert "TIGHTENING" in out


# ── Verdict-icon ladder (most-alarming first) ─────────────────────────


def test_disconnected_uses_loudest_icon():
    """DISCONNECTED is the most alarming — uses ⚠️ (matches the
    ``_realized_vs_unrealized_line`` LEAKING_PAPER precedent)."""
    payload = {
        "verdict": "DISCONNECTED",
        "unexecuted_usd": 500.0,
        "headline": "DISCONNECTED",
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        out = reporter._plan_execution_debt_line()
    assert out.startswith("⚠️")


def test_drifting_uses_middle_icon():
    payload = {
        "verdict": "DRIFTING",
        "unexecuted_usd": 200.0,
        "headline": "DRIFTING",
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        out = reporter._plan_execution_debt_line()
    assert out.startswith("🔶")


def test_tightening_uses_lightest_icon():
    payload = {
        "verdict": "TIGHTENING",
        "unexecuted_usd": 100.0,
        "headline": "TIGHTENING",
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        out = reporter._plan_execution_debt_line()
    assert out.startswith("🟡")


# ── SSOT — headline shipped verbatim ──────────────────────────────────


def test_headline_shipped_verbatim_ssot():
    """Invariant #10 — the headline is the endpoint's own, never
    re-derived in the reporter. A sentinel string must come through
    unchanged so a future builder refactor stays in lockstep with
    the Discord line."""
    payload = {
        "verdict": "DISCONNECTED",
        "unexecuted_usd": 500.0,
        "headline": "SENTINEL_PLAN_DEBT_HEADLINE_DO_NOT_CHANGE",
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        out = reporter._plan_execution_debt_line()
    assert "SENTINEL_PLAN_DEBT_HEADLINE_DO_NOT_CHANGE" in out


# ── Failure modes ────────────────────────────────────────────────────


def test_silent_on_urlopen_exception():
    """Dashboard down / port not bound / network blip — silent.
    Discord-path discipline: a transient endpoint failure must never
    take down the summary."""
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError):
        assert reporter._plan_execution_debt_line() == ""


def test_silent_on_non_200_status():
    """The endpoint returns 500 on builder fault — silent."""
    payload = {"verdict": "DISCONNECTED", "unexecuted_usd": 200.0,
               "headline": "x"}
    with patch("urllib.request.urlopen",
               return_value=_mock_urlopen(payload, status=500)):
        assert reporter._plan_execution_debt_line() == ""


def test_silent_on_malformed_json():
    """A response body that is not valid JSON — silent."""
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = b"not valid json{{"
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    with patch("urllib.request.urlopen", return_value=cm):
        assert reporter._plan_execution_debt_line() == ""


def test_silent_on_non_dict_response():
    """The endpoint returns a JSON array instead of a dict — silent."""
    with patch("urllib.request.urlopen",
               return_value=_mock_urlopen(["unexpected", "shape"])):
        assert reporter._plan_execution_debt_line() == ""


def test_silent_on_empty_headline():
    """If the endpoint returns a fireable verdict with an empty
    headline, the helper must NOT emit a bare PLAN DEBT block with
    no body — same discipline as ``_deployment_plan_line``."""
    payload = {"verdict": "DISCONNECTED", "unexecuted_usd": 500.0,
               "headline": ""}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert reporter._plan_execution_debt_line() == ""


def test_silent_on_non_numeric_unexecuted_usd():
    """Defensive — a non-numeric unexecuted_usd (corrupt JSON / future
    schema drift) must not crash the floor check; silent."""
    payload = {"verdict": "DISCONNECTED", "unexecuted_usd": "not a number",
               "headline": "x"}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        assert reporter._plan_execution_debt_line() == ""


def test_helper_never_raises():
    """Any internal exception path must collapse to ``""`` — Discord
    summary protection (the helper sits inside the hot path)."""
    with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
        out = reporter._plan_execution_debt_line()
    assert out == ""


# ── Wiring into hourly + daily close ─────────────────────────────────


def test_hourly_summary_invokes_plan_execution_debt_helper():
    """Regression guard: the helper must actually be called by
    ``send_hourly_summary``. If a future refactor accidentally drops
    the call, this test fails."""
    called = {"n": 0}

    def fake_helper():
        called["n"] += 1
        return ""

    with patch.object(reporter, "_plan_execution_debt_line",
                      side_effect=fake_helper), \
         patch.object(reporter, "_send", return_value=True):
        try:
            reporter.send_hourly_summary()
        except Exception:
            pass
    assert called["n"] >= 1, \
        "send_hourly_summary must call _plan_execution_debt_line"


def test_daily_close_invokes_plan_execution_debt_helper():
    """Same regression guard for daily close — both surfaces must
    surface the plan-execution debt."""
    called = {"n": 0}

    def fake_helper():
        called["n"] += 1
        return ""

    with patch.object(reporter, "_plan_execution_debt_line",
                      side_effect=fake_helper), \
         patch.object(reporter, "_send", return_value=True):
        try:
            reporter.send_daily_close()
        except Exception:
            pass
    assert called["n"] >= 1, \
        "send_daily_close must call _plan_execution_debt_line"
