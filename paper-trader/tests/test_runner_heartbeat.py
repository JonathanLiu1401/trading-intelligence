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
    _no_decision_cause,
    _dominant_cause,
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


def test_endpoint_surfaces_degraded_discord_delivery(client, monkeypatch):
    """Additive (2026-05-18): /api/runner-heartbeat exposes Discord delivery
    health so the silent-channel class of failure (the 2026-05-17 `env node`
    PATH outage — loop alive, operator surface dark) is visible. The pre-
    existing liveness verdict is unchanged."""
    c, s = client
    s.record_decision(True, 3, "BUY NVDA → FILLED", "{}", 1000.0, 100.0)
    from paper_trader import reporter as _reporter
    monkeypatch.setattr(_reporter, "_notify_state", {
        "last_attempt_ts": None, "last_ok_ts": None, "last_result": None,
        "consecutive_failures": 0, "last_error": "",
    })
    for _ in range(3):
        _reporter._record_send_outcome(False, "/usr/bin/env: 'node' not found")
    r = c.get("/api/runner-heartbeat")
    assert r.status_code == 200
    d = r.get_json()
    assert d["verdict"] == "HEALTHY"  # loop liveness unchanged
    assert d["notify"]["verdict"] == "DEGRADED"
    assert d["notify"]["consecutive_failures"] == 3
    assert d["notify"]["restart_recommended"] is True


def test_endpoint_discord_delivery_healthy(client, monkeypatch):
    c, s = client
    s.record_decision(True, 3, "BUY NVDA → FILLED", "{}", 1000.0, 100.0)
    from paper_trader import reporter as _reporter
    monkeypatch.setattr(_reporter, "_notify_state", {
        "last_attempt_ts": None, "last_ok_ts": None, "last_result": None,
        "consecutive_failures": 0, "last_error": "",
    })
    _reporter._record_send_outcome(True)
    r = c.get("/api/runner-heartbeat")
    d = r.get_json()
    assert d["notify"]["verdict"] == "HEALTHY"
    assert d["notify"]["restart_recommended"] is False


# ═════════════ decision-efficacy overlay (additive, 2026-05-18) ═════════════
#
# Why this exists: the bare cadence verdict calls a loop that cycles
# perfectly but emits NO_DECISION every cycle "HEALTHY, restart_recommended:
# false" — the exact live regime (~60% lifetime NO_DECISION, 5-in-a-row
# under host-load storms). A trader reading the heartbeat (its primary use)
# is then actively reassured the engine is fine while it is brain-dead.
# These pin the additive overlay: byte-identical when omitted, IDLE_STORM
# folds restart+headline but NEVER the liveness verdict enum, the
# elevated/producing/no-data bands, the exact storm-threshold boundary, and
# the canonical-predicate drift lock.

from paper_trader.analytics.runner_heartbeat import (  # noqa: E402
    NO_DECISION_STORM_THRESHOLD,
    _is_no_decision,
)

ND = "NO_DECISION"


def test_storm_threshold_constant_is_the_spec():
    # Retune-proof (the OPEN_INTERVAL_S precedent) AND must mirror the
    # runner's auto-recovery breaker so the heartbeat recommends a restart
    # at the same wedge the breaker actually fires on.
    assert NO_DECISION_STORM_THRESHOLD == 5
    from paper_trader import runner as _runner
    assert NO_DECISION_STORM_THRESHOLD == _runner.CONSECUTIVE_NO_DECISION_LIMIT


def test_is_no_decision_mirrors_forensics():
    """Drift lock (single source of truth, invariant #10): the inlined
    predicate must stay byte-equivalent to the canonical
    decision_forensics._is_no_decision across every shape it sees."""
    from paper_trader.analytics.decision_forensics import (
        _is_no_decision as canonical,
    )
    for s in (ND, "NO_DECISION", "", "  ", None, "HOLD MU → HOLD",
              "BUY NVDA → FILLED", "SELL X → BLOCKED", "no_decision",
              " NO_DECISION ", "NO_DECISIONX"):
        assert _is_no_decision(s) == canonical(s), repr(s)


def test_recent_actions_omitted_is_byte_identical():
    """The additive contract: with recent_actions omitted the output is
    exactly what it was before the parameter existed — no decision_efficacy
    key, headline/verdict/restart untouched."""
    base = build_runner_heartbeat(_ago(600), market_open=True, now=NOW)
    assert "decision_efficacy" not in base
    assert base["verdict"] == "HEALTHY"
    assert base["restart_recommended"] is False
    # Passing recent_actions=None is identical to omitting it.
    assert build_runner_heartbeat(
        _ago(600), market_open=True, now=NOW, recent_actions=None) == base


def test_idle_storm_folds_restart_and_headline_not_verdict():
    out = build_runner_heartbeat(
        _ago(600), market_open=True, now=NOW,
        recent_actions=[ND] * 5 + ["BUY NVDA → FILLED"])
    # liveness verdict enum untouched (the separation contract)
    assert out["verdict"] == "HEALTHY"
    # but the trader-facing signals now reflect the brain-dead engine
    assert out["restart_recommended"] is True
    assert "NO_DECISION" in out["headline"] and "restart" in out["headline"]
    eff = out["decision_efficacy"]
    assert eff["verdict"] == "IDLE_STORM"
    assert eff["consecutive_no_decision"] == 5
    assert eff["window"] == 6
    assert eff["no_decision_pct"] == round(5 / 6 * 100.0, 1)


def test_storm_threshold_boundary_off_by_one():
    """threshold-1 consecutive is NOT a storm; exactly threshold IS — locks
    the `>=` against a `>` regression that would mute a real 5-run wedge."""
    k = NO_DECISION_STORM_THRESHOLD
    just_under = build_runner_heartbeat(
        _ago(600), market_open=True, now=NOW,
        recent_actions=[ND] * (k - 1) + ["HOLD MU → HOLD"])
    assert just_under["decision_efficacy"]["verdict"] != "IDLE_STORM"
    assert just_under["restart_recommended"] is False
    exactly = build_runner_heartbeat(
        _ago(600), market_open=True, now=NOW, recent_actions=[ND] * k)
    assert exactly["decision_efficacy"]["verdict"] == "IDLE_STORM"
    assert exactly["restart_recommended"] is True


def test_elevated_but_not_storm_is_degraded_no_restart():
    """Latest cycle produced a decision (consec=0) but the window is mostly
    NO_DECISION → DEGRADED: informational, NO restart, verdict/headline of
    the liveness layer untouched."""
    out = build_runner_heartbeat(
        _ago(600), market_open=True, now=NOW,
        recent_actions=["BUY NVDA → FILLED"] + [ND] * 3)  # 75% ND, consec=0
    assert out["verdict"] == "HEALTHY"
    assert out["restart_recommended"] is False
    assert "NO_DECISION" not in out["headline"]  # liveness headline unchanged
    eff = out["decision_efficacy"]
    assert eff["verdict"] == "DEGRADED"
    assert eff["consecutive_no_decision"] == 0
    assert eff["no_decision_pct"] == 75.0


def test_producing_when_engine_actually_decides():
    out = build_runner_heartbeat(
        _ago(600), market_open=True, now=NOW,
        recent_actions=["BUY NVDA → FILLED", "HOLD MU → HOLD",
                        ND, "SELL MU → FILLED"])  # 25% ND, consec=0
    assert out["restart_recommended"] is False
    assert out["decision_efficacy"]["verdict"] == "PRODUCING"
    assert "decision_efficacy" in out and "NO_DECISION" not in out["headline"]


def test_empty_recent_actions_is_no_data_efficacy():
    out = build_runner_heartbeat(
        _ago(600), market_open=True, now=NOW, recent_actions=[])
    assert out["decision_efficacy"]["verdict"] == "NO_DATA"
    assert out["restart_recommended"] is False
    assert out["verdict"] == "HEALTHY"


def test_stalled_with_storm_stays_stalled_and_restart():
    """A STALLED loop that is ALSO storming: verdict stays STALLED (cadence),
    restart already True, efficacy still reports the storm so the operator
    sees both root causes."""
    out = build_runner_heartbeat(
        _ago(4 * 3600), market_open=True, now=NOW,
        recent_actions=[ND] * 8)
    assert out["verdict"] == "STALLED"
    assert out["restart_recommended"] is True
    assert out["decision_efficacy"]["verdict"] == "IDLE_STORM"


def test_endpoint_idle_storm_surfaces_restart_keeps_verdict(client):
    """End-to-end: a fresh-cadence loop (HEALTHY) whose recent decisions are
    all NO_DECISION must surface restart_recommended + the storm in
    decision_efficacy while the liveness verdict stays HEALTHY."""
    c, s = client
    for _ in range(6):
        s.record_decision(False, 0, "NO_DECISION",
                          "claude returned no response (timeout/empty)",
                          972.0, 6.0)
    r = c.get("/api/runner-heartbeat")
    assert r.status_code == 200
    d = r.get_json()
    assert d["verdict"] == "HEALTHY"            # cadence is fine
    assert d["restart_recommended"] is True     # but engine is wedged
    assert d["decision_efficacy"]["verdict"] == "IDLE_STORM"
    assert d["decision_efficacy"]["consecutive_no_decision"] == 6


def test_endpoint_producing_when_recent_decision_real(client):
    c, s = client
    s.record_decision(True, 3, "BUY NVDA → FILLED", "{}", 1000.0, 100.0)
    d = c.get("/api/runner-heartbeat").get_json()
    assert d["verdict"] == "HEALTHY"
    assert d["restart_recommended"] is False
    assert d["decision_efficacy"]["verdict"] == "PRODUCING"


# ───────── cause-aware IDLE_STORM restart advice (recent_reasons) ──────────
# A host-saturation / quota storm is NOT cleared by a restart — restarting
# just adds another Opus process. These pin that build_runner_heartbeat
# diagnoses the cause from decisions.reasoning and stops misdirecting the
# operator into the harmful restart.

_HOST_SAT = "skipped claude call — host saturated: 8 concurrent Opus (>4)"
_QUOTA = "claude quota/usage limit exhausted (no decision)"
_TIMEOUT = "claude returned no response (timeout)"


def test_no_decision_cause_buckets():
    assert _no_decision_cause(_HOST_SAT) == "host_saturated"
    assert _no_decision_cause("skipped claude call — host saturated: swap 98%") \
        == "host_saturated"
    assert _no_decision_cause(_QUOTA) == "quota"
    assert _no_decision_cause("claude err: usage limit reached") == "quota"
    assert _no_decision_cause(_TIMEOUT) == "other"
    assert _no_decision_cause("parse_failed: {garbage") == "other"
    assert _no_decision_cause(None) == "other"
    assert _no_decision_cause("") == "other"


def test_dominant_cause_majority_and_tiebreak():
    # Clear majority.
    assert _dominant_cause([_HOST_SAT, _HOST_SAT, _TIMEOUT]) == "host_saturated"
    assert _dominant_cause([_QUOTA, _QUOTA, _TIMEOUT]) == "quota"
    # Tie between host_saturated and quota → host_saturated wins (never
    # under-warn the operator into a useless restart).
    assert _dominant_cause([_HOST_SAT, _QUOTA]) == "host_saturated"
    # Tie between quota and other → quota wins over other.
    assert _dominant_cause([_QUOTA, _TIMEOUT]) == "quota"
    # Empty → other (legacy: assume a restart could help).
    assert _dominant_cause([]) == "other"


def test_idle_storm_host_saturation_does_not_recommend_restart():
    out = build_runner_heartbeat(
        _ago(600), market_open=True, now=NOW,
        recent_actions=[ND] * 6,
        recent_reasons=[_HOST_SAT] * 6)
    eff = out["decision_efficacy"]
    assert eff["verdict"] == "IDLE_STORM"
    assert eff["dominant_cause"] == "host_saturated"
    assert eff["restart_helps"] is False
    # The whole point: a host-saturation storm must NOT recommend a restart.
    assert out["restart_recommended"] is False
    assert "host saturation" in out["headline"]
    assert "will NOT help" in out["headline"]
    assert "wedged Claude CLI" not in out["headline"]


def test_idle_storm_quota_does_not_recommend_restart():
    out = build_runner_heartbeat(
        _ago(600), market_open=True, now=NOW,
        recent_actions=[ND] * 5,
        recent_reasons=[_QUOTA] * 5)
    eff = out["decision_efficacy"]
    assert eff["verdict"] == "IDLE_STORM"
    assert eff["dominant_cause"] == "quota"
    assert eff["restart_helps"] is False
    assert out["restart_recommended"] is False
    assert "quota" in out["headline"] and "will NOT help" in out["headline"]


def test_idle_storm_wedged_cli_still_recommends_restart():
    out = build_runner_heartbeat(
        _ago(600), market_open=True, now=NOW,
        recent_actions=[ND] * 5,
        recent_reasons=[_TIMEOUT] * 5)
    eff = out["decision_efficacy"]
    assert eff["verdict"] == "IDLE_STORM"
    assert eff["dominant_cause"] == "other"
    assert eff["restart_helps"] is True
    # A genuine wedged-CLI storm IS cleared by a restart — advice preserved.
    assert out["restart_recommended"] is True
    assert "wedged Claude CLI" in out["headline"]


def test_idle_storm_only_leading_run_diagnosed():
    # The leading consec run (newest 5) is all host-saturated; an older
    # timeout sits past a real decision and must not dilute the diagnosis.
    out = build_runner_heartbeat(
        _ago(600), market_open=True, now=NOW,
        recent_actions=[ND] * 5 + ["BUY NVDA → FILLED", ND],
        recent_reasons=[_HOST_SAT] * 5 + ["{}", _TIMEOUT])
    eff = out["decision_efficacy"]
    assert eff["consecutive_no_decision"] == 5
    assert eff["dominant_cause"] == "host_saturated"
    assert out["restart_recommended"] is False


def test_idle_storm_omitting_reasons_is_byte_identical():
    """Backward compat: with recent_reasons omitted the IDLE_STORM output is
    byte-identical to before the cause-aware feature — restart still
    recommended, the legacy 'wedged Claude CLI' headline, no new keys."""
    out = build_runner_heartbeat(
        _ago(600), market_open=True, now=NOW, recent_actions=[ND] * 5)
    assert out["restart_recommended"] is True
    assert "wedged Claude CLI" in out["headline"]
    eff = out["decision_efficacy"]
    assert "dominant_cause" not in eff
    assert "restart_helps" not in eff


def test_endpoint_idle_storm_host_saturation_no_restart(client):
    """End-to-end: NO_DECISION rows whose reasoning is host saturation must
    surface IDLE_STORM with restart_recommended False — the endpoint now
    passes recent_reasons so the heartbeat agrees with /api/no-decision-
    reasons that a restart will not help."""
    c, s = client
    for _ in range(6):
        s.record_decision(False, 0, "NO_DECISION", _HOST_SAT, 1000.0, 341.0)
    d = c.get("/api/runner-heartbeat").get_json()
    assert d["verdict"] == "HEALTHY"
    assert d["decision_efficacy"]["verdict"] == "IDLE_STORM"
    assert d["decision_efficacy"]["dominant_cause"] == "host_saturated"
    assert d["restart_recommended"] is False
    assert "will NOT help" in d["headline"]
