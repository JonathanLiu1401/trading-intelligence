"""Unit tests for paper_trader.analytics.concurrent_opus_attribution.

Pure-builder asserts on a synthetic process table — tests never touch
``/proc``. Verifies the verdict ladder, parent-chain walk, group
ordering, legitimacy logic, dominant-culprit selection, kill-command
templates, and the never-raises / advisory contract.

Plus one Flask test-client test that exercises the endpoint wiring
without depending on the live host's /proc state (the project memory
``paper-trader analytics verification`` notes — verify via test client,
not CLI smoke).
"""
from __future__ import annotations

from paper_trader.analytics.concurrent_opus_attribution import (
    BACKTEST_LEGIT_MAX,
    BENIGN_MAX,
    DEFAULT_OPUS_MARKER,
    ELEVATED_MAX,
    build_concurrent_opus_attribution,
)


def _opus_proc(pid: int, ppid: int) -> dict:
    return {
        "pid": pid,
        "ppid": ppid,
        "cmdline": (
            "claude --model claude-opus-4-7 --permission-mode "
            "bypassPermissions --print BEFORE STARTING: Read AGENTS"
        ),
    }


def _bash_proc(pid: int, ppid: int, *, kind: str) -> dict:
    cmd_by_kind = {
        "hourly_review": (
            "bash /home/zeph/trading-intelligence/paper-trader/scripts/"
            "hourly_review.sh"
        ),
        "backtest": (
            "python3 /home/zeph/trading-intelligence/paper-trader/"
            "run_continuous_backtests.py"
        ),
        "runner": "python3 -m paper_trader.runner",
        "daemon": (
            "python3 /home/zeph/trading-intelligence/digital-intern/daemon.py"
        ),
        "interactive": "-bash",  # plain login shell — chain falls off
    }
    return {"pid": pid, "ppid": ppid, "cmdline": cmd_by_kind[kind]}


# ── verdict ladder ─────────────────────────────────────────────────────


def test_empty_input_yields_no_opus():
    out = build_concurrent_opus_attribution([])
    assert out["verdict"] == "NO_OPUS"
    assert out["state"] == "READY"
    assert out["n_opus"] == 0
    assert out["groups"] == []
    assert out["dominant_culprit"] is None
    assert out["recommendation"] == ""


def test_none_input_yields_no_opus():
    out = build_concurrent_opus_attribution(None)
    assert out["verdict"] == "NO_OPUS"
    assert out["n_opus"] == 0


def test_single_runner_opus_is_clean():
    # Live runner spawns ONE Opus per cycle — that's the legitimate baseline.
    procs = [
        _bash_proc(100, 1, kind="runner"),
        _opus_proc(200, 100),
    ]
    out = build_concurrent_opus_attribution(procs)
    assert out["verdict"] == "CLEAN"
    assert out["state"] == "READY"
    assert out["n_opus"] == 1
    assert len(out["groups"]) == 1
    g = out["groups"][0]
    assert g["parent_marker"] == "runner"
    assert g["is_legitimate"] is True
    assert g["pids"] == [200]
    assert out["dominant_culprit"] is None
    assert out["recommendation"] == ""


def test_benign_at_max_legit_backtest():
    # Backtest committee at its _CLAUDE_SEM cap (3) is legitimate.
    procs = [_bash_proc(100, 1, kind="backtest")]
    procs += [_opus_proc(200 + i, 100) for i in range(BACKTEST_LEGIT_MAX)]
    out = build_concurrent_opus_attribution(procs)
    assert out["n_opus"] == BACKTEST_LEGIT_MAX
    assert out["verdict"] == "BENIGN"
    assert out["state"] == "READY"
    g = out["groups"][0]
    assert g["parent_marker"] == "backtest"
    assert g["is_legitimate"] is True
    assert out["dominant_culprit"] is None
    assert out["recommendation"] == ""


def test_backtest_above_cap_is_illegitimate():
    # Backtest committee slipped its semaphore cap — flagged.
    procs = [_bash_proc(100, 1, kind="backtest")]
    n = BACKTEST_LEGIT_MAX + 1
    procs += [_opus_proc(200 + i, 100) for i in range(n)]
    out = build_concurrent_opus_attribution(procs)
    assert out["n_opus"] == n
    g = out["groups"][0]
    assert g["parent_marker"] == "backtest"
    assert g["is_legitimate"] is False
    assert out["dominant_culprit"] is not None
    assert out["dominant_culprit"]["parent_marker"] == "backtest"


def test_elevated_5_to_8_opus():
    procs = [_bash_proc(100, 1, kind="hourly_review")]
    procs += [_opus_proc(200 + i, 100) for i in range(5)]
    out = build_concurrent_opus_attribution(procs)
    assert out["n_opus"] == 5
    assert out["verdict"] == "ELEVATED"
    assert out["state"] == "DEGRADED"


def test_saturated_9_plus_opus():
    procs = [_bash_proc(100, 1, kind="hourly_review")]
    procs += [_opus_proc(200 + i, 100) for i in range(ELEVATED_MAX + 1)]
    out = build_concurrent_opus_attribution(procs)
    assert out["n_opus"] == ELEVATED_MAX + 1
    assert out["verdict"] == "SATURATED"
    assert out["state"] == "STORM"


def test_seventeen_opus_storm_live_footprint():
    # Reproduces the 2026-05-23 live state: 17 Opus all rooted in
    # hourly_review.sh.
    procs = [_bash_proc(100, 1, kind="hourly_review")]
    procs += [_opus_proc(1000 + i, 100) for i in range(17)]
    out = build_concurrent_opus_attribution(procs)
    assert out["n_opus"] == 17
    assert out["verdict"] == "SATURATED"
    assert len(out["groups"]) == 1
    g = out["groups"][0]
    assert g["parent_marker"] == "hourly_review"
    assert g["n_opus"] == 17
    assert g["is_legitimate"] is False
    assert "pkill -f scripts/hourly_review.sh" in g["kill_command"]
    assert out["dominant_culprit"]["parent_marker"] == "hourly_review"
    assert "pkill" in out["recommendation"]


# ── parent-chain walk ──────────────────────────────────────────────────


def test_parent_chain_walks_through_nested_bash():
    # Real /proc footprint: Opus → bash hourly_review.sh → bash
    # hourly_review.sh → init. The walker must hit the script marker on
    # the *second* hop, not give up after one.
    procs = [
        _bash_proc(50, 1, kind="hourly_review"),   # outer script
        _bash_proc(100, 50, kind="hourly_review"), # inner script
        _opus_proc(200, 100),                       # leaf Opus
    ]
    out = build_concurrent_opus_attribution(procs)
    assert out["n_opus"] == 1
    assert out["groups"][0]["parent_marker"] == "hourly_review"


def test_chain_falls_off_to_unknown():
    # Opus parented by a plain interactive shell (no recognized marker) →
    # unknown bucket. The kill command falls back to explicit PIDs.
    procs = [
        _bash_proc(100, 1, kind="interactive"),
        _opus_proc(200, 100),
    ]
    out = build_concurrent_opus_attribution(procs)
    assert out["n_opus"] == 1
    g = out["groups"][0]
    assert g["parent_marker"] == "unknown"
    assert g["is_legitimate"] is False
    # Unknown groups expose the PID list (no pkill template available).
    assert g["kill_command"] == "kill 200"


def test_parent_chain_terminates_on_init():
    # ppid=1 immediately — chain terminates without walking further.
    procs = [_opus_proc(200, 1)]
    out = build_concurrent_opus_attribution(procs)
    assert out["groups"][0]["parent_marker"] == "unknown"


def test_parent_chain_caps_hops_under_cycle():
    # Synthetic cycle: 100 → 101 → 100 → ... The walker must terminate
    # via the seen-set guard, classifying as unknown.
    procs = [
        {"pid": 100, "ppid": 101, "cmdline": "bash"},
        {"pid": 101, "ppid": 100, "cmdline": "bash"},
        _opus_proc(200, 100),
    ]
    out = build_concurrent_opus_attribution(procs)
    assert out["groups"][0]["parent_marker"] == "unknown"


# ── grouping + ordering ────────────────────────────────────────────────


def test_groups_sorted_by_count_desc():
    # Two illegit groups: hourly_review with 5, daemon with 2. Bigger
    # first.
    procs = [
        _bash_proc(50, 1, kind="hourly_review"),
        _bash_proc(60, 1, kind="daemon"),
    ]
    procs += [_opus_proc(100 + i, 50) for i in range(5)]
    procs += [_opus_proc(200 + i, 60) for i in range(2)]
    out = build_concurrent_opus_attribution(procs)
    assert out["n_opus"] == 7
    assert out["groups"][0]["parent_marker"] == "hourly_review"
    assert out["groups"][0]["n_opus"] == 5
    assert out["groups"][1]["parent_marker"] == "daemon"
    assert out["groups"][1]["n_opus"] == 2
    # Recommendation prescribes the bigger culprit first.
    assert "hourly_review" in out["recommendation"]
    assert out["recommendation"].find("hourly") < out["recommendation"].find(
        "daemon"
    )


def test_runner_plus_illegit_storm_dominant_skips_runner():
    # 1 legit runner Opus + 6 illegit hourly_review Opus → dominant
    # culprit is the hourly_review group, NOT the runner.
    procs = [
        _bash_proc(50, 1, kind="runner"),
        _bash_proc(60, 1, kind="hourly_review"),
    ]
    procs.append(_opus_proc(100, 50))            # legit
    procs += [_opus_proc(200 + i, 60) for i in range(6)]  # illegit
    out = build_concurrent_opus_attribution(procs)
    assert out["n_opus"] == 7
    assert out["verdict"] == "ELEVATED"
    assert out["dominant_culprit"]["parent_marker"] == "hourly_review"
    # Runner group remains legitimate.
    runner_g = next(g for g in out["groups"] if g["parent_marker"] == "runner")
    assert runner_g["is_legitimate"] is True


def test_pids_are_sorted_within_group():
    procs = [_bash_proc(100, 1, kind="hourly_review")]
    # Insert Opus PIDs in scrambled order — output must still sort.
    for pid in (500, 100_001, 200, 300):
        procs.append(_opus_proc(pid, 100))
    out = build_concurrent_opus_attribution(procs)
    g = out["groups"][0]
    assert g["pids"] == [200, 300, 500, 100_001]


# ── never-raises contract ──────────────────────────────────────────────


def test_garbage_rows_dropped_silently():
    procs = [
        None,
        {},
        {"pid": "notanint", "ppid": 1, "cmdline": "x"},
        {"pid": -1, "ppid": 1, "cmdline": "x"},
        "not a dict",
        _opus_proc(200, 1),
    ]
    out = build_concurrent_opus_attribution(procs)
    # The valid Opus row survives; everything else is discarded.
    assert out["n_opus"] == 1


def test_marker_override_disables_attribution():
    # If the marker doesn't match any cmdline, n_opus == 0.
    procs = [_opus_proc(200, 1)]
    out = build_concurrent_opus_attribution(procs, marker="nonexistent")
    assert out["n_opus"] == 0
    assert out["verdict"] == "NO_OPUS"


def test_default_marker_matches_live_opus_cmdline():
    # Documents the canonical marker the live runner uses; if this ever
    # drifts, both host_guard and this builder need to be updated together.
    assert "claude --model claude-opus" == DEFAULT_OPUS_MARKER


# ── endpoint wiring ────────────────────────────────────────────────────


def test_endpoint_wiring_synthetic(monkeypatch):
    # Inject a synthetic /proc table into scan_proc_table so the endpoint
    # does not depend on the host's actual state. Verifies the dashboard
    # route is reachable, the JSON shape is preserved, and the builder
    # is called.
    from paper_trader import dashboard
    from paper_trader.analytics import concurrent_opus_attribution as mod

    synthetic = [
        _bash_proc(100, 1, kind="hourly_review"),
        _opus_proc(200, 100),
        _opus_proc(201, 100),
    ]
    monkeypatch.setattr(mod, "scan_proc_table", lambda: synthetic)

    client = dashboard.app.test_client()
    resp = client.get("/api/concurrent-opus-attribution")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["n_opus"] == 2
    assert body["verdict"] == "BENIGN"
    assert body["groups"][0]["parent_marker"] == "hourly_review"
    assert body["groups"][0]["is_legitimate"] is False
    assert "pkill -f scripts/hourly_review.sh" in body["groups"][0]["kill_command"]
    assert body["as_of"] is not None
