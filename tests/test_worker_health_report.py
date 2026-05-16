"""Worker health snapshot/report: dead workers must be identifiable.

Regression guard for the observability bug where the per-worker log line
hardcoded "alive state=ok" for *every* worker, including the ones counted in
``dead=N``. With state=ok but no success ping in >15min a worker was rolled
up as dead yet logged as alive, making the dead workers impossible to find.
"""
from __future__ import annotations

import logging

import daemon


def _reset_worker_globals(monkeypatch, names):
    monkeypatch.setattr(daemon, "ALL_WORKERS", tuple(names))
    monkeypatch.setattr(daemon, "_worker_crashes", {})
    monkeypatch.setattr(daemon, "_worker_state", {})
    monkeypatch.setattr(daemon, "_worker_disabled_until", {})
    monkeypatch.setattr(daemon, "_worker_total_crashes", {})
    monkeypatch.setattr(daemon, "_worker_last_exception", {})
    monkeypatch.setattr(daemon, "_worker_last_ok", {})


def test_snapshot_marks_stale_ok_worker_dead(monkeypatch):
    now = 1_000_000.0
    _reset_worker_globals(monkeypatch, ["fresh", "stale_ok"])
    # both report supervisor state "ok"; only one has a recent success ping
    daemon._worker_state.update({"fresh": "ok", "stale_ok": "ok"})
    daemon._worker_last_ok.update({
        "fresh": now - 60,            # pinged a minute ago -> alive
        "stale_ok": now - 20 * 60,    # silent 20min, state=ok -> DEAD
    })

    snap = daemon._worker_health_snapshot(now=now)
    by_name = {w["name"]: w for w in snap["workers"]}

    assert snap["workers_ok"] == 1
    assert snap["workers_dead"] == 1
    assert by_name["fresh"]["alive"] is True
    # The core regression: state=ok but stale must be flagged not-alive.
    assert by_name["stale_ok"]["alive"] is False
    assert by_name["stale_ok"]["state"] == "ok"


def test_report_logs_dead_worker_at_warning(monkeypatch, caplog):
    now = 2_000_000.0
    _reset_worker_globals(monkeypatch, ["dead_one"])
    daemon._worker_state.update({"dead_one": "ok"})
    daemon._worker_last_ok.update({"dead_one": now - 30 * 60})

    monkeypatch.setattr(daemon.time, "time", lambda: now)
    monkeypatch.setattr(daemon, "_write_supervisor_state", lambda snap: None)

    with caplog.at_level(logging.INFO, logger=daemon.log.name):
        daemon._worker_health_report()

    dead_recs = [r for r in caplog.records
                 if getattr(r, "worker", None) == "dead_one"]
    assert dead_recs, "expected a per-worker record for dead_one"
    rec = dead_recs[-1]
    assert rec.levelno == logging.WARNING
    assert getattr(rec, "alive") is False
    assert getattr(rec, "event") == "worker_dead"
    assert "DEAD" in rec.getMessage()
