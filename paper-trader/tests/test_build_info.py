"""/api/build-info — code-freshness probe.

Long-running dashboard/runner processes silently serve pre-deploy bytecode:
the scorer-clamp fix (commit cd17c16) was committed while the :8090 process
was already up, so production kept extrapolating to ±700% for hours. This
endpoint exposes the git SHA the process booted with vs the on-disk HEAD so
an operator can see "you're running stale code — restart".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def client():
    from paper_trader import dashboard
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c, dashboard


def test_build_info_reports_stale_when_boot_differs_from_head(client, monkeypatch):
    c, dashboard = client
    monkeypatch.setattr(dashboard, "_BOOT_SHA", "aaaaaaa")
    monkeypatch.setattr(dashboard, "_head_sha_and_behind", lambda: ("bbbbbbb", 4))
    data = c.get("/api/build-info").get_json()
    assert data["service"] == "paper_trader"
    assert data["boot_sha"] == "aaaaaaa"
    assert data["head_sha"] == "bbbbbbb"
    assert data["behind"] == 4
    assert data["stale"] is True


def test_build_info_not_stale_when_in_sync(client, monkeypatch):
    c, dashboard = client
    monkeypatch.setattr(dashboard, "_BOOT_SHA", "abc1234")
    monkeypatch.setattr(dashboard, "_head_sha_and_behind", lambda: ("abc1234", 0))
    data = c.get("/api/build-info").get_json()
    assert data["stale"] is False
    assert data["behind"] == 0


def test_build_info_handles_unknown_sha(client, monkeypatch):
    """If git isn't resolvable, never falsely claim staleness."""
    c, dashboard = client
    monkeypatch.setattr(dashboard, "_BOOT_SHA", None)
    monkeypatch.setattr(dashboard, "_head_sha_and_behind", lambda: (None, 0))
    data = c.get("/api/build-info").get_json()
    assert data["stale"] is False
