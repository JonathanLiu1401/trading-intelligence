"""Locks for analytics/supervision.build_supervision + the refactored
/api/supervision endpoint + reporter._supervision_line.

The supervision verdict (orphan / stale-code / no-restart-net) is the #1
recurring HIGH live finding and is operator-facing on TWO surfaces now
(/api/supervision JSON and the hourly/daily Discord line). These tests pin
the EXACT verdict + recommendation strings verbatim so a stray space in an
f-string concatenation can never silently break a scraper or the Discord
operator, and assert the endpoint delegates to the builder byte-identically.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.supervision import build_supervision, is_actionable

_T0 = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


# ── exact verdict/recommendation strings (verbatim — DO NOT import) ────────
HEALTHY_REC = "Supervised and current — no action."
STALE_REC = (
    "Supervised but running old code (boot aaaaaaa vs head bbbbbbb, "
    "behind 3). `systemctl --user restart paper-trader` to deploy the "
    "committed fixes.")
UNSUP_REC = (
    "Running current code but with NO restart safety net (orphan / unit "
    "not active+enabled). A clean exit (git-watcher restart, deadman) or "
    "crash leaves the trader DOWN. `systemctl --user enable --now "
    "paper-trader`.")
UNSUP_STALE_REC = (
    "NO restart safety net AND on old code (boot aaaaaaa vs head bbbbbbb, "
    "behind 3). This is an orphan / un-managed run; the moment its "
    "git-watcher or deadman does os._exit(0) the trader stays DOWN. "
    "Re-attach supervision: `systemctl --user enable --now paper-trader` "
    "(it boots on current code).")
UNKNOWN_REC = (
    "Could not read systemd user state from inside the process (user bus "
    "may be unreachable). Verify manually: `systemctl --user is-active "
    "paper-trader; systemctl --user is-enabled paper-trader`.")


class TestVerdictMatrix:
    def test_healthy(self):
        r = build_supervision(pid=42, ppid=99, unit_active="active",
                              unit_enabled="enabled", boot_sha="abc1234",
                              head_sha="abc1234", behind=0, now=_T0)
        assert r["verdict"] == "HEALTHY"
        assert r["recommendation"] == HEALTHY_REC
        assert r["supervised"] is True
        assert r["orphan"] is False
        assert r["stale"] is False
        assert r["actionable"] is False
        assert r["as_of"] == "2026-05-18T12:00:00+00:00"
        assert r["pid"] == 42 and r["ppid"] == 99
        assert r["systemd"] == {"active": "active", "enabled": "enabled"}

    def test_stale_supervised(self):
        r = build_supervision(pid=1, ppid=99, unit_active="active",
                              unit_enabled="enabled", boot_sha="aaaaaaa",
                              head_sha="bbbbbbb", behind=3, now=_T0)
        assert r["verdict"] == "STALE"
        assert r["recommendation"] == STALE_REC
        assert r["supervised"] is True
        assert r["stale"] is True
        assert r["actionable"] is True

    def test_unsupervised_orphan_current(self):
        r = build_supervision(pid=1, ppid=1, unit_active="inactive",
                              unit_enabled="disabled", boot_sha="abc1234",
                              head_sha="abc1234", behind=0, now=_T0)
        assert r["verdict"] == "UNSUPERVISED"
        assert r["recommendation"] == UNSUP_REC
        assert r["supervised"] is False
        assert r["orphan"] is True
        assert r["stale"] is False
        assert r["actionable"] is True

    def test_unsupervised_stale_worst_case(self):
        r = build_supervision(pid=1, ppid=1, unit_active="inactive",
                              unit_enabled="disabled", boot_sha="aaaaaaa",
                              head_sha="bbbbbbb", behind=3, now=_T0)
        assert r["verdict"] == "UNSUPERVISED_STALE"
        assert r["recommendation"] == UNSUP_STALE_REC
        assert r["supervised"] is False
        assert r["orphan"] is True
        assert r["stale"] is True
        assert r["actionable"] is True

    def test_unknown_when_bus_unreadable_non_orphan(self):
        r = build_supervision(pid=1, ppid=99, unit_active="unknown",
                              unit_enabled="unknown", boot_sha="abc1234",
                              head_sha="abc1234", behind=0, now=_T0)
        assert r["verdict"] == "UNKNOWN"
        assert r["recommendation"] == UNKNOWN_REC
        assert r["supervised"] is None
        assert r["actionable"] is True

    def test_unknown_when_only_one_unit_probe_unknown(self):
        # is-active resolves but is-enabled does not → indeterminate.
        r = build_supervision(pid=1, ppid=99, unit_active="active",
                              unit_enabled="unknown", boot_sha="x",
                              head_sha="x", behind=0, now=_T0)
        assert r["verdict"] == "UNKNOWN"
        assert r["supervised"] is None


class TestOrphanPrecedence:
    def test_orphan_beats_systemctl_says_active(self):
        """PPID==1 is the deterministic signal: even if systemctl wrongly
        reports active+enabled (a stale unit cache), an orphan reparented to
        init will NOT be revived — verdict must be UNSUPERVISED, not HEALTHY."""
        r = build_supervision(pid=1, ppid=1, unit_active="active",
                              unit_enabled="enabled", boot_sha="s",
                              head_sha="s", behind=0, now=_T0)
        assert r["verdict"] == "UNSUPERVISED"
        assert r["supervised"] is False

    def test_orphan_with_unknown_units_still_unsupervised(self):
        # orphan precedence is checked BEFORE the unknown→None branch.
        r = build_supervision(pid=1, ppid=1, unit_active="unknown",
                              unit_enabled="unknown", boot_sha="s",
                              head_sha="s", behind=0, now=_T0)
        assert r["verdict"] == "UNSUPERVISED"
        assert r["supervised"] is False

    def test_non_orphan_active_not_enabled_is_unsupervised(self):
        r = build_supervision(pid=1, ppid=99, unit_active="active",
                              unit_enabled="disabled", boot_sha="s",
                              head_sha="s", behind=0, now=_T0)
        assert r["verdict"] == "UNSUPERVISED"
        assert r["supervised"] is False


class TestStaleDerivation:
    def test_no_boot_sha_never_stale(self):
        r = build_supervision(pid=1, ppid=99, unit_active="active",
                              unit_enabled="enabled", boot_sha=None,
                              head_sha="bbbbbbb", behind=9, now=_T0)
        assert r["stale"] is False
        assert r["verdict"] == "HEALTHY"

    def test_no_head_sha_never_stale(self):
        r = build_supervision(pid=1, ppid=99, unit_active="active",
                              unit_enabled="enabled", boot_sha="aaaaaaa",
                              head_sha=None, behind=0, now=_T0)
        assert r["stale"] is False

    def test_equal_shas_not_stale(self):
        r = build_supervision(pid=1, ppid=99, unit_active="active",
                              unit_enabled="enabled", boot_sha="deadbee",
                              head_sha="deadbee", behind=0, now=_T0)
        assert r["stale"] is False

    def test_differing_shas_stale(self):
        r = build_supervision(pid=1, ppid=99, unit_active="active",
                              unit_enabled="enabled", boot_sha="deadbee",
                              head_sha="f00f00f", behind=1, now=_T0)
        assert r["stale"] is True


class TestSafeContract:
    def test_never_raises_returns_unknown_on_bad_clock(self):
        class BadClock:
            def isoformat(self, *a, **k):
                raise RuntimeError("boom")
        r = build_supervision(pid=1, ppid=1, unit_active="active",
                              unit_enabled="enabled", boot_sha="a",
                              head_sha="b", behind=2, now=BadClock())
        assert r["verdict"] == "UNKNOWN"
        assert r["actionable"] is True
        assert "failed" in r["recommendation"]

    def test_is_actionable_only_healthy_suppressed(self):
        assert is_actionable("HEALTHY") is False
        for v in ("STALE", "UNSUPERVISED", "UNSUPERVISED_STALE", "UNKNOWN"):
            assert is_actionable(v) is True
        assert is_actionable(None) is False


class TestEndpointDelegatesToBuilder:
    """The refactored /api/supervision must return exactly what
    build_supervision returns for the same probed inputs (single source of
    truth — invariant #10)."""

    @pytest.fixture
    def client(self):
        from paper_trader import dashboard
        dashboard.app.config["TESTING"] = True
        with dashboard.app.test_client() as c:
            yield c, dashboard

    def test_endpoint_matches_builder_unsupervised_stale(self, client,
                                                          monkeypatch):
        c, dashboard = client
        monkeypatch.setattr(dashboard, "_BOOT_SHA", "aaaaaaa")
        monkeypatch.setattr(dashboard, "_head_sha_and_behind",
                            lambda: ("bbbbbbb", 3))
        monkeypatch.setattr(dashboard._os, "getpid", lambda: 4242)
        monkeypatch.setattr(dashboard._os, "getppid", lambda: 1)

        class _Fake:
            def __init__(self, out):
                self.stdout, self.stderr = out, ""

        def _fake_run(args, **kw):
            verb = args[2]  # ["systemctl","--user",<verb>,"paper-trader"]
            return _Fake("inactive" if verb == "is-active" else "disabled")

        monkeypatch.setattr(dashboard.subprocess, "run", _fake_run)
        data = c.get("/api/supervision").get_json()

        expected = build_supervision(
            pid=4242, ppid=1, unit_active="inactive",
            unit_enabled="disabled", boot_sha="aaaaaaa",
            head_sha="bbbbbbb", behind=3)
        # as_of is wall-clock; assert every other key is byte-identical.
        for k, v in expected.items():
            if k == "as_of":
                continue
            assert data[k] == v, f"endpoint/builder drift on {k!r}"
        assert data["verdict"] == "UNSUPERVISED_STALE"
        assert data["recommendation"] == UNSUP_STALE_REC


class TestReporterSupervisionLine:
    def test_suppressed_when_healthy(self, monkeypatch):
        from paper_trader import reporter
        import paper_trader.analytics.supervision as sup
        monkeypatch.setattr(sup, "build_supervision",
                            lambda **kw: {"verdict": "HEALTHY",
                                          "recommendation": HEALTHY_REC,
                                          "actionable": False})
        monkeypatch.setattr(reporter, "_systemctl_user", lambda v: "active")
        assert reporter._supervision_line() == ""

    def test_surfaced_when_unsupervised_stale(self, monkeypatch):
        from paper_trader import reporter
        import paper_trader.analytics.supervision as sup
        monkeypatch.setattr(sup, "build_supervision",
                            lambda **kw: {"verdict": "UNSUPERVISED_STALE",
                                          "recommendation": UNSUP_STALE_REC,
                                          "actionable": True})
        monkeypatch.setattr(reporter, "_systemctl_user", lambda v: "inactive")
        line = reporter._supervision_line()
        assert line.startswith("⚠️ **SUPERVISION** ◈ UNSUPERVISED_STALE")
        assert UNSUP_STALE_REC in line
        assert "\n> " in line

    def test_surfaced_when_unknown(self, monkeypatch):
        from paper_trader import reporter
        import paper_trader.analytics.supervision as sup
        monkeypatch.setattr(sup, "build_supervision",
                            lambda **kw: {"verdict": "UNKNOWN",
                                          "recommendation": UNKNOWN_REC,
                                          "actionable": True})
        monkeypatch.setattr(reporter, "_systemctl_user", lambda v: "unknown")
        assert reporter._supervision_line().startswith(
            "⚠️ **SUPERVISION** ◈ UNKNOWN")

    def test_builder_exception_degrades_to_empty(self, monkeypatch):
        from paper_trader import reporter
        import paper_trader.analytics.supervision as sup

        def _boom(**kw):
            raise RuntimeError("builder fault")

        monkeypatch.setattr(sup, "build_supervision", _boom)
        monkeypatch.setattr(reporter, "_systemctl_user", lambda v: "active")
        # Additive failure contract: a fault drops THIS line, never raises.
        assert reporter._supervision_line() == ""

    def test_empty_recommendation_degrades_to_empty(self, monkeypatch):
        from paper_trader import reporter
        import paper_trader.analytics.supervision as sup
        monkeypatch.setattr(sup, "build_supervision",
                            lambda **kw: {"verdict": "UNKNOWN",
                                          "recommendation": "",
                                          "actionable": True})
        monkeypatch.setattr(reporter, "_systemctl_user", lambda v: "unknown")
        assert reporter._supervision_line() == ""
