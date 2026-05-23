"""Locks `reporter._starvation_trend_line` — the temporal companion to
`_host_pulse_line`. `_host_pulse_line` says "is the box saturated?";
this says "is the storm WORSENING or RECOVERING?". An operator deciding
whether to KILL parallel Opus jobs vs WAIT for the storm to pass needs
THIS direction-of-travel signal, not the aggregate snapshot.

Test contracts mirror the `_host_pulse_line` suite (same builder shape,
single source of truth: the builder's headline is rendered verbatim)."""

from __future__ import annotations

import pytest

from paper_trader import reporter


class TestStarvationTrendLine:
    """`_starvation_trend_line` direct unit coverage."""

    _WORSENING = {
        "state": "WORSENING",
        "older_rate": 0.40, "newer_rate": 0.85,
        "older_n": 30, "newer_n": 30, "delta": 0.45,
        "headline": ("Starvation WORSENING: 40%→85% (+45pp over the last 30 "
                     "cycles); the storm is intensifying — the bot cannot "
                     "resolve this by trading."),
        "ok": True,
    }
    _RECOVERING = {
        "state": "RECOVERING",
        "older_rate": 0.95, "newer_rate": 0.10,
        "older_n": 30, "newer_n": 30, "delta": -0.85,
        "headline": ("Starvation RECOVERING: 95%→10% (-85pp over the last 30 "
                     "cycles); the box is clearing — give it a few more "
                     "cycles before intervening."),
        "ok": True,
    }
    _STABLE_LOW = {
        "state": "STABLE",
        "older_rate": 0.05, "newer_rate": 0.07,
        "older_n": 30, "newer_n": 30, "delta": 0.02,
        "headline": ("Starvation STABLE: 5%→7% (+2pp over the last 30 "
                     "cycles); rate is not moving."),
        "ok": True,
    }
    _STABLE_HIGH = {
        "state": "STABLE",
        "older_rate": 1.0, "newer_rate": 1.0,
        "older_n": 60, "newer_n": 60, "delta": 0.0,
        "headline": ("Starvation STABLE: 100%→100% (+0pp over the last 60 "
                     "cycles); rate is not moving."),
        "ok": True,
    }
    _STABLE_MEDIUM = {  # boundary: 30% exactly should fire (max(.., ..) == 0.30)
        "state": "STABLE",
        "older_rate": 0.30, "newer_rate": 0.30,
        "older_n": 30, "newer_n": 30, "delta": 0.0,
        "headline": ("Starvation STABLE: 30%→30% (+0pp over the last 30 "
                     "cycles); rate is not moving."),
        "ok": True,
    }
    _INSUFFICIENT = {
        "state": "INSUFFICIENT",
        "older_rate": 0.0, "newer_rate": 0.0,
        "older_n": 3, "newer_n": 3, "delta": 0.0,
        "headline": ("Only 3/3 (newer/older) decision(s) — need ≥10 per "
                     "half for a trend verdict."),
        "ok": True,
    }

    def _stub_trend(self, monkeypatch, payload):
        from paper_trader import host_guard
        monkeypatch.setattr(host_guard, "recent_starvation_trend",
                            lambda *a, **k: payload)

    def test_worsening_fires_with_warning_tag(self, monkeypatch):
        self._stub_trend(monkeypatch, self._WORSENING)
        line = reporter._starvation_trend_line()
        # Tag carries the WORSENING warning emoji so the operator's eye
        # snaps to it in a long Discord summary.
        assert line.startswith("**STARVATION TREND** ◈ ⚠️ WORSENING")
        # Builder headline rendered VERBATIM — single source of truth.
        assert self._WORSENING["headline"] in line
        # The "act now" discriminator that distinguishes it from STABLE.
        assert "intensifying" in line

    def test_recovering_fires_with_success_tag(self, monkeypatch):
        self._stub_trend(monkeypatch, self._RECOVERING)
        line = reporter._starvation_trend_line()
        assert line.startswith("**STARVATION TREND** ◈ ✅ RECOVERING")
        assert self._RECOVERING["headline"] in line
        # The "wait it out" discriminator.
        assert "before intervening" in line

    def test_stable_low_baseline_is_silent(self, monkeypatch):
        """Silence-when-nothing-actionable — a 5%→7% stable rate is normal
        noise; surfacing it makes the summary its own lying green light."""
        self._stub_trend(monkeypatch, self._STABLE_LOW)
        assert reporter._starvation_trend_line() == ""

    def test_stable_high_baseline_fires(self, monkeypatch):
        """A 100%→100% stable rate is a sustained storm wall — distinct
        operator signal from the `_host_pulse_line` snapshot."""
        self._stub_trend(monkeypatch, self._STABLE_HIGH)
        line = reporter._starvation_trend_line()
        assert line.startswith("**STARVATION TREND** ◈ STABLE")
        assert self._STABLE_HIGH["headline"] in line

    def test_stable_30pct_boundary_inclusive(self, monkeypatch):
        """The 30% baseline cutoff is inclusive — a flat 30% storm fires."""
        self._stub_trend(monkeypatch, self._STABLE_MEDIUM)
        line = reporter._starvation_trend_line()
        assert line.startswith("**STARVATION TREND** ◈ STABLE")

    def test_stable_just_below_threshold_silent(self, monkeypatch):
        """Boundary discipline: 29% (just under 30%) is silent."""
        payload = dict(self._STABLE_HIGH)
        payload.update(older_rate=0.29, newer_rate=0.29,
                       headline="Starvation STABLE: 29%→29% ...")
        self._stub_trend(monkeypatch, payload)
        assert reporter._starvation_trend_line() == ""

    def test_insufficient_data_is_silent(self, monkeypatch):
        """Never publish a verdict on a tiny sample (mirrors the builder's
        INSUFFICIENT bucket — verdict withheld)."""
        self._stub_trend(monkeypatch, self._INSUFFICIENT)
        assert reporter._starvation_trend_line() == ""

    def test_missing_state_is_silent(self, monkeypatch):
        """Degenerate payload (no state) — safe silence."""
        self._stub_trend(monkeypatch, {"state": None, "headline": "..."})
        assert reporter._starvation_trend_line() == ""

    def test_missing_headline_is_silent(self, monkeypatch):
        """Builder said state but no headline — never fabricate text."""
        self._stub_trend(monkeypatch, {"state": "WORSENING", "headline": ""})
        assert reporter._starvation_trend_line() == ""

    def test_non_dict_response_is_silent(self, monkeypatch):
        """Defense against a builder regression returning None / wrong type."""
        self._stub_trend(monkeypatch, None)
        assert reporter._starvation_trend_line() == ""

    def test_degrades_to_empty_on_builder_fault(self, monkeypatch):
        """Additive failure contract — a fault drops this line, never
        raises (the `_host_pulse_line` precedent)."""
        from paper_trader import host_guard

        def _boom(*a, **k):
            raise RuntimeError("host_guard.recent_starvation_trend blew up")

        monkeypatch.setattr(host_guard, "recent_starvation_trend", _boom)
        assert reporter._starvation_trend_line() == ""

    def test_non_numeric_rates_are_silent_for_stable(self, monkeypatch):
        """Defensive: a malformed STABLE payload with garbage rates degrades
        to silence rather than raising or fabricating a verdict."""
        self._stub_trend(monkeypatch, {
            "state": "STABLE",
            "older_rate": "garbage", "newer_rate": None,
            "headline": "Starvation STABLE: ...",
            "ok": True,
        })
        assert reporter._starvation_trend_line() == ""


class TestStarvationTrendHourlyWiring:
    """End-to-end wiring of `_starvation_trend_line` into
    `send_hourly_summary` / `send_daily_close`."""

    _WORSENING_PAYLOAD = {
        "state": "WORSENING",
        "older_rate": 0.20, "newer_rate": 0.80,
        "older_n": 30, "newer_n": 30, "delta": 0.60,
        "headline": "Starvation WORSENING: 20%→80% — the storm is intensifying",
        "ok": True,
    }

    def _stub_market(self, monkeypatch):
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5100.0)

    def test_hourly_emits_starvation_trend_when_worsening(
            self, fresh_store, monkeypatch):
        from paper_trader import host_guard
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        self._stub_market(monkeypatch)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        monkeypatch.setattr(host_guard, "recent_starvation_trend",
                            lambda *a, **k: self._WORSENING_PAYLOAD)
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**STARVATION TREND** ◈ ⚠️ WORSENING" in body
        # Builder's own headline carried verbatim.
        assert "the storm is intensifying" in body

    def test_hourly_silent_when_stable_low(self, fresh_store, monkeypatch):
        from paper_trader import host_guard
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        self._stub_market(monkeypatch)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        # Low-baseline STABLE — must not appear in body.
        monkeypatch.setattr(host_guard, "recent_starvation_trend",
                            lambda *a, **k: {
                                "state": "STABLE",
                                "older_rate": 0.0, "newer_rate": 0.05,
                                "older_n": 30, "newer_n": 30, "delta": 0.05,
                                "headline": "Starvation STABLE: 0%→5%",
                                "ok": True,
                            })
        assert reporter.send_hourly_summary() is True
        assert "**STARVATION TREND**" not in captured[0]

    def test_daily_close_includes_starvation_trend_when_worsening(
            self, fresh_store, monkeypatch):
        from paper_trader import host_guard
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        self._stub_market(monkeypatch)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        monkeypatch.setattr(host_guard, "recent_starvation_trend",
                            lambda *a, **k: self._WORSENING_PAYLOAD)
        assert reporter.send_daily_close() is True
        body = captured[0]
        assert "**DAILY CLOSE**" in body
        assert "**STARVATION TREND** ◈ ⚠️ WORSENING" in body

    def test_starvation_trend_appears_right_after_host(
            self, fresh_store, monkeypatch):
        """The load-bearing ORDER: STARVATION TREND sits IMMEDIATELY after
        the HOST line in the body. A top-down read goes "is the box
        saturated? — yes; which way is it going? — worsening". A reordering
        regression that floated TREND above HOST or below IDLE would put
        the action signal in the wrong place."""
        from paper_trader import host_guard
        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        self._stub_market(monkeypatch)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        # Force HOST + TREND both to fire; capital line forced to fire too
        # so the TREND<CAPITAL ordering can be checked.
        monkeypatch.setattr(host_guard, "pulse", lambda *a, **k: {
            "state": "SATURATED",
            "headline": "Opus is starved by the box"})
        monkeypatch.setattr(host_guard, "recent_starvation_trend",
                            lambda *a, **k: self._WORSENING_PAYLOAD)
        monkeypatch.setattr(reporter, "_capital_pulse_line",
                            lambda store: "**CAPITAL** ◈ PINNED\n> ~98%")
        assert reporter.send_hourly_summary() is True
        body = captured[0]
        assert "**HOST**" in body
        assert "**STARVATION TREND**" in body
        assert "**CAPITAL**" in body
        assert body.index("**HOST**") < body.index("**STARVATION TREND**")
        assert body.index("**STARVATION TREND**") < body.index("**CAPITAL**")

    def test_hourly_silent_when_starvation_trend_fault(
            self, fresh_store, monkeypatch):
        """Builder fault on the new trend line drops just that line,
        never the whole hourly summary (the additive contract)."""
        from paper_trader import host_guard

        def _boom(*a, **k):
            raise RuntimeError("starvation-trend builder blew up")

        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        self._stub_market(monkeypatch)
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        monkeypatch.setattr(host_guard, "recent_starvation_trend", _boom)
        # The hourly itself must still send.
        assert reporter.send_hourly_summary() is True
        assert "**STARVATION TREND**" not in captured[0]


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    """In-memory-ish Store backed by a temp file so each test starts clean.
    Mirrors the shared conftest fixture used by other reporter tests."""
    from paper_trader import store as store_mod

    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = store_mod.Store()
    yield s
    s.close()
