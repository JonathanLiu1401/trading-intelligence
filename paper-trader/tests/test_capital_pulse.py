"""Tests for reporter._capital_pulse_line + its hourly/daily wiring.

The #2 documented live pathology (AGENTS.md pass #14 #4) is a capital-
paralysed book: pinned near 98% deployed, ~$18 free, unable to act for a
day while involuntary NO_DECISION droughts bleed alpha. `capital_paralysis`
serves this on the dashboard and `buying_power` reaches the Opus prompt, but
the operator — who lives in Discord — had no hourly/daily signal that the
desk was frozen. `_capital_pulse_line` closes that surface, composing
`build_capital_paralysis` **verbatim** (single source of truth, invariant
#10) with the additive failure / suppression contract every other reporter
block uses.

These lock the contract with exact assertions:
  * suppression (NO_DATA, healthy-FREE-not-bleeding) → "";
  * PINNED and FREE-but-BLEEDING (the live 2026-05-18 state) → surfaced;
  * the builder's headline / verdict_reason are reproduced VERBATIM
    (no re-derivation — single source of truth);
  * a builder fault degrades to "" and NEVER raises;
  * the block is actually wired into send_hourly_summary AND
    send_daily_close (end-to-end on a real temp Store).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paper_trader.analytics.capital_paralysis as cp_mod
import paper_trader.reporter as reporter
import paper_trader.store as store_mod
from paper_trader.store import Store


class _StubStore:
    """Minimal store — _capital_pulse_line forwards these straight into the
    (monkeypatched) builder, so their contents are irrelevant here."""

    def get_portfolio(self):
        return {"cash": 0.0, "total_value": 0.0, "positions": []}

    def open_positions(self):
        return []

    def recent_trades(self, limit=50):
        return []

    def recent_decisions(self, limit=20):
        return []

    def equity_curve(self, limit=500):
        return []


def _patch_builder(monkeypatch, payload_or_exc):
    def _fake(*a, **k):
        if isinstance(payload_or_exc, Exception):
            raise payload_or_exc
        return payload_or_exc
    monkeypatch.setattr(cp_mod, "build_capital_paralysis", _fake)


# ── Live-state reference payloads (shapes mirror /api/capital-paralysis) ──
_PINNED = {
    "state": "PINNED",
    "headline": ("PINNED — $18.49 cash (1.9%) across 2 position(s); "
                 "selling LITE frees $592.13 → can act again. Staying "
                 "pinned has bled -2.21% alpha across 6 paralysis "
                 "drought(s)."),
    "recommended_unlock": {"ticker": "LITE", "frees_usd": 592.13,
                           "pl_pct": -0.6},
    "paralysis": {"verdict": "BLEEDING",
                  "verdict_reason": ("2.21% of alpha lost across 6 "
                                     "involuntary (parse-failure) droughts")},
}
_FREE_BLEEDING = {
    "state": "FREE",
    "headline": ("FREE — $18.49 cash (1.9%) available; the book can act "
                 "on a new signal without selling."),
    "recommended_unlock": None,
    "paralysis": {"verdict": "BLEEDING",
                  "verdict_reason": ("2.21% of alpha lost across 6 "
                                     "involuntary (parse-failure) droughts")},
}
_FREE_HEALTHY = {
    "state": "FREE",
    "headline": "FREE — $800.00 cash (80%) available; can act freely.",
    "recommended_unlock": None,
    "paralysis": {"verdict": "HEALTHY", "verdict_reason": ""},
}


class TestSuppression:
    def test_no_data_is_silent(self, monkeypatch):
        _patch_builder(monkeypatch, {"state": "NO_DATA", "headline": "x"})
        assert reporter._capital_pulse_line(_StubStore()) == ""

    def test_missing_state_is_silent(self, monkeypatch):
        _patch_builder(monkeypatch, {"headline": "x"})  # state None
        assert reporter._capital_pulse_line(_StubStore()) == ""

    def test_free_and_not_bleeding_is_silent(self, monkeypatch):
        _patch_builder(monkeypatch, _FREE_HEALTHY)
        assert reporter._capital_pulse_line(_StubStore()) == ""

    def test_non_dict_payload_is_silent(self, monkeypatch):
        _patch_builder(monkeypatch, None)
        assert reporter._capital_pulse_line(_StubStore()) == ""

    def test_empty_headline_is_silent(self, monkeypatch):
        _patch_builder(monkeypatch, {"state": "PINNED", "headline": "",
                                     "paralysis": {}})
        assert reporter._capital_pulse_line(_StubStore()) == ""


class TestSurfaced:
    def test_pinned_renders_verbatim_with_unlock_and_reason(self, monkeypatch):
        _patch_builder(monkeypatch, _PINNED)
        out = reporter._capital_pulse_line(_StubStore())
        # header + builder headline VERBATIM (single source of truth)
        assert out.startswith("**CAPITAL** ◈ PINNED\n> ")
        assert _PINNED["headline"] in out
        # the unlock pick is rendered with the exact frees_usd
        assert "> unlock — sell LITE frees $592.13" in out
        # the involuntary-bleed verdict reason is reproduced verbatim
        assert "> " + _PINNED["paralysis"]["verdict_reason"] in out

    def test_free_but_bleeding_is_surfaced_live_state(self, monkeypatch):
        """The live 2026-05-18 book: can_act→FREE yet droughts BLEEDING.
        A FREE book is only suppressed when it is NOT bleeding; this one
        IS, so the operator must see it."""
        _patch_builder(monkeypatch, _FREE_BLEEDING)
        out = reporter._capital_pulse_line(_StubStore())
        assert out.startswith("**CAPITAL** ◈ FREE")
        assert _FREE_BLEEDING["headline"] in out
        assert _FREE_BLEEDING["paralysis"]["verdict_reason"] in out
        # no unlock pick when the builder recommends none
        assert "unlock — sell" not in out

    def test_pinned_without_unlock_or_reason_still_minimal(self, monkeypatch):
        _patch_builder(monkeypatch, {
            "state": "PINNED",
            "headline": "PINNED — $0.50 cash across 1 position(s).",
            "recommended_unlock": None,
            "paralysis": {"verdict": "WATCH", "verdict_reason": "x"},
        })
        out = reporter._capital_pulse_line(_StubStore())
        assert out == ("**CAPITAL** ◈ PINNED\n"
                       "> PINNED — $0.50 cash across 1 position(s).")

    def test_unlock_garbage_frees_usd_does_not_crash(self, monkeypatch):
        _patch_builder(monkeypatch, {
            "state": "PINNED", "headline": "PINNED — pinned.",
            "recommended_unlock": {"ticker": "MU", "frees_usd": "not-a-num"},
            "paralysis": {},
        })
        out = reporter._capital_pulse_line(_StubStore())
        assert "> unlock — sell MU frees $0.00" in out


class TestFailureContract:
    def test_builder_exception_degrades_to_empty_never_raises(self,
                                                              monkeypatch):
        _patch_builder(monkeypatch, RuntimeError("boom"))
        # must NOT raise — a diagnostics fault drops the line, never the
        # whole Discord summary
        assert reporter._capital_pulse_line(_StubStore()) == ""

    def test_store_method_exception_degrades_to_empty(self, monkeypatch):
        class _Boom:
            def get_portfolio(self):
                raise RuntimeError("store down")
        # build_capital_paralysis is real here; the store read explodes first
        assert reporter._capital_pulse_line(_Boom()) == ""


class TestWiredIntoReports:
    """End-to-end on a real temp Store: the PINNED block must appear in BOTH
    the hourly and daily-close Discord bodies (the wiring is the feature)."""

    @pytest.fixture
    def fresh_store(self, tmp_path, monkeypatch):
        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        try:
            yield s
        finally:
            s.close()

    def _capture(self, monkeypatch):
        sent = {}

        def _fake_send(body):
            sent["body"] = body
            return True
        monkeypatch.setattr(reporter, "_send", _fake_send)
        # no network in the report path
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda *a, **k: 5800.0)
        _patch_builder(monkeypatch, _PINNED)
        return sent

    def test_hourly_includes_capital_block(self, fresh_store, monkeypatch):
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        sent = self._capture(monkeypatch)
        assert reporter.send_hourly_summary() is True
        body = sent["body"]
        assert "**CAPITAL** ◈ PINNED" in body
        assert "> unlock — sell LITE frees $592.13" in body

    def test_daily_close_includes_capital_block(self, fresh_store,
                                                monkeypatch):
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        sent = self._capture(monkeypatch)
        assert reporter.send_daily_close() is True
        assert "**CAPITAL** ◈ PINNED" in sent["body"]

    def test_healthy_free_book_adds_no_capital_block(self, fresh_store,
                                                     monkeypatch):
        """Regression: the additive contract must not inject noise into a
        healthy book's summary (the existing reporter-test fixtures rely on
        this — a non-suppressed block here would break exact-body tests)."""
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        sent = {}

        def _fake_send(body):
            sent["body"] = body
            return True
        monkeypatch.setattr(reporter, "_send", _fake_send)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda *a, **k: 5800.0)
        _patch_builder(monkeypatch, _FREE_HEALTHY)
        assert reporter.send_hourly_summary() is True
        assert "**CAPITAL**" not in sent["body"]
