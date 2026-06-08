"""Tests for the desk-pulse digest (analytics/desk_pulse.py + the
/api/desk-pulse endpoint).

These assert *exact behaviour*, not "it returned 200":

* the money block reproduces the **canonical** round-trip metrics
  byte-for-byte on the same fixed ledger /api/analytics uses, and the
  endpoint and /api/analytics are cross-checked to agree (single source of
  truth, AGENTS.md #10 — a re-derived win-split would diverge here);
* concentration is the exact stored-mark market-value recipe incl. the
  option ×100 multiplier and the current_price→avg_cost fallback, and is
  honestly ``None`` (not faked) on an empty book;
* the router precedence is locked at every boundary — a dead loop beats a
  stale SHA beats a behavioural flag beats a lagging loop — with the chosen
  axis's headline forwarded verbatim;
* invariant #12: the builder uses the *passed* initial_cash, never a
  hardcoded 1000 (a literal would fail the −43.5% case);
* the _safe contract: a constituent that raises degrades that block, never
  500s the lifeline endpoint.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import store as store_mod
from paper_trader.analytics import desk_pulse as dp
from paper_trader.analytics.desk_pulse import build_desk_pulse
from paper_trader.store import Store

_NOW = datetime(2026, 5, 17, 18, 0, 0, tzinfo=timezone.utc)


def _ledger() -> list[dict]:
    """The exact 4-round-trip ledger from test_core_analytics, store-native
    NEWEST-FIRST (build_desk_pulse reverses internally, like /api/analytics).
    pnl_usd = [+20, -20, +30, +100] → 3 wins / 1 loss, realised +130."""
    rows = [
        # oldest → newest, then reversed below to mimic recent_trades()
        {"ticker": "AAPL", "action": "BUY", "qty": 10, "price": 10.0,
         "value": 100.0, "option_type": None, "strike": None, "expiry": None,
         "id": 1, "timestamp": "2026-05-01T10:00:00+00:00"},
        {"ticker": "AAPL", "action": "SELL", "qty": 10, "price": 12.0,
         "value": 120.0, "option_type": None, "strike": None, "expiry": None,
         "id": 2, "timestamp": "2026-05-02T10:00:00+00:00"},
        {"ticker": "MSFT", "action": "BUY", "qty": 5, "price": 20.0,
         "value": 100.0, "option_type": None, "strike": None, "expiry": None,
         "id": 3, "timestamp": "2026-05-03T10:00:00+00:00"},
        {"ticker": "MSFT", "action": "SELL", "qty": 5, "price": 16.0,
         "value": 80.0, "option_type": None, "strike": None, "expiry": None,
         "id": 4, "timestamp": "2026-05-04T10:00:00+00:00"},
        {"ticker": "NVDA", "action": "BUY", "qty": 2, "price": 50.0,
         "value": 100.0, "option_type": None, "strike": None, "expiry": None,
         "id": 5, "timestamp": "2026-05-05T10:00:00+00:00"},
        {"ticker": "NVDA", "action": "SELL", "qty": 1, "price": 60.0,
         "value": 60.0, "option_type": None, "strike": None, "expiry": None,
         "id": 6, "timestamp": "2026-05-06T10:00:00+00:00"},
        {"ticker": "NVDA", "action": "SELL", "qty": 1, "price": 70.0,
         "value": 70.0, "option_type": None, "strike": None, "expiry": None,
         "id": 7, "timestamp": "2026-05-07T10:00:00+00:00"},
        {"ticker": "TSLA", "action": "BUY_CALL", "qty": 1, "price": 2.0,
         "value": 200.0, "option_type": "call", "strike": 100.0,
         "expiry": "2026-06-19", "id": 8,
         "timestamp": "2026-05-08T10:00:00+00:00"},
        {"ticker": "TSLA", "action": "SELL_CALL", "qty": 1, "price": 3.0,
         "value": 300.0, "option_type": "call", "strike": 100.0,
         "expiry": "2026-06-19", "id": 9,
         "timestamp": "2026-05-09T10:00:00+00:00"},
    ]
    return list(reversed(rows))


def _fresh_decisions():
    """One very recent decision → heartbeat HEALTHY by default."""
    return [{"timestamp": (_NOW - timedelta(seconds=30)).isoformat(),
             "action_taken": "HOLD", "id": 1}]


# ─────────────────────────── pure money block ───────────────────────────

class TestMoneyBlock:
    def test_round_trip_metrics_match_canonical(self):
        rep = build_desk_pulse(
            {"total_value": 1130.0, "cash": 1130.0}, [], _ledger(),
            _fresh_decisions(), [], build_info={}, market_open=True,
            initial_cash=1000.0, now=_NOW)
        m = rep["money"]
        assert m["n_round_trips"] == 4
        assert m["win_rate_pct"] == 75.0           # 3 of 4 (strict > 0)
        assert m["profit_factor"] == 7.5           # (20+30+100)/20
        assert m["realized_pl_usd"] == 130.0       # +20-20+30+100
        assert m["equity_usd"] == 1130.0
        assert m["total_return_pct"] == 13.0       # (1130-1000)/1000*100

    def test_initial_cash_is_not_hardcoded(self):
        """Invariant #12: a literal 1000 would yield +13.0, not −43.5."""
        rep = build_desk_pulse(
            {"total_value": 1130.0, "cash": 1130.0}, [], _ledger(),
            _fresh_decisions(), [], build_info={}, market_open=True,
            initial_cash=2000.0, now=_NOW)
        assert rep["money"]["total_return_pct"] == -43.5

    def test_external_deposit_is_not_reported_as_return(self):
        """A manual top-up changes capital basis, not trading performance."""
        eq = [
            {"timestamp": "2026-06-08T09:00:00+00:00",
             "total_value": 1000.0, "cash": 1000.0, "sp500_price": 7400.0},
            {"timestamp": "2026-06-08T09:05:00+00:00",
             "total_value": 10000.0, "cash": 10000.0, "sp500_price": 7400.0},
        ]
        rep = build_desk_pulse(
            {"total_value": 10000.0, "cash": 10000.0}, [], [],
            _fresh_decisions(), eq, build_info={}, market_open=True,
            initial_cash=1000.0, now=_NOW)
        m = rep["money"]
        assert m["total_return_pct"] == 0.0
        assert m["deposit_adjusted_return_pct"] == 0.0
        assert m["deposit_adjusted_pnl"] == 0.0
        assert m["capital_basis"] == 10000.0
        assert m["net_external_cash_flow"] == 9000.0
        assert m["raw_total_return_pct"] == 900.0

    def test_unrealized_is_sum_of_marks_null_safe(self):
        positions = [
            {"ticker": "NVDA", "type": "stock", "qty": 10,
             "current_price": 60.0, "avg_cost": 50.0, "unrealized_pl": 100.0},
            {"ticker": "AMD", "type": "stock", "qty": 5,
             "current_price": None, "avg_cost": 40.0, "unrealized_pl": None},
        ]
        rep = build_desk_pulse({"total_value": 1000.0, "cash": 0.0},
                               positions, [], _fresh_decisions(), [],
                               build_info={}, market_open=True, now=_NOW)
        # NULL unrealized contributes 0, never raises.
        assert rep["money"]["unrealized_pl_usd"] == 100.0
        assert rep["money"]["realized_pl_usd"] == 0.0   # no closed trips
        assert rep["money"]["win_rate_pct"] is None     # honest, not 0


# ───────────────────────── pure concentration ──────────────────────────

class TestConcentration:
    def test_top_weight_and_gross_exact(self):
        positions = [
            {"ticker": "NVDA", "type": "stock", "qty": 10,
             "current_price": 60.0, "avg_cost": 50.0},   # mv 600
            {"ticker": "AMD", "type": "stock", "qty": 10,
             "current_price": 40.0, "avg_cost": 30.0},    # mv 400
        ]
        c = build_desk_pulse({"total_value": 1000.0, "cash": 0.0},
                             positions, [], _fresh_decisions(), [],
                             build_info={}, market_open=True,
                             now=_NOW)["concentration"]
        assert c["n_open_positions"] == 2
        assert c["top_name"] == "NVDA"
        assert c["top_weight_pct"] == 60.0          # 600 / 1000
        assert c["gross_exposure_usd"] == 1000.0

    def test_option_x100_and_avg_cost_fallback(self):
        positions = [
            # option: mv = price 5 * qty 1 * 100 = 500 (current_price NULL →
            # avg_cost fallback, the /api/correlation recipe)
            {"ticker": "TSLA", "type": "call", "qty": 1,
             "current_price": None, "avg_cost": 5.0},
            {"ticker": "AAPL", "type": "stock", "qty": 10,
             "current_price": 10.0, "avg_cost": 9.0},      # mv 100
        ]
        c = build_desk_pulse({"total_value": 1.0, "cash": 0.0},
                             positions, [], _fresh_decisions(), [],
                             build_info={}, market_open=True,
                             now=_NOW)["concentration"]
        assert c["top_name"] == "TSLA"
        assert c["top_weight_pct"] == round(500 / 600 * 100, 2)  # 83.33
        assert c["gross_exposure_usd"] == 600.0

    def test_empty_book_is_honest_none(self):
        c = build_desk_pulse({"total_value": 1000.0, "cash": 1000.0}, [],
                             [], _fresh_decisions(), [], build_info={},
                             market_open=True, now=_NOW)["concentration"]
        assert c["top_weight_pct"] is None      # undefined, not faked 0
        assert c["top_name"] is None
        assert c["n_open_positions"] == 0


# ───────────────────────── router precedence ───────────────────────────

class TestRouterPrecedence:
    def _stalled_decisions(self):
        # Market open → expected 1800s; >2× (>3600s) ⇒ STALLED.
        return [{"timestamp": (_NOW - timedelta(hours=3)).isoformat(),
                 "action_taken": "HOLD", "id": 1}]

    def test_stalled_loop_beats_stale_code(self, monkeypatch):
        """A dead loop dominates even a stale SHA + behavioural flags."""
        monkeypatch.setattr(dp, "build_trader_scorecard",
                            lambda *a, **k: {"state": "FLAGS_PRESENT",
                                             "headline": "flagging",
                                             "focus": {"name": "churn"}})
        rep = build_desk_pulse(
            {"total_value": 900.0, "cash": 900.0}, [], _ledger(),
            self._stalled_decisions(), [],
            build_info={"stale": True, "behind": 9}, market_open=True,
            initial_cash=1000.0, now=_NOW)
        assert rep["state"] == "LOOP_STALLED"
        assert rep["liveness"]["verdict"] == "STALLED"
        # headline forwarded verbatim from runner_heartbeat
        assert rep["headline"] == rep["liveness"]["headline"]
        assert rep["liveness"]["restart_recommended"] is True

    def test_stale_code_beats_behavioural_flags(self, monkeypatch):
        monkeypatch.setattr(dp, "build_trader_scorecard",
                            lambda *a, **k: {"state": "FLAGS_PRESENT",
                                             "headline": "2 checks flagging",
                                             "focus": {"name": "churn"}})
        rep = build_desk_pulse(
            {"total_value": 1130.0, "cash": 1130.0}, [], _ledger(),
            _fresh_decisions(), [],
            build_info={"stale": True, "behind": 1}, market_open=True,
            initial_cash=1000.0, now=_NOW)
        assert rep["state"] == "CODE_STALE"
        assert "stale code" in rep["headline"]
        assert "1 commit behind" in rep["headline"]   # singular grammar
        # focus is still forwarded for the operator even when not the router
        assert rep["focus"] == {"name": "churn"}

    def test_behavioural_flags_when_loop_ok_and_code_current(self,
                                                             monkeypatch):
        monkeypatch.setattr(dp, "build_trader_scorecard",
                            lambda *a, **k: {"state": "FLAGS_PRESENT",
                                             "headline": "2 checks flagging",
                                             "focus": {"name": "churn",
                                                       "headline": "churny"}})
        rep = build_desk_pulse(
            {"total_value": 1130.0, "cash": 1130.0}, [], _ledger(),
            _fresh_decisions(), [], build_info={"stale": False, "behind": 0},
            market_open=True, initial_cash=1000.0, now=_NOW)
        assert rep["state"] == "BEHAVIOURAL_FLAGS"
        assert rep["headline"] == "2 checks flagging"

    def test_healthy_when_all_clear(self, monkeypatch):
        monkeypatch.setattr(dp, "build_trader_scorecard",
                            lambda *a, **k: {"state": "ALIGNED_HEALTHY",
                                             "headline": "all good",
                                             "focus": None})
        rep = build_desk_pulse(
            {"total_value": 1130.0, "cash": 1130.0}, [], _ledger(),
            _fresh_decisions(), [], build_info={"stale": False, "behind": 0},
            market_open=True, initial_cash=1000.0, now=_NOW)
        assert rep["state"] == "HEALTHY"

    def test_integrity_unknown_when_not_checked(self, monkeypatch):
        """Honesty: build_info=None (the CLI path) must NOT claim 'code
        current' — it never checked. status=UNKNOWN, and UNKNOWN never
        trips the CODE_STALE branch."""
        monkeypatch.setattr(dp, "build_trader_scorecard",
                            lambda *a, **k: {"state": "ALIGNED_HEALTHY",
                                             "headline": "ok", "focus": None})
        rep = build_desk_pulse(
            {"total_value": 1130.0, "cash": 1130.0}, [], _ledger(),
            _fresh_decisions(), [], build_info=None, market_open=True,
            initial_cash=1000.0, now=_NOW)
        assert rep["integrity"]["status"] == "UNKNOWN"
        assert rep["integrity"]["code_stale"] is False
        assert rep["state"] == "HEALTHY"
        assert "code current" not in rep["headline"]
        assert "not checked" in rep["headline"]

    def test_integrity_status_current_vs_stale(self, monkeypatch):
        monkeypatch.setattr(dp, "build_trader_scorecard",
                            lambda *a, **k: {"state": "ALIGNED_HEALTHY",
                                             "headline": "ok", "focus": None})
        cur = build_desk_pulse(
            {"total_value": 1130.0, "cash": 1130.0}, [], _ledger(),
            _fresh_decisions(), [], build_info={"stale": False, "behind": 0},
            market_open=True, initial_cash=1000.0, now=_NOW)
        assert cur["integrity"]["status"] == "CURRENT"
        stale = build_desk_pulse(
            {"total_value": 1130.0, "cash": 1130.0}, [], _ledger(),
            _fresh_decisions(), [], build_info={"stale": True, "behind": 3},
            market_open=True, initial_cash=1000.0, now=_NOW)
        assert stale["integrity"]["status"] == "STALE"

    def test_no_data_when_no_trades_or_decisions(self):
        rep = build_desk_pulse({"total_value": 1000.0, "cash": 1000.0}, [],
                               [], [], [], build_info={"stale": False},
                               market_open=False, initial_cash=1000.0,
                               now=_NOW)
        assert rep["state"] == "NO_DATA"


# ─────────────────────────── never raises ──────────────────────────────

class TestSafeContract:
    def test_constituent_fault_degrades_not_raises(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("builder exploded")
        monkeypatch.setattr(dp, "build_runner_heartbeat", _boom)
        monkeypatch.setattr(dp, "build_trader_scorecard", _boom)
        # Must not raise; liveness degrades to the ERROR marker.
        rep = build_desk_pulse(
            {"total_value": 1130.0, "cash": 1130.0}, [], _ledger(),
            _fresh_decisions(), [], build_info={"stale": False, "behind": 0},
            market_open=True, initial_cash=1000.0, now=_NOW)
        assert rep["liveness"]["verdict"] == "ERROR"
        assert rep["money"]["realized_pl_usd"] == 130.0   # money still good


# ─────────────────────── Flask endpoint (e2e) ──────────────────────────

@pytest.fixture
def seeded_client(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    s.record_trade("AAPL", "BUY", 10, 10.0)
    s.record_trade("AAPL", "SELL", 10, 12.0)
    s.record_trade("MSFT", "BUY", 5, 20.0)
    s.record_trade("MSFT", "SELL", 5, 16.0)
    s.record_trade("NVDA", "BUY", 2, 50.0)
    s.record_trade("NVDA", "SELL", 1, 60.0)
    s.record_trade("NVDA", "SELL", 1, 70.0)
    s.record_trade("TSLA", "BUY_CALL", 1, 2.0, expiry="2026-06-19",
                   strike=100.0, option_type="call")
    s.record_trade("TSLA", "SELL_CALL", 1, 3.0, expiry="2026-06-19",
                   strike=100.0, option_type="call")
    s.record_decision(True, 3, "HOLD → no fill", "waiting", 1130.0, 1130.0)
    s.update_portfolio(cash=1130.0, total_value=1130.0, positions=[])

    from paper_trader import dashboard
    dashboard.app.config["TESTING"] = True
    try:
        with dashboard.app.test_client() as client:
            yield client, s
    finally:
        s.close()


class TestDeskPulseEndpoint:
    def test_endpoint_ok_and_agrees_with_analytics(self, seeded_client):
        client, _ = seeded_client
        dp_resp = client.get("/api/desk-pulse")
        an_resp = client.get("/api/analytics")
        assert dp_resp.status_code == 200
        d = dp_resp.get_json()
        a = an_resp.get_json()
        assert "error" not in d, d
        # Single source of truth: the digest's realised metrics must equal
        # /api/analytics on the same ledger — a drifted copy fails here.
        assert d["money"]["n_round_trips"] == a["n_round_trips"] == 4
        assert d["money"]["win_rate_pct"] == a["win_rate_pct"] == 75.0
        assert d["money"]["realized_pl_usd"] == 130.0

    def test_endpoint_uses_store_initial_cash_constant(self, seeded_client):
        """Endpoint wires store.INITIAL_CASH (invariant #12) — read the live
        constant so an INITIAL_CASH retune can't false-fail this."""
        client, _ = seeded_client
        d = client.get("/api/desk-pulse").get_json()
        expected = round((1130.0 - store_mod.INITIAL_CASH)
                         / store_mod.INITIAL_CASH * 100, 2)
        assert d["money"]["total_return_pct"] == expected

    def test_endpoint_wires_build_info_integrity(self, seeded_client):
        client, _ = seeded_client
        d = client.get("/api/desk-pulse").get_json()
        integ = d["integrity"]
        # The route fills these from _head_sha_and_behind()/_BOOT_SHA.
        for k in ("status", "code_stale", "commits_behind", "boot_sha",
                  "head_sha"):
            assert k in integ
        # The endpoint always supplies a build_info dict WITH a "stale" key,
        # so status is decided (never the CLI-only UNKNOWN).
        assert integ["status"] in ("CURRENT", "STALE")
        assert isinstance(d["liveness"]["verdict"], str)  # not NO_DATA: 1 dec
