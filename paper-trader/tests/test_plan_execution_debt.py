"""Tests for the plan-execution-debt audit.

The builder is pure (no DB) so the tests are deterministic fixtures
over the same plan-row shape ``build_deployment_plan`` emits and the
same trade-row shape ``store.recent_trades`` emits. The Flask endpoint
contract is exercised via test_client with the dashboard's actual
deployment_plan_api patched to return fixture rows + the store's
recent_trades patched to return fixture trades.

Coverage:

* Verdict ladder: NO_PLAN / ALIGNED / TIGHTENING / DRIFTING /
  DISCONNECTED at the exact threshold boundaries.
* Per-ticker status ladder: UNEXECUTED / PARTIAL / EXECUTED at the
  exact boundaries.
* Window filter: BUYs outside ``window_hours`` excluded.
* Action filter: only ``BUY`` rows contribute (SELL / HOLD ignored).
* Ticker filter: BUYs for tickers NOT in the plan ignored.
* Garbage rows never raise.
* Headline names the worst-gap ticker.
* Endpoint contract via Flask test_client.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from paper_trader.analytics.plan_execution_debt import (
    ALIGNED_FLOOR_PCT,
    DEFAULT_WINDOW_HOURS,
    DRIFTING_FLOOR_PCT,
    EXECUTED_FLOOR_PCT,
    PARTIAL_FLOOR_PCT,
    TIGHTENING_FLOOR_PCT,
    build_plan_execution_debt,
)


FIXED_NOW = datetime(2026, 5, 28, 23, 30, 0, tzinfo=timezone.utc)


def _plan_row(ticker: str, alloc_usd: float, **kwargs) -> dict:
    return {
        "ticker": ticker,
        "alloc_usd": alloc_usd,
        "alloc_pct_of_book": kwargs.get("alloc_pct_of_book", 25.0),
        "scorer_verdict": kwargs.get("scorer_verdict", "STRONG_HOLD"),
        "sector": kwargs.get("sector", "semis"),
        "is_leveraged": kwargs.get("is_leveraged", False),
        "pred_5d_return_pct": kwargs.get("pred_5d_return_pct", 5.0),
    }


def _trade(ticker: str, value: float, hours_ago: float = 1.0,
           action: str = "BUY", now: datetime | None = None) -> dict:
    # ``now`` lets smoke tests anchor trade timestamps to real wall clock
    # (the endpoint uses real time); unit tests default to FIXED_NOW so the
    # builder-side ``now=FIXED_NOW`` argument keeps the window math hermetic.
    base = now if now is not None else FIXED_NOW
    ts = (base - timedelta(hours=hours_ago)).isoformat()
    return {
        "id": 1,
        "ticker": ticker,
        "action": action,
        "value": value,
        "qty": 1.0,
        "price": value,
        "timestamp": ts,
    }


class TestVerdictLadder:
    def test_no_plan_empty_list(self):
        r = build_plan_execution_debt([], [], now=FIXED_NOW)
        assert r["verdict"] == "NO_PLAN"
        assert r["n_plan_rows"] == 0
        assert r["plan_total_usd"] == 0.0
        assert r["by_ticker"] == []

    def test_no_plan_none_input(self):
        r = build_plan_execution_debt(None, None, now=FIXED_NOW)  # type: ignore[arg-type]
        assert r["verdict"] == "NO_PLAN"

    def test_no_plan_all_zero_alloc(self):
        # Both rows have alloc_usd <= 0 → coalesced out.
        rows = [_plan_row("MUU", 0.0), _plan_row("KLAC", -5.0)]
        r = build_plan_execution_debt(rows, [], now=FIXED_NOW)
        assert r["verdict"] == "NO_PLAN"
        assert r["n_plan_rows"] == 0

    def test_disconnected_zero_executed(self):
        plan = [_plan_row("MUU", 294.67), _plan_row("KLAC", 294.67)]
        r = build_plan_execution_debt(plan, [], now=FIXED_NOW)
        assert r["verdict"] == "DISCONNECTED"
        assert r["execution_pct"] == 0.0
        assert r["unexecuted_usd"] == 589.34
        assert r["n_unexecuted"] == 2

    def test_drifting_at_floor(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [_trade("MUU", 200.0)]   # 20% of 1000
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["execution_pct"] == 20.0
        assert r["verdict"] == "DRIFTING"

    def test_disconnected_just_below_drifting_floor(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [_trade("MUU", 199.0)]   # 19.9% < 20%
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["execution_pct"] == 19.9
        assert r["verdict"] == "DISCONNECTED"

    def test_tightening_at_floor(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [_trade("MUU", 500.0)]   # 50%
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["execution_pct"] == 50.0
        assert r["verdict"] == "TIGHTENING"

    def test_aligned_at_floor(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [_trade("MUU", 800.0)]   # 80%
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["execution_pct"] == 80.0
        assert r["verdict"] == "ALIGNED"

    def test_aligned_overfilled_capped_at_100(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [_trade("MUU", 1500.0)]   # 150%
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["execution_pct"] == 100.0  # rollup capped at 100
        assert r["verdict"] == "ALIGNED"
        # Per-ticker pct is NOT capped — operator wants to see over-fill.
        assert r["by_ticker"][0]["executed_pct"] == 150.0


class TestThresholdConstants:
    """Drift-locks. If anyone bumps the floors, these tests break and
    force a deliberate doc update — same precedent as
    test_held_wire_balance.TestPerTickerVerdict.test_threshold_pin."""

    def test_aligned_floor_pinned(self):
        assert ALIGNED_FLOOR_PCT == 80.0

    def test_tightening_floor_pinned(self):
        assert TIGHTENING_FLOOR_PCT == 50.0

    def test_drifting_floor_pinned(self):
        assert DRIFTING_FLOOR_PCT == 20.0

    def test_executed_floor_pinned(self):
        assert EXECUTED_FLOOR_PCT == 90.0

    def test_partial_floor_pinned(self):
        assert PARTIAL_FLOOR_PCT == 10.0

    def test_default_window_hours_pinned(self):
        assert DEFAULT_WINDOW_HOURS == 24.0


class TestPerTickerStatus:
    def test_unexecuted_zero_trades(self):
        plan = [_plan_row("MUU", 295.0)]
        r = build_plan_execution_debt(plan, [], now=FIXED_NOW)
        row = r["by_ticker"][0]
        assert row["status"] == "UNEXECUTED"
        assert row["executed_usd"] == 0.0
        assert row["gap_usd"] == 295.0

    def test_partial_at_floor(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [_trade("MUU", 100.0)]   # 10% → PARTIAL
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["by_ticker"][0]["status"] == "PARTIAL"

    def test_unexecuted_just_below_partial_floor(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [_trade("MUU", 99.9)]    # 9.99% → UNEXECUTED
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["by_ticker"][0]["status"] == "UNEXECUTED"

    def test_executed_at_floor(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [_trade("MUU", 900.0)]   # 90% → EXECUTED
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["by_ticker"][0]["status"] == "EXECUTED"

    def test_partial_just_below_executed_floor(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [_trade("MUU", 890.0)]   # 89% → PARTIAL
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["by_ticker"][0]["status"] == "PARTIAL"

    def test_multiple_buys_summed(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [
            _trade("MUU", 200.0, hours_ago=2.0),
            _trade("MUU", 300.0, hours_ago=5.0),
        ]
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        row = r["by_ticker"][0]
        assert row["executed_usd"] == 500.0
        assert row["status"] == "PARTIAL"


class TestWindowFilter:
    def test_buys_outside_window_excluded(self):
        plan = [_plan_row("MUU", 1000.0)]
        # 25h ago — outside default 24h window
        trades = [_trade("MUU", 900.0, hours_ago=25.0)]
        r = build_plan_execution_debt(plan, trades,
                                      window_hours=24.0, now=FIXED_NOW)
        assert r["executed_total_usd"] == 0.0
        assert r["verdict"] == "DISCONNECTED"

    def test_buys_at_exact_boundary_included(self):
        plan = [_plan_row("MUU", 1000.0)]
        # 24h ago — at boundary, included (ts >= cutoff)
        trades = [_trade("MUU", 900.0, hours_ago=24.0)]
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["executed_total_usd"] == 900.0
        assert r["verdict"] == "ALIGNED"

    def test_custom_window(self):
        plan = [_plan_row("MUU", 1000.0)]
        # 10h ago — outside 5h window
        trades = [_trade("MUU", 900.0, hours_ago=10.0)]
        r = build_plan_execution_debt(plan, trades,
                                      window_hours=5.0, now=FIXED_NOW)
        assert r["executed_total_usd"] == 0.0

    def test_window_clamped_low(self):
        plan = [_plan_row("MUU", 1000.0)]
        r = build_plan_execution_debt(plan, [],
                                      window_hours=0.001, now=FIXED_NOW)
        assert r["window_hours"] == 0.5

    def test_window_clamped_high(self):
        plan = [_plan_row("MUU", 1000.0)]
        r = build_plan_execution_debt(plan, [],
                                      window_hours=99999.0, now=FIXED_NOW)
        assert r["window_hours"] == 720.0


class TestActionFilter:
    def test_sell_ignored(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [_trade("MUU", 500.0, action="SELL")]
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["executed_total_usd"] == 0.0
        assert r["verdict"] == "DISCONNECTED"

    def test_unknown_action_ignored(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [_trade("MUU", 500.0, action="HOLD")]
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["executed_total_usd"] == 0.0

    def test_buy_lowercase_normalized(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [_trade("MUU", 800.0, action="buy")]
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["executed_total_usd"] == 800.0


class TestTickerFilter:
    def test_buy_for_non_plan_ticker_ignored(self):
        plan = [_plan_row("MUU", 1000.0)]
        # AMD not in plan — ignored
        trades = [_trade("AMD", 500.0)]
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["executed_total_usd"] == 0.0
        assert r["by_ticker"][0]["ticker"] == "MUU"
        assert r["by_ticker"][0]["status"] == "UNEXECUTED"

    def test_ticker_case_normalization(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [_trade("muu", 800.0)]  # lowercase trade ticker
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["executed_total_usd"] == 800.0


class TestGarbageSafety:
    def test_non_list_plan(self):
        r = build_plan_execution_debt("garbage", [], now=FIXED_NOW)  # type: ignore[arg-type]
        assert r["verdict"] == "NO_PLAN"

    def test_non_list_trades(self):
        plan = [_plan_row("MUU", 1000.0)]
        r = build_plan_execution_debt(plan, "garbage", now=FIXED_NOW)  # type: ignore[arg-type]
        assert r["verdict"] == "DISCONNECTED"
        assert r["executed_total_usd"] == 0.0

    def test_non_dict_plan_row_skipped(self):
        plan = [_plan_row("MUU", 1000.0), "garbage", 42]  # type: ignore[list-item]
        r = build_plan_execution_debt(plan, [], now=FIXED_NOW)
        assert r["n_plan_rows"] == 1

    def test_non_dict_trade_row_skipped(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [_trade("MUU", 800.0), "garbage", 42]  # type: ignore[list-item]
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["executed_total_usd"] == 800.0

    def test_missing_ticker_skipped(self):
        plan = [{"alloc_usd": 1000.0, "scorer_verdict": "STRONG_HOLD"}]
        r = build_plan_execution_debt(plan, [], now=FIXED_NOW)
        assert r["verdict"] == "NO_PLAN"

    def test_missing_alloc_skipped(self):
        plan = [{"ticker": "MUU", "scorer_verdict": "STRONG_HOLD"}]
        r = build_plan_execution_debt(plan, [], now=FIXED_NOW)
        assert r["verdict"] == "NO_PLAN"

    def test_malformed_alloc_usd(self):
        plan = [{"ticker": "MUU", "alloc_usd": "not-a-number"}]
        r = build_plan_execution_debt(plan, [], now=FIXED_NOW)
        assert r["verdict"] == "NO_PLAN"

    def test_malformed_trade_timestamp(self):
        plan = [_plan_row("MUU", 1000.0)]
        bad_trade = _trade("MUU", 800.0)
        bad_trade["timestamp"] = "not-a-timestamp"
        r = build_plan_execution_debt(plan, [bad_trade], now=FIXED_NOW)
        assert r["executed_total_usd"] == 0.0

    def test_missing_timestamp_skipped(self):
        plan = [_plan_row("MUU", 1000.0)]
        bad_trade = _trade("MUU", 800.0)
        bad_trade["timestamp"] = None
        r = build_plan_execution_debt(plan, [bad_trade], now=FIXED_NOW)
        assert r["executed_total_usd"] == 0.0

    def test_negative_trade_value_skipped(self):
        plan = [_plan_row("MUU", 1000.0)]
        trades = [_trade("MUU", -500.0)]
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert r["executed_total_usd"] == 0.0


class TestHeadlineAndSortOrder:
    def test_headline_names_worst_gap(self):
        # MUU 295 with 150 executed → gap 145
        # KLAC 200 with 0 executed → gap 200 (largest)
        plan = [_plan_row("MUU", 295.0), _plan_row("KLAC", 200.0)]
        trades = [_trade("MUU", 150.0)]
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        # Sorted by largest gap first → KLAC first
        assert r["by_ticker"][0]["ticker"] == "KLAC"
        assert "KLAC" in r["headline"]

    def test_headline_no_clause_when_fully_filled(self):
        plan = [_plan_row("MUU", 500.0)]
        trades = [_trade("MUU", 500.0)]
        r = build_plan_execution_debt(plan, trades, now=FIXED_NOW)
        assert "Largest miss" not in r["headline"]
        assert r["verdict"] == "ALIGNED"


class TestRowSchema:
    """Every documented field present so downstream consumers can rely
    on the shape — same precedent as test_bootstrap_ci_skill_log."""

    def test_rollup_has_all_fields(self):
        plan = [_plan_row("MUU", 500.0)]
        r = build_plan_execution_debt(plan, [], now=FIXED_NOW)
        for k in (
            "verdict", "headline", "as_of", "window_hours",
            "n_plan_rows", "plan_total_usd", "executed_total_usd",
            "execution_pct", "unexecuted_usd", "partial_gap_usd",
            "n_executed", "n_partial", "n_unexecuted",
            "by_ticker", "thresholds",
        ):
            assert k in r, f"missing field: {k}"

    def test_by_ticker_row_has_all_fields(self):
        plan = [_plan_row("MUU", 500.0)]
        r = build_plan_execution_debt(plan, [_trade("MUU", 100.0)],
                                       now=FIXED_NOW)
        row = r["by_ticker"][0]
        for k in (
            "ticker", "rec_alloc_usd", "rec_alloc_pct_of_book",
            "executed_usd", "executed_pct", "gap_usd",
            "status", "scorer_verdict", "sector",
            "is_leveraged", "pred_5d_return_pct",
        ):
            assert k in row, f"missing field: {k}"


class TestEndpointSmoke:
    """Flask test_client per the analytics-verification memory: module
    __main__ would hit the live data/ DB; the endpoint contract is
    verified by patching ``deployment_plan_api`` + ``store.recent_trades``."""

    def _fixture_plan_resp(self):
        from flask import jsonify
        return jsonify({
            "plan": [_plan_row("MUU", 294.67),
                     _plan_row("KLAC", 294.67)],
            "verdict": "GATED",
            "headline": "test plan",
            "regime": "bull",
            "cash_available_usd": 660.56,
            "deployable_usd": 594.5,
        })

    def test_endpoint_smoke_aligned(self, monkeypatch):
        from paper_trader.dashboard import app
        from paper_trader import dashboard as dash_mod

        # Patch the deployment_plan route handler to return our fixture.
        monkeypatch.setattr(dash_mod, "deployment_plan_api",
                            self._fixture_plan_resp)

        # Patch the store to return fixture trades. Anchored to real wall
        # clock so the endpoint's real-time 24h window includes them.
        _now = datetime.now(timezone.utc)
        class _FakeStore:
            def recent_trades(self, n):
                return [_trade("MUU", 294.67, hours_ago=2.0, now=_now),
                        _trade("KLAC", 294.67, hours_ago=1.0, now=_now)]
        monkeypatch.setattr(dash_mod, "get_store", lambda: _FakeStore())

        client = app.test_client()
        rv = client.get("/api/plan-execution-debt")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["verdict"] == "ALIGNED"
        assert data["execution_pct"] == 100.0
        assert data["n_executed"] == 2
        # Planner context surfaced.
        assert data["planner_verdict"] == "GATED"
        assert data["regime"] == "bull"
        assert data["cash_available_usd"] == 660.56

    def test_endpoint_smoke_disconnected_no_trades(self, monkeypatch):
        from paper_trader.dashboard import app
        from paper_trader import dashboard as dash_mod

        monkeypatch.setattr(dash_mod, "deployment_plan_api",
                            self._fixture_plan_resp)

        class _FakeStore:
            def recent_trades(self, n):
                return []
        monkeypatch.setattr(dash_mod, "get_store", lambda: _FakeStore())

        client = app.test_client()
        rv = client.get("/api/plan-execution-debt")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["verdict"] == "DISCONNECTED"
        assert data["unexecuted_usd"] == 589.34
        assert data["n_unexecuted"] == 2

    def test_endpoint_smoke_window_param(self, monkeypatch):
        from paper_trader.dashboard import app
        from paper_trader import dashboard as dash_mod

        monkeypatch.setattr(dash_mod, "deployment_plan_api",
                            self._fixture_plan_resp)

        # 30h-ago BUY — excluded with window_hours=24, included with
        # window_hours=48. Anchored to real wall clock so the endpoint's
        # real-time cutoff matches the trade's `hours_ago` offset.
        _now = datetime.now(timezone.utc)
        class _FakeStore:
            def recent_trades(self, n):
                return [_trade("MUU", 294.67, hours_ago=30.0, now=_now)]
        monkeypatch.setattr(dash_mod, "get_store", lambda: _FakeStore())

        client = app.test_client()
        rv24 = client.get("/api/plan-execution-debt?window_hours=24")
        rv48 = client.get("/api/plan-execution-debt?window_hours=48")
        assert rv24.status_code == 200
        assert rv48.status_code == 200
        d24 = rv24.get_json()
        d48 = rv48.get_json()
        # 24h window: MUU BUY excluded → 0 executed.
        assert d24["executed_total_usd"] == 0.0
        # 48h window: MUU BUY included → MUU fully filled.
        assert d48["executed_total_usd"] > 0.0

    def test_endpoint_smoke_no_plan(self, monkeypatch):
        from paper_trader.dashboard import app
        from paper_trader import dashboard as dash_mod
        from flask import jsonify

        monkeypatch.setattr(dash_mod, "deployment_plan_api",
                            lambda: jsonify({"plan": [], "verdict": "EMPTY"}))

        class _FakeStore:
            def recent_trades(self, n):
                return []
        monkeypatch.setattr(dash_mod, "get_store", lambda: _FakeStore())

        client = app.test_client()
        rv = client.get("/api/plan-execution-debt")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["verdict"] == "NO_PLAN"

    def test_endpoint_smoke_garbage_window_param(self, monkeypatch):
        from paper_trader.dashboard import app
        from paper_trader import dashboard as dash_mod

        monkeypatch.setattr(dash_mod, "deployment_plan_api",
                            self._fixture_plan_resp)

        class _FakeStore:
            def recent_trades(self, n):
                return []
        monkeypatch.setattr(dash_mod, "get_store", lambda: _FakeStore())

        client = app.test_client()
        # Non-numeric window_hours falls back to DEFAULT_WINDOW_HOURS.
        rv = client.get("/api/plan-execution-debt?window_hours=garbage")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["window_hours"] == DEFAULT_WINDOW_HOURS
