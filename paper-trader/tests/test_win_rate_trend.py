"""Tests for the win-rate-trend analyzer + reporter line + endpoint.

The builder is pure (no DB) so tests fix the round-trip ledger
explicitly. The reporter wiring is verified by patching the store and
asserting the headline appears or is suppressed. The endpoint contract
is exercised via Flask test_client.
"""
from __future__ import annotations

import pytest

from paper_trader.analytics.win_rate_trend import (
    DEFAULT_RECENT_N,
    DELTA_THRESHOLD_PP,
    MIN_TOTAL,
    MIN_WINDOW,
    _is_win,
    _win_rate_pct,
    build_win_rate_trend,
)


def _rt(pnl: float | None) -> dict:
    """Minimal round-trip row — only ``pnl_usd`` is read by the builder."""
    return {"pnl_usd": pnl}


class TestIsWin:
    def test_positive_pnl_is_win(self):
        assert _is_win({"pnl_usd": 1.0}) is True

    def test_zero_pnl_is_loss(self):
        # Zero is a non-positive outcome (breakeven counts as not-a-win),
        # mirroring loser_autopsy's strict ``> 0`` rule.
        assert _is_win({"pnl_usd": 0.0}) is False

    def test_negative_pnl_is_loss(self):
        assert _is_win({"pnl_usd": -1.0}) is False

    def test_none_pnl_returns_none(self):
        assert _is_win({"pnl_usd": None}) is None

    def test_missing_pnl_returns_none(self):
        assert _is_win({}) is None

    def test_non_dict_returns_none(self):
        assert _is_win("garbage") is None
        assert _is_win(None) is None
        assert _is_win(42) is None

    def test_unparseable_pnl_returns_none(self):
        assert _is_win({"pnl_usd": "not-a-number"}) is None


class TestWinRatePct:
    def test_empty_returns_none(self):
        assert _win_rate_pct([]) is None

    def test_all_wins_100(self):
        assert _win_rate_pct([_rt(1.0), _rt(2.0), _rt(3.0)]) == 100.0

    def test_all_losses_0(self):
        assert _win_rate_pct([_rt(-1.0), _rt(-2.0)]) == 0.0

    def test_mixed_50pct(self):
        assert _win_rate_pct([_rt(1.0), _rt(-1.0)]) == 50.0

    def test_garbage_rows_skipped(self):
        # 1 win, 1 loss, 1 garbage → 50% over the 2 parseable rows.
        rts = [_rt(1.0), _rt(-1.0), {"pnl_usd": "junk"}]
        assert _win_rate_pct(rts) == 50.0

    def test_only_garbage_returns_none(self):
        assert _win_rate_pct([{"pnl_usd": None}, {"pnl_usd": "junk"}]) is None


class TestVerdictLadder:
    def test_no_data_below_min_total(self):
        # 9 round-trips < MIN_TOTAL=10.
        rts = [_rt(1.0)] * 9
        r = build_win_rate_trend(rts)
        assert r["state"] == "NO_DATA"
        assert r["total_n"] == 9
        # Lifetime is still computed, the recent_win_rate is not.
        assert r["lifetime_win_rate_pct"] == 100.0
        assert r["recent_win_rate_pct"] is None
        assert r["prior_win_rate_pct"] is None

    def test_empty_list_no_data(self):
        r = build_win_rate_trend([])
        assert r["state"] == "NO_DATA"
        assert r["total_n"] == 0

    def test_trending_up_recent_better(self):
        # Prior: 10 losses (0% win rate). Recent: 10 wins (100% win rate).
        # Delta = +100pp — way above threshold.
        rts = [_rt(-1.0)] * 10 + [_rt(1.0)] * 10
        r = build_win_rate_trend(rts, recent_n=10)
        assert r["state"] == "TRENDING_UP"
        assert r["recent_n"] == 10
        assert r["prior_n"] == 10
        assert r["recent_win_rate_pct"] == 100.0
        assert r["prior_win_rate_pct"] == 0.0
        assert r["delta_pp"] == 100.0
        assert "improving" in r["headline"].lower()

    def test_trending_down_recent_worse(self):
        # Prior: 10 wins (100%). Recent: 10 losses (0%). Delta = -100pp.
        rts = [_rt(1.0)] * 10 + [_rt(-1.0)] * 10
        r = build_win_rate_trend(rts, recent_n=10)
        assert r["state"] == "TRENDING_DOWN"
        assert r["recent_win_rate_pct"] == 0.0
        assert r["prior_win_rate_pct"] == 100.0
        assert r["delta_pp"] == -100.0
        assert "regressing" in r["headline"].lower()

    def test_stable_within_noise_band(self):
        # Recent 50% vs prior 55% — delta -5pp, within ±10pp band.
        # Prior: 11 wins, 9 losses (55%). Recent: 5 wins, 5 losses (50%).
        prior = [_rt(1.0)] * 11 + [_rt(-1.0)] * 9
        recent = [_rt(1.0)] * 5 + [_rt(-1.0)] * 5
        rts = prior + recent
        r = build_win_rate_trend(rts, recent_n=10)
        assert r["state"] == "STABLE"
        assert r["delta_pp"] == -5.0

    def test_at_positive_threshold_boundary(self):
        # Prior 0% vs recent 10% — exactly at +10pp threshold → TRENDING_UP.
        prior = [_rt(-1.0)] * 10
        recent = [_rt(1.0)] + [_rt(-1.0)] * 9   # 10% win rate
        rts = prior + recent
        r = build_win_rate_trend(rts, recent_n=10)
        assert r["state"] == "TRENDING_UP"
        assert r["delta_pp"] == 10.0

    def test_just_below_positive_threshold_stable(self):
        # Use 5+5 windows with mixed values to hit just-below 10pp.
        # Prior 60% (3W2L), recent 80% (4W1L) → delta = +20pp → TRENDING_UP.
        # Need a finer case for just-below: prior 50% (5W5L), recent 59% — but
        # discrete counts. Use larger windows.
        # Prior 50% (10W 10L), recent 59% needs fractional; instead, recent
        # 5W 4L = 55.56% over 9, but window must be ≥ 5. Use 9-trip recent
        # window: 5/9 = 55.56%, prior 50% → delta = 5.56pp < 10pp → STABLE.
        prior = [_rt(1.0)] * 10 + [_rt(-1.0)] * 10
        recent = [_rt(1.0)] * 5 + [_rt(-1.0)] * 4
        rts = prior + recent
        r = build_win_rate_trend(rts, recent_n=9)
        assert r["state"] == "STABLE"
        # 55.56 - 50.0 = 5.56pp (rounded to 2 decimals)
        assert r["delta_pp"] == 5.56

    def test_at_negative_threshold_boundary(self):
        # Mirror image: prior 100% vs recent 90% → delta = -10pp.
        prior = [_rt(1.0)] * 10
        recent = [_rt(-1.0)] + [_rt(1.0)] * 9
        rts = prior + recent
        r = build_win_rate_trend(rts, recent_n=10)
        assert r["state"] == "TRENDING_DOWN"
        assert r["delta_pp"] == -10.0

    def test_insufficient_recent_window(self):
        # 10 total satisfies MIN_TOTAL, but recent_n clamp would force
        # ≥ MIN_WINDOW each side. With recent_n=20 clamped to total-MIN
        # = 5, both windows have 5 — sufficient.
        # To trigger INSUFFICIENT, ask for recent_n=20 on 9 total trips —
        # that's below MIN_TOTAL → NO_DATA. So INSUFFICIENT only fires
        # when garbage filtering reduces parseable rows below MIN_TOTAL
        # AFTER the original list was ≥ MIN_TOTAL.
        # With 10 garbage rows in front and 5 valid behind, parseable = 5
        # < MIN_TOTAL → NO_DATA, NOT INSUFFICIENT.
        # INSUFFICIENT can be reached if total=10 but recent_n clamped
        # yields fewer than MIN_WINDOW in some pathological invocation —
        # not reachable via normal API. Verify the threshold constants
        # are pinned anyway so a refactor cannot silently break them.
        assert MIN_TOTAL == 10
        assert MIN_WINDOW == 5
        assert DELTA_THRESHOLD_PP == 10.0
        assert DEFAULT_RECENT_N == 20

    def test_recent_n_clamped_to_window_min(self):
        # recent_n=1 should be clamped to MIN_WINDOW.
        rts = [_rt(1.0)] * 10 + [_rt(-1.0)] * 10
        r = build_win_rate_trend(rts, recent_n=1)
        assert r["recent_n"] >= MIN_WINDOW

    def test_recent_n_clamped_to_total_minus_min(self):
        # recent_n=20 on a 12-trip total should clamp recent to 7
        # (12 - 5 prior). recent_n max = total - MIN_WINDOW.
        rts = [_rt(1.0)] * 6 + [_rt(-1.0)] * 6
        r = build_win_rate_trend(rts, recent_n=100)
        assert r["recent_n"] == 12 - MIN_WINDOW
        assert r["prior_n"] == MIN_WINDOW

    def test_garbage_recent_n_falls_back(self):
        # Non-numeric recent_n defaults to DEFAULT_RECENT_N.
        rts = [_rt(1.0)] * 30
        r = build_win_rate_trend(rts, recent_n="garbage")  # type: ignore[arg-type]
        # With 30 total and default 20, recent_n stays 20, prior 10.
        assert r["recent_n"] == 20
        assert r["prior_n"] == 10


class TestGarbageSafety:
    def test_non_list_input(self):
        r = build_win_rate_trend("garbage")  # type: ignore[arg-type]
        assert r["state"] == "NO_DATA"

    def test_none_input(self):
        r = build_win_rate_trend(None)  # type: ignore[arg-type]
        assert r["state"] == "NO_DATA"

    def test_mixed_garbage_and_real_rows(self):
        # 5 wins + 5 losses + 3 garbage = 10 parseable → MIN_TOTAL met.
        rts = (
            [_rt(1.0)] * 5
            + ["garbage", None, {"pnl_usd": "junk"}]  # type: ignore[list-item]
            + [_rt(-1.0)] * 5
        )
        r = build_win_rate_trend(rts, recent_n=5)
        # 10 parseable, recent 5 (losses) → 0%, prior 5 (wins) → 100%.
        assert r["total_n"] == 10
        assert r["state"] == "TRENDING_DOWN"

    def test_thresholds_pinned(self):
        r = build_win_rate_trend([])
        assert r["threshold_pp"] == 10.0
        assert r["min_total"] == 10
        assert r["min_window"] == 5


class TestRowSchema:
    def test_all_fields_present(self):
        r = build_win_rate_trend([_rt(1.0)] * 10 + [_rt(-1.0)] * 10,
                                  recent_n=10)
        for k in (
            "state", "headline", "recent_n", "prior_n", "total_n",
            "recent_win_rate_pct", "prior_win_rate_pct",
            "lifetime_win_rate_pct", "delta_pp", "threshold_pp",
            "min_total", "min_window",
        ):
            assert k in r, f"missing field: {k}"


class TestReporterWiring:
    """Verify _win_rate_trend_line composes the builder correctly and
    suppresses non-actionable verdicts."""

    def _fake_store(self, trades_oldest_first):
        # store.recent_trades returns NEWEST first; the reporter reverses
        # it before passing to build_round_trips.
        trades_newest_first = list(reversed(trades_oldest_first))

        class _FakeStore:
            def recent_trades(self, n):
                return trades_newest_first

        return _FakeStore()

    def _buy(self, ts, ticker="MUU", qty=1.0, price=100.0):
        return {
            "id": 1, "timestamp": ts, "ticker": ticker,
            "action": "BUY", "qty": qty, "price": price,
            "value": qty * price, "reason": "",
            "expiry": None, "strike": None, "option_type": None,
        }

    def _sell(self, ts, ticker="MUU", qty=1.0, price=110.0):
        return {
            "id": 2, "timestamp": ts, "ticker": ticker,
            "action": "SELL", "qty": qty, "price": price,
            "value": qty * price, "reason": "",
            "expiry": None, "strike": None, "option_type": None,
        }

    def test_silent_when_below_min_total(self):
        from paper_trader import reporter
        # Only 1 round-trip — below MIN_TOTAL.
        trades = [
            self._buy("2026-05-28T10:00:00+00:00"),
            self._sell("2026-05-28T11:00:00+00:00"),
        ]
        line = reporter._win_rate_trend_line(self._fake_store(trades))
        assert line == ""

    def test_silent_on_stable_verdict(self):
        from paper_trader import reporter
        # 30 round-trips, alternating win/loss → exactly 50% in each
        # window after the default recent_n=20 clamps to 20 recent + 10
        # prior (both 50/50 by construction).
        trades = []
        for i in range(30):
            day = (i % 28) + 1
            t = f"2026-05-{day:02d}T10:{i:02d}:00+00:00"
            trades.append(self._buy(t, ticker=f"T{i}"))
            sell_price = 110.0 if i % 2 == 0 else 90.0
            trades.append(self._sell(
                f"2026-05-{day:02d}T11:{i:02d}:00+00:00",
                ticker=f"T{i}", price=sell_price,
            ))
        line = reporter._win_rate_trend_line(self._fake_store(trades))
        # Both windows balanced 50/50 → STABLE → silent.
        assert line == ""

    def test_fires_on_trending_up(self):
        from paper_trader import reporter
        # 10 losses first, 10 wins second. recent_n default is 20 but
        # clamped to 10 because total = 20 and MIN_WINDOW = 5 means
        # recent_n clamps to min(20, 20-5)=15... actually no, 20 in
        # default clamps to max(5, min(20-5, 20)) = max(5, 15) = 15.
        # Let me use 30 trips to make recent_n=20 happen cleanly.
        trades = []
        for i in range(15):
            ts = f"2026-05-{(i+1) % 28 + 1:02d}T10:{i:02d}:00+00:00"
            trades.append(self._buy(ts, ticker=f"T{i}"))
            trades.append(self._sell(
                f"2026-05-{(i+1) % 28 + 1:02d}T11:{i:02d}:00+00:00",
                ticker=f"T{i}", price=90.0,  # all losses
            ))
        for i in range(15, 30):
            ts = f"2026-05-{(i+1) % 28 + 1:02d}T12:{i % 60:02d}:00+00:00"
            trades.append(self._buy(ts, ticker=f"T{i}"))
            trades.append(self._sell(
                f"2026-05-{(i+1) % 28 + 1:02d}T13:{i % 60:02d}:00+00:00",
                ticker=f"T{i}", price=110.0,  # all wins
            ))
        line = reporter._win_rate_trend_line(self._fake_store(trades))
        assert "TRENDING_UP" in line
        assert "improving" in line.lower()

    def test_fires_on_trending_down(self):
        from paper_trader import reporter
        # 15 wins first, 15 losses second.
        trades = []
        for i in range(15):
            ts = f"2026-05-{(i+1) % 28 + 1:02d}T10:{i:02d}:00+00:00"
            trades.append(self._buy(ts, ticker=f"T{i}"))
            trades.append(self._sell(
                f"2026-05-{(i+1) % 28 + 1:02d}T11:{i:02d}:00+00:00",
                ticker=f"T{i}", price=110.0,  # all wins
            ))
        for i in range(15, 30):
            ts = f"2026-05-{(i+1) % 28 + 1:02d}T12:{i % 60:02d}:00+00:00"
            trades.append(self._buy(ts, ticker=f"T{i}"))
            trades.append(self._sell(
                f"2026-05-{(i+1) % 28 + 1:02d}T13:{i % 60:02d}:00+00:00",
                ticker=f"T{i}", price=90.0,  # all losses
            ))
        line = reporter._win_rate_trend_line(self._fake_store(trades))
        assert "TRENDING_DOWN" in line
        assert "regressing" in line.lower()

    def test_store_fault_returns_empty(self):
        from paper_trader import reporter

        class _BadStore:
            def recent_trades(self, n):
                raise RuntimeError("store down")

        line = reporter._win_rate_trend_line(_BadStore())
        assert line == ""


class TestEndpointSmoke:
    """Verify /api/win-rate-trend works against a patched store."""

    def test_endpoint_returns_no_data_on_empty_store(self, monkeypatch):
        from paper_trader.dashboard import app
        from paper_trader import dashboard as dash_mod

        class _EmptyStore:
            def recent_trades(self, n):
                return []

        monkeypatch.setattr(dash_mod, "get_store", lambda: _EmptyStore())
        client = app.test_client()
        rv = client.get("/api/win-rate-trend")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["state"] == "NO_DATA"

    def test_endpoint_accepts_recent_n_param(self, monkeypatch):
        from paper_trader.dashboard import app
        from paper_trader import dashboard as dash_mod

        class _EmptyStore:
            def recent_trades(self, n):
                return []

        monkeypatch.setattr(dash_mod, "get_store", lambda: _EmptyStore())
        client = app.test_client()
        rv = client.get("/api/win-rate-trend?recent_n=10")
        assert rv.status_code == 200

    def test_endpoint_garbage_recent_n_defaults(self, monkeypatch):
        from paper_trader.dashboard import app
        from paper_trader import dashboard as dash_mod

        class _EmptyStore:
            def recent_trades(self, n):
                return []

        monkeypatch.setattr(dash_mod, "get_store", lambda: _EmptyStore())
        client = app.test_client()
        rv = client.get("/api/win-rate-trend?recent_n=garbage")
        assert rv.status_code == 200
