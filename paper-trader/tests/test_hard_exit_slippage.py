"""Tests for analytics.hard_exit_slippage + /api/hard-exit-slippage.

Exercises the operator-facing calibration view that follows
``/api/hard-exit-summary``. Where the summary asks "is mechanical
discipline firing?", this asks "are the THRESHOLDS themselves
calibrated against realized volatility?".

Live evidence motivating the builder (2026-05-26): MU HARD_TP fired at
threshold $773.31 with a fill of $889.50 — +15% slippage past the
trigger. A "lucky overshoot" the threshold did not capture; the tape
did. The test corpus locks the verdict ladder against that case + the
"clean fills" baseline + edge cases (empty, insufficient, parse fail,
non-numeric).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.hard_exit_slippage import (
    HARD_SL_MARKER,
    HARD_TP_MARKER,
    LUCKY_TP_THRESHOLD_PCT,
    MIN_FOR_VERDICT,
    UNLUCKY_SL_THRESHOLD_PCT,
    _classify_exit_reason,
    _parse_threshold,
    _percentile,
    _slippage_pct,
    _verdict,
    build_hard_exit_slippage,
)


def _make_trade(
    *,
    action: str = "SELL",
    ticker: str = "NVDA",
    qty: float = 1.0,
    price: float = 100.0,
    reason: str = "",
    timestamp: str = "2026-05-26T12:00:00+00:00",
) -> dict:
    return {
        "action": action,
        "ticker": ticker,
        "qty": qty,
        "price": price,
        "value": qty * price,
        "reason": reason,
        "timestamp": timestamp,
    }


# ──────────────────────── _classify_exit_reason ────────────────────────


class TestClassifyExitReason:
    def test_hard_sl_marker(self):
        assert _classify_exit_reason(
            "HARD_SL: price 97.00 <= threshold 98.00"
        ) == "HARD_SL"

    def test_hard_tp_marker(self):
        assert _classify_exit_reason(
            "HARD_TP: price 103.00 >= threshold 100.00"
        ) == "HARD_TP"

    def test_discretionary_returns_none(self):
        assert _classify_exit_reason("Thesis broken; rotating") is None

    def test_empty_and_none(self):
        assert _classify_exit_reason("") is None
        assert _classify_exit_reason(None) is None

    def test_substring_safe(self):
        # SL marker anywhere in the reason still classifies — the live
        # emitter always writes "HARD_SL:" at the start, but the
        # substring discipline keeps the test surface honest.
        assert _classify_exit_reason("comment HARD_SL mid") == "HARD_SL"

    def test_sl_wins_when_both_markers_present(self):
        # Pathological string with both markers — SL checked first.
        assert _classify_exit_reason("HARD_SL after HARD_TP retry") == "HARD_SL"


# ──────────────────────────── _parse_threshold ────────────────────────────


class TestParseThreshold:
    def test_canonical_tp(self):
        # Live MU reason from 2026-05-26.
        price, thr = _parse_threshold(
            "HARD_TP: price 889.50 >= threshold 773.31"
        )
        assert price == 889.50
        assert thr == 773.31

    def test_canonical_sl(self):
        price, thr = _parse_threshold(
            "HARD_SL: price 97.00 <= threshold 98.00"
        )
        assert price == 97.0
        assert thr == 98.0

    def test_no_marker_returns_none(self):
        assert _parse_threshold("discretionary rotate") == (None, None)

    def test_empty_returns_none(self):
        assert _parse_threshold("") == (None, None)
        assert _parse_threshold(None) == (None, None)

    def test_malformed_returns_none(self):
        # Marker present, but threshold token missing.
        assert _parse_threshold("HARD_TP: fired") == (None, None)


# ──────────────────────────── _slippage_pct ─────────────────────────────


class TestSlippagePct:
    def test_tp_slippage_positive(self):
        # Fill above threshold = positive TP slippage.
        slip = _slippage_pct(fill_price=110.0, threshold=100.0, is_sl=False)
        assert slip == pytest.approx(10.0)

    def test_sl_slippage_positive(self):
        # Fill below threshold = positive SL slippage.
        slip = _slippage_pct(fill_price=90.0, threshold=100.0, is_sl=True)
        assert slip == pytest.approx(10.0)

    def test_at_threshold_zero(self):
        assert _slippage_pct(fill_price=100.0, threshold=100.0, is_sl=False) == 0.0
        assert _slippage_pct(fill_price=100.0, threshold=100.0, is_sl=True) == 0.0

    def test_live_mu_case(self):
        # MU TP fire 2026-05-26: fill 889.50, threshold 773.31.
        slip = _slippage_pct(
            fill_price=889.50, threshold=773.31, is_sl=False
        )
        # ((889.50 - 773.31) / 773.31) * 100 ≈ 15.03%
        assert slip == pytest.approx(15.025, abs=0.01)

    def test_zero_threshold_returns_none(self):
        assert _slippage_pct(
            fill_price=100.0, threshold=0.0, is_sl=False
        ) is None

    def test_negative_threshold_returns_none(self):
        assert _slippage_pct(
            fill_price=100.0, threshold=-50.0, is_sl=True
        ) is None

    def test_non_numeric_returns_none(self):
        assert _slippage_pct(
            fill_price="bad", threshold=100.0, is_sl=False
        ) is None


# ─────────────────────────────── _verdict ───────────────────────────────


class TestVerdict:
    def test_no_hard_exits(self):
        v, h = _verdict(n_tp=0, n_sl=0, tp_slip_median=None, sl_slip_median=None)
        assert v == "NO_HARD_EXITS"
        assert "untested" in h

    def test_insufficient_below_min(self):
        # Total = 2 < MIN_FOR_VERDICT (3).
        v, h = _verdict(n_tp=1, n_sl=1, tp_slip_median=10.0, sl_slip_median=5.0)
        assert v == "INSUFFICIENT"
        assert "Only 2" in h

    def test_lucky_overshoots_at_threshold(self):
        v, h = _verdict(
            n_tp=3, n_sl=0,
            tp_slip_median=LUCKY_TP_THRESHOLD_PCT, sl_slip_median=None,
        )
        assert v == "LUCKY_OVERSHOOTS"
        assert "TPs gapping" in h

    def test_unlucky_gaps_at_threshold(self):
        v, h = _verdict(
            n_tp=0, n_sl=3,
            tp_slip_median=None, sl_slip_median=UNLUCKY_SL_THRESHOLD_PCT,
        )
        assert v == "UNLUCKY_GAPS"
        assert "blowing through" in h

    def test_clean_fills(self):
        # Both well below threshold with sufficient sample.
        v, h = _verdict(
            n_tp=3, n_sl=3, tp_slip_median=0.5, sl_slip_median=0.5,
        )
        assert v == "CLEAN_FILLS"
        assert "Threshold settings holding" in h

    def test_lucky_wins_when_both_breach(self):
        # Both medians above their thresholds — LUCKY_OVERSHOOTS takes
        # precedence per the documented ladder. Lock that ordering.
        v, _ = _verdict(
            n_tp=3, n_sl=3,
            tp_slip_median=LUCKY_TP_THRESHOLD_PCT + 1.0,
            sl_slip_median=UNLUCKY_SL_THRESHOLD_PCT + 1.0,
        )
        assert v == "LUCKY_OVERSHOOTS"

    def test_lucky_requires_per_direction_min(self):
        # 5 hard exits total but only 2 TPs — LUCKY_OVERSHOOTS cannot
        # publish even with high TP slippage; needs 3+ per direction.
        v, _ = _verdict(
            n_tp=2, n_sl=3,
            tp_slip_median=10.0, sl_slip_median=0.5,
        )
        assert v == "CLEAN_FILLS"


# ──────────────────────────────── _percentile ────────────────────────────


class TestPercentile:
    def test_empty_returns_none(self):
        assert _percentile([], 0.5) is None

    def test_single_value(self):
        assert _percentile([7.5], 0.9) == 7.5

    def test_p50_two_values(self):
        # Linear interpolation between 0 and 10 at q=0.5 → 5.0.
        assert _percentile([0.0, 10.0], 0.5) == pytest.approx(5.0)

    def test_p90_ten_values(self):
        # Sorted 1..10, q=0.9 → idx = 8.1 → 9 + 0.1*(10-9) = 9.1.
        result = _percentile(list(range(1, 11)), 0.9)
        assert result == pytest.approx(9.1)

    def test_unsorted_input_handled(self):
        # The function sorts internally; unsorted should not break the
        # answer.
        assert _percentile([10.0, 0.0, 5.0], 0.5) == pytest.approx(5.0)


# ────────────────────────── build_hard_exit_slippage ────────────────────


class TestBuildHardExitSlippage:
    def test_empty_ledger_returns_no_hard_exits(self):
        snap = build_hard_exit_slippage([])
        assert snap["state"] == "OK"
        assert snap["verdict"] == "NO_HARD_EXITS"
        assert snap["n_hard_tp"] == 0
        assert snap["n_hard_sl"] == 0
        assert snap["n_total_hard"] == 0
        assert snap["tp_slippage_median_pct"] is None
        assert snap["sl_slippage_median_pct"] is None
        assert snap["last_hard_tp"] is None
        assert snap["last_hard_sl"] is None
        assert snap["per_ticker"] == []

    def test_buys_ignored(self):
        trades = [
            _make_trade(
                action="BUY", ticker="NVDA",
                reason="HARD_TP appears in buy reason text",
            ),
        ]
        snap = build_hard_exit_slippage(trades)
        assert snap["n_hard_tp"] == 0
        assert snap["n_hard_sl"] == 0

    def test_discretionary_sells_ignored(self):
        trades = [
            _make_trade(reason="Thesis broken — rotate to AMD"),
        ]
        snap = build_hard_exit_slippage(trades)
        assert snap["n_hard_tp"] == 0
        assert snap["n_hard_sl"] == 0
        assert snap["verdict"] == "NO_HARD_EXITS"

    def test_live_mu_tp_lucky_overshoot(self):
        # Exact live row from 2026-05-26 — locks the calibration view
        # against the very case the module was designed for. Three
        # copies because INSUFFICIENT below MIN_FOR_VERDICT.
        trades = [
            _make_trade(
                ticker="MU", qty=1.3, price=889.50,
                reason="HARD_TP: price 889.50 >= threshold 773.31",
                timestamp=f"2026-05-26T17:13:5{i}.000000+00:00",
            )
            for i in range(3)
        ]
        snap = build_hard_exit_slippage(trades)
        assert snap["state"] == "OK"
        assert snap["n_hard_tp"] == 3
        assert snap["n_hard_sl"] == 0
        # Median slippage is 15.03% — well above the 2.0% LUCKY threshold.
        assert snap["tp_slippage_median_pct"] == pytest.approx(15.025, abs=0.01)
        assert snap["verdict"] == "LUCKY_OVERSHOOTS"
        assert snap["last_hard_tp"]["ticker"] == "MU"
        assert snap["last_hard_tp"]["fill_price"] == pytest.approx(889.50)
        assert snap["last_hard_tp"]["threshold"] == pytest.approx(773.31)

    def test_clean_fills_at_threshold(self):
        # Fill price ≈ threshold → near-zero slippage → CLEAN_FILLS.
        trades = [
            _make_trade(
                ticker="NVDA", price=100.0,
                reason="HARD_TP: price 100.00 >= threshold 100.00",
                timestamp=f"2026-05-26T1{i}:00:00+00:00",
            )
            for i in range(3)
        ]
        snap = build_hard_exit_slippage(trades)
        assert snap["verdict"] == "CLEAN_FILLS"
        assert snap["tp_slippage_median_pct"] == pytest.approx(0.0)

    def test_unlucky_gaps_on_sl(self):
        # SL fill below threshold = positive slippage past the stop.
        trades = [
            _make_trade(
                ticker="AMD", price=95.0,
                reason="HARD_SL: price 95.00 <= threshold 100.00",
                timestamp=f"2026-05-26T1{i}:00:00+00:00",
            )
            for i in range(3)
        ]
        snap = build_hard_exit_slippage(trades)
        # ((100 - 95) / 100) * 100 = 5.0% — above 2.0% UNLUCKY threshold.
        assert snap["sl_slippage_median_pct"] == pytest.approx(5.0)
        assert snap["verdict"] == "UNLUCKY_GAPS"
        assert snap["last_hard_sl"]["ticker"] == "AMD"

    def test_insufficient_below_min(self):
        trades = [
            _make_trade(
                reason="HARD_TP: price 110.00 >= threshold 100.00",
            ),
        ]
        snap = build_hard_exit_slippage(trades)
        assert snap["n_hard_tp"] == 1
        assert snap["verdict"] == "INSUFFICIENT"
        # The lone trade is still recorded as the last TP.
        assert snap["last_hard_tp"] is not None

    def test_parse_failure_counted_but_does_not_crash(self):
        # The reason has the marker but no parseable numeric fields →
        # n_parse_failed increments, not counted toward n_hard_*.
        trades = [
            _make_trade(reason="HARD_TP: malformed (no numbers)"),
        ]
        snap = build_hard_exit_slippage(trades)
        assert snap["state"] == "OK"
        assert snap["n_parse_failed"] == 1
        assert snap["n_hard_tp"] == 0

    def test_per_ticker_breakdown_sorted_by_total(self):
        trades = [
            _make_trade(
                ticker="NVDA",
                reason="HARD_TP: price 110.00 >= threshold 100.00",
            ),
            _make_trade(
                ticker="NVDA",
                reason="HARD_SL: price 90.00 <= threshold 100.00",
            ),
            _make_trade(
                ticker="AMD",
                reason="HARD_TP: price 60.00 >= threshold 50.00",
            ),
        ]
        snap = build_hard_exit_slippage(trades)
        # NVDA: 2 fires (1 TP + 1 SL); AMD: 1 fire — NVDA must rank first.
        assert snap["per_ticker"][0]["ticker"] == "NVDA"
        assert snap["per_ticker"][0]["n_total"] == 2
        assert snap["per_ticker"][0]["n_tp"] == 1
        assert snap["per_ticker"][0]["n_sl"] == 1
        assert snap["per_ticker"][1]["ticker"] == "AMD"

    def test_last_hard_tp_picks_newest_with_newest_first_input(self):
        # Input ordering is store-native newest-first; the FIRST row in
        # iteration order is the newest.
        trades = [
            _make_trade(
                ticker="NEW",
                reason="HARD_TP: price 110.00 >= threshold 100.00",
                timestamp="2026-05-26T18:00:00+00:00",
            ),
            _make_trade(
                ticker="OLD",
                reason="HARD_TP: price 105.00 >= threshold 100.00",
                timestamp="2026-05-25T10:00:00+00:00",
            ),
        ]
        snap = build_hard_exit_slippage(trades)
        assert snap["last_hard_tp"]["ticker"] == "NEW"

    def test_p90_and_max_present_in_output(self):
        # Three rows with TP slippage 5%, 10%, 15% — p90 between 10–15,
        # max 15.
        trades = [
            _make_trade(
                reason=f"HARD_TP: price {p:.2f} >= threshold 100.00",
                timestamp=f"2026-05-26T1{i}:00:00+00:00",
            )
            for i, p in enumerate((105.0, 110.0, 115.0))
        ]
        snap = build_hard_exit_slippage(trades)
        assert snap["tp_slippage_max_pct"] == pytest.approx(15.0)
        assert snap["tp_slippage_p90_pct"] == pytest.approx(14.0, abs=0.01)

    def test_none_input_degrades_to_empty_not_error(self):
        # ``None`` is normalised to ``[]`` via the builder's
        # ``trades_newest_first or []`` guard — this is graceful empty
        # handling, NOT an error path. A trader querying the endpoint
        # before any trades exist must see NO_HARD_EXITS, not ERROR.
        snap = build_hard_exit_slippage(None)  # type: ignore[arg-type]
        assert snap["state"] == "OK"
        assert snap["verdict"] == "NO_HARD_EXITS"

    def test_state_error_on_iteration_fault(self):
        # The builder must never raise — a class that raises on
        # iteration degrades to an ERROR envelope so the endpoint never
        # 500s with a missing payload.
        class _ExplodingIterable:
            def __iter__(self):
                raise RuntimeError("simulated iteration fault")

        snap = build_hard_exit_slippage(_ExplodingIterable())  # type: ignore[arg-type]
        assert snap["state"] == "ERROR"
        assert snap["verdict"] == "ERROR"
        assert "simulated iteration fault" in snap["error"]
        # Field shape must match the success envelope so a UI binding
        # never sees missing keys.
        assert snap["n_hard_tp"] == 0
        assert snap["n_hard_sl"] == 0
        assert snap["per_ticker"] == []

    def test_thresholds_echoed_for_dashboard_consumer(self):
        # Dashboard renders the live thresholds beside the verdict so a
        # future tweak surfaces without re-deploying the panel.
        snap = build_hard_exit_slippage([])
        assert snap["lucky_tp_threshold_pct"] == LUCKY_TP_THRESHOLD_PCT
        assert snap["unlucky_sl_threshold_pct"] == UNLUCKY_SL_THRESHOLD_PCT
        assert snap["min_for_verdict"] == MIN_FOR_VERDICT

    def test_non_dict_trades_skipped(self):
        # Defensive: a stray non-dict row (a string, an int) must not
        # crash the builder — the live store has only ever returned
        # dicts, but the additive contract requires no-crash on garbage.
        trades = [
            "garbage",  # type: ignore[list-item]
            _make_trade(
                reason="HARD_TP: price 110.00 >= threshold 100.00",
            ),
        ]
        snap = build_hard_exit_slippage(trades)
        assert snap["state"] == "OK"
        assert snap["n_hard_tp"] == 1


# ──────────────────── /api/hard-exit-slippage endpoint ──────────────────


class TestHardExitSlippageEndpoint:
    def test_endpoint_returns_json_envelope(self, tmp_path, monkeypatch):
        # Build a minimal store fixture so we don't touch the live DB.
        from paper_trader import dashboard
        from paper_trader.store import Store

        class _StubStore:
            def __init__(self, trades):
                self._trades = trades

            def recent_trades(self, limit):
                return list(self._trades)

        stub = _StubStore(
            [
                _make_trade(
                    ticker="MU", price=889.50,
                    reason="HARD_TP: price 889.50 >= threshold 773.31",
                )
                for _ in range(3)
            ]
        )
        monkeypatch.setattr(dashboard, "get_store", lambda: stub)

        with dashboard.app.test_client() as client:
            r = client.get("/api/hard-exit-slippage")
        assert r.status_code == 200
        payload = r.get_json()
        assert payload["service"] == "paper_trader"
        assert payload["state"] == "OK"
        assert payload["verdict"] == "LUCKY_OVERSHOOTS"
        assert payload["n_hard_tp"] == 3

    def test_endpoint_degrades_to_error_envelope_on_store_fault(
        self, monkeypatch
    ):
        from paper_trader import dashboard

        class _BrokenStore:
            def recent_trades(self, limit):
                raise RuntimeError("simulated store fault")

        monkeypatch.setattr(dashboard, "get_store", lambda: _BrokenStore())
        with dashboard.app.test_client() as client:
            r = client.get("/api/hard-exit-slippage")
        # 500 with a valid envelope — the operator panel can render the
        # ERROR state instead of seeing an empty 5xx page.
        assert r.status_code == 500
        payload = r.get_json()
        assert payload["state"] == "ERROR"
        assert payload["verdict"] == "ERROR"
        assert "simulated store fault" in payload["error"]
        # Field shape must match the success envelope so a UI binding
        # never sees missing keys.
        assert payload["n_hard_tp"] == 0
        assert payload["n_hard_sl"] == 0
        assert payload["per_ticker"] == []

    def test_endpoint_empty_ledger(self, monkeypatch):
        from paper_trader import dashboard

        class _EmptyStore:
            def recent_trades(self, limit):
                return []

        monkeypatch.setattr(dashboard, "get_store", lambda: _EmptyStore())
        with dashboard.app.test_client() as client:
            r = client.get("/api/hard-exit-slippage")
        assert r.status_code == 200
        payload = r.get_json()
        assert payload["state"] == "OK"
        assert payload["verdict"] == "NO_HARD_EXITS"
