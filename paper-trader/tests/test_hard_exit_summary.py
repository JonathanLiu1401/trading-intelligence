"""Tests for analytics.hard_exit_summary + /api/hard-exit-summary.

Exercises the operator-facing aggregation of the freshly-landed (commit
3176d2f) hard SL/TP enforcement. Locks both the pure builder's verdict
ladder and the dashboard endpoint's envelope shape + caching semantics."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.hard_exit_summary import (
    HARD_SL_MARKER,
    HARD_TP_MARKER,
    MIN_FOR_VERDICT,
    _classify_exit_reason,
    _verdict,
    build_hard_exit_summary,
)


def _make_trade(
    *,
    action: str = "SELL",
    ticker: str = "NVDA",
    qty: float = 1.0,
    price: float = 100.0,
    value: float | None = None,
    reason: str = "",
    timestamp: str = "2026-05-24T12:00:00+00:00",
    option_type: str | None = None,
    strike: float | None = None,
    expiry: str | None = None,
) -> dict:
    return {
        "action": action,
        "ticker": ticker,
        "qty": qty,
        "price": price,
        "value": value if value is not None else qty * price,
        "reason": reason,
        "timestamp": timestamp,
        "option_type": option_type,
        "strike": strike,
        "expiry": expiry,
    }


# ──────────────────────────── _classify_exit_reason ────────────────────────────


class TestClassifyExitReason:
    def test_hard_sl_marker(self):
        assert _classify_exit_reason(
            "HARD_SL: price 97.00 <= threshold 98.00"
        ) == "HARD_SL"

    def test_hard_tp_marker(self):
        assert _classify_exit_reason(
            "HARD_TP: price 103.00 >= threshold 103.00"
        ) == "HARD_TP"

    def test_discretionary_reason_returns_none(self):
        assert _classify_exit_reason(
            "Thesis broken; rotating to AMD"
        ) is None

    def test_empty_string(self):
        assert _classify_exit_reason("") is None

    def test_none(self):
        assert _classify_exit_reason(None) is None

    def test_markers_are_substring_safe(self):
        """Defensive: a reason that contains the marker anywhere should
        still classify correctly."""
        assert _classify_exit_reason(
            "trailing comment HARD_SL appears mid-string"
        ) == "HARD_SL"


# ──────────────────────────────── _verdict ────────────────────────────────


class TestVerdict:
    def test_no_hard_exits(self):
        v, h = _verdict(0, 0)
        assert v == "NO_HARD_EXITS"
        assert "armed but untested" in h

    def test_insufficient_below_min(self):
        v, h = _verdict(1, 1)
        assert v == "INSUFFICIENT"
        assert "Only 2" in h

    def test_tp_heavy(self):
        # 5 TP, 1 SL → TP share 5/6 = 0.833 >= 0.65
        v, h = _verdict(1, 5)
        assert v == "TP_HEAVY"
        assert "Winners dominate" in h

    def test_sl_heavy(self):
        # 5 SL, 1 TP → TP share 1/6 = 0.167 <= 0.35
        v, h = _verdict(5, 1)
        assert v == "SL_HEAVY"
        assert "Losers dominate" in h

    def test_balanced(self):
        # 2 SL, 2 TP → TP share 0.5
        v, h = _verdict(2, 2)
        assert v == "BALANCED"
        assert "Mixed" in h

    def test_min_for_verdict_inclusive(self):
        """At MIN_FOR_VERDICT exact the verdict promotes — locks the
        boundary so a future tweak to MIN_FOR_VERDICT is testable."""
        # 0 SL, 3 TP → 3 total >= MIN_FOR_VERDICT (3) → tp_share=1.0 → TP_HEAVY
        v, _ = _verdict(0, MIN_FOR_VERDICT)
        assert v == "TP_HEAVY"


# ────────────────────────── build_hard_exit_summary ──────────────────────────


class TestBuildHardExitSummary:
    def test_empty_ledger_returns_no_hard_exits_state(self):
        snap = build_hard_exit_summary([])
        assert snap["state"] == "OK"
        assert snap["verdict"] == "NO_HARD_EXITS"
        assert snap["n_hard_sl"] == 0
        assert snap["n_hard_tp"] == 0
        assert snap["n_discretionary_sells"] == 0
        assert snap["realized_sl_usd"] == 0.0
        assert snap["realized_tp_usd"] == 0.0
        assert snap["discipline_ratio"] is None
        assert snap["mechanical_share"] is None
        assert snap["last_hard_sl"] is None
        assert snap["last_hard_tp"] is None
        assert snap["top_tickers"] == []

    def test_buys_are_ignored(self):
        trades = [
            _make_trade(action="BUY", ticker="NVDA",
                        reason="strong setup"),
            _make_trade(action="BUY", ticker="AMD",
                        reason="HARD_SL appears in BUY reason text"),
        ]
        snap = build_hard_exit_summary(trades)
        assert snap["n_hard_sl"] == 0
        assert snap["n_hard_tp"] == 0
        assert snap["n_discretionary_sells"] == 0

    def test_one_hard_sl_counts_and_credits_notional(self):
        trades = [
            _make_trade(
                action="SELL", ticker="NVDA", qty=2.0, price=97.0,
                reason="HARD_SL: price 97.00 <= threshold 98.00",
            ),
        ]
        snap = build_hard_exit_summary(trades)
        assert snap["n_hard_sl"] == 1
        assert snap["n_hard_tp"] == 0
        assert snap["realized_sl_usd"] == pytest.approx(2 * 97.0)
        assert snap["net_hard_notional_usd"] == pytest.approx(-194.0)
        assert snap["verdict"] == "INSUFFICIENT"  # < MIN_FOR_VERDICT
        assert snap["last_hard_sl"]["ticker"] == "NVDA"
        assert snap["last_hard_tp"] is None

    def test_one_hard_tp(self):
        trades = [
            _make_trade(
                action="SELL", ticker="AMD", qty=3.0, price=51.5,
                reason="HARD_TP: price 51.50 >= threshold 51.50",
            ),
        ]
        snap = build_hard_exit_summary(trades)
        assert snap["n_hard_tp"] == 1
        assert snap["realized_tp_usd"] == pytest.approx(3 * 51.5)
        assert snap["last_hard_tp"]["ticker"] == "AMD"

    def test_tp_heavy_verdict_at_threshold(self):
        # 5 TPs, 0 SLs → tp_share=1.0 ≥ 0.65 → TP_HEAVY
        trades = [
            _make_trade(reason="HARD_TP: ...") for _ in range(5)
        ]
        snap = build_hard_exit_summary(trades)
        assert snap["verdict"] == "TP_HEAVY"
        assert snap["discipline_ratio"] == 1.0

    def test_sl_heavy_verdict(self):
        trades = [
            _make_trade(reason="HARD_SL: ...") for _ in range(5)
        ]
        snap = build_hard_exit_summary(trades)
        assert snap["verdict"] == "SL_HEAVY"
        assert snap["discipline_ratio"] == 0.0

    def test_balanced_verdict(self):
        # 3 SL, 3 TP → tp_share=0.5 → BALANCED
        trades = [
            _make_trade(reason=f"HARD_SL: lot {i}") for i in range(3)
        ] + [
            _make_trade(reason=f"HARD_TP: lot {i}") for i in range(3)
        ]
        snap = build_hard_exit_summary(trades)
        assert snap["verdict"] == "BALANCED"
        assert snap["discipline_ratio"] == 0.5

    def test_mechanical_share_with_discretionary_mix(self):
        # 2 hard exits + 3 discretionary SELLs → mechanical_share = 2/5 = 0.4
        trades = [
            _make_trade(reason="HARD_SL: x"),
            _make_trade(reason="HARD_TP: x"),
            _make_trade(reason="discretionary rotate 1"),
            _make_trade(reason="discretionary rotate 2"),
            _make_trade(reason="discretionary rotate 3"),
        ]
        snap = build_hard_exit_summary(trades)
        assert snap["n_total_hard"] == 2
        assert snap["n_discretionary_sells"] == 3
        assert snap["mechanical_share"] == pytest.approx(0.4)

    def test_last_hard_sl_picks_newest_with_newest_first_input(self):
        """Input is newest-first (the store-native order), so the FIRST
        matching row in iteration order is the newest hard SL."""
        trades = [
            _make_trade(reason="HARD_SL: newest", ticker="NEW",
                        timestamp="2026-05-24T14:00:00+00:00"),
            _make_trade(reason="HARD_SL: older", ticker="OLD",
                        timestamp="2026-05-24T10:00:00+00:00"),
        ]
        snap = build_hard_exit_summary(trades)
        assert snap["last_hard_sl"]["ticker"] == "NEW"

    def test_per_ticker_breakdown_sorted_by_total(self):
        trades = [
            _make_trade(ticker="NVDA", reason="HARD_SL: x"),
            _make_trade(ticker="NVDA", reason="HARD_SL: x"),
            _make_trade(ticker="NVDA", reason="HARD_TP: x"),
            _make_trade(ticker="AMD", reason="HARD_SL: x"),
        ]
        snap = build_hard_exit_summary(trades)
        # NVDA: 3 total (2 SL + 1 TP), AMD: 1 total
        assert snap["top_tickers"][0]["ticker"] == "NVDA"
        assert snap["top_tickers"][0]["n_sl"] == 2
        assert snap["top_tickers"][0]["n_tp"] == 1
        assert snap["top_tickers"][0]["n_total"] == 3
        assert snap["top_tickers"][1]["ticker"] == "AMD"

    def test_non_numeric_value_degrades_to_zero(self):
        trades = [
            _make_trade(reason="HARD_SL: x", value=None),
            _make_trade(reason="HARD_SL: x", value="garbage"),
            _make_trade(reason="HARD_SL: x", value=50.0),
        ]
        # The first uses default qty*price = 100, the second degrades to 0
        # via the try/except, the third is 50. Total = 150.
        snap = build_hard_exit_summary(trades)
        assert snap["realized_sl_usd"] == pytest.approx(150.0)

    def test_option_sells_skipped_when_no_marker(self):
        """An option SELL without a HARD marker is still a discretionary
        sell (no special handling for options in this builder — the hard
        exits are stock-only by store contract, but counting an option
        discretionary sell is fine)."""
        trades = [
            _make_trade(action="SELL_CALL", option_type="call",
                        strike=200.0, expiry="2026-06-15",
                        reason="rotate to puts"),
        ]
        snap = build_hard_exit_summary(trades)
        # SELL_CALL starts with "SELL" → discretionary counter
        assert snap["n_discretionary_sells"] == 1

    def test_state_ok_on_normal_run(self):
        snap = build_hard_exit_summary([])
        assert snap["state"] == "OK"

    def test_state_error_on_exception(self):
        """The builder must never raise — a malformed input degrades to
        an ERROR envelope."""
        # Passing something that's not iterable triggers the except path.
        snap = build_hard_exit_summary(None)  # type: ignore[arg-type]
        assert snap["state"] == "ERROR"
        assert snap["verdict"] == "ERROR"
        assert "headline" in snap

    def test_thresholds_echoed(self):
        """Trader / dashboard consumer reads the live thresholds — echo
        them so a future tweak surfaces on the panel without re-deploy."""
        snap = build_hard_exit_summary([])
        assert snap["tp_heavy_threshold"] == 0.65
        assert snap["sl_heavy_threshold"] == 0.35
        assert snap["min_for_verdict"] == MIN_FOR_VERDICT

    def test_as_of_uses_injected_now(self):
        fixed = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)
        snap = build_hard_exit_summary([], now=fixed)
        assert snap["as_of"].startswith("2026-05-24T12:00:00")


# ───────────────────────── dashboard /api/hard-exit-summary ─────────────────────────


class TestHardExitSummaryEndpoint:
    """Smoke-test the dashboard endpoint to lock the envelope shape."""

    def test_endpoint_returns_valid_envelope(self, tmp_path, monkeypatch):
        from paper_trader import dashboard
        from paper_trader import store as store_mod
        from paper_trader.store import Store

        # Isolated store with no trades — endpoint must return NO_HARD_EXITS
        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        # Ensure the dashboard uses the freshly-built store
        s = Store()
        monkeypatch.setattr(store_mod, "_singleton", s)
        try:
            client = dashboard.app.test_client()
            r = client.get("/api/hard-exit-summary")
            assert r.status_code == 200
            body = r.get_json()
            assert body["service"] == "paper_trader"
            assert body["state"] == "OK"
            assert body["verdict"] == "NO_HARD_EXITS"
            assert body["n_hard_sl"] == 0
            assert body["n_hard_tp"] == 0
        finally:
            s.close()
            monkeypatch.setattr(store_mod, "_singleton", None)
