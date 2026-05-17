"""Tests for the scorer-vs-Opus disagreement helpers in dashboard.py."""
from __future__ import annotations

import pytest

from paper_trader.dashboard import (
    _classify_disagreement,
    _concentration_severity,
    _parse_action_ticker,
)


class TestParseActionTicker:
    def test_buy_with_status(self):
        assert _parse_action_ticker("BUY NVDA → FILLED") == ("BUY", "NVDA")

    def test_hold_with_arrow(self):
        assert _parse_action_ticker("HOLD MU → HOLD") == ("HOLD", "MU")

    def test_sell_call_compound_verb(self):
        # The verb stays as one token; everything between verb and arrow
        # is treated as args. We currently only need (verb, ticker).
        assert _parse_action_ticker("SELL AMD → FILLED") == ("SELL", "AMD")

    def test_no_decision_sentinel(self):
        assert _parse_action_ticker("NO_DECISION") == ("NO_DECISION", None)

    def test_blocked_sentinel(self):
        assert _parse_action_ticker("BLOCKED") == ("BLOCKED", None)

    def test_empty_string(self):
        assert _parse_action_ticker("") == ("", None)

    def test_cash_pseudo_ticker_is_none(self):
        # 'HOLD CASH' appears in the DB but isn't a real ticker, so we
        # nullify so it doesn't show up as a position-level disagreement.
        verb, tk = _parse_action_ticker("HOLD CASH → HOLD")
        assert verb == "HOLD"
        assert tk is None

    def test_hold_none_is_none(self):
        verb, tk = _parse_action_ticker("HOLD NONE → HOLD")
        assert tk is None


class TestClassifyDisagreement:
    def test_scorer_exit_but_opus_holds_is_high(self):
        sev, label = _classify_disagreement("EXIT", "HOLD")
        assert sev == "HIGH"
        assert "exit" in label.lower()

    def test_scorer_trim_but_opus_buys_is_high(self):
        sev, _ = _classify_disagreement("TRIM", "BUY")
        assert sev == "HIGH"

    def test_scorer_strong_hold_but_opus_sells_is_medium(self):
        sev, _ = _classify_disagreement("STRONG_HOLD", "SELL")
        assert sev == "MEDIUM"

    def test_scorer_hold_but_opus_sells_is_medium(self):
        sev, _ = _classify_disagreement("HOLD", "SELL")
        assert sev == "MEDIUM"

    def test_scorer_neutral_but_opus_adds_is_medium(self):
        sev, _ = _classify_disagreement("NEUTRAL", "BUY")
        assert sev == "MEDIUM"

    def test_both_bullish_is_aligned(self):
        sev, _ = _classify_disagreement("STRONG_HOLD", "HOLD")
        assert sev == "ALIGNED"

    def test_both_bearish_is_aligned(self):
        sev, _ = _classify_disagreement("EXIT", "SELL")
        assert sev == "ALIGNED"

    def test_missing_action_is_aligned_by_default(self):
        # No action history for the ticker → can't claim disagreement.
        sev, _ = _classify_disagreement("EXIT", None)
        assert sev == "ALIGNED"

    def test_case_insensitive_verb(self):
        sev, _ = _classify_disagreement("EXIT", "hold")
        assert sev == "HIGH"

    def test_buy_call_treated_as_bullish(self):
        sev, _ = _classify_disagreement("EXIT", "BUY_CALL")
        assert sev == "HIGH"

    def test_sell_put_treated_as_bearish(self):
        # SELL_PUT is short-volatility but functionally adds long exposure —
        # however our verb-set classifies SELL_PUT as a bearish-direction
        # action. Document the current behavior; if the desk decides
        # SELL_PUT belongs in the bullish set later, this test will catch it.
        sev, _ = _classify_disagreement("STRONG_HOLD", "SELL_PUT")
        assert sev == "MEDIUM"


class TestConcentrationSeverity:
    def test_balanced_book_is_low(self):
        sev, warn = _concentration_severity(top1_pct=25.0, top3_pct=60.0)
        assert sev == "LOW"
        assert warn is False

    def test_top1_over_40_is_medium(self):
        sev, warn = _concentration_severity(top1_pct=45.0, top3_pct=60.0)
        assert sev == "MEDIUM"
        assert warn is True

    def test_top3_over_75_is_medium(self):
        sev, warn = _concentration_severity(top1_pct=30.0, top3_pct=80.0)
        assert sev == "MEDIUM"
        assert warn is True

    def test_top1_over_60_is_high(self):
        sev, warn = _concentration_severity(top1_pct=66.0, top3_pct=80.0)
        assert sev == "HIGH"
        assert warn is True

    def test_top3_over_90_is_high(self):
        sev, warn = _concentration_severity(top1_pct=35.0, top3_pct=92.0)
        assert sev == "HIGH"
        assert warn is True

    def test_high_beats_medium_when_both_thresholds_hit(self):
        # top1 ≥ 60 AND top3 ≥ 75 — we return HIGH, not MEDIUM.
        sev, _ = _concentration_severity(top1_pct=70.0, top3_pct=88.0)
        assert sev == "HIGH"

    def test_empty_book_is_low(self):
        sev, warn = _concentration_severity(top1_pct=0.0, top3_pct=0.0)
        assert sev == "LOW"
        assert warn is False
