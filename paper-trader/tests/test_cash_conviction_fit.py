"""Tests for paper_trader.analytics.cash_conviction_fit.

Pins:
* the IDLE_DESPITE_SURGE × OVERDEPLOYED × IDLE_LOW_CONVICTION ×
  BALANCED × NO_DATA verdict matrix
* recent-fill disambiguation (active fill within recent_fill_max_min
  prevents IDLE_DESPITE_SURGE from firing — the idle reading is
  transient by construction in an active loop)
* threshold-override forwarding
* envelope key stability across every verdict
* defensive: malformed signals, no decision, missing fields all
  degrade — never raise
* verb extraction from the free-text decision string
  (`"BUY NVDA → FILLED"` → BUY)
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.cash_conviction_fit import (
    BALANCED,
    IDLE_DESPITE_SURGE,
    IDLE_LOW_CONVICTION,
    NO_DATA,
    OVERDEPLOYED,
    build_cash_conviction_fit,
)


def _now():
    return datetime(2026, 5, 21, 6, 30, 0, tzinfo=timezone.utc)


def _portfolio(cash, total_value, n_positions=1):
    return {
        "cash": cash,
        "total_value": total_value,
        "cash_pct": (cash / total_value * 100.0) if total_value else None,
        "n_positions": n_positions,
    }


def _signal(ticker, score, *, urgency=1, source="rss", held=False):
    return {
        "ticker": ticker, "ai_score": score, "urgency": urgency,
        "source": source, "held": held,
    }


def _decision(verb, *, age_min=10, ticker="NVDA"):
    ts = (_now() - timedelta(minutes=age_min)).isoformat()
    return {
        "timestamp": ts,
        "action_taken": f"{verb} {ticker} → FILLED" if verb in ("BUY", "SELL") else verb,
    }


_ENVELOPE_KEYS = {
    "as_of", "verdict", "headline", "portfolio",
    "top_signal", "last_decision", "thresholds",
}


class TestEnvelopeStability:
    def test_no_data_no_portfolio(self):
        out = build_cash_conviction_fit(None, [], None, now=_now())
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == NO_DATA

    def test_no_data_no_signals(self):
        out = build_cash_conviction_fit(
            _portfolio(500, 1000), [], None, now=_now(),
        )
        assert out["verdict"] == NO_DATA

    def test_balanced_envelope(self):
        # Cash 25% (between idle and overdeployed) ⇒ BALANCED.
        out = build_cash_conviction_fit(
            _portfolio(250, 1000), [_signal("AAPL", 9.0)],
            None, now=_now(),
        )
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == BALANCED

    def test_all_verdicts_emit_full_envelope(self):
        # Walk every verdict path and confirm structure.
        cases = [
            (None, [], None, NO_DATA),
            (_portfolio(500, 1000), [], None, NO_DATA),
            (_portfolio(500, 1000), [_signal("AAPL", 9.5)],
             _decision("HOLD"), IDLE_DESPITE_SURGE),
            (_portfolio(50, 1000), [_signal("AAPL", 9.5)],
             _decision("HOLD"), OVERDEPLOYED),
            (_portfolio(500, 1000), [_signal("AAPL", 4.0)],
             _decision("HOLD"), IDLE_LOW_CONVICTION),
            (_portfolio(250, 1000), [_signal("AAPL", 9.5)],
             _decision("HOLD"), BALANCED),
        ]
        for pf, sigs, dec, expected_verdict in cases:
            out = build_cash_conviction_fit(pf, sigs, dec, now=_now())
            assert set(out.keys()) >= _ENVELOPE_KEYS
            # last_decision and top_signal sub-dicts must always exist
            # (UI binding contract).
            assert isinstance(out["last_decision"], dict)
            assert isinstance(out["top_signal"], dict)
            assert isinstance(out["portfolio"], dict)
            assert isinstance(out["thresholds"], dict)
            assert out["verdict"] == expected_verdict


class TestIdleDespiteSurge:
    def test_high_cash_high_score_passive_decision(self):
        out = build_cash_conviction_fit(
            _portfolio(500, 1000),
            [_signal("NVDA", 9.5)],
            _decision("HOLD"),
            now=_now(),
        )
        assert out["verdict"] == IDLE_DESPITE_SURGE
        assert "9.5" in out["headline"]
        assert "NVDA" in out["headline"]
        assert out["top_signal"]["ticker"] == "NVDA"
        assert out["last_decision"]["verb"] == "HOLD"

    def test_no_decision_counts_as_passive(self):
        # "NO_DECISION" (no ticker, just the verb) — must still trigger.
        out = build_cash_conviction_fit(
            _portfolio(500, 1000),
            [_signal("NVDA", 8.5)],
            {"timestamp": (_now() - timedelta(minutes=5)).isoformat(),
             "action_taken": "NO_DECISION"},
            now=_now(),
        )
        assert out["verdict"] == IDLE_DESPITE_SURGE
        assert out["last_decision"]["verb"] == "NO_DECISION"

    def test_recent_fill_overrides_idle_despite_surge(self):
        # A FILL 5 minutes ago — the cash idleness is transient.
        out = build_cash_conviction_fit(
            _portfolio(500, 1000),
            [_signal("NVDA", 9.5)],
            _decision("BUY", age_min=5),
            now=_now(),
        )
        assert out["verdict"] == BALANCED
        assert out["last_decision"]["recent_fill"] is True

    def test_old_fill_does_not_override(self):
        # FILL 60 minutes ago — beyond the recent_fill_max_min (30m).
        out = build_cash_conviction_fit(
            _portfolio(500, 1000),
            [_signal("NVDA", 9.5)],
            _decision("BUY", age_min=60),
            now=_now(),
        )
        assert out["verdict"] == IDLE_DESPITE_SURGE
        assert out["last_decision"]["recent_fill"] is False

    def test_no_decision_history_treated_as_passive(self):
        # None of the disambiguation should kick in — verdict still
        # IDLE_DESPITE_SURGE when last_decision is None.
        out = build_cash_conviction_fit(
            _portfolio(500, 1000),
            [_signal("NVDA", 9.5)],
            None,
            now=_now(),
        )
        assert out["verdict"] == IDLE_DESPITE_SURGE
        assert out["last_decision"]["verb"] is None


class TestOverdeployed:
    def test_low_cash_high_score_overdeployed(self):
        out = build_cash_conviction_fit(
            _portfolio(50, 1000),
            [_signal("NVDA", 9.5)],
            _decision("HOLD"),
            now=_now(),
        )
        assert out["verdict"] == OVERDEPLOYED
        assert "50" not in out["headline"]  # we render percent, not dollars
        assert "5%" in out["headline"]

    def test_low_cash_low_score_balanced(self):
        # Low cash + low conviction ⇒ BALANCED (no add-headroom alarm
        # is warranted when nothing is screaming).
        out = build_cash_conviction_fit(
            _portfolio(50, 1000),
            [_signal("NVDA", 4.0)],
            _decision("HOLD"),
            now=_now(),
        )
        assert out["verdict"] == BALANCED


class TestIdleLowConviction:
    def test_high_cash_low_score_correctly_idle(self):
        out = build_cash_conviction_fit(
            _portfolio(500, 1000),
            [_signal("NVDA", 4.0)],
            _decision("HOLD"),
            now=_now(),
        )
        assert out["verdict"] == IDLE_LOW_CONVICTION
        assert "correct" in out["headline"]


class TestTopSignalPicking:
    def test_loudest_wins(self):
        out = build_cash_conviction_fit(
            _portfolio(500, 1000),
            [
                _signal("AAPL", 6.0),
                _signal("NVDA", 9.5),
                _signal("MSFT", 7.0),
            ],
            _decision("HOLD"),
            now=_now(),
        )
        assert out["top_signal"]["ticker"] == "NVDA"
        assert out["top_signal"]["ai_score"] == 9.5

    def test_tie_breaks_on_urgency(self):
        # Same score, different urgency — higher urgency wins.
        out = build_cash_conviction_fit(
            _portfolio(500, 1000),
            [
                _signal("AAPL", 9.0, urgency=1),
                _signal("NVDA", 9.0, urgency=2),
            ],
            _decision("HOLD"),
            now=_now(),
        )
        assert out["top_signal"]["ticker"] == "NVDA"

    def test_malformed_signals_skipped(self):
        out = build_cash_conviction_fit(
            _portfolio(500, 1000),
            [
                None,
                "garbage",
                {"ticker": "NVDA", "ai_score": "not-a-number"},
                {"ticker": "AAPL", "ai_score": float("nan")},
                _signal("MSFT", 8.0),
            ],
            _decision("HOLD"),
            now=_now(),
        )
        assert out["top_signal"]["ticker"] == "MSFT"


class TestThresholdOverrides:
    def test_idle_cash_threshold_override(self):
        # Default idle=40%; cash 30% would BALANCED at default but
        # IDLE_DESPITE_SURGE with idle=25%.
        out = build_cash_conviction_fit(
            _portfolio(300, 1000),
            [_signal("NVDA", 9.5)],
            _decision("HOLD"),
            now=_now(),
        )
        assert out["verdict"] == BALANCED

        out = build_cash_conviction_fit(
            _portfolio(300, 1000),
            [_signal("NVDA", 9.5)],
            _decision("HOLD"),
            now=_now(),
            idle_cash_pct=25.0,
        )
        assert out["verdict"] == IDLE_DESPITE_SURGE
        assert out["thresholds"]["idle_cash_pct"] == 25.0

    def test_high_conviction_threshold_override(self):
        # Default high=8.0; raise to 9.8 so a 9.5 no longer triggers.
        out = build_cash_conviction_fit(
            _portfolio(500, 1000),
            [_signal("NVDA", 9.5)],
            _decision("HOLD"),
            now=_now(),
            high_conviction_score=9.8,
        )
        # 500/1000 = 50% cash, top 9.5, high threshold 9.8 ⇒ neither
        # IDLE_DESPITE_SURGE nor IDLE_LOW_CONVICTION fires ⇒ BALANCED.
        assert out["verdict"] == BALANCED

    def test_recent_fill_threshold_override(self):
        # Push the window down to 1 minute — a 10-minute-old FILL is
        # now stale and IDLE_DESPITE_SURGE fires.
        out = build_cash_conviction_fit(
            _portfolio(500, 1000),
            [_signal("NVDA", 9.5)],
            _decision("BUY", age_min=10),
            now=_now(),
            recent_fill_max_min=1.0,
        )
        assert out["verdict"] == IDLE_DESPITE_SURGE


class TestVerbExtraction:
    def test_extracts_first_token(self):
        out = build_cash_conviction_fit(
            _portfolio(500, 1000),
            [_signal("NVDA", 9.5)],
            {"timestamp": (_now() - timedelta(minutes=5)).isoformat(),
             "action_taken": "HOLD NVDA → HOLD"},
            now=_now(),
        )
        assert out["last_decision"]["verb"] == "HOLD"
        # HOLD is a passive verb — so this is still IDLE_DESPITE_SURGE
        # (the fill_actions check looks for fill verbs, not HOLD).
        assert out["verdict"] == IDLE_DESPITE_SURGE

    def test_handles_empty_action_taken(self):
        out = build_cash_conviction_fit(
            _portfolio(500, 1000),
            [_signal("NVDA", 9.5)],
            {"timestamp": _now().isoformat(), "action_taken": ""},
            now=_now(),
        )
        assert out["last_decision"]["verb"] is None
        assert out["verdict"] == IDLE_DESPITE_SURGE


class TestPortfolioNormalisation:
    def test_recomputes_cash_pct_when_absent(self):
        out = build_cash_conviction_fit(
            {"cash": 250.0, "total_value": 1000.0, "n_positions": 1},
            [_signal("NVDA", 7.0)],
            _decision("HOLD"),
            now=_now(),
        )
        assert out["portfolio"]["cash_pct"] == 25.0

    def test_uses_explicit_cash_pct_when_given(self):
        # Explicit cash_pct wins over the implicit derive — useful for
        # the live trader's portfolio_snapshot shape where cash_pct
        # ships pre-computed.
        out = build_cash_conviction_fit(
            {"cash": 250.0, "total_value": 1000.0, "cash_pct": 99.9,
             "n_positions": 1},
            [_signal("NVDA", 4.0)],
            _decision("HOLD"),
            now=_now(),
        )
        assert out["portfolio"]["cash_pct"] == 99.9
        # 99.9% cash + score 4.0 ⇒ IDLE_LOW_CONVICTION
        assert out["verdict"] == IDLE_LOW_CONVICTION
