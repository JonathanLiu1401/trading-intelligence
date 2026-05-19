"""Unit tests for analytics/shadow_vs_claude.py — the right-now snapshot
comparing the deterministic shadow rec (from /api/suggestions) to the most
recent Claude decision.

These are pure-function tests (no Flask, no DB, no I/O), so they run fast
and don't depend on the live trader state."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics.shadow_vs_claude import (
    DIRECTIONAL_ACTIONS,
    STRONG_CONVICTION,
    _classify_claude_action,
    _claude_ticker,
    _top_directional,
    build_shadow_vs_claude,
)


NOW = datetime(2026, 5, 19, 2, 30, tzinfo=timezone.utc)


def _claude(action_taken, ts_minutes_ago=5, confidence=0.7):
    return {
        "action_taken": action_taken,
        "timestamp": (NOW - timedelta(minutes=ts_minutes_ago)).isoformat(),
        "confidence": confidence,
        "reasoning": "test",
    }


def _shadow_row(action, ticker, conviction, **extra):
    base = {
        "action": action,
        "ticker": ticker,
        "conviction": conviction,
        "reasons": ["URGENT news"],
        "rsi": 60.0,
        "macd": "bullish",
        "news_urgent": True,
        "news_max_score": 8.0,
        "top_headline": "Test headline",
    }
    base.update(extra)
    return base


# ─────────────────────── action classification ────────────────────────────


def test_classify_claude_action_buckets_no_decision():
    assert _classify_claude_action("NO_DECISION") == "NO_DECISION"


def test_classify_claude_action_extracts_verb_from_filled_action():
    assert _classify_claude_action("BUY NVDA → FILLED") == "BUY"


def test_classify_claude_action_extracts_verb_from_hold():
    assert _classify_claude_action("HOLD NVDA → HOLD") == "HOLD"


def test_classify_claude_action_blocked_is_its_own_bucket():
    assert _classify_claude_action("BLOCKED: oversell") == "BLOCKED"


def test_classify_claude_action_handles_none():
    assert _classify_claude_action(None) == "UNKNOWN"
    assert _classify_claude_action("") == "UNKNOWN"


def test_claude_ticker_extracts_from_buy_filled():
    assert _claude_ticker("BUY NVDA → FILLED") == "NVDA"


def test_claude_ticker_returns_none_for_no_decision():
    assert _claude_ticker("NO_DECISION") is None


def test_claude_ticker_returns_none_for_cash_pseudo_ticker():
    # CASH/NONE pseudo-tickers must not leak into the alignment check.
    assert _claude_ticker("HOLD CASH → HOLD") is None
    assert _claude_ticker("HOLD NONE → HOLD") is None


# ─────────────────────── top directional pick ─────────────────────────────


def test_top_directional_returns_none_on_only_hold_watch():
    suggestions = [
        _shadow_row("HOLD", "NVDA", 0.4),
        _shadow_row("WATCH", "AMD", 0.55),
    ]
    assert _top_directional(suggestions) is None


def test_top_directional_picks_highest_conviction_directional():
    suggestions = [
        _shadow_row("BUY", "MU", 0.84),
        _shadow_row("BUY", "AMD", 0.60),
        _shadow_row("WATCH", "LITE", 0.95),  # higher conv but not directional
    ]
    top = _top_directional(suggestions)
    assert top is not None
    assert top["ticker"] == "MU"
    assert top["action"] == "BUY"


def test_top_directional_includes_all_directional_actions():
    # DIRECTIONAL_ACTIONS contract: BUY/ADD/TRIM/EXIT.
    for act in ["BUY", "ADD", "TRIM", "EXIT"]:
        assert act in DIRECTIONAL_ACTIONS
    for act in ["HOLD", "WATCH"]:
        assert act not in DIRECTIONAL_ACTIONS


# ─────────────────────── verdict ladder ───────────────────────────────────


def test_missed_opportunity_when_no_decision_and_strong_shadow_buy():
    # Live scenario as of 2026-05-19: 72% NO_DECISION rate while shadow has
    # MU BUY conv=0.84 (this is from the real /api/suggestions response).
    shadow = [_shadow_row("BUY", "MU", 0.84)]
    claude = _claude("NO_DECISION", ts_minutes_ago=5)
    r = build_shadow_vs_claude(shadow, claude, now=NOW)
    assert r["verdict"] == "MISSED_OPPORTUNITY"
    assert r["shadow"]["strong"] is True
    assert r["shadow"]["ticker"] == "MU"
    assert "MISSED_OPPORTUNITY" in r["headline"]
    assert "MU" in r["headline"]


def test_drought_ok_when_no_decision_but_shadow_quiet():
    # Claude NO_DECISION but rules engine also has no directional rec.
    shadow = [
        _shadow_row("HOLD", "NVDA", 0.4),
        _shadow_row("WATCH", "AMD", 0.55),
    ]
    claude = _claude("NO_DECISION")
    r = build_shadow_vs_claude(shadow, claude, now=NOW)
    assert r["verdict"] == "DROUGHT_OK"


def test_drought_ok_when_shadow_directional_but_weak():
    # Conviction below STRONG_CONVICTION should NOT trigger MISSED_OPPORTUNITY.
    shadow = [_shadow_row("BUY", "MU", STRONG_CONVICTION - 0.01)]
    claude = _claude("NO_DECISION")
    r = build_shadow_vs_claude(shadow, claude, now=NOW)
    assert r["verdict"] == "DROUGHT_OK"


def test_aligned_when_both_long_same_ticker():
    shadow = [_shadow_row("ADD", "NVDA", 0.8)]
    claude = _claude("BUY NVDA → FILLED")
    r = build_shadow_vs_claude(shadow, claude, now=NOW)
    assert r["verdict"] == "ALIGNED"
    assert r["aligned"] is True


def test_divergent_when_both_directional_but_different_ticker():
    shadow = [_shadow_row("BUY", "MU", 0.84)]
    claude = _claude("BUY NVDA → FILLED")
    r = build_shadow_vs_claude(shadow, claude, now=NOW)
    assert r["verdict"] == "DIVERGENT"
    assert r["aligned"] is False


def test_divergent_when_opposite_verbs_on_same_ticker():
    # Claude SELL but shadow says BUY — clear divergence.
    shadow = [_shadow_row("BUY", "NVDA", 0.8)]
    claude = _claude("SELL NVDA → FILLED")
    r = build_shadow_vs_claude(shadow, claude, now=NOW)
    assert r["verdict"] == "DIVERGENT"
    assert r["aligned"] is False


def test_claude_holds_while_shadow_signals():
    shadow = [_shadow_row("BUY", "MU", 0.84)]
    claude = _claude("HOLD NVDA → HOLD")
    r = build_shadow_vs_claude(shadow, claude, now=NOW)
    assert r["verdict"] == "CLAUDE_HOLDS"
    # CLAUDE_HOLDS deliberately doesn't set aligned True/False — the holds
    # action isn't directional, so we make no alignment claim.


def test_no_claude_data_when_last_decision_missing():
    shadow = [_shadow_row("BUY", "MU", 0.84)]
    r = build_shadow_vs_claude(shadow, None, now=NOW)
    assert r["verdict"] == "NO_CLAUDE_DATA"


def test_no_shadow_data_when_suggestions_empty_and_claude_acted():
    claude = _claude("BUY NVDA → FILLED")
    r = build_shadow_vs_claude([], claude, now=NOW)
    assert r["verdict"] == "NO_SHADOW_DATA"


def test_drought_ok_when_both_sides_quiet():
    r = build_shadow_vs_claude([], _claude("NO_DECISION"), now=NOW)
    assert r["verdict"] == "DROUGHT_OK"


# ─────────────────────── snapshot shape contract ──────────────────────────


def test_snapshot_carries_all_required_top_level_keys():
    r = build_shadow_vs_claude(
        [_shadow_row("BUY", "MU", 0.84)],
        _claude("NO_DECISION"),
        now=NOW,
    )
    for key in ("as_of", "shadow", "claude", "aligned", "verdict", "headline"):
        assert key in r, f"missing top-level key {key!r}"


def test_shadow_payload_carries_strong_flag_and_reasons():
    r = build_shadow_vs_claude(
        [_shadow_row("BUY", "MU", 0.84)],
        _claude("NO_DECISION"),
        now=NOW,
    )
    assert r["shadow"]["strong"] is True
    assert isinstance(r["shadow"]["reasons"], list)
    assert r["shadow"]["reasons"]  # non-empty
    # Conviction normalised, ticker upper-cased.
    assert r["shadow"]["ticker"] == "MU"


def test_claude_payload_includes_minutes_ago():
    r = build_shadow_vs_claude(
        [_shadow_row("BUY", "MU", 0.84)],
        _claude("BUY NVDA → FILLED", ts_minutes_ago=12),
        now=NOW,
    )
    assert r["claude"]["minutes_ago"] == pytest.approx(12.0, abs=0.1)


def test_invalid_timestamp_does_not_raise():
    # Per the build contract: never raises. A junk timestamp degrades to None.
    bad = {"action_taken": "NO_DECISION", "timestamp": "not-a-date", "confidence": None}
    r = build_shadow_vs_claude(
        [_shadow_row("BUY", "MU", 0.84)],
        bad,
        now=NOW,
    )
    assert r["claude"]["minutes_ago"] is None
    assert r["verdict"] == "MISSED_OPPORTUNITY"


def test_naive_timestamp_treated_as_utc():
    # Some store rows are naive ISO strings; the builder must coerce to UTC.
    naive = {
        "action_taken": "NO_DECISION",
        "timestamp": (NOW - timedelta(minutes=8)).replace(tzinfo=None).isoformat(),
        "confidence": None,
    }
    r = build_shadow_vs_claude(
        [_shadow_row("BUY", "MU", 0.84)],
        naive,
        now=NOW,
    )
    assert r["claude"]["minutes_ago"] == pytest.approx(8.0, abs=0.1)


def test_buy_add_treated_as_equivalent_long_for_alignment():
    # _equiv: BUY ≡ ADD on the same name should be ALIGNED, not DIVERGENT.
    shadow = [_shadow_row("ADD", "NVDA", 0.8)]
    claude = _claude("BUY NVDA → FILLED")
    r = build_shadow_vs_claude(shadow, claude, now=NOW)
    assert r["verdict"] == "ALIGNED"


def test_sell_trim_exit_treated_as_equivalent_down_for_alignment():
    # Claude SELL ≡ shadow TRIM on the same name → ALIGNED.
    shadow = [_shadow_row("TRIM", "NVDA", 0.7)]
    claude = _claude("SELL NVDA → FILLED")
    r = build_shadow_vs_claude(shadow, claude, now=NOW)
    assert r["verdict"] == "ALIGNED"


def test_builder_never_raises_on_garbage_inputs():
    # Contract: pure function, never raises.
    garbage = {"action_taken": object(), "timestamp": 12345, "confidence": "nope"}
    r = build_shadow_vs_claude(
        [{"action": None, "ticker": None, "conviction": "bad"}],
        garbage,
        now=NOW,
    )
    assert "verdict" in r
