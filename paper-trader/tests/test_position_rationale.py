"""Tests for paper_trader.analytics.position_rationale.

Locks the most-recent-rationale-per-ticker contract and exercises the
edges that would silently break the trader-facing answer: garbage
reasoning JSON, NO_DECISION/BLOCKED filtering, ordering when a position
has no rationale at all, the 600-char cap, and the (verb, ticker) parse
shared with position_attention.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics.position_rationale import (
    build_position_rationale,
    _extract_decision,
    _MAX_REASON_CHARS,
)


NOW = datetime(2026, 5, 20, 10, 0, 0, tzinfo=timezone.utc)


def _ts(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _envelope(action: str, reasoning: str, confidence: float) -> str:
    return json.dumps({
        "decision": {
            "action": action,
            "ticker": "NVDA",
            "confidence": confidence,
            "reasoning": reasoning,
        },
        "detail": "ok",
        "fallback_used": False,
    })


class TestMostRecentPerTicker:
    def test_returns_most_recent_decision_per_position(self):
        positions = [{"ticker": "NVDA", "type": "stock", "qty": 3.0,
                      "opened_at": _ts(48.0)}]
        decisions = [
            # newest first — what store.recent_decisions returns
            {"timestamp": _ts(1.0),
             "action_taken": "HOLD NVDA → HOLD",
             "reasoning": _envelope("HOLD",
                                     "earnings in 0.8d, hold through print",
                                     0.7)},
            {"timestamp": _ts(2.0),
             "action_taken": "HOLD NVDA → HOLD",
             "reasoning": _envelope("HOLD", "older reasoning", 0.6)},
        ]
        rep = build_position_rationale(positions, decisions, now=NOW)
        assert len(rep["positions"]) == 1
        r = rep["positions"][0]
        assert r["ticker"] == "NVDA"
        assert r["last_decision_verb"] == "HOLD"
        # MUST pick the newest, not the second.
        assert r["last_decision_reasoning"] == \
            "earnings in 0.8d, hold through print"
        assert r["last_decision_confidence"] == 0.7
        assert r["hours_since_last_decision"] == pytest.approx(1.0)
        assert r["days_held"] == pytest.approx(2.0)
        assert rep["n_with_rationale"] == 1
        assert rep["verdict"] == "OK"

    def test_no_decision_and_blocked_rows_are_skipped(self):
        """A NO_DECISION / BLOCKED row newer than the real one must NOT win
        — those rows don't carry a rationale for the position."""
        positions = [{"ticker": "NVDA", "type": "stock", "qty": 1.0,
                      "opened_at": _ts(10.0)}]
        decisions = [
            {"timestamp": _ts(0.5),
             "action_taken": "NO_DECISION",
             "reasoning": "claude returned no response (timeout)"},
            {"timestamp": _ts(1.0),
             "action_taken": "SELL NVDA → BLOCKED",
             "reasoning": "no price for NVDA"},
            {"timestamp": _ts(3.0),
             "action_taken": "HOLD NVDA → HOLD",
             "reasoning": _envelope("HOLD", "thesis intact", 0.8)},
        ]
        rep = build_position_rationale(positions, decisions, now=NOW)
        r = rep["positions"][0]
        # NO_DECISION isn't tied to a ticker — must not match. The BLOCKED
        # row IS tied to NVDA and IS picked (it's still a real Opus
        # decision the trader wants to see).
        assert r["last_decision_verb"] == "SELL"
        # ... but it carries no JSON envelope, so reasoning is None.
        assert r["last_decision_reasoning"] is None
        assert r["last_decision_confidence"] is None
        # n_with_rationale counts only positions WITH a reasoning string
        assert rep["n_with_rationale"] == 0


class TestUnparseable:
    def test_legacy_text_reasoning_degrades_to_empty(self):
        """Pre-JSON reasoning (parse_failed / retry_failed / the raw
        no-response text) must NOT crash — surface verb + ts, omit
        reasoning."""
        positions = [{"ticker": "NVDA", "type": "stock", "qty": 1.0,
                      "opened_at": _ts(5.0)}]
        decisions = [
            {"timestamp": _ts(1.0),
             "action_taken": "HOLD NVDA → HOLD",
             "reasoning": "parse_failed: { not actually json"},
        ]
        rep = build_position_rationale(positions, decisions, now=NOW)
        r = rep["positions"][0]
        assert r["last_decision_verb"] == "HOLD"
        assert r["last_decision_reasoning"] is None
        assert r["last_decision_confidence"] is None

    def test_empty_string_reasoning_does_not_raise(self):
        assert _extract_decision("") == {}
        assert _extract_decision(None) == {}
        assert _extract_decision("   ") == {}

    def test_envelope_missing_inner_decision_dict(self):
        # Envelope is a dict but has no `decision` key — caller envelopes
        # written by an earlier code path.
        blob = json.dumps({"detail": "ok"})
        assert _extract_decision(blob) == {}

    def test_envelope_with_nondict_decision(self):
        # JSON shape changed under us — degrade gracefully.
        blob = json.dumps({"decision": "HOLD"})
        assert _extract_decision(blob) == {}

    def test_envelope_with_non_string_reasoning(self):
        # A non-string reasoning field is dropped (the rest of the
        # envelope still surfaces).
        blob = json.dumps({"decision": {"action": "HOLD",
                                          "confidence": 0.5,
                                          "reasoning": 42}})
        out = _extract_decision(blob)
        assert out.get("confidence") == 0.5
        assert "reasoning" not in out


class TestReasoningCap:
    def test_long_reasoning_is_truncated_to_max(self):
        # Opus emits a 2000-char reasoning — the cap keeps the response
        # compact.
        long_reason = "x" * (_MAX_REASON_CHARS + 500)
        positions = [{"ticker": "MU", "type": "stock", "qty": 1.0,
                      "opened_at": _ts(1.0)}]
        decisions = [{"timestamp": _ts(0.1),
                       "action_taken": "HOLD MU → HOLD",
                       "reasoning": json.dumps({
                           "decision": {"action": "HOLD",
                                         "confidence": 0.5,
                                         "reasoning": long_reason}
                       })}]
        rep = build_position_rationale(positions, decisions, now=NOW)
        r = rep["positions"][0]
        assert r["last_decision_reasoning"] is not None
        assert len(r["last_decision_reasoning"]) == _MAX_REASON_CHARS


class TestMissingRationale:
    def test_position_with_no_decisions_returns_none_row(self):
        positions = [{"ticker": "TQQQ", "type": "stock", "qty": 4.0,
                      "opened_at": _ts(30.0)}]
        decisions = [
            # Real decision on a DIFFERENT ticker — no match for TQQQ
            {"timestamp": _ts(1.0),
             "action_taken": "HOLD NVDA → HOLD",
             "reasoning": _envelope("HOLD", "NVDA thesis", 0.7)},
        ]
        rep = build_position_rationale(positions, decisions, now=NOW)
        r = rep["positions"][0]
        assert r["ticker"] == "TQQQ"
        assert r["last_decision_ts"] is None
        assert r["last_decision_verb"] is None
        assert r["last_decision_reasoning"] is None
        assert r["days_held"] == pytest.approx(30.0 / 24.0, abs=0.01)
        # Verdict reflects the missing rationale.
        assert rep["verdict"] == "MISSING_RATIONALE"
        assert rep["n_with_rationale"] == 0

    def test_missing_rationale_positions_sort_to_top(self):
        positions = [
            {"ticker": "NVDA", "type": "stock", "qty": 1.0,
             "opened_at": _ts(48.0)},
            {"ticker": "TQQQ", "type": "stock", "qty": 4.0,
             "opened_at": _ts(30.0)},
        ]
        decisions = [
            {"timestamp": _ts(0.5),
             "action_taken": "HOLD NVDA → HOLD",
             "reasoning": _envelope("HOLD", "NVDA still bullish", 0.7)},
        ]
        rep = build_position_rationale(positions, decisions, now=NOW)
        # TQQQ (no rationale) MUST sort above NVDA (has rationale).
        assert rep["positions"][0]["ticker"] == "TQQQ"
        assert rep["positions"][0]["last_decision_reasoning"] is None
        assert rep["positions"][1]["ticker"] == "NVDA"
        assert rep["positions"][1]["last_decision_reasoning"] is not None


class TestEmpty:
    def test_no_open_positions(self):
        rep = build_position_rationale([], [], now=NOW)
        assert rep["positions"] == []
        assert rep["n_positions"] == 0
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_garbage_timestamp_skipped_not_raised(self):
        # An unparseable timestamp must not crash — the decision row is
        # simply skipped (the next real one is still found).
        positions = [{"ticker": "NVDA", "type": "stock", "qty": 1.0,
                      "opened_at": _ts(2.0)}]
        decisions = [
            {"timestamp": "definitely-not-iso",
             "action_taken": "HOLD NVDA → HOLD",
             "reasoning": _envelope("HOLD", "stale ts", 0.5)},
            {"timestamp": _ts(1.0),
             "action_taken": "HOLD NVDA → HOLD",
             "reasoning": _envelope("HOLD", "good ts", 0.7)},
        ]
        rep = build_position_rationale(positions, decisions, now=NOW)
        # First (bad ts) was skipped; second (good ts) won.
        assert rep["positions"][0]["last_decision_reasoning"] == "good ts"

    def test_all_with_rationale_returns_ok(self):
        positions = [
            {"ticker": "NVDA", "type": "stock", "qty": 1.0,
             "opened_at": _ts(48.0)},
            {"ticker": "MU", "type": "stock", "qty": 2.0,
             "opened_at": _ts(20.0)},
        ]
        decisions = [
            {"timestamp": _ts(0.5),
             "action_taken": "HOLD NVDA → HOLD",
             "reasoning": _envelope("HOLD", "NVDA hold", 0.7)},
            {"timestamp": _ts(1.0),
             "action_taken": "HOLD MU → HOLD",
             "reasoning": _envelope("HOLD", "MU hold", 0.55)},
        ]
        rep = build_position_rationale(positions, decisions, now=NOW)
        assert rep["verdict"] == "OK"
        assert rep["n_with_rationale"] == 2
        # Both rows have a reasoning; oldest-since-last-decision sorts
        # MU (1.0h) ahead of NVDA (0.5h) within the has-rationale bucket.
        assert rep["positions"][0]["ticker"] == "MU"
        assert rep["positions"][1]["ticker"] == "NVDA"
