"""Tests for paper_trader.reporter._classify_block_reason /
_blocked_reasons_line and its wiring into send_hourly_summary /
send_daily_close.

The helper is silence-by-default — it surfaces a line ONLY when at least one
decision was BLOCKED in the window. A clean execution path produces nothing,
so the summary never grows a lying green light.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from paper_trader.reporter import (
    _blocked_reasons_line,
    _classify_block_reason,
    send_daily_close,
    send_hourly_summary,
)


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc)
            - timedelta(hours=hours_ago)).isoformat()


def _blocked_row(detail: str, hours_ago: float = 0.1,
                 action: str = "BUY NVDA") -> dict:
    """A decisions-table row exactly as ``strategy.decide`` writes a BLOCKED
    decision: ``action_taken`` ends ``→ BLOCKED`` and ``reasoning`` is the
    JSON blob carrying ``detail``."""
    return {
        "timestamp": _iso(hours_ago),
        "action_taken": f"{action} → BLOCKED",
        "reasoning": json.dumps({
            "decision": {"action": action.split()[0], "ticker": "NVDA"},
            "auto_exits": [],
            "detail": detail,
            "fallback_used": False,
        }),
    }


def _store(decisions: list[dict]):
    store = MagicMock()
    store.recent_decisions.return_value = decisions
    return store


# ── _classify_block_reason ───────────────────────────────────────────────
class TestClassifyBlockReason:
    @pytest.mark.parametrize("detail,expected", [
        ("insufficient cash (have $12.00, need $480.00)", "insufficient cash"),
        ("no price for NVDA", "no price"),
        ("no option price for NVDA 2026-06-19 900 call", "no option price"),
        ("sell qty 5.0 exceeds held 2.0 for NVDA stock", "oversell"),
        ("ambiguous call close for NVDA; specify strike+expiry", "ambiguous option close"),
        ("no matching open call for NVDA", "no position to close"),
        ("no open stock position in NVDA to close", "no position to close"),
        ("option trade missing strike/expiry", "malformed option"),
        ("strike not numeric: 'ATM'", "malformed field"),
        ("qty must be > 0", "malformed field"),
        ("unknown action FOO", "unknown action"),
        ("some brand new reason nobody anticipated", "other"),
        ("", "other"),
        (None, "other"),
    ])
    def test_buckets(self, detail, expected):
        assert _classify_block_reason(detail) == expected

    def test_no_option_price_wins_over_no_price(self):
        # "no option price" contains "price" — ordering must not misclassify
        # it as the plain-stock "no price" bucket.
        assert _classify_block_reason(
            "no option price for X 2026-06-19 5 put") == "no option price"

    def test_case_insensitive(self):
        assert _classify_block_reason(
            "INSUFFICIENT CASH (have $0)") == "insufficient cash"


# ── _blocked_reasons_line ────────────────────────────────────────────────
class TestBlockedReasonsLine:
    def test_no_decisions_is_silent(self):
        assert _blocked_reasons_line(_store([]), 1.0, "1h") == ""

    def test_no_blocked_rows_is_silent(self):
        rows = [
            {"timestamp": _iso(0.1), "action_taken": "BUY NVDA → FILLED",
             "reasoning": "{}"},
            {"timestamp": _iso(0.2), "action_taken": "HOLD MU → HOLD",
             "reasoning": "{}"},
            {"timestamp": _iso(0.3), "action_taken": "NO_DECISION",
             "reasoning": "claude returned no response (timeout)"},
        ]
        assert _blocked_reasons_line(_store(rows), 1.0, "1h") == ""

    def test_single_blocked_uses_singular_noun(self):
        out = _blocked_reasons_line(
            _store([_blocked_row("insufficient cash (have $1)")]), 1.0, "1h")
        assert out.startswith("**BLOCKED** ◈ 1 blocked decision last 1h")
        assert "insufficient cash ×1" in out

    def test_multiple_blocked_uses_plural_and_counts(self):
        rows = [
            _blocked_row("insufficient cash (have $1)"),
            _blocked_row("insufficient cash (have $2)"),
            _blocked_row("no price for NVDA"),
        ]
        out = _blocked_reasons_line(_store(rows), 1.0, "1h")
        assert "3 blocked decisions last 1h" in out
        # Most-frequent bucket first.
        assert "insufficient cash ×2, no price ×1" in out

    def test_ordering_ties_broken_alphabetically(self):
        rows = [
            _blocked_row("no price for NVDA"),
            _blocked_row("unknown action FOO"),
        ]
        out = _blocked_reasons_line(_store(rows), 1.0, "1h")
        # Both count 1 → alphabetical: "no price" before "unknown action".
        assert "no price ×1, unknown action ×1" in out

    def test_window_excludes_old_blocked_rows(self):
        rows = [
            _blocked_row("no price for NVDA", hours_ago=0.1),   # in window
            _blocked_row("insufficient cash", hours_ago=48.0),  # out of window
        ]
        out = _blocked_reasons_line(_store(rows), 1.0, "1h")
        assert "1 blocked decision last 1h" in out
        assert "no price ×1" in out
        assert "insufficient cash" not in out

    def test_24h_window_includes_older_rows(self):
        rows = [_blocked_row("no price for NVDA", hours_ago=12.0)]
        out = _blocked_reasons_line(_store(rows), 24.0, "24h")
        assert "1 blocked decision last 24h" in out

    def test_corrupt_reasoning_json_counts_as_other(self):
        rows = [{
            "timestamp": _iso(0.1),
            "action_taken": "BUY NVDA → BLOCKED",
            "reasoning": "not-json-at-all",
        }]
        out = _blocked_reasons_line(_store(rows), 1.0, "1h")
        assert "1 blocked decision last 1h" in out
        assert "other ×1" in out

    def test_missing_detail_field_counts_as_other(self):
        rows = [{
            "timestamp": _iso(0.1),
            "action_taken": "SELL MU → BLOCKED",
            "reasoning": json.dumps({"decision": {}, "fallback_used": False}),
        }]
        out = _blocked_reasons_line(_store(rows), 1.0, "1h")
        assert "other ×1" in out

    def test_store_exception_degrades_to_empty(self):
        store = MagicMock()
        store.recent_decisions.side_effect = RuntimeError("db locked")
        assert _blocked_reasons_line(store, 1.0, "1h") == ""

    def test_blocked_substring_match_is_case_insensitive(self):
        # action_taken is free text (invariant #11) — match must not depend
        # on exact casing of the BLOCKED token.
        rows = [{
            "timestamp": _iso(0.1),
            "action_taken": "buy nvda → blocked",
            "reasoning": json.dumps({"detail": "no price for NVDA"}),
        }]
        out = _blocked_reasons_line(_store(rows), 1.0, "1h")
        assert "no price ×1" in out


# ── wiring into the hourly / daily reports ───────────────────────────────
class TestBlockedReasonsWiring:
    def test_hourly_includes_blocked_block_when_blocked(self):
        with patch("paper_trader.reporter._send") as send, \
             patch("paper_trader.reporter._blocked_reasons_line",
                   return_value="**BLOCKED** ◈ 2 blocked decisions last 1h"):
            send.return_value = True
            send_hourly_summary()
            body = send.call_args[0][0]
        assert "**BLOCKED** ◈ 2 blocked decisions last 1h" in body

    def test_hourly_omits_blocked_block_when_silent(self):
        with patch("paper_trader.reporter._send") as send, \
             patch("paper_trader.reporter._blocked_reasons_line",
                   return_value=""):
            send.return_value = True
            send_hourly_summary()
            body = send.call_args[0][0]
        assert "**BLOCKED**" not in body

    def test_daily_close_includes_blocked_block_when_blocked(self):
        with patch("paper_trader.reporter._send") as send, \
             patch("paper_trader.reporter._blocked_reasons_line",
                   return_value="**BLOCKED** ◈ 5 blocked decisions last 24h"):
            send.return_value = True
            send_daily_close()
            body = send.call_args[0][0]
        assert "**BLOCKED** ◈ 5 blocked decisions last 24h" in body

    def test_daily_close_passes_24h_window(self):
        with patch("paper_trader.reporter._send") as send, \
             patch("paper_trader.reporter._blocked_reasons_line") as bl:
            send.return_value = True
            bl.return_value = ""
            send_daily_close()
        # The daily close must request the 24h window / label.
        assert bl.call_args[0][1] == 24.0
        assert bl.call_args[0][2] == "24h"

    def test_hourly_passes_1h_window(self):
        with patch("paper_trader.reporter._send") as send, \
             patch("paper_trader.reporter._blocked_reasons_line") as bl:
            send.return_value = True
            bl.return_value = ""
            send_hourly_summary()
        assert bl.call_args[0][1] == 1.0
        assert bl.call_args[0][2] == "1h"
