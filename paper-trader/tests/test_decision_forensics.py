"""Tests for analytics/decision_forensics.py — the NO_DECISION failure taxonomy.

These assert the *classification* and *aggregation* logic, not just "no
crash": a wrong precedence rule or a miscounted window will fail here.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics.decision_forensics import (
    build_decision_forensics,
    classify_failure,
)

NOW = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)


def _dec(action="NO_DECISION", reasoning="", mins_ago=1, market_open=True,
         signal_count=0):
    return {
        "timestamp": (NOW - timedelta(minutes=mins_ago)).isoformat(),
        "action_taken": action,
        "reasoning": reasoning,
        "market_open": market_open,
        "signal_count": signal_count,
    }


class TestClassifyFailure:
    def test_timeout_empty(self):
        c = classify_failure("claude returned no response (timeout/empty)")
        assert c["mode"] == "TIMEOUT_EMPTY"
        assert c["tag"] == "no_response"
        assert c["excerpt"] == ""

    def test_legacy_no_parseable_json(self):
        c = classify_failure("claude returned no parseable JSON")
        assert c["mode"] == "LEGACY_UNKNOWN"
        assert c["tag"] == "legacy"
        assert c["excerpt"] == ""

    def test_parse_failed_truncated(self):
        # '{' opened but never closed → response was cut off mid-object.
        c = classify_failure('parse_failed: {"action": "BUY", "ticker": "NV')
        assert c["mode"] == "TRUNCATED"
        assert c["tag"] == "parse_failed"
        assert c["excerpt"].startswith('{"action"')

    def test_retry_failed_keeps_tag(self):
        c = classify_failure('retry_failed: here you go {"action":"HOLD"}')
        assert c["tag"] == "retry_failed"
        # balanced braces, prose before '{' → prose-wrapped
        assert c["mode"] == "PROSE_WRAPPED"

    def test_parse_failed_no_json_is_refusal(self):
        c = classify_failure("parse_failed: I cannot provide trading advice.")
        assert c["mode"] == "NO_JSON"

    def test_parse_failed_empty_payload(self):
        c = classify_failure("parse_failed: ")
        assert c["mode"] == "EMPTY"

    def test_parse_failed_fenced(self):
        c = classify_failure('parse_failed: ```json\n{"action": "BUY"}\n```')
        assert c["mode"] == "FENCED"

    def test_parse_failed_malformed_balanced(self):
        # Starts at '{', braces balanced, but invalid JSON (single quotes /
        # trailing comma) — a syntax problem, not truncation.
        c = classify_failure("parse_failed: {'action': 'BUY',}")
        assert c["mode"] == "MALFORMED_JSON"

    def test_truncation_outranks_fence(self):
        # A fenced AND cut-off response: truncation is the actionable signal
        # (raise the timeout), so it must win the precedence.
        c = classify_failure('parse_failed: ```json\n{"action": "BUY", "qty')
        assert c["mode"] == "TRUNCATED"

    def test_non_failure_decision_json(self):
        c = classify_failure('{"decision": {"action": "HOLD"}, "detail": "x"}')
        assert c["tag"] == "not_a_failure"

    def test_none_and_empty(self):
        assert classify_failure(None)["mode"] == "EMPTY"
        assert classify_failure("")["tag"] == "none"

    def test_excerpt_is_capped(self):
        big = "parse_failed: " + ("z" * 5000)
        c = classify_failure(big)
        assert len(c["excerpt"]) <= 280


class TestBuildForensicsBasics:
    def test_empty_list(self):
        r = build_decision_forensics([], now=NOW)
        assert r["verdict"] == "NO_DATA"
        assert r["n_decisions"] == 0
        assert r["mode_mix"] == []

    def test_only_failures_counted(self):
        rows = [
            _dec("NO_DECISION", "claude returned no response (timeout/empty)"),
            _dec("HOLD NVDA → HOLD", '{"decision":{"action":"HOLD"}}'),
            _dec("BUY MU → FILLED", '{"decision":{"action":"BUY"}}'),
            _dec("", "parse_failed: oops"),  # blank action == NO_DECISION
        ]
        r = build_decision_forensics(rows, now=NOW)
        assert r["n_decisions"] == 4
        assert r["n_failures"] == 2
        assert r["failure_rate_pct"] == 50.0

    def test_mode_mix_counts_and_sort(self):
        rows = [
            _dec(reasoning="claude returned no response (timeout/empty)"),
            _dec(reasoning="claude returned no response (timeout/empty)"),
            _dec(reasoning="claude returned no response (timeout/empty)"),
            _dec(reasoning="parse_failed: I refuse."),  # NO_JSON
        ]
        r = build_decision_forensics(rows, now=NOW)
        modes = {m["mode"]: m for m in r["mode_mix"]}
        assert modes["TIMEOUT_EMPTY"]["n"] == 3
        assert modes["TIMEOUT_EMPTY"]["pct"] == 75.0
        assert modes["NO_JSON"]["n"] == 1
        # most frequent mode first
        assert r["mode_mix"][0]["mode"] == "TIMEOUT_EMPTY"
        assert r["dominant_mode"] == "TIMEOUT_EMPTY"
        assert r["hint"]  # actionable hint is non-empty

    def test_retry_exhausted_count(self):
        rows = [
            _dec(reasoning="retry_failed: still bad {"),
            _dec(reasoning="parse_failed: bad {"),
            _dec(reasoning="retry_failed: nope"),
        ]
        r = build_decision_forensics(rows, now=NOW)
        assert r["retry_exhausted"] == 2

    def test_by_market_split(self):
        rows = [
            _dec(reasoning="parse_failed: x", market_open=True),
            _dec("HOLD X → HOLD", '{"decision":{}}', market_open=True),
            _dec(reasoning="parse_failed: y", market_open=False),
            _dec(reasoning="parse_failed: z", market_open=False),
        ]
        r = build_decision_forensics(rows, now=NOW)
        assert r["by_market"]["open"]["total"] == 2
        assert r["by_market"]["open"]["failures"] == 1
        assert r["by_market"]["open"]["fail_pct"] == 50.0
        assert r["by_market"]["closed"]["total"] == 2
        assert r["by_market"]["closed"]["fail_pct"] == 100.0

    def test_recent_failures_have_excerpt(self):
        rows = [_dec(reasoning='parse_failed: {"action": "BU')]
        r = build_decision_forensics(rows, now=NOW)
        rf = r["recent_failures"]
        assert len(rf) == 1
        assert rf[0]["mode"] == "TRUNCATED"
        assert rf[0]["market_open"] is True
        assert rf[0]["excerpt"].startswith('{"action"')

    def test_hourly_buckets(self):
        rows = [
            _dec(reasoning="parse_failed: a", mins_ago=10),   # hour 11:50 → bucket 11:00
            _dec("HOLD → HOLD", '{"decision":{}}', mins_ago=20),
            _dec(reasoning="parse_failed: b", mins_ago=70),   # 10:50 → bucket 10:00
        ]
        r = build_decision_forensics(rows, now=NOW)
        hourly = r["hourly"]
        assert len(hourly) == 2  # two distinct hours
        total = sum(h["total"] for h in hourly)
        failures = sum(h["failures"] for h in hourly)
        assert total == 3 and failures == 2
        # oldest hour first
        assert hourly[0]["hour"] < hourly[1]["hour"]


class TestVerdictThresholds:
    def _many(self, fail_n, ok_n):
        rows = [_dec(reasoning="parse_failed: bad {", mins_ago=i + 1)
                for i in range(fail_n)]
        rows += [_dec("HOLD X → HOLD", '{"decision":{}}', mins_ago=fail_n + i + 1)
                 for i in range(ok_n)]
        return build_decision_forensics(rows, now=NOW)

    def test_critical_over_50(self):
        r = self._many(fail_n=8, ok_n=4)  # 12 in 24h, 66% fail
        assert r["verdict"] == "CRITICAL"
        assert r["verdict_window"] == "24h"

    def test_degraded_25_to_50(self):
        r = self._many(fail_n=4, ok_n=12)  # 16, 25% fail
        assert r["verdict"] == "DEGRADED"

    def test_healthy_under_25(self):
        r = self._many(fail_n=2, ok_n=18)  # 20, 10% fail
        assert r["verdict"] == "HEALTHY"

    def test_no_failures_is_healthy(self):
        rows = [_dec("HOLD X → HOLD", '{"decision":{}}', mins_ago=i + 1)
                for i in range(15)]
        r = build_decision_forensics(rows, now=NOW)
        assert r["verdict"] == "HEALTHY"
        assert r["n_failures"] == 0
