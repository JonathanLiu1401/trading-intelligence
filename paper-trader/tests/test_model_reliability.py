"""Tests for paper_trader.analytics.model_reliability.

These assert SPECIFIC numbers and verdicts against hand-built decision
ledgers — they exist to catch real logic regressions (legacy-row
miscounting, outcome-prefix parsing, FILLED-from-fallback attribution,
verdict bands, trend direction, sample-size gate), not just "it runs".
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics.model_reliability import (
    build_model_reliability,
    _classify,
    _outcome,
    _MIN_ATTRIBUTABLE,
)

NOW = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)


def _made(action_taken: str, fallback_used, ts: datetime,
          confidence: float = 0.8) -> dict:
    """A made-decision row exactly as store.record_decision writes it."""
    inner = {"decision": {"action": action_taken.split()[0],
                          "reasoning": "x", "confidence": confidence},
             "auto_exits": [], "detail": "", "fallback_used": fallback_used}
    return {"timestamp": ts.isoformat(), "action_taken": action_taken,
            "reasoning": json.dumps(inner)}


def _legacy_made(action_taken: str, ts: datetime) -> dict:
    """A made row from before the fallback_used key existed (no key)."""
    inner = {"decision": {"action": action_taken.split()[0]},
             "auto_exits": [], "detail": ""}
    return {"timestamp": ts.isoformat(), "action_taken": action_taken,
            "reasoning": json.dumps(inner)}


def _no_decision(reason: str, ts: datetime) -> dict:
    return {"timestamp": ts.isoformat(), "action_taken": "NO_DECISION",
            "reasoning": reason}


# ─── _outcome ────────────────────────────────────────────────────────────
class TestOutcome:
    def test_filled(self):
        assert _outcome("BUY NVDA → FILLED") == "FILLED"

    def test_hold(self):
        assert _outcome("HOLD MU → HOLD") == "HOLD"

    def test_blocked(self):
        assert _outcome("SELL X → BLOCKED") == "BLOCKED"

    def test_bare_no_decision(self):
        assert _outcome("NO_DECISION") == "NO_DECISION"

    def test_empty_is_no_decision(self):
        assert _outcome("") == "NO_DECISION"
        assert _outcome(None) == "NO_DECISION"

    def test_unrecognised_tail_is_no_decision(self):
        # An arrow with a garbage tail must not be miscounted as a real outcome.
        assert _outcome("BUY X → WAT") == "NO_DECISION"


# ─── _classify ───────────────────────────────────────────────────────────
class TestClassify:
    def test_opus_when_fallback_false(self):
        assert _classify(_made("BUY NVDA → FILLED", False, NOW)) == "opus"

    def test_sonnet_when_fallback_true(self):
        assert _classify(_made("BUY NVDA → FILLED", True, NOW)) == "sonnet_fallback"

    def test_legacy_when_key_missing(self):
        # The exact live failure mode: 279 pre-instrumentation rows read back
        # with no fallback_used key. They must NOT be counted as Opus.
        assert _classify(_legacy_made("HOLD MU → HOLD", NOW)) == "legacy_unknown"

    def test_legacy_when_fallback_none(self):
        assert _classify(_made("HOLD MU → HOLD", None, NOW)) == "legacy_unknown"

    def test_non_json_reasoning_made_is_legacy(self):
        row = {"timestamp": NOW.isoformat(),
               "action_taken": "BUY X → FILLED", "reasoning": "freeform note"}
        assert _classify(row) == "legacy_unknown"

    def test_timeout_no_decision(self):
        assert _classify(_no_decision(
            "claude returned no response (timeout/empty)", NOW)) == "timeout"

    def test_parse_failed_no_decision(self):
        assert _classify(_no_decision(
            "parse_failed: {garbled", NOW)) == "parse_failed"

    def test_retry_failed_no_decision(self):
        assert _classify(_no_decision(
            "retry_failed: still bad", NOW)) == "retry_failed"

    def test_unknown_no_decision_reason(self):
        assert _classify(_no_decision("weird", NOW)) == "other_no_dec"


# ─── build_model_reliability: empty / insufficient ───────────────────────
class TestStates:
    def test_no_data(self):
        rep = build_model_reliability([], now=NOW)
        assert rep["state"] == "NO_DATA"
        assert rep["verdict"] is None
        assert rep["n_decisions"] == 0

    def test_insufficient_below_threshold(self):
        # 9 attributable < _MIN_ATTRIBUTABLE (10) → verdict withheld.
        rows = [_made("BUY X → FILLED", False, NOW - timedelta(minutes=i))
                for i in range(_MIN_ATTRIBUTABLE - 1)]
        rep = build_model_reliability(rows, now=NOW)
        assert rep["state"] == "INSUFFICIENT"
        assert rep["verdict"] is None

    def test_legacy_rows_do_not_satisfy_threshold(self):
        # 50 legacy rows + 3 opus → still INSUFFICIENT (only 3 attributable).
        rows = [_legacy_made("HOLD X → HOLD", NOW - timedelta(minutes=i))
                for i in range(50)]
        rows += [_made("BUY X → FILLED", False, NOW - timedelta(minutes=100 + i))
                 for i in range(3)]
        rep = build_model_reliability(rows, now=NOW)
        assert rep["state"] == "INSUFFICIENT"
        assert rep["windows"]["all"]["legacy_unknown"] == 50
        assert rep["windows"]["all"]["attributable"] == 3


# ─── verdict bands ───────────────────────────────────────────────────────
class TestVerdictBands:
    def test_opus_healthy_at_full_opus(self):
        rows = [_made("BUY X → FILLED", False, NOW - timedelta(minutes=i))
                for i in range(20)]
        rep = build_model_reliability(rows, now=NOW)
        assert rep["state"] == "OK"
        assert rep["verdict"] == "OPUS_HEALTHY"
        assert rep["windows"]["all"]["opus_share_pct"] == 100.0

    def test_degraded_band(self):
        # 16 opus / 4 fallback = 80% opus → DEGRADED (70 ≤ x < 90).
        rows = [_made("BUY X → FILLED", False, NOW - timedelta(minutes=i))
                for i in range(16)]
        rows += [_made("BUY X → FILLED", True, NOW - timedelta(minutes=100 + i))
                 for i in range(4)]
        rep = build_model_reliability(rows, now=NOW)
        assert rep["verdict"] == "DEGRADED"
        assert rep["windows"]["all"]["opus_share_pct"] == 80.0

    def test_failing_band(self):
        # 10 opus / 10 fallback = 50% opus → FAILING (< 70).
        rows = [_made("BUY X → FILLED", False, NOW - timedelta(minutes=i))
                for i in range(10)]
        rows += [_made("BUY X → FILLED", True, NOW - timedelta(minutes=100 + i))
                 for i in range(10)]
        rep = build_model_reliability(rows, now=NOW)
        assert rep["verdict"] == "FAILING"
        assert rep["windows"]["all"]["opus_share_pct"] == 50.0


# ─── the money metric: executed trades placed by the fallback ────────────
class TestFilledFromFallback:
    def test_only_filled_counts_not_hold(self):
        rows = [
            _made("BUY A → FILLED", True, NOW - timedelta(minutes=1)),   # fb fill
            _made("HOLD B → HOLD", True, NOW - timedelta(minutes=2)),    # fb but not a fill
            _made("BUY C → FILLED", False, NOW - timedelta(minutes=3)),  # opus fill
        ] + [_made("HOLD Z → HOLD", False, NOW - timedelta(minutes=10 + i))
             for i in range(10)]
        w = build_model_reliability(rows, now=NOW)["windows"]["all"]
        assert w["filled_total"] == 2          # only the two FILLED rows
        assert w["filled_fallback"] == 1       # one of them was the fallback
        assert w["filled_fallback_pct"] == 50.0


# ─── windowing ───────────────────────────────────────────────────────────
class TestWindows:
    def test_24h_excludes_old_rows(self):
        rows = [_made("BUY X → FILLED", False, NOW - timedelta(hours=1))
                for _ in range(12)]
        rows += [_made("BUY X → FILLED", True, NOW - timedelta(hours=48))
                 for _ in range(12)]
        rep = build_model_reliability(rows, now=NOW)
        assert rep["windows"]["24h"]["opus"] == 12
        assert rep["windows"]["24h"]["sonnet_fallback"] == 0
        assert rep["windows"]["all"]["sonnet_fallback"] == 12

    def test_no_decision_pct_counts_all_cycles(self):
        rows = [_made("HOLD X → HOLD", False, NOW - timedelta(minutes=i))
                for i in range(15)]
        rows += [_no_decision("claude returned no response (timeout/empty)",
                              NOW - timedelta(minutes=100 + i))
                 for i in range(5)]
        w = build_model_reliability(rows, now=NOW)["windows"]["all"]
        assert w["total"] == 20
        assert w["timeout"] == 5
        assert w["no_decision_pct"] == 25.0     # 5 / 20


# ─── trend ───────────────────────────────────────────────────────────────
class TestTrend:
    def test_worsening_when_recent_half_more_fallback(self):
        # decisions newest-first: recent half all fallback, older half all
        # opus → recent opus_share 0%, older 100% → worsening.
        recent = [_made("BUY X → FILLED", True, NOW - timedelta(minutes=i))
                  for i in range(10)]
        older = [_made("BUY X → FILLED", False, NOW - timedelta(hours=5, minutes=i))
                 for i in range(10)]
        rep = build_model_reliability(recent + older, now=NOW)
        t = rep["trend"]
        assert t["recent_opus_share_pct"] == 0.0
        assert t["older_opus_share_pct"] == 100.0
        assert t["direction"] == "worsening"

    def test_flat_when_consistent(self):
        rows = [_made("BUY X → FILLED", False, NOW - timedelta(minutes=i))
                for i in range(20)]
        assert build_model_reliability(rows, now=NOW)["trend"]["direction"] == "flat"


# ─── robustness ──────────────────────────────────────────────────────────
class TestRobustness:
    def test_never_raises_on_garbage_rows(self):
        rows = [
            {"timestamp": None, "action_taken": None, "reasoning": None},
            {"timestamp": "not-a-date", "action_taken": 123, "reasoning": 456},
            {},
            _made("BUY X → FILLED", False, NOW),
        ]
        rep = build_model_reliability(rows, now=NOW)   # must not raise
        assert "windows" in rep and "all" in rep["windows"]

    def test_headline_is_a_nonempty_string_in_every_state(self):
        for rows in ([], [_made("BUY X → FILLED", False, NOW)],
                     [_made("BUY X → FILLED", False, NOW - timedelta(minutes=i))
                      for i in range(20)]):
            rep = build_model_reliability(rows, now=NOW)
            assert isinstance(rep["headline"], str) and rep["headline"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
