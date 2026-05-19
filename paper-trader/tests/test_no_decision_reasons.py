"""Tests for paper_trader.analytics.no_decision_reasons.

The point of this surface is to tell the operator WHICH lever to pull —
restart, kill parallel Opus jobs, or wait for the quota. Every assertion
below is on a *specific* expected bucket / state / recommendation; a
silently-broken bucketing would otherwise misdirect the operator on a
live storm.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics import no_decision_reasons as ndr  # noqa: E402


def _nd(reason: str) -> dict:
    """Helper — a NO_DECISION decision row with the given reasoning."""
    return {"action_taken": "NO_DECISION", "reasoning": reason,
            "timestamp": "2026-05-19T10:00:00+00:00"}


def _filled() -> dict:
    return {"action_taken": "BUY NVDA → FILLED", "reasoning": "",
            "timestamp": "2026-05-19T10:00:00+00:00"}


class TestBucketFor:
    def test_quota_markers(self):
        for r in (
            "claude quota/usage limit exhausted (no decision)",
            "Quota Exceeded for the month",
            "you've hit your org's monthly usage limit",
        ):
            assert ndr._bucket_for(r) == "quota_exhausted", r

    def test_host_saturated_markers(self):
        for r in (
            "skipped claude call — host saturated: 7 concurrent Opus (>4)",
            "host saturated mid-call: swap 90% (>90%)",
        ):
            assert ndr._bucket_for(r) == "host_saturated", r

    def test_model_empty_marker(self):
        assert (ndr._bucket_for(
            "claude returned no response (timeout/empty)")
            == "model_empty")

    def test_parse_and_retry_failed(self):
        assert ndr._bucket_for("parse_failed: garbage prose returned") \
            == "parse_failed"
        assert ndr._bucket_for("retry_failed: still no JSON") == "retry_failed"

    def test_unknown_falls_back_to_other(self):
        assert ndr._bucket_for("something we don't recognize") == "other"

    def test_none_and_empty_are_other(self):
        assert ndr._bucket_for(None) == "other"
        assert ndr._bucket_for("") == "other"
        assert ndr._bucket_for("   ") == "other"

    def test_quota_wins_over_other_markers(self):
        # If a quota outage message echoes the timeout phrase too,
        # the more-operator-actionable bucket wins.
        r = "claude quota/usage limit exhausted: returned no response"
        assert ndr._bucket_for(r) == "quota_exhausted"


class TestIsNoDecision:
    def test_canonical_no_decision_string(self):
        assert ndr._is_no_decision("NO_DECISION")

    def test_empty_or_none_treated_as_no_decision(self):
        # Mirrors decision_forensics / runner_heartbeat — an empty
        # action_taken row is documented as a NO_DECISION (a cycle that
        # failed to write a verb).
        assert ndr._is_no_decision(None)
        assert ndr._is_no_decision("")
        assert ndr._is_no_decision("   ")

    def test_filled_is_not_no_decision(self):
        assert not ndr._is_no_decision("BUY NVDA → FILLED")
        assert not ndr._is_no_decision("HOLD MU → HOLD")
        assert not ndr._is_no_decision("SELL X → BLOCKED")


class TestNoData:
    def test_empty_decisions_returns_no_data(self):
        r = ndr.build_no_decision_reasons([])
        assert r["state"] == "NO_DATA"
        assert r["n_no_decision"] == 0
        assert r["buckets"] == {}
        assert r["dominant_bucket"] is None
        assert r["recommendation"] == ""

    def test_only_filled_returns_no_data(self):
        r = ndr.build_no_decision_reasons([_filled()] * 5)
        assert r["state"] == "NO_DATA"
        assert r["n_no_decision"] == 0
        assert r["no_decision_pct"] == 0.0


class TestDominant:
    def test_pure_host_saturated_storm(self):
        # 10 NO_DECISION rows, all host_saturated.
        rows = [_nd("skipped claude call — host saturated: 7 concurrent "
                    "Opus (>4)") for _ in range(10)]
        r = ndr.build_no_decision_reasons(rows, window=20)
        assert r["state"] == "DOMINANT"
        assert r["dominant_bucket"] == "host_saturated"
        assert r["dominant_pct"] == 100.0
        assert r["n_no_decision"] == 10
        # The headline names the bucket and folds in the recommendation.
        assert "host_saturated" in r["headline"]
        assert "restart does NOT help" in r["headline"]
        # The recommendation must NOT tell the operator to restart for a
        # host_saturated storm — that is the *exact* misdirection this
        # builder exists to prevent.
        assert "restart" not in r["recommendation"].split(";")[0].lower()
        assert r["recommendation"] != ""

    def test_quota_dominant_recommends_waiting(self):
        rows = [_nd("claude quota/usage limit exhausted (no decision)")
                for _ in range(7)]
        r = ndr.build_no_decision_reasons(rows, window=20)
        assert r["dominant_bucket"] == "quota_exhausted"
        assert r["state"] == "DOMINANT"
        assert "wait" in r["recommendation"].lower()
        # Restart never appears as the recommended fix for a quota outage —
        # restarting does nothing about the org-level usage limit.
        assert "restart the runner" not in r["recommendation"].lower()

    def test_model_empty_dominant_recommends_restart(self):
        rows = [_nd("claude returned no response (timeout/empty)")
                for _ in range(6)]
        r = ndr.build_no_decision_reasons(rows, window=20)
        assert r["dominant_bucket"] == "model_empty"
        assert "restart" in r["recommendation"].lower()


class TestMixed:
    def test_50_50_split_is_mixed(self):
        # 4 model_empty + 4 host_saturated — neither hits 50% strictly
        # higher; with the default 50.0 threshold the top is at exactly 50.0
        # and qualifies as DOMINANT. We pick a 6:4 split for an UNambiguous
        # DOMINANT, and a 4:4:2 split for the MIXED case below.
        rows = (
            [_nd("claude returned no response (timeout/empty)")] * 6
            + [_nd("skipped claude call — host saturated")] * 4
        )
        r = ndr.build_no_decision_reasons(rows, window=20)
        assert r["state"] == "DOMINANT"
        assert r["dominant_bucket"] == "model_empty"

    def test_balanced_three_way_is_mixed(self):
        # 4:4:2 — no bucket clears 50%.
        rows = (
            [_nd("claude returned no response (timeout/empty)")] * 4
            + [_nd("skipped claude call — host saturated")] * 4
            + [_nd("parse_failed: prose")] * 2
        )
        r = ndr.build_no_decision_reasons(rows, window=20)
        assert r["state"] == "MIXED"
        assert r["dominant_bucket"] is None
        # No single fix is recommended — the operator must triage.
        assert r["recommendation"] == ""
        assert "no single dominant cause" in r["headline"]
        # The mix listing names every present bucket.
        for b in ("model_empty", "host_saturated", "parse_failed"):
            assert b in r["headline"]


class TestWindow:
    def test_window_truncates_to_newest(self):
        # 30 host_saturated newest + 30 quota_exhausted older. A window
        # of 30 must see ONLY the newest 30.
        rows = (
            [_nd("skipped claude call — host saturated")] * 30
            + [_nd("claude quota/usage limit exhausted (no decision)")] * 30
        )
        r = ndr.build_no_decision_reasons(rows, window=30)
        assert r["n_decisions"] == 30
        assert r["dominant_bucket"] == "host_saturated"
        assert r["dominant_pct"] == 100.0

    def test_filled_in_window_lowers_no_decision_pct(self):
        rows = [_filled()] * 4 + [_nd("parse_failed: x")] * 6
        r = ndr.build_no_decision_reasons(rows, window=10)
        assert r["n_decisions"] == 10
        assert r["n_no_decision"] == 6
        assert r["no_decision_pct"] == 60.0
        assert r["dominant_bucket"] == "parse_failed"

    def test_bogus_window_falls_back_to_default(self):
        rows = [_nd("parse_failed: x")] * 3
        r = ndr.build_no_decision_reasons(rows, window="oops")  # type: ignore[arg-type]
        # Window normalised; everything still works.
        assert r["state"] == "DOMINANT"
        assert r["window"] == ndr.DEFAULT_WINDOW


class TestNeverRaises:
    def test_none_decisions_is_safe(self):
        r = ndr.build_no_decision_reasons(None)
        assert r["state"] == "NO_DATA"

    def test_decisions_with_garbage_reasoning_bucket_other(self):
        rows = [{"action_taken": "NO_DECISION", "reasoning": None}] * 2
        r = ndr.build_no_decision_reasons(rows)
        assert r["dominant_bucket"] == "other"
        assert r["buckets"] == {"other": 2}


class TestReporterIntegration:
    """Pin the Discord reporter line shape and suppression contract."""

    def test_dominant_emits_two_line_block_with_recommendation(self):
        from paper_trader import reporter

        class Store:
            def recent_decisions(self, limit=20):
                return [{"action_taken": "NO_DECISION",
                         "reasoning": "skipped claude call — host saturated",
                         "timestamp": "2026-05-19T10:00:00+00:00"}] * 8

        line = reporter._no_decision_reasons_line(Store())
        assert "NO_DECISION CAUSE" in line
        assert "HOST_SATURATED" in line
        # Builder's headline + (possibly) a separate recommendation line.
        assert "host_saturated" in line.lower()

    def test_no_data_suppresses_line(self):
        from paper_trader import reporter

        class Store:
            def recent_decisions(self, limit=20):
                return [{"action_taken": "BUY NVDA → FILLED",
                         "reasoning": "",
                         "timestamp": "2026-05-19T10:00:00+00:00"}]

        assert reporter._no_decision_reasons_line(Store()) == ""

    def test_mixed_suppresses_line(self):
        from paper_trader import reporter

        class Store:
            def recent_decisions(self, limit=20):
                return (
                    [{"action_taken": "NO_DECISION",
                      "reasoning": "claude returned no response "
                                   "(timeout/empty)",
                      "timestamp": "2026-05-19T10:00:00+00:00"}] * 4
                    + [{"action_taken": "NO_DECISION",
                        "reasoning": "skipped claude call — host saturated",
                        "timestamp": "2026-05-19T09:00:00+00:00"}] * 4
                    + [{"action_taken": "NO_DECISION",
                        "reasoning": "parse_failed: prose returned",
                        "timestamp": "2026-05-19T08:00:00+00:00"}] * 2
                )

        # MIXED state — no single fix is appropriate, line is suppressed
        # so the hourly summary doesn't tell the operator "restart" on a
        # storm that's actually multi-cause.
        assert reporter._no_decision_reasons_line(Store()) == ""

    def test_store_exception_degrades_to_empty(self):
        from paper_trader import reporter

        class Store:
            def recent_decisions(self, limit=20):
                raise RuntimeError("DB locked")

        # A reporter helper must NEVER raise (it would drop the entire
        # hourly summary). Degrade to empty so the rest of the report
        # still goes out.
        assert reporter._no_decision_reasons_line(Store()) == ""
