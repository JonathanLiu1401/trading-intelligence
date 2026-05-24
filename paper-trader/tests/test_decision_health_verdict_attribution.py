"""Verdict-reason attribution for build_decision_health.

The legacy verdict_reason text always read "Opus output is failing to parse"
whenever the NO_DECISION rate crossed the DEGRADED/CRITICAL threshold. But the
canonical NO_DECISION ``reasoning`` taxonomy (see
``paper_trader/analytics/no_decision_reasons.py``) covers FIVE distinct causes
— quota exhaustion, host saturation, model timeout, parse failure, retry
failure — and each has a different operator action. Falsely attributing all
NO_DECISIONs as parse failures points the operator at the wrong fix (restart
the runner / inspect the prompt) when the live-trader-dominant cause is
actually host saturation (parallel Opus subprocesses).

These tests pin the new attribution-aware verdict_reason: the text must name
the *dominant* cause from the actual reasoning strings, not always the
parser. The verdict ladder thresholds (HEALTHY < 25% < DEGRADED < 50% <
CRITICAL) are preserved verbatim.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

import pytest

from paper_trader.analytics.decision_health import (
    _BUCKET_LABEL,
    _dominant_no_decision_cause,
    build_decision_health,
    hours_in_window,
)


def _fresh_rows(*, n_nd: int, nd_reason: str, n_real: int = 3,
                minutes_apart: int = 15) -> list[dict]:
    """Build a synthetic decisions ledger: ``n_nd`` NO_DECISION rows with the
    given ``reasoning``, plus ``n_real`` HOLD rows. All within last 24h so the
    24h window judges the verdict (matches the ladder's freshest-window rule)."""
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(n_nd):
        rows.append({
            "timestamp": (now - timedelta(minutes=15 + i * minutes_apart)).isoformat(),
            "action_taken": "NO_DECISION",
            "reasoning": nd_reason,
            "signal_count": 5,
        })
    for i in range(n_real):
        rows.append({
            "timestamp": (now - timedelta(minutes=2 + i * 5)).isoformat(),
            "action_taken": "HOLD NVDA → HOLD",
            "reasoning": json.dumps({"decision": {"confidence": 0.7}}),
            "signal_count": 10,
        })
    return rows


class TestHostSaturationAttribution:
    """The actual live-trader bug: 97% NO_DECISION from host saturation was
    being attributed to a parser regression."""

    def test_critical_host_saturated_names_saturation_not_parser(self):
        rows = _fresh_rows(
            n_nd=80, n_real=3,
            nd_reason="skipped claude call — host saturated: 8 concurrent Opus (>4)",
        )
        rep = build_decision_health(rows)
        assert rep["verdict"] == "CRITICAL"
        assert "host saturation" in rep["verdict_reason"].lower()
        assert "parse" not in rep["verdict_reason"].lower()
        assert "opus output" not in rep["verdict_reason"].lower()

    def test_degraded_host_saturated_still_names_saturation(self):
        rows = _fresh_rows(
            n_nd=10, n_real=23,  # 10/33 = 30% (DEGRADED band)
            nd_reason="skipped claude call — host saturated: 8 concurrent Opus (>4)",
        )
        rep = build_decision_health(rows)
        assert rep["verdict"] == "DEGRADED"
        assert "host saturation" in rep["verdict_reason"].lower()
        assert "parse_decision" not in rep["verdict_reason"]


class TestQuotaAttribution:
    def test_critical_quota_names_quota(self):
        rows = _fresh_rows(
            n_nd=60, n_real=3,
            nd_reason="claude quota/usage limit exhausted (no decision)",
        )
        rep = build_decision_health(rows)
        assert rep["verdict"] == "CRITICAL"
        assert "quota" in rep["verdict_reason"].lower()


class TestParseFailureAttribution:
    """When the actual cause IS the parser, we must still report it as such —
    don't lose the original signal in the rewrite."""

    def test_critical_parse_failed_names_parser(self):
        rows = _fresh_rows(
            n_nd=60, n_real=3,
            nd_reason="parse_failed: { malformed json from opus",
        )
        rep = build_decision_health(rows)
        assert rep["verdict"] == "CRITICAL"
        assert "parse" in rep["verdict_reason"].lower()


class TestModelTimeoutAttribution:
    def test_critical_timeout_names_timeout(self):
        rows = _fresh_rows(
            n_nd=60, n_real=3,
            nd_reason="claude returned no response (timeout)",
        )
        rep = build_decision_health(rows)
        assert rep["verdict"] == "CRITICAL"
        # "timed out" or "timeout" both acceptable; label is "Claude CLI timed out"
        assert "timed out" in rep["verdict_reason"].lower() or "timeout" in rep["verdict_reason"].lower()


class TestHealthyVerdictUnchanged:
    """Healthy verdict text is unchanged — no false-positive cause attribution."""

    def test_healthy_low_fail_rate_has_no_cause_attribution(self):
        rows = _fresh_rows(
            n_nd=1, n_real=20,  # ~5% NO_DECISION rate
            nd_reason="skipped claude call — host saturated",
        )
        rep = build_decision_health(rows)
        assert rep["verdict"] == "HEALTHY"
        # Verdict reason should not be alarming or naming a cause —
        # the rate is within normal range.
        assert "host saturation" not in rep["verdict_reason"].lower()


class TestVerdictThresholdsPreserved:
    """The 25% / 50% verdict ladder thresholds must be byte-preserved by the
    refactor — this is the legacy contract the existing dashboard polls."""

    @pytest.mark.parametrize("n_nd,n_real,expected", [
        (0, 20, "HEALTHY"),         # 0% → HEALTHY
        (5, 95, "HEALTHY"),         # 5% → HEALTHY
        (24, 76, "HEALTHY"),        # 24% → HEALTHY (< 25%)
        (25, 75, "DEGRADED"),       # 25% → DEGRADED (>= 25%)
        (49, 51, "DEGRADED"),       # 49% → DEGRADED
        (50, 50, "CRITICAL"),       # 50% → CRITICAL (>= 50%)
        (97, 3, "CRITICAL"),        # 97% → CRITICAL
    ])
    def test_threshold_ladder(self, n_nd, n_real, expected):
        rows = _fresh_rows(
            n_nd=n_nd, n_real=n_real,
            nd_reason="skipped claude call — host saturated",
            minutes_apart=1,
        )
        rep = build_decision_health(rows)
        assert rep["verdict"] == expected, (
            f"{n_nd}/{n_nd+n_real} NO_DECISION = "
            f"{rep['windows']['24h']['parse_fail_pct']}% → expected {expected}, "
            f"got {rep['verdict']} ({rep['verdict_reason']})"
        )


class TestNoDataAndEmpty:
    def test_no_decisions_no_data_verdict(self):
        rep = build_decision_health([])
        assert rep["verdict"] == "NO_DATA"
        assert rep["verdict_reason"] == "no decisions recorded yet"

    def test_unparseable_timestamps_excluded_from_window(self):
        """A row whose timestamp won't parse must not crash the build."""
        rows = _fresh_rows(n_nd=10, n_real=10,
                          nd_reason="skipped claude call — host saturated")
        rows.append({
            "timestamp": "not a valid iso string",
            "action_taken": "NO_DECISION",
            "reasoning": "skipped claude call — host saturated",
            "signal_count": None,
        })
        rep = build_decision_health(rows)
        # Must not raise; verdict should still derive from the valid rows.
        assert rep["verdict"] in {"HEALTHY", "DEGRADED", "CRITICAL"}


class TestDominantCauseHelper:
    """Direct coverage of the new _dominant_no_decision_cause helper."""

    def test_empty_rows_returns_safe_default(self):
        from paper_trader.analytics.no_decision_reasons import _bucket_for
        out = _dominant_no_decision_cause([], _bucket_for)
        assert out == "no NO_DECISION rows in window"

    def test_real_decisions_skipped_from_attribution(self):
        """A HOLD row must not contribute to NO_DECISION cause attribution."""
        from paper_trader.analytics.no_decision_reasons import _bucket_for
        rows = [
            {"action_taken": "HOLD NVDA → HOLD",
             "reasoning": "decided to wait"},  # NOT a NO_DECISION cause
        ]
        out = _dominant_no_decision_cause(rows, _bucket_for)
        assert out == "no NO_DECISION rows in window"

    def test_dominant_bucket_named_and_counted(self):
        from paper_trader.analytics.no_decision_reasons import _bucket_for
        rows = [
            {"action_taken": "NO_DECISION",
             "reasoning": "skipped claude call — host saturated"}
            for _ in range(5)
        ] + [
            {"action_taken": "NO_DECISION",
             "reasoning": "claude returned no response (timeout)"}
        ]
        out = _dominant_no_decision_cause(rows, _bucket_for)
        # Expect "host saturation ... (5/6 = 83%)" or similar.
        assert "host saturation" in out.lower()
        assert "5/6" in out

    def test_ties_break_by_operator_actionable_rank(self):
        """quota_exhausted outranks host_saturated when they tie."""
        from paper_trader.analytics.no_decision_reasons import _bucket_for
        rows = [
            {"action_taken": "NO_DECISION",
             "reasoning": "claude quota/usage limit exhausted"},
            {"action_taken": "NO_DECISION",
             "reasoning": "claude quota/usage limit exhausted"},
            {"action_taken": "NO_DECISION",
             "reasoning": "skipped claude call — host saturated"},
            {"action_taken": "NO_DECISION",
             "reasoning": "skipped claude call — host saturated"},
        ]
        out = _dominant_no_decision_cause(rows, _bucket_for)
        assert "quota" in out.lower()


class TestHoursInWindow:
    """The window membership predicate must agree with _window."""

    def test_all_window_includes_everything(self):
        now = datetime.now(timezone.utc)
        assert hours_in_window(now - timedelta(days=365), now, "all") is True
        # None ts → False (matches _window, which silently drops unparseable
        # rows). For the "all" window with no time gate, we still get True.
        assert hours_in_window(None, now, "all") is True

    def test_24h_window_excludes_older(self):
        now = datetime.now(timezone.utc)
        assert hours_in_window(now - timedelta(hours=23), now, "24h") is True
        assert hours_in_window(now - timedelta(hours=25), now, "24h") is False

    def test_7d_window_includes_within_week(self):
        now = datetime.now(timezone.utc)
        assert hours_in_window(now - timedelta(days=6), now, "7d") is True
        assert hours_in_window(now - timedelta(days=8), now, "7d") is False

    def test_none_ts_excluded_from_bounded_window(self):
        now = datetime.now(timezone.utc)
        assert hours_in_window(None, now, "24h") is False
        assert hours_in_window(None, now, "7d") is False


class TestRecentTapeCauseEnrichment:
    """Each NO_DECISION row in the recent[] tape carries its bucket cause.

    Without this, a trader scrolling the dashboard's recent-decisions tape
    sees a wall of "NO_DECISION" rows but cannot tell whether they're
    clustered by host saturation, model timeout, or parser failure — the
    triage signal that determines what to do next.
    """

    def test_no_decision_row_carries_cause_and_label(self):
        rows = _fresh_rows(
            n_nd=1, n_real=0,
            nd_reason="skipped claude call — host saturated: 8 concurrent",
        )
        rep = build_decision_health(rows)
        nd_recents = [r for r in rep["recent"] if r["category"] == "NO_DECISION"]
        assert len(nd_recents) == 1
        assert nd_recents[0]["cause"] == "host_saturated"
        assert "host saturation" in nd_recents[0]["cause_label"].lower()

    def test_quota_rows_bucket_to_quota_exhausted(self):
        rows = _fresh_rows(
            n_nd=1, n_real=0,
            nd_reason="claude quota/usage limit exhausted (no decision)",
        )
        rep = build_decision_health(rows)
        nd = [r for r in rep["recent"] if r["category"] == "NO_DECISION"]
        assert nd[0]["cause"] == "quota_exhausted"
        assert "quota" in nd[0]["cause_label"].lower()

    def test_each_no_decision_bucket_distinct_in_tape(self):
        """Mixed-cause tape: every row preserves its own cause attribution."""
        now = datetime.now(timezone.utc)
        rows = [
            {"timestamp": (now - timedelta(minutes=1)).isoformat(),
             "action_taken": "NO_DECISION",
             "reasoning": "skipped claude call — host saturated"},
            {"timestamp": (now - timedelta(minutes=2)).isoformat(),
             "action_taken": "NO_DECISION",
             "reasoning": "claude returned no response (timeout)"},
            {"timestamp": (now - timedelta(minutes=3)).isoformat(),
             "action_taken": "NO_DECISION",
             "reasoning": "parse_failed: { malformed"},
            {"timestamp": (now - timedelta(minutes=4)).isoformat(),
             "action_taken": "NO_DECISION",
             "reasoning": "claude quota/usage limit exhausted"},
        ]
        rep = build_decision_health(rows)
        causes = [r.get("cause") for r in rep["recent"] if r["category"] == "NO_DECISION"]
        assert causes == [
            "host_saturated", "model_timeout", "parse_failed", "quota_exhausted",
        ]

    def test_hold_filled_rows_keep_legacy_shape_no_cause(self):
        """A HOLD or FILLED row must NOT carry a cause field — the
        enrichment is purely NO_DECISION-specific. Existing dashboard JS
        that doesn't know about `cause` for non-NO_DECISION rows must
        continue to work unchanged."""
        rows = _fresh_rows(
            n_nd=0, n_real=3,
            nd_reason="ignored",
        )
        rep = build_decision_health(rows)
        for r in rep["recent"]:
            if r["category"] != "NO_DECISION":
                assert "cause" not in r
                assert "cause_label" not in r

    def test_unknown_reasoning_buckets_to_other(self):
        """A NO_DECISION with reasoning we don't know how to classify still
        gets a cause — the 'other' bucket — so the column is never None."""
        rows = [
            {"timestamp": datetime.now(timezone.utc).isoformat(),
             "action_taken": "NO_DECISION",
             "reasoning": "something totally unprecedented happened"},
        ]
        rep = build_decision_health(rows)
        nd = [r for r in rep["recent"] if r["category"] == "NO_DECISION"]
        assert nd[0]["cause"] == "other"
        assert nd[0]["cause_label"]  # non-empty label


class TestBucketLabelsCoverAllBuckets:
    """The verdict_reason renders a label per canonical bucket — any new
    bucket added to no_decision_reasons but missing a label here would fall
    back to the raw bucket name (less readable)."""

    def test_all_known_buckets_have_a_label(self):
        from paper_trader.analytics.no_decision_reasons import _RECOMMENDATIONS
        # Every bucket no_decision_reasons knows about must have a label.
        for bucket in _RECOMMENDATIONS:
            assert bucket in _BUCKET_LABEL, (
                f"bucket {bucket!r} from no_decision_reasons has no label in "
                f"decision_health._BUCKET_LABEL — verdict_reason will render "
                f"the raw bucket name. Add a label."
            )
