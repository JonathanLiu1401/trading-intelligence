"""Escalation logic for the ML retrain failure guard.

Regression context: a silent UnboundLocalError in ml.trainer once kept
ArticleNet from retraining for an entire daemon lifetime while the hourly
healthcheck still reported errors=0 (it greps ERROR/CRITICAL; retrain
failures log at WARNING). These tests pin the decision logic that escalates
a persistently broken trainer to Discord so that blind spot cannot recur.
"""
from __future__ import annotations

from core.retrain_guard import (
    ML_RETRAIN_FAIL_ALERT_THRESHOLD,
    alert_message,
    should_alert,
)


class TestShouldAlert:
    def test_no_alert_before_threshold(self):
        for n in range(ML_RETRAIN_FAIL_ALERT_THRESHOLD):
            assert should_alert(n) is False

    def test_alert_exactly_at_threshold(self):
        assert should_alert(ML_RETRAIN_FAIL_ALERT_THRESHOLD) is True

    def test_no_alert_between_thresholds(self):
        # 4 and 5 sit between the first (3) and second (6) escalation points.
        t = ML_RETRAIN_FAIL_ALERT_THRESHOLD
        assert should_alert(t + 1) is False
        assert should_alert(t + 2) is False

    def test_re_alerts_every_threshold_multiple(self):
        t = ML_RETRAIN_FAIL_ALERT_THRESHOLD
        assert should_alert(t * 2) is True
        assert should_alert(t * 3) is True

    def test_zero_and_negative_never_alert(self):
        assert should_alert(0) is False
        assert should_alert(-1) is False

    def test_non_positive_threshold_disables_alerts(self):
        # Defensive: a misconfigured threshold must not divide-by-zero or
        # alert on every cycle.
        assert should_alert(10, threshold=0) is False
        assert should_alert(10, threshold=-3) is False


class TestAlertMessage:
    def test_includes_count_and_error(self):
        msg = alert_message(3, "cannot access local variable 'texts'")
        assert "3" in msg
        assert "texts" in msg

    def test_truncates_long_error(self):
        msg = alert_message(6, "x" * 5000)
        # Body must stay well under Discord's 2000-char limit.
        assert len(msg) < 500
