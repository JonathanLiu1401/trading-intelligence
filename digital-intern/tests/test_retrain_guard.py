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
    is_retrain_failure,
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


class TestIsRetrainFailure:
    """``ml.trainer.train()`` returns a status dict instead of raising on
    internal failure, so the trainer worker's try/except never sees it. These
    pin that a returned-error cycle is classified as a failure (it MUST count
    toward escalation) while ``ok``/``skipped`` are not — the regression that
    let a subprocess_timeout-every-cycle trainer escalate nothing."""

    def test_ok_is_not_a_failure(self):
        assert is_retrain_failure(
            {"status": "ok", "final_loss": 0.12, "n": 900}
        ) is False

    def test_skipped_is_not_a_failure(self):
        # too_few_samples / trainer_busy are benign no-ops — the loop is fine.
        assert is_retrain_failure(
            {"status": "skipped", "reason": "too_few_samples", "n": 12}
        ) is False
        assert is_retrain_failure(
            {"status": "skipped", "reason": "trainer_busy", "n": 900}
        ) is False

    def test_subprocess_timeout_is_a_failure(self):
        # The exact live-observed case (2026-05-22): train() returned this
        # WITHOUT raising, so consec_fail never moved.
        assert is_retrain_failure(
            {"status": "error", "reason": "subprocess_timeout", "elapsed_s": 659.5}
        ) is True

    def test_other_error_statuses_are_failures(self):
        for reason in ("no_result", "child_exception: boom"):
            assert is_retrain_failure(
                {"status": "error", "reason": reason}
            ) is True

    def test_unknown_or_missing_status_is_a_failure(self):
        # Defensive: an unexpected shape means train() did not clearly succeed.
        assert is_retrain_failure({"n": 5}) is True
        assert is_retrain_failure({"status": "weird"}) is True

    def test_non_dict_is_a_failure(self):
        assert is_retrain_failure(None) is True
        assert is_retrain_failure("error") is True

    def test_failure_classification_feeds_escalation(self):
        # End-to-end: N consecutive returned-error cycles must reach the
        # should_alert threshold, exactly as N raised exceptions would.
        consec = 0
        for _ in range(ML_RETRAIN_FAIL_ALERT_THRESHOLD):
            metrics = {"status": "error", "reason": "subprocess_timeout"}
            if is_retrain_failure(metrics):
                consec += 1
        assert consec == ML_RETRAIN_FAIL_ALERT_THRESHOLD
        assert should_alert(consec) is True
        # A subsequent ok cycle resets the counter.
        if not is_retrain_failure({"status": "ok", "final_loss": 0.1}):
            consec = 0
        assert consec == 0
