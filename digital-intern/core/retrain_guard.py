"""Consecutive-failure escalation for the ML retrain loop.

Background: a silent ``UnboundLocalError`` in :mod:`ml.trainer` once kept
ArticleNet from retraining for an entire daemon lifetime. The failure logged
at WARNING every cycle, and the hourly healthcheck only greps ERROR/CRITICAL,
so the model went stale for hours with no signal raised.

This module owns the decision of *when* a persistently broken retrain loop
should escalate to Discord. The trainer worker keeps the tiny failure
counter; the policy here is pure and unit tested so the blind spot cannot
silently recur.
"""
from __future__ import annotations

# Escalate after this many back-to-back retrain failures, then re-alert once
# every THRESHOLD further failures (3, 6, 9, …). Re-alerting on a multiple
# rather than every cycle keeps a flapping trainer from flooding the channel
# while still re-pinging if it stays broken.
ML_RETRAIN_FAIL_ALERT_THRESHOLD = 3


def should_alert(consecutive_failures: int,
                 threshold: int = ML_RETRAIN_FAIL_ALERT_THRESHOLD) -> bool:
    """Whether the current consecutive-failure count warrants a Discord alert.

    Fires exactly at ``threshold`` and then on every ``threshold`` multiple
    after it. Returns ``False`` for non-positive counts and for any
    misconfigured non-positive ``threshold`` (defensive: never divide by zero
    or alert on every cycle)."""
    if threshold <= 0:
        return False
    if consecutive_failures < threshold:
        return False
    return consecutive_failures % threshold == 0


def alert_message(consecutive_failures: int, last_error: str) -> str:
    """Discord alert body for a stuck ML retrain loop."""
    return (
        f"🚨 ML TRAINER STUCK: {consecutive_failures} consecutive retrain "
        f"failures — ArticleNet is not learning new labels. "
        f"Last error: {last_error[:300]}"
    )


# Return-value statuses ``ml.trainer.train()`` produces that are NOT failures:
#   - "ok"      — a real training cycle completed.
#   - "skipped" — benign no-op (``too_few_samples`` before bootstrap has
#                 enough labels, or ``trainer_busy`` when the heavy trainer
#                 holds the lock). The retrain loop is healthy; nothing stuck.
# Everything else ("error" — ``subprocess_timeout`` / ``no_result`` /
# ``child_exception`` — or an unexpected shape) is a failure.
_RETRAIN_OK_STATUSES = ("ok", "skipped")


def is_retrain_failure(metrics) -> bool:
    """True when a ``ml.trainer.train()`` return value represents a FAILED
    retrain cycle that should count toward consecutive-failure escalation.

    ``train()`` catches every internal error and RETURNS a status dict instead
    of raising — ``{"status": "error", "reason": "subprocess_timeout"}``,
    ``no_result``, ``child_exception``. The trainer worker's ``try/except``
    therefore never observes these (no exception is raised), so without this
    classifier a trainer that times out on EVERY cycle increments no failure
    counter and never escalates to Discord: the exact silent-staleness blind
    spot this module exists to close, reopened on the return-value path
    (observed live 2026-05-22: ``subprocess_timeout`` after 659.5s).

    A non-dict, or a dict with a missing/unknown ``status``, is treated as a
    failure — defensive: an unexpected shape means ``train()`` did not clearly
    succeed, and a false "stuck" alert is far cheaper than silent staleness."""
    if not isinstance(metrics, dict):
        return True
    return metrics.get("status") not in _RETRAIN_OK_STATUSES
