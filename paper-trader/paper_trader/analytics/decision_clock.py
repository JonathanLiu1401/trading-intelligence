"""Per-hour-of-day decision distribution — pure builder.

The dashboard's ``/api/decision-clock`` already buckets the last N days of
decisions by NY local hour and computes a NO_DECISION concentration
verdict. That logic lives inline in ``dashboard.decision_clock_api`` and
is dashboard-only — the operator who lives in Discord never sees the
HOURLY_CONCENTRATION verdict that flags a recurring saturation window
(e.g. "hour 20:00 ET has 80% NO_DECISION").

This module pulls the categorisation + verdict logic into a pure
builder so:

  1. The reporter can call it from ``_decision_clock_line`` and emit
     the HOURLY_CONCENTRATION verdict to Discord (the established
     dashboard→Discord trajectory ``_host_pulse_line`` /
     ``_capital_pulse_line`` / ``_singleton_lock_line`` each followed,
     one operator-surface gap at a time).
  2. The verdict is offline-testable with exact-value asserts.

This builder is **independently maintained** from the inline endpoint
logic — both are tested at exact-value parity in
``tests/test_decision_clock_builder.py`` so a drift in either layer
fails loudly (the ``hold_discipline`` ↔ ``loser_autopsy`` no-drift
discipline applied to a builder ↔ endpoint pair).

Pure: no I/O, no network, no DB read — the caller passes decision
rows. Never raises (the ``_safe`` contract — a garbage row degrades
the row, never sinks the whole verdict).

State ladder, identical to the inline endpoint:
  * ``INSUFFICIENT_DATA`` — fewer than ``MIN_TOTAL_DECISIONS`` in window
  * ``HOURLY_CONCENTRATION`` — at least one bucket (with
    ``>= MIN_WORST_BUCKET_SAMPLES`` samples) has
    ``>= HOURLY_CONCENTRATION_PCT`` NO_DECISION
  * ``EVEN_DISTRIBUTION`` — otherwise

Bucket precedence within NO_DECISION (load-bearing, ordered most-specific
first; mirror of the inline endpoint after the 2026-05-18 quota fix):
``quota`` → ``host saturated`` / ``skipped claude call`` →
``no response`` → ``parse_failed`` / ``retry_failed`` → ``other``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover — zoneinfo is stdlib
    NY = timezone.utc

# Thresholds — module-level constants so tests read live values instead
# of hardcoding (the digital-intern "tests read live module constants"
# discipline).
MIN_TOTAL_DECISIONS = 5
MIN_WORST_BUCKET_SAMPLES = 3
HOURLY_CONCENTRATION_PCT = 50.0


def _empty_buckets() -> list[dict]:
    return [
        {
            "hour": h, "total": 0, "filled": 0, "no_decision": 0,
            "host_saturated": 0, "empty_response": 0,
            "parse_failed": 0, "quota_exhausted": 0,
            "other_no_decision": 0,
        }
        for h in range(24)
    ]


def _classify_no_decision(reasoning: str) -> str:
    """Bucket a NO_DECISION reasoning string. Branches MUST be mutually
    exclusive — quota first (most specific; doesn't contain the other
    tokens but a future format change must not silently re-merge them).
    """
    if "quota" in reasoning:
        return "quota_exhausted"
    if "host saturated" in reasoning or "skipped claude call" in reasoning:
        return "host_saturated"
    if "no response" in reasoning:
        return "empty_response"
    if "parse_failed" in reasoning or "retry_failed" in reasoning:
        return "parse_failed"
    return "other_no_decision"


def build_decision_clock(decisions: list[dict],
                         now: datetime | None = None,
                         days: int = 7,
                         tz=NY) -> dict:
    """Pure per-hour-of-day decision distribution + verdict.

    Args:
        decisions: ``store.recent_decisions(limit=...)`` rows
            (newest-first; any order is fine — we re-bucket).
        now: injectable for deterministic tests; defaults to UTC now.
        days: window in days. Clamped 1..30. Decisions older than
            ``now - days`` are excluded.
        tz: target timezone for hour bucketing (defaults to NY market).

    Returns a dict identical in shape to ``/api/decision-clock``:
        {as_of, days, tz, total_decisions, buckets[24],
         worst_hour_local, verdict, headline}

    ``verdict`` is an enum-clean string (``INSUFFICIENT_DATA`` /
    ``HOURLY_CONCENTRATION`` / ``EVEN_DISTRIBUTION``) — the inline
    endpoint's ``verdict`` field embeds detail in the HOURLY_CONCENTRATION
    case for backwards-compat; here ``verdict`` is enum-only and
    ``headline`` carries the human string. The reporter's Discord line
    keys off ``verdict``, not ``headline``, so an enum is required.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    days = max(1, min(int(days or 7), 30))
    cutoff = now - timedelta(days=days)

    buckets = _empty_buckets()
    total = 0

    for d in decisions or []:
        try:
            ts_raw = d.get("timestamp") if isinstance(d, dict) else None
            if not ts_raw:
                continue
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        if ts < cutoff:
            continue
        try:
            hour = ts.astimezone(tz).hour
        except Exception:
            hour = ts.hour
        if not 0 <= hour <= 23:
            continue
        b = buckets[hour]
        b["total"] += 1
        total += 1
        action = (d.get("action_taken") or "")
        reasoning = (d.get("reasoning") or "")
        if "FILLED" in action:
            b["filled"] += 1
        elif action == "NO_DECISION":
            b["no_decision"] += 1
            b[_classify_no_decision(reasoning)] += 1

    for b in buckets:
        n = b["total"]
        b["fill_rate_pct"] = round(100.0 * b["filled"] / n, 1) if n else 0.0
        b["no_decision_pct"] = round(100.0 * b["no_decision"] / n, 1) if n else 0.0
        b["host_saturated_pct"] = round(100.0 * b["host_saturated"] / n, 1) if n else 0.0

    worst = None
    for b in buckets:
        if b["total"] < MIN_WORST_BUCKET_SAMPLES:
            continue
        if worst is None or b["no_decision_pct"] > worst["no_decision_pct"]:
            worst = b
    worst_hour = worst["hour"] if worst else None

    if total < MIN_TOTAL_DECISIONS:
        verdict = "INSUFFICIENT_DATA"
        headline = (f"INSUFFICIENT_DATA — only {total} decisions in the "
                    f"last {days}d; verdict withheld below "
                    f"{MIN_TOTAL_DECISIONS}.")
    elif worst and worst["no_decision_pct"] >= HOURLY_CONCENTRATION_PCT:
        verdict = "HOURLY_CONCENTRATION"
        # Breakdown of the worst bucket's NO_DECISION sub-buckets so the
        # operator can immediately see *why* this hour is starved (host /
        # quota / empty / parse) instead of just "it's bad".
        parts = []
        for k, label in (("host_saturated", "host"),
                         ("quota_exhausted", "quota"),
                         ("empty_response", "empty"),
                         ("parse_failed", "parse"),
                         ("other_no_decision", "other")):
            if worst[k]:
                parts.append(f"{worst[k]} {label}")
        breakdown = (", " + ", ".join(parts)) if parts else ""
        headline = (f"hour {worst['hour']:02d}:00 ET has "
                    f"{worst['no_decision_pct']:.0f}% NO_DECISION over "
                    f"{worst['total']} samples{breakdown} — recurring "
                    f"saturation window; investigate concurrent jobs.")
    else:
        verdict = "EVEN_DISTRIBUTION"
        headline = (f"EVEN_DISTRIBUTION — no hour-of-day has "
                    f"≥{HOURLY_CONCENTRATION_PCT:.0f}% NO_DECISION over "
                    f"the last {days}d ({total} total).")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "days": days,
        "tz": getattr(tz, "key", str(tz)),
        "total_decisions": total,
        "buckets": buckets,
        "worst_hour_local": worst_hour,
        "verdict": verdict,
        "headline": headline,
    }
