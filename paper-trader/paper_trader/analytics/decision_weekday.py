"""Per-day-of-week decision distribution — pure builder.

The orthogonal complement to ``decision_clock.build_decision_clock``: that
module buckets the last N days of decisions by NY local **hour-of-day**
and flags an ``HOURLY_CONCENTRATION`` window (e.g. "20:00 ET has 80%
NO_DECISION"). It says nothing about *which weekday* is starved — a
Monday-after-open quota slump or a Friday-close parse-storm is invisible
to it because the same hour-of-day on the off-day washes the bucket out.

``build_decision_weekday`` buckets the same decision rows by NY local
**day-of-week** (Mon..Sun) and re-runs the identical NO_DECISION
sub-classification (``quota`` → ``host saturated`` → ``empty`` →
``parse_failed`` → ``other``) so the operator can see, for example, that
*Fridays* run 60% NO_DECISION because the quota budget is consistently
spent by close, distinct from the *evening hour* concentration
``decision_clock`` already flags.

State ladder mirrors ``decision_clock`` exactly:
  * ``INSUFFICIENT_DATA`` — fewer than ``MIN_TOTAL_DECISIONS`` rows
  * ``WEEKDAY_CONCENTRATION`` — at least one weekday (with
    ``>= MIN_WORST_BUCKET_SAMPLES`` samples) has
    ``>= WEEKDAY_CONCENTRATION_PCT`` NO_DECISION
  * ``EVEN_DISTRIBUTION`` — otherwise

Pure: no I/O, no DB, never raises. Caller passes
``store.recent_decisions(...)`` rows. NO_DECISION sub-classification is
delegated to ``decision_clock._classify_no_decision`` so the bucket
precedence (load-bearing, mutually exclusive) can never drift between
the hour and weekday surfaces — the same parity discipline
``hold_discipline`` ↔ ``loser_autopsy`` and ``decision_clock`` builder ↔
endpoint already follow.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .decision_clock import _classify_no_decision

try:
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover — zoneinfo is stdlib
    NY = timezone.utc

MIN_TOTAL_DECISIONS = 7
MIN_WORST_BUCKET_SAMPLES = 3
WEEKDAY_CONCENTRATION_PCT = 50.0

_WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _empty_buckets() -> list[dict]:
    return [
        {
            "weekday": i, "name": _WEEKDAY_NAMES[i],
            "total": 0, "filled": 0, "no_decision": 0,
            "host_saturated": 0, "empty_response": 0,
            "parse_failed": 0, "quota_exhausted": 0,
            "other_no_decision": 0,
        }
        for i in range(7)
    ]


def build_decision_weekday(decisions: list[dict],
                           now: datetime | None = None,
                           days: int = 28,
                           tz=NY) -> dict:
    """Pure per-day-of-week decision distribution + verdict.

    Args:
        decisions: ``store.recent_decisions(limit=...)`` rows.
        now: injectable for deterministic tests; defaults to UTC now.
        days: window in days. Clamped 7..90 (a week minimum so every
            weekday has at least one observation in steady state).
        tz: target timezone for weekday bucketing (defaults to NY).
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    days = max(7, min(int(days or 28), 90))
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
            wd = ts.astimezone(tz).weekday()
        except Exception:
            wd = ts.weekday()
        if not 0 <= wd <= 6:
            continue
        b = buckets[wd]
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

    worst = None
    for b in buckets:
        if b["total"] < MIN_WORST_BUCKET_SAMPLES:
            continue
        if worst is None or b["no_decision_pct"] > worst["no_decision_pct"]:
            worst = b
    worst_weekday = worst["weekday"] if worst else None

    if total < MIN_TOTAL_DECISIONS:
        verdict = "INSUFFICIENT_DATA"
        headline = (f"INSUFFICIENT_DATA — only {total} decisions in the "
                    f"last {days}d; verdict withheld below "
                    f"{MIN_TOTAL_DECISIONS}.")
    elif worst and worst["no_decision_pct"] >= WEEKDAY_CONCENTRATION_PCT:
        verdict = "WEEKDAY_CONCENTRATION"
        parts = []
        for k, label in (("host_saturated", "host"),
                         ("quota_exhausted", "quota"),
                         ("empty_response", "empty"),
                         ("parse_failed", "parse"),
                         ("other_no_decision", "other")):
            if worst[k]:
                parts.append(f"{worst[k]} {label}")
        breakdown = (", " + ", ".join(parts)) if parts else ""
        headline = (f"{worst['name']} has {worst['no_decision_pct']:.0f}% "
                    f"NO_DECISION over {worst['total']} samples"
                    f"{breakdown} — recurring weekday starvation.")
    else:
        verdict = "EVEN_DISTRIBUTION"
        headline = (f"EVEN_DISTRIBUTION — no weekday has "
                    f"≥{WEEKDAY_CONCENTRATION_PCT:.0f}% NO_DECISION over "
                    f"the last {days}d ({total} total).")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "days": days,
        "tz": getattr(tz, "key", str(tz)),
        "total_decisions": total,
        "buckets": buckets,
        "worst_weekday_local": worst_weekday,
        "verdict": verdict,
        "headline": headline,
    }
