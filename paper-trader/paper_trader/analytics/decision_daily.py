"""Per-calendar-day NO_DECISION timeseries — pure builder.

The orthogonal **time** complement to ``decision_clock`` (hour-of-day) and
``decision_weekday`` (day-of-week). Those two surface *which recurring
slot* is starved; neither answers the question an operator asks after a
fix lands: **is the storm getting better or worse day over day?** Without a
calendar-day series, a 50% drop in NO_DECISION rate after a commit lands is
invisible — the hour and weekday buckets continue showing the *historical*
saturation window for as long as the window stays in scope.

``build_decision_daily`` buckets the last N calendar days of decisions in
NY local time and re-runs the identical NO_DECISION sub-classification
(``quota`` → ``host saturated`` → ``empty`` → ``parse_failed`` →
``other``) delegated to ``decision_clock._classify_no_decision`` — so the
bucket precedence cannot drift across the three decision-* surfaces.

Trend verdict, distinct from the "find the worst bucket" verdict the
hour/weekday surfaces emit. A calendar-day series is a *timeseries*; the
operator signal is **direction**:

  * ``INSUFFICIENT_DATA`` — fewer than ``MIN_TOTAL_DECISIONS`` rows total,
    or fewer than ``MIN_TREND_DAYS`` days with any decisions.
  * ``TREND_WORSENING`` — the more-recent half of the window has
    NO_DECISION rate ``>= TREND_DELTA_PCT`` points above the earlier half
    (and earlier-half had at least ``MIN_HALF_SAMPLES`` samples so the
    baseline is real).
  * ``TREND_IMPROVING`` — the more-recent half is ``>= TREND_DELTA_PCT``
    points below the earlier half (same sample gate on earlier half).
  * ``STABLE`` — neither.

A "find the worst day" verdict was deliberately rejected: a single bad
day in a low-volume window can dominate without representing a real
operational change. The split-halves trend is robust to one-day spikes.

Pure: no I/O, no DB, never raises. Caller passes
``store.recent_decisions(limit=...)`` rows; bucket precedence is shared
with the hour/weekday surfaces via ``_classify_no_decision`` (the same
parity discipline ``hold_discipline`` ↔ ``loser_autopsy`` and
``decision_clock`` builder ↔ endpoint already follow).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .decision_clock import _classify_no_decision

try:
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover — zoneinfo is stdlib
    NY = timezone.utc

MIN_TOTAL_DECISIONS = 10
MIN_TREND_DAYS = 4
MIN_HALF_SAMPLES = 5
TREND_DELTA_PCT = 15.0
DEFAULT_DAYS = 14
MAX_DAYS = 60


def _empty_bucket(date_str: str) -> dict:
    return {
        "date": date_str,
        "total": 0, "filled": 0, "no_decision": 0,
        "host_saturated": 0, "empty_response": 0,
        "parse_failed": 0, "quota_exhausted": 0,
        "other_no_decision": 0,
    }


def build_decision_daily(decisions: list[dict],
                         now: datetime | None = None,
                         days: int = DEFAULT_DAYS,
                         tz=NY) -> dict:
    """Pure per-calendar-day decision distribution + trend verdict.

    Args:
        decisions: ``store.recent_decisions(limit=...)`` rows (any order).
        now: injectable for deterministic tests; defaults to UTC now.
        days: window in calendar days. Clamped ``1..MAX_DAYS``.
        tz: target timezone for date bucketing (defaults to NY market).

    Returns:
        {
          "as_of": ISO timestamp,
          "days": int,
          "tz": str,
          "total_decisions": int,
          "buckets": [{date, total, filled, no_decision, host_saturated,
                       empty_response, parse_failed, quota_exhausted,
                       other_no_decision, fill_rate_pct, no_decision_pct,
                       host_saturated_pct}, ...],   # oldest → newest
          "earlier_no_decision_pct": float | None,
          "recent_no_decision_pct": float | None,
          "delta_pct": float | None,
          "verdict": "INSUFFICIENT_DATA" | "TREND_WORSENING" |
                     "TREND_IMPROVING" | "STABLE",
          "headline": str,
        }
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    days = max(1, min(int(days or DEFAULT_DAYS), MAX_DAYS))

    try:
        now_local = now.astimezone(tz)
    except Exception:
        now_local = now
    end_date = now_local.date()
    start_date = end_date - timedelta(days=days - 1)

    date_strs = [(start_date + timedelta(days=i)).isoformat()
                 for i in range(days)]
    buckets_by_date: dict[str, dict] = {
        ds: _empty_bucket(ds) for ds in date_strs
    }
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
        try:
            local_date = ts.astimezone(tz).date().isoformat()
        except Exception:
            local_date = ts.date().isoformat()
        b = buckets_by_date.get(local_date)
        if b is None:
            continue
        b["total"] += 1
        total += 1
        action = (d.get("action_taken") or "")
        reasoning = (d.get("reasoning") or "")
        if "FILLED" in action:
            b["filled"] += 1
        elif action == "NO_DECISION":
            b["no_decision"] += 1
            b[_classify_no_decision(reasoning)] += 1

    buckets = [buckets_by_date[ds] for ds in date_strs]
    for b in buckets:
        n = b["total"]
        b["fill_rate_pct"] = round(100.0 * b["filled"] / n, 1) if n else 0.0
        b["no_decision_pct"] = round(100.0 * b["no_decision"] / n, 1) if n else 0.0
        b["host_saturated_pct"] = round(100.0 * b["host_saturated"] / n, 1) if n else 0.0

    active_days = sum(1 for b in buckets if b["total"] > 0)
    half = len(buckets) // 2
    earlier_slice = buckets[:half]
    recent_slice = buckets[half:]

    earlier_total = sum(b["total"] for b in earlier_slice)
    recent_total = sum(b["total"] for b in recent_slice)
    earlier_nd = sum(b["no_decision"] for b in earlier_slice)
    recent_nd = sum(b["no_decision"] for b in recent_slice)

    earlier_pct = (round(100.0 * earlier_nd / earlier_total, 1)
                   if earlier_total else None)
    recent_pct = (round(100.0 * recent_nd / recent_total, 1)
                  if recent_total else None)

    delta_pct: float | None
    if earlier_pct is not None and recent_pct is not None:
        delta_pct = round(recent_pct - earlier_pct, 1)
    else:
        delta_pct = None

    if (total < MIN_TOTAL_DECISIONS
            or active_days < MIN_TREND_DAYS
            or earlier_total < MIN_HALF_SAMPLES
            or recent_total < MIN_HALF_SAMPLES
            or delta_pct is None):
        verdict = "INSUFFICIENT_DATA"
        headline = (f"INSUFFICIENT_DATA — {total} decisions across "
                    f"{active_days} active day(s) in the last {days}d; "
                    f"trend verdict requires ≥{MIN_TOTAL_DECISIONS} total "
                    f"and ≥{MIN_HALF_SAMPLES} samples in each half.")
    elif delta_pct >= TREND_DELTA_PCT:
        verdict = "TREND_WORSENING"
        headline = (f"TREND_WORSENING — NO_DECISION rate up "
                    f"{delta_pct:+.1f}pts from {earlier_pct:.1f}% "
                    f"(earlier {earlier_total} samples) to "
                    f"{recent_pct:.1f}% (recent {recent_total} samples) "
                    f"over the last {days}d.")
    elif delta_pct <= -TREND_DELTA_PCT:
        verdict = "TREND_IMPROVING"
        headline = (f"TREND_IMPROVING — NO_DECISION rate down "
                    f"{delta_pct:+.1f}pts from {earlier_pct:.1f}% "
                    f"(earlier {earlier_total} samples) to "
                    f"{recent_pct:.1f}% (recent {recent_total} samples) "
                    f"over the last {days}d.")
    else:
        verdict = "STABLE"
        headline = (f"STABLE — NO_DECISION rate change "
                    f"{delta_pct:+.1f}pts (earlier {earlier_pct:.1f}% / "
                    f"recent {recent_pct:.1f}%) below the "
                    f"±{TREND_DELTA_PCT:.0f}pt threshold over {days}d.")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "days": days,
        "tz": getattr(tz, "key", str(tz)),
        "total_decisions": total,
        "active_days": active_days,
        "buckets": buckets,
        "earlier_no_decision_pct": earlier_pct,
        "recent_no_decision_pct": recent_pct,
        "earlier_samples": earlier_total,
        "recent_samples": recent_total,
        "delta_pct": delta_pct,
        "verdict": verdict,
        "headline": headline,
    }
