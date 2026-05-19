"""Quota-exhaustion burn-rate — rolling-window view of the dominant
NO_DECISION cause.

``decision_clock`` buckets by hour-of-day; ``decision_weekday`` by day-of-week.
Both surface *which* hour or weekday is starved, but neither tells the
operator the answer to the question they actually ask first: **"right now —
in the last 6h / 24h / 72h — is quota exhaustion my dominant failure
mode, or is something else going on?"** When the answer is "yes, quota",
the lever is upgrade plan / wait / reduce concurrency; when the answer is
"no, host saturated", the lever is kill review agents. Conflating the two
(the documented historical bug, MEMORY: "NO_DECISION = quota, not JSON")
costs the operator engineering time on the wrong root cause.

This builder is the rolling-window orthogonal complement: for each
configured window it reports total decisions, NO_DECISION count, and the
split into ``quota_exhausted`` vs ``other_no_decision`` (with the same
``decision_clock._classify_no_decision`` precedence so the bucket
definitions never drift across surfaces — the parity discipline
``hold_discipline``↔``loser_autopsy`` already follow).

Pure: no I/O, no DB, never raises. Caller passes
``store.recent_decisions(...)`` rows (newest-first; any order works — we
re-filter by timestamp). ``now`` is injectable for deterministic tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .decision_clock import _classify_no_decision

DEFAULT_WINDOWS_HOURS = (6, 24, 72)
# At least this many NO_DECISION samples in a window before we report a
# quota_pct verdict — below this the percentage is dominated by sample
# noise and would mislead the operator.
MIN_NO_DECISION_SAMPLES = 3
# A window's quota share at-or-above this percentage flips the verdict to
# QUOTA_DOMINANT — the documented historical bug (NO_DECISION blamed on
# JSON parsing while quota was the actual cause) reliably presents as
# >=70% quota share in the affected window.
QUOTA_DOMINANT_PCT = 70.0


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _window(decisions: list[dict], cutoff: datetime) -> dict:
    """Aggregate one window (decisions newer than ``cutoff``)."""
    total = 0
    filled = 0
    no_decision = 0
    quota = 0
    host = 0
    empty = 0
    parse = 0
    other = 0
    for d in decisions:
        ts = _parse_ts(d.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        total += 1
        action = (d.get("action_taken") or "").strip()
        if "FILLED" in action.upper():
            filled += 1
            continue
        if action == "NO_DECISION":
            no_decision += 1
            bucket = _classify_no_decision(d.get("reasoning") or "")
            if bucket == "quota_exhausted":
                quota += 1
            elif bucket == "host_saturated":
                host += 1
            elif bucket == "empty_response":
                empty += 1
            elif bucket == "parse_failed":
                parse += 1
            else:
                other += 1
    return {
        "total": total,
        "filled": filled,
        "no_decision": no_decision,
        "quota_exhausted": quota,
        "host_saturated": host,
        "empty_response": empty,
        "parse_failed": parse,
        "other_no_decision": other,
    }


def _verdict(agg: dict) -> tuple[str, float | None, str]:
    """Return (verdict, quota_pct, headline) for an aggregated window."""
    nd = agg["no_decision"]
    if nd < MIN_NO_DECISION_SAMPLES:
        return ("LOW_SAMPLES", None,
                f"Only {nd} NO_DECISION samples in this window — not enough "
                f"to call a dominant cause.")
    quota_pct = round(agg["quota_exhausted"] / nd * 100.0, 1)
    if quota_pct >= QUOTA_DOMINANT_PCT:
        return ("QUOTA_DOMINANT", quota_pct,
                f"QUOTA_DOMINANT — {quota_pct:.0f}% of the {nd} NO_DECISIONs "
                f"in this window were quota-exhausted; the lever is upgrade "
                f"plan / wait / reduce concurrent Opus agents (NOT a parser "
                f"fix).")
    return ("MIXED", quota_pct,
            f"MIXED — {quota_pct:.0f}% of {nd} NO_DECISIONs were quota; "
            f"other causes dominate (host_saturated / empty / parse).")


def build_quota_burnrate(
    decisions: list[dict],
    now: datetime | None = None,
    windows_hours: tuple[int, ...] = DEFAULT_WINDOWS_HOURS,
) -> dict:
    """Per-rolling-window quota-exhaustion burn-rate verdict.

    Args:
        decisions: ``store.recent_decisions(limit=...)`` rows. Newest-first
            preferred; the builder filters by timestamp so any order works.
        now: injectable for deterministic tests; defaults to UTC now.
        windows_hours: rolling lookback windows to aggregate, in hours
            (default 6h / 24h / 72h).

    Returns a JSON-ready dict::

      {
        "as_of": "<iso>",
        "windows": [
          {
            "hours": 6,
            "total": int, "filled": int, "no_decision": int,
            "quota_exhausted": int, "host_saturated": int,
            "empty_response": int, "parse_failed": int,
            "other_no_decision": int,
            "quota_pct": float | None,
            "verdict": "QUOTA_DOMINANT" | "MIXED" | "LOW_SAMPLES",
            "headline": "...",
          },
          ...
        ],
      }
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    windows: list[dict] = []
    for h in windows_hours:
        try:
            hi = int(h)
        except (TypeError, ValueError):
            continue
        if hi <= 0:
            continue
        cutoff = now - timedelta(hours=hi)
        agg = _window(decisions or [], cutoff)
        verdict, quota_pct, headline = _verdict(agg)
        windows.append({
            "hours": hi,
            **agg,
            "quota_pct": quota_pct,
            "verdict": verdict,
            "headline": headline,
        })
    return {
        "as_of": now.isoformat(timespec="seconds"),
        "windows": windows,
    }
