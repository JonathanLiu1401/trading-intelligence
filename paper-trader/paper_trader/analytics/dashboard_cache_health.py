"""Dashboard SWR-cache health — operator view of which cached endpoints are
healthy, stale, or silently failing.

The dashboard wraps every slow JSON endpoint with ``@swr_cached`` (see
``dashboard.swr_cached`` / ``_SWR_STATE``). Each entry tracks the last
successful build timestamp, a per-key ``fail_count`` for consecutive
background-rebuild failures, and the last error text. When a background
build keeps raising (a yfinance hang, a digital-intern DB lock, an
analytics-builder regression) the panel stays on its *last good* payload
indefinitely while the failures accumulate — silently. The existing
``_SWR_FAIL_LOG_EVERY`` throttle writes a stderr line on the 1st and every
10th consecutive failure, but the operator must be tailing the log to see
it; nothing surfaces this on the dashboard itself.

This builder takes the raw ``_SWR_STATE`` dict (the dashboard module's
authoritative cache table) plus a wall-clock and produces a
verdict-tagged snapshot — same advisory contract as the existing
``alarm_latch_state`` / ``notify_health`` surfaces: pure read, never
raises, intentionally NOT persisted (cache health is a property of the
running process; a fresh process re-establishes it on its first poll).

Per-entry verdict ladder:

  * ``NEVER_BUILT``  — entry has no ``data``; the endpoint has not been
    polled since boot. Distinct from FAILING — nothing has tried yet, so
    there is nothing to be sick about.
  * ``FAILING``     — ``fail_count >= _FAIL_THRESHOLD`` consecutive
    background rebuild failures since the last good build. The panel
    is serving a stale-but-frozen payload and will NOT self-heal until
    the underlying handler stops raising. The operator-actionable bucket.
  * ``STALE``       — entry has data and the last successful build is
    older than ``_STALE_AFTER_S``. Cause is usually "nothing is polling
    this panel" (TTL refresh fires on demand, not periodically), so the
    last good payload is older than its TTL suggests. Distinct from
    FAILING — no errors have been recorded.
  * ``HEALTHY``     — entry has data, last good build is fresh, fail_count
    below the threshold. Default state for an actively-served panel.

Aggregate verdict:

  * ``HEALTHY``    — no FAILING entries.
  * ``DEGRADED``   — at least one FAILING entry but at least one HEALTHY.
  * ``FAILED``     — every entry that has data is FAILING (catastrophic
    backend-wide regression — the dashboard is effectively dark).
  * ``NO_DATA``    — empty state (a fresh-boot dashboard before any poll).

Thresholds are module-level constants so future tuning is one edit.

Pure: takes ``_SWR_STATE``-shaped input + optional ``now`` (test
injection). Does NOT import dashboard.py — that would be a circular
import; the dashboard module imports this helper, not the other way.
Degrade-safe: any per-entry parse fault is caught and the entry is
marked ``ERROR`` (a fifth verdict, distinct from FAILING — the error is
in OUR code, not the cached endpoint's), never raises. Mirrors the
``passive_signal_density`` / ``alarm_latch_state`` discipline.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

# Per-entry verdict thresholds.

# Number of consecutive background-rebuild failures before an entry is
# tagged ``FAILING``. Single failures are common (a transient yfinance
# blip / digital-intern lock); the persistent class is what the operator
# needs to act on. Matches ``dashboard._SWR_FAIL_LOG_EVERY``'s "1st then
# every 10th" logging-throttle intent — 3 is "this is sustained, not a
# blip".
_FAIL_THRESHOLD = 3

# Age (seconds) past which a successful-build is considered STALE. The
# longest swr_cached TTL the dashboard currently sets is 60s
# (watchlist-opportunities). 600s = 10 min gives a comfortable
# 10x-TTL margin so a healthy panel that simply isn't being polled
# regularly doesn't tip into STALE noise.
_STALE_AFTER_S = 600.0

# Aggregate-verdict thresholds — see the module docstring.
_AGG_FAILED_RATIO = 1.0   # 100% of entries with data are failing
_AGG_DEGRADED_MIN = 1     # at least one failing entry

# Verdict names — exported so callers can match by string (the
# ``passive_signal_density`` precedent).
VERDICT_HEALTHY = "HEALTHY"
VERDICT_STALE = "STALE"
VERDICT_FAILING = "FAILING"
VERDICT_NEVER_BUILT = "NEVER_BUILT"
VERDICT_ERROR = "ERROR"


def _safe_float(x) -> float | None:
    """Coerce to float or return ``None`` (mirrors ``_safe_int`` /
    ``_coerce_signal_count`` pattern elsewhere in this stack)."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _classify_entry(entry: dict, now_ts: float) -> dict:
    """Classify a single ``_SWR_STATE`` entry. Returns a dict with the
    verdict + diagnostic fields. Never raises — any parse fault is
    surfaced as ``ERROR``."""
    try:
        has_data = entry.get("data") is not None
        ts = _safe_float(entry.get("ts"))
        last_ok_ts = _safe_float(entry.get("last_ok_ts"))
        last_error_ts = _safe_float(entry.get("last_error_ts"))
        fail_count = entry.get("fail_count") or 0
        try:
            fail_count = int(fail_count)
        except (TypeError, ValueError):
            fail_count = 0
        last_error = entry.get("last_error")
        if not isinstance(last_error, str):
            last_error = None

        # Prefer the explicit ``last_ok_ts`` (tracks every successful build,
        # never overwritten by failures); fall back to ``ts`` (the cache-
        # write timestamp, which equals last good build's ts in steady state
        # but legacy entries may lack last_ok_ts).
        ok_ref = last_ok_ts if last_ok_ts is not None else ts
        ok_age_s = (now_ts - ok_ref) if ok_ref is not None else None
        # A wall-clock step-back (the documented clock-skew hazard) clamps
        # to 0 rather than rendering a negative age.
        if ok_age_s is not None and ok_age_s < 0:
            ok_age_s = 0.0
        err_age_s = (now_ts - last_error_ts) if last_error_ts is not None else None
        if err_age_s is not None and err_age_s < 0:
            err_age_s = 0.0

        if not has_data:
            verdict = VERDICT_NEVER_BUILT
        elif fail_count >= _FAIL_THRESHOLD:
            verdict = VERDICT_FAILING
        elif ok_age_s is not None and ok_age_s > _STALE_AFTER_S:
            verdict = VERDICT_STALE
        else:
            verdict = VERDICT_HEALTHY

        return {
            "verdict": verdict,
            "has_data": has_data,
            "last_ok_age_s": round(ok_age_s, 1) if ok_age_s is not None else None,
            "fail_count": fail_count,
            "last_error": last_error,
            "last_error_age_s": (round(err_age_s, 1) if err_age_s is not None
                                 else None),
        }
    except Exception as e:
        # An entry whose shape we cannot parse — surface as ERROR (distinct
        # from FAILING; the failure is in our code, not the cached endpoint).
        return {
            "verdict": VERDICT_ERROR,
            "has_data": False,
            "last_ok_age_s": None,
            "fail_count": 0,
            "last_error": f"classify error: {type(e).__name__}: {e}"[:200],
            "last_error_age_s": None,
        }


def build_cache_health(swr_state: dict | None,
                       now: datetime | None = None) -> dict:
    """Compose a dashboard-cache-health snapshot from the raw ``_SWR_STATE``.

    ``swr_state`` — the dashboard module's ``_SWR_STATE`` dict (keys are
    ``"<endpoint-name>?<query-string>"``; values are the per-entry state
    dict). Pass ``None`` / empty for the NO_DATA verdict.

    ``now`` — wall-clock for age calculations (test injection). Defaults
    to ``datetime.now(timezone.utc)`` and is converted to a monotonic-ish
    float via ``.timestamp()`` so the per-entry ``ts`` floats (set by
    ``time.time()`` in the dashboard module) can be subtracted directly.
    A naive ``now`` is treated as UTC, mirroring ``_signal_age_str`` /
    ``_hold_age_str``.

    Returned shape mirrors the existing ``alarm_latch_state`` /
    ``passive_signal_density`` snapshots — ``as_of``, ``state``,
    ``verdict``, ``headline``, ``entries`` (per-key list, sorted by
    verdict-then-key for deterministic test assertions), ``summary``
    counter dict, and pinned thresholds.

    Pure read; never raises; every field is degrade-safe.
    """
    if now is None:
        now_dt = datetime.now(timezone.utc)
    else:
        now_dt = now
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
    now_ts = now_dt.timestamp()
    as_of = now_dt.isoformat(timespec="seconds")

    # Defensive — caller may pass None or a non-dict.
    if not isinstance(swr_state, dict) or not swr_state:
        return {
            "as_of": as_of,
            "state": "NO_DATA",
            "verdict": "NO_DATA",
            "headline": (
                "No cached endpoints yet — dashboard has not "
                "served any @swr_cached panel since boot."
            ),
            "entries": [],
            "summary": {
                "total": 0,
                "healthy": 0,
                "stale": 0,
                "failing": 0,
                "never_built": 0,
                "error": 0,
            },
            "fail_threshold": _FAIL_THRESHOLD,
            "stale_after_s": _STALE_AFTER_S,
        }

    entries: list[dict] = []
    counts = {
        VERDICT_HEALTHY: 0,
        VERDICT_STALE: 0,
        VERDICT_FAILING: 0,
        VERDICT_NEVER_BUILT: 0,
        VERDICT_ERROR: 0,
    }
    for key, entry in swr_state.items():
        if not isinstance(entry, dict):
            # A non-dict value in the cache table — unrecognisable shape;
            # surface as ERROR so the operator sees the corruption.
            classified = {
                "verdict": VERDICT_ERROR,
                "has_data": False,
                "last_ok_age_s": None,
                "fail_count": 0,
                "last_error": f"non-dict cache entry: {type(entry).__name__}",
                "last_error_age_s": None,
            }
        else:
            classified = _classify_entry(entry, now_ts)
        entries.append({"key": str(key), **classified})
        counts[classified["verdict"]] = counts.get(classified["verdict"], 0) + 1

    total = len(entries)
    n_failing = counts[VERDICT_FAILING]
    n_healthy = counts[VERDICT_HEALTHY]
    n_stale = counts[VERDICT_STALE]
    n_never = counts[VERDICT_NEVER_BUILT]
    n_error = counts[VERDICT_ERROR]

    # Aggregate verdict — see module docstring.
    n_with_data = n_healthy + n_stale + n_failing
    if n_failing == 0:
        verdict = VERDICT_HEALTHY
    elif n_with_data > 0 and n_failing / n_with_data >= _AGG_FAILED_RATIO:
        verdict = "FAILED"
    else:
        verdict = "DEGRADED"

    if verdict == VERDICT_HEALTHY:
        headline = (
            f"{n_healthy}/{total} cached endpoints healthy"
            + (f" ({n_stale} stale)" if n_stale else "")
            + (f" ({n_never} never-built)" if n_never else "")
            + (f" ({n_error} ERROR)" if n_error else "")
            + "."
        )
    elif verdict == "FAILED":
        headline = (
            f"FAILED — every cached endpoint with data is FAILING "
            f"({n_failing}/{n_with_data} consecutive-failures ≥ "
            f"{_FAIL_THRESHOLD}); dashboard is effectively dark."
        )
    else:
        headline = (
            f"DEGRADED — {n_failing} endpoint{'s' if n_failing != 1 else ''} "
            f"FAILING (≥{_FAIL_THRESHOLD} consecutive errors); "
            f"{n_healthy}/{total} still healthy."
        )

    # Sort entries verdict-first (so FAILING surfaces at the top of the
    # list), then by key for deterministic ordering.
    _VERDICT_ORDER = {
        VERDICT_FAILING: 0,
        VERDICT_ERROR: 1,
        VERDICT_STALE: 2,
        VERDICT_NEVER_BUILT: 3,
        VERDICT_HEALTHY: 4,
    }
    entries.sort(key=lambda e: (_VERDICT_ORDER.get(e["verdict"], 99), e["key"]))

    return {
        "as_of": as_of,
        "state": "STABLE",
        "verdict": verdict,
        "headline": headline,
        "entries": entries,
        "summary": {
            "total": total,
            "healthy": n_healthy,
            "stale": n_stale,
            "failing": n_failing,
            "never_built": n_never,
            "error": n_error,
        },
        "fail_threshold": _FAIL_THRESHOLD,
        "stale_after_s": _STALE_AFTER_S,
    }


if __name__ == "__main__":  # smoke test against the live dashboard state
    import json

    from paper_trader.dashboard import _SWR_STATE  # type: ignore

    print(json.dumps(build_cache_health(_SWR_STATE),
                     indent=2, default=str))
