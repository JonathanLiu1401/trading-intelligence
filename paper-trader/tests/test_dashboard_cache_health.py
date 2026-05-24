"""Tests for ``analytics.dashboard_cache_health.build_cache_health``.

The builder takes a ``dashboard._SWR_STATE``-shaped dict and emits an
operator-actionable cache-health snapshot. These tests pin the verdict
ladder, the aggregate roll-up, the deterministic ordering of the entries
list, and the degrade-safe contract (a corrupt entry surfaces as ERROR,
never crashes).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from paper_trader.analytics.dashboard_cache_health import (
    _FAIL_THRESHOLD,
    _STALE_AFTER_S,
    VERDICT_ERROR,
    VERDICT_FAILING,
    VERDICT_HEALTHY,
    VERDICT_NEVER_BUILT,
    VERDICT_STALE,
    build_cache_health,
)


def _entry(*, data=b"{}", ts=None, last_ok_ts=None, fail_count=0,
           last_error=None, last_error_ts=None) -> dict:
    """Build one ``_SWR_STATE``-shaped entry. Defaults match what
    ``dashboard._swr_entry`` creates."""
    return {
        "data": data,
        "status": 200,
        "ct": "application/json",
        "ts": ts if ts is not None else 0.0,
        "fut": None,
        "fail_count": fail_count,
        "last_error": last_error,
        "last_error_ts": (last_error_ts if last_error_ts is not None
                          else 0.0),
        "last_ok_ts": (last_ok_ts if last_ok_ts is not None else 0.0),
    }


_FIXED_NOW = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)
_FIXED_TS = _FIXED_NOW.timestamp()


class TestEmptyAndShapes:
    """NO_DATA + safe handling of malformed input."""

    def test_none_input_returns_no_data(self):
        out = build_cache_health(None, now=_FIXED_NOW)
        assert out["state"] == "NO_DATA"
        assert out["verdict"] == "NO_DATA"
        assert out["entries"] == []
        assert out["summary"]["total"] == 0

    def test_empty_dict_returns_no_data(self):
        out = build_cache_health({}, now=_FIXED_NOW)
        assert out["verdict"] == "NO_DATA"
        assert "No cached endpoints" in out["headline"]

    def test_non_dict_input_returns_no_data(self):
        # Defensive — a misuse must degrade, not crash.
        out = build_cache_health([1, 2, 3], now=_FIXED_NOW)  # type: ignore
        assert out["verdict"] == "NO_DATA"

    def test_thresholds_surfaced_in_snapshot(self):
        out = build_cache_health({}, now=_FIXED_NOW)
        assert out["fail_threshold"] == _FAIL_THRESHOLD
        assert out["stale_after_s"] == _STALE_AFTER_S


class TestPerEntryVerdict:
    """Per-entry verdict ladder boundaries — pinned in tests so future
    threshold tuning is explicit."""

    def test_never_built_when_data_is_none(self):
        state = {"state?": _entry(data=None)}
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["entries"][0]["verdict"] == VERDICT_NEVER_BUILT
        assert out["summary"]["never_built"] == 1
        assert out["summary"]["healthy"] == 0

    def test_healthy_fresh_entry(self):
        # Last good build 5s ago, no failures → HEALTHY.
        state = {"state?": _entry(ts=_FIXED_TS - 5.0,
                                   last_ok_ts=_FIXED_TS - 5.0)}
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["entries"][0]["verdict"] == VERDICT_HEALTHY
        assert out["entries"][0]["last_ok_age_s"] == 5.0

    def test_stale_just_past_threshold(self):
        # 601s > _STALE_AFTER_S (600s) → STALE.
        state = {"risk?": _entry(ts=_FIXED_TS - 601.0,
                                  last_ok_ts=_FIXED_TS - 601.0)}
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["entries"][0]["verdict"] == VERDICT_STALE

    def test_stale_threshold_is_strict(self):
        # Exactly at the threshold counts as HEALTHY (the ``> `` boundary
        # contract — only STRICTLY older than the threshold is STALE).
        state = {"k?": _entry(ts=_FIXED_TS - _STALE_AFTER_S,
                               last_ok_ts=_FIXED_TS - _STALE_AFTER_S)}
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["entries"][0]["verdict"] == VERDICT_HEALTHY

    def test_failing_at_threshold(self):
        # _FAIL_THRESHOLD consecutive errors → FAILING (>=, not >).
        state = {"k?": _entry(ts=_FIXED_TS - 5.0,
                               last_ok_ts=_FIXED_TS - 5.0,
                               fail_count=_FAIL_THRESHOLD,
                               last_error="boom",
                               last_error_ts=_FIXED_TS - 2.0)}
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["entries"][0]["verdict"] == VERDICT_FAILING
        assert out["entries"][0]["fail_count"] == _FAIL_THRESHOLD
        assert out["entries"][0]["last_error"] == "boom"
        assert out["entries"][0]["last_error_age_s"] == 2.0

    def test_failing_below_threshold_is_healthy(self):
        # Two consecutive failures (below 3) but data still fresh → HEALTHY.
        # Transient blips must NOT trip the persistent-failure verdict.
        state = {"k?": _entry(ts=_FIXED_TS - 5.0,
                               last_ok_ts=_FIXED_TS - 5.0,
                               fail_count=_FAIL_THRESHOLD - 1,
                               last_error="blip",
                               last_error_ts=_FIXED_TS - 1.0)}
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["entries"][0]["verdict"] == VERDICT_HEALTHY

    def test_failing_dominates_stale(self):
        # Old data AND high fail_count → FAILING wins (the actionable verdict).
        state = {"k?": _entry(ts=_FIXED_TS - 999.0,
                               last_ok_ts=_FIXED_TS - 999.0,
                               fail_count=10,
                               last_error="x",
                               last_error_ts=_FIXED_TS - 1.0)}
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["entries"][0]["verdict"] == VERDICT_FAILING

    def test_clock_skew_clamps_negative_age_to_zero(self):
        # Future ts (wall-clock stepped back) clamps to 0.0, not negative.
        state = {"k?": _entry(ts=_FIXED_TS + 5.0,
                               last_ok_ts=_FIXED_TS + 5.0)}
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["entries"][0]["last_ok_age_s"] == 0.0
        assert out["entries"][0]["verdict"] == VERDICT_HEALTHY

    def test_falls_back_to_ts_when_last_ok_ts_missing(self):
        # Legacy entries may lack last_ok_ts — must still age correctly.
        e = _entry(ts=_FIXED_TS - 50.0, last_ok_ts=None)
        # Drop the last_ok_ts key entirely to simulate a legacy entry.
        del e["last_ok_ts"]
        state = {"k?": e}
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["entries"][0]["verdict"] == VERDICT_HEALTHY
        assert out["entries"][0]["last_ok_age_s"] == 50.0

    def test_malformed_fail_count_coerces_to_zero(self):
        # A string "two" in fail_count must NOT crash the classifier.
        e = _entry(ts=_FIXED_TS - 5.0, last_ok_ts=_FIXED_TS - 5.0)
        e["fail_count"] = "two"  # corrupted
        state = {"k?": e}
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["entries"][0]["fail_count"] == 0
        assert out["entries"][0]["verdict"] == VERDICT_HEALTHY

    def test_non_dict_entry_surfaces_as_error(self):
        # A garbage value in the cache table — distinct ERROR verdict.
        state = {"k?": "not a dict"}
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["entries"][0]["verdict"] == VERDICT_ERROR
        assert out["summary"]["error"] == 1


class TestAggregateVerdict:
    """The roll-up verdict — what the operator sees first."""

    def test_all_healthy_is_healthy(self):
        state = {
            "a?": _entry(ts=_FIXED_TS - 1.0, last_ok_ts=_FIXED_TS - 1.0),
            "b?": _entry(ts=_FIXED_TS - 2.0, last_ok_ts=_FIXED_TS - 2.0),
        }
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["verdict"] == VERDICT_HEALTHY
        assert "2/2" in out["headline"]

    def test_one_failing_among_healthy_is_degraded(self):
        state = {
            "good?": _entry(ts=_FIXED_TS - 1.0, last_ok_ts=_FIXED_TS - 1.0),
            "bad?": _entry(ts=_FIXED_TS - 1.0, last_ok_ts=_FIXED_TS - 1.0,
                            fail_count=_FAIL_THRESHOLD,
                            last_error="x", last_error_ts=_FIXED_TS - 1.0),
        }
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["verdict"] == "DEGRADED"
        assert "1" in out["headline"]
        # The failing entry must surface FIRST in the entries list.
        assert out["entries"][0]["key"] == "bad?"
        assert out["entries"][0]["verdict"] == VERDICT_FAILING

    def test_all_failing_with_data_is_failed(self):
        state = {
            "a?": _entry(ts=_FIXED_TS - 1.0, last_ok_ts=_FIXED_TS - 1.0,
                          fail_count=_FAIL_THRESHOLD,
                          last_error="x", last_error_ts=_FIXED_TS - 1.0),
            "b?": _entry(ts=_FIXED_TS - 1.0, last_ok_ts=_FIXED_TS - 1.0,
                          fail_count=_FAIL_THRESHOLD * 5,
                          last_error="y", last_error_ts=_FIXED_TS - 1.0),
        }
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["verdict"] == "FAILED"
        assert "every cached endpoint" in out["headline"]

    def test_never_built_does_not_count_against_aggregate(self):
        # NEVER_BUILT entries are not "sick" — only data-bearing entries
        # can be FAILING/HEALTHY. Aggregate ignores NEVER_BUILT for the
        # ratio.
        state = {
            "cold?": _entry(data=None),
            "good?": _entry(ts=_FIXED_TS - 1.0, last_ok_ts=_FIXED_TS - 1.0),
        }
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["verdict"] == VERDICT_HEALTHY

    def test_stale_does_not_count_as_failing_for_aggregate(self):
        # A panel that's stale (last poll >10 min ago) is not failing —
        # likely just idle. Aggregate stays HEALTHY.
        state = {
            "stale?": _entry(ts=_FIXED_TS - 999.0, last_ok_ts=_FIXED_TS - 999.0),
            "fresh?": _entry(ts=_FIXED_TS - 1.0, last_ok_ts=_FIXED_TS - 1.0),
        }
        out = build_cache_health(state, now=_FIXED_NOW)
        assert out["verdict"] == VERDICT_HEALTHY
        assert out["summary"]["stale"] == 1
        assert out["summary"]["healthy"] == 1


class TestEntryOrdering:
    """Deterministic ordering — FAILING first, then ERROR, STALE,
    NEVER_BUILT, HEALTHY; ties broken by key. The operator's eye lands
    on the actionable entries first."""

    def test_failing_sorts_before_healthy(self):
        state = {
            "z-healthy?": _entry(ts=_FIXED_TS - 1.0, last_ok_ts=_FIXED_TS - 1.0),
            "a-failing?": _entry(ts=_FIXED_TS - 1.0, last_ok_ts=_FIXED_TS - 1.0,
                                  fail_count=_FAIL_THRESHOLD,
                                  last_error="x", last_error_ts=_FIXED_TS - 1.0),
        }
        out = build_cache_health(state, now=_FIXED_NOW)
        verdicts = [e["verdict"] for e in out["entries"]]
        assert verdicts == [VERDICT_FAILING, VERDICT_HEALTHY]

    def test_keys_sorted_within_verdict_group(self):
        state = {
            "z-healthy?": _entry(ts=_FIXED_TS - 1.0, last_ok_ts=_FIXED_TS - 1.0),
            "a-healthy?": _entry(ts=_FIXED_TS - 2.0, last_ok_ts=_FIXED_TS - 2.0),
        }
        out = build_cache_health(state, now=_FIXED_NOW)
        keys = [e["key"] for e in out["entries"]]
        assert keys == ["a-healthy?", "z-healthy?"]

    def test_full_ordering(self):
        # One of each verdict; mixed key ordering. Verdict order takes
        # priority over key order.
        state = {
            "z-fail?": _entry(ts=_FIXED_TS - 1.0, last_ok_ts=_FIXED_TS - 1.0,
                               fail_count=_FAIL_THRESHOLD, last_error="x",
                               last_error_ts=_FIXED_TS - 1.0),
            "a-cold?": _entry(data=None),
            "m-stale?": _entry(ts=_FIXED_TS - 999.0, last_ok_ts=_FIXED_TS - 999.0),
            "b-good?": _entry(ts=_FIXED_TS - 1.0, last_ok_ts=_FIXED_TS - 1.0),
        }
        out = build_cache_health(state, now=_FIXED_NOW)
        verdicts = [e["verdict"] for e in out["entries"]]
        keys = [e["key"] for e in out["entries"]]
        assert verdicts[0] == VERDICT_FAILING
        assert verdicts[1] == VERDICT_STALE
        assert verdicts[2] == VERDICT_NEVER_BUILT
        assert verdicts[3] == VERDICT_HEALTHY
        assert keys == ["z-fail?", "m-stale?", "a-cold?", "b-good?"]


class TestSummaryCounts:
    """Per-bucket totals — used by operator dashboards / Discord lines."""

    def test_counts_sum_to_total(self):
        state = {
            "a?": _entry(ts=_FIXED_TS - 1.0, last_ok_ts=_FIXED_TS - 1.0),
            "b?": _entry(ts=_FIXED_TS - 1.0, last_ok_ts=_FIXED_TS - 1.0,
                          fail_count=5, last_error="x",
                          last_error_ts=_FIXED_TS - 1.0),
            "c?": _entry(data=None),
            "d?": _entry(ts=_FIXED_TS - 999.0, last_ok_ts=_FIXED_TS - 999.0),
        }
        out = build_cache_health(state, now=_FIXED_NOW)
        s = out["summary"]
        assert s["total"] == 4
        assert (s["healthy"] + s["stale"] + s["failing"]
                + s["never_built"] + s["error"]) == s["total"]
        assert s["healthy"] == 1
        assert s["failing"] == 1
        assert s["never_built"] == 1
        assert s["stale"] == 1


class TestRealStateShape:
    """Smoke against the real ``dashboard._SWR_STATE`` factory shape so
    a future field-name change in dashboard.py is caught here."""

    def test_dashboard_swr_entry_consumed_correctly(self):
        # Reconstruct exactly what dashboard._swr_entry creates for a
        # fresh key. The builder must accept this shape without surfacing
        # an ERROR verdict.
        fresh_entry = {
            "data": None,
            "status": 200,
            "ct": "application/json",
            "ts": 0.0,
            "fut": None,
            "fail_count": 0,
            "last_error": None,
            "last_error_ts": 0.0,
            "last_ok_ts": 0.0,
        }
        out = build_cache_health({"healthz?": fresh_entry}, now=_FIXED_NOW)
        # Empty entry has data=None → NEVER_BUILT (not ERROR).
        assert out["entries"][0]["verdict"] == VERDICT_NEVER_BUILT
        assert out["summary"]["error"] == 0

    def test_default_now_uses_real_clock(self):
        # When ``now`` is omitted, the as_of stamp matches the real clock.
        out = build_cache_health({})
        # as_of is an ISO-8601 string in the recent past.
        as_of = datetime.fromisoformat(out["as_of"])
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        delta = (datetime.now(timezone.utc) - as_of).total_seconds()
        assert -2.0 < delta < 5.0


class TestDashboardEndpointWiring:
    """The ``/api/swr-cache-health`` endpoint must be registered and
    return the builder's snapshot verbatim (with the ``service`` field
    added — the existing notify_health / alarm_latches precedent). Pins
    the route name + envelope shape so a future refactor that drops the
    handler is caught by CI."""

    def test_endpoint_registered_and_returns_200(self):
        from paper_trader import dashboard
        client = dashboard.app.test_client()
        r = client.get("/api/swr-cache-health")
        assert r.status_code == 200
        body = r.get_json()
        assert isinstance(body, dict)
        # Envelope shape — the keys callers will read.
        assert body["service"] == "paper_trader"
        assert "verdict" in body
        assert "headline" in body
        assert "entries" in body
        assert "summary" in body
        assert "fail_threshold" in body
        assert "stale_after_s" in body

    def test_endpoint_reads_module_swr_state(self, monkeypatch):
        # Inject a synthetic _SWR_STATE so we can assert the endpoint
        # exposes its actual contents, not a stub.
        from paper_trader import dashboard
        synthetic = {
            "test-key?": {
                "data": b"{}",
                "status": 200,
                "ct": "application/json",
                "ts": time.time(),
                "fut": None,
                "fail_count": 5,  # FAILING — above threshold
                "last_error": "synthetic",
                "last_error_ts": time.time(),
                "last_ok_ts": time.time() - 1.0,
            }
        }
        monkeypatch.setattr(dashboard, "_SWR_STATE", synthetic)
        client = dashboard.app.test_client()
        r = client.get("/api/swr-cache-health")
        body = r.get_json()
        assert body["summary"]["failing"] == 1
        # FAILING entry surfaces FIRST in entries (the operator-actionable
        # ordering).
        assert body["entries"][0]["key"] == "test-key?"
        assert body["entries"][0]["verdict"] == VERDICT_FAILING
        # Aggregate verdict — every entry with data is failing → FAILED.
        assert body["verdict"] == "FAILED"
