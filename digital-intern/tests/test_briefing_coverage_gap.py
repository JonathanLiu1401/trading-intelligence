"""Adaptive briefing lookback + coverage-gap banner.

Phase-3 evidence (this repo's live `briefings` table): consecutive heartbeat
posts arrived 41.2h and 31.9h apart vs the 5h target under OOM-restart churn.
When a briefing finally lands after a 32h blackout the daemon still pulled
only the last 5h of articles and the digest gave the consuming analyst *no
indication they had been dark* — the "last 5h" framing silently understated a
32h coverage hole.

These three pure helpers (mirroring the `_format_source_health_summary` /
`_initial_heartbeat_last` style — deterministic, live module constants, no
threads) make a late briefing both honest and useful:

* ``_briefing_gap_hours``        — hours since the last persisted briefing,
  ``None`` when unknown/unparseable/future (degrade to normal behaviour).
* ``_briefing_lookback_hours``   — article window: unchanged 5h on a healthy
  cadence (no regression on the common path); widens to cover the real gap
  when overdue, hard-capped at 24h (the same ceiling
  ``get_top_for_briefing`` already enforces via the published-staleness
  filter — no new stale-news risk, all four load-bearing invariants
  untouched: this path never writes articles / ai_score / ml_score /
  score_source).
* ``_coverage_gap_banner``       — one-line analyst warning, empty on a
  healthy cadence so an on-time briefing stays clean.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import daemon


_NOW = datetime(2026, 5, 17, 13, 0, 0, tzinfo=timezone.utc)


def _ago(hours: float) -> str:
    return (_NOW - timedelta(hours=hours)).isoformat()


# ── _briefing_gap_hours ──────────────────────────────────────────────────────

def test_gap_none_when_no_prior_briefing():
    assert daemon._briefing_gap_hours(None, _NOW) is None
    assert daemon._briefing_gap_hours("", _NOW) is None


def test_gap_none_when_unparseable_or_future():
    assert daemon._briefing_gap_hours("not-a-date", _NOW) is None
    # A future ts (clock skew / bad row) must not yield a negative gap.
    assert daemon._briefing_gap_hours(_ago(-3), _NOW) is None


def test_gap_hours_accurate_for_known_ts():
    assert daemon._briefing_gap_hours(_ago(31.9), _NOW) == 31.9
    # naive ts assumed UTC; Z-suffix accepted
    naive = (_NOW - timedelta(hours=6)).replace(tzinfo=None).isoformat()
    assert round(daemon._briefing_gap_hours(naive, _NOW), 3) == 6.0
    zsuffix = (_NOW - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert round(daemon._briefing_gap_hours(zsuffix, _NOW), 3) == 5.0


# ── _briefing_lookback_hours ─────────────────────────────────────────────────

def test_lookback_stays_5h_on_healthy_cadence():
    # Unknown gap or on-time → unchanged 5h window (no healthy-path regression).
    assert daemon._briefing_lookback_hours(None) == 5
    assert daemon._briefing_lookback_hours(5.0) == 5
    assert daemon._briefing_lookback_hours(4.2) == 5


def test_lookback_widens_when_overdue_capped_at_24h():
    assert daemon._briefing_lookback_hours(9.0) == 9
    assert daemon._briefing_lookback_hours(31.9) == 24   # hard cap
    assert daemon._briefing_lookback_hours(41.2) == 24
    # never below the 5h floor
    assert daemon._briefing_lookback_hours(6.4) == 6


# ── _coverage_gap_banner ─────────────────────────────────────────────────────

def test_banner_empty_on_healthy_cadence():
    assert daemon._coverage_gap_banner(None) == ""
    assert daemon._coverage_gap_banner(5.0) == ""
    assert daemon._coverage_gap_banner(6.9) == ""  # below the warn threshold


def test_banner_present_and_exact_when_materially_overdue():
    b = daemon._coverage_gap_banner(31.9)
    assert b == (
        "⚠ COVERAGE GAP: first briefing in 31.9h (target 5h) — this digest "
        "spans the backlog, not the usual 5h window"
    )
    # Derives the target from the live HEARTBEAT_INTERVAL constant, not a
    # hardcoded 5, so a retune can't silently desync the message.
    assert f"target {daemon.HEARTBEAT_INTERVAL // 3600}h" in b
    assert daemon._coverage_gap_banner(41.2).startswith(
        "⚠ COVERAGE GAP: first briefing in 41.2h"
    )
