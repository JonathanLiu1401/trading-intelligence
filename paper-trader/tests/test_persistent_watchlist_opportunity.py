"""Pinning tests for the persistent-watchlist-opportunity builder.

Discriminating locks (per AGENTS.md analytics-test convention — assert
verdicts, sample-size discipline, and pure/never-raises contract):

- **TIME-aged discriminator**: the same article landing 1h ago vs 12h ago
  must produce different ``current_run_hours`` — that gradient is the entire
  reason this module exists alongside the snapshot
  ``watchlist_opportunities`` and the drought-gated ``idle_opportunity``.
- **held filter**: a ticker is excluded the moment it appears in ``held``,
  even with monster heat — this surface is "what the book is missing", not
  "what's hot".
- **persistence threshold**: a name with 5h of contiguous heat must NOT
  surface when threshold is 6h. Without this lock the panel becomes noise.
- **current vs longest**: a name that was hot 24-36h ago but is cold now
  must NOT surface (longest_run is informational; current_run is the
  trigger). Mirrors the "current_drought" / "longest_passive_24h" split in
  ``decision_paralysis``.
- **never raises**: garbage rows (non-dict articles, malformed timestamps,
  None ai_score) degrade row-by-row to skip, never propagate.
- **silence verdicts**: empty universe / signals / no-persistence collapse
  to a stable ``state`` ladder (NO_DATA / NO_PERSISTENT / FLAG) — chat
  wrappers and dashboards switch on these.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.persistent_watchlist_opportunity import (
    DEFAULT_MIN_AI_SCORE,
    DEFAULT_MIN_PERSISTENCE_HOURS,
    DEFAULT_WINDOW_HOURS,
    build_persistent_watchlist_opportunity,
)

NOW = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


def _art(ticker, *, hours_ago, score=7.0, title="", source="rss",
         url=None):
    return {
        "id": f"{ticker}-{hours_ago}",
        "url": url or f"https://example.test/{ticker}/{hours_ago}",
        "title": title or f"{ticker} update {hours_ago}h ago",
        "source": source,
        "ai_score": score,
        "urgency": 0,
        "first_seen": (NOW - timedelta(hours=hours_ago)).isoformat(),
        "summary": "",
        "tickers": [ticker],
    }


# ── pure / total contract ───────────────────────────────────────────────
def test_empty_inputs_no_data():
    out = build_persistent_watchlist_opportunity([], [], [], now=NOW)
    assert out["state"] == "NO_DATA"
    assert out["opportunities"] == []
    assert out["n_persistent"] == 0
    assert "no signals" in out["headline"].lower()


def test_empty_watchlist_no_data():
    out = build_persistent_watchlist_opportunity(
        [], [], [_art("NVDA", hours_ago=1)], now=NOW)
    assert out["state"] == "NO_DATA"
    assert out["opportunities"] == []


def test_empty_signals_no_data_when_watchlist_present():
    # Universe exists but no signals — collapses to NO_DATA (nothing to
    # scan against). state-ladder branch covered explicitly so chat
    # wrappers can switch on it.
    out = build_persistent_watchlist_opportunity(
        ["NVDA"], [], [], now=NOW)
    assert out["state"] == "NO_DATA"
    assert out["n_scanned"] == 1


@pytest.mark.parametrize("bad_watchlist", [None, [None, ""], [123, "AMD"]])
def test_garbage_watchlist_never_raises(bad_watchlist):
    out = build_persistent_watchlist_opportunity(
        bad_watchlist, [], [_art("AMD", hours_ago=1)], now=NOW)
    # AMD should still be scannable when present; pure builder absorbs the
    # garbage entries silently.
    assert isinstance(out, dict)
    assert "state" in out


def test_garbage_articles_never_raise():
    sigs = [
        None,
        "not-a-dict",
        42,
        {"first_seen": None, "ai_score": 7.0, "tickers": ["NVDA"]},
        {"first_seen": "not-a-date", "ai_score": 7.0, "tickers": ["NVDA"]},
        {"first_seen": NOW.isoformat(), "ai_score": "bad", "tickers": ["NVDA"]},
    ]
    out = build_persistent_watchlist_opportunity(
        ["NVDA"], [], sigs, now=NOW)
    assert isinstance(out, dict)
    # Every row was unusable — should collapse to NO_PERSISTENT (universe
    # present, but no bins counted) or NO_DATA. Either is a valid silence
    # verdict; the contract is "does not raise" + "non-FLAG state".
    assert out["state"] in ("NO_DATA", "NO_PERSISTENT")


# ── held filter ─────────────────────────────────────────────────────────
def test_held_ticker_excluded_even_when_persistent():
    # NVDA hot for 12 contiguous hours — but the book HOLDS it. Must not
    # surface; that's the missed-opportunity-only contract.
    sigs = [_art("NVDA", hours_ago=h, score=8.0) for h in range(12)]
    out = build_persistent_watchlist_opportunity(
        ["NVDA"], ["NVDA"], sigs, now=NOW)
    assert out["state"] == "NO_PERSISTENT"
    assert out["n_scanned"] == 0
    assert out["n_persistent"] == 0


def test_held_case_insensitive():
    sigs = [_art("AMD", hours_ago=h, score=8.0) for h in range(12)]
    out = build_persistent_watchlist_opportunity(
        ["AMD"], ["amd"], sigs, now=NOW)
    assert out["state"] == "NO_PERSISTENT"


# ── persistence threshold (the central discriminator) ───────────────────
def test_persistent_run_surfaces():
    # NVDA scoring ≥6 every hour for 10 contiguous hours → current_run=10,
    # above the 6h threshold → must surface as FLAG.
    sigs = [_art("NVDA", hours_ago=h, score=7.5) for h in range(10)]
    out = build_persistent_watchlist_opportunity(
        ["NVDA"], [], sigs, now=NOW)
    assert out["state"] == "FLAG"
    assert out["n_persistent"] == 1
    row = out["opportunities"][0]
    assert row["ticker"] == "NVDA"
    assert row["current_run_hours"] >= 6.0
    assert "NVDA" in out["headline"]


def test_subthreshold_run_does_not_surface():
    # Same shape, but only 5h of contiguous heat → below 6h threshold →
    # must collapse to NO_PERSISTENT. Without this lock the panel
    # duplicates the snapshot one.
    sigs = [_art("AMD", hours_ago=h, score=7.5) for h in range(5)]
    out = build_persistent_watchlist_opportunity(
        ["AMD"], [], sigs, now=NOW)
    assert out["state"] == "NO_PERSISTENT"
    assert out["n_persistent"] == 0


def test_persistence_threshold_is_inclusive():
    # Exactly meeting the threshold must surface (≥, not >).
    sigs = [_art("MU", hours_ago=h, score=6.0)
            for h in range(int(DEFAULT_MIN_PERSISTENCE_HOURS))]
    out = build_persistent_watchlist_opportunity(
        ["MU"], [], sigs, now=NOW)
    assert out["state"] == "FLAG"


# ── current vs longest run (TIME-aged discriminator) ────────────────────
def test_old_run_does_not_surface_when_cold_now():
    # NVDA had 8h of heat ending 24h ago — but the last 24h are cold.
    # current_run=0 → must NOT surface. longest_run preserved as info.
    sigs = [_art("NVDA", hours_ago=24 + h, score=8.0) for h in range(8)]
    out = build_persistent_watchlist_opportunity(
        ["NVDA"], [], sigs, now=NOW)
    assert out["state"] == "NO_PERSISTENT"


def test_short_recent_run_below_threshold_no_flag_even_with_long_history():
    # 3h of recent heat + 10h of older heat with a cold gap → current_run=3,
    # longest_run=10. current_run drives the verdict; longest_run is info.
    sigs = (
        [_art("NVDA", hours_ago=h, score=8.0) for h in range(3)]  # 0..2h
        + [_art("NVDA", hours_ago=h, score=8.0)
           for h in range(15, 25)]                                # 15..24h
    )
    out = build_persistent_watchlist_opportunity(
        ["NVDA"], [], sigs, now=NOW)
    assert out["state"] == "NO_PERSISTENT"


def test_score_below_threshold_does_not_count_as_hot_bin():
    # 10 contiguous bins but every score below min_score → bin is cold →
    # no persistent run.
    sigs = [_art("AMD", hours_ago=h,
                 score=DEFAULT_MIN_AI_SCORE - 1.0) for h in range(10)]
    out = build_persistent_watchlist_opportunity(
        ["AMD"], [], sigs, now=NOW)
    assert out["state"] == "NO_PERSISTENT"


# ── ordering / cap ──────────────────────────────────────────────────────
def test_multiple_persistent_sort_by_current_run_then_score():
    # Two persistent names; NVDA has the longer current run → must lead.
    sigs = [_art("NVDA", hours_ago=h, score=7.0) for h in range(10)]
    sigs += [_art("AMD", hours_ago=h, score=9.0) for h in range(7)]
    out = build_persistent_watchlist_opportunity(
        ["NVDA", "AMD"], [], sigs, now=NOW)
    assert out["state"] == "FLAG"
    assert out["n_persistent"] == 2
    assert out["opportunities"][0]["ticker"] == "NVDA"   # 10h beats 7h
    assert out["opportunities"][1]["ticker"] == "AMD"


def test_tie_breaks_by_max_score_then_ticker():
    # Same current run, AMD has higher score → AMD leads.
    sigs = [_art("NVDA", hours_ago=h, score=6.5) for h in range(8)]
    sigs += [_art("AMD", hours_ago=h, score=9.0) for h in range(8)]
    out = build_persistent_watchlist_opportunity(
        ["NVDA", "AMD"], [], sigs, now=NOW)
    assert out["opportunities"][0]["ticker"] == "AMD"


def test_limit_caps_output():
    sigs = []
    tickers = ["NVDA", "AMD", "MU", "INTC", "TSM"]
    for tk in tickers:
        sigs += [_art(tk, hours_ago=h, score=7.0) for h in range(10)]
    out = build_persistent_watchlist_opportunity(
        tickers, [], sigs, now=NOW, limit=2)
    assert len(out["opportunities"]) == 2
    assert out["n_persistent"] == 2  # the cap is the surfaced count


# ── headline contracts (chat-wrapper SSOT precondition) ─────────────────
def test_headline_no_data_voice():
    out = build_persistent_watchlist_opportunity([], [], [], now=NOW)
    assert out["headline"].startswith("Persistent watchlist opportunity:")


def test_headline_flag_names_top_ticker_and_run():
    sigs = [_art("NVDA", hours_ago=h, score=8.5) for h in range(12)]
    out = build_persistent_watchlist_opportunity(
        ["NVDA"], [], sigs, now=NOW)
    h = out["headline"]
    assert "NVDA" in h
    assert "12.0h" in h or "12h" in h
    assert "8.5" in h


def test_headline_flag_indicates_count_when_multiple():
    sigs = [_art("NVDA", hours_ago=h, score=7.0) for h in range(10)]
    sigs += [_art("AMD", hours_ago=h, score=7.0) for h in range(8)]
    out = build_persistent_watchlist_opportunity(
        ["NVDA", "AMD"], [], sigs, now=NOW)
    assert "+1 more" in out["headline"]


# ── parameter passthrough / payload shape ───────────────────────────────
def test_payload_carries_thresholds_for_chat_audit():
    out = build_persistent_watchlist_opportunity(
        ["NVDA"], [], [_art("NVDA", hours_ago=1)],
        now=NOW, min_score=5.5, min_persistence_hours=4.0,
        window_hours=24.0)
    assert out["min_score"] == 5.5
    assert out["min_persistence_hours"] == 4.0
    assert out["window_hours"] == 24.0


def test_defaults_match_module_constants():
    out = build_persistent_watchlist_opportunity(
        ["NVDA"], [], [_art("NVDA", hours_ago=1)], now=NOW)
    assert out["min_score"] == DEFAULT_MIN_AI_SCORE
    assert out["min_persistence_hours"] == DEFAULT_MIN_PERSISTENCE_HOURS
    assert out["window_hours"] == DEFAULT_WINDOW_HOURS


def test_articles_outside_window_ignored():
    # Heat 100h ago (way outside 48h default window) must not produce a run.
    sigs = [_art("NVDA", hours_ago=100 + h, score=9.0) for h in range(8)]
    out = build_persistent_watchlist_opportunity(
        ["NVDA"], [], sigs, now=NOW)
    assert out["state"] == "NO_PERSISTENT"


def test_now_is_injectable_for_determinism():
    fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    sigs = [{
        "first_seen": (fixed - timedelta(hours=h)).isoformat(),
        "ai_score": 7.0,
        "tickers": ["NVDA"],
        "title": "t",
        "source": "rss",
        "url": f"u/{h}",
    } for h in range(8)]
    out = build_persistent_watchlist_opportunity(
        ["NVDA"], [], sigs, now=fixed)
    assert out["state"] == "FLAG"
    assert out["as_of"] == fixed.isoformat()


def test_naive_now_is_assumed_utc():
    # Same shape with a naive datetime — must be treated as UTC, never
    # raise on aware-vs-naive arithmetic.
    naive = NOW.replace(tzinfo=None)
    sigs = [_art("NVDA", hours_ago=h, score=7.0) for h in range(8)]
    out = build_persistent_watchlist_opportunity(
        ["NVDA"], [], sigs, now=naive)
    assert out["state"] == "FLAG"
