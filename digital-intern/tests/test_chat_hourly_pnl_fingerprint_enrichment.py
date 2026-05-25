"""Pure-helper tests for the /api/chat hourly-PnL-fingerprint enrichment.

``_hourly_pnl_fingerprint_chat_lines`` renders paper-trader's
``/api/hourly-pnl-fingerprint`` (the per-hour-of-day alpha-vs-SPY
fingerprint over the bot's equity-curve history) into compact chat
lines so the analyst can answer "is now a good time to lean into this
signal?" with the empirical hour-of-day verdict — the chat has
~50 paper-trader analytics blocks but no temporal-edge surface.

The surrounding chat handler is one large inline closure, so per the
established design (cf. ``_feed_health_chat_lines`` /
``_all_cash_streak_chat_lines``) the logic is a total/pure function
unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` string passes through UNCHANGED — no
  chat-side re-derived verdict.
- **flat = silence**: FLAT_CLOCK / INSUFFICIENT_DATA / NO_SPY_DATA /
  ERROR collapse to ``[]`` — a no-edge clock must never become chat
  filler; INSUFFICIENT_DATA / NO_SPY_DATA are probe-side defects.
- **detail line fields**: when actionable, the detail line restates
  the builder's own ``best_hour`` / ``worst_hour`` /
  ``alpha_spread_pp`` / ``n_alpha_samples`` verbatim — never a
  recomputation.
- **pure/total**: non-dict / missing keys / unparseable numbers never
  raise and degrade to silence or the safe subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _hourly_pnl_fingerprint_chat_lines


def _rep(
    verdict="MORNING_EDGE",
    *,
    headline=None,
    alpha_spread_pp=0.85,
    n_alpha_samples=90,
    best_hour=None,
    worst_hour=None,
):
    if best_hour is None:
        best_hour = {
            "hour": 11,
            "label": "MORNING_EDGE",
            "mean_alpha_pct": 0.0367,
            "n_alpha_samples": 18,
        }
    if worst_hour is None:
        worst_hour = {
            "hour": 10,
            "label": "MORNING_EDGE",
            "mean_alpha_pct": -0.0993,
            "n_alpha_samples": 13,
        }
    if headline is None:
        headline = (
            f"{verdict} — alpha spread {alpha_spread_pp}pp; "
            f"best hour {best_hour.get('hour')}"
        )
    return {
        "verdict": verdict,
        "headline": headline,
        "alpha_spread_pp": alpha_spread_pp,
        "n_alpha_samples": n_alpha_samples,
        "n_total_samples": n_alpha_samples + 3,
        "best_hour": best_hour,
        "worst_hour": worst_hour,
        "buckets": [],
        "thresholds": {
            "alpha_spread_pp": 0.5,
            "min_bucket_alpha_samples": 8,
            "min_total_samples": 60,
            "tz": "America/New_York",
        },
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, 1.5, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _hourly_pnl_fingerprint_chat_lines(bad) == []


def test_empty_dict_is_silence():
    assert _hourly_pnl_fingerprint_chat_lines({}) == []


# ── silence on non-actionable verdicts ──────────────────────────────────
@pytest.mark.parametrize(
    "v", ["FLAT_CLOCK", "INSUFFICIENT_DATA", "NO_SPY_DATA", "ERROR", "", None, "OK"],
)
def test_non_actionable_verdicts_collapse_to_silence(v):
    assert _hourly_pnl_fingerprint_chat_lines(_rep(verdict=v)) == []


@pytest.mark.parametrize(
    "v", ["MORNING_EDGE", "MIDDAY_EDGE", "AFTERNOON_EDGE", "OFF_HOURS_EDGE"],
)
def test_actionable_verdicts_emit_lines(v):
    lines = _hourly_pnl_fingerprint_chat_lines(_rep(verdict=v))
    assert lines, f"{v} must produce ≥1 line"


# ── headline verbatim (SSOT) ────────────────────────────────────────────
def test_headline_passes_through_verbatim():
    rep = _rep(
        verdict="MORNING_EDGE",
        headline=(
            "MORNING_EDGE — alpha spread 0.85pp across 6 hours ≥ 0.50pp "
            "floor; best alpha at 11:00 NY +0.04%/cycle × 18."
        ),
    )
    lines = _hourly_pnl_fingerprint_chat_lines(rep)
    assert lines[0] == rep["headline"]
    assert "MORNING_EDGE — alpha spread 0.85pp" in lines[0]


def test_offhours_edge_headline_passes_through_verbatim():
    rep = _rep(
        verdict="OFF_HOURS_EDGE",
        headline=(
            "OFF_HOURS_EDGE — best alpha hour 04:00 NY (+0.5%/cycle × 12) "
            "is outside session — likely mark-to-market timing artifact."
        ),
    )
    assert _hourly_pnl_fingerprint_chat_lines(rep)[0] == rep["headline"]


def test_missing_headline_degrades_to_detail_only():
    rep = _rep()
    rep["headline"] = None
    lines = _hourly_pnl_fingerprint_chat_lines(rep)
    assert len(lines) == 1
    assert lines[0].startswith("  ")
    assert "best hour 11" in lines[0]


def test_empty_string_headline_is_dropped():
    rep = _rep()
    rep["headline"] = "   "
    lines = _hourly_pnl_fingerprint_chat_lines(rep)
    assert all(not ln.strip() == "" for ln in lines)
    assert lines[0].startswith("  ")


# ── detail line composition ─────────────────────────────────────────────
def test_detail_line_full_morning_edge():
    rep = _rep(
        verdict="MORNING_EDGE",
        alpha_spread_pp=0.85,
        n_alpha_samples=90,
        best_hour={
            "hour": 11, "label": "MORNING_EDGE",
            "mean_alpha_pct": 0.0367, "n_alpha_samples": 18,
        },
        worst_hour={
            "hour": 10, "label": "MORNING_EDGE",
            "mean_alpha_pct": -0.0993, "n_alpha_samples": 13,
        },
    )
    detail = _hourly_pnl_fingerprint_chat_lines(rep)[1]
    assert detail.startswith("  ")
    assert "best hour 11" in detail
    assert "alpha +0.037%" in detail
    assert "n=18" in detail
    assert "worst hour 10" in detail
    assert "alpha -0.099%" in detail
    assert "n=13" in detail
    assert "spread 0.85pp" in detail
    assert "n_alpha=90" in detail


def test_detail_line_omits_missing_best_hour():
    rep = _rep()
    rep["best_hour"] = None
    lines = _hourly_pnl_fingerprint_chat_lines(rep)
    detail = lines[1]
    assert "best hour" not in detail
    assert "worst hour" in detail


def test_detail_line_omits_missing_worst_hour():
    rep = _rep()
    rep["worst_hour"] = {}
    lines = _hourly_pnl_fingerprint_chat_lines(rep)
    detail = lines[1]
    assert "best hour" in detail
    assert "worst hour" not in detail


def test_detail_line_omits_missing_spread():
    rep = _rep()
    rep["alpha_spread_pp"] = None
    detail = _hourly_pnl_fingerprint_chat_lines(rep)[1]
    assert "spread" not in detail


def test_detail_line_omits_missing_n_alpha():
    rep = _rep()
    rep["n_alpha_samples"] = None
    detail = _hourly_pnl_fingerprint_chat_lines(rep)[1]
    assert "n_alpha" not in detail


def test_detail_line_when_only_hours_missing_falls_back():
    rep = _rep()
    rep["best_hour"] = None
    rep["worst_hour"] = None
    lines = _hourly_pnl_fingerprint_chat_lines(rep)
    # Only spread + n_alpha survive
    detail = lines[1]
    assert "spread" in detail
    assert "n_alpha" in detail
    assert "hour" not in detail


def test_detail_line_all_fields_missing_suppresses_detail():
    rep = {
        "verdict": "MORNING_EDGE",
        "headline": "MORNING_EDGE — sample text",
    }
    lines = _hourly_pnl_fingerprint_chat_lines(rep)
    assert len(lines) == 1
    assert lines[0] == "MORNING_EDGE — sample text"


# ── defensive: bool / unparseable numerics ──────────────────────────────
def test_bool_alpha_treated_as_missing():
    """bool is_a int in Python; must never slip through as an alpha."""
    rep = _rep()
    rep["best_hour"]["mean_alpha_pct"] = True
    detail = _hourly_pnl_fingerprint_chat_lines(rep)[1]
    assert "alpha True" not in detail
    assert "alpha +1" not in detail


def test_bool_hour_treated_as_missing():
    rep = _rep()
    rep["best_hour"]["hour"] = True
    detail = _hourly_pnl_fingerprint_chat_lines(rep)[1]
    assert "best hour" not in detail


def test_bool_spread_treated_as_missing():
    rep = _rep()
    rep["alpha_spread_pp"] = False
    detail = _hourly_pnl_fingerprint_chat_lines(rep)[1]
    assert "spread" not in detail


def test_string_alpha_treated_as_missing():
    rep = _rep()
    rep["best_hour"]["mean_alpha_pct"] = "0.04"
    detail = _hourly_pnl_fingerprint_chat_lines(rep)[1]
    assert "alpha 0.04" not in detail
    assert "best hour 11" in detail  # hour still survives


def test_negative_hour_zero_renders_as_two_digit():
    """Hour 0 (midnight) is legal and must render '00', not '0'."""
    rep = _rep(verdict="OFF_HOURS_EDGE")
    rep["best_hour"]["hour"] = 0
    detail = _hourly_pnl_fingerprint_chat_lines(rep)[1]
    assert "hour 00" in detail


# ── live-fixture regression ─────────────────────────────────────────────
def test_live_2026_05_25_flat_clock_is_silent():
    """The current live FLAT_CLOCK response — confirm we don't push
    chat filler when no hour-of-day edge has emerged yet."""
    rep = {
        "verdict": "FLAT_CLOCK",
        "headline": (
            "FLAT_CLOCK — alpha spread 0.14pp across 6 hours < 0.50pp floor"
        ),
        "alpha_spread_pp": 0.136,
        "n_alpha_samples": 90,
        "n_total_samples": 93,
        "best_hour": {
            "hour": 11, "label": "MORNING_EDGE",
            "mean_alpha_pct": 0.0367, "n_alpha_samples": 18,
        },
        "worst_hour": {
            "hour": 10, "label": "MORNING_EDGE",
            "mean_alpha_pct": -0.0993, "n_alpha_samples": 13,
        },
    }
    assert _hourly_pnl_fingerprint_chat_lines(rep) == []


def test_morning_edge_fixture_full_render():
    """The shape the helper exists to surface — locked end-to-end."""
    rep = {
        "verdict": "MORNING_EDGE",
        "headline": (
            "MORNING_EDGE — alpha spread 0.85pp across 6 hours ≥ 0.50pp "
            "floor; best alpha at 11:00 NY +0.04%/cycle × 18."
        ),
        "alpha_spread_pp": 0.85,
        "n_alpha_samples": 90,
        "best_hour": {
            "hour": 11, "label": "MORNING_EDGE",
            "mean_alpha_pct": 0.04, "n_alpha_samples": 18,
        },
        "worst_hour": {
            "hour": 15, "label": "AFTERNOON_EDGE",
            "mean_alpha_pct": -0.4, "n_alpha_samples": 14,
        },
    }
    lines = _hourly_pnl_fingerprint_chat_lines(rep)
    assert lines[0] == rep["headline"]
    detail = lines[1]
    assert "best hour 11 alpha +0.040% n=18" in detail
    assert "worst hour 15 alpha -0.400% n=14" in detail
    assert "spread 0.85pp" in detail
    assert "n_alpha=90" in detail
