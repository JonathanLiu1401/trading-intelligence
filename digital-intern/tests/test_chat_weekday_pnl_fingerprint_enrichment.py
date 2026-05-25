"""Pure-helper tests for the /api/chat weekday-PnL-fingerprint enrichment.

``_weekday_pnl_fingerprint_chat_lines`` renders paper-trader's
``/api/weekday-pnl-fingerprint`` (the per-weekday alpha-vs-SPY
fingerprint over the bot's equity-curve history) into compact chat
lines so the analyst can answer "is today historically a good day
for this bot vs SPY?" — the chat has no DOW-edge surface anywhere
else.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` string passes through UNCHANGED — no
  chat-side re-derived verdict.
- **flat = silence**: FLAT_WEEK / INSUFFICIENT_DATA / NO_SPY_DATA /
  ERROR collapse to ``[]`` — a no-edge week must never become chat
  filler; INSUFFICIENT_DATA / NO_SPY_DATA are probe-side defects.
- **detail line fields**: when actionable, the detail line restates
  the builder's own ``best_weekday`` / ``worst_weekday`` /
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

from dashboard.web_server import _weekday_pnl_fingerprint_chat_lines


def _rep(
    verdict="WEEKDAY_EDGE",
    *,
    headline=None,
    alpha_spread_pp=0.85,
    n_alpha_samples=90,
    best_weekday=None,
    worst_weekday=None,
):
    if best_weekday is None:
        best_weekday = {
            "weekday": 2, "weekday_name": "Wed",
            "mean_alpha_pct": 0.0305, "n_alpha_samples": 25,
        }
    if worst_weekday is None:
        worst_weekday = {
            "weekday": 4, "weekday_name": "Fri",
            "mean_alpha_pct": -0.1273, "n_alpha_samples": 13,
        }
    if headline is None:
        headline = (
            f"{verdict} — best Wed, worst Fri; spread {alpha_spread_pp}pp"
        )
    return {
        "verdict": verdict,
        "headline": headline,
        "alpha_spread_pp": alpha_spread_pp,
        "n_alpha_samples": n_alpha_samples,
        "n_total_samples": n_alpha_samples + 3,
        "best_weekday": best_weekday,
        "worst_weekday": worst_weekday,
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
    assert _weekday_pnl_fingerprint_chat_lines(bad) == []


def test_empty_dict_is_silence():
    assert _weekday_pnl_fingerprint_chat_lines({}) == []


# ── silence on non-actionable verdicts ──────────────────────────────────
@pytest.mark.parametrize(
    "v", ["FLAT_WEEK", "INSUFFICIENT_DATA", "NO_SPY_DATA", "ERROR", "", None, "OK"],
)
def test_non_actionable_verdicts_collapse_to_silence(v):
    assert _weekday_pnl_fingerprint_chat_lines(_rep(verdict=v)) == []


@pytest.mark.parametrize("v", ["WEEKDAY_EDGE", "WEEKEND_EDGE"])
def test_actionable_verdicts_emit_lines(v):
    lines = _weekday_pnl_fingerprint_chat_lines(_rep(verdict=v))
    assert lines, f"{v} must produce ≥1 line"


# ── headline verbatim (SSOT) ────────────────────────────────────────────
def test_headline_passes_through_verbatim():
    rep = _rep(
        verdict="WEEKDAY_EDGE",
        headline=(
            "WEEKDAY_EDGE — alpha spread 0.85pp across 4 weekdays ≥ 0.50pp "
            "floor; best alpha Wed +0.03%/cycle × 25."
        ),
    )
    assert _weekday_pnl_fingerprint_chat_lines(rep)[0] == rep["headline"]


def test_weekend_edge_headline_passes_through_verbatim():
    rep = _rep(
        verdict="WEEKEND_EDGE",
        headline=(
            "WEEKEND_EDGE — best alpha weekday Sat (+0.5%/cycle × 8) is "
            "weekend — likely after-hours mark drift, not real trading edge."
        ),
    )
    assert _weekday_pnl_fingerprint_chat_lines(rep)[0] == rep["headline"]


def test_missing_headline_degrades_to_detail_only():
    rep = _rep()
    rep["headline"] = None
    lines = _weekday_pnl_fingerprint_chat_lines(rep)
    assert len(lines) == 1
    assert lines[0].startswith("  ")
    assert "best Wed" in lines[0]


# ── detail line composition ─────────────────────────────────────────────
def test_detail_line_full_weekday_edge():
    rep = _rep(
        verdict="WEEKDAY_EDGE",
        alpha_spread_pp=0.85,
        n_alpha_samples=90,
        best_weekday={
            "weekday": 2, "weekday_name": "Wed",
            "mean_alpha_pct": 0.0305, "n_alpha_samples": 25,
        },
        worst_weekday={
            "weekday": 4, "weekday_name": "Fri",
            "mean_alpha_pct": -0.1273, "n_alpha_samples": 13,
        },
    )
    detail = _weekday_pnl_fingerprint_chat_lines(rep)[1]
    assert detail.startswith("  ")
    assert "best Wed" in detail
    assert "alpha +0.030%" in detail
    assert "n=25" in detail
    assert "worst Fri" in detail
    assert "alpha -0.127%" in detail
    assert "n=13" in detail
    assert "spread 0.85pp" in detail
    assert "n_alpha=90" in detail


def test_detail_line_uses_weekday_name_not_number():
    """The detail line MUST use the human weekday name, never the
    builder's integer code — operators read 'Mon' not '0'."""
    rep = _rep()
    rep["best_weekday"]["weekday_name"] = "Mon"
    rep["best_weekday"]["weekday"] = 0
    detail = _weekday_pnl_fingerprint_chat_lines(rep)[1]
    assert "best Mon" in detail
    # We never emit a bare integer weekday position
    assert "best 0" not in detail


def test_detail_line_omits_missing_weekday_name():
    rep = _rep()
    rep["best_weekday"]["weekday_name"] = None
    lines = _weekday_pnl_fingerprint_chat_lines(rep)
    detail = lines[1]
    assert "best" not in detail
    assert "worst Fri" in detail


def test_detail_line_empty_string_name_dropped():
    rep = _rep()
    rep["best_weekday"]["weekday_name"] = "   "
    detail = _weekday_pnl_fingerprint_chat_lines(rep)[1]
    assert "best" not in detail


def test_detail_line_omits_missing_spread():
    rep = _rep()
    rep["alpha_spread_pp"] = None
    detail = _weekday_pnl_fingerprint_chat_lines(rep)[1]
    assert "spread" not in detail


def test_detail_line_all_fields_missing_suppresses_detail():
    rep = {
        "verdict": "WEEKDAY_EDGE",
        "headline": "WEEKDAY_EDGE — sample text",
    }
    lines = _weekday_pnl_fingerprint_chat_lines(rep)
    assert len(lines) == 1
    assert lines[0] == "WEEKDAY_EDGE — sample text"


# ── defensive: bool / unparseable numerics ──────────────────────────────
def test_bool_alpha_treated_as_missing():
    rep = _rep()
    rep["best_weekday"]["mean_alpha_pct"] = True
    detail = _weekday_pnl_fingerprint_chat_lines(rep)[1]
    assert "alpha True" not in detail
    assert "alpha +1" not in detail


def test_bool_n_alpha_treated_as_missing():
    rep = _rep()
    rep["n_alpha_samples"] = True
    detail = _weekday_pnl_fingerprint_chat_lines(rep)[1]
    assert "n_alpha=" not in detail


def test_string_alpha_treated_as_missing():
    rep = _rep()
    rep["best_weekday"]["mean_alpha_pct"] = "0.04"
    detail = _weekday_pnl_fingerprint_chat_lines(rep)[1]
    assert "alpha 0.04" not in detail
    assert "best Wed" in detail


# ── live-fixture regression ─────────────────────────────────────────────
def test_live_2026_05_25_flat_week_is_silent():
    """The current live FLAT_WEEK response — confirm we don't push
    chat filler when no DOW edge has emerged yet."""
    rep = {
        "verdict": "FLAT_WEEK",
        "headline": (
            "FLAT_WEEK — alpha spread 0.16pp across 4 weekdays < 0.50pp floor"
        ),
        "alpha_spread_pp": 0.1578,
        "n_alpha_samples": 90,
        "n_total_samples": 93,
        "best_weekday": {
            "weekday": 2, "weekday_name": "Wed",
            "mean_alpha_pct": 0.0305, "n_alpha_samples": 25,
        },
        "worst_weekday": {
            "weekday": 4, "weekday_name": "Fri",
            "mean_alpha_pct": -0.1273, "n_alpha_samples": 13,
        },
    }
    assert _weekday_pnl_fingerprint_chat_lines(rep) == []


def test_weekday_edge_fixture_full_render():
    rep = {
        "verdict": "WEEKDAY_EDGE",
        "headline": (
            "WEEKDAY_EDGE — alpha spread 0.85pp across 4 weekdays ≥ 0.50pp "
            "floor; best alpha Wed +0.03%/cycle × 25."
        ),
        "alpha_spread_pp": 0.85,
        "n_alpha_samples": 90,
        "best_weekday": {
            "weekday": 2, "weekday_name": "Wed",
            "mean_alpha_pct": 0.03, "n_alpha_samples": 25,
        },
        "worst_weekday": {
            "weekday": 4, "weekday_name": "Fri",
            "mean_alpha_pct": -0.55, "n_alpha_samples": 13,
        },
    }
    lines = _weekday_pnl_fingerprint_chat_lines(rep)
    assert lines[0] == rep["headline"]
    detail = lines[1]
    assert "best Wed alpha +0.030% n=25" in detail
    assert "worst Fri alpha -0.550% n=13" in detail
    assert "spread 0.85pp" in detail
    assert "n_alpha=90" in detail
