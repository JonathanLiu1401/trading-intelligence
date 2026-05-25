"""Pure-helper tests for the /api/chat feed-health enrichment.

`_feed_health_chat_lines` renders paper-trader's `/api/feed-health`
(the live-news pipeline fitness surface) into compact chat-context
lines so the analyst can flag "the bot is BLIND / STALE_FEED /
UNSCORED right now" — the gating-context layer every downstream
decision/book/skill block silently assumes is healthy.

A CASH_REDEPLOYMENT=STALLED verdict means something fundamentally
different when the bot is BLIND vs when the wire is live. Without
this block, the analyst will recommend "trim NVDA" when the right
answer is "restart the scorer" or "wait for the feed to recover".

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_cash_drag_chat_lines` /
`_decision_paralysis_chat_lines`) the logic is a total/pure function
unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` string passes through UNCHANGED — no
  chat-side re-derived verdict. The UNSCORED-clause sub-message
  composed by the builder under BLIND / STALE_FEED is preserved
  intact.
- **healthy = silence**: HEALTHY / NO_DATA collapse to ``[]`` —
  a working feed must not become chat filler; NO_DATA is a probe-side
  defect and not actionable.
- **detail line fields**: when actionable, the detail line restates
  the builder's own ``resolved_newest_age_h`` / ``resolved_live_2h``
  / ``resolved_scored_2h`` / ``blind_streak`` / ``unscored_feed`` /
  ``restart_recommended`` verbatim — never a recomputation.
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

from dashboard.web_server import _feed_health_chat_lines


def _rep(verdict="BLIND", *, headline=None, blind_streak=37,
         resolved_live_2h=184, resolved_scored_2h=0,
         resolved_newest_age_h=0.5, unscored_feed=True,
         restart_recommended=False):
    if headline is None:
        if verdict == "BLIND":
            headline = (
                f"BLIND — {blind_streak} consecutive decision(s) with 0 "
                f"signals; {resolved_live_2h} live article(s) in the last 2h "
                f"but {resolved_scored_2h} scored — restart the scorer, not "
                f"the collector."
            )
        elif verdict == "STALE_FEED":
            headline = (
                f"STALE_FEED — newest live article is "
                f"{resolved_newest_age_h:.1f}h old; "
                f"{resolved_live_2h} in 2h."
            )
        elif verdict == "HEALTHY":
            headline = (
                f"HEALTHY — newest live article {resolved_newest_age_h:.1f}h "
                f"old; {resolved_live_2h} live article(s) in the last 2h."
            )
        else:
            headline = "NO_DATA — feed probe returned nothing."
    return {
        "as_of": "2026-05-25T00:12:41+00:00",
        "verdict": verdict,
        "headline": headline,
        "blind_streak": blind_streak,
        "blind_streak_min": 3,
        "resolved_live_2h": resolved_live_2h,
        "resolved_live_24h": 2400,
        "resolved_scored_2h": resolved_scored_2h,
        "resolved_newest_age_h": resolved_newest_age_h,
        "resolved_newest": "2026-05-25T00:08:39+00:00",
        "resolved_path": "/home/zeph/trading-intelligence/digital-intern/data/articles.db",
        "live_min_score": 4.0,
        "n_decisions": 421,
        "stale_hours": 6.0,
        "split_brain": False,
        "split_brain_gap_h": 6.0,
        "unscored_feed": unscored_feed,
        "restart_recommended": restart_recommended,
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, 1.5, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _feed_health_chat_lines(bad) == []


def test_empty_dict_is_silence():
    assert _feed_health_chat_lines({}) == []


# ── silence on non-actionable verdicts ──────────────────────────────────
@pytest.mark.parametrize("v", ["HEALTHY", "NO_DATA", "", None, "OK"])
def test_non_actionable_verdicts_collapse_to_silence(v):
    assert _feed_health_chat_lines(_rep(verdict=v)) == []


@pytest.mark.parametrize("v", ["BLIND", "STALE_FEED"])
def test_actionable_verdicts_emit_lines(v):
    lines = _feed_health_chat_lines(_rep(verdict=v))
    assert lines, f"{v} must produce ≥1 line"


# ── headline verbatim (SSOT) ────────────────────────────────────────────
def test_blind_headline_passes_through_verbatim():
    """The UNSCORED clause is part of the builder's headline — must
    NEVER be reformatted or stripped by the chat side."""
    rep = _rep(
        verdict="BLIND",
        headline=(
            "BLIND — 37 consecutive decision(s) with 0 signals; 184 live "
            "article(s) in the last 2h but 0 scored — restart the scorer, "
            "not the collector."
        ),
    )
    lines = _feed_health_chat_lines(rep)
    assert lines[0] == rep["headline"]
    assert "restart the scorer, not the collector" in lines[0]


def test_stale_feed_headline_passes_through_verbatim():
    rep = _rep(
        verdict="STALE_FEED",
        headline=(
            "STALE_FEED — newest live article in /…/articles.db is 7.2h "
            "old; only 0 live in 2h; collector likely down."
        ),
    )
    assert _feed_health_chat_lines(rep)[0] == rep["headline"]


def test_missing_headline_degrades_to_detail_only():
    rep = _rep()
    rep["headline"] = None
    lines = _feed_health_chat_lines(rep)
    assert len(lines) == 1
    assert lines[0].startswith("  ")


# ── detail line composition ─────────────────────────────────────────────
def test_detail_line_full_blind_pathology():
    """The full live BLIND+UNSCORED detail line — the most diagnostic
    failure mode this enrichment exists to surface."""
    rep = _rep(verdict="BLIND", blind_streak=37, resolved_live_2h=184,
               resolved_scored_2h=0, resolved_newest_age_h=0.3,
               unscored_feed=True, restart_recommended=True)
    detail = _feed_health_chat_lines(rep)[1]
    assert detail.startswith("  ")
    assert "newest age 0.3h" in detail
    assert "live_2h=184" in detail
    assert "scored_2h=0" in detail
    assert "blind_streak=37" in detail
    assert "unscored_feed=YES" in detail
    assert "restart_recommended=YES" in detail


def test_detail_line_omits_zero_blind_streak():
    """A 0 blind_streak under BLIND would be incoherent; omit rather
    than emit a noisy `blind_streak=0` chunk."""
    rep = _rep(blind_streak=0)
    detail = _feed_health_chat_lines(rep)[1]
    assert "blind_streak=0" not in detail


def test_detail_line_omits_unscored_when_false():
    """STALE_FEED can occur without the UNSCORED clause — must not
    fabricate `unscored_feed=NO` noise then."""
    rep = _rep(verdict="STALE_FEED", unscored_feed=False)
    detail = _feed_health_chat_lines(rep)[1]
    assert "unscored_feed" not in detail


def test_detail_line_omits_restart_when_false():
    rep = _rep(restart_recommended=False)
    detail = _feed_health_chat_lines(rep)[1]
    assert "restart_recommended" not in detail


def test_detail_line_omits_missing_fields():
    rep = _rep()
    for k in (
        "resolved_newest_age_h", "resolved_live_2h",
        "resolved_scored_2h", "blind_streak", "restart_recommended",
    ):
        rep[k] = None
    rep["unscored_feed"] = False
    lines = _feed_health_chat_lines(rep)
    # Headline still present; detail line is suppressed entirely because
    # no safe field survives — no empty "  " marker line either.
    assert len(lines) == 1


def test_bool_count_treated_as_missing():
    """Defensive: bool is_a int in Python; never let True/False slip
    through as a count."""
    rep = _rep()
    rep["resolved_live_2h"] = True
    detail = _feed_health_chat_lines(rep)[1]
    assert "live_2h" not in detail


def test_unscored_only_flag_is_strict_true():
    """`unscored_feed=YES` must require boolean True — a truthy
    non-True (e.g. the int 1) shouldn't accidentally trip it; the
    builder only writes a strict bool."""
    rep = _rep()
    rep["unscored_feed"] = 1
    detail = _feed_health_chat_lines(rep)[1]
    assert "unscored_feed=YES" not in detail


# ── live-fixture regression ─────────────────────────────────────────────
def test_blind_with_unscored_fixture():
    """The 2026-05-24 pass #8 pathology — articles arriving but
    ML scorer down — locked end-to-end through the chat helper."""
    rep = {
        "as_of": "2026-05-24T00:00:00+00:00",
        "verdict": "BLIND",
        "headline": (
            "BLIND — 37 consecutive decision(s) with 0 signals; "
            "184 live article(s) in the last 2h but 0 scored — "
            "restart the scorer, not the collector."
        ),
        "blind_streak": 37,
        "resolved_live_2h": 184,
        "resolved_scored_2h": 0,
        "resolved_newest_age_h": 0.5,
        "unscored_feed": True,
        "restart_recommended": True,
        "live_min_score": 4.0,
    }
    lines = _feed_health_chat_lines(rep)
    assert lines[0] == rep["headline"]
    detail = lines[1]
    assert "unscored_feed=YES" in detail
    assert "blind_streak=37" in detail
    assert "live_2h=184" in detail
    assert "scored_2h=0" in detail


def test_healthy_live_fixture_2026_05_25_is_silent():
    """The current live HEALTHY response — confirm we don't push
    chat filler when the feed is fine."""
    rep = {
        "verdict": "HEALTHY",
        "headline": (
            "HEALTHY — newest live article 0.0h old; 493 live "
            "article(s) in the last 2h, 2457 in 24h; the most-recent "
            "decision received signals."
        ),
        "blind_streak": 0,
        "resolved_live_2h": 493,
        "resolved_scored_2h": 21,
        "resolved_newest_age_h": 0.0,
        "unscored_feed": False,
        "restart_recommended": False,
    }
    assert _feed_health_chat_lines(rep) == []
