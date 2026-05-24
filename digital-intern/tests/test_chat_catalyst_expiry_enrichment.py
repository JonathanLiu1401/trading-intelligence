"""Pure-helper tests for the /api/chat catalyst-expiry enrichment.

`_catalyst_expiry_chat_lines` renders paper-trader's
`/api/catalyst-expiry-skill` (per-open-position catalyst-class + age vs
catalyst-type expiry window) into compact chat-context lines so the
analyst can answer "which positions are sitting on a STALE thesis?" —
the catalyst-clock follow-up that thesis_drift (P/L verdict) and
hold_discipline (losers overstayed) both leave unanswered.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_cash_drag_chat_lines` /
`_no_decision_reasons_chat_lines` / `_round_trip_postmortem_chat_lines`)
the logic is a total/pure function unit-tested here — no Flask, no
:8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own ``headline`` string passes through UNCHANGED — no
  chat-side re-derived verdict. The worst-zombie detail line restates
  the position's *own* ticker / days_held / catalyst_class fields,
  never a recomputation.
- **healthy / unscored = silence**: ALL_FRESH / STRUCTURAL_BOOK /
  MIXED_BOOK / NO_DATA collapse to ``[]`` — only ZOMBIE_HOLDINGS is
  actionable.
- **worst-position selection**: when multiple positions are ZOMBIE, the
  chat detail line surfaces the one with the LARGEST days_held (ties
  broken alphabetically by ticker for deterministic chat output).
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

from dashboard.web_server import _catalyst_expiry_chat_lines


def _pos(
    ticker="NVDA",
    verdict="ZOMBIE",
    *,
    days_held=5.4,
    catalyst_class="EARNINGS",
    reason="Q1 results crushed expectations.",
):
    return {
        "ticker": ticker,
        "verdict": verdict,
        "days_held": days_held,
        "catalyst_class": catalyst_class,
        "reason": reason,
    }


def _rep(
    verdict="ZOMBIE_HOLDINGS",
    *,
    headline=None,
    positions=None,
):
    if headline is None:
        headline = (
            "ZOMBIE_HOLDINGS — 1 position sitting on a stale catalyst "
            "(NVDA EARNINGS at 5.4d held, >3d zombie floor)."
        )
    if positions is None:
        positions = [_pos()]
    return {
        "verdict": verdict,
        "headline": headline,
        "positions": positions,
        "counts": {
            "ZOMBIE": sum(1 for p in positions if isinstance(p, dict)
                          and p.get("verdict") == "ZOMBIE"),
            "FRESH_CATALYST": 0,
            "STRUCTURAL": 0,
            "UNCATEGORIZED": 0,
            "NO_REASON": 0,
        },
        "n_positions": len(positions),
        "thresholds": {"fresh_days_ceil": 2.0, "zombie_days_floor": 3.0},
    }


# ── pure/total contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, 1.5, [], ["a"], (), object()])
def test_non_dict_is_silence(bad):
    assert _catalyst_expiry_chat_lines(bad) == []


def test_missing_verdict_is_silence():
    assert _catalyst_expiry_chat_lines({}) == []
    assert _catalyst_expiry_chat_lines({"headline": "anything"}) == []


# ── healthy / unscored = silence ────────────────────────────────────────
@pytest.mark.parametrize(
    "verdict",
    [
        "ALL_FRESH",
        "STRUCTURAL_BOOK",
        "MIXED_BOOK",
        "NO_DATA",
        None,
        "",
        "UNKNOWN_VERDICT",
    ],
)
def test_non_actionable_verdicts_silence(verdict):
    rep = _rep(verdict=verdict)
    assert _catalyst_expiry_chat_lines(rep) == []


# ── verbatim headline SSOT (invariant #10) ──────────────────────────────
def test_headline_passes_through_verbatim():
    custom = (
        "ZOMBIE_HOLDINGS — 3 positions aged past their catalyst expiry. "
        "Worst: AMD EARNINGS at 11.2d held."
    )
    out = _catalyst_expiry_chat_lines(_rep(headline=custom))
    assert out[0] == custom            # exact char-for-char passthrough


# ── worst-zombie detail line ────────────────────────────────────────────
def test_worst_zombie_picks_largest_days_held():
    rep = _rep(positions=[
        _pos("NVDA", "ZOMBIE", days_held=3.5, catalyst_class="EARNINGS"),
        _pos("AMD", "ZOMBIE", days_held=11.2, catalyst_class="PRODUCT"),
        _pos("MU", "ZOMBIE", days_held=7.0, catalyst_class="MACRO"),
        _pos("LRCX", "FRESH_CATALYST", days_held=0.5,
             catalyst_class="EARNINGS"),
    ])
    out = _catalyst_expiry_chat_lines(rep)
    body = "\n".join(out)
    # AMD has the largest days_held (11.2) — it must be surfaced.
    assert "AMD" in body
    assert "11.2d held" in body
    assert "catalyst PRODUCT" in body
    # Smaller-days zombies must NOT be the worst sample.
    assert "NVDA" not in body or "11.2" in body.split("worst zombie")[1]


def test_worst_zombie_ties_broken_alphabetically():
    rep = _rep(positions=[
        _pos("ZZZZ", "ZOMBIE", days_held=5.0, catalyst_class="EARNINGS"),
        _pos("AAAA", "ZOMBIE", days_held=5.0, catalyst_class="MACRO"),
        _pos("MMMM", "ZOMBIE", days_held=5.0, catalyst_class="PRODUCT"),
    ])
    out = _catalyst_expiry_chat_lines(rep)
    body = "\n".join(out)
    # All tied at 5.0d; alphabetical first (AAAA) wins for deterministic
    # chat output across runs.
    assert "worst zombie → AAAA" in body
    assert "catalyst MACRO" in body


def test_non_zombie_positions_skipped_in_detail():
    rep = _rep(positions=[
        _pos("AMD", "FRESH_CATALYST", days_held=0.5,
             catalyst_class="EARNINGS"),
        _pos("LRCX", "STRUCTURAL", days_held=20.0,
             catalyst_class="TECHNICAL"),
        _pos("NVDA", "ZOMBIE", days_held=4.0, catalyst_class="EARNINGS"),
    ])
    out = _catalyst_expiry_chat_lines(rep)
    body = "\n".join(out)
    assert "worst zombie → NVDA" in body
    assert "4.0d held" in body
    # The STRUCTURAL position is older but not a zombie — must NOT be
    # surfaced.
    assert "LRCX" not in body


def test_no_zombie_positions_omits_detail_but_keeps_headline():
    """ZOMBIE_HOLDINGS verdict with zero ZOMBIE rows in `positions` is
    an edge case (trader endpoint could in principle report this if the
    classifier disagrees with the aggregate verdict). The chat block
    should trust the trader's top-level verdict — headline still emits
    — but omit the detail line since there is no zombie sample to
    surface."""
    rep = _rep(positions=[
        _pos("AMD", "FRESH_CATALYST", days_held=0.5,
             catalyst_class="EARNINGS"),
    ])
    out = _catalyst_expiry_chat_lines(rep)
    assert len(out) == 1
    assert not out[0].startswith("  ")
    assert "worst zombie" not in out[0]


# ── garbage-input robustness ────────────────────────────────────────────
def test_garbage_positions_skipped():
    rep = _rep(positions=[
        "not-a-dict",
        None,
        42,
        _pos("NVDA", "ZOMBIE", days_held=6.0, catalyst_class="EARNINGS"),
    ])
    out = _catalyst_expiry_chat_lines(rep)
    body = "\n".join(out)
    assert "worst zombie → NVDA" in body
    assert "6.0d held" in body


def test_unparseable_days_held_skipped():
    rep = _rep(positions=[
        _pos("AMD", "ZOMBIE", days_held="x", catalyst_class="EARNINGS"),
        _pos("NVDA", "ZOMBIE", days_held=True,  # bool must be rejected
             catalyst_class="EARNINGS"),
        _pos("MU", "ZOMBIE", days_held=4.5, catalyst_class="MACRO"),
    ])
    out = _catalyst_expiry_chat_lines(rep)
    body = "\n".join(out)
    # Only MU has a usable days_held — it must be the worst.
    assert "worst zombie → MU" in body
    assert "4.5d held" in body


def test_missing_catalyst_class_omits_class_part():
    rep = _rep(
        headline="ZOMBIE_HOLDINGS — 1 stale position past expiry.",
        positions=[
            _pos("NVDA", "ZOMBIE", days_held=5.0, catalyst_class=None),
        ],
    )
    out = _catalyst_expiry_chat_lines(rep)
    detail = next((ln for ln in out if ln.startswith("  worst zombie")), "")
    # Detail line must restate the position but omit the catalyst part
    # when the class is None — restated fields are localized to the
    # detail line, never re-derived from the headline.
    assert "NVDA" in detail
    assert "5.0d held" in detail
    assert "catalyst" not in detail


def test_non_list_positions_omits_detail():
    rep = _rep()
    rep["positions"] = "not-a-list"
    out = _catalyst_expiry_chat_lines(rep)
    # Headline-only, never raises.
    assert len(out) == 1


def test_missing_positions_omits_detail():
    rep = _rep()
    rep.pop("positions", None)
    out = _catalyst_expiry_chat_lines(rep)
    assert len(out) == 1


def test_empty_headline_omits_first_line_but_detail_still_renders():
    rep = _rep(headline="")
    out = _catalyst_expiry_chat_lines(rep)
    body = "\n".join(out)
    assert "worst zombie → NVDA" in body


def test_garbage_headline_omits_first_line():
    rep = _rep(headline=42)
    out = _catalyst_expiry_chat_lines(rep)
    body = "\n".join(out)
    assert "worst zombie → NVDA" in body


# ── shape stability ─────────────────────────────────────────────────────
def test_returns_list_always():
    assert isinstance(_catalyst_expiry_chat_lines({}), list)
    assert isinstance(_catalyst_expiry_chat_lines(None), list)
    assert isinstance(_catalyst_expiry_chat_lines(_rep()), list)


def test_no_positions_state_is_silence():
    """The live state right now: book is 100% cash → NO_DATA verdict.
    The chat must collapse to silence — never chat filler when there
    are no positions to grade."""
    rep = _rep(verdict="NO_DATA", positions=[])
    assert _catalyst_expiry_chat_lines(rep) == []
