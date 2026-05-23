"""Pure-helper tests for the /api/chat inverse-pair-conflict enrichment.

``_inverse_pair_conflict_chat_lines`` renders paper-trader's
``/api/inverse-pair-conflict-skill`` (leveraged-long + leveraged-inverse
ETFs of the same underlying family simultaneously held — the carry-
waste pathology) into compact chat-context lines.

Discriminating locks:

- **verbatim SSOT** (paper-trader invariant #10): the builder's own
  ``headline`` passes through UNCHANGED as the chat headline — no chat-
  side re-derived verdict.
- **healthy book = silence**: CLEAN / NO_BOOK / OPPOSING_UNLEVERED all
  collapse to ``[]``, exactly the ``_persona_book_fit_chat_lines``
  silence precedent — never chat filler.
- **CARRY_WASTE is loud**: actionable verdict emits the headline + a
  detail line restating the WORST-family fields (cancelled_delta_usd,
  daily_drag_estimate_usd, severity) verbatim.
- **pure/total**: non-dict / missing keys / malformed sub-rows never
  raise and degrade to the safe subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _inverse_pair_conflict_chat_lines


def _rep(
    verdict="CARRY_WASTE",
    headline="CARRY_WASTE — QQQ family: TQQQ + SQQQ",
    *,
    family_label="Nasdaq-100 (QQQ family)",
    cancelled=3000.0,
    drag=1.2,
    severity="HIGH",
    classification="CARRY_WASTE",
):
    return {
        "verdict": verdict,
        "headline": headline,
        "conflicts": [{
            "family": "QQQ",
            "family_label": family_label,
            "classification": classification,
            "severity": severity,
            "cancelled_delta_usd": cancelled,
            "daily_drag_estimate_usd": drag,
        }],
    }


# ─── silence on non-actionable verdicts ──────────────────────────────────


def test_non_dict_input_returns_empty():
    assert _inverse_pair_conflict_chat_lines(None) == []
    assert _inverse_pair_conflict_chat_lines("string") == []
    assert _inverse_pair_conflict_chat_lines(42) == []
    assert _inverse_pair_conflict_chat_lines([]) == []


def test_clean_collapses_to_silence():
    assert _inverse_pair_conflict_chat_lines(_rep(verdict="CLEAN")) == []


def test_no_book_collapses_to_silence():
    assert _inverse_pair_conflict_chat_lines(_rep(verdict="NO_BOOK")) == []


def test_opposing_unlevered_collapses_to_silence():
    # OPPOSING_UNLEVERED is operationally distinct from CARRY_WASTE
    # (single decay tab not two) and not worth the chat slot.
    assert _inverse_pair_conflict_chat_lines(
        _rep(verdict="OPPOSING_UNLEVERED", classification="OPPOSING_UNLEVERED")
    ) == []


def test_unknown_verdict_collapses_to_silence():
    assert _inverse_pair_conflict_chat_lines(_rep(verdict="GARBAGE")) == []


# ─── CARRY_WASTE renders headline verbatim ───────────────────────────────


def test_carry_waste_emits_verbatim_headline_first():
    out = _inverse_pair_conflict_chat_lines(_rep())
    assert len(out) >= 1
    # Verbatim SSOT — the builder's own headline must NOT be paraphrased.
    assert out[0] == "CARRY_WASTE — QQQ family: TQQQ + SQQQ"


def test_carry_waste_detail_contains_all_three_fields():
    out = _inverse_pair_conflict_chat_lines(_rep())
    assert len(out) == 2
    detail = out[1]
    # The detail line restates the builder's own per-family fields.
    assert "cancelled Δ $3000" in detail
    assert "~$1.2/day drag" in detail
    assert "severity HIGH" in detail


def test_carry_waste_with_medium_severity():
    out = _inverse_pair_conflict_chat_lines(_rep(severity="MEDIUM"))
    assert "severity MEDIUM" in out[-1]


# ─── worst-family selection ──────────────────────────────────────────────


def test_picks_first_carry_waste_when_multiple_conflicts():
    rep = {
        "verdict": "CARRY_WASTE",
        "headline": "CARRY_WASTE — Semis family: SOXL + SOXS",
        "conflicts": [
            {  # OPPOSING_UNLEVERED listed first — must be skipped
                "family": "SP500",
                "family_label": "S&P 500",
                "classification": "OPPOSING_UNLEVERED",
                "severity": "MEDIUM",
                "cancelled_delta_usd": 999.0,
                "daily_drag_estimate_usd": 0.3,
            },
            {
                "family": "SEMIS",
                "family_label": "Semis (SOXX family)",
                "classification": "CARRY_WASTE",
                "severity": "HIGH",
                "cancelled_delta_usd": 1234.5,
                "daily_drag_estimate_usd": 0.55,
            },
        ],
    }
    out = _inverse_pair_conflict_chat_lines(rep)
    # Headline is verbatim from the builder (semis).
    assert out[0].endswith("SOXS")
    detail = out[1]
    # Detail must restate the CARRY_WASTE row's fields (1234.5 cancelled),
    # NOT the OPPOSING_UNLEVERED row's 999.0.
    assert "1234.5" in detail
    assert "999" not in detail


# ─── degradation: malformed sub-rows ─────────────────────────────────────


def test_missing_headline_drops_first_line():
    rep = _rep(headline=None)
    out = _inverse_pair_conflict_chat_lines(rep)
    # Headline absent ⇒ only the detail line remains.
    assert all("$" in line or "severity" in line for line in out)


def test_missing_conflicts_returns_only_headline():
    rep = {"verdict": "CARRY_WASTE", "headline": "CARRY_WASTE — sample"}
    out = _inverse_pair_conflict_chat_lines(rep)
    assert out == ["CARRY_WASTE — sample"]


def test_conflicts_not_a_list_returns_only_headline():
    rep = {"verdict": "CARRY_WASTE", "headline": "h", "conflicts": "garbage"}
    out = _inverse_pair_conflict_chat_lines(rep)
    assert out == ["h"]


def test_garbage_conflict_row_no_carry_waste_match():
    rep = {
        "verdict": "CARRY_WASTE",
        "headline": "h",
        "conflicts": [None, "x", {"classification": "OTHER"}],
    }
    out = _inverse_pair_conflict_chat_lines(rep)
    assert out == ["h"]


def test_partial_fields_only_emit_present():
    rep = {
        "verdict": "CARRY_WASTE",
        "headline": "h",
        "conflicts": [{
            "classification": "CARRY_WASTE",
            "cancelled_delta_usd": 100.0,
            # daily_drag missing; severity missing
        }],
    }
    out = _inverse_pair_conflict_chat_lines(rep)
    assert len(out) == 2
    assert "cancelled" in out[1]
    assert "drag" not in out[1]
    assert "severity" not in out[1]


def test_non_numeric_fields_silently_dropped():
    rep = {
        "verdict": "CARRY_WASTE",
        "headline": "h",
        "conflicts": [{
            "classification": "CARRY_WASTE",
            "cancelled_delta_usd": "not-a-number",
            "daily_drag_estimate_usd": True,   # bool is excluded
            "severity": "HIGH",
        }],
    }
    out = _inverse_pair_conflict_chat_lines(rep)
    detail = out[1]
    assert "severity HIGH" in detail
    assert "cancelled" not in detail
    assert "drag" not in detail
