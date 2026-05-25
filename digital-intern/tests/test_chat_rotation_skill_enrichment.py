"""Pin ``_rotation_skill_chat_lines`` chat enrichment.

Mirror `test_chat_cash_redeployment_enrichment.py` — silence-on-healthy,
verbatim headline pass-through, detail line restates only the endpoint's own
`stats` fields, never re-derived.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.web_server import _rotation_skill_chat_lines


# ─── pure / total contract ──────────────────────────────────────────────────

class TestIsTotal:
    @pytest.mark.parametrize("bad", [None, 42, "x", [], (), True, False])
    def test_non_dict_collapses_to_empty(self, bad):
        assert _rotation_skill_chat_lines(bad) == []

    def test_empty_dict_collapses_to_empty(self):
        assert _rotation_skill_chat_lines({}) == []

    def test_missing_stats_does_not_raise(self):
        rep = {"verdict": "LAZY_ROTATION", "headline": "headline"}
        out = _rotation_skill_chat_lines(rep)
        assert out == ["headline"]

    def test_non_dict_stats_does_not_raise(self):
        rep = {"verdict": "LAZY_ROTATION", "headline": "h",
               "stats": "not-a-dict"}
        out = _rotation_skill_chat_lines(rep)
        assert out == ["h"]


# ─── silence-on-healthy ─────────────────────────────────────────────────────

class TestSilenceOnHealthy:
    @pytest.mark.parametrize("verdict", [
        "SKILLED_ROTATION", "NET_POSITIVE", "NEUTRAL",
        "INSUFFICIENT_DATA", "ERROR",
    ])
    def test_non_actionable_collapses_to_empty(self, verdict):
        rep = {
            "verdict": verdict,
            "headline": "anything",
            "stats": {"median_alpha_pp": 5.0, "n_negative": 1,
                      "n_pairs_scored": 4, "negative_alpha_pct": 25.0},
        }
        assert _rotation_skill_chat_lines(rep) == []


# ─── actionable verdicts ────────────────────────────────────────────────────

class TestActionableEmits:
    def test_lazy_rotation_emits_headline_verbatim(self):
        rep = {
            "verdict": "LAZY_ROTATION",
            "headline": "lazy: median alpha -2.50pp; 8/10 rotations destroyed value (80%)",
            "stats": {
                "median_alpha_pp": -2.5,
                "p25_alpha_pp": -4.5,
                "p75_alpha_pp": -1.0,
                "n_negative": 8,
                "n_pairs_scored": 10,
                "negative_alpha_pct": 80.0,
            },
        }
        out = _rotation_skill_chat_lines(rep)
        # SSOT — headline verbatim
        assert out[0] == rep["headline"]
        # Detail line restates the SAME fields the endpoint's stats carries
        assert "p25/median/p75" in out[1]
        assert "-4.50" in out[1] or "-4.5" in out[1]
        assert "-2.50" in out[1] or "-2.5" in out[1]
        assert "8/10 rotations destroyed value" in out[1]
        assert "80%" in out[1]

    def test_net_negative_emits_headline(self):
        rep = {
            "verdict": "NET_NEGATIVE",
            "headline": "net-negative: median alpha -0.50pp across 6 rotations",
            "stats": {
                "median_alpha_pp": -0.5,
                "n_negative": 4,
                "n_pairs_scored": 6,
                "negative_alpha_pct": 66.0,
            },
        }
        out = _rotation_skill_chat_lines(rep)
        assert out[0] == rep["headline"]
        # Without p25/p75, falls back to median-only restate
        assert "median alpha -0.50pp" in out[1] or "median alpha -0.5pp" in out[1]
        assert "4/6 rotations destroyed value" in out[1]

    def test_missing_median_skips_alpha_line(self):
        rep = {
            "verdict": "LAZY_ROTATION",
            "headline": "h",
            "stats": {"n_negative": 5, "n_pairs_scored": 8,
                      "negative_alpha_pct": 62.5},
        }
        out = _rotation_skill_chat_lines(rep)
        assert out[0] == "h"
        # Still emits the count line even without alpha numerics
        assert "5/8 rotations destroyed value" in out[1]


# ─── defensive numeric handling ─────────────────────────────────────────────

class TestNumericDefense:
    def test_bool_as_int_ignored(self):
        rep = {
            "verdict": "LAZY_ROTATION",
            "headline": "h",
            "stats": {
                "median_alpha_pp": True,    # bool — must not print as "+1.00pp"
                "n_negative": True,
                "n_pairs_scored": True,
            },
        }
        out = _rotation_skill_chat_lines(rep)
        # Headline survives; detail line is empty / clean
        assert out[0] == "h"
        if len(out) > 1:
            assert "+1.00" not in out[1]
            assert "1/1 rotations destroyed value" not in out[1]

    def test_zero_pairs_scored_skips_counts(self):
        rep = {
            "verdict": "LAZY_ROTATION",
            "headline": "h",
            "stats": {"median_alpha_pp": -2.0,
                      "n_negative": 0, "n_pairs_scored": 0},
        }
        out = _rotation_skill_chat_lines(rep)
        assert out[0] == "h"
        # No count line because n_pairs_scored is 0
        detail = out[1] if len(out) > 1 else ""
        assert "rotations destroyed value" not in detail
