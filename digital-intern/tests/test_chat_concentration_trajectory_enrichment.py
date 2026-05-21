"""Pure-helper tests for the /api/chat concentration-trajectory enrichment.

`_concentration_trajectory_chat_lines` renders paper-trader's
`/api/concentration-trajectory` (the daily-snapshot slope view of
single-name concentration) into compact chat-context lines so the analyst
can answer the first-derivative question no other concentration block
answers: "has the book's top-1 weight been RISING, FALLING, or STEADY
over the last N days?"

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_decision_paralysis_chat_lines` /
`_macro_calendar_chat_lines` / `_cash_redeployment_chat_lines`) the
logic is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own `headline` string passes through UNCHANGED.
- **healthy = silence**: DECONCENTRATING / DIVERSIFIED / BALANCED /
  INSUFFICIENT_DATA / NO_DATA verdicts collapse to `[]`, matching the
  `_decision_paralysis_chat_lines` silence precedent — a chat must not
  carry "you're fine" filler when the trajectory is good (DECONCENTRATING
  is positive news; DIVERSIFIED / BALANCED are healthy states).
- **pure/total**: non-dict / missing keys / unparseable values never
  raise and degrade to silence or the safe subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import _concentration_trajectory_chat_lines


def _rep(verdict="RAMPING_UP", *, headline=None,
         top1_pct=65.0, top1_ticker="NVDA", top3_pct=65.0,
         n_positions=1, delta_top1_pct=20.0, window_days=7):
    if headline is None:
        headline = (
            f"{verdict} — {top1_ticker} climbed 45.0% → {top1_pct:.1f}% "
            f"(top-1 of {n_positions} name(s)) over {window_days} day(s) "
            f"— concentration creep into one name.")
    return {
        "as_of": "2026-05-21T11:00:00+00:00",
        "verdict": verdict,
        "headline": headline,
        "window_days": window_days,
        "n_trades_walked": 12,
        "current": {
            "top1_pct": top1_pct,
            "top1_ticker": top1_ticker,
            "top3_pct": top3_pct,
            "hhi": 0.42,
            "effective_positions": 2.4,
            "n_positions": n_positions,
            "deployed_usd": 656.04,
        },
        "delta_top1_pct": delta_top1_pct,
        "max_top1_pct": top1_pct,
        "min_top1_pct": top1_pct - delta_top1_pct,
    }


class TestPureTotalContract:
    @pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
    def test_non_dict_is_silence(self, bad):
        assert _concentration_trajectory_chat_lines(bad) == []

    def test_missing_verdict_is_silence(self):
        assert _concentration_trajectory_chat_lines({}) == []
        assert _concentration_trajectory_chat_lines({"headline": "x"}) == []


class TestSilenceOnNonActionable:
    @pytest.mark.parametrize("verdict",
                             ["DECONCENTRATING", "DIVERSIFIED", "BALANCED",
                              "INSUFFICIENT_DATA", "NO_DATA",
                              None, "OTHER", ""])
    def test_non_actionable_verdicts_silence(self, verdict):
        rep = _rep(verdict=verdict)
        assert _concentration_trajectory_chat_lines(rep) == []


class TestVerbatimHeadlineSSOT:
    """Invariant #10 — chat must not re-derive the verdict."""

    @pytest.mark.parametrize("verdict",
                             ["CONCENTRATION_SPIKE", "RAMPING_UP",
                              "CONCENTRATED_STEADY"])
    def test_headline_passes_through_verbatim(self, verdict):
        custom = (
            f"{verdict} — custom test string with $42.42 [exact match] "
            "from build_concentration_trajectory")
        rep = _rep(verdict=verdict, headline=custom)
        lines = _concentration_trajectory_chat_lines(rep)
        assert lines[0] == custom

    def test_blank_headline_is_skipped_but_detail_kept(self):
        rep = _rep(verdict="RAMPING_UP", headline="")
        lines = _concentration_trajectory_chat_lines(rep)
        for ln in lines:
            assert ln  # not empty
        joined = "\n".join(lines)
        assert "NVDA" in joined and "top-1" in joined

    def test_whitespace_only_headline_skipped(self):
        rep = _rep(verdict="RAMPING_UP", headline="   ")
        lines = _concentration_trajectory_chat_lines(rep)
        # Skipped headline, but detail still present.
        joined = "\n".join(lines)
        assert "NVDA" in joined


class TestDetailLineComposition:
    def test_detail_restates_top1_ticker_and_pct(self):
        rep = _rep(verdict="RAMPING_UP", top1_ticker="NVDA",
                   top1_pct=65.8, n_positions=1)
        lines = _concentration_trajectory_chat_lines(rep)
        joined = "\n".join(lines)
        assert "NVDA" in joined
        assert "65.8%" in joined
        # n_positions=1 → singular "name"
        assert "1 name)" in joined

    def test_plural_names_when_multiple_positions(self):
        rep = _rep(verdict="RAMPING_UP", n_positions=3)
        lines = _concentration_trajectory_chat_lines(rep)
        joined = "\n".join(lines)
        assert "3 names)" in joined

    def test_ramping_up_shows_delta_with_window(self):
        rep = _rep(verdict="RAMPING_UP", delta_top1_pct=20.5, window_days=7)
        lines = _concentration_trajectory_chat_lines(rep)
        joined = "\n".join(lines)
        assert "+20.5pp" in joined
        assert "7d" in joined

    def test_spike_shows_delta(self):
        rep = _rep(verdict="CONCENTRATION_SPIKE", delta_top1_pct=39.8,
                   window_days=3)
        lines = _concentration_trajectory_chat_lines(rep)
        joined = "\n".join(lines)
        assert "+39.8pp" in joined
        assert "3d" in joined

    def test_steady_omits_delta_clause(self):
        # CONCENTRATED_STEADY is by definition a low-spread regime —
        # the delta clause would be a near-zero number that adds no
        # signal, so we skip it for the steady verdict.
        rep = _rep(verdict="CONCENTRATED_STEADY", delta_top1_pct=0.5,
                   window_days=10)
        lines = _concentration_trajectory_chat_lines(rep)
        joined = "\n".join(lines)
        # No "+0.5pp over 10d" clause on STEADY
        assert "pp over" not in joined
        # But top-1 and top-3 fields still present
        assert "top-1" in joined

    def test_detail_includes_top3(self):
        rep = _rep(verdict="RAMPING_UP", top1_pct=50.0, top3_pct=85.0,
                   top1_ticker="NVDA", n_positions=3)
        lines = _concentration_trajectory_chat_lines(rep)
        joined = "\n".join(lines)
        assert "85.0%" in joined  # top-3

    def test_missing_fields_degrade_silently(self):
        rep = {"verdict": "RAMPING_UP", "headline": "RU test"}
        # No current / delta — only the headline survives.
        lines = _concentration_trajectory_chat_lines(rep)
        assert lines == ["RU test"]

    def test_garbage_numeric_fields_skip_not_raise(self):
        rep = {
            "verdict": "RAMPING_UP",
            "headline": "ru",
            "current": {
                "top1_pct": "x",
                "top1_ticker": "NVDA",
                "top3_pct": None,
                "n_positions": True,        # bool must NOT pass _num filter
            },
            "delta_top1_pct": "nope",
        }
        lines = _concentration_trajectory_chat_lines(rep)
        # No detail line because no usable numerics.
        assert lines == ["ru"]

    def test_non_dict_current_degrades(self):
        rep = {"verdict": "RAMPING_UP", "headline": "ru",
               "current": "not-a-dict"}
        lines = _concentration_trajectory_chat_lines(rep)
        assert lines == ["ru"]


class TestAllActionableVerdictsFire:
    @pytest.mark.parametrize("verdict",
                             ["CONCENTRATION_SPIKE", "RAMPING_UP",
                              "CONCENTRATED_STEADY"])
    def test_each_actionable_emits_at_least_headline(self, verdict):
        rep = _rep(verdict=verdict)
        lines = _concentration_trajectory_chat_lines(rep)
        assert lines
        assert lines[0].startswith(verdict)

    def test_live_shape_from_actual_endpoint(self):
        # Smoke against the exact response shape pulled from the live
        # /api/concentration-trajectory on 2026-05-21 (the live NVDA
        # ramping-up case that motivated this endpoint).
        rep = {
            "as_of": "2026-05-21T15:18:18+00:00",
            "current": {
                "deployed_usd": 655.9845,
                "effective_positions": 1.0,
                "hhi": 1.0,
                "n_positions": 1,
                "top1_pct": 100.0,
                "top1_ticker": "NVDA",
                "top3_pct": 100.0,
            },
            "delta_top1_pct": 39.8013,
            "headline": ("RAMPING_UP — NVDA climbed 60.2% → 100.0% "
                         "(top-1 of 1 name(s)) over 3 day(s) — "
                         "concentration creep into one name."),
            "max_top1_pct": 100.0,
            "min_top1_pct": 60.1987,
            "n_trades_walked": 12,
            "verdict": "RAMPING_UP",
            "window_days": 3,
        }
        lines = _concentration_trajectory_chat_lines(rep)
        assert lines
        # Verbatim headline first.
        assert lines[0].startswith("RAMPING_UP — NVDA climbed")
        # Detail line restates NVDA at 100.0%.
        joined = "\n".join(lines)
        assert "NVDA" in joined
        assert "100.0%" in joined
        assert "+39.8pp" in joined
        assert "3d" in joined
