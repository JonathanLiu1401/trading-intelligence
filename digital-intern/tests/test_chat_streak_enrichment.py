"""Pure-helper tests for the /api/chat streak enrichment.

`_streak_chat_lines` renders paper-trader's `/api/streak` (current
win/loss run + historical extremes on the closed round-trip series) into
compact chat-context lines so the analyst can answer behavioural-edge
questions no other chat surface answers: "am I on a HOT_HAND right now,
or on a TILT_RISK loss-cluster?"

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_decision_paralysis_chat_lines` /
`_macro_calendar_chat_lines` / `_cash_redeployment_chat_lines`) the
logic is a total/pure function unit-tested here — no Flask, no :8090.

Discriminating locks:

- **verbatim SSOT composition** (paper-trader invariant #10): the
  builder's own `headline` string passes through UNCHANGED.
- **healthy = silence**: NEUTRAL / None (EMERGING / NO_DATA states have
  `verdict=None`) collapse to `[]`, matching the
  `_decision_paralysis_chat_lines` silence precedent — the verdict is
  gated to STABLE (n_round_trips >= 8) by the builder, so a 3-trip
  "streak" never reaches the chat anyway.
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

from dashboard.web_server import _streak_chat_lines


def _rep(verdict="HOT_HAND", *, headline=None,
         cur_kind="WIN", cur_len=4, longest_win=4, longest_loss=2,
         n_round_trips=12):
    if headline is None:
        headline = (
            f"{verdict} — on a {cur_len}-"
            f"{'win' if cur_kind == 'WIN' else 'loss'} run (threshold 4). "
            f"longest W={longest_win}, longest L={longest_loss} "
            f"across {n_round_trips} round-trips.")
    return {
        "as_of": "2026-05-21T11:00:00+00:00",
        "state": "STABLE",
        "verdict": verdict,
        "headline": headline,
        "n_round_trips": n_round_trips,
        "n_wins": 7,
        "n_losses": 5,
        "n_flats": 0,
        "current_streak": {
            "kind": cur_kind,
            "length": cur_len,
            "since_ts": "2026-05-20T14:00:00+00:00",
        },
        "longest_win_streak": longest_win,
        "longest_loss_streak": longest_loss,
        "recent_sequence": ["W", "L", "W", "W", "W", "W"],
        "stable_min_round_trips": 8,
        "hot_hand_min": 4,
        "tilt_risk_min": 4,
    }


class TestPureTotalContract:
    @pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
    def test_non_dict_is_silence(self, bad):
        assert _streak_chat_lines(bad) == []

    def test_missing_verdict_is_silence(self):
        assert _streak_chat_lines({}) == []
        assert _streak_chat_lines({"headline": "x"}) == []


class TestSilenceOnNonActionable:
    @pytest.mark.parametrize("verdict",
                             ["NEUTRAL", None, "EMERGING", "NO_DATA",
                              "", "OTHER"])
    def test_non_actionable_verdicts_silence(self, verdict):
        rep = _rep(verdict=verdict)
        assert _streak_chat_lines(rep) == []

    def test_emerging_state_with_null_verdict_silence(self):
        # State=EMERGING produces verdict=None; helper sees only verdict.
        rep = {
            "state": "EMERGING",
            "verdict": None,
            "headline": "Emerging — 3 of 8 round-trips for a stable read.",
            "n_round_trips": 3,
            "current_streak": {"kind": "WIN", "length": 2,
                               "since_ts": None},
            "longest_win_streak": 2,
            "longest_loss_streak": 1,
        }
        assert _streak_chat_lines(rep) == []


class TestVerbatimHeadlineSSOT:
    """Invariant #10 — chat must not re-derive the verdict."""

    @pytest.mark.parametrize("verdict", ["HOT_HAND", "TILT_RISK"])
    def test_headline_passes_through_verbatim(self, verdict):
        custom = (
            f"{verdict} — custom test string with $42.42 [exact match] "
            "from build_streak")
        rep = _rep(verdict=verdict, headline=custom)
        lines = _streak_chat_lines(rep)
        assert lines[0] == custom

    def test_blank_headline_is_skipped_but_detail_kept(self):
        rep = _rep(verdict="HOT_HAND", headline="")
        lines = _streak_chat_lines(rep)
        for ln in lines:
            assert ln  # not empty
        joined = "\n".join(lines)
        assert "current run" in joined or "round-trip" in joined


class TestDetailLineComposition:
    def test_hot_hand_emits_win_run_in_detail(self):
        rep = _rep(verdict="HOT_HAND", cur_kind="WIN", cur_len=4)
        lines = _streak_chat_lines(rep)
        joined = "\n".join(lines)
        assert "current run: 4 wins" in joined

    def test_tilt_risk_emits_loss_run_in_detail(self):
        rep = _rep(verdict="TILT_RISK", cur_kind="LOSS", cur_len=4)
        lines = _streak_chat_lines(rep)
        joined = "\n".join(lines)
        assert "current run: 4 losses" in joined

    def test_singular_one_win_one_loss(self):
        rep_w = _rep(verdict="HOT_HAND", cur_kind="WIN", cur_len=1)
        lines = _streak_chat_lines(rep_w)
        joined = "\n".join(lines)
        assert "current run: 1 win" in joined
        assert "1 wins" not in joined

        rep_l = _rep(verdict="TILT_RISK", cur_kind="LOSS", cur_len=1)
        lines = _streak_chat_lines(rep_l)
        joined = "\n".join(lines)
        assert "current run: 1 loss" in joined
        assert "1 losses" not in joined

    def test_detail_includes_longest_w_l(self):
        rep = _rep(verdict="HOT_HAND", longest_win=6, longest_loss=3)
        lines = _streak_chat_lines(rep)
        joined = "\n".join(lines)
        assert "longest W=6" in joined
        assert "L=3" in joined

    def test_detail_includes_n_round_trips(self):
        rep = _rep(verdict="HOT_HAND", n_round_trips=15)
        lines = _streak_chat_lines(rep)
        joined = "\n".join(lines)
        assert "15 round-trips" in joined

    def test_n_round_trips_singular(self):
        # Won't actually happen at HOT_HAND (gated to >=8) but the helper
        # must handle the singular form gracefully if ever called. Use a
        # synthetic headline that doesn't itself contain "round-trips" so
        # the assertion locks the helper's *detail-line* pluralization
        # (rather than being contaminated by the fixture's headline text).
        rep = _rep(verdict="HOT_HAND", n_round_trips=1,
                   headline="HOT_HAND — synthetic")
        lines = _streak_chat_lines(rep)
        # The detail line (lines[1]) is what we lock — the helper composes
        # it from the builder's own n_round_trips field with correct
        # singular/plural agreement.
        detail = lines[1] if len(lines) > 1 else ""
        assert "1 round-trip" in detail
        # No "round-trips" plural anywhere in the detail line.
        assert "round-trips" not in detail

    def test_missing_fields_degrade_silently(self):
        rep = {"verdict": "HOT_HAND", "headline": "HH test"}
        # No current_streak / longest_* / n_round_trips — only headline.
        lines = _streak_chat_lines(rep)
        assert lines == ["HH test"]

    def test_garbage_numeric_fields_skip_not_raise(self):
        rep = {
            "verdict": "TILT_RISK",
            "headline": "tr",
            "current_streak": {"kind": "LOSS", "length": "x"},
            "longest_win_streak": None,
            "longest_loss_streak": True,        # bool must NOT pass _num
            "n_round_trips": "many",
        }
        lines = _streak_chat_lines(rep)
        # No detail line because no usable numerics.
        assert lines == ["tr"]

    def test_non_dict_current_streak_degrades(self):
        rep = {
            "verdict": "TILT_RISK", "headline": "tr",
            "current_streak": "not-a-dict",
            "longest_win_streak": 2, "longest_loss_streak": 4,
            "n_round_trips": 10,
        }
        lines = _streak_chat_lines(rep)
        joined = "\n".join(lines)
        # Headline + a detail line that has longest + n_round_trips but
        # NOT current-run (since current_streak was unusable).
        assert "tr" in lines[0]
        assert "current run" not in joined
        assert "longest W=2" in joined
        assert "10 round-trips" in joined

    def test_unexpected_streak_kind_omits_current_run_clause(self):
        # Helper only emits the run clause for kind ∈ {WIN, LOSS}.
        # An unexpected "NONE" or anything else just gets skipped.
        rep = _rep(verdict="HOT_HAND", cur_kind="NONE", cur_len=4)
        lines = _streak_chat_lines(rep)
        joined = "\n".join(lines)
        assert "current run" not in joined
        # But longest + n_round_trips still appear.
        assert "longest W=" in joined


class TestAllActionableVerdictsFire:
    @pytest.mark.parametrize("verdict", ["HOT_HAND", "TILT_RISK"])
    def test_each_actionable_emits_at_least_headline(self, verdict):
        rep = _rep(verdict=verdict,
                   cur_kind="WIN" if verdict == "HOT_HAND" else "LOSS")
        lines = _streak_chat_lines(rep)
        assert lines
        assert lines[0].startswith(verdict)
