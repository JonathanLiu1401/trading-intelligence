"""Tests for paper_trader.analytics.reasoning_action_verbs.

Grades the internal-consistency check between a decision's structured
``action`` and its natural-language reasoning. Pins:

  * cue scanning (BUY / SELL / HOLD verbs, all conjugations)
  * negation handling ("would not add" → no bullish vote)
  * hedge handling ("would add IF earnings beat" → no vote)
  * option-verb direction mapping (BUY_CALL → BUY, SELL_PUT → BUY,
    SELL_CALL → SELL, BUY_PUT → SELL)
  * verdict mapping for HOLD / BUY / SELL / NO_DECISION
  * state ladder (INSUFFICIENT < 10, CLEAN < 5%, MILD 5-15%,
    NOTABLE 15-30%, ALARMING >= 30%)
  * inner-JSON action preference over outer FILLED-suffixed verb
  * unparseable rows counted separately, never raise
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.reasoning_action_verbs import (
    MIN_SAMPLES_FOR_VERDICT,
    _action_verb,
    _classify_leaning,
    _extract_inner_reasoning,
    _is_rejected,
    _scan_cues,
    _state,
    _tokenize,
    _verdict,
    _BEAR_PATS,
    _BULL_PATS,
    _HOLD_PATS,
    build_reasoning_action_verbs,
)


# ─────────────────────── tokenizer + helpers ──────────────────────


class TestTokenize:
    def test_basic_words(self):
        toks = _tokenize("Buy NVDA today")
        assert [w for _, w in toks] == ["buy", "nvda", "today"]

    def test_contractions_preserved(self):
        # _WORD_RE allows apostrophes inside words.
        toks = _tokenize("I wouldn't add here")
        assert [w for _, w in toks] == ["i", "wouldn't", "add", "here"]


class TestActionVerb:
    def test_extracts_first_token(self):
        assert _action_verb("BUY NVDA → FILLED") == "BUY"

    def test_handles_option_verbs(self):
        assert _action_verb("BUY_CALL NVDA → FILLED") == "BUY_CALL"

    def test_no_decision_intact(self):
        assert _action_verb("NO_DECISION") == "NO_DECISION"

    def test_empty_returns_unknown(self):
        assert _action_verb("") == "UNKNOWN"
        assert _action_verb(None) == "UNKNOWN"
        assert _action_verb("   ") == "UNKNOWN"


# ─────────────────────── cue scanner ──────────────────────


class TestScanCues:
    def test_adds_buy_cue(self):
        hits = _scan_cues("I'm adding to NVDA today.", _BULL_PATS)
        assert len(hits) == 1
        assert hits[0][0].lower() == "adding"

    def test_word_boundary_excludes_additional(self):
        # "additional" must NOT match "add" — word boundaries are anchored
        # to start of the cue. Confirm the regex does not mis-fire.
        hits = _scan_cues("Additional risk only.", _BULL_PATS)
        assert hits == []

    def test_sell_cue_detected(self):
        hits = _scan_cues("Selling the position now.", _BEAR_PATS)
        assert len(hits) == 1

    def test_hold_cue_detected(self):
        hits = _scan_cues("Waiting on earnings before deciding.", _HOLD_PATS)
        assert len(hits) >= 1

    def test_negated_buy_dropped(self):
        # "would not add" is negated within _NEG_WINDOW
        hits = _scan_cues("I would not add here.", _BULL_PATS)
        assert hits == []

    def test_negated_sell_dropped(self):
        hits = _scan_cues("I wouldn't trim into this dip.", _BEAR_PATS)
        # "wouldn't" appears in _NEGATIONS as a full token
        assert hits == []

    def test_avoid_negation_dropped(self):
        # The token "avoid" is a registered negation
        hits = _scan_cues("Avoid selling into weakness.", _BEAR_PATS)
        assert hits == []

    def test_hedged_buy_dropped_with_if(self):
        # Hedge tokens preceding the cue within _HEDGE_WINDOW cancel it.
        # ``If we beat, adding aggressively`` — "if" is 3 words before
        # "adding", inside the window.
        hits = _scan_cues("If we beat, adding aggressively.", _BULL_PATS)
        assert hits == []

    def test_unhedged_following_if_still_counts(self):
        # The hedge check is one-directional (preceding only). A subordinate
        # "if" clause AFTER the cue does NOT cancel — pin so a future
        # symmetric check doesn't silently make every reasoning consistent.
        hits = _scan_cues(
            "Will adding if NVDA beats — solid catalyst.", _BULL_PATS
        )
        assert len(hits) == 1

    def test_hedged_buy_dropped_with_would(self):
        # "would" is in _HEDGES
        hits = _scan_cues("would add tomorrow.", _BULL_PATS)
        assert hits == []

    def test_hedged_sell_dropped_with_unless(self):
        hits = _scan_cues("Will not trim unless macro flips.", _BEAR_PATS)
        # "not" precedes "trim" → negated; even without that, the hedge
        # would fire.
        assert hits == []

    def test_multiple_buy_cues_all_counted(self):
        hits = _scan_cues(
            "Adding here. Also buying a starter at the lows.",
            _BULL_PATS,
        )
        assert len(hits) == 2

    def test_empty_string_returns_empty(self):
        assert _scan_cues("", _BULL_PATS) == []


# ─────────────────────── classify_leaning ──────────────────────


class TestClassifyLeaning:
    def test_no_directional_returns_holding(self):
        assert _classify_leaning(0, 0, 0) == "HOLDING"
        assert _classify_leaning(0, 0, 5) == "HOLDING"

    def test_bull_majority_returns_bullish(self):
        assert _classify_leaning(3, 1, 0) == "BULLISH"

    def test_bear_majority_returns_bearish(self):
        assert _classify_leaning(0, 2, 0) == "BEARISH"

    def test_equal_nonzero_returns_mixed(self):
        # Equal bull / bear AND both non-zero → MIXED
        assert _classify_leaning(2, 2, 0) == "MIXED"


# ─────────────────────── verdict mapping ──────────────────────


class TestVerdict:
    def test_hold_action_consistent_with_holding(self):
        assert _verdict("HOLD", "HOLDING") == "CONSISTENT"

    def test_hold_action_bullish_inside_hold(self):
        assert _verdict("HOLD", "BULLISH") == "BULLISH_INSIDE_HOLD"

    def test_hold_action_bearish_inside_hold(self):
        assert _verdict("HOLD", "BEARISH") == "BEARISH_INSIDE_HOLD"

    def test_hold_action_mixed_is_consistent(self):
        # MIXED leaning is not a strong signal one way or the other,
        # so a HOLD action with MIXED leaning isn't flagged.
        assert _verdict("HOLD", "MIXED") == "CONSISTENT"

    def test_buy_action_bearish_reasoning_flagged(self):
        assert _verdict("BUY", "BEARISH") == "BEARISH_INSIDE_BUY"

    def test_buy_action_bullish_is_consistent(self):
        assert _verdict("BUY", "BULLISH") == "CONSISTENT"

    def test_buy_action_holding_reasoning_is_consistent(self):
        # A BUY with neutral wording isn't "bearish_inside_buy" — only
        # a directly opposing direction is flagged.
        assert _verdict("BUY", "HOLDING") == "CONSISTENT"

    def test_sell_action_bullish_flagged(self):
        assert _verdict("SELL", "BULLISH") == "BULLISH_INSIDE_SELL"

    def test_buy_call_maps_to_buy(self):
        # BUY_CALL is bullish-direction; bearish leaning flags it
        assert _verdict("BUY_CALL", "BEARISH") == "BEARISH_INSIDE_BUY"

    def test_sell_put_maps_to_buy(self):
        # Sell-puts is a bullish strategy
        assert _verdict("SELL_PUT", "BEARISH") == "BEARISH_INSIDE_BUY"

    def test_sell_call_maps_to_sell(self):
        # Sell-call is bearish-direction; bullish leaning flags it
        assert _verdict("SELL_CALL", "BULLISH") == "BULLISH_INSIDE_SELL"

    def test_buy_put_maps_to_sell(self):
        assert _verdict("BUY_PUT", "BULLISH") == "BULLISH_INSIDE_SELL"

    def test_no_decision_directional_flagged(self):
        assert _verdict("NO_DECISION", "BULLISH") == "DIRECTION_INSIDE_NO_DECISION"
        assert _verdict("NO_DECISION", "BEARISH") == "DIRECTION_INSIDE_NO_DECISION"

    def test_no_decision_holding_is_consistent(self):
        assert _verdict("NO_DECISION", "HOLDING") == "CONSISTENT"

    def test_blocked_unknown_actions_consistent(self):
        # Non-directional actions never produce a mismatch verdict
        assert _verdict("BLOCKED", "BULLISH") == "CONSISTENT"
        assert _verdict("REBALANCE", "BEARISH") == "CONSISTENT"
        assert _verdict("UNKNOWN", "BULLISH") == "CONSISTENT"


# ─────────────────────── state ladder ──────────────────────


class TestStateLadder:
    def test_insufficient_under_threshold(self):
        assert _state(0.5, 0) == "INSUFFICIENT"
        assert _state(0.5, MIN_SAMPLES_FOR_VERDICT - 1) == "INSUFFICIENT"

    def test_clean_under_5pct(self):
        assert _state(0.0, MIN_SAMPLES_FOR_VERDICT) == "CLEAN"
        assert _state(0.04, 100) == "CLEAN"

    def test_mild_5_to_15pct(self):
        assert _state(0.05, 100) == "MILD"
        assert _state(0.14, 100) == "MILD"

    def test_notable_15_to_30pct(self):
        assert _state(0.15, 100) == "NOTABLE"
        assert _state(0.29, 100) == "NOTABLE"

    def test_alarming_at_or_above_30pct(self):
        assert _state(0.30, 100) == "ALARMING"
        assert _state(0.99, 100) == "ALARMING"


# ─────────────────────── inner-reasoning extraction ──────────────────────


class TestExtractInnerReasoning:
    def test_canonical_envelope(self):
        blob = json.dumps({
            "decision": {
                "action": "BUY",
                "ticker": "NVDA",
                "reasoning": "Strong earnings catalyst.",
            },
            "auto_exits": [],
            "detail": "BUY 1 NVDA @ 100",
        })
        text, action, ticker = _extract_inner_reasoning(blob)
        assert text == "Strong earnings catalyst."
        assert action == "BUY"
        assert ticker == "NVDA"

    def test_falls_back_to_top_level_reasoning(self):
        blob = json.dumps({"reasoning": "Older row format."})
        text, action, ticker = _extract_inner_reasoning(blob)
        assert text == "Older row format."
        assert action is None
        assert ticker is None

    def test_falls_back_to_detail(self):
        blob = json.dumps({"detail": "stored in detail key only"})
        text, action, ticker = _extract_inner_reasoning(blob)
        assert text == "stored in detail key only"

    def test_parse_failed_prefix_stripped(self):
        text, action, ticker = _extract_inner_reasoning(
            "parse_failed: I think we should hold."
        )
        # Not valid JSON after stripping → raw prose is returned.
        assert text == "I think we should hold."
        assert action is None

    def test_retry_failed_prefix_stripped(self):
        blob = "retry_failed: " + json.dumps(
            {"decision": {"action": "HOLD", "reasoning": "wait it out"}}
        )
        text, action, ticker = _extract_inner_reasoning(blob)
        assert text == "wait it out"
        assert action == "HOLD"

    def test_malformed_json_returns_raw_string_as_reasoning(self):
        # Not parseable JSON, no prefix — use the raw text as reasoning.
        text, action, ticker = _extract_inner_reasoning("just some prose")
        assert text == "just some prose"
        assert action is None

    def test_empty_blob_returns_empty(self):
        assert _extract_inner_reasoning("") == ("", None, None)
        assert _extract_inner_reasoning(None) == ("", None, None)
        assert _extract_inner_reasoning("   ") == ("", None, None)

    def test_non_dict_json_returns_empty(self):
        text, action, ticker = _extract_inner_reasoning('["not", "a", "dict"]')
        assert text == ""
        assert action is None

    def test_inner_action_normalized_upper(self):
        blob = json.dumps({
            "decision": {"action": "buy", "ticker": "nvda", "reasoning": "x"}
        })
        text, action, ticker = _extract_inner_reasoning(blob)
        assert action == "BUY"
        assert ticker == "NVDA"

    def test_markdown_fence_stripped(self):
        # Some legacy rows wrap the JSON in ```json fences
        blob = "```json\n" + json.dumps(
            {"decision": {"action": "BUY", "reasoning": "go"}}
        ) + "\n```"
        text, action, ticker = _extract_inner_reasoning(blob)
        assert action == "BUY"
        assert text == "go"


# ─────────────────────── end-to-end builder ──────────────────────


def _decision(ts: str, action_taken: str, inner_action: str,
              reasoning: str, ticker: str = "NVDA",
              decision_id: int | None = None) -> dict:
    """Construct a synthetic decision row in the canonical store shape."""
    blob = json.dumps({
        "decision": {
            "action": inner_action,
            "ticker": ticker,
            "reasoning": reasoning,
        },
        "detail": f"{action_taken} @ test",
    })
    return {
        "id": decision_id,
        "timestamp": ts,
        "action_taken": action_taken,
        "reasoning": blob,
    }


class TestBuilderEndToEnd:
    def test_empty_input(self):
        r = build_reasoning_action_verbs([])
        assert r["state"] == "INSUFFICIENT"
        assert r["n_decisions"] == 0
        assert r["n_parsed"] == 0
        assert r["mismatches"] == []
        assert "No parseable" in r["headline"]

    def test_none_input(self):
        r = build_reasoning_action_verbs(None)
        assert r["state"] == "INSUFFICIENT"
        assert r["n_decisions"] == 0

    def test_consistent_hold_does_not_flag(self):
        rows = [
            _decision(
                f"2026-05-21T{h:02d}:00:00+00:00",
                "HOLD",
                "HOLD",
                "Staying patient and waiting for the print.",
            )
            for h in range(12)
        ]
        r = build_reasoning_action_verbs(rows)
        assert r["state"] == "CLEAN"
        assert r["n_parsed"] == 12
        assert r["n_mismatched"] == 0
        assert r["by_verdict"].get("CONSISTENT") == 12

    def test_hold_with_buy_verbs_flagged(self):
        # Outer action HOLD but reasoning says "I'm adding here" → flag
        row = _decision(
            "2026-05-21T10:00:00+00:00",
            "HOLD MU",
            "HOLD",
            "I'm adding NVDA aggressively at these levels.",
        )
        r = build_reasoning_action_verbs([row])
        assert r["n_parsed"] == 1
        assert r["n_mismatched"] == 1
        assert r["by_verdict"]["BULLISH_INSIDE_HOLD"] == 1
        assert len(r["mismatches"]) == 1
        m = r["mismatches"][0]
        assert m["leaning"] == "BULLISH"
        assert m["verdict"] == "BULLISH_INSIDE_HOLD"
        assert m["action"] == "HOLD"

    def test_buy_with_bearish_reasoning_alarming(self):
        row = _decision(
            "2026-05-21T10:00:00+00:00",
            "BUY NVDA → FILLED",
            "BUY",
            "Selling the position would be prudent here.",
        )
        r = build_reasoning_action_verbs([row])
        assert r["by_verdict"].get("BEARISH_INSIDE_BUY") == 1

    def test_sell_with_bullish_reasoning_flagged(self):
        row = _decision(
            "2026-05-21T10:00:00+00:00",
            "SELL NVDA → FILLED",
            "SELL",
            "Adding to the position remains compelling here.",
        )
        r = build_reasoning_action_verbs([row])
        assert r["by_verdict"].get("BULLISH_INSIDE_SELL") == 1

    def test_negated_cues_do_not_count(self):
        # "would not add" must not contribute a bullish vote — outer HOLD
        # with no real direction → CONSISTENT
        row = _decision(
            "2026-05-21T10:00:00+00:00",
            "HOLD NVDA",
            "HOLD",
            "I would not add to NVDA here. Staying patient.",
        )
        r = build_reasoning_action_verbs([row])
        assert r["by_verdict"].get("CONSISTENT") == 1

    def test_hedged_cues_do_not_count(self):
        # "would add IF X" is hedged — no directional vote, stays CONSISTENT
        row = _decision(
            "2026-05-21T10:00:00+00:00",
            "HOLD NVDA",
            "HOLD",
            "If earnings beat handily, would consider adding.",
        )
        r = build_reasoning_action_verbs([row])
        assert r["by_verdict"].get("CONSISTENT") == 1

    def test_unparseable_rows_counted(self):
        # A row with no reasoning JSON / no inner text counts as unparseable
        rows = [
            {"id": 1, "timestamp": "2026-05-21T10:00:00+00:00",
             "action_taken": "HOLD", "reasoning": None},
            {"id": 2, "timestamp": "2026-05-21T11:00:00+00:00",
             "action_taken": "HOLD", "reasoning": ""},
        ]
        r = build_reasoning_action_verbs(rows)
        assert r["n_parsed"] == 0
        assert r["n_unparseable"] == 2

    def test_non_dict_rows_counted_unparseable(self):
        r = build_reasoning_action_verbs(["not-a-dict", 42, None])
        assert r["n_unparseable"] == 3
        assert r["n_parsed"] == 0

    def test_mismatches_sorted_newest_first(self):
        rows = [
            _decision(
                "2026-05-21T10:00:00+00:00", "HOLD", "HOLD",
                "Adding more NVDA aggressively.",
            ),
            _decision(
                "2026-05-21T15:00:00+00:00", "HOLD", "HOLD",
                "Adding to LITE — high conviction.",
            ),
            _decision(
                "2026-05-21T12:00:00+00:00", "HOLD", "HOLD",
                "Adding to MU here.",
            ),
        ]
        r = build_reasoning_action_verbs(rows)
        assert r["n_mismatched"] == 3
        # Newest-first by ts
        ts_order = [m["ts"] for m in r["mismatches"]]
        assert ts_order == sorted(ts_order, reverse=True)

    def test_per_action_breakdown(self):
        # 3 HOLDs (2 flagged), 1 BUY (consistent)
        rows = [
            _decision("2026-05-21T10:00:00+00:00", "HOLD", "HOLD",
                      "Adding aggressively here."),
            _decision("2026-05-21T11:00:00+00:00", "HOLD", "HOLD",
                      "Adding more on the dip."),
            _decision("2026-05-21T12:00:00+00:00", "HOLD", "HOLD",
                      "Sitting tight, waiting."),
            _decision("2026-05-21T13:00:00+00:00", "BUY NVDA → FILLED",
                      "BUY", "Adding aggressively here."),
        ]
        r = build_reasoning_action_verbs(rows)
        hold_stats = r["by_action"]["HOLD"]
        assert hold_stats["n"] == 3
        assert hold_stats["n_mismatched"] == 2
        # rate is 2/3 = 66.67%
        assert abs(hold_stats["mismatch_rate_pct"] - 66.67) < 0.5
        buy_stats = r["by_action"]["BUY"]
        assert buy_stats["n"] == 1
        assert buy_stats["n_mismatched"] == 0

    def test_alarming_state_at_30pct(self):
        # 4 flagged + 6 consistent = 40% mismatch over 10 parsed → ALARMING
        rows = []
        for i in range(4):
            rows.append(_decision(
                f"2026-05-21T{i:02d}:00:00+00:00", "HOLD", "HOLD",
                "Adding aggressively to the position here."))
        for i in range(4, 10):
            rows.append(_decision(
                f"2026-05-21T{i:02d}:00:00+00:00", "HOLD", "HOLD",
                "Sitting tight and waiting for the print."))
        r = build_reasoning_action_verbs(rows)
        assert r["state"] == "ALARMING"
        assert r["n_parsed"] == 10
        assert r["n_mismatched"] == 4

    def test_inner_action_overrides_outer(self):
        # The inner JSON action should win when outer is FILLED/BLOCKED
        # appended. A "HOLD" outer with "BUY" inner should grade as BUY.
        row = _decision(
            "2026-05-21T10:00:00+00:00",
            "HOLD MU → BLOCKED",        # outer says HOLD
            "BUY",                       # inner says BUY
            "Selling would be better.",  # bearish reasoning
        )
        r = build_reasoning_action_verbs([row])
        # Grades against inner=BUY + leaning=BEARISH = BEARISH_INSIDE_BUY
        assert r["by_verdict"].get("BEARISH_INSIDE_BUY") == 1


class TestPureNeverRaises:
    def test_garbage_inputs_do_not_raise(self):
        # Pure: any garbage degrades to INSUFFICIENT
        assert build_reasoning_action_verbs(
            [{"weird": 1}, "string-row", None, 42]
        )["state"] == "INSUFFICIENT"

    def test_action_taken_with_no_arrow(self):
        # Free-form "NO_DECISION" rows (no arrow, no ticker)
        rows = [{
            "id": 1, "timestamp": "2026-05-21T10:00:00+00:00",
            "action_taken": "NO_DECISION",
            "reasoning": "claude returned no response (timeout)",
        }] * 12
        r = build_reasoning_action_verbs(rows)
        # The reasoning has no directional verbs → leaning=HOLDING
        # action=NO_DECISION + HOLDING → CONSISTENT, so CLEAN state.
        assert r["state"] == "CLEAN"
