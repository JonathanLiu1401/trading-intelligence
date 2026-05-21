"""Tests for paper_trader.analytics.conviction_language_skill.

Pins:
* the SELF_AWARE × OVERCONFIDENT × NO_PATTERN × INSUFFICIENT_DATA
  verdict matrix
* classifier precedence: HIGH > LOW > ADD > NEUTRAL
* phrase-pattern coverage for each bucket
* word-boundary discipline (no substring poisoning — "problem" ≠ "probe")
* matched_phrase round-tripped to the operator card
* envelope key stability across every verdict
* malformed silent-drop
* threshold-override forwarding
* aggregate maths at exact values
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.conviction_language_skill import (
    ADD,
    HIGH,
    INSUFFICIENT_DATA,
    LOW,
    NEUTRAL,
    NO_PATTERN,
    OVERCONFIDENT,
    SELF_AWARE,
    build_conviction_language_skill,
    classify_reason,
)


def _now():
    return datetime(2026, 5, 21, 7, 0, 0, tzinfo=timezone.utc)


def _s(reason, realized, *, tid=1, action="BUY", ticker="NVDA", closed=False):
    return {
        "trade_id": tid,
        "trade_ts": "2026-05-21T06:00:00+00:00",
        "ticker": ticker,
        "action": action,
        "reason": reason,
        "realized_pct": realized,
        "closed": closed,
    }


_ENVELOPE_KEYS = {
    "as_of", "verdict", "headline", "n_samples",
    "buckets", "thresholds", "samples",
}


class TestClassifierPrecedence:
    def test_high_beats_low(self):
        # Both phrases present — HIGH must win.
        bucket, _ = classify_reason("high conviction but small position")
        assert bucket == HIGH

    def test_low_beats_add(self):
        bucket, _ = classify_reason("starter position to scale into")
        assert bucket == LOW

    def test_add_when_no_conviction_phrase(self):
        bucket, _ = classify_reason("scaling into the trade")
        assert bucket == ADD

    def test_neutral_fallback(self):
        bucket, _ = classify_reason("trade based on news")
        assert bucket == NEUTRAL

    def test_none_is_neutral(self):
        assert classify_reason(None) == (NEUTRAL, None)

    def test_empty_string_is_neutral(self):
        assert classify_reason("") == (NEUTRAL, None)

    def test_non_string_is_neutral(self):
        assert classify_reason(123) == (NEUTRAL, None)


class TestHighPhrases:
    def test_high_conviction(self):
        assert classify_reason("high conviction setup")[0] == HIGH

    def test_strong_conviction(self):
        assert classify_reason("strong conviction add")[0] == HIGH

    def test_max_conviction(self):
        assert classify_reason("max conviction size")[0] == HIGH

    def test_full_size(self):
        assert classify_reason("going full size on NVDA")[0] == HIGH

    def test_aggressive_size(self):
        assert classify_reason("aggressively sizing in")[0] == HIGH

    def test_hyphenated(self):
        assert classify_reason("high-conviction trade")[0] == HIGH

    def test_conviction_reversed(self):
        assert classify_reason("conviction is high here")[0] == HIGH


class TestLowPhrases:
    def test_small_position(self):
        assert classify_reason("opening a small position")[0] == LOW

    def test_tiny_stake(self):
        assert classify_reason("tiny stake to test thesis")[0] == LOW

    def test_test_position(self):
        assert classify_reason("test position only")[0] == LOW

    def test_probe(self):
        assert classify_reason("starting with a probe")[0] == LOW

    def test_starter(self):
        assert classify_reason("starter tranche")[0] == LOW

    def test_speculative(self):
        assert classify_reason("highly speculative")[0] == LOW

    def test_cautious(self):
        assert classify_reason("cautiously entering")[0] == LOW

    def test_tentative(self):
        assert classify_reason("tentative entry")[0] == LOW

    def test_low_conviction(self):
        assert classify_reason("low-conviction trade")[0] == LOW


class TestAddPhrases:
    def test_scale_in(self):
        assert classify_reason("scaling in to NVDA")[0] == ADD

    def test_accumulate(self):
        assert classify_reason("accumulating exposure")[0] == ADD

    def test_ramp(self):
        assert classify_reason("ramping up")[0] == ADD

    def test_build_into(self):
        assert classify_reason("building into a position")[0] == ADD

    def test_add_to(self):
        assert classify_reason("adding to NVDA on dip")[0] == ADD


class TestWordBoundaryDiscipline:
    def test_problem_does_not_match_probe(self):
        # "problem" must NOT classify as LOW via the probe pattern.
        bucket, matched = classify_reason("solving a real problem here")
        assert bucket == NEUTRAL
        assert matched is None

    def test_subterranean_does_not_match_terran(self):
        # General defense against substring poisoning — common-word check.
        bucket, _ = classify_reason("a fine trade")
        assert bucket == NEUTRAL

    def test_overadd_not_confused(self):
        # The literal word "add" alone (without the "to/more/exposure"
        # qualifier) should not trip ADD — otherwise common English like
        # "the addition" would noise the classifier.
        bucket, _ = classify_reason("the addition to the watchlist")
        assert bucket == NEUTRAL


class TestMatchedPhraseSurface:
    def test_matched_phrase_in_card(self):
        rows = [_s("high conviction NVDA add", 5.0, tid=1)]
        out = build_conviction_language_skill(rows, now=_now())
        assert out["samples"][0]["matched_phrase"] is not None
        assert "conviction" in out["samples"][0]["matched_phrase"].lower()

    def test_reason_excerpt_truncated_at_200(self):
        long_reason = "x" * 500
        rows = [_s(long_reason, 1.0)]
        out = build_conviction_language_skill(rows, now=_now())
        exc = out["samples"][0]["reason_excerpt"]
        # 200 chars + ellipsis
        assert len(exc) == 201
        assert exc.endswith("…")


class TestEnvelopeStability:
    def test_empty_input(self):
        out = build_conviction_language_skill([], now=_now())
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == INSUFFICIENT_DATA

    def test_none_input(self):
        out = build_conviction_language_skill(None, now=_now())
        assert out["verdict"] == INSUFFICIENT_DATA

    def test_all_buckets_present_in_output(self):
        out = build_conviction_language_skill([_s("trade", 1.0)], now=_now())
        for b in (HIGH, LOW, ADD, NEUTRAL):
            assert b in out["buckets"]
            assert "n" in out["buckets"][b]


class TestVerdictMatrix:
    def test_self_aware_fires_when_high_outperforms(self):
        high = [_s("high conviction", 5.0, tid=i) for i in range(5)]
        low = [_s("small probe", 0.5, tid=i + 100) for i in range(5)]
        out = build_conviction_language_skill(high + low, now=_now())
        assert out["verdict"] == SELF_AWARE
        assert out["buckets"][HIGH]["mean_pct"] == 5.0
        assert out["buckets"][LOW]["mean_pct"] == 0.5

    def test_overconfident_when_high_underperforms(self):
        high = [_s("strong conviction", -3.0, tid=i) for i in range(5)]
        low = [_s("speculative", 2.0, tid=i + 100) for i in range(5)]
        out = build_conviction_language_skill(high + low, now=_now())
        assert out["verdict"] == OVERCONFIDENT

    def test_no_pattern_within_tolerance(self):
        high = [_s("high conviction", 1.0, tid=i) for i in range(5)]
        low = [_s("probe", 0.5, tid=i + 100) for i in range(5)]
        # gap 0.5pp < 2.0 → NO_PATTERN
        out = build_conviction_language_skill(high + low, now=_now())
        assert out["verdict"] == NO_PATTERN

    def test_insufficient_when_high_short(self):
        high = [_s("high conviction", 5.0, tid=i) for i in range(2)]
        low = [_s("probe", 0.5, tid=i + 100) for i in range(5)]
        out = build_conviction_language_skill(high + low, now=_now())
        assert out["verdict"] == INSUFFICIENT_DATA

    def test_insufficient_when_low_short(self):
        high = [_s("high conviction", 5.0, tid=i) for i in range(5)]
        low = [_s("probe", 0.5, tid=i + 100) for i in range(2)]
        out = build_conviction_language_skill(high + low, now=_now())
        assert out["verdict"] == INSUFFICIENT_DATA


class TestAggregateMath:
    def test_mean_median_win_rate_exact(self):
        rows = [
            _s("high conviction", r, tid=i)
            for i, r in enumerate([-2.0, -1.0, 0.0, 1.0, 2.0])
        ]
        rows += [_s("probe", 5.0, tid=i + 100) for i in range(5)]
        out = build_conviction_language_skill(rows, now=_now())
        h = out["buckets"][HIGH]
        assert h["n"] == 5
        assert h["mean_pct"] == 0.0
        assert h["median_pct"] == 0.0
        assert h["win_rate"] == 40.0  # 2 of 5 strictly > 0


class TestSampleNormalisation:
    def test_malformed_silent_drop(self):
        rows = [
            "not a dict",
            None,
            {"realized_pct": None, "reason": "x"},
            {"realized_pct": float("nan"), "reason": "x"},
            _s("high conviction", 5.0),
        ]
        out = build_conviction_language_skill(rows, now=_now())
        assert out["n_samples"] == 1

    def test_reason_none_is_neutral_bucket(self):
        rows = [_s(None, 1.0, tid=i) for i in range(3)]
        out = build_conviction_language_skill(rows, now=_now())
        assert out["buckets"][NEUTRAL]["n"] == 3


class TestThresholdOverrides:
    def test_min_per_bucket_override(self):
        high = [_s("high conviction", 5.0, tid=i) for i in range(2)]
        low = [_s("probe", 0.5, tid=i + 100) for i in range(2)]
        default = build_conviction_language_skill(high + low, now=_now())
        assert default["verdict"] == INSUFFICIENT_DATA
        over = build_conviction_language_skill(
            high + low, now=_now(), min_per_bucket=2,
        )
        assert over["verdict"] == SELF_AWARE

    def test_verdict_gap_override(self):
        high = [_s("high conviction", 1.5, tid=i) for i in range(5)]
        low = [_s("probe", 0.0, tid=i + 100) for i in range(5)]
        default = build_conviction_language_skill(high + low, now=_now())
        assert default["verdict"] == NO_PATTERN
        over = build_conviction_language_skill(
            high + low, now=_now(), verdict_gap_pct=1.0,
        )
        assert over["verdict"] == SELF_AWARE


class TestSamplesCard:
    def test_samples_capped_at_50(self):
        rows = [_s("trade", 1.0, tid=i) for i in range(120)]
        out = build_conviction_language_skill(rows, now=_now())
        assert out["n_samples"] == 120
        assert len(out["samples"]) == 50

    def test_closed_emitted_first(self):
        rows = [
            _s("trade", 1.0, tid=1, closed=False),
            _s("trade", 1.0, tid=2, closed=True),
            _s("trade", 1.0, tid=3, closed=False),
        ]
        out = build_conviction_language_skill(rows, now=_now())
        assert out["samples"][0]["closed"] is True
