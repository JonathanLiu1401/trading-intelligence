"""``[echo]`` calibration tag on briefing newswire rows.

A cluster of N>=ECHO_MIN_COPIES copies all carrying the SAME ``source`` key
is one outlet repeating itself — NOT independent cross-outlet corroboration.
The existing ``[syndicated xN]`` tag tells Opus how many copies exist; this
qualifies it so a GDELT-GKG host self-syndicating slight title variants of
one wire does not read as positive corroboration. Same shape as the other
calibration tags (``[model]`` / ``[ALERTED]`` / ``[BOOK:]``): additive,
pure-read-side, never mutates input, leaves all four load-bearing invariants
intact by construction (no DB write, no ai_score/ml_score/score_source/
urgency touch).

Discriminating locks:
  * threshold N>=3 (a 2-copy single-source cluster stays quiet — likely a
    retitle, not a firehose)
  * distinct-source counting uses the literal ``source`` key (no host
    normalisation), so ``gdelt_gkg/iheart.com`` + ``rss`` = 2 distinct ⇒ NO
    echo, while two copies tagged ``gdelt_gkg/iheart.com`` = 1 distinct ⇒
    echo-eligible (waits for 3rd to fire — see threshold)
  * the prepended PORTFOLIO/OPTIONS snapshot rows pass through with
    ``_corroboration==1`` and (after this change) no ``_distinct_sources``
    field — defaulting to corro=1 they MUST NOT light up
  * a row that bypassed ``_collapse_syndicated`` entirely (legacy callers,
    or a future code path) has no ``_distinct_sources`` either: the helper
    must default to the corroboration count so an unrelated row keeping
    ``_corroboration=1`` never echoes
  * SYSTEM_PROMPT rule names the down-weight consequence (the analyst's
    biggest noise-vs-signal calibration pain), in lockstep with the
    ``[syndicated xN]`` rule above it
"""
from __future__ import annotations

import pytest

from analysis import claude_analyst as ca
from analysis.claude_analyst import (
    ECHO_MIN_COPIES,
    _collapse_syndicated,
    _is_echo_row,
    _build_payload,
)


# ── _is_echo_row pure-helper unit tests ─────────────────────────────────────

class TestIsEchoRow:
    def test_threshold_not_met_returns_false(self):
        for corro in (1, 2):
            assert _is_echo_row({"_corroboration": corro,
                                 "_distinct_sources": 1}) is False, corro

    def test_threshold_met_single_source_returns_true(self):
        assert _is_echo_row({"_corroboration": ECHO_MIN_COPIES,
                             "_distinct_sources": 1}) is True

    def test_threshold_met_two_distinct_returns_false(self):
        """3 copies but 2 distinct outlets — real (though narrow)
        corroboration, NOT echo."""
        assert _is_echo_row({"_corroboration": 5,
                             "_distinct_sources": 2}) is False

    def test_missing_distinct_defaults_to_corroboration(self):
        """A row that bypassed _collapse_syndicated has no
        ``_distinct_sources``; the default must be the corro count, so a
        single-copy row never lights up."""
        assert _is_echo_row({"_corroboration": 1}) is False
        # A 5-copy row with no _distinct_sources defaults distinct=5 — wide,
        # so NOT echo. The conservative direction: don't tag what we can't
        # verify is single-source.
        assert _is_echo_row({"_corroboration": 5}) is False

    def test_non_dict_returns_false(self):
        assert _is_echo_row(None) is False
        assert _is_echo_row("string") is False
        assert _is_echo_row(42) is False

    def test_garbage_fields_swallowed(self):
        assert _is_echo_row({"_corroboration": "many",
                             "_distinct_sources": 1}) is False
        assert _is_echo_row({"_corroboration": 5,
                             "_distinct_sources": "one"}) is False

    def test_threshold_constant_pinned(self):
        """A tuning change here ripples to every briefing — pin the value
        explicitly so it can't drift silently. 3 is the deliberate floor: a
        2-copy single-source cluster is more likely a retitle than a
        firehose."""
        assert ECHO_MIN_COPIES == 3


# ── _collapse_syndicated must surface _distinct_sources ─────────────────────

def _art(title, source="rss", ai_score=8.0):
    return {"title": title, "source": source, "ai_score": ai_score,
            "link": f"https://x.com/{abs(hash((title, source))) % 99999}"}


class TestCollapseDistinctSources:
    def test_single_row_distinct_sources_one(self):
        out = _collapse_syndicated([_art("MU plunges on inventory glut")])
        assert len(out) == 1
        assert out[0]["_distinct_sources"] == 1
        assert out[0]["_corroboration"] == 1

    def test_three_copies_same_source_is_echo(self):
        """Same first-8-token signature, identical ``source`` — the canonical
        single-source self-syndication pattern. The wire-prefix stripper
        removes "UPDATE 2-" etc., so adding/changing those leaves the
        signature identical, while the title is technically distinct."""
        base = "MU plunges 5 on inventory glut new analyst note today"
        arts = [
            _art(base, source="gdelt_gkg/iheart.com"),
            _art(f"UPDATE 1- {base}",
                 source="gdelt_gkg/iheart.com", ai_score=7.0),
            _art(f"RPT- {base}",
                 source="gdelt_gkg/iheart.com", ai_score=7.5),
        ]
        out = _collapse_syndicated(arts)
        assert len(out) == 1, f"expected 1 cluster, got {len(out)}"
        assert out[0]["_corroboration"] == 3
        assert out[0]["_distinct_sources"] == 1
        assert _is_echo_row(out[0]) is True

    def test_three_copies_three_sources_is_not_echo(self):
        """3 wire copies from 3 distinct outlets — real broad corroboration."""
        base = "NVDA earnings beat Q3 by big margin analyst note today"
        arts = [
            _art(base, source="rss"),
            _art(f"UPDATE 1- {base}",
                 source="gdelt_gkg/reuters.com", ai_score=7.0),
            _art(f"RPT- {base}",
                 source="gdelt_gkg/bloomberg.com", ai_score=7.5),
        ]
        out = _collapse_syndicated(arts)
        assert len(out) == 1
        assert out[0]["_corroboration"] == 3
        assert out[0]["_distinct_sources"] == 3
        assert _is_echo_row(out[0]) is False

    def test_mixed_single_and_dual_source_below_threshold(self):
        """Two copies from one source — under the 3-copy floor, no echo."""
        base = "Apple Q4 service revenue tops 25 billion analyst note"
        arts = [
            _art(base, source="rss"),
            _art(f"UPDATE 1- {base}", source="rss"),
        ]
        out = _collapse_syndicated(arts)
        assert len(out) == 1
        assert out[0]["_corroboration"] == 2
        assert out[0]["_distinct_sources"] == 1
        assert _is_echo_row(out[0]) is False

    def test_empty_source_treated_as_distinct_key(self):
        """Two copies with no source attribution — empty string is the key,
        so distinct=1 (still under the 3-copy threshold here)."""
        base = "Generic headline copy one twenty seven sample text"
        arts = [
            _art(base, source=""),
            _art(f"UPDATE 1- {base}", source=""),
        ]
        out = _collapse_syndicated(arts)
        assert out[0]["_corroboration"] == 2
        assert out[0]["_distinct_sources"] == 1

    def test_input_list_not_mutated(self):
        """Pure: caller's dicts must NOT gain ``_distinct_sources`` —
        ``_collapse_syndicated`` only writes onto its NEW shallow copies."""
        base = "Headline X collapse test row signature stable here"
        arts = [
            _art(base, source="rss"),
            _art(f"UPDATE 1- {base}", source="rss"),
            _art(f"RPT- {base}", source="rss"),
        ]
        _collapse_syndicated(arts)
        for a in arts:
            assert "_distinct_sources" not in a, (
                "_collapse_syndicated mutated the input dicts — pure-read-"
                "side contract broken"
            )
            assert "_corroboration" not in a


# ── _build_payload renders the [echo] tag correctly ─────────────────────────

class TestBuildPayloadEchoRender:
    @staticmethod
    def _three_same_source(score=8.0):
        base = "Some wire alpha story version distinct keyword pattern here"
        return [
            _art(base, source="gdelt_gkg/iheart.com", ai_score=score),
            _art(f"UPDATE 1- {base}",
                 source="gdelt_gkg/iheart.com", ai_score=score),
            _art(f"RPT- {base}",
                 source="gdelt_gkg/iheart.com", ai_score=score),
        ]

    def test_echo_tag_appears_on_single_source_cluster(self):
        arts = self._three_same_source()
        out = _build_payload(arts, {}, [])
        assert "[syndicated x3]" in out
        assert "[echo]" in out
        # And specifically — the [echo] is in the SAME line as
        # [syndicated x3] (one row, both calibration tags).
        for line in out.splitlines():
            if "[syndicated x3]" in line:
                assert "[echo]" in line, (
                    "[echo] must render on the same line as "
                    "[syndicated xN] it qualifies; got line:\n" + line
                )
                break
        else:
            pytest.fail("no [syndicated x3] line found in output")

    def test_no_echo_tag_on_multi_outlet_corroboration(self):
        base = "Broad cross outlet wire story body distinctive sample"
        arts = [
            _art(base, source="rss"),
            _art(f"UPDATE 1- {base}",
                 source="gdelt_gkg/reuters.com"),
            _art(f"RPT- {base}",
                 source="gdelt_gkg/bloomberg.com"),
        ]
        out = _build_payload(arts, {}, [])
        assert "[syndicated x3]" in out
        # The cross-outlet case must NOT carry [echo] — full corroboration
        # credit applies.
        for line in out.splitlines():
            if "[syndicated x3]" in line:
                assert "[echo]" not in line, (
                    "broad cross-outlet 3-copy cluster falsely tagged "
                    "[echo]; line:\n" + line
                )

    def test_no_echo_tag_on_two_copy_cluster(self):
        """Threshold floor: a 2-copy single-source cluster is below the bar."""
        base = "Two copy single source cluster headline distinct sample"
        arts = [
            _art(base, source="rss"),
            _art(f"UPDATE 1- {base}", source="rss"),
        ]
        out = _build_payload(arts, {}, [])
        assert "[syndicated x2]" in out
        for line in out.splitlines():
            if "[syndicated x2]" in line:
                assert "[echo]" not in line, (
                    "2-copy cluster falsely tagged [echo] under the 3-copy "
                    "threshold; line:\n" + line
                )

    def test_no_echo_tag_on_singletons(self):
        """A single-copy row carries no [syndicated] tag and must not carry
        [echo] either (defaults to corro=1, distinct defaults to corro)."""
        arts = [_art("Lone headline never tagged anything", source="rss")]
        out = _build_payload(arts, {}, [])
        assert "[syndicated" not in out
        assert "[echo]" not in out


# ── SYSTEM_PROMPT rule presence ─────────────────────────────────────────────

class TestSystemPromptRule:
    def test_echo_rule_present_with_downweight_consequence(self):
        prompt = ca.SYSTEM_PROMPT
        assert "[echo]" in prompt, "SYSTEM_PROMPT missing the [echo] rule"
        # The rule must name the down-weight consequence; without it Opus
        # has no direction to apply the tag. Pin the specific intent
        # phrases — a future edit that softens to a vague "consider" would
        # break this and force a deliberate review.
        for phrase in (
            "ONE source",
            "NOT independent corroboration",
            "Down-weight",
        ):
            assert phrase in prompt, (
                f"SYSTEM_PROMPT [echo] rule missing key phrase {phrase!r}"
            )
