"""Syndication dedup: signature normalization and merge bookkeeping.

Focus is the wire-prefix normalization (UPDATE n-, RPT-, BREAKING:, ...) that
lets verbatim wire revisions collapse into the bare headline, plus the
invariants the caller relies on (best representative wins, every collapsed id
is returned for marking, input order of survivors preserved).
"""
from __future__ import annotations

import pytest

from watchers.alert_dedup import _signature, alerted_ids, dedupe_urgent


class TestSignatureNormalization:
    def test_bare_and_attributed_match(self):
        bare = _signature("Micron shares surge after Q3 earnings blowout")
        assert _signature("Micron shares surge after Q3 earnings blowout - Reuters") == bare
        assert _signature("Micron shares surge after Q3 earnings blowout (Bloomberg)") == bare

    @pytest.mark.parametrize(
        "headline",
        [
            "UPDATE 1-Micron shares surge after Q3 earnings blowout",
            "UPDATE 2-Micron shares surge after Q3 earnings blowout",
            "RPT-Micron shares surge after Q3 earnings blowout",
            "REFILE-Micron shares surge after Q3 earnings blowout",
            "CORRECTED-Micron shares surge after Q3 earnings blowout",
            "EXCLUSIVE-Micron shares surge after Q3 earnings blowout",
            "WRAPUP 1-Micron shares surge after Q3 earnings blowout",
            "BREAKING: Micron shares surge after Q3 earnings blowout",
            "JUST IN: Micron shares surge after Q3 earnings blowout",
            "RPT-UPDATE 3-Micron shares surge after Q3 earnings blowout",
        ],
    )
    def test_wire_prefixes_collapse_to_bare(self, headline):
        assert _signature(headline) == _signature(
            "Micron shares surge after Q3 earnings blowout"
        )

    def test_real_allcaps_word_not_eaten(self):
        # No trailing separator after the leading token => not a wire marker.
        assert _signature("NVIDIA earnings crush estimates again") == "nvidia earnings crush estimates again"
        # A genuine acronym headline must not be mistaken for "TABLE-"/"POLL-".
        assert _signature("AMD and TSMC expand foundry pact") == "amd and tsmc expand foundry pact"

    def test_distinct_stories_stay_distinct(self):
        assert _signature("Fed holds rates steady amid inflation concerns") != _signature(
            "Micron shares surge after Q3 earnings blowout"
        )

    def test_empty_and_none(self):
        assert _signature(None) == ""
        assert _signature("") == ""
        assert _signature("   ") == ""


class TestSignatureFrontAttribution:
    """Regression: ``_signature`` used to do ``_SOURCE_SEP.split(head)[0]``,
    which silently collapsed every front-attributed copy of a wire headline to
    just the publisher tag.

    Live evidence (2026-05-19): one canonical NVDA earnings-preview story fired
    THREE BREAKING pushes within 2.5h (03:21 ``GN: Nvidia``, 05:16
    ``GDELT/markets.financialcontent``, 05:42 ``GN: Nvidia``) — they SHOULD
    have been cross-cycle-suppressed by ``alert_recency`` (TTL=6h), but the
    front-attributed copy collapsed to ``'financialcontent'`` and never
    matched the canonical ``'nvidia nvda reports earnings tomorrow ...'``
    signature. Every downstream gate that keys on this signature was bypassed
    on every front-attributed copy:

      * ``alert_recency.partition_already_alerted`` (cross-cycle dedup)
      * ``dedupe_urgent`` (in-batch dedup) — for same-batch front-attrib copies
      * ``analysis.claude_analyst._signature`` → briefing's ``[ALERTED]`` tag

    Fix: pick the LONGEST split part by word count (publisher tag is 1-2
    tokens, real headline is more). The convention of trailing attribution
    ("Micron shares ... - Reuters") is unchanged — the leading headline is
    still longer than the trailing publisher — so the existing
    ``test_bare_and_attributed_match`` invariant holds, locked below by
    ``test_canonical_trailing_attribution_unchanged``.
    """

    def test_front_attributed_matches_bare(self):
        # The live failure that drove this fix: three BREAKING fires on one
        # event in 2.5h — without this fix only #2 would have been suppressed.
        canonical = _signature(
            "Nvidia (NVDA) reports earnings tomorrow: What to expect"
        )
        # Should be a real multi-token signature, NOT collapsed to a single
        # publisher tag.
        assert len(canonical.split()) >= 5, (
            f"canonical signature too short: {canonical!r}"
        )
        front_attributed = [
            "FinancialContent - Nvidia ( NVDA ) Reports Earnings Tomorrow : What To Expect",
            "GDELT - Nvidia (NVDA) reports earnings tomorrow: What to expect",
        ]
        for h in front_attributed:
            assert _signature(h) == canonical, (
                f"front-attributed copy did not collapse to canonical: {h!r} "
                f"-> {_signature(h)!r}"
            )

    def test_front_attribution_never_collapses_to_publisher_tag(self):
        # The specific bug fingerprint: a front-attribution split would
        # produce ``parts[0] = 'FinancialContent'`` and the signature would
        # become a single token. That is what bypassed cross-cycle dedup.
        for h, must_not_be in (
            ("FinancialContent - Nvidia (NVDA) Reports Earnings Tomorrow",
             "financialcontent"),
            ("Zacks - MU beats Q3 estimates handily", "zacks"),
            ("Reuters - Fed surprises with 50bp cut", "reuters"),
            ("Motley Fool - Why I'm bullish on Micron", "motley fool"),
        ):
            sig = _signature(h)
            assert sig != must_not_be, (
                f"signature collapsed to publisher tag {must_not_be!r} for {h!r}"
            )
            # And the canonical (headline-only) form must match.
            canonical = h.split(" - ", 1)[1]
            assert _signature(canonical) == sig

    def test_canonical_trailing_attribution_unchanged(self):
        """The original convention ("Headline - Publisher") must still match
        the bare form (this is the original docstring guarantee and the
        existing ``test_bare_and_attributed_match`` invariant — the fix MUST
        NOT regress it)."""
        bare = _signature("Micron shares surge after Q3 earnings blowout")
        assert _signature(
            "Micron shares surge after Q3 earnings blowout - Reuters"
        ) == bare
        assert _signature(
            "Micron shares surge after Q3 earnings blowout - Bloomberg"
        ) == bare

    def test_three_part_publisher_sandwich(self):
        """Multi-publisher attribution at both ends: pick the meaningful
        middle. "Reuters - MU beats Q3 - Bloomberg" must collapse to the
        same signature as the bare "MU beats Q3"."""
        sig = _signature("Reuters - MU beats Q3 estimates - Bloomberg")
        assert sig == _signature("MU beats Q3 estimates")

    def test_no_separator_unchanged(self):
        """Single-part headlines (no `_SOURCE_SEP` match) must be byte-
        identical to before — covers every test in
        ``test_real_allcaps_word_not_eaten`` indirectly."""
        for h, expected in (
            ("NVIDIA earnings crush estimates again",
             "nvidia earnings crush estimates again"),
            ("Fed holds rates steady amid inflation concerns",
             "fed holds rates steady amid inflation concerns"),
            ("AMD and TSMC expand foundry pact",
             "amd and tsmc expand foundry pact"),
        ):
            assert _signature(h) == expected


class TestDedupeMerge:
    def test_wire_revisions_merge_into_best_representative(self):
        arts = [
            {"_id": "a", "title": "Micron shares surge after Q3 earnings blowout", "ai_score": 7.0},
            {"_id": "b", "title": "UPDATE 2-Micron shares surge after Q3 earnings blowout", "ai_score": 8.5},
            {"_id": "c", "title": "RPT-Micron shares surge after Q3 earnings blowout - Reuters", "ai_score": 6.0},
        ]
        out = dedupe_urgent(arts)
        assert len(out) == 1
        assert out[0]["_id"] == "b"          # highest ai_score wins
        assert out[0]["dup_count"] == 3
        assert sorted(out[0]["_dup_ids"]) == ["a", "c"]
        assert sorted(alerted_ids(out)) == ["a", "b", "c"]

    def test_survivor_order_preserved(self):
        arts = [
            {"_id": "1", "title": "Fed holds rates steady amid inflation concerns", "ai_score": 9.0},
            {"_id": "2", "title": "UPDATE 1-Micron shares surge after Q3 earnings blowout", "ai_score": 5.0},
            {"_id": "3", "title": "Fed holds rates steady amid inflation concerns - AP", "ai_score": 4.0},
        ]
        out = dedupe_urgent(arts)
        assert [a["_id"] for a in out] == ["1", "2"]
        assert out[0]["dup_count"] == 2

    def test_untitled_never_merged(self):
        arts = [
            {"_id": "x", "title": None, "ai_score": 5.0},
            {"_id": "y", "title": None, "ai_score": 6.0},
        ]
        out = dedupe_urgent(arts)
        assert {a["_id"] for a in out} == {"x", "y"}

    def test_queued_dups_stay_urgent(self):
        # Marking only the batched survivor must still sweep its collapsed
        # copies, but not touch a different story still in the queue.
        arts = [
            {"_id": "a", "title": "Micron shares surge after Q3 earnings blowout", "ai_score": 7.0},
            {"_id": "b", "title": "UPDATE 2-Micron shares surge after Q3 earnings blowout", "ai_score": 8.5},
            {"_id": "z", "title": "Fed holds rates steady amid inflation concerns", "ai_score": 9.0},
        ]
        out = dedupe_urgent(arts)
        batch = [a for a in out if a["_id"] == "b"]
        assert sorted(alerted_ids(batch)) == ["a", "b"]


class TestWinnerBranchIdRobustness:
    """Regression: the winner branch promoted the *new* higher-scored copy and
    carried the displaced representative's id forward via the hard subscript
    ``cur["_id"]``. The loser branch and alerted_ids() both guard with .get(),
    but this branch did not, so:

      * a displaced representative dict with no ``_id`` key → KeyError, which
        send_urgent_alert's broad ``except`` swallows → the WHOLE urgent batch
        is dropped and nothing is marked alerted → urgent alerts silently fail
        that cycle (the exact failure class the _fmt defensive-access comment
        in alert_agent.py documents for non-canonical / manual-replay rows);
      * a displaced representative with ``_id=None`` → a literal None leaks
        into _dup_ids → alerted_ids → mark_alerted_batch's ``WHERE id=?``.

    Both must not happen; canonical input behaviour must be byte-identical.
    """

    def test_displaced_representative_missing_id_does_not_raise(self):
        # First (lower-scored) copy has NO _id key at all; the second copy is
        # higher-scored so it wins and must carry the displaced one forward.
        arts = [
            {"title": "Micron shares surge after Q3 earnings blowout", "ai_score": 6.0},
            {"_id": "b", "title": "UPDATE 2-Micron shares surge after Q3 earnings blowout", "ai_score": 9.0},
        ]
        out = dedupe_urgent(arts)              # pre-fix: raised KeyError('_id')
        assert len(out) == 1
        assert out[0]["_id"] == "b"
        assert out[0]["dup_count"] == 2
        # The displaced copy had no id, so there is nothing to mark — but the
        # winner itself must still be returned for marking.
        assert out[0]["_dup_ids"] == []
        assert alerted_ids(out) == ["b"]

    def test_displaced_representative_none_id_not_injected(self):
        arts = [
            {"_id": None, "title": "Micron shares surge after Q3 earnings blowout", "ai_score": 6.0},
            {"_id": "b", "title": "RPT-Micron shares surge after Q3 earnings blowout", "ai_score": 9.0},
        ]
        out = dedupe_urgent(arts)
        assert out[0]["_id"] == "b"
        assert out[0]["dup_count"] == 2
        assert None not in out[0]["_dup_ids"], "None leaked into _dup_ids"
        assert None not in alerted_ids(out), "None would hit mark_alerted's WHERE id=?"

    def test_canonical_input_behaviour_unchanged(self):
        # Same scenario as the module __main__ smoke test — every row has an
        # _id; the fix must not alter the established merge bookkeeping.
        arts = [
            {"_id": "a", "title": "Micron shares surge after Q3 earnings blowout", "ai_score": 7.0},
            {"_id": "b", "title": "Micron shares surge after Q3 earnings blowout", "ai_score": 8.5},
            {"_id": "c", "title": "Micron shares surge after Q3 earnings blowout - Reuters", "ai_score": 6.0},
            {"_id": "f", "title": "UPDATE 2-Micron shares surge after Q3 earnings blowout", "ai_score": 4.0},
            {"_id": "g", "title": "RPT-UPDATE 3-Micron shares surge after Q3 earnings blowout (Reuters)", "ai_score": 3.0},
        ]
        out = dedupe_urgent(arts)
        assert len(out) == 1
        assert out[0]["_id"] == "b" and out[0]["dup_count"] == 5
        assert sorted(out[0]["_dup_ids"]) == ["a", "c", "f", "g"]
