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
