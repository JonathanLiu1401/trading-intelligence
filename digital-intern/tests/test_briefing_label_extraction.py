"""Briefing-label extraction (``daemon._extract_briefing_labels``).

This is the producer side of the briefing-boost training pipeline. Every 5h the
heartbeat worker generates an Opus briefing, then runs each source article's
title through ``_extract_briefing_labels`` to decide which ones become
``score_source='briefing_boost'`` training labels.

``tests/test_briefing_boost.py`` covers the *consumer*
(``store.update_scores_from_labels``) but does not exercise the extractor — a
bug here (false-positive on empty titles, missed case mismatches, KeyError on
synthetic snapshot rows) silently poisons the training pool with no test
failure. The extractor is the canonical match between titles in the briefing
text and rows in articles.db, so its invariants are load-bearing:

  1. A title prefix shorter than 12 chars NEVER matches — the empty-string-
     substring trap ('' in any_text → True) would otherwise mark every
     untitled row in_briefing.
  2. Synthetic PORTFOLIO/OPTIONS snapshot rows (no url) MUST be skipped — a
     KeyError or false match here would either crash the worker mid-briefing
     or poison `update_scores_from_labels` with a bogus url=''.
  3. Matching is case-insensitive on the title PREFIX (the first 40 chars),
     mirroring ``trainer._fetch_briefing_samples`` exactly.
  4. The 40-char prefix bound is non-trivial: a long mid-headline match must
     NOT count unless the prefix itself appears.
  5. The returned URL is the article's ``link`` field (the DB column the
     consumer matches against), with ``url`` honoured for callers that use
     that alias.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def _extract():
    """Return ``daemon._extract_briefing_labels``.

    Importing daemon top-level pulls every collector + the ML stack, which is
    heavy but acceptable here — the function under test lives in daemon.py
    and there is no lighter import path that exposes it.
    """
    import daemon
    return daemon._extract_briefing_labels


class TestEmptyOrShortTitles:
    def test_empty_title_never_matches(self, _extract):
        """The empty-string-substring trap: ``"" in "anything"`` is True. The
        12-char guard must block it before that ever happens — otherwise every
        untitled row would land in the training pool tagged in_briefing=True."""
        labels = _extract(
            "Opus briefing text mentioning lots of real headlines here.",
            [{"title": "", "link": "https://x.com/empty"}],
        )
        assert len(labels) == 1
        assert labels[0]["in_briefing"] is False, (
            "empty title silently matched — every untitled row would poison the pool"
        )

    def test_short_title_never_matches_even_if_present(self, _extract):
        """An 11-char title that LITERALLY appears in the briefing must still
        be rejected — the 12-char threshold is the guard against generic short
        labels like 'Stocks fell' or 'Bitcoin up'. Mirror
        ``trainer._fetch_briefing_samples``'s identical 12-char minimum."""
        labels = _extract(
            "Stocks fell on Fed comments today, Bitcoin up sharply.",
            [{"title": "Stocks fell", "link": "https://x.com/short"}],
        )
        assert len(labels) == 1
        assert labels[0]["in_briefing"] is False, (
            "an 11-char generic phrase was matched; the 12-char floor regressed"
        )

    def test_exactly_12_char_title_can_match(self, _extract):
        """The boundary — at exactly 12 chars the match must work (otherwise the
        guard is too strict and real labels are missed)."""
        labels = _extract(
            "Today MU earnings beat analyst estimates by a wide margin.",
            [{"title": "MU earnings ", "link": "https://x.com/12"}],
        )
        assert labels[0]["in_briefing"] is True


class TestSyntheticSnapshotRows:
    def test_no_url_row_is_skipped(self, _extract):
        """PORTFOLIO P&L SNAPSHOT / OPTIONS SNAPSHOT rows are prepended to the
        briefing input but carry no url. ``_extract_briefing_labels`` must
        ``continue`` past them silently — neither crash on KeyError nor emit a
        bogus url='' entry the consumer would try to UPDATE on."""
        labels = _extract(
            "PORTFOLIO P&L SNAPSHOT shows MSFT up 1.2% today.",
            [
                {"title": "PORTFOLIO P&L SNAPSHOT", "source": "portfolio"},
                {"title": "OPTIONS SNAPSHOT", "source": "options_monitor"},
                {"title": "MSFT closes up 1.2% on AI demand",
                 "link": "https://reuters.com/msft"},
            ],
        )
        # Only the real article (with url) survives.
        assert len(labels) == 1, (
            f"snapshot rows leaked into labels: got {[l.get('title') for l in labels]}"
        )
        assert labels[0]["url"] == "https://reuters.com/msft"

    def test_empty_url_string_is_also_skipped(self, _extract):
        """Belt-and-braces: an empty-string url (rather than missing key) must
        also be skipped. A url='' would otherwise survive into the labels list
        and the consumer's ``WHERE url IN (...)`` would match every empty-url
        row in the DB."""
        labels = _extract(
            "A real story is mentioned here in the briefing text now.",
            [{"title": "Real headline that is long enough", "link": ""}],
        )
        assert labels == [], "row with empty link should be dropped, not retained"


class TestPrefixMatching:
    def test_case_insensitive_match(self, _extract):
        """Opus may render the headline in mixed case in the briefing prose
        while the DB row carries the original wire casing. Match must be
        case-insensitive on both sides (the punctuation/spacing must still
        line up — the 40-char prefix is a literal substring, not a fuzzy
        match)."""
        labels = _extract(
            "MICRON EARNINGS BEAT estimates by 12% on AI memory.",
            [{"title": "Micron Earnings Beat estimates",
              "link": "https://x.com/mu"}],
        )
        assert labels[0]["in_briefing"] is True, (
            "case-insensitive prefix match regressed — real wires get missed"
        )

    def test_unrelated_title_does_not_match(self, _extract):
        """Sanity: a title that's nowhere in the briefing must report
        in_briefing=False."""
        labels = _extract(
            "Today's market was driven by Fed rate cuts and earnings.",
            [{"title": "Completely unrelated news headline here",
              "link": "https://x.com/no"}],
        )
        assert labels[0]["in_briefing"] is False

    def test_prefix_required_not_mid_substring(self, _extract):
        """A long title whose tail (past char 40) appears in the briefing must
        NOT count — only the first 40 chars are checked. Otherwise the trainer
        ingests a label from a coincidental tail-word collision."""
        # Title is 80 chars. The first 40 are "absolutely-unique-prefix-no-overlap-here-x"
        # — the briefing does NOT contain that prefix. The tail "Fed rate" does
        # appear in the briefing but is past the 40-char window.
        title = (
            "absolutely-unique-prefix-no-overlap-here Fed rate cuts will hit Q3"
        )
        assert len(title) > 40
        labels = _extract(
            "Briefing says the Fed rate cuts dominate this week's news cycle.",
            [{"title": title, "link": "https://x.com/tail"}],
        )
        assert labels[0]["in_briefing"] is False, (
            "tail-substring matched past the 40-char prefix bound — the trainer "
            "would ingest noise labels from coincidental tail collisions"
        )


class TestUrlAliasFallback:
    def test_link_field_is_used_when_url_missing(self, _extract):
        """``get_top_for_briefing`` returns rows with ``link`` (no ``url`` key);
        the extractor must read ``link`` so the resulting labels carry a real
        URL the consumer can match in articles.db. Pinned because a previous
        accidental rename to ``art["url"]`` (KeyError) would crash the worker."""
        labels = _extract(
            "Headline here is fairly long and survives the prefix gate.",
            [{"title": "Headline here is fairly long and",
              "link": "https://reuters.com/x"}],
        )
        assert labels[0]["url"] == "https://reuters.com/x"

    def test_url_field_honoured_when_provided(self, _extract):
        """Some callers (manual replay, test fixtures) use ``url`` — the alias
        must still resolve. ``art.get("url") or art.get("link", "")`` is the
        accepted convention; this pins the OR-precedence so future cleanups
        don't drop one of the keys."""
        labels = _extract(
            "Headline here is fairly long and survives the prefix gate.",
            [{"title": "Headline here is fairly long and",
              "url": "https://bloomberg.com/y"}],
        )
        assert labels[0]["url"] == "https://bloomberg.com/y"


class TestMixedBatch:
    def test_only_in_briefing_rows_are_truthy(self, _extract):
        """End-to-end shape: a batch with snapshot rows, a mention, and an
        unrelated article must produce ONE in_briefing=True entry and one
        in_briefing=False entry — with the snapshot row dropped entirely.

        The matched title's first 40 chars MUST appear verbatim (case-
        insensitive) in the briefing prose. ``MU earnings beat estimates
        topped guidance`` truncates to ``mu earnings beat estimates topped
        guidan`` so the briefing must contain that 40-char prefix verbatim
        for in_briefing=True. This is the same 12-char-min + first-40-char
        contract the trainer's ``_fetch_briefing_samples`` uses — exact
        substring, not fuzzy match — pinned here so a future "smarter"
        rewrite of the extractor doesn't silently relax it (false-positive
        labels are far more damaging than a missed boost)."""
        labels = _extract(
            "Discussion of mu earnings beat estimates topped guidance today.",
            [
                {"title": "PORTFOLIO P&L SNAPSHOT"},  # snapshot — no url, drop
                {"title": "MU earnings beat estimates topped guidance",
                 "link": "https://reuters.com/mu"},  # match (40-char prefix in briefing)
                {"title": "Tesla files paperwork in Texas for new factory",
                 "link": "https://x.com/tsla"},  # no overlap
            ],
        )
        assert len(labels) == 2, (
            f"snapshot row leaked: {[l.get('title') for l in labels]}"
        )
        by_url = {l["url"]: l["in_briefing"] for l in labels}
        assert by_url["https://reuters.com/mu"] is True
        assert by_url["https://x.com/tsla"] is False
