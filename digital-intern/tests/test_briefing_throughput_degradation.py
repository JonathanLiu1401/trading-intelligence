"""``analysis.claude_analyst._throughput_degradation_lines`` and its wiring
into ``_build_payload`` — the early-warning complement to COVERAGE GAP.

COVERAGE GAP only surfaces sources the FAILURE_THRESHOLD has pushed to
``disabled`` (a binary, late signal). A live source can be quietly losing
most of its throughput without ever crossing that bar; the analyst's
"stale sources" complaint applies equally to a marginally-alive source as
to a fully dark one. ``ArticleStore.source_throughput`` already detects
this; this feature finally surfaces it to Opus.

Tests pin the operational contract with specific numbers (not "no crash"):
  * the renderer respects ``min_prior`` (tiny baseline → no line, no
    wall-of-text alarm for sources with insignificant prior flow);
  * it respects ``min_decel_pct`` (mild slowdowns stay silent);
  * it skips ``decel_pct=None`` (no-baseline / brand-new sources have no
    measurable degradation);
  * sort order is largest absolute loss first, tiebreak on higher prior
    (a 50→0 source matters more than a 20→0 source — both 100% decel);
  * ``_build_payload`` emits the section ONLY when a throughput report is
    explicitly supplied — the no-arg path stays byte-deterministic, same
    discipline as ``source_health_report`` / ``prior_digest``;
  * the SYSTEM_PROMPT reproduces the new section (a non-prompt that
    silently dropped the input would defeat the whole feature).
"""
from __future__ import annotations

import pytest

from analysis import claude_analyst


class TestThroughputDegradationLines:
    def test_below_min_prior_is_skipped(self):
        """A 5→0 drop is 100% decel but only 5 lost rows in an hour — a tiny
        baseline. The renderer must not produce a line: tiny absolute loss
        is exactly the noise this section is designed to suppress."""
        rows = [{"source": "tiny", "recent": 0, "prior": 5,
                 "delta": -5, "decel_pct": 100.0}]
        assert claude_analyst._throughput_degradation_lines(rows) == []

    def test_below_min_decel_pct_is_skipped(self):
        """A 100→70 drop is only 30% decel — a normal news-rate fluctuation,
        not an analyst-actionable degradation."""
        rows = [{"source": "mild", "recent": 70, "prior": 100,
                 "delta": -30, "decel_pct": 30.0}]
        assert claude_analyst._throughput_degradation_lines(rows) == []

    def test_no_baseline_is_skipped(self):
        """``decel_pct=None`` means the source had no prior baseline — it
        is either brand-new or just-recovered; nothing to report.
        ``source_throughput`` already documents this convention."""
        rows = [{"source": "fresh", "recent": 5, "prior": 0,
                 "delta": 5, "decel_pct": None}]
        assert claude_analyst._throughput_degradation_lines(rows) == []

    def test_accelerating_source_is_skipped(self):
        """An accelerating source has negative decel_pct — the opposite of
        a degradation signal."""
        rows = [{"source": "hot", "recent": 50, "prior": 10,
                 "delta": 40, "decel_pct": -400.0}]
        assert claude_analyst._throughput_degradation_lines(rows) == []

    def test_significant_degradation_produces_line(self):
        """The flagship case: a productive source (prior=40) collapsed to 5
        in the recent window (decel=87.5%) — the analyst must be told.
        Format mirrors COVERAGE GAP for renderer parity."""
        rows = [{"source": "RSS Reuters Markets", "recent": 5, "prior": 40,
                 "delta": -35, "decel_pct": 87.5}]
        lines = claude_analyst._throughput_degradation_lines(rows)
        assert lines == [
            "RSS Reuters Markets — 5 in last 60min "
            "(vs 40 prior; -88%)"
        ]

    def test_orders_by_absolute_loss_desc(self):
        """A 50→0 source matters more than a 20→0 source — both are 100%
        decel, but the absolute loss is 2.5× greater. The renderer must
        sort largest absolute loss first so the biggest blind spot is
        always at the top of the section.

        And ties on absolute loss break on higher prior (which encodes
        baseline magnitude when recent counts differ but loss is equal)."""
        rows = [
            {"source": "small", "recent": 0, "prior": 20,
             "delta": -20, "decel_pct": 100.0},
            {"source": "big",   "recent": 0, "prior": 50,
             "delta": -50, "decel_pct": 100.0},
            {"source": "mid",   "recent": 0, "prior": 30,
             "delta": -30, "decel_pct": 100.0},
        ]
        lines = claude_analyst._throughput_degradation_lines(rows)
        # Largest absolute loss first.
        assert lines[0].startswith("big —")
        assert lines[1].startswith("mid —")
        assert lines[2].startswith("small —")

    def test_caps_at_max_lines(self):
        """No matter how many sources degrade, the section must cap at
        ``_MAX_DEGRADATION_LINES`` so it can never itself become noise."""
        rows = [
            {"source": f"s{i}", "recent": 0, "prior": 100 - i,
             "delta": -(100 - i), "decel_pct": 100.0}
            for i in range(20)
        ]
        lines = claude_analyst._throughput_degradation_lines(rows)
        assert len(lines) == claude_analyst._MAX_DEGRADATION_LINES

    def test_empty_input_returns_empty(self):
        assert claude_analyst._throughput_degradation_lines([]) == []
        assert claude_analyst._throughput_degradation_lines(None) == []  # type: ignore

    def test_malformed_rows_are_skipped(self):
        """A None/non-dict entry in the list (a future producer change, a
        manual replay) must NOT crash — just be skipped, same robustness
        discipline as ``_coverage_gap_lines``."""
        rows = [
            None,
            "not a dict",
            {"source": "good", "recent": 5, "prior": 50,
             "delta": -45, "decel_pct": 90.0},
        ]
        lines = claude_analyst._throughput_degradation_lines(rows)  # type: ignore
        assert len(lines) == 1
        assert lines[0].startswith("good —")

    def test_ties_on_loss_and_prior_do_not_crash(self):
        """Two rows with identical (abs_loss, prior) must sort cleanly — the
        tuple's trailing element used to be the row dict, so Python fell back
        to comparing dicts and raised ``TypeError: '<' not supported between
        instances of 'dict' and 'dict'`` for the entire ``_throughput_
        degradation_lines`` call (live evidence: bubbled to ``analyze()`` and
        blanked a 5h heartbeat). The tiebreaker is now the source name so a
        stable string comparison resolves the tie deterministically."""
        rows = [
            {"source": "rss_b", "recent": 0, "prior": 50,
             "delta": -50, "decel_pct": 100.0},
            {"source": "rss_a", "recent": 0, "prior": 50,
             "delta": -50, "decel_pct": 100.0},
        ]
        lines = claude_analyst._throughput_degradation_lines(rows)
        assert len(lines) == 2
        # Alphabetical tiebreak on identical abs_loss/prior — both lines emit.
        assert lines[0].startswith("rss_a —")
        assert lines[1].startswith("rss_b —")


class TestBuildPayloadWiring:
    def test_section_emitted_when_throughput_provided(self):
        """When throughput data is explicitly passed AND has a qualifying
        row, the section appears with the canonical header."""
        rows = [{"source": "RSS Reuters", "recent": 5, "prior": 40,
                 "delta": -35, "decel_pct": 87.5}]
        payload = claude_analyst._build_payload(
            articles=[], stock_data={}, earnings=[],
            source_throughput=rows,
        )
        assert "THROUGHPUT DEGRADATION" in payload, (
            "explicit throughput data with qualifying rows must produce "
            "the analyst-facing section"
        )
        assert "RSS Reuters — 5 in last 60min" in payload

    def test_section_omitted_when_no_throughput_arg(self):
        """The no-arg path must stay byte-deterministic — same discipline
        as source_health_report / prior_digest. A test that builds a
        payload without throughput context must not see the section."""
        payload = claude_analyst._build_payload(
            articles=[], stock_data={}, earnings=[],
        )
        assert "THROUGHPUT DEGRADATION" not in payload

    def test_section_omitted_when_empty_throughput(self):
        """Explicit [] means "we checked and nothing qualifies" — the
        section must NOT appear (a header with no body is itself noise)."""
        payload = claude_analyst._build_payload(
            articles=[], stock_data={}, earnings=[],
            source_throughput=[],
        )
        assert "THROUGHPUT DEGRADATION" not in payload

    def test_section_omitted_when_all_rows_below_threshold(self):
        """Throughput data was supplied but no source qualifies — the
        section must NOT appear (same discipline as COVERAGE GAP when
        every channel is healthy)."""
        rows = [
            # Below min_prior:
            {"source": "tiny", "recent": 0, "prior": 5,
             "delta": -5, "decel_pct": 100.0},
            # Below min_decel_pct:
            {"source": "mild", "recent": 70, "prior": 100,
             "delta": -30, "decel_pct": 30.0},
        ]
        payload = claude_analyst._build_payload(
            articles=[], stock_data={}, earnings=[],
            source_throughput=rows,
        )
        assert "THROUGHPUT DEGRADATION" not in payload


class TestSystemPromptCoverage:
    """The SYSTEM_PROMPT must actually instruct Opus to reproduce the new
    section AND list it in the OUTPUT FORMAT skeleton. A silent input
    block Opus had no rule for would defeat the entire feature."""

    def test_rule_mentions_throughput_degradation(self):
        assert "THROUGHPUT DEGRADATION" in claude_analyst.SYSTEM_PROMPT
        # Must be paired with the omit-when-absent discipline (same shape
        # as COVERAGE GAP / BOOK HEAT / AGING TOP ROWS / PRIOR DIGEST).
        idx = claude_analyst.SYSTEM_PROMPT.find("THROUGHPUT DEGRADATION")
        rule_excerpt = claude_analyst.SYSTEM_PROMPT[idx:idx + 800]
        assert "Omit the section entirely if no degradation block is provided" in rule_excerpt
