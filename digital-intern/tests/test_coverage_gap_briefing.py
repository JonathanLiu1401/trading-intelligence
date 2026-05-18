"""analysis/claude_analyst.py — COVERAGE GAP briefing intelligence.

A news analyst's worst failure is a *silent* one: a high-value intel channel
(e.g. SEC 8-K filings) goes dark and the 5h briefing simply contains nothing
from it, so the absence reads as "no news" instead of "blind here". Live
inspection (2026-05) found sec_edgar/sec_edgar_ft with 900+ consecutive empty
polls and ZERO filings delivered, with no signal anywhere in the briefing.

These tests pin the new behaviour with specific-value assertions:

  * `_coverage_gap_lines` — only curated high-value disabled channels surface,
    ranked filings-first, with correct dark-hours math and the
    "0 delivered all session" annotation; uncurated/healthy keys excluded;
    capped so a fully-degraded host can't itself become noise.
  * `_collect_source_health` — best-effort: any failure yields {} and never
    propagates into the briefing path.
  * `_build_payload` — emits the block ONLY when an explicit report is passed
    (None ⇒ deterministic, no live DB read, section omitted).
  * `analyze` — wires the live health read into the prompt Opus receives.
  * SYSTEM_PROMPT — carries the reproduce-verbatim rule + output section.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from analysis import claude_analyst


def _report(**sources):
    """Build a source-health report dict. Each kwarg value is
    (disabled, dark_hours_or_None, consecutive_failures, total_articles)."""
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    rep = {}
    for key, (disabled, dark_h, fails, tot) in sources.items():
        if dark_h is None:
            last_seen = "not-a-timestamp"
        else:
            last_seen = (now - timedelta(hours=dark_h)).isoformat()
        rep[key] = {
            "last_seen": last_seen,
            "consecutive_failures": fails,
            "total_articles": tot,
            "disabled": disabled,
        }
    return rep, now


class TestCoverageGapLines:
    def test_disabled_sec_edgar_surfaces_with_exact_text(self):
        rep, now = _report(sec_edgar=(True, 4.0, 913, 0))
        lines = claude_analyst._coverage_gap_lines(rep, now=now)
        assert len(lines) == 1
        line = lines[0]
        assert "SEC 8-K filings" in line
        assert "DARK 4.0h" in line
        assert "913 empty polls" in line
        # tot==0 → explicit "blind all session" annotation
        assert "0 delivered all session" in line

    def test_delivered_positive_omits_zero_annotation(self):
        rep, now = _report(finnhub=(True, 2.5, 28, 1904))
        lines = claude_analyst._coverage_gap_lines(rep, now=now)
        assert len(lines) == 1
        assert "Finnhub company news" in lines[0]
        assert "DARK 2.5h" in lines[0]
        assert "0 delivered" not in lines[0]

    def test_healthy_source_excluded(self):
        """disabled=False must never appear, even for a curated key."""
        rep, now = _report(sec_edgar=(False, 0.1, 0, 5000))
        assert claude_analyst._coverage_gap_lines(rep, now=now) == []

    def test_uncurated_disabled_key_excluded(self):
        """Per-query gdelt junk / unknown hosts are noise — never surfaced
        even when disabled. Only the curated analyst-meaningful set lists."""
        rep, now = _report(**{
            "gdelt:nvidia earnings": (True, 9.0, 50, 0),
            "gdelt_gkg/iheart.com": (True, 9.0, 50, 0),
        })
        assert claude_analyst._coverage_gap_lines(rep, now=now) == []

    def test_ranking_filings_first_then_by_dark_time(self):
        """priority 0 (filings) before priority 1 (finnhub) before priority 2
        (reddit), regardless of dark-time."""
        rep, now = _report(
            reddit=(True, 99.0, 10, 0),       # pri 2, very dark
            finnhub=(True, 1.0, 5, 100),      # pri 1
            sec_edgar=(True, 0.5, 900, 0),    # pri 0, least dark
        )
        lines = claude_analyst._coverage_gap_lines(rep, now=now)
        assert len(lines) == 3
        assert "SEC 8-K filings" in lines[0]
        assert "Finnhub company news" in lines[1]
        assert "Reddit retail sentiment" in lines[2]

    def test_same_priority_longest_dark_first(self):
        """Within a priority tier, the channel dark longest sorts first."""
        rep, now = _report(
            rss=(True, 2.0, 3, 10),     # pri 1
            web=(True, 8.0, 3, 10),     # pri 1, darker
        )
        lines = claude_analyst._coverage_gap_lines(rep, now=now)
        assert "Web-scrape wire" in lines[0]
        assert "RSS feed bundle" in lines[1]

    def test_unparseable_last_seen_is_unknown_not_crash(self):
        rep, now = _report(polygon=(True, None, 800, 0))
        lines = claude_analyst._coverage_gap_lines(rep, now=now)
        assert len(lines) == 1
        assert "DARK unknown" in lines[0]
        assert "Polygon market news" in lines[0]

    def test_capped_at_max_lines(self):
        # 11 curated keys, all disabled — must clamp to _MAX_COVERAGE_LINES.
        keys = ["sec_edgar", "sec_edgar_ft", "finnhub", "polygon", "gdelt",
                "rss", "web", "alphavantage", "newsapi", "google_news",
                "reddit"]
        rep, now = _report(**{k: (True, 3.0, 5, 0) for k in keys})
        lines = claude_analyst._coverage_gap_lines(rep, now=now)
        assert len(lines) == claude_analyst._MAX_COVERAGE_LINES == 8

    def test_empty_or_garbage_report_returns_empty(self):
        assert claude_analyst._coverage_gap_lines({}) == []
        assert claude_analyst._coverage_gap_lines(None) == []
        assert claude_analyst._coverage_gap_lines("nope") == []


class TestCollectSourceHealthIsBestEffort:
    def test_returns_empty_on_import_or_query_failure(self, monkeypatch):
        """A missing/locked source_health.db must NOT break the briefing."""
        import collectors.source_health as sh

        def boom():
            raise RuntimeError("source_health.db locked")

        monkeypatch.setattr(sh, "get_health_report", boom)
        assert claude_analyst._collect_source_health() == {}

    def test_returns_report_dict_on_success(self, monkeypatch):
        import collectors.source_health as sh
        sentinel = {"sec_edgar": {"disabled": True, "last_seen": None,
                                  "consecutive_failures": 1,
                                  "total_articles": 0}}
        monkeypatch.setattr(sh, "get_health_report", lambda: sentinel)
        assert claude_analyst._collect_source_health() == sentinel


class TestBuildPayloadIntegration:
    def test_none_report_omits_section_entirely(self):
        """Backward-compatible / deterministic: the 3-arg call path must not
        emit a COVERAGE GAP block and must not touch the live DB."""
        payload = claude_analyst._build_payload(
            [{"title": "x", "source": "rss", "ai_score": 9, "summary": ""}],
            {"macro": [], "equities": []}, [],
        )
        assert "COVERAGE GAP" not in payload

    def test_explicit_report_with_gap_emits_block(self):
        rep, _ = _report(sec_edgar=(True, 4.0, 913, 0))
        payload = claude_analyst._build_payload(
            [], {"macro": [], "equities": []}, [], source_health_report=rep
        )
        assert "=== COVERAGE GAP" in payload
        assert "SEC 8-K filings" in payload
        assert "913 empty polls" in payload

    def test_explicit_report_no_curated_gap_omits_block(self):
        """A report where nothing curated is disabled → no section."""
        rep, _ = _report(sec_edgar=(False, 0.1, 0, 9))
        payload = claude_analyst._build_payload(
            [], {"macro": [], "equities": []}, [], source_health_report=rep
        )
        assert "COVERAGE GAP" not in payload


class TestAnalyzeWiresHealthIntoPrompt:
    def test_analyze_includes_gap_in_prompt_sent_to_opus(self):
        rep, _ = _report(sec_edgar=(True, 6.0, 900, 0))
        captured = {}

        def fake_claude_call(prompt, **kw):
            captured["prompt"] = prompt
            return "**DIGITAL INTERN** ok"

        with patch.object(claude_analyst, "_collect_source_health",
                          return_value=rep), \
             patch.object(claude_analyst, "claude_call",
                          side_effect=fake_claude_call):
            out = claude_analyst.analyze([], {"macro": [], "equities": []}, [])

        assert out == "**DIGITAL INTERN** ok"
        assert "=== COVERAGE GAP" in captured["prompt"]
        assert "SEC 8-K filings" in captured["prompt"]

    def test_analyze_still_returns_placeholder_when_claude_empty(self):
        """Wiring the health read in must not regress the retry sentinel."""
        with patch.object(claude_analyst, "_collect_source_health",
                          return_value={}), \
             patch.object(claude_analyst, "claude_call", return_value=None):
            out = claude_analyst.analyze([], {}, [])
        assert out == "[analyst] No response from Claude."


class TestSystemPromptCarriesRule:
    def test_prompt_has_reproduce_rule_and_output_section(self):
        sp = claude_analyst.SYSTEM_PROMPT
        assert "COVERAGE GAP" in sp
        # The rule must instruct verbatim reproduction + omit-if-absent.
        assert "intel channels" in sp
        assert "Omit the section entirely" in sp
