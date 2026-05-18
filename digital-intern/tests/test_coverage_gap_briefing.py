"""analysis/claude_analyst.py — COVERAGE GAP briefing intelligence.

A news analyst's worst failure is a *silent* one: a high-value intel channel
(e.g. SEC 8-K filings) goes dark and the 5h briefing simply contains nothing
from it, so the absence reads as "no news" instead of "blind here". Live
inspection (2026-05) found sec_edgar/sec_edgar_ft with 900+ consecutive empty
polls and ZERO filings delivered, with no signal anywhere in the briefing.

CORRECTED DARK-DURATION CONTRACT (2026-05-18). The prior revision of these
tests modelled ``last_seen`` as "last delivery time" (``now - dark_hours``)
and asserted ``DARK {(now-last_seen)}h``. That contract does NOT match
production: ``collectors.source_health.record_result`` rewrites
``last_seen = now`` on *every* poll, including the empty polls of a disabled
channel (it is "last poll", and ``get_stale_sources`` legitimately depends on
that). So in the live daemon ``now - last_seen`` is ≈0 for any
actively-polled disabled source — the 5h briefing literally read
"SEC 8-K filings — DARK 0.0h (932 empty polls, 0 delivered all session)",
telling the analyst a channel blind the *entire* session was negligible. The
old tests fabricated a scenario source_health never produces, which is why
the bug shipped invisibly. ``_coverage_gap_lines`` now estimates the dark
duration from ``consecutive_failures × the channel's poll cadence``
(``_COVERAGE_POLL_SECS``) and prefixes it with ``~`` to flag it as an
estimate. These assertions are updated to the production-accurate contract;
``test_production_last_seen_is_now_high_fails_still_reports_long_dark`` is the
discriminating regression that the prior suite was missing.

These tests pin the behaviour with specific-value assertions:

  * `_coverage_gap_lines` — only curated high-value disabled channels surface,
    ranked filings-first, with the corrected estimated-dark math and the
    "0 delivered all session" annotation; uncurated/healthy keys excluded;
    capped so a fully-degraded host can't itself become noise.
  * `_COVERAGE_POLL_SECS` — keys are a superset of `_COVERAGE_LABELS` (a
    labelled channel without a cadence silently degrades to "DARK unknown").
  * `_collect_source_health` — best-effort: any failure yields {} and never
    propagates into the briefing path.
  * `_build_payload` — emits the block ONLY when an explicit report is passed
    (None ⇒ deterministic, no live DB read, section omitted).
  * `analyze` — wires the live health read into the prompt Opus receives.
  * SYSTEM_PROMPT — carries the reproduce-verbatim rule + output section.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from analysis import claude_analyst


def _report(**sources):
    """Build a source-health report dict shaped exactly as
    ``source_health.get_health_report()`` produces it: ``last_seen`` is the
    *last poll* time, which for an actively-polled disabled channel is ≈now.

    Each kwarg value is ``(disabled, consecutive_failures, total_articles)``
    or ``(disabled, consecutive_failures, total_articles, last_seen_override)``
    — the override only exercises that ``last_seen`` no longer influences the
    dark estimate at all (it is robustly ignored)."""
    now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    rep = {}
    for key, val in sources.items():
        if len(val) == 4:
            disabled, fails, tot, last_seen = val
        else:
            disabled, fails, tot = val
            last_seen = now.isoformat()  # production: last poll just happened
        rep[key] = {
            "last_seen": last_seen,
            "consecutive_failures": fails,
            "total_articles": tot,
            "disabled": disabled,
        }
    return rep, now


def _expected_dark(key: str, fails: int) -> str:
    """The ~Xh string _coverage_gap_lines must now produce for (key, fails)."""
    secs = claude_analyst._COVERAGE_POLL_SECS[key]
    return f"~{fails * secs / 3600.0:.1f}h"


class TestCoveragePollSecsParity:
    def test_poll_secs_is_superset_of_labels(self):
        """Every curated label MUST have a cadence or it silently degrades to
        'DARK unknown' in production (the exact failure class this fix
        addresses, just one layer up)."""
        labels = set(claude_analyst._COVERAGE_LABELS)
        cadences = set(claude_analyst._COVERAGE_POLL_SECS)
        assert labels <= cadences, (
            f"labels missing a poll cadence: {labels - cadences}"
        )

    def test_cadences_are_positive_ints(self):
        for k, v in claude_analyst._COVERAGE_POLL_SECS.items():
            assert isinstance(v, int) and v > 0, (k, v)


class TestCoverageGapLines:
    def test_disabled_sec_edgar_surfaces_with_exact_text(self):
        rep, now = _report(sec_edgar=(True, 913, 0))
        lines = claude_analyst._coverage_gap_lines(rep, now=now)
        assert len(lines) == 1
        line = lines[0]
        assert "SEC 8-K filings" in line
        # 913 empty polls × 300s cadence ≈ 76.1h dark — the HONEST estimate,
        # not the misleading "DARK 0.0h" the last_seen-delta produced live.
        assert f"DARK {_expected_dark('sec_edgar', 913)}" in line  # ~76.1h
        assert "913 empty polls" in line
        # tot==0 → explicit "blind all session" annotation
        assert "0 delivered all session" in line

    def test_delivered_positive_omits_zero_annotation(self):
        rep, now = _report(finnhub=(True, 28, 1904))
        lines = claude_analyst._coverage_gap_lines(rep, now=now)
        assert len(lines) == 1
        assert "Finnhub company news" in lines[0]
        assert f"DARK {_expected_dark('finnhub', 28)}" in lines[0]  # ~2.3h
        assert "0 delivered" not in lines[0]

    def test_healthy_source_excluded(self):
        """disabled=False must never appear, even for a curated key."""
        rep, now = _report(sec_edgar=(False, 0, 5000))
        assert claude_analyst._coverage_gap_lines(rep, now=now) == []

    def test_uncurated_disabled_key_excluded(self):
        """Per-query gdelt junk / unknown hosts are noise — never surfaced
        even when disabled. Only the curated analyst-meaningful set lists."""
        rep, now = _report(**{
            "gdelt:nvidia earnings": (True, 50, 0),
            "gdelt_gkg/iheart.com": (True, 50, 0),
        })
        assert claude_analyst._coverage_gap_lines(rep, now=now) == []

    def test_ranking_filings_first_then_by_dark_time(self):
        """priority 0 (filings) before priority 1 (finnhub) before priority 2
        (reddit), regardless of dark-time."""
        rep, now = _report(
            reddit=(True, 10, 0),       # pri 2
            finnhub=(True, 5, 100),     # pri 1
            sec_edgar=(True, 900, 0),   # pri 0
        )
        lines = claude_analyst._coverage_gap_lines(rep, now=now)
        assert len(lines) == 3
        assert "SEC 8-K filings" in lines[0]
        assert "Finnhub company news" in lines[1]
        assert "Reddit retail sentiment" in lines[2]

    def test_same_priority_longest_dark_first(self):
        """Within a priority tier, the channel dark longest sorts first.
        web: 20×60s = 0.33h ; rss: 3×30s = 0.025h → web first."""
        rep, now = _report(
            rss=(True, 3, 10),     # pri 1, 0.025h
            web=(True, 20, 10),    # pri 1, 0.33h — darker
        )
        lines = claude_analyst._coverage_gap_lines(rep, now=now)
        assert "Web-scrape wire" in lines[0]
        assert "RSS feed bundle" in lines[1]

    def test_zero_failures_is_unknown_not_crash(self):
        """No consecutive_failures yet (or an unknown cadence) → 'DARK
        unknown', never a crash and never a fabricated 0.0h."""
        rep, now = _report(polygon=(True, 0, 0))
        lines = claude_analyst._coverage_gap_lines(rep, now=now)
        assert len(lines) == 1
        assert "DARK unknown" in lines[0]
        assert "Polygon market news" in lines[0]

    def test_garbage_last_seen_is_ignored_not_crashing(self):
        """last_seen no longer feeds the estimate at all — even a non-string /
        unparseable value must not crash and must not change the ~Xh math."""
        rep, now = _report(sec_edgar=(True, 100, 0, "not-a-timestamp"))
        lines = claude_analyst._coverage_gap_lines(rep, now=now)
        assert len(lines) == 1
        assert f"DARK {_expected_dark('sec_edgar', 100)}" in lines[0]

    def test_capped_at_max_lines(self):
        # 11 curated keys, all disabled — must clamp to _MAX_COVERAGE_LINES.
        keys = ["sec_edgar", "sec_edgar_ft", "finnhub", "polygon", "gdelt",
                "rss", "web", "alphavantage", "newsapi", "google_news",
                "reddit"]
        rep, now = _report(**{k: (True, 5, 0) for k in keys})
        lines = claude_analyst._coverage_gap_lines(rep, now=now)
        assert len(lines) == claude_analyst._MAX_COVERAGE_LINES == 8

    def test_empty_or_garbage_report_returns_empty(self):
        assert claude_analyst._coverage_gap_lines({}) == []
        assert claude_analyst._coverage_gap_lines(None) == []
        assert claude_analyst._coverage_gap_lines("nope") == []

    def test_production_last_seen_is_now_high_fails_still_reports_long_dark(
        self,
    ):
        """DISCRIMINATING REGRESSION (the suite previously lacked this).

        Reproduce the exact production shape: source_health rewrote
        last_seen=now on the most recent (empty) poll, yet the channel has
        932 consecutive failures and delivered nothing all session. The OLD
        last_seen-delta code produced 'DARK 0.0h' here (observed live). The
        line MUST instead report a large, honest estimate.
        """
        now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
        rep = {"sec_edgar": {
            "last_seen": now.isoformat(),       # last poll = right now
            "consecutive_failures": 932,
            "total_articles": 0,
            "disabled": True,
        }}
        lines = claude_analyst._coverage_gap_lines(rep, now=now)
        assert len(lines) == 1
        line = lines[0]
        assert "DARK 0.0h" not in line
        assert "DARK ~0.0h" not in line
        # 932 × 300s / 3600 = 77.666… → ~77.7h
        assert f"DARK {_expected_dark('sec_edgar', 932)}" in line
        assert "~77.7h" in line
        assert "0 delivered all session" in line


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
        rep, _ = _report(sec_edgar=(True, 913, 0))
        payload = claude_analyst._build_payload(
            [], {"macro": [], "equities": []}, [], source_health_report=rep
        )
        assert "=== COVERAGE GAP" in payload
        assert "SEC 8-K filings" in payload
        assert "913 empty polls" in payload

    def test_explicit_report_no_curated_gap_omits_block(self):
        """A report where nothing curated is disabled → no section."""
        rep, _ = _report(sec_edgar=(False, 0, 9))
        payload = claude_analyst._build_payload(
            [], {"macro": [], "equities": []}, [], source_health_report=rep
        )
        assert "COVERAGE GAP" not in payload


class TestAnalyzeWiresHealthIntoPrompt:
    def test_analyze_includes_gap_in_prompt_sent_to_opus(self):
        rep, _ = _report(sec_edgar=(True, 900, 0))
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
