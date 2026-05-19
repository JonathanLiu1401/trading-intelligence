"""``analysis.claude_analyst._alert_velocity_lines`` and its wiring into
``_build_payload`` — the BREAKING-wire firing-rate magnitude hint.

The 🚨 BREAKING alert path is the analyst's most time-critical product. Its
RAW firing rate over a 5h window vs the prior 5h window carries a magnitude
signal no individual story score can express: 24 alerts vs 8 prior tells
Opus the wire is materially hot (a real macro event under way — Fed
surprise, geopolitical escalation, broad selloff). 2 vs 12 tells Opus the
wire is unusually quiet, so individual stories this window deserve closer
scrutiny than the same scores in a busy window.

Tests pin the operational contract with SPECIFIC numbers (not "no crash"):
  * tiny totals stay silent (a 1→3 swing on a sleepy wire is noise);
  * mild changes stay silent (a 25% swing is normal variance);
  * a materially-hot wire emits a hot line with exact recent/prior/% values;
  * a materially-cooling wire emits a cooling line, ditto;
  * the newly-lit / newly-silent edges bypass the percentage gate;
  * ``_build_payload`` emits the section ONLY when a velocity dict is
    explicitly supplied — the no-arg path stays byte-deterministic (same
    discipline as ``source_throughput`` / ``source_health_report`` /
    ``prior_digest``);
  * the SYSTEM_PROMPT reproduces the new section (a non-prompt that silently
    dropped the input would defeat the whole feature).
"""
from __future__ import annotations

import pytest

from analysis import claude_analyst


class TestAlertVelocityLines:
    def test_returns_empty_on_none_or_non_dict(self):
        assert claude_analyst._alert_velocity_lines(None) == []
        assert claude_analyst._alert_velocity_lines([]) == []  # type: ignore[arg-type]
        assert claude_analyst._alert_velocity_lines("x") == []  # type: ignore[arg-type]

    def test_returns_empty_when_total_below_minimum(self):
        """1→3 is +200% but only 4 alerts total — a sleepy wire's micro-
        fluctuation, not analyst-actionable. Must stay silent."""
        assert claude_analyst._alert_velocity_lines(
            {"recent": 3, "prior": 1, "window_h": 5}
        ) == []

    def test_returns_empty_when_change_below_min_delta(self):
        """100→80 is -20%, normal news-rate variance. Must stay silent."""
        assert claude_analyst._alert_velocity_lines(
            {"recent": 80, "prior": 100, "window_h": 5}
        ) == []

    def test_hot_wire_emits_exact_message(self):
        """24 vs 8 is +200% — well above the 50% gate. The output line MUST
        carry the exact numbers and "+200%" + "hot" verdict."""
        lines = claude_analyst._alert_velocity_lines(
            {"recent": 24, "prior": 8, "window_h": 5}
        )
        assert len(lines) == 1
        assert lines[0] == (
            "BREAKING wire fired 24 alerts in last 5h vs 8 in prior 5h "
            "(+200%) — wire materially hot"
        )

    def test_cooling_wire_emits_exact_message(self):
        """2 vs 12 is -83% (recent+prior=14 >= 5; |delta|>=50). Cooling
        verdict, no leading '+' on negative delta."""
        lines = claude_analyst._alert_velocity_lines(
            {"recent": 2, "prior": 12, "window_h": 5}
        )
        assert len(lines) == 1
        assert lines[0] == (
            "BREAKING wire fired 2 alerts in last 5h vs 12 in prior 5h "
            "(-83%) — wire materially cooling"
        )

    def test_newly_lit_wire_bypasses_percentage_gate(self):
        """recent=5, prior=0 — ratio undefined, but the absolute change IS
        the signal. Must emit a "newly active" line."""
        lines = claude_analyst._alert_velocity_lines(
            {"recent": 5, "prior": 0, "window_h": 5}
        )
        assert len(lines) == 1
        assert lines[0] == (
            "BREAKING wire fired 5 alert(s) in last 5h vs 0 in prior 5h — "
            "wire newly active"
        )

    def test_newly_silent_wire_bypasses_percentage_gate(self):
        """recent=0, prior=8 — the wire went dark. Must emit a "silent" line
        (the percentage would be -100% but we handle this edge explicitly
        so the wording reads naturally — "0 alerts" not "wire materially
        cooling")."""
        lines = claude_analyst._alert_velocity_lines(
            {"recent": 0, "prior": 8, "window_h": 5}
        )
        assert len(lines) == 1
        assert lines[0] == (
            "BREAKING wire fired 0 alerts in last 5h vs 8 in prior 5h — "
            "wire silent"
        )

    def test_newly_lit_below_min_total_stays_silent(self):
        """recent=2, prior=0 — too few alerts to be a real wire-event signal.
        Must stay silent (the special-case branch must respect min_total)."""
        assert claude_analyst._alert_velocity_lines(
            {"recent": 2, "prior": 0, "window_h": 5}
        ) == []

    def test_newly_silent_below_min_total_stays_silent(self):
        """recent=0, prior=3 — the prior wire was already too quiet for
        the absence to be meaningful. Must stay silent."""
        assert claude_analyst._alert_velocity_lines(
            {"recent": 0, "prior": 3, "window_h": 5}
        ) == []

    def test_doubling_at_threshold_emits(self):
        """6 vs 4 is +50% exactly; recent+prior=10 (>=5). At-threshold must
        emit (the comparison is >=, not >). Pinned so a future thresh tweak
        is visible."""
        lines = claude_analyst._alert_velocity_lines(
            {"recent": 6, "prior": 4, "window_h": 5}
        )
        assert len(lines) == 1
        assert "+50%" in lines[0]
        assert "hot" in lines[0]

    def test_window_hours_is_reflected_in_text(self):
        """A 1h window briefing variant must render '1h' in the output (the
        period label is data-driven, not hardcoded)."""
        lines = claude_analyst._alert_velocity_lines(
            {"recent": 12, "prior": 4, "window_h": 1}
        )
        assert len(lines) == 1
        assert "in last 1h" in lines[0]
        assert "in prior 1h" in lines[0]

    def test_malformed_dict_returns_empty(self):
        """A garbage payload (non-numeric, missing keys, negative values)
        must degrade gracefully to [] — best-effort upstream means the dict
        could legitimately be missing fields."""
        assert claude_analyst._alert_velocity_lines({}) == []
        assert claude_analyst._alert_velocity_lines(
            {"recent": "x", "prior": 8, "window_h": 5}
        ) == []
        assert claude_analyst._alert_velocity_lines(
            {"recent": -1, "prior": 8, "window_h": 5}
        ) == []
        assert claude_analyst._alert_velocity_lines(
            {"recent": 10, "prior": 5, "window_h": 0}
        ) == []
        assert claude_analyst._alert_velocity_lines(
            {"recent": 10, "prior": 5, "window_h": -3}
        ) == []


class TestBuildPayloadAlertVelocityWiring:
    def test_emit_when_velocity_signals_hot(self):
        out = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
            alert_velocity={"recent": 24, "prior": 8, "window_h": 5},
        )
        assert "=== ALERT VELOCITY" in out
        # The exact rendered line must appear so a non-conforming Opus could
        # still recover it via the prompt's "verbatim" rule.
        assert (
            "BREAKING wire fired 24 alerts in last 5h vs 8 in prior 5h "
            "(+200%) — wire materially hot"
        ) in out

    def test_omit_when_velocity_below_threshold(self):
        """A 10→8 wire (-20%) emits NO velocity section — same "below the
        bar means silent" discipline as COVERAGE GAP / THROUGHPUT
        DEGRADATION, so an unremarkable window stays clean."""
        out = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
            alert_velocity={"recent": 8, "prior": 10, "window_h": 5},
        )
        assert "ALERT VELOCITY" not in out

    def test_omit_entirely_when_arg_is_none(self):
        """The default path (no alert_velocity kwarg) MUST be byte-identical
        to the pre-feature behaviour for that section — every caller / test
        that doesn't pass it stays unaffected. Same discipline as
        source_health_report / prior_digest / source_throughput."""
        out_default = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
        )
        out_explicit_none = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
            alert_velocity=None,
        )
        assert "ALERT VELOCITY" not in out_default
        assert out_default == out_explicit_none

    def test_emit_for_cooling_wire(self):
        out = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
            alert_velocity={"recent": 2, "prior": 12, "window_h": 5},
        )
        assert "=== ALERT VELOCITY" in out
        assert "wire materially cooling" in out
        assert "(-83%)" in out

    def test_emit_for_newly_lit_wire(self):
        out = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
            alert_velocity={"recent": 7, "prior": 0, "window_h": 5},
        )
        assert "=== ALERT VELOCITY" in out
        assert "wire newly active" in out


class TestSystemPromptCoverage:
    def test_system_prompt_includes_alert_velocity_rule(self):
        """The prompt MUST instruct Opus to reproduce the new section,
        otherwise the input block is silently dropped from the output and
        the analyst never sees it. Same protection as the other reproduced
        sections (COVERAGE GAP, THROUGHPUT DEGRADATION)."""
        assert "ALERT VELOCITY" in claude_analyst.SYSTEM_PROMPT
        # The reproduction rule must be explicit and bidirectional (emit
        # when present, omit when absent) so Opus doesn't fabricate it.
        assert "Omit the section entirely if no velocity block" in (
            claude_analyst.SYSTEM_PROMPT
        )
