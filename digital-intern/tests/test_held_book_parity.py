"""Cross-prompt held-book parity audit — three prompts, one held set.

The three Claude prompts (urgency_scorer SCORE_PROMPT, alert_agent
ALERT_PROMPT, claude_analyst SYSTEM_PROMPT) each interpolate the analyst's
held universe. Per-prompt regression guards (``test_urgency_portfolio_prompt``
/ ``test_alert_held_book_prompt`` / ``test_briefing_held_book_prompt``) pin
that each prompt sees the SSOT (``ml.features.LIVE_PORTFOLIO_TICKERS``).
This file pins the structural counterpart: the three sets agree with each
other.

If any future change makes one prompt's enumeration drift from the others —
say a refactor adds a fourth prompt that hardcodes its own list, or a
helper accidentally filters the SSOT through a different transform —
``audit()['verdict']`` flips to "DRIFT" and these tests fail loudly.
"""
from __future__ import annotations

import json
from unittest.mock import patch

from analytics import held_book_parity
from ml.features import LIVE_PORTFOLIO_TICKERS


class TestAuditReportShape:
    def test_audit_returns_expected_keys(self):
        r = held_book_parity.audit()
        assert set(r.keys()) >= {"ssot_size", "prompts", "pairwise_diffs", "verdict"}
        assert isinstance(r["ssot_size"], int)
        assert isinstance(r["prompts"], dict)
        assert isinstance(r["pairwise_diffs"], dict)
        assert r["verdict"] in {"OK", "DRIFT"}

    def test_audit_covers_all_three_prompts(self):
        r = held_book_parity.audit()
        names = set(r["prompts"].keys())
        assert {"urgency_scorer", "alert_agent", "claude_analyst"} <= names

    def test_audit_reports_size_equal_to_ssot_for_alert_and_urgency(self):
        """alert_agent + urgency_scorer enumerate exactly the SSOT — extras
        would mean a literal slipped back in. (claude_analyst is a superset
        because it adds the static core; see audit().)"""
        r = held_book_parity.audit()
        ssot_size = len(LIVE_PORTFOLIO_TICKERS)
        assert r["prompts"]["alert_agent"]["size"] == ssot_size
        assert r["prompts"]["urgency_scorer"]["size"] == ssot_size

    def test_pairwise_diffs_present_for_every_pair(self):
        r = held_book_parity.audit()
        assert "alert_agent_vs_claude_analyst" in r["pairwise_diffs"]
        assert "alert_agent_vs_urgency_scorer" in r["pairwise_diffs"]
        assert "claude_analyst_vs_urgency_scorer" in r["pairwise_diffs"]


class TestParityVerdictHonest:
    """The verdict must actually flip when the three sets disagree — without
    these, a regression that silently drifts one prompt would pass."""

    def test_live_state_is_ok(self):
        """Sanity: today, the three prompts agree."""
        r = held_book_parity.audit()
        assert r["verdict"] == "OK", (
            f"Parity drift: {json.dumps(r, indent=2)}"
        )

    def test_verdict_drift_when_a_prompt_is_missing_ssot_ticker(self):
        """Simulate one prompt regressing to a frozen literal that doesn't
        include a live SSOT ticker — the audit must flag DRIFT."""
        # Pick any one live ticker as the canary missing from alert_agent.
        assert LIVE_PORTFOLIO_TICKERS, "fixture: SSOT is empty"
        canary = sorted(LIVE_PORTFOLIO_TICKERS)[0]
        live_minus_canary = LIVE_PORTFOLIO_TICKERS - {canary}
        # Force alert_agent's helper to return a phrase missing the canary;
        # the audit must catch it as missing-from-prompt and flip to DRIFT.
        with patch("watchers.alert_agent._held_book_phrase",
                   return_value="/".join(sorted(live_minus_canary))):
            r = held_book_parity.audit()
        assert r["verdict"] == "DRIFT"
        assert canary in r["prompts"]["alert_agent"]["missing_from_prompt"]

    def test_verdict_drift_when_a_prompt_has_alien_ticker(self):
        """Simulate one prompt with an extra ticker the SSOT doesn't carry —
        a stale hardcoded literal that outlived the SSOT. Audit must catch."""
        injected = "ZZZTEST"
        assert injected not in LIVE_PORTFOLIO_TICKERS, "fixture polluted"
        injected_phrase = "/".join(sorted(LIVE_PORTFOLIO_TICKERS | {injected}))
        with patch("watchers.alert_agent._held_book_phrase",
                   return_value=injected_phrase):
            r = held_book_parity.audit()
        assert r["verdict"] == "DRIFT"
        assert injected in r["prompts"]["alert_agent"]["extra_in_prompt"]


class TestStrictExitCode:
    """The --strict flag is the CI gate contract — exit 1 on DRIFT."""

    def test_strict_returns_zero_when_ok(self):
        # No patching; live state is OK per TestParityVerdictHonest.
        code = held_book_parity.main(["--json", "--strict"])
        assert code == 0

    def test_strict_returns_one_on_drift(self):
        canary = sorted(LIVE_PORTFOLIO_TICKERS)[0]
        live_minus = LIVE_PORTFOLIO_TICKERS - {canary}
        with patch("watchers.alert_agent._held_book_phrase",
                   return_value="/".join(sorted(live_minus))):
            code = held_book_parity.main(["--strict"])
        assert code == 1


class TestSSOTSource:
    def test_ssot_set_equals_live_portfolio_tickers(self):
        """The SSOT used by the audit MUST be the same import all three prompt
        helpers consume — otherwise the audit could pass while every prompt
        silently uses a different source."""
        assert held_book_parity._ssot_set() == LIVE_PORTFOLIO_TICKERS
