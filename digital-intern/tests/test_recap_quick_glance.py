"""Recap-template gate — the Zacks "A Quick Glance at Key Metrics" fingerprint.

Live evidence (2026-05-21 NVDA earnings night, articles.db urgency=2 set):
the Zacks post-earnings recap-mill title

    "NVIDIA Earnings: A Quick Glance at Key Metrics"

reached urgency=2 THREE times — YahooFinance/NVDA (ml_score 9.9),
yfinance/Zacks (ml_score 9.7), and GN: Nvidia (ai_score 9.0 — Sonnet itself
over-scored it). All three publishers are above the 0.45
ALERT_MIN_LONE_SOURCE_CRED bar so the source-authority gate does not catch
them; the content type IS the failure — a "Quick Glance" summary is written
AFTER the print crossed the wire, so it is retrospective recap, never breaking.

This pins the new ``quick_glance_metrics`` fingerprint on both consumed
products:
  - the 🚨 BREAKING alert path (``watchers.alert_agent``), which also feeds
    ``watchers.urgency_scorer``'s pre-floor via the SSOT import; and
  - the 5h Opus heartbeat briefing path (``analysis.claude_analyst``).
Plus the must-survive corpus (real earnings movers are NEVER caught) and an
end-to-end ``send_urgent_alert`` integration.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from watchers import alert_agent
from analysis import claude_analyst


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


# The verbatim live-failure titles (2026-05-21) plus the Zacks variant that
# uses "Key Financial Metrics" — all must be caught with the same name.
_LIVE_QUICK_GLANCE = [
    "NVIDIA Earnings: A Quick Glance at Key Metrics",
    "NVIDIA Earnings: A Quick Glance at Key Metrics - Zacks Investment Research",
    "Micron Technology Q3 Earnings: A Quick Glance at Key Financial Metrics",
    "Oracle Earnings: A Quick Glance at Metrics",
]

# Real breaking earnings headlines — content the analyst MUST still receive.
_MUST_SURVIVE = [
    "Nvidia Q1 revenue rises 22% to $35.1 billion, beats estimates",
    "MU earnings blow past estimates as DRAM pricing surges",
    "NVIDIA Earnings Today: Wall Street Expects EPS to Jump",
    "Fed cuts rates 50bp in surprise emergency move",
    "Micron halts shipments to China after new export ban",
    "Oracle Q2 earnings preview: what analysts expect",
]


class TestAlertGateCatchesQuickGlance:
    def test_live_titles_caught_with_correct_name(self):
        for title in _LIVE_QUICK_GLANCE:
            hit, name = alert_agent._looks_like_recap_template({"title": title})
            assert hit, f"alert gate missed Zacks recap: {title!r}"
            assert name == "quick_glance_metrics", (
                f"wrong fingerprint for {title!r}: got {name!r}"
            )

    def test_must_survive_corpus_not_caught(self):
        for title in _MUST_SURVIVE:
            hit, name = alert_agent._looks_like_recap_template({"title": title})
            assert not hit, (
                f"alert gate WRONGLY suppressed real headline {title!r} "
                f"as {name!r}"
            )


class TestBriefingGateCatchesQuickGlance:
    def test_live_titles_caught_with_correct_name(self):
        for title in _LIVE_QUICK_GLANCE:
            hit, name = claude_analyst._looks_like_recap_template(
                {"title": title}
            )
            assert hit, f"briefing gate missed Zacks recap: {title!r}"
            assert name == "quick_glance_metrics", (
                f"wrong fingerprint for {title!r}: got {name!r}"
            )

    def test_must_survive_corpus_not_caught(self):
        for title in _MUST_SURVIVE:
            hit, _ = claude_analyst._looks_like_recap_template({"title": title})
            assert not hit, (
                f"briefing gate WRONGLY suppressed real headline {title!r}"
            )


def test_alert_and_briefing_gates_agree_on_quick_glance():
    """Lockstep parity: the new fingerprint must fire identically on both
    consumed products with the same fingerprint name — a regex edit to one
    gate but not the other is caught here."""
    for title in _LIVE_QUICK_GLANCE:
        a_hit, a_name = alert_agent._looks_like_recap_template({"title": title})
        b_hit, b_name = claude_analyst._looks_like_recap_template(
            {"title": title}
        )
        assert a_hit and b_hit, f"gate disagreement on {title!r}"
        assert a_name == b_name == "quick_glance_metrics", (
            f"fingerprint name drifted for {title!r}: "
            f"alert={a_name!r} briefing={b_name!r}"
        )


def test_urgency_scorer_prefloors_quick_glance_via_ssot():
    """``watchers.urgency_scorer`` imports ``_looks_like_recap_template`` from
    alert_agent (SSOT). The new fingerprint must therefore reach the
    urgency_scorer pre-floor path with no local fork."""
    from watchers import urgency_scorer as us
    assert us._looks_like_recap_template is alert_agent._looks_like_recap_template
    hit, name = us._looks_like_recap_template(
        {"title": "NVIDIA Earnings: A Quick Glance at Key Metrics"}
    )
    assert hit and name == "quick_glance_metrics"


class _StoreSpy:
    """Records mark_alerted calls without touching SQLite."""

    def __init__(self):
        self.marked: list[str] = []

    def mark_alerted_batch(self, ids):
        self.marked.extend(ids)

    def mark_alerted(self, aid):
        self.marked.append(aid)


def _row(_id="x", title="generic", source="GN: Nvidia", **kw) -> dict:
    base = {
        "_id": _id, "link": f"https://news.example.com/{_id}",
        "title": title, "source": source, "ai_score": 9.0,
        "summary": "", "published": _iso(0.2), "first_seen": _iso(0.1),
    }
    base.update(kw)
    return base


class TestSendUrgentAlertIntegration:
    def test_all_quick_glance_batch_never_reaches_claude(self, monkeypatch):
        """The exact 2026-05-21 live failure case end-to-end: a batch made
        entirely of "A Quick Glance at Key Metrics" recap rows must (a) never
        reach Claude/Discord, (b) be marked alerted so each row exits the
        urgent queue instead of churning every 20s."""
        spy = _StoreSpy()
        batch = [
            _row(_id="qg1",
                 title="NVIDIA Earnings: A Quick Glance at Key Metrics",
                 source="YahooFinance/NVDA"),
            _row(_id="qg2",
                 title="NVIDIA Earnings: A Quick Glance at Key Metrics - "
                       "Zacks Investment Research",
                 source="yfinance/Zacks"),
        ]
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert(batch, spy)
        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert sorted(spy.marked) == ["qg1", "qg2"], (
            "quick-glance recap rows not marked alerted — would re-fetch "
            "every 20s and re-fire on the next batch"
        )

    def test_mixed_batch_only_real_story_reaches_prompt(self, monkeypatch):
        """A real breaking row in the same batch as a quick-glance recap:
        only the real story feeds the alert prompt; the recap is still
        marked alerted alongside it."""
        spy = _StoreSpy()
        real = _row(_id="real",
                    title="Micron halts shipments to China after export ban",
                    source="reuters")
        recap = _row(_id="qg",
                     title="NVIDIA Earnings: A Quick Glance at Key Metrics",
                     source="yfinance/Zacks")
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="ALERT BODY") as mock_claude, \
             patch("notifier.discord_notifier.send", return_value=True):
            ok = alert_agent.send_urgent_alert([real, recap], spy)
        assert ok is True
        mock_claude.assert_called_once()
        prompt = mock_claude.call_args[0][0]
        assert "Micron halts shipments" in prompt
        assert "Quick Glance" not in prompt, (
            "recap row leaked into the alert prompt"
        )
        assert "qg" in spy.marked and "real" in spy.marked
