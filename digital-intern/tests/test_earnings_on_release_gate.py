"""Earnings BREAKING alerts must page on release, not hours later."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from watchers import alert_agent


def _iso_hours_ago(h: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=h)).isoformat()


class TestEarningsOnReleaseGate:
    def test_fresh_earnings_print_passes(self):
        art = {
            "title": "AMD reports Q2 earnings, beats EPS",
            "summary": "Revenue beat consensus.",
            "published": _iso_hours_ago(0.1),
        }
        assert alert_agent._is_earnings_article(art) is True
        assert alert_agent._article_age_ok(art) is True

    def test_hours_old_earnings_blocked(self):
        # Exact failure mode: "Reported ~1.6h ago" should never page as BREAKING.
        art = {
            "title": "AMD Q2 FY2026 earnings results beat estimates",
            "summary": "Company posted quarterly results after the close.",
            "published": _iso_hours_ago(1.6),
        }
        assert alert_agent._is_earnings_article(art) is True
        assert alert_agent._article_age_ok(art) is False

    def test_non_earnings_still_allowed_within_24h(self):
        art = {
            "title": "Fed officials signal patience on cuts",
            "summary": "Macro colour only.",
            "published": _iso_hours_ago(16.0),
        }
        assert alert_agent._is_earnings_article(art) is False
        assert alert_agent._article_age_ok(art) is True

    def test_category_tag_marks_earnings(self):
        art = {
            "title": "Company posts strong numbers",
            "summary": "",
            "category": "EARNINGS",
            "published": _iso_hours_ago(0.2),
        }
        assert alert_agent._is_earnings_article(art) is True
        assert alert_agent._article_age_ok(art) is True

    def test_stale_earnings_marked_and_not_sent(self, monkeypatch):
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/wh")
        art = {
            "_id": 42,
            "title": "MU Q3 earnings beat Street estimates",
            "summary": "EPS and revenue above consensus.",
            "source": "reuters",
            "link": "https://example.com/mu-earnings",
            "published": _iso_hours_ago(2.0),
            "ai_score": 0.95,
            "urgency": 1,
        }

        class Store:
            def __init__(self):
                self.marked = []

            def mark_alerted_batch(self, ids):
                self.marked.extend(ids)

        store = Store()
        with patch("notifier.discord_notifier.send") as mock_send, patch.object(
            alert_agent, "claude_call"
        ) as mock_claude:
            ok = alert_agent.send_urgent_alert([art], store)

        assert ok is False
        assert mock_send.call_count == 0
        assert mock_claude.call_count == 0
        assert 42 in store.marked


class TestBookOpenVsWatchAnnotation:
    def test_watchlist_only_gets_book_watch_not_holds(self, monkeypatch):
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/wh")
        # AMD is sector_watchlist in portfolio.json, not an open qty position.
        monkeypatch.setattr(
            alert_agent,
            "_open_book_tickers",
            lambda: {"MUU", "MU", "ADBG", "ADBE", "NOK"},
        )
        art = {
            "_id": 7,
            "title": "AMD unveils new AI accelerator roadmap",
            "summary": "No earnings language here.",
            "source": "reuters",
            "link": "https://example.com/amd",
            "published": _iso_hours_ago(0.2),
            "ai_score": 0.9,
            "urgency": 1,
        }

        class Store:
            def mark_alerted_batch(self, ids):
                return None

        captured = {}

        def fake_claude(prompt, **kwargs):
            captured["prompt"] = prompt
            return "🚨 BREAKING ◈ SUPPLY CHAIN ◈ AMD"

        with patch.object(alert_agent, "claude_call", side_effect=fake_claude), patch(
            "notifier.discord_notifier.send", return_value=True
        ), patch.object(
            alert_agent, "_grok_urgency_gate",
            return_value={
                "urgent": True,
                "reason": "test",
                "keep": [art],
                "headline": "AMD",
            },
        ):
            # Some paths require gate; if gate helper missing, ignore.
            try:
                ok = alert_agent.send_urgent_alert([art], Store())
            except Exception:
                # Fall back: call formatter path via send without gate patch if needed
                with patch.object(alert_agent, "claude_call", side_effect=fake_claude), patch(
                    "notifier.discord_notifier.send", return_value=True
                ):
                    ok = alert_agent.send_urgent_alert([art], Store())

        assert "prompt" in captured
        prompt = captured["prompt"]
        assert "book:" in prompt
        assert "book_watch:" in prompt
        assert "AMD" in prompt
        assert "do NOT claim these are held positions" in prompt
        assert "analyst HOLDS/watches these" not in prompt
