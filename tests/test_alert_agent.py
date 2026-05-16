"""Live alert formatter guards — ``watchers.alert_agent.send_urgent_alert``.

This is the most safety-critical hop in the system: it is the single function
that turns a DB row into a Bloomberg-style Discord alert. Two of its guards
were only ever exercised *implicitly* (via the end-to-end integration test or
the store-level isolation tests), never asserted at the agent boundary:

  1. Staleness — an article whose ``published`` date is > 24h old must NOT
     fire as breaking news, even though ``get_unalerted_urgent`` returned it
     (the store SQL filters on ``first_seen``, not ``published``; the agent's
     ``_article_age_ok`` is the only thing standing between a 3-day-old
     re-syndicated headline and a "🚨 BREAKING" Discord post).
  2. The webhook / no-parseable-date early-outs must short-circuit BEFORE the
     Sonnet call — otherwise every cycle burns a Claude call and risks posting
     to a ``None`` webhook.

These tests pin the agent's own contract: when it drops a batch it must touch
neither Claude nor Discord nor the store, and when it sends it must mark
exactly the alerted ids so the article cannot re-fire. A regression that
weakens the staleness guard (or removes the webhook check) passes every other
suite but resurfaces stale alerts in production — exactly the class of bug this
file exists to catch.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from watchers import alert_agent


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _insert_urgent(store, *, id, url="https://reuters.com/x",
                    title="MU earnings blow past Q3 estimates sharply",
                    source="rss", ai_score=9.0, published="", first_seen=None):
    """Insert a single live, urgency=1 row exactly as the scoring path would
    leave it for the alerter to pick up."""
    if first_seen is None:
        first_seen = _iso(0.08)  # ~5 min ago — inside the 24h first_seen window
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            " first_seen, cycle, ml_score, score_source, full_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, published, 1.0, ai_score, 1,
             first_seen, 0, None, "llm", None),
        )
        store.conn.commit()


def _urgency_of(store, aid):
    return store.conn.execute(
        "SELECT urgency FROM articles WHERE id=?", (aid,)
    ).fetchone()[0]


class TestStalenessGuard:
    def test_stale_published_article_is_not_alerted(self, store, monkeypatch):
        """A live urgent row whose ``published`` is 72h old is returned by
        ``get_unalerted_urgent`` (its ``first_seen`` is recent) but the agent
        MUST drop it: no Claude call, no Discord post, urgency stays 1."""
        _insert_urgent(store, id="stale", published=_iso(72), first_seen=_iso(0.1))
        urgent = store.get_unalerted_urgent()
        assert len(urgent) == 1, "precondition: store returns the recent-first_seen row"

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert _urgency_of(store, "stale") == 1, "stale row was wrongly marked alerted"

    def test_unparseable_dates_block_the_alert(self, monkeypatch):
        """Neither field parses → the agent blocks rather than risk a stale
        alert (documented 'no parseable date — dropping to be safe' path).
        Driven directly so the branch is isolated from store SQL quirks."""
        art = {
            "_id": "junk", "link": "https://reuters.com/junk",
            "title": "Totally real urgent headline about MU here",
            "source": "rss", "ai_score": 9.0, "summary": "",
            "published": "not-a-date", "first_seen": "also-not-a-date",
        }

        class _StoreSpy:
            marked = []

            def mark_alerted_batch(self, ids):
                self.marked.extend(ids)

        spy = _StoreSpy()
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert([art], spy)

        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert spy.marked == []


class TestWebhookEarlyOut:
    def test_missing_webhook_short_circuits_before_claude(self, store, monkeypatch):
        """No DISCORD_WEBHOOK configured → return False immediately, WITHOUT
        spending a Sonnet call. A regression here silently burns Claude quota
        every alert cycle and POSTs to an empty URL."""
        _insert_urgent(store, id="fresh", published=_iso(1))
        urgent = store.get_unalerted_urgent()
        assert len(urgent) == 1

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert _urgency_of(store, "fresh") == 1


class TestHappyPathMarksAlerted:
    def test_fresh_live_article_alerts_and_marks_alerted_exactly_once(
        self, store, monkeypatch
    ):
        """A fresh (published 1h ago) live urgent row is sent: Sonnet composes
        the alert, Discord accepts it, and the agent marks PRECISELY that id
        urgency=2 so it can never re-fire on the next 20s alert cycle."""
        _insert_urgent(store, id="go", published=_iso(1), ai_score=9.0)
        urgent = store.get_unalerted_urgent()
        assert len(urgent) == 1

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU") as mock_claude, \
             patch("notifier.discord_notifier.send",
                   return_value=True) as mock_send:
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is True
        mock_claude.assert_called_once()
        mock_send.assert_called_once()
        # Marked alerted (urgency 1 → 2) and now invisible to the alerter.
        assert _urgency_of(store, "go") == 2
        assert store.get_unalerted_urgent() == []

    def test_discord_failure_leaves_article_requeued(self, store, monkeypatch):
        """If Discord delivery fails, the article must stay urgency=1 so the
        next cycle retries it — marking it alerted on a failed POST would
        silently lose the alert forever."""
        _insert_urgent(store, id="retry", published=_iso(1))
        urgent = store.get_unalerted_urgent()
        assert len(urgent) == 1

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call", return_value="alert body"), \
             patch("notifier.discord_notifier.send", return_value=False):
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is False
        assert _urgency_of(store, "retry") == 1, "failed POST must not mark alerted"
