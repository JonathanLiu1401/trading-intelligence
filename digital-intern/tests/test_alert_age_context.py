"""watchers/alert_agent.py — freshness context in the 🚨 BREAKING alert.

A news analyst reacting to a Discord BREAKING push must be able to tell a
4-minute-old NVDA 8-K (act NOW) from a 16-hour-old reused headline (already
priced in). The store SQL only guarantees the row is < 24h by ``first_seen``;
``_article_age_ok`` drops > 24h, but anything inside the wide 0..24h band
fired identically with zero recency signal in the prompt. This pins the new
``_article_age_hours`` / ``_article_age_str`` helpers and that the ``age``
line + the RECENCY rule actually reach the Sonnet prompt — without regressing
any of the formatter's load-bearing guards (read-only on the alert path: no
ai_score / ml_score / score_source / backtest-isolation surface touched).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import patch

from watchers import alert_agent


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


class _StoreSpy:
    def __init__(self):
        self.marked = []

    def mark_alerted_batch(self, ids):
        self.marked.extend(ids)

    def mark_alerted(self, aid):
        self.marked.append(aid)


# ── _article_age_hours (pure) ────────────────────────────────────────────────

class TestArticleAgeHours:
    def test_iso_published_two_hours_ago(self):
        h = alert_agent._article_age_hours({"published": _iso(2.0)})
        assert h is not None and abs(h - 2.0) < 0.05

    def test_rfc822_published_parsed(self):
        dt = datetime.now(timezone.utc) - timedelta(hours=3)
        h = alert_agent._article_age_hours({"published": format_datetime(dt)})
        assert h is not None and abs(h - 3.0) < 0.05

    def test_published_preferred_over_first_seen(self):
        # First parseable field wins — same convention as _article_age_ok.
        h = alert_agent._article_age_hours(
            {"published": _iso(5.0), "first_seen": _iso(0.1)}
        )
        assert abs(h - 5.0) < 0.05

    def test_falls_back_to_first_seen_when_published_empty(self):
        h = alert_agent._article_age_hours(
            {"published": "", "first_seen": _iso(7.0)}
        )
        assert abs(h - 7.0) < 0.05

    def test_falls_back_to_first_seen_when_published_garbage(self):
        h = alert_agent._article_age_hours(
            {"published": "not-a-date", "first_seen": _iso(4.0)}
        )
        assert abs(h - 4.0) < 0.05

    def test_none_when_both_unparseable_or_absent(self):
        assert alert_agent._article_age_hours(
            {"published": "nope", "first_seen": "also-nope"}
        ) is None
        assert alert_agent._article_age_hours({}) is None

    def test_naive_timestamp_assumed_utc(self):
        naive = (datetime.now(timezone.utc) - timedelta(hours=2)
                 ).replace(tzinfo=None).isoformat()
        h = alert_agent._article_age_hours({"published": naive})
        assert h is not None and abs(h - 2.0) < 0.05

    def test_future_timestamp_clamped_to_zero(self):
        h = alert_agent._article_age_hours({"published": _iso(-3.0)})
        assert h == 0.0


# ── _article_age_str (pure, exact formatting) ────────────────────────────────

class TestArticleAgeStr:
    def test_minutes_under_one_hour(self):
        assert alert_agent._article_age_str(
            {"published": _iso(4.0 / 60.0)}) == "4m"

    def test_sub_minute_is_lt_1m(self):
        assert alert_agent._article_age_str(
            {"published": _iso(20.0 / 3600.0)}) == "<1m"

    def test_one_decimal_below_ten_hours(self):
        assert alert_agent._article_age_str(
            {"published": _iso(3.2)}) == "3.2h"
        assert alert_agent._article_age_str(
            {"published": _iso(1.0)}) == "1.0h"

    def test_integer_hours_at_or_above_ten(self):
        assert alert_agent._article_age_str(
            {"published": _iso(16.0)}) == "16h"
        assert alert_agent._article_age_str(
            {"published": _iso(10.0)}) == "10h"

    def test_none_when_age_unknown(self):
        assert alert_agent._article_age_str(
            {"published": "garbage", "first_seen": ""}) is None


# ── end-to-end: age line + RECENCY rule reach the Sonnet prompt ──────────────

class TestAgeReachesSonnetPrompt:
    def _send(self, art, monkeypatch):
        spy = _StoreSpy()
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/wh")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU"
                          ) as mock_claude, \
             patch("notifier.discord_notifier.send",
                   return_value=True) as mock_send:
            ok = alert_agent.send_urgent_alert([art], spy)
        return ok, spy, mock_claude, mock_send

    def test_fresh_row_emits_minute_age_line(self, monkeypatch):
        art = {
            "_id": "fresh", "link": "https://reuters.com/x",
            "title": "MU guides Q4 revenue sharply above the Street",
            "source": "rss", "ai_score": 9.0, "summary": "",
            "published": _iso(4.0 / 60.0), "first_seen": _iso(0.05),
        }
        ok, spy, mock_claude, mock_send = self._send(art, monkeypatch)
        assert ok is True
        mock_claude.assert_called_once()
        prompt = mock_claude.call_args.args[0]
        assert "age: 4m (time since publication)" in prompt
        assert "RECENCY:" in prompt
        assert "alert send time, not the event time" in prompt
        # Read-only contract preserved: exactly the live row marked alerted.
        assert spy.marked == ["fresh"]

    def test_old_but_not_stale_row_emits_hour_age_line(self, monkeypatch):
        # 16h old: _article_age_ok keeps it (< 24h) so it reaches _fmt; the
        # analyst now sees it is NOT just-broke.
        art = {
            "_id": "old", "link": "https://reuters.com/y",
            "title": "Fed minutes reveal split on the next rate decision",
            "source": "rss", "ai_score": 8.0, "summary": "",
            "published": _iso(16.0), "first_seen": _iso(0.1),
        }
        ok, spy, mock_claude, _ = self._send(art, monkeypatch)
        assert ok is True
        prompt = mock_claude.call_args.args[0]
        assert "age: 16h (time since publication)" in prompt
        assert spy.marked == ["old"]

    def test_age_uses_first_seen_when_published_garbage(self, monkeypatch):
        # published unparseable but first_seen fresh → _article_age_ok keeps
        # it (first_seen branch) and the age line falls back to first_seen.
        art = {
            "_id": "fb", "link": "https://reuters.com/z",
            "title": "Samsung HBM4 supply update lands on the wire",
            "source": "rss", "ai_score": 9.0, "summary": "",
            "published": "not-a-date", "first_seen": _iso(0.5),
        }
        ok, spy, mock_claude, _ = self._send(art, monkeypatch)
        assert ok is True
        prompt = mock_claude.call_args.args[0]
        assert "age: 30m (time since publication)" in prompt

    def test_unknown_age_omits_line_silently_not_a_fake_zero(self):
        # Pure-helper level: an undated dict yields no line (None), never "0m".
        assert alert_agent._article_age_str(
            {"published": "", "first_seen": ""}) is None
