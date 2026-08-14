"""BREAKING channel posts may still land; @-page + Sao DM must not.

Failure class (2026-08-04 Segro/PLD): off-book M&A with IMPACT: WATCH was
packaged as 🚨 BREAKING and pinged Jonathan + Sao. News-desk "breaking" is not
the same as "page the desk."

Contract:
  - page when any approved article hits an OPEN held name (qty>0 / option),
    OR IMPACT is BUY/SELL
  - quiet channel post (no mention prefix, no Sao DM, is_alert=False) for
    watchlist-only WATCH colour (PLD on XLRE watch is not a page)
  - channel delivery still happens either way
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from watchers import alert_agent


@pytest.fixture(autouse=True)
def _no_real_sao_dm(monkeypatch):
    monkeypatch.setattr(alert_agent, "_send_sao_breaking_dm", lambda message: True)


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _watch_body(tickers: str = "SGRO.L / PLD") -> str:
    return (
        "🚨 BREAKING  ◈  M&A  ◈  2026-08-04 07:37 UTC\n"
        "SEGRO BOARD RECOMMENDS £14BN TAKEOVER BID FROM PROLOGIS\n"
        f"TICKERS:   {tickers}\n"
        "IMPACT:    WATCH — PLD final approach for London-listed logistics rival\n"
        "SOURCE:    Reuters\n"
        "<https://example.com/segro>\n"
    )


def _buy_body(tickers: str = "MU") -> str:
    return (
        "🚨 BREAKING  ◈  EARNINGS  ◈  2026-08-04 07:37 UTC\n"
        "MU GUIDES Q4 SHARPLY ABOVE STREET\n"
        f"TICKERS:   {tickers}\n"
        "IMPACT:    BUY — beat-and-raise with HBM upside\n"
        "SOURCE:    Reuters\n"
        "<https://example.com/mu>\n"
    )


class TestAlertImpactSide:
    def test_parses_watch_buy_sell(self):
        assert alert_agent._alert_impact_side(_watch_body()) == "WATCH"
        assert alert_agent._alert_impact_side(_buy_body()) == "BUY"
        assert alert_agent._alert_impact_side(
            "IMPACT: SELL — thesis broken"
        ) == "SELL"

    def test_missing_impact_empty(self):
        assert alert_agent._alert_impact_side("no impact line here") == ""


class TestShouldPageBreaking:
    def test_off_book_watch_is_quiet(self, monkeypatch):
        monkeypatch.setattr(alert_agent, "BREAKING_PING_ENABLED", True)
        batch = [{
            "title": "Segro board recommends Prologis takeover",
            "summary": "SGRO.L / PLD logistics bid",
        }]
        assert alert_agent._should_page_breaking(_watch_body(), batch) is False

    def test_open_book_hit_pages_even_on_watch(self, monkeypatch):
        monkeypatch.setattr(alert_agent, "BREAKING_PING_ENABLED", True)
        # MU is open via MUU levered wrapper / NOK is open option underlying.
        batch = [{
            "title": "MU supply note hits the wire",
            "summary": "",
        }]
        assert alert_agent._should_page_breaking(_watch_body("MU"), batch) is True

    def test_watchlist_only_pld_watch_is_quiet(self, monkeypatch):
        monkeypatch.setattr(alert_agent, "BREAKING_PING_ENABLED", True)
        batch = [{
            "title": "PLD approaches Segro in final bid talks",
            "summary": "Prologis REIT M&A colour",
        }]
        assert alert_agent._should_page_breaking(_watch_body("PLD"), batch) is False

    def test_off_book_buy_still_pages(self, monkeypatch):
        monkeypatch.setattr(alert_agent, "BREAKING_PING_ENABLED", True)
        batch = [{
            "title": "Random industrial name beats estimates hard",
            "summary": "",
        }]
        assert alert_agent._should_page_breaking(
            _buy_body("XYZ"), batch
        ) is True

    def test_ping_disabled_never_pages(self, monkeypatch):
        monkeypatch.setattr(alert_agent, "BREAKING_PING_ENABLED", False)
        batch = [{"title": "MU guides above", "summary": ""}]
        assert alert_agent._should_page_breaking(_buy_body(), batch) is False


class TestWithBreakingPingGate:
    def test_quiet_watch_strips_prefix(self, monkeypatch):
        monkeypatch.setattr(alert_agent, "BREAKING_PING_ENABLED", True)
        monkeypatch.setattr(
            alert_agent, "JONATHAN_DISCORD_USER_ID", "111"
        )
        monkeypatch.setattr(alert_agent, "SAO_DISCORD_USER_ID", "222")
        body = _watch_body()
        out = alert_agent._with_breaking_ping(
            body,
            batch=[{"title": "Segro / Prologis bid", "summary": ""}],
        )
        assert out == body.strip()
        assert "<@111>" not in out
        assert "<@222>" not in out

    def test_book_buy_keeps_prefix(self, monkeypatch):
        monkeypatch.setattr(alert_agent, "BREAKING_PING_ENABLED", True)
        monkeypatch.setattr(
            alert_agent, "JONATHAN_DISCORD_USER_ID", "111"
        )
        monkeypatch.setattr(alert_agent, "SAO_DISCORD_USER_ID", "222")
        body = _buy_body()
        out = alert_agent._with_breaking_ping(
            body,
            batch=[{"title": "MU guides Q4 above", "summary": ""}],
        )
        assert out.startswith("<@111>")
        assert "<@222>" in out
        assert "MU GUIDES" in out


class TestSendUrgentAlertPageGate:
    def _art(self, *, aid: str, title: str) -> dict:
        return {
            "_id": aid,
            "link": f"https://reuters.com/{aid}",
            "title": title,
            "source": "rss",
            "ai_score": 9.0,
            "summary": "",
            "published": _iso(0.1),
            "first_seen": _iso(0.05),
        }

    def test_off_book_watch_posts_quietly_no_sao_dm(self, store, monkeypatch):
        """Segro/PLD class: channel gets the body, no ping / DM / alert TTS."""
        art = self._art(
            aid="segro",
            title="Segro board recommends £14bn Prologis takeover bid",
        )
        _insert = store  # keep store fixture live for mark path
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/wh")
        monkeypatch.setattr(alert_agent, "BREAKING_PING_ENABLED", True)
        monkeypatch.setattr(
            alert_agent, "JONATHAN_DISCORD_USER_ID", "454961974048980992"
        )
        monkeypatch.setattr(
            alert_agent, "SAO_DISCORD_USER_ID", "702863115276124211"
        )
        gate = {
            "urgent": True,
            "reason": "M&A colour",
            "keep": [art],
            "headline": "SEGRO / PROLOGIS BID",
        }
        sao = []
        monkeypatch.setattr(
            alert_agent, "_send_sao_breaking_dm",
            lambda message: sao.append(message) or True,
        )
        with patch.object(
            alert_agent, "_grok_urgency_gate", return_value=gate
        ), patch.object(
            alert_agent, "claude_call", return_value=_watch_body()
        ), patch(
            "notifier.discord_notifier.send", return_value=True
        ) as mock_send:
            ok = alert_agent.send_urgent_alert([art], store)

        assert ok is True
        mock_send.assert_called_once()
        sent_msg, kwargs = mock_send.call_args.args[0], mock_send.call_args.kwargs
        # kwargs may be positional is_alert
        if not kwargs and len(mock_send.call_args.args) > 1:
            is_alert = mock_send.call_args.args[1]
        else:
            is_alert = kwargs.get("is_alert", None)
        assert "<@454961974048980992>" not in sent_msg
        assert "<@702863115276124211>" not in sent_msg
        assert "IMPACT:" in sent_msg and "WATCH" in sent_msg
        assert is_alert is False
        assert sao == [], "quiet WATCH must not fan out to Sao DM"
        del _insert

    def test_book_hit_still_pages_and_dms(self, store, monkeypatch):
        art = self._art(
            aid="mu1",
            title="MU guides Q4 revenue sharply above the Street",
        )
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/wh")
        monkeypatch.setattr(alert_agent, "BREAKING_PING_ENABLED", True)
        monkeypatch.setattr(
            alert_agent, "JONATHAN_DISCORD_USER_ID", "454961974048980992"
        )
        monkeypatch.setattr(
            alert_agent, "SAO_DISCORD_USER_ID", "702863115276124211"
        )
        gate = {
            "urgent": True,
            "reason": "held earnings",
            "keep": [art],
            "headline": "MU BEAT",
        }
        sao = []
        monkeypatch.setattr(
            alert_agent, "_send_sao_breaking_dm",
            lambda message: sao.append(message) or True,
        )
        with patch.object(
            alert_agent, "_grok_urgency_gate", return_value=gate
        ), patch.object(
            alert_agent, "claude_call", return_value=_buy_body()
        ), patch(
            "notifier.discord_notifier.send", return_value=True
        ) as mock_send:
            ok = alert_agent.send_urgent_alert([art], store)

        assert ok is True
        sent_msg = mock_send.call_args.args[0]
        is_alert = (
            mock_send.call_args.kwargs.get("is_alert")
            if mock_send.call_args.kwargs
            else mock_send.call_args.args[1]
        )
        assert sent_msg.startswith("<@454961974048980992>")
        assert "<@702863115276124211>" in sent_msg
        assert is_alert is True
        assert len(sao) == 1
