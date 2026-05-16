"""Discord notifier guards: empty bodies must not POST or fire TTS.

Discord rejects an empty `content` with HTTP 400, and the pre-guard code
returned True for `send("")` — a silent false success that also wasted a
TTS call. These tests pin the no-op behaviour and confirm a real message
still posts exactly once.
"""
from __future__ import annotations

from unittest import mock

import pytest

from notifier import discord_notifier


@pytest.mark.parametrize("body", ["", "   ", "\n", "  \n\t  "])
def test_empty_or_whitespace_message_is_noop(body, monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/webhook")
    with mock.patch.object(discord_notifier.requests, "post") as post, \
            mock.patch("notifier.tts.speak_async") as tts:
        assert discord_notifier.send(body) is False
        post.assert_not_called()
        tts.assert_not_called()


def test_real_message_posts_once_and_fires_tts(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/webhook")
    resp = mock.Mock(status_code=204, text="")
    with mock.patch.object(discord_notifier.requests, "post", return_value=resp) as post, \
            mock.patch("notifier.tts.speak_async") as tts:
        assert discord_notifier.send("Micron raises DRAM guidance") is True
        post.assert_called_once()
        tts.assert_called_once()


def test_empty_message_skips_before_webhook_lookup(monkeypatch):
    # Even with no webhook configured, the empty guard short-circuits first
    # and the return is still a clean False (not an exception).
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    with mock.patch.object(discord_notifier.requests, "post") as post:
        assert discord_notifier.send("") is False
        post.assert_not_called()
