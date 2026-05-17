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
        # A custom User-Agent must be sent: Discord 403-filters bare default
        # library UAs, so a missing/empty header would silently drop alerts.
        headers = post.call_args.kwargs.get("headers") or {}
        ua = headers.get("User-Agent", "")
        assert ua and "python-requests" not in ua.lower()


def test_persistent_429_exhausts_retries_and_logs_giveup(monkeypatch, capsys):
    # A 429 storm that never clears must NOT be a silent drop: after
    # _MAX_ATTEMPTS the chunk is abandoned, send() returns False, TTS does
    # not fire, and an explicit "gave up" line is printed so the failure is
    # visible in journalctl during a rate-limit incident.
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/webhook")
    rate_limited = mock.Mock(
        status_code=429,
        text="rate limited",
        headers={"Retry-After": "0"},
    )
    rate_limited.json.return_value = {"retry_after": 0}
    with mock.patch.object(discord_notifier.requests, "post",
                           return_value=rate_limited) as post, \
            mock.patch.object(discord_notifier.time, "sleep") as sleep, \
            mock.patch("notifier.tts.speak_async") as tts:
        assert discord_notifier.send("Micron DRAM alert") is False
        assert post.call_count == discord_notifier._MAX_ATTEMPTS
        tts.assert_not_called()
        sleep.assert_called()  # honored the retry_after each attempt
    out = capsys.readouterr().out
    assert "gave up on chunk 1/1" in out
    assert f"after {discord_notifier._MAX_ATTEMPTS} attempts" in out


def test_empty_message_skips_before_webhook_lookup(monkeypatch):
    # Even with no webhook configured, the empty guard short-circuits first
    # and the return is still a clean False (not an exception).
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    with mock.patch.object(discord_notifier.requests, "post") as post:
        assert discord_notifier.send("") is False
        post.assert_not_called()
