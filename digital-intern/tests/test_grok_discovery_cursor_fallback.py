"""Grok multi-agent discovery: Cursor CLI is fallback-only, never primary."""
from __future__ import annotations

import collectors.grok_multiagent_discovery as discovery


def test_xai_missing_token_uses_cursor_fallback(monkeypatch):
    calls = []

    monkeypatch.setattr(discovery, "_load_xai_token", lambda: None)
    monkeypatch.setattr(
        discovery,
        "_cursor_chat",
        lambda prompt, timeout_s=30: calls.append(prompt) or '[{"title":"T","url":"https://ex.com/a"}]',
    )
    out = discovery._xai_chat("find news", timeout_s=5)
    assert out is not None
    assert len(calls) == 1


def test_xai_failure_falls_back_to_cursor(monkeypatch):
    calls = []

    monkeypatch.setattr(discovery, "_load_xai_token", lambda: "tok")
    monkeypatch.setattr(discovery, "ENABLE_WEB_SEARCH", False)
    monkeypatch.setattr(discovery, "ENABLE_X_SEARCH", False)

    def boom(*_a, **_k):
        raise RuntimeError("xAI down")

    monkeypatch.setattr(discovery.urllib.request, "urlopen", boom)
    monkeypatch.setattr(
        discovery,
        "_cursor_chat",
        lambda prompt, timeout_s=30: calls.append("cursor") or "[]",
    )
    out = discovery._xai_chat("find news", model="grok-4.6", timeout_s=5)
    assert out == "[]"
    assert calls == ["cursor"]


def test_cursor_fallback_can_be_disabled(monkeypatch):
    monkeypatch.setattr(discovery, "CURSOR_FALLBACK", False)
    monkeypatch.setattr(discovery, "_load_xai_token", lambda: None)
    assert discovery._xai_chat("find news", timeout_s=5) is None
