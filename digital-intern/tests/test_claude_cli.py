"""Quota circuit-breaker + Grok default routing for core.claude_cli."""
from __future__ import annotations

import json
import types
import urllib.error

import pytest

from core import claude_cli


@pytest.fixture(autouse=True)
def _fresh_breaker(monkeypatch):
    """Start every test with the breaker shut and default Grok model."""
    claude_cli.reset_quota_breaker()
    monkeypatch.setenv("DIGITAL_INTERN_LLM_MODEL", "grok-4.5")
    monkeypatch.setattr(claude_cli, "DEFAULT_LLM_MODEL", "grok-4.5")
    # Keep Cursor fallback off unless a test explicitly enables it.
    monkeypatch.setenv("DIGITAL_INTERN_CURSOR_FALLBACK", "0")
    monkeypatch.setattr(claude_cli, "CURSOR_FALLBACK", False)
    yield
    claude_cli.reset_quota_breaker()


def _fake_run(returncode=0, stdout="", stderr=""):
    calls = {"n": 0}

    def runner(*_a, **_k):
        calls["n"] += 1
        return types.SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr
        )

    runner.calls = calls
    return runner


def test_quota_error_trips_breaker_and_skips_next_http(monkeypatch):
    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            url="https://api.x.ai/v1/chat/completions",
            code=429,
            msg="rate limit",
            hdrs=None,
            fp=types.SimpleNamespace(read=lambda: b"rate limit exceeded"),
        )

    monkeypatch.setattr(claude_cli, "_load_xai_access_token", lambda: "tok")
    monkeypatch.setattr(claude_cli.urllib.request, "urlopen", boom)

    assert claude_cli.claude_call("p", model="grok-4.5") is None
    assert claude_cli.quota_blocked() is True
    assert claude_cli.claude_call("p", model="grok-4.5") is None
    assert claude_cli.claude_call("p", model="grok-4.5") is None
    assert calls["n"] == 1


def test_non_quota_failure_does_not_trip_breaker(monkeypatch):
    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            url="https://api.x.ai/v1/chat/completions",
            code=500,
            msg="server",
            hdrs=None,
            fp=types.SimpleNamespace(read=lambda: b"some transient parse error"),
        )

    monkeypatch.setattr(claude_cli, "_load_xai_access_token", lambda: "tok")
    monkeypatch.setattr(claude_cli.urllib.request, "urlopen", boom)

    assert claude_cli.claude_call("p", model="grok-4.5") is None
    assert claude_cli.quota_blocked() is False
    assert claude_cli.claude_call("p", model="grok-4.5") is None
    assert calls["n"] == 2


def test_rate_limit_string_also_trips(monkeypatch):
    def boom(*_a, **_k):
        raise urllib.error.HTTPError(
            url="https://api.x.ai/v1/chat/completions",
            code=429,
            msg="too many",
            hdrs=None,
            fp=types.SimpleNamespace(read=lambda: b"Error: rate limit exceeded"),
        )

    monkeypatch.setattr(claude_cli, "_load_xai_access_token", lambda: "tok")
    monkeypatch.setattr(claude_cli.urllib.request, "urlopen", boom)

    assert claude_cli.claude_call("p", model="grok-4.5") is None
    assert claude_cli.quota_blocked() is True


def test_breaker_self_heals_after_cooldown(monkeypatch):
    state = {"n": 0}

    def boom(*_a, **_k):
        state["n"] += 1
        if state["n"] == 1:
            raise urllib.error.HTTPError(
                url="https://api.x.ai/v1/chat/completions",
                code=429,
                msg="limit",
                hdrs=None,
                fp=types.SimpleNamespace(read=lambda: b"monthly usage limit reached"),
            )
        class Resp:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": "result body"}}]
                }).encode()
        return Resp()

    monkeypatch.setattr(claude_cli, "_load_xai_access_token", lambda: "tok")
    monkeypatch.setattr(claude_cli.urllib.request, "urlopen", boom)

    assert claude_cli.claude_call("p", model="grok-4.5") is None
    assert claude_cli.quota_blocked() is True

    monkeypatch.setattr(
        claude_cli.time,
        "time",
        lambda: claude_cli._quota_blocked_until + 1,
    )
    assert claude_cli.quota_blocked() is False
    assert claude_cli.claude_call("p", model="grok-4.5") == "result body"
    assert state["n"] == 2


def test_success_passes_through_unchanged(monkeypatch):
    class Resp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "  hello  "}}]
            }).encode()

    monkeypatch.setattr(claude_cli, "_load_xai_access_token", lambda: "tok")
    monkeypatch.setattr(
        claude_cli.urllib.request,
        "urlopen",
        lambda *_a, **_k: Resp(),
    )
    assert claude_cli.claude_call("p", model="grok-4.5") == "hello"
    assert claude_cli.quota_blocked() is False


def test_default_model_is_grok():
    assert claude_cli.DEFAULT_LLM_MODEL.startswith("grok-") or claude_cli.DEFAULT_LLM_MODEL.startswith("xai/")


def test_cursor_fallback_used_when_xai_circuit_open(monkeypatch):
    """Cursor Grok is fallback-only: used when xAI quota breaker is open."""
    monkeypatch.setattr(claude_cli, "CURSOR_FALLBACK", True)
    monkeypatch.setattr(claude_cli, "CURSOR_MODEL", "cursor-grok-4.5-high")
    claude_cli._quota_blocked_until = claude_cli.time.time() + 3600
    monkeypatch.setattr(
        claude_cli,
        "_cursor_http_call",
        lambda prompt, timeout: "cursor-ok",
    )
    monkeypatch.setattr(
        claude_cli,
        "_xai_http_call",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("xAI must not run")),
    )
    assert claude_cli.claude_call("p", model="grok-4.5") == "cursor-ok"


def test_cursor_fallback_used_after_xai_http_failure(monkeypatch):
    """Primary xAI failure still falls through to Cursor when enabled."""
    monkeypatch.setattr(claude_cli, "CURSOR_FALLBACK", True)
    monkeypatch.setattr(claude_cli, "_xai_http_call", lambda *_a, **_k: None)
    monkeypatch.setattr(
        claude_cli,
        "_cursor_http_call",
        lambda prompt, timeout: "cursor-after-miss",
    )
    assert claude_cli.claude_call("p", model="grok-4.5") == "cursor-after-miss"


def test_codex_gpt_models_still_pass_prompt_on_stdin(monkeypatch):
    """Legacy explicit gpt-* override still uses codex stdin path."""
    seen = {}

    def runner(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["input"] = kwargs.get("input")
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess_module := __import__("subprocess"), "run", runner)
    monkeypatch.setattr(claude_cli.subprocess, "run", runner)

    assert claude_cli.claude_call("urgent wire", model="gpt-5.5") == "ok"
    assert seen["cmd"][0].endswith("codex") or seen["cmd"][0] == "/usr/bin/codex"
    assert seen["cmd"][-1] == "-"
    assert seen["input"] == "urgent wire"
