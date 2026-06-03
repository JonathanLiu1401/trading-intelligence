"""Tests for /api/chat LLM backend fallback helpers."""

from dashboard import web_server


def test_chat_model_candidates_include_both_backends_by_default(monkeypatch):
    monkeypatch.delenv("DIGITAL_INTERN_CHAT_MODELS", raising=False)
    monkeypatch.delenv("DIGITAL_INTERN_LLM_MODEL", raising=False)

    assert web_server._chat_model_candidates() == [
        "gpt-5.5",
        "claude-sonnet-4-6",
    ]


def test_chat_model_candidates_preserve_override_then_add_fallbacks(monkeypatch):
    monkeypatch.setenv(
        "DIGITAL_INTERN_CHAT_MODELS",
        "claude-opus-4-7, gpt-5.5, claude-opus-4-7",
    )

    assert web_server._chat_model_candidates() == [
        "claude-opus-4-7",
        "gpt-5.5",
        "claude-sonnet-4-6",
    ]


def test_call_chat_llm_falls_through_to_second_backend(monkeypatch):
    calls = []

    def fake_call(prompt, model, timeout):
        calls.append((prompt, model, timeout))
        if model == "gpt-5.5":
            return None
        return "answer"

    monkeypatch.delenv("DIGITAL_INTERN_CHAT_MODELS", raising=False)
    monkeypatch.delenv("DIGITAL_INTERN_LLM_MODEL", raising=False)
    monkeypatch.setattr(web_server, "_claude_cli_call", fake_call)

    text, model, failures = web_server._call_chat_llm("prompt", timeout=7)

    assert text == "answer"
    assert model == "claude-sonnet-4-6"
    assert failures == ["gpt-5.5"]
    assert calls == [
        ("prompt", "gpt-5.5", 7),
        ("prompt", "claude-sonnet-4-6", 7),
    ]


def test_unavailable_response_is_user_visible_not_error_json():
    response = web_server._chat_backend_unavailable_response(
        "what is next?",
        [{"title": "NVDA launches new accelerator", "source": "Wire", "ai_score": 8.4}],
        "Equity $1000\nCash $500",
        ["gpt-5.5", "claude-sonnet-4-6"],
    )

    assert "LLM backends are unavailable" in response
    assert "NVDA launches new accelerator" in response
    assert "Paper trader snapshot" in response
    assert "tries both Codex and Claude" in response


def test_chat_deep_context_detection_only_for_bot_questions():
    assert not web_server._chat_needs_deep_context("what is moving markets today?")
    assert not web_server._chat_needs_deep_context("next explosive stocks")

    assert web_server._chat_needs_deep_context("why did the bot buy NVDA?")
    assert web_server._chat_needs_deep_context("paper trader deep diagnosis")
