"""llm_adapter routes Grok models through strategy (xAI + Cursor fallback)."""

from paper_trader import llm_adapter


def test_is_grok_model_accepts_aliases():
    assert llm_adapter._is_grok_model("grok-4.6") is True
    assert llm_adapter._is_grok_model("xai/grok-4.6") is True
    assert llm_adapter._is_grok_model("cursor-grok-4.6-xhigh") is True
    assert llm_adapter._is_grok_model("cursor-cli/cursor-grok-4.6-xhigh") is True
    assert llm_adapter._is_grok_model("hf/org/model") is False
    assert llm_adapter._is_grok_model("claude-sonnet-4-6") is False


def test_call_llm_routes_grok_to_strategy(monkeypatch):
    calls = []

    def fake_claude_call(prompt, timeout_s=90, model="grok-4.6"):
        calls.append((prompt, timeout_s, model))
        return "ok"

    monkeypatch.setattr(
        "paper_trader.strategy._claude_call",
        fake_claude_call,
    )
    out = llm_adapter.call_llm("grok-4.6", "hello", timeout=12)
    assert out == "ok"
    assert calls == [("hello", 12, "grok-4.6")]


def test_call_llm_blocks_claude(monkeypatch):
    assert llm_adapter.call_llm("claude-sonnet-4-6", "nope") is None
