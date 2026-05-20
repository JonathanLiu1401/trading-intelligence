import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import patch, MagicMock


def test_call_llm_raises_on_unknown_model():
    from paper_trader.llm_adapter import call_llm
    with pytest.raises(ValueError, match="Unknown model_id"):
        call_llm("unknown/model", "prompt")


def test_call_llm_routes_claude_to_subprocess():
    from paper_trader import llm_adapter
    with patch("paper_trader.llm_adapter._claude_call") as mock_claude:
        mock_claude.return_value = '{"action":"HOLD"}'
        result = llm_adapter.call_llm("claude-opus-4-7", "test prompt")
        mock_claude.assert_called_once_with("claude-opus-4-7", "test prompt")
        assert result == '{"action":"HOLD"}'


def test_call_llm_routes_hf_prefix():
    from paper_trader import llm_adapter
    with patch("paper_trader.llm_adapter._hf_call") as mock_hf:
        mock_hf.return_value = '{"action":"BUY"}'
        result = llm_adapter.call_llm("hf/deepseek-ai/DeepSeek-R1", "test prompt")
        # strips "hf/" prefix before passing to _hf_call
        mock_hf.assert_called_once_with("deepseek-ai/DeepSeek-R1", "test prompt", 90)
        assert result == '{"action":"BUY"}'


def test_hf_call_returns_none_when_no_token(monkeypatch):
    from paper_trader import llm_adapter
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Patch _load_hf_token to return None (no .env file in test env)
    with patch("paper_trader.llm_adapter._load_hf_token", return_value=None):
        result = llm_adapter._hf_call("deepseek-ai/DeepSeek-R1", "prompt", 10)
    assert result is None


def test_hf_call_sends_correct_payload(monkeypatch):
    import json
    from paper_trader import llm_adapter
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "test-token")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"action":"HOLD"}'}}]
    }

    with patch("paper_trader.llm_adapter.requests.post", return_value=mock_resp) as mock_post:
        result = llm_adapter._hf_call("Qwen/Qwen3-32B", "buy or sell?", 30)

    assert result == '{"action":"HOLD"}'
    call_kwargs = mock_post.call_args
    payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
    assert payload["model"] == "Qwen/Qwen3-32B"
    assert payload["messages"][0]["content"] == "buy or sell?"


def test_hf_call_retries_on_500(monkeypatch):
    from paper_trader import llm_adapter
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "test-token")

    fail_resp = MagicMock()
    fail_resp.status_code = 500

    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

    with patch("paper_trader.llm_adapter.requests.post", side_effect=[fail_resp, ok_resp]), \
         patch("paper_trader.llm_adapter.time.sleep"):  # don't actually sleep in tests
        result = llm_adapter._hf_call("Qwen/Qwen3-8B", "prompt", 30)

    assert result == "ok"
