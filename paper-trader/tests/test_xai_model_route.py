from paper_trader import strategy


def test_uses_xai_http_for_grok_models():
    assert strategy._uses_xai_http("grok-4.6") is True
    assert strategy._uses_xai_http("xai/grok-4.6") is True
    assert strategy._uses_xai_http("gpt-5.5") is False
    assert strategy._uses_xai_http("claude-sonnet-4-6") is False


def test_normalize_model_name_strips_provider_prefix():
    assert strategy._normalize_model_name("xai/grok-4.6") == "grok-4.6"
    assert strategy._normalize_model_name("grok-4.6") == "grok-4.6"


def test_default_model_is_grok():
    assert strategy.MODEL == "grok-4.6"
    assert strategy.FALLBACK_MODEL == "grok-4.6"
