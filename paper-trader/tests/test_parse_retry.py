"""Tests for the parse-failure retry + raw-capture path in strategy.decide().

These tests don't touch yfinance, claude, or the live paper_trader.db. They
monkeypatch every external call to keep the suite hermetic — see
strategy._claude_call, strategy._portfolio_snapshot, signals.get_top_signals,
and friends. Each test asserts a specific behavior (retry fires / doesn't,
raw is captured to reasoning, etc.) instead of just "no crash"."""
from __future__ import annotations

from unittest import mock

import pytest

import paper_trader.strategy as strategy


@pytest.fixture
def stub_decide_inputs(monkeypatch):
    """Replace every external call in decide() with a no-op stub.

    Yields the mocked store so tests can inspect what was recorded."""
    # _build_payload reads several keys from the snapshot — give it the
    # shape _portfolio_snapshot would produce so the test exercises the
    # real prompt-construction path.
    snap = {
        "total_value": 1000.0,
        "cash": 1000.0,
        "open_value": 0.0,
        "positions": [],
    }

    def fake_snapshot(store):
        return snap

    fake_store = mock.MagicMock()
    fake_store.record_decision = mock.MagicMock()
    fake_store.record_equity_point = mock.MagicMock()

    monkeypatch.setattr(strategy, "_portfolio_snapshot", fake_snapshot)
    monkeypatch.setattr(strategy, "get_store", lambda: fake_store)
    monkeypatch.setattr(strategy.signals, "get_top_signals", lambda *a, **k: [])
    monkeypatch.setattr(strategy.signals, "get_urgent_articles", lambda *a, **k: [])
    monkeypatch.setattr(strategy.signals, "ticker_sentiments", lambda *a, **k: [])
    monkeypatch.setattr(strategy.market, "is_market_open", lambda: True)
    monkeypatch.setattr(strategy.market, "get_prices", lambda *a, **k: {})
    monkeypatch.setattr(strategy.market, "get_futures_price", lambda *a, **k: None)
    monkeypatch.setattr(strategy.market, "benchmark_sp500", lambda: 5000.0)
    monkeypatch.setattr(strategy, "get_quant_signals_live", lambda *a, **k: {})
    return fake_store


def test_should_retry_skips_none_response():
    # Timeout / empty stdout: retrying buys nothing. Same prompt → same failure.
    assert strategy._should_retry_parse(None) is False
    assert strategy._should_retry_parse("") is False


def test_should_retry_fires_on_non_empty_unparseable():
    # Prose-wrapped reply with no JSON object at all → retry should fire.
    assert strategy._should_retry_parse("Here is my answer: I think we should hold.") is True


def test_should_retry_skips_already_parseable():
    # If parse succeeded the caller shouldn't be asking us — but if it does,
    # we must say "no, don't retry" so we don't double-call on a good response.
    assert strategy._should_retry_parse('{"action": "HOLD"}') is False


def test_retry_fires_when_first_response_is_unparseable(stub_decide_inputs):
    """Non-empty prose first → retry with stronger nudge → parseable second."""
    calls: list[str] = []

    def fake_claude(prompt, timeout_s=strategy.DECISION_TIMEOUT_S):
        calls.append(prompt)
        if len(calls) == 1:
            return "Sure, here's what I think we should do today: hold everything."
        return '{"action": "HOLD", "ticker": "NVDA", "confidence": 0.5, "reasoning": "stub"}'

    with mock.patch.object(strategy, "_claude_call", side_effect=fake_claude):
        result = strategy.decide()

    assert len(calls) == 2, "retry must call claude twice"
    assert strategy._RETRY_SUFFIX in calls[1], "retry must include JSON-only nudge"
    assert result["retried"] is True
    assert result["decision"] is not None
    assert result["decision"]["action"] == "HOLD"


def test_no_retry_when_first_response_is_none(stub_decide_inputs):
    """Timeout / CLI failure: no JSON-nudge *retry* of the SAME prompt.

    The invariant under test is narrow: a None first response must NOT trigger
    the ``_RETRY_SUFFIX`` JSON-only retry (same prompt → same wall). The Sonnet
    fallback added later is a DISTINCT mechanism — a different model with a
    *condensed* prompt — so a second `_claude_call` is now expected and correct;
    what must never appear is a third call carrying ``_RETRY_SUFFIX``. When the
    fallback also returns None, decide() records NO_DECISION with no retry."""
    calls = []

    # Must accept the fallback's model= / timeout_s= kwargs — a fake that only
    # took (prompt) silently regressed when the Sonnet fallback was added.
    def fake_claude(prompt, **kwargs):
        calls.append(prompt)
        return None

    with mock.patch.object(strategy, "_claude_call", side_effect=fake_claude):
        result = strategy.decide()

    # Opus attempt + Sonnet fallback = exactly 2 calls. NOT 3 (no JSON retry).
    assert len(calls) == 2, "expect Opus attempt + Sonnet fallback, no JSON retry"
    assert all(strategy._RETRY_SUFFIX not in p for p in calls), (
        "a None first response must NOT trigger the same-prompt JSON-nudge retry"
    )
    assert result["retried"] is False
    assert result["fallback_used"] is False    # fallback ran but returned None
    assert result["status"] == "NO_DECISION"
    # The no-response branch must record a recognizable reason string.
    fake_store = stub_decide_inputs
    args, _ = fake_store.record_decision.call_args
    reason = args[3]
    assert "timeout/empty" in reason or "no response" in reason


def test_no_retry_when_first_response_parses_cleanly(stub_decide_inputs):
    """Happy path — single Claude call is enough."""
    calls = []

    def fake_claude(prompt, timeout_s=strategy.DECISION_TIMEOUT_S):
        calls.append(prompt)
        return '{"action": "HOLD", "ticker": "NVDA", "confidence": 0.6, "reasoning": "x"}'

    with mock.patch.object(strategy, "_claude_call", side_effect=fake_claude):
        result = strategy.decide()

    assert len(calls) == 1
    assert result["retried"] is False
    assert result["decision"]["action"] == "HOLD"


def test_failed_parse_captures_raw_excerpt_in_reasoning(stub_decide_inputs):
    """The whole point of this feature: when parse fails we must persist what
    Claude actually said, not a generic 'no parseable JSON' line."""
    bad_response = "I refuse to answer this hypothetical question about trading."

    with mock.patch.object(strategy, "_claude_call", return_value=bad_response):
        strategy.decide()

    fake_store = stub_decide_inputs
    args, _ = fake_store.record_decision.call_args
    reason = args[3]
    # Tagged so operators can grep DB for parse vs retry failures
    assert reason.startswith("retry_failed:") or reason.startswith("parse_failed:")
    assert "I refuse to answer" in reason


def test_raw_excerpt_is_truncated_to_cap(stub_decide_inputs):
    """Don't fill the DB with 50KB blobs of model rambling."""
    huge = "x" * (strategy.RAW_CAPTURE_CHARS * 5)

    with mock.patch.object(strategy, "_claude_call", return_value=huge):
        strategy.decide()

    fake_store = stub_decide_inputs
    args, _ = fake_store.record_decision.call_args
    reason = args[3]
    # tag prefix + ': ' + up to RAW_CAPTURE_CHARS chars of payload
    overhead = len("retry_failed: ")
    assert len(reason) <= strategy.RAW_CAPTURE_CHARS + overhead + 5


def test_retry_uses_shorter_timeout(stub_decide_inputs):
    """Retry must use RETRY_TIMEOUT_S so a parse-failure rescue can't blow
    past the 60s open-market cycle cadence."""
    timeouts: list[int] = []

    def fake_claude(prompt, timeout_s=strategy.DECISION_TIMEOUT_S):
        timeouts.append(timeout_s)
        return "prose with no json"

    with mock.patch.object(strategy, "_claude_call", side_effect=fake_claude):
        strategy.decide()

    assert timeouts == [strategy.DECISION_TIMEOUT_S, strategy.RETRY_TIMEOUT_S]


def test_decision_pipeline_still_works_end_to_end(stub_decide_inputs, monkeypatch):
    """Sanity: a clean HOLD passes through to record_decision with the right shape."""
    monkeypatch.setattr(
        strategy, "_execute", lambda decision, snap, store: ("HOLD", "no action")
    )
    fake_store = stub_decide_inputs
    # _portfolio_snapshot is monkeypatched once at fixture scope; for the
    # post-execute "final mark" the strategy calls it again — same stub keeps
    # working since it just returns the same snap dict.

    with mock.patch.object(
        strategy,
        "_claude_call",
        return_value='{"action": "HOLD", "ticker": "NVDA", "confidence": 0.7, "reasoning": "x"}',
    ):
        result = strategy.decide()

    assert result["status"] == "HOLD"
    assert result["decision"]["action"] == "HOLD"
    # The success path records a structured JSON in reasoning, not a parse-failed tag
    args, _ = fake_store.record_decision.call_args
    assert args[2] == "HOLD NVDA → HOLD"
