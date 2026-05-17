"""`/api/chat` session-delta context block (web_server.py).

The chat is the trader's main conversational surface; every other context
stream it assembles is a *current-state snapshot*. The session-delta block is
the one "what materially changed since you last looked" view. These tests pin:

1. an ACTIVE session-delta is injected into the Claude prompt, headline +
   ranked event summaries, positioned after the PAPER TRADER LIVE STATE block;
2. the network sub-fetch is guarded exactly like its 4 siblings — an
   unreachable :8090 must NOT raise into the chat: the request still returns
   200 and the prompt simply omits the section (the `+ (… if block else "")`
   pattern);
3. a QUIET/NO_DATA window is silence (ACTIVE-only contract — matches the
   unified :8888 chat's _fetch_session_delta so the two surfaces stay
   consistent).

The chat ends in a Claude CLI call; we stub it and capture the assembled
prompt (memory: verify via the Flask test client, not a module __main__
smoke that hits a different/empty DB).
"""
from __future__ import annotations

import json

import pytest

from dashboard.web_server import create_app


class _FakeResp:
    """Minimal context-manager HTTP response (urlopen replacement)."""

    def __init__(self, payload: dict):
        self._b = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._b


_ACTIVE_SD = {
    "state": "ACTIVE",
    "headline": ("Since 03:07 UTC (6h): 1 fill; equity $+12.40 (+1.24%) "
                 "vs SPY +0.10% → +1.14pp."),
    "events": [
        {"kind": "TRADE", "severity": "HIGH",
         "summary": "BUY NVDA 0.4 @ $225.79 ($90.32)"},
        {"kind": "EQUITY_MOVE", "severity": "HIGH",
         "summary": "Equity $+12.40 (+1.24%) vs SPY +0.10% → +1.14pp"},
    ],
}


def _client(store, monkeypatch, *, sd_payload, sd_raises=False):
    """Build the chat app with the Claude call stubbed (prompt captured) and
    urlopen routed: the session-delta URL returns ``sd_payload`` (or raises if
    ``sd_raises``); every other sub-fetch raises so it degrades to empty."""
    app = create_app(store)
    captured: dict = {}

    def _fake_claude(prompt, **kw):
        captured["prompt"] = prompt
        return "stub-response"

    monkeypatch.setattr("dashboard.web_server._claude_cli_call", _fake_claude)

    import urllib.error
    import urllib.request

    def _fake_urlopen(url, timeout=None):
        u = url if isinstance(url, str) else getattr(url, "full_url", "")
        if "/api/session-delta" in u:
            if sd_raises:
                raise urllib.error.URLError("session-delta down (test)")
            return _FakeResp(sd_payload)
        # Every other sub-fetch (state/greeks/heatmap/analytics/earnings)
        # degrades to empty — the documented swallow behaviour.
        raise urllib.error.URLError("upstream down (test)")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    return app.test_client(), captured


def test_active_session_delta_injected_after_live_state(store, monkeypatch):
    client, captured = _client(store, monkeypatch, sd_payload=_ACTIVE_SD)
    r = client.post("/api/chat", json={"message": "what changed?",
                                       "history": []})
    assert r.status_code == 200
    prompt = captured["prompt"]
    hdr = "WHAT MATERIALLY CHANGED SINCE YOU LAST LOOKED"
    assert hdr in prompt
    # Headline restated verbatim + each ranked event summary bulleted.
    assert _ACTIVE_SD["headline"] in prompt
    assert "BUY NVDA 0.4 @ $225.79 ($90.32)" in prompt
    assert "• Equity $+12.40 (+1.24%)" in prompt
    # Framed right after the live-state snapshot (before deep analytics).
    assert prompt.index("PAPER TRADER LIVE STATE") < prompt.index(hdr)


def test_unreachable_session_delta_omits_block_but_chat_still_answers(
        store, monkeypatch):
    """The sub-fetch must never raise into the chat (sibling contract)."""
    client, captured = _client(store, monkeypatch,
                               sd_payload=_ACTIVE_SD, sd_raises=True)
    r = client.post("/api/chat", json={"message": "status?", "history": []})
    assert r.status_code == 200            # chat still answers
    assert r.get_json()["response"] == "stub-response"
    assert "WHAT MATERIALLY CHANGED SINCE YOU LAST LOOKED" \
        not in captured["prompt"]


@pytest.mark.parametrize("state", ["QUIET", "NO_DATA"])
def test_non_active_window_is_silence(store, monkeypatch, state):
    """A QUIET/NO_DATA window contributes no actionable content to a
    data-driven analyst prompt — suppressed (ACTIVE-only, matches the
    unified :8888 chat)."""
    payload = {"state": state, "headline": "Quiet since 03:00 UTC (6h).",
               "events": []}
    client, captured = _client(store, monkeypatch, sd_payload=payload)
    r = client.post("/api/chat", json={"message": "anything?",
                                       "history": []})
    assert r.status_code == 200
    assert "WHAT MATERIALLY CHANGED SINCE YOU LAST LOOKED" \
        not in captured["prompt"]
    assert "Quiet since 03:00 UTC" not in captured["prompt"]
