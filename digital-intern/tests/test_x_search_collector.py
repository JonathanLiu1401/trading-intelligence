"""xAI x_search intern worker seams — mocked HTTP, no live secrets."""
from __future__ import annotations

import json
import types

from collectors import x_search_collector as xs


def test_load_twitter_accounts_from_sources_json(tmp_path):
    src = tmp_path / "sources.json"
    src.write_text(json.dumps({
        "twitter_accounts": ["@KobeissiLetter", "Forbes", "@business"]
    }))
    assert xs.load_twitter_accounts(src) == ["KobeissiLetter", "Forbes", "business"]


def test_load_twitter_accounts_falls_back_to_defaults(tmp_path):
    src = tmp_path / "missing.json"
    accounts = xs.load_twitter_accounts(src)
    assert "KobeissiLetter" in accounts
    assert "WSJmarkets" in accounts


def test_build_payload_uses_x_search_and_handles():
    payload = xs.build_x_search_payload(["@KobeissiLetter", "Reuters"], from_date="2026-08-13")
    assert payload["tools"] == [{
        "type": "x_search",
        "allowed_x_handles": ["KobeissiLetter", "Reuters"],
        "from_date": "2026-08-13",
    }]
    assert payload["model"]
    user = payload["input"][0]["content"]
    assert "@KobeissiLetter" in user
    assert "@Reuters" in user


def test_articles_from_json_and_citations():
    data = {
        "output_text": json.dumps([
            {
                "handle": "KobeissiLetter",
                "text": "CPI prints hotter than expected",
                "url": "https://x.com/KobeissiLetter/status/1234567890123456789",
                "published": "2026-08-14T12:00:00Z",
                "tickers": ["SPY"],
            }
        ]),
        "citations": [
            "https://x.com/Forbes/status/9876543210987654321",
            "https://x.ai/news",
        ],
    }
    arts = xs.articles_from_response(data)
    links = {a["link"] for a in arts}
    assert "https://x.com/KobeissiLetter/status/1234567890123456789" in links
    assert "https://x.com/Forbes/status/9876543210987654321" in links
    kobe = next(a for a in arts if "KobeissiLetter" in a["source"])
    assert kobe["title"].startswith("@KobeissiLetter:")
    assert "CPI" in kobe["summary"]
    assert kobe["tickers"] == ["SPY"]


def test_collect_x_search_mocked_http(monkeypatch):
    captured = {}

    class Resp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return json.dumps({
                "output": [{
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": json.dumps([{
                            "handle": "business",
                            "text": "Oil jumps on supply scare",
                            "url": "https://x.com/business/status/1111111111111111111",
                            "published": "",
                            "tickers": [],
                        }]),
                    }],
                }],
            }).encode()

    def fake_urlopen(req, timeout=None, context=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode())
        return Resp()

    monkeypatch.setattr("core.claude_cli._load_xai_access_token", lambda: "tok-not-a-secret")
    monkeypatch.setattr(xs.urllib.request, "urlopen", fake_urlopen)
    arts = xs.collect_x_search(accounts=["business"])
    assert captured["url"].endswith("/responses")
    assert captured["body"]["tools"][0]["type"] == "x_search"
    assert captured["body"]["tools"][0]["allowed_x_handles"] == ["business"]
    auth = captured["headers"].get("Authorization") or captured["headers"].get("authorization")
    assert auth == "Bearer tok-not-a-secret"
    assert len(arts) == 1
    assert arts[0]["link"] == "https://x.com/business/status/1111111111111111111"


def test_missing_token_returns_empty(monkeypatch):
    monkeypatch.setattr("core.claude_cli._load_xai_access_token", lambda: None)

    def boom(*_a, **_k):
        raise AssertionError("must not call xAI without a token")

    monkeypatch.setattr(xs.urllib.request, "urlopen", boom)
    assert xs.collect_x_search(accounts=["Reuters"]) == []


def test_daemon_registers_x_search_worker():
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "daemon.py").read_text()
    tree = ast.parse(src)
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "x_search_worker" in names
    assert "x_search" in src
    assert "collect_x_search" in src


def test_collect_fans_out_per_account(monkeypatch):
    calls = []

    class Resp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            handle = calls[-1]
            tid = "1111111111111111111" if handle == "business" else "2222222222222222222"
            return json.dumps({
                "output_text": json.dumps([{
                    "handle": handle,
                    "text": "ping",
                    "url": f"https://x.com/{handle}/status/{tid}",
                    "published": "",
                    "tickers": [],
                }])
            }).encode()

    def fake_urlopen(req, timeout=None, context=None):
        body = json.loads(req.data.decode())
        calls.append(body["tools"][0]["allowed_x_handles"][0])
        return Resp()

    monkeypatch.setattr("core.claude_cli._load_xai_access_token", lambda: "tok-not-a-secret")
    monkeypatch.setattr(xs.urllib.request, "urlopen", fake_urlopen)
    arts = xs.collect_x_search(accounts=["business", "Reuters"])
    assert calls == ["business", "Reuters"]
    assert {a["source"] for a in arts} == {"twitter_xsearch/@business", "twitter_xsearch/@Reuters"}
