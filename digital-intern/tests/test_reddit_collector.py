"""Reddit ingest: RSS parse + 429 circuit. No live Reddit keys."""
from __future__ import annotations

import types

from collectors import reddit_collector as rc


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>stocks</title>
  <entry>
    <title>NVDA earnings reaction</title>
    <link rel="alternate" href="https://www.reddit.com/r/stocks/comments/abc/nvda/"/>
    <updated>2026-08-14T16:00:00+00:00</updated>
    <content type="html">&lt;p&gt;Chip names bid after the print.&lt;/p&gt;</content>
  </entry>
</feed>
"""

RSS2 = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>investing</title>
  <item>
    <title>Fed minutes</title>
    <link>https://www.reddit.com/r/investing/comments/def/fed/</link>
    <description>Dovish lean in the minutes.</description>
    <pubDate>Fri, 14 Aug 2026 12:00:00 +0000</pubDate>
  </item>
</channel></rss>
"""


def test_parse_atom_feed():
    items = rc._parse_rss_xml(ATOM, "stocks")
    assert len(items) == 1
    assert items[0]["title"] == "NVDA earnings reaction"
    assert items[0]["link"].endswith("/nvda/")
    assert items[0]["source"] == "reddit/r/stocks"
    assert "Chip names" in items[0]["summary"]


def test_parse_rss2_feed():
    items = rc._parse_rss_xml(RSS2, "investing")
    assert len(items) == 1
    assert items[0]["title"] == "Fed minutes"
    assert items[0]["source"] == "reddit/r/investing"


class _FakeResp:
    def __init__(self, status, text="", headers=None):
        self.status_code = status
        self.text = text
        self.headers = headers or {}
        self.content = text.encode()

    def json(self):
        raise ValueError("not json")


def test_429_trips_circuit_and_skips_later_gets(monkeypatch):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return _FakeResp(429, headers={"Retry-After": "0"})

    monkeypatch.setattr(rc.requests, "get", fake_get)
    monkeypatch.setattr(rc.time, "sleep", lambda *_a, **_k: None)
    rc._rate_limited = False
    rc._http_fail_logged = 0
    assert rc._fetch_listing("stocks", "hot") == []
    assert rc._rate_limited is True
    first = len(calls)
    assert rc._fetch_listing("investing", "hot") == []
    assert len(calls) == first  # circuit open, no more HTTP


def test_rss_success_path(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        assert "old.reddit.com" in url
        assert url.endswith(".rss")
        return _FakeResp(200, ATOM)

    monkeypatch.setattr(rc.requests, "get", fake_get)
    rc._rate_limited = False
    rc._http_fail_logged = 0
    items = rc._fetch_listing("stocks", "hot")
    assert len(items) == 1
    assert items[0]["title"] == "NVDA earnings reaction"


def test_oauth_used_when_env_present(monkeypatch):
    seen = {}

    def fake_post(url, data=None, auth=None, headers=None, timeout=None):
        seen["auth"] = auth
        return types.SimpleNamespace(
            status_code=200,
            json=lambda: {"access_token": "reddit-app-token"},
        )

    def fake_get(url, headers=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        return types.SimpleNamespace(
            status_code=200,
            text="",
            headers={},
            json=lambda: {
                "data": {
                    "children": [{
                        "data": {
                            "title": "OAuth post",
                            "url": "https://www.reddit.com/r/stocks/comments/xyz/",
                            "score": 10,
                            "selftext": "hi",
                            "created_utc": 1,
                            "permalink": "/r/stocks/comments/xyz/",
                        }
                    }]
                }
            },
        )

    monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csecret")
    monkeypatch.setattr(rc.requests, "post", fake_post)
    monkeypatch.setattr(rc.requests, "get", fake_get)
    rc._rate_limited = False
    token = rc._reddit_oauth_token()
    assert token == "reddit-app-token"
    items = rc._fetch_listing("stocks", "hot", token)
    assert seen["url"].startswith("https://oauth.reddit.com/r/stocks/hot.json")
    assert seen["headers"]["Authorization"] == "Bearer reddit-app-token"
    assert items[0]["title"] == "OAuth post"


def test_no_oauth_without_env(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    assert rc._reddit_oauth_token() is None
