"""Unit tests for collectors.hackernews_collector.

Fully mocked — no network. Asserts the standard collector dict shape and the
specific behaviours the daemon relies on: link fallback, tz-aware published
stamp, front-page relevance filtering, query hits kept unfiltered, dedup
across endpoints, and graceful [] on network / non-200 failures.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from collectors import hackernews_collector as hn


class _FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


# A known epoch: 2026-05-18T09:00:00+00:00
_EPOCH = int(datetime(2026, 5, 18, 9, 0, 0, tzinfo=timezone.utc).timestamp())

_FRONT_HIT_RELEVANT = {
    "objectID": "111",
    "title": "Acme Corp misses Q1 earnings, stock drops 12%",
    "url": "https://example.com/acme-earnings",
    "points": 240,
    "num_comments": 88,
    "created_at_i": _EPOCH,
}
_FRONT_HIT_IRRELEVANT = {
    "objectID": "222",
    "title": "Show HN: my weekend houseplant watering robot",
    "url": "https://example.com/plant-bot",
    "points": 30,
    "num_comments": 4,
    "created_at_i": _EPOCH,
}
_QUERY_HIT_NO_URL = {
    "objectID": "333",
    "title": "Ask HN: thoughts on the macro outlook",  # no obvious keyword
    "url": "",  # Ask HN posts have no external url
    "points": 15,
    "num_comments": 60,
    "created_at_i": _EPOCH,
}
_QUERY_HIT_DUP_OF_FRONT = dict(_FRONT_HIT_RELEVANT)  # same objectID 111


def _router(front_hits, search_hits):
    """Return a fake requests.get that serves front-page vs search payloads."""
    def _get(url, **kwargs):
        if "front_page" in url:
            return _FakeResp(200, {"hits": front_hits})
        return _FakeResp(200, {"hits": search_hits})
    return _get


def test_standard_shape_and_values(monkeypatch):
    monkeypatch.setattr(hn.requests, "get",
                        _router([_FRONT_HIT_RELEVANT], []))
    arts = hn.collect_hackernews()
    assert len(arts) == 1
    a = arts[0]
    assert set(a) >= {"title", "link", "summary", "published", "source"}
    assert a["source"] == "hackernews"
    assert a["title"] == "Acme Corp misses Q1 earnings, stock drops 12%"
    assert a["link"] == "https://example.com/acme-earnings"
    assert a["_hn_id"] == "111"
    assert a["_hn_points"] == 240
    assert a["_hn_comments"] == 88


def test_link_falls_back_to_hn_discussion(monkeypatch):
    # No url -> link must be the HN item discussion page. Routed via a query
    # endpoint (require_relevant=False) so it isn't dropped by the filter.
    monkeypatch.setattr(hn.requests, "get",
                        _router([], [_QUERY_HIT_NO_URL]))
    arts = hn.collect_hackernews()
    assert len(arts) == 1
    assert arts[0]["link"] == "https://news.ycombinator.com/item?id=333"
    assert arts[0]["_hn_discussion"] == "https://news.ycombinator.com/item?id=333"


def test_published_is_tz_aware_and_roundtrips(monkeypatch):
    monkeypatch.setattr(hn.requests, "get",
                        _router([_FRONT_HIT_RELEVANT], []))
    pub = hn.collect_hackernews()[0]["published"]
    assert pub.endswith("+00:00"), f"expected tz-aware UTC, got {pub!r}"
    parsed = datetime.fromisoformat(pub)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0
    assert parsed == datetime(2026, 5, 18, 9, 0, 0, tzinfo=timezone.utc)


def test_published_falls_back_to_now_on_bad_epoch(monkeypatch):
    monkeypatch.setattr(hn.requests, "get",
                        _router([dict(_FRONT_HIT_RELEVANT, created_at_i=None)], []))
    pub = hn.collect_hackernews()[0]["published"]
    assert pub.endswith("+00:00")
    # Within a minute of now (test wall clock).
    delta = abs((datetime.now(timezone.utc)
                 - datetime.fromisoformat(pub)).total_seconds())
    assert delta < 60


def test_front_page_relevance_filter(monkeypatch):
    # Relevant + irrelevant on the front page -> only the relevant one kept.
    monkeypatch.setattr(
        hn.requests, "get",
        _router([_FRONT_HIT_RELEVANT, _FRONT_HIT_IRRELEVANT], []))
    arts = hn.collect_hackernews()
    ids = {a["_hn_id"] for a in arts}
    assert ids == {"111"}


def test_query_hits_kept_without_relevance_filter(monkeypatch):
    # _QUERY_HIT_NO_URL has no finance keyword in its title; coming from a
    # search query it must still be kept (the query already made it topical).
    monkeypatch.setattr(hn.requests, "get",
                        _router([], [_QUERY_HIT_NO_URL]))
    arts = hn.collect_hackernews()
    assert [a["_hn_id"] for a in arts] == ["333"]


def test_dedup_by_objectid_across_endpoints(monkeypatch):
    # Same story on the front page and returned by a query -> one article.
    monkeypatch.setattr(
        hn.requests, "get",
        _router([_FRONT_HIT_RELEVANT], [_QUERY_HIT_DUP_OF_FRONT]))
    arts = hn.collect_hackernews()
    assert len(arts) == 1
    assert arts[0]["_hn_id"] == "111"


def test_network_error_returns_empty(monkeypatch):
    monkeypatch.setattr(hn.time, "sleep", lambda *_: None)  # no real backoff

    def _boom(*a, **k):
        raise ConnectionError("network down")

    monkeypatch.setattr(hn.requests, "get", _boom)
    assert hn.collect_hackernews() == []


def test_non_200_returns_empty(monkeypatch):
    monkeypatch.setattr(hn.time, "sleep", lambda *_: None)
    monkeypatch.setattr(hn.requests, "get",
                        lambda *a, **k: _FakeResp(503, {"hits": ["x"]}))
    assert hn.collect_hackernews() == []
