"""Regression lock (2026-05-17 review): GDELT permanent-vs-transient errors.

``GDELTFetcher.fetch`` must distinguish a *deterministic permanent* failure
(a pre-2017 date the GDELT DOC 2.0 index will NEVER cover — it raises
"The query was not valid … Invalid query start date") from a *transient*
one (rate-limit / connection drop):

- **Permanent** → exactly one attempt, ZERO backoff sleeps, and an empty
  result negative-cached so every later cycle is a pure disk hit. Without
  this the continuous loop (windows back to 1993) burned 20+40+60s of
  backoff per (date,keyword) and re-attempted it every cycle for hours.
- **Transient** → keep the full 3-retry escalating-backoff path and NEVER
  negative-cache (poisoning a temporarily-failing date for the loop's life).
- A covered date with genuinely no articles still caches ``[]`` (the
  pre-existing behaviour the refactor preserves).

Offline & deterministic — ``article_search`` is monkeypatched, ``time.sleep``
is stubbed, and the GDELT cache dir is the conftest tmp redirect.
"""
from __future__ import annotations

import json
from datetime import date

import paper_trader.backtest as bt


class TestGdeltPermanentError:
    def test_permanent_error_caches_empty_and_does_not_retry(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(bt.time, "sleep", lambda s: sleeps.append(s))
        f = bt.GDELTFetcher()
        calls: list[int] = []

        def _boom(_filters):
            calls.append(1)
            raise ValueError("The query was not valid. The API error "
                             "message was: Invalid query start date.")

        monkeypatch.setattr(f._client, "article_search", _boom)
        d, kw = date(2001, 6, 15), "stock market earnings semiconductor"

        res = f.fetch(d, kw)
        assert res == []
        # A deterministic permanent error: exactly ONE attempt, ZERO backoff.
        assert len(calls) == 1
        assert sleeps == []
        cache = f._cache_key(d, kw)
        assert cache.exists()
        assert json.loads(cache.read_text()) == []   # negative-cached

        # Second call is a pure disk hit — article_search not invoked again.
        assert f.fetch(d, kw) == []
        assert len(calls) == 1

    def test_transient_error_retries_and_is_not_cached(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(bt.time, "sleep", lambda s: sleeps.append(s))
        f = bt.GDELTFetcher()
        calls: list[int] = []

        def _conn_err(_filters):
            calls.append(1)
            raise ConnectionError("Connection aborted. RemoteDisconnected")

        monkeypatch.setattr(f._client, "article_search", _conn_err)
        d, kw = date(2024, 6, 18), "currency forex dollar euro yen"

        res = f.fetch(d, kw)
        assert res == []
        # Full retry budget exhausted with the escalating 20/40/60s backoff
        # each round (interleaved rate-limit pre-sleeps may add more entries —
        # the load-bearing property is that all three backoffs fired, i.e.
        # the transient error was NOT short-circuited like a permanent one).
        assert len(calls) == 3
        assert {20.0, 40.0, 60.0}.issubset(set(sleeps))
        # A transient failure must NEVER be negative-cached.
        assert not f._cache_key(d, kw).exists()

    def test_successful_empty_result_is_cached(self, monkeypatch):
        """A covered date that genuinely has no articles still caches []
        (pre-existing behaviour preserved by the refactor)."""
        monkeypatch.setattr(bt.time, "sleep", lambda *_a: None)
        f = bt.GDELTFetcher()

        class _EmptyDF:
            empty = True

        monkeypatch.setattr(f._client, "article_search", lambda _f: _EmptyDF())
        d, kw = date(2024, 6, 18), "semiconductor chip AI earnings beat"
        assert f.fetch(d, kw) == []
        assert json.loads(f._cache_key(d, kw).read_text()) == []
