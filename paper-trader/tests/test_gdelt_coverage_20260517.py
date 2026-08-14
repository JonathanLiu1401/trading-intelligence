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
        # Post-coverage date so we exercise the API-error short-circuit,
        # not the new pre-coverage skip (that path never calls article_search).
        d, kw = date(2016, 6, 15), "stock market earnings semiconductor"

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


class TestGdeltPreCoverageSkip:
    def test_pre_coverage_date_skips_network_and_caches_empty(self, monkeypatch):
        """Dates before GDELT_COVERAGE_START must not hit the API at all."""
        sleeps: list[float] = []
        monkeypatch.setattr(bt.time, "sleep", lambda s: sleeps.append(s))
        bt.GDELTFetcher._last_request_ts = 0.0
        f = bt.GDELTFetcher()
        calls: list[int] = []
        monkeypatch.setattr(f._client, "article_search",
                            lambda _f: calls.append(1))
        d, kw = date(2001, 6, 15), "stock market earnings semiconductor"
        assert f.fetch(d, kw) == []
        assert calls == []
        assert sleeps == []
        cache = f._cache_key(d, kw)
        assert cache.exists()
        assert json.loads(cache.read_text()) == []
        # Second call is a disk hit.
        assert f.fetch(d, kw) == []
        assert calls == []

    def test_coverage_monday_before_start_is_skipped(self, monkeypatch):
        """2015-02-16 (Monday of the coverage-start week) used to be queried
        by weekly prewarm and then cached as 'outside coverage' after 3
        rate-limit retries. Must skip with zero API calls."""
        bt.GDELTFetcher._last_request_ts = 0.0
        f = bt.GDELTFetcher()
        calls: list[int] = []
        monkeypatch.setattr(f._client, "article_search",
                            lambda _f: calls.append(1))
        d = date(2015, 2, 16)
        assert d < bt.GDELT_COVERAGE_START
        assert f.fetch(d, "SP500 market rally selloff") == []
        assert calls == []


class TestGdeltSharedLimiter:
    def test_two_instances_share_one_lock_and_timestamp(self):
        a = bt.GDELTFetcher()
        b = bt.GDELTFetcher()
        assert a._request_lock is b._request_lock
        assert a._request_lock is bt.GDELTFetcher._request_lock
        # Class-level timestamp, not per-instance.
        bt.GDELTFetcher._last_request_ts = 123.0
        assert a._last_request_ts == 123.0
        assert b._last_request_ts == 123.0
        bt.GDELTFetcher._last_request_ts = 0.0


class TestGdeltWeeklyWarmClamp:
    def test_weekly_warm_does_not_start_before_coverage(self, monkeypatch):
        import paper_trader.historical_collector as hc
        fetched: list[date] = []

        class _FakeFetcher:
            def _cache_key(self, d, kw):
                return bt.GDELT_CACHE / f"{d.isoformat()}_x.json"

            def fetch(self, d, kw):
                fetched.append(d)
                return []

        monkeypatch.setattr(hc, "GDELTFetcher", _FakeFetcher)
        monkeypatch.setattr(hc, "KEYWORD_GROUPS", ["kw-only"])
        n = hc.warm_gdelt_weekly(date(2015, 2, 19), date(2015, 2, 25))
        assert date(2015, 2, 16) not in fetched
        assert all(d >= bt.GDELT_COVERAGE_START for d in fetched)
        assert n == len(fetched) >= 1
