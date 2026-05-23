"""Atomic JSON write contract — 2026-05-23 review pass #22.

GDELT, AV cache, and the AV cross-restart quota tracker used bare
``path.write_text(json.dumps(...))`` for years. A SIGKILL / OOM mid-write
left a torn JSON file: the next read's ``except Exception: pass`` guard
silently treated the corruption as "no cache", forcing a network refetch
that burned the 5s GDELT rate limit, the 22/day AV quota, or — worst —
silently reset the AV cross-restart quota counter to 0 (defeating the
explicit guarantee in the AV_QUOTA_PATH comment).

These tests pin the post-fix invariants:

1. The new ``_atomic_write_json`` helper writes via tmp + ``Path.replace``
   so a torn intermediate state is never visible to readers.
2. ``_inc_quota`` round-trips through the helper — the on-disk file is
   parseable JSON after any number of bumps.
3. The GDELT cache and AV cache writes use the helper too — same
   contract.
4. Atomic write degrades cleanly on serialization failure (no exception
   propagates, matches the bare-write best-effort behaviour).
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import paper_trader.backtest as bt
from paper_trader.backtest import (
    AlphaVantageNewsFetcher,
    GDELTFetcher,
    _atomic_write_json,
)


class TestAtomicWriteJsonHelper:
    def test_round_trips_payload(self, tmp_path):
        path = tmp_path / "out.json"
        payload = {"date": "2026-05-23", "calls": 7, "extra": [1, 2, 3]}
        _atomic_write_json(path, payload)
        assert json.loads(path.read_text()) == payload

    def test_tmp_file_is_removed_after_replace(self, tmp_path):
        """The tmp file used during the write should not linger."""
        path = tmp_path / "out.json"
        _atomic_write_json(path, {"k": 1})
        # The tmp suffix is `.json.tmp`. After a successful `Path.replace`,
        # the tmp is renamed onto the destination — it should not exist as
        # a separate file.
        tmp = path.with_suffix(".json.tmp")
        assert not tmp.exists()
        assert path.exists()

    def test_overwrites_existing_file_atomically(self, tmp_path):
        path = tmp_path / "out.json"
        path.write_text(json.dumps({"old": "data"}))
        _atomic_write_json(path, {"new": "data"})
        assert json.loads(path.read_text()) == {"new": "data"}

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "out.json"
        _atomic_write_json(path, {"k": 1})
        assert json.loads(path.read_text()) == {"k": 1}

    def test_does_not_raise_on_serialization_failure(self, tmp_path, capsys):
        """A non-serializable payload should print a warning but not
        propagate — the call sites are all best-effort and the bare
        ``write_text`` had the same behaviour."""
        path = tmp_path / "out.json"

        class _Unserializable:
            pass

        # Should NOT raise — best-effort contract.
        _atomic_write_json(path, _Unserializable())
        # And the destination file should still NOT exist (the tmp
        # serialization failed before the atomic replace).
        assert not path.exists()
        captured = capsys.readouterr()
        assert "atomic_json" in captured.out

    def test_no_torn_intermediate_visible_to_concurrent_reader(self, tmp_path):
        """Replace is atomic: a reader sees either the OLD complete file
        or the NEW complete file, never a half-written tmp.

        Simulated by checking that the destination file at every step
        between two atomic writes is a parseable complete JSON document.
        """
        path = tmp_path / "out.json"
        payloads = [
            {"calls": i, "date": "2026-05-23"} for i in range(20)
        ]
        for p in payloads:
            _atomic_write_json(path, p)
            # The on-disk state must always be the complete latest payload.
            parsed = json.loads(path.read_text())
            assert parsed == p


class TestIncQuotaAtomicWrite:
    def test_inc_quota_writes_parseable_json(self):
        """The end-to-end on-disk file is valid JSON after every bump.

        Pre-fix: ``write_text`` was not atomic; a kill mid-write left a
        torn file. This test does not simulate the kill (impossible in a
        single-process test) but verifies that the on-disk format is
        valid JSON after ``_inc_quota`` — the atomic helper guarantees
        this is exactly the LAST-completed payload, never a partial one.
        """
        f = AlphaVantageNewsFetcher()
        for _ in range(5):
            f._inc_quota()
        # Direct read — bypassing _quota()'s broad except so a corrupt
        # write surfaces as a test failure, not a silent reset.
        raw = json.loads(bt.AV_QUOTA_PATH.read_text())
        assert raw == {"date": date.today().isoformat(), "calls": 5}

    def test_inc_quota_no_lingering_tmp_after_bump(self):
        """After ``_inc_quota`` returns, no ``.tmp`` sibling should exist
        (the atomic replace consumed it)."""
        f = AlphaVantageNewsFetcher()
        f._inc_quota()
        tmp = bt.AV_QUOTA_PATH.with_suffix(".json.tmp")
        assert not tmp.exists()
        assert bt.AV_QUOTA_PATH.exists()


class TestGdeltCacheAtomicWrite:
    def test_gdelt_cache_write_is_atomic(self, monkeypatch):
        """After a successful fetch, the cache file is a complete list
        of dicts with no ``.tmp`` sibling."""
        # Make GDELT_CACHE point under a tmp (the conftest fixture already
        # does this) and bypass network — patch the client.
        fetcher = GDELTFetcher()

        # Patch the underlying gdeltdoc client to return a fake non-empty
        # result so the success branch fires.
        import pandas as pd

        fake_df = pd.DataFrame([
            {"title": "Fake headline", "url": "http://x/1",
             "domain": "x.com", "seendate": "2026-05-23T00:00:00Z"},
        ])
        fetcher._client.article_search = MagicMock(return_value=fake_df)

        d = date(2026, 5, 23)
        articles = fetcher.fetch(d, "test keyword group")

        assert isinstance(articles, list) and len(articles) == 1
        # On-disk file is a complete list of dicts.
        cache_path = fetcher._cache_key(d, "test keyword group")
        assert cache_path.exists()
        cached = json.loads(cache_path.read_text())
        assert isinstance(cached, list)
        assert len(cached) == 1
        assert cached[0]["title"] == "Fake headline"
        # No lingering tmp.
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        assert not tmp.exists()

    def test_gdelt_cache_negative_cached_atomically(self, monkeypatch):
        """A permanent failure (e.g. pre-coverage date) caches an empty
        list — verify that is also via the atomic helper."""
        fetcher = GDELTFetcher()
        # Simulate a permanent ValueError.
        fetcher._client.article_search = MagicMock(
            side_effect=ValueError("The query was not valid: Invalid query start date"))
        d = date(2010, 1, 1)  # pre-GDELT coverage
        articles = fetcher.fetch(d, "ancient kw")
        assert articles == []
        cache_path = fetcher._cache_key(d, "ancient kw")
        assert cache_path.exists()
        # Empty list is the legitimate negative-cache value.
        assert json.loads(cache_path.read_text()) == []
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        assert not tmp.exists()


class TestAvCacheAtomicWrite:
    def test_av_cache_write_is_atomic(self, monkeypatch):
        """An AV NEWS_SENTIMENT cache entry is a complete list of items
        with no torn intermediate."""
        # Force the AV fetcher to have an API key so the network branch
        # would fire — then patch requests.get.
        monkeypatch.setenv("ALPHA_VANTAGE_KEY", "fake_key")
        fetcher = AlphaVantageNewsFetcher()
        assert fetcher._key == "fake_key"

        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            "feed": [
                {"title": "AV item 1", "url": "http://av/1",
                 "source": "AV"},
                {"title": "AV item 2", "url": "http://av/2",
                 "source": "AV"},
                # A row without a title — should be filtered.
                {"title": "", "url": "http://av/3", "source": "AV"},
            ]
        }
        monkeypatch.setattr(bt.requests, "get",
                            MagicMock(return_value=fake_resp))
        # Speed: bypass the 1.2s rate-limit sleep.
        monkeypatch.setattr(bt.time, "sleep", lambda *a, **k: None)

        d = date(2026, 5, 23)
        items = fetcher.fetch(["NVDA"], d)
        # Two real items returned.
        assert len(items) == 2
        cache_path = fetcher._cache_path("NVDA", d)
        assert cache_path.exists()
        cached = json.loads(cache_path.read_text())
        # Cached items survived the title-filter (entry 3 dropped).
        assert isinstance(cached, list) and len(cached) == 2
        assert all(isinstance(x, dict) and x.get("title") for x in cached)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        assert not tmp.exists()

        # And the quota tracker was bumped exactly once for this fetch.
        raw = json.loads(bt.AV_QUOTA_PATH.read_text())
        assert raw["calls"] == 1
