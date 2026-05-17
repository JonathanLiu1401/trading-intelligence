"""Pytest fixtures: redirect ArticleStore to a per-test sqlite file."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable when pytest is invoked from elsewhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    """Return a callable that builds a fresh ArticleStore backed by tmp_path.

    Monkeypatches storage.article_store._get_db_path so the in-test store never
    touches the real production DB. The store class opens its connection in
    __init__; the patch must be in place *before* the first instantiation.
    """
    from storage import article_store

    def _factory():
        # Each invocation gets its own filename so tests can spin up multiple
        # independent stores without cross-contamination.
        db_path = tmp_path / f"articles_{_factory.counter}.db"
        _factory.counter += 1
        monkeypatch.setattr(
            article_store, "_get_db_path", lambda: db_path
        )
        return article_store.ArticleStore()

    _factory.counter = 0
    return _factory


@pytest.fixture
def store(store_factory):
    """Most tests want a single store — provide it directly."""
    return store_factory()
