"""Pytest fixtures: redirect ArticleStore to a per-test sqlite file."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable when pytest is invoked from elsewhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# tests/test_alert_history.py is an orphan: it imports ``watchers.alert_history``
# which has NEVER existed in git history (``git log --all -- watchers/alert_history.py``
# is empty). It was written against an earlier design that shipped instead as
# ``watchers.alert_recency`` (committed 8410f05) and is exercised by the tracked
# ``tests/test_alert_recency.py``. Left in place, its import error aborts
# collection of the ENTIRE suite (484 tests silently never run). Ignore the
# orphan so the real suite collects; the file itself is left untouched.
collect_ignore = ["test_alert_history.py"]


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


@pytest.fixture(autouse=True)
def _isolate_alert_recency(tmp_path, monkeypatch):
    """Redirect watchers.alert_recency to a per-test SQLite file.

    ``send_urgent_alert`` records the canonical signature of every fired
    story to a *persistent* ``data/alert_recency.db`` for cross-cycle
    duplicate suppression (intended in production). Without this redirect,
    every alert-path test would write to that one real file and the next
    test reusing a default headline would be cross-suppressed — a state leak
    across tests and into the repo's data dir. This is the exact analogue of
    ``store_factory`` redirecting ``article_store._get_db_path``; it isolates
    the new persistent store and changes no test's assertions. Autouse so
    suites that never heard of alert_recency are still isolated.
    """
    try:
        from watchers import alert_recency
    except Exception:
        return
    monkeypatch.setattr(
        alert_recency, "DB_PATH", tmp_path / "alert_recency.db"
    )
