"""get_stale_sources: a source not polled within the window is 'stale'.

Stale means the worker stopped calling record_result at all (crash / the
collector raising before recording). This is the complement of `disabled`
(polled but empty) and is the signal surfaced on the [source_health] log
line consumed by the hourly audit.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from collectors import source_health


@pytest.fixture
def sh(tmp_path, monkeypatch):
    """Point source_health at an isolated DB and clear its path cache."""
    db = tmp_path / "source_health.db"
    monkeypatch.setattr(source_health, "_db_path_cache", db)
    return source_health


def _backdate(sh_mod, source: str, age_secs: int) -> None:
    """Force a source's last_seen to age_secs in the past."""
    ts = (datetime.now(timezone.utc) - timedelta(seconds=age_secs)).isoformat(
        timespec="seconds"
    )
    conn = sh_mod._connect()
    try:
        conn.execute(
            "UPDATE source_health SET last_seen = ? WHERE source = ?", (ts, source)
        )
        conn.commit()
    finally:
        conn.close()


def test_fresh_source_not_stale(sh):
    sh.record_result("rss", 5)
    assert sh.get_stale_sources() == []


def test_old_source_is_stale(sh):
    sh.record_result("rss", 5)
    _backdate(sh, "rss", source_health.DEFAULT_STALE_SECS + 60)
    assert sh.get_stale_sources() == ["rss"]


def test_threshold_boundary(sh):
    sh.record_result("rss", 1)
    # Just inside the window -> healthy.
    _backdate(sh, "rss", source_health.DEFAULT_STALE_SECS - 120)
    assert sh.get_stale_sources() == []
    # Just outside -> stale.
    _backdate(sh, "rss", source_health.DEFAULT_STALE_SECS + 120)
    assert sh.get_stale_sources() == ["rss"]


def test_custom_max_age(sh):
    sh.record_result("web", 3)
    _backdate(sh, "web", 600)
    assert sh.get_stale_sources(max_age_secs=300) == ["web"]
    assert sh.get_stale_sources(max_age_secs=900) == []


def test_unparseable_last_seen_is_stale(sh):
    sh.record_result("nitter", 0)
    conn = sh._connect()
    try:
        conn.execute(
            "UPDATE source_health SET last_seen = ? WHERE source = ?",
            ("not-a-timestamp", "nitter"),
        )
        conn.commit()
    finally:
        conn.close()
    assert "nitter" in sh.get_stale_sources()


def test_result_sorted_and_multi(sh):
    sh.record_result("zsrc", 1)
    sh.record_result("asrc", 1)
    _backdate(sh, "zsrc", source_health.DEFAULT_STALE_SECS + 60)
    _backdate(sh, "asrc", source_health.DEFAULT_STALE_SECS + 60)
    assert sh.get_stale_sources() == ["asrc", "zsrc"]


def test_disabled_but_recently_polled_is_not_stale(sh):
    # 3 zero passes -> disabled, but last_seen is fresh: empty, not dead.
    for _ in range(source_health.FAILURE_THRESHOLD):
        sh.record_result("finnhub", 0)
    assert "finnhub" in sh.get_disabled_sources()
    assert sh.get_stale_sources() == []
