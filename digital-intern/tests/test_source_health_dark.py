"""get_dark_sources: a source still polling but producing nothing.

This is the chronic state of sec_edgar / polygon / newsapi / nitter per the
operator's standing observation (memory: di-chronic-dark-collectors). They
are NOT stale (workers actively poll) and the `disabled` bit flips on/off as
the 3-consecutive-failure counter resets — neither signal answers the analyst
question "has this source actually given me any news today?". The new
last_success column + get_dark_sources do.

Tests pin the gate's exact contract so the two adjacent failure signals
(stale / disabled / dark) can never silently merge:
  1. last_success is stamped on a productive pass and NEVER moves on a zero
     pass — the "darkness clock" measures from the last real article.
  2. A never-succeeded source surfaces with dark_secs = -1 sentinel.
  3. A stale source (worker stopped polling) is EXCLUDED — get_stale_sources
     owns that signal.
  4. Sort: never-succeeded first, then by descending dark_secs, then alpha.
  5. last_success surfaces in get_health_report payload.
  6. Migration: a pre-existing DB without the column gets it added on
     _connect, without losing prior rows.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from collectors import source_health


@pytest.fixture
def sh(tmp_path, monkeypatch):
    db = tmp_path / "source_health.db"
    monkeypatch.setattr(source_health, "_db_path_cache", db)
    monkeypatch.setattr(source_health, "_schema_ready_path", None)
    return source_health


def _backdate(sh_mod, source: str, *, last_seen_age=None, last_success_age=None) -> None:
    now = datetime.now(timezone.utc)
    conn = sh_mod._connect()
    try:
        if last_seen_age is not None:
            ts = (now - timedelta(seconds=last_seen_age)).isoformat(timespec="seconds")
            conn.execute(
                "UPDATE source_health SET last_seen = ? WHERE source = ?", (ts, source)
            )
        if last_success_age is not None:
            ts = (now - timedelta(seconds=last_success_age)).isoformat(timespec="seconds")
            conn.execute(
                "UPDATE source_health SET last_success = ? WHERE source = ?",
                (ts, source),
            )
        conn.commit()
    finally:
        conn.close()


# ── last_success bookkeeping ─────────────────────────────────────────────────

def test_last_success_stamped_on_productive_pass(sh):
    sh.record_result("rss", 7)
    report = sh.get_health_report()
    assert report["rss"]["last_success"] is not None
    # Must be parseable as an ISO timestamp
    datetime.fromisoformat(report["rss"]["last_success"])


def test_last_success_null_on_zero_first_pass(sh):
    sh.record_result("nitter", 0)
    report = sh.get_health_report()
    assert report["nitter"]["last_success"] is None
    assert report["nitter"]["total_articles"] == 0


def test_last_success_frozen_on_subsequent_zero_pass(sh):
    """A productive pass stamps last_success; later zero passes MUST NOT
    advance it — the darkness clock measures from the last real article."""
    sh.record_result("polygon", 5)
    first_success = sh.get_health_report()["polygon"]["last_success"]
    # Several zero passes
    for _ in range(5):
        sh.record_result("polygon", 0)
    later_success = sh.get_health_report()["polygon"]["last_success"]
    assert later_success == first_success, (
        "last_success advanced on a zero pass — darkness clock broken"
    )


def test_last_success_refreshed_on_recovery(sh):
    sh.record_result("rss", 3)
    # Backdate last_success an hour into the past, then record a fresh
    # productive pass: the new pass MUST advance last_success past the
    # backdated marker (i.e., recovery resets the darkness clock).
    _backdate(sh, "rss", last_success_age=3600)
    backdated = sh.get_health_report()["rss"]["last_success"]
    sh.record_result("rss", 1)
    later = sh.get_health_report()["rss"]["last_success"]
    assert later != backdated
    assert later > backdated  # ISO timestamps sort lexicographically


# ── get_dark_sources ──────────────────────────────────────────────────────────

def test_fresh_productive_source_not_dark(sh):
    sh.record_result("rss", 5)
    assert sh.get_dark_sources() == []


def test_dark_source_with_old_last_success_surfaces(sh):
    sh.record_result("polygon", 1)
    # Make last_success ancient but keep last_seen fresh (still polling).
    _backdate(sh, "polygon", last_success_age=source_health.DEFAULT_DARK_SECS + 600)
    sh.record_result("polygon", 0)  # fresh poll, still 0
    dark = sh.get_dark_sources()
    assert len(dark) == 1
    src, secs = dark[0]
    assert src == "polygon"
    assert secs >= source_health.DEFAULT_DARK_SECS


def test_never_succeeded_returns_sentinel(sh):
    """last_success NULL but actively polling -> dark_secs = -1 sentinel."""
    sh.record_result("sec_edgar", 0)
    sh.record_result("sec_edgar", 0)
    sh.record_result("sec_edgar", 0)
    dark = sh.get_dark_sources()
    assert dark == [("sec_edgar", -1)]


def test_stale_source_excluded(sh):
    """Stale (worker not polling) -> NOT reported here; get_stale_sources owns it."""
    sh.record_result("dead", 1)
    _backdate(
        sh,
        "dead",
        last_seen_age=source_health.DEFAULT_STALE_SECS + 600,
        last_success_age=source_health.DEFAULT_STALE_SECS + 600,
    )
    assert "dead" in sh.get_stale_sources()
    # MUST NOT also show up in dark_sources: the two channels are disjoint.
    assert sh.get_dark_sources() == []


def test_threshold_boundary(sh):
    sh.record_result("polygon", 1)
    # Just inside threshold -> not dark.
    _backdate(sh, "polygon", last_success_age=source_health.DEFAULT_DARK_SECS - 600)
    assert sh.get_dark_sources() == []
    # Just outside -> dark.
    _backdate(sh, "polygon", last_success_age=source_health.DEFAULT_DARK_SECS + 600)
    dark = sh.get_dark_sources()
    assert len(dark) == 1 and dark[0][0] == "polygon"


def test_custom_min_dark_secs(sh):
    sh.record_result("alphavantage", 1)
    _backdate(sh, "alphavantage", last_success_age=3600)  # 1h dark
    # Threshold 30min -> reported.
    assert sh.get_dark_sources(min_dark_secs=1800)[0][0] == "alphavantage"
    # Threshold 2h -> not yet reported.
    assert sh.get_dark_sources(min_dark_secs=7200) == []


def test_sort_order_never_first_then_by_dark_desc(sh):
    """Sentinel rows surface FIRST; aged-out rows next by darkest-first; alpha tie-break."""
    # Two never-succeeded
    sh.record_result("newsapi", 0)
    sh.record_result("nitter", 0)
    # Three aged-out with different dark ages
    for s in ("polygon", "sec_edgar", "finnhub"):
        sh.record_result(s, 1)
    _backdate(sh, "polygon", last_success_age=2 * source_health.DEFAULT_DARK_SECS)
    _backdate(sh, "sec_edgar", last_success_age=5 * source_health.DEFAULT_DARK_SECS)
    _backdate(sh, "finnhub", last_success_age=3 * source_health.DEFAULT_DARK_SECS)

    dark = sh.get_dark_sources()
    # Never-succeeded come first, alpha-sorted within their bucket.
    assert dark[0] == ("newsapi", -1)
    assert dark[1] == ("nitter", -1)
    # Aged-out: darkest (sec_edgar) -> finnhub -> polygon.
    assert dark[2][0] == "sec_edgar"
    assert dark[3][0] == "finnhub"
    assert dark[4][0] == "polygon"


def test_unparseable_last_success_treated_as_never(sh):
    sh.record_result("rss", 1)
    conn = sh._connect()
    try:
        conn.execute(
            "UPDATE source_health SET last_success = ? WHERE source = ?",
            ("not-a-timestamp", "rss"),
        )
        conn.commit()
    finally:
        conn.close()
    assert sh.get_dark_sources() == [("rss", -1)]


# ── Schema migration ─────────────────────────────────────────────────────────

def test_migration_adds_last_success_to_legacy_db(tmp_path, monkeypatch):
    """A DB created before last_success existed gets the column added by _connect,
    preserving prior rows. Validates the ALTER TABLE migration is idempotent."""
    db = tmp_path / "legacy.db"
    # Hand-create the pre-migration schema and seed a row.
    raw = sqlite3.connect(str(db))
    raw.execute(
        """CREATE TABLE source_health (
            source TEXT PRIMARY KEY,
            last_seen TEXT,
            consecutive_failures INTEGER DEFAULT 0,
            total_articles INTEGER DEFAULT 0,
            disabled INTEGER DEFAULT 0
        )"""
    )
    raw.execute(
        "INSERT INTO source_health (source, last_seen, total_articles) VALUES (?, ?, ?)",
        ("rss", "2025-01-01T00:00:00+00:00", 42),
    )
    raw.commit()
    raw.close()

    monkeypatch.setattr(source_health, "_db_path_cache", db)
    monkeypatch.setattr(source_health, "_schema_ready_path", None)

    # First connect should add the column without dropping existing data.
    report = source_health.get_health_report()
    assert "rss" in report
    assert report["rss"]["total_articles"] == 42
    assert report["rss"]["last_success"] is None

    # And the new column is now usable: a productive pass stamps it.
    source_health.record_result("rss", 1)
    report2 = source_health.get_health_report()
    assert report2["rss"]["last_success"] is not None
    assert report2["rss"]["total_articles"] == 43
