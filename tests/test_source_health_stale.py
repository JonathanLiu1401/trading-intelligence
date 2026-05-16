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


# ── _format_source_health_line: steady-state log-bloat behaviour ─────────────
#
# A single flaky source (reddit/nitter) flapping must NOT dump the full
# ~258-item down-list on every change. The full list is reserved for the
# first observation after restart and an hourly safety-net re-dump.

from daemon import _format_source_health_line, SOURCE_HEALTH_FULL_DUMP_SECS

# Simulated "mostly permanently down" gdelt-style universe.
BIG = sorted(f"gdelt_q{i}" for i in range(258))


def test_steady_state_change_has_no_full_list():
    """(a) A changed-set warning in steady state omits `list=`."""
    prev = tuple(BIG)
    cur = sorted(BIG + ["reddit"])
    line, level, new_sig, new_dump = _format_source_health_line(
        disabled=cur, stale=[], down=cur,
        last_down_sig=prev,
        last_full_dump=1000.0,
        now=1000.0,  # full dump just happened -> not due
    )
    assert level == "warning"
    assert "list=" not in line
    assert "newly_down=['reddit']" in line
    assert new_sig == tuple(cur)
    assert new_dump == 1000.0  # untouched, no full dump emitted


def test_first_observation_emits_full_list():
    """(b) last_down_sig is None -> full list IS emitted."""
    line, level, new_sig, new_dump = _format_source_health_line(
        disabled=BIG, stale=[], down=BIG,
        last_down_sig=None,
        last_full_dump=0.0,
        now=5000.0,
    )
    assert level == "warning"
    assert f"list={BIG}" in line
    assert new_sig == tuple(BIG)
    assert new_dump == 5000.0  # full-dump timestamp reset


def test_recovered_and_newly_down_tokens():
    """(c) recovered=[...] when sources come back, newly_down=[...] when broken."""
    prev = tuple(sorted(BIG + ["reddit"]))
    cur = sorted(BIG + ["nitter"])  # reddit recovered, nitter broke
    line, level, _, _ = _format_source_health_line(
        disabled=cur, stale=[], down=cur,
        last_down_sig=prev,
        last_full_dump=9999.0,
        now=9999.0,
    )
    assert level == "warning"
    assert "newly_down=['nitter']" in line
    assert "recovered=['reddit']" in line
    assert "list=" not in line
    # Both tokens present, exactly one space between, no stray whitespace.
    assert "newly_down=['nitter'] recovered=['reddit']" in line
    assert line == line.strip()
    assert "  " not in line


def test_hourly_safety_net_redumps_on_change():
    """Changed set + >1h since last full dump -> full list re-emitted."""
    prev = tuple(BIG)
    cur = sorted(BIG + ["reddit"])
    line, level, _, new_dump = _format_source_health_line(
        disabled=cur, stale=[], down=cur,
        last_down_sig=prev,
        last_full_dump=0.0,
        now=SOURCE_HEALTH_FULL_DUMP_SECS + 1,  # safety net due
    )
    assert level == "warning"
    assert f"list={cur}" in line
    assert new_dump == SOURCE_HEALTH_FULL_DUMP_SECS + 1


def test_unchanged_within_hour_is_concise_info():
    """Unchanged set, <1h since dump -> concise INFO, no list."""
    sig = tuple(BIG)
    line, level, new_sig, new_dump = _format_source_health_line(
        disabled=BIG, stale=[], down=BIG,
        last_down_sig=sig,
        last_full_dump=100.0,
        now=200.0,
    )
    assert level == "info"
    assert "(unchanged)" in line
    assert "list=" not in line
    assert new_sig == sig
    assert new_dump == 100.0  # untouched


def test_unchanged_after_hour_redumps_as_warning():
    """Unchanged set, >1h since dump -> safety-net WARNING with full list."""
    sig = tuple(BIG)
    line, level, _, new_dump = _format_source_health_line(
        disabled=BIG, stale=[], down=BIG,
        last_down_sig=sig,
        last_full_dump=0.0,
        now=SOURCE_HEALTH_FULL_DUMP_SECS + 5,
    )
    assert level == "warning"
    assert "(unchanged)" in line
    assert f"list={BIG}" in line
    assert new_dump == SOURCE_HEALTH_FULL_DUMP_SECS + 5


# ── _connect: schema init is one-time-per-DB on the hot path ─────────────────
#
# _connect() runs on every collector pass across ~20 worker threads. The
# schema executescript + PRAGMA migration probe must run once per DB, not on
# every call, while still re-initializing when the DB path changes (tests).


def _table_exists(conn) -> bool:
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='source_health'"
    ).fetchone() is not None


def test_schema_init_runs_once_then_is_skipped(sh, monkeypatch):
    """Once the guard is set, _connect() must NOT re-run the schema script.

    Proven behaviorally: drop the table out-of-band, reconnect, and confirm
    it was *not* recreated (executescript was skipped). Resetting the guard
    makes the next connect rebuild it.
    """
    monkeypatch.setattr(source_health, "_schema_ready_path", None)

    conn = source_health._connect()  # first connect: schema created
    try:
        assert _table_exists(conn)
        assert source_health._schema_ready_path is not None
        conn.execute("DROP TABLE source_health")
        conn.commit()
    finally:
        conn.close()

    conn = source_health._connect()  # guard set -> schema NOT re-run
    try:
        assert not _table_exists(conn)
    finally:
        conn.close()

    # Clearing the guard restores one-time init on the next connect.
    monkeypatch.setattr(source_health, "_schema_ready_path", None)
    sh.record_result("rss", 3)
    assert sh.get_health_report()["rss"]["total_articles"] == 3


def test_schema_reinitializes_when_db_path_changes(tmp_path, monkeypatch):
    """A new DB path re-runs schema init even after a prior path was ready."""
    monkeypatch.setattr(source_health, "_schema_ready_path", None)

    db_a = tmp_path / "a" / "source_health.db"
    monkeypatch.setattr(source_health, "_db_path_cache", db_a)
    source_health.record_result("rss", 1)
    assert "rss" in source_health.get_health_report()

    db_b = tmp_path / "b" / "source_health.db"
    monkeypatch.setattr(source_health, "_db_path_cache", db_b)
    # Fresh DB: must have its own schema created, not skipped by the guard.
    source_health.record_result("web", 2)
    report_b = source_health.get_health_report()
    assert "web" in report_b and "rss" not in report_b


def test_no_trailing_space_when_only_newly_down():
    """Concise path with only newly_down: no stray trailing/leading space."""
    line, _, _, _ = _format_source_health_line(
        disabled=["reddit"], stale=[], down=["reddit"],
        last_down_sig=(),  # not None, not changed-from-None first-seen path
        last_full_dump=1e9,
        now=1e9,
    )
    assert "list=" not in line
    assert line == line.strip()
    assert "  " not in line
