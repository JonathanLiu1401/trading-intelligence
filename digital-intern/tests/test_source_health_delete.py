"""delete_sources(prefix): purge legacy/high-cardinality source_health rows.

Per-query ``gdelt:<query>`` keys were recorded under cross-query dedup, so
almost every key tripped the 3-failure disable threshold while GDELT itself
was healthy — burying the few genuinely-down sources in the hourly
[source_health] alert. The collector now records a single aggregate
``gdelt`` key; this helper lets purge_worker sweep away the stale per-query
rows so the alert reflects reality.
"""
from __future__ import annotations

import pytest

from collectors import source_health


@pytest.fixture
def sh(tmp_path, monkeypatch):
    """Point source_health at an isolated DB and clear its path cache."""
    db = tmp_path / "source_health.db"
    monkeypatch.setattr(source_health, "_db_path_cache", db)
    monkeypatch.setattr(source_health, "_schema_ready_path", None)
    return source_health


def test_delete_sources_removes_only_matching_prefix(sh):
    sh.record_result("gdelt", 4)
    sh.record_result("gdelt:DRAM memory", 0)
    sh.record_result("gdelt:NAND flash", 0)
    sh.record_result("rss", 3)

    removed = sh.delete_sources("gdelt:")

    assert removed == 2
    report = sh.get_health_report()
    assert set(report) == {"gdelt", "rss"}


def test_delete_sources_is_idempotent(sh):
    sh.record_result("gdelt:foo", 0)
    assert sh.delete_sources("gdelt:") == 1
    # Second call has nothing left to remove and must not raise.
    assert sh.delete_sources("gdelt:") == 0


def test_delete_sources_empty_prefix_is_noop(sh):
    sh.record_result("rss", 1)
    assert sh.delete_sources("") == 0
    assert set(sh.get_health_report()) == {"rss"}


def test_aggregate_gdelt_key_accumulates_across_sweeps(sh):
    # Mirrors the new gdelt_worker contract: one key, cumulative totals,
    # never disabled while the sweep yields articles.
    for _ in range(3):
        sh.record_result("gdelt", 5)

    report = sh.get_health_report()
    assert set(report) == {"gdelt"}
    assert report["gdelt"]["total_articles"] == 15
    assert report["gdelt"]["disabled"] is False
    assert sh.get_disabled_sources() == []
