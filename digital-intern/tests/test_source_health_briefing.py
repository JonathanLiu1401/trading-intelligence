"""Source-health blind-spot visibility in the 5h heartbeat briefing.

Before this, the briefing health line only reported four worker threads'
liveness and never source_health's disabled/stale set, so a disabled
SEC-EDGAR / wire collector left the digest silently missing that source's
news with no signal to the analyst (6 collectors incl. sec_edgar were
observed disabled in production while the briefing said nothing).

``daemon._format_source_health_summary`` is pure and deterministic — exact
strings pinned here — and ``_build_health_line`` must append it only when
something is actually down (a healthy briefing stays clean).
"""
from __future__ import annotations

import daemon


def test_empty_when_all_healthy():
    assert daemon._format_source_health_summary([], []) == ""


def test_disabled_sorted():
    assert (
        daemon._format_source_health_summary(["sec_edgar", "polygon"], [])
        == "⚠ Sources down (2): polygon, sec_edgar"
    )


def test_dedups_stale_against_disabled():
    # 'polygon' is both disabled and stale — counted once, as disabled.
    assert (
        daemon._format_source_health_summary(["polygon"], ["polygon", "rss"])
        == "⚠ Sources down (2): polygon, rss"
    )


def test_lists_disabled_before_stale():
    # The union is NOT globally sorted: disabled (harder failure) come first
    # even though 'alpha' < 'zeta' alphabetically.
    assert (
        daemon._format_source_health_summary(["zeta"], ["alpha"])
        == "⚠ Sources down (2): zeta, alpha"
    )


def test_truncates_with_overflow_marker():
    disabled = ["massive", "newsapi", "nitter", "polygon",
                "sec_edgar", "sec_edgar_ft"]
    assert (
        daemon._format_source_health_summary(disabled, [])
        == "⚠ Sources down (6): massive, newsapi, nitter, polygon, +2"
    )


def test_hard_char_cap():
    long_names = [f"collector_with_a_very_long_name_{i}" for i in range(8)]
    out = daemon._format_source_health_summary(long_names, [], max_chars=60)
    assert len(out) <= 60
    assert out.endswith("…")
    assert out.startswith("⚠ Sources down (8): ")


def test_build_health_line_appends_source_health(store, monkeypatch):
    monkeypatch.setattr(
        daemon.source_health, "get_disabled_sources",
        lambda: ["sec_edgar", "polygon"],
    )
    monkeypatch.setattr(
        daemon.source_health, "get_stale_sources", lambda: [],
    )
    line = daemon._build_health_line(store)
    assert line.startswith("⚙ Workers: ")
    assert "\n⚠ Sources down (2): polygon, sec_edgar" in line


def test_build_health_line_clean_when_all_sources_healthy(store, monkeypatch):
    monkeypatch.setattr(
        daemon.source_health, "get_disabled_sources", lambda: [])
    monkeypatch.setattr(
        daemon.source_health, "get_stale_sources", lambda: [])
    line = daemon._build_health_line(store)
    assert "Sources down" not in line
    assert "\n" not in line


def test_build_health_line_survives_source_health_error(store, monkeypatch):
    """A source_health probe failure must not break the briefing — the line
    degrades to workers-only, never raises."""
    def _boom():
        raise RuntimeError("health db locked")

    monkeypatch.setattr(daemon.source_health, "get_disabled_sources", _boom)
    monkeypatch.setattr(daemon.source_health, "get_stale_sources", _boom)
    line = daemon._build_health_line(store)
    assert line.startswith("⚙ Workers: ")
    assert "Sources down" not in line
