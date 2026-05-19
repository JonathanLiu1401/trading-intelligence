"""storage/ingestion_latency.py — per-source ingestion latency monitor.

These tests pin the behaviour an operator relies on: latency is computed
from real ``published``/``first_seen`` pairs only (unparseable timestamps
surface as ``skipped_no_published`` rather than silently inflating fresh
counts); negative latencies clamp to zero (clock skew is a real ingestion);
implausibly large latencies (>7d) are bucketed separately so a backfill row
does not dominate the percentile; ingestion-volume queries exclude
backtest/opus rows; and the canonical backtest-isolation clause has not
drifted from article_store's copy.

All DB tests use in-memory SQLite. No external calls.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from storage import ingestion_latency

NOW = datetime(2026, 5, 19, 21, 0, 0, tzinfo=timezone.utc)

_SCHEMA = """
CREATE TABLE articles (
    id TEXT PRIMARY KEY, url TEXT NOT NULL, title TEXT NOT NULL, source TEXT,
    published TEXT, kw_score REAL DEFAULT 0, ai_score REAL DEFAULT 0,
    urgency INTEGER DEFAULT 0, full_text BLOB, first_seen TEXT NOT NULL,
    cycle INTEGER DEFAULT 0, time_sensitivity REAL
);
"""


def _iso(seconds_ago: float = 0.0, hours_ago: float = 0.0) -> str:
    return (NOW - timedelta(seconds=seconds_ago, hours=hours_ago)).isoformat()


def _conn(rows: list[tuple[str, str, str | None, str]]) -> sqlite3.Connection:
    """rows = list of (source, url, published, first_seen_iso)."""
    c = sqlite3.connect(":memory:")
    c.executescript(_SCHEMA)
    for i, (src, url, pub, fs) in enumerate(rows):
        c.execute(
            "INSERT INTO articles (id, url, title, source, published, first_seen) "
            "VALUES (?,?,?,?,?,?)",
            (f"id{i}", url, f"title {i}", src, pub, fs),
        )
    c.commit()
    return c


# ── canonical-clause drift guard ────────────────────────────────────────────

def test_live_only_clause_matches_canonical():
    """If article_store's clause changes, this duplicate MUST be updated too."""
    from storage import article_store
    assert ingestion_latency.LIVE_ONLY_CLAUSE == article_store._LIVE_ONLY_CLAUSE


# ── parse_published format coverage ─────────────────────────────────────────

def test_parse_published_iso_and_rfc2822():
    # ISO-8601 with Z
    iso_dt = ingestion_latency.parse_published("2026-05-19T20:30:00Z")
    assert iso_dt is not None and iso_dt.tzinfo is not None
    assert iso_dt == datetime(2026, 5, 19, 20, 30, tzinfo=timezone.utc)
    # ISO-8601 with offset
    off = ingestion_latency.parse_published("2026-05-19T16:30:00-04:00")
    assert off == datetime(2026, 5, 19, 20, 30, tzinfo=timezone.utc)
    # RFC 2822 (RSS standard)
    rfc = ingestion_latency.parse_published("Tue, 19 May 2026 20:30:00 GMT")
    assert rfc == datetime(2026, 5, 19, 20, 30, tzinfo=timezone.utc)
    # Naive ISO → assumed UTC
    naive = ingestion_latency.parse_published("2026-05-19T20:30:00")
    assert naive == datetime(2026, 5, 19, 20, 30, tzinfo=timezone.utc)


def test_parse_published_handles_garbage_and_empty():
    for bad in [None, "", "   ", "not-a-date", "12345", "yesterday"]:
        assert ingestion_latency.parse_published(bad) is None


# ── compute_latency_stats — the pure contract ───────────────────────────────

def test_compute_latency_stats_basic_percentiles():
    # rss: latencies 60s, 120s, 600s, 1200s, 3600s → median 600, p90 ~2640
    rows = [
        ("rss", _iso(seconds_ago=120), _iso(seconds_ago=60)),     # 60s
        ("rss", _iso(seconds_ago=240), _iso(seconds_ago=120)),    # 120s
        ("rss", _iso(seconds_ago=1200), _iso(seconds_ago=600)),   # 600s
        ("rss", _iso(seconds_ago=2400), _iso(seconds_ago=1200)),  # 1200s
        ("rss", _iso(seconds_ago=7200), _iso(seconds_ago=3600)),  # 3600s
    ]
    stats = ingestion_latency.compute_latency_stats(rows)
    rss = stats["rss"]
    assert rss["n"] == 5
    assert rss["median_sec"] == 600.0
    # mean = (60+120+600+1200+3600)/5 = 1116
    assert rss["mean_sec"] == 1116.0
    assert rss["max_sec"] == 3600.0
    # p90 over [60,120,600,1200,3600], idx = 0.9*4 = 3.6 → 1200 + 0.6*(3600-1200) = 2640
    assert rss["p90_sec"] == 2640.0


def test_compute_latency_stats_clamps_negative_to_zero():
    # published in the future relative to first_seen (upstream clock skew).
    # Should be clamped to 0 — a real ingestion, not a parsing miss.
    rows = [
        ("nitter", _iso(seconds_ago=-30), _iso(seconds_ago=60)),   # delta = -90 → 0
        ("nitter", _iso(seconds_ago=300), _iso(seconds_ago=60)),   # 240s
    ]
    stats = ingestion_latency.compute_latency_stats(rows)
    assert stats["nitter"]["n"] == 2
    assert stats["nitter"]["max_sec"] == 240.0
    # mean = (0 + 240)/2 = 120
    assert stats["nitter"]["mean_sec"] == 120.0


def test_compute_latency_stats_skips_implausible_above_seven_days():
    eight_days = 8 * 24 * 3600
    rows = [
        ("sec_edgar", _iso(seconds_ago=eight_days), _iso(seconds_ago=60)),  # 8d-old backfill
        ("sec_edgar", _iso(seconds_ago=120), _iso(seconds_ago=60)),         # 60s fresh
    ]
    stats = ingestion_latency.compute_latency_stats(rows)
    sec = stats["sec_edgar"]
    assert sec["n"] == 1
    assert sec["max_sec"] == 60.0   # the 8d row did NOT inflate the max
    assert sec["skipped_implausible"] == 1
    assert "skipped_no_published" not in sec


def test_compute_latency_stats_skips_unparseable_published():
    rows = [
        ("reddit", "not-a-date", _iso(seconds_ago=60)),
        ("reddit", "", _iso(seconds_ago=60)),
        ("reddit", None, _iso(seconds_ago=60)),
        ("reddit", _iso(seconds_ago=120), _iso(seconds_ago=60)),  # one good sample
    ]
    stats = ingestion_latency.compute_latency_stats(rows)
    red = stats["reddit"]
    assert red["n"] == 1
    assert red["skipped_no_published"] == 3
    assert red["median_sec"] == 60.0


def test_compute_latency_stats_keeps_source_with_only_skipped_rows():
    """A source whose every row had unparseable timestamps must still appear,
    so the operator sees the metadata-coverage gap rather than silent absence.
    """
    rows = [
        ("substack", "not-a-date", _iso(seconds_ago=60)),
        ("substack", "garbage", _iso(seconds_ago=120)),
    ]
    stats = ingestion_latency.compute_latency_stats(rows)
    sub = stats["substack"]
    assert sub["n"] == 0
    assert sub["median_sec"] is None
    assert sub["mean_sec"] is None
    assert sub["p90_sec"] is None
    assert sub["max_sec"] is None
    assert sub["skipped_no_published"] == 2


def test_compute_latency_stats_null_source_becomes_question_mark():
    rows = [(None, _iso(seconds_ago=120), _iso(seconds_ago=60))]
    stats = ingestion_latency.compute_latency_stats(rows)
    assert "?" in stats
    assert stats["?"]["n"] == 1


def test_compute_latency_stats_single_sample_percentiles_collapse():
    rows = [("polygon", _iso(seconds_ago=120), _iso(seconds_ago=60))]
    stats = ingestion_latency.compute_latency_stats(rows)
    p = stats["polygon"]
    assert p["n"] == 1
    assert p["median_sec"] == p["p90_sec"] == p["mean_sec"] == p["max_sec"] == 60.0


# ── DB shell: latency_rows window + live-only filter ────────────────────────

def test_latency_rows_excludes_backtest_and_respects_window():
    c = _conn([
        ("rss", "https://a.com/1", _iso(seconds_ago=120), _iso(seconds_ago=60)),       # in
        ("web", "https://b.com/2", _iso(seconds_ago=240), _iso(seconds_ago=120)),      # in
        ("rss", "https://c.com/3", _iso(hours_ago=25), _iso(hours_ago=25)),            # OUT of 24h
        ("rss", "backtest://r7/NVDA", _iso(seconds_ago=60), _iso(seconds_ago=30)),     # excluded: url
        ("backtest_run_42_winner", "https://d.com/4", _iso(seconds_ago=60),
         _iso(seconds_ago=30)),                                                        # excluded: source
        ("opus_annotation_cycle_3", "https://e.com/5", _iso(seconds_ago=60),
         _iso(seconds_ago=30)),                                                        # excluded: source
    ])
    rows = ingestion_latency.latency_rows(c, hours=24.0, now=NOW)
    assert len(rows) == 2
    sources = {r[0] for r in rows}
    assert sources == {"rss", "web"}


# ── latency_report integration ──────────────────────────────────────────────

def test_latency_report_degrades_gracefully_without_db(tmp_path):
    rep = ingestion_latency.latency_report(
        db_path=tmp_path / "nope.db", hours=24.0, now=NOW
    )
    assert "error" in rep
    assert rep["per_source"] == {}
    assert rep["db_path"].endswith("nope.db")
    assert rep["window_hours"] == 24.0


def test_resolve_db_path_prefers_usb_then_local_no_side_effects(tmp_path, monkeypatch):
    usb = tmp_path / "usb"
    usb.mkdir()
    (usb / "articles.db").write_bytes(b"x")
    monkeypatch.setattr(ingestion_latency, "_USB_PATH", usb)
    assert ingestion_latency.resolve_db_path() == usb / "articles.db"

    monkeypatch.setattr(ingestion_latency, "_USB_PATH", tmp_path / "absent")
    monkeypatch.setattr(ingestion_latency, "_LOCAL_PATH", tmp_path / "data")
    assert ingestion_latency.resolve_db_path() == tmp_path / "data" / "articles.db"
    # strictly read-only: must NOT create the fallback dir
    assert not (tmp_path / "data").exists()
