"""Tests for analytics.publish_lag_audit.

The audit is read-only and operates against a tmp SQLite file with the
minimum ``articles`` columns it actually selects on. We do not stand up a
full ``ArticleStore`` — its migrations are irrelevant here and would only
slow the test.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from analytics import publish_lag_audit


# ── helpers ─────────────────────────────────────────────────────────────

_MIN_COLS = """
CREATE TABLE articles (
    id          TEXT PRIMARY KEY,
    url         TEXT NOT NULL,
    title       TEXT,
    source      TEXT,
    published   TEXT,
    first_seen  TEXT
)
"""


def _build_db(path: Path, rows: list[tuple]) -> None:
    """rows: (id, url, source, published, first_seen)."""
    conn = sqlite3.connect(str(path))
    conn.execute(_MIN_COLS)
    conn.executemany(
        "INSERT INTO articles (id, url, title, source, published, first_seen) "
        "VALUES (?,?,?,?,?,?)",
        [(rid, url, "t", src, pub, fs) for (rid, url, src, pub, fs) in rows],
    )
    conn.commit()
    conn.close()


@pytest.fixture
def patched_db(tmp_path, monkeypatch):
    """Return ``(db_path, build)`` where ``build(rows)`` creates the DB."""
    db_path = tmp_path / "lag.db"

    def build(rows):
        _build_db(db_path, rows)
        monkeypatch.setattr(publish_lag_audit, "_get_db_path", lambda: db_path)
        # SNAPSHOT_PATH defaults to a real operator dir; redirect it.
        monkeypatch.setattr(
            publish_lag_audit, "SNAPSHOT_PATH", tmp_path / "snap.json"
        )

    return db_path, build


# ── tests ───────────────────────────────────────────────────────────────


def test_empty_db_yields_zero_collectors(patched_db):
    db, build = patched_db
    build([])
    report = publish_lag_audit.compute()
    assert report["scanned"] == 0
    assert report["rows_with_parseable_lag"] == 0
    assert report["collectors"] == {}
    assert report["ranked_freshest"] == []
    assert report["ranked_stalest"] == []


def test_lag_summary_is_correct_for_known_inputs(patched_db):
    """One collector with 5 controlled lag values (0, 1, 5, 30, 90 min):

      * median = 5 min
      * p90 (linear-interpolated) = 30 + (90-30)*0.6 = 66 min
      * fresh (<5m) = 2/5 = 40%
      * stale (>60m) = 1/5 = 20%
    """
    db, build = patched_db
    base = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    lags_min = [0, 1, 5, 30, 90]
    rows = []
    for i, lm in enumerate(lags_min):
        pub = base.isoformat()
        fs = (base + timedelta(minutes=lm)).isoformat()
        rows.append((f"id{i}", f"https://x/{i}", "rss", pub, fs))
    build(rows)

    report = publish_lag_audit.compute()
    coll = report["collectors"]["rss"]
    assert coll["n"] == 5
    assert coll["median_lag_min"] == pytest.approx(5.0)
    assert coll["p90_lag_min"] == pytest.approx(66.0)
    assert coll["mean_lag_min"] == pytest.approx(sum(lags_min) / 5.0)
    assert coll["fresh_5m_pct"] == pytest.approx(40.0)
    assert coll["stale_60m_pct"] == pytest.approx(20.0)


def test_subsource_collapses_into_collector_family(patched_db):
    """gdelt_gkg/iheart.com and gdelt_gkg/reuters.com both roll up to
    ``gdelt_gkg`` — matches stale_source_alerter's granularity."""
    db, build = patched_db
    base = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i, sub in enumerate(["a", "b", "c", "d", "e"]):
        pub = base.isoformat()
        fs = (base + timedelta(minutes=10)).isoformat()
        rows.append((f"id{i}", f"https://x/{i}", f"gdelt_gkg/{sub}.com", pub, fs))
    build(rows)
    report = publish_lag_audit.compute()
    assert "gdelt_gkg" in report["collectors"]
    assert report["collectors"]["gdelt_gkg"]["n"] == 5


def test_synthetic_rows_excluded(patched_db):
    """Backtest/opus_annotation rows are training-only and must not skew
    the freshness picture — they should never enter the audit."""
    db, build = patched_db
    base = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    pub = base.isoformat()
    fs = (base + timedelta(minutes=10)).isoformat()
    rows = []
    # 5 live rows (qualifies for reporting).
    for i in range(5):
        rows.append((f"live{i}", f"https://x/{i}", "rss", pub, fs))
    # Synthetic — both URL and source variants — should be filtered.
    rows.append(("bt1", "backtest://run_1/foo", "backtest_run_1_winner", pub, fs))
    rows.append(("bt2", "https://x/y", "backtest_run_2_rank1", pub, fs))
    rows.append(("op1", "https://x/z", "opus_annotation_cycle_42", pub, fs))
    build(rows)

    report = publish_lag_audit.compute()
    assert set(report["collectors"].keys()) == {"rss"}
    assert report["collectors"]["rss"]["n"] == 5
    # Only the 5 live rows were scanned at all.
    assert report["scanned"] == 5


def test_clock_skew_and_unparseable_rejected(patched_db):
    """Rows with absurd lag (more than 1h in the future) or unparseable
    ``published`` must NOT contribute to the median/p90."""
    db, build = patched_db
    base = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    pub = base.isoformat()
    fs_ok = (base + timedelta(minutes=10)).isoformat()

    rows = []
    # 5 valid rows at 10-min lag.
    for i in range(5):
        rows.append((f"ok{i}", f"https://x/{i}", "rss", pub, fs_ok))
    # 3-hour skew into the future — rejected.
    fs_future = (base - timedelta(hours=3)).isoformat()
    rows.append(("skew", "https://x/skew", "rss", pub, fs_future))
    # Garbage `published` — rejected by _parse_published.
    rows.append(("junk", "https://x/junk", "rss", "not-a-date", fs_ok))
    build(rows)

    report = publish_lag_audit.compute()
    assert report["collectors"]["rss"]["n"] == 5
    assert report["collectors"]["rss"]["median_lag_min"] == pytest.approx(10.0)


def test_min_per_collector_threshold(patched_db):
    """A collector below MIN_PER_COLLECTOR (default 5) must be omitted from
    the report — its median would be too noisy to act on."""
    db, build = patched_db
    base = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    pub = base.isoformat()
    fs = (base + timedelta(minutes=5)).isoformat()
    rows = []
    # 5 'rss' (reported)
    for i in range(5):
        rows.append((f"r{i}", f"https://x/r{i}", "rss", pub, fs))
    # 2 'reddit' (below threshold, dropped)
    for i in range(2):
        rows.append((f"d{i}", f"https://x/d{i}", "reddit", pub, fs))
    build(rows)
    report = publish_lag_audit.compute()
    assert "rss" in report["collectors"]
    assert "reddit" not in report["collectors"]


def test_ranked_lists_are_sorted_by_median(patched_db):
    """ranked_freshest should be ascending median, ranked_stalest its
    reverse — they're the headline ranking the operator reads."""
    db, build = patched_db
    base = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    rows = []
    # collector A: 1-min lag x5  (freshest)
    for i in range(5):
        pub = base.isoformat()
        fs = (base + timedelta(minutes=1)).isoformat()
        rows.append((f"a{i}", f"https://a/{i}", "fastfeed", pub, fs))
    # collector B: 30-min lag x5
    for i in range(5):
        pub = base.isoformat()
        fs = (base + timedelta(minutes=30)).isoformat()
        rows.append((f"b{i}", f"https://b/{i}", "midfeed", pub, fs))
    # collector C: 120-min lag x5  (stalest)
    for i in range(5):
        pub = base.isoformat()
        fs = (base + timedelta(minutes=120)).isoformat()
        rows.append((f"c{i}", f"https://c/{i}", "slowfeed", pub, fs))
    build(rows)
    report = publish_lag_audit.compute()
    fresh_order = [e["collector"] for e in report["ranked_freshest"]]
    assert fresh_order == ["fastfeed", "midfeed", "slowfeed"]
    stale_order = [e["collector"] for e in report["ranked_stalest"]]
    assert stale_order == ["slowfeed", "midfeed", "fastfeed"]


def test_write_snapshot_round_trips(patched_db):
    db, build = patched_db
    base = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    pub = base.isoformat()
    fs = (base + timedelta(minutes=2)).isoformat()
    rows = [(f"x{i}", f"https://x/{i}", "rss", pub, fs) for i in range(5)]
    build(rows)
    report = publish_lag_audit.compute()
    out = publish_lag_audit.write_snapshot(report)
    payload = json.loads(out.read_text())
    assert payload["collectors"]["rss"]["n"] == 5
    assert "generated_at" in payload
