"""Unit tests for ml.llm_promotion_audit.

Covers the pure ``compute_promotion_stats`` rollup contract and the
``load_rows`` SQL boundary — particularly the live-only filter that keeps
backtest/opus_annotation synthetic rows OUT of this read-side diagnostic.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ml import llm_promotion_audit as lpa


# ── compute_promotion_stats (pure) ───────────────────────────────────────────


def test_empty_returns_zero_skeleton() -> None:
    r = lpa.compute_promotion_stats([])
    assert r["n"] == 0
    assert r["n_promoted"] == 0
    assert r["promotion_rate_pct"] == 0.0
    assert r["n_alerted"] == 0
    assert r["overall_alert_yield_pct"] == 0.0
    assert r["by_source"] == []


def _row(source: str, score_source: str | None, ai: float, urg: int) -> dict:
    return {
        "source": source,
        "score_source": score_source,
        "ai_score": ai,
        "urgency": urg,
        "first_seen": "2026-05-18T12:00:00Z",
    }


def test_promotion_rate_and_alert_yield_per_source() -> None:
    # rss: 10 total, 4 promoted, 2 of those alerted
    rows = (
        [_row("rss", "llm", 7.5, 1)] * 1  # promoted + alerted
        + [_row("rss", "llm", 8.0, 2)] * 1  # promoted + alerted (urgency=2 counts)
        + [_row("rss", "llm", 4.0, 0)] * 2  # promoted, not alerted
        + [_row("rss", None, 0.0, 0)] * 6  # not promoted
        # web: 6 total, 6 promoted, 0 alerted — full LLM spend but no alerts
        + [_row("web", "llm", 3.0, 0)] * 6
    )

    r = lpa.compute_promotion_stats(rows)

    assert r["n"] == 16
    assert r["n_promoted"] == 10
    assert r["promotion_rate_pct"] == round(100.0 * 10 / 16, 2)
    # Two rows have urgency >= 1.
    assert r["n_alerted"] == 2
    # Overall yield = alerted_promoted (2) / total_promoted (10) = 20.0%
    assert r["overall_alert_yield_pct"] == 20.0

    by = {b["source"]: b for b in r["by_source"]}
    assert set(by) == {"rss", "web"}

    assert by["rss"]["total"] == 10
    assert by["rss"]["promoted"] == 4
    assert by["rss"]["promotion_rate_pct"] == 40.0
    # Mean ai over the 4 promoted rss rows = (7.5 + 8.0 + 4.0 + 4.0) / 4 = 5.875
    assert by["rss"]["mean_ai_on_promoted"] == 5.875
    # 2 of 4 promoted rss rows are alerted -> 50%
    assert by["rss"]["alert_yield_pct"] == 50.0

    assert by["web"]["promoted"] == 6
    assert by["web"]["promotion_rate_pct"] == 100.0
    # All web rows have ai=3.0
    assert by["web"]["mean_ai_on_promoted"] == 3.0
    # No web row is alerted -> 0%
    assert by["web"]["alert_yield_pct"] == 0.0


def test_small_sources_aggregate_into_other_bucket() -> None:
    # rss has 5 rows (the min threshold) and stays. Three tiny sources with 1
    # row each collapse into _other so a singleton's 100% rate is not noise.
    rows = (
        [_row("rss", "llm", 6.0, 1)] * 5
        + [_row("nitter", "llm", 9.0, 1)]
        + [_row("substack", None, 0.0, 0)]
        + [_row("massive", "llm", 2.0, 0)]
    )
    r = lpa.compute_promotion_stats(rows)
    by = {b["source"]: b for b in r["by_source"]}
    assert "rss" in by
    assert "_other" in by
    assert by["_other"]["total"] == 3
    assert by["_other"]["promoted"] == 2  # nitter + massive
    # Of the 2 promoted in _other, only nitter (urg=1) was alerted -> 50%
    assert by["_other"]["alert_yield_pct"] == 50.0
    # The tiny sources do NOT appear as their own buckets.
    assert "nitter" not in by
    assert "substack" not in by
    assert "massive" not in by


def test_non_llm_score_source_is_not_a_promotion() -> None:
    # 'briefing_boost' is a curation nudge for the heartbeat, not a Sonnet
    # grade — only score_source='llm' counts as LLM-promoted spend.
    rows = [
        _row("rss", "briefing_boost", 6.0, 0),
        _row("rss", "ml", 4.0, 0),
        _row("rss", None, 0.0, 0),
        _row("rss", "llm", 8.0, 1),
        _row("rss", "llm", 7.0, 0),
    ]
    r = lpa.compute_promotion_stats(rows)
    assert r["n"] == 5
    assert r["n_promoted"] == 2
    assert r["promotion_rate_pct"] == 40.0
    by = {b["source"]: b for b in r["by_source"]}
    assert by["rss"]["promoted"] == 2
    # 1 of 2 promoted is alerted -> 50%
    assert by["rss"]["alert_yield_pct"] == 50.0


def test_blank_or_missing_source_falls_into_unknown() -> None:
    rows = [_row("", "llm", 6.0, 0) for _ in range(5)] + [
        {"source": None, "score_source": "llm", "ai_score": 5.0, "urgency": 0}
        for _ in range(5)
    ]
    r = lpa.compute_promotion_stats(rows)
    by = {b["source"]: b for b in r["by_source"]}
    # Both blank and None source coalesce to _unknown; with 10 rows together
    # they exceed the per-source threshold and stay as one bucket.
    assert "_unknown" in by
    assert by["_unknown"]["total"] == 10
    assert by["_unknown"]["promoted"] == 10


def test_urgency_2_counts_as_alerted() -> None:
    # alert_worker advances 1 -> 2 asynchronously; the spend-to-outcome
    # question we want is "did the LLM-promoted row reach an alertable
    # state?" — urgency >= 1 is the right boundary.
    rows = [
        _row("rss", "llm", 7.0, 0),
        _row("rss", "llm", 7.0, 1),
        _row("rss", "llm", 7.0, 2),
        _row("rss", "llm", 7.0, 0),
        _row("rss", "llm", 7.0, 0),
    ]
    r = lpa.compute_promotion_stats(rows)
    by = {b["source"]: b for b in r["by_source"]}
    # 2 of 5 promoted (urg=1 and urg=2) -> 40%
    assert by["rss"]["alert_yield_pct"] == 40.0
    assert r["n_alerted"] == 2


def test_by_source_is_sorted_by_promoted_count_desc() -> None:
    rows = (
        [_row("a", "llm", 5.0, 0)] * 2  # 2 promoted of 5
        + [_row("a", None, 0.0, 0)] * 3
        + [_row("b", "llm", 5.0, 0)] * 6  # 6 promoted of 6
        + [_row("c", "llm", 5.0, 0)] * 4  # 4 promoted of 5
        + [_row("c", None, 0.0, 0)] * 1
    )
    r = lpa.compute_promotion_stats(rows)
    order = [b["source"] for b in r["by_source"]]
    # Ranked by promoted DESC: b(6), c(4), a(2)
    assert order == ["b", "c", "a"]


# ── load_rows (SQL boundary, including the live-only filter) ─────────────────


def _make_test_db(tmp_path: Path) -> Path:
    """Build a minimal articles-table fixture matching the production shape."""
    db = tmp_path / "articles.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE articles (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT,
            published TEXT,
            kw_score REAL DEFAULT 0,
            ai_score REAL DEFAULT 0,
            ml_score REAL,
            score_source TEXT,
            urgency INTEGER DEFAULT 0,
            full_text BLOB,
            first_seen TEXT NOT NULL,
            cycle INTEGER DEFAULT 0,
            time_sensitivity REAL
        )
        """
    )
    rows = [
        # 3 live rows
        ("a1", "https://x/a1", "live 1", "rss",       "llm",  7.0, 1,
         "datetime('now','-30 minutes')"),
        ("a2", "https://x/a2", "live 2", "rss",       None,    0.0, 0,
         "datetime('now','-2 hours')"),
        ("a3", "https://x/a3", "live 3", "web",       "llm",  6.0, 0,
         "datetime('now','-3 hours')"),
        # 1 backtest URL — must be filtered out
        ("b1", "backtest://run_1/2024-01-01/BUY/AAPL", "bt", "rss",
         None, 5.0, 0, "datetime('now','-1 hour')"),
        # 1 backtest source tag — must be filtered out
        ("b2", "https://x/b2", "bt-src", "backtest_run_42_winner",
         None, 5.0, 0, "datetime('now','-1 hour')"),
        # 1 opus_annotation source tag — must be filtered out
        ("b3", "https://x/b3", "opus", "opus_annotation_cycle_7",
         None, 5.0, 0, "datetime('now','-1 hour')"),
        # 1 OLD live row — outside the 24h window
        ("o1", "https://x/o1", "old", "rss", "llm", 8.0, 1,
         "datetime('now','-48 hours')"),
    ]
    for rid, url, title, src, ss, ai, urg, ts_expr in rows:
        conn.execute(
            f"INSERT INTO articles (id, url, title, source, score_source, "
            f"ai_score, urgency, first_seen) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, {ts_expr})",
            (rid, url, title, src, ss, ai, urg),
        )
    conn.commit()
    conn.close()
    return db


def test_load_rows_excludes_backtest_and_old(tmp_path: Path) -> None:
    db = _make_test_db(tmp_path)
    rows = lpa.load_rows(db, hours=24)
    # 3 live rows in window; 3 synthetic + 1 old must be excluded.
    assert len(rows) == 3
    sources = {r["source"] for r in rows}
    assert sources == {"rss", "web"}
    # Specifically confirm no synthetic source leaked.
    assert not any(s.startswith("backtest_") for s in sources)
    assert not any(s.startswith("opus_annotation") for s in sources)


def test_load_rows_respects_hours_window(tmp_path: Path) -> None:
    db = _make_test_db(tmp_path)
    # A 1h window must drop the rows at -2h and -3h.
    rows = lpa.load_rows(db, hours=1)
    # The -1h rows include a1 plus the synthetic b1/b2/b3 which are filtered
    # by the live-only clause regardless of the time bound.
    sources_seen = {r["source"] for r in rows}
    assert sources_seen == {"rss"}
    assert len(rows) == 1


def test_run_writes_json_and_returns_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_test_db(tmp_path)
    monkeypatch.setattr(lpa, "_db_path", lambda: db)
    out_path = tmp_path / "report.json"
    monkeypatch.setattr(lpa, "OUTPUT_PATH", out_path)
    report = lpa.run(write=True, hours=24)
    assert report["n"] == 3
    assert report["n_promoted"] == 2
    assert report["window_hours"] == 24
    assert "generated_at" in report
    # Persistence path went through the atomic .tmp -> replace dance.
    assert out_path.exists()
    import json
    persisted = json.loads(out_path.read_text())
    assert persisted["n"] == 3
    assert persisted["n_promoted"] == 2
