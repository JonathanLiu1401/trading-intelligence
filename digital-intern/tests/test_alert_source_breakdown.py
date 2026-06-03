"""Pin the per-source alert-funnel breakdown contract.

Calibration parity: the ``by_source`` map and ``llm_fraction`` semantics
must match ``ArticleStore.urgency_label_split`` byte-for-byte so the two
audits never disagree on what a "vetted" alert is. Backtest isolation:
synthetic rows must never reach the report. Sort order: alerted desc,
source asc — stable for downstream consumers that key on the first row.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from analytics import alert_source_breakdown as mod
from storage.article_store import SCHEMA as _SCHEMA


# ── Pure compute_breakdown contract ─────────────────────────────────────


def test_compute_breakdown_empty_returns_empty():
    assert mod.compute_breakdown([]) == []


def test_compute_breakdown_counts_by_source_and_score_source():
    rows = [
        ("rss", "llm"),
        ("rss", "ml"),
        ("rss", "ml"),
        ("rss", "briefing_boost"),
        ("rss", None),  # legacy untagged
        ("finnhub", "llm"),
        ("finnhub", "llm"),
        ("finnhub", "ml"),
    ]
    out = mod.compute_breakdown(rows)
    by_src = {r["source"]: r for r in out}

    rss = by_src["rss"]
    assert rss["alerted"] == 5
    assert rss["by_source"] == {
        "llm": 1, "ml": 2, "briefing_boost": 1, "null": 1
    }
    # llm_fraction = (llm + briefing_boost) / alerted = 2/5 = 0.4
    assert rss["llm_fraction"] == 0.4

    finnhub = by_src["finnhub"]
    assert finnhub["alerted"] == 3
    # 2 llm of 3 = 0.6667
    assert finnhub["llm_fraction"] == 0.6667


def test_compute_breakdown_unknown_score_source_buckets_to_null():
    """Any score_source not in {llm, ml, briefing_boost} (e.g. a stray
    legacy tag or future value) MUST bucket into ``null`` — same fallback
    discipline as urgency_label_split, so a forward-compat addition can't
    silently inflate llm_fraction."""
    rows = [("rss", "experimental_tag_xyz"), ("rss", "llm")]
    out = mod.compute_breakdown(rows)
    assert out[0]["by_source"] == {
        "llm": 1, "ml": 0, "briefing_boost": 0, "null": 1
    }
    assert out[0]["llm_fraction"] == 0.5


def test_compute_breakdown_sorts_by_alerted_desc():
    rows = (
        [("a", "llm")]
        + [("b", "llm")] * 5
        + [("c", "ml")] * 3
    )
    out = mod.compute_breakdown(rows)
    assert [r["source"] for r in out] == ["b", "c", "a"]
    assert [r["alerted"] for r in out] == [5, 3, 1]


def test_compute_breakdown_tie_break_alphabetical():
    rows = [("zoo", "llm"), ("ape", "llm"), ("mid", "llm")]
    out = mod.compute_breakdown(rows)
    # All tied at alerted=1; secondary sort is source ascending.
    assert [r["source"] for r in out] == ["ape", "mid", "zoo"]


def test_compute_breakdown_empty_source_becomes_unknown():
    rows = [("", "llm"), (None, "ml")]
    out = mod.compute_breakdown(rows)
    # Both empty-source rows collapse into 'unknown'.
    assert len(out) == 1
    assert out[0]["source"] == "unknown"
    assert out[0]["alerted"] == 2


def test_compute_breakdown_min_per_source_drops_below_floor():
    rows = [("rss", "llm")] * 3 + [("tiny", "llm")] * 1
    out = mod.compute_breakdown(rows, min_per_source=2)
    sources = {r["source"] for r in out}
    assert sources == {"rss"}, sources


# ── build_report aggregate parity with urgency_label_split ──────────────


def test_build_report_aggregate_llm_fraction_matches_per_source_sum():
    breakdown = [
        {"source": "rss", "alerted": 4,
         "by_source": {"llm": 2, "ml": 1, "briefing_boost": 1, "null": 0},
         "llm_fraction": 0.75},
        {"source": "finnhub", "alerted": 2,
         "by_source": {"llm": 0, "ml": 2, "briefing_boost": 0, "null": 0},
         "llm_fraction": 0.0},
    ]
    report = mod.build_report(breakdown, hours=24)
    assert report["total_alerted"] == 6
    # vetted = 2 (rss llm) + 1 (rss boost) + 0 = 3 of 6 → 0.5
    assert report["aggregate_llm_fraction"] == 0.5
    assert report["window_hours"] == 24


def test_build_report_zero_total_does_not_div_zero():
    report = mod.build_report(breakdown=[], hours=6)
    assert report["total_alerted"] == 0
    assert report["aggregate_llm_fraction"] == 0.0


# ── load_alerted_rows SQL contract: backtest + urgency + window ─────────


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    db = tmp_path / "articles.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_SCHEMA)
    now = datetime.now(timezone.utc)
    fresh = (now - timedelta(minutes=30)).isoformat()
    stale = (now - timedelta(hours=48)).isoformat()
    rows = [
        # (id, url, title, source, urgency, first_seen, score_source)
        # Real alerted live row — must appear.
        ("live-1", "https://wire/1", "Live alerted", "rss",
         2, fresh, "llm"),
        # urgency=1 (queued) — must NOT appear (only urgency=2 = fired).
        ("queued-1", "https://wire/2", "Queued", "rss",
         1, fresh, "llm"),
        # urgency=0 (normal) — must NOT appear.
        ("normal-1", "https://wire/3", "Normal", "rss",
         0, fresh, "ml"),
        # Stale alerted row — must NOT appear (outside 24h window).
        ("stale-1", "https://wire/4", "Stale alert", "rss",
         2, stale, "llm"),
        # Backtest URL synthetic — urgency=2 + fresh; must NOT appear
        # (live-only clause).
        ("bt-url", "backtest://run_1/foo", "Synthetic", "rss",
         2, fresh, "llm"),
        # backtest_* source synthetic — must NOT appear.
        ("bt-src", "https://internal/foo", "Synthetic 2", "backtest_winner",
         2, fresh, "llm"),
        # opus_annotation* synthetic — must NOT appear.
        ("op-src", "https://internal/op", "Synthetic 3",
         "opus_annotation_cycle_3", 2, fresh, "llm"),
    ]
    conn.executemany(
        "INSERT INTO articles (id, url, title, source, urgency, "
        "first_seen, score_source) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db


def test_load_alerted_rows_excludes_non_fired_and_synthetic(seeded_db: Path):
    rows = mod.load_alerted_rows(seeded_db, hours=24)
    # Only the one fresh urgency=2 live row passes every filter.
    assert rows == [("rss", "llm")], rows


def test_run_end_to_end_excludes_synthetic_from_report(
    seeded_db: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(mod, "OUT", tmp_path / "alert_breakdown.json")
    report = mod.run(db_path=seeded_db, hours=24, write=True)
    assert report["total_alerted"] == 1
    assert len(report["sources"]) == 1
    s = report["sources"][0]
    assert s["source"] == "rss"
    assert s["alerted"] == 1
    assert s["by_source"]["llm"] == 1
    assert s["llm_fraction"] == 1.0
    # Verify no synthetic markers leaked into the persisted JSON either.
    text = (tmp_path / "alert_breakdown.json").read_text()
    assert "backtest://" not in text
    assert "backtest_" not in text
    assert "opus_annotation" not in text


def test_calibration_parity_with_urgency_label_split(
    seeded_db: Path, tmp_path: Path, monkeypatch
):
    """Cross-product anti-drift parity: this module counts urgency==2
    (alert ACTUALLY fired), ``urgency_label_split`` counts urgency>=1
    (queued + fired). The two read the SAME table with the SAME backtest
    filter, so summing this module's by_source across all sources must
    equal split.by_source MINUS the queued-only (urgency=1) contribution.
    The seeded DB has exactly one fresh urgency=1 row ("queued-1", score
    source "llm") and one fresh urgency=2 row — so split.by_source must
    equal summed-breakdown + {llm: 1}, the same shape urgency_label_split's
    own tests assert. If a future change drifts the calibration keys this
    parity check fails first."""
    from storage import article_store

    monkeypatch.setattr(article_store, "_get_db_path", lambda: seeded_db)
    store = article_store.ArticleStore()
    try:
        split = store.urgency_label_split(hours=24)
    finally:
        store.close()
    breakdown_report = mod.run(db_path=seeded_db, hours=24, write=False)

    # Sum per-source breakdown across all sources.
    summed = {"llm": 0, "ml": 0, "briefing_boost": 0, "null": 0}
    for r in breakdown_report["sources"]:
        for k, v in r["by_source"].items():
            summed[k] += v
    # The fired (urgency=2) total in this module equals the fired part
    # of the split's total: split.total == summed_breakdown_total + 1
    # (the one queued-1 urgency=1 row).
    assert (
        split["total"] == breakdown_report["total_alerted"] + 1
    ), (
        f"split.total={split['total']} breakdown.total_alerted="
        f"{breakdown_report['total_alerted']} — expected to differ by exactly "
        "the one queued urgency=1 row in the fixture"
    )
    # By_source parity: split = breakdown + the queued llm row.
    expected = dict(summed)
    expected["llm"] += 1
    assert split["by_source"] == expected, (
        f"urgency_label_split.by_source={split['by_source']} "
        f"breakdown summed={summed} (expected split=breakdown + the queued "
        "urgency=1 row that this module deliberately excludes)"
    )
