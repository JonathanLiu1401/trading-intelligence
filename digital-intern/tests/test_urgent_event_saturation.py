"""Tests for analytics.urgent_event_saturation — per-(held-ticker × event-
class) URGENT-QUEUE saturation audit over articles.db.

Queued-side sibling of pushed_alert_event_concentration. The pure builder
``build_saturation_report`` is the unit-tested contract; the live failure
shape that motivated it is the NVDA × BUYBACK cluster (live evidence
2026-05-25, articles.db 24h window: 14+ urgent rows about the same NVDA
$80B buyback event across multiple syndication channels).

The four load-bearing invariants are pinned here explicitly — same
discipline as test_pushed_alert_event_concentration and test_label_audit.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from analytics.urgent_event_saturation import (
    build_saturation_report,
    _load_urgent_rows,
    _LIVE_ONLY_CLAUSE,
    HEAVY_THRESHOLD,
    SATURATION_THRESHOLD,
)
from analytics import urgent_event_saturation as ues


HELD = {"NVDA", "MU", "MSFT", "AXTI", "LITE", "ORCL"}


# ── build_saturation_report: pure-builder contract ───────────────────────────


def test_empty_input_returns_no_data_verdict():
    """Zero-data input degrades to NO_DATA — refuses to make any claim
    about saturation when there is no input to measure. Same discipline
    as briefing_health's NO_DATA branch and label_production_rate's
    NO_DATA / DARK separation."""
    report = build_saturation_report([], HELD, window_h=24.0)
    assert report["verdict"] == "NO_DATA"
    assert report["total_urgent"] == 0
    assert report["distinct_pairs"] == 0
    assert report["by_pair"] == []
    assert report["saturation_alerts"] == []


def test_healthy_when_no_pair_meets_threshold():
    """Below-threshold concentration → HEALTHY. One urgent NVDA buyback
    row alone is not saturation; the analyst persona needs at least
    SATURATION_THRESHOLD same-event signals."""
    rows = [
        {"title": "Nvidia unveils $80B buyback plan", "age_hours": 1.0,
         "urgency": 2, "score_source": "llm"},
        {"title": "MU lowers full-year guidance after Q3", "age_hours": 2.0,
         "urgency": 2, "score_source": "llm"},
    ]
    report = build_saturation_report(rows, HELD)
    assert report["verdict"] == "HEALTHY"
    assert report["urgent_held_x_class"] == 2
    # 2 distinct pairs (NVDA,BUYBACK) and (MU,GUIDANCE), each count=1
    pairs = {(p["ticker"], p["event_class"], p["urgent_count"])
             for p in report["by_pair"]}
    assert ("NVDA", "BUYBACK", 1) in pairs
    assert ("MU", "GUIDANCE", 1) in pairs


def test_watch_when_pair_at_saturation_threshold():
    """Exactly SATURATION_THRESHOLD same-event rows → WATCH (not yet
    SATURATED). Mirrors the verdict-ladder semantics of briefing_health:
    a single STALE-bordering reading is WATCH, not DEAD."""
    rows = [
        {"title": "Nvidia announces $80B buyback", "age_hours": 1.0,
         "urgency": 2, "score_source": "llm"},
        {"title": "Nvidia unveils $80B buyback plan - MSN", "age_hours": 2.0,
         "urgency": 2, "score_source": "ml"},
    ]
    report = build_saturation_report(rows, HELD, saturation_threshold=2,
                                      heavy_threshold=5)
    assert report["verdict"] == "WATCH"
    # The single pair carries both rows.
    p = report["by_pair"][0]
    assert p["ticker"] == "NVDA"
    assert p["event_class"] == "BUYBACK"
    assert p["urgent_count"] == 2
    assert p["alerted_count"] == 2  # both urgency=2


def test_saturated_when_pair_at_heavy_threshold():
    """The live failure case: 14 urgent rows on (NVDA, BUYBACK) →
    SATURATED. This is the buyback-saturation pattern the audit exists
    to surface."""
    rows = [
        {"title": f"Nvidia unveils $80B buyback variant {i}",
         "age_hours": float(i),
         "urgency": 2, "score_source": "llm" if i % 2 == 0 else "ml"}
        for i in range(14)
    ]
    report = build_saturation_report(rows, HELD)
    assert report["verdict"] == "SATURATED"
    assert len(report["saturation_alerts"]) == 1
    p = report["by_pair"][0]
    assert p["urgent_count"] == 14
    # score_sources surfaces the calibration mix: 7 llm, 7 ml
    assert p["score_sources"] == {"llm": 7, "ml": 7, "briefing_boost": 0, "null": 0}


def test_multi_ticker_row_contributes_to_each_pair():
    """A row mentioning two held tickers contributes to BOTH pairs —
    same multi-attribution convention as cross_book_event_pulse and
    pushed_alert_event_concentration."""
    rows = [
        {"title": "Nvidia and MU report blowout earnings", "age_hours": 1.0,
         "urgency": 2, "score_source": "llm"},
    ]
    report = build_saturation_report(rows, HELD)
    pairs = {(p["ticker"], p["event_class"]) for p in report["by_pair"]}
    assert ("NVDA", "EARNINGS") in pairs
    assert ("MU", "EARNINGS") in pairs
    # The row counts once per pair, not aggregated total.
    for p in report["by_pair"]:
        assert p["urgent_count"] == 1


def test_title_without_closed_vocab_class_does_not_bucket():
    """Conservative under-claim: a title with no closed-vocab event class
    is counted in total_urgent but NEVER appears in by_pair. Mirrors
    pushed_alert_event_concentration's same discipline."""
    rows = [
        {"title": "Nvidia stock continues to struggle today",
         "age_hours": 1.0, "urgency": 2, "score_source": "ml"},
    ]
    report = build_saturation_report(rows, HELD)
    assert report["total_urgent"] == 1
    assert report["urgent_with_class"] == 0
    assert report["by_pair"] == []


def test_held_ticker_required_for_pair_bucketing():
    """A title with a class but no held ticker mention is counted in
    urgent_with_class but NEVER appears in by_pair — gating by the
    HELD universe is the audit's purpose."""
    rows = [
        {"title": "Boeing unveils $10B buyback", "age_hours": 1.0,
         "urgency": 2, "score_source": "llm"},
    ]
    report = build_saturation_report(rows, HELD)
    assert report["urgent_with_class"] == 1
    assert report["urgent_held_x_class"] == 0
    assert report["by_pair"] == []


def test_titles_sorted_newest_first():
    """Per-pair titles list is sorted newest-first (lowest age first),
    same display convention as build_concentration_report and the
    briefing's recency ranker."""
    rows = [
        {"title": "Nvidia announces $80B buyback A", "age_hours": 5.0,
         "urgency": 2, "score_source": "llm"},
        {"title": "Nvidia unveils $80B buyback B", "age_hours": 1.0,
         "urgency": 2, "score_source": "llm"},
        {"title": "Nvidia $80B buyback C", "age_hours": 3.0,
         "urgency": 2, "score_source": "llm"},
    ]
    report = build_saturation_report(rows, HELD)
    p = report["by_pair"][0]
    assert p["titles"][0].endswith("B")  # newest
    assert p["titles"][1].endswith("C")
    assert p["titles"][2].endswith("A")  # oldest
    # newest_title also tracks the newest entry, not just the first.
    assert p["newest_title"].endswith("B")
    assert p["newest_age_h"] == 1.0


def test_alerted_count_separates_urg_1_from_urg_2():
    """alerted_count counts ONLY urgency=2 rows (queue-exited via formatter
    or paraphrase suppression), so the analyst can see how many urgent
    rows were queued vs how many actually reached the alert formatter."""
    rows = [
        {"title": "Nvidia $80B buyback A", "age_hours": 1.0,
         "urgency": 1, "score_source": "llm"},        # queued only
        {"title": "Nvidia $80B buyback B", "age_hours": 2.0,
         "urgency": 2, "score_source": "llm"},        # exited queue
        {"title": "Nvidia $80B buyback C", "age_hours": 3.0,
         "urgency": 2, "score_source": "llm"},        # exited queue
    ]
    report = build_saturation_report(rows, HELD)
    p = report["by_pair"][0]
    assert p["urgent_count"] == 3
    assert p["alerted_count"] == 2


def test_unknown_score_source_lands_in_null_bucket():
    """A score_source outside the canonical set (llm/ml/briefing_boost)
    lands in the null bucket — mirrors urgency_label_split's discipline.
    Defensive against unexpected legacy values in the column."""
    rows = [
        {"title": "Nvidia $80B buyback", "age_hours": 1.0,
         "urgency": 2, "score_source": None},
        {"title": "Nvidia $80B buyback 2", "age_hours": 2.0,
         "urgency": 2, "score_source": "weird_unexpected"},
    ]
    report = build_saturation_report(rows, HELD)
    p = report["by_pair"][0]
    assert p["score_sources"]["null"] == 2


def test_sort_descending_urgent_then_alerted_then_alpha():
    """Worst-first deterministic sort: urgent_count desc → alerted_count
    desc → ticker → event_class. Same tiebreak convention as
    urgency_label_split_by_source and the push-side audit."""
    rows = [
        # NVDA × EARNINGS: 3 urgent, 1 alerted
        {"title": "Nvidia Q1 earnings beat estimates A", "age_hours": 1.0,
         "urgency": 1, "score_source": "llm"},
        {"title": "Nvidia Q1 earnings beat estimates B", "age_hours": 2.0,
         "urgency": 1, "score_source": "llm"},
        {"title": "Nvidia Q1 earnings beat estimates C", "age_hours": 3.0,
         "urgency": 2, "score_source": "llm"},
        # MU × EARNINGS: 3 urgent, 3 alerted (same count, more alerted → ranks higher)
        {"title": "MU Q1 earnings beat estimates A", "age_hours": 1.0,
         "urgency": 2, "score_source": "llm"},
        {"title": "MU Q1 earnings beat estimates B", "age_hours": 2.0,
         "urgency": 2, "score_source": "llm"},
        {"title": "MU Q1 earnings beat estimates C", "age_hours": 3.0,
         "urgency": 2, "score_source": "llm"},
    ]
    report = build_saturation_report(rows, HELD)
    # Same urgent_count: MU has more alerted → ranks first.
    assert report["by_pair"][0]["ticker"] == "MU"
    assert report["by_pair"][1]["ticker"] == "NVDA"


def test_by_pair_capped_at_max_rows():
    """A degraded window (many pairs) must not emit a wall-of-text — same
    anti-noise capping discipline as BRIEFING_MAX_PER_DOMAIN."""
    rows = []
    # Create 30 distinct (ticker, class) pairs across a held universe.
    # Use BUYBACK on all so we just vary the ticker.
    held = {f"AA{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    held.add("NVDA")
    for i, t in enumerate(sorted(held)):
        rows.append({
            "title": f"{t} announces $10B buyback", "age_hours": float(i),
            "urgency": 2, "score_source": "llm",
        })
    report = build_saturation_report(rows, held, max_by_pair_rows=5)
    # 27+ distinct pairs but only 5 returned in by_pair (distinct_pairs
    # reports the true count).
    assert report["distinct_pairs"] >= 20
    assert len(report["by_pair"]) == 5


def test_clamp_window_h_to_positive():
    """Defensive: window_h <= 0 clamps to a tiny positive — no divide-by-
    zero in the saturation_alerts formatting."""
    report = build_saturation_report([], HELD, window_h=0.0)
    assert report["window_h"] >= 0.01
    report2 = build_saturation_report([], HELD, window_h=-5.0)
    assert report2["window_h"] >= 0.01


def test_saturation_alerts_human_readable_lines():
    """saturation_alerts emits one line per flagged pair with the
    ticker × event_class summary — usable as a Discord-style report."""
    rows = [
        {"title": f"Nvidia $80B buyback {i}", "age_hours": float(i),
         "urgency": 2, "score_source": "llm"}
        for i in range(3)
    ]
    report = build_saturation_report(rows, HELD, saturation_threshold=2)
    assert len(report["saturation_alerts"]) == 1
    line = report["saturation_alerts"][0]
    assert "NVDA" in line and "BUYBACK" in line
    assert "3 urgent" in line


# ── Invariant 1: BACKTEST ISOLATION ──────────────────────────────────────────
# The articles.db SELECT applies _LIVE_ONLY_CLAUSE — synthetic backtest/opus
# rows can never inflate the saturation figure. Pinned both at the SQL
# clause level AND at a roundtrip test against a real sqlite DB.


def test_live_only_clause_matches_storage_layer():
    """_LIVE_ONLY_CLAUSE must match storage.article_store._LIVE_ONLY_CLAUSE
    byte-for-byte (whitespace-normalized) — anti-drift between the two
    backtest-isolation surfaces."""
    from storage.article_store import _LIVE_ONLY_CLAUSE as storage_clause
    normalize = lambda s: re.sub(r"\s+", " ", s).strip()
    assert normalize(_LIVE_ONLY_CLAUSE) == normalize(storage_clause)


def test_load_urgent_rows_excludes_backtest_urls(tmp_path, monkeypatch):
    """The SQL load path must NEVER return rows with backtest:// URLs or
    backtest_/opus_annotation* sources, even if they have urgency >= 1.
    This is invariant #1 (backtest isolation, CLAUDE.md §5) tested at the
    database level."""
    db = tmp_path / "articles.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE articles (
            id TEXT PRIMARY KEY, url TEXT, title TEXT, source TEXT,
            published TEXT, kw_score REAL, ai_score REAL, urgency INTEGER,
            full_text BLOB, first_seen TEXT, cycle INTEGER,
            time_sensitivity REAL, ml_score REAL, score_source TEXT
        )
    """)
    now = "2026-05-25T10:00:00+00:00"
    # 1 LIVE urgent row (should appear)
    conn.execute(
        "INSERT INTO articles (id, url, title, source, urgency, first_seen) "
        "VALUES ('a1', 'https://reuters.com/x', 'NVDA buyback announced', "
        "'reuters', 2, ?)", (now,))
    # 3 SYNTHETIC urgent rows (must NOT appear)
    conn.execute(
        "INSERT INTO articles (id, url, title, source, urgency, first_seen) "
        "VALUES ('b1', 'backtest://run42/winner/NVDA', 'NVDA buyback BACKTEST', "
        "'rss', 2, ?)", (now,))
    conn.execute(
        "INSERT INTO articles (id, url, title, source, urgency, first_seen) "
        "VALUES ('b2', 'https://x.com', 'NVDA buyback BACKTEST 2', "
        "'backtest_run_42_winner', 2, ?)", (now,))
    conn.execute(
        "INSERT INTO articles (id, url, title, source, urgency, first_seen) "
        "VALUES ('b3', 'https://x.com/y', 'NVDA buyback OPUS', "
        "'opus_annotation_cycle_1', 2, ?)", (now,))
    conn.commit()
    conn.close()

    monkeypatch.setattr(ues, "_resolve_articles_db", lambda: db)
    rows = _load_urgent_rows(hours=72.0)
    titles = {r["title"] for r in rows}
    assert "NVDA buyback announced" in titles
    assert all("BACKTEST" not in t and "OPUS" not in t for t in titles)
    assert len(rows) == 1


# ── Invariants 2/3: ml_score vs ai_score / score_source — NO MUTATION ────────
# The audit is read-only. No DB writes, no score_source / ai_score /
# ml_score / urgency mutation. Verified by checking the row state on the
# DB is byte-identical before and after a CLI run.


def test_audit_is_read_only_no_db_mutation(tmp_path, monkeypatch):
    """Invariants #2 + #3: the audit performs NO DB writes — ai_score,
    ml_score, score_source, urgency are all unchanged after the load.
    Pinned by reading the row state pre and post."""
    db = tmp_path / "articles.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE articles (
            id TEXT PRIMARY KEY, url TEXT, title TEXT, source TEXT,
            published TEXT, kw_score REAL, ai_score REAL, urgency INTEGER,
            full_text BLOB, first_seen TEXT, cycle INTEGER,
            time_sensitivity REAL, ml_score REAL, score_source TEXT
        )
    """)
    now = "2026-05-25T10:00:00+00:00"
    conn.execute(
        "INSERT INTO articles (id, url, title, source, ai_score, ml_score, "
        "urgency, first_seen, score_source) "
        "VALUES ('a1', 'https://x.com', 'NVDA $80B buyback', 'reuters', "
        "0, 9.5, 2, ?, 'ml')", (now,))
    conn.commit()

    before = conn.execute(
        "SELECT ai_score, ml_score, urgency, score_source FROM articles"
    ).fetchall()
    conn.close()

    monkeypatch.setattr(ues, "_resolve_articles_db", lambda: db)
    _ = _load_urgent_rows(hours=72.0)
    # Build a report too — anything that writes would show.
    rows = _load_urgent_rows(hours=72.0)
    _ = build_saturation_report(rows, HELD)

    conn2 = sqlite3.connect(str(db))
    after = conn2.execute(
        "SELECT ai_score, ml_score, urgency, score_source FROM articles"
    ).fetchall()
    conn2.close()
    assert before == after


# ── Invariant 4: score_source pass-through accuracy ──────────────────────────


def test_score_source_pass_through_not_mutated():
    """The audit reports score_source per pair via the score_sources dict
    but never modifies the per-row tag. A row tagged 'ml' arrives in the
    'ml' bucket — verified by direct count."""
    rows = [
        {"title": "Nvidia $80B buyback A", "age_hours": 1.0,
         "urgency": 2, "score_source": "llm"},
        {"title": "Nvidia $80B buyback B", "age_hours": 2.0,
         "urgency": 2, "score_source": "ml"},
        {"title": "Nvidia $80B buyback C", "age_hours": 3.0,
         "urgency": 2, "score_source": "briefing_boost"},
        {"title": "Nvidia $80B buyback D", "age_hours": 4.0,
         "urgency": 1, "score_source": None},
    ]
    report = build_saturation_report(rows, HELD)
    p = report["by_pair"][0]
    assert p["score_sources"] == {
        "llm": 1, "ml": 1, "briefing_boost": 1, "null": 1
    }


# ── CLI / load-path degrade-gracefully ───────────────────────────────────────


def test_load_urgent_rows_missing_db_returns_empty(monkeypatch, tmp_path):
    """A missing articles.db must not crash the audit; returns []. Same
    discipline as pushed_alert_event_concentration._load_pushed."""
    nonexistent = tmp_path / "no_such.db"
    monkeypatch.setattr(ues, "_resolve_articles_db", lambda: nonexistent)
    rows = _load_urgent_rows(hours=24.0)
    assert rows == []


def test_load_urgent_rows_corrupted_db_returns_empty(monkeypatch, tmp_path):
    """A corrupted file (not sqlite) is treated as a missing DB — best-
    effort degrade-gracefully, matches the push-side audit's contract."""
    bad = tmp_path / "bad.db"
    bad.write_text("this is not a sqlite database")
    monkeypatch.setattr(ues, "_resolve_articles_db", lambda: bad)
    rows = _load_urgent_rows(hours=24.0)
    assert rows == []


# ── Live-failure pin: NVDA buyback saturation ────────────────────────────────


def test_live_failure_nvda_buyback_saturation():
    """Pin the live failure shape that motivated this audit. Live evidence
    (2026-05-25, articles.db 24h window): 14 urgency>=1 rows about the
    same NVDA $80B buyback event. The report MUST flag this as SATURATED
    with the (NVDA, BUYBACK) pair leading by_pair."""
    titles = [
        "Nvidia unveils $80B buyback, 25x dividend hike on record earnings - MSN",
        "Nvidia posts record $81.6B quarter, unveils $80B buyback and dividend hike - MSN",
        "Nvidia posts $81.6B quarter, unveils $80B buyback plan - MSN",
        "Nvidia's board just authorized an additional $80 billion buyback. Here's what th...",
        "NVIDIA projects $91B Q2 revenue while outlining $80B buyback and a $0.25 quarter",
        "Nvidia posts record $81.6B revenue, unveils $80B buyback plan - MSN",
        "KLA (KLAC) Is Up 7.5% After Stock Split, Buyback, Dividend Hike",
        "Nvidia unveils $80B buyback, 25x dividend hike on record earnings - MSN",
        "Nvidia $80B buyback plan announced",
    ]
    rows = [
        {"title": t, "age_hours": float(i) * 0.5,
         "urgency": 2 if i % 3 != 0 else 1,
         "score_source": "ml" if i % 2 == 0 else "llm"}
        for i, t in enumerate(titles)
    ]
    report = build_saturation_report(rows, HELD)
    assert report["verdict"] == "SATURATED"
    top = report["by_pair"][0]
    assert top["ticker"] == "NVDA"
    assert top["event_class"] == "BUYBACK"
    # Most of the NVDA-buyback titles map to (NVDA, BUYBACK); KLAC is not
    # in HELD so does not enter the pair table, NVDIA aliases via the
    # SSOT _held_tickers_in_title.
    assert top["urgent_count"] >= HEAVY_THRESHOLD
    assert any("NVDA" in line and "BUYBACK" in line
               for line in report["saturation_alerts"])
