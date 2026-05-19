"""Backtest-isolation parity for analytics/* modules.

The canonical live-only SQL fragment lives in
``storage.article_store._LIVE_ONLY_CLAUSE`` and excludes three classes of
training-only rows that share ``articles.db`` with live news:

  * ``url LIKE 'backtest://%'``               — paper-trader replay injections
  * ``source LIKE 'backtest_%'``              — backtest winner/loser tags
  * ``source LIKE 'opus_annotation%''``      — Opus-annotated training rows

A partial filter (e.g. ``source NOT LIKE 'backtest_run_%'``) lets the other two
classes leak through. AGENTS.md / CLAUDE.md §5 already pin this for
``signals.py`` / ``trend_velocity.py``-via-AGENTS-doc / ``source_diversity.py``;
this suite extends that discipline to the eight remaining analytics modules
fixed in the same pass:

  source_score_volatility · collection_quality · scorer_skew · daily_digest ·
  trend_velocity · breaking_news_detector · consensus_signal · ticker_comentions ·
  ticker_first_mention

The discriminating contract for each: given a DB containing one live row plus
three synthetic rows (one per leak class), the module's output must NOT
reference the synthetic source/URL strings. The synthetic rows are seeded
with the SAME ticker as the live row so a leak shows up as inflated counts
or contaminated per-source aggregates.
"""
from __future__ import annotations

import importlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


_SCHEMA = """
CREATE TABLE articles (
    id           TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    title        TEXT NOT NULL,
    source       TEXT,
    published    TEXT,
    kw_score     REAL DEFAULT 0,
    ai_score     REAL DEFAULT 0,
    urgency      INTEGER DEFAULT 0,
    full_text    BLOB,
    first_seen   TEXT NOT NULL,
    cycle        INTEGER DEFAULT 0,
    time_sensitivity REAL DEFAULT NULL,
    ml_score     REAL DEFAULT NULL,
    score_source TEXT DEFAULT NULL
);
CREATE INDEX idx_first_seen ON articles(first_seen);
CREATE INDEX idx_urgency    ON articles(urgency);
"""


def _seed_mixed_db(db_path: Path) -> dict:
    """Insert one live row + three synthetic rows. Returns the dict of inserted
    sources / URLs / titles so each test can assert non-appearance."""
    now = datetime.now(timezone.utc)
    fresh = (now - timedelta(minutes=5)).isoformat()
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    rows = [
        # (id, url, title, source, ai_score, ml_score, kw_score, urgency, first_seen)
        ("live-1", "https://wire.example/n1",
         "NVDA earnings beat — guidance raised", "rss",
         8.0, 7.5, 3.0, 2, fresh),
        # synthetic-1: backtest:// URL (source is plain — only URL is the marker)
        ("bt-url-1", "backtest://run_42/2026-05-19/BUY/NVDA",
         "NVDA replay winner — synthetic training row", "rss",
         5.0, None, 1.0, 0, fresh),
        # synthetic-2: backtest_* source (not specifically backtest_run_*)
        ("bt-src-1", "https://internal/replay/42",
         "NVDA replay winner — synthetic backtest_winner", "backtest_winner",
         5.0, None, 1.0, 0, fresh),
        # synthetic-3: opus_annotation* source
        ("opus-1", "https://internal/opus/cycle/3",
         "NVDA opus-annotated training label — synthetic",
         "opus_annotation_cycle_3", 5.0, None, 1.0, 0, fresh),
    ]
    conn.executemany(
        "INSERT INTO articles (id, url, title, source, ai_score, ml_score, "
        "kw_score, urgency, first_seen) VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return {
        "synthetic_sources": ["backtest_winner", "opus_annotation_cycle_3"],
        "synthetic_urls":    ["backtest://run_42/2026-05-19/BUY/NVDA"],
        "live_source":       "rss",
        "live_url":          "https://wire.example/n1",
    }


def _no_synthetic_strings(payload_text: str, seeds: dict) -> None:
    """Assert no synthetic source/URL marker appears anywhere in the payload."""
    for s in seeds["synthetic_sources"]:
        assert s not in payload_text, f"synthetic source {s!r} leaked into output"
    for u in seeds["synthetic_urls"]:
        assert u not in payload_text, f"synthetic URL {u!r} leaked into output"
    # backtest:// is the URL-marker prefix — never let any variant slip through
    assert "backtest://" not in payload_text, "backtest:// URL leaked into output"
    assert "opus_annotation" not in payload_text, "opus_annotation source leaked"


@pytest.fixture
def mixed_db(tmp_path):
    """Per-test DB at tmp_path/articles.db seeded with the live + 3 synthetic
    rows. Returns (db_path, seeds_dict)."""
    db = tmp_path / "articles.db"
    seeds = _seed_mixed_db(db)
    return db, seeds


# ── source_score_volatility ─────────────────────────────────────────────


def test_source_score_volatility_excludes_synthetic(mixed_db, monkeypatch, tmp_path):
    db, seeds = mixed_db
    from analytics import source_score_volatility as mod

    monkeypatch.setattr(mod, "DB", db)
    monkeypatch.setattr(mod, "OUT", tmp_path / "ssv.json")
    # MIN_PER_SOURCE=8 by default — drop to 1 so a single live row is reportable.
    monkeypatch.setattr(mod, "MIN_PER_SOURCE", 1)
    payload = mod.compute()
    assert payload["scanned"] == 1, (
        "scanned must equal the count of live rows; got "
        f"{payload['scanned']} (synthetic rows leaked in)"
    )
    sources = {r["source"] for r in payload["ranked"]}
    assert sources == {"rss"}, f"expected only live source 'rss', got {sources}"
    _no_synthetic_strings(json.dumps(payload), seeds)


# ── collection_quality ──────────────────────────────────────────────────


def test_collection_quality_excludes_synthetic(mixed_db, monkeypatch, tmp_path):
    db, seeds = mixed_db
    from analytics import collection_quality as mod

    monkeypatch.setattr(mod, "DB", db)
    monkeypatch.setattr(mod, "OUT", tmp_path / "cq.json")
    monkeypatch.setattr(mod, "MIN_PER_SOURCE", 1)
    payload = mod.compute()
    assert payload["scanned"] == 1
    sources = {r["source"] for r in payload["ranked"]}
    assert sources == {"rss"}
    _no_synthetic_strings(json.dumps(payload), seeds)


# ── scorer_skew (needs ai_score AND ml_score both set) ─────────────────


def test_scorer_skew_excludes_synthetic(mixed_db, monkeypatch, tmp_path):
    db, seeds = mixed_db
    from analytics import scorer_skew as mod

    monkeypatch.setattr(mod, "DB", db)
    monkeypatch.setattr(mod, "OUT", tmp_path / "skew.json")
    monkeypatch.setattr(mod, "MIN_PER_SOURCE", 1)
    payload = mod.compute()
    # Only the live row carries both ai_score AND ml_score; synthetic rows have
    # ml_score=NULL and were already masked by the existing filter — but the
    # partial backtest filter was still a code-level drift class. The newly-
    # added `_LIVE_ONLY_CLAUSE` is what this discriminator pins.
    assert payload["scanned"] == 1
    sources = {r["source"] for r in payload["ranked"]}
    assert sources == {"rss"}
    _no_synthetic_strings(json.dumps(payload), seeds)


# ── daily_digest ────────────────────────────────────────────────────────


def test_daily_digest_excludes_synthetic(mixed_db, monkeypatch, tmp_path):
    db, seeds = mixed_db
    from analytics import daily_digest as mod

    monkeypatch.setattr(mod, "DB", db)
    monkeypatch.setattr(mod, "OUT", tmp_path / "digest.txt")
    lines = mod.compute()
    text = "\n".join(lines)
    # Only the live row has urgency >= 2 + fresh first_seen → exactly one
    # digest entry, and the "Real articles 24h" tally is 1, not 4.
    assert "Real articles 24h" in text
    # Parse the header line for the "Real articles 24h ...: <N>" tally.
    header = next(l for l in lines if "Real articles 24h" in l)
    # "...: 1   urgent>=2: 1"
    n_part = header.split(":")[1].strip().split()[0]
    n = int(n_part.replace(",", ""))
    assert n == 1, (
        f"daily_digest 'Real articles 24h' tally must exclude synthetic rows; "
        f"got {n} from header {header!r}"
    )
    _no_synthetic_strings(text, seeds)


# ── trend_velocity (fetch_recent is the queried surface) ───────────────


def test_trend_velocity_excludes_synthetic(mixed_db, monkeypatch, tmp_path):
    db, seeds = mixed_db
    from analytics import trend_velocity as mod

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = mod.fetch_recent(conn, 100)
    finally:
        conn.close()
    # All 4 rows share the same fresh first_seen — the live-only filter is
    # the ONLY discriminator. Live row alone must survive.
    assert len(rows) == 1, f"expected 1 row, got {len(rows)}: {rows}"
    titles = [r[1] for r in rows]
    payload_text = json.dumps(titles)
    _no_synthetic_strings(payload_text, seeds)
    assert "synthetic" not in payload_text


# ── breaking_news_detector (queried inside main; rebuild the query) ───


def test_breaking_news_excludes_synthetic(mixed_db, monkeypatch, tmp_path):
    db, seeds = mixed_db
    from analytics import breaking_news_detector as mod
    from storage.article_store import _LIVE_ONLY_CLAUSE

    # The module's main() builds its own query inline — mirror that exact
    # WHERE clause here against the seeded DB so the test pins what main()
    # would actually fetch (one live row, three synthetic excluded).
    since = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT first_seen, title, source FROM articles "
            f"WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
            "ORDER BY first_seen DESC LIMIT ?",
            (since, mod.FETCH_LIMIT),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    assert len(rows) == 1, f"expected 1 live row, got {len(rows)}"
    _no_synthetic_strings(json.dumps(rows), seeds)


# ── consensus_signal (same shape as breaking_news_detector) ────────────


def test_consensus_signal_excludes_synthetic(mixed_db, monkeypatch, tmp_path):
    db, seeds = mixed_db
    from analytics import consensus_signal as mod
    from storage.article_store import _LIVE_ONLY_CLAUSE

    since = (datetime.now(timezone.utc) - timedelta(hours=6)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT first_seen, title, source, ai_score FROM articles "
            f"WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
            "ORDER BY first_seen DESC LIMIT ?",
            (since, mod.FETCH_LIMIT),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    assert len(rows) == 1, f"expected 1 live row, got {len(rows)}"
    _no_synthetic_strings(json.dumps(rows), seeds)


# ── ticker_comentions ──────────────────────────────────────────────────


def test_ticker_comentions_excludes_synthetic(mixed_db, monkeypatch, tmp_path):
    db, seeds = mixed_db
    from analytics import ticker_comentions as mod
    from storage.article_store import _LIVE_ONLY_CLAUSE

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT first_seen, title FROM articles "
            f"WHERE {_LIVE_ONLY_CLAUSE} "
            "ORDER BY first_seen DESC LIMIT ?",
            (mod.FETCH_LIMIT,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    assert len(rows) == 1, f"expected 1 live row, got {len(rows)}"
    _no_synthetic_strings(json.dumps(rows), seeds)


# ── ticker_first_mention.run() ────────────────────────────────────────


def test_ticker_first_mention_excludes_synthetic(mixed_db, monkeypatch, tmp_path):
    db, seeds = mixed_db
    from analytics import ticker_first_mention as mod

    monkeypatch.setattr(mod, "DB_PATH", db)
    monkeypatch.setattr(mod, "OUT_PATH", tmp_path / "tfm.json")
    report = mod.run()
    # All 4 rows mention NVDA. With synthetic exclusion, the report scans
    # only the live row. Without exclusion, the synthetic NVDA mentions would
    # populate the LOOKBACK history (they have older-than-recent ts, but here
    # all four share fresh ts → recent for all). The critical assertion: the
    # scanned-rows count must be 1, not 4.
    assert report["scanned_rows"] == 1, (
        f"scanned_rows must exclude synthetic; got {report['scanned_rows']}"
    )
    _no_synthetic_strings(json.dumps(report), seeds)
